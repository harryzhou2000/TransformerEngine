/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_runtime.h>
#include <transformer_engine/fused_router.h>

#include "../common.h"
#include "../util/logging.h"
#include "async_loader.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_router {

template <typename DataType, typename BiasType, TopkFuncType TopkFunc = TopkFuncType::Naive>
__global__ void fused_topk_with_score_function_forward_kernel(
    const DataType *logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const BiasType *expert_bias, DataType *probs, bool *routing_map, CompType *intermediate_output,
    int num_buffers) {
  /***
     * Section: Global Variables/Addresses init
     *
     * Shmem layout — logits stored in original DataType for cp.async on all dtypes:
     *   [logits_raw: NB * E * W * sizeof(DataType)]  — async double-buffered
     *   [scores:     E * W * sizeof(CompType)]        — work area (score func + topk)
     *   [topk_scores: K * W * sizeof(CompType)]
     *   [topk_indices: K * W * sizeof(int)]
     *   (if group_topk > 0:)
     *     [masked_scores: E * W * sizeof(CompType)]
     *     [group_scores: G * W * sizeof(CompType)]
     *
     * Flow: async load DataType logits → wait → convert+score_func into scores → topk
     */
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ char shmem_raw[];

  // --- Async-loaded logits (DataType, double-buffered) ---
  char *shmem_ptr = shmem_raw;
  DataType *logits_shmem_base = reinterpret_cast<DataType *>(shmem_ptr);
  RawAsyncLoader<DataType> loader(logits_shmem_base, warp_id, num_experts, num_token_per_block,
                                  num_buffers);
  shmem_ptr += RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);

  // --- Scores work area (CompType, single-buffered) ---
  CompType *scores_buf = reinterpret_cast<CompType *>(shmem_ptr);
  shmem_ptr += num_experts * num_token_per_block * sizeof(CompType);

  // --- Topk scratch ---
  CompType *topk_scores_buf = reinterpret_cast<CompType *>(shmem_ptr);
  CompType *group_scores_buf = nullptr, *masked_scores_buf = nullptr;
  int *topk_indices_buf = nullptr;
  if (group_topk > 0) {
    masked_scores_buf = topk_scores_buf + topk * num_token_per_block;
    group_scores_buf = masked_scores_buf + num_experts * num_token_per_block;
    topk_indices_buf = reinterpret_cast<int *>(group_scores_buf + num_groups * num_token_per_block);
  } else {
    topk_indices_buf = reinterpret_cast<int *>(topk_scores_buf + topk * num_token_per_block);
  }

  // Per-warp pointers
  CompType *scores = scores_buf + warp_id * num_experts;
  CompType *topk_scores = topk_scores_buf + warp_id * topk;
  CompType *masked_scores =
      (masked_scores_buf != nullptr) ? masked_scores_buf + warp_id * num_experts : nullptr;
  CompType *group_scores =
      (group_scores_buf != nullptr) ? group_scores_buf + warp_id * num_groups : nullptr;
  int *topk_indices = topk_indices_buf + warp_id * topk;

  /***
     * Section: Main Loop with async double-buffered load
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  int first_round = blockIdx.x;
  if (first_round >= total_round) return;

  // Kick off first async load
  {
    int first_token = first_round * num_token_per_block + warp_id;
    if (first_token < num_tokens) {
      loader.load_current(logits + first_token * num_experts, num_experts, lane_id);
    }
  }

  for (int round = first_round; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    if (token_offset_cur_warp >= num_tokens) break;

    // Wait for current buffer's async load to complete
    loader.wait();
    DataType *raw_logits = loader.current_buf();

    // Kick off async prefetch for next round (overlaps with all compute below)
    int next_round = round + gridDim.x;
    if (next_round < total_round) {
      int next_token = next_round * num_token_per_block + warp_id;
      if (next_token < num_tokens) {
        loader.start_load(logits + next_token * num_experts, num_experts, lane_id);
      }
    }

    int pos_offset = token_offset_cur_warp * num_experts;

    // Clear the probs/routing_map for this token
    vec_fill_global(probs + pos_offset, static_cast<DataType>(0), num_experts, lane_id);
    vec_fill_global(routing_map + pos_offset, false, num_experts, lane_id);

    /***
         * Section: Fused Preprocess
         * Convert DataType→CompType + apply score function + write intermediate_output
         * all in one pass over the raw logits.
         */
    if (use_pre_softmax && score_function == 1) {
      // Softmax: need all values converted first, then 2-pass softmax
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        scores[i] = static_cast<CompType>(raw_logits[i]);
      }
      __syncwarp();
      apply_softmax_on_float(scores, num_experts, lane_id);
      __syncwarp();
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = scores[i];
      }
    } else if (score_function == 0) {
      // Sigmoid: fused convert + apply + save + bias
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float val = sigmoid_scalar(static_cast<CompType>(raw_logits[i]));
        intermediate_output[pos_offset + i] = val;
        if (expert_bias) val += static_cast<CompType>(expert_bias[i]);
        scores[i] = val;
      }
    } else if (score_function == 2) {
      // Sqrtsoftplus: fused convert + save-logits + apply + bias
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float logit = static_cast<CompType>(raw_logits[i]);
        intermediate_output[pos_offset + i] = logit;
        float val = sqrtsoftplus_scalar(logit);
        if (expert_bias) val += static_cast<CompType>(expert_bias[i]);
        scores[i] = val;
      }
    } else if (!use_pre_softmax && score_function == 1) {
      // Post-softmax: convert logits to scores, init intermediate_output to -inf
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        scores[i] = static_cast<CompType>(raw_logits[i]);
        intermediate_output[pos_offset + i] = -std::numeric_limits<CompType>::infinity();
      }
    }
    __syncwarp();

    // If group_topk > 0, init the masked_scores to -inf
    if (group_topk > 0) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        masked_scores[i] = -std::numeric_limits<CompType>::infinity();
      }
      __syncwarp();
    }

    /***
         * Section: Topk
         * Get the topk indices (same logic as before, no change)
         */
    if (group_topk > 0) {
      int group_size = num_experts / num_groups;
      for (int i = 0; i < num_groups; i++) {
        topk_and_mask<TopkFunc>(
            /*scores ptr = */ scores + i * group_size,
            /*data size = */ group_size,
            /*topk = */ topk / group_topk,
            /*topk indices ptr = */ topk_indices,
            /*topk scores ptr = */ topk_scores,
            /*lane id = */ lane_id);
        __syncwarp();
        if (lane_id == 0) {
          CompType tmp = 0.0;
          for (int j = 0; j < topk / group_topk; j++) {
            tmp = tmp + topk_scores[j];
          }
          group_scores[i] = tmp;
        }
        __syncwarp();
      }

      topk_and_mask<TopkFunc>(group_scores, num_groups, group_topk, topk_indices, topk_scores,
                              lane_id);
      __syncwarp();
      for (int i = 0; i < group_topk; i++) {
        int st = topk_indices[i] * group_size;
        int ed = st + group_size;
        for (int j = st + lane_id; j < ed; j += kThreadsPerWarp) {
          masked_scores[j] = scores[j];
        }
      }
      __syncwarp();
      topk_and_mask<TopkFunc>(masked_scores, num_experts, topk, topk_indices, topk_scores, lane_id);
    } else {
      topk_and_mask<TopkFunc>(scores, num_experts, topk, topk_indices, topk_scores, lane_id);
    }
    __syncwarp();

    /***
         * Section: Postprocess
         * - Revert Expert bias
         * - Softmax / Sigmoid / Sqrtsoftplus post-processing
         * - Write the result with scaling_factor
         */
    // Revert expert bias from topk scores
    if (expert_bias && (score_function == 0 || score_function == 2)) {
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        topk_scores[i] = topk_scores[i] - static_cast<CompType>(expert_bias[topk_indices[i]]);
      }
      __syncwarp();
    }

    // Post-softmax on topk scores
    if (!use_pre_softmax && score_function == 1) {
      apply_softmax_on_float(topk_scores, topk, lane_id);
      __syncwarp();
      // Save softmax output for backward (scatter — topk positions only)
      for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + topk_indices[i]] = topk_scores[i];
      }
      __syncwarp();
    }

    // Sigmoid/Sqrtsoftplus normalization when topk > 1
    if (score_function == 0 || score_function == 2) {
      if (topk > 1) {
        CompType sum_scores = warp_reduce_on_shmem(topk_scores, topk, ReduceFuncType::SUM, lane_id);
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          topk_scores[i] = topk_scores[i] / (sum_scores + epsilon);
        }
      }
      __syncwarp();
    }

    // Write the probs/routing_map to the output tensor
    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
      probs[pos_offset + topk_indices[i]] = scaling_factor * topk_scores[i];
    }
    __syncwarp();

    // Flip double buffer for next round
    loader.flip();
  }
}

template <typename DataType, typename BiasType>
void fused_topk_with_score_function_forward_kernel_launcher(
    const DataType *logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const BiasType *expert_bias, DataType *probs, bool *routing_map, CompType *intermediate_output,
    cudaStream_t stream) {
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t total_blocks = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  // Shmem: raw logits (DataType, double-buffered) + scores work (CompType) + topk scratch
  size_t scores_shmem = num_experts * num_token_per_block * sizeof(CompType);  // scores work buf
  size_t scratch_shmem = topk * num_token_per_block * sizeof(CompType)         // topk_scores
                         + topk * num_token_per_block * sizeof(int);           // topk_indices
  if (group_topk > 0) {
    scratch_shmem += num_groups * num_token_per_block * sizeof(CompType);   // group_scores
    scratch_shmem += num_experts * num_token_per_block * sizeof(CompType);  // masked_scores
  }
  size_t other_shmem = scores_shmem + scratch_shmem;
  // Decide single vs double buffer for the raw logits loader
  size_t logits_single_buf =
      RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, 1);
  int num_buffers = choose_num_buffers(logits_single_buf, other_shmem);
  size_t logits_raw_shmem =
      RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);
  size_t shared_memory_size = logits_raw_shmem + other_shmem;
  check_shared_memory_capacity_num_experts(shared_memory_size, num_experts);
  // Radix selection is O(E), independent of K, but it needs 4 passes for 32-bit float;
  // switch at K=16 where naive O(K^2*E) starts to dominate
  if (topk < 16) {
    auto kernel =
        fused_topk_with_score_function_forward_kernel<DataType, BiasType, TopkFuncType::Naive>;
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shared_memory_size));
    size_t grid_size =
        compute_persistent_grid(kernel, kThreadsPerBlock, shared_memory_size, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
        logits, num_tokens, num_experts, topk, use_pre_softmax, num_groups, group_topk,
        scaling_factor, score_function, expert_bias, probs, routing_map, intermediate_output,
        num_buffers);
  } else {
    auto kernel =
        fused_topk_with_score_function_forward_kernel<DataType, BiasType, TopkFuncType::Radix>;
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shared_memory_size));
    size_t grid_size =
        compute_persistent_grid(kernel, kThreadsPerBlock, shared_memory_size, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
        logits, num_tokens, num_experts, topk, use_pre_softmax, num_groups, group_topk,
        scaling_factor, score_function, expert_bias, probs, routing_map, intermediate_output,
        num_buffers);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_topk_with_score_function_forward(const Tensor logits, int num_tokens, int num_experts,
                                            int topk, bool use_pre_softmax, int num_groups,
                                            int group_topk, float scaling_factor,
                                            int score_function, const Tensor expert_bias,
                                            Tensor probs, Tensor routing_map,
                                            Tensor intermediate_output, cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      logits.data.dtype, DataType,
      TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
          expert_bias.data.dtype, BiasType,
          fused_topk_with_score_function_forward_kernel_launcher<DataType, BiasType>(
              reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,
              use_pre_softmax, num_groups, group_topk, scaling_factor, score_function,
              reinterpret_cast<BiasType *>(expert_bias.data.dptr),
              reinterpret_cast<DataType *>(probs.data.dptr),
              reinterpret_cast<bool *>(routing_map.data.dptr),
              reinterpret_cast<CompType *>(intermediate_output.data.dptr), stream);););
}

// Streaming 1-pass backward (for sigmoid/sqrtsoftplus with topk == 1).
// When topk == 1, no normalization is needed — pure element-wise ops with no reduction.
// Single read from global, all computation in registers, single write to global.
//
// Streaming 2-pass backward (for all other cases: topk > 1 or softmax).
// Pass 1: Read all inputs → compute reduction sums → warp shuffle.
// Pass 2: Read same inputs (L1-cached) → apply backward → write output.
//
// score_function is templated to eliminate dead branches at compile time.
template <typename DataType, int ScoreFunc>
__global__ void fused_topk_with_score_function_backward_kernel(
    const bool *routing_map, const CompType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    DataType *grad_logits) {
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;

  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_idx = round * num_token_per_block + warp_id;
    if (token_idx >= num_tokens) break;
    int pos = token_idx * num_experts;

    // ---- Pass 1: Compute reduction sums (only when needed) ----
    CompType sum_act = 0.0f;
    CompType sum_grad_act = 0.0f;
    CompType sum_output_x_grad = 0.0f;

    // Sigmoid/sqrtsoftplus normalization bwd needs reductions only when topk > 1
    bool need_norm_reduce = (ScoreFunc == 0 || ScoreFunc == 2) && topk > 1;
    // Softmax bwd always needs reduction
    bool need_softmax_reduce = (ScoreFunc == 1);

    if (need_norm_reduce || need_softmax_reduce) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        CompType g = static_cast<CompType>(grad_probs[pos + i]) * scaling_factor;
        CompType act = intermediate_output[pos + i];
        bool routed = routing_map[pos + i];

        if constexpr (ScoreFunc == 0) {
          if (routed) {
            sum_act += act;
            sum_grad_act += g * act;
          }
        } else if constexpr (ScoreFunc == 2) {
          if (routed) {
            CompType act_val = sqrtsoftplus_scalar(act);
            sum_act += act_val;
            sum_grad_act += g * act_val;
          }
        } else if constexpr (ScoreFunc == 1) {
          if (!use_pre_softmax) {
            if (routed) sum_output_x_grad += g * act;
          } else {
            CompType masked_g = routed ? g : 0.0f;
            sum_output_x_grad += masked_g * act;
          }
        }
      }

#pragma unroll
      for (int s = 16; s > 0; s >>= 1) {
        if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
          sum_act += __shfl_xor_sync(0xffffffff, sum_act, s);
          sum_grad_act += __shfl_xor_sync(0xffffffff, sum_grad_act, s);
        }
        if constexpr (ScoreFunc == 1) {
          sum_output_x_grad += __shfl_xor_sync(0xffffffff, sum_output_x_grad, s);
        }
      }
    }

    // ---- Pass 2 (or only pass if no reduction needed): Apply backward + write ----
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      CompType g = static_cast<CompType>(grad_probs[pos + i]);
      CompType act = intermediate_output[pos + i];
      bool routed = routing_map[pos + i];

      g *= scaling_factor;

      // Normalization backward (sigmoid/sqrtsoftplus, topk > 1)
      if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
        if (topk > 1) {
          CompType act_val = act;
          if constexpr (ScoreFunc == 2) act_val = sqrtsoftplus_scalar(act);
          if (routed) {
            CompType denom = sum_act + epsilon;
            g = g / denom - sum_grad_act / (denom * denom);
          } else {
            g = 0.0f;
          }
        }
      }

      // Post-softmax backward
      if constexpr (ScoreFunc == 1) {
        if (!use_pre_softmax) {
          g = routed ? act * (g - sum_output_x_grad) : 0.0f;
        }
      }

      // Topk backward: mask unselected
      if (!routed) g = 0.0f;

      // Pre-softmax backward
      if constexpr (ScoreFunc == 1) {
        if (use_pre_softmax) {
          g = act * (g - sum_output_x_grad);
        }
      }

      // Activation backward
      if constexpr (ScoreFunc == 0) {
        g = g * act * (1.0f - act);
      } else if constexpr (ScoreFunc == 2) {
        CompType y = sqrtsoftplus_scalar(act);
        CompType dy_dx = (act > 20.0f) ? (1.0f / (2.0f * y + epsilon))
                                       : (sigmoid_scalar(act) / (2.0f * y + epsilon));
        g = g * dy_dx;
      }

      grad_logits[pos + i] = static_cast<DataType>(g);
    }
  }
}

template <typename DataType>
void fused_topk_with_score_function_backward_kernel_launcher(
    const bool *routing_map, const CompType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, DataType *grad_logits, cudaStream_t stream) {
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t total_blocks = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = 0;

  // Dispatch on score_function to eliminate dead branches at compile time
  auto launch = [&](auto kernel) {
    size_t grid_size =
        compute_persistent_grid(kernel, kThreadsPerBlock, shared_memory_size, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
        routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
        use_pre_softmax, scaling_factor, grad_logits);
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  switch (score_function) {
    case 0:
      launch(fused_topk_with_score_function_backward_kernel<DataType, 0>);
      break;
    case 1:
      launch(fused_topk_with_score_function_backward_kernel<DataType, 1>);
      break;
    case 2:
      launch(fused_topk_with_score_function_backward_kernel<DataType, 2>);
      break;
    default:
      NVTE_ERROR("Unsupported score_function: " + std::to_string(score_function));
  }
}

void fused_topk_with_score_function_backward(const Tensor &routing_map,
                                             const Tensor &intermediate_output,
                                             const Tensor &grad_probs, int num_tokens,
                                             int num_experts, int topk, bool use_pre_softmax,
                                             float scaling_factor, int score_function,
                                             Tensor &grad_logits, cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_logits.data.dtype, DataType,
      fused_topk_with_score_function_backward_kernel_launcher<DataType>(
          reinterpret_cast<bool *>(routing_map.data.dptr),
          reinterpret_cast<CompType *>(intermediate_output.data.dptr),
          reinterpret_cast<DataType *>(grad_probs.data.dptr), num_tokens, num_experts, topk,
          use_pre_softmax, scaling_factor, score_function,
          reinterpret_cast<DataType *>(grad_logits.data.dptr), stream););
}

}  // namespace fused_router
}  // namespace transformer_engine

void nvte_fused_topk_with_score_function_forward(
    const NVTETensor logits, int num_tokens, int num_experts, int topk, int use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor, int score_function,
    const NVTETensor expert_bias, NVTETensor probs, NVTETensor routing_map,
    NVTETensor intermediate_output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_forward);
  using namespace transformer_engine;
  fused_router::fused_topk_with_score_function_forward(
      *convertNVTETensorCheck(logits), num_tokens, num_experts, topk,
      static_cast<bool>(use_pre_softmax), num_groups, group_topk, scaling_factor, score_function,
      *convertNVTETensorCheck(expert_bias), *convertNVTETensorCheck(probs),
      *convertNVTETensorCheck(routing_map), *convertNVTETensorCheck(intermediate_output), stream);
}

void nvte_fused_topk_with_score_function_backward(const NVTETensor routing_map,
                                                  const NVTETensor intermediate_output,
                                                  const NVTETensor grad_probs, int num_tokens,
                                                  int num_experts, int topk, int use_pre_softmax,
                                                  float scaling_factor, int score_function,
                                                  NVTETensor grad_logits, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_topk_with_score_function_backward);
  using namespace transformer_engine;
  fused_router::fused_topk_with_score_function_backward(
      *convertNVTETensorCheck(routing_map), *convertNVTETensorCheck(intermediate_output),
      *convertNVTETensorCheck(grad_probs), num_tokens, num_experts, topk,
      static_cast<bool>(use_pre_softmax), scaling_factor, score_function,
      *convertNVTETensorCheck(grad_logits), stream);
}
