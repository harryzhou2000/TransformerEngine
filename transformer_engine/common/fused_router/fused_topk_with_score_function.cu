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

// Forward: logits → score_function → topk → normalize → probs + routing_map.
//
// Template params:
//   DataType  — logits/probs element type (bf16/fp16/fp32)
//   BiasType  — expert_bias element type
//   TopkFunc  — Naive (K<16) or Radix (K>=16) selection
//   ScoreFunc — 0=sigmoid, 1=softmax, 2=sqrtsoftplus
//
// Shmem layout (W = warps/block, B = num_buffers):
//   logits_raw:    B × E × W × sizeof(DataType)  — double-buffered cp.async
//   scores:        E × W × sizeof(CompType)       — work area
//   topk_scratch:  K × W × sizeof(CompType) + K × W × sizeof(int)
//   (+ group_topk scratch if group_topk > 0)
template <typename DataType, typename BiasType, TopkFuncType TopkFunc = TopkFuncType::Naive,
          int ScoreFunc = 0>
__global__ void fused_topk_with_score_function_forward_kernel(
    const DataType *logits, int num_tokens, int num_experts, int topk, bool use_pre_softmax,
    int num_groups, int group_topk, float scaling_factor,
    const BiasType *expert_bias, DataType *probs, bool *routing_map, CompType *intermediate_output,
    int num_buffers) {
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ char shmem_raw[];

  // Shmem setup
  char *shmem_ptr = shmem_raw;
  DataType *logits_shmem_base = reinterpret_cast<DataType *>(shmem_ptr);
  RawAsyncLoader<DataType> loader(logits_shmem_base, warp_id, num_experts, num_token_per_block,
                                  num_buffers);
  shmem_ptr += RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);

  CompType *scores_buf = reinterpret_cast<CompType *>(shmem_ptr);
  shmem_ptr += num_experts * num_token_per_block * sizeof(CompType);

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

  CompType *scores = scores_buf + warp_id * num_experts;
  CompType *topk_scores = topk_scores_buf + warp_id * topk;
  CompType *masked_scores =
      (masked_scores_buf != nullptr) ? masked_scores_buf + warp_id * num_experts : nullptr;
  CompType *group_scores =
      (group_scores_buf != nullptr) ? group_scores_buf + warp_id * num_groups : nullptr;
  int *topk_indices = topk_indices_buf + warp_id * topk;

  // Main loop — persistent grid with double-buffered async load
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  int first_round = blockIdx.x;
  if (first_round >= total_round) return;

  {  // First async load
    int first_token = first_round * num_token_per_block + warp_id;
    if (first_token < num_tokens) {
      loader.load_current(logits + first_token * num_experts, num_experts, lane_id);
    }
  }

  for (int round = first_round; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    if (token_offset_cur_warp >= num_tokens) break;

    loader.wait();
    DataType *raw_logits = loader.current_buf();

    // Prefetch next round (overlaps with compute below)
    int next_round = round + gridDim.x;
    if (next_round < total_round) {
      int next_token = next_round * num_token_per_block + warp_id;
      if (next_token < num_tokens) {
        loader.start_load(logits + next_token * num_experts, num_experts, lane_id);
      }
    }

    int pos_offset = token_offset_cur_warp * num_experts;

    vec_fill_global(probs + pos_offset, static_cast<DataType>(0), num_experts, lane_id);
    vec_fill_global(routing_map + pos_offset, false, num_experts, lane_id);

    // Preprocess: convert DataType→CompType, apply score function, save intermediate
    if constexpr (ScoreFunc == 1) {
      if (use_pre_softmax) {
        // Pre-softmax: convert, softmax, save
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          scores[i] = static_cast<CompType>(raw_logits[i]);
        }
        __syncwarp();
        apply_softmax_on_float(scores, num_experts, lane_id);
        __syncwarp();
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          intermediate_output[pos_offset + i] = scores[i];
        }
      } else {
        // Post-softmax: convert logits to scores, init intermediate_output to -inf
        for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
          scores[i] = static_cast<CompType>(raw_logits[i]);
          intermediate_output[pos_offset + i] = -std::numeric_limits<CompType>::infinity();
        }
      }
    } else if constexpr (ScoreFunc == 0) {
      // Sigmoid: fused convert + apply + save + bias
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float val = sigmoid_scalar(static_cast<CompType>(raw_logits[i]));
        intermediate_output[pos_offset + i] = val;
        if (expert_bias) val += static_cast<CompType>(expert_bias[i]);
        scores[i] = val;
      }
    } else if constexpr (ScoreFunc == 2) {
      // Sqrtsoftplus: fused convert + save-logits + apply + bias
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float logit = static_cast<CompType>(raw_logits[i]);
        intermediate_output[pos_offset + i] = logit;
        float val = sqrtsoftplus_scalar(logit);
        if (expert_bias) val += static_cast<CompType>(expert_bias[i]);
        scores[i] = val;
      }
    }
    __syncwarp();

    if (group_topk > 0) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        masked_scores[i] = -std::numeric_limits<CompType>::infinity();
      }
      __syncwarp();
    }

    // Topk selection
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

    // Postprocess: revert bias, normalize, write output
    if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
      if (expert_bias) {
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          topk_scores[i] = topk_scores[i] - static_cast<CompType>(expert_bias[topk_indices[i]]);
        }
        __syncwarp();
      }
    }

    if constexpr (ScoreFunc == 1) {
      if (!use_pre_softmax) {
        apply_softmax_on_float(topk_scores, topk, lane_id);
        __syncwarp();
        // Save softmax output for backward (scatter — topk positions only)
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          intermediate_output[pos_offset + topk_indices[i]] = topk_scores[i];
        }
        __syncwarp();
      }
    }

    if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
      if (topk > 1) {
        CompType sum_scores = warp_reduce_on_shmem(topk_scores, topk, ReduceFuncType::SUM, lane_id);
        for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
          topk_scores[i] = topk_scores[i] / (sum_scores + epsilon);
        }
      }
      __syncwarp();
    }

    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
      probs[pos_offset + topk_indices[i]] = scaling_factor * topk_scores[i];
    }
    __syncwarp();

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
  size_t scores_shmem = num_experts * num_token_per_block * sizeof(CompType);
  size_t scratch_shmem = topk * num_token_per_block * sizeof(CompType)
                         + topk * num_token_per_block * sizeof(int);
  if (group_topk > 0) {
    scratch_shmem += num_groups * num_token_per_block * sizeof(CompType);
    scratch_shmem += num_experts * num_token_per_block * sizeof(CompType);
  }
  size_t other_shmem = scores_shmem + scratch_shmem;
  size_t logits_single_buf =
      RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, 1);
  int num_buffers = choose_num_buffers(logits_single_buf, other_shmem);
  size_t logits_raw_shmem =
      RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);
  size_t shared_memory_size = logits_raw_shmem + other_shmem;
  check_shared_memory_capacity_num_experts(shared_memory_size, num_experts);

  auto launch = [&](auto kernel) {
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shared_memory_size));
    size_t grid_size =
        compute_persistent_grid(kernel, kThreadsPerBlock, shared_memory_size, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
        logits, num_tokens, num_experts, topk, use_pre_softmax, num_groups, group_topk,
        scaling_factor, expert_bias, probs, routing_map, intermediate_output,
        num_buffers);
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  if (topk < 16) {
    switch (score_function) {
      case 0:
        launch(fused_topk_with_score_function_forward_kernel<DataType, BiasType,
                                                              TopkFuncType::Naive, 0>);
        break;
      case 1:
        launch(fused_topk_with_score_function_forward_kernel<DataType, BiasType,
                                                              TopkFuncType::Naive, 1>);
        break;
      case 2:
        launch(fused_topk_with_score_function_forward_kernel<DataType, BiasType,
                                                              TopkFuncType::Naive, 2>);
        break;
      default:
        NVTE_ERROR("Unsupported score_function: " + std::to_string(score_function));
    }
  } else {
    switch (score_function) {
      case 0:
        launch(fused_topk_with_score_function_forward_kernel<DataType, BiasType,
                                                              TopkFuncType::Radix, 0>);
        break;
      case 1:
        launch(fused_topk_with_score_function_forward_kernel<DataType, BiasType,
                                                              TopkFuncType::Radix, 1>);
        break;
      case 2:
        launch(fused_topk_with_score_function_forward_kernel<DataType, BiasType,
                                                              TopkFuncType::Radix, 2>);
        break;
      default:
        NVTE_ERROR("Unsupported score_function: " + std::to_string(score_function));
    }
  }
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

// Backward: grad_probs + intermediate_output + routing_map → grad_logits.
//
// Double-buffered cp.async loads all 3 inputs in original types.  Compute
// happens in registers: reduction pass (warp shuffle) then element-wise pass.
//
// Template params:
//   DataType  — grad_probs/grad_logits element type (bf16/fp16/fp32)
//   ScoreFunc — 0=sigmoid, 1=softmax, 2=sqrtsoftplus
//
// Shmem layout (B = 2, W = warps/block):
//   grad_raw:  B × E × W × sizeof(DataType)
//   act_buf:   B × E × W × sizeof(CompType)
//   mask_buf:  B × E × W × sizeof(bool)
constexpr int kBwdNumBuffers = 2;

template <typename DataType, int ScoreFunc>
__global__ void fused_topk_with_score_function_backward_kernel(
    const bool *routing_map, const CompType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    DataType *grad_logits) {
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;

  extern __shared__ char shmem_bwd[];
  char *shmem_ptr = shmem_bwd;

  DataType *grad_shmem_base = reinterpret_cast<DataType *>(shmem_ptr);
  RawAsyncLoader<DataType> grad_loader(grad_shmem_base, warp_id, num_experts,
                                       num_token_per_block, kBwdNumBuffers);
  shmem_ptr += RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block,
                                                     kBwdNumBuffers);

  CompType *act_shmem_base = reinterpret_cast<CompType *>(shmem_ptr);
  RawAsyncLoader<CompType> act_loader(act_shmem_base, warp_id, num_experts,
                                      num_token_per_block, kBwdNumBuffers);
  shmem_ptr += RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block,
                                                     kBwdNumBuffers);

  bool *mask_shmem_base = reinterpret_cast<bool *>(shmem_ptr);
  RawAsyncLoader<bool> mask_loader(mask_shmem_base, warp_id, num_experts,
                                   num_token_per_block, kBwdNumBuffers);

  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  int first_round = blockIdx.x;
  if (first_round >= total_round) return;

  {  // First async load
    int first_token = first_round * num_token_per_block + warp_id;
    if (first_token < num_tokens) {
      int pos = first_token * num_experts;
      grad_loader.load_current(grad_probs + pos, num_experts, lane_id);
      act_loader.load_current(intermediate_output + pos, num_experts, lane_id);
      mask_loader.load_current(routing_map + pos, num_experts, lane_id);
    }
  }

  for (int round = first_round; round < total_round; round += gridDim.x) {
    int token_idx = round * num_token_per_block + warp_id;
    if (token_idx >= num_tokens) break;
    int pos = token_idx * num_experts;

    grad_loader.wait();
    act_loader.wait();
    mask_loader.wait();

    DataType *raw_grad = grad_loader.current_buf();
    CompType *raw_act = act_loader.current_buf();
    bool *raw_mask = mask_loader.current_buf();

    // Prefetch next round
    int next_round = round + gridDim.x;
    if (next_round < total_round) {
      int next_token = next_round * num_token_per_block + warp_id;
      if (next_token < num_tokens) {
        int next_pos = next_token * num_experts;
        grad_loader.start_load(grad_probs + next_pos, num_experts, lane_id);
        act_loader.start_load(intermediate_output + next_pos, num_experts, lane_id);
        mask_loader.start_load(routing_map + next_pos, num_experts, lane_id);
      }
    }

    // Reduction pass
    CompType sum_act = 0.0f;
    CompType sum_grad_act = 0.0f;
    CompType sum_output_x_grad = 0.0f;

    bool need_reduce = ((ScoreFunc == 0 || ScoreFunc == 2) && topk > 1) || (ScoreFunc == 1);

    if (need_reduce) {
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        CompType g = static_cast<CompType>(raw_grad[i]) * scaling_factor;
        CompType act = raw_act[i];
        bool routed = raw_mask[i];

        if constexpr (ScoreFunc == 0) {
          if (routed) { sum_act += act; sum_grad_act += g * act; }
        } else if constexpr (ScoreFunc == 2) {
          if (routed) {
            CompType v = sqrtsoftplus_scalar(act);
            sum_act += v; sum_grad_act += g * v;
          }
        } else if constexpr (ScoreFunc == 1) {
          if (!use_pre_softmax) {
            if (routed) sum_output_x_grad += g * act;
          } else {
            sum_output_x_grad += (routed ? g : 0.0f) * act;
          }
        }
      }
      if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
        sum_act = warp_allreduce_sum(sum_act);
        sum_grad_act = warp_allreduce_sum(sum_grad_act);
      }
      if constexpr (ScoreFunc == 1) {
        sum_output_x_grad = warp_allreduce_sum(sum_output_x_grad);
      }
    }

    // Element-wise pass
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      CompType g = static_cast<CompType>(raw_grad[i]) * scaling_factor;
      CompType act = raw_act[i];
      bool routed = raw_mask[i];

      // Normalization backward (sigmoid/sqrtsoftplus with topk > 1)
      if constexpr (ScoreFunc == 0 || ScoreFunc == 2) {
        if (topk > 1) {
          g = normalize_bwd_scalar(g, routed, sum_act, sum_grad_act);
        }
      }

      // Post-softmax backward (routed subset only)
      if constexpr (ScoreFunc == 1) {
        if (!use_pre_softmax) {
          g = routed ? softmax_bwd_scalar(g, act, sum_output_x_grad) : 0.0f;
        }
      }

      // Topk mask: zero non-routed
      if (!routed) g = 0.0f;

      // Pre-softmax backward (all experts)
      if constexpr (ScoreFunc == 1) {
        if (use_pre_softmax) {
          g = softmax_bwd_scalar(g, act, sum_output_x_grad);
        }
      }

      // Activation backward
      if constexpr (ScoreFunc == 0) {
        g = sigmoid_bwd_scalar(g, act);
      } else if constexpr (ScoreFunc == 2) {
        g = sqrtsoftplus_bwd_scalar(g, act, sqrtsoftplus_scalar(act));
      }

      grad_logits[pos + i] = static_cast<DataType>(g);
    }

    grad_loader.flip();
    act_loader.flip();
    mask_loader.flip();
  }
}

template <typename DataType>
void fused_topk_with_score_function_backward_kernel_launcher(
    const bool *routing_map, const CompType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, DataType *grad_logits, cudaStream_t stream) {
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t total_blocks = (num_tokens + num_token_per_block - 1) / num_token_per_block;

  size_t shmem_bytes =
      RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, kBwdNumBuffers) +
      RawAsyncLoader<CompType>::shmem_bytes(num_experts, num_token_per_block, kBwdNumBuffers) +
      RawAsyncLoader<bool>::shmem_bytes(num_experts, num_token_per_block, kBwdNumBuffers);
  check_shared_memory_capacity_num_experts(shmem_bytes, num_experts);

  auto launch = [&](auto kernel) {
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_bytes));
    size_t grid_size =
        compute_persistent_grid(kernel, kThreadsPerBlock, shmem_bytes, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, shmem_bytes, stream>>>(
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
