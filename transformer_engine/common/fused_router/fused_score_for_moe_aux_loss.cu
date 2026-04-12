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
#include "../utils.cuh"
#include "async_loader.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_router {

template <typename DataType, TopkFuncType TopkFunc = TopkFuncType::Naive>
__global__ void fused_score_for_moe_aux_loss_forward_kernel(
    const DataType *logits, int num_tokens, int num_experts, int topk, int score_function,
    float *scores, bool *routing_map, CompType *intermediate_output, int num_buffers) {
  /***
     * Shmem layout — logits stored in original DataType for cp.async on all dtypes:
     *   [logits_raw: NB * E * W * sizeof(DataType)]  — async double-buffered
     *   [local_logits: E * W * sizeof(CompType)]      — work area (score func + topk)
     *   [topk_logits: K * W * sizeof(CompType)]
     *   [topk_indices: K * W * sizeof(int)]
     */
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ char shmem_raw_aux[];

  char *shmem_ptr = shmem_raw_aux;
  DataType *logits_shmem_base = reinterpret_cast<DataType *>(shmem_ptr);
  RawAsyncLoader<DataType> loader(logits_shmem_base, warp_id, num_experts, num_token_per_block,
                                  num_buffers);
  shmem_ptr += RawAsyncLoader<DataType>::shmem_bytes(num_experts, num_token_per_block, num_buffers);

  CompType *logits_work_buf = reinterpret_cast<CompType *>(shmem_ptr);
  shmem_ptr += num_experts * num_token_per_block * sizeof(CompType);

  CompType *topk_logits_buf = reinterpret_cast<CompType *>(shmem_ptr);
  int *topk_indices_buf = reinterpret_cast<int *>(topk_logits_buf + topk * num_token_per_block);

  CompType *local_logits = logits_work_buf + warp_id * num_experts;
  CompType *topk_logits = topk_logits_buf + warp_id * topk;
  int *topk_indices = topk_indices_buf + warp_id * topk;

  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  int first_round = blockIdx.x;
  if (first_round >= total_round) return;

  {
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

    // Async prefetch next round
    int next_round = round + gridDim.x;
    if (next_round < total_round) {
      int next_token = next_round * num_token_per_block + warp_id;
      if (next_token < num_tokens) {
        loader.start_load(logits + next_token * num_experts, num_experts, lane_id);
      }
    }

    int pos_offset = token_offset_cur_warp * num_experts;

    // Fused convert + score function + intermediate_output write
    if (score_function == 1) {  // softmax
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_logits[i] = static_cast<CompType>(raw_logits[i]);
      }
      __syncwarp();
      apply_softmax_on_float(local_logits, num_experts, lane_id);
      __syncwarp();
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        intermediate_output[pos_offset + i] = local_logits[i];
      }
    } else if (score_function == 0) {  // sigmoid
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float val = sigmoid_scalar(static_cast<CompType>(raw_logits[i]));
        intermediate_output[pos_offset + i] = val;
        local_logits[i] = val;
      }
    } else if (score_function == 2) {  // sqrtsoftplus
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float logit = static_cast<CompType>(raw_logits[i]);
        intermediate_output[pos_offset + i] = logit;
        local_logits[i] = sqrtsoftplus_scalar(logit);
      }
    }
    __syncwarp();

    // Sigmoid/Sqrtsoftplus post-processing (normalize by sum)
    if (score_function == 0 || score_function == 2) {
      auto sum_logits =
          warp_reduce_on_shmem(local_logits, num_experts, ReduceFuncType::SUM, lane_id);
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_logits[i] /= (sum_logits + epsilon);
      }
      __syncwarp();
    }

    // Topk
    topk_and_mask<TopkFunc>(local_logits, num_experts, topk, topk_indices, topk_logits, lane_id);
    __syncwarp();

    // Write results
    vec_fill_global(routing_map + pos_offset, false, num_experts, lane_id);
    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
    }
    vec_store_global(scores + pos_offset, local_logits, num_experts, lane_id);
    __syncwarp();

    loader.flip();
  }
}

template <typename DataType>
void fused_score_for_moe_aux_loss_forward_kernel_launcher(
    const DataType *logits, int num_tokens, int num_experts, int topk, int score_function,
    float *scores, bool *routing_map, CompType *intermediate_output, cudaStream_t stream) {
  // Meta data for the kernel
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t total_blocks = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  // Shmem: raw logits (DataType, double-buffered) + scores work (CompType) + topk scratch
  size_t scores_shmem = num_experts * num_token_per_block * sizeof(CompType);
  size_t scratch_shmem = topk * num_token_per_block * sizeof(CompType)
                         + topk * num_token_per_block * sizeof(int);
  size_t other_shmem = scores_shmem + scratch_shmem;
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
    auto kernel = fused_score_for_moe_aux_loss_forward_kernel<DataType, TopkFuncType::Naive>;
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shared_memory_size));
    size_t grid_size =
        compute_persistent_grid(kernel, kThreadsPerBlock, shared_memory_size, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
        logits, num_tokens, num_experts, topk, score_function, scores, routing_map,
        intermediate_output, num_buffers);
  } else {
    auto kernel = fused_score_for_moe_aux_loss_forward_kernel<DataType, TopkFuncType::Radix>;
    NVTE_CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shared_memory_size));
    size_t grid_size =
        compute_persistent_grid(kernel, kThreadsPerBlock, shared_memory_size, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
        logits, num_tokens, num_experts, topk, score_function, scores, routing_map,
        intermediate_output, num_buffers);
  }
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void fused_score_for_moe_aux_loss_forward(const Tensor &logits, int num_tokens, int num_experts,
                                          int topk, int score_function, Tensor &scores,
                                          Tensor &routing_map, Tensor &intermediate_output,
                                          cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      logits.data.dtype, DataType,
      fused_score_for_moe_aux_loss_forward_kernel_launcher<DataType>(
          reinterpret_cast<DataType *>(logits.data.dptr), num_tokens, num_experts, topk,
          score_function, reinterpret_cast<float *>(scores.data.dptr),
          reinterpret_cast<bool *>(routing_map.data.dptr),
          reinterpret_cast<CompType *>(intermediate_output.data.dptr), stream););
}

template <typename DataType, int ScoreFunc>
__global__ void fused_score_for_moe_aux_loss_backward_kernel(const CompType *intermediate_output,
                                                              const float *grad_scores,
                                                              int num_tokens, int num_experts,
                                                              int topk,
                                                              DataType *grad_logits) {
  // Streaming 2-pass backward, no shmem. score_function templated to eliminate dead code.
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;

  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_idx = round * num_token_per_block + warp_id;
    if (token_idx >= num_tokens) break;
    int pos = token_idx * num_experts;

    // ---- Pass 1: Compute reduction sums ----
    CompType sum_act = 0.0f;
    CompType sum_grad_act = 0.0f;
    CompType sum_output_x_grad = 0.0f;

    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      CompType g = static_cast<CompType>(grad_scores[pos + i]);
      CompType act = intermediate_output[pos + i];

      if constexpr (ScoreFunc == 0) {
        sum_act += act;
        sum_grad_act += g * act;
      } else if constexpr (ScoreFunc == 2) {
        CompType act_val = sqrtsoftplus_scalar(act);
        sum_act += act_val;
        sum_grad_act += g * act_val;
      } else if constexpr (ScoreFunc == 1) {
        sum_output_x_grad += g * act;
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

    // ---- Pass 2: Apply backward + write ----
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      CompType g = static_cast<CompType>(grad_scores[pos + i]);
      CompType act = intermediate_output[pos + i];

      if constexpr (ScoreFunc == 0) {
        CompType denom = sum_act + epsilon;
        g = g / denom - sum_grad_act / (denom * denom);
        g = g * act * (1.0f - act);
      } else if constexpr (ScoreFunc == 2) {
        CompType act_val = sqrtsoftplus_scalar(act);
        CompType denom = sum_act + epsilon;
        g = g / denom - sum_grad_act / (denom * denom);
        CompType dy_dx = (act > 20.0f) ? (1.0f / (2.0f * act_val + epsilon))
                                       : (sigmoid_scalar(act) / (2.0f * act_val + epsilon));
        g = g * dy_dx;
      } else if constexpr (ScoreFunc == 1) {
        g = act * (g - sum_output_x_grad);
      }

      grad_logits[pos + i] = static_cast<DataType>(g);
    }
  }
}

template <typename DataType>
void fused_score_for_moe_aux_loss_backward_kernel_launcher(
    const CompType *intermediate_output, const float *grad_scores, int num_tokens, int num_experts,
    int topk, int score_function, DataType *grad_logits, cudaStream_t stream) {
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t total_blocks = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = 0;

  auto launch = [&](auto kernel) {
    size_t grid_size =
        compute_persistent_grid(kernel, kThreadsPerBlock, shared_memory_size, total_blocks);
    kernel<<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
        intermediate_output, grad_scores, num_tokens, num_experts, topk, grad_logits);
    NVTE_CHECK_CUDA(cudaGetLastError());
  };

  switch (score_function) {
    case 0:
      launch(fused_score_for_moe_aux_loss_backward_kernel<DataType, 0>);
      break;
    case 1:
      launch(fused_score_for_moe_aux_loss_backward_kernel<DataType, 1>);
      break;
    case 2:
      launch(fused_score_for_moe_aux_loss_backward_kernel<DataType, 2>);
      break;
    default:
      NVTE_ERROR("Unsupported score_function: " + std::to_string(score_function));
  }
}

void fused_score_for_moe_aux_loss_backward(const Tensor &intermediate_output,
                                           const Tensor &grad_scores, int num_tokens,
                                           int num_experts, int topk, int score_function,
                                           Tensor &grad_logits, cudaStream_t stream) {
  TE_ROUTER_PROBS_TYPE_SWITCH_ALL(
      grad_logits.data.dtype, DataType,
      fused_score_for_moe_aux_loss_backward_kernel_launcher<DataType>(
          reinterpret_cast<CompType *>(intermediate_output.data.dptr),
          reinterpret_cast<float *>(grad_scores.data.dptr), num_tokens, num_experts, topk,
          score_function, reinterpret_cast<DataType *>(grad_logits.data.dptr), stream););
}

}  // namespace fused_router
}  // namespace transformer_engine

void nvte_fused_score_for_moe_aux_loss_forward(const NVTETensor logits, int num_tokens,
                                               int num_experts, int topk, int score_function,
                                               NVTETensor scores, const NVTETensor routing_map,
                                               const NVTETensor intermediate_output,
                                               cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_score_for_moe_aux_loss_forward);
  using namespace transformer_engine;
  fused_router::fused_score_for_moe_aux_loss_forward(
      *convertNVTETensorCheck(logits), num_tokens, num_experts, topk, score_function,
      *convertNVTETensorCheck(scores), *convertNVTETensorCheck(routing_map),
      *convertNVTETensorCheck(intermediate_output), stream);
}

void nvte_fused_score_for_moe_aux_loss_backward(const NVTETensor intermediate_output,
                                                const NVTETensor grad_scores, int num_tokens,
                                                int num_experts, int topk, int score_function,
                                                NVTETensor grad_logits, cudaStream_t stream) {
  NVTE_API_CALL(nvte_fused_score_for_moe_aux_loss_backward);
  using namespace transformer_engine;
  fused_router::fused_score_for_moe_aux_loss_backward(
      *convertNVTETensorCheck(intermediate_output), *convertNVTETensorCheck(grad_scores),
      num_tokens, num_experts, topk, score_function, *convertNVTETensorCheck(grad_logits), stream);
}
