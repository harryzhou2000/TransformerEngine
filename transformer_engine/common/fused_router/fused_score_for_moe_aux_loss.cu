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
     * Section: Global Variables/Addresses init
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     *
     * Shared memory layout (1x or 2x buffered logits):
     *   [logits_buf: num_buffers * E * W]  -- 1 or 2 buffers
     *   [topk_logits_buf: K * W]
     *   [topk_indices_buf: K * W]          -- (as int)
     */
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem_scores_for_aux_loss[];

  // --- Scores region (1x or 2x buffered) ---
  CompType *logits_db_base = reinterpret_cast<CompType *>(shmem_scores_for_aux_loss);
  WarpAsyncLoader loader(logits_db_base, warp_id, num_experts, num_token_per_block, num_buffers);

  // --- Scratch arrays (after the scores region) ---
  CompType *scratch_base = logits_db_base + num_buffers * num_experts * num_token_per_block;
  CompType *topk_logits_buf = scratch_base;
  int *topk_indices_buf = reinterpret_cast<int *>(topk_logits_buf + topk * num_token_per_block);
  // Per-warp pointers
  CompType *topk_logits = topk_logits_buf + warp_id * topk;
  int *topk_indices = topk_indices_buf + warp_id * topk;

  /***
     * Section: Main Loop with async load (double-buffered when shmem permits)
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  int first_round = blockIdx.x;
  if (first_round >= total_round) return;

  // Kick off first load
  {
    int first_token = first_round * num_token_per_block + warp_id;
    if (first_token < num_tokens) {
      loader.load_current<DataType>(logits + first_token * num_experts, num_experts, lane_id);
    }
  }

  for (int round = first_round; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    if (token_offset_cur_warp >= num_tokens) break;

    // --- Wait for current buffer to be ready ---
    loader.wait(lane_id);
    CompType *local_logits = loader.current_buf();

    // --- Prefetch next round into the other buffer ---
    int next_round = round + gridDim.x;
    if (next_round < total_round) {
      int next_token = next_round * num_token_per_block + warp_id;
      if (next_token < num_tokens) {
        loader.start_load<DataType>(logits + next_token * num_experts, num_experts, lane_id);
      }
    }

    int pos_offset = token_offset_cur_warp * num_experts;

    /***
         * Section: Fused Preprocess
         * Apply score function in-place on shmem local_logits, and write intermediate_output
         * to global memory in the same pass (for sigmoid and sqrtsoftplus).
         * For softmax, fuse the intermediate_output write into the normalize pass.
         */
    if (score_function == 1) {  // softmax
      apply_softmax_on_float(local_logits, num_experts, lane_id);
      __syncwarp();
      // Fused: save softmax output to global
      vec_store_global(intermediate_output + pos_offset, local_logits, num_experts, lane_id);
    } else if (score_function == 0) {  // sigmoid: fused apply + save
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float val = sigmoid_scalar(local_logits[i]);
        intermediate_output[pos_offset + i] = val;
        local_logits[i] = val;
      }
    } else if (score_function == 2) {  // sqrtsoftplus: fused save-logits + apply
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float logit = local_logits[i];
        intermediate_output[pos_offset + i] = logit;  // Save original for backward
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

    /***
         * Section: Topk
         */
    topk_and_mask<TopkFunc>(local_logits, num_experts, topk, topk_indices, topk_logits, lane_id);
    __syncwarp();

    // Write the routing_map (zero-init + scatter) and scores (dense) to global
    vec_fill_global(routing_map + pos_offset, false, num_experts, lane_id);
    for (int i = lane_id; i < topk; i += kThreadsPerWarp) {
      routing_map[pos_offset + topk_indices[i]] = true;
    }
    vec_store_global(scores + pos_offset, local_logits, num_experts, lane_id);
    __syncwarp();

    // Flip double buffer for next round
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
  // Compute scratch size (everything except the scores buffer)
  size_t scratch_shmem = topk * num_token_per_block * sizeof(CompType)  // topk_logits
                         + topk * num_token_per_block * sizeof(int);    // topk_indices
  // Decide single vs double buffer based on shmem budget
  int num_buffers =
      WarpAsyncLoader::choose_num_buffers(num_experts, num_token_per_block, scratch_shmem);
  size_t shared_memory_size =
      WarpAsyncLoader::scores_shmem_bytes_n(num_experts, num_token_per_block, num_buffers) +
      scratch_shmem;
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

template <typename DataType>
__global__ void fused_score_for_moe_aux_loss_backward_kernel(const CompType *intermediate_output,
                                                             const float *grad_scores,
                                                             int num_tokens, int num_experts,
                                                             int topk, int score_function,
                                                             DataType *grad_logits) {
  /***
     * Section: Global Variables/Addresses init
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     */
  // Used variables/addresses init
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem[];
  CompType *grad_scores_buf = reinterpret_cast<CompType *>(shmem);
  // To store the output of softmax/sigmoid from fwd, or original logits for sqrtsoftplus
  CompType *act_from_fwd_buf = grad_scores_buf + num_experts * num_token_per_block;
  CompType *comp_buf = act_from_fwd_buf + num_experts * num_token_per_block;
  // The address of buffers on the current warp
  CompType *local_grad = grad_scores_buf + warp_id * num_experts;
  CompType *local_act_from_fwd = act_from_fwd_buf + warp_id * num_experts;
  CompType *local_comp_buf = comp_buf + warp_id * num_experts;

  /***
     * Section: Main Loop
     * - Each warp is responsible for one token
     */
  int total_round = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  for (int round = blockIdx.x; round < total_round; round += gridDim.x) {
    int token_offset_cur_warp = round * num_token_per_block + warp_id;
    // Each warp is responsible for one token
    if (token_offset_cur_warp >= num_tokens) break;

    /***
         * Section: Init buffer
         * - Clear the global buffer which will accept the result of this round
         * - Clear/Init the shmem buffer used by current warp this round
         * - Load the dgrad/output_from_fwd to shmem
         */
    int pos_offset = token_offset_cur_warp * num_experts;
    // Load the dgrad/output_from_fwd to shmem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      local_grad[i] = grad_scores[pos_offset + i];
      local_act_from_fwd[i] = intermediate_output[pos_offset + i];
    }
    __threadfence_block();
    __syncwarp();

    /***
         * Section: Backward of ops before the topk
         * - Pre-softmax bwd
         * - Sigmoid/Sqrtsoftplus Post-processing bwd when topk > 1
         * - Sigmoid bwd
         * - Sqrtsoftplus bwd
         * - Write the grad_logits to the global mem
         */
    // Sqrtsoftplus: First compute sqrtsoftplus output from original logits
    // (needed for both post-processing bwd and activation bwd, compute once here)
    // For sqrtsoftplus, intermediate_output stores original logits
    if (score_function == 2) {
      // Copy original logits to local_comp_buf and apply sqrtsoftplus in-place
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_comp_buf[i] = local_act_from_fwd[i];
      }
      __syncwarp();
      apply_sqrtsoftplus_on_float(local_comp_buf, num_experts, lane_id);
      __syncwarp();
    }

    // Sigmoid/Sqrtsoftplus Post-processing bwd (normalization backward)
    if (score_function == 0 || score_function == 2) {
      // Select the correct activation output buffer:
      // - Sigmoid: local_act_from_fwd already contains sigmoid output
      // - Sqrtsoftplus: local_comp_buf contains sqrtsoftplus output computed above
      CompType *act_output = (score_function == 0) ? local_act_from_fwd : local_comp_buf;

      auto sum_fwd_input =
          warp_reduce_on_shmem(act_output, num_experts, ReduceFuncType::SUM, lane_id);
      // Compute sum of output * grad using registers
      CompType local_sum_Output_x_Grad = 0.0;
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_sum_Output_x_Grad += local_grad[i] * act_output[i];
      }
      // Warp reduce the sum
      for (int s = 16; s > 0; s /= 2) {
        local_sum_Output_x_Grad += __shfl_xor_sync(0xffffffff, local_sum_Output_x_Grad, s);
      }
      CompType sum_Output_x_Grad = local_sum_Output_x_Grad;
      // In-place update
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        local_grad[i] = local_grad[i] / (sum_fwd_input + epsilon) -
                        sum_Output_x_Grad / ((sum_fwd_input + epsilon) * (sum_fwd_input + epsilon));
      }
      __syncwarp();
    }

    // Pre-softmax bwd
    if (score_function == 1) {
      apply_softmax_bwd_on_float(local_grad, local_act_from_fwd, local_comp_buf, nullptr,
                                 num_experts, lane_id);
      __syncwarp();
    }
    // Sigmoid bwd
    if (score_function == 0) {
      apply_sigmoid_bwd_on_float(local_grad, local_act_from_fwd, num_experts, lane_id);
      __syncwarp();
    }
    // Sqrtsoftplus bwd
    // For sqrtsoftplus, local_comp_buf already contains sqrtsoftplus output computed earlier
    // Now compute gradient: dy/dx = sigmoid(x) / (2 * y)
    if (score_function == 2) {
      apply_sqrtsoftplus_bwd_on_float(local_grad, local_comp_buf, local_act_from_fwd, num_experts,
                                      lane_id);
      __syncwarp();
    }
    // Write the grad_logits to the global mem
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      grad_logits[pos_offset + i] = static_cast<DataType>(local_grad[i]);
    }
    __syncwarp();
  }
}

template <typename DataType>
void fused_score_for_moe_aux_loss_backward_kernel_launcher(
    const CompType *intermediate_output, const float *grad_scores, int num_tokens, int num_experts,
    int topk, int score_function, DataType *grad_logits, cudaStream_t stream) {
  // Meta data for the kernel
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(CompType)  // grad_scores
                              +
                              num_experts * num_token_per_block * sizeof(CompType)  // act_from_fwd
                              + num_experts * num_token_per_block * sizeof(CompType);  // comp_buf
  check_shared_memory_capacity_num_experts(shared_memory_size, num_experts);
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(fused_score_for_moe_aux_loss_backward_kernel<DataType>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       shared_memory_size));
  fused_score_for_moe_aux_loss_backward_kernel<DataType>
      <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
          intermediate_output, grad_scores, num_tokens, num_experts, topk, score_function,
          grad_logits);
  NVTE_CHECK_CUDA(cudaGetLastError());
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
