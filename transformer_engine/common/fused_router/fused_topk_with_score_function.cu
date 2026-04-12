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
     * - Each warp is responsible for one token, and has own shared memory buffer.
     *   Then __syncwarp() is used instead of __syncthreads()
     *
     * Shared memory layout (1x or 2x buffered scores):
     *   [scores_buf: num_buffers * E * W]  -- 1 or 2 buffers for logits/scores
     *   [topk_scores_buf: K * W]
     *   [topk_indices_buf: K * W]          -- (as int)
     *   (if group_topk > 0:)
     *     [masked_scores_buf: E * W]
     *     [group_scores_buf: G * W]
     *
     * where E = num_experts, K = topk, W = num_token_per_block (warps),
     *       G = num_groups.
     */
  int num_token_per_block = blockDim.x / kThreadsPerWarp;
  int warp_id = threadIdx.x / kThreadsPerWarp;
  int lane_id = threadIdx.x % kThreadsPerWarp;
  extern __shared__ float shmem[];

  // --- Scores region (1x or 2x buffered) ---
  CompType *scores_db_base = reinterpret_cast<CompType *>(shmem);
  WarpAsyncLoader loader(scores_db_base, warp_id, num_experts, num_token_per_block, num_buffers);

  // --- Scratch arrays (after the scores region) ---
  CompType *scratch_base = scores_db_base + num_buffers * num_experts * num_token_per_block;
  CompType *topk_scores_buf = scratch_base;
  CompType *group_scores_buf = nullptr, *masked_scores_buf = nullptr;
  int *topk_indices_buf = nullptr;
  if (group_topk > 0) {
    masked_scores_buf = topk_scores_buf + topk * num_token_per_block;
    group_scores_buf = masked_scores_buf + num_experts * num_token_per_block;
    topk_indices_buf = reinterpret_cast<int *>(group_scores_buf + num_groups * num_token_per_block);
  } else {
    topk_indices_buf = reinterpret_cast<int *>(topk_scores_buf + topk * num_token_per_block);
  }
  // Per-warp pointers into scratch
  CompType *topk_scores = topk_scores_buf + warp_id * topk;
  CompType *masked_scores =
      (masked_scores_buf != nullptr) ? masked_scores_buf + warp_id * num_experts : nullptr;
  CompType *group_scores =
      (group_scores_buf != nullptr) ? group_scores_buf + warp_id * num_groups : nullptr;
  int *topk_indices = topk_indices_buf + warp_id * topk;

  /***
     * Section: Main Loop with double-buffered async load
     *
     * Structure:
     *   1. Kick off async load for the first round into current_buf
     *   2. For each round:
     *      a. Wait for async load to complete
     *      b. Get pointer to current scores buffer
     *      c. If there is a next round, kick off async load into next_buf
     *      d. Fused preprocess: apply score function + write intermediate_output
     *      e. Topk selection
     *      f. Fused postprocess + dense global store (probs, routing_map)
     *      g. Flip buffers
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
    CompType *scores = loader.current_buf();

    // --- Prefetch next round into the other buffer ---
    int next_round = round + gridDim.x;
    if (next_round < total_round) {
      int next_token = next_round * num_token_per_block + warp_id;
      if (next_token < num_tokens) {
        loader.start_load<DataType>(logits + next_token * num_experts, num_experts, lane_id);
      }
    }

    int pos_offset = token_offset_cur_warp * num_experts;

    // Clear the probs/routing_map for this token (will be scatter-written after topk)
    vec_fill_global(probs + pos_offset, static_cast<DataType>(0), num_experts, lane_id);
    vec_fill_global(routing_map + pos_offset, false, num_experts, lane_id);

    /***
         * Section: Fused Preprocess
         * Apply score function in-place on shmem scores, and write intermediate_output
         * to global memory in the same pass (for sigmoid and sqrtsoftplus).
         * For softmax (which is not element-wise), we still need the 2-pass softmax
         * on shmem but fuse the intermediate_output write into the normalize pass.
         */
    if (use_pre_softmax && score_function == 1) {
      // Softmax: 2-pass (cannot fuse element-wise, but the load is already async)
      apply_softmax_on_float(scores, num_experts, lane_id);
      __syncwarp();
      // Fused: save softmax output to global intermediate_output
      vec_store_global(intermediate_output + pos_offset, scores, num_experts, lane_id);
    } else if (score_function == 0) {
      // Sigmoid: fused apply + save + bias in one pass
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float val = sigmoid_scalar(scores[i]);
        intermediate_output[pos_offset + i] = val;  // Save sigmoid output for backward
        if (expert_bias) {
          val += static_cast<CompType>(expert_bias[i]);
        }
        scores[i] = val;
      }
    } else if (score_function == 2) {
      // Sqrtsoftplus: fused save-logits + apply + bias in one pass
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        float logit = scores[i];
        intermediate_output[pos_offset + i] = logit;  // Save original logits for backward
        float val = sqrtsoftplus_scalar(logit);
        if (expert_bias) {
          val += static_cast<CompType>(expert_bias[i]);
        }
        scores[i] = val;
      }
    } else if (!use_pre_softmax && score_function == 1) {
      // Post-softmax: logits stay as-is for topk, intermediate_output written later
      // Init intermediate_output to -inf for backward
      vec_fill_global(intermediate_output + pos_offset, -std::numeric_limits<CompType>::infinity(),
                      num_experts, lane_id);
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
  // Compute scratch size (everything except the scores buffer)
  size_t scratch_shmem = topk * num_token_per_block * sizeof(CompType)  // topk_scores
                         + topk * num_token_per_block * sizeof(int);    // topk_indices
  if (group_topk > 0) {
    scratch_shmem += num_groups * num_token_per_block * sizeof(CompType);   // group_scores
    scratch_shmem += num_experts * num_token_per_block * sizeof(CompType);  // masked_scores
  }
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

template <typename DataType>
__global__ void fused_topk_with_score_function_backward_kernel(
    // Inputs tensor
    const bool *routing_map, const CompType *intermediate_output, const DataType *grad_probs,
    // Other parameters
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function,
    // Output tensor
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
  CompType *grad_probs_buf = reinterpret_cast<CompType *>(shmem);
  // To store the output of softmax/sigmoid from fwd, or original logits for sqrtsoftplus
  CompType *act_from_fwd_buf = grad_probs_buf + num_experts * num_token_per_block;
  CompType *comp_buf = act_from_fwd_buf + num_experts * num_token_per_block;
  // To store the routing_map from the fwd
  bool *routing_map_buf = reinterpret_cast<bool *>(comp_buf + num_experts * num_token_per_block);
  // The address of buffers on the current warp
  CompType *local_grad = grad_probs_buf + warp_id * num_experts;
  CompType *local_act_from_fwd = act_from_fwd_buf + warp_id * num_experts;
  CompType *local_comp_buf = comp_buf + warp_id * num_experts;
  bool *local_routing_map = routing_map_buf + warp_id * num_experts;

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
      local_grad[i] = grad_probs[pos_offset + i];
      local_act_from_fwd[i] = intermediate_output[pos_offset + i];
      local_routing_map[i] = routing_map[pos_offset + i];
    }
    __threadfence_block();
    __syncwarp();

    /***
         * Section: Backward of ops after the topk
         * - Backward of the used scaling_factor
         * - Sigmoid/Sqrtsoftplus Post-processing bwd when topk > 1
         * - Softmax bwd if use_pre_softmax is false
         */
    // Backward of the used scaling_factor
    // In-place update
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      if (local_routing_map[i]) {
        local_grad[i] = local_grad[i] * scaling_factor;
      }
    }
    __syncwarp();

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

    // Sigmoid/Sqrtsoftplus Post-processing bwd when topk > 1 (normalization backward)
    if (topk > 1 && (score_function == 0 || score_function == 2)) {
      // Select the correct activation output buffer:
      // - Sigmoid: local_act_from_fwd already contains sigmoid output
      // - Sqrtsoftplus: local_comp_buf contains sqrtsoftplus output computed above
      CompType *act_output = (score_function == 0) ? local_act_from_fwd : local_comp_buf;

      CompType sum_fwd_input = masked_warp_reduce_on_shmem(
          /*data ptr = */ act_output,
          /*mask ptr = */ local_routing_map,
          /*data size = */ num_experts,
          /*reduce func = */ ReduceFuncType::SUM, lane_id);
      // Compute sum of output * grad using registers
      CompType local_sum_Output_x_Grad = 0.0;
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        if (local_routing_map[i]) {
          local_sum_Output_x_Grad += local_grad[i] * act_output[i];
        }
      }
      // Warp reduce the sum
      for (int s = 16; s > 0; s /= 2) {
        local_sum_Output_x_Grad += __shfl_xor_sync(0xffffffff, local_sum_Output_x_Grad, s);
      }
      CompType sum_Output_x_Grad = local_sum_Output_x_Grad;
      // In-place update
      for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
        if (local_routing_map[i]) {
          local_grad[i] =
              local_grad[i] / (sum_fwd_input + epsilon) -
              sum_Output_x_Grad / ((sum_fwd_input + epsilon) * (sum_fwd_input + epsilon));
        } else {
          local_grad[i] = 0.0;
        }
      }
      __syncwarp();
    }

    // Softmax bwd if use_pre_softmax is false
    if (!use_pre_softmax && score_function == 1) {
      apply_softmax_bwd_on_float(local_grad, local_act_from_fwd, local_comp_buf, local_routing_map,
                                 num_experts, lane_id);
      __syncwarp();
    }

    /***
         * Section: Backward of topk
         * mask the unselected position in the grad
         */
    for (int i = lane_id; i < num_experts; i += kThreadsPerWarp) {
      if (!local_routing_map[i]) {
        local_grad[i] = 0.0;
      }
    }
    __syncwarp();

    /***
         * Section: Backward of ops before the topk
         * - Pre-softmax bwd
         * - Sigmoid bwd
         * - Sqrtsoftplus bwd
         * - Write the grad_logits to the global mem
         */
    // Pre-softmax bwd
    if (score_function == 1 && use_pre_softmax) {
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
      grad_logits[pos_offset + i] = local_grad[i];
    }
    __syncwarp();
  }
}

template <typename DataType>
void fused_topk_with_score_function_backward_kernel_launcher(
    const bool *routing_map, const CompType *intermediate_output, const DataType *grad_probs,
    int num_tokens, int num_experts, int topk, bool use_pre_softmax, float scaling_factor,
    int score_function, DataType *grad_logits, cudaStream_t stream) {
  // Meta data for the kernel
  size_t num_token_per_block = kThreadsPerBlock / kThreadsPerWarp;
  size_t grid_size = (num_tokens + num_token_per_block - 1) / num_token_per_block;
  size_t shared_memory_size = num_experts * num_token_per_block * sizeof(CompType)  // grad_probs
                              +
                              num_experts * num_token_per_block * sizeof(CompType)  // act_from_fwd
                              + num_experts * num_token_per_block * sizeof(CompType)  // comp_buf
                              + num_experts * num_token_per_block * sizeof(bool);     // routing_map
  check_shared_memory_capacity_num_experts(shared_memory_size, num_experts);
  NVTE_CHECK_CUDA(cudaFuncSetAttribute(fused_topk_with_score_function_backward_kernel<DataType>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       shared_memory_size));
  fused_topk_with_score_function_backward_kernel<DataType>
      <<<grid_size, kThreadsPerBlock, shared_memory_size, stream>>>(
          routing_map, intermediate_output, grad_probs, num_tokens, num_experts, topk,
          use_pre_softmax, scaling_factor, score_function, grad_logits);
  NVTE_CHECK_CUDA(cudaGetLastError());
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
