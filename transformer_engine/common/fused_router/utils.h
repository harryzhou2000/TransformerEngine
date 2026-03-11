/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ROUTER_UTILS_H_
#define TRANSFORMER_ENGINE_FUSED_ROUTER_UTILS_H_

#include <assert.h>
#include <limits>

#include "transformer_engine/transformer_engine.h"

namespace transformer_engine {

constexpr size_t kThreadsPerWarp = 32;
constexpr int kThreadsPerBlock =
    128;  // Using 4 warps in 1 CTA, Each warp is responsible for 1 token.
constexpr float epsilon = 1e-20;

template <typename T>
__device__ inline T max(T a, T b) {
  return a > b ? a : b;
}

template <typename T>
__device__ inline T sum(T a, T b) {
  return a + b;
}

enum ReduceFuncType {
  SUM,
  MAX,
};

template <typename T>
__device__ inline T warp_reduce_on_shmem(T *data_ptr, int data_size, ReduceFuncType type,
                                         int lane_id) {
  T (*reduce_func)(T, T);
  double default_val = 0;
  if (type == ReduceFuncType::SUM) {
    reduce_func = sum;
    default_val = 0;
  } else if (type == ReduceFuncType::MAX) {
    reduce_func = max;
    default_val = -std::numeric_limits<double>::infinity();
  }

  // Some value is hanlded in local thread
  // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
  // Reduce the value in local thread
  volatile double val = lane_id < data_size ? static_cast<double>(data_ptr[lane_id]) : default_val;
  for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
    val = reduce_func(val, data_ptr[i]);
  }

  // Warp shuffle between threads
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 1));
  __syncwarp();
  return T(val);
}

template <typename DataType>
__device__ inline void apply_sigmoid_on_float(DataType *scores, int data_size, int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = static_cast<float>(1.0f / (1.0f + exp(-static_cast<float>(scores[i]))));
  }
}

template <typename T>
__device__ inline T masked_warp_reduce_on_shmem(T *data_ptr, bool *mask, int data_size,
                                                ReduceFuncType type, int lane_id) {
  T (*reduce_func)(T, T);
  double default_val = 0;
  if (type == ReduceFuncType::SUM) {
    reduce_func = sum;
    default_val = 0;
  } else if (type == ReduceFuncType::MAX) {
    reduce_func = max;
    default_val = -std::numeric_limits<double>::infinity();
  }

  // Some value is hanlded in local thread
  // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
  // Reduce the value in local thread
  volatile double val =
      lane_id < data_size && mask[lane_id] ? static_cast<double>(data_ptr[lane_id]) : default_val;
  for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
    if (mask[i]) {
      val = reduce_func(val, data_ptr[i]);
    }
  }

  // Warp shuffle between threads
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 16));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 8));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 4));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 2));
  val = reduce_func(val, __shfl_xor_sync(0xffffffff, val, 1));
  __syncwarp();
  return T(val);
}

template <typename DataType>
__device__ inline void apply_sigmoid_bwd_on_float(DataType *grad, DataType *fwd_output,
                                                  int data_size, int lane_id) {
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    grad[i] = static_cast<double>(grad[i]) * static_cast<double>(fwd_output[i]) *
              (1 - static_cast<double>(fwd_output[i]));
  }
}

template <typename DataType>
__device__ inline void apply_softmax_bwd_on_float(DataType *grad, DataType *fwd_output,
                                                  DataType *comp_buf, bool *mask, int data_size,
                                                  int lane_id) {
  // Put the result of output * grad to the comp_buf
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    if (mask) {
      if (mask[i])
        comp_buf[i] = static_cast<float>(grad[i]) * static_cast<float>(fwd_output[i]);
      else
        comp_buf[i] = 0.0f;
    } else {
      comp_buf[i] = static_cast<float>(grad[i]) * static_cast<float>(fwd_output[i]);
    }
  }
  __syncwarp();
  float sum_Output_x_Grad = warp_reduce_on_shmem(
      /*data ptr = */ comp_buf,
      /*data size = */ data_size,
      /*reduce func = */ ReduceFuncType::SUM, lane_id);
  // In-place update
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    if (mask) {
      if (mask[i])
        grad[i] =
            static_cast<float>(fwd_output[i]) * (static_cast<float>(grad[i]) - sum_Output_x_Grad);
      else
        grad[i] = 0.0f;
    } else {
      grad[i] =
          static_cast<float>(fwd_output[i]) * (static_cast<float>(grad[i]) - sum_Output_x_Grad);
    }
  }
}

template <typename DataType>
__device__ inline void apply_softmax_on_float(DataType *scores, int data_size, int lane_id) {
  // 1. compute the max of value
  float max_val =
      static_cast<float>(warp_reduce_on_shmem(scores, data_size, ReduceFuncType::MAX, lane_id));
  // 2. value -> exp_value
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = static_cast<float>(exp(static_cast<float>(scores[i]) - max_val));
  }
  __syncwarp();
  // 3. compute the sum of exp_value
  float sum_val =
      static_cast<float>(warp_reduce_on_shmem(scores, data_size, ReduceFuncType::SUM, lane_id));
  // 4. update the softmax value
  for (int i = lane_id; i < data_size; i += kThreadsPerWarp) {
    scores[i] = static_cast<float>(scores[i]) / sum_val;
  }
  __syncwarp();
}

/*******************************************************************************
 * naive_topk_and_mask_v1 — Warp-level bitonic-sort based top-K
 *
 * Supports topk up to N_REGS * 32 (max 128 with N_REGS=4) by giving each of
 * the 32 warp lanes N_REGS register pairs (value, index).  The registers form
 * a virtual array of N_REGS*32 elements in row-major order:
 *
 *   virtual_pos(reg_r, lane_l) = r * 32 + l
 *
 * A full bitonic sort over this virtual array uses:
 *   - __shfl_xor_sync  for partner distances j < 32  (cross-lane, same reg)
 *   - local register CAS for distances j >= 32        (same lane, cross-reg)
 *
 * After sorting descending, virtual positions 0..topk-1 hold the result.
 *
 * Streaming merge
 * ---------------
 * When data_size > N_REGS*32, we stream in chunks.  The first
 * keep_regs = ceil(topk/32) register rows are the "keep set" and are never
 * overwritten.  The remaining (N_REGS - keep_regs) rows accept new data each
 * chunk, are merged via a full sort, and the tail is discarded.
 *
 * Complexity: O( (E / new_slots) * N_REGS * 32 * log²(N_REGS*32) )
 *           ≈ O( E * log²(N_REGS*32) )  — no K² term.
 *
 * Stability: ties broken by original index ascending (lower index wins),
 *            matching the original naive_topk_and_mask behavior.
 *
 * Constraints:
 *   - topk <= N_REGS * 32  (i.e. topk <= 128 for N_REGS=4)
 *   - If data_size > N_REGS*32, then topk < N_REGS*32 (need spare slots)
 *   - N_REGS must be a power of two: {1, 2, 4}.  Non-power-of-two values
 *     (e.g. 3) cause the bitonic sort's cross-register CAS steps to
 *     reference non-existent registers, silently skipping comparisons
 *     and breaking sort correctness.
 ******************************************************************************/

// Compare-and-swap helper for descending sort by (value DESC, index ASC).
// "Descending" means: the element that should come first (lower virtual position)
// has a higher value, or on tie, a lower original index.
__device__ inline void bitonic_cas_descending(float &val_a, int &idx_a, float &val_b, int &idx_b) {
  bool swap = (val_a < val_b) || (val_a == val_b && idx_a > idx_b);
  if (swap) {
    float tmp_val = val_a;
    val_a = val_b;
    val_b = tmp_val;
    int tmp_idx = idx_a;
    idx_a = idx_b;
    idx_b = tmp_idx;
  }
}

/*******************************************************************************
 * Warp-level bitonic sort over N_REGS * 32 elements (N_REGS in {1,2,3,4}).
 *
 * Each lane holds vals[0..N_REGS-1] and idxs[0..N_REGS-1].  The virtual
 * position of (reg r, lane l) is  r*32 + l.  After this function returns,
 * virtual position 0 holds the global maximum and position N_REGS*32-1 the
 * minimum.
 *
 * The total virtual size TOTAL = N_REGS * 32 need not be a power of two for
 * N_REGS = 3 (96 elements).  We round up to the next power of two (128) for
 * the bitonic network and pad with sentinels, but this is handled implicitly
 * by initializing unused registers to (-inf, -1) before entry.
 *
 * Implementation note: to keep the code compact and avoid per-N_REGS template
 * specializations for every CAS step, we use helper lambdas that map a
 * virtual position to (reg, lane) and vice-versa.
 ******************************************************************************/

// Core bitonic sort over N registers × 32 lanes.
// vals[0..N-1] and idxs[0..N-1] are the per-lane register arrays.
// TOTAL_P2 is the padded power-of-2 total size (>= N*32).
// Virtual positions >= N*32 are assumed to hold sentinels already.
template <int N>
__device__ inline void warp_bitonic_sort_N_descending(float (&vals)[N], int (&idxs)[N], int lane_id,
                                                      int total_p2) {
  // Bitonic sort: iterate over stages k = 2, 4, 8, ... total_p2
  // and sub-stages j = k/2, k/4, ... 1.
  // For each (k, j), every virtual position p is paired with p ^ j.
  // The sort direction for position p within stage k:
  //   descending if ((p & k) == 0), ascending otherwise.
  for (int k = 2; k <= total_p2; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      // Determine which register pairs are involved.
      // Partner of virtual pos (r, lane_id) = r*32 + lane_id is at
      //   partner_pos = (r*32 + lane_id) ^ j
      //   partner_reg = partner_pos / 32,  partner_lane = partner_pos % 32
      // Two cases:
      //   j < 32:  partner_reg == r (same reg), partner_lane = lane_id ^ j
      //   j >= 32: partner_lane == lane_id (same lane), partner_reg = r ^ (j/32)

      if (j < 32) {
        // Cross-lane exchange within each register.
        // Both lanes in a pair receive each other's value via the shuffle.
        // Each lane independently decides whether to keep its own value or
        // the partner's, based on whether it is the "low" or "high" position
        // in the comparator and the sort direction for this sub-sequence.
#pragma unroll
        for (int r = 0; r < N; r++) {
          int vpos = r * 32 + lane_id;
          float partner_val = __shfl_xor_sync(0xffffffff, vals[r], j);
          int partner_idx = __shfl_xor_sync(0xffffffff, idxs[r], j);

          // Determine sort direction: descending if this position's
          // k-th bit is 0, ascending otherwise.
          bool descending = ((vpos & k) == 0);
          // Am I the lower virtual position in this pair?
          bool is_low = (lane_id & j) == 0;

          // The low position keeps the "winner" (larger for descending,
          // smaller for ascending); the high position keeps the other.
          // Equivalently: low+descending → keep max; low+ascending → keep min;
          //               high+descending → keep min; high+ascending → keep max.
          // "want_larger" means this lane should hold the larger element.
          bool want_larger = (is_low == descending);

          // Check if we need to swap: does the partner have what we want?
          bool partner_is_larger =
              (partner_val > vals[r]) || (partner_val == vals[r] && partner_idx < idxs[r]);
          if (want_larger == partner_is_larger) {
            vals[r] = partner_val;
            idxs[r] = partner_idx;
          }
        }
      } else {
        // Same-lane exchange across registers
        int reg_dist = j / 32;  // distance in register indices
#pragma unroll
        for (int r = 0; r < N; r++) {
          int partner_r = r ^ reg_dist;
          if (partner_r > r && partner_r < N) {
            int vpos = r * 32 + lane_id;
            bool descending = ((vpos & k) == 0);
            if (descending) {
              bitonic_cas_descending(vals[r], idxs[r], vals[partner_r], idxs[partner_r]);
            } else {
              bitonic_cas_descending(vals[partner_r], idxs[partner_r], vals[r], idxs[r]);
            }
          }
        }
      }
      __syncwarp();
    }
  }
}

template <typename T, int N_REGS>
__device__ inline void naive_topk_and_mask_v1_impl(T *scores, int data_size, int topk,
                                                   int *topk_indices, T *topk_scores, int lane_id) {
  constexpr float NEG_INF = std::numeric_limits<float>::lowest();
  constexpr int INVALID_IDX = -1;
  constexpr int TOTAL = N_REGS * 32;

  // Padded power-of-2 total for bitonic network.
  // N_REGS must be a power of two (1, 2, or 4) so TOTAL == TOTAL_P2.
  // N_REGS: 1->32, 2->64, 4->128
  constexpr int TOTAL_P2 = (N_REGS <= 1) ? 32 : (N_REGS <= 2) ? 64 : 128;

  // ---- Precondition checks (device-side asserts) ----
  // topk must fit within the N_REGS register slots
  assert(topk > 0 && "naive_topk_and_mask_v1: topk must be positive");
  assert(topk <= TOTAL && "naive_topk_and_mask_v1: topk exceeds N_REGS * 32 capacity");
  assert(topk <= data_size && "naive_topk_and_mask_v1: topk exceeds data_size");
  // Streaming requires at least 32 spare slots (one register row) for new data
  assert((data_size <= TOTAL || topk <= (N_REGS - 1) * 32) &&
         "naive_topk_and_mask_v1: data_size > N_REGS*32 but topk leaves no spare register "
         "row for streaming; increase N_REGS or reduce topk");

  // Register arrays: each lane holds N_REGS (value, index) pairs
  float vals[N_REGS];
  int idxs[N_REGS];

  // Initialize all to sentinel
#pragma unroll
  for (int r = 0; r < N_REGS; r++) {
    vals[r] = NEG_INF;
    idxs[r] = INVALID_IDX;
  }

  // Number of register rows dedicated to the keep set
  int keep_regs = (topk + 31) / 32;  // ceil(topk / 32)

  // Number of new element slots per chunk (the non-keep rows)
  int new_slots = (N_REGS - keep_regs) * 32;

  if (data_size <= TOTAL) {
    // ---- Fast path: everything fits in registers, single sort ----
#pragma unroll
    for (int r = 0; r < N_REGS; r++) {
      int elem_idx = r * 32 + lane_id;
      if (elem_idx < data_size) {
        vals[r] = static_cast<float>(scores[elem_idx]);
        idxs[r] = elem_idx;
      }
    }

    // For N_REGS=3, positions 96..127 are virtual (TOTAL_P2=128)
    // They stay as sentinel, which is correct (sorted to the bottom).
    warp_bitonic_sort_N_descending<N_REGS>(vals, idxs, lane_id, TOTAL_P2);

  } else {
    // ---- Streaming path: merge chunks into the running top-K ----
    // First chunk: fill all N_REGS rows
#pragma unroll
    for (int r = 0; r < N_REGS; r++) {
      int elem_idx = r * 32 + lane_id;
      if (elem_idx < data_size) {
        vals[r] = static_cast<float>(scores[elem_idx]);
        idxs[r] = elem_idx;
      }
    }
    warp_bitonic_sort_N_descending<N_REGS>(vals, idxs, lane_id, TOTAL_P2);

    // Discard tail (positions >= topk): reset non-keep slots to sentinel
#pragma unroll
    for (int r = keep_regs; r < N_REGS; r++) {
      // For the boundary register row (r == keep_regs-1 is kept fully;
      // r == keep_regs may be partially kept if topk is not a multiple of 32)
      vals[r] = NEG_INF;
      idxs[r] = INVALID_IDX;
    }
    // Handle partial keep in the last keep row
    if (topk % 32 != 0 && keep_regs > 0) {
      int last_keep = keep_regs - 1;
      if (lane_id >= (topk % 32)) {
        vals[last_keep] = NEG_INF;
        idxs[last_keep] = INVALID_IDX;
      }
    }
    __syncwarp();

    // Subsequent chunks
    for (int base = TOTAL; base < data_size; base += new_slots) {
      // Load new data into the non-keep register rows
#pragma unroll
      for (int r = keep_regs; r < N_REGS; r++) {
        int slot_in_new = (r - keep_regs) * 32 + lane_id;
        int elem_idx = base + slot_in_new;
        if (elem_idx < data_size) {
          vals[r] = static_cast<float>(scores[elem_idx]);
          idxs[r] = elem_idx;
        } else {
          vals[r] = NEG_INF;
          idxs[r] = INVALID_IDX;
        }
      }

      warp_bitonic_sort_N_descending<N_REGS>(vals, idxs, lane_id, TOTAL_P2);

      // Discard tail again
#pragma unroll
      for (int r = keep_regs; r < N_REGS; r++) {
        vals[r] = NEG_INF;
        idxs[r] = INVALID_IDX;
      }
      if (topk % 32 != 0 && keep_regs > 0) {
        int last_keep = keep_regs - 1;
        if (lane_id >= (topk % 32)) {
          vals[last_keep] = NEG_INF;
          idxs[last_keep] = INVALID_IDX;
        }
      }
      __syncwarp();
    }
  }

  // Write results: virtual positions 0..topk-1 hold the top-K descending
#pragma unroll
  for (int r = 0; r < N_REGS; r++) {
    int vpos = r * 32 + lane_id;
    if (vpos < topk) {
      topk_indices[vpos] = idxs[r];
      topk_scores[vpos] = static_cast<T>(vals[r]);
    }
  }
  __syncwarp();
}

// Dispatch wrapper: selects N_REGS based on topk AND data_size at runtime.
//
// N_REGS must satisfy two constraints:
//   1. topk  <= N_REGS * 32        (results must fit in register file)
//   2. If data_size > N_REGS * 32 (streaming path), we need at least one
//      spare register row for incoming data, i.e.  keep_regs < N_REGS
//      where keep_regs = ceil(topk / 32).  Equivalently:
//        N_REGS >= ceil(topk / 32) + 1
//
// We compute min_n_regs satisfying both constraints, then round up to a
// power of two.  The bitonic sort network operates over TOTAL_P2 virtual
// positions (next power-of-two >= N_REGS*32).  When N_REGS is not a
// power of two (e.g. N_REGS=3, TOTAL_P2=128), cross-register CAS steps
// reference register indices that don't exist (e.g. register 3), causing
// those comparisons to be silently skipped and breaking the bitonic
// invariants.  Restricting N_REGS to {1, 2, 4} avoids the problem
// entirely since N_REGS*32 is always a power of two.
//
// Supported N_REGS: 1, 2, 4.
// Maximum topk:
//   - If data_size <= 128:  topk <= 128  (no streaming, N_REGS=4 suffices)
//   - If data_size >  128:  topk <=  96  (streaming, need spare row, N_REGS=4
//                                          gives keep_regs<=3 < 4)
//
// Preconditions:
//   - 0 < topk <= 128
//   - 0 < topk <= data_size
template <typename T>
__device__ inline void naive_topk_and_mask_v1(T *scores, int data_size, int topk,
                                              int *topk_indices, T *topk_scores, int lane_id) {
  assert(topk > 0 && "naive_topk_and_mask_v1: topk must be positive");
  assert(topk <= 128 && "naive_topk_and_mask_v1: topk exceeds maximum supported value (128)");
  assert(data_size >= topk && "naive_topk_and_mask_v1: data_size must be >= topk");

  // Minimum N_REGS to hold the topk results
  int keep_regs = (topk + 31) / 32;  // ceil(topk / 32)

  // If streaming is needed, we need at least one extra register row
  // for incoming data.  Compute the minimum total required.
  //
  // min_n_regs = keep_regs           if data_size <= keep_regs * 32
  //            = keep_regs + 1       otherwise (need streaming)
  bool needs_streaming_at_min = (data_size > keep_regs * 32);
  int min_n_regs = needs_streaming_at_min ? keep_regs + 1 : keep_regs;

  // Round up to the next power of two to ensure the bitonic sort network
  // has no missing register partners in cross-register CAS steps.
  // 1 → 1, 2 → 2, 3 → 4, 4 → 4
  if (min_n_regs == 3) {
    min_n_regs = 4;
  }

  assert(min_n_regs <= 4 &&
         "naive_topk_and_mask_v1: topk/data_size combination requires N_REGS > 4 "
         "(topk too large for streaming with 4 register rows)");

  if (min_n_regs <= 1) {
    naive_topk_and_mask_v1_impl<T, 1>(scores, data_size, topk, topk_indices, topk_scores, lane_id);
  } else if (min_n_regs <= 2) {
    naive_topk_and_mask_v1_impl<T, 2>(scores, data_size, topk, topk_indices, topk_scores, lane_id);
  } else {
    naive_topk_and_mask_v1_impl<T, 4>(scores, data_size, topk, topk_indices, topk_scores, lane_id);
  }
}

template <typename T>
__device__ inline void naive_topk_and_mask(T *scores, int data_size, int topk, int *topk_indices,
                                           T *topk_scores, int lane_id) {
  // Check if the index is masked by the later iteration
  auto is_masked = [&topk_indices](int k, int index) {
    if (k == 0) return false;
    for (int i = 0; i < k; i++) {
      if (topk_indices[i] == index) return true;
    }
    return false;
  };
  // Topk Times: Find the max value and its index
  // Then mask it, and record the index in the topk_indices
  // After looping topk times, the topk_indices will be the topk indices
  for (int k = 0; k < topk; k++) {
    // Find the max value and its index
    volatile double val = (lane_id < data_size && !is_masked(k, lane_id))
                              ? static_cast<double>(scores[lane_id])
                              : -std::numeric_limits<double>::infinity();
    volatile int index = (lane_id < data_size) ? lane_id : 0;
    // Some value is hanlded in local thread
    // Thread 0 is responsible for the: 0-th, 32-th, 64-th, 96-th ...
    // Reduce the value in local thread
    for (int i = lane_id + kThreadsPerWarp; i < data_size; i += kThreadsPerWarp) {
      volatile double cur_val = (is_masked(k, i)) ? -std::numeric_limits<double>::infinity()
                                                  : static_cast<double>(scores[i]);
      if (cur_val > val) {
        val = cur_val;
        index = i;
      }
    }
    // Warp shuffle between threads
    for (int s = 16; s > 0; s /= 2) {
      volatile auto shuffled_val = __shfl_xor_sync(0xffffffff, val, s);
      volatile auto shuffled_index = __shfl_xor_sync(0xffffffff, index, s);
      if (shuffled_val > val) {
        val = shuffled_val;
        index = shuffled_index;
      }
    }
    if (lane_id == 0) {
      topk_indices[k] = index;
      topk_scores[k] = val;
    }
    __syncwarp();
  }
}

// Current TE only support float32/bf16/fp16, float64 probs should be considered in the future
#define TE_ROUTER_PROBS_TYPE_SWITCH_ALL(dtype, type, ...) \
  switch (dtype) {                                        \
    using namespace transformer_engine;                   \
    case DType::kFloat32: {                               \
      using type = float;                                 \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kFloat16: {                               \
      using type = fp16;                                  \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kBFloat16: {                              \
      using type = bf16;                                  \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    default:                                              \
      NVTE_ERROR("Invalid type.");                        \
  }

#define TE_ROUTER_INDEX_TYPE_SWITCH_ALL(dtype, type, ...) \
  switch (dtype) {                                        \
    using namespace transformer_engine;                   \
    case DType::kInt32: {                                 \
      using type = int32_t;                               \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kInt64: {                                 \
      using type = int64_t;                               \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kBFloat16: {                              \
      using type = bf16;                                  \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    case DType::kFloat32: {                               \
      using type = float;                                 \
      { __VA_ARGS__ }                                     \
    } break;                                              \
    default:                                              \
      NVTE_ERROR("Invalid type.");                        \
  }
}  // namespace transformer_engine
#endif
