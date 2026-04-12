/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_FUSED_ROUTER_ASYNC_LOADER_H_
#define TRANSFORMER_ENGINE_FUSED_ROUTER_ASYNC_LOADER_H_

#include <cuda_pipeline.h>

#include <type_traits>

#include "../utils.cuh"
#include "utils.h"

/*******************************************************************************
 * WarpAsyncLoader — Double-buffered global→shmem loader for fused router
 *
 * Overview
 * --------
 * Each warp in the fused router kernels processes one token at a time.  The
 * dominant global load is the logits vector (num_experts floats per token).
 * This utility class manages:
 *   1. A double-buffered shared memory region for the scores array,
 *      enabling overlap of compute on buffer N with the async load of
 *      buffer N+1.
 *   2. Arch-dispatched loads:
 *        sm_80+  → cp.async  (16-byte global→shmem, bypasses L1)
 *        sm_70   → Vec<>     (vectorized register load + shmem store)
 *        fallback→ scalar    (when pointer/size is not 16-byte aligned)
 *   3. Host-side shared memory size calculation that accounts for the
 *      double buffer and all other per-warp scratch arrays.
 *
 * Usage (device side)
 * -------------------
 *   // In kernel:
 *   WarpAsyncLoader loader(shmem_base, warp_id, num_experts, num_warps);
 *
 *   // Kick off first load (before entering main loop)
 *   loader.start_load<DataType>(src_ptr, num_experts, lane_id);
 *
 *   for (int round = ...) {
 *     // Wait for current buffer to be ready
 *     loader.wait(lane_id);
 *     CompType *scores = loader.current_buf();
 *
 *     // Kick off next round's load into the other buffer (if needed)
 *     loader.start_load<DataType>(next_src_ptr, num_experts, lane_id);
 *
 *     // ... compute on scores ...
 *
 *     loader.flip();
 *   }
 *
 * Usage (host side)
 * -----------------
 *   size_t scores_size = WarpAsyncLoader::scores_shmem_bytes(num_experts, num_warps);
 *   size_t total_shmem = scores_size + other_scratch_bytes;
 ******************************************************************************/

namespace transformer_engine {
namespace fused_router {

// ============================================================================
// Persistent kernel grid size computation
// ============================================================================

// Compute a persistent grid size: min(total_blocks_needed, SMs * max_blocks_per_SM).
// When double buffering is enabled (num_buffers == 2), the grid must be smaller than
// the total work so each block processes multiple rounds and the prefetch overlaps
// with compute.  When single-buffered, we can still use persistent launch but the
// benefit is just reduced launch overhead for large token counts.
//
// `kernel_func` is a pointer to the __global__ function.
// `block_size` is kThreadsPerBlock.
// `shmem_bytes` is the dynamic shared memory per block.
// `total_blocks` is ceil(num_tokens / tokens_per_block).
template <typename KernelFunc>
inline size_t compute_persistent_grid(KernelFunc kernel_func, int block_size, size_t shmem_bytes,
                                      size_t total_blocks) {
  int blocks_per_sm = 0;
  NVTE_CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel_func,
                                                                block_size, shmem_bytes));
  if (blocks_per_sm <= 0) {
    // Fallback: non-persistent (launch everything)
    return total_blocks;
  }
  int device_id;
  NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
  int num_sms;
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id));

  size_t max_resident = static_cast<size_t>(num_sms) * blocks_per_sm;
  return (total_blocks < max_resident) ? total_blocks : max_resident;
}

// ============================================================================
// Vectorized store helpers (global memory, using Vec<> from utils.cuh)
// ============================================================================

// Number of elements per 16-byte vector for a given type.
template <typename T>
struct VecTraits {
  // 16 bytes / sizeof(T) elements, clamped to at least 1
  static constexpr int kVecSize = (sizeof(T) <= 16) ? (16 / sizeof(T)) : 1;
};

// Vectorized store: write `count` elements from shmem/registers to global memory.
// Uses 16-byte vector stores for the aligned bulk, scalar for the tail.
// Works on all architectures.
template <typename T>
__device__ inline void vec_store_global(T *__restrict__ dst, const T *__restrict__ src, int count,
                                        int lane_id) {
  constexpr int kVecSize = VecTraits<T>::kVecSize;
  using VecType = typename BytesToType<sizeof(T) * kVecSize>::Type;

  // Check alignment of dst pointer for vectorized path
  bool aligned = (reinterpret_cast<uint64_t>(dst) % (sizeof(T) * kVecSize) == 0);
  int aligned_count = (count / kVecSize) * kVecSize;

  if (aligned && aligned_count > 0) {
    // Vectorized bulk
    int vec_count = aligned_count / kVecSize;
    for (int vi = lane_id; vi < vec_count; vi += kThreadsPerWarp) {
      VecType v;
      // Load from src (shmem or register-spilled array) — element by element into vec
      T *v_elts = reinterpret_cast<T *>(&v);
#pragma unroll
      for (int e = 0; e < kVecSize; e++) {
        v_elts[e] = src[vi * kVecSize + e];
      }
      reinterpret_cast<VecType *>(dst)[vi] = v;
    }
    // Scalar tail
    for (int i = aligned_count + lane_id; i < count; i += kThreadsPerWarp) {
      dst[i] = src[i];
    }
  } else {
    // Fully scalar fallback (unaligned pointer or tiny count)
    for (int i = lane_id; i < count; i += kThreadsPerWarp) {
      dst[i] = src[i];
    }
  }
}

// Vectorized fill: write `val` to `count` elements of global memory.
template <typename T>
__device__ inline void vec_fill_global(T *__restrict__ dst, T val, int count, int lane_id) {
  constexpr int kVecSize = VecTraits<T>::kVecSize;
  using VecType = typename BytesToType<sizeof(T) * kVecSize>::Type;

  bool aligned = (reinterpret_cast<uint64_t>(dst) % (sizeof(T) * kVecSize) == 0);
  int aligned_count = (count / kVecSize) * kVecSize;

  if (aligned && aligned_count > 0) {
    // Build a vector of repeated val
    VecType v;
    T *v_elts = reinterpret_cast<T *>(&v);
#pragma unroll
    for (int e = 0; e < kVecSize; e++) {
      v_elts[e] = val;
    }
    int vec_count = aligned_count / kVecSize;
    for (int vi = lane_id; vi < vec_count; vi += kThreadsPerWarp) {
      reinterpret_cast<VecType *>(dst)[vi] = v;
    }
    for (int i = aligned_count + lane_id; i < count; i += kThreadsPerWarp) {
      dst[i] = val;
    }
  } else {
    for (int i = lane_id; i < count; i += kThreadsPerWarp) {
      dst[i] = val;
    }
  }
}

// ============================================================================
// Vectorized global→shmem load (warp-strided, cast to CompType)
// ============================================================================

// Load `count` elements from global `src` (type SrcType) into shmem `dst`
// (type CompType = float), with type cast.  Uses vectorized loads when aligned.
template <typename SrcType>
__device__ inline void vec_load_global_to_shmem(const SrcType *__restrict__ src,
                                                CompType *__restrict__ dst, int count,
                                                int lane_id) {
  // For the source pointer: vectorize based on SrcType
  constexpr int kVecSize = VecTraits<SrcType>::kVecSize;

  bool src_aligned = (reinterpret_cast<uint64_t>(src) % (sizeof(SrcType) * kVecSize) == 0);
  int aligned_count = (count / kVecSize) * kVecSize;

  if (src_aligned && aligned_count > 0) {
    using SrcVecType = typename BytesToType<sizeof(SrcType) * kVecSize>::Type;
    int vec_count = aligned_count / kVecSize;
    for (int vi = lane_id; vi < vec_count; vi += kThreadsPerWarp) {
      // Vector load from global
      SrcVecType sv = reinterpret_cast<const SrcVecType *>(src)[vi];
      const SrcType *elts = reinterpret_cast<const SrcType *>(&sv);
#pragma unroll
      for (int e = 0; e < kVecSize; e++) {
        dst[vi * kVecSize + e] = static_cast<CompType>(elts[e]);
      }
    }
    // Scalar tail
    for (int i = aligned_count + lane_id; i < count; i += kThreadsPerWarp) {
      dst[i] = static_cast<CompType>(src[i]);
    }
  } else {
    // Scalar fallback
    for (int i = lane_id; i < count; i += kThreadsPerWarp) {
      dst[i] = static_cast<CompType>(src[i]);
    }
  }
}

// ============================================================================
// cp.async wrappers
// ============================================================================

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

// Issue a 16-byte cp.async from global to shared memory.
// Both `dst` (shmem) and `src` (global) must be 16-byte aligned.
__device__ __forceinline__ void cp_async_16B(void *__restrict__ dst, const void *__restrict__ src) {
  // Use __pipeline_memcpy_async for 16-byte aligned copy
  __pipeline_memcpy_async(dst, src, 16);
}

// Commit the current batch of cp.async operations.
__device__ __forceinline__ void cp_async_commit() { __pipeline_commit(); }

// Wait for all prior cp.async groups to complete.
__device__ __forceinline__ void cp_async_wait_all() { __pipeline_wait_prior(0); }

// Wait for all but the most recent `N` cp.async groups.
template <int N>
__device__ __forceinline__ void cp_async_wait_prior() {
  __pipeline_wait_prior(N);
}

#endif  // __CUDA_ARCH__ >= 800

// ============================================================================
// WarpAsyncLoader — the double-buffer manager
// ============================================================================

class WarpAsyncLoader {
 public:
  // ----- Host-side: shared memory size calculation -----

  // Returns the shared memory bytes needed for the double-buffered scores array.
  // All other scratch arrays (topk_scores, topk_indices, etc.) are separate.
  //
  // With double buffering, each warp needs 2 × num_experts × sizeof(CompType)
  // for the scores buffer.  Total = 2 × num_experts × num_warps × sizeof(CompType).
  static inline size_t scores_shmem_bytes(int num_experts, int num_warps) {
    return 2 * static_cast<size_t>(num_experts) * num_warps * sizeof(CompType);
  }

  // Single-buffer version (for backward kernels or when double buffering is disabled).
  static inline size_t scores_shmem_bytes_single(int num_experts, int num_warps) {
    return static_cast<size_t>(num_experts) * num_warps * sizeof(CompType);
  }

  // Decide at host side whether to use double buffering based on shmem budget.
  // Returns the number of score buffers (1 or 2).
  //
  // Strategy: ensure at least kMinBlocksPerSM blocks can co-reside on one SM.
  // With too few blocks per SM, occupancy collapses and the scheduler cannot
  // hide memory latency (register spills, global loads).
  //
  // For the target config E=2304, W=4:
  //   double-buf shmem ≈ 75 KB → 233/75 = 3 blocks/SM → 18% occupancy (BAD)
  //   single-buf shmem ≈ 38 KB → 233/38 = 6 blocks/SM → 37% occupancy (OK)
  static inline int choose_num_buffers(int num_experts, int num_warps, int other_shmem_bytes) {
    constexpr int kMinBlocksPerSM = 4;

    size_t single_buf = scores_shmem_bytes_single(num_experts, num_warps) + other_shmem_bytes;
    size_t double_buf = scores_shmem_bytes(num_experts, num_warps) + other_shmem_bytes;

    int device_id;
    NVTE_CHECK_CUDA(cudaGetDevice(&device_id));
    int max_smem;
    NVTE_CHECK_CUDA(
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device_id));

    // How many blocks fit per SM with each option?
    int blocks_double = (double_buf > 0) ? static_cast<int>(max_smem / double_buf) : 0;
    int blocks_single = (single_buf > 0) ? static_cast<int>(max_smem / single_buf) : 0;

    // Use double buffer only if it still allows adequate occupancy.
    // If double buffering gives < kMinBlocksPerSM but single does >= kMinBlocksPerSM,
    // prefer single.  If both are below threshold, pick whichever gives more blocks.
    if (blocks_double >= kMinBlocksPerSM) {
      return 2;
    }
    if (blocks_single >= kMinBlocksPerSM) {
      return 1;
    }
    // Both below threshold: pick the one with better occupancy
    return (blocks_double >= blocks_single) ? 2 : 1;
  }

  // Returns shmem bytes for the scores region given num_buffers (1 or 2).
  static inline size_t scores_shmem_bytes_n(int num_experts, int num_warps, int num_buffers) {
    return static_cast<size_t>(num_buffers) * num_experts * num_warps * sizeof(CompType);
  }

  // ----- Device-side: initialization -----

  // Construct from the base of the scores region in shared memory.
  // `num_buffers` is 1 (single) or 2 (double).  When 1, both buf_[0] and
  // buf_[1] point to the same region — start_load becomes a synchronous
  // overwrite and flip() is a no-op in effect.
  __device__ WarpAsyncLoader(CompType *buf_base, int warp_id, int num_experts, int num_warps,
                             int num_buffers)
      : num_experts_(num_experts), phase_(0), double_buf_(num_buffers == 2) {
    int per_buffer = num_experts * num_warps;
    buf_[0] = buf_base + warp_id * num_experts;
    if (num_buffers == 2) {
      buf_[1] = buf_base + per_buffer + warp_id * num_experts;
    } else {
      buf_[1] = buf_[0];  // single buffer: both point to same memory
    }
  }

  // Get the current buffer (the one that has data ready for compute).
  __device__ __forceinline__ CompType *current_buf() { return buf_[phase_]; }

  // Get the next buffer (the one being loaded into).
  __device__ __forceinline__ CompType *next_buf() { return buf_[phase_ ^ 1]; }

  // Flip buffers (call after compute on current_buf is done and before next wait).
  // In single-buffer mode this is a no-op (both buffers point to same memory).
  __device__ __forceinline__ void flip() {
    if (double_buf_) phase_ ^= 1;
  }

  // Whether double buffering is active.
  __device__ __forceinline__ bool is_double_buffered() const { return double_buf_; }

  // ----- Async load dispatch -----

  // Start loading `count` elements from global `src` (SrcType) into the NEXT
  // buffer, casting to CompType.  The caller should call `wait()` before
  // accessing the buffer that was loaded.
  //
  // On sm_80+: uses cp.async for 16-byte chunks, scalar for tail.
  // On sm_70:  uses vectorized register load + shmem store.
  template <typename SrcType>
  __device__ void start_load(const SrcType *__restrict__ src, int count, int lane_id) {
    CompType *dst = next_buf();
    load_impl<SrcType>(src, dst, count, lane_id);
  }

  // Load directly into the current buffer (for the very first load before the
  // main loop, when there's no previous compute to overlap with).
  template <typename SrcType>
  __device__ void load_current(const SrcType *__restrict__ src, int count, int lane_id) {
    CompType *dst = current_buf();
    load_impl<SrcType>(src, dst, count, lane_id);
  }

  // Wait for the pending async load to complete.
  // On sm_80+: waits on the pipeline.  On sm_70: no-op (loads were synchronous).
  __device__ __forceinline__ void wait(int lane_id) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    cp_async_wait_all();
#endif
    __syncwarp();
  }

 private:
  CompType *buf_[2];
  int num_experts_;
  int phase_;
  bool double_buf_;

  template <typename SrcType>
  __device__ void load_impl(const SrcType *__restrict__ src, CompType *__restrict__ dst, int count,
                            int lane_id) {
    // Decide which load path to use based on architecture and data types.
    //
    // cp.async loads 16 bytes at a time directly from global to shared memory.
    // It requires both src and dst to be 16-byte aligned, and only works when
    // SrcType == CompType (no in-flight type cast — cp.async is a raw memcpy).
    // When SrcType != CompType (e.g., bf16→float), we must go through registers
    // anyway for the cast, so we use the vectorized register path instead.

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    if constexpr (std::is_same<SrcType, CompType>::value) {
      // cp.async path: direct global→shmem, 16 bytes at a time
      constexpr int kBytesPerCopy = 16;
      constexpr int kEltsPerCopy = kBytesPerCopy / sizeof(CompType);  // 4 for float
      static_assert(kEltsPerCopy > 0, "CompType too large for 16-byte cp.async");

      bool src_aligned = (reinterpret_cast<uint64_t>(src) % kBytesPerCopy == 0);
      bool dst_aligned = (reinterpret_cast<uint64_t>(dst) % kBytesPerCopy == 0);
      int aligned_count = (count / kEltsPerCopy) * kEltsPerCopy;

      if (src_aligned && dst_aligned && aligned_count > 0) {
        int vec_count = aligned_count / kEltsPerCopy;
        for (int vi = lane_id; vi < vec_count; vi += kThreadsPerWarp) {
          cp_async_16B(dst + vi * kEltsPerCopy, src + vi * kEltsPerCopy);
        }
        // Scalar tail (must be synchronous)
        for (int i = aligned_count + lane_id; i < count; i += kThreadsPerWarp) {
          dst[i] = static_cast<CompType>(src[i]);
        }
        cp_async_commit();
        return;
      }
    }
    // Fall through to vectorized register path for non-float src or unaligned data
#endif  // __CUDA_ARCH__ >= 800

    // Vectorized register-based load with type cast
    vec_load_global_to_shmem<SrcType>(src, dst, count, lane_id);
  }
};

}  // namespace fused_router
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_FUSED_ROUTER_ASYNC_LOADER_H_
