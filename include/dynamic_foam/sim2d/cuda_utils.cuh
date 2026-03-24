// =============================================================================
// cuda_utils.cuh
// Shared CUDA utility macros and helpers used across all CUDA translation units.
// =============================================================================

#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif // __CUDACC__

// -----------------------------------------------------------------------------
// Error checking
// -----------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// -----------------------------------------------------------------------------
// CUB two-pass helper
// Eliminates the boilerplate of querying temp size, allocating, executing,
// and freeing for every CUB device-wide call.
//
// Usage:
//   CUB_CALL(cub::DeviceScan::ExclusiveSum, d_in, d_out, n);
//   CUB_CALL(cub::DeviceRadixSort::SortPairs, keys_in, keys_out, vals_in, vals_out, n);
// -----------------------------------------------------------------------------

#define CUB_CALL(fn, ...)                                                       \
    do {                                                                        \
        void*  _cub_temp       = nullptr;                                       \
        size_t _cub_temp_bytes = 0;                                             \
        CUDA_CHECK(fn(_cub_temp, _cub_temp_bytes, __VA_ARGS__));                \
        CUDA_CHECK(cudaMalloc(&_cub_temp, _cub_temp_bytes));                    \
        CUDA_CHECK(fn(_cub_temp, _cub_temp_bytes, __VA_ARGS__));                \
        CUDA_CHECK(cudaFree(_cub_temp));                                        \
    } while (0)

// -----------------------------------------------------------------------------
// Grid size helper
// Computes the number of blocks needed to cover n elements at a given
// block size (default 256).
// -----------------------------------------------------------------------------

inline int grid_size(int n, int block_size = 256) {
    return (n + block_size - 1) / block_size;
}

// -----------------------------------------------------------------------------
// Realloc helper
// Reallocates a device buffer only when the new size exceeds the current
// capacity, using a doubling strategy to amortize allocations.
// Sets *capacity to the new capacity when a reallocation occurs.
// -----------------------------------------------------------------------------

template<typename T>
inline void cuda_realloc_if_needed(T** ptr, size_t* capacity, size_t required) {
    if (required <= *capacity) return;
    if (*ptr) CUDA_CHECK(cudaFree(*ptr));
    *capacity = required * 2;
    CUDA_CHECK(cudaMalloc(ptr, *capacity * sizeof(T)));
}
