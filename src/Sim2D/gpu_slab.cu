// =============================================================================
// gpu_slab.cu
// nvcc-compiled implementation of GpuSlabAllocator::biasCsrOffsets.
//
// Separated from gpu_slab.cuh so the declaration is visible to MSVC-compiled
// translation units (simulation.cpp) without requiring __CUDACC__ guards.
// =============================================================================

#include <cuda.h>          // must precede glm when GLM_FORCE_CUDA is active
#include "dynamic_foam/Sim2D/gpu_slab.cuh"
#include "dynamic_foam/Sim2D/cuda_utils.cuh"

namespace DynamicFoam::Sim2D {

// Adds a scalar bias to every element of a uint32 device array.
// Used to convert 0-based node_offsets written by buildGPUAdjacencyList into
// global indices into d_csr_nbrs after they are placed in a slab slice.
static __global__ void k_bias_u32(uint32_t* arr, int n, uint32_t bias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] += bias;
}

void GpuSlabAllocator::biasCsrOffsets(int foam_id) {
    const FoamSlot& s = slots.at(foam_id);
    const int n = s.csr_node_capacity; // N+1 entries written by buildGPUAdjacencyList
    if (n <= 0 || s.csr_edge_offset == 0) return;
    k_bias_u32<<<grid_size(n), 256>>>(
        d_csr_node_offsets + s.csr_node_offset,
        n,
        static_cast<uint32_t>(s.csr_edge_offset));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace DynamicFoam::Sim2D
