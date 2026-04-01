// =============================================================================
// simulation_gpu.cu
// Thin free-function wrappers for CUDA-guarded operations that Simulation needs
// to call from simulation.cpp (compiled by MSVC, not nvcc).
//
// These wrappers are intentionally minimal and do NOT include entt's registry
// header, which is incompatible with nvcc's template instantiation.
// =============================================================================

#include <cub/cub.cuh>
#include <entt/entity/entity.hpp>  // entt::entity type only — no registry
#include "dynamic_foam/Sim2D/adjacency.cuh"
#include "dynamic_foam/Sim2D/gpu_slab.cuh"

namespace DynamicFoam::Sim2D {

// Called from simulation.cpp during slab initialization.
// Wraps AdjacencyList<entt::entity>::buildGPUAdjacencyListIntoSlice which is
// guarded by #ifdef __CUDACC__ and thus invisible to MSVC.
void buildAdjacencyIntoSlabSlice(
    AdjacencyList<entt::entity>& adj,
    AdjacencyListGPU<entt::entity>& out,
    uint32_t* d_nbrs_slice,       size_t nbrs_cap,
    uint32_t* d_node_offsets_slice, size_t node_offsets_cap)
{
    adj.buildGPUAdjacencyListIntoSlice(
        out,
        d_nbrs_slice,         nbrs_cap,
        d_node_offsets_slice, node_offsets_cap);
}

// Called from simulation.cpp after buildAdjacencyIntoSlabSlice.
// Wraps GpuSlabAllocator::biasCsrOffsets which launches k_bias_u32 and is
// thus guarded by #ifdef __CUDACC__.
void biasSlabCsrOffsets(GpuSlabAllocator& slab, int foam_id) {
    slab.biasCsrOffsets(foam_id);
}

} // namespace DynamicFoam::Sim2D
