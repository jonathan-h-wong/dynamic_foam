// =============================================================================
// gpu_slab.cu
// nvcc-compiled implementations for GpuSlabAllocator.
//
//   biasCsrOffsets      — bias 0-based CSR node_offsets by the slab edge start.
//   bulkMortonSort     — reorder all per-particle slab buffers by Morton code.
//
// Separated from gpu_slab.cuh so declarations are visible to MSVC-compiled
// translation units (simulation.cpp) without requiring __CUDACC__ guards.
// =============================================================================

#include <cuda.h>          // must precede glm when GLM_FORCE_CUDA is active
#include <cub/cub.cuh>
#include "dynamic_foam/Sim2D/gpu_slab.cuh"
#include "dynamic_foam/Sim2D/cuda_utils.cuh"

namespace DynamicFoam::Sim2D {

// Adds a scalar bias to every element of a uint32 device array.
// Used to convert 0-based node_offsets written by buildGPUAdjacencyList into
// global indices into d_csr_colidx after they are placed in a slab slice.
static __global__ void k_bias_u32(uint32_t* arr, int n, uint32_t bias) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] += bias;
}

void GpuSlabAllocator::biasCsrOffsets(int foam_id) {
    const FoamSlot& s = slots.at(foam_id);
    const int n = s.csr_node_capacity; // N+1 entries written by buildGPUAdjacencyList
    if (n <= 0 || s.csr_edge_offset == 0) return;
    k_bias_u32<<<grid_size(n), 256>>>(
        d_csr_rowptr + s.csr_node_offset,
        n,
        static_cast<uint32_t>(s.csr_edge_offset));
    CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// Foam AABB computation
// =============================================================================

void GpuSlabAllocator::computeFoamAABB(int foam_id, int n, AABB& out_bbox) {
    const FoamSlot& s = slots.at(foam_id);
    AABB* d_result = nullptr;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(AABB)));
    AABB identity{};
    CUB_CALL(cub::DeviceReduce::Reduce,
        d_particle_aabbs + s.particle_offset, d_result, n, AABBUnion{}, identity);
    // Store local-space AABB into the foam-indexed device buffer.
    CUDA_CHECK(cudaMemcpy(d_foam_aabbs + foam_id, d_result,
        sizeof(AABB), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(&out_bbox, d_result, sizeof(AABB), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_result));
}

// =============================================================================
// Bulk Morton sort — reorders ALL per-particle slab buffers together
// =============================================================================

// Computes a 30-bit Morton code for each of N particles using the centroid
// of its local-space AABB, normalized into [0,1]^3 against the foam AABB.
static __global__ void k_compute_all_morton(
    const AABB*  __restrict__ aabbs,
    uint32_t*    __restrict__ morton_out,
    int          n,
    glm::vec3    scene_min,
    glm::vec3    scene_extent_inv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const glm::vec3 c = aabbs[i].centroid();
    const float nx = fminf(fmaxf((c.x - scene_min.x) * scene_extent_inv.x, 0.f), 1.f);
    const float ny = fminf(fmaxf((c.y - scene_min.y) * scene_extent_inv.y, 0.f), 1.f);
    const float nz = fminf(fmaxf((c.z - scene_min.z) * scene_extent_inv.z, 0.f), 1.f);
    morton_out[i] = morton3D(nx, ny, nz);
}

// Fills arr[i] = i for i in [0, n).
static __global__ void k_iota(uint32_t* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = static_cast<uint32_t>(i);
}

// Generic gather: dst[i] = src[indices[i]].
template<typename T>
static __global__ void k_gather(
    const T*        __restrict__ src,
    const uint32_t* __restrict__ indices,
    T*              __restrict__ dst,
    int             n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[indices[i]];
}

void GpuSlabAllocator::bulkMortonSort(int foam_id, int n_particles)
{
    if (n_particles <= 0) return;
    const FoamSlot& s = slots.at(foam_id);

    // Step 1: compute the foam's local-space AABB for centroid normalization.
    AABB foam_bbox;
    computeFoamAABB(foam_id, n_particles, foam_bbox);
    const glm::vec3 extent = foam_bbox.max_pt - foam_bbox.min_pt;
    const glm::vec3 extent_inv = {
        extent.x > 0.f ? 1.f / extent.x : 1.f,
        extent.y > 0.f ? 1.f / extent.y : 1.f,
        extent.z > 0.f ? 1.f / extent.z : 1.f,
    };

    // Step 2: compute Morton codes for all N particles.
    uint32_t* d_morton_in  = nullptr;
    uint32_t* d_morton_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_morton_in,  n_particles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_morton_out, n_particles * sizeof(uint32_t)));
    k_compute_all_morton<<<grid_size(n_particles), 256>>>(
        d_particle_aabbs + s.particle_offset,
        d_morton_in, n_particles,
        foam_bbox.min_pt, extent_inv);
    CUDA_CHECK(cudaGetLastError());

    // Step 3: sort identity permutation [0..N-1] by Morton code to get the
    //         permutation that maps sorted position → original particle index.
    uint32_t* d_perm_in  = nullptr;
    uint32_t* d_perm_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_perm_in,  n_particles * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_perm_out, n_particles * sizeof(uint32_t)));
    k_iota<<<grid_size(n_particles), 256>>>(d_perm_in, n_particles);
    CUDA_CHECK(cudaGetLastError());
    CUB_CALL(cub::DeviceRadixSort::SortPairs,
        d_morton_in, d_morton_out,
        d_perm_in,   d_perm_out,
        n_particles);

    // Step 4: gather every per-particle slab buffer into Morton order.
    AABB*      d_aabb_tmp = nullptr;
    glm::vec4* d_col_tmp  = nullptr;
    glm::vec3* d_pos_tmp  = nullptr;
    uint8_t*   d_mask_tmp = nullptr;
    uint32_t*  d_ids_tmp  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_aabb_tmp, n_particles * sizeof(AABB)));
    CUDA_CHECK(cudaMalloc(&d_col_tmp,  n_particles * sizeof(glm::vec4)));
    CUDA_CHECK(cudaMalloc(&d_pos_tmp,  n_particles * sizeof(glm::vec3)));
    CUDA_CHECK(cudaMalloc(&d_mask_tmp, n_particles * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&d_ids_tmp,  n_particles * sizeof(uint32_t)));

    k_gather<AABB>    <<<grid_size(n_particles), 256>>>(
        d_particle_aabbs     + s.particle_offset, d_perm_out, d_aabb_tmp, n_particles);
    k_gather<glm::vec4><<<grid_size(n_particles), 256>>>(
        d_particle_colors    + s.particle_offset, d_perm_out, d_col_tmp,  n_particles);
    k_gather<glm::vec3><<<grid_size(n_particles), 256>>>(
        d_particle_positions + s.particle_offset, d_perm_out, d_pos_tmp,  n_particles);
    k_gather<uint8_t>  <<<grid_size(n_particles), 256>>>(
        d_particle_surface_mask + s.particle_offset, d_perm_out, d_mask_tmp, n_particles);
    k_gather<uint32_t> <<<grid_size(n_particles), 256>>>(
        d_active_ids         + s.active_offset,   d_perm_out, d_ids_tmp,  n_particles);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(d_particle_aabbs     + s.particle_offset, d_aabb_tmp,
        n_particles * sizeof(AABB),      cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_particle_colors    + s.particle_offset, d_col_tmp,
        n_particles * sizeof(glm::vec4), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_particle_positions + s.particle_offset, d_pos_tmp,
        n_particles * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_particle_surface_mask + s.particle_offset, d_mask_tmp,
        n_particles * sizeof(uint8_t),   cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_active_ids         + s.active_offset,   d_ids_tmp,
        n_particles * sizeof(uint32_t),  cudaMemcpyDeviceToDevice));

    slots.at(foam_id).active_count = n_particles;

    // Free all temporaries.
    CUDA_CHECK(cudaFree(d_morton_in));    CUDA_CHECK(cudaFree(d_morton_out));
    CUDA_CHECK(cudaFree(d_perm_in));      CUDA_CHECK(cudaFree(d_perm_out));
    CUDA_CHECK(cudaFree(d_aabb_tmp));     CUDA_CHECK(cudaFree(d_col_tmp));
    CUDA_CHECK(cudaFree(d_pos_tmp));      CUDA_CHECK(cudaFree(d_mask_tmp));
    CUDA_CHECK(cudaFree(d_ids_tmp));
    CUDA_CHECK(cudaDeviceSynchronize());
}

// =============================================================================
// updateFoamData — apply FoamUpdate (deletions then insertions)
// =============================================================================

// For each particle at position i, set d_flags[i] = 1 (keep) unless its entity
// ID (from d_active_ids) appears in the del_ids array, in which case = 0 (drop).
static __global__ void k_build_del_flags(
    const uint32_t* __restrict__ active_ids,
    const uint32_t* __restrict__ del_ids,
    uint8_t*        __restrict__ d_flags,
    int             n_particles,
    int             n_dels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    const uint32_t id = active_ids[i];
    uint8_t keep = 1;
    for (int j = 0; j < n_dels; ++j) {
        if (del_ids[j] == id) { keep = 0; break; }
    }
    d_flags[i] = keep;
}

void GpuSlabAllocator::updateFoamData(int foam_id, const FoamUpdate& update)
{
    auto it = slots.find(foam_id);
    if (it == slots.end() || it->second.dead) return;

    const int n_dels = static_cast<int>(update.particle_id_dels.size());
    const int n_ins  = static_cast<int>(update.particle_position_ins.size());
    if (n_dels == 0 && n_ins == 0) return;

    int cur_count = it->second.active_count;

    // -------------------------------------------------------------------------
    // Phase 1: Deletions
    // -------------------------------------------------------------------------
    if (n_dels > 0 && cur_count > 0) {
        FoamSlot& s = slots.at(foam_id);

        // Upload deletion ID list to device.
        uint32_t* d_del_ids = nullptr;
        CUDA_CHECK(cudaMalloc(&d_del_ids, n_dels * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_del_ids, update.particle_id_dels.data(),
            n_dels * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Build per-particle keep-flag array (1 = keep, 0 = discard).
        uint8_t* d_flags = nullptr;
        CUDA_CHECK(cudaMalloc(&d_flags, cur_count * sizeof(uint8_t)));
        k_build_del_flags<<<grid_size(cur_count), 256>>>(
            d_active_ids + s.active_offset,
            d_del_ids, d_flags, cur_count, n_dels);
        CUDA_CHECK(cudaGetLastError());

        // Temporary compacted output buffers.
        AABB*      d_aabb_tmp = nullptr;
        glm::vec4* d_col_tmp  = nullptr;
        glm::vec3* d_pos_tmp  = nullptr;
        uint8_t*   d_mask_tmp = nullptr;
        uint32_t*  d_ids_tmp  = nullptr;
        CUDA_CHECK(cudaMalloc(&d_aabb_tmp, cur_count * sizeof(AABB)));
        CUDA_CHECK(cudaMalloc(&d_col_tmp,  cur_count * sizeof(glm::vec4)));
        CUDA_CHECK(cudaMalloc(&d_pos_tmp,  cur_count * sizeof(glm::vec3)));
        CUDA_CHECK(cudaMalloc(&d_mask_tmp, cur_count * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_ids_tmp,  cur_count * sizeof(uint32_t)));

        // Device-side output count (only needs to be read once, after the last call).
        int* d_num_selected = nullptr;
        CUDA_CHECK(cudaMalloc(&d_num_selected, sizeof(int)));

        // Stream-compact all five particle arrays using the same flag array.
        CUB_CALL(cub::DeviceSelect::Flagged,
            d_particle_aabbs        + s.particle_offset, d_flags, d_aabb_tmp, d_num_selected, cur_count);
        CUB_CALL(cub::DeviceSelect::Flagged,
            d_particle_colors       + s.particle_offset, d_flags, d_col_tmp,  d_num_selected, cur_count);
        CUB_CALL(cub::DeviceSelect::Flagged,
            d_particle_positions    + s.particle_offset, d_flags, d_pos_tmp,  d_num_selected, cur_count);
        CUB_CALL(cub::DeviceSelect::Flagged,
            d_particle_surface_mask + s.particle_offset, d_flags, d_mask_tmp, d_num_selected, cur_count);
        CUB_CALL(cub::DeviceSelect::Flagged,
            d_active_ids            + s.active_offset,   d_flags, d_ids_tmp,  d_num_selected, cur_count);

        // Read back the surviving particle count.
        CUDA_CHECK(cudaMemcpy(&cur_count, d_num_selected,
            sizeof(int), cudaMemcpyDeviceToHost));

        // Write compacted arrays back into the slab slot in-place.
        CUDA_CHECK(cudaMemcpy(d_particle_aabbs        + s.particle_offset, d_aabb_tmp,
            cur_count * sizeof(AABB),      cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_particle_colors       + s.particle_offset, d_col_tmp,
            cur_count * sizeof(glm::vec4), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_particle_positions    + s.particle_offset, d_pos_tmp,
            cur_count * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_particle_surface_mask + s.particle_offset, d_mask_tmp,
            cur_count * sizeof(uint8_t),   cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(d_active_ids            + s.active_offset,   d_ids_tmp,
            cur_count * sizeof(uint32_t),  cudaMemcpyDeviceToDevice));

        s.active_count = cur_count;

        CUDA_CHECK(cudaFree(d_del_ids));
        CUDA_CHECK(cudaFree(d_flags));
        CUDA_CHECK(cudaFree(d_num_selected));
        CUDA_CHECK(cudaFree(d_aabb_tmp)); CUDA_CHECK(cudaFree(d_col_tmp));
        CUDA_CHECK(cudaFree(d_pos_tmp));  CUDA_CHECK(cudaFree(d_mask_tmp));
        CUDA_CHECK(cudaFree(d_ids_tmp));
    }

    // -------------------------------------------------------------------------
    // Phase 2: Insertions
    // -------------------------------------------------------------------------
    if (n_ins > 0) {
        const int new_count = cur_count + n_ins;
        FoamSlot& s = slots.at(foam_id);

        if (new_count > s.particle_capacity) {
            // Slot is too small.  Tombstone and re-allocate a larger one at the
            // watermark, preserving the surviving particle data via D→D copy.
            const int old_particle_offset = s.particle_offset;
            const int old_active_offset   = s.active_offset;

            // Halve the stored (doubled) capacities back to the original counts
            // so that allocate() doubles them again correctly.
            const int bvh_nodes = s.bvh_capacity      / 2;
            const int csr_nodes = s.csr_node_capacity / 2;
            const int csr_edges = s.csr_edge_capacity / 2;

            s.dead = true;
            allocate(foam_id, bvh_nodes, csr_nodes, csr_edges, new_count);
            FoamSlot& ns = slots.at(foam_id); // re-fetch: allocate() overwrites the map entry

            // Copy post-deletion particles from old slot to new slot.
            if (cur_count > 0) {
                CUDA_CHECK(cudaMemcpy(d_particle_aabbs        + ns.particle_offset,
                    d_particle_aabbs        + old_particle_offset, cur_count * sizeof(AABB),      cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_particle_colors       + ns.particle_offset,
                    d_particle_colors       + old_particle_offset, cur_count * sizeof(glm::vec4), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_particle_positions    + ns.particle_offset,
                    d_particle_positions    + old_particle_offset, cur_count * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_particle_surface_mask + ns.particle_offset,
                    d_particle_surface_mask + old_particle_offset, cur_count * sizeof(uint8_t),   cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_active_ids            + ns.active_offset,
                    d_active_ids            + old_active_offset,   cur_count * sizeof(uint32_t),  cudaMemcpyDeviceToDevice));
            }
        }

        // Append new particles immediately after the surviving ones.
        FoamSlot& sn = slots.at(foam_id);
        const int insert_particle_pos = sn.particle_offset + cur_count;
        const int insert_active_pos   = sn.active_offset   + cur_count;

        CUDA_CHECK(cudaMemcpy(d_particle_aabbs        + insert_particle_pos,
            update.particle_aabb_ins.data(),            n_ins * sizeof(AABB),      cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_particle_colors       + insert_particle_pos,
            update.particle_color_ins.data(),           n_ins * sizeof(glm::vec4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_particle_positions    + insert_particle_pos,
            update.particle_position_ins.data(),        n_ins * sizeof(glm::vec3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_particle_surface_mask + insert_particle_pos,
            update.particle_surface_mask_ins.data(),    n_ins * sizeof(uint8_t),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_active_ids            + insert_active_pos,
            update.particle_active_ids_ins.data(),      n_ins * sizeof(uint32_t),  cudaMemcpyHostToDevice));

        sn.active_count = new_count;
    }
}
} // namespace DynamicFoam::Sim2D