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
// Reparent — shift particle positions and AABBs to a new local origin
// =============================================================================

// Subtracts `origin` from both the position and the AABB of each particle.
// This is an in-place operation; no auxiliary memory is required.
static __global__ void k_reparent(
    glm::vec3* __restrict__ positions,
    AABB*      __restrict__ aabbs,
    int        n,
    glm::vec3  origin)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    positions[i]    -= origin;
    aabbs[i].min_pt -= origin;
    aabbs[i].max_pt -= origin;
}

void GpuSlabAllocator::reparentFoamData(int foam_id, glm::vec3 new_origin)
{
    auto it = slots.find(foam_id);
    if (it == slots.end() || it->second.dead) return;
    const FoamSlot& s = it->second;
    const int n = s.active_count;
    if (n <= 0) return;
    k_reparent<<<grid_size(n), 256>>>(
        d_particle_positions + s.particle_offset,
        d_particle_aabbs     + s.particle_offset,
        n, new_origin);
    CUDA_CHECK(cudaGetLastError());
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

// Generic per-particle flag kernel.
// d_flags[i] = match_value   if active_ids[i] is found in id_set,
// d_flags[i] = 1-match_value otherwise.
// Set match_value=0 to flag matched particles for deletion (keep=1 by default).
// Set match_value=1 to flag matched particles for retention (keep=0 by default).
static __global__ void k_build_id_match_flags(
    const uint32_t* __restrict__ active_ids,
    const uint32_t* __restrict__ id_set,
    uint8_t*        __restrict__ d_flags,
    int             n_particles,
    int             n_ids,
    uint8_t         match_value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles) return;
    const uint32_t id = active_ids[i];
    uint8_t flag = 1 - match_value;
    for (int j = 0; j < n_ids; ++j) {
        if (id_set[j] == id) { flag = match_value; break; }
    }
    d_flags[i] = flag;
}

// Generic per-edge flag kernel.
// d_flags[i] = match_value   if coo_src[i] or coo_dst[i] is found in id_set,
// d_flags[i] = 1-match_value otherwise.
static __global__ void k_build_edge_id_match_flags(
    const uint32_t* __restrict__ coo_src,
    const uint32_t* __restrict__ coo_dst,
    const uint32_t* __restrict__ id_set,
    uint8_t*        __restrict__ d_flags,
    int             n_edges,
    int             n_ids,
    uint8_t         match_value)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_edges) return;
    const uint32_t src = coo_src[i];
    const uint32_t dst = coo_dst[i];
    uint8_t flag = 1 - match_value;
    for (int j = 0; j < n_ids; ++j) {
        if (id_set[j] == src || id_set[j] == dst) { flag = match_value; break; }
    }
    d_flags[i] = flag;
}

void GpuSlabAllocator::updateFoamData(int foam_id, const FoamUpdate& update)
{
    auto it = slots.find(foam_id);
    if (it == slots.end() || it->second.dead) return;

    const int n_dels    = static_cast<int>(update.particle_id_dels.size());
    const int n_ins     = static_cast<int>(update.particle_position_ins.size());
    const int n_coo_ins = static_cast<int>(update.coo_src_ins.size());
    if (n_dels == 0 && n_ins == 0 && n_coo_ins == 0) return;

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
        k_build_id_match_flags<<<grid_size(cur_count), 256>>>(
            d_active_ids + s.active_offset,
            d_del_ids, d_flags, cur_count, n_dels, /*match_value=*/0);
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

        // -----------------------------------------------------------------
        // COO edge deletion — compact both COO arrays, dropping any edge
        // whose src or dst ID appears in the deletion set.
        // -----------------------------------------------------------------
        const int cur_coo_count = s.coo_count;
        if (cur_coo_count > 0) {
            uint8_t* d_edge_flags = nullptr;
            CUDA_CHECK(cudaMalloc(&d_edge_flags, cur_coo_count * sizeof(uint8_t)));
            k_build_edge_id_match_flags<<<grid_size(cur_coo_count), 256>>>(
                d_coo_src + s.coo_offset,
                d_coo_dst + s.coo_offset,
                d_del_ids, d_edge_flags, cur_coo_count, n_dels, /*match_value=*/0);
            CUDA_CHECK(cudaGetLastError());

            uint32_t* d_coo_src_tmp = nullptr;
            uint32_t* d_coo_dst_tmp = nullptr;
            CUDA_CHECK(cudaMalloc(&d_coo_src_tmp, cur_coo_count * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_coo_dst_tmp, cur_coo_count * sizeof(uint32_t)));
            int* d_coo_selected = nullptr;
            CUDA_CHECK(cudaMalloc(&d_coo_selected, sizeof(int)));

            CUB_CALL(cub::DeviceSelect::Flagged,
                d_coo_src + s.coo_offset, d_edge_flags, d_coo_src_tmp, d_coo_selected, cur_coo_count);
            CUB_CALL(cub::DeviceSelect::Flagged,
                d_coo_dst + s.coo_offset, d_edge_flags, d_coo_dst_tmp, d_coo_selected, cur_coo_count);

            int new_coo_count = 0;
            CUDA_CHECK(cudaMemcpy(&new_coo_count, d_coo_selected, sizeof(int), cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaMemcpy(d_coo_src + s.coo_offset, d_coo_src_tmp,
                new_coo_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_coo_dst + s.coo_offset, d_coo_dst_tmp,
                new_coo_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            s.coo_count = new_coo_count;

            CUDA_CHECK(cudaFree(d_edge_flags));
            CUDA_CHECK(cudaFree(d_coo_src_tmp));
            CUDA_CHECK(cudaFree(d_coo_dst_tmp));
            CUDA_CHECK(cudaFree(d_coo_selected));
        }

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
            const int old_coo_offset      = s.coo_offset;
            const int old_coo_count       = s.coo_count;

            // Halve the stored (doubled) capacities back to the original counts
            // so that allocate() doubles them again correctly.
            // BVH and CSR node counts are derived from new_count (not the old
            // slot) since new_count may exceed 2 * old slot capacity.
            const int bvh_nodes = (new_count > 1) ? 2 * new_count - 1 : 1;
            const int csr_nodes = new_count + 1;
            const int csr_edges = s.csr_edge_capacity / 2;
            // Ensure the new COO slice can hold existing + incoming edges so
            // Phase 3 (COO insertion) won't need a second resize.
            const int coo_need  = old_coo_count + n_coo_ins;

            s.dead = true;
            allocate(foam_id, bvh_nodes, csr_nodes, csr_edges, new_count, coo_need);
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
            // Copy surviving COO edges from old slot to new slot.
            if (old_coo_count > 0) {
                CUDA_CHECK(cudaMemcpy(d_coo_src + ns.coo_offset,
                    d_coo_src + old_coo_offset, old_coo_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_coo_dst + ns.coo_offset,
                    d_coo_dst + old_coo_offset, old_coo_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            }
            ns.coo_count = old_coo_count;
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

    // -------------------------------------------------------------------------
    // Phase 3: COO edge insertions
    // -------------------------------------------------------------------------
    if (n_coo_ins > 0) {
        FoamSlot& s = slots.at(foam_id);
        const int new_coo_count = s.coo_count + n_coo_ins;

        if (new_coo_count > s.coo_capacity) {
            // COO slice is too small.  Tombstone and re-allocate a larger one,
            // preserving both the surviving particle data and COO edge data.
            const int old_particle_offset = s.particle_offset;
            const int old_active_offset   = s.active_offset;
            const int old_coo_offset      = s.coo_offset;
            const int old_coo_count       = s.coo_count;
            const int cur_particle_count  = s.active_count;

            const int bvh_nodes  = s.bvh_capacity      / 2;
            const int csr_nodes  = s.csr_node_capacity / 2;
            const int csr_edges  = s.csr_edge_capacity / 2;
            const int part_nodes = s.particle_capacity / 2;

            s.dead = true;
            allocate(foam_id, bvh_nodes, csr_nodes, csr_edges, part_nodes, new_coo_count);
            FoamSlot& ns = slots.at(foam_id);

            // Copy surviving particle data from old slot to new slot.
            if (cur_particle_count > 0) {
                CUDA_CHECK(cudaMemcpy(d_particle_aabbs        + ns.particle_offset,
                    d_particle_aabbs        + old_particle_offset, cur_particle_count * sizeof(AABB),      cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_particle_colors       + ns.particle_offset,
                    d_particle_colors       + old_particle_offset, cur_particle_count * sizeof(glm::vec4), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_particle_positions    + ns.particle_offset,
                    d_particle_positions    + old_particle_offset, cur_particle_count * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_particle_surface_mask + ns.particle_offset,
                    d_particle_surface_mask + old_particle_offset, cur_particle_count * sizeof(uint8_t),   cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_active_ids            + ns.active_offset,
                    d_active_ids            + old_active_offset,   cur_particle_count * sizeof(uint32_t),  cudaMemcpyDeviceToDevice));
            }
            ns.active_count = cur_particle_count;

            // Copy surviving COO edges from old slot to new slot.
            if (old_coo_count > 0) {
                CUDA_CHECK(cudaMemcpy(d_coo_src + ns.coo_offset,
                    d_coo_src + old_coo_offset, old_coo_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_coo_dst + ns.coo_offset,
                    d_coo_dst + old_coo_offset, old_coo_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            }
            ns.coo_count = old_coo_count;
        }

        // Append new COO edges immediately after the surviving ones.
        FoamSlot& sn = slots.at(foam_id);
        const int insert_coo_pos = sn.coo_offset + sn.coo_count;
        CUDA_CHECK(cudaMemcpy(d_coo_src + insert_coo_pos,
            update.coo_src_ins.data(), n_coo_ins * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coo_dst + insert_coo_pos,
            update.coo_dst_ins.data(), n_coo_ins * sizeof(uint32_t), cudaMemcpyHostToDevice));
        sn.coo_count = new_coo_count;
    }
}

// =============================================================================
// copyFoamData — D→D clone of one foam slot into a new slot.
// =============================================================================

void GpuSlabAllocator::copyFoamData(int src_foam_id, int dst_foam_id,
                                     const uint32_t* h_particle_ids,
                                     int n_ids)
{
    // Snapshot all src values before any allocate() call, since allocate()
    // can rehash the unordered_map and invalidate any reference into it.
    const FoamSlot src_snap = slots.at(src_foam_id);
    const int src_n_particles  = src_snap.active_count;
    const int src_n_coo        = src_snap.coo_count;
    const int src_bvh_off      = src_snap.bvh_offset;
    const int src_csr_node_off = src_snap.csr_node_offset;
    const int src_csr_edge_off = src_snap.csr_edge_offset;
    const int src_part_off     = src_snap.particle_offset;
    const int src_active_off   = src_snap.active_offset;
    const int src_coo_off      = src_snap.coo_offset;
    const int src_csr_edges    = src_snap.csr_edge_capacity / 2; // actual directed-edge count estimate

    if (h_particle_ids == nullptr || n_ids == 0) {
        // =====================================================================
        // Full copy — transfer all particle data, COO edges, BVH, and CSR.
        // =====================================================================
        const int bvh_nodes = (src_n_particles > 0) ? (2 * src_n_particles - 1) : 1;
        const int csr_nodes = src_n_particles + 1;
        allocate(dst_foam_id, bvh_nodes, csr_nodes, src_csr_edges,
                 src_n_particles, src_n_coo);
        const FoamSlot& dst = slots.at(dst_foam_id);

        // Particle arrays.
        if (src_n_particles > 0) {
            CUDA_CHECK(cudaMemcpy(d_particle_aabbs        + dst.particle_offset,
                d_particle_aabbs        + src_part_off,
                src_n_particles * sizeof(AABB),      cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_particle_colors       + dst.particle_offset,
                d_particle_colors       + src_part_off,
                src_n_particles * sizeof(glm::vec4), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_particle_positions    + dst.particle_offset,
                d_particle_positions    + src_part_off,
                src_n_particles * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_particle_surface_mask + dst.particle_offset,
                d_particle_surface_mask + src_part_off,
                src_n_particles * sizeof(uint8_t),   cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_active_ids            + dst.active_offset,
                d_active_ids            + src_active_off,
                src_n_particles * sizeof(uint32_t),  cudaMemcpyDeviceToDevice));
        }

        // COO edge arrays.
        if (src_n_coo > 0) {
            CUDA_CHECK(cudaMemcpy(d_coo_src + dst.coo_offset,
                d_coo_src + src_coo_off, src_n_coo * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_coo_dst + dst.coo_offset,
                d_coo_dst + src_coo_off, src_n_coo * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        }

        // BVH nodes.
        if (bvh_nodes > 0) {
            CUDA_CHECK(cudaMemcpy(d_bvh_nodes + dst.bvh_offset,
                d_bvh_nodes + src_bvh_off,
                bvh_nodes * sizeof(BVHNode), cudaMemcpyDeviceToDevice));
        }

        // CSR rowptr (n+1 entries, already biased by src's csr_edge_offset;
        // re-bias for the new offset so indices remain globally valid).
        if (csr_nodes > 0) {
            CUDA_CHECK(cudaMemcpy(d_csr_rowptr + dst.csr_node_offset,
                d_csr_rowptr + src_csr_node_off,
                csr_nodes * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            // Adjust bias: subtract src edge offset, add dst edge offset.
            const int bias_delta = dst.csr_edge_offset - src_csr_edge_off;
            if (bias_delta != 0) {
                k_bias_u32<<<grid_size(csr_nodes), 256>>>(
                    d_csr_rowptr + dst.csr_node_offset,
                    csr_nodes,
                    static_cast<uint32_t>(bias_delta));
                CUDA_CHECK(cudaGetLastError());
            }
        }

        // CSR colidx.
        if (src_csr_edges > 0) {
            CUDA_CHECK(cudaMemcpy(d_csr_colidx + dst.csr_edge_offset,
                d_csr_colidx + src_csr_edge_off,
                src_csr_edges * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        }

        FoamSlot& dst_mut = slots.at(dst_foam_id);
        dst_mut.active_count = src_n_particles;
        dst_mut.coo_count    = src_n_coo;

    } else {
        // =====================================================================
        // Conditional copy — filter by the supplied particle ID set.
        // =====================================================================

        // Upload ID set to device.
        uint32_t* d_id_set = nullptr;
        CUDA_CHECK(cudaMalloc(&d_id_set, n_ids * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_id_set, h_particle_ids,
            n_ids * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // -----------------------------------------------------------------
        // Step 1: compact particle arrays by ID membership.
        // -----------------------------------------------------------------
        uint8_t* d_part_flags = nullptr;
        CUDA_CHECK(cudaMalloc(&d_part_flags, src_n_particles * sizeof(uint8_t)));
        if (src_n_particles > 0) {
            k_build_id_match_flags<<<grid_size(src_n_particles), 256>>>(
                d_active_ids + src_active_off,
                d_id_set, d_part_flags, src_n_particles, n_ids, /*match_value=*/1);
            CUDA_CHECK(cudaGetLastError());
        }

        AABB*      d_aabb_tmp = nullptr;
        glm::vec4* d_col_tmp  = nullptr;
        glm::vec3* d_pos_tmp  = nullptr;
        uint8_t*   d_mask_tmp = nullptr;
        uint32_t*  d_ids_tmp  = nullptr;
        int*       d_n_sel    = nullptr;
        if (src_n_particles > 0) {
            CUDA_CHECK(cudaMalloc(&d_aabb_tmp, src_n_particles * sizeof(AABB)));
            CUDA_CHECK(cudaMalloc(&d_col_tmp,  src_n_particles * sizeof(glm::vec4)));
            CUDA_CHECK(cudaMalloc(&d_pos_tmp,  src_n_particles * sizeof(glm::vec3)));
            CUDA_CHECK(cudaMalloc(&d_mask_tmp, src_n_particles * sizeof(uint8_t)));
            CUDA_CHECK(cudaMalloc(&d_ids_tmp,  src_n_particles * sizeof(uint32_t)));
        }
        CUDA_CHECK(cudaMalloc(&d_n_sel, sizeof(int)));

        int n_surviving = 0;
        if (src_n_particles > 0) {
            CUB_CALL(cub::DeviceSelect::Flagged,
                d_particle_aabbs        + src_part_off,   d_part_flags, d_aabb_tmp, d_n_sel, src_n_particles);
            CUB_CALL(cub::DeviceSelect::Flagged,
                d_particle_colors       + src_part_off,   d_part_flags, d_col_tmp,  d_n_sel, src_n_particles);
            CUB_CALL(cub::DeviceSelect::Flagged,
                d_particle_positions    + src_part_off,   d_part_flags, d_pos_tmp,  d_n_sel, src_n_particles);
            CUB_CALL(cub::DeviceSelect::Flagged,
                d_particle_surface_mask + src_part_off,   d_part_flags, d_mask_tmp, d_n_sel, src_n_particles);
            CUB_CALL(cub::DeviceSelect::Flagged,
                d_active_ids            + src_active_off, d_part_flags, d_ids_tmp,  d_n_sel, src_n_particles);
            CUDA_CHECK(cudaMemcpy(&n_surviving, d_n_sel, sizeof(int), cudaMemcpyDeviceToHost));
        }

        // -----------------------------------------------------------------
        // Step 2: compact COO edges — keep if either endpoint is in the set.
        // -----------------------------------------------------------------
        uint32_t* d_coo_src_tmp = nullptr;
        uint32_t* d_coo_dst_tmp = nullptr;
        int n_surviving_edges = 0;
        if (src_n_coo > 0) {
            uint8_t* d_edge_flags = nullptr;
            int*     d_e_sel      = nullptr;
            CUDA_CHECK(cudaMalloc(&d_edge_flags, src_n_coo * sizeof(uint8_t)));
            CUDA_CHECK(cudaMalloc(&d_e_sel, sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_coo_src_tmp, src_n_coo * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_coo_dst_tmp, src_n_coo * sizeof(uint32_t)));

            k_build_edge_id_match_flags<<<grid_size(src_n_coo), 256>>>(
                d_coo_src + src_coo_off,
                d_coo_dst + src_coo_off,
                d_id_set, d_edge_flags, src_n_coo, n_ids, /*match_value=*/1);
            CUDA_CHECK(cudaGetLastError());

            CUB_CALL(cub::DeviceSelect::Flagged,
                d_coo_src + src_coo_off, d_edge_flags, d_coo_src_tmp, d_e_sel, src_n_coo);
            CUB_CALL(cub::DeviceSelect::Flagged,
                d_coo_dst + src_coo_off, d_edge_flags, d_coo_dst_tmp, d_e_sel, src_n_coo);
            CUDA_CHECK(cudaMemcpy(&n_surviving_edges, d_e_sel, sizeof(int), cudaMemcpyDeviceToHost));

            CUDA_CHECK(cudaFree(d_edge_flags));
            CUDA_CHECK(cudaFree(d_e_sel));
        }

        // -----------------------------------------------------------------
        // Step 3: allocate dst slot sized to surviving counts.
        //   BVH/CSR regions are reserved (caller must rebuild) but not filled.
        // -----------------------------------------------------------------
        const int bvh_nodes = (n_surviving > 0) ? (2 * n_surviving - 1) : 1;
        const int csr_nodes = n_surviving + 1;
        // Re-use the src foam's CSR edge density as a rough upper bound for
        // the new slot's CSR edge capacity.
        const int csr_edges_est = (src_n_particles > 0)
            ? static_cast<int>(static_cast<int64_t>(src_csr_edges) * n_surviving
                               / src_n_particles) + 1
            : 0;
        allocate(dst_foam_id, bvh_nodes, csr_nodes, csr_edges_est,
                 n_surviving, n_surviving_edges);
        FoamSlot& dst = slots.at(dst_foam_id);

        // Copy compacted particle arrays into the new slot.
        if (n_surviving > 0) {
            CUDA_CHECK(cudaMemcpy(d_particle_aabbs        + dst.particle_offset,
                d_aabb_tmp, n_surviving * sizeof(AABB),      cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_particle_colors       + dst.particle_offset,
                d_col_tmp,  n_surviving * sizeof(glm::vec4), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_particle_positions    + dst.particle_offset,
                d_pos_tmp,  n_surviving * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_particle_surface_mask + dst.particle_offset,
                d_mask_tmp, n_surviving * sizeof(uint8_t),   cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_active_ids            + dst.active_offset,
                d_ids_tmp,  n_surviving * sizeof(uint32_t),  cudaMemcpyDeviceToDevice));
        }

        // Copy compacted COO edges into the new slot.
        if (n_surviving_edges > 0) {
            CUDA_CHECK(cudaMemcpy(d_coo_src + dst.coo_offset,
                d_coo_src_tmp, n_surviving_edges * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_coo_dst + dst.coo_offset,
                d_coo_dst_tmp, n_surviving_edges * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        }

        FoamSlot& dst_mut = slots.at(dst_foam_id);
        dst_mut.active_count = n_surviving;
        dst_mut.coo_count    = n_surviving_edges;

        // Free temporaries.
        CUDA_CHECK(cudaFree(d_id_set));
        CUDA_CHECK(cudaFree(d_part_flags));
        CUDA_CHECK(cudaFree(d_n_sel));
        if (d_aabb_tmp)     CUDA_CHECK(cudaFree(d_aabb_tmp));
        if (d_col_tmp)      CUDA_CHECK(cudaFree(d_col_tmp));
        if (d_pos_tmp)      CUDA_CHECK(cudaFree(d_pos_tmp));
        if (d_mask_tmp)     CUDA_CHECK(cudaFree(d_mask_tmp));
        if (d_ids_tmp)      CUDA_CHECK(cudaFree(d_ids_tmp));
        if (d_coo_src_tmp)  CUDA_CHECK(cudaFree(d_coo_src_tmp));
        if (d_coo_dst_tmp)  CUDA_CHECK(cudaFree(d_coo_dst_tmp));
    }
}

} // namespace DynamicFoam::Sim2D