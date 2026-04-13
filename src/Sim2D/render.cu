// =============================================================================
// render.cu
// GPU ray tracer implementation for 2D foam simulation.
//
// Pipeline per frame:
//   Broadphase  — ray vs foam AABB
//   Narrowphase — ray vs per-foam BVH, interior particles masked out
//   Sort        — narrowphase hits ordered by (ray_idx, t)
//   Compaction  — run-length encode + exclusive scan → compact offsets
//   Exact       — generalized Voronoi slab test, one thread per hit ray
// =============================================================================

#include <cub/cub.cuh>
#include "dynamic_foam/Sim2D/render.cuh"
#include "dynamic_foam/Sim2D/components.h"
#include "dynamic_foam/Sim2D/cuda_utils.cuh"

namespace DynamicFoam::Sim2D {

// -----------------------------------------------------------------------------
// AABB transform kernel — one thread per foam.
// Reads the local-space AABB from the slab (indexed by foam_id), applies the
// foam's world transform (8-corner method), and writes the world-space AABB
// into world_aabbs_out[foam_id].  Runs before broadphase each frame so that
// no D→H→D copy is needed for geometry data.
// -----------------------------------------------------------------------------
__global__ void k_transform_foam_aabbs(
    const AABB*      local_aabbs,
    const glm::mat4* transforms,
    AABB*            world_aabbs_out,
    int              num_foams
) {
    int fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= num_foams) return;
    const AABB&      local = local_aabbs[fid];
    const glm::mat4& tx    = transforms[fid];
    const glm::vec3  mn    = local.min_pt, mx = local.max_pt;
    glm::vec3 c0   = glm::vec3(tx * glm::vec4(mn, 1.f));
    glm::vec3 wmin = c0, wmax = c0;
    for (int j = 1; j < 8; ++j) {
        glm::vec3 c(j & 1 ? mx.x : mn.x, j & 2 ? mx.y : mn.y, j & 4 ? mx.z : mn.z);
        c    = glm::vec3(tx * glm::vec4(c, 1.f));
        wmin = glm::min(wmin, c);
        wmax = glm::max(wmax, c);
    }
    world_aabbs_out[fid] = AABB(wmin, wmax);
}

// -----------------------------------------------------------------------------
// Broadphase — one thread per ray, iterates over all foam AABBs
// -----------------------------------------------------------------------------

__global__ void k_broadphase_collision(
    CameraParams      camera,
    int               width,
    int               height,
    const AABB*       foam_aabbs,
    const int*        foam_ids,
    BroadphaseHit*    hits,
    int*              hit_counter,
    int               num_rays,
    int               num_foams
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= num_rays) return;

    glm::vec3 origin, dir;
    generateRay(camera, ray_idx, width, height, origin, dir);
    const glm::vec3 inv_dir = 1.0f / dir;

    for (int i = 0; i < num_foams; ++i) {
        float t_min = 0.f, t_max = 1e30f, t_hit;
        if (foam_aabbs[i].intersect(origin, inv_dir, t_min, t_max, t_hit)) {
            int hit_idx = atomicAdd(hit_counter, 1);
            hits[hit_idx] = {ray_idx, foam_ids[i], t_hit};
        }
    }
}

// -----------------------------------------------------------------------------
// Narrowphase — one thread per broadphase hit, traverses per-foam BVH.
// Interior particles (surface_mask == 0) are skipped at the leaf.
// -----------------------------------------------------------------------------

__global__ void k_narrowphase_collision(
    CameraParams         camera,
    int                  width,
    int                  height,
    const BroadphaseHit* broadphase_hits,
    int                  num_broadphase_hits,
    const BVHNode*       bvh_nodes,
    const int*           bvh_offsets,
    const uint8_t*       surface_mask,
    const glm::mat4*     foam_inv_transforms,
    // foam_particle_offsets[fid] = start of foam fid's particles in the flat
    // position/color/surface_mask buffers. Used to bias prim_idx so that
    // NarrowphaseHit::particle_id is a global flat index.
    const int*           foam_particle_offsets,
    NarrowphaseHit*      narrowphase_hits,
    int*                 hit_counter
) {
    int hit_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hit_idx >= num_broadphase_hits) return;

    const BroadphaseHit b_hit   = broadphase_hits[hit_idx];
    const int           ray_idx = b_hit.ray_idx;
    const int           fid     = b_hit.foam_id;

    // Generate the world-space ray for this pixel, then transform into the
    // foam's local BVH space. BVH nodes were built in local particle coordinates.
    glm::vec3 world_origin, world_dir;
    generateRay(camera, ray_idx, width, height, world_origin, world_dir);
    const glm::mat4& inv_xform = foam_inv_transforms[fid];
    const glm::vec3  origin    = glm::vec3(inv_xform * glm::vec4(world_origin, 1.0f));
    const glm::vec3  dir       = glm::vec3(inv_xform * glm::vec4(world_dir,    0.0f));
    const glm::vec3  inv_dir   = 1.0f / dir;

    const BVHNode* foam_bvh      = &bvh_nodes[bvh_offsets[fid]];
    const int      particle_base = foam_particle_offsets[fid];

    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        const BVHNode& node = foam_bvh[stack[--stack_ptr]];

        float t_min = 0.f, t_max = 1e30f, t_hit;
        if (node.bbox.intersect(origin, inv_dir, t_min, t_max, t_hit)) {
            if (node.prim_idx >= 0) {
                // prim_idx is a foam-local sorted position. The surface_mask
                // and all flat particle buffers are indexed by global flat
                // index, so bias by particle_base before the mask read and
                // when writing the hit.
                const int global_pid = particle_base + node.prim_idx;
                if (surface_mask[global_pid]) {
                    int narrow_hit_idx = atomicAdd(hit_counter, 1);
                    narrowphase_hits[narrow_hit_idx] = {ray_idx, global_pid, fid, t_hit};
                }
            } else {
                if (stack_ptr < 62) {
                    stack[stack_ptr++] = node.right;
                    stack[stack_ptr++] = node.left;
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Pack (ray_idx, t) into a uint64 sort key.
// Upper 32 bits = ray_idx, lower 32 bits = __float_as_uint(t).
// __float_as_uint preserves ordering for positive floats, so a single radix
// sort over the packed key sorts by ray_idx first, then by t within each ray.
// -----------------------------------------------------------------------------

__global__ void k_pack_sort_key(
    const NarrowphaseHit* hits, uint64_t* keys, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t t_bits = __float_as_uint(hits[i].t);
    keys[i] = ((uint64_t)hits[i].ray_idx << 32) | (uint64_t)t_bits;
}

// -----------------------------------------------------------------------------
// Extract the upper 32 bits (ray_idx) from sorted packed keys, producing
// a sorted array of ray indices ready for run-length encoding.
// -----------------------------------------------------------------------------

__global__ void k_extract_ray_idx(
    const uint64_t* packed_keys, int* ray_indices, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ray_indices[i] = (int)(packed_keys[i] >> 32);
}

// -----------------------------------------------------------------------------
// Exact collision — one thread per ray that received narrowphase hits.
//
// Uses a flat concatenated CSR (all foams packed end-to-end). Each
// NarrowphaseHit carries a global flat particle_id (biased by
// foam_particle_offsets[fid] at narrowphase write time) and the foam_id it
// came from, so the kernel can recover the foam-local sorted index and look up
// the correct CSR neighbor run without any per-particle indirection table.
//
// Neighbor indices in csr_nbrs[] are foam-local sorted positions; adding
// foam_particle_offsets[fid] converts them to global flat indices.
//
// Voronoi slab test:
//   For each candidate surface particle P, intersect all bisector half-planes
//   defined by P and each neighbor N. The ray hits P's Voronoi cell if the
//   resulting t-interval [t_enter, t_exit] is non-empty and t_enter >= 0.
//   Candidates are visited in approximate front-to-back order (sorted by BVH t)
//   and we take the first cell that passes.
// -----------------------------------------------------------------------------
__global__ void k_exact_collision(
    CameraParams          camera,
    int                   width,
    int                   height,
    const NarrowphaseHit* narrowphase_hits,
    const int*            unique_ray_ids,
    const int*            ray_hit_offsets,
    const int*            ray_hit_counts,
    // Flat concatenated CSR — all foams packed end-to-end.
    // csr_node_offsets[csr_offsets[fid] + local_sorted_pos] gives the
    // neighbor run start; csr_nbrs entries are foam-local sorted positions.
    const uint32_t*       csr_node_offsets,
    const uint32_t*       csr_nbrs,
    // csr_offsets[fid]           = start of foam fid's node_offsets in csr_node_offsets
    // foam_particle_offsets[fid] = start of foam fid's particles in positions/colors
    const int*            csr_offsets,
    const int*            foam_particle_offsets,
    const glm::vec3*      particle_positions,
    const glm::vec4*      particle_colors,
    glm::vec4*            output_buffer,
    int                   num_unique,
    RenderOverlayParams   overlay
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_unique) return;

    const int ray_idx    = unique_ray_ids[i];
    const int hit_count  = ray_hit_counts[i];
    const int hit_offset = ray_hit_offsets[i];

    glm::vec3 origin, dir;
    generateRay(camera, ray_idx, width, height, origin, dir);

    float    best_t        = 1e30f;
    int      best_particle = -1;
    int      best_foam_id  = -1;
    uint32_t best_enter_k  = ~0u; // CSR index of the neighbor whose bisector produced t_enter

    for (int j = 0; j < hit_count; ++j) {
        const NarrowphaseHit hit   = narrowphase_hits[hit_offset + j];
        const int            fid   = hit.foam_id;

        // particle_id is already a global flat index (biased by foam offset
        // when the narrowphase hit was written).
        const int       pid   = hit.particle_id;
        const glm::vec3 p_pos = particle_positions[pid];

        // Recover foam-local sorted position from the global flat index.
        const int fp_offset  = foam_particle_offsets[fid];
        const int local_idx  = pid - fp_offset;

        // Look up neighbor run in the flat CSR.
        const int      csr_base  = csr_offsets[fid];
        const uint32_t adj_start = csr_node_offsets[csr_base + local_idx];
        const uint32_t adj_end   = csr_node_offsets[csr_base + local_idx + 1];

        float    t_enter = 0.f;
        float    t_exit  = 1e30f;
        bool     valid   = true;
        uint32_t enter_k = ~0u; // CSR index that gave t_enter for this candidate

        for (uint32_t k = adj_start; k < adj_end; ++k) {
            // csr_nbrs[k] is a foam-local sorted-position index.
            // Bias by foam_particle_offsets to get the global flat index.
            const int       nid   = fp_offset + (int)csr_nbrs[k];
            const glm::vec3 n_pos = particle_positions[nid];

            const glm::vec3 plane_normal = n_pos - p_pos;
            const glm::vec3 midpoint     = (p_pos + n_pos) * 0.5f;

            const float denom = glm::dot(dir, plane_normal);
            const float numer = glm::dot(midpoint - origin, plane_normal);

            if (fabsf(denom) < 1e-6f) {
                if (numer < 0.f) { valid = false; break; }
            } else {
                const float t_plane = numer / denom;
                // Constraint: t * denom <= numer  (ray must stay on P's side)
                // denom > 0 → t <= t_plane → upper bound on t → tighten t_exit
                // denom < 0 → t >= t_plane → lower bound on t → tighten t_enter
                if (denom > 0.f) {
                    t_exit = fminf(t_exit, t_plane);
                } else {
                    if (t_plane > t_enter) {
                        t_enter = t_plane;
                        enter_k = k; // record which neighbor is the entry face
                    }
                }
                if (t_enter > t_exit) { valid = false; break; }
            }
        }

        if (valid && t_enter <= t_exit && t_enter < best_t) {
            best_t        = t_enter;
            best_particle = pid;
            best_foam_id  = fid;
            best_enter_k  = enter_k;
        }
    }

    if (best_particle >= 0) {
        glm::vec4 out_color = particle_colors[best_particle];

        if (overlay.show_edges || overlay.show_centers) {
            const glm::vec3 p_hit = origin + best_t * dir;
            const glm::vec3 p_pos = particle_positions[best_particle];

            // --- Edge overlay ---
            // Compute the 3D signed distance from p_hit to each Voronoi bisector
            // plane. The margin for neighbor N is:
            //   dot(p_hit − midpt, N−P) / |N−P|
            // which is negative when p_hit is on P's side (interior) and
            // approaches 0 near the bisector face.
            //
            // p_hit = origin + t_enter * dir lies exactly on the entry bisector
            // plane by construction, so that neighbor always gives margin = 0
            // and must be skipped. best_enter_k records which CSR index produced
            // t_enter in the slab test, so we can exclude exactly that neighbor.
            if (overlay.show_edges) {
                const int      fp_offset = foam_particle_offsets[best_foam_id];
                const int      local_idx = best_particle - fp_offset;
                const int      csr_base  = csr_offsets[best_foam_id];
                const uint32_t adj_start = csr_node_offsets[csr_base + local_idx];
                const uint32_t adj_end   = csr_node_offsets[csr_base + local_idx + 1];

                float max_margin = -1e30f;
                for (uint32_t k = adj_start; k < adj_end; ++k) {
                    // Skip the entry bisector — its margin is 0 by construction.
                    if (k == best_enter_k) continue;
                    const int       nid    = fp_offset + (int)csr_nbrs[k];
                    const glm::vec3 n_pos  = particle_positions[nid];
                    const glm::vec3 diff3d = n_pos - p_pos;
                    const float     len3d  = glm::length(diff3d);
                    if (len3d < 1e-6f) continue;
                    const glm::vec3 midpt3d = (p_pos + n_pos) * 0.5f;
                    // Negative = inside P's cell, 0 = on face, positive = outside.
                    const float margin = glm::dot(p_hit - midpt3d, diff3d) / len3d;
                    max_margin = fmaxf(max_margin, margin);
                }
                // max_margin > -edge_threshold: p_hit is within edge_threshold
                // world units of the nearest non-entry Voronoi face.
                if (max_margin > -overlay.edge_threshold)
                    out_color = overlay.edge_color;
            }

            // --- Center overlay (drawn on top of edge overlay) ---
            // Calculate distance between hit site and midpoint to neighbor is within overlay.center_radius
            if (overlay.show_centers && best_enter_k != ~0u) {
                const int       fp_offset_c = foam_particle_offsets[best_foam_id];
                const int       nid         = fp_offset_c + (int)csr_nbrs[best_enter_k];
                const glm::vec3 n_pos_c     = particle_positions[nid];
                const glm::vec3 midpoint    = (p_pos + n_pos_c) * 0.5f;
                if (glm::length(p_hit - midpoint) < overlay.center_radius)
                    out_color = overlay.center_color;
            }
        }

        output_buffer[ray_idx] = out_color;
    }
}

// =============================================================================
// Render host methods
// =============================================================================

Render::Render() {
    CUDA_CHECK(cudaMalloc(&d_broadphase_hit_counter_,  sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_narrowphase_hit_counter_, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_num_unique_,              sizeof(int)));
}

Render::~Render() {
    free_device_memory();
}

void Render::update(
    const GpuSlabAllocator&                   slab,
    const std::unordered_map<int, glm::mat4>& foamTransforms,
    const CameraParams&                       camera,
    const glm::ivec2&                         windowSize,
    const RenderOverlayParams&                overlay
) {
    const int num_rays  = windowSize.x * windowSize.y;
    const int num_foams = slab.num_foams; // authoritative count from slab

    // ------------------------------------------------------------------
    // Collect live foam IDs (cheap host-side metadata walk; no geometry).
    // ------------------------------------------------------------------

    std::vector<int> h_foam_ids;
    h_foam_ids.reserve(slab.slots.size());
    for (const auto& [id, slot] : slab.slots) {
        if (!slot.dead && id >= 0 && id < num_foams)
            h_foam_ids.push_back(id);
    }
    const int num_visible_foams = static_cast<int>(h_foam_ids.size());

    cuda_realloc_if_needed(&d_foam_aabbs_, &cap_foams_,    num_foams);
    cuda_realloc_if_needed(&d_foam_ids_,   &cap_foam_ids_, num_visible_foams);
    CUDA_CHECK(cudaMemcpy(d_foam_ids_, h_foam_ids.data(),
                          num_visible_foams * sizeof(int), cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // Per-foam transforms (per-frame: dynamic foams rotate/translate).
    // Upload both the forward transform (for AABB) and its inverse (for rays).
    // ------------------------------------------------------------------

    std::vector<glm::mat4> h_foam_transforms    (num_foams, glm::mat4(1.0f));
    std::vector<glm::mat4> h_foam_inv_transforms(num_foams, glm::mat4(1.0f));
    for (const auto& [id, mat] : foamTransforms) {
        if (id >= 0 && id < num_foams) {
            h_foam_transforms[id]     = mat;
            h_foam_inv_transforms[id] = glm::inverse(mat);
        }
    }

    cuda_realloc_if_needed(&d_foam_transforms_,     &cap_foam_transforms_,     num_foams);
    cuda_realloc_if_needed(&d_foam_inv_transforms_, &cap_foam_inv_transforms_, num_foams);
    CUDA_CHECK(cudaMemcpy(d_foam_transforms_, h_foam_transforms.data(),
                          num_foams * sizeof(glm::mat4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_foam_inv_transforms_, h_foam_inv_transforms.data(),
                          num_foams * sizeof(glm::mat4), cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // World-space AABB computation — entirely on GPU, no D→H copy.
    // k_transform_foam_aabbs reads local AABBs from the slab and writes
    // world-space results into d_foam_aabbs_, ready for broadphase.
    // ------------------------------------------------------------------

    if (num_foams > 0) {
        k_transform_foam_aabbs<<<grid_size(num_foams), 256>>>(
            slab.d_foam_aabbs, d_foam_transforms_,
            d_foam_aabbs_, num_foams);
        CUDA_CHECK(cudaGetLastError());
    }

    // ------------------------------------------------------------------
    // Broadphase
    // ------------------------------------------------------------------

    const size_t max_broadphase_hits = (size_t)num_rays * num_visible_foams;
    cuda_realloc_if_needed(&d_broadphase_hits_, &cap_broadphase_hits_,
                           max_broadphase_hits);
    CUDA_CHECK(cudaMemset(d_broadphase_hit_counter_, 0, sizeof(int)));

    k_broadphase_collision<<<grid_size(num_rays), 256>>>(
        camera, windowSize.x, windowSize.y,
        d_foam_aabbs_,  d_foam_ids_,
        d_broadphase_hits_, d_broadphase_hit_counter_,
        num_rays, num_visible_foams
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_broadphase_hits = 0;
    CUDA_CHECK(cudaMemcpy(&num_broadphase_hits, d_broadphase_hit_counter_,
                          sizeof(int), cudaMemcpyDeviceToHost));
    if (num_broadphase_hits == 0) return;

    // ------------------------------------------------------------------
    // Narrowphase — surface particles only; BVH/mask live in slab
    // ------------------------------------------------------------------

    const size_t max_narrowphase_hits = (size_t)num_broadphase_hits * 32;
    cuda_realloc_if_needed(&d_narrowphase_hits_, &cap_narrowphase_hits_,
                           max_narrowphase_hits);
    CUDA_CHECK(cudaMemset(d_narrowphase_hit_counter_, 0, sizeof(int)));

    k_narrowphase_collision<<<grid_size(num_broadphase_hits), 256>>>(
        camera, windowSize.x, windowSize.y,
        d_broadphase_hits_, num_broadphase_hits,
        slab.d_bvh_nodes,        slab.d_foam_bvh_start,
        slab.d_particle_surface_mask,
        d_foam_inv_transforms_,
        slab.d_foam_particle_start,
        d_narrowphase_hits_, d_narrowphase_hit_counter_
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_narrowphase_hits = 0;
    CUDA_CHECK(cudaMemcpy(&num_narrowphase_hits, d_narrowphase_hit_counter_,
                          sizeof(int), cudaMemcpyDeviceToHost));
    if (num_narrowphase_hits == 0) return;

    // ------------------------------------------------------------------
    // Sort narrowphase hits by (ray_idx, t) via packed uint64 radix sort
    // ------------------------------------------------------------------

    cuda_realloc_if_needed(&d_sort_keys_in_,  &cap_sort_keys_,     num_narrowphase_hits);
    cuda_realloc_if_needed(&d_sort_keys_out_, &cap_sort_keys_out_, num_narrowphase_hits);
    cuda_realloc_if_needed(&d_hits_sorted_,   &cap_hits_sorted_,   num_narrowphase_hits);

    k_pack_sort_key<<<grid_size(num_narrowphase_hits), 256>>>(
        d_narrowphase_hits_, d_sort_keys_in_, num_narrowphase_hits);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUB_CALL(cub::DeviceRadixSort::SortPairs,
             d_sort_keys_in_,     d_sort_keys_out_,
             d_narrowphase_hits_, d_hits_sorted_,
             num_narrowphase_hits);

    // ------------------------------------------------------------------
    // Extract ray indices, run-length encode → per-ray hit counts
    // ------------------------------------------------------------------

    cuda_realloc_if_needed(&d_ray_idx_keys_,   &cap_rle_,            num_narrowphase_hits);
    cuda_realloc_if_needed(&d_unique_ray_ids_, &cap_unique_ray_ids_, num_narrowphase_hits);
    cuda_realloc_if_needed(&d_rle_counts_,     &cap_rle_counts_,     num_narrowphase_hits);
    if (!d_num_unique_) { CUDA_CHECK(cudaMalloc(&d_num_unique_, sizeof(int))); }

    k_extract_ray_idx<<<grid_size(num_narrowphase_hits), 256>>>(
        d_sort_keys_out_, d_ray_idx_keys_, num_narrowphase_hits);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUB_CALL(cub::DeviceRunLengthEncode::Encode,
             d_ray_idx_keys_,
             d_unique_ray_ids_, d_rle_counts_, d_num_unique_,
             num_narrowphase_hits);

    int num_unique = 0;
    CUDA_CHECK(cudaMemcpy(&num_unique, d_num_unique_,
                          sizeof(int), cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------
    // Exclusive scan over per-ray hit counts → offsets into sorted hits
    // ------------------------------------------------------------------

    cuda_realloc_if_needed(&d_ray_hit_offsets_, &cap_ray_hit_offsets_, num_unique);

    CUB_CALL(cub::DeviceScan::ExclusiveSum,
             d_rle_counts_, d_ray_hit_offsets_, num_unique);

    // ------------------------------------------------------------------
    // Output buffer
    // ------------------------------------------------------------------

    cuda_realloc_if_needed(&d_output_buffer_, &cap_output_buffer_, num_rays);
    CUDA_CHECK(cudaMemset(d_output_buffer_, 0, num_rays * sizeof(glm::vec4)));

    // ------------------------------------------------------------------
    // Exact Voronoi slab test — CSR and particle data live in slab
    // ------------------------------------------------------------------

    k_exact_collision<<<grid_size(num_unique), 256>>>(
        camera, windowSize.x, windowSize.y,
        d_hits_sorted_,
        d_unique_ray_ids_,
        d_ray_hit_offsets_,
        d_rle_counts_,
        slab.d_csr_rowptr,
        slab.d_csr_colidx,
        slab.d_foam_rowptr_start,
        slab.d_foam_particle_start,
        slab.d_particle_positions,
        slab.d_particle_colors,
        d_output_buffer_,
        num_unique,
        overlay
    );
    CUDA_CHECK(cudaDeviceSynchronize());
}

// -----------------------------------------------------------------------------
// Destructor
// -----------------------------------------------------------------------------

void Render::free_device_memory() {
    auto f = [](void* p) { if (p) cudaFree(p); };

    f(d_foam_aabbs_);                d_foam_aabbs_                = nullptr;
    f(d_foam_ids_);                  d_foam_ids_                  = nullptr;
    f(d_foam_transforms_);           d_foam_transforms_           = nullptr;
    f(d_broadphase_hits_);           d_broadphase_hits_           = nullptr;
    f(d_broadphase_hit_counter_);    d_broadphase_hit_counter_    = nullptr;
    f(d_narrowphase_hits_);          d_narrowphase_hits_          = nullptr;
    f(d_narrowphase_hit_counter_);   d_narrowphase_hit_counter_   = nullptr;
    f(d_sort_keys_in_);              d_sort_keys_in_              = nullptr;
    f(d_sort_keys_out_);             d_sort_keys_out_             = nullptr;
    f(d_hits_sorted_);               d_hits_sorted_               = nullptr;
    f(d_ray_idx_keys_);              d_ray_idx_keys_              = nullptr;
    f(d_unique_ray_ids_);            d_unique_ray_ids_            = nullptr;
    f(d_rle_counts_);                d_rle_counts_                = nullptr;
    f(d_num_unique_);                d_num_unique_                = nullptr;
    f(d_ray_hit_offsets_);           d_ray_hit_offsets_           = nullptr;
    f(d_foam_inv_transforms_);       d_foam_inv_transforms_       = nullptr;
    f(d_output_buffer_);             d_output_buffer_             = nullptr;
}

} // namespace DynamicFoam::Sim2D
