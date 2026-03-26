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
    int                   num_unique
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_unique) return;

    const int ray_idx    = unique_ray_ids[i];
    const int hit_count  = ray_hit_counts[i];
    const int hit_offset = ray_hit_offsets[i];

    glm::vec3 origin, dir;
    generateRay(camera, ray_idx, width, height, origin, dir);

    float best_t        = 1e30f;
    int   best_particle = -1;

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

        float t_enter = 0.f;
        float t_exit  = 1e30f;
        bool  valid   = true;

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
                if (denom > 0.f) t_exit  = fminf(t_exit,  t_plane);
                else             t_enter = fmaxf(t_enter, t_plane);
                if (t_enter > t_exit) { valid = false; break; }
            }
        }

        if (valid && t_enter <= t_exit && t_enter < best_t) {
            best_t        = t_enter;
            best_particle = pid;
        }
    }

    if (best_particle >= 0)
        output_buffer[ray_idx] = particle_colors[best_particle];
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
    const std::unordered_map<int, AABB>&                         foamAABBs,
    const std::unordered_map<int, BVH>&                          foamBVHs,
    const std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists,
    const entt::registry&                                        particleRegistry,
    const std::unordered_map<int, glm::mat4>&                    foamTransforms,
    const CameraParams&                                          camera,
    const glm::ivec2&                                            windowSize
) {
    const int num_rays  = windowSize.x * windowSize.y;
    const int num_foams = static_cast<int>(foamAABBs.size());

    // ------------------------------------------------------------------
    // Flat particle buffers + adjacency
    //
    // All foams are concatenated in foam iteration order. For each foam we
    // walk its ordered node list (getOrderedNodeIds), which gives the same
    // order used by buildGPUAdjacencyList when no Morton sort is provided.
    // This guarantees that foam-local sorted position i maps to global flat
    // index (foam_particle_offsets[fid] + i) with no extra indirection.
    //
    // foam_particle_offsets[fid] — start of foam fid in the flat arrays.
    // csr_offsets[fid]           — start of foam fid's node_offsets block
    //                              in the flat csr_node_offsets array.
    //
    // Rebuilt in full every frame (positions always move; colors/mask only
    // on topology change, but at <=100k particles the memcpy is cheap).
    // ------------------------------------------------------------------

    // Build CSR, renderer is only place in application where GPU CSR is needed
    for (const auto& [foam_id, adj] : foamAdjacencyLists) {
        auto& gpu_adj = foam_gpu_adj_[foam_id];
        if (gpu_adj.num_nodes == 0)
            adj.buildGPUAdjacencyList(gpu_adj);
    }

    // Compute per-foam offsets (prefix sum over particle counts).
    std::vector<int> h_foam_particle_offsets(num_foams, 0);
    std::vector<int> h_csr_offsets(num_foams, 0);
    int total_particles = 0;
    int total_csr_nodes = 0;
    {
        int poffset = 0;
        int coffset = 0;
        for (const auto& [foam_id, adj] : foamAdjacencyLists) {
            h_foam_particle_offsets[foam_id] = poffset;
            h_csr_offsets[foam_id]           = coffset;
            const uint32_t n = adj.nodeCount();
            poffset += static_cast<int>(n);
            coffset += static_cast<int>(n) + 1; // node_offsets has N+1 entries
        }
        total_particles = poffset;
        total_csr_nodes = coffset;
    }

    // Build flat host buffers by iterating each foam's ordered node list.
    std::vector<glm::vec3> h_positions(total_particles,    glm::vec3(0.f));
    std::vector<glm::vec4> h_colors(total_particles,       glm::vec4(1.f));
    std::vector<uint8_t>   h_surface_mask(total_particles, 0);
    std::vector<uint32_t>  h_csr_node_offsets(total_csr_nodes, 0);
    std::vector<uint32_t>  h_csr_nbrs;
    h_csr_nbrs.reserve(total_particles * 6); // rough upper bound

    {
        for (const auto& [foam_id, adj] : foamAdjacencyLists) {
            const int pbase  = h_foam_particle_offsets[foam_id];
            const int cbase  = h_csr_offsets[foam_id];
            const auto& gpu  = foam_gpu_adj_[foam_id];
            const auto nodes = adj.getOrderedNodeIds(); // host-side ordered list

            // Positions, colors, surface mask — in sorted order.
            for (int li = 0; li < static_cast<int>(nodes.size()); ++li) {
                const entt::entity e = nodes[li];
                const int gidx = pbase + li;

                if (const auto* p = particleRegistry.try_get<ParticleWorldPosition>(e))
                    h_positions[gidx] = p->value;
                if (const auto* c = particleRegistry.try_get<ParticleColor>(e))
                    if (const auto* o = particleRegistry.try_get<ParticleOpacity>(e))
                        h_colors[gidx] = glm::vec4(c->rgb, o->value);
                if (particleRegistry.all_of<Surface>(e))
                    h_surface_mask[gidx] = 1;
            }

            // CSR node_offsets — pull from GPU, fix up nbr base offset into
            // h_csr_nbrs, then append nbrs.
            if (gpu.num_nodes > 0) {
                // Download node_offsets (N+1 entries) from this foam's GPU CSR.
                std::vector<uint32_t> local_offsets(gpu.num_nodes + 1);
                CUDA_CHECK(cudaMemcpy(local_offsets.data(), gpu.node_offsets,
                    (gpu.num_nodes + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

                // The local offsets are relative to the start of this foam's
                // nbrs block. Bias them by the current size of h_csr_nbrs so
                // they index correctly into the global flat nbrs array.
                const uint32_t nbrs_base = static_cast<uint32_t>(h_csr_nbrs.size());
                for (uint32_t ni = 0; ni <= gpu.num_nodes; ++ni)
                    h_csr_node_offsets[cbase + ni] = local_offsets[ni] + nbrs_base;

                // Download and append this foam's nbrs.
                std::vector<uint32_t> local_nbrs(gpu.num_edges);
                CUDA_CHECK(cudaMemcpy(local_nbrs.data(), gpu.nbrs,
                    gpu.num_edges * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                h_csr_nbrs.insert(h_csr_nbrs.end(), local_nbrs.begin(), local_nbrs.end());
            }
        }
    }

    // Upload flat particle buffers.
    cuda_realloc_if_needed(&d_particle_positions_, &cap_particles_,       total_particles);
    cuda_realloc_if_needed(&d_particle_colors_,    &cap_particle_colors_, total_particles);
    cuda_realloc_if_needed(&d_surface_mask_,       &cap_surface_mask_, total_particles);
    CUDA_CHECK(cudaMemcpy(d_particle_positions_, h_positions.data(),
                          total_particles * sizeof(glm::vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particle_colors_, h_colors.data(),
                          total_particles * sizeof(glm::vec4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_surface_mask_, h_surface_mask.data(),
                          total_particles * sizeof(uint8_t),   cudaMemcpyHostToDevice));

    // Upload flat CSR.
    const size_t nbrs_count = h_csr_nbrs.size();
    cuda_realloc_if_needed(&d_csr_node_offsets_, &cap_csr_node_offsets_, total_csr_nodes);
    cuda_realloc_if_needed(&d_csr_nbrs_,         &cap_csr_nbrs_,         nbrs_count);
    CUDA_CHECK(cudaMemcpy(d_csr_node_offsets_, h_csr_node_offsets.data(),
                          total_csr_nodes * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_nbrs_, h_csr_nbrs.data(),
                          nbrs_count * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Upload offset tables.
    cuda_realloc_if_needed(&d_foam_particle_offsets_, &cap_foam_offsets_, num_foams);
    cuda_realloc_if_needed(&d_csr_offsets_,           &cap_csr_offsets_,  num_foams);
    CUDA_CHECK(cudaMemcpy(d_foam_particle_offsets_, h_foam_particle_offsets.data(),
                          num_foams * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_offsets_, h_csr_offsets.data(),
                          num_foams * sizeof(int), cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // Foam AABBs
    // ------------------------------------------------------------------

    std::vector<AABB> h_foam_aabbs(num_foams);
    std::vector<int>  h_foam_ids(num_foams);
    {
        int i = 0;
        for (const auto& [id, aabb] : foamAABBs) {
            h_foam_aabbs[i] = aabb;
            h_foam_ids[i]   = id;
            ++i;
        }
    }

    cuda_realloc_if_needed(&d_foam_aabbs_, &cap_foams_,    num_foams);
    cuda_realloc_if_needed(&d_foam_ids_,   &cap_foam_ids_, num_foams);
    CUDA_CHECK(cudaMemcpy(d_foam_aabbs_, h_foam_aabbs.data(),
                          num_foams * sizeof(AABB), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_foam_ids_,   h_foam_ids.data(),
                          num_foams * sizeof(int),  cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // Per-foam inverse transforms
    // ------------------------------------------------------------------

    std::vector<glm::mat4> h_foam_inv_transforms(num_foams, glm::mat4(1.0f));
    for (const auto& [id, mat] : foamTransforms) {
        if (id >= 0 && id < num_foams)
            h_foam_inv_transforms[id] = glm::inverse(mat);
    }

    cuda_realloc_if_needed(&d_foam_inv_transforms_, &cap_foam_inv_transforms_, num_foams);
    CUDA_CHECK(cudaMemcpy(d_foam_inv_transforms_, h_foam_inv_transforms.data(),
                          num_foams * sizeof(glm::mat4), cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // BVH nodes consolidated into a single flat buffer
    // ------------------------------------------------------------------

    std::vector<BVHNode> all_bvh_nodes;
    std::vector<int>     h_bvh_offsets(num_foams + 1, 0);
    {
        int current_offset = 0;
        for (const auto& [id, bvh] : foamBVHs) {
            h_bvh_offsets[id] = current_offset;
            const int num_nodes = (bvh.num_primitives() > 1)
                                  ? 2 * bvh.num_primitives() - 1 : 1;
            std::vector<BVHNode> host_nodes(num_nodes);
            CUDA_CHECK(cudaMemcpy(host_nodes.data(), bvh.export_nodes(),
                                  num_nodes * sizeof(BVHNode),
                                  cudaMemcpyDeviceToHost));
            all_bvh_nodes.insert(all_bvh_nodes.end(),
                                 host_nodes.begin(), host_nodes.end());
            current_offset += num_nodes;
        }
    }

    cuda_realloc_if_needed(&d_bvh_nodes_,   &cap_bvh_nodes_,   all_bvh_nodes.size());
    cuda_realloc_if_needed(&d_bvh_offsets_, &cap_bvh_offsets_, h_bvh_offsets.size());
    CUDA_CHECK(cudaMemcpy(d_bvh_nodes_, all_bvh_nodes.data(),
                          all_bvh_nodes.size() * sizeof(BVHNode),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bvh_offsets_, h_bvh_offsets.data(),
                          h_bvh_offsets.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // Broadphase
    // ------------------------------------------------------------------

    const size_t max_broadphase_hits = (size_t)num_rays * num_foams;
    cuda_realloc_if_needed(&d_broadphase_hits_, &cap_broadphase_hits_,
                           max_broadphase_hits);
    CUDA_CHECK(cudaMemset(d_broadphase_hit_counter_, 0, sizeof(int)));

    k_broadphase_collision<<<grid_size(num_rays), 256>>>(
        camera, windowSize.x, windowSize.y,
        d_foam_aabbs_,  d_foam_ids_,
        d_broadphase_hits_, d_broadphase_hit_counter_,
        num_rays, num_foams
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_broadphase_hits = 0;
    CUDA_CHECK(cudaMemcpy(&num_broadphase_hits, d_broadphase_hit_counter_,
                          sizeof(int), cudaMemcpyDeviceToHost));
    if (num_broadphase_hits == 0) return;

    // ------------------------------------------------------------------
    // Narrowphase — surface particles only
    // ------------------------------------------------------------------

    const size_t max_narrowphase_hits = (size_t)num_broadphase_hits * 32;
    cuda_realloc_if_needed(&d_narrowphase_hits_, &cap_narrowphase_hits_,
                           max_narrowphase_hits);
    CUDA_CHECK(cudaMemset(d_narrowphase_hit_counter_, 0, sizeof(int)));

    k_narrowphase_collision<<<grid_size(num_broadphase_hits), 256>>>(
        camera, windowSize.x, windowSize.y,
        d_broadphase_hits_, num_broadphase_hits,
        d_bvh_nodes_,   d_bvh_offsets_,
        d_surface_mask_,
        d_foam_inv_transforms_,
        d_foam_particle_offsets_,
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
    cuda_realloc_if_needed(&d_hits_sorted_,   &cap_hits_sorted_, num_narrowphase_hits);

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

    cuda_realloc_if_needed(&d_ray_idx_keys_,   &cap_rle_,        num_narrowphase_hits);
    cuda_realloc_if_needed(&d_unique_ray_ids_, &cap_unique_ray_ids_, num_narrowphase_hits);
    cuda_realloc_if_needed(&d_rle_counts_,     &cap_rle_counts_, num_narrowphase_hits);
    // d_num_unique_ is a single device int — allocate once on first use.
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
    // Exact Voronoi slab test
    // ------------------------------------------------------------------

    k_exact_collision<<<grid_size(num_unique), 256>>>(
        camera, windowSize.x, windowSize.y,
        d_hits_sorted_,
        d_unique_ray_ids_,
        d_ray_hit_offsets_,
        d_rle_counts_,
        d_csr_node_offsets_,
        d_csr_nbrs_,
        d_csr_offsets_,
        d_foam_particle_offsets_,
        d_particle_positions_,
        d_particle_colors_,
        d_output_buffer_,
        num_unique
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
    f(d_bvh_nodes_);                 d_bvh_nodes_                 = nullptr;
    f(d_bvh_offsets_);               d_bvh_offsets_               = nullptr;
    f(d_surface_mask_);              d_surface_mask_              = nullptr;
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
    f(d_particle_positions_);        d_particle_positions_        = nullptr;
    f(d_particle_colors_);           d_particle_colors_           = nullptr;
    f(d_csr_node_offsets_);          d_csr_node_offsets_          = nullptr;
    f(d_csr_nbrs_);                  d_csr_nbrs_                  = nullptr;
    f(d_csr_offsets_);               d_csr_offsets_               = nullptr;
    f(d_foam_particle_offsets_);     d_foam_particle_offsets_     = nullptr;
    f(d_foam_inv_transforms_);       d_foam_inv_transforms_       = nullptr;
    f(d_output_buffer_);             d_output_buffer_             = nullptr;

    for (auto& [id, gpu_adj] : foam_gpu_adj_)
        gpu_adj.free();
    foam_gpu_adj_.clear();
}

} // namespace DynamicFoam::Sim2D
