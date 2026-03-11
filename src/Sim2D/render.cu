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
#include "dynamic_foam/sim2d/render.cuh"
#include "dynamic_foam/sim2d/components.h"
#include "dynamic_foam/cuda_utils.cuh"

namespace DynamicFoam::Sim2D {

// -----------------------------------------------------------------------------
// Broadphase — one thread per ray, iterates over all foam AABBs
// -----------------------------------------------------------------------------

__global__ void k_broadphase_collision(
    const glm::vec3*  ray_origins,
    const glm::vec3*  ray_dirs,
    const AABB*       foam_aabbs,
    const int*        foam_ids,
    BroadphaseHit*    hits,
    int*              hit_counter,
    int               num_rays,
    int               num_foams
) {
    int ray_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_idx >= num_rays) return;

    const glm::vec3 origin  = ray_origins[ray_idx];
    const glm::vec3 dir     = ray_dirs[ray_idx];
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
    const glm::vec3*     ray_origins,
    const glm::vec3*     ray_dirs,
    const BroadphaseHit* broadphase_hits,
    int                  num_broadphase_hits,
    const BVHNode*       bvh_nodes,
    const int*           bvh_offsets,
    const uint8_t*       surface_mask,
    NarrowphaseHit*      narrowphase_hits,
    int*                 hit_counter
) {
    int hit_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (hit_idx >= num_broadphase_hits) return;

    const BroadphaseHit b_hit   = broadphase_hits[hit_idx];
    const int           ray_idx = b_hit.ray_idx;

    const glm::vec3 origin  = ray_origins[ray_idx];
    const glm::vec3 dir     = ray_dirs[ray_idx];
    const glm::vec3 inv_dir = 1.0f / dir;

    const BVHNode* foam_bvh = &bvh_nodes[bvh_offsets[b_hit.foam_id]];

    int stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        const BVHNode& node = foam_bvh[stack[--stack_ptr]];

        float t_min = 0.f, t_max = 1e30f, t_hit;
        if (node.bbox.intersect(origin, inv_dir, t_min, t_max, t_hit)) {
            if (node.prim_idx >= 0) {
                if (surface_mask[node.prim_idx]) {
                    int narrow_hit_idx = atomicAdd(hit_counter, 1);
                    narrowphase_hits[narrow_hit_idx] = {ray_idx, node.prim_idx, t_hit};
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
// For each candidate surface particle, performs a generalized Voronoi slab
// test: the Voronoi cell of a particle P is the intersection of half-spaces,
// one per neighbour N, defined by the perpendicular bisector plane between P
// and N. A point X lies inside P's cell iff for every neighbour N:
//
//   dot(X - midpoint(P,N), N - P) <= 0
//
// Equivalently for a ray O + t*D, we solve for the interval of t that
// satisfies all half-space constraints simultaneously (slab intersection).
// If the interval [t_enter, t_exit] is non-empty and t_enter >= 0, the ray
// hits this Voronoi cell and we record t_enter as the intersection.
//
// Candidates are sorted by AABB t, so we iterate in approximate front-to-back
// order and take the first cell whose slab test passes.
// -----------------------------------------------------------------------------

__global__ void k_exact_collision(
    const glm::vec3*                      ray_origins,
    const glm::vec3*                      ray_dirs,
    const NarrowphaseHit*                 narrowphase_hits,
    const int*                            unique_ray_ids,
    const int*                            ray_hit_offsets,
    const int*                            ray_hit_counts,
    const AdjacencyListGPU<entt::entity>* adj_lists,
    const int*                            adj_list_offsets,
    const glm::vec3*                      particle_positions,
    const glm::vec4*                      particle_colors,
    glm::vec4*                            output_buffer,
    int                                   num_unique
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_unique) return;

    const int ray_idx    = unique_ray_ids[i];
    const int hit_count  = ray_hit_counts[i];
    const int hit_offset = ray_hit_offsets[i];

    const glm::vec3 origin = ray_origins[ray_idx];
    const glm::vec3 dir    = ray_dirs[ray_idx];

    float best_t = 1e30f;
    int   best_particle = -1;

    for (int j = 0; j < hit_count; ++j) {
        const NarrowphaseHit hit        = narrowphase_hits[hit_offset + j];
        const int            pid        = hit.particle_id;
        const glm::vec3      p_pos      = particle_positions[pid];

        // Retrieve this particle's adjacency list
        const int                      adj_offset = adj_list_offsets[pid];
        const AdjacencyListGPU<entt::entity>& adj = adj_lists[adj_offset];

        // Slab test: accumulate the t-interval [t_enter, t_exit] over all
        // bisector half-planes. Each neighbour contributes one slab.
        float t_enter = 0.f;
        float t_exit  = 1e30f;
        bool  valid   = true;

        for (int k = 0; k < adj.size; ++k) {
            const int       nid   = static_cast<int>(
                                        entt::to_integral(adj.neighbours[k]));
            const glm::vec3 n_pos = particle_positions[nid];

            // Bisector plane between p and neighbour n:
            //   normal = n_pos - p_pos  (points from p toward n)
            //   point on plane = midpoint(p_pos, n_pos)
            const glm::vec3 plane_normal = n_pos - p_pos;
            const glm::vec3 midpoint     = (p_pos + n_pos) * 0.5f;

            // Ray-plane intersection: dot(O + t*D - midpoint, normal) = 0
            //   t = dot(midpoint - O, normal) / dot(D, normal)
            const float denom  = glm::dot(dir, plane_normal);
            const float numer  = glm::dot(midpoint - origin, plane_normal);

            if (fabsf(denom) < 1e-6f) {
                // Ray is parallel to this bisector plane.
                // Check which side of the plane the ray origin lies on.
                // If it's on the far side (numer < 0), the ray never enters
                // this cell — bail out immediately.
                if (numer < 0.f) {
                    valid = false;
                    break;
                }
                // Otherwise the ray is entirely on the correct side for this
                // plane; no t constraint needed.
            } else {
                const float t_plane = numer / denom;
                if (denom > 0.f) {
                    // Ray crosses into the half-space at t_plane
                    t_enter = fmaxf(t_enter, t_plane);
                } else {
                    // Ray exits the half-space at t_plane
                    t_exit = fminf(t_exit, t_plane);
                }

                if (t_enter > t_exit) {
                    valid = false;
                    break;
                }
            }
        }

        if (valid && t_enter <= t_exit && t_enter < best_t) {
            best_t        = t_enter;
            best_particle = pid;
        }
    }

    if (best_particle >= 0) {
        output_buffer[ray_idx] = particle_colors[best_particle];
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
    const entt::registry& particleRegistry,
    const std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists,
    const std::unordered_map<int, BVH>& foamBVHs,
    const std::unordered_map<int, AABB>& foamAABBs,
    const OrthographicCamera& camera,
    const glm::ivec2& windowSize
) {
    const int num_rays      = windowSize.x * windowSize.y;
    const int num_foams     = static_cast<int>(foamAABBs.size());
    const int num_particles = static_cast<int>(particleRegistry.alive());

    // ------------------------------------------------------------------
    // Ray buffer
    // ------------------------------------------------------------------

    const glm::vec3 forward = glm::normalize(camera.lookAt - camera.origin);
    const glm::vec3 right   = glm::normalize(glm::cross(forward, camera.up));
    const glm::vec3 up      = glm::cross(right, forward);

    std::vector<glm::vec3> origins(num_rays);
    std::vector<glm::vec3> dirs(num_rays);

    for (int y = 0; y < windowSize.y; ++y) {
        for (int x = 0; x < windowSize.x; ++x) {
            const int   index = y * windowSize.x + x;
            const float u     = (float(x) / float(windowSize.x)) - 0.5f;
            const float v     = (float(y) / float(windowSize.y)) - 0.5f;
            origins[index] = camera.origin
                           + u * camera.width  * right
                           + v * camera.height * up;
            dirs[index] = forward;
        }
    }

    cuda_realloc_if_needed(&d_ray_origins_, &cap_rays_, num_rays);
    cuda_realloc_if_needed(&d_ray_dirs_,    &cap_rays_, num_rays);
    CUDA_CHECK(cudaMemcpy(d_ray_origins_, origins.data(),
                          num_rays * sizeof(glm::vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ray_dirs_, dirs.data(),
                          num_rays * sizeof(glm::vec3), cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // Particle positions, colors, and surface mask — all built in one
    // pass over the registry to avoid redundant iteration.
    //
    // ParticleWorldPosition holds the current simulation position.
    // ParticleColor holds the RGB; we pack it to vec4 with ParticleOpacity.
    // Surface tag sets the mask entry to 1 for surface particles.
    // ------------------------------------------------------------------

    std::vector<glm::vec3> h_positions(num_particles, glm::vec3(0.f));
    std::vector<glm::vec4> h_colors(num_particles, glm::vec4(1.f));
    std::vector<uint8_t>   h_surface_mask(num_particles, 0);

    particleRegistry.view<const ParticleWorldPosition,
                          const ParticleColor,
                          const ParticleOpacity>()
        .each([&](entt::entity e,
                  const ParticleWorldPosition& pos,
                  const ParticleColor&         col,
                  const ParticleOpacity&       opacity)
        {
            const int pid = static_cast<int>(entt::to_integral(e));
            if (pid >= num_particles) return;
            h_positions[pid] = pos.value;
            h_colors[pid]    = glm::vec4(col.rgb, opacity.value);
        });

    particleRegistry.view<const Surface>().each(
        [&](entt::entity e) {
            const int pid = static_cast<int>(entt::to_integral(e));
            if (pid < num_particles)
                h_surface_mask[pid] = 1;
        });

    cuda_realloc_if_needed(&d_particle_positions_, &cap_particles_, num_particles);
    cuda_realloc_if_needed(&d_particle_colors_,    &cap_particles_, num_particles);
    cuda_realloc_if_needed(&d_surface_mask_,       &cap_surface_mask_, num_particles);

    CUDA_CHECK(cudaMemcpy(d_particle_positions_, h_positions.data(),
                          num_particles * sizeof(glm::vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particle_colors_, h_colors.data(),
                          num_particles * sizeof(glm::vec4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_surface_mask_, h_surface_mask.data(),
                          num_particles * sizeof(uint8_t), cudaMemcpyHostToDevice));

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

    cuda_realloc_if_needed(&d_foam_aabbs_, &cap_foams_, num_foams);
    cuda_realloc_if_needed(&d_foam_ids_,   &cap_foams_, num_foams);
    CUDA_CHECK(cudaMemcpy(d_foam_aabbs_, h_foam_aabbs.data(),
                          num_foams * sizeof(AABB), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_foam_ids_, h_foam_ids.data(),
                          num_foams * sizeof(int),  cudaMemcpyHostToDevice));

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

    cuda_realloc_if_needed(&d_bvh_nodes_,   &cap_bvh_nodes_,
                           all_bvh_nodes.size());
    cuda_realloc_if_needed(&d_bvh_offsets_, &cap_bvh_offsets_,
                           h_bvh_offsets.size());
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
        d_ray_origins_, d_ray_dirs_,
        d_foam_aabbs_,  d_foam_ids_,
        d_broadphase_hits_, d_broadphase_hit_counter_,
        num_rays, num_foams
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_broadphase_hits;
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
        d_ray_origins_, d_ray_dirs_,
        d_broadphase_hits_, num_broadphase_hits,
        d_bvh_nodes_,   d_bvh_offsets_,
        d_surface_mask_,
        d_narrowphase_hits_, d_narrowphase_hit_counter_
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_narrowphase_hits;
    CUDA_CHECK(cudaMemcpy(&num_narrowphase_hits, d_narrowphase_hit_counter_,
                          sizeof(int), cudaMemcpyDeviceToHost));
    if (num_narrowphase_hits == 0) return;

    // ------------------------------------------------------------------
    // Sort narrowphase hits by (ray_idx, t) via packed uint64 radix sort
    // ------------------------------------------------------------------

    cuda_realloc_if_needed(&d_sort_keys_in_,  &cap_sort_keys_, num_narrowphase_hits);
    cuda_realloc_if_needed(&d_sort_keys_out_, &cap_sort_keys_, num_narrowphase_hits);
    cuda_realloc_if_needed(&d_hits_sorted_,   &cap_hits_sorted_, num_narrowphase_hits);

    k_pack_sort_key<<<grid_size(num_narrowphase_hits), 256>>>(
        d_narrowphase_hits_, d_sort_keys_in_, num_narrowphase_hits);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUB_CALL(cub::DeviceRadixSort::SortPairs,
             d_sort_keys_in_,     d_sort_keys_out_,
             d_narrowphase_hits_, d_hits_sorted_,
             num_narrowphase_hits);

    // ------------------------------------------------------------------
    // Extract ray indices from sorted keys, then run-length encode to
    // get the count of hits per unique ray
    // ------------------------------------------------------------------

    cuda_realloc_if_needed(&d_ray_idx_keys_,   &cap_rle_, num_narrowphase_hits);
    cuda_realloc_if_needed(&d_unique_ray_ids_,  &cap_rle_, num_narrowphase_hits);
    cuda_realloc_if_needed(&d_rle_counts_,      &cap_rle_, num_narrowphase_hits);

    k_extract_ray_idx<<<grid_size(num_narrowphase_hits), 256>>>(
        d_sort_keys_out_, d_ray_idx_keys_, num_narrowphase_hits);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUB_CALL(cub::DeviceRunLengthEncode::Encode,
             d_ray_idx_keys_,
             d_unique_ray_ids_, d_rle_counts_, d_num_unique_,
             num_narrowphase_hits);

    int num_unique;
    CUDA_CHECK(cudaMemcpy(&num_unique, d_num_unique_,
                          sizeof(int), cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------
    // Exclusive scan over per-ray hit counts → compact offsets into the
    // sorted hit buffer. No scatter needed since k_exact_collision is
    // launched over num_unique threads and uses unique_ray_ids to write
    // directly into output_buffer at the correct ray index.
    // ------------------------------------------------------------------

    cuda_realloc_if_needed(&d_ray_hit_offsets_, &cap_ray_hit_offsets_, num_unique);

    CUB_CALL(cub::DeviceScan::ExclusiveSum,
             d_rle_counts_, d_ray_hit_offsets_, num_unique);

    // ------------------------------------------------------------------
    // Output buffer — memset to zero fills background for rays with no hits
    // ------------------------------------------------------------------

    cuda_realloc_if_needed(&d_output_buffer_, &cap_output_buffer_, num_rays);
    CUDA_CHECK(cudaMemset(d_output_buffer_, 0, num_rays * sizeof(glm::vec4)));

    // ------------------------------------------------------------------
    // Exact Voronoi slab test — one thread per ray that received hits
    // ------------------------------------------------------------------

    k_exact_collision<<<grid_size(num_unique), 256>>>(
        d_ray_origins_,
        d_ray_dirs_,
        d_hits_sorted_,
        d_unique_ray_ids_,
        d_ray_hit_offsets_,
        d_rle_counts_,
        d_adj_lists_,
        d_adj_list_offsets_,
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

    f(d_ray_origins_);             d_ray_origins_             = nullptr;
    f(d_ray_dirs_);                d_ray_dirs_                = nullptr;
    f(d_foam_aabbs_);              d_foam_aabbs_              = nullptr;
    f(d_foam_ids_);                d_foam_ids_                = nullptr;
    f(d_bvh_nodes_);               d_bvh_nodes_               = nullptr;
    f(d_bvh_offsets_);             d_bvh_offsets_             = nullptr;
    f(d_surface_mask_);            d_surface_mask_            = nullptr;
    f(d_broadphase_hits_);         d_broadphase_hits_         = nullptr;
    f(d_broadphase_hit_counter_);  d_broadphase_hit_counter_  = nullptr;
    f(d_narrowphase_hits_);        d_narrowphase_hits_        = nullptr;
    f(d_narrowphase_hit_counter_); d_narrowphase_hit_counter_ = nullptr;
    f(d_sort_keys_in_);            d_sort_keys_in_            = nullptr;
    f(d_sort_keys_out_);           d_sort_keys_out_           = nullptr;
    f(d_hits_sorted_);             d_hits_sorted_             = nullptr;
    f(d_ray_idx_keys_);            d_ray_idx_keys_            = nullptr;
    f(d_unique_ray_ids_);          d_unique_ray_ids_          = nullptr;
    f(d_rle_counts_);              d_rle_counts_              = nullptr;
    f(d_num_unique_);              d_num_unique_              = nullptr;
    f(d_ray_hit_offsets_);         d_ray_hit_offsets_         = nullptr;
    f(d_particle_positions_);      d_particle_positions_      = nullptr;
    f(d_particle_colors_);         d_particle_colors_         = nullptr;
    f(d_adj_lists_);               d_adj_lists_               = nullptr;
    f(d_adj_list_offsets_);        d_adj_list_offsets_        = nullptr;
    f(d_output_buffer_);           d_output_buffer_           = nullptr;
}

} // namespace DynamicFoam::Sim2D
