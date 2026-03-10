// =============================================================================
// render.cu
// GPU ray tracer implementation for 2D foam simulation.
//
// Pipeline per frame:
//   1. Upload rays + scene data (reusing preallocated buffers)
//   2. Broadphase: ray vs foam AABB
//   3. Narrowphase: ray vs per-foam BVH
//   4. Sort narrowphase hits by (ray_idx, t) via packed uint64 radix sort
//   5. Run-length encode to get per-ray hit counts
//   6. Exclusive scan over RLE counts → per-ray offsets (compact)
//   7. Exact collision: one thread per hit ray (Voronoi slab test placeholder)
// =============================================================================

#include <cub/cub.cuh>
#include "dynamic_foam/sim2d/render.cuh"
#include "dynamic_foam/sim2d/components.h"
#include "dynamic_foam/cuda_utils.cuh"

namespace DynamicFoam::Sim2D {

// -----------------------------------------------------------------------------
// Broadphase kernel — one thread per ray, iterates over all foam AABBs
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
// Narrowphase kernel — one thread per broadphase hit, traverses per-foam BVH
// -----------------------------------------------------------------------------

__global__ void k_narrowphase_collision(
    const glm::vec3*     ray_origins,
    const glm::vec3*     ray_dirs,
    const BroadphaseHit* broadphase_hits,
    int                  num_broadphase_hits,
    const BVHNode*       bvh_nodes,
    const int*           bvh_offsets,
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
                int narrow_hit_idx = atomicAdd(hit_counter, 1);
                narrowphase_hits[narrow_hit_idx] = {ray_idx, node.prim_idx, t_hit};
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
// Pack (ray_idx, t) into a single uint64 sort key.
// Upper 32 bits = ray_idx (primary sort key).
// Lower 32 bits = __float_as_uint(t) (secondary; valid because t >= 0 always,
// so the IEEE 754 bit pattern preserves float ordering).
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
// Extract the upper 32 bits (ray_idx) from sorted packed keys.
// Result is a sorted array of ray indices, ready for RLE.
// -----------------------------------------------------------------------------

__global__ void k_extract_ray_idx(
    const uint64_t* packed_keys, int* ray_indices, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ray_indices[i] = (int)(packed_keys[i] >> 32);
}

// -----------------------------------------------------------------------------
// Scatter RLE counts into a dense per-ray array.
// One thread per unique ray that received at least one hit.
// -----------------------------------------------------------------------------

__global__ void k_scatter_counts(
    int*       ray_hit_counts,
    const int* unique_ray_ids,
    const int* rle_counts,
    int        num_unique)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_unique) return;
    ray_hit_counts[unique_ray_ids[i]] = rle_counts[i];
}

// -----------------------------------------------------------------------------
// Exact collision kernel — one thread per ray that received hits.
// Hits are pre-sorted by t, so the first valid Voronoi intersection is closest.
//
// unique_ray_ids[i]    = actual ray index for thread i
// ray_hit_offsets[i]   = index into narrowphase_hits for thread i's first hit
// ray_hit_counts[i]    = number of hits for thread i
// (both offset/count arrays are compact, indexed by thread i, not ray_idx)
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

    for (int j = 0; j < hit_count; ++j) {
        const NarrowphaseHit hit = narrowphase_hits[hit_offset + j];

        // TODO: Voronoi slab test against hit.particle_id's cell planes.
        // Hits are sorted by t so the first passing test is the closest cell.
        output_buffer[ray_idx] = particle_colors[hit.particle_id];
        return;
    }
}

// =============================================================================
// Render host methods
// =============================================================================

Render::Render() {
    // Allocate the single-int device counters once — they are never resized.
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
    const int num_rays = windowSize.x * windowSize.y;
    const int num_foams = static_cast<int>(foamAABBs.size());

    // ------------------------------------------------------------------
    // 1. Build ray buffer on the host, upload to device
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
            origins[index] = camera.origin + u * camera.width * right
                                           + v * camera.height * up;
            dirs[index]    = forward;
        }
    }

    cuda_realloc_if_needed(&d_ray_origins_, &cap_rays_, num_rays);
    cuda_realloc_if_needed(&d_ray_dirs_,    &cap_rays_, num_rays);
    CUDA_CHECK(cudaMemcpy(d_ray_origins_, origins.data(),
                          num_rays * sizeof(glm::vec3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ray_dirs_,    dirs.data(),
                          num_rays * sizeof(glm::vec3), cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // 2. Upload foam AABBs
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
    CUDA_CHECK(cudaMemcpy(d_foam_ids_,   h_foam_ids.data(),
                          num_foams * sizeof(int),  cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // 3. Consolidate BVH nodes from all foams into a single flat buffer
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
    // 4. Broadphase
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
    // 5. Narrowphase
    // ------------------------------------------------------------------

    const size_t max_narrowphase_hits = (size_t)num_broadphase_hits * 32;
    cuda_realloc_if_needed(&d_narrowphase_hits_, &cap_narrowphase_hits_,
                           max_narrowphase_hits);
    CUDA_CHECK(cudaMemset(d_narrowphase_hit_counter_, 0, sizeof(int)));

    k_narrowphase_collision<<<grid_size(num_broadphase_hits), 256>>>(
        d_ray_origins_, d_ray_dirs_,
        d_broadphase_hits_, num_broadphase_hits,
        d_bvh_nodes_,   d_bvh_offsets_,
        d_narrowphase_hits_, d_narrowphase_hit_counter_
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    int num_narrowphase_hits;
    CUDA_CHECK(cudaMemcpy(&num_narrowphase_hits, d_narrowphase_hit_counter_,
                          sizeof(int), cudaMemcpyDeviceToHost));

    if (num_narrowphase_hits == 0) return;

    // ------------------------------------------------------------------
    // Pack (ray_idx, t) → uint64 sort key
    // ------------------------------------------------------------------

    cuda_realloc_if_needed(&d_sort_keys_in_,  &cap_sort_keys_, num_narrowphase_hits);
    cuda_realloc_if_needed(&d_sort_keys_out_, &cap_sort_keys_, num_narrowphase_hits);
    cuda_realloc_if_needed(&d_hits_sorted_,   &cap_hits_sorted_, num_narrowphase_hits);

    k_pack_sort_key<<<grid_size(num_narrowphase_hits), 256>>>(
        d_narrowphase_hits_, d_sort_keys_in_, num_narrowphase_hits);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // Radix sort by packed key — sorts by ray_idx then t
    // ------------------------------------------------------------------

    CUB_CALL(cub::DeviceRadixSort::SortPairs,
             d_sort_keys_in_,  d_sort_keys_out_,
             d_narrowphase_hits_, d_hits_sorted_,
             num_narrowphase_hits);

    // d_hits_sorted_ is now the canonical sorted hit buffer
    // d_sort_keys_out_ holds the sorted packed keys

    // ------------------------------------------------------------------
    // Extract ray_idx from sorted keys for RLE input
    // ------------------------------------------------------------------

    cuda_realloc_if_needed(&d_ray_idx_keys_, &cap_rle_, num_narrowphase_hits);

    k_extract_ray_idx<<<grid_size(num_narrowphase_hits), 256>>>(
        d_sort_keys_out_, d_ray_idx_keys_, num_narrowphase_hits);
    CUDA_CHECK(cudaDeviceSynchronize());

    // ------------------------------------------------------------------
    // Run-length encode sorted ray indices

    //   d_unique_ray_ids_[i] = which ray
    //   d_rle_counts_[i]     = how many hits it has
    //   d_num_unique_        = how many rays got any hit
    // ------------------------------------------------------------------

    cuda_realloc_if_needed(&d_unique_ray_ids_, &cap_rle_, num_narrowphase_hits);
    cuda_realloc_if_needed(&d_rle_counts_,     &cap_rle_, num_narrowphase_hits);

    CUB_CALL(cub::DeviceRunLengthEncode::Encode,
             d_ray_idx_keys_,
             d_unique_ray_ids_, d_rle_counts_, d_num_unique_,
             num_narrowphase_hits);

    int num_unique;
    CUDA_CHECK(cudaMemcpy(&num_unique, d_num_unique_,
                          sizeof(int), cudaMemcpyDeviceToHost));

    // ------------------------------------------------------------------
    // Exclusive scan over compact RLE counts → compact offsets.
    // No scatter needed — k_exact_collision is launched over num_unique
    // threads and uses unique_ray_ids to index into output_buffer.
    // ------------------------------------------------------------------

    cuda_realloc_if_needed(&d_ray_hit_offsets_, &cap_ray_hit_offsets_, num_unique);

    CUB_CALL(cub::DeviceScan::ExclusiveSum,
             d_rle_counts_, d_ray_hit_offsets_, num_unique);

    // ------------------------------------------------------------------
    // Prepare output buffer (memset fills background for rays with no hits)
    // ------------------------------------------------------------------

    cuda_realloc_if_needed(&d_output_buffer_, &cap_output_buffer_, num_rays);

    // Background colour: dark grey
    // glm::vec4(0.1, 0.1, 0.1, 1.0) bit pattern filled via memset is
    // approximate — use a kernel if exact background colour is required.
    CUDA_CHECK(cudaMemset(d_output_buffer_, 0, num_rays * sizeof(glm::vec4)));

    // ------------------------------------------------------------------
    // 6. Exact collision — one thread per ray that received hits
    // ------------------------------------------------------------------

    k_exact_collision<<<grid_size(num_unique), 256>>>(
        d_ray_origins_,
        d_ray_dirs_,
        d_hits_sorted_,
        d_unique_ray_ids_,
        d_ray_hit_offsets_,
        d_rle_counts_,          // compact hit counts, indexed by thread i
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
// Destructor helper — frees every device pointer
// -----------------------------------------------------------------------------

void Render::free_device_memory() {
    auto f = [](void* p) { if (p) cudaFree(p); };

    f(d_ray_origins_);           d_ray_origins_           = nullptr;
    f(d_ray_dirs_);              d_ray_dirs_               = nullptr;
    f(d_foam_aabbs_);            d_foam_aabbs_             = nullptr;
    f(d_foam_ids_);              d_foam_ids_               = nullptr;
    f(d_bvh_nodes_);             d_bvh_nodes_              = nullptr;
    f(d_bvh_offsets_);           d_bvh_offsets_            = nullptr;
    f(d_broadphase_hits_);       d_broadphase_hits_        = nullptr;
    f(d_broadphase_hit_counter_);d_broadphase_hit_counter_ = nullptr;
    f(d_narrowphase_hits_);      d_narrowphase_hits_       = nullptr;
    f(d_narrowphase_hit_counter_);d_narrowphase_hit_counter_= nullptr;
    f(d_sort_keys_in_);          d_sort_keys_in_           = nullptr;
    f(d_sort_keys_out_);         d_sort_keys_out_          = nullptr;
    f(d_hits_sorted_);           d_hits_sorted_            = nullptr;
    f(d_ray_idx_keys_);          d_ray_idx_keys_           = nullptr;
    f(d_unique_ray_ids_);        d_unique_ray_ids_         = nullptr;
    f(d_rle_counts_);            d_rle_counts_             = nullptr;
    f(d_num_unique_);            d_num_unique_             = nullptr;
    f(d_ray_hit_offsets_);       d_ray_hit_offsets_        = nullptr;
    f(d_adj_lists_);             d_adj_lists_              = nullptr;
    f(d_adj_list_offsets_);      d_adj_list_offsets_       = nullptr;
    f(d_particle_positions_);    d_particle_positions_     = nullptr;
    f(d_particle_colors_);       d_particle_colors_        = nullptr;
    f(d_output_buffer_);         d_output_buffer_          = nullptr;
}

} // namespace DynamicFoam::Sim2D
