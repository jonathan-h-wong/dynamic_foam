// =============================================================================
// render.cuh
// GPU ray tracer for 2D foam simulation.
// Implements a three-phase pipeline: broadphase AABB → narrowphase BVH →
// exact Voronoi cell intersection.
// =============================================================================

#pragma once

#define GLM_FORCE_CUDA
#include <entt/entt.hpp>
#include <glm/glm.hpp>
#include <cuda_runtime.h>

#include "dynamic_foam/sim2d/adjacency.h"
#include "dynamic_foam/sim2d/bvh.cuh"
#include "dynamic_foam/cuda_utils.cuh"

namespace DynamicFoam::Sim2D {

// -----------------------------------------------------------------------------
// Camera
// -----------------------------------------------------------------------------

struct OrthographicCamera {
    glm::vec3 origin;
    glm::vec3 lookAt;
    glm::vec3 up;
    float width;
    float height;
};

// -----------------------------------------------------------------------------
// Hit record types
// -----------------------------------------------------------------------------

struct BroadphaseHit {
    int   ray_idx;
    int   foam_id;
    float t;
};

struct NarrowphaseHit {
    int   ray_idx;
    int   particle_id;
    float t;
};

// -----------------------------------------------------------------------------
// Kernel declarations
// -----------------------------------------------------------------------------

__global__ void k_broadphase_collision(
    const glm::vec3*    ray_origins,
    const glm::vec3*    ray_dirs,
    const AABB*         foam_aabbs,
    const int*          foam_ids,
    BroadphaseHit*      hits,
    int*                hit_counter,
    int                 num_rays,
    int                 num_foams
);

__global__ void k_narrowphase_collision(
    const glm::vec3*    ray_origins,
    const glm::vec3*    ray_dirs,
    const BroadphaseHit* broadphase_hits,
    int                 num_broadphase_hits,
    const BVHNode*      bvh_nodes,
    const int*          bvh_offsets,
    NarrowphaseHit*     narrowphase_hits,
    int*                hit_counter
);

__global__ void k_pack_sort_key(
    const NarrowphaseHit* hits,
    uint64_t*             keys,
    int                   n
);

__global__ void k_extract_ray_idx(
    const uint64_t* packed_keys,
    int*            ray_indices,
    int             n
);

__global__ void k_scatter_counts(
    int*       ray_hit_counts,
    const int* unique_ray_ids,
    const int* rle_counts,
    int        num_unique
);

// Operates over only the num_unique rays that received hits.
// unique_ray_ids maps thread index → actual ray index.
// ray_hit_offsets and ray_hit_counts are compact arrays of length num_unique,
// indexed by thread index (not ray index).
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
);

// -----------------------------------------------------------------------------
// Render class
// -----------------------------------------------------------------------------

class Render {
public:
    Render();
    ~Render();

    void update(
        const entt::registry&                                    particleRegistry,
        const std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists,
        const std::unordered_map<int, BVH>&                      foamBVHs,
        const std::unordered_map<int, AABB>&                     foamAABBs,
        const OrthographicCamera&                                camera,
        const glm::ivec2&                                        windowSize
    );

private:
    void free_device_memory();

    // ------------------------------------------------------------------
    // Persistent device buffers — allocated once, reused every frame.
    // Capacities track current allocation size for realloc-if-needed.
    // ------------------------------------------------------------------

    // Ray buffers (capacity = max pixels)
    glm::vec3* d_ray_origins_  = nullptr;
    glm::vec3* d_ray_dirs_     = nullptr;
    size_t     cap_rays_       = 0;

    // Foam AABB buffers (capacity = max foam count)
    AABB* d_foam_aabbs_ = nullptr;
    int*  d_foam_ids_   = nullptr;
    size_t cap_foams_   = 0;

    // BVH node/offset buffers (capacity = total nodes across all foams)
    BVHNode* d_bvh_nodes_   = nullptr;
    int*     d_bvh_offsets_ = nullptr;
    size_t   cap_bvh_nodes_   = 0;
    size_t   cap_bvh_offsets_ = 0;

    // Broadphase hit buffer + atomic counter
    BroadphaseHit* d_broadphase_hits_        = nullptr;
    int*           d_broadphase_hit_counter_ = nullptr;
    size_t         cap_broadphase_hits_      = 0;

    // Narrowphase hit buffer + atomic counter
    NarrowphaseHit* d_narrowphase_hits_        = nullptr;
    int*            d_narrowphase_hit_counter_ = nullptr;
    size_t          cap_narrowphase_hits_      = 0;

    // Sort/compaction temporaries (reused across frames)
    uint64_t* d_sort_keys_in_  = nullptr;
    uint64_t* d_sort_keys_out_ = nullptr;
    size_t    cap_sort_keys_   = 0;

    NarrowphaseHit* d_hits_sorted_ = nullptr;
    size_t          cap_hits_sorted_ = 0;

    int*   d_ray_idx_keys_   = nullptr;
    int*   d_unique_ray_ids_ = nullptr;
    int*   d_rle_counts_     = nullptr;
    int*   d_num_unique_     = nullptr;    // single int on device
    size_t cap_rle_          = 0;

    // Per-ray compact offset array (length = num_unique each frame)
    int*   d_ray_hit_offsets_ = nullptr;
    size_t cap_ray_hit_offsets_ = 0;

    // Adjacency list / particle data (uploaded once when sim state changes)
    AdjacencyListGPU<entt::entity>* d_adj_lists_        = nullptr;
    int*                            d_adj_list_offsets_ = nullptr;
    glm::vec3*                      d_particle_positions_ = nullptr;
    glm::vec4*                      d_particle_colors_    = nullptr;

    // Final pixel output
    glm::vec4* d_output_buffer_  = nullptr;
    size_t     cap_output_buffer_ = 0;
};

} // namespace DynamicFoam::Sim2D
