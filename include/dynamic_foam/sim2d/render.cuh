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
#include <unordered_map>

#include "dynamic_foam/Sim2D/adjacency.h"
#include "dynamic_foam/Sim2D/bvh.cuh"
#include "dynamic_foam/Sim2D/cuda_utils.cuh"

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

// foam_id carried through from broadphase so k_exact_collision can index
// the correct per-foam CSR adjacency buffers.
struct NarrowphaseHit {
    int   ray_idx;
    int   particle_id;
    int   foam_id;
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

// surface_mask is a per-particle boolean array (uint8_t, 1 = surface).
// Leaf hits against interior particles (mask == 0) are discarded, so only
// surface particles reach the exact collision stage.
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

// One thread per ray that received narrowphase hits.
// Performs a generalized Voronoi slab test against each candidate particle,
// using per-foam CSR adjacency buffers passed as device pointer tables.
//
// foam_node_offsets   Device array of per-foam node_offsets pointers.
//                     foam_node_offsets[foam_id][i] is the start of node i's
//                     neighbor run in foam_nbrs[foam_id].
// foam_nbrs           Device array of per-foam neighbor index arrays.
//                     Indices are global dense particle IDs.
// particle_to_sorted  Maps global dense particle index -> local sorted
//                     position within that particle's foam CSR.
__global__ void k_exact_collision(
    const glm::vec3*      ray_origins,
    const glm::vec3*      ray_dirs,
    const NarrowphaseHit* narrowphase_hits,
    const int*            unique_ray_ids,
    const int*            ray_hit_offsets,
    const int*            ray_hit_counts,
    const uint32_t**      foam_node_offsets,
    const uint32_t**      foam_nbrs,
    const int*            particle_to_sorted,
    const glm::vec3*      particle_positions,
    const glm::vec4*      particle_colors,
    glm::vec4*            output_buffer,
    int                   num_unique
);

// -----------------------------------------------------------------------------
// Render class
// -----------------------------------------------------------------------------

class Render {
public:
    Render();
    ~Render();

    // foamAdjacencyLists is non-const: dirty flags are cleared after each
    // GPU rebuild via adj.clearDirty().
    void update(
        const entt::registry&                                  particleRegistry,
        std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists,
        const std::unordered_map<int, BVH>&                   foamBVHs,
        const std::unordered_map<int, AABB>&                  foamAABBs,
        const OrthographicCamera&                             camera,
        const glm::ivec2&                                     windowSize
    );

private:
    void free_device_memory();

    // ------------------------------------------------------------------
    // Persistent device buffers — allocated once, reused every frame.
    // Capacities track current allocation size for realloc-if-needed.
    // ------------------------------------------------------------------

    // Ray buffers
    glm::vec3* d_ray_origins_ = nullptr;
    glm::vec3* d_ray_dirs_    = nullptr;
    size_t     cap_rays_      = 0;

    // Foam AABB buffers
    AABB*  d_foam_aabbs_ = nullptr;
    int*   d_foam_ids_   = nullptr;
    size_t cap_foams_    = 0;

    // BVH node/offset buffers
    BVHNode* d_bvh_nodes_     = nullptr;
    int*     d_bvh_offsets_   = nullptr;
    size_t   cap_bvh_nodes_   = 0;
    size_t   cap_bvh_offsets_ = 0;

    // Surface mask — one uint8_t per particle: 1 = surface, 0 = interior.
    // Built from entt Surface tag each frame and used in k_narrowphase_collision
    // to skip interior particles before they reach exact collision.
    uint8_t* d_surface_mask_   = nullptr;
    size_t   cap_surface_mask_ = 0;

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

    NarrowphaseHit* d_hits_sorted_   = nullptr;
    size_t          cap_hits_sorted_ = 0;

    int*   d_ray_idx_keys_   = nullptr;
    int*   d_unique_ray_ids_ = nullptr;
    int*   d_rle_counts_     = nullptr;
    int*   d_num_unique_     = nullptr;  // single int on device
    size_t cap_rle_          = 0;

    // Compact per-ray offsets into the sorted hit buffer (length = num_unique)
    int*   d_ray_hit_offsets_   = nullptr;
    size_t cap_ray_hit_offsets_ = 0;

    // Particle data buffers — uploaded each frame from the entt registry
    glm::vec3* d_particle_positions_ = nullptr;
    glm::vec4* d_particle_colors_    = nullptr;
    size_t     cap_particles_        = 0;

    // Per-foam GPU adjacency lists — persistent across frames, rebuilt only
    // when a foam's topology changes (isCOODirty()).
    std::unordered_map<int, AdjacencyListGPU<entt::entity>> foam_gpu_adj_;

    // Device pointer tables — one entry per foam, pointing into the
    // corresponding foam's AdjacencyListGPU buffers.
    // Uploaded whenever foam_gpu_adj_ changes.
    uint32_t** d_foam_node_offsets_ptrs_ = nullptr;
    uint32_t** d_foam_nbrs_ptrs_         = nullptr;
    size_t     cap_foam_ptr_table_       = 0;

    // Maps global dense particle index -> local sorted position in its foam's
    // CSR. Rebuilt whenever any foam adjacency list is dirty.
    int*   d_particle_to_sorted_   = nullptr;
    size_t cap_particle_to_sorted_ = 0;

    // Final pixel output
    glm::vec4* d_output_buffer_   = nullptr;
    size_t     cap_output_buffer_ = 0;
};

} // namespace DynamicFoam::Sim2D
