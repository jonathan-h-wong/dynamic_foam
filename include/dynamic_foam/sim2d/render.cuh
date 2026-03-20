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
#include <glm/gtc/matrix_inverse.hpp>
#include <cuda_runtime.h>
#include <unordered_map>

#include "dynamic_foam/Sim2D/adjacency.cuh"
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
// foam_inv_transforms is a per-foam array of inverse world-to-local transforms
// indexed by foam_id. Rays are transformed into each foam's local space before
// BVH traversal since BVH nodes are built in local particle coordinates.
__global__ void k_narrowphase_collision(
    const glm::vec3*     ray_origins,
    const glm::vec3*     ray_dirs,
    const BroadphaseHit* broadphase_hits,
    int                  num_broadphase_hits,
    const BVHNode*       bvh_nodes,
    const int*           bvh_offsets,
    const uint8_t*       surface_mask,
    const glm::mat4*     foam_inv_transforms,
    const int*           foam_particle_offsets,
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
// using flat concatenated CSR buffers (all foams packed end-to-end).
//
// csr_node_offsets    Flat array of per-node neighbor run starts.
//                     csr_node_offsets[csr_offsets[fid] + local_idx] gives
//                     the start of that node's neighbor run.
// csr_nbrs            Flat array of neighbor indices (foam-local sorted pos).
// csr_offsets         csr_offsets[fid] = start of foam fid's block in
//                     csr_node_offsets.
// foam_particle_offsets  foam_particle_offsets[fid] = start of foam fid's
//                        particles in the flat position/color buffers.
__global__ void k_exact_collision(
    const glm::vec3*      ray_origins,
    const glm::vec3*      ray_dirs,
    const NarrowphaseHit* narrowphase_hits,
    const int*            unique_ray_ids,
    const int*            ray_hit_offsets,
    const int*            ray_hit_counts,
    const uint32_t*       csr_node_offsets,
    const uint32_t*       csr_nbrs,
    const int*            csr_offsets,
    const int*            foam_particle_offsets,
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

    // All arguments are const: the render subsystem only reads simulation state.
    // foamTransforms maps foam_id -> world transform (position * rotation mat4).
    // Their inverses are computed on the host and uploaded so the narrowphase
    // kernel can transform rays into each foam's local BVH space.
    void update(
        const std::unordered_map<int, AABB>&                         foamAABBs,
        const std::unordered_map<int, BVH>&                          foamBVHs,
        const std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists,
        const entt::registry&                                        particleRegistry,
        const std::unordered_map<int, glm::mat4>&                    foamTransforms,
        const OrthographicCamera&                                    camera,
        const glm::ivec2&                                            windowSize
    );

    const glm::vec4* deviceOutputBuffer() const { return d_output_buffer_; }

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

    // Per-foam GPU adjacency lists
    std::unordered_map<int, AdjacencyListGPU<entt::entity>> foam_gpu_adj_;

    // Flat concatenated CSR — all foams packed end-to-end.
    // csr_node_offsets[csr_offsets[fid] + local_idx] = neighbor run start.
    // csr_nbrs entries are foam-local sorted positions.
    // Rebuilt from host each frame alongside particle buffers.
    uint32_t* d_csr_node_offsets_   = nullptr;
    size_t    cap_csr_node_offsets_ = 0;
    uint32_t* d_csr_nbrs_           = nullptr;
    size_t    cap_csr_nbrs_         = 0;

    // Per-foam offset tables (one int per foam).
    // d_csr_offsets_[fid]           = start of foam fid in d_csr_node_offsets_.
    // d_foam_particle_offsets_[fid] = start of foam fid in flat particle arrays.
    int*   d_csr_offsets_            = nullptr;
    int*   d_foam_particle_offsets_  = nullptr;
    size_t cap_foam_offsets_         = 0;

    // Per-foam inverse world-to-local transforms — one mat4 per foam, indexed
    // by foam_id. Used in k_narrowphase_collision to transform rays into the
    // foam's local BVH coordinate space.
    glm::mat4* d_foam_inv_transforms_   = nullptr;
    size_t     cap_foam_inv_transforms_ = 0;

    // Final pixel output
    glm::vec4* d_output_buffer_   = nullptr;
    size_t     cap_output_buffer_ = 0;
};

} // namespace DynamicFoam::Sim2D
