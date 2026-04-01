// =============================================================================
// render.cuh
// GPU ray tracer for 2D foam simulation.
// Implements a three-phase pipeline: broadphase AABB → narrowphase BVH →
// exact Voronoi cell intersection.
// =============================================================================

#pragma once

#define GLM_FORCE_CUDA
#include <cuda_runtime.h>
#include <entt/entity/registry.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <unordered_map>
#include <unordered_set>

#include "dynamic_foam/Sim2D/adjacency.cuh"
#include "dynamic_foam/Sim2D/bvh.cuh"
#include "dynamic_foam/Sim2D/cuda_utils.cuh"
#include "dynamic_foam/Sim2D/gpu_slab.cuh"

namespace DynamicFoam::Sim2D {

// -----------------------------------------------------------------------------
// Camera
// -----------------------------------------------------------------------------

enum class ProjectionType : int { Orthographic = 0, Perspective = 1 };

// Plain-old-data camera descriptor passed by value to GPU kernels.
// For orthographic cameras, width/height are world-space viewport dimensions.
// For perspective cameras, fovY is the vertical field-of-view in radians;
// width/height are unused (aspect ratio is derived from the render resolution).
struct CameraParams {
    glm::vec3      origin  = {0.f,  0.f, -5.f};
    glm::vec3      lookAt  = {0.f,  0.f,  0.f};
    glm::vec3      up      = {0.f,  1.f,  0.f};
    float          width   = 3.f;
    float          height  = 3.f;
    float          fovY    = 0.785398f; // 45 degrees in radians
    ProjectionType type    = ProjectionType::Orthographic;
};

// Generates a ray for pixel (ray_idx % img_width, ray_idx / img_width).
// Callable from both host and device code — no global-memory ray buffers needed.
__host__ __device__ inline void generateRay(
    const CameraParams& cam,
    int ray_idx, int img_width, int img_height,
    glm::vec3& out_origin, glm::vec3& out_dir
) {
    const int   px = ray_idx % img_width;
    const int   py = ray_idx / img_width;
    // u: [-0.5, +0.5] left to right.
    // v: flip py so screen row 0 (ImGui top) maps to +0.5 (world +Y up).
    const float u  =        float(px) / float(img_width)  - 0.5f;
    const float v  = 0.5f - float(py) / float(img_height);
    const glm::vec3 forward = glm::normalize(cam.lookAt - cam.origin);
    // Standard right-hand basis: right = cross(up, forward).
    const glm::vec3 right   = glm::normalize(glm::cross(cam.up, forward));
    const glm::vec3 up_vec  = glm::cross(forward, right);

    if (cam.type == ProjectionType::Orthographic) {
        out_origin = cam.origin + u * cam.width * right + v * cam.height * up_vec;
        out_dir    = forward;
    } else {
        const float half_h = tanf(cam.fovY * 0.5f);
        const float half_w = half_h * (float(img_width) / float(img_height));
        out_origin = cam.origin;
        out_dir    = glm::normalize(forward
                     + (u * 2.f * half_w) * right
                     + (v * 2.f * half_h) * up_vec);
    }
}

// Converts an ImGui pixel coordinate (top-left origin) to a world-space point
// by inverting the camera projection. For orthographic cameras this equals the
// ray origin; for perspective it lies on the image plane at the camera origin.
// Used by handleUserInput to map mouse positions to world space.
inline glm::vec3 unprojectPixel(
    const CameraParams& cam,
    glm::vec2 pixel, glm::ivec2 window
) {
    glm::vec3 origin, dir;
    generateRay(cam,
                static_cast<int>(pixel.y) * window.x + static_cast<int>(pixel.x),
                window.x, window.y,
                origin, dir);
    return origin;
}

// -----------------------------------------------------------------------------
// Overlay parameters passed to k_exact_collision.
// All fields are plain-old-data so the struct can be passed by value to CUDA
// kernels. Distances are in world-space units.
// -----------------------------------------------------------------------------
struct RenderOverlayParams {
    bool      show_centers   = false;
    bool      show_edges     = false;
    float     center_radius  = 0.005f;             // world-space dot radius
    float     edge_threshold = 0.001f;             // world-space edge half-width
    glm::vec4 center_color   = {1.f, 0.f, 0.f, 1.f};
    glm::vec4 edge_color     = {1.f, 0.f, 0.f, 1.f};
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
    CameraParams        camera,
    int                 width,
    int                 height,
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
    CameraParams         camera,
    int                  width,
    int                  height,
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
    CameraParams          camera,
    int                   width,
    int                   height,
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
    int                   num_unique,
    RenderOverlayParams   overlay
);

// -----------------------------------------------------------------------------
// Render class
// -----------------------------------------------------------------------------

class Render {
public:
    Render();
    ~Render();

    // All arguments are const: the render subsystem only reads simulation state.
    //
    // slab           Holds pre-built BVH nodes, CSR, and particle data for all
    //                foams.  Particle positions must already have been uploaded
    //                to the slab by the caller (Simulation) before this call.
    // foamTransforms maps foam_id -> world transform (position * rotation mat4).
    //                Their inverses are computed on the host and uploaded so the
    //                narrowphase kernel can transform rays into each foam's local
    //                BVH space.
    void update(
        const std::unordered_map<int, AABB>&      foamAABBs,
        const GpuSlabAllocator&                   slab,
        const std::unordered_map<int, glm::mat4>& foamTransforms,
        const CameraParams&                       camera,
        const glm::ivec2&                         windowSize,
        const RenderOverlayParams&                overlay = {}
    );

    const glm::vec4* deviceOutputBuffer() const { return d_output_buffer_; }

private:
    void free_device_memory();

    // ------------------------------------------------------------------
    // Persistent device buffers — allocated once, reused every frame.
    // Capacities track current allocation size for realloc-if-needed.
    // ------------------------------------------------------------------

    // Foam AABB buffers
    AABB*  d_foam_aabbs_ = nullptr;
    int*   d_foam_ids_   = nullptr;
    size_t cap_foams_    = 0;
    size_t cap_foam_ids_ = 0;

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
    size_t    cap_sort_keys_     = 0;
    size_t    cap_sort_keys_out_ = 0;

    NarrowphaseHit* d_hits_sorted_   = nullptr;
    size_t          cap_hits_sorted_ = 0;

    int*   d_ray_idx_keys_    = nullptr;
    int*   d_unique_ray_ids_  = nullptr;
    int*   d_rle_counts_      = nullptr;
    int*   d_num_unique_      = nullptr;  // single int on device
    size_t cap_rle_             = 0;
    size_t cap_unique_ray_ids_  = 0;
    size_t cap_rle_counts_      = 0;

    // Compact per-ray offsets into the sorted hit buffer (length = num_unique)
    int*   d_ray_hit_offsets_   = nullptr;
    size_t cap_ray_hit_offsets_ = 0;

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
