// =============================================================================
// bvh.cuh
// GPU BVH implementation using the Karras 2012 algorithm over AABB primitives.
//
// Reference: "Maximizing Parallelism in the Construction of BVHs, Octrees,
//             and k-d Trees" — Tero Karras, HPG 2012
//
// Pipeline (all on GPU):
//   1. Compute scene AABB  (parallel reduction via CUB)
//   2. Compute Morton codes (one thread per primitive)
//   3. Sort by Morton code  (CUB radix sort)
//   4. Build tree topology  (one thread per internal node, Karras algorithm)
//   5. Propagate bboxes     (bottom-up via atomic flags, one thread per leaf)
//
// Node layout:
//   N primitives → N-1 internal nodes [0, N-2] + N leaf nodes [N-1, 2N-2]
//
// See bvh.cu for all kernel and method implementations.
// =============================================================================

#pragma once

#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// -----------------------------------------------------------------------------
// Error checking
// -----------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(1);                                                           \
        }                                                                      \
    } while(0)

// -----------------------------------------------------------------------------
// AABB (device-compatible)
// -----------------------------------------------------------------------------

struct AABB {
    glm::vec3 min_pt;
    glm::vec3 max_pt;

    __host__ __device__ AABB()
        : min_pt( 1e30f,  1e30f,  1e30f)
        , max_pt(-1e30f, -1e30f, -1e30f)
    {}

    __host__ __device__ AABB(const glm::vec3& mn, const glm::vec3& mx)
        : min_pt(mn), max_pt(mx)
    {}

    __host__ __device__ glm::vec3 centroid() const {
        return (min_pt + max_pt) * 0.5f;
    }

    __host__ __device__ AABB merge(const AABB& o) const {
        return { glm::min(min_pt, o.min_pt), glm::max(max_pt, o.max_pt) };
    }

    __device__ bool intersect(const glm::vec3& origin, const glm::vec3& inv_dir,
                              float t_min, float t_max) const
    {
        glm::vec3 t0 = (min_pt - origin) * inv_dir;
        glm::vec3 t1 = (max_pt - origin) * inv_dir;

        t_min = fmaxf(t_min, fmaxf(fminf(t0.x, t1.x), fmaxf(fminf(t0.y, t1.y), fminf(t0.z, t1.z))));
        t_max = fminf(t_max, fminf(fmaxf(t0.x, t1.x), fminf(fmaxf(t0.y, t1.y), fmaxf(t0.z, t1.z))));

        return t_min <= t_max;
    }
};

// CUB reduction operator for AABB union
struct AABBUnion {
    __host__ __device__ AABB operator()(const AABB& a, const AABB& b) const {
        return a.merge(b);
    }
};

// -----------------------------------------------------------------------------
// BVH Node (flat GPU layout — no virtual dispatch)
// -----------------------------------------------------------------------------

struct BVHNode {
    AABB bbox;
    int  left     = -1;
    int  right    = -1;
    int  parent   = -1;
    int  prim_idx = -1;  // >= 0 → leaf
};

// -----------------------------------------------------------------------------
// Ray traversal output
// -----------------------------------------------------------------------------

struct RayHit {
    int prim_ids[32];  // hit primitive IDs (capacity capped at 32)
    int count;         // number of hits recorded
};

// -----------------------------------------------------------------------------
// Morton code utilities (inline — required in header for device inlining)
// -----------------------------------------------------------------------------

__host__ __device__ inline uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__host__ __device__ inline uint32_t morton3D(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.f, 0.f), 1023.f);
    y = fminf(fmaxf(y * 1024.f, 0.f), 1023.f);
    z = fminf(fmaxf(z * 1024.f, 0.f), 1023.f);
    const uint32_t ix = static_cast<uint32_t>(x);
    const uint32_t iy = static_cast<uint32_t>(y);
    const uint32_t iz = static_cast<uint32_t>(z);
    return (expand_bits(ix) << 2) | (expand_bits(iy) << 1) | expand_bits(iz);
}

// -----------------------------------------------------------------------------
// Kernel declarations
// -----------------------------------------------------------------------------

__global__ void k_compute_morton(
    const AABB* __restrict__ primitives,
    uint32_t*   __restrict__ morton_codes,
    int*        __restrict__ indices,
    int n,
    glm::vec3 scene_min,
    glm::vec3 scene_extent_inv);

__global__ void k_init_leaves(
    const AABB* __restrict__ primitives,
    const int*  __restrict__ sorted_indices,
    BVHNode*    __restrict__ nodes,
    int n);

__global__ void k_build_topology(
    const uint32_t* __restrict__ morton_codes,
    BVHNode*        __restrict__ nodes,
    int n);

__global__ void k_propagate_bboxes(
    BVHNode* __restrict__ nodes,
    int*     __restrict__ flags,
    int n);

__global__ void k_traverse(
    const BVHNode*   __restrict__ nodes,
    const glm::vec3* __restrict__ ray_origins,
    const glm::vec3* __restrict__ ray_inv_dirs,
    float t_min, float t_max,
    RayHit* __restrict__ hits,
    int num_rays,
    int root);

// -----------------------------------------------------------------------------
// BVH: host-side manager
// -----------------------------------------------------------------------------

class BVH {
public:
    BVH() = default;
    ~BVH();

    // Upload primitives and build the BVH entirely on the GPU.
    void build(const AABB* primitives_host, int n);

    // Launch ray traversal on the GPU.
    // Returns a device-side RayHit array; the caller is responsible for freeing it.
    RayHit* traverse_rays(const glm::vec3* d_origins, const glm::vec3* d_inv_dirs,
                          int num_rays, float t_min = 0.f, float t_max = 1e30f);

    int num_primitives() const { return n_; }

private:
    void alloc_device(int n);
    void free_device();

    int       n_                = 0;
    AABB*     d_primitives_     = nullptr;
    BVHNode*  d_nodes_          = nullptr;
    uint32_t* d_morton_codes_   = nullptr;
    uint32_t* d_morton_sorted_  = nullptr;
    int*      d_indices_        = nullptr;
    int*      d_indices_sorted_ = nullptr;
    int*      d_flags_          = nullptr;
    AABB*     d_scene_bbox_     = nullptr;
};
