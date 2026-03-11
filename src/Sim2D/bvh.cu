// =============================================================================
// bvh.cu
// GPU kernel and BVH method implementations.
//
// Reference: "Maximizing Parallelism in the Construction of BVHs, Octrees,
//             and k-d Trees" — Tero Karras, HPG 2012
//
// See bvh.cuh for struct definitions and declarations.
// =============================================================================

#include "dynamic_foam/Sim2D/bvh.cuh"

#include <cassert>

// -----------------------------------------------------------------------------
// delta: longest common prefix length between Morton codes i and j.
// Returns -1 for out-of-range j (boundary sentinel).
// -----------------------------------------------------------------------------

__device__ inline int delta(
    const uint32_t* __restrict__ codes, int i, int j, int n)
{
    if (j < 0 || j >= n) return -1;
    if (codes[i] == codes[j])
        return 32 + __clz(static_cast<uint32_t>(i ^ j));
    return __clz(codes[i] ^ codes[j]);
}

// -----------------------------------------------------------------------------
// Kernel 1: compute Morton codes (one thread per primitive)
// -----------------------------------------------------------------------------

__global__ void k_compute_morton(
    const AABB* __restrict__ primitives,
    uint32_t*   __restrict__ morton_codes,
    int*        __restrict__ indices,
    int n,
    glm::vec3 scene_min,
    glm::vec3 scene_extent_inv)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const glm::vec3 c  = primitives[i].centroid();
    const float nx = (c.x - scene_min.x) * scene_extent_inv.x;
    const float ny = (c.y - scene_min.y) * scene_extent_inv.y;
    const float nz = (c.z - scene_min.z) * scene_extent_inv.z;

    morton_codes[i] = morton3D(nx, ny, nz);
    indices[i]      = i;
}

// -----------------------------------------------------------------------------
// Kernel 2: initialize leaf nodes after Morton sort
// -----------------------------------------------------------------------------

__global__ void k_init_leaves(
    const AABB* __restrict__ primitives,
    const int*  __restrict__ sorted_indices,
    BVHNode*    __restrict__ nodes,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int leaf_node = (n - 1) + i;
    const int prim      = sorted_indices[i];

    nodes[leaf_node].prim_idx = prim;
    nodes[leaf_node].bbox     = primitives[prim];
    nodes[leaf_node].left     = -1;
    nodes[leaf_node].right    = -1;
    nodes[leaf_node].parent   = -1;
}

// -----------------------------------------------------------------------------
// Kernel 3: Karras tree topology (one thread per internal node)
// -----------------------------------------------------------------------------

__global__ void k_build_topology(
    const uint32_t* __restrict__ morton_codes,
    BVHNode*        __restrict__ nodes,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1) return;

    // Determine direction: which neighbour shares a longer common prefix?
    const int d = (delta(morton_codes, i, i + 1, n) -
                   delta(morton_codes, i, i - 1, n)) >= 0 ? 1 : -1;

    const int delta_min = delta(morton_codes, i, i - d, n);

    // Find upper bound on range length
    int l_max = 2;
    while (delta(morton_codes, i, i + d * l_max, n) > delta_min)
        l_max <<= 1;

    // Binary search for exact range length
    int l = 0;
    for (int t = l_max >> 1; t >= 1; t >>= 1) {
        if (delta(morton_codes, i, i + d * (l + t), n) > delta_min)
            l += t;
    }

    const int j     = i + d * l;
    const int first = min(i, j);
    const int last  = max(i, j);

    // Find split position within [first, last]
    const int delta_node = delta(morton_codes, first, last, n);
    int split = first;
    int step  = last - first;
    do {
        step = (step + 1) >> 1;
        const int mid = split + step;
        if (mid < last && delta(morton_codes, first, mid, n) > delta_node)
            split = mid;
    } while (step > 1);

    const int left_child  = (split     == first) ? (n - 1 + split)     : split;
    const int right_child = (split + 1 == last)  ? (n - 1 + split + 1) : (split + 1);

    nodes[i].left  = left_child;
    nodes[i].right = right_child;

    nodes[left_child].parent  = i;
    nodes[right_child].parent = i;
}

// -----------------------------------------------------------------------------
// Kernel 4: bottom-up bounding box propagation (one thread per leaf)
// -----------------------------------------------------------------------------

__global__ void k_propagate_bboxes(
    BVHNode* __restrict__ nodes,
    int*     __restrict__ flags,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int idx    = (n - 1) + i;
    int parent = nodes[idx].parent;

    while (parent != -1) {
        const int old = atomicAdd(&flags[parent], 1);
        if (old == 0) return;

        const AABB left_bbox  = nodes[nodes[parent].left ].bbox;
        const AABB right_bbox = nodes[nodes[parent].right].bbox;
        nodes[parent].bbox = left_bbox.merge(right_bbox);

        __threadfence();

        idx    = parent;
        parent = nodes[idx].parent;
    }
}

// -----------------------------------------------------------------------------
// BVH host-side method implementations
// -----------------------------------------------------------------------------

BVH::~BVH() {
    free_device();
}

void BVH::build(const AABB* primitives_host, int n) {
    assert(n > 0);
    n_ = n;

    free_device();
    alloc_device(n);

    CUDA_CHECK(cudaMemcpy(d_primitives_, primitives_host,
                          n * sizeof(AABB), cudaMemcpyHostToDevice));

    const int block = 256;

    // Step 1: reduce all primitive AABBs to a single scene AABB
    AABB identity{};
    CUB_CALL(cub::DeviceReduce::Reduce,
             d_primitives_, d_scene_bbox_, n, AABBUnion{}, identity);

    AABB scene_bbox;
    CUDA_CHECK(cudaMemcpy(&scene_bbox, d_scene_bbox_,
                          sizeof(AABB), cudaMemcpyDeviceToHost));

    const glm::vec3 extent = scene_bbox.max_pt - scene_bbox.min_pt;
    const glm::vec3 extent_inv = {
        extent.x > 0.f ? 1.f / extent.x : 1.f,
        extent.y > 0.f ? 1.f / extent.y : 1.f,
        extent.z > 0.f ? 1.f / extent.z : 1.f
    };

    // Step 2: compute Morton codes
    k_compute_morton<<<grid_size(n, block), block>>>(
        d_primitives_, d_morton_codes_, d_indices_, n,
        scene_bbox.min_pt, extent_inv);
    CUDA_CHECK(cudaGetLastError());

    // Step 3: sort by Morton code
    CUB_CALL(cub::DeviceRadixSort::SortPairs,
             d_morton_codes_, d_morton_sorted_,
             d_indices_,      d_indices_sorted_,
             n);

    // Step 4: initialize leaf nodes
    k_init_leaves<<<grid_size(n, block), block>>>(
        d_primitives_, d_indices_sorted_, d_nodes_, n);
    CUDA_CHECK(cudaGetLastError());

    if (n > 1) {
        // Step 5: build tree topology
        k_build_topology<<<grid_size(n - 1, block), block>>>(
            d_morton_sorted_, d_nodes_, n);
        CUDA_CHECK(cudaGetLastError());

        // Step 6: propagate bounding boxes bottom-up
        CUDA_CHECK(cudaMemset(d_flags_, 0, (n - 1) * sizeof(int)));
        k_propagate_bboxes<<<grid_size(n, block), block>>>(
            d_nodes_, d_flags_, n);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

BVHNode* BVH::export_nodes() const {
    return d_nodes_;
}

void BVH::alloc_device(int n) {
    const int num_nodes = (n > 1) ? 2 * n - 1 : 1;

    CUDA_CHECK(cudaMalloc(&d_primitives_,     n                    * sizeof(AABB)));
    CUDA_CHECK(cudaMalloc(&d_nodes_,          num_nodes            * sizeof(BVHNode)));
    CUDA_CHECK(cudaMalloc(&d_morton_codes_,   n                    * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_morton_sorted_,  n                    * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_indices_,        n                    * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_indices_sorted_, n                    * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flags_,          (n > 1 ? n - 1 : 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scene_bbox_,     sizeof(AABB)));
}

void BVH::free_device() {
    auto f = [](void* p) { if (p) cudaFree(p); };

    f(d_primitives_);     d_primitives_     = nullptr;
    f(d_nodes_);          d_nodes_          = nullptr;
    f(d_morton_codes_);   d_morton_codes_   = nullptr;
    f(d_morton_sorted_);  d_morton_sorted_  = nullptr;
    f(d_indices_);        d_indices_        = nullptr;
    f(d_indices_sorted_); d_indices_sorted_ = nullptr;
    f(d_flags_);          d_flags_          = nullptr;
    f(d_scene_bbox_);     d_scene_bbox_     = nullptr;
}
