#pragma once

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>
#ifdef __CUDACC__
#include <cub/cub.cuh>
#include <thrust/scatter.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include "cuda_utils.cuh"
#endif // __CUDACC__

namespace DynamicFoam::Sim2D {

// =============================================================================
// AdjacencyListGPU
//
// Three-buffer CSR layout. All pointers are device-side and owned by this
// struct. Call free() when done.
//
//   nodes[i]
//       Original node ID at sorted position i.
//
//   nbrs[node_offsets[i] .. node_offsets[i+1] - 1]
//       Sorted-position indices of node i's neighbors.
//
//   node_offsets[i]
//       Start of node i's neighbor run in nbrs.
//   node_offsets[num_nodes]
//       Sentinel equal to num_edges.
// =============================================================================
template <typename T>
struct AdjacencyListGPU {
    T*        nodes        = nullptr;
    uint32_t* nbrs         = nullptr;
    uint32_t* node_offsets = nullptr;

    uint32_t num_nodes = 0;
    uint32_t num_edges = 0;

    size_t nodes_capacity        = 0;
    size_t nbrs_capacity         = 0;
    size_t node_offsets_capacity = 0;

    // Ownership flags.  When false the pointer is a slice owned by an external
    // allocator (e.g. GpuSlabAllocator); free() must not cudaFree it.
    bool nodes_owned        = true;
    bool nbrs_owned         = true;
    bool node_offsets_owned = true;

    void free() {
        if (nodes        && nodes_owned)        { cudaFree(nodes);        }
        if (nbrs         && nbrs_owned)         { cudaFree(nbrs);         }
        if (node_offsets && node_offsets_owned) { cudaFree(node_offsets); }
        nodes        = nullptr;  nodes_owned        = true;  nodes_capacity        = 0;
        nbrs         = nullptr;  nbrs_owned         = true;  nbrs_capacity         = 0;
        node_offsets = nullptr;  node_offsets_owned = true;  node_offsets_capacity = 0;
        num_nodes = num_edges = 0;
    }
};

// =============================================================================
// Kernels — compiled by NVCC only
// =============================================================================
#ifdef __CUDACC__
namespace {

// Remap raw node IDs in a COO buffer to their sorted positions.
// inverse_map[original_id] = sorted_position
template <typename T>
__global__ void remapCOOKernel(
    const T*        __restrict__ raw,
    uint32_t*       __restrict__ remapped,
    const uint32_t* __restrict__ inverse_map,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) remapped[i] = inverse_map[static_cast<uint32_t>(raw[i])];
}

// Build inverse map from sorted index array.
// scatter: inverse_map[sorted_ids[i]] = i
__global__ void buildInverseMapKernel(
    const uint32_t* __restrict__ sorted_ids,
    uint32_t*       __restrict__ inverse_map,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) inverse_map[sorted_ids[i]] = i;
}

} // anonymous namespace
#endif // __CUDACC__

// =============================================================================
// AdjacencyList
//
// Stores an undirected graph as an adjacency set map plus a flat COO cache.
//
//   adj         — node -> unordered_set of neighbor IDs
//   edgeIndex   — packed 64-bit key set for O(1) existence checks
//   coo_src/dst — flat directed edge list kept current by all mutations
// =============================================================================
template <typename T>
class AdjacencyList {
public:

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    AdjacencyList() = default;

    explicit AdjacencyList(const std::vector<T>& nodeIds) {
        for (auto id : nodeIds) addNode(id);
    }

    // Type-converting copy constructor
    template <typename U>
    AdjacencyList(const AdjacencyList<U>& other,
                  const std::function<T(U)>& converter)
    {
        for (const auto& [node, neighbors] : other.getAdjList())
            for (auto nbr : neighbors)
                addEdge(converter(node), converter(nbr));
    }

    // -------------------------------------------------------------------------
    // Mutation
    // -------------------------------------------------------------------------

    void addNode(T nodeId) {
        adj.try_emplace(nodeId);
    }

    // Add an undirected edge between a and b. No-op if already exists.
    void addEdge(T a, T b) {
        addNode(a);
        addNode(b);

        uint64_t key = edgeKey(a, b);
        if (edgeIndex.count(key)) return;
        edgeIndex.insert(key);

        adj[a].insert(b);
        adj[b].insert(a);

        coo_src.push_back(a);
        coo_dst.push_back(b);
        coo_src.push_back(b);
        coo_dst.push_back(a);
    }

    void addNodeEdges(T nodeId, const std::vector<T>& connections) {
        for (auto conn : connections)
            addEdge(nodeId, conn);
    }

    // Delete a node and all its incident edges.
    void deleteNode(T nodeId) {
        auto it = adj.find(nodeId);
        if (it == adj.end()) return;

        for (T nbr : it->second) {
            adj[nbr].erase(nodeId);
            edgeIndex.erase(edgeKey(nodeId, nbr));
        }
        adj.erase(it);
        rebuildCOO();
    }

    // Merge another adjacency list into this one
    void graphEdit(const AdjacencyList<T>& other) {
        for (const auto& [node, neighbors] : other.adj)
            for (T nbr : neighbors)
                addEdge(node, nbr);
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    bool hasEdge(T a, T b) const {
        return edgeIndex.count(edgeKey(a, b)) > 0;
    }

    template <typename Func>
    void forEachNeighbor(T nodeId, Func fn) const {
        auto it = adj.find(nodeId);
        if (it == adj.end()) return;
        for (T nbr : it->second)
            fn(nbr);
    }

    const std::unordered_map<T, std::unordered_set<T>>& getAdjList() const {
        return adj;
    }

    // Returns the flat directed COO edge arrays. Always current — no lazy
    // rebuild needed since mutations update the COO eagerly.
    const std::vector<T>& getCOOSrc() const { return coo_src; }
    const std::vector<T>& getCOODst() const { return coo_dst; }

    // Returns node IDs in insertion order. Used by the render layer to build
    // the per-foam slice of the flat position/color/surface-mask buffers in
    // a deterministic order that matches the GPU CSR's sorted node layout.
    // Call after buildGPUAdjacencyList so that insertion order matches the
    // order nodes were added (which is the same order used for the nodes[]
    // buffer when d_morton_sorted_ids is null).
    std::vector<T> getOrderedNodeIds() const {
        std::vector<T> ids;
        ids.reserve(adj.size());
        for (const auto& [node, _] : adj)
            ids.push_back(node);
        return ids;
    }

    uint32_t nodeCount() const { return static_cast<uint32_t>(adj.size()); }

    // Returns the subset of the COO where both endpoints are in subset_ids.
    // Call this on the CPU before buildGPUAdjacencyList to avoid uploading and
    // filtering the full edge list on the GPU.
    std::pair<std::vector<T>, std::vector<T>>
    getSubsetCOO(const std::vector<T>& subset_ids) const {
        std::unordered_set<T> subset_set(subset_ids.begin(), subset_ids.end());
        std::vector<T> src, dst;
        for (T node : subset_ids) {
            forEachNeighbor(node, [&](T nbr) {
                if (subset_set.count(nbr)) {
                    src.push_back(node);
                    dst.push_back(nbr);
                }
            });
        }
        return {src, dst};
    }

    // -------------------------------------------------------------------------
    // buildGPUAdjacencyList — compiled by NVCC only
    //
    // Constructs a CSR adjacency list entirely on the GPU from the current
    // COO edge list.
    //
    // sorted_ids           CPU-side node IDs in desired sort order (e.g. from
    //                      a Morton sort pass). Pass an empty vector to use
    //                      insertion order. When non-empty, only intra-subset
    //                      edges are uploaded; filtering is done on the CPU
    //                      via getSubsetCOO before any GPU work begins.
    //
    // out                  Reused across frames. Device buffers are grown with
    //                      the doubling realloc helper and never shrunk.
    //
    // stream               All GPU work is issued onto this stream.
    // -------------------------------------------------------------------------
#ifdef __CUDACC__
    void buildGPUAdjacencyList(
        AdjacencyListGPU<T>&   out,
        const std::vector<T>&  sorted_ids = {},
        cudaStream_t           stream = 0) const
    {
        const bool     use_subset = !sorted_ids.empty();
        const uint32_t N = use_subset
                           ? static_cast<uint32_t>(sorted_ids.size())
                           : static_cast<uint32_t>(adj.size());

        // Filter edges on the CPU — only intra-subset edges are uploaded.
        // For the full-graph path the internal coo_src/coo_dst are used as-is.
        std::pair<std::vector<T>, std::vector<T>> subset_coo;
        if (use_subset)
            subset_coo = getSubsetCOO(sorted_ids);

        const std::vector<T>& src_buf = use_subset ? subset_coo.first  : coo_src;
        const std::vector<T>& dst_buf = use_subset ? subset_coo.second : coo_dst;
        const uint32_t E = static_cast<uint32_t>(src_buf.size());

        if (N == 0 || E == 0) return;

        out.num_nodes = N;
        out.num_edges = E;

        // ------------------------------------------------------------------
        // Grow persistent output buffers as needed.
        // Non-owned buffers (slab slices) must already have enough capacity;
        // we assert rather than reallocate them.
        // ------------------------------------------------------------------
        cuda_realloc_if_needed(&out.nodes, &out.nodes_capacity, N);
        if (out.nbrs_owned) {
            cuda_realloc_if_needed(&out.nbrs, &out.nbrs_capacity, E);
        } else {
            assert(out.nbrs_capacity >= E &&
                   "slab nbrs slice is too small for this foam");
        }
        if (out.node_offsets_owned) {
            cuda_realloc_if_needed(&out.node_offsets, &out.node_offsets_capacity, N + 1);
        } else {
            assert(out.node_offsets_capacity >= N + 1 &&
                   "slab node_offsets slice is too small for this foam");
        }

        // ------------------------------------------------------------------
        // Step 1 — upload node ordering (pure H2D, no D2H anywhere)
        // ------------------------------------------------------------------
        if (use_subset) {
            CUDA_CHECK(cudaMemcpy(
                out.nodes, sorted_ids.data(),
                N * sizeof(T), cudaMemcpyHostToDevice));
        } else {
            std::vector<T> h_nodes;
            h_nodes.reserve(N);
            for (const auto& [node, _] : adj)
                h_nodes.push_back(node);
            CUDA_CHECK(cudaMemcpy(
                out.nodes, h_nodes.data(),
                N * sizeof(T), cudaMemcpyHostToDevice));
        }

        // ------------------------------------------------------------------
        // Step 2 — build inverse map: node_id -> sorted position
        //
        // Entity IDs are assigned globally by entt (not per-foam), so the max
        // ID can be >> N. Allocate the map to max_entity_id + 1 and zero-fill
        // unused slots to avoid out-of-bounds writes from buildInverseMapKernel.
        // When using a subset the COO only contains subset IDs, so sizing to
        // the subset max is sufficient. We still scan adj as a safe
        // upper bound that covers both paths.
        // ------------------------------------------------------------------
        uint32_t max_entity_id = 0;
        for (const auto& [node, _] : adj)
            max_entity_id = std::max(max_entity_id, static_cast<uint32_t>(node));

        uint32_t* d_inverse_map = nullptr;
        const uint32_t inv_map_size = max_entity_id + 1;
        CUDA_CHECK(cudaMalloc(&d_inverse_map, inv_map_size * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_inverse_map, 0, inv_map_size * sizeof(uint32_t)));

        buildInverseMapKernel<<<grid_size(N), 256, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(out.nodes),
            d_inverse_map, N);

        // ------------------------------------------------------------------
        // Step 3 — upload COO and remap both ends to sorted positions
        //
        // src_buf/dst_buf are either the full graph COO or a caller-supplied
        // pre-filtered subset COO built by getSubsetCOO() on the CPU.
        // ------------------------------------------------------------------
        uint32_t* d_coo_src = nullptr;
        uint32_t* d_coo_dst = nullptr;
        CUDA_CHECK(cudaMalloc(&d_coo_src, E * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_coo_dst, E * sizeof(uint32_t)));

        CUDA_CHECK(cudaMemcpy(d_coo_src, src_buf.data(),
            E * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coo_dst, dst_buf.data(),
            E * sizeof(T), cudaMemcpyHostToDevice));

        remapCOOKernel<T><<<grid_size(E), 256, 0, stream>>>(
            reinterpret_cast<const T*>(d_coo_src), d_coo_src, d_inverse_map, E);
        remapCOOKernel<T><<<grid_size(E), 256, 0, stream>>>(
            reinterpret_cast<const T*>(d_coo_dst), d_coo_dst, d_inverse_map, E);

        // ------------------------------------------------------------------
        // Step 4 — radix sort COO by source
        //
        // After sorting, d_coo_dst_sorted is already the final nbrs buffer
        // (neighbor indices grouped by node, in sorted-position space).
        // ------------------------------------------------------------------
        uint32_t* d_coo_src_sorted = nullptr;
        uint32_t* d_coo_dst_sorted = nullptr;
        CUDA_CHECK(cudaMalloc(&d_coo_src_sorted, E * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_coo_dst_sorted, E * sizeof(uint32_t)));

        CUB_CALL(cub::DeviceRadixSort::SortPairs,
            d_coo_src, d_coo_src_sorted,
            d_coo_dst, d_coo_dst_sorted,
            E, 0, sizeof(uint32_t) * 8, stream);

        CUDA_CHECK(cudaMemcpy(out.nbrs, d_coo_dst_sorted,
            E * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

        // ------------------------------------------------------------------
        // Step 5 — run-length encode sorted src -> per-node degrees
        //
        // Because src is sorted, RLE gives exact degrees with no atomics.
        // Isolated nodes (degree 0) won't appear in the output -- we handle
        // them by zeroing d_degrees first and scattering only the live runs.
        // ------------------------------------------------------------------
        uint32_t* d_unique_nodes = nullptr;
        uint32_t* d_run_lengths  = nullptr;
        uint32_t* d_num_runs     = nullptr;
        CUDA_CHECK(cudaMalloc(&d_unique_nodes, N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_run_lengths,  N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_num_runs,     sizeof(uint32_t)));

        CUB_CALL(cub::DeviceRunLengthEncode::Encode,
            d_coo_src_sorted,
            d_unique_nodes,
            d_run_lengths,
            d_num_runs,
            E, stream);

        uint32_t h_num_runs = 0;
        CUDA_CHECK(cudaMemcpy(&h_num_runs, d_num_runs,
            sizeof(uint32_t), cudaMemcpyDeviceToHost));

        uint32_t* d_degrees = nullptr;
        CUDA_CHECK(cudaMalloc(&d_degrees, N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_degrees, 0, N * sizeof(uint32_t)));

        // Scatter run lengths into degree array at their sorted positions.
        // d_unique_nodes[i] is already a sorted-position index (post-remap).
        thrust::scatter(
            thrust::cuda::par.on(stream),
            thrust::device_ptr<uint32_t>(d_run_lengths),
            thrust::device_ptr<uint32_t>(d_run_lengths + h_num_runs),
            thrust::device_ptr<uint32_t>(d_unique_nodes),
            thrust::device_ptr<uint32_t>(d_degrees));

        // ------------------------------------------------------------------
        // Step 6 — exclusive scan over degrees -> node_offsets
        // ------------------------------------------------------------------
        CUB_CALL(cub::DeviceScan::ExclusiveSum,
            d_degrees, out.node_offsets, N, stream);

        // Write sentinel: node_offsets[N] = total edge count
        CUDA_CHECK(cudaMemcpy(out.node_offsets + N, &E,
            sizeof(uint32_t), cudaMemcpyHostToDevice));

        // ------------------------------------------------------------------
        // Cleanup temporaries
        // ------------------------------------------------------------------
        CUDA_CHECK(cudaFree(d_inverse_map));
        CUDA_CHECK(cudaFree(d_coo_src));
        CUDA_CHECK(cudaFree(d_coo_dst));
        CUDA_CHECK(cudaFree(d_coo_src_sorted));
        CUDA_CHECK(cudaFree(d_coo_dst_sorted));
        CUDA_CHECK(cudaFree(d_unique_nodes));
        CUDA_CHECK(cudaFree(d_run_lengths));
        CUDA_CHECK(cudaFree(d_num_runs));
        CUDA_CHECK(cudaFree(d_degrees));
    }
#endif // __CUDACC__

    // -------------------------------------------------------------------------
    // buildGPUAdjacencyListIntoSlice
    //
    // Slab variant: d_nbrs_slice and d_node_offsets_slice are pre-allocated
    // device memory owned by an external allocator (GpuSlabAllocator).  The
    // adjacency list writes directly into these slices and will NOT free them.
    //
    // nbrs_cap        must be >= E (num directed edges in this foam).
    // node_offsets_cap must be >= N + 1 (num nodes + sentinel).
    //
    // After this call, out.nbrs and out.node_offsets point at the slices with
    // nbrs_owned = false / node_offsets_owned = false so that out.free() is
    // safe to call without leaking.
    // -------------------------------------------------------------------------
#ifdef __CUDACC__
    void buildGPUAdjacencyListIntoSlice(
        AdjacencyListGPU<T>&   out,
        uint32_t*              d_nbrs_slice,
        size_t                 nbrs_cap,
        uint32_t*              d_node_offsets_slice,
        size_t                 node_offsets_cap,
        const std::vector<T>&  sorted_ids = {},
        cudaStream_t           stream = 0) const
    {
        // Free any previously owned nbrs / node_offsets buffers, then adopt
        // the slices without taking ownership.
        if (out.nbrs         && out.nbrs_owned)         { cudaFree(out.nbrs);         }
        if (out.node_offsets && out.node_offsets_owned) { cudaFree(out.node_offsets); }

        out.nbrs                  = d_nbrs_slice;
        out.nbrs_capacity         = nbrs_cap;
        out.nbrs_owned            = false;

        out.node_offsets          = d_node_offsets_slice;
        out.node_offsets_capacity = node_offsets_cap;
        out.node_offsets_owned    = false;

        // Delegate to the main builder which will skip realloc for non-owned
        // buffers (capacity is already set to the slice size).
        buildGPUAdjacencyList(out, sorted_ids, stream);
    }
#endif // __CUDACC__

private:

    // node -> set of neighbor IDs
    std::unordered_map<T, std::unordered_set<T>> adj;

    // O(1) existence check (canonical undirected key)
    std::unordered_set<uint64_t> edgeIndex;

    // COO cache — kept current eagerly by all mutation methods
    std::vector<T> coo_src;
    std::vector<T> coo_dst;

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    // Canonical edge key: always pack smaller ID into high bits
    static uint64_t edgeKey(T a, T b) {
        if (a > b) std::swap(a, b);
        return (uint64_t(a) << 32) | uint64_t(b);
    }

    // Remove a single undirected edge (a, b).
    void removeEdge(T a, T b) {
        uint64_t key = edgeKey(a, b);
        if (!edgeIndex.count(key)) return;
        edgeIndex.erase(key);
        adj[a].erase(b);
        adj[b].erase(a);
        rebuildCOO();
    }

    void rebuildCOO() {
        coo_src.clear();
        coo_dst.clear();
        for (const auto& [node, neighbors] : adj)
            for (T nbr : neighbors) {
                coo_src.push_back(node);
                coo_dst.push_back(nbr);
            }
    }
};

} // namespace DynamicFoam::Sim2D
