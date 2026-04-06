#pragma once

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stack>

#include <cuda_runtime.h>
#ifdef __CUDACC__
#include <cub/cub.cuh>
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
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
// Stores an undirected graph. Each undirected edge is stored as two half-edges
// (one per direction) in a flat array. Deleted edge slots are pushed onto a
// free list and reclaimed by subsequent additions, so the array stays near its
// high-water mark when adds and deletes are balanced -- no compaction needed.
//
// Three supporting structures are kept in sync with the edge array:
//
//   nodeHeads   — node -> index of its first outgoing half-edge (-1 if none)
//   edgeIndex   — packed 64-bit key set for O(1) existence checks
//   coo_src/dst — flat directed edge list rebuilt lazily for GPU upload
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
        if (nodeHeads.find(nodeId) == nodeHeads.end())
            nodeHeads[nodeId] = -1;
    }

    // Add an undirected edge between a and b. No-op if already exists.
    void addEdge(T a, T b) {
        addNode(a);
        addNode(b);

        uint64_t key = edgeKey(a, b);
        if (edgeIndex.count(key)) return;
        edgeIndex.insert(key);

        // Allocate two half-edge slots, reusing free list slots first
        int idxAB = allocSlot();
        int idxBA = allocSlot();

        edges[idxAB] = { a, b, idxBA, nodeHeads[a], true };
        edges[idxBA] = { b, a, idxAB, nodeHeads[b], true };

        nodeHeads[a] = idxAB;
        nodeHeads[b] = idxBA;

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
        auto it = nodeHeads.find(nodeId);
        if (it == nodeHeads.end()) return;

        // Snapshot neighbor list before modifying it
        std::vector<T> neighbors;
        int idx = it->second;
        while (idx != -1) {
            neighbors.push_back(edges[idx].dst);
            idx = edges[idx].next_out;
        }

        for (T nbr : neighbors)
            removeEdge(nodeId, nbr);

        nodeHeads.erase(it);
    }

    // Merge another adjacency list into this one
    void graphEdit(const AdjacencyList<T>& other) {
        for (const auto& [node, head] : other.nodeHeads) {
            int idx = head;
            while (idx != -1) {
                const HalfEdge& e = other.edges[idx];
                addEdge(e.src, e.dst);
                idx = e.next_out;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    bool hasEdge(T a, T b) const {
        return edgeIndex.count(edgeKey(a, b)) > 0;
    }

    template <typename Func>
    void forEachNeighbor(T nodeId, Func fn) const {
        auto it = nodeHeads.find(nodeId);
        if (it == nodeHeads.end()) return;
        int idx = it->second;
        while (idx != -1) {
            fn(edges[idx].dst);
            idx = edges[idx].next_out;
        }
    }

    // Reconstructs an unordered_map view for compatibility. Allocates -- avoid
    // calling this every frame.
    std::unordered_map<T, std::unordered_set<T>> getAdjList() const {
        std::unordered_map<T, std::unordered_set<T>> result;
        for (const auto& [node, head] : nodeHeads) {
            result[node] = {};
            int idx = head;
            while (idx != -1) {
                result[node].insert(edges[idx].dst);
                idx = edges[idx].next_out;
            }
        }
        return result;
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
        ids.reserve(nodeHeads.size());
        for (const auto& [node, _] : nodeHeads)
            ids.push_back(node);
        return ids;
    }

    uint32_t nodeCount() const { return static_cast<uint32_t>(nodeHeads.size()); }

    // -------------------------------------------------------------------------
    // buildGPUAdjacencyList — compiled by NVCC only
    //
    // Constructs a CSR adjacency list entirely on the GPU from the current
    // COO edge list.
    //
    // d_morton_sorted_ids  Device pointer to N node IDs in desired sort order
    //                      (e.g. from an upstream Morton sort pass). Pass
    //                      nullptr to use insertion order.
    //
    // out                  Reused across frames. Device buffers are grown with
    //                      the doubling realloc helper and never shrunk.
    //
    // stream               All GPU work is issued onto this stream.
    // -------------------------------------------------------------------------
#ifdef __CUDACC__
    void buildGPUAdjacencyList(
        AdjacencyListGPU<T>& out,
        const uint32_t*      d_morton_sorted_ids = nullptr,
        uint32_t             num_sorted_ids = 0,
        cudaStream_t         stream = 0) const
    {

        const bool     use_subset = (d_morton_sorted_ids != nullptr) && (num_sorted_ids > 0);
        const uint32_t N_all      = static_cast<uint32_t>(nodeHeads.size());
        const uint32_t N          = use_subset ? num_sorted_ids : N_all;
        const uint32_t E_all      = static_cast<uint32_t>(coo_src.size());

        if (N == 0 || E_all == 0) return;

        out.num_nodes = N;
        // out.num_edges is set after COO filtering below

        // ------------------------------------------------------------------
        // Grow persistent output buffers as needed.
        // Non-owned buffers (slab slices) must already have enough capacity;
        // we assert rather than reallocate them.
        // ------------------------------------------------------------------
        cuda_realloc_if_needed(&out.nodes, &out.nodes_capacity, N);
        if (out.nbrs_owned) {
            cuda_realloc_if_needed(&out.nbrs, &out.nbrs_capacity, E_all);
        } else {
            assert(out.nbrs_capacity >= E_all &&
                   "slab nbrs slice is too small for this foam");
        }
        if (out.node_offsets_owned) {
            cuda_realloc_if_needed(&out.node_offsets, &out.node_offsets_capacity, N + 1);
        } else {
            assert(out.node_offsets_capacity >= N + 1 &&
                   "slab node_offsets slice is too small for this foam");
        }

        // ------------------------------------------------------------------
        // Step 1 — upload node ordering
        //
        // Build h_nodes unconditionally so we can compute max_entity_id below
        // (entity IDs are globally assigned by entt and may be much larger than N).
        // ------------------------------------------------------------------
        if (d_morton_sorted_ids) {
            // Stay entirely on device — no D2H transfer needed.
            CUDA_CHECK(cudaMemcpyAsync(
                out.nodes, d_morton_sorted_ids,
                N * sizeof(T), cudaMemcpyDeviceToDevice, stream));
        } else {
            std::vector<T> h_nodes;
            h_nodes.reserve(N);
            for (const auto& [node, _] : nodeHeads)
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
        // ------------------------------------------------------------------
        // Size the inverse map to cover ALL node IDs (not just the subset),
        // since the COO contains edges from the full graph and remapCOOKernel
        // will index into this map with any node ID present in the COO.
        uint32_t max_entity_id = 0;
        for (const auto& [node, _] : nodeHeads)
            max_entity_id = std::max(max_entity_id, static_cast<uint32_t>(node));

        uint32_t* d_inverse_map = nullptr;
        const uint32_t inv_map_size = max_entity_id + 1;
        CUDA_CHECK(cudaMalloc(&d_inverse_map, inv_map_size * sizeof(uint32_t)));
        // When building for a subset, fill with 0xFF (UINT32_MAX per slot) so
        // out-of-subset nodes produce a sentinel detectable in step 3b.
        CUDA_CHECK(cudaMemset(d_inverse_map, use_subset ? 0xFF : 0x00,
            inv_map_size * sizeof(uint32_t)));

        buildInverseMapKernel<<<grid_size(N), 256, 0, stream>>>(
            reinterpret_cast<const uint32_t*>(out.nodes),
            d_inverse_map, N);

        // ------------------------------------------------------------------
        // Step 3 — upload COO and remap both ends to sorted positions
        // ------------------------------------------------------------------
        uint32_t* d_coo_src = nullptr;
        uint32_t* d_coo_dst = nullptr;
        CUDA_CHECK(cudaMalloc(&d_coo_src, E * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_coo_dst, E * sizeof(uint32_t)));

        CUDA_CHECK(cudaMemcpy(d_coo_src, coo_src.data(),
            E * sizeof(T), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coo_dst, coo_dst.data(),
            E * sizeof(T), cudaMemcpyHostToDevice));

        remapCOOKernel<T><<<grid_size(E_all), 256, 0, stream>>>(
            reinterpret_cast<const T*>(d_coo_src), d_coo_src, d_inverse_map, E_all);
        remapCOOKernel<T><<<grid_size(E_all), 256, 0, stream>>>(
            reinterpret_cast<const T*>(d_coo_dst), d_coo_dst, d_inverse_map, E_all);

        // ------------------------------------------------------------------
        // Step 3b — filter COO to subset (only when use_subset)
        //
        // Nodes outside the subset have d_inverse_map entries == UINT32_MAX.
        // Discard any edge where either remapped endpoint is UINT32_MAX.
        // ------------------------------------------------------------------
        uint32_t E = E_all;
        if (use_subset) {
            uint32_t* d_coo_src_filt = nullptr;
            uint32_t* d_coo_dst_filt = nullptr;
            CUDA_CHECK(cudaMalloc(&d_coo_src_filt, E_all * sizeof(uint32_t)));
            CUDA_CHECK(cudaMalloc(&d_coo_dst_filt, E_all * sizeof(uint32_t)));

            auto src_in  = thrust::device_ptr<uint32_t>(d_coo_src);
            auto dst_in  = thrust::device_ptr<uint32_t>(d_coo_dst);
            auto src_out = thrust::device_ptr<uint32_t>(d_coo_src_filt);
            auto dst_out = thrust::device_ptr<uint32_t>(d_coo_dst_filt);

            auto in_begin  = thrust::make_zip_iterator(thrust::make_tuple(src_in,  dst_in));
            auto out_begin = thrust::make_zip_iterator(thrust::make_tuple(src_out, dst_out));

            auto new_end = thrust::copy_if(
                thrust::cuda::par.on(stream),
                in_begin, in_begin + E_all, out_begin,
                [] __device__ (thrust::tuple<uint32_t, uint32_t> t) {
                    return thrust::get<0>(t) != 0xFFFFFFFFu
                        && thrust::get<1>(t) != 0xFFFFFFFFu;
                });

            E = static_cast<uint32_t>(new_end - out_begin);

            CUDA_CHECK(cudaFree(d_coo_src));
            CUDA_CHECK(cudaFree(d_coo_dst));
            d_coo_src = d_coo_src_filt;
            d_coo_dst = d_coo_dst_filt;
        }

        out.num_edges = E;

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
        AdjacencyListGPU<T>& out,
        uint32_t*            d_nbrs_slice,
        size_t               nbrs_cap,
        uint32_t*            d_node_offsets_slice,
        size_t               node_offsets_cap,
        const uint32_t*      d_morton_sorted_ids = nullptr,
        uint32_t             num_sorted_ids = 0,
        cudaStream_t         stream = 0) const
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
        buildGPUAdjacencyList(out, d_morton_sorted_ids, num_sorted_ids, stream);
    }
#endif // __CUDACC__

private:

    // -------------------------------------------------------------------------
    // Half-edge storage
    // -------------------------------------------------------------------------

    struct HalfEdge {
        T    src;
        T    dst;
        int  twin;      // index of reverse half-edge in edges[]
        int  next_out;  // next outgoing half-edge from src (-1 if last)
        bool alive;
    };

    std::vector<HalfEdge> edges;
    std::stack<int>       freeList;  // indices of dead slots ready to reuse

    // node -> index of first outgoing half-edge (-1 if isolated)
    std::unordered_map<T, int> nodeHeads;

    // O(1) existence check. Does not store neighbor data.
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

    // Return a slot index, reusing from free list before appending
    int allocSlot() {
        if (!freeList.empty()) {
            int idx = freeList.top();
            freeList.pop();
            return idx;
        }
        edges.push_back({});
        return static_cast<int>(edges.size()) - 1;
    }

    // Remove a single undirected edge (a, b), unlinking both half-edges and
    // returning their slots to the free list.
    void removeEdge(T a, T b) {
        uint64_t key = edgeKey(a, b);
        if (!edgeIndex.count(key)) return;
        edgeIndex.erase(key);

        // Walk a's outgoing list to find the a->b half-edge, then splice it
        // out. Simultaneously locate and splice the b->a twin from b's list.
        int* cursor = &nodeHeads[a];
        while (*cursor != -1) {
            HalfEdge& e = edges[*cursor];
            if (e.dst == b) {
                int idxAB = *cursor;
                int idxBA = e.twin;

                // Unlink a->b from a's list
                *cursor = e.next_out;

                // Unlink b->a from b's list
                int* bcursor = &nodeHeads[b];
                while (*bcursor != -1 && *bcursor != idxBA)
                    bcursor = &edges[*bcursor].next_out;
                if (*bcursor == idxBA)
                    *bcursor = edges[idxBA].next_out;

                edges[idxAB].alive = false;
                edges[idxBA].alive = false;
                freeList.push(idxAB);
                freeList.push(idxBA);

                rebuildCOO();
                return;
            }
            cursor = &e.next_out;
        }
    }

    void rebuildCOO() {
        coo_src.clear();
        coo_dst.clear();
        for (const auto& e : edges) {
            if (e.alive) {
                coo_src.push_back(e.src);
                coo_dst.push_back(e.dst);
            }
        }
    }
};

} // namespace DynamicFoam::Sim2D
