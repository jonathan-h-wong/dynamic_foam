#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

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
struct AdjacencyListGPU {
    uint32_t* nodes        = nullptr;
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
// AdjacencyList
//
// Stores an undirected graph as an adjacency set map plus a flat COO cache.
//
//   adj         — node -> unordered_set of neighbor IDs
//   edgeIndex   — packed 64-bit key set for O(1) existence checks
//   coo_src/dst — flat directed edge list kept current by all mutations
// =============================================================================
class AdjacencyList {
public:

    // -------------------------------------------------------------------------
    // Construction
    // -------------------------------------------------------------------------

    AdjacencyList() = default;

    explicit AdjacencyList(const std::vector<uint32_t>& nodeIds) {
        for (auto id : nodeIds) addNode(id);
    }

    // Returns a new AdjacencyList with all node IDs remapped via idMap.
    AdjacencyList remap(const std::unordered_map<uint32_t, uint32_t>& idMap) const {
        AdjacencyList result;
        for (const auto& [node, neighbors] : adj)
            for (uint32_t nbr : neighbors)
                result.addEdge(idMap.at(node), idMap.at(nbr));
        return result;
    }

    // -------------------------------------------------------------------------
    // Mutation
    // -------------------------------------------------------------------------

    void addNode(uint32_t nodeId) {
        adj.try_emplace(nodeId);
    }

    // Add an undirected edge between a and b. No-op if already exists.
    void addEdge(uint32_t a, uint32_t b) {
        addNode(a);
        addNode(b);

        uint64_t key = edgeKey(a, b);
        if (edgeIndex.count(key)) return;
        edgeIndex.insert(key);

        adj[a].insert(b);
        adj[b].insert(a);

        coo_dirty = true;
    }

    void addNodeEdges(uint32_t nodeId, const std::vector<uint32_t>& connections) {
        for (auto conn : connections)
            addEdge(nodeId, conn);
    }

    // Delete a node and all its incident edges.
    void deleteNode(uint32_t nodeId) {
        auto it = adj.find(nodeId);
        if (it == adj.end()) return;

        for (uint32_t nbr : it->second) {
            adj[nbr].erase(nodeId);
            edgeIndex.erase(edgeKey(nodeId, nbr));
        }
        adj.erase(it);
        coo_dirty = true;
    }

    // Merge another adjacency list into this one
    void graphEdit(const AdjacencyList& other) {
        for (const auto& [node, neighbors] : other.adj)
            for (uint32_t nbr : neighbors)
                addEdge(node, nbr);
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

    bool hasEdge(uint32_t a, uint32_t b) const {
        return edgeIndex.count(edgeKey(a, b)) > 0;
    }

    template <typename Func>
    void forEachNeighbor(uint32_t nodeId, Func fn) const {
        auto it = adj.find(nodeId);
        if (it == adj.end()) return;
        for (uint32_t nbr : it->second)
            fn(nbr);
    }

    const std::unordered_map<uint32_t, std::unordered_set<uint32_t>>& getAdjList() const {
        return adj;
    }

    // Returns the flat directed COO edge arrays. Rebuilt lazily on first
    // access after any mutation that sets coo_dirty.
    const std::vector<uint32_t>& getCOOSrc() const { if (coo_dirty) rebuildCOO(); return coo_src; }
    const std::vector<uint32_t>& getCOODst() const { if (coo_dirty) rebuildCOO(); return coo_dst; }

    // Returns node IDs in insertion order. Used by the render layer to build
    // the per-foam slice of the flat position/color/surface-mask buffers in
    // a deterministic order that matches the GPU CSR's sorted node layout.
    // Call after buildGPUAdjacencyList so that insertion order matches the
    // order nodes were added (which is the same order used for the nodes[]
    // buffer when d_morton_sorted_ids is null).
    std::vector<uint32_t> getOrderedNodeIds() const {
        std::vector<uint32_t> ids;
        ids.reserve(adj.size());
        for (const auto& [node, _] : adj)
            ids.push_back(node);
        return ids;
    }

    uint32_t nodeCount() const { return static_cast<uint32_t>(adj.size()); }

    // Returns the subset of the COO where both endpoints are in subset_ids.
    // Call this on the CPU before buildGPUAdjacencyList to avoid uploading and
    // filtering the full edge list on the GPU.
    std::pair<std::vector<uint32_t>, std::vector<uint32_t>>
    getSubsetCOO(const std::vector<uint32_t>& subset_ids) const {
        std::unordered_set<uint32_t> subset_set(subset_ids.begin(), subset_ids.end());
        std::vector<uint32_t> src, dst;
        for (uint32_t node : subset_ids) {
            forEachNeighbor(node, [&](uint32_t nbr) {
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
    // Builds a CSR adjacency list directly into a pre-allocated slab slice.
    // d_nbrs_slice and d_node_offsets_slice are device memory owned by an
    // external allocator (GpuSlabAllocator); they will NOT be freed by this
    // call or by out.free().
    //
    // nbrs_cap         must be >= E (num directed edges in this foam).
    // node_offsets_cap must be >= N + 1 (num nodes + sentinel).
    //
    // sorted_ids        CPU-side node IDs in desired sort order. Pass an empty
    //                   vector to use insertion order.
    //
    // stream            All GPU work is issued onto this stream.
    // -------------------------------------------------------------------------
    void buildGPUAdjacencyList(
        AdjacencyListGPU&            out,
        uint32_t*                    d_nbrs_slice,
        size_t                       nbrs_cap,
        uint32_t*                    d_node_offsets_slice,
        size_t                       node_offsets_cap,
        const std::vector<uint32_t>& sorted_ids = {},
        cudaStream_t                 stream = 0) const;

private:

    // node -> set of neighbor IDs
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> adj;

    // O(1) existence check (canonical undirected key)
    std::unordered_set<uint64_t> edgeIndex;

    // COO cache — rebuilt lazily when coo_dirty is set by a mutation.
    mutable std::vector<uint32_t> coo_src;
    mutable std::vector<uint32_t> coo_dst;
    mutable bool coo_dirty = false;

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    // Core GPU CSR builder — operates on buffers already wired into out.
    void buildGPUAdjacencyListImpl(
        AdjacencyListGPU&            out,
        const std::vector<uint32_t>& sorted_ids,
        cudaStream_t                 stream) const;

    // Canonical edge key: always pack smaller ID into high bits
    static uint64_t edgeKey(uint32_t a, uint32_t b) {
        if (a > b) std::swap(a, b);
        return (uint64_t(a) << 32) | uint64_t(b);
    }

    // Remove a single undirected edge (a, b).
    void removeEdge(uint32_t a, uint32_t b) {
        uint64_t key = edgeKey(a, b);
        if (!edgeIndex.count(key)) return;
        edgeIndex.erase(key);
        adj[a].erase(b);
        adj[b].erase(a);
        coo_dirty = true;
    }

    void rebuildCOO() const {
        coo_src.clear();
        coo_dst.clear();
        for (const auto& [node, neighbors] : adj)
            for (uint32_t nbr : neighbors) {
                coo_src.push_back(node);
                coo_dst.push_back(nbr);
            }
        coo_dirty = false;
    }
};

} // namespace DynamicFoam::Sim2D
