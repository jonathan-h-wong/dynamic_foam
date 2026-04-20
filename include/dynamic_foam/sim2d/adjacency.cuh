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

    // Fast-path edge insert for callers that guarantee:
    //   (a) both endpoints are already registered as nodes, and
    //   (b) each undirected edge is submitted exactly once (no duplicates).
    // Skips the existence check and the redundant addNode try_emplace calls.
    // ~2x faster than addEdge per call. Use for CGAL finite_edges output.
    void addEdgeUnique(uint32_t a, uint32_t b) {
        adj[a].insert(b);
        adj[b].insert(a);
        edgeIndex.insert(edgeKey(a, b));
        coo_dirty = true;
    }

    // Pre-size the edge dedup index. Call before a bulk addEdgeUnique sequence
    // when the expected edge count is known (e.g. dt.number_of_finite_edges()).
    void reserveEdges(size_t numEdges) {
        edgeIndex.reserve(numEdges);
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

// =============================================================================
// buildGPUAdjacencyList — fully GPU-resident CSR adjacency builder.
//
// All inputs are device pointers. The only CPU↔GPU traffic is one scalar
// D2H read (run-length encode count).
//
//   d_sorted_ids          — N Morton-sorted entity IDs
//                           (d_active_ids + slot.active_offset).
//   N                     — particle / node count.
//   d_coo_src / d_coo_dst — E directed edge pairs in raw entity-ID space
//                           (d_coo_src/dst + slot.coo_offset).
//   E                     — directed edge count.
//   d_nbrs_slice          — slab colidx slice (size >= E); NOT freed by out.free().
//   d_node_offsets_slice  — slab rowptr slice (size >= N+1); NOT freed by out.free().
//   stream                — CUDA stream for all kernel/cub work.
// =============================================================================
void buildGPUAdjacencyList(
    AdjacencyListGPU& out,
    const uint32_t*   d_sorted_ids,
    uint32_t          N,
    const uint32_t*   d_coo_src,
    const uint32_t*   d_coo_dst,
    uint32_t          E,
    uint32_t*         d_nbrs_slice,
    size_t            nbrs_cap,
    uint32_t*         d_node_offsets_slice,
    size_t            node_offsets_cap,
    cudaStream_t      stream = 0);

} // namespace DynamicFoam::Sim2D
