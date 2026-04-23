#pragma once

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

namespace DynamicFoam::Sim2D {

// =============================================================================
// AdjacencyList
//
// Stores an undirected graph as an adjacency set map plus a flat COO cache.
//
//   adj         — node -> unordered_set of neighbor IDs
//   edgeIndex   — packed 64-bit key set for O(1) existence checks
// =============================================================================
// Note: COO edge data is built on-demand via buildCOO() and is not cached.
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

    }

    void addNodeEdges(uint32_t nodeId, const std::vector<uint32_t>& connections) {
        for (auto conn : connections)
            addEdge(nodeId, conn);
    }

    // Pre-size the edge dedup index. Call before a bulk addEdge sequence
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
    }

    // Strict induced-subgraph copy: copy only nodes that pass `nodeFilter`,
    // and only edges where BOTH endpoints pass. No stray neighbours are
    // introduced, so the caller controls the exact node set explicitly.
    // Use a prior BFS expansion to grow the seed set before calling if you
    // want to include a surrounding shell.
    //
    // Example:
    //   child.copyNodesFrom(parent, [&](uint32_t id){ return keep.count(id); });
    template <typename Pred>
    void copyNodesFrom(const AdjacencyList& other, Pred nodeFilter) {
        for (const auto& [node, neighbors] : other.adj) {
            if (!nodeFilter(node)) continue;
            addNode(node);
            for (uint32_t nbr : neighbors)
                if (nodeFilter(nbr))
                    addEdge(node, nbr);
        }
    }

    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------

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

    // Build and return all directed COO edges as a src/dst pair.
    // Called once during initialisation; result is not cached.
    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> buildCOO() const {
        std::vector<uint32_t> src, dst;
        for (const auto& [node, neighbors] : adj)
            for (uint32_t nbr : neighbors) {
                src.push_back(node);
                dst.push_back(nbr);
            }
        return {src, dst};
    }

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

private:

    // node -> set of neighbor IDs
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> adj;

    // O(1) existence check (canonical undirected key)
    std::unordered_set<uint64_t> edgeIndex;

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
