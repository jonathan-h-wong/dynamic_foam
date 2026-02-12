#pragma once
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <entt/entt.hpp>

namespace dynamic_foam::sim2d {
    class AdjacencyList {
    public:
        // Constructor: initializes with a list of node IDs
        AdjacencyList(const std::vector<entt::entity>& nodeIds);

        // Add Node: adds a node to the adjacency list
        void addNode(entt::entity nodeId);

        // Add Node Edges: adds bidirectional edges for a list of connections
        void addNodeEdges(entt::entity nodeId, const std::vector<entt::entity>& connections);

        // Delete Node: removes node and all its edges
        void deleteNode(entt::entity nodeId);

        // Graph Edit: merges another adjacency list into this one
        void graphEdit(const AdjacencyList& other);

        // Accessor for adjacency list
        const std::unordered_map<entt::entity, std::unordered_set<entt::entity>>& getAdjList() const;

    private:
        std::unordered_map<entt::entity, std::unordered_set<entt::entity>> adjList;
    };
}
