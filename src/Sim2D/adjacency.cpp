#include "dynamic_foam/sim2d/adjacency.h"
#include <entt/entt.hpp>

namespace dynamic_foam::sim2d {
    // Constructor: initializes with a list of node IDs
    AdjacencyList::AdjacencyList(const std::vector<entt::entity>& nodeIds) {
        for (auto id : nodeIds) {
            adjList[id] = {};
        }
    }

    // Add Node: adds a node to the adjacency list
    void AdjacencyList::addNode(entt::entity nodeId) {
        // Only add if not already present
        if (adjList.find(nodeId) == adjList.end()) {
            adjList[nodeId] = {};
        }
    }

    // Add Node Edges: adds bidirectional edges for a list of connections
    void AdjacencyList::addNodeEdges(entt::entity nodeId, const std::vector<entt::entity>& connections) {
        // Ensure node exists
        addNode(nodeId);
        for (auto conn : connections) {
            addNode(conn); // Ensure connection exists
            adjList[nodeId].insert(conn);
            adjList[conn].insert(nodeId);
        }
    }

    // Delete Node: removes node and all its edges
    void AdjacencyList::deleteNode(entt::entity nodeId) {
        if (adjList.find(nodeId) != adjList.end()) {
            for (auto neighbor : adjList[nodeId]) {
                adjList[neighbor].erase(nodeId);
            }
            adjList.erase(nodeId);
        }
    }

    // Graph Edit: merges another adjacency list into this one
    void AdjacencyList::graphEdit(const AdjacencyList& other) {
        for (const auto& [node, neighbors] : other.adjList) {
            addNodeEdges(node, std::vector<entt::entity>(neighbors.begin(), neighbors.end()));
        }
    }

    // Accessor for adjacency list
    const std::unordered_map<entt::entity, std::unordered_set<entt::entity>>& AdjacencyList::getAdjList() const {
        return adjList;
    }
}