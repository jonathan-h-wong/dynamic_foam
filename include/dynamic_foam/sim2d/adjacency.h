#pragma once
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace dynamic_foam::sim2d {
template <typename T> class AdjacencyList {
public:
  // Constructor: initializes with a list of node IDs
  AdjacencyList(const std::vector<T> &nodeIds) {
    for (auto id : nodeIds) {
      adjList[id] = {};
    }
  }

  // Add Node: adds a node to the adjacency list
  void addNode(T nodeId) {
    // Only add if not already present
    if (adjList.find(nodeId) == adjList.end()) {
      adjList[nodeId] = {};
    }
  }

  // Add Node Edges: adds bidirectional edges for a list of connections
  void addNodeEdges(T nodeId, const std::vector<T> &connections) {
    // Ensure node exists
    addNode(nodeId);
    for (auto conn : connections) {
      addNode(conn); // Ensure connection exists
      adjList[nodeId].insert(conn);
      adjList[conn].insert(nodeId);
    }
  }

  // Delete Node: removes node and all its edges
  void deleteNode(T nodeId) {
    if (adjList.find(nodeId) != adjList.end()) {
      for (auto neighbor : adjList[nodeId]) {
        adjList[neighbor].erase(nodeId);
      }
      adjList.erase(nodeId);
    }
  }

  // Graph Edit: merges another adjacency list into this one
  void graphEdit(const AdjacencyList<T> &other) {
    for (const auto &[node, neighbors] : other.adjList) {
      addNodeEdges(node, std::vector<T>(neighbors.begin(), neighbors.end()));
    }
  }

  // Accessor for adjacency list
  const std::unordered_map<T, std::unordered_set<T>> &getAdjList() const {
    return adjList;
  }

private:
  std::unordered_map<T, std::unordered_set<T>> adjList;
};
} // namespace dynamic_foam::sim2d
