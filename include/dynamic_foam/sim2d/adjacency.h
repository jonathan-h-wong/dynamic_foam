#pragma once
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace DynamicFoam::Sim2D {
  template <typename T> class AdjacencyList {
  public:

    // Constructor: initializes with a list of node IDs
    AdjacencyList(const std::vector<T> &nodeIds) {
      for (auto id : nodeIds) {
        adjList[id] = {};
      }
    }

    // Templated constructor for mapping from another adjacency list type
    template <typename U, typename Func>
    AdjacencyList(const AdjacencyList<U>& other, Func map_func) {
        adj.resize(other.adj.size());
        for (size_t i = 0; i < other.adj.size(); ++i) {
            for (const auto& neighbor : other.adj[i]) {
                adj[i].push_back(map_func(neighbor));
            }
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

    // Templated copy constructor for type conversion
    template <typename U>
    AdjacencyList(const AdjacencyList<U>& other, const std::function<T(U)>& converter) {
        const auto& otherAdjList = other.getAdjList();
        for (const auto& [node, neighbors] : otherAdjList) {
            T newNode = converter(node);
            addNode(newNode);
            std::vector<T> newNeighbors;
            for (const auto& neighbor : neighbors) {
                newNeighbors.push_back(converter(neighbor));
            }
            addNodeEdges(newNode, newNeighbors);
        }
    }

    // Export to GPU-friendly format
    AdjacencyListGPU<T> exportToGPU() const {
        AdjacencyListGPU<T> gpuList;
        gpuList.num_nodes = adjList.size();
        
        std::vector<T> connections;
        std::vector<int> offsets;
        offsets.push_back(0);

        for (const auto& pair : adjList) {
            for (const auto& neighbor : pair.second) {
                connections.push_back(neighbor);
            }
            offsets.push_back(connections.size());
        }

        gpuList.num_connections = connections.size();

        cudaMalloc(&gpuList.d_connections, gpuList.num_connections * sizeof(T));
        cudaMemcpy(gpuList.d_connections, connections.data(), gpuList.num_connections * sizeof(T), cudaMemcpyHostToDevice);

        cudaMalloc(&gpuList.d_offsets, (gpuList.num_nodes + 1) * sizeof(int));
        cudaMemcpy(gpuList.d_offsets, offsets.data(), (gpuList.num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);

        return gpuList;
    }

  private:
    std::unordered_map<T, std::unordered_set<T>> adjList;
  };

  // GPU-friendly representation of the adjacency list
  template <typename T>
  struct AdjacencyListGPU {
      T* d_connections;
      int* d_offsets;
      int num_nodes;
      int num_connections;
  };
}