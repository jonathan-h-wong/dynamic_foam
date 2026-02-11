#include "dynamic_foam/sim2d/adjacency.h"

namespace DynamicFoam {
    namespace Sim2D {
        // Constructor: initializes with a list of node IDs
        AdjacencyList::AdjacencyList(const std::vector<int>& nodeIds) {
            for (int id : nodeIds) {
                adjList[id] = std::unordered_set<int>();
            }
        }

        // Add Node: adds a node to the adjacency list
        void AdjacencyList::addNode(int nodeId) {
            // Only add if not already present
            if (adjList.find(nodeId) == adjList.end()) {
                adjList[nodeId] = std::unordered_set<int>();
            }
        }

        // Add Node Edges: adds bidirectional edges for a list of connections
        void AdjacencyList::addNodeEdges(int nodeId, const std::vector<int>& connections) {
            // Ensure node exists
            addNode(nodeId);
            for (int conn : connections) {
                addNode(conn); // Ensure connection exists
                adjList[nodeId].insert(conn);
                adjList[conn].insert(nodeId);
            }
        }

        // Delete Node: removes node and all its edges
        void AdjacencyList::deleteNode(int nodeId) {
            if (adjList.find(nodeId) != adjList.end()) {
                for (int neighbor : adjList[nodeId]) {
                    adjList[neighbor].erase(nodeId);
                }
                adjList.erase(nodeId);
            }
        }

        // Graph Edit: merges another adjacency list into this one
        void AdjacencyList::graphEdit(const AdjacencyList& other) {
            for (const auto& [node, neighbors] : other.adjList) {
                addNodeEdges(node, std::vector<int>(neighbors.begin(), neighbors.end()));
            }
        }

        // Accessor for adjacency list
        const std::unordered_map<int, std::unordered_set<int>>& AdjacencyList::getAdjList() const {
            return adjList;
        }
    }
}