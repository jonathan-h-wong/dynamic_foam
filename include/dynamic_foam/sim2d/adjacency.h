#pragma once
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace DynamicFoam {
    namespace Sim2D {
        class AdjacencyList {
        public:
            // Constructor: initializes with a list of node IDs
            AdjacencyList(const std::vector<int>& nodeIds);

            // Add Node: adds a node to the adjacency list
            void addNode(int nodeId);

            // Add Node Edges: adds bidirectional edges for a list of connections
            void addNodeEdges(int nodeId, const std::vector<int>& connections);

            // Delete Node: removes node and all its edges
            void deleteNode(int nodeId);

            // Graph Edit: merges another adjacency list into this one
            void graphEdit(const AdjacencyList& other);

            // Accessor for adjacency list
            const std::unordered_map<int, std::unordered_set<int>>& getAdjList() const;

        private:
            std::unordered_map<int, std::unordered_set<int>> adjList;
        };
    }
}
