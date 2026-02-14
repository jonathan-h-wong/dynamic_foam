# include "dynamic_foam/sim2d/sample_foams.h"

namespace DynamicFoam::Sim2D {
    Foam generateFoamBar() {
        // Example foam generation logic
        AdjacencyList<int> adjList;
        std::unordered_map<int, bool> stencil;
        std::unordered_map<int, bool> mutable_map;
        std::unordered_map<int, glm::vec3> position;
        std::unordered_map<int, glm::vec3> mass;
        std::unordered_map<int, glm::vec3> color;
        std::unordered_map<int, float> opacity;

        // Populate the above data structures with foam data

        return Foam(adjList, stencil, mutable_map, position, mass, color, opacity);
    }

    Foam generateFoamPointCursor() {
        // Example foam generation logic
        AdjacencyList<int> adjList;
        std::unordered_map<int, bool> stencil;
        std::unordered_map<int, bool> mutable_map;
        std::unordered_map<int, glm::vec3> position;
        std::unordered_map<int, glm::vec3> mass;
        std::unordered_map<int, glm::vec3> color;
        std::unordered_map<int, float> opacity;

        // Populate the above data structures with foam data

        return Foam(adjList, stencil, mutable_map, position, mass, color, opacity);
    }
}