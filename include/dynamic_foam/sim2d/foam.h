#pragma once

#include <glm/glm.hpp>
#include <stdexcept>
#include <unordered_map>
#include "dynamic_foam/sim2d/adjacency.h"

namespace DynamicFoam::Sim2D {
    class Foam {
    public:
        // Topology
        AdjacencyList<int> adjacencyList;
        std::unordered_map<int, bool> isStencil;
        std::unordered_map<int, bool> isMutable;

        // Local Space Position
        std::unordered_map<int, glm::vec3> particlePosition;

        // Physics
        float density;
        std::unordered_map<int, glm::vec3> particleMass;

        // Rendering
        std::unordered_map<int, glm::vec3> particleColor;
        std::unordered_map<int, float> particleOpacity;

        Foam(
            const AdjacencyList<int>& adjList,
            const std::unordered_map<int, bool>& stencil,
            const std::unordered_map<int, bool>& mutable_map,
            const std::unordered_map<int, glm::vec3>& position,
            const std::unordered_map<int, glm::vec3>& mass,
            const std::unordered_map<int, glm::vec3>& color,
            const std::unordered_map<int, float>& opacity,
            float density = 1.0f
        ) : adjacencyList(adjList),
            isStencil(stencil),
            isMutable(mutable_map),
            particlePosition(position),
            particleMass(mass),
            particleColor(color),
            particleOpacity(opacity),
            density(density) {
            validate();
        }

    private:
        /*
        Validation logic for foam types based on their stencil and mutable properties:
        Stencil | Mutable
        Y        Y       Not allowed, validated here
        Y        N       Default stencil
        N        Y       Default subject foam
        N        N       Default background foam
        */
        void validate() {
            for (const auto& [id, stencil] : isStencil) {
                if (stencil) {
                    auto mutableIt = isMutable.find(id);
                    if (mutableIt != isMutable.end() && mutableIt->second) {
                        throw std::runtime_error("A particle cannot be both a stencil and mutable.");
                    }
                }
            }
        }
    };
}