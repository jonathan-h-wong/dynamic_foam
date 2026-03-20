#pragma once
#include <vector>
#include <entt/entt.hpp>
#include "dynamic_foam/Sim2D/adjacency.cuh"

namespace DynamicFoam::Sim2D {

class Physics {
    public:
        Physics() = default;
        ~Physics() = default;

        // Returns the entity ids of all foam bodies that were updated this tick.
        std::vector<entt::entity> update(
            entt::registry& foamRegistry,
            entt::registry& particleRegistry,
            const std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists,
            float deltaTime
        );
    };
}
