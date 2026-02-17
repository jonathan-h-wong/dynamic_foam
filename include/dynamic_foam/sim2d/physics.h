#pragma once
#include <entt/entt.hpp>

namespace DynamicFoam::Sim2D {

class Physics {
    public:
        Physics() = default;
        ~Physics() = default;

        void update(
            entt::registry& foamRegistry,
            entt::registry& particleRegistry,
            const std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists,
            float deltaTime
        );
    };
}
