#pragma once
#include <entt/entt.hpp>

namespace DynamicFoam::Sim2D {

class Render {
    public:
        Render() = default;
        ~Render() = default;

        void update(
            entt::registry& foamRegistry,
            entt::registry& particleRegistry,
            const std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists,
        );
    };
}
