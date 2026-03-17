#pragma once
#include <entt/entt.hpp>
#include "dynamic_foam/Sim2D/adjacency.cuh"

namespace DynamicFoam::Sim2D {
    class Topology {
    public:
        Topology() = default;
        ~Topology() = default;

        void update(
            entt::registry& foamRegistry,
            entt::registry& particleRegistry,
            std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists
        );
    };
}
