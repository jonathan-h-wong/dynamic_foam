#pragma once
#include <vector>
#include <entt/entity/registry.hpp>
#include "dynamic_foam/Sim2D/adjacency.cuh"
#include "dynamic_foam/Sim2D/bvh.cuh"

namespace DynamicFoam::Sim2D {

class Physics {
    public:
        Physics() = default;
        ~Physics() = default;

        // Returns the entity ids of all foam bodies that were updated this tick.
        std::vector<entt::entity> update(
            const std::unordered_map<int, AABB>&                         foamAABBs,
            const std::unordered_map<int, BVH>&                          foamBVHs,
            const std::unordered_map<int, AdjacencyList>&                foamAdjacencyLists,
            entt::registry&                                              foamRegistry,
            const entt::registry&                                        particleRegistry,
            float                                                        deltaTime
        );
    };
}
