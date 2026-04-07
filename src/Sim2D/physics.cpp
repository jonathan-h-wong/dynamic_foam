#include "dynamic_foam/Sim2D/physics.h"

namespace DynamicFoam::Sim2D {
    std::vector<entt::entity> Physics::update(
        const std::unordered_map<int, AABB>&                         foamAABBs,
        const std::unordered_map<int, BVH>&                          foamBVHs,
        const std::unordered_map<int, AdjacencyList>& foamAdjacencyLists,
        entt::registry&                                              foamRegistry,
        const entt::registry&                                        particleRegistry,
        float                                                        deltaTime
    ) {
        std::vector<entt::entity> updatedFoams;
        // TODO: Implement physics update logic
        return updatedFoams;
    }
}