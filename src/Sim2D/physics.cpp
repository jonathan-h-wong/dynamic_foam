#include "dynamic_foam/sim2d/physics.h"

namespace DynamicFoam::Sim2D {
    void Physics::update(
        entt::registry& foamRegistry,
        entt::registry& particleRegistry,
        const std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists,
        float deltaTime
    ) {
        // TODO: Implement physics update logic
    }
}