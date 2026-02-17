#include "dynamic_foam/sim2d/render.h"

namespace DynamicFoam::Sim2D {
    void Render::update(
        entt::registry& foamRegistry,
        entt::registry& particleRegistry,
        const std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists
    ) {
        //TODO: Implement rendering logic
    }
}