#pragma once
#include <glm/glm.hpp>
#include <entt/entt.hpp>
#include "dynamic_foam/sim2d/adjacency.h"
#include "dynamic_foam/sim2d/scenegraph.h"
#include "dynamic_foam/sim2d/user_input.h"

namespace DynamicFoam::Sim2D {

class Simulation {
    public:
        Simulation(
            const SceneGraph& sceneGraph, 
            const UserInput& input,
            const glm::ivec2& windowSize
        );
        ~Simulation() = default;

        // Subsystems
        void handleUserInput(const UserInput& input);
        void updateTopology(
            entt::registry& foamRegistry,
            entt::registry& particleRegistry,
            std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists);
        void updatePhysics(
            entt::registry& foamRegistry,
            entt::registry& particleRegistry,
            float deltaTime);
        void render(
            entt::registry& foamRegistry,
            entt::registry& particleRegistry
        );
        void step(float deltaTime);

    private:
        entt::registry foamRegistry;
        entt::registry particleRegistry;
        std::unordered_map<int, AdjacencyList<entt::entity>> foamAdjacencyLists;
    };

}
