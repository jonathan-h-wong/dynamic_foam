#pragma once
#include <glm/glm.hpp>
#include <entt/entt.hpp>
#include "dynamic_foam/sim2d/adjacency.h"
#include "dynamic_foam/sim2d/bvh.cuh"
#include "dynamic_foam/sim2d/scenegraph.h"
#include "dynamic_foam/sim2d/user_input.h"
#include <dynamic_foam/sim2d/topology.h>
#include <dynamic_foam/sim2d/physics.h>
#include <dynamic_foam/sim2d/render.h>

namespace DynamicFoam::Sim2D {

class Simulation {
    public:
        Simulation(
            const SceneGraph& sceneGraph, 
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
        void step(const UserInput& input, float deltaTime);

    private:
        void applyForwardKinematics(entt::entity controllerFoam);
        
        entt::registry foamRegistry;
        entt::registry particleRegistry;
        std::unordered_map<int, AdjacencyList<entt::entity>> foamAdjacencyLists;
        std::unordered_map<int, BVH> foamBVHs;
        std::unordered_map<int, AABB> foamAABBs;
        std::unordered_map<int, AABB> particleAABBs;

        glm::ivec2 windowSize;
        
        Topology topologySubsystem;
        Physics physicsSubsystem;
        Render renderSubsystem;
    };

}
