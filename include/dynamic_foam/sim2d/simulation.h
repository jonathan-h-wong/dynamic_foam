#pragma once
#include <glm/glm.hpp>
#include <entt/entt.hpp>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include "dynamic_foam/Sim2D/adjacency.cuh"
#include "dynamic_foam/Sim2D/bvh.cuh"
#include "dynamic_foam/Sim2D/scenegraph.h"
#include "dynamic_foam/Sim2D/user_input.h"
#include "dynamic_foam/Sim2D/topology.h"
#include "dynamic_foam/Sim2D/physics.h"
#include "dynamic_foam/Sim2D/render.cuh"

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
            const entt::registry& particleRegistry, 
            const std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists,
            const std::unordered_map<int, BVH>& foamBVHs,
            const std::unordered_map<int, AABB>& foamAABBs
        );
        void step(const UserInput& input, float deltaTime);

    private:
        void applyForwardKinematics(
            entt::entity foamEntity,
            const std::optional<std::unordered_set<entt::entity>>& particleSubset = std::nullopt
        );
        
        entt::registry foamRegistry;
        entt::registry particleRegistry;
        std::unordered_map<int, AdjacencyList<entt::entity>> foamAdjacencyLists;
        std::unordered_map<int, BVH> foamBVHs;
        std::unordered_map<int, AABB> foamAABBs;
        
        glm::ivec2 windowSize_;

        Topology topologySubsystem;
        Physics physicsSubsystem;
        Render renderSubsystem;
    };

}
