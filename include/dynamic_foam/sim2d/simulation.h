#pragma once
#include <glm/glm.hpp>
#include <entt/entity/registry.hpp>
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
            const std::unordered_map<int, AABB>&                         foamAABBs,
            const std::unordered_map<int, BVH>&                          foamBVHs,
            std::unordered_map<int, AdjacencyList<entt::entity>>&        foamAdjacencyLists);
        void updatePhysics(
            const std::unordered_map<int, AABB>&                         foamAABBs,
            const std::unordered_map<int, BVH>&                          foamBVHs,
            float deltaTime);
        void render(
            const std::unordered_map<int, AABB>&                         foamAABBs,
            const std::unordered_map<int, BVH>&                          foamBVHs,
            const std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists,
            const entt::registry&                                        particleRegistry
        );
        void step(const UserInput& input, float deltaTime);

        // Returns the device-side RGBA output buffer produced by the last render call.
        // Valid only after the first call to step(). Lifetime is managed by the Render subsystem.
        const glm::vec4* deviceOutputBuffer() const { return renderSubsystem.deviceOutputBuffer(); }

    private:
        void applyForwardKinematics(
            entt::entity foamEntity,
            const std::optional<std::unordered_set<entt::entity>>& particleSubset = std::nullopt
        );

        // Rebuilds the BVH for a foam body by reading ParticleVertices directly
        // from the particle registry, ordered by getOrderedNodeIds() so that
        // BVH prim_idx matches the foam-local sorted position used by the narrowphase kernel.
        void buildBVH(entt::entity foamEntity);

        // Rebuilds the world-space AABB for a foam body by collecting
        // ParticleWorldPosition values for all particles in getOrderedNodeIds() order.
        void buildAABB(entt::entity foamEntity);
        
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
