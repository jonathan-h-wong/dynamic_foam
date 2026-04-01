#pragma once
#include <glm/glm.hpp>
#include <entt/entity/registry.hpp>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include "dynamic_foam/Sim2D/adjacency.cuh"
#include "dynamic_foam/Sim2D/bvh.cuh"
#include "dynamic_foam/Sim2D/gpu_slab.cuh"
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
        void handleUserInput(const UserInput& input, float deltaTime);
        void updateTopology(
            const std::unordered_map<int, AABB>&                         foamAABBs,
            const std::unordered_map<int, BVH>&                          foamBVHs,
            std::unordered_map<int, AdjacencyList<entt::entity>>&        foamAdjacencyLists);
        void updatePhysics(
            const std::unordered_map<int, AABB>&                         foamAABBs,
            const std::unordered_map<int, BVH>&                          foamBVHs,
            float deltaTime);
        void render();
        void step(const UserInput& input, float deltaTime);

        // Returns the device-side RGBA output buffer produced by the last render call.
        // Valid only after the first call to step(). Lifetime is managed by the Render subsystem.
        const glm::vec4* deviceOutputBuffer() const { return renderSubsystem.deviceOutputBuffer(); }

        // Persistent render overlay configuration — mutate freely between steps.
        RenderOverlayParams overlayParams;

        // Camera descriptor — mutate freely between steps.
        // origin/lookAt/up/width/type/fovY are logical parameters;
        // height is recomputed from the aspect ratio each frame inside render().
        CameraParams camera_;

    private:
        void applyForwardKinematics(
            entt::entity foamEntity,
            const std::optional<std::unordered_set<entt::entity>>& particleSubset = std::nullopt
        );

        // Resolves the latest cursor screen position to world space and snaps
        // all Controller foam bodies to it.  Separated from handleUserInput so
        // it can be called a second time immediately before render() to minimise
        // input-to-display latency without re-applying camera or click logic.
        void applyControllerCursor(ImVec2 mouse_pos);

        // Rebuilds the BVH for a foam body by reading ParticleVertices directly
        // from the particle registry, ordered by getOrderedNodeIds() so that
        // BVH prim_idx matches the foam-local sorted position used by the narrowphase kernel.
        // If a slab slot already exists for the foam, builds directly into the slab slice.
        void buildBVH(entt::entity foamEntity);

        // Rebuilds the world-space AABB for a foam body by collecting
        // ParticleWorldPosition values for all particles in getOrderedNodeIds() order.
        void buildAABB(entt::entity foamEntity);

        // Allocates and populates the GpuSlabAllocator from the CPU-side foam
        // structures.  Implemented in simulation_gpu.cu (requires nvcc).
        // Called once at the end of the constructor.
        void initSlab();

        // Rebuilds every live foam's GPU CSR adjacency (node_offsets + nbrs)
        // from the CPU-side adjacency lists into the current slab slices.
        // Must be called after gpuSlab.compact() to fix the CSR node_offsets,
        // which are invalidated when csr_edge_offsets shift during compaction.
        void rebuildAllSlabCsr();
        
        entt::registry foamRegistry;
        entt::registry particleRegistry;
        std::unordered_map<int, AdjacencyList<entt::entity>> foamAdjacencyLists;
        std::unordered_map<int, BVH>  foamBVHs;
        std::unordered_map<int, AABB> foamAABBs;

        // GPU slab allocator — flat device buffers for BVH, CSR, and particle
        // render data for all foams.  Built once in the constructor; updated
        // incrementally when topology changes.
        GpuSlabAllocator gpuSlab;
        // Per-foam GPU adjacency list handles (nbrs/node_offsets are slab slices).
        std::unordered_map<int, AdjacencyListGPU<entt::entity>> foamGpuAdj;
        
        glm::ivec2 windowSize_;

        Topology topologySubsystem;
        Physics physicsSubsystem;
        Render renderSubsystem;
    };

}
