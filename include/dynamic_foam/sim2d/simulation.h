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
        void updateTopology();
        void updatePhysics(float deltaTime);
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

        // Bulk upload for a single foam: builds CPU arrays for AABBs, colors,
        // local positions, surface mask, and active IDs from the particle registry,
        // uploads all five buffers to the GPU slab via gpuSlab.stageParticleData in
        // getOrderedNodeIds() order, then calls gpuSlab.bulkMortonSort to reorder
        // every buffer together by local-space Morton code so that d_active_ids is
        // Morton-sorted and ready for buildGPUAdjacencyListFromSlab.
        // d_particle_positions stores local (object) space; the renderer transforms
        // rays per-foam using foam_inv_transforms in k_exact_collision.
        // Must be called after a foam's particle data is fully populated and
        // after gpuSlab.allocate() has reserved a valid slot for the foam.
        void stageParticleData(entt::entity foamEntity);

        // Allocates and populates the GpuSlabAllocator from the CPU-side foam
        // structures.  Implemented in simulation_gpu.cu (requires nvcc).
        // Called once at the end of the constructor.
        void initSlab();

        // Rebuilds the GPU CSR for one foam using the device-resident Morton-sorted
        // d_active_ids and pre-staged d_coo_src/dst buffers.  No H2D transfers.
        // Call after stageParticleData (which populates d_active_ids) and whenever
        // the adjacency or slab layout has changed (resize, compact).
        void rebuildSlabAdj(int foam_id);

        // Applies a FoamUpdate (deletions then insertions) to the GPU particle
        // slab buffers for the given foam.  Delegates to gpuSlab.updateFoamData.
        // The caller is responsible for rebuilding the BVH and CSR adjacency
        // after this call if the particle set has changed.
        void updateParticleData(int foam_id, const FoamUpdate& update);
        
        entt::registry foamRegistry;
        entt::registry particleRegistry;
        std::unordered_map<int, AdjacencyList> foamAdjacencyLists;
        std::unordered_map<int, BVH>  foamBVHs;
        // foamAABBs removed: world-space AABBs are computed on-demand from
        // the GPU slab's d_foam_aabbs (local-space) + foam world transforms.

        // GPU slab allocator — flat device buffers for BVH, CSR, and particle
        // render data for all foams.  Built once in the constructor; updated
        // incrementally when topology changes.
        GpuSlabAllocator gpuSlab;
        // Per-foam GPU adjacency list handles (nbrs/node_offsets are slab slices).
        std::unordered_map<int, AdjacencyListGPU> foamGpuAdj;
        
        glm::ivec2 windowSize_;

        Topology topologySubsystem;
        Physics physicsSubsystem;
        Render renderSubsystem;
    };

}
