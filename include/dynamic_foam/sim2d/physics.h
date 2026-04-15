#pragma once
#include <vector>
#include <unordered_map>
#include <entt/entity/registry.hpp>
#include <glm/glm.hpp>
#include "dynamic_foam/Sim2D/adjacency.cuh"
#include "dynamic_foam/Sim2D/gpu_slab.cuh"
#include "dynamic_foam/Sim2D/components.h"

namespace DynamicFoam::Sim2D {

    // Per-foam rigid-body state produced by Physics::update.
    // Simulation's post-processing step publishes these values into the
    // foamRegistry, keeping exclusive registry write privileges in Simulation.
    struct FoamPhysicsUpdate {
        entt::entity    foamId;
        Position        position;
        Velocity        velocity;
        Orientation     orientation;
        AngularVelocity angularVelocity;
    };

class Physics {
    public:
        Physics() = default;
        ~Physics() = default;

        // Returns one result per foam body whose state was integrated this tick.
        // foamTransforms is the same read-only world-transform map built by
        // Simulation::render — keyed by static_cast<int>(foamEntity).
        // The physics subsystem must not write to the entt registries directly;
        // results are applied by Simulation's post-processing step.
        std::vector<FoamPhysicsUpdate> update(
            const GpuSlabAllocator&                              gpuSlab,
            const std::unordered_map<int, AdjacencyList>&        foamAdjacencyLists,
            const std::unordered_map<int, glm::mat4>&            foamTransforms,
            const entt::registry&                                particleRegistry,
            float                                                deltaTime
        );
    };
}
