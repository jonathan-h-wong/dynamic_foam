#pragma once
#include <vector>
#include <entt/entity/registry.hpp>
#include "dynamic_foam/Sim2D/adjacency.cuh"
#include "dynamic_foam/Sim2D/gpu_slab.cuh"
#include "dynamic_foam/Sim2D/components.h"

namespace DynamicFoam::Sim2D {

    // Describes a single foam that was structurally modified by Topology::update.
    struct FoamTopologyUpdate {
        // The foam whose topology changed.
        entt::entity foamId;
        // The particle insertions and deletions to apply to the GPU slab for this foam.
        FoamUpdate foamUpdate;
    };

    class Topology {
    public:
        Topology() = default;
        ~Topology() = default;

        // Returns one result per foam that was structurally modified this tick.
        std::vector<FoamTopologyUpdate> update(
            const GpuSlabAllocator&                              gpuSlab,
            std::unordered_map<int, AdjacencyList>&              foamAdjacencyLists,
            const std::unordered_map<int, glm::mat4>&            foamTransforms,
            entt::registry&                                      particleRegistry
        );
    };
}
