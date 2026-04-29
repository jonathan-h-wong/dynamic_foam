#pragma once
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <entt/entity/registry.hpp>
#include "dynamic_foam/Sim2D/adjacency.cuh"
#include "dynamic_foam/Sim2D/bvh.cuh"
#include "dynamic_foam/Sim2D/collision.cuh"
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

    // A foamB-local point that landed inside a mutable cell during boundary sampling.
    struct BoundaryHit {
        glm::vec3    localPoint;
        entt::entity mutParticle;
    };

    // Aggregated output of the first three pipeline steps (collision filtering,
    // boundary sampling, spanning AABB filter).
    struct StencilWorkResult {
        // stencil -> foamB -> candidate mutable cell ids (from AABB collision).
        std::unordered_map<entt::entity,
            std::unordered_map<int, std::vector<entt::entity>>>    cellContacts;
        // stencil -> foamB -> AABB over all foamB-local boundary samples.
        std::unordered_map<entt::entity,
            std::unordered_map<int, AABB>>                         effectiveAABB;
        // stencil -> foamB -> boundary hits that pass the spanning-axis filter.
        std::unordered_map<entt::entity,
            std::unordered_map<int, std::vector<BoundaryHit>>>     validWork;
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
            const entt::registry&                                foamRegistry,
            entt::registry&                                      particleRegistry
        );

    private:
        // Steps 1–3: collect collisions, sample boundary points, filter by spanning AABB.
        static StencilWorkResult buildStencilWork(
            const std::vector<FoamCollision>&                    collisions,
            const std::unordered_map<int, AdjacencyList>&        foamAdjacencyLists,
            const std::unordered_map<int, glm::mat4>&            foamTransforms,
            const entt::registry&                                particleRegistry
        );

        // Returns true if point P lies inside the Voronoi cell centered at C.
        static bool voronoiContains(
            const glm::vec3&              P,
            const glm::vec3&              C,
            const std::vector<glm::vec3>& neighborCenters
        );

        // If P is closer to the C↔N bisector plane than C itself, returns P mirrored
        // across that plane; otherwise returns std::nullopt.
        static std::optional<glm::vec3> faceReflect(
            const glm::vec3& P,
            const glm::vec3& C,
            const glm::vec3& N
        );

        // Create a new interior particle inheriting appearance from cellId.
        // Registers the new particle in stencilDirtyCells and cellSamples.
        static entt::entity addInteriorParticle(
            entt::entity                                                          stencilId,
            entt::entity                                                          cellId,
            glm::vec3                                                             localPos,
            entt::registry&                                                       registry,
            std::unordered_map<entt::entity, std::unordered_set<entt::entity>>&  stencilDirtyCells,
            std::unordered_map<entt::entity, std::unordered_set<entt::entity>>&  cellSamples
        );

        // Create a zero-opacity air ghost particle.
        // Registers the new particle in cellSamples.
        static entt::entity addAirParticle(
            entt::entity                                                          cellId,
            glm::vec3                                                             localPos,
            entt::registry&                                                       registry,
            std::unordered_map<entt::entity, std::unordered_set<entt::entity>>&  cellSamples
        );
    };
}
