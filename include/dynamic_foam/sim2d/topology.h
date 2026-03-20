#pragma once
#include <optional>
#include <vector>
#include <entt/entt.hpp>
#include "dynamic_foam/Sim2D/adjacency.cuh"
#include "dynamic_foam/Sim2D/bvh.cuh"
#include "dynamic_foam/Sim2D/components.h"

namespace DynamicFoam::Sim2D {

    // Immutable snapshot of a foam's rigid body state captured before any
    // topology mutations are applied. Using a snapshot rather than reading
    // live registry state ensures that post-processing is correct even when
    // the topology subsystem reuses entity ids or mutates the parent in-place.
    struct FoamSnapshot {
        entt::entity foamId;
        // Persistent
        Density         density;
        InertiaTensor   inertiaTensor;
        bool            isStatic     = false;
        bool            isDynamic    = false;
        bool            isController = false;
        // Transient
        Position        position;
        Velocity        velocity;
        Orientation     orientation;
        AngularVelocity angularVelocity;
    };

    // Describes a single foam that was structurally modified by Topology::update.
    struct TopologyUpdateResult {
        // The foam whose topology changed.
        entt::entity foamId;
        // Snapshot of the parent foam taken before mutations were applied.
        // nullopt when foamId is a pre-existing foam with no parent (e.g. a simple
        // deformation rather than a split).
        std::optional<FoamSnapshot> parentFoamSnapshot;
        // Particles within foamId whose world positions need refreshing.
        std::vector<entt::entity> updatedParticles;
    };

    class Topology {
    public:
        Topology() = default;
        ~Topology() = default;

        // Returns one result per foam that was structurally modified this tick.
        std::vector<TopologyUpdateResult> update(
            const std::unordered_map<int, AABB>&                         foamAABBs,
            const std::unordered_map<int, BVH>&                          foamBVHs,
            std::unordered_map<int, AdjacencyList<entt::entity>>&        foamAdjacencyLists,
            entt::registry&                                              foamRegistry,
            entt::registry&                                              particleRegistry
        );
    };
}
