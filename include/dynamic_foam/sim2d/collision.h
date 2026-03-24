#pragma once

// =============================================================================
// collision.h
// Broadphase / Narrowphase / Exact Voronoi-cell collision detection.
//
// Pipeline per step:
//   1. Broadphase  — world-space AABB pairs that overlap.
//   2. Narrowphase — dual BVH traversal: boxes of each foam's BVH (local
//                    space) are transformed into the other foam's local space
//                    to cull non-overlapping leaf pairs.
//   3. Exact       — GJK + EPA on the world-space Voronoi cell vertices of
//                    each candidate leaf pair.  Produces a FoamCollision for
//                    every penetrating pair.
// =============================================================================

#include <vector>
#include <unordered_map>

#include <entt/entity/registry.hpp>
#include <glm/glm.hpp>

#include "dynamic_foam/Sim2D/bvh.cuh"
#include "dynamic_foam/Sim2D/adjacency.cuh"

namespace DynamicFoam::Sim2D {

// -----------------------------------------------------------------------------
// FoamCollision – one per penetrating Voronoi-cell pair
// -----------------------------------------------------------------------------

/**
 * Describes a detected collision between two individual Voronoi cells, one
 * belonging to foam A and one to foam B.
 *
 * Conventions
 * -----------
 *  - foamIdA / foamIdB       : keys into the foamAABBs / foamBVHs / foamTransforms maps.
 *  - particleA / particleB   : EnTT entities in the shared particleRegistry.
 *  - contactPoint            : world-space point on the interface surface where the
 *                              two Voronoi cells first touch/overlap.
 *  - normal                  : world-space unit vector pointing from B's cell toward
 *                              A's cell (i.e. the direction A must move to separate).
 *  - penetrationDepth        : signed overlap distance along `normal`.  Positive means
 *                              the cells are actually penetrating.
 */
struct FoamCollision {
    int          foamIdA;
    int          foamIdB;
    entt::entity particleA;
    entt::entity particleB;
    glm::vec3    contactPoint;    ///< world-space
    glm::vec3    normal;          ///< world-space, B → A
    float        penetrationDepth;
};

// -----------------------------------------------------------------------------
// detectCollisions – main entry point
// -----------------------------------------------------------------------------

/**
 * @brief Detect all Voronoi-cell collisions across every pair of foams.
 *
 * @param foamAABBs          World-space AABBs for each foam (keyed by foam int id).
 * @param foamBVHs           Local-space BVHs for each foam (keyed by foam int id).
 *                           Leaf prim_idx values index into the ordered particle
 *                           sequence returned by foamAdjacencyLists[id].getOrderedNodeIds().
 * @param foamTransforms     World transforms (T * R, glm::mat4) for each foam.
 * @param foamAdjacencyLists Adjacency lists whose getOrderedNodeIds() yields the
 *                           particle entities in the same order the BVH was built.
 * @param particleRegistry   Read-only registry used to fetch ParticleVertices per particle.
 * @return                   All penetrating Voronoi-cell pairs across all foam pairs.
 */
std::vector<FoamCollision> detectCollisions(
    const std::unordered_map<int, AABB>&                         foamAABBs,
    const std::unordered_map<int, BVH>&                          foamBVHs,
    const std::unordered_map<int, glm::mat4>&                    foamTransforms,
    const std::unordered_map<int, AdjacencyList<entt::entity>>&  foamAdjacencyLists,
    const entt::registry&                                        particleRegistry
);

} // namespace DynamicFoam::Sim2D
