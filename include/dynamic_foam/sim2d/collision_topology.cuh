// =============================================================================
// collision_topology.cuh
// Point-in-cell containment detection for the topology subsystem.
//
// Purpose
// -------
// Given the output of detectCollisions() (cell-cell pairs), this stage answers:
//   "Which neighbor particles of a colliding cell lie inside the opposing cell?"
//
// For each FoamCollision (cellA, cellB):
//   - Gather the Voronoi neighbors of cellA; test each against cellB.
//   - Gather the Voronoi neighbors of cellB; test each against cellA.
//
// Containment test (Voronoi half-plane)
// --------------------------------------
// A point P is inside Voronoi cell C (with neighbor centers N_i) iff P is
// closer to C than to every neighbor:
//
//   sdf_i = dot(P - C, N_i - C) - 0.5 * |N_i - C|^2  <= 0  for all i
//
// The neighbor whose sdf_i is least negative gives the contact normal and depth.
// No vertex buffers, GJK, or EPA required.
//
// Implementation paths
// --------------------
//   CPU  -- used when collisions.size() < kPointCellGpuThreshold.
//           O(pairs * neighbors^2), all on the calling thread.
//
//   GPU  -- used for dense contact (collisions.size() >= kPointCellGpuThreshold).
//           CPU flattens the test list; one GPU thread per (point, cell) pair.
//           A single D2H transfer brings back only the contacts.
//
// Callers should not rely on which path runs; behavior is identical.
// =============================================================================

#pragma once

#define GLM_FORCE_CUDA
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include <unordered_map>
#include <vector>

#include <entt/entity/registry.hpp>

#include "dynamic_foam/Sim2D/adjacency.cuh"
#include "dynamic_foam/Sim2D/collision.cuh"
#include "dynamic_foam/Sim2D/components.h"
#include "dynamic_foam/Sim2D/cuda_utils.cuh"

namespace DynamicFoam::Sim2D {

// Collision count above which the GPU path is preferred over the CPU path.
inline constexpr int kPointCellGpuThreshold = 300;

// -----------------------------------------------------------------------------
// PointCellContact -- one per neighbor particle found inside an opposing cell.
// -----------------------------------------------------------------------------
struct PointCellContact {
    entt::entity pointEntity;   ///< the neighbor particle that penetrated
    entt::entity cellEntity;    ///< the cell it is inside
    int          foamIdPoint;
    int          foamIdCell;
    glm::vec3    normal;        ///< world-space, direction to eject the point
    float        depth;         ///< penetration depth (positive = inside cell)
};

// -----------------------------------------------------------------------------
// Internal GPU test descriptor -- one per (point, cell) pair.
// Built on the CPU during flattening, then uploaded before the kernel launch.
//
// The test reads cellCenter and the neighbor centers stored in a flat buffer
// at [neighborOffset, neighborOffset + neighborCount).
// -----------------------------------------------------------------------------
struct PointCellTest {
    glm::vec3  pointWorld;      ///< world-space position of the candidate point
    glm::vec3  cellCenter;      ///< world-space center of the cell being tested
    uint32_t   pointEntity;     ///< EnTT entity (cast to uint32_t for GPU)
    uint32_t   cellEntity;      ///< EnTT entity of the cell
    int        foamIdPoint;
    int        foamIdCell;
    int        neighborOffset;  ///< start index into flat neighbor-center buffer
    int        neighborCount;   ///< number of neighbors for this cell
};

// =============================================================================
// GPU kernel -- one thread per PointCellTest.
// Runs Voronoi half-plane containment, writes PointCellContact on hit.
// =============================================================================
__global__ void k_pcc_containment(
    const PointCellTest*  tests,
    int                   num_tests,
    const glm::vec3*      neighbor_centers, ///< flat world-space neighbor buffer
    PointCellContact*     contacts_out,
    int*                  contact_counter,
    int                   max_contacts
);

// =============================================================================
// Host entry point
// =============================================================================

/**
 * @brief Detect which neighbor particles of each colliding cell pair lie inside
 * the opposing cell, using Voronoi half-plane tests.
 *
 * @param collisions         Cell-cell pairs from detectCollisions().
 * @param foamAdjacencyLists Voronoi neighbor graph per foam (keyed by foam_id).
 * @param foamTransforms     World transforms per foam (keyed by foam_id).
 * @param particleRegistry   Read-only registry (ParticleLocalPosition per particle).
 * @param maxContacts        GPU output buffer capacity. Ignored on the CPU path.
 * @param forceGpu           Override automatic path selection.
 */
std::vector<PointCellContact> detectPointCellContainment(
    const std::vector<FoamCollision>&              collisions,
    const std::unordered_map<int, AdjacencyList>&  foamAdjacencyLists,
    const std::unordered_map<int, glm::mat4>&      foamTransforms,
    const entt::registry&                          particleRegistry,
    int                                            maxContacts = 65536,
    bool                                           forceGpu   = false
);

} // namespace DynamicFoam::Sim2D
