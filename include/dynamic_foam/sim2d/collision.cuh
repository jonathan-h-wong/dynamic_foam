// =============================================================================
// collision.cuh
// GPU-accelerated broadphase + narrowphase, CPU GJK+EPA exact phase.
// Intended as a full replacement for collision.cu / collision.h.
//
// Pipeline
// --------
//   1. AABB transform   — k_col_transform_foam_aabbs  (GPU)
//                         One thread per foam: transforms the local-space foam
//                         AABB stored in the slab into world space.
//
//   2. Broadphase       — k_col_broadphase_foam_pairs  (GPU)
//                         One thread per (i < j) foam pair: emits a FoamPair
//                         for every pair whose world AABBs overlap.
//
//   3. Narrowphase      — k_col_narrowphase_particle_bvh  (GPU)
//                         One thread per broadphase-surviving pair: dual BVH
//                         traversal with B's nodes transformed into A-local
//                         space.  Emits CollisionCandidates for leaf-leaf hits.
//
//   4. Exact            — CPU GJK + EPA  (CPU)
//                         CollisionCandidates are downloaded (O(100) pairs, so
//                         negligible transfer).  GJK confirms penetration;
//                         EPA extracts the contact normal, depth, and point.
//                         Returns one FoamCollision per penetrating pair.
//
// Public surface
// --------------
//   detectCandidates()   — GPU phases 1–3 only; for callers that drive phase 4
//                          themselves.
//   detectCollisions()   — full pipeline (phases 1–4); drop-in replacement for
//                          the function of the same name in collision.h.
// =============================================================================

#pragma once

#define GLM_FORCE_CUDA
#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include <unordered_map>
#include <vector>

#include <entt/entity/registry.hpp>

#include "dynamic_foam/Sim2D/bvh.cuh"
#include "dynamic_foam/Sim2D/components.h"
#include "dynamic_foam/Sim2D/cuda_utils.cuh"
#include "dynamic_foam/Sim2D/gpu_slab.cuh"

namespace DynamicFoam::Sim2D {

// -----------------------------------------------------------------------------
// FoamCollision — one per penetrating Voronoi-cell pair (phase 4 output).
//
// Conventions
// -----------
//  foamIdA / foamIdB       : integer foam IDs matching the keys passed to
//                            detectCollisions / detectCandidates.
//                            When detectCollisions is called with a non-empty
//                            primaryFoamIds, foamIdA is always a primary foam.
//                            When primaryFoamIds is empty (all-pairs mode) the
//                            ordering follows the foamIds index (i < j).
//  particleA / particleB   : EnTT entities in the shared particleRegistry.
//  contactPoint            : world-space midpoint of the two deepest support
//                            points along the penetration normal.
//  normal                  : world-space unit vector pointing from B's cell
//                            toward A's cell (direction A must move to separate).
//  penetrationDepth        : positive overlap distance along normal.
// -----------------------------------------------------------------------------
struct FoamCollision {
    int          foamIdA;
    int          foamIdB;
    entt::entity particleA;
    entt::entity particleB;
    glm::vec3    contactPoint;
    glm::vec3    normal;          ///< world-space, B → A
    float        penetrationDepth;
};

// -----------------------------------------------------------------------------
// FoamPair — broadphase output: a pair of foam IDs whose world AABBs overlap.
// -----------------------------------------------------------------------------
struct FoamPair {
    int foam_id_a;
    int foam_id_b;
};

// -----------------------------------------------------------------------------
// CollisionCandidate — narrowphase + entity-resolve output.
// Written by k_col_narrowphase_particle_bvh (prim_idx fields), then updated
// in-place by k_col_resolve_entity_ids (entity_id fields) before D2H transfer,
// so the single download carries everything phase 4 needs.
//
// prim_idx_a / prim_idx_b   Morton-sorted BVH leaf positions.
// entity_id_a / entity_id_b EnTT entity handles, resolved on-device from
//                           d_active_ids[active_offset[fid] + prim_idx].
// -----------------------------------------------------------------------------
struct CollisionCandidate {
    int      foam_id_a;
    int      foam_id_b;
    int      prim_idx_a;    ///< BVH leaf prim_idx within foam A.
    int      prim_idx_b;    ///< BVH leaf prim_idx within foam B.
    uint32_t entity_id_a;   ///< EnTT entity for particle A, resolved on GPU.
    uint32_t entity_id_b;   ///< EnTT entity for particle B, resolved on GPU.
};

// =============================================================================
// Kernel declarations
// =============================================================================

// Phase 1 — transform local-space foam AABBs to world space.
// One thread per foam.  Reads slab->d_foam_aabbs[fid] (local space) and the
// corresponding transform, writes the world-space result into world_aabbs_out.
__global__ void k_col_transform_foam_aabbs(
    const AABB*      local_aabbs,        ///< slab->d_foam_aabbs (local space)
    const glm::mat4* transforms,         ///< per-foam world transforms, indexed by foam_id
    AABB*            world_aabbs_out,    ///< device buffer, size >= num_foams
    int              num_foams
);

// Phase 2 — broadphase: one thread per ordered foam pair (i < j).
// Emits a FoamPair for every pair whose world-space AABBs overlap.
// Launch with a 1-D grid of num_foams * num_foams threads; threads where
// thread_id >= num_pairs or i >= j are discarded.
__global__ void k_col_broadphase_foam_pairs(
    const AABB* world_aabbs,             ///< world-space AABBs, indexed by foam_id
    const int*  foam_ids,                ///< active foam IDs, length num_foams
    FoamPair*   pairs_out,               ///< output broadphase pairs
    int*        pair_counter,            ///< atomic counter
    int         num_foams
);

// Phase 3a — narrowphase: one thread per (broadphase pair × particle in foam A).
// 2-D launch: blockIdx.x = pair index, blockIdx.y*blockDim.x+threadIdx.x = particle
// index within foam A's Morton-sorted active set.
//
// Each thread transforms its particle A's local-space AABB into foam B-local
// space and traverses foam B's BVH, emitting a CollisionCandidate for every
// leaf AABB that overlaps.  Compared to the dual-BVH single-thread-per-pair
// formulation, this maximises GPU occupancy and lets all threads on the same
// foam-pair block share foam B's upper BVH levels from L1 cache.
//
// particle_aabbs           slab->d_particle_aabbs  (local-space, Morton order)
// foam_particle_start      slab->d_foam_particle_start (particle_offset per foam)
// foam_particle_counts     active_count per foam, indexed by foam_id
// primary_particle_ids     optional sorted array of EnTT entity IDs to filter on;
//                          nullptr / num_primary=0 disables the filter (all particles).
//                          Resolved via d_active_ids at thread startup; linear scan
//                          over at most num_primary entries before any BVH work.
// d_active_ids             slab->d_active_ids (Morton-sorted entity IDs per foam)
// d_foam_active_start      slab->d_foam_active_start
__global__ void k_col_narrowphase_particle_bvh(
    const FoamPair*         broadphase_pairs,      ///< surviving broadphase pairs
    int                     num_broadphase_pairs,
    const BVHNode*          bvh_nodes,             ///< slab->d_bvh_nodes
    const int*              bvh_offsets,           ///< slab->d_foam_bvh_start
    const AABB*             particle_aabbs,        ///< slab->d_particle_aabbs
    const int*              foam_particle_start,   ///< slab->d_foam_particle_start
    const int*              foam_particle_counts,  ///< active_count per foam
    const glm::mat4*        transforms,
    const glm::mat4*        inv_transforms,
    const uint32_t*         primary_particle_ids,  ///< entity IDs to filter; nullptr = all
    int                     num_primary_particles, ///< 0 = disable filter
    const uint32_t*         d_active_ids,          ///< slab->d_active_ids
    const int*              d_foam_active_start,   ///< slab->d_foam_active_start
    CollisionCandidate*     candidates_out,
    int*                    candidate_counter,
    int                     max_candidates
);

// Phase 3b — entity resolve: one thread per narrowphase candidate.
// Reads d_active_ids[d_foam_active_start[fid] + prim_idx] for both sides of
// each candidate and writes the results into candidate.entity_id_a/b in place.
// Must run after k_col_narrowphase_particle_bvh, before the D2H candidates copy.
__global__ void k_col_resolve_entity_ids(
    CollisionCandidate* candidates,         ///< in/out: entity fields written
    int                 num_candidates,
    const uint32_t*     d_active_ids,       ///< slab->d_active_ids
    const int*          d_foam_active_start ///< slab->d_foam_active_start
);

// =============================================================================
// Host-side entry points
// =============================================================================

/**
 * @brief GPU phases 1–3 only: broadphase + narrowphase.
 *
 * Phase 1 runs on the GPU (AABB transform).  The resulting world-space AABBs
 * are downloaded to the CPU, which generates the candidate pair list filtered
 * to pairs where at least one side appears in @p primaryFoamIds, then tests
 * AABB overlap.  The surviving pairs are uploaded and the narrowphase runs on
 * the GPU.  The caller is responsible for running GJK+EPA (phase 4) on each
 * returned candidate.
 *
 * @param gpuSlab         Slab allocator that owns all GPU BVH / particle data.
 * @param foamTransforms  Host-side world transforms keyed by foam_id.
 * @param foamIds         Full set of active foam IDs.
 * @param primaryFoamIds  Restrict to pairs where at least one side is in this
 *                        set.  Pass an empty vector (the default) to test all
 *                        i < j pairs — identical to the original behaviour.
 * @param primaryParticleIds  Restrict the A-side of each candidate to this set
 *                            of EnTT entity IDs.  Pass an empty vector (the
 *                            default) to test all particles in foam A.
 *                            Intended for small sets (≤50 IDs); the kernel
 *                            performs a linear scan at thread startup.
 * @param maxCandidates   Device buffer capacity for narrowphase output.
 *                        Defaults to 64 K; increase for dense contact scenes.
 * @return CollisionCandidates that survived the AABB broadphase and narrowphase.
 */
std::vector<CollisionCandidate> detectCandidates(
    const GpuSlabAllocator&                   gpuSlab,
    const std::unordered_map<int, glm::mat4>& foamTransforms,
    const std::vector<int>&                   foamIds,
    const std::vector<int>&                   primaryFoamIds     = {},
    const std::vector<uint32_t>&              primaryParticleIds = {},
    int                                       maxCandidates      = 65536
);

/**
 * @brief Full pipeline (phases 1–4): GPU broadphase + narrowphase, CPU GJK+EPA.
 *
 * prim_idx values from the narrowphase are resolved to EnTT entity handles by
 * reading d_active_ids directly from the slab (Morton-sorted, in sync with the
 * BVH via bulkMortonSort).
 *
 * @param gpuSlab          Slab allocator that owns all GPU BVH / particle data.
 * @param foamTransforms   Host-side world transforms keyed by foam_id.
 * @param particleRegistry Read-only registry used to fetch ParticleVertices.
 * @param foamIds          Full set of active foam IDs.
 * @param primaryFoamIds   Restrict to pairs where at least one side is in this
 *                         set.  Pass an empty vector (the default) to test all
 *                         i < j pairs — identical to the original behaviour.
 * @param primaryParticleIds  Restrict the A-side of each candidate to this set
 *                            of EnTT entity IDs.  Pass an empty vector (the
 *                            default) to test all particles in foam A.
 *                            Intended for small sets (≤50 IDs); the kernel
 *                            performs a linear scan at thread startup.
 * @param maxCandidates    Narrowphase GPU buffer capacity (default 64 K).
 * @return All penetrating Voronoi-cell pairs that involve at least one primary
 *         foam (or all pairs if @p primaryFoamIds is empty).
 */
std::vector<FoamCollision> detectCollisions(
    const GpuSlabAllocator&                        gpuSlab,
    const std::unordered_map<int, glm::mat4>&      foamTransforms,
    const entt::registry&                          particleRegistry,
    const std::vector<int>&                        foamIds,
    const std::vector<int>&                        primaryFoamIds     = {},
    const std::vector<uint32_t>&                   primaryParticleIds = {},
    int                                            maxCandidates      = 65536
);

} // namespace DynamicFoam::Sim2D
