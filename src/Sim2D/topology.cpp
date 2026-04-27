#include "dynamic_foam/Sim2D/topology.h"

namespace DynamicFoam::Sim2D {
    std::vector<FoamTopologyUpdate> Topology::update(
        const GpuSlabAllocator&                              gpuSlab,
        std::unordered_map<int, AdjacencyList>&              foamAdjacencyLists,
        const std::unordered_map<int, glm::mat4>&            foamTransforms,
        const entt::registry&                                foamRegistry,
        entt::registry&                                      particleRegistry
    ) {
        std::vector<FoamTopologyUpdate> results;

        // Collect active foam IDs from the adjacency list map.
        std::vector<int> foamIds;
        foamIds.reserve(foamAdjacencyLists.size());
        for (const auto& [id, _] : foamAdjacencyLists)
            foamIds.push_back(id);

        // Primary foam IDs: foams with the Controller component.
        std::vector<int> primaryFoamIds;
        for (auto entity : foamRegistry.view<Controller>())
            primaryFoamIds.push_back(static_cast<int>(entity));

        // Primary particle IDs: particles with the Stencil component.
        std::vector<uint32_t> primaryParticleIds;
        for (auto entity : particleRegistry.view<Stencil>())
            primaryParticleIds.push_back(static_cast<uint32_t>(entity));

        auto collisions = detectCollisions(gpuSlab, foamTransforms, particleRegistry, foamIds,
                                           primaryFoamIds, primaryParticleIds);

        // 1) Collect stencil cell to cell collisions
        // stencil particleA -> foamIdB -> intersected mutable particle ids
        std::unordered_map<entt::entity, std::unordered_map<int, std::vector<entt::entity>>> stencilContacts;
        // stencil particle -> the foam it belongs to (needed for neighbor lookup below)
        std::unordered_map<entt::entity, int> stencilFoamId;
        for (const auto& col : collisions) {
            if (particleRegistry.all_of<Stencil>(col.particleA) &&
                particleRegistry.all_of<Mutable, Surface>(col.particleB))
            {
                stencilContacts[col.particleA][col.foamIdB].push_back(col.particleB);
                stencilFoamId.emplace(col.particleA, col.foamIdA);
            }
        }

        // Voronoi half-plane containment test (CPU, local-space).
        // Returns true if point P is inside the Voronoi cell centered at C.
        auto voronoiContains = [](
            const glm::vec3& P,
            const glm::vec3& C,
            const std::vector<glm::vec3>& neighborCenters) -> bool
        {
            for (const auto& N : neighborCenters) {
                glm::vec3 edge = N - C;
                float     len  = glm::length(edge);
                if (len < 1e-6f) continue;
                glm::vec3 n       = edge / len;
                float     normSdf = glm::dot(P - C, n) - 0.5f * len;
                if (normSdf > 0.0f) return false;
            }
            return !neighborCenters.empty();
        };

        // 2) Collect stencil boundary point to cell collisions
        // For each (stencil particle, foamB) pair, sample two points per Voronoi axis
        // at ±kStencilBoundaryOffset from the edge midpoint (world space), convert them
        // into foamB local space, then test each sample against the candidate mutable
        // cells inside foamB.
        //
        // Result: stencil -> foamId -> [(foamB-local boundary sample, collided mutable entity)]
        constexpr float kStencilBoundaryOffset = 0.05f; // perturbation around t=0.5
        struct BoundaryHit { glm::vec3 localPoint; entt::entity mutParticle; };
        std::unordered_map<entt::entity,
            std::unordered_map<int, std::vector<BoundaryHit>>> stencilBoundaryContacts;

        // effectiveStencilAABB: stencil -> foamB -> AABB over ALL foamB-local boundary
        // samples (regardless of whether they hit a cell). Used in step 3 to test
        // whether the intersection region spans the full stencil boundary on any axis.
        std::unordered_map<entt::entity, std::unordered_map<int, AABB>> effectiveStencilAABB;

        for (const auto& [stencil, foamBMap] : stencilContacts) {
            int foamA = stencilFoamId.at(stencil);
            const glm::mat4& txA = foamTransforms.at(foamA);

            // Stencil center in world space.
            glm::vec3 stencilWorld =
                glm::vec3(txA * glm::vec4(
                    particleRegistry.get<ParticleLocalPosition>(stencil).value, 1.0f));

            // Collect two world-space boundary samples per stencil axis (±offset from midpoint).
            std::vector<glm::vec3> boundaryPointsWorld;
            const AdjacencyList& adjA = foamAdjacencyLists.at(foamA);
            adjA.forEachNeighbor(static_cast<uint32_t>(stencil), [&](uint32_t nbId) {
                entt::entity nb = static_cast<entt::entity>(nbId);
                if (!particleRegistry.all_of<ParticleLocalPosition>(nb)) return;
                glm::vec3 nbWorld = glm::vec3(txA * glm::vec4(
                    particleRegistry.get<ParticleLocalPosition>(nb).value, 1.0f));
                glm::vec3 axis = nbWorld - stencilWorld;
                // Sample at t = 0.5 ± kStencilBoundaryOffset along the stencil->neighbor segment.
                boundaryPointsWorld.push_back(stencilWorld + (0.5f + kStencilBoundaryOffset) * axis);
                boundaryPointsWorld.push_back(stencilWorld + (0.5f - kStencilBoundaryOffset) * axis);
            });

            for (const auto& [foamB, candidates] : foamBMap) {
                const glm::mat4  invTxB = glm::inverse(foamTransforms.at(foamB));
                const AdjacencyList& adjB = foamAdjacencyLists.at(foamB);

                for (const glm::vec3& sampleWorld : boundaryPointsWorld) {
                    // Convert boundary sample into foamB local space.
                    glm::vec3 sampleLocal = glm::vec3(invTxB * glm::vec4(sampleWorld, 1.0f));

                    // Expand the effective stencil AABB over every sample (hit or not).
                    AABB& effAABB = effectiveStencilAABB[stencil][foamB];
                    effAABB.min_pt = glm::min(effAABB.min_pt, sampleLocal);
                    effAABB.max_pt = glm::max(effAABB.max_pt, sampleLocal);

                    for (entt::entity mutParticle : candidates) {
                        if (!particleRegistry.all_of<ParticleLocalPosition>(mutParticle))
                            continue;

                        // Cell center and neighbor centers are already in foamB local space.
                        glm::vec3 cellCenter =
                            particleRegistry.get<ParticleLocalPosition>(mutParticle).value;

                        std::vector<glm::vec3> cellNeighborCenters;
                        adjB.forEachNeighbor(static_cast<uint32_t>(mutParticle), [&](uint32_t nbId) {
                            entt::entity nb = static_cast<entt::entity>(nbId);
                            if (!particleRegistry.all_of<ParticleLocalPosition>(nb)) return;
                            cellNeighborCenters.push_back(
                                particleRegistry.get<ParticleLocalPosition>(nb).value);
                        });

                        if (voronoiContains(sampleLocal, cellCenter, cellNeighborCenters))
                            stencilBoundaryContacts[stencil][foamB].push_back({sampleLocal, mutParticle});
                    }
                }
            }
        }

        // 3) Filter stencilBoundaryContacts to entries where the intersection AABB
        // matches the effectiveStencilAABB on at least one axis, meaning the stencil
        // boundary fully spans that axis within foamB — indicating a genuine crossing.
        //
        // Result: stencil -> foamB -> boundary hits (same structure, subset of above)
        std::unordered_map<entt::entity,
            std::unordered_map<int, std::vector<BoundaryHit>>> validStencilWork;

        for (const auto& [stencil, foamBHits] : stencilBoundaryContacts) {
            for (const auto& [foamB, hits] : foamBHits) {
                // Build AABB over the hit points only.
                AABB intersectionAABB{};
                for (const auto& hit : hits) {
                    intersectionAABB.min_pt = glm::min(intersectionAABB.min_pt, hit.localPoint);
                    intersectionAABB.max_pt = glm::max(intersectionAABB.max_pt, hit.localPoint);
                }

                const AABB& effAABB = effectiveStencilAABB.at(stencil).at(foamB);

                // Keep this (stencil, foamB) entry if the intersection AABB matches
                // the effective stencil AABB on at least one axis.
                bool spans =
                    (intersectionAABB.min_pt.x == effAABB.min_pt.x && intersectionAABB.max_pt.x == effAABB.max_pt.x) ||
                    (intersectionAABB.min_pt.y == effAABB.min_pt.y && intersectionAABB.max_pt.y == effAABB.max_pt.y) ||
                    (intersectionAABB.min_pt.z == effAABB.min_pt.z && intersectionAABB.max_pt.z == effAABB.max_pt.z);

                if (spans)
                    validStencilWork[stencil][foamB] = hits;
            }
        }

        return results;
    }
}