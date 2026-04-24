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

        // stencil particleA -> foamIdB -> intersected mutable particle ids
        std::unordered_map<entt::entity, std::unordered_map<int, std::vector<entt::entity>>> stencilContacts;
        // stencil particle -> the foam it belongs to (needed for neighbor lookup below)
        std::unordered_map<entt::entity, int> stencilFoamId;
        for (const auto& col : collisions) {
            if (particleRegistry.all_of<Stencil>(col.particleA) &&
                particleRegistry.all_of<Mutable>(col.particleB))
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

        // For each (stencil particle, foamB) pair, compute the midpoints between
        // the stencil particle and each of its Voronoi neighbors (world space),
        // convert them into foamB local space, then test each midpoint against
        // the candidate mutable cells inside foamB.
        //
        // Result: stencil -> foamId -> [(foamB-local midpoint, collided mutable entity)]
        struct MidpointHit { glm::vec3 localMidpoint; entt::entity mutParticle; };
        std::unordered_map<entt::entity,
            std::unordered_map<int, std::vector<MidpointHit>>> stencilBoundaryContacts;

        for (const auto& [stencil, foamBMap] : stencilContacts) {
            int foamA = stencilFoamId.at(stencil);
            const glm::mat4& txA = foamTransforms.at(foamA);

            // Stencil center in world space.
            glm::vec3 stencilWorld =
                glm::vec3(txA * glm::vec4(
                    particleRegistry.get<ParticleLocalPosition>(stencil).value, 1.0f));

            // Collect world-space midpoints along each stencil axis (one per neighbor).
            std::vector<glm::vec3> midpointsWorld;
            const AdjacencyList& adjA = foamAdjacencyLists.at(foamA);
            adjA.forEachNeighbor(static_cast<uint32_t>(stencil), [&](uint32_t nbId) {
                entt::entity nb = static_cast<entt::entity>(nbId);
                if (!particleRegistry.all_of<ParticleLocalPosition>(nb)) return;
                glm::vec3 nbWorld = glm::vec3(txA * glm::vec4(
                    particleRegistry.get<ParticleLocalPosition>(nb).value, 1.0f));
                midpointsWorld.push_back((stencilWorld + nbWorld) * 0.5f);
            });

            for (const auto& [foamB, candidates] : foamBMap) {
                const glm::mat4  invTxB = glm::inverse(foamTransforms.at(foamB));
                const AdjacencyList& adjB = foamAdjacencyLists.at(foamB);

                for (const glm::vec3& midWorld : midpointsWorld) {
                    // Convert midpoint into foamB local space.
                    glm::vec3 midLocal = glm::vec3(invTxB * glm::vec4(midWorld, 1.0f));

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

                        if (voronoiContains(midLocal, cellCenter, cellNeighborCenters))
                            stencilBoundaryContacts[stencil][foamB].push_back({midLocal, mutParticle});
                    }
                }
            }
        }



        return results;
    }
}