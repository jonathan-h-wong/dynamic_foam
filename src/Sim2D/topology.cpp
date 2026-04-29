#include "dynamic_foam/Sim2D/topology.h"
#include <optional>
#include <queue>
#include <unordered_set>

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
        std::unordered_map<entt::entity, std::unordered_map<int, std::vector<entt::entity>>> stencilCellContacts;
        // stencil particle -> the foam it belongs to (needed for neighbor lookup below)
        std::unordered_map<entt::entity, int> stencilFoamId;
        for (const auto& col : collisions) {
            if (particleRegistry.all_of<Stencil>(col.particleA) &&
                particleRegistry.all_of<Mutable, Surface>(col.particleB))
            {
                stencilCellContacts[col.particleA][col.foamIdB].push_back(col.particleB);
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

        for (const auto& [stencil, foamBMap] : stencilCellContacts) {
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
        
        // 4) Point creation
        std::unordered_map<entt::entity, std::unordered_set<entt::entity>> stencilDirtyCells;
        std::unordered_map<entt::entity, std::unordered_set<entt::entity>> cellSamples;
        for (const auto& [stencil, foamBHits] : validStencilWork) {
            for (const auto& [foamB, hits] : foamBHits) {

                // First, produce overlap set to filter downstream graph traversal.
                // overlap_set: contact cells whose Voronoi AABB is entirely contained
                // within the effectiveStencilAABB for this (stencil, foamB) pair.
                std::unordered_set<entt::entity> overlap_set;

                // Add covered particles
                const AABB& effAABB = effectiveStencilAABB.at(stencil).at(foamB);
                if (stencilCellContacts.count(stencil) &&
                    stencilCellContacts.at(stencil).count(foamB))
                {
                    for (entt::entity cellEnt : stencilCellContacts.at(stencil).at(foamB)) {
                        if (!particleRegistry.all_of<ParticleVertices>(cellEnt)) continue;

                        const auto& verts =
                            particleRegistry.get<ParticleVertices>(cellEnt).vertices;
                        if (verts.empty()) continue;

                        // Build AABB directly from the Voronoi cell's local-space vertices.
                        AABB cellAABB{};
                        cellAABB.min_pt = verts[0];
                        cellAABB.max_pt = verts[0];
                        for (const auto& v : verts) {
                            cellAABB.min_pt = glm::min(cellAABB.min_pt, v);
                            cellAABB.max_pt = glm::max(cellAABB.max_pt, v);
                        }

                        // Deposit cell if its AABB is entirely covered by effAABB.
                        bool covered =
                            cellAABB.min_pt.x >= effAABB.min_pt.x &&
                            cellAABB.max_pt.x <= effAABB.max_pt.x &&
                            cellAABB.min_pt.y >= effAABB.min_pt.y &&
                            cellAABB.max_pt.y <= effAABB.max_pt.y &&
                            cellAABB.min_pt.z >= effAABB.min_pt.z &&
                            cellAABB.max_pt.z <= effAABB.max_pt.z;

                        if (covered)
                            overlap_set.insert(cellEnt);
                    }
                }

                // Add boundary particles
                for (const auto& hit : hits)
                    overlap_set.insert(hit.mutParticle);

                // Utility: create a new interior particle inheriting color, opacity, Surface,
                // and Mutable from the intersected cell; mass and vertices are left as defaults
                // to be populated after triangulation.
                auto addInteriorParticle = [&](entt::entity stencilId,
                                               entt::entity cellId,
                                               glm::vec3    localPos) -> entt::entity
                {
                    entt::entity p = particleRegistry.create();
                    particleRegistry.emplace<ParticleColor>(p, particleRegistry.get<ParticleColor>(cellId));
                    particleRegistry.emplace<ParticleOpacity>(p, particleRegistry.get<ParticleOpacity>(cellId));
                    particleRegistry.emplace<Surface>(p);
                    particleRegistry.emplace<Mutable>(p);
                    particleRegistry.emplace<ParticleMass>(p);
                    particleRegistry.emplace<ParticleVertices>(p);
                    particleRegistry.emplace<ParticleLocalPosition>(p, ParticleLocalPosition{ localPos });
                    stencilDirtyCells[stencilId].insert(cellId);
                    cellSamples[cellId].insert(p);
                    return p;
                };

                // Utility: test whether point P is near the Voronoi bisector between
                // cell center C and a neighbor at N (sdf(P) > sdf(C)).
                // Returns the reflected point across that bisector, or std::nullopt.
                auto faceReflect = [](const glm::vec3& P,
                                      const glm::vec3& C,
                                      const glm::vec3& N) -> std::optional<glm::vec3>
                {
                    glm::vec3 edge = N - C;
                    float     len  = glm::length(edge);
                    if (len < 1e-6f) return std::nullopt;
                    glm::vec3 n   = edge / len;
                    float sdf     = glm::dot(P - C, n) - 0.5f * len;
                    if (sdf <= -0.5f * len) return std::nullopt; // not closer than center
                    return P - 2.0f * sdf * n;
                };

                // Utility: create a zero-opacity air particle reflected across a Voronoi
                // bisector plane; mass and vertices are left as defaults.
                auto addAirParticle = [&](entt::entity cellId,
                                          glm::vec3    localPos) -> entt::entity
                {
                    entt::entity p = particleRegistry.create();
                    particleRegistry.emplace<ParticleColor>(p);
                    particleRegistry.emplace<ParticleOpacity>(p, ParticleOpacity{ 0.0f });
                    particleRegistry.emplace<ParticleMass>(p, ParticleMass{ 0.0f });
                    particleRegistry.emplace<ParticleVertices>(p);
                    particleRegistry.emplace<Mutable>(p);
                    particleRegistry.emplace<ParticleLocalPosition>(p, ParticleLocalPosition{ localPos });
                    cellSamples[cellId].insert(p);
                    return p;
                };

                // Densification
                for (const auto& hit: hits) {
                    // Surface Cell Scaffolding
                    entt::entity newParticle = addInteriorParticle(stencil, hit.mutParticle, hit.localPoint);
                    const AdjacencyList& adjB = foamAdjacencyLists.at(foamB);
                    const glm::vec3 cellCenter =
                        particleRegistry.get<ParticleLocalPosition>(hit.mutParticle).value;

                    // Reflected air particles across air-facing bisectors of hit.mutParticle.
                    adjB.forEachNeighbor(static_cast<uint32_t>(hit.mutParticle), [&](uint32_t nbId) {
                        entt::entity nb = static_cast<entt::entity>(nbId);
                        if (!particleRegistry.all_of<ParticleLocalPosition, ParticleOpacity>(nb)) return;
                        if (particleRegistry.get<ParticleOpacity>(nb).value != 0.0f) return;

                        glm::vec3 nbCenter = particleRegistry.get<ParticleLocalPosition>(nb).value;
                        if (auto reflected = faceReflect(hit.localPoint, cellCenter, nbCenter))
                            addAirParticle(hit.mutParticle, *reflected);
                    });

                    // Surface Cell Neighbor Scaffolding
                    // For each surface neighbor of hit.mutParticle outside the overlap region,
                    // mirror hit.localPoint into that neighbor's cell to preserve the shared
                    // edge under triangulation.
                    adjB.forEachNeighbor(static_cast<uint32_t>(hit.mutParticle), [&](uint32_t nbrId) {
                        entt::entity nbr = static_cast<entt::entity>(nbrId);
                        if (!particleRegistry.all_of<Surface, ParticleLocalPosition>(nbr)) return;
                        if (overlap_set.count(nbr)) return;

                        glm::vec3 nbrCenter = particleRegistry.get<ParticleLocalPosition>(nbr).value;
                        auto reflectedIntoNbr = faceReflect(hit.localPoint, cellCenter, nbrCenter);
                        if (!reflectedIntoNbr) return;

                        addInteriorParticle(stencil, nbr, *reflectedIntoNbr);

                        // For each air neighbor of nbr, produce a reflected air ghost.
                        adjB.forEachNeighbor(nbrId, [&](uint32_t airNbrId) {
                            entt::entity airNbr = static_cast<entt::entity>(airNbrId);
                            if (!particleRegistry.all_of<Surface, ParticleLocalPosition, ParticleOpacity>(airNbr)) return;
                            if (particleRegistry.get<ParticleOpacity>(airNbr).value != 0.0f) return;

                            glm::vec3 airNbrCenter = particleRegistry.get<ParticleLocalPosition>(airNbr).value;
                            if (auto reflectedAir = faceReflect(*reflectedIntoNbr, nbrCenter, airNbrCenter))
                                addAirParticle(nbr, *reflectedAir);
                        });
                    });

                    // Internal Densification
                    // BFS from hit.mutParticle into non-surface, opaque interior cells.
                    // At each visited cell, deposit N uniformly distributed sample points.
                    constexpr int kDensifyDepth   = 2;
                    constexpr int kSamplesPerCell = 2;

                    struct BfsNode { entt::entity cell; int depth; };
                    std::unordered_set<entt::entity> visited;
                    std::queue<BfsNode> bfsQueue;

                    visited.insert(hit.mutParticle);
                    bfsQueue.push({ hit.mutParticle, 0 });

                    while (!bfsQueue.empty()) {
                        auto [current, depth] = bfsQueue.front();
                        bfsQueue.pop();
                        if (particleRegistry.all_of<ParticleLocalPosition, ParticleVertices>(current)) {
                            const glm::vec3& currentCenter =
                                particleRegistry.get<ParticleLocalPosition>(current).value;
                            const auto& verts =
                                particleRegistry.get<ParticleVertices>(current).vertices;

                            if (!verts.empty()) {
                                if constexpr (kSamplesPerCell == 1) {
                                    // Single sample: preserve the cell center for a clean
                                    // 1-to-1 replacement under triangulation.
                                    addInteriorParticle(stencil, current, currentCenter);
                                } else {
                                    // Multiple samples: stride evenly across the vertex list,
                                    // placing each point at the midpoint of center-to-vertex.
                                    int stride = static_cast<int>(verts.size()) / kSamplesPerCell;
                                    if (stride < 1) stride = 1;
                                    for (int s = 0; s < kSamplesPerCell; ++s) {
                                        int vi = (s * stride) % static_cast<int>(verts.size());
                                        addInteriorParticle(stencil, current,
                                            0.5f * (currentCenter + verts[vi]));
                                    }
                                }
                            }
                        }

                        if (depth < kDensifyDepth) {
                            adjB.forEachNeighbor(static_cast<uint32_t>(current), [&](uint32_t nbId) {
                                entt::entity nb = static_cast<entt::entity>(nbId);
                                if (visited.count(nb)) return;
                                if (particleRegistry.all_of<Surface>(nb)) return;
                                if (!particleRegistry.all_of<ParticleOpacity>(nb)) return;
                                if (particleRegistry.get<ParticleOpacity>(nb).value == 0.0f) return;
                                visited.insert(nb);
                                bfsQueue.push({ nb, depth + 1 });
                            });
                        }
                    }
                }
            }
        }

        // 5) Triangulation 

        // 6) CPU/GPU data structure updates

        return results;
    }
}