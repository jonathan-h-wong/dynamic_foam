#include "dynamic_foam/Sim2D/topology.h"
#include <future>
#include <queue>

namespace DynamicFoam::Sim2D {

// =============================================================================
// Static helpers
// =============================================================================

bool Topology::voronoiContains(
    const glm::vec3&              P,
    const glm::vec3&              C,
    const std::vector<glm::vec3>& neighborCenters)
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
}

std::optional<glm::vec3> Topology::faceReflect(
    const glm::vec3& P,
    const glm::vec3& C,
    const glm::vec3& N)
{
    glm::vec3 edge = N - C;
    float     len  = glm::length(edge);
    if (len < 1e-6f) return std::nullopt;
    glm::vec3 n   = edge / len;
    float     sdf = glm::dot(P - C, n) - 0.5f * len;
    // sdf(C) = -0.5*len; only proceed if P is closer to the face than C.
    if (sdf <= -0.5f * len) return std::nullopt;
    return P - 2.0f * sdf * n;
}

entt::entity Topology::addInteriorParticle(
    StencilFoamKey                                                                           key,
    entt::entity                                                                             cellId,
    glm::vec3                                                                                localPos,
    entt::registry&                                                                          registry,
    const AdjacencyList&                                                                     adjList,
    StencilFoamMap<std::unordered_set<entt::entity>>&                                        stencilROI,
    std::unordered_map<entt::entity, std::unordered_set<entt::entity>>&                     cellSamples)
{
    entt::entity p = registry.create();
    registry.emplace<ParticleColor>(p,         registry.get<ParticleColor>(cellId));
    registry.emplace<ParticleOpacity>(p,       registry.get<ParticleOpacity>(cellId));
    registry.emplace<Surface>(p);
    registry.emplace<Mutable>(p);
    registry.emplace<ParticleMass>(p);
    registry.emplace<ParticleVertices>(p);
    registry.emplace<ParticleLocalPosition>(p, ParticleLocalPosition{ localPos });
    stencilROI[key].insert(cellId);
    cellSamples[cellId].insert(p);

    // If cellId is a surface cell, add its air neighbors into cellSamples so that
    // triangulation preserves the existing surface boundary across the bisector.
    if (registry.all_of<Surface>(cellId)) {
        adjList.forEachNeighbor(static_cast<uint32_t>(cellId), [&](uint32_t nbId) {
            entt::entity nb = static_cast<entt::entity>(nbId);
            if (!registry.all_of<ParticleOpacity>(nb)) return;
            if (registry.get<ParticleOpacity>(nb).value != 0.0f) return;
            cellSamples[cellId].insert(nb);
        });
    }

    return p;
}

entt::entity Topology::addAirParticle(
    entt::entity                                                         cellId,
    glm::vec3                                                            localPos,
    entt::registry&                                                      registry,
    std::unordered_map<entt::entity, std::unordered_set<entt::entity>>& cellSamples)
{
    entt::entity p = registry.create();
    registry.emplace<ParticleColor>(p);
    registry.emplace<ParticleOpacity>(p,       ParticleOpacity{ 0.0f });
    registry.emplace<ParticleMass>(p,          ParticleMass{ 0.0f });
    registry.emplace<ParticleVertices>(p);
    registry.emplace<Mutable>(p);
    registry.emplace<ParticleLocalPosition>(p, ParticleLocalPosition{ localPos });
    cellSamples[cellId].insert(p);
    return p;
}

// =============================================================================
// Steps 1-3: build valid stencil work
// =============================================================================

StencilWorkResult Topology::buildStencilWork(
    const std::vector<FoamCollision>&             collisions,
    const std::unordered_map<int, AdjacencyList>& foamAdjacencyLists,
    const std::unordered_map<int, glm::mat4>&     foamTransforms,
    const entt::registry&                         particleRegistry)
{
    StencilWorkResult result;

    // 1) Collect stencil cell-to-cell collisions.
    // stencil int -> foamA id (needed for boundary sampling in step 2)
    std::unordered_map<int, int> stencilFoamId;
    // stencil int -> list of foamB ids encountered (for step 2 outer grouping)
    std::unordered_map<int, std::vector<int>> stencilToFoams;
    for (const auto& col : collisions) {
        if (particleRegistry.all_of<Stencil>(col.particleA) &&
            particleRegistry.all_of<Mutable, Surface>(col.particleB))
        {
            int stencilInt = static_cast<int>(col.particleA);
            StencilFoamKey key{ stencilInt, col.foamIdB };
            result.cellContacts[key].push_back(col.particleB);
            stencilFoamId.emplace(stencilInt, col.foamIdA);
            stencilToFoams[stencilInt].push_back(col.foamIdB);
        }
    }

    // 2) For each (stencil, foamB) pair, sample boundary points and Voronoi-test them
    //    against candidate cells, building stencilBoundaryContacts and effectiveAABB.
    constexpr float kStencilBoundaryOffset = 0.05f;

    StencilFoamMap<std::vector<BoundaryHit>> stencilBoundaryContacts;

    for (const auto& [stencilInt, foamList] : stencilToFoams) {
        entt::entity stencil = static_cast<entt::entity>(stencilInt);
        int foamA = stencilFoamId.at(stencilInt);
        const glm::mat4& txA = foamTransforms.at(foamA);

        glm::vec3 stencilWorld =
            glm::vec3(txA * glm::vec4(
                particleRegistry.get<ParticleLocalPosition>(stencil).value, 1.0f));

        // Two world-space boundary samples per stencil axis (+-offset from midpoint).
        std::vector<glm::vec3> boundaryPointsWorld;
        const AdjacencyList& adjA = foamAdjacencyLists.at(foamA);
        adjA.forEachNeighbor(static_cast<uint32_t>(stencil), [&](uint32_t nbId) {
            entt::entity nb = static_cast<entt::entity>(nbId);
            if (!particleRegistry.all_of<ParticleLocalPosition>(nb)) return;
            glm::vec3 nbWorld = glm::vec3(txA * glm::vec4(
                particleRegistry.get<ParticleLocalPosition>(nb).value, 1.0f));
            glm::vec3 axis = nbWorld - stencilWorld;
            boundaryPointsWorld.push_back(stencilWorld + (0.5f + kStencilBoundaryOffset) * axis);
            boundaryPointsWorld.push_back(stencilWorld + (0.5f - kStencilBoundaryOffset) * axis);
        });

        for (int targetFoamId : foamList) {
            StencilFoamKey key{ stencilInt, targetFoamId };
            const glm::mat4      invTxB    = glm::inverse(foamTransforms.at(targetFoamId));
            const AdjacencyList& targetAdj = foamAdjacencyLists.at(targetFoamId);
            const auto& candidates = result.cellContacts.at(key);

            for (const glm::vec3& sampleWorld : boundaryPointsWorld) {
                glm::vec3 sampleLocal = glm::vec3(invTxB * glm::vec4(sampleWorld, 1.0f));

                AABB& effAABB = result.effectiveAABB[key];
                effAABB.min_pt = glm::min(effAABB.min_pt, sampleLocal);
                effAABB.max_pt = glm::max(effAABB.max_pt, sampleLocal);

                for (entt::entity mutParticle : candidates) {
                    if (!particleRegistry.all_of<ParticleLocalPosition>(mutParticle)) continue;

                    glm::vec3 cellCenter =
                        particleRegistry.get<ParticleLocalPosition>(mutParticle).value;

                    std::vector<glm::vec3> cellNeighborCenters;
                    targetAdj.forEachNeighbor(static_cast<uint32_t>(mutParticle), [&](uint32_t nbId) {
                        entt::entity nb = static_cast<entt::entity>(nbId);
                        if (!particleRegistry.all_of<ParticleLocalPosition>(nb)) return;
                        cellNeighborCenters.push_back(
                            particleRegistry.get<ParticleLocalPosition>(nb).value);
                    });

                    if (voronoiContains(sampleLocal, cellCenter, cellNeighborCenters))
                        stencilBoundaryContacts[key].push_back({sampleLocal, mutParticle});
                }
            }
        }
    }

    // 3) Filter to entries whose intersection AABB spans the effective stencil AABB on
    //    at least one axis, indicating a genuine crossing.
    for (const auto& [key, hits] : stencilBoundaryContacts) {
        AABB intersectionAABB{};
        for (const auto& hit : hits) {
            intersectionAABB.min_pt = glm::min(intersectionAABB.min_pt, hit.localPoint);
            intersectionAABB.max_pt = glm::max(intersectionAABB.max_pt, hit.localPoint);
        }

        const AABB& effAABB = result.effectiveAABB.at(key);

        bool spans =
            (intersectionAABB.min_pt.x == effAABB.min_pt.x && intersectionAABB.max_pt.x == effAABB.max_pt.x) ||
            (intersectionAABB.min_pt.y == effAABB.min_pt.y && intersectionAABB.max_pt.y == effAABB.max_pt.y) ||
            (intersectionAABB.min_pt.z == effAABB.min_pt.z && intersectionAABB.max_pt.z == effAABB.max_pt.z);

        if (spans)
            result.validWork[key] = hits;
    }

    return result;
}

// =============================================================================
// update
// =============================================================================

std::vector<FoamTopologyUpdate> Topology::update(
    const GpuSlabAllocator&                              gpuSlab,
    std::unordered_map<int, AdjacencyList>&              foamAdjacencyLists,
    const std::unordered_map<int, glm::mat4>&            foamTransforms,
    const entt::registry&                                foamRegistry,
    entt::registry&                                      particleRegistry)
{
    // Collect IDs needed by detectCollisions.
    std::vector<int> foamIds;
    foamIds.reserve(foamAdjacencyLists.size());
    for (const auto& [id, _] : foamAdjacencyLists)
        foamIds.push_back(id);

    std::vector<int> primaryFoamIds;
    for (auto entity : foamRegistry.view<Controller>())
        primaryFoamIds.push_back(static_cast<int>(entity));

    std::vector<uint32_t> primaryParticleIds;
    for (auto entity : particleRegistry.view<Stencil>())
        primaryParticleIds.push_back(static_cast<uint32_t>(entity));

    auto collisions = detectCollisions(gpuSlab, foamTransforms, particleRegistry,
                                       foamIds, primaryFoamIds, primaryParticleIds);

    // Steps 1-3: collision filtering, boundary sampling, spanning-axis filter.
    auto [cellContacts, effectiveAABB, validWork] =
        buildStencilWork(collisions, foamAdjacencyLists, foamTransforms, particleRegistry);

    // 4) Point creation
    StencilFoamMap<std::unordered_set<entt::entity>> stencilROI;
    StencilFoamMap<std::unordered_set<entt::entity>> stencilPerimeter;
    StencilFoamMap<std::unordered_set<entt::entity>> stencilOverlap;
    std::unordered_map<entt::entity, std::unordered_set<entt::entity>> cellSamples;

    for (const auto& [key, hits] : validWork) {
        auto [stencilInt, targetFoamId] = key;
        const AdjacencyList& targetAdj = foamAdjacencyLists.at(targetFoamId);
        const AABB&          effAABB   = effectiveAABB.at(key);

        if (cellContacts.count(key)) {
            for (entt::entity cellEnt : cellContacts.at(key)) {
                if (!particleRegistry.all_of<ParticleVertices>(cellEnt)) continue;
                const auto& verts = particleRegistry.get<ParticleVertices>(cellEnt).vertices;
                if (verts.empty()) continue;

                AABB cellAABB{ verts[0], verts[0] };
                for (const auto& v : verts) {
                    cellAABB.min_pt = glm::min(cellAABB.min_pt, v);
                    cellAABB.max_pt = glm::max(cellAABB.max_pt, v);
                }

                if (cellAABB.min_pt.x >= effAABB.min_pt.x && cellAABB.max_pt.x <= effAABB.max_pt.x &&
                    cellAABB.min_pt.y >= effAABB.min_pt.y && cellAABB.max_pt.y <= effAABB.max_pt.y &&
                    cellAABB.min_pt.z >= effAABB.min_pt.z && cellAABB.max_pt.z <= effAABB.max_pt.z)
                    stencilOverlap[key].insert(cellEnt);
            }
        }
        for (const auto& hit : hits)
            stencilOverlap[key].insert(hit.mutParticle);

        // Densification loop
        for (const auto& hit : hits) {
            const glm::vec3 cellCenter =
                particleRegistry.get<ParticleLocalPosition>(hit.mutParticle).value;

            // Surface Cell Scaffolding: place a new particle at the boundary hit point.
            addInteriorParticle(key, hit.mutParticle, hit.localPoint,
                                particleRegistry, targetAdj, stencilROI, cellSamples);

            // Reflected air particles across air-facing bisectors of hit.mutParticle.
            targetAdj.forEachNeighbor(static_cast<uint32_t>(hit.mutParticle), [&](uint32_t nbId) {
                entt::entity nb = static_cast<entt::entity>(nbId);
                if (!particleRegistry.all_of<ParticleLocalPosition, ParticleOpacity>(nb)) return;
                if (particleRegistry.get<ParticleOpacity>(nb).value != 0.0f) return;

                glm::vec3 nbCenter = particleRegistry.get<ParticleLocalPosition>(nb).value;
                if (auto reflected = faceReflect(hit.localPoint, cellCenter, nbCenter))
                    addAirParticle(hit.mutParticle, *reflected, particleRegistry, cellSamples);
            });

            // Surface Cell Neighbor Scaffolding: mirror hit.localPoint into each
            // surface neighbor outside the overlap region to preserve shared edges.
            targetAdj.forEachNeighbor(static_cast<uint32_t>(hit.mutParticle), [&](uint32_t nbrId) {
                entt::entity nbr = static_cast<entt::entity>(nbrId);
                if (!particleRegistry.all_of<Surface, ParticleLocalPosition>(nbr)) return;
                if (stencilOverlap[key].count(nbr)) return;

                glm::vec3 nbrCenter = particleRegistry.get<ParticleLocalPosition>(nbr).value;
                auto reflectedIntoNbr = faceReflect(hit.localPoint, cellCenter, nbrCenter);
                if (!reflectedIntoNbr) return;

                addInteriorParticle(key, nbr, *reflectedIntoNbr,
                                    particleRegistry, targetAdj, stencilROI, cellSamples);

                // Air ghosts for each air neighbor of nbr.
                targetAdj.forEachNeighbor(nbrId, [&](uint32_t airNbrId) {
                    entt::entity airNbr = static_cast<entt::entity>(airNbrId);
                    if (!particleRegistry.all_of<Surface, ParticleLocalPosition, ParticleOpacity>(airNbr)) return;
                    if (particleRegistry.get<ParticleOpacity>(airNbr).value != 0.0f) return;

                    glm::vec3 airNbrCenter = particleRegistry.get<ParticleLocalPosition>(airNbr).value;
                    if (auto reflectedAir = faceReflect(*reflectedIntoNbr, nbrCenter, airNbrCenter))
                        addAirParticle(nbr, *reflectedAir, particleRegistry, cellSamples);
                });
            });

            // Internal Densification: BFS into non-surface opaque interior cells,
            // depositing N uniformly distributed sample points at each.
            constexpr int kDensifyDepth   = 1;
            constexpr int kSamplesPerCell = 2;

            struct BfsNode { entt::entity cell; int depth; };
            std::unordered_map<entt::entity, int> nodeDepth;
            std::queue<BfsNode>                   bfsQueue;

            nodeDepth[hit.mutParticle] = 0;
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
                            // Single sample: preserve the cell center.
                            addInteriorParticle(key, current, currentCenter,
                                                particleRegistry, targetAdj, stencilROI, cellSamples);
                        } else {
                            // Multiple samples: stride evenly across the vertex list,
                            // placing each point at the midpoint of center-to-vertex.
                            int stride = static_cast<int>(verts.size()) / kSamplesPerCell;
                            if (stride < 1) stride = 1;
                            for (int s = 0; s < kSamplesPerCell; ++s) {
                                int vi = (s * stride) % static_cast<int>(verts.size());
                                addInteriorParticle(key, current,
                                                    0.5f * (currentCenter + verts[vi]),
                                                    particleRegistry, targetAdj, stencilROI, cellSamples);
                            }
                        }
                    }
                }

                if (depth < kDensifyDepth) {
                    targetAdj.forEachNeighbor(static_cast<uint32_t>(current), [&](uint32_t nbId) {
                        entt::entity nb = static_cast<entt::entity>(nbId);
                        if (nodeDepth.count(nb)) return;
                        if (particleRegistry.all_of<Surface>(nb)) return;
                        if (!particleRegistry.all_of<ParticleOpacity>(nb)) return;
                        if (particleRegistry.get<ParticleOpacity>(nb).value == 0.0f) return;
                        nodeDepth[nb] = depth + 1;
                        bfsQueue.push({ nb, depth + 1 });
                    });
                }
            }

            // Perimeter = nodes at the maximum depth actually reached by the BFS,
            // plus all surface cells in the stencil ROI (they form the boundary
            // between the intersection region and the surrounding foam).
            int maxDepth = 0;
            for (const auto& [e, d] : nodeDepth)
                maxDepth = std::max(maxDepth, d);
            for (const auto& [e, d] : nodeDepth)
                if (d == maxDepth)
                    stencilPerimeter[key].insert(e);
            for (entt::entity roiCell : stencilROI[key])
                if (particleRegistry.all_of<Surface>(roiCell))
                    stencilPerimeter[key].insert(roiCell);
        }
    }

    // 5) Triangulation — one async task per (stencil, foamB) key.
    // Each task is independent: it reads only from registry/adjacency data that is
    // not mutated after step 4, and writes to its own local TriResult.
    using TriResult = std::tuple<AdjacencyList, std::unordered_map<uint32_t, std::vector<glm::vec3>>>;
    StencilFoamMap<TriResult> triangulationResults;

    // Lambda: build the vertex set for one key and triangulate it.
    auto triangulateKey = [&](const StencilFoamKey& key) -> TriResult {
        auto [stencilInt, targetFoamId] = key;
        const AdjacencyList& targetAdj = foamAdjacencyLists.at(targetFoamId);
        std::unordered_map<uint32_t, glm::vec3> triangulationVertices;

        // Add all samples generated within the stencil ROI.
        for (entt::entity cell : stencilROI.at(key)) {
            for (entt::entity sample : cellSamples.at(cell)) {
                triangulationVertices[static_cast<uint32_t>(sample)] =
                    particleRegistry.get<ParticleLocalPosition>(sample).value;
            }
        }

        // Add geometric support beyond the perimeter: for each neighbor of a perimeter
        // cell, classify and include it so the triangulation has proper boundary context:
        //   - Air neighbor (opacity == 0): include its position directly.
        //   - Non-ROI, non-sampled neighbor: include its center position directly.
        //   - Otherwise (sampled by another stencil's ROI): include all its cellSamples.
        const auto& roiCells = stencilROI.at(key);
        for (entt::entity perimCell : stencilPerimeter.at(key)) {
            targetAdj.forEachNeighbor(static_cast<uint32_t>(perimCell), [&](uint32_t nbId) {
                entt::entity nb = static_cast<entt::entity>(nbId);
                if (roiCells.count(nb)) return;
                if (!particleRegistry.all_of<ParticleLocalPosition>(nb)) return;

                const glm::vec3 nbPos = particleRegistry.get<ParticleLocalPosition>(nb).value;

                if (particleRegistry.all_of<ParticleOpacity>(nb) &&
                    particleRegistry.get<ParticleOpacity>(nb).value == 0.0f) {
                    // Air particle — include its position as a boundary anchor.
                    triangulationVertices[static_cast<uint32_t>(nb)] = nbPos;
                } else if (!cellSamples.count(nb)) {
                    // Interior cell with no generated samples — use its center.
                    triangulationVertices[static_cast<uint32_t>(nb)] = nbPos;
                } else {
                    // Cell has samples from another stencil's ROI — include all of them.
                    for (entt::entity sample : cellSamples.at(nb)) {
                        if (particleRegistry.all_of<ParticleLocalPosition>(sample))
                            triangulationVertices[static_cast<uint32_t>(sample)] =
                                particleRegistry.get<ParticleLocalPosition>(sample).value;
                    }
                }
            });
        }

        // Convert to parallel vectors and triangulate.
        std::vector<uint32_t>  triIds;
        std::vector<glm::vec3> triPositions;
        triIds.reserve(triangulationVertices.size());
        triPositions.reserve(triangulationVertices.size());
        for (const auto& [id, pos] : triangulationVertices) {
            triIds.push_back(id);
            triPositions.push_back(pos);
        }
        return triangulateVoronoiCells(triPositions, triIds);
    };

    // Dispatch one async task per key, then gather.
    StencilFoamMap<std::future<TriResult>> trifutures;
    trifutures.reserve(validWork.size());
    for (const auto& [key, _] : validWork)
        trifutures[key] = std::async(std::launch::async, triangulateKey, key);

    for (auto& [key, fut] : trifutures)
        triangulationResults[key] = fut.get();
    

    // 6) CPU/GPU data structure updates
    std::vector<FoamTopologyUpdate> results;

    // Accumulate per-foam changes across all (stencil, foam) keys before
    // building the final FoamTopologyUpdate structs.
    std::unordered_map<int, std::unordered_set<uint32_t>>                          foamParticleAdditions;
    std::unordered_map<int, std::unordered_set<uint32_t>>                          foamParticleDeletions;
    std::unordered_map<int, std::vector<std::pair<uint32_t, uint32_t>>> foamEdgeAdditions;

    for (const auto& [key, triResult] : triangulationResults) {
        // Update ParticleVertices and ParticleMass for all newly added particles
        auto [stencilInt, targetFoamId] = key;
        const auto& voronoiVertices = std::get<1>(triResult);

        // Retrieve the foam's density so we can approximate each cell's mass.
        entt::entity foamEnt = static_cast<entt::entity>(targetFoamId);
        float density = 1.0f;
        if (foamRegistry.all_of<Density>(foamEnt))
            density = foamRegistry.get<Density>(foamEnt).value;

        for (const auto& [rawId, verts] : voronoiVertices) {
            if (verts.empty()) continue;

            entt::entity particle = static_cast<entt::entity>(rawId);

            // Update Voronoi vertices.
            particleRegistry.get<ParticleVertices>(particle).vertices = verts;

            // Approximate volume from Voronoi AABB and multiply by density.
            glm::vec3 aabbMin = verts[0];
            glm::vec3 aabbMax = verts[0];
            for (const auto& v : verts) {
                aabbMin = glm::min(aabbMin, v);
                aabbMax = glm::max(aabbMax, v);
            }
            glm::vec3 extent = aabbMax - aabbMin;
            float approxVolume = extent.x * extent.y * extent.z;
            particleRegistry.get<ParticleMass>(particle).value = approxVolume * density;
        }

        // Update CPU adjacency list.
        AdjacencyList& foamAdj = foamAdjacencyLists.at(targetFoamId);

        // Deletions: remove every node in the stencil ROI and overlap region.
        // stencilROI holds the original cells that were resampled; stencilOverlap
        // holds cells fully inside the stencil AABB. Both sets are replaced by
        // the triangulation's output nodes.
        for (entt::entity e : stencilROI.at(key))
            foamAdj.deleteNode(static_cast<uint32_t>(e));
        for (entt::entity e : stencilOverlap.at(key))
            foamAdj.deleteNode(static_cast<uint32_t>(e));

        // Additions: merge all nodes from the triangulation result into the foam's
        // adjacency list; the predicate admits every node unconditionally.
        const AdjacencyList& resultAdj = std::get<0>(triResult);
        foamAdj.copyNodesFrom(resultAdj, [](uint32_t) { return true; });

        // Particle additions: collect every sample generated inside the ROI cells.
        auto& additions = foamParticleAdditions[targetFoamId];
        for (entt::entity roiCell : stencilROI.at(key)) {
            auto it = cellSamples.find(roiCell);
            if (it == cellSamples.end()) continue;
            for (entt::entity sample : it->second)
                additions.insert(static_cast<uint32_t>(sample));
        }

        // Particle deletions: the original ROI nodes are being replaced.
        auto& deletions = foamParticleDeletions[targetFoamId];
        for (entt::entity roiCell : stencilROI.at(key))
            deletions.insert(static_cast<uint32_t>(roiCell));

        // Edge additions: union all edges from the triangulation result.
        auto [cooSrc, cooDst] = resultAdj.buildCOO();
        auto& edges = foamEdgeAdditions[targetFoamId];
        for (size_t i = 0; i < cooSrc.size(); ++i)
            edges.emplace_back(cooSrc[i], cooDst[i]);
    }

    // Construct one FoamTopologyUpdate per foam that was touched.
    // Gather the union of all foam ids across the three maps.
    std::unordered_set<int> touchedFoams;
    for (const auto& [id, _] : foamParticleAdditions) touchedFoams.insert(id);
    for (const auto& [id, _] : foamParticleDeletions)  touchedFoams.insert(id);
    for (const auto& [id, _] : foamEdgeAdditions)      touchedFoams.insert(id);

    for (int foamId : touchedFoams) {
        std::vector<glm::vec3> positions;
        std::vector<glm::vec4> colors;
        std::vector<uint8_t>   surface_masks;
        std::vector<AABB>      aabbs;
        std::vector<uint32_t>  active_ids;

        if (foamParticleAdditions.count(foamId)) {
            for (uint32_t rawId : foamParticleAdditions.at(foamId)) {
                entt::entity p = static_cast<entt::entity>(rawId);

                // Position
                positions.push_back(
                    particleRegistry.get<ParticleLocalPosition>(p).value);

                // Color (rgb) + opacity packed into vec4
                const auto& col = particleRegistry.get<ParticleColor>(p);
                float opacity   = particleRegistry.get<ParticleOpacity>(p).value;
                colors.push_back(glm::vec4(col.rgb, opacity));

                // Surface mask
                surface_masks.push_back(
                    particleRegistry.all_of<Surface>(p) ? uint8_t(1) : uint8_t(0));

                // AABB from Voronoi vertices
                const auto& verts = particleRegistry.get<ParticleVertices>(p).vertices;
                if (verts.empty()) {
                    const glm::vec3& pos = positions.back();
                    aabbs.push_back(AABB{ pos, pos });
                } else {
                    AABB aabb{ verts[0], verts[0] };
                    for (const auto& v : verts) {
                        aabb.min_pt = glm::min(aabb.min_pt, v);
                        aabb.max_pt = glm::max(aabb.max_pt, v);
                    }
                    aabbs.push_back(aabb);
                }

                active_ids.push_back(rawId);
            }
        }

        // Deletions
        std::vector<uint32_t> del_ids;
        if (foamParticleDeletions.count(foamId))
            del_ids.assign(
                foamParticleDeletions.at(foamId).begin(),
                foamParticleDeletions.at(foamId).end());

        // COO edge buffers — deserialize vector<pair> into parallel src/dst vectors
        std::vector<uint32_t> coo_src;
        std::vector<uint32_t> coo_dst;
        if (foamEdgeAdditions.count(foamId)) {
            const auto& edgePairs = foamEdgeAdditions.at(foamId);
            coo_src.reserve(edgePairs.size());
            coo_dst.reserve(edgePairs.size());
            for (const auto& [s, d] : edgePairs) {
                coo_src.push_back(s);
                coo_dst.push_back(d);
            }
        }

        results.push_back(FoamTopologyUpdate{
            static_cast<entt::entity>(foamId),
            FoamUpdate(
                std::move(positions),
                std::move(colors),
                std::move(surface_masks),
                std::move(aabbs),
                std::move(active_ids),
                std::move(del_ids),
                std::move(coo_src),
                std::move(coo_dst))
        });
    }

    return results;
}

} // namespace DynamicFoam::Sim2D
