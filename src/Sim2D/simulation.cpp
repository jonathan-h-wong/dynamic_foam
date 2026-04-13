#include "dynamic_foam/Sim2D/components.h"
#include "dynamic_foam/Sim2D/simulation.h"
#include "dynamic_foam/Sim2D/utils.h"
#include <entt/entity/registry.hpp>

namespace DynamicFoam::Sim2D {

    Simulation::Simulation(
        const SceneGraph& sceneGraph, 
        const glm::ivec2& windowSize
    ) : renderSubsystem(), windowSize_(windowSize) {
        // Creates rigid body and particle registries
        // For each foam, this creates 1 rigid body and N particles, and populates registries
        for (const auto& [foamId, foam] : sceneGraph.foams) {
            auto rigidBody = foamRegistry.create();

            // Set RB position from the scene graph
            const auto& transform = sceneGraph.worldTransforms.at(foamId);
            foamRegistry.emplace<Position>(rigidBody, glm::vec3(transform[3]));

            // Set RB transient components
            foamRegistry.emplace<Velocity>(rigidBody, glm::vec3(0.0f));
            foamRegistry.emplace<Orientation>(rigidBody, glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
            foamRegistry.emplace<AngularVelocity>(rigidBody, glm::vec3(0.0f));

            // Set RB foam type
            if (sceneGraph.isController.at(foamId)) {
                foamRegistry.emplace<Controller>(rigidBody);
            } else if (sceneGraph.isDynamic.at(foamId)) {
                foamRegistry.emplace<Dynamic>(rigidBody);
            } else {
                foamRegistry.emplace<Static>(rigidBody);
            }

            // Set RB density, intertiaTensor
            foamRegistry.emplace<Density>(rigidBody, foam.density);
            foamRegistry.emplace<InertiaTensor>(rigidBody, foam.intertiaTensor);

            std::unordered_map<uint32_t, entt::entity> particleMap;
            for (const auto& [particleId, localPos] : foam.particlePosition) {
                particleMap[particleId] = particleRegistry.create();
            }

            std::unordered_map<uint32_t, uint32_t> particleRemap;
            for (const auto& [particleId, entity] : particleMap)
                particleRemap[particleId] = static_cast<uint32_t>(entity);

            // Add RB to central Adjacency List registry
            foamAdjacencyLists[static_cast<int>(rigidBody)] = foam.adjacencyList.remap(particleRemap);

            // Set Particle components
            for (const auto& [particleId, localPos] : foam.particlePosition) {
                auto particle = particleMap.at(particleId);

                particleRegistry.emplace<ParticleLocalPosition>(particle, localPos);
                particleRegistry.emplace<ParticleColor>(particle, foam.particleColor.at(particleId));
                particleRegistry.emplace<ParticleOpacity>(particle, foam.particleOpacity.at(particleId));
                
                float mass = foam.particleMass.at(particleId);
                particleRegistry.emplace<ParticleMass>(particle, mass);
                const auto& p_verts = foam.particleVertices.at(particleId);
                particleRegistry.emplace<ParticleVertices>(particle, p_verts);

                glm::vec3 worldPos = glm::vec3(transform * glm::vec4(localPos, 1.0f));
                particleRegistry.emplace<ParticleWorldPosition>(particle, worldPos);

                if (foam.isStencil.at(particleId)) {
                    particleRegistry.emplace<Stencil>(particle);
                } else if (foam.isMutable.at(particleId)) {
                    particleRegistry.emplace<Mutable>(particle);
                } else {
                    particleRegistry.emplace<Immutable>(particle);
                }
            }

            // Find and emplace surface particles
            auto surface_cells = findSurfaceCells(foam.adjacencyList, foam.particleOpacity);
            for(const auto& p_id : surface_cells){
                particleRegistry.emplace<Surface>(particleMap.at(p_id));
            }
        }

        // Initialise the GPU slab (allocates device buffers, builds BVH and
        // CSR adjacency into slab slices, uploads static particle data).
        initSlab();
    }

    // -------------------------------------------------------------------------
    // initSlab
    // -------------------------------------------------------------------------
    void Simulation::initSlab() {
        // Tally buffer totals across all foams.
        int total_bvh_nodes = 0;
        int total_csr_nodes = 0; // Σ (N_i + 1)
        int total_csr_edges = 0; // Σ directed edge count
        int total_particles = 0;

        for (const auto& [foam_id, adj] : foamAdjacencyLists) {
            const int N = adj.nodeCount();
            const int E = static_cast<int>(adj.getCOOSrc().size());
            total_bvh_nodes += (N > 1) ? 2 * N - 1 : 1;
            total_csr_nodes += N + 1;
            total_csr_edges += E;
            total_particles += N;
        }

        // Allocate flat device buffers.
        gpuSlab.init(total_bvh_nodes, total_csr_nodes, total_csr_edges, total_particles);

        // Per-foam slice setup.
        for (auto& [foam_id, adj] : foamAdjacencyLists) {
            const int N             = static_cast<int>(adj.nodeCount());
            const int E             = static_cast<int>(adj.getCOOSrc().size());
            const int num_bvh_nodes = (N > 1) ? 2 * N - 1 : 1;

            // Allocate a 2×-overcommitted slab slot.
            gpuSlab.allocate(foam_id, num_bvh_nodes, N + 1, E, N);
            const FoamSlot& slot = gpuSlab.slots.at(foam_id);

            // Build GPU CSR into slab slice and bias offsets.
            adj.buildGPUAdjacencyList(
                foamGpuAdj[foam_id],
                gpuSlab.d_csr_colidx + slot.csr_edge_offset,
                static_cast<size_t>(slot.csr_edge_capacity),
                gpuSlab.d_csr_rowptr + slot.csr_node_offset,
                static_cast<size_t>(slot.csr_node_capacity));
            gpuSlab.biasCsrOffsets(foam_id);

            // Bulk-upload all particle data (AABBs, colors, positions, surface mask)
            // and Morton-sort every buffer together in a single pass.
            stageParticleData(static_cast<entt::entity>(foam_id));

            // Build BVH using the now Morton-sorted particle AABBs.
            foamBVHs[foam_id].build(
                gpuSlab.d_particle_aabbs + slot.particle_offset, N,
                gpuSlab.d_bvh_nodes + slot.bvh_offset,
                slot.bvh_capacity);

            // Rebuild the CSR in Morton order so that CSR row i and neighbor
            // column indices are both Morton positions, matching prim_idx from
            // the BVH leaves and the layout of all other slab particle arrays.
            rebuildSlabCsrMortonOrder(foam_id);
        }
    }

    // -------------------------------------------------------------------------
    // rebuildSlabCsrMortonOrder
    //
    // Rebuilds the GPU CSR for one foam using the Morton permutation stored in
    // foamMortonPerms[foam_id].  After bulkMortonSort all particle slab buffers
    // (AABBs, positions, colors, surface mask) are reordered so that GPU buffer
    // index i corresponds to Morton position i.  The BVH is then built on top
    // of that Morton-sorted AABB array, so every leaf's prim_idx is also a
    // Morton position.  The CSR must use the same indexing convention:
    //   - CSR row i  → node at Morton position i
    //   - CSR column indices → Morton positions of neighbors
    // buildGPUAdjacencyList accepts a sorted_ids parameter that re-orders the
    // build entirely, so we pass the Morton-permuted node-ID list directly.
    // -------------------------------------------------------------------------
    void Simulation::rebuildSlabCsrMortonOrder(int foam_id) {
        auto it = gpuSlab.slots.find(foam_id);
        if (it == gpuSlab.slots.end() || it->second.dead) return;
        const auto permIt = foamMortonPerms.find(foam_id);
        if (permIt == foamMortonPerms.end()) return;

        const auto& adj     = foamAdjacencyLists.at(foam_id);
        const auto& ordered = adj.getOrderedNodeIds();
        const int   N       = static_cast<int>(ordered.size());
        const auto& perm    = permIt->second;

        // Build the Morton-sorted node ID list:
        //   morton_sorted_ids[i] = node whose particle data sits at slab position i.
        std::vector<uint32_t> morton_sorted_ids(N);
        for (int i = 0; i < N; ++i)
            morton_sorted_ids[i] = ordered[perm[i]];

        const FoamSlot& slot = it->second;
        adj.buildGPUAdjacencyList(
            foamGpuAdj[foam_id],
            gpuSlab.d_csr_colidx + slot.csr_edge_offset,
            static_cast<size_t>(slot.csr_edge_capacity),
            gpuSlab.d_csr_rowptr + slot.csr_node_offset,
            static_cast<size_t>(slot.csr_node_capacity),
            morton_sorted_ids);
        gpuSlab.biasCsrOffsets(foam_id);
    }

    // -------------------------------------------------------------------------
    // rebuildAllSlabCsr
    //
    // Rebuilds the GPU CSR adjacency for every live foam in Morton-sorted order.
    // Must be called after gpuSlab.compact() so that the biased CSR
    // node_offsets (which carry the old csr_edge_offset) are corrected for the
    // new slab layout produced by compaction.
    // -------------------------------------------------------------------------
    void Simulation::rebuildAllSlabCsr() {
        for (const auto& [foam_id, adj] : foamAdjacencyLists)
            rebuildSlabCsrMortonOrder(foam_id);
    }

    // -------------------------------------------------------------------------
    // stageParticleData
    //
    // Single entry point for all per-particle GPU data for one foam.
    // Builds CPU arrays for local-space AABBs, RGBA colors, world positions,
    // surface mask, and active mask (surface + 1-hop neighbours), uploads all
    // four static buffers to the GPU slab in getOrderedNodeIds() order, then
    // calls gpuSlab.bulkMortonSort which:
    //   - reorders every slab buffer (AABBs, colors, positions, surface mask)
    //     together by local-space Morton code in a single pass,
    //   - compacts active particle positions into d_active_ids,
    //   - writes the permutation to foamMortonPerms[foam_id] so render() can
    //     upload per-frame positions in the same Morton-sorted order.
    // -------------------------------------------------------------------------
    void Simulation::stageParticleData(entt::entity foamEntity) {
        const int foam_id = static_cast<int>(foamEntity);
        const auto it = gpuSlab.slots.find(foam_id);
        if (it == gpuSlab.slots.end() || it->second.dead) return;

        const auto& adj     = foamAdjacencyLists.at(foam_id);
        const auto& ordered = adj.getOrderedNodeIds();
        const int   N       = static_cast<int>(ordered.size());
        if (N == 0) return;

        // Build all per-particle arrays in the same getOrderedNodeIds() order.
        std::vector<AABB>      h_aabbs(N);
        std::vector<glm::vec4> h_colors(N);
        std::vector<glm::vec3> h_positions(N);
        std::vector<uint8_t>   h_surface(N, 0);
        std::vector<uint32_t>  h_active_ids(N);

        for (int li = 0; li < N; ++li) {
            const entt::entity e = static_cast<entt::entity>(ordered[li]);

            const auto& verts = particleRegistry.get<ParticleVertices>(e).vertices;
            auto [mn, mx] = calculateAABB(verts);
            h_aabbs[li] = AABB(mn, mx);

            const auto* c = particleRegistry.try_get<ParticleColor>(e);
            const auto* o = particleRegistry.try_get<ParticleOpacity>(e);
            if (c && o) h_colors[li] = glm::vec4(c->rgb, o->value);

            h_positions[li] = particleRegistry.get<ParticleWorldPosition>(e).value;

            if (particleRegistry.all_of<Surface>(e)) h_surface[li] = 1;

            h_active_ids[li] = ordered[li];
        }

        // Bulk upload to slab in getOrderedNodeIds() order.
        gpuSlab.stageParticleAABBs(foam_id, h_aabbs.data(), N);
        gpuSlab.stageParticleColors(foam_id, h_colors.data(), N);
        gpuSlab.stageParticlePositions(foam_id, h_positions.data(), N);
        gpuSlab.stageParticleSurfaceMask(foam_id, h_surface.data(), N);
        gpuSlab.stageParticleActiveIds(foam_id, h_active_ids.data(), N);

        // Bulk Morton sort: reorders all slab buffers together (including
        // d_active_ids) and returns the permutation for per-frame position uploads.
        foamMortonPerms[foam_id].resize(N);
        gpuSlab.bulkMortonSort(foam_id, N, foamMortonPerms[foam_id]);
    }

    void Simulation::buildBVH(entt::entity foamEntity) {
        const int foam_id = static_cast<int>(foamEntity);
        auto& adj = foamAdjacencyLists.at(foam_id);
        const int N = static_cast<int>(adj.getOrderedNodeIds().size());
        const int E = static_cast<int>(adj.getCOOSrc().size());

        auto it = gpuSlab.slots.find(foam_id);
        if (it == gpuSlab.slots.end() || it->second.dead) return;

        const int num_bvh_nodes = (N > 1) ? 2 * N - 1 : 1;

        // If particle count has grown beyond the 2x overcommit, tombstone the
        // old slot and allocate a larger one.  The CSR must also be rebuilt
        // into the new slice since the edge-block start offset has changed.
        if (gpuSlab.needsResize(foam_id, num_bvh_nodes, N + 1, E, N)) {
            gpuSlab.resize(foam_id, num_bvh_nodes, N + 1, E, N);
            const FoamSlot& newSlot = gpuSlab.slots.at(foam_id);
            adj.buildGPUAdjacencyList(
                foamGpuAdj[foam_id],
                gpuSlab.d_csr_colidx + newSlot.csr_edge_offset,
                static_cast<size_t>(newSlot.csr_edge_capacity),
                gpuSlab.d_csr_rowptr + newSlot.csr_node_offset,
                static_cast<size_t>(newSlot.csr_node_capacity));
            gpuSlab.biasCsrOffsets(foam_id);
        }

        // Bulk-upload all particle data (AABBs, colors, positions, surface mask)
        // and Morton-sort every buffer together in a single pass.
        stageParticleData(foamEntity);

        // Build BVH using the now Morton-sorted particle AABBs.
        const FoamSlot& slot = gpuSlab.slots.at(foam_id);
        foamBVHs[foam_id].build(
            gpuSlab.d_particle_aabbs + slot.particle_offset, N,
            gpuSlab.d_bvh_nodes + slot.bvh_offset,
            slot.bvh_capacity);

        // Rebuild the CSR in Morton order so that CSR row / column indices
        // are Morton positions, consistent with prim_idx and all other slab buffers.
        rebuildSlabCsrMortonOrder(foam_id);
    }

    void Simulation::applyForwardKinematics(
        entt::entity foamEntity,
        const std::optional<std::unordered_set<entt::entity>>& particleSubset
    ) {
        const auto& pos = foamRegistry.get<Position>(foamEntity);
        const auto& orient = foamRegistry.get<Orientation>(foamEntity);

        glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), pos.value);
        glm::mat4 rotationMatrix = glm::mat4_cast(orient.value);
        glm::mat4 transform = translationMatrix * rotationMatrix;

        const int foamId = static_cast<int>(foamEntity);
        if (!foamAdjacencyLists.count(foamId)) return;

        const auto& adjList = foamAdjacencyLists.at(foamId);
        const auto allParticles = adjList.getOrderedNodeIds();  // std::vector<uint32_t>

        // Build the update buffer: validate and use the subset if provided, otherwise use all particles.
        std::vector<uint32_t> effectiveBuffer;
        if (particleSubset.has_value()) {
            const std::unordered_set<uint32_t> foamParticleSet(allParticles.begin(), allParticles.end());
            for (const auto& particle : *particleSubset) {
                uint32_t id = static_cast<uint32_t>(particle);
                if (!foamParticleSet.count(id)) {
                    throw std::invalid_argument(
                        "Particle entity " + std::to_string(id) +
                        " does not belong to foam " + std::to_string(foamId)
                    );
                }
                effectiveBuffer.push_back(id);
            }
        } else {
            effectiveBuffer = allParticles;
        }

        for (uint32_t pid : effectiveBuffer) {
            auto& worldPos = particleRegistry.get<ParticleWorldPosition>(static_cast<entt::entity>(pid));
            const auto& localPos = particleRegistry.get<ParticleLocalPosition>(static_cast<entt::entity>(pid));
            worldPos.value = glm::vec3(transform * glm::vec4(localPos.value, 1.0f));
        }
    }

    void Simulation::handleUserInput(const UserInput& input, float deltaTime) {
        // --- Orbital camera: WASD moves along the tangent plane of the upper hemisphere ---
        {
            const float     r     = glm::length(camera_.origin);
            const glm::vec3 r_hat = camera_.origin / r;
            const glm::vec3 worldUp = glm::vec3(0.f, 1.f, 0.f);

            // Skip when at the degenerate north-pole (cross product undefined).
            if (glm::length(glm::vec2(r_hat.x, r_hat.z)) > 1e-6f) {
                // Tangent-plane basis at the current camera position:
                //   camRight   — azimuthal (+X when at south face), drives A/D
                //   camForward — polar toward north pole, drives W/S
                //   camUp      — outward radial normal (not used for movement)
                const glm::vec3 camRight   = glm::normalize(glm::cross(worldUp, r_hat));
                const glm::vec3 camForward = glm::cross(r_hat, camRight);
                const glm::vec3 camUp      = r_hat; // outward normal, kept for reference
                (void)camUp;

                const float speed = 1.5f; // world-space units per second
                const float fwd   = (input.key_w ? 1.f : 0.f) - (input.key_s ? 1.f : 0.f);
                const float right = (input.key_d ? 1.f : 0.f) - (input.key_a ? 1.f : 0.f);

                if (fwd != 0.f || right != 0.f) {
                    // Step along the tangent plane then re-project onto the sphere.
                    glm::vec3 new_pos = glm::normalize(
                        camera_.origin + speed * deltaTime * (fwd * camForward + right * camRight)
                    ) * r;

                    // Clamp to upper hemisphere (y >= 0); re-project onto equatorial circle.
                    if (new_pos.y < 0.f) {
                        new_pos.y = 0.f;
                        const float xz_len = glm::length(glm::vec2(new_pos.x, new_pos.z));
                        if (xz_len > 1e-6f)
                            new_pos = glm::vec3(new_pos.x / xz_len, 0.f, new_pos.z / xz_len) * r;
                    }

                    camera_.origin = new_pos;
                }
            }
        }

        applyControllerCursor(input.mouse_pos);
    }

    void Simulation::applyControllerCursor(ImVec2 mouse_pos) {
        // Guard: ImGui reports (-FLT_MAX, -FLT_MAX) when the mouse is outside the window.
        if (mouse_pos.x < -1e6f || mouse_pos.y < -1e6f) return;

        // Convert from ImGui pixel coordinates (top-left origin) to world space
        // using the camera's projection. unprojectPixel delegates to generateRay,
        // so the mapping is consistent with the GPU rendering kernels.
        const glm::vec3 world   = unprojectPixel(camera_, glm::vec2(mouse_pos.x, mouse_pos.y), windowSize_);
        const float     world_x = world.x;
        const float     world_y = world.y;

        foamRegistry.view<Controller, Position>().each([&](auto entity, auto& pos) {
            pos.value = glm::vec3(world_x, world_y, 0.0f);
            applyForwardKinematics(entity);
            // World AABB is computed on-demand from slab + transforms in step().
        });
    }

    void Simulation::updateTopology() {
        const auto results = topologySubsystem.update(gpuSlab, foamAdjacencyLists, foamRegistry, particleRegistry);
        for (const auto& result : results) {
            // Refresh world positions for the affected particles.
            const std::unordered_set<entt::entity> particleSubset(
                result.updatedParticles.begin(), result.updatedParticles.end());
            applyForwardKinematics(result.foamId, particleSubset);

            // Rebuild BVH from current ParticleVertices in the registry.
            // buildBVH also calls stageParticleData which reorders all slab
            // buffers by Morton, refreshes active IDs, and writes the foam's
            // local AABB into slab.d_foam_aabbs (via computeFoamAABB).
            buildBVH(result.foamId);

            // A parent foam snapshot indicates a new foam was spawned from a topology change.
            if (result.parentFoamSnapshot.has_value()) {
                const FoamSnapshot& snap = *result.parentFoamSnapshot;

                // --- Populate the new foam's rigid body components ---
                // Gather per-particle data needed for several derived quantities.
                std::unordered_map<entt::entity, glm::vec3> localPositions;
                std::unordered_map<entt::entity, glm::vec3> worldPositions;
                std::unordered_map<entt::entity, float> masses;
                for (const auto& particle : result.updatedParticles) {
                    localPositions[particle] = particleRegistry.get<ParticleLocalPosition>(particle).value;
                    worldPositions[particle] = particleRegistry.get<ParticleWorldPosition>(particle).value;
                    masses[particle] = particleRegistry.get<ParticleMass>(particle).value;
                }

                // InertiaTensor: derived from the particles' local positions and masses.
                foamRegistry.emplace_or_replace<InertiaTensor>(result.foamId,
                    calculateInertiaTensor(localPositions, masses));

                // Density: sourced from the parent snapshot.
                foamRegistry.emplace_or_replace<Density>(result.foamId, snap.density.value);

                // Foam type: spawned foams inherit Dynamic from the parent snapshot.
                // Static and Controller types are not applicable here.
                if (snap.isDynamic)
                    foamRegistry.emplace_or_replace<Dynamic>(result.foamId);

                // Position: center of mass computed from particle world positions and masses.
                const glm::vec3 childPos = calculateCenterOfMass(worldPositions, masses);
                foamRegistry.emplace_or_replace<Position>(result.foamId, childPos);

                // Velocity: Chasles decomposition — v(p) = v_cm + ω × (p − p_cm).
                // Derived entirely from the parent snapshot to avoid stale registry reads.
                foamRegistry.emplace_or_replace<Velocity>(result.foamId,
                    snap.velocity.value + glm::cross(snap.angularVelocity.value, childPos - snap.position.value));

                // Orientation: sourced from the parent snapshot.
                foamRegistry.emplace_or_replace<Orientation>(result.foamId, snap.orientation.value);

                // AngularVelocity: sourced from the parent snapshot.
                // By Chasles decomposition every sub-body of a rigid body shares the same ω.
                foamRegistry.emplace_or_replace<AngularVelocity>(result.foamId, snap.angularVelocity.value);
            }
        }
    }

    void Simulation::updatePhysics(float deltaTime) {
        const auto updatedFoams = physicsSubsystem.update(gpuSlab, foamAdjacencyLists, foamRegistry, particleRegistry, deltaTime);
    }

    void Simulation::render() {
        // Upload per-frame particle world positions to the slab in Morton-sorted
        // order so the layout matches the sorted AABBs/colors/mask already on device.
        // foamMortonPerms[foam_id][i] is the original getOrderedNodeIds() index of
        // the particle at Morton-sorted slab position i.
        for (const auto& [foam_id, adj] : foamAdjacencyLists) {
            const auto& ordered = adj.getOrderedNodeIds();
            const int   N       = static_cast<int>(ordered.size());
            const auto& perm    = foamMortonPerms.at(foam_id);
            std::vector<glm::vec3> h_positions(N);
            for (int i = 0; i < N; ++i)
                h_positions[i] = particleRegistry.get<ParticleWorldPosition>(
                    static_cast<entt::entity>(ordered[perm[i]])).value;
            gpuSlab.stageParticlePositions(foam_id, h_positions.data(), N);
        }

        // Build per-foam world transforms from position and orientation.
        std::unordered_map<int, glm::mat4> foamTransforms;
        foamRegistry.view<const Position, const Orientation>().each(
            [&](entt::entity e, const Position& pos, const Orientation& orient) {
                const glm::mat4 t = glm::translate(glm::mat4(1.0f), pos.value);
                const glm::mat4 r = glm::mat4_cast(orient.value);
                foamTransforms[static_cast<int>(e)] = t * r;
            });

        // Recompute viewport height from current aspect ratio before rendering.
        camera_.height = camera_.width * (float(windowSize_.y) / float(windowSize_.x));
        renderSubsystem.update(gpuSlab, foamTransforms,
                               camera_, windowSize_, overlayParams);
    }

    void Simulation::step(const UserInput& input, float deltaTime) {
        handleUserInput(input, deltaTime);
        updateTopology();

        // Compact dead slab regions accumulated from prior resizes.
        // compact() D->D packs BVH/CSR/particle data and resets watermarks,
        // but invalidates the biased CSR node_offsets (they were biased by the
        // old csr_edge_offsets); rebuildAllSlabCsr() rewrites them correctly.
        // BVH nodes contain only foam-local indices so their D->D move is valid
        // as-is; no BVH rebuild is needed after compaction.
        if (gpuSlab.needsCompaction()) {
            gpuSlab.compact();
            rebuildAllSlabCsr();
        }

        updatePhysics(deltaTime);
        // Re-sample the cursor position after the expensive subsystems so that
        // Controller foam is placed at its true on-screen location when the GPU
        // kernel fires, minimising input-to-display latency.
        applyControllerCursor(ImGui::GetIO().MousePos);
        render();
    }
};