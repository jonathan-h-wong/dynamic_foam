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
        // Build COO data once per foam; reuse across the tally and upload loops.
        struct FoamCOO { std::vector<uint32_t> src, dst; };
        std::unordered_map<int, FoamCOO> foamCOOs;
        foamCOOs.reserve(foamAdjacencyLists.size());
        for (const auto& [foam_id, adj] : foamAdjacencyLists) {
            auto [src, dst] = adj.buildCOO();
            foamCOOs[foam_id] = FoamCOO{ std::move(src), std::move(dst) };
        }

        // Tally buffer totals across all foams.
        int total_bvh_nodes = 0;
        int total_csr_nodes = 0; // Σ (N_i + 1)
        int total_csr_edges = 0; // Σ directed edge count
        int total_particles = 0;

        for (const auto& [foam_id, adj] : foamAdjacencyLists) {
            const int N = adj.nodeCount();
            const int E = static_cast<int>(foamCOOs.at(foam_id).src.size());
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
            const auto& [coo_src, coo_dst] = foamCOOs.at(foam_id);
            const int E             = static_cast<int>(coo_src.size());
            const int num_bvh_nodes = (N > 1) ? 2 * N - 1 : 1;

            // Allocate a 2×-overcommitted slab slot.
            gpuSlab.allocate(foam_id, num_bvh_nodes, N + 1, E, N);

            // Stage COO edge data into the slab so GPU adjacency kernels can
            // consume it directly without a per-build H2D transfer.
            gpuSlab.stageCOOData(foam_id, coo_src.data(), coo_dst.data(), E);

            // Bulk-upload particle data and Morton-sort every buffer together.
            // After this call d_active_ids is Morton-sorted and slot.active_count is set.
            {
                const auto& adjMap = adj.getAdjList();
                const int   Np     = static_cast<int>(adjMap.size());

                std::vector<AABB>      h_aabbs(Np);
                std::vector<glm::vec4> h_colors(Np);
                std::vector<glm::vec3> h_positions(Np);
                std::vector<uint8_t>   h_surface(Np, 0);
                std::vector<uint32_t>  h_active_ids(Np);

                int li = 0;
                for (const auto& [nodeId, _] : adjMap) {
                    const entt::entity e = static_cast<entt::entity>(nodeId);

                    const auto& verts = particleRegistry.get<ParticleVertices>(e).vertices;
                    h_positions[li] = particleRegistry.get<ParticleLocalPosition>(e).value;
                    if (verts.empty()) {
                        // Boundary air particle: no Voronoi cell, assign a point-AABB at
                        // its local position instead of a degenerate zero-volume box at origin.
                        h_aabbs[li] = AABB(h_positions[li], h_positions[li]);
                    } else {
                        auto [mn, mx] = calculateAABB(verts);
                        h_aabbs[li] = AABB(mn, mx);
                    }

                    const auto* c = particleRegistry.try_get<ParticleColor>(e);
                    const auto* o = particleRegistry.try_get<ParticleOpacity>(e);
                    if (c && o) h_colors[li] = glm::vec4(c->rgb, o->value);

                    if (particleRegistry.all_of<Surface>(e)) h_surface[li] = 1;

                    h_active_ids[li] = nodeId;
                    ++li;
                }

                gpuSlab.stageParticleData(foam_id,
                    h_aabbs.data(), h_colors.data(), h_positions.data(),
                    h_surface.data(), h_active_ids.data(), Np);

                gpuSlab.bulkMortonSort(foam_id, Np);
            }

            // Build BVH using the now Morton-sorted particle AABBs.
            const FoamSlot& slot = gpuSlab.slots.at(foam_id);
            foamBVHs[foam_id].build(
                gpuSlab.d_particle_aabbs + slot.particle_offset, N,
                gpuSlab.d_bvh_nodes + slot.bvh_offset,
                slot.bvh_capacity);

            // Build the GPU CSR from the Morton-sorted d_active_ids and the
            // pre-staged d_coo_src/dst — no H2D uploads needed.
            buildAdj(foam_id);
        }
    }

    // -------------------------------------------------------------------------
    // buildAdj
    //
    // Builds the GPU CSR for one foam directly from the slab's device-resident
    // data: d_active_ids (Morton-sorted, written by bulkMortonSort) and
    // d_coo_src/dst (staged by stageCOOData).  No H2D transfers.
    // -------------------------------------------------------------------------
    void Simulation::buildAdj(int foam_id) {
        auto it = gpuSlab.slots.find(foam_id);
        if (it == gpuSlab.slots.end() || it->second.dead) return;
        const FoamSlot& slot = it->second;
        const uint32_t N = static_cast<uint32_t>(slot.active_count);
        const uint32_t E = static_cast<uint32_t>(slot.coo_count);
        if (N == 0 || E == 0) return;

        buildGPUAdjacencyList(
            gpuSlab.d_active_ids + slot.active_offset,
            N,
            gpuSlab.d_coo_src + slot.coo_offset,
            gpuSlab.d_coo_dst + slot.coo_offset,
            E,
            gpuSlab.d_csr_colidx + slot.csr_edge_offset,
            static_cast<size_t>(slot.csr_edge_capacity),
            gpuSlab.d_csr_rowptr + slot.csr_node_offset,
            static_cast<size_t>(slot.csr_node_capacity));
        gpuSlab.biasCsrOffsets(foam_id);
    }

    // -------------------------------------------------------------------------
    // buildBVH — builds the BVH for one foam from the slab's current
    // Morton-sorted particle AABBs.
    // -------------------------------------------------------------------------
    void Simulation::buildBVH(entt::entity foamEntity) {
        const int foam_id = static_cast<int>(foamEntity);
        const int N = static_cast<int>(foamAdjacencyLists.at(foam_id).nodeCount());
        auto it = gpuSlab.slots.find(foam_id);
        if (it == gpuSlab.slots.end() || it->second.dead) return;
        const FoamSlot& slot = it->second;
        foamBVHs[foam_id].build(
            gpuSlab.d_particle_aabbs + slot.particle_offset, N,
            gpuSlab.d_bvh_nodes + slot.bvh_offset,
            slot.bvh_capacity);
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
            // Local-space particle positions are static; only the rigid-body
            // Position component needs updating. The renderer reads the foam
            // transform each frame, so no stageParticleData call is required.
        });
    }

    void Simulation::updateTopology() {
        std::unordered_map<int, glm::mat4> foamTransforms;
        foamRegistry.view<const Position, const Orientation>().each(
            [&](entt::entity e, const Position& pos, const Orientation& orient) {
                foamTransforms[static_cast<int>(e)] =
                    glm::translate(glm::mat4(1.f), pos.value) * glm::mat4_cast(orient.value);
            });

        const auto results = topologySubsystem.update(gpuSlab, foamAdjacencyLists, foamTransforms, foamRegistry, particleRegistry);

        for (const auto& result : results) {
            const int foam_id = static_cast<int>(result.foamId);
            const AdjacencyList& adj = foamAdjacencyLists.at(foam_id);

            // Build opacity map for this foam from the (now-updated) particle registry.
            std::unordered_map<uint32_t, float> opacityMap;
            for (const auto& [node, _] : adj.getAdjList()) {
                const entt::entity e = static_cast<entt::entity>(node);
                const auto* o = particleRegistry.try_get<ParticleOpacity>(e);
                opacityMap[node] = o ? o->value : 0.f;
            }

            const auto surfaceIds      = findSurfaceCells(adj, opacityMap);
            const int  cycles          = countCycles(adj, surfaceIds, opacityMap);
            const auto componentLabels = findConnectedComponents(adj, opacityMap);
            (void)cycles;

            int num_components = 0;
            for (const auto& [node, comp_id] : componentLabels)
                num_components = std::max(num_components, comp_id);
            
            // Update regardless of split or no-split
            gpuSlab.updateFoamData(foam_id, result.foamUpdate);
            const int N = gpuSlab.slots.at(foam_id).active_count;
            gpuSlab.bulkMortonSort(foam_id, N);
            buildBVH(result.foamId);
            buildAdj(foam_id);

            if (num_components > 1) {
                // Split — the foam has fragmented into num_components pieces.
                // Build surfaceMap required by findSharedAirParticles.
                std::unordered_map<uint32_t, bool> surfaceMap;
                for (const auto& [node, _] : adj.getAdjList())
                    surfaceMap[node] = false;
                for (uint32_t sid : surfaceIds)
                    surfaceMap[sid] = true;

                // Shared air particles: air cells adjacent to 2+ components.
                // Maps component_id -> [shared air particle ids].
                const auto sharedAirByComponent = findSharedAirParticles(
                    adj, surfaceMap, opacityMap, componentLabels);

                // Collect the full set of shared air particles for later cleanup.
                std::unordered_set<uint32_t> allSharedAir;
                for (const auto& [comp_id, air_ids] : sharedAirByComponent)
                    for (uint32_t aid : air_ids)
                        allSharedAir.insert(aid);

                // Group component-member particle IDs (non-air).
                std::unordered_map<int, std::vector<uint32_t>> byComponent;
                for (const auto& [node, comp_id] : componentLabels)
                    byComponent[comp_id].push_back(node);

                // Snapshot parent kinematic state before mutating the registry.
                const glm::vec3 parentPos     = foamRegistry.get<Position>(result.foamId).value;
                const glm::vec3 parentVel     = foamRegistry.get<Velocity>(result.foamId).value;
                const glm::quat parentOrient  = foamRegistry.get<Orientation>(result.foamId).value;
                const glm::vec3 parentOmega   = foamRegistry.get<AngularVelocity>(result.foamId).value;
                const float     parentDensity = foamRegistry.get<Density>(result.foamId).value;
                const bool      parentDynamic = foamRegistry.all_of<Dynamic>(result.foamId);

                for (auto& [comp_id, particle_ids] : byComponent) {
                    // (1) Create a new foam entity and adjacency list.
                    const entt::entity child_entity = foamRegistry.create();
                    const int          child_id     = static_cast<int>(child_entity);
                    AdjacencyList&     child_adj    = foamAdjacencyLists[child_id];

                    // (2) Collect this component's shared air neighbours via one BFS step 
                    std::unordered_set<uint32_t> member_set(particle_ids.begin(), particle_ids.end());

                    const auto sharedIt = sharedAirByComponent.find(comp_id);
                    std::vector<uint32_t> original_air_ids;
                    if (sharedIt != sharedAirByComponent.end())
                        original_air_ids = sharedIt->second;

                    // One BFS step outward: expand member_set by all direct neighbours.
                    std::unordered_set<uint32_t> copy_set = member_set;
                    for (uint32_t pid : particle_ids)
                        adj.forEachNeighbor(pid, [&](uint32_t nbr) { copy_set.insert(nbr); });

                    child_adj.copyNodesFrom(adj, [&](uint32_t id) {
                        return copy_set.count(id) > 0;
                    });

                    // FoamUpdate accumulators — built during steps (3)/(4) where all
                    // clone entity IDs and properties are defined.  Positions are stored in
                    // parent-local space here and shifted to child-local space after the CoM
                    // is known in step (5), immediately before the GPU update in step (6).
                    std::vector<uint32_t>  clone_del_ids;
                    std::vector<glm::vec3> clone_positions;
                    std::vector<glm::vec4> clone_colors;
                    std::vector<uint8_t>   clone_surface_masks;
                    std::vector<AABB>      clone_aabbs;
                    std::vector<uint32_t>  clone_active_ids;
                    std::vector<uint32_t>  clone_coo_src;
                    std::vector<uint32_t>  clone_coo_dst;

                    // (3) Remove the original shared-air nodes from the child — they will be
                    //     replaced by per-component clones in step (4).
                    for (uint32_t aid : original_air_ids) {
                        child_adj.deleteNode(aid);
                        clone_del_ids.push_back(aid); // stage for GPU deletion
                    }

                    // (4) Create duplicate air particles with the same properties
                    //     and add them to the child adjacency list and particle registry.
                    for (uint32_t orig_aid : original_air_ids) {
                        const entt::entity orig_e   = static_cast<entt::entity>(orig_aid);
                        const entt::entity clone_e  = particleRegistry.create();
                        const uint32_t     clone_id = static_cast<uint32_t>(clone_e);

                        // Copy all particle components from the original air particle.
                        if (auto* c = particleRegistry.try_get<ParticleLocalPosition>(orig_e))
                            particleRegistry.emplace<ParticleLocalPosition>(clone_e, *c);
                        if (auto* c = particleRegistry.try_get<ParticleColor>(orig_e))
                            particleRegistry.emplace<ParticleColor>(clone_e, *c);
                        if (auto* c = particleRegistry.try_get<ParticleOpacity>(orig_e))
                            particleRegistry.emplace<ParticleOpacity>(clone_e, *c);
                        if (auto* c = particleRegistry.try_get<ParticleMass>(orig_e))
                            particleRegistry.emplace<ParticleMass>(clone_e, *c);
                        if (auto* c = particleRegistry.try_get<ParticleVertices>(orig_e))
                            particleRegistry.emplace<ParticleVertices>(clone_e, *c);
                        if (particleRegistry.all_of<Surface>(orig_e))
                            particleRegistry.emplace<Surface>(clone_e);
                        if (particleRegistry.all_of<Mutable>(orig_e))
                            particleRegistry.emplace<Mutable>(clone_e);
                        else if (particleRegistry.all_of<Immutable>(orig_e))
                            particleRegistry.emplace<Immutable>(clone_e);
                        else if (particleRegistry.all_of<Stencil>(orig_e))
                            particleRegistry.emplace<Stencil>(clone_e);

                        // Wire the clone into the child adjacency list.
                        child_adj.addNode(clone_id);
                        adj.forEachNeighbor(orig_aid, [&](uint32_t nbr) {
                            if (member_set.count(nbr))
                                child_adj.addEdge(clone_id, nbr);
                        });

                        // Populate GPU FoamUpdate insertion buffers for this clone.
                        // Positions are in parent-local space for now; subtraction of
                        // childCoM_local happens after step (5) below.
                        // All four components are guaranteed present: every particle
                        // entering the registry (constructor or topology) receives them
                        // unconditionally, and the clone was just copied from such a particle.
                        {
                            const glm::vec3 pos  = particleRegistry.get<ParticleLocalPosition>(clone_e).value;
                            const auto&     col_c = particleRegistry.get<ParticleColor>(clone_e);
                            const float     alpha = particleRegistry.get<ParticleOpacity>(clone_e).value;
                            const glm::vec4 col  = glm::vec4(col_c.rgb, alpha);
                            const uint8_t   surf = particleRegistry.all_of<Surface>(clone_e) ? 1u : 0u;
                            const auto& pv_verts = particleRegistry.get<ParticleVertices>(clone_e).vertices;
                            // Boundary air particles have an empty vertex list; use a point-AABB
                            // at their local position rather than a degenerate box at the origin.
                            const AABB aabb = pv_verts.empty()
                                ? AABB(pos, pos)
                                : [&]{ auto [mn, mx] = calculateAABB(pv_verts); return AABB(mn, mx); }();
                            clone_positions.push_back(pos);
                            clone_colors.push_back(col);
                            clone_surface_masks.push_back(surf);
                            clone_aabbs.push_back(aabb);
                            clone_active_ids.push_back(clone_id);

                            // Directed COO edges: clone -> each member-set neighbour (both directions).
                            adj.forEachNeighbor(orig_aid, [&](uint32_t nbr) {
                                if (member_set.count(nbr)) {
                                    clone_coo_src.push_back(clone_id); clone_coo_dst.push_back(nbr);
                                    clone_coo_src.push_back(nbr);      clone_coo_dst.push_back(clone_id);
                                }
                            });
                        }
                    }

                    // (5) Register foam physics for this child, deriving kinematic
                    //     state from the parent foam's snapshot.
                    // Since the parent's origin coincides with its CoM, particle local
                    // positions are already expressed relative to the parent CoM.  We
                    // only need local-space data — no per-particle world transform needed.
                    std::unordered_map<entt::entity, glm::vec3> localPositions;
                    std::unordered_map<entt::entity, float>     masses;
                    for (const auto& [pid, _] : child_adj.getAdjList()) {
                        const entt::entity e = static_cast<entt::entity>(pid);
                        localPositions[e] = particleRegistry.get<ParticleLocalPosition>(e).value;
                        masses[e]         = particleRegistry.get<ParticleMass>(e).value;
                    }
                    // Child CoM in local (parent) space; rotate into world space to get
                    // the world-space offset from the parent CoM.
                    const glm::vec3 childCoM_local = calculateCenterOfMass(localPositions, masses);
                    const glm::vec3 childCoM_world = parentPos + glm::mat3_cast(parentOrient) * childCoM_local;

                    foamRegistry.emplace_or_replace<InertiaTensor>(child_entity,
                        calculateInertiaTensor(localPositions, masses));
                    foamRegistry.emplace_or_replace<Density>(child_entity, parentDensity);
                    if (parentDynamic)
                        foamRegistry.emplace_or_replace<Dynamic>(child_entity);
                    foamRegistry.emplace_or_replace<Position>(child_entity, childCoM_world);
                    // Chasles decomposition: v(child) = v_parent_cm + ω × (R · child_cm_local)
                    foamRegistry.emplace_or_replace<Velocity>(child_entity,
                        parentVel + glm::cross(parentOmega, childCoM_world - parentPos));
                    foamRegistry.emplace_or_replace<Orientation>(child_entity, parentOrient);
                    foamRegistry.emplace_or_replace<AngularVelocity>(child_entity, parentOmega);

                    // (6) GPU mirror — replicate the CPU topology split in the slab.
                    {
                        // (6a) Conditional copy: snapshot all copy_set particles from the
                        // parent slab into a freshly allocated child slot.  Particle data
                        // (positions, colors, AABBs) arrives in parent-local space.
                        std::vector<uint32_t> copy_set_ids(copy_set.begin(), copy_set.end());
                        gpuSlab.copyFoamData(foam_id, child_id,
                            copy_set_ids.data(), static_cast<int>(copy_set_ids.size()));

                        // (6b) Reparent: shift every copied particle's position and AABB
                        // so that childCoM_local becomes the new local origin (0,0,0).
                        gpuSlab.reparentFoamData(child_id, childCoM_local);

                        // (6c) Adjust clone insertion positions/AABBs to child-local space
                        // now that childCoM_local is known.
                        for (auto& p : clone_positions)  p -= childCoM_local;
                        for (auto& a : clone_aabbs) {
                            a.min_pt -= childCoM_local;
                            a.max_pt -= childCoM_local;
                        }

                        // (6d) Submit the FoamUpdate: delete the copied original-air entries
                        // and insert the per-component clone particles with their COO edges.
                        FoamUpdate childUpdate(
                            std::move(clone_positions),
                            std::move(clone_colors),
                            std::move(clone_surface_masks),
                            std::move(clone_aabbs),
                            std::move(clone_active_ids),
                            std::move(clone_del_ids),
                            std::move(clone_coo_src),
                            std::move(clone_coo_dst));
                        gpuSlab.updateFoamData(child_id, childUpdate);

                        // (6e) Re-sort, refit BVH, and rebuild CSR for the new child slot.
                        const int child_N = gpuSlab.slots.at(child_id).active_count;
                        gpuSlab.bulkMortonSort(child_id, child_N);
                        buildBVH(child_entity);
                        buildAdj(child_id);
                    }
                }

                // Post-loop cleanup: erase parent from all registries.
                foamAdjacencyLists.erase(foam_id);
                foamRegistry.destroy(result.foamId);

                // Destroy all shared air particles (replaced by per-child clones).
                for (uint32_t aid : allSharedAir)
                    particleRegistry.destroy(static_cast<entt::entity>(aid));

                // Tombstone the parent GPU slot — its data has been distributed into
                // the child slots and is no longer needed.
                if (gpuSlab.slots.count(foam_id))
                    gpuSlab.slots.at(foam_id).dead = true;

                // Eagerly compact if the dead parent slot pushes waste past threshold.
                if (gpuSlab.needsCompaction()) {
                    gpuSlab.compact();
                    for (const auto& [fid, _] : foamAdjacencyLists)
                        buildAdj(fid);
                }
            }
        }
    }

    void Simulation::updatePhysics(float deltaTime) {
        // Build the same read-only transform map used by render() and updateTopology().
        std::unordered_map<int, glm::mat4> foamTransforms;
        foamRegistry.view<const Position, const Orientation>().each(
            [&](entt::entity e, const Position& pos, const Orientation& orient) {
                foamTransforms[static_cast<int>(e)] =
                    glm::translate(glm::mat4(1.f), pos.value) * glm::mat4_cast(orient.value);
            });

        const auto results = physicsSubsystem.update(gpuSlab, foamAdjacencyLists, foamTransforms, particleRegistry, deltaTime);

        // Publish the integrated state back into the registry.
        // Simulation holds exclusive write privileges over foamRegistry.
        for (const auto& r : results) {
            foamRegistry.emplace_or_replace<Position>(r.foamId,        r.position);
            foamRegistry.emplace_or_replace<Velocity>(r.foamId,        r.velocity);
            foamRegistry.emplace_or_replace<Orientation>(r.foamId,     r.orientation);
            foamRegistry.emplace_or_replace<AngularVelocity>(r.foamId, r.angularVelocity);
        }
    }

    void Simulation::render() {
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
        // compact() D->D packs BVH/CSR/COO/particle data and resets watermarks,
        // but invalidates the biased CSR node_offsets; rebuildSlabAdj() rewrites
        // them correctly using the already-moved d_active_ids and d_coo_src/dst.
        if (gpuSlab.needsCompaction()) {
            gpuSlab.compact();
            for (const auto& [foam_id, _] : foamAdjacencyLists)
                buildAdj(foam_id);
        }

        updatePhysics(deltaTime);
        // Re-sample the cursor position after the expensive subsystems so that
        // Controller foam is placed at its true on-screen location when the GPU
        // kernel fires, minimising input-to-display latency.
        applyControllerCursor(ImGui::GetIO().MousePos);
        render();
    }
};