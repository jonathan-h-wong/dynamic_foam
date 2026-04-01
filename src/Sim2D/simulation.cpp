#include "dynamic_foam/Sim2D/components.h"
#include "dynamic_foam/Sim2D/simulation.h"
#include "dynamic_foam/Sim2D/utils.h"
#include <entt/entity/registry.hpp>

namespace DynamicFoam::Sim2D {

// Forward declarations for the CUDA helpers implemented in simulation_gpu.cu.
// These are free functions that wrap __CUDACC__-guarded template methods so
// they can be called from MSVC-compiled simulation.cpp without triggering the
// nvcc/entt registry compatibility issue.
void buildAdjacencyIntoSlabSlice(
    AdjacencyList<entt::entity>& adj,
    AdjacencyListGPU<entt::entity>& out,
    uint32_t* d_nbrs_slice,         size_t nbrs_cap,
    uint32_t* d_node_offsets_slice, size_t node_offsets_cap);

void biasSlabCsrOffsets(GpuSlabAllocator& slab, int foam_id);

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

            std::unordered_map<int, entt::entity> particleMap;
            for (const auto& [particleId, localPos] : foam.particlePosition) {
                particleMap[particleId] = particleRegistry.create();
            }

            // Add RB to central Adjacency List registry
            foamAdjacencyLists[static_cast<int>(rigidBody)] = AdjacencyList<entt::entity>(
                foam.adjacencyList,
                std::function<entt::entity(int)>([&](int p_id) { return particleMap.at(p_id); })
            );

            std::unordered_map<int, glm::vec3> local_positions;
            std::unordered_map<int, float> masses;

            // Set Particle components
            for (const auto& [particleId, localPos] : foam.particlePosition) {
                auto particle = particleMap.at(particleId);

                particleRegistry.emplace<ParticleLocalPosition>(particle, localPos);
                particleRegistry.emplace<ParticleColor>(particle, foam.particleColor.at(particleId));
                particleRegistry.emplace<ParticleOpacity>(particle, foam.particleOpacity.at(particleId));
                
                float mass = foam.particleMass.at(particleId);
                particleRegistry.emplace<ParticleMass>(particle, mass);
                masses[particleId] = mass;
                const auto& p_verts = foam.particleVertices.at(particleId);
                particleRegistry.emplace<ParticleVertices>(particle, p_verts);

                glm::vec3 worldPos = glm::vec3(transform * glm::vec4(localPos, 1.0f));
                particleRegistry.emplace<ParticleWorldPosition>(particle, worldPos);
                local_positions[particleId] = localPos;

                if (foam.isStencil.at(particleId)) {
                    particleRegistry.emplace<Stencil>(particle);
                } else if (foam.isMutable.at(particleId)) {
                    particleRegistry.emplace<Mutable>(particle);
                } else {
                    particleRegistry.emplace<Immutable>(particle);
                }
            }

            // Build BVH for the foam's particles.
            buildBVH(rigidBody);

            // Define worldspace AABB
            buildAABB(rigidBody);

            // Find and emplace surface particles
            auto surface_cells = findSurfaceCells(foam.adjacencyList, foam.particleOpacity);
            for(const auto& p_id : surface_cells){
                particleRegistry.emplace<Surface>(particleMap.at(p_id));
            }
        }

        // Initialise the GPU slab (allocates device buffers, builds BVH and
        // CSR adjacency into slab slices, uploads static particle data).
        // Must be called after all foams have been fully constructed above.
        // Implemented in simulation_gpu.cu (requires nvcc / __CUDACC__).
        initSlab();
    }

    void Simulation::buildAABB(entt::entity foamEntity) {
        const auto& ordered = foamAdjacencyLists.at(static_cast<int>(foamEntity)).getOrderedNodeIds();
        std::vector<glm::vec3> world_positions;
        world_positions.reserve(ordered.size());
        for (auto e : ordered)
            world_positions.push_back(particleRegistry.get<ParticleWorldPosition>(e).value);
        if (!world_positions.empty()) {
            auto [min, max] = calculateAABB(world_positions);
            foamAABBs[static_cast<int>(foamEntity)] = AABB(min, max);
        }
    }

    // -------------------------------------------------------------------------
    // initSlab
    //
    // Allocates and populates the GpuSlabAllocator from the CPU-side with foam data
    // structures built during the constructor. This function is intentionally
    // in simulation.cpp (MSVC) so it can access entt registry members; the two
    // CUDA-kernel-guarded calls are delegated to free functions in
    // simulation_gpu.cu via the forward declarations at the top of this file.
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

            // Rebuild BVH into the slab slice.
            const auto& ordered = adj.getOrderedNodeIds();
            std::vector<AABB> ordered_aabbs;
            ordered_aabbs.reserve(N);
            for (auto e : ordered) {
                const auto& verts = particleRegistry.get<ParticleVertices>(e).vertices;
                auto [mn, mx] = calculateAABB(verts);
                ordered_aabbs.push_back(AABB(mn, mx));
            }
            foamBVHs.at(foam_id).build(
                ordered_aabbs.data(), N,
                gpuSlab.d_bvh_nodes + slot.bvh_offset,
                slot.bvh_capacity);

            // Build GPU CSR into slab slices and bias offsets.
            // Uses free-function wrappers in simulation_gpu.cu (nvcc-compiled)
            // to avoid entt/nvcc registry incompatibility in .cu files.
            buildAdjacencyIntoSlabSlice(
                adj,
                foamGpuAdj[foam_id],
                gpuSlab.d_csr_nbrs         + slot.csr_edge_offset,
                static_cast<size_t>(slot.csr_edge_capacity),
                gpuSlab.d_csr_node_offsets + slot.csr_node_offset,
                static_cast<size_t>(slot.csr_node_capacity));

            biasSlabCsrOffsets(gpuSlab, foam_id);

            // Upload static particle data (colors + surface mask) once.
            std::vector<glm::vec4> h_colors(N);
            std::vector<uint8_t>   h_mask(N, 0);
            for (int li = 0; li < N; ++li) {
                const entt::entity e = ordered[li];
                const auto* c = particleRegistry.try_get<ParticleColor>(e);
                const auto* o = particleRegistry.try_get<ParticleOpacity>(e);
                if (c && o) h_colors[li] = glm::vec4(c->rgb, o->value);
                if (particleRegistry.all_of<Surface>(e)) h_mask[li] = 1;
            }
            gpuSlab.uploadParticleColors(foam_id, h_colors.data(), N);
            gpuSlab.uploadSurfaceMask(foam_id, h_mask.data(), N);
        }
    }

    // -------------------------------------------------------------------------
    // rebuildAllSlabCsr
    //
    // Rebuilds the GPU CSR adjacency for every live foam directly into its
    // current slab slice, then re-biases the node_offsets.
    //
    // This must be called after gpuSlab.compact().  compact() D->D moves the
    // raw CSR node_offsets values but does not update the bias they carry: each
    // value was written as (0-based_offset + old_csr_edge_offset).  After
    // compaction csr_edge_offset has changed, so the values are wrong.
    // Rebuilding from the always-valid CPU adjacency lists corrects this in one
    // pass without any extra kernel.
    // -------------------------------------------------------------------------
    void Simulation::rebuildAllSlabCsr() {
        for (auto& [foam_id, adj] : foamAdjacencyLists) {
            auto it = gpuSlab.slots.find(foam_id);
            if (it == gpuSlab.slots.end() || it->second.dead) continue;
            const FoamSlot& slot = it->second;
            buildAdjacencyIntoSlabSlice(
                adj, foamGpuAdj[foam_id],
                gpuSlab.d_csr_nbrs         + slot.csr_edge_offset,
                static_cast<size_t>(slot.csr_edge_capacity),
                gpuSlab.d_csr_node_offsets + slot.csr_node_offset,
                static_cast<size_t>(slot.csr_node_capacity));
            biasSlabCsrOffsets(gpuSlab, foam_id);
        }
    }

    void Simulation::buildBVH(entt::entity foamEntity) {
        const int foam_id   = static_cast<int>(foamEntity);
        auto& adj           = foamAdjacencyLists.at(foam_id);
        const auto& ordered = adj.getOrderedNodeIds();
        const int N = static_cast<int>(ordered.size());
        const int E = static_cast<int>(adj.getCOOSrc().size());

        std::vector<AABB> ordered_aabbs;
        ordered_aabbs.reserve(N);
        for (auto e : ordered) {
            const auto& verts = particleRegistry.get<ParticleVertices>(e).vertices;
            auto [mn, mx] = calculateAABB(verts);
            ordered_aabbs.push_back(AABB(mn, mx));
        }

        auto it = gpuSlab.slots.find(foam_id);
        if (it != gpuSlab.slots.end() && !it->second.dead) {
            const int num_bvh_nodes = (N > 1) ? 2 * N - 1 : 1;

            // If particle count has grown beyond the 2x overcommit, tombstone the
            // old slot and allocate a larger one.  The CSR must also be rebuilt
            // into the new slice since the edge-block start offset has changed.
            if (gpuSlab.needsResize(foam_id, num_bvh_nodes, N + 1, E, N)) {
                gpuSlab.resize(foam_id, num_bvh_nodes, N + 1, E, N);
                const FoamSlot& newSlot = gpuSlab.slots.at(foam_id);
                buildAdjacencyIntoSlabSlice(
                    adj, foamGpuAdj[foam_id],
                    gpuSlab.d_csr_nbrs         + newSlot.csr_edge_offset,
                    static_cast<size_t>(newSlot.csr_edge_capacity),
                    gpuSlab.d_csr_node_offsets + newSlot.csr_node_offset,
                    static_cast<size_t>(newSlot.csr_node_capacity));
                biasSlabCsrOffsets(gpuSlab, foam_id);
            }

            const FoamSlot& slot = gpuSlab.slots.at(foam_id);
            foamBVHs[foam_id].build(ordered_aabbs.data(), N,
                                    gpuSlab.d_bvh_nodes + slot.bvh_offset,
                                    slot.bvh_capacity);

        } else {
            // Slab not yet initialised (first construction pass) — normal alloc.
            BVH bvh;
            bvh.build(ordered_aabbs.data(), ordered_aabbs.size());
            foamBVHs[foam_id] = std::move(bvh);
        }
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
        const auto allParticles = adjList.getOrderedNodeIds();

        // Build the update buffer: validate and use the subset if provided, otherwise use all particles.
        const std::unordered_set<entt::entity> foamParticleSet(allParticles.begin(), allParticles.end());
        const std::vector<entt::entity>* updateBuffer = &allParticles;
        std::vector<entt::entity> subsetBuffer;
        if (particleSubset.has_value()) {
            for (const auto& particle : *particleSubset) {
                if (!foamParticleSet.count(particle)) {
                    throw std::invalid_argument(
                        "Particle entity " + std::to_string(static_cast<uint32_t>(particle)) +
                        " does not belong to foam " + std::to_string(foamId)
                    );
                }
            }
            subsetBuffer.assign(particleSubset->begin(), particleSubset->end());
            updateBuffer = &subsetBuffer;
        }

        for (const auto& particle : *updateBuffer) {
            auto& worldPos = particleRegistry.get<ParticleWorldPosition>(particle);
            const auto& localPos = particleRegistry.get<ParticleLocalPosition>(particle);
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
            buildBVH(entity);   // keep narrowphase BVH in sync with particle world positions
            buildAABB(entity);  // keep broadphase AABB in sync with particle world positions
        });
    }

    void Simulation::updateTopology(
        const std::unordered_map<int, AABB>&                         foamAABBs,
        const std::unordered_map<int, BVH>&                          foamBVHs,
        std::unordered_map<int, AdjacencyList<entt::entity>>&        foamAdjacencyLists
    ) {
        const auto results = topologySubsystem.update(foamAABBs, foamBVHs, foamAdjacencyLists, foamRegistry, particleRegistry);
        for (const auto& result : results) {
            // Refresh world positions for the affected particles.
            const std::unordered_set<entt::entity> particleSubset(
                result.updatedParticles.begin(), result.updatedParticles.end());
            applyForwardKinematics(result.foamId, particleSubset);

            // Rebuild BVH from current ParticleVertices in the registry.
            buildBVH(result.foamId);

            // A parent foam snapshot indicates a new foam was spawned from a topology change.
            if (result.parentFoamSnapshot.has_value()) {
                const FoamSnapshot& snap = *result.parentFoamSnapshot;

                // Rebuild AABB of the parent using already-updated world positions.
                buildAABB(snap.foamId);

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

    void Simulation::updatePhysics(
        const std::unordered_map<int, AABB>&  foamAABBs,
        const std::unordered_map<int, BVH>&   foamBVHs,
        float deltaTime) {
        const auto updatedFoams = physicsSubsystem.update(foamAABBs, foamBVHs, foamAdjacencyLists, foamRegistry, particleRegistry, deltaTime);
        for (const auto foamEntity : updatedFoams)
            buildAABB(foamEntity);
    }

    void Simulation::render() {
        // Upload per-frame particle world positions to the slab.
        // Physics runs on the CPU, so positions change every frame for dynamic
        // foams.  Static foams' entries in the slab are overwritten with the
        // same data each frame, which is cheap (~few KB memcpy).
        for (const auto& [foam_id, adj] : foamAdjacencyLists) {
            const auto& ordered = adj.getOrderedNodeIds();
            std::vector<glm::vec3> h_positions;
            h_positions.reserve(ordered.size());
            for (auto e : ordered)
                h_positions.push_back(particleRegistry.get<ParticleWorldPosition>(e).value);
            gpuSlab.uploadParticlePositions(
                foam_id, h_positions.data(), static_cast<int>(h_positions.size()));
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
        renderSubsystem.update(foamAABBs, gpuSlab, foamTransforms,
                               camera_, windowSize_, overlayParams);
    }

    void Simulation::step(const UserInput& input, float deltaTime) {
        handleUserInput(input, deltaTime);
        updateTopology(foamAABBs, foamBVHs, foamAdjacencyLists);

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

        updatePhysics(foamAABBs, foamBVHs, deltaTime);
        // Re-sample the cursor position after the expensive subsystems so that
        // Controller foam is placed at its true on-screen location when the GPU
        // kernel fires, minimising input-to-display latency.
        applyControllerCursor(ImGui::GetIO().MousePos);
        render();
    }
};