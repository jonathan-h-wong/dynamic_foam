#include "dynamic_foam/Sim2D/components.h"
#include "dynamic_foam/Sim2D/simulation.h"
#include "dynamic_foam/Sim2D/utils.h"
#include <entt/entt.hpp>

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

    void Simulation::buildBVH(entt::entity foamEntity) {
        const auto& ordered = foamAdjacencyLists.at(static_cast<int>(foamEntity)).getOrderedNodeIds();
        std::vector<AABB> ordered_aabbs;
        ordered_aabbs.reserve(ordered.size());
        for (auto e : ordered) {
            const auto& verts = particleRegistry.get<ParticleVertices>(e).value;
            auto [min, max] = calculateAABB(verts);
            ordered_aabbs.push_back(AABB(min, max));
        }
        BVH bvh;
        bvh.build(ordered_aabbs.data(), ordered_aabbs.size());
        foamBVHs[static_cast<int>(foamEntity)] = std::move(bvh);
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

    void Simulation::handleUserInput(const UserInput& input) {
        // Convert from ImGui screen coordinates (top-left origin) to world coordinates (center origin)
        float world_x = input.mouse_pos.x - windowSize_.x / 2.0f;
        float world_y = -(input.mouse_pos.y - windowSize_.y / 2.0f); // Flip y-axis

        foamRegistry.view<Controller, Position>().each([&](auto entity, auto& pos) {
            pos.value = glm::vec3(world_x, world_y, 0.0f);
            applyForwardKinematics(entity);
        });
    }

    void Simulation::updateTopology(
        entt::registry& foamRegistry,
        entt::registry& particleRegistry,
        std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists
    ) {
        const auto results = topologySubsystem.update(foamRegistry, particleRegistry, foamAdjacencyLists);
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
        entt::registry& foamRegistry,
        entt::registry& particleRegistry,
        float deltaTime) {
        const auto updatedFoams = physicsSubsystem.update(foamRegistry, particleRegistry, foamAdjacencyLists, deltaTime);
        for (const auto foamEntity : updatedFoams)
            buildAABB(foamEntity);
    }

    void Simulation::render(
        const entt::registry& particleRegistry,
        const std::unordered_map<int, AdjacencyList<entt::entity>>& foamAdjacencyLists,
        const std::unordered_map<int, BVH>& foamBVHs,
        const std::unordered_map<int, AABB>& foamAABBs
    ) {
        // Build per-foam world transforms from position and orientation.
        // These are passed to the render subsystem so it can compute inverse
        // transforms for transforming rays into each foam's local BVH space.
        std::unordered_map<int, glm::mat4> foamTransforms;
        foamRegistry.view<const Position, const Orientation>().each(
            [&](entt::entity e, const Position& pos, const Orientation& orient) {
                const glm::mat4 t = glm::translate(glm::mat4(1.0f), pos.value);
                const glm::mat4 r = glm::mat4_cast(orient.value);
                foamTransforms[static_cast<int>(e)] = t * r;
            });

        OrthographicCamera camera;
        camera.origin = glm::vec3(0.0f, 0.0f, -5.0f);
        camera.lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
        camera.up = glm::vec3(0.0f, 1.0f, 0.0f);
        camera.width = windowSize_.x;
        camera.height = windowSize_.y;
        renderSubsystem.update(particleRegistry, foamAdjacencyLists, foamBVHs, foamAABBs, foamTransforms, camera, windowSize_);
    }

    void Simulation::step(const UserInput& input, float deltaTime) {
        handleUserInput(input);
        updateTopology(foamRegistry, particleRegistry, foamAdjacencyLists);
        updatePhysics(foamRegistry, particleRegistry, deltaTime);
        render(particleRegistry, foamAdjacencyLists, foamBVHs, foamAABBs);
    }
};