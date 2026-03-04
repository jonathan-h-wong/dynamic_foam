#include <dynamic_foam/sim2d/components.h>
#include <dynamic_foam/sim2d/simulation.h>
#include <dynamic_foam/sim2d/utils.h>
#include <entt/entt.hpp>

namespace DynamicFoam::Sim2D {
    Simulation::Simulation(
        const SceneGraph& sceneGraph, 
        const glm::ivec2& windowSize
    ) : windowSize(windowSize) {
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
                [&](int p_id) { return particleMap.at(p_id); }
            );

            std::vector<glm::vec3> world_positions;
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
                particleRegistry.emplace<ParticleVertices>(particle, foam.particleVertices.at(particleId));

                glm::vec3 worldPos = glm::vec3(transform * glm::vec4(localPos, 1.0f));
                particleRegistry.emplace<ParticleWorldPosition>(particle, worldPos);
                world_positions.push_back(worldPos);
                local_positions[particleId] = localPos;

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

            // Set RB AABB (worldspace)
            auto [min, max] = calculateAABB(world_positions);
            foamRegistry.emplace<AABB>(rigidBody, min, max);
        }
    }

    void Simulation::applyForwardKinematics(entt::entity controllerFoam) {
        const auto& pos = foamRegistry.get<Position>(controllerFoam);
        const auto& orient = foamRegistry.get<Orientation>(controllerFoam);

        glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), pos.value);
        glm::mat4 rotationMatrix = glm::mat4_cast(orient.value);
        glm::mat4 transform = translationMatrix * rotationMatrix;

        auto foamId = static_cast<int>(controllerFoam);
        if (foamAdjacencyLists.count(foamId)) {
            const auto& adjList = foamAdjacencyLists.at(foamId).getAdjList();
            for (const auto& [particleEntity, neighbors] : adjList) {
                auto& worldPos = particleRegistry.get<ParticleWorldPosition>(particleEntity);
                const auto& localPos = particleRegistry.get<ParticleLocalPosition>(particleEntity);
                worldPos.value = glm::vec3(transform * glm::vec4(localPos.value, 1.0f));
            }
        }
    }

    void Simulation::handleUserInput(const UserInput& input) {
        // Convert from ImGui screen coordinates (top-left origin) to world coordinates (center origin)
        float world_x = input.mouse_pos.x - windowSize.x / 2.0f;
        float world_y = -(input.mouse_pos.y - windowSize.y / 2.0f); // Flip y-axis

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
        topologySubsystem.update(foamRegistry, particleRegistry, foamAdjacencyLists);
    }

    void Simulation::updatePhysics(
        entt::registry& foamRegistry,
        entt::registry& particleRegistry,
        float deltaTime) {
        physicsSubsystem.update(foamRegistry, particleRegistry, foamAdjacencyLists, deltaTime);
    }

    void Simulation::render(
        entt::registry& foamRegistry,
        entt::registry& particleRegistry
    ) {
        renderSubsystem.update(foamRegistry, particleRegistry, foamAdjacencyLists);
    }

    void Simulation::step(const UserInput& input, float deltaTime) {
        handleUserInput(input);
        updateTopology(foamRegistry, particleRegistry, foamAdjacencyLists);
        updatePhysics(foamRegistry, particleRegistry, deltaTime);
        render(foamRegistry, particleRegistry);
    }
};