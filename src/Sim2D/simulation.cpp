#include <dynamic_foam/sim2d/components.h>
#include <dynamic_foam/sim2d/simulation.h>
#include <dynamic_foam/sim2d/utils.h>
#include <entt/entt.hpp>

namespace DynamicFoam::Sim2D {
    Simulation::Simulation(
        const SceneGraph& sceneGraph, 
        const UserInput& input,
        const glm::ivec2& windowSize
    ) {
        // Creates rigid body and particle registries
        // For each foam, this creates 1 rigid body and N particles, and populates registries
        for (const auto& [foamId, foam] : sceneGraph.foams) {
            auto rigidBody = foamRegistry.create();

            // Set rigid body's position from the scene graph
            const auto& transform = sceneGraph.worldTransforms.at(foamId);
            foamRegistry.emplace<Position>(rigidBody, glm::vec3(transform[3]));

            // Set default transient components
            foamRegistry.emplace<Velocity>(rigidBody, glm::vec3(0.0f));
            foamRegistry.emplace<Orientation>(rigidBody, glm::quat(1.0f, 0.0f, 0.0f, 0.0f));
            foamRegistry.emplace<AngularVelocity>(rigidBody, glm::vec3(0.0f));

            // Set foam type
            if (sceneGraph.isController.at(foamId)) {
                foamRegistry.emplace<Controller>(rigidBody);
            } else if (sceneGraph.isDynamic.at(foamId)) {
                foamRegistry.emplace<Dynamic>(rigidBody);
            } else {
                foamRegistry.emplace<Static>(rigidBody);
            }

            // Set density
            foamRegistry.emplace<Density>(rigidBody, foam.density);

            std::unordered_map<int, entt::entity> particleMap;
            std::vector<glm::vec3> world_positions;
            std::unordered_map<int, glm::vec3> local_positions;
            std::unordered_map<int, float> masses;

            for (const auto& [particleId, localPos] : foam.particlePosition) {
                auto particle = particleRegistry.create();
                particleMap[particleId] = particle;

                particleRegistry.emplace<ParticleLocalPosition>(particle, localPos);
                particleRegistry.emplace<ParticleColor>(particle, foam.particleColor.at(particleId));
                particleRegistry.emplace<ParticleOpacity>(particle, foam.particleOpacity.at(particleId));
                
                // Assuming mass is stored in the x-component of particleMass
                float mass = foam.particleMass.at(particleId).x;
                particleRegistry.emplace<ParticleMass>(particle, mass);
                masses[particleId] = mass;

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

            // Calculate and emplace CenterOfMass, InertiaTensor, and AABB
            glm::vec3 com = calculateCenterOfMass(local_positions, masses);
            foamRegistry.emplace<CenterOfMass>(rigidBody, com);

            std::unordered_map<int, glm::vec3> local_positions_com;
            for(const auto& [id, pos] : local_positions){
                local_positions_com[id] = pos - com;
            }
            glm::mat3 inertia = calculateInertiaTensor(local_positions_com, masses);
            foamRegistry.emplace<InertiaTensor>(rigidBody, inertia);

            auto [min, max] = calculateAABB(world_positions);
            foamRegistry.emplace<AABB>(rigidBody, min, max);

            // Find and emplace surface particles
            auto surface_cells = findSurfaceCells(foam.adjacencyList, foam.particleOpacity);
            for(const auto& p_id : surface_cells){
                particleRegistry.emplace<Surface>(particleMap.at(p_id));
            }
        }
    }

    void Simulation::handleUserInput(const UserInput& input) {
        // Process user input and update simulation state accordingly
    }

    void Simulation::updateTopology() {
        topologySubsystem.update(foamRegistry, particleRegistry, foamAdjacencyLists);
    }

    void Simulation::updatePhysics(float deltaTime) {
        physicsSubsystem.update(foamRegistry, particleRegistry, foamAdjacencyLists, deltaTime);
    }

    void Simulation::render() {
        renderSubsystem.update(foamRegistry, particleRegistry, foamAdjacencyLists);
    }

    void Simulation::step(float deltaTime) {
        handleUserInput(/* current user input */);
        updateTopology();
        updatePhysics(deltaTime);
        render();
    }
};