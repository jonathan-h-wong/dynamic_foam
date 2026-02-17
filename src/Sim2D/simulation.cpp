#include <dynamic_foam/sim2d/components.h>
#include <dynamic_foam/sim2d/simulation.h>

namespace DynamicFoam::Sim2D {
    Simulation::Simulation(
        const SceneGraph& sceneGraph, 
        const UserInput& input,
        const glm::ivec2& windowSize
    ) {
        // Initialize simulation with user input and scene graph
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