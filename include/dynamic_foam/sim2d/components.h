#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace DynamicFoam::Sim2D {
    // ============================================================================
    // Rigid Body Components
    // ============================================================================

    // Persistent Components
    // Initalized once per foam
    struct InertiaTensor {
        glm::mat3 value{1.0f}; // Identity matrix by default
    };
    struct Density {
        float value{1.0f};
    };
    
    // Persistent Foam Types
    // Initialzed once per foam
    struct Static {};
    struct Dynamic {};
    struct Controller {};

    // Transient Components
    struct Position {
        glm::vec3 value{0.0f};
    };
    struct Velocity {
        glm::vec3 value{0.0f};
    };
    struct Orientation {
        glm::quat value{1.0f, 0.0f, 0.0f, 0.0f}; // Identity quaternion (w, x, y, z)
    };
    struct AngularVelocity {
        glm::vec3 value{0.0f};
    };

    // ============================================================================
    // Particle Components
    // ============================================================================

    // Persistent Components
    // Initalized once per particle
    struct ParticleColor {
        glm::vec3 rgb{1.0f}; // White by default
    };
    struct ParticleOpacity {
        float value{1.0f};
    };
    struct ParticleMass {
        float value{1.0f};
    };
    // Local-space vertices of the Voronoi cell; used as the collision primitive for GJK-EPA.
    struct ParticleVertices {
        std::vector<glm::vec3> vertices;
    };
    struct Surface {};

    // Persistent Particle Types
    // Initialized once per particle
    struct Stencil {};
    struct Mutable {};
    struct Immutable {};

    // Transient Components
    struct ParticleLocalPosition {
        glm::vec3 value{0.0f};
    };
    struct ParticleWorldPosition {
        glm::vec3 value{0.0f};
    };

}