#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace DynamicFoam::Sim2D {
    // ============================================================================
    // Rigid Body Components
    // ============================================================================

    // Persistent Components
    struct CenterOfMass {
        glm::vec3 value{0.0f};
    };
    struct InertiaTensor {
        glm::mat3 value{1.0f}; // Identity matrix by default
    };

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
    struct AABB {
        glm::vec3 min{0.0f};
        glm::vec3 max{0.0f};
    };

    // Persistent Foam Types
    struct Static {};
    struct Dynamic {};
    struct Controller {};

    // ============================================================================
    // Particle Components
    // ============================================================================

    // Persistent Components
    struct ParticleLocalPosition {
        glm::vec3 value{0.0f};
    };
    struct ParticleColor {
        glm::vec3 rgb{1.0f}; // White by default
    };
    struct ParticleOpacity {
        float value{1.0f};
    };

    // Transient Components
    struct ParticleWorldPosition {
        glm::vec3 value{0.0f};
    };
    struct ParticleMass {
        float value{1.0f};
    };
    struct Surface {};

    // Persistent Particle Types
    struct Stencil {};
    struct Mutable {};
    struct Immutable {};

}