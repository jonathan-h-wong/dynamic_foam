#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace dynamic_foam::sim2d {

// ============================================================================
// Rigid Body Components
// ============================================================================

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

struct CenterOfMass {
    glm::vec3 value{0.0f};
};

struct InertiaTensor {
    glm::mat3 value{1.0f}; // Identity matrix by default
};


// ============================================================================
// Particle Components
// ============================================================================

struct ParticlePosition {
    glm::vec3 value{0.0f};
};

struct ParticleVelocity {
    glm::vec3 value{0.0f};
};

struct ParticleColor {
    glm::vec3 rgb{1.0f}; // White by default
};

struct ParticleOpacity {
    float value{1.0f};
};

struct ParticleMass {
    float value{1.0f};
};

struct IsBlade {
    bool value{true};
};

struct IsCuttable {
    bool value{true};
};

} // namespace dynamic_foam::sim2d
