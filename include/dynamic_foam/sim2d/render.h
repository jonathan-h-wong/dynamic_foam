#pragma once
#include <entt/entt.hpp>

namespace DynamicFoam::Sim2D {

class Render {
    public:
        Render() = default;
        ~Render() = default;

        void update(
            entt::registry& foamRegistry,
            entt::registry& particleRegistry
        );
    };
}
