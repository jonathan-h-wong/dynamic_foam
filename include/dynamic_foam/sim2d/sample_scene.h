#pragma once

#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "dynamic_foam/Sim2D/foam.h"
#include "dynamic_foam/Sim2D/scenegraph.h"

namespace DynamicFoam::Sim2D {
    Foam generateFoamBar(
        float width,
        float height,
        float depth,
        int widthParticles,
        int heightParticles,
        int depthParticles,
        float density,
        const glm::vec3& color_param
    );
    Foam generateFoamPointCursor();
    SceneGraph createSampleSceneGraph();
}