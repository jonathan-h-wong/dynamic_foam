#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "dynamic_foam/sim2d/foam.h"
#include "dynamic_foam/sim2d/scenegraph.h"

namespace DynamicFoam::Sim2D {
    Foam generateFoamBar(
        int width, 
        int height, 
        int widthParticles, 
        int heightParticles, 
        float density, 
        const glm::vec3& color_param
    );
    Foam generateFoamPointCursor();
    SceneGraph createSampleSceneGraph();
}