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

    /**
     * @brief Generate a flat foam floor plane.
     *
     * Produces a 100x100 grid of interior particles centred at the origin in
     * the XZ plane (y = 0), surrounded by a single ring of ghost (opacity = 0)
     * boundary particles in XZ, plus a ghost cap layer immediately above
     * (y = +dy) and below (y = -dy) to give the 3-D Delaunay triangulation
     * proper convex extent and avoid degenerate coplanar tetrahedra.
     *
     * All particles are non-stencil and immutable; the foam is registered in
     * the SceneGraph as non-controller and static.
     *
     * @param widthX         Total span of the plane along X (default 10 m).
     * @param widthZ         Total span of the plane along Z (default 10 m).
     * @param xParticles     Interior particle count along X (default 100).
     * @param zParticles     Interior particle count along Z (default 100).
     * @param density        Foam density (default 1.0).
     * @param color_param    RGB colour (default grey).
     */
    Foam generateFoamFloorPlane(
        float widthX       = 10.0f,
        float widthZ       = 10.0f,
        int   xParticles   = 20,
        int   zParticles   = 20,
        float density      = 1.0f,
        const glm::vec3& color_param = glm::vec3(0.5f, 0.5f, 0.5f)
    );

    SceneGraph createSampleSceneGraph();
}