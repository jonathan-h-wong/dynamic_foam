#include "dynamic_foam/Sim2D/sample_scene.h"

namespace DynamicFoam::Sim2D {
    Foam generateFoamBar(
        float width, 
        float height, 
        float depth,
        int widthParticles, 
        int heightParticles, 
        int depthParticles,
        float density, 
        const glm::vec3& color_param) {
        
        std::vector<uint32_t> particleIds;
        std::vector<glm::vec3> positions;
        std::unordered_map<uint32_t, bool> stencil;
        std::unordered_map<uint32_t, bool> mutable_map;
        std::unordered_map<uint32_t, glm::vec3> color;
        std::unordered_map<uint32_t, float> opacity;

        int paddedWidth = widthParticles + 2;
        int paddedHeight = heightParticles + 2;
        int paddedDepth = depthParticles + 2;
        float dx = width / (widthParticles - 1);
        float dy = height / (heightParticles - 1);
        float dz = depth / (depthParticles - 1);

        for (int i = 0; i < paddedWidth; ++i) {
            for (int j = 0; j < paddedHeight; ++j) {
                for (int k = 0; k < paddedDepth; ++k) {
                    int id = i * (paddedHeight * paddedDepth) + j * paddedDepth + k;
                    particleIds.push_back(static_cast<uint32_t>(id));
                    positions.push_back(glm::vec3(
                        (i - 1) * dx - width / 2.0f,
                        (j - 1) * dy - height / 2.0f,
                        (k - 1) * dz - depth / 2.0f
                    ));

                    bool isPadding = (i == 0 || i == paddedWidth - 1 || j == 0 || j == paddedHeight - 1 || k == 0 || k == paddedDepth - 1);

                    stencil[static_cast<uint32_t>(id)] = false;
                    mutable_map[static_cast<uint32_t>(id)] = true;
                    color[static_cast<uint32_t>(id)] = color_param;
                    opacity[static_cast<uint32_t>(id)] = isPadding ? 0.0f : 1.0f;
                }
            }
        }

        return Foam(particleIds, positions, stencil, mutable_map, color, opacity, density);
    }

    Foam generateFoamPointCursor() {
        std::vector<uint32_t> particleIds;
        std::vector<glm::vec3> positions;
        std::unordered_map<uint32_t, bool> stencil;
        std::unordered_map<uint32_t, bool> mutable_map;
        std::unordered_map<uint32_t, glm::vec3> color;
        std::unordered_map<uint32_t, float> opacity;

        // Central particle
        uint32_t centerId = 0;
        particleIds.push_back(centerId);
        positions.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
        opacity[centerId] = 1.0f;

        // Surrounding particles on a sphere
        int numSurrounding = 32;
        float radius = 0.2f;
        uint32_t id_counter = 1;

        // Use Fibonacci sphere algorithm for evenly distributed points
        float phi = glm::pi<float>() * (3.0f - sqrt(5.0f)); // Golden angle in radians

        for (int i = 0; i < numSurrounding; ++i) {
            float y = 1 - (i / static_cast<float>(numSurrounding - 1)) * 2; // y goes from 1 to -1
            float radius_at_y = sqrt(1 - y * y); // radius at y

            float theta = phi * i; // golden angle increment

            float x = cos(theta) * radius_at_y;
            float z = sin(theta) * radius_at_y;

            uint32_t id = id_counter++;
            particleIds.push_back(id);
            positions.push_back(glm::vec3(x * radius, y * radius, z * radius));
            opacity[id] = 0.0f; // Padding particles
        }

        for (const auto& id : particleIds) {
            stencil[id] = true;
            mutable_map[id] = false;
            color[id] = glm::vec3(1.0f); // White
        }

        return Foam(particleIds, positions, stencil, mutable_map, color, opacity, 1.0f);
    }

    Foam generateFoamFloorPlane(
        float widthX,
        float widthZ,
        int   xParticles,
        int   zParticles,
        float density,
        const glm::vec3& color_param)
    {
        std::vector<uint32_t>                       particleIds;
        std::vector<glm::vec3>                 positions;
        std::unordered_map<uint32_t, bool>          stencil;
        std::unordered_map<uint32_t, bool>          mutable_map;
        std::unordered_map<uint32_t, glm::vec3>     color;
        std::unordered_map<uint32_t, float>         opacity;

        // Padded grid: one ghost ring in XZ, plus one ghost cap above/below in Y.
        //   paddedX = xParticles + 2  (ghost at i=0 and i=paddedX-1)
        //   paddedZ = zParticles + 2  (ghost at k=0 and k=paddedZ-1)
        //   paddedY = 3               (j=0 ghost below, j=1 real layer at y=0,
        //                              j=2 ghost above)
        const int paddedX = xParticles + 2;
        const int paddedZ = zParticles + 2;
        const int paddedY = 3;

        const float dx = widthX / static_cast<float>(xParticles - 1);
        const float dz = widthZ / static_cast<float>(zParticles - 1);
        // Y ghost offset is the average in-plane spacing so the ghost caps sit
        // just one cell-width away from the main layer.
        const float dy = (dx + dz) * 0.5f;

        for (int i = 0; i < paddedX; ++i) {
            for (int j = 0; j < paddedY; ++j) {
                for (int k = 0; k < paddedZ; ++k) {
                    const uint32_t id = static_cast<uint32_t>(i * (paddedY * paddedZ) + j * paddedZ + k);
                    particleIds.push_back(id);

                    // (i-1) so that i=1 maps to x=-widthX/2, i=xParticles maps
                    // to x=+widthX/2.  Same convention as generateFoamBar.
                    const float x = static_cast<float>(i - 1) * dx - widthX * 0.5f;
                    const float y = static_cast<float>(j - 1) * dy; // j=0→-dy, j=1→0, j=2→+dy
                    const float z = static_cast<float>(k - 1) * dz - widthZ * 0.5f;
                    positions.push_back(glm::vec3(x, y, z));

                    const bool isGhost =
                        (i == 0 || i == paddedX - 1 ||
                         j == 0 || j == paddedY - 1 ||
                         k == 0 || k == paddedZ - 1);

                    stencil[id]     = false;       // non-stencil
                    mutable_map[id] = false;       // immutable
                    color[id]       = color_param; // grey
                    opacity[id]     = isGhost ? 0.0f : 1.0f;
                }
            }
        }

        return Foam(particleIds, positions, stencil, mutable_map, color, opacity, density);
    }

    SceneGraph createSampleSceneGraph() {
        std::unordered_map<int, Foam> foams;
        std::unordered_map<int, glm::mat4> transforms;
        std::unordered_map<int, bool> controller_map;
        std::unordered_map<int, bool> dynamic_map;

        foams[0] = generateFoamBar(1.0f, 0.5f, 0.2f, 10, 5, 2, 1.0f, glm::vec3(0.0f, 0.5f, 1.0f));
        transforms[0] = glm::mat4(1.0f); // Identity transform
        controller_map[0] = false; // Not a controller
        dynamic_map[0] = true; // Dynamic foam

        foams[1] = generateFoamPointCursor();
        transforms[1] = glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, 0.0f, 0.0f)); // Positioned to the right of the bar
        controller_map[1] = true; // Controller foam
        dynamic_map[1] = false; // Not dynamic

        // Floor plane — immobile, immutable, static, non-controller.
        // 20x20 interior particles centred at the origin at y=0.
        foams[2]          = generateFoamFloorPlane(
            10.0f, 10.0f,                    // 10 m × 10 m
            20, 20,                           // 20 × 20 interior centres
            1.0f,                             // density
            glm::vec3(0.5f, 0.5f, 0.5f));    // grey
        transforms[2]     = glm::mat4(1.0f); // identity — already at origin
        controller_map[2] = false;           // not a controller
        dynamic_map[2]    = false;           // static

        return SceneGraph(foams, transforms, controller_map, dynamic_map);
    }
}