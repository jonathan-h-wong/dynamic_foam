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
        
        std::vector<int> particleIds;
        std::vector<glm::vec3> positions;
        std::unordered_map<int, bool> stencil;
        std::unordered_map<int, bool> mutable_map;
        std::unordered_map<int, glm::vec3> color;
        std::unordered_map<int, float> opacity;

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
                    particleIds.push_back(id);
                    positions.push_back(glm::vec3(
                        (i - 1) * dx - width / 2.0f,
                        (j - 1) * dy - height / 2.0f,
                        (k - 1) * dz - depth / 2.0f
                    ));

                    bool isPadding = (i == 0 || i == paddedWidth - 1 || j == 0 || j == paddedHeight - 1 || k == 0 || k == paddedDepth - 1);

                    stencil[id] = false;
                    mutable_map[id] = true;
                    color[id] = color_param;
                    opacity[id] = isPadding ? 0.0f : 1.0f;
                }
            }
        }

        return Foam(particleIds, positions, stencil, mutable_map, color, opacity, density);
    }

    Foam generateFoamPointCursor() {
        std::vector<int> particleIds;
        std::vector<glm::vec3> positions;
        std::unordered_map<int, bool> stencil;
        std::unordered_map<int, bool> mutable_map;
        std::unordered_map<int, glm::vec3> color;
        std::unordered_map<int, float> opacity;

        // Central particle
        int centerId = 0;
        particleIds.push_back(centerId);
        positions.push_back(glm::vec3(0.0f, 0.0f, 0.0f));
        opacity[centerId] = 1.0f;

        // Surrounding particles on a sphere
        int numSurrounding = 32;
        float radius = 0.2f;
        int id_counter = 1;

        // Use Fibonacci sphere algorithm for evenly distributed points
        float phi = glm::pi<float>() * (3.0f - sqrt(5.0f)); // Golden angle in radians

        for (int i = 0; i < numSurrounding; ++i) {
            float y = 1 - (i / static_cast<float>(numSurrounding - 1)) * 2; // y goes from 1 to -1
            float radius_at_y = sqrt(1 - y * y); // radius at y

            float theta = phi * i; // golden angle increment

            float x = cos(theta) * radius_at_y;
            float z = sin(theta) * radius_at_y;

            int id = id_counter++;
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

        return SceneGraph(foams, transforms, controller_map, dynamic_map);
    }
}