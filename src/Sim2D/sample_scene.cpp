#include "dynamic_foam/sim2d/sample_scene.h"
#include "dynamic_foam/sim2d/utils.h"

namespace DynamicFoam::Sim2D {
    Foam generateFoamBar(
        int width, 
        int height, 
        int widthParticles, 
        int heightParticles, 
        float density, 
        const glm::vec3& color_param) {
        
        std::vector<int> particleIds;
        std::vector<glm::vec2> positions_vec2;
        std::unordered_map<int, glm::vec3> position;
        std::unordered_map<int, bool> stencil;
        std::unordered_map<int, bool> mutable_map;
        std::unordered_map<int, glm::vec3> mass;
        std::unordered_map<int, glm::vec3> color;
        std::unordered_map<int, float> opacity;

        int paddedWidth = widthParticles + 2;
        int paddedHeight = heightParticles + 2;
        float dx = static_cast<float>(width) / (widthParticles - 1);
        float dy = static_cast<float>(height) / (heightParticles - 1);

        for (int i = 0; i < paddedWidth; ++i) {
            for (int j = 0; j < paddedHeight; ++j) {
                int id = i * paddedHeight + j;
                particleIds.push_back(id);

                float x = (i - 1) * dx - width / 2.0f;
                float y = (j - 1) * dy - height / 2.0f;
                position[id] = glm::vec3(x, y, 0.0f);
                positions_vec2.push_back(glm::vec2(x, y));

                bool isPadding = (i == 0 || i == paddedWidth - 1 || j == 0 || j == paddedHeight - 1);
                
                stencil[id] = false;
                mutable_map[id] = true;
                color[id] = color_param;
                opacity[id] = isPadding ? 0.0f : 1.0f;
            }
        }

        auto [adjList, areaMap] = triangulateWithAreaIntegration(positions_vec2, particleIds);

        for (const auto& id : particleIds) {
            mass[id] = glm::vec3(areaMap[id] * density);
        }

        return Foam(adjList, stencil, mutable_map, position, mass, color, opacity, density);
    }

    Foam generateFoamPointCursor() {
        std::vector<int> particleIds;
        std::vector<glm::vec2> positions_vec2;
        std::unordered_map<int, glm::vec3> position;
        std::unordered_map<int, bool> stencil;
        std::unordered_map<int, bool> mutable_map;
        std::unordered_map<int, glm::vec3> mass;
        std::unordered_map<int, glm::vec3> color;
        std::unordered_map<int, float> opacity;

        // Central particle
        int centerId = 0;
        particleIds.push_back(centerId);
        position[centerId] = glm::vec3(0.0f, 0.0f, 0.0f);
        positions_vec2.push_back(glm::vec2(0.0f, 0.0f));
        opacity[centerId] = 1.0f;

        // Surrounding particles
        int numSurrounding = 8;
        float radius = 0.2f;
        for (int i = 0; i < numSurrounding; ++i) {
            int id = i + 1;
            particleIds.push_back(id);
            float angle = 2.0f * glm::pi<float>() * static_cast<float>(i) / numSurrounding;
            float x = radius * cos(angle);
            float y = radius * sin(angle);
            position[id] = glm::vec3(x, y, 0.0f);
            positions_vec2.push_back(glm::vec2(x, y));
            opacity[id] = 0.0f; // Padding particles
        }

        AdjacencyList<int> adjList = triangulate(positions_vec2, particleIds);

        for (const auto& id : particleIds) {
            stencil[id] = true;
            mutable_map[id] = false;
            mass[id] = glm::vec3(1.0f);
            color[id] = glm::vec3(1.0f); // White
        }

        return Foam(adjList, stencil, mutable_map, position, mass, color, opacity, 1.0f);
    }

    SceneGraph createSampleSceneGraph() {
        std::unordered_map<int, Foam> foams;
        std::unordered_map<int, glm::mat4> transforms;
        std::unordered_map<int, bool> controller_map;
        std::unordered_map<int, bool> dynamic_map;

        foams[0] = createFoamBar(1.0f, 0.5f, 10, 5, 1.0f, glm::vec3(0.0f, 0.5f, 1.0f));
        transforms[0] = glm::mat4(1.0f); // Identity transform
        controller_map[0] = false; // Not a controller
        dynamic_map[0] = true; // Dynamic foam

        foams[1] = createFoamPointCursor();
        transforms[1] = glm::translate(glm::mat4(1.0f), glm::vec3(0.5f, 0.0f, 0.0f)); // Positioned to the right of the bar
        controller_map[1] = true; // Controller foam
        dynamic_map[1] = false; // Not dynamic

        return SceneGraph(foams, transforms, controller_map, dynamic_map);
    }
}