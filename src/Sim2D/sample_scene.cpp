# include "dynamic_foam/sim2d/sample_scene.h"

namespace DynamicFoam::Sim2D {
    Foam generateFoamBar(
        int width, 
        int height, 
        int widthParticles, 
        int heightParticles, 
        float density, 
        const glm::vec3& color) {
        // Example foam generation logic
        AdjacencyList<int> adjList;
        std::unordered_map<int, bool> stencil;
        std::unordered_map<int, bool> mutable_map;
        std::unordered_map<int, glm::vec3> position;
        std::unordered_map<int, glm::vec3> mass;
        std::unordered_map<int, glm::vec3> color;
        std::unordered_map<int, float> opacity;

        // TODO
        // Create width/height grid of particles centered on origin. 

        // Stencil map is entirely false
        // Mutable map is entirely true
        // Position map is grid layout
        // TEMP: Mass map is uniform  (Write MC Integration utility)
        // Color map is enitrely the input color
        // Opacity map is entirely 1.0f

        // Okay, I believe I need to write the MC utility first
        // That is the workhorse for topology subsystem anyway
        // Essentially, pass in particle positions to DT. 
        
        // triangulate()
        // I: Particle positions
        // O: AdjList
        
        // triangulateWithAreaIntegration()
        // I: Particle positions
        // O: AdjList, vol map

        return Foam(adjList, stencil, mutable_map, position, mass, color, opacity, density);
    }

    Foam generateFoamPointCursor() {
        // Example foam generation logic
        AdjacencyList<int> adjList;
        std::unordered_map<int, bool> stencil;
        std::unordered_map<int, bool> mutable_map;
        std::unordered_map<int, glm::vec3> position;
        std::unordered_map<int, glm::vec3> mass;
        std::unordered_map<int, glm::vec3> color;
        std::unordered_map<int, float> opacity;

        // Populate the above data structures with foam data
        
        // TODO
        // Create single particle at origin surrounded by particles (6-8). 
        
        // Stencil map is entirely true
        // Mutable map is entirely false
        // Position map is radial layout
        // Mass is negligible-- can be entirely 1.0f
        // Color map is entirely white
        // Opacity map is entirely 1.0f

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