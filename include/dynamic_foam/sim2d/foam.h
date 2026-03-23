#pragma once

#include <glm/glm.hpp>
#include <stdexcept>
#include <unordered_map>
#include <vector>
#include "dynamic_foam/Sim2D/adjacency.cuh"
#include "dynamic_foam/Sim2D/utils.h"
#include "dynamic_foam/Sim2D/triangulation.h"

namespace DynamicFoam::Sim2D {
    class Foam {
    public:
        // Topology
        AdjacencyList<int> adjacencyList;
        std::unordered_map<int, bool> isStencil;
        std::unordered_map<int, bool> isMutable;

        // Local Space Geometry
        std::unordered_map<int, glm::vec3> particlePosition;
        std::unordered_map<int, ConvexPolytope> particlePolytopes;

        // Physics
        float density;
        std::unordered_map<int, float> particleMass;
        glm::mat3 intertiaTensor;

        // Rendering
        std::unordered_map<int, glm::vec3> particleColor;
        std::unordered_map<int, float> particleOpacity;

        Foam(
            const std::vector<int>& particleIds,
            const std::vector<glm::vec3>& positions,
            const std::unordered_map<int, bool>& stencil,
            const std::unordered_map<int, bool>& mutable_map,
            const std::unordered_map<int, glm::vec3>& color,
            const std::unordered_map<int, float>& opacity,
            float density = 1.0f
        ) : isStencil(stencil),
            isMutable(mutable_map),
            particleColor(color),
            particleOpacity(opacity),
            density(density) {
            // Triangulate in original space (translation-invariant topology/volumes)
            auto [adjList, volumeMap, voronoiPolytopes] = triangulateVoronoiCells(positions, particleIds);
            adjacencyList = adjList;
            particlePolytopes = voronoiPolytopes;

            // Build position and mass maps
            for (size_t i = 0; i < particleIds.size(); ++i) {
                int id = particleIds[i];
                particlePosition[id] = positions[i];
                particleMass[id] = (volumeMap.count(id) && volumeMap.at(id) > 0.0f)
                    ? volumeMap.at(id) * density
                    : 0.0f;
            }

            validate();
            parentToCenterOfMass();
            intertiaTensor = calculateInertiaTensor(particlePosition, particleMass);
        }

    private:
        void parentToCenterOfMass() {
            glm::vec3 com = calculateCenterOfMass(particlePosition, particleMass);
            for (auto& [id, pos] : particlePosition) {
                pos -= com;
            }
            // Shift voronoi vertices so they are also in CoM-local space.
            // Normals are direction vectors and are unaffected by translation.
            // Distances encode dot(normal, point_on_plane), so shifting every
            // vertex by -com reduces each face distance to: d -= dot(normal, com).
            for (auto& [id, polytope] : particlePolytopes) {
                for (auto& v : polytope.vertices) {
                    v -= com;
                }
                for (auto& face : polytope.faces) {
                    face.distance -= glm::dot(face.normal, com);
                }
            }
        }

        /*
        Validation logic for foam types based on their stencil and mutable properties:
        Stencil | Mutable
        Y        Y       Not allowed, validated here
        Y        N       Default stencil
        N        Y       Default subject foam
        N        N       Default background foam
        */
        void validate() {
            for (const auto& [id, stencil] : isStencil) {
                if (stencil) {
                    auto mutableIt = isMutable.find(id);
                    if (mutableIt != isMutable.end() && mutableIt->second) {
                        throw std::runtime_error("A particle cannot be both a stencil and mutable.");
                    }
                }
            }
        }
    };
}