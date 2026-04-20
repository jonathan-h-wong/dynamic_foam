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
        // Default constructor — needed for use as std::unordered_map value type
        Foam() = default;

        // Topology
        AdjacencyList adjacencyList;
        std::unordered_map<uint32_t, bool> isStencil;
        std::unordered_map<uint32_t, bool> isMutable;

        // Local Space Geometry
        std::unordered_map<uint32_t, glm::vec3> particlePosition;
        std::unordered_map<uint32_t, std::vector<glm::vec3>> particleVertices;

        // Physics
        float density;
        std::unordered_map<uint32_t, float> particleMass;
        glm::mat3 intertiaTensor;

        // Rendering
        std::unordered_map<uint32_t, glm::vec3> particleColor;
        std::unordered_map<uint32_t, float> particleOpacity;

#ifndef __CUDACC__
        Foam(
            const std::vector<uint32_t>& particleIds,
            const std::vector<glm::vec3>& positions,
            const std::unordered_map<uint32_t, bool>& stencil,
            const std::unordered_map<uint32_t, bool>& mutable_map,
            const std::unordered_map<uint32_t, glm::vec3>& color,
            const std::unordered_map<uint32_t, float>& opacity,
            float density = 1.0f
        ) : isStencil(stencil),
            isMutable(mutable_map),
            particleColor(color),
            particleOpacity(opacity),
            density(density) {
            // Triangulate in original space (translation-invariant topology)
            auto [adjList, voronoiVertices] = triangulateVoronoiCells(positions, particleIds);
            adjacencyList = adjList;
            particleVertices = voronoiVertices;

            // Build position and mass maps.
            // Mass is approximated from the AABB of the Voronoi cell vertices
            // (volume = AABB width * height * depth * density).  Boundary
            // particles (empty vertex list) get zero mass.
            for (size_t i = 0; i < particleIds.size(); ++i) {
                uint32_t id = particleIds[i];
                particlePosition[id] = positions[i];

                const auto& verts = voronoiVertices[id];
                if (!verts.empty()) {
                    glm::vec3 lo = verts[0], hi = verts[0];
                    for (const auto& v : verts) {
                        lo = glm::min(lo, v);
                        hi = glm::max(hi, v);
                    }
                    glm::vec3 ext = hi - lo;
                    particleMass[id] = ext.x * ext.y * ext.z * density;
                } else {
                    particleMass[id] = 0.0f;
                }
            }

            validate();
            parentToCenterOfMass();
            intertiaTensor = calculateInertiaTensor(particlePosition, particleMass);
        }
#endif // !__CUDACC__

    private:
        void parentToCenterOfMass() {
            glm::vec3 com = calculateCenterOfMass(particlePosition, particleMass);
            for (auto& [id, pos] : particlePosition) {
                pos -= com;
            }
            // Shift Voronoi vertices into CoM-local space.
            for (auto& [id, verts] : particleVertices) {
                for (auto& v : verts) {
                    v -= com;
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