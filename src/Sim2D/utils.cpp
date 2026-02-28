// TODO: 
// Add position + quaternion => 4x4 matrix conversion utilities

#pragma once
#include <random>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>

#include "dynamic_foam/sim2d/utils.h"
#include "dynamic_foam/sim2d/adjacency.h"

namespace DynamicFoam::Sim2D {

/**
 * Compute the signed area of a triangle in 2D
 */
inline float signedTriangleArea(
    float ax, float ay,
    float bx, float by,
    float cx, float cy
) {
    return 0.5f * ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay));
}

/**
 * Triangulate with analytical Voronoi area integration via circumcenter fan
 *
 * For each particle, collects the ordered ring of circumcenters of incident
 * Delaunay faces (= Voronoi cell vertices), then sums signed triangle areas
 * formed with the generator point as the fan origin.
 *
 * Boundary particles (with infinite incident faces) are assigned area = -1
 * to signal that their cell is unbounded. Clip to a domain first if you
 * need areas for those.
 *
 * @param positions:   Particle positions (.x, .y)
 * @param particleIds: Particle IDs (must be 1-to-1 with positions)
 * @return Pair of (AdjacencyList, area map).
 *         Unbounded cells have area -1.
 */
template <typename T, typename Vec2>
std::tuple<
    AdjacencyList<T>,
    std::unordered_map<T, float>,
    std::unordered_map<T, std::vector<glm::vec2>>
>
triangulateWithMetadata(
    const std::vector<Vec2>& positions,
    const std::vector<T>&    particleIds
) {
    size_t numParticles = particleIds.size();

    // ------------------------------------------------------------------ //
    // 1. Build Delaunay triangulation, keep vertex handles                //
    // ------------------------------------------------------------------ //
    Delaunay_2 dt;
    std::unordered_map<Delaunay_2::Vertex_handle, size_t> vertexToIndex;

    for (size_t i = 0; i < numParticles; ++i) {
        auto vh = dt.insert(Point_2(positions[i].x, positions[i].y));
        vertexToIndex[vh] = i;
    }

    // ------------------------------------------------------------------ //
    // 2. Build adjacency list from finite Delaunay edges                  //
    // ------------------------------------------------------------------ //
    AdjacencyList<T> adjList(particleIds);

    for (auto eit = dt.finite_edges_begin(); eit != dt.finite_edges_end(); ++eit) {
        auto face = eit->first;
        int  idx  = eit->second;

        auto v1 = face->vertex((idx + 1) % 3);
        auto v2 = face->vertex((idx + 2) % 3);

        T id1 = particleIds[vertexToIndex[v1]];
        T id2 = particleIds[vertexToIndex[v2]];

        adjList.addNodeEdges(id1, {id2});
    }

    // ------------------------------------------------------------------ //
    // 3. Analytical area per cell via circumcenter fan                    //
    // ------------------------------------------------------------------ //
    std::unordered_map<T, float> areaMap;
    areaMap.reserve(numParticles);
    std::unordered_map<T, std::vector<glm::vec2>> voronoiVertices;
    voronoiVertices.reserve(numParticles);

    for (auto vit = dt.finite_vertices_begin();
              vit != dt.finite_vertices_end(); ++vit)
    {
        size_t i  = vertexToIndex[vit];
        T      id = particleIds[i];

        float genX = static_cast<float>(vit->point().x());
        float genY = static_cast<float>(vit->point().y());

        // Collect the ordered ring of incident faces and check for unbounded cells
        Delaunay_2::Face_circulator fc = dt.incident_faces(vit), done = fc;
        bool is_bounded = !dt.is_infinite(fc);
        if (is_bounded) {
            do {
                ++fc;
                if (dt.is_infinite(fc)) {
                    is_bounded = false;
                    break;
                }
            } while (fc != done);
        }

        if (!is_bounded) {
            areaMap[id] = -1.0f;
            voronoiVertices[id] = {}; // No vertices for unbounded cell
            continue;
        }

        // Reset circulator for the main loop
        fc = done;

        // Sum signed areas of fan triangles:
        //   fan origin = generator point
        //   each triangle = (generator, circumcenter[k], circumcenter[k+1])
        // Because incident_faces gives a cyclic order, adjacent circumcenters
        // are already the correct consecutive Voronoi vertices.
        float area = 0.0f;
        std::vector<glm::vec2> vertices;
        do {
            Point_2 cc = dt.circumcenter(fc);
            float   cx = static_cast<float>(cc.x());
            float   cy = static_cast<float>(cc.y());
            vertices.emplace_back(cx, cy);

            // Peek at next face's circumcenter
            auto next = fc; ++next;
            Point_2 ccNext = dt.circumcenter(next);
            float   nx     = static_cast<float>(ccNext.x());
            float   ny     = static_cast<float>(ccNext.y());

            area += signedTriangleArea(genX, genY, cx, cy, nx, ny);
            ++fc;
        } while (fc != done);

        areaMap[id] = std::abs(area);
        voronoiVertices[id] = vertices;
    }

    return {adjList, areaMap, voronoiVertices};
}

// Find leaf nodes in the adjacency list
template <typename T>
std::unordered_map<T, int> findConnectedComponents(
    const AdjacencyList<T>& adjList,
    const std::unordered_map<T, float>& opacityMap
) {
    std::unordered_map<T, int> componentLabels;
    std::unordered_set<T> visited;
    int componentId = 0;

    for (const auto& pair : opacityMap) {
        const T& startNode = pair.first;
        float opacity = pair.second;

        if (opacity > 0.0f && visited.find(startNode) == visited.end()) {
            // Start of a new component
            componentId++;
            std::vector<T> stack;

            stack.push_back(startNode);
            visited.insert(startNode);
            componentLabels[startNode] = componentId;

            while (!stack.empty()) {
                T currentNode = stack.back();
                stack.pop_back();

                for (const auto& neighbor : adjList.getNeighbors(currentNode)) {
                    auto it = opacityMap.find(neighbor);
                    if (it != opacityMap.end() && it->second > 0.0f && visited.find(neighbor) == visited.end()) {
                        visited.insert(neighbor);
                        componentLabels[neighbor] = componentId;
                        stack.push_back(neighbor);
                    }
                }
            }
        }
    }

    return componentLabels;
}

// Find surface cells
template <typename T>
std::vector<T> findSurfaceCells(
    const AdjacencyList<T>& adjList,
    const std::unordered_map<T, float>& opacityMap
) {
    std::vector<T> surfaceCells;
    for (const auto& pair : opacityMap) {
        const T& nodeId = pair.first;
        float opacity = pair.second;

        if (opacity > 0.0f) {
            bool isSurfaceCell = false;
            for (const auto& neighborId : adjList.getNeighbors(nodeId)) {
                auto it = opacityMap.find(neighborId);
                if (it == opacityMap.end() || it->second == 0.0f) {
                    isSurfaceCell = true;
                    break;
                }
            }
            if (isSurfaceCell) {
                surfaceCells.push_back(nodeId);
            }
        }
    }
    return surfaceCells;
}

// Calculate center of mass
template <typename T>
glm::vec3 calculateCenterOfMass(
    const std::unordered_map<T, glm::vec3>& positions,
    const std::unordered_map<T, float>& masses
) {
    glm::vec3 com(0.0f);
    float totalMass = 0.0f;

    for (const auto& pair : positions) {
        const T& id = pair.first;
        const glm::vec3& pos = pair.second;
        
        auto massIt = masses.find(id);
        if (massIt != masses.end()) {
            float mass = massIt->second;
            com += pos * mass;
            totalMass += mass;
        }
    }

    if (totalMass > 0.0f) {
        com /= totalMass;
    }

    return com;
}

// Calculate inertia tensor
template <typename T>
glm::mat3 calculateInertiaTensor(
    const std::unordered_map<T, glm::vec3>& localPositions,
    const std::unordered_map<T, float>& masses
) {
    glm::mat3 inertiaTensor(0.0f);

    for (const auto& pair : localPositions) {
        const T& id = pair.first;
        const glm::vec3& r = pair.second; // Local position
        
        auto massIt = masses.find(id);
        if (massIt != masses.end()) {
            float mass = massIt->second;
            
            // Diagonal elements
            inertiaTensor[0][0] += mass * (r.y * r.y + r.z * r.z);
            inertiaTensor[1][1] += mass * (r.x * r.x + r.z * r.z);
            inertiaTensor[2][2] += mass * (r.x * r.x + r.y * r.y);
            
            // Off-diagonal elements
            inertiaTensor[0][1] -= mass * r.x * r.y;
            inertiaTensor[0][2] -= mass * r.x * r.z;
            inertiaTensor[1][2] -= mass * r.y * r.z;
        }
    }

    // Symmetrize the matrix
    inertiaTensor[1][0] = inertiaTensor[0][1];
    inertiaTensor[2][0] = inertiaTensor[0][2];
    inertiaTensor[2][1] = inertiaTensor[1][2];

    return inertiaTensor;
}

// Calculate AABB
std::pair<glm::vec3, glm::vec3> calculateAABB(
    const std::vector<glm::vec3>& positions
) {
    if (positions.empty()) {
        return {glm::vec3(0.0f), glm::vec3(0.0f)};
    }

    glm::vec3 min = positions[0];
    glm::vec3 max = positions[0];

    for (size_t i = 1; i < positions.size(); ++i) {
        const glm::vec3& pos = positions[i];
        min.x = std::min(min.x, pos.x);
        min.y = std::min(min.y, pos.y);
        min.z = std::min(min.z, pos.z);
        max.x = std::max(max.x, pos.x);
        max.y = std::max(max.y, pos.y);
        max.z = std::max(max.z, pos.z);
    }

    return {min, max};
}

// Count cycles in the air cell subgraph
template <typename T>
int countCycles(
    const AdjacencyList<T>& adjList,
    const std::vector<T>& surfaceIds,
    const std::unordered_map<T, float>& opacityMap
) {
    // Extract all air cells (neighbors of surface cells with opacity == 0)
    std::unordered_set<T> airCells;
    for (const T& surfaceId : surfaceIds) {
        for (const T& neighborId : adjList.getNeighbors(surfaceId)) {
            auto it = opacityMap.find(neighborId);
            if (it != opacityMap.end() && it->second == 0.0f) {
                airCells.insert(neighborId);
            }
        }
    }

    if (airCells.empty()) {
        return 0;
    }

    int V = airCells.size();
    int E = 0;
    for (const T& airCell : airCells) {
        for (const T& neighborId : adjList.getNeighbors(airCell)) {
            if (airCells.count(neighborId) && airCell < neighborId) {
                E++;
            }
        }
    }

    // Count connected components (C) in the air cell subgraph
    // int C = 0;
    // std::unordered_set<T> visited;
    // for (const T& airCell : airCells) {
    //     if (visited.find(airCell) == visited.end()) {
    //         C++;
    //         std::vector<T> stack;
    //         stack.push_back(airCell);
    //         visited.insert(airCell);
    //         while (!stack.empty()) {
    //             T current = stack.back();
    //             stack.pop_back();
    //             for (const T& neighbor : adjList.getNeighbors(current)) {
    //                 if (airCells.count(neighbor) && visited.find(neighbor) == visited.end()) {
    //                     visited.insert(neighbor);
    //                     stack.push_back(neighbor);
    //                 }
    //             }
    //         }
    //     }
    // }
    int C = 1; // Assuming all air cells are connected for simplicity

    return E - V + C;
}

} // namespace DynamicFoam::Sim2D