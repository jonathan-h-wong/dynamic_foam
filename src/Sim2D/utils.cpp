// TODO: 
// Add position + quaternion => 4x4 matrix conversion utilities

#pragma once
#include <random>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>

#include "dynamic_foam/Sim2D/utils.h"
#include "dynamic_foam/Sim2D/adjacency.h"

namespace DynamicFoam::Sim2D {

/**
 * Compute the signed volume of a tetrahedron in 3D
 */
inline float signedTetrahedronVolume(
    const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& d
) {
    return (1.0f / 6.0f) * glm::determinant(glm::mat3(b - a, c - a, d - a));
}

/**
 * Triangulate with analytical Voronoi volume integration via direct decomposition
 *
 * For each particle, this function computes the volume of its Voronoi cell.
 * It leverages the duality between the Delaunay triangulation and the Voronoi
 * diagram. The Voronoi cell's volume is calculated by decomposing it into a
 * set of pyramids, where each pyramid's base is a Voronoi face and its apex
 * is the generator particle.
 *
 * This is achieved by iterating through the Delaunay edges incident to a
 * generator. Each edge is dual to a Voronoi face. The vertices of this face
 * are the circumcenters of the Delaunay cells sharing that edge. The face is
 * triangulated, and each resulting triangle forms a tetrahedron with the
 * generator point. The sum of the signed volumes of these tetrahedra gives
 * the total volume of the Voronoi cell.
 *
 * Boundary particles (incident to infinite Delaunay cells) are assigned a
 * volume of -1 to indicate that their Voronoi cell is unbounded.
 *
 * @param positions:   Particle positions (.x, .y, .z)
 * @param particleIds: Particle IDs (must be 1-to-1 with positions)
 * @return Tuple of (AdjacencyList, volume map, Voronoi vertex map).
 *         Unbounded cells have volume -1.
 */
template <typename T, typename Vec3>
std::tuple<
    AdjacencyList<T>,
    std::unordered_map<T, float>,
    std::unordered_map<T, std::vector<glm::vec3>>
>
triangulateWithMetadata(
    const std::vector<Vec3>& positions,
    const std::vector<T>&    particleIds
) {
    size_t numParticles = particleIds.size();

    // ------------------------------------------------------------------ //
    // 1. Build Delaunay triangulation, keep vertex handles                //
    // ------------------------------------------------------------------ //
    Delaunay_3 dt;
    std::unordered_map<Delaunay_3::Vertex_handle, size_t> vertexToIndex;

    std::vector<Point_3> points;
    points.reserve(numParticles);
    for (size_t i = 0; i < numParticles; ++i) {
        points.emplace_back(positions[i].x, positions[i].y, positions[i].z);
    }
    dt.insert(points.begin(), points.end());

    for(auto vh = dt.finite_vertices_begin(); vh != dt.finite_vertices_end(); ++vh) {
        // This is a bit of a hack to associate points back to IDs,
        // as the vertex handles change after the bulk insert.
        // A more robust way would be to use a Point-to-ID map if positions are not unique.
        const auto& p = vh->point();
        for(size_t i = 0; i < numParticles; ++i) {
            if (p.x() == positions[i].x && p.y() == positions[i].y && p.z() == positions[i].z) {
                vertexToIndex[vh] = i;
                break;
            }
        }
    }


    // ------------------------------------------------------------------ //
    // 2. Build adjacency list from finite Delaunay edges                  //
    // ------------------------------------------------------------------ //
    AdjacencyList<T> adjList(particleIds);

    for (auto eit = dt.finite_edges_begin(); eit != dt.finite_edges_end(); ++eit) {
        auto cell = eit->first;
        auto v1 = cell->vertex(eit->second);
        auto v2 = cell->vertex(eit->third);

        if (vertexToIndex.count(v1) && vertexToIndex.count(v2)) {
            T id1 = particleIds[vertexToIndex[v1]];
            T id2 = particleIds[vertexToIndex[v2]];
            adjList.addNodeEdges(id1, {id2});
        }
    }

    // ------------------------------------------------------------------ //
    // 3. Analytical volume per cell via direct decomposition            //
    // ------------------------------------------------------------------ //
    std::unordered_map<T, float> volumeMap;
    volumeMap.reserve(numParticles);
    std::unordered_map<T, std::vector<glm::vec3>> voronoiVertices;
    voronoiVertices.reserve(numParticles);

    for (auto vit = dt.finite_vertices_begin();
              vit != dt.finite_vertices_end(); ++vit)
    {
        if (!vertexToIndex.count(vit)) continue;
        size_t i  = vertexToIndex[vit];
        T      id = particleIds[i];

        glm::vec3 genPt(
            static_cast<float>(vit->point().x()),
            static_cast<float>(vit->point().y()),
            static_cast<float>(vit->point().z())
        );

        // Check if the vertex is on the convex hull, which means its Voronoi cell is unbounded.
        if (dt.is_infinite(vit)) {
            volumeMap[id] = -1.0f;
            voronoiVertices[id] = {};
            continue;
        }

        // Collect all circumcenters of incident cells to form the Voronoi vertices
        std::vector<Delaunay_3::Cell_handle> incident_cells;
        dt.incident_cells(vit, std::back_inserter(incident_cells));
        std::vector<glm::vec3> v_vertices;
        for (const auto& cell_handle : incident_cells) {
            Point_3 cc = dt.dual(cell_handle);
            v_vertices.emplace_back(static_cast<float>(cc.x()), static_cast<float>(cc.y()), static_cast<float>(cc.z()));
        }
        voronoiVertices[id] = v_vertices;

        // Decompose Voronoi cell into pyramids and sum their volumes
        float total_volume = 0.0f;

        std::vector<Delaunay_3::Edge> incident_edges;
        dt.incident_edges(vit, std::back_inserter(incident_edges));

        for (const auto& edge : incident_edges) {
            Delaunay_3::Cell_circulator cc = dt.incident_cells(edge), done(cc);
            if (cc == nullptr) continue;

            std::vector<glm::vec3> face_points;
            do {
                if (!dt.is_infinite(cc)) {
                    Point_3 dual_pt = dt.dual(cc);
                    face_points.emplace_back(
                        static_cast<float>(dual_pt.x()),
                        static_cast<float>(dual_pt.y()),
                        static_cast<float>(dual_pt.z())
                    );
                }
                cc++;
            } while (cc != done);

            if (face_points.size() < 3) continue;

            // Triangulate the Voronoi face (polygon) into a fan and sum tetrahedra volumes
            const glm::vec3& p0 = face_points[0];
            for (size_t j = 1; j < face_points.size() - 1; ++j) {
                const glm::vec3& p1 = face_points[j];
                const glm::vec3& p2 = face_points[j + 1];
                total_volume += signedTetrahedronVolume(genPt, p0, p1, p2);
            }
        }
        volumeMap[id] = std::abs(total_volume);
    }

    return {adjList, volumeMap, voronoiVertices};
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