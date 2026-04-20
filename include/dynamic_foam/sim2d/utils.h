#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <tuple>
#include <random>
#include <algorithm>

#define NOMINMAX
#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>

#include "dynamic_foam/Sim2D/adjacency.cuh"

#ifndef __CUDACC__
// CGAL header-only setup with faster kernel
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#endif // !__CUDACC__

namespace DynamicFoam::Sim2D {

#ifndef __CUDACC__
// CGAL typedefs for use in function signatures.
// Vertex base stores the original particle index (size_t) as user info,
// eliminating the O(N^2) coordinate-scan vertex→index lookup.
using K    = CGAL::Exact_predicates_inexact_constructions_kernel;
using Vb   = CGAL::Triangulation_vertex_base_with_info_3<size_t, K>;
using Cb   = CGAL::Delaunay_triangulation_cell_base_3<K>;
using Tds  = CGAL::Triangulation_data_structure_3<Vb, Cb>;
using Delaunay_3 = CGAL::Delaunay_triangulation_3<K, Tds>;
using Point_3    = K::Point_3;
#endif // !__CUDACC__

#ifndef __CUDACC__
/**
 * @brief Triangulate and compute Voronoi cell metadata (volume, vertices).
 *
 * For each particle, this function computes the vertices of its Voronoi cell,
 * which are the circumcenters of the incident Delaunay tetrahedra. The volume
 * of the convex Voronoi cell is then calculated by decomposing it into a fan
 * of tetrahedra originating from the particle's position.
 *
 * Boundary particles (incident to infinite Delaunay cells) are assigned a
 * volume of -1 to indicate that their Voronoi cell is unbounded.
 *
 * @param positions:   Particle positions (.x, .y, .z)
 * @param particleIds: Particle IDs (must be 1-to-1 with positions)
 * @return Tuple of (AdjacencyList, volume map, Voronoi vertex map).
 *         Unbounded cells have volume -1.
 */
template <typename Vec3>
std::tuple<
    AdjacencyList,
    std::unordered_map<uint32_t, float>,
    std::unordered_map<uint32_t, std::vector<glm::vec3>>
>
triangulateWithMetadata(
    const std::vector<Vec3>&     positions,
    const std::vector<uint32_t>& particleIds
) {
    size_t numParticles = particleIds.size();

    // Insert points with their original index stored directly in the vertex.
    // This eliminates the O(N^2) coordinate-scan that previously populated
    // vertexToIndex; all lookups become O(1) via vh->info().
    Delaunay_3 dt;
    {
        std::vector<std::pair<Point_3, size_t>> indexed_points;
        indexed_points.reserve(numParticles);
        for (size_t i = 0; i < numParticles; ++i)
            indexed_points.emplace_back(
                Point_3(positions[i].x, positions[i].y, positions[i].z), i);
        dt.insert(indexed_points.begin(), indexed_points.end());
    }

    // Build adjacency using O(1) vh->info() index lookup.
    // CGAL's finite_edges iterator emits each undirected edge exactly once.
    AdjacencyList adjList(particleIds);
    adjList.reserveEdges(static_cast<size_t>(dt.number_of_finite_edges()));
    for (auto eit = dt.finite_edges_begin(); eit != dt.finite_edges_end(); ++eit) {
        auto v1 = eit->first->vertex(eit->second);
        auto v2 = eit->first->vertex(eit->third);
        adjList.addEdge(particleIds[v1->info()], particleIds[v2->info()]);
    }

    // Pre-compute every finite cell's circumcenter once.
    // Storing by Cell_handle (pointer) avoids redundant dt.dual() calls in the
    // per-vertex Voronoi loop, which previously recomputed the same circumcenters
    // once per incident edge per vertex.
    std::unordered_map<Delaunay_3::Cell_handle, glm::vec3> ccCache;
    ccCache.reserve(static_cast<size_t>(dt.number_of_finite_cells()));
    for (auto cit = dt.finite_cells_begin(); cit != dt.finite_cells_end(); ++cit) {
        Point_3 cc = dt.dual(cit);
        ccCache[cit] = glm::vec3(
            static_cast<float>(cc.x()),
            static_cast<float>(cc.y()),
            static_cast<float>(cc.z()));
    }

    std::unordered_map<uint32_t, float> volumeMap;
    volumeMap.reserve(numParticles);
    std::unordered_map<uint32_t, std::vector<glm::vec3>> voronoiVertices;
    voronoiVertices.reserve(numParticles);

    for (auto vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit) {
        size_t   i  = vit->info();
        uint32_t id = particleIds[i];

        glm::vec3 genPt(
            static_cast<float>(vit->point().x()),
            static_cast<float>(vit->point().y()),
            static_cast<float>(vit->point().z()));

        if (dt.is_infinite(vit)) {
            volumeMap[id] = -1.0f;
            voronoiVertices[id] = {};
            continue;
        }

        // Collect Voronoi vertices (circumcenters of incident finite cells)
        // from the cache — no new dt.dual() calls needed.
        std::vector<Delaunay_3::Cell_handle> incident_cells;
        dt.incident_cells(vit, std::back_inserter(incident_cells));
        std::vector<glm::vec3> v_vertices;
        v_vertices.reserve(incident_cells.size());
        for (const auto& ch : incident_cells) {
            auto it = ccCache.find(ch);
            if (it != ccCache.end())
                v_vertices.push_back(it->second);
        }
        voronoiVertices[id] = v_vertices;

        // Compute Voronoi cell volume: for each incident Delaunay edge the dual
        // is a Voronoi face polygon; decompose each polygon into a fan of
        // tetrahedra from genPt. All circumcenters are read from cache.
        float total_volume = 0.0f;
        std::vector<Delaunay_3::Edge> incident_edges;
        dt.incident_edges(vit, std::back_inserter(incident_edges));

        for (const auto& edge : incident_edges) {
            Delaunay_3::Cell_circulator cc = dt.incident_cells(edge), done(cc);
            if (cc == nullptr) continue;

            std::vector<glm::vec3> face_points;
            do {
                auto it = ccCache.find(cc);
                if (it != ccCache.end())
                    face_points.push_back(it->second);
                ++cc;
            } while (cc != done);

            if (face_points.size() < 3) continue;
            const glm::vec3& p0 = face_points[0];
            for (size_t j = 1; j + 1 < face_points.size(); ++j) {
                total_volume += (1.0f / 6.0f) * glm::determinant(
                    glm::mat3(p0 - genPt, face_points[j] - genPt, face_points[j + 1] - genPt));
            }
        }
        volumeMap[id] = std::abs(total_volume);
    }

    return {adjList, volumeMap, voronoiVertices};
}
#endif // !__CUDACC__

/**
 * @brief Finds connected components in a graph based on an opacity map.
 *
 * This function performs a graph traversal (like BFS or DFS) to identify
 * connected components. The traversal is bounded by vertices with an opacity
 * of 0.0f. Each component is assigned a unique integer ID.
 *
 * @tparam T The type of the node IDs.
 * @param adjList The adjacency list representing the graph.
 * @param opacityMap A map from node IDs to their opacity values (0.0f to 1.0f).
 * @return A map from node IDs to their component ID. Nodes with 0.0f opacity
 *         or nodes that are not part of any component will not be in the map.
 */
inline std::unordered_map<uint32_t, int> findConnectedComponents(
    const AdjacencyList& adjList,
    const std::unordered_map<uint32_t, float>& opacityMap
) {
    std::unordered_map<uint32_t, int> componentLabels;
    std::unordered_set<uint32_t> visited;
    int componentId = 0;

    for (const auto& pair : opacityMap) {
        const uint32_t& startNode = pair.first;
        float opacity = pair.second;

        if (opacity > 0.0f && visited.find(startNode) == visited.end()) {
            componentId++;
            std::vector<uint32_t> stack;
            stack.push_back(startNode);
            visited.insert(startNode);
            componentLabels[startNode] = componentId;

            while (!stack.empty()) {
                uint32_t currentNode = stack.back();
                stack.pop_back();
                adjList.forEachNeighbor(currentNode, [&](const uint32_t& neighbor) {
                    auto it = opacityMap.find(neighbor);
                    if (it != opacityMap.end() && it->second > 0.0f
                            && visited.find(neighbor) == visited.end()) {
                        visited.insert(neighbor);
                        componentLabels[neighbor] = componentId;
                        stack.push_back(neighbor);
                    }
                });
            }
        }
    }
    return componentLabels;
}

/**
 * @brief Finds surface cells in a graph based on an opacity map.
 *
 * A surface cell is defined as a vertex with an opacity greater than 0
 * that is a neighbor to any vertex with an opacity of 0.
 *
 * @tparam T The type of the node IDs.
 * @param adjList The adjacency list representing the graph.
 * @param opacityMap A map from node IDs to their opacity values (0.0f to 1.0f).
 * @return A vector of IDs of the surface cells.
 */
inline std::vector<uint32_t> findSurfaceCells(
    const AdjacencyList& adjList,
    const std::unordered_map<uint32_t, float>& opacityMap
) {
    std::vector<uint32_t> surfaceCells;
    for (const auto& pair : opacityMap) {
        const uint32_t& nodeId = pair.first;
        if (pair.second > 0.0f) {
            bool isSurfaceCell = false;
            adjList.forEachNeighbor(nodeId, [&](const uint32_t& neighborId) {
                auto it = opacityMap.find(neighborId);
                if (it == opacityMap.end() || it->second == 0.0f)
                    isSurfaceCell = true;
            });
            if (isSurfaceCell)
                surfaceCells.push_back(nodeId);
        }
    }
    return surfaceCells;
}

/**
 * @brief Calculates the center of mass for a collection of particles.
 *
 * @tparam T The type of the particle IDs.
 * @param positions A map from particle IDs to their positions (glm::vec3).
 * @param masses A map from particle IDs to their masses (float).
 * @return glm::vec3 The calculated center of mass.
 */
template <typename T>
glm::vec3 calculateCenterOfMass(
    const std::unordered_map<T, glm::vec3>& positions,
    const std::unordered_map<T, float>& masses
) {
    glm::vec3 com(0.0f);
    float totalMass = 0.0f;
    for (const auto& pair : positions) {
        auto massIt = masses.find(pair.first);
        if (massIt != masses.end()) {
            com += pair.second * massIt->second;
            totalMass += massIt->second;
        }
    }
    if (totalMass > 0.0f) com /= totalMass;
    return com;
}

/**
 * @brief Calculates the inertia tensor for a collection of particles.
 *
 * @tparam T The type of the particle IDs.
 * @param localPositions A map from particle IDs to their local positions relative to the center of mass.
 * @param masses A map from particle IDs to their masses.
 * @return glm::mat3 The calculated inertia tensor.
 */
template <typename T>
glm::mat3 calculateInertiaTensor(
    const std::unordered_map<T, glm::vec3>& localPositions,
    const std::unordered_map<T, float>& masses
) {
    glm::mat3 I(0.0f);
    for (const auto& pair : localPositions) {
        auto massIt = masses.find(pair.first);
        if (massIt == masses.end()) continue;
        float m = massIt->second;
        const glm::vec3& r = pair.second;
        I[0][0] += m * (r.y*r.y + r.z*r.z);
        I[1][1] += m * (r.x*r.x + r.z*r.z);
        I[2][2] += m * (r.x*r.x + r.y*r.y);
        I[0][1] -= m * r.x * r.y;
        I[0][2] -= m * r.x * r.z;
        I[1][2] -= m * r.y * r.z;
    }
    I[1][0] = I[0][1];
    I[2][0] = I[0][2];
    I[2][1] = I[1][2];
    return I;
}

/**
 * @brief Calculates the Axis-Aligned Bounding Box (AABB) for a collection of positions.
 *
 * @param positions A vector of positions (glm::vec3).
 * @return A pair containing the min and max corners of the AABB as glm::vec3.
 */
inline std::pair<glm::vec3, glm::vec3> calculateAABB(
    const std::vector<glm::vec3>& positions
) {
    if (positions.empty())
        return {glm::vec3(0.0f), glm::vec3(0.0f)};
    glm::vec3 lo = positions[0], hi = positions[0];
    for (size_t i = 1; i < positions.size(); ++i) {
        lo.x = std::min(lo.x, positions[i].x);
        lo.y = std::min(lo.y, positions[i].y);
        lo.z = std::min(lo.z, positions[i].z);
        hi.x = std::max(hi.x, positions[i].x);
        hi.y = std::max(hi.y, positions[i].y);
        hi.z = std::max(hi.z, positions[i].z);
    }
    return {lo, hi};
}

/**
 * @brief Counts cycles in the air cell subgraph using cyclomatic complexity.
 *
 * This function computes the cyclomatic complexity (E - V + C) of the subgraph
 * formed by air cells (vertices with opacity == 0 that are neighbors of surface cells).
 * The number of edges counts only those connecting two air cells.
 *
 * @tparam T The type of the node IDs.
 * @param adjList The adjacency list representing the graph.
 * @param surfaceIds A vector of surface cell IDs.
 * @param opacityMap A map from node IDs to their opacity values (0.0f to 1.0f).
 * @return The cycle count (number of independent cycles in the air cell subgraph).
 */
inline int countCycles(
    const AdjacencyList& adjList,
    const std::vector<uint32_t>& surfaceIds,
    const std::unordered_map<uint32_t, float>& opacityMap
) {
    std::unordered_set<uint32_t> airCells;
    for (const uint32_t& surfaceId : surfaceIds) {
        adjList.forEachNeighbor(surfaceId, [&](const uint32_t& neighborId) {
            auto it = opacityMap.find(neighborId);
            if (it != opacityMap.end() && it->second == 0.0f)
                airCells.insert(neighborId);
        });
    }
    if (airCells.empty()) return 0;

    int V = static_cast<int>(airCells.size());
    int E = 0;
    for (const uint32_t& airCell : airCells) {
        adjList.forEachNeighbor(airCell, [&](const uint32_t& neighborId) {
            if (airCells.count(neighborId) && airCell < neighborId)
                E++;
        });
    }
    int C = 1; // Assuming all air cells are connected
    return E - V + C;
}

/**
 * @brief Finds shared air particles — air cells that are adjacent to at least
 * two distinct connected components.
 *
 * A shared air particle is an air particle (opacity == 0) whose non-air
 * neighbours belong to two or more different connected components.  Because a
 * single air particle can be shared by several components simultaneously the
 * output map may contain the same particle ID under multiple component keys.
 *
 * @param adjList         The adjacency list representing the graph.
 * @param surfaceMap      Map from particle ID → true if the particle is a
 *                        surface cell (i.e. a non-air particle on the boundary
 *                        of a foam cluster).  Used to restrict the neighbour
 *                        search to particles that are topologically meaningful.
 * @param opacityMap      Map from particle ID → opacity (0.0f = air, >0 = foam).
 * @param componentLabels Output of findConnectedComponents: particle ID →
 *                        component ID (only non-air particles are present).
 * @return An unordered_map<component_id, vector<air_particle_ids>> where every
 *         entry lists the air particles shared with that component.  A particle
 *         that spans N components appears once in each of those N entries.
 */
inline std::unordered_map<int, std::vector<uint32_t>> findSharedAirParticles(
    const AdjacencyList& adjList,
    const std::unordered_map<uint32_t, bool>& surfaceMap,
    const std::unordered_map<uint32_t, float>& opacityMap,
    const std::unordered_map<uint32_t, int>& componentLabels
) {
    std::unordered_map<int, std::vector<uint32_t>> result;

    // Collect candidate air particles: those adjacent to at least one surface
    // particle.  Using surfaceMap as the seed avoids scanning the entire graph.
    std::unordered_set<uint32_t> candidates;
    for (const auto& [id, isSurface] : surfaceMap) {
        if (!isSurface) continue;
        // Only seed from particles that actually belong to a component.
        if (!componentLabels.count(id)) continue;
        adjList.forEachNeighbor(id, [&](uint32_t neighborId) {
            auto opIt = opacityMap.find(neighborId);
            if (opIt != opacityMap.end() && opIt->second == 0.0f)
                candidates.insert(neighborId);
        });
    }

    // For each candidate air particle, collect the set of component IDs that
    // its non-air neighbours belong to.
    for (uint32_t airId : candidates) {
        std::unordered_set<int> touchedComponents;
        adjList.forEachNeighbor(airId, [&](uint32_t neighborId) {
            auto compIt = componentLabels.find(neighborId);
            if (compIt != componentLabels.end())
                touchedComponents.insert(compIt->second);
        });

        // Only an air bridge if it spans at least two distinct components.
        if (touchedComponents.size() < 2) continue;

        for (int compId : touchedComponents)
            result[compId].push_back(airId);
    }

    return result;
}

} // namespace DynamicFoam::Sim2D
