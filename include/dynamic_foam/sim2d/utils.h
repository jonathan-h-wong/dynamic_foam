#pragma once

#include <vector>
#include <unordered_map>
#include <utility>
#include <tuple>

#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>

#include "dynamic_foam/Sim2D/adjacency.cuh"

// CGAL header-only setup with faster kernel
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

namespace DynamicFoam::Sim2D {

// CGAL typedefs for use in function signatures
// Using Exact_predicates_inexact_constructions_kernel for faster compilation
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Delaunay_3 = CGAL::Delaunay_triangulation_3<K>;
using Point_3 = K::Point_3;

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
template <typename T, typename Vec3>
std::tuple<
    AdjacencyList<T>,
    std::unordered_map<T, float>,
    std::unordered_map<T, std::vector<glm::vec3>>
>
triangulateWithMetadata(
    const std::vector<Vec3>& positions,
    const std::vector<T>& particleIds
);

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
template <typename T>
std::unordered_map<T, int> findConnectedComponents(
    const AdjacencyList<T>& adjList,
    const std::unordered_map<T, float>& opacityMap
);

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
template <typename T>
std::vector<T> findSurfaceCells(
    const AdjacencyList<T>& adjList,
    const std::unordered_map<T, float>& opacityMap
);

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
);

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
);

/**
 * @brief Calculates the Axis-Aligned Bounding Box (AABB) for a collection of positions.
 *
 * @param positions A vector of positions (glm::vec3).
 * @return A pair containing the min and max corners of the AABB as glm::vec3.
 */
std::pair<glm::vec3, glm::vec3> calculateAABB(
    const std::vector<glm::vec3>& positions
);

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
template <typename T>
int countCycles(
    const AdjacencyList<T>& adjList,
    const std::vector<T>& surfaceIds,
    const std::unordered_map<T, float>& opacityMap
);

} // namespace DynamicFoam::Sim2D
