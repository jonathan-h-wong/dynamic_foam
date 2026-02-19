#pragma once

#include <vector>
#include <unordered_map>
#include <utility>
#include <tuple>

#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>

#include "dynamic_foam/sim2d/adjacency_list.h"
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

namespace DynamicFoam::Sim2D {

// CGAL typedefs for use in function signatures
using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Delaunay_2 = CGAL::Delaunay_triangulation_2<K>;
using Point_2 = K::Point_2;

/**
 * @brief Triangulates a set of 2D particles using Delaunay triangulation.
 *
 * This function takes a set of 2D particle positions and their corresponding IDs,
 * and computes the Delaunay triangulation. It returns the adjacency list representing
 * the connectivity of the triangulation.
 *
 * @tparam T The type of the particle IDs.
 * @tparam Vec2 The type of the 2D position vectors (must have .x and .y members).
 * @param positions A vector of particle positions.
 * @param particleIds A vector of particle IDs.
 * @return AdjacencyList<T>: The graph of particle neighbors.
 */
template <typename T, typename Vec2>
AdjacencyList<T>
triangulate(
    const std::vector<Vec2>& positions,
    const std::vector<T>& particleIds
);

/**
 * @brief Performs triangulation and calculates Voronoi cell areas using GPU-accelerated Monte Carlo integration.
 *
 * This function first triangulates the particles and then computes the area of each particle's
 * Voronoi cell. The area calculation is accelerated using a CUDA kernel that performs
 * Monte Carlo sampling.
 *
 * @tparam T The type of the particle IDs.
 * @tparam Vec2 The type of the 2D position vectors (must have .x and .y members).
 * @param positions A vector of particle positions.
 * @param particleIds A vector of particle IDs.
 * @param samplesPerCell The number of Monte Carlo samples to use for each cell's area calculation.
 * @param seed The random seed for the Monte Carlo simulation for reproducibility.
 * @return A pair containing:
 *         - AdjacencyList<T>: The graph of particle neighbors.
 *         - std::unordered_map<T, float>: A map from particle IDs to their Voronoi cell areas.
 */
template <typename T, typename Vec2>
std::pair<AdjacencyList<T>, std::unordered_map<T, float>>
triangulateWithAreaIntegration(
    const std::vector<Vec2>& positions,
    const std::vector<T>& particleIds,
    int samplesPerCell = 10000,
    unsigned int seed = 42
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

} // namespace DynamicFoam::Sim2D
