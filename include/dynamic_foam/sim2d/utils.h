#pragma once

#include <vector>
#include <unordered_map>
#include <utility>
#include <tuple>

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

} // namespace DynamicFoam::Sim2D
