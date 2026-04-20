#pragma once

#include <vector>
#include <unordered_map>
#include <tuple>

#include <glm/glm.hpp>

#include "dynamic_foam/Sim2D/adjacency.cuh"

#ifndef __CUDACC__
// CGAL header-only setup with faster kernel
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Triangulation_data_structure_3.h>
#endif // !__CUDACC__

namespace DynamicFoam::Sim2D {

#ifndef __CUDACC__
// CGAL typedefs — vertex carries particle array index as info
using K_tri        = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_tri    = K_tri::Point_3;
using Vb_tri       = CGAL::Triangulation_vertex_base_with_info_3<size_t, K_tri>;
using Cb_tri       = CGAL::Delaunay_triangulation_cell_base_3<K_tri>;
using Tds_tri      = CGAL::Triangulation_data_structure_3<Vb_tri, Cb_tri>;
using Delaunay_tri = CGAL::Delaunay_triangulation_3<K_tri, Tds_tri>;
#endif // !__CUDACC__

// ============================================================================
// Triangulation
// ============================================================================

#ifndef __CUDACC__
/**
 * @brief Build Delaunay triangulation, extract adjacency and a Voronoi
 *        vertex buffer for every particle.
 *
 * Volume is NOT computed.  Callers that need per-particle mass should
 * approximate it from the AABB of the returned voronoiVertices (or from a
 * uniform sphere/particle radius).
 *
 * Boundary particles (incident to at least one infinite Delaunay cell) receive
 * an empty vertex list.
 *
 * @tparam Vec3    Position type exposing .x / .y / .z members.
 * @param positions   Particle positions in world / local space.
 * @param particleIds Particle IDs, one-to-one with positions.
 * @return Tuple of
 *         - AdjacencyList built from finite Delaunay edges,
 *         - per-particle Voronoi vertex buffer (empty for unbounded cells).
 */
template <typename Vec3>
std::tuple<
    AdjacencyList,
    std::unordered_map<uint32_t, std::vector<glm::vec3>>
>
triangulateVoronoiCells(
    const std::vector<Vec3>&     positions,
    const std::vector<uint32_t>& particleIds)
{
    const size_t numParticles = particleIds.size();

    // ------------------------------------------------------------------
    // 1. Build Delaunay triangulation — vertex carries array index as info
    // ------------------------------------------------------------------
    std::vector<std::pair<Point_tri, size_t>> indexedPoints;
    indexedPoints.reserve(numParticles);
    for (size_t i = 0; i < numParticles; ++i)
        indexedPoints.emplace_back(
            Point_tri(positions[i].x, positions[i].y, positions[i].z), i);

    Delaunay_tri dt;
    dt.insert(indexedPoints.begin(), indexedPoints.end());

    // ------------------------------------------------------------------
    // 2. Adjacency list from finite Delaunay edges
    // ------------------------------------------------------------------
    AdjacencyList adjList(particleIds);
    adjList.reserveEdges(dt.number_of_finite_edges());

    for (auto eit = dt.finite_edges_begin();
              eit != dt.finite_edges_end(); ++eit) {
        const auto& cell = eit->first;
        uint32_t id1 = particleIds[cell->vertex(eit->second)->info()];
        uint32_t id2 = particleIds[cell->vertex(eit->third)->info()];
        adjList.addEdgeUnique(id1, id2);
    }

    // ------------------------------------------------------------------
    // 3. Circumcenter cache — one dt.dual() per finite cell
    // ------------------------------------------------------------------
    std::unordered_map<Delaunay_tri::Cell_handle, glm::vec3> ccCache;
    ccCache.reserve(dt.number_of_finite_cells());
    for (auto cit = dt.finite_cells_begin();
              cit != dt.finite_cells_end(); ++cit) {
        Point_tri cc = dt.dual(cit);
        ccCache[cit] = glm::vec3(
            static_cast<float>(cc.x()),
            static_cast<float>(cc.y()),
            static_cast<float>(cc.z()));
    }

    // ------------------------------------------------------------------
    // 4. Per-particle Voronoi vertex buffer (no volume)
    // ------------------------------------------------------------------
    std::unordered_map<uint32_t, std::vector<glm::vec3>> voronoiVertices;
    voronoiVertices.reserve(numParticles);

    for (auto vit = dt.finite_vertices_begin();
              vit != dt.finite_vertices_end(); ++vit) {
        uint32_t id = particleIds[vit->info()];

        std::vector<Delaunay_tri::Cell_handle> incCells;
        dt.incident_cells(vit, std::back_inserter(incCells));

        bool unbounded = false;
        for (const auto& ch : incCells) {
            if (dt.is_infinite(ch)) { unbounded = true; break; }
        }
        if (unbounded) {
            voronoiVertices[id] = {};
            continue;
        }

        std::vector<glm::vec3> verts;
        verts.reserve(incCells.size());
        for (const auto& ch : incCells)
            verts.push_back(ccCache.at(ch));

        voronoiVertices[id] = std::move(verts);
    }

    return {adjList, voronoiVertices};
}
#endif // !__CUDACC__

} // namespace DynamicFoam::Sim2D
