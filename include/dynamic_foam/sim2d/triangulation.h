#pragma once

#include <vector>
#include <unordered_map>
#include <tuple>
#include <map>

#include <glm/glm.hpp>

#include "dynamic_foam/Sim2D/adjacency.cuh"

// CGAL header-only setup with faster kernel
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

namespace DynamicFoam::Sim2D {

// CGAL typedefs (shared with utils.h; re-declared here for standalone use)
using K_tri       = CGAL::Exact_predicates_inexact_constructions_kernel;
using Delaunay_tri = CGAL::Delaunay_triangulation_3<K_tri>;
using Point_tri   = K_tri::Point_3;

// ============================================================================
// Triangulation – internal helpers
// ============================================================================

namespace detail {

inline float signedTetVolume(
    const glm::vec3& a, const glm::vec3& b,
    const glm::vec3& c, const glm::vec3& d)
{
    return (1.0f / 6.0f) * glm::determinant(glm::mat3(b - a, c - a, d - a));
}

} // namespace detail

// ============================================================================
// Triangulation
// ============================================================================

/**
 * @brief Build Delaunay triangulation, extract adjacency, volumes, and
 *        a vertex buffer for every Voronoi cell.
 *
 * For each particle the Voronoi cell is computed via the dual of the 3-D
 * Delaunay triangulation.  The unique circumcenter vertices of the incident
 * Delaunay tetrahedra form the convex hull of the Voronoi cell and are
 * returned as a plain vertex list suitable for GJK support queries.
 *
 * Boundary particles (incident to at least one infinite Delaunay cell) are
 * assigned volume -1 and an empty vertex list.
 *
 * @tparam T       Particle ID type (e.g. int).
 * @tparam Vec3    Position type exposing .x / .y / .z members.
 * @param positions   Particle positions in world / local space.
 * @param particleIds Particle IDs, one-to-one with positions.
 * @return Tuple of
 *         - AdjacencyList built from finite Delaunay edges,
 *         - per-particle volume map (volume = -1 for unbounded cells),
 *         - per-particle vertex buffer (empty for unbounded cells).
 */
template <typename T, typename Vec3>
std::tuple<
    AdjacencyList<T>,
    std::unordered_map<T, float>,
    std::unordered_map<T, std::vector<glm::vec3>>
>
triangulateVoronoiCells(
    const std::vector<Vec3>& positions,
    const std::vector<T>&    particleIds)
{
    size_t numParticles = particleIds.size();

    // ------------------------------------------------------------------
    // 1. Build Delaunay triangulation
    // ------------------------------------------------------------------
    Delaunay_tri dt;
    std::unordered_map<Delaunay_tri::Vertex_handle, size_t> vertexToIndex;

    std::vector<Point_tri> points;
    points.reserve(numParticles);
    for (size_t i = 0; i < numParticles; ++i)
        points.emplace_back(positions[i].x, positions[i].y, positions[i].z);
    dt.insert(points.begin(), points.end());

    for (auto vh = dt.finite_vertices_begin();
              vh != dt.finite_vertices_end(); ++vh) {
        const auto& p = vh->point();
        for (size_t i = 0; i < numParticles; ++i) {
            if (p.x() == positions[i].x &&
                p.y() == positions[i].y &&
                p.z() == positions[i].z) {
                vertexToIndex[vh] = i;
                break;
            }
        }
    }

    // ------------------------------------------------------------------
    // 2. Adjacency list from finite Delaunay edges
    // ------------------------------------------------------------------
    AdjacencyList<T> adjList(particleIds);

    for (auto eit = dt.finite_edges_begin();
              eit != dt.finite_edges_end(); ++eit) {
        auto cell = eit->first;
        auto v1   = cell->vertex(eit->second);
        auto v2   = cell->vertex(eit->third);
        if (vertexToIndex.count(v1) && vertexToIndex.count(v2)) {
            T id1 = particleIds[vertexToIndex[v1]];
            T id2 = particleIds[vertexToIndex[v2]];
            adjList.addNodeEdges(id1, {id2});
        }
    }

    // ------------------------------------------------------------------
    // 3. Per-particle Voronoi volume + vertex buffer
    // ------------------------------------------------------------------
    std::unordered_map<T, float>                  volumeMap;
    std::unordered_map<T, std::vector<glm::vec3>> voronoiVertices;
    volumeMap.reserve(numParticles);
    voronoiVertices.reserve(numParticles);

    for (auto vit = dt.finite_vertices_begin();
              vit != dt.finite_vertices_end(); ++vit) {
        if (!vertexToIndex.count(vit)) continue;

        size_t i  = vertexToIndex[vit];
        T      id = particleIds[i];

        glm::vec3 genPt(
            static_cast<float>(vit->point().x()),
            static_cast<float>(vit->point().y()),
            static_cast<float>(vit->point().z()));

        {
            std::vector<Delaunay_tri::Cell_handle> incCells;
            dt.incident_cells(vit, std::back_inserter(incCells));
            bool unbounded = false;
            for (const auto& ch : incCells) {
                if (dt.is_infinite(ch)) { unbounded = true; break; }
            }
            if (unbounded) {
                volumeMap[id]       = -1.0f;
                voronoiVertices[id] = {};
                continue;
            }
        }

        std::vector<glm::vec3> mergedVerts;
        std::map<std::tuple<float, float, float>, int> vertIdx;

        auto getOrAdd = [&](const glm::vec3& v) -> int {
            auto key = std::make_tuple(v.x, v.y, v.z);
            auto it  = vertIdx.find(key);
            if (it != vertIdx.end()) return it->second;
            int idx = static_cast<int>(mergedVerts.size());
            mergedVerts.push_back(v);
            vertIdx[key] = idx;
            return idx;
        };

        float totalVolume = 0.0f;

        std::vector<Delaunay_tri::Edge> incEdges;
        dt.incident_edges(vit, std::back_inserter(incEdges));

        for (const auto& edge : incEdges) {
            Delaunay_tri::Cell_circulator cc =
                dt.incident_cells(edge), done(cc);
            if (cc == nullptr) continue;

            bool faceIsFinite = true;
            std::vector<int> faceIdx;

            do {
                if (dt.is_infinite(cc)) {
                    faceIsFinite = false;
                    break;
                }
                Point_tri dual = dt.dual(cc);
                glm::vec3 v(
                    static_cast<float>(dual.x()),
                    static_cast<float>(dual.y()),
                    static_cast<float>(dual.z()));
                faceIdx.push_back(getOrAdd(v));
                ++cc;
            } while (cc != done);

            if (!faceIsFinite || faceIdx.size() < 3) continue;

            for (size_t j = 1; j + 1 < faceIdx.size(); ++j) {
                const glm::vec3& p0 = mergedVerts[faceIdx[0]];
                const glm::vec3& p1 = mergedVerts[faceIdx[j]];
                const glm::vec3& p2 = mergedVerts[faceIdx[j + 1]];
                totalVolume += detail::signedTetVolume(genPt, p0, p1, p2);
            }
        }

        volumeMap[id]       = std::abs(totalVolume);
        voronoiVertices[id] = std::move(mergedVerts);
    }

    return {adjList, volumeMap, voronoiVertices};
}

} // namespace DynamicFoam::Sim2D
