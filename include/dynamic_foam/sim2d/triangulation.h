#pragma once

#include <vector>
#include <unordered_map>
#include <tuple>
#include <map>

#include <glm/glm.hpp>

#include "dynamic_foam/Sim2D/adjacency.cuh"

#ifndef __CUDACC__
// CGAL header-only setup with faster kernel
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#endif // !__CUDACC__

namespace DynamicFoam::Sim2D {

#ifndef __CUDACC__
// CGAL typedefs (shared with utils.h; re-declared here for standalone use)
using K_tri       = CGAL::Exact_predicates_inexact_constructions_kernel;
using Delaunay_tri = CGAL::Delaunay_triangulation_3<K_tri>;
using Point_tri   = K_tri::Point_3;
#endif // !__CUDACC__

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

#ifndef __CUDACC__
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
 * @tparam Vec3    Position type exposing .x / .y / .z members.
 * @param positions   Particle positions in world / local space.
 * @param particleIds Particle IDs, one-to-one with positions.
 * @return Tuple of
 *         - AdjacencyList built from finite Delaunay edges,
 *         - per-particle volume map (volume = -1 for unbounded cells),
 *         - per-particle vertex buffer (empty for unbounded cells).
 */
template <typename Vec3>
std::tuple<
    AdjacencyList,
    std::unordered_map<uint32_t, float>,
    std::unordered_map<uint32_t, std::vector<glm::vec3>>
>
triangulateVoronoiCells(
    const std::vector<Vec3>&     positions,
    const std::vector<uint32_t>& particleIds)
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

    // Build an O(1) position → index lookup so the subsequent vertex-handle
    // mapping is O(n) rather than O(n²).  The previous inner linear scan was
    // unnoticeable for small foams but causes multi-second stalls for large
    // particle grids (e.g. the floor plane with ~31k particles).
    struct TripleHash {
        size_t operator()(const std::tuple<double, double, double>& t) const noexcept {
            size_t h = 0;
            auto combine = [&](double v) {
                h ^= std::hash<double>{}(v) + 0x9e3779b9ull + (h << 6) + (h >> 2);
            };
            combine(std::get<0>(t));
            combine(std::get<1>(t));
            combine(std::get<2>(t));
            return h;
        }
    };
    std::unordered_map<std::tuple<double, double, double>, size_t, TripleHash> posToIndex;
    posToIndex.reserve(numParticles);
    for (size_t i = 0; i < numParticles; ++i)
        posToIndex[{positions[i].x, positions[i].y, positions[i].z}] = i;

    for (auto vh = dt.finite_vertices_begin();
              vh != dt.finite_vertices_end(); ++vh) {
        const auto& p = vh->point();
        auto it = posToIndex.find({p.x(), p.y(), p.z()});
        if (it != posToIndex.end())
            vertexToIndex[vh] = it->second;
    }

    // ------------------------------------------------------------------
    // 2. Adjacency list from finite Delaunay edges
    // ------------------------------------------------------------------
    AdjacencyList adjList(particleIds);

    for (auto eit = dt.finite_edges_begin();
              eit != dt.finite_edges_end(); ++eit) {
        auto cell = eit->first;
        auto v1   = cell->vertex(eit->second);
        auto v2   = cell->vertex(eit->third);
        if (vertexToIndex.count(v1) && vertexToIndex.count(v2)) {
            uint32_t id1 = particleIds[vertexToIndex[v1]];
            uint32_t id2 = particleIds[vertexToIndex[v2]];
            adjList.addNodeEdges(id1, {id2});
        }
    }

    // ------------------------------------------------------------------
    // 3. Per-particle Voronoi volume + vertex buffer
    // ------------------------------------------------------------------
    std::unordered_map<uint32_t, float>                  volumeMap;
    std::unordered_map<uint32_t, std::vector<glm::vec3>> voronoiVertices;
    volumeMap.reserve(numParticles);
    voronoiVertices.reserve(numParticles);

    for (auto vit = dt.finite_vertices_begin();
              vit != dt.finite_vertices_end(); ++vit) {
        if (!vertexToIndex.count(vit)) continue;

        size_t   i  = vertexToIndex[vit];
        uint32_t id = particleIds[i];

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
#endif // !__CUDACC__

} // namespace DynamicFoam::Sim2D
