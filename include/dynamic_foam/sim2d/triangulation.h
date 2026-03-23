#pragma once

#include <vector>
#include <unordered_map>
#include <tuple>
#include <array>
#include <cstdint>
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
// Half-Edge Convex Polytope Data Structures
// ============================================================================

/**
 * A single directed half-edge in a half-edge mesh.
 * Each undirected edge is represented by two opposite half-edges (twins).
 */
struct HalfEdge {
    int vertex;  ///< Index into ConvexPolytope::vertices (destination vertex)
    int twin;    ///< Index of the opposite half-edge (-1 if boundary / unmatched)
    int next;    ///< Index of the next half-edge around the same face (CCW)
    int face;    ///< Index of the owning face in ConvexPolytope::faces
};

/**
 * A triangular face of the convex polytope, carrying precomputed geometry
 * needed by the EPA (Expanding Polytope Algorithm).
 */
struct Face {
    int halfEdge;       ///< Index of any half-edge belonging to this face
    glm::vec3 normal;   ///< Outward-facing unit normal
    float distance;     ///< Signed distance from origin (dot(normal, vertex))
};

/**
 * A convex polytope represented as a half-edge mesh.
 * Constructed from the Voronoi cell of a simulation particle and used as
 * the collision primitive for GJK-EPA narrow-phase detection.
 */
struct ConvexPolytope {
    std::vector<glm::vec3> vertices;   ///< Unique vertex positions (local space)
    std::vector<HalfEdge>  halfEdges;  ///< Directed half-edges (triangulated)
    std::vector<Face>      faces;      ///< Triangular faces with normals/distances
};

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

inline uint64_t encodeEdge(int src, int dst)
{
    return (static_cast<uint64_t>(static_cast<uint32_t>(src)) << 32)
         |  static_cast<uint64_t>(static_cast<uint32_t>(dst));
}

inline ConvexPolytope buildConvexPolytope(
    const std::vector<glm::vec3>&          verts,
    const std::vector<std::array<int, 3>>& triangles)
{
    ConvexPolytope poly;
    poly.vertices = verts;

    std::unordered_map<uint64_t, int> edgeMap;
    edgeMap.reserve(triangles.size() * 3);

    for (int fi = 0; fi < static_cast<int>(triangles.size()); ++fi) {
        const auto& tri = triangles[fi];
        int heBase = static_cast<int>(poly.halfEdges.size());

        const glm::vec3& v0 = verts[tri[0]];
        const glm::vec3& v1 = verts[tri[1]];
        const glm::vec3& v2 = verts[tri[2]];

        glm::vec3 n = glm::cross(v1 - v0, v2 - v0);
        float len   = glm::length(n);
        n = (len > 1e-9f) ? (n / len) : glm::vec3(0.0f, 0.0f, 1.0f);
        float dist  = glm::dot(n, v0);

        Face face;
        face.halfEdge = heBase;
        face.normal   = n;
        face.distance = dist;
        poly.faces.push_back(face);

        for (int i = 0; i < 3; ++i) {
            HalfEdge he;
            he.vertex = tri[(i + 1) % 3];
            he.next   = heBase + (i + 1) % 3;
            he.face   = fi;
            he.twin   = -1;
            poly.halfEdges.push_back(he);

            int src = tri[i];
            int dst = tri[(i + 1) % 3];
            edgeMap[encodeEdge(src, dst)] = heBase + i;
        }
    }

    for (int fi = 0; fi < static_cast<int>(triangles.size()); ++fi) {
        const auto& tri = triangles[fi];
        int heBase = fi * 3;
        for (int i = 0; i < 3; ++i) {
            int src = tri[i];
            int dst = tri[(i + 1) % 3];
            auto it = edgeMap.find(encodeEdge(dst, src));
            if (it != edgeMap.end())
                poly.halfEdges[heBase + i].twin = it->second;
        }
    }

    return poly;
}

} // namespace detail

// ============================================================================
// Triangulation
// ============================================================================

/**
 * @brief Build Delaunay triangulation, extract adjacency / volumes, and
 *        construct a half-edge ConvexPolytope for every Voronoi cell.
 *
 * For each particle the Voronoi cell is computed via the dual of the 3-D
 * Delaunay triangulation.  Each cell is then tessellated into triangles and
 * wrapped in a ConvexPolytope whose HalfEdge connectivity enables the EPA
 * to cheaply walk around faces during collision resolution.
 *
 * Boundary particles (incident to at least one infinite Delaunay cell) are
 * assigned volume -1 and an empty ConvexPolytope.
 *
 * @tparam T       Particle ID type (e.g. int).
 * @tparam Vec3    Position type exposing .x / .y / .z members.
 * @param positions   Particle positions in world / local space.
 * @param particleIds Particle IDs, one-to-one with positions.
 * @return Tuple of
 *         - AdjacencyList built from finite Delaunay edges,
 *         - per-particle volume map (volume = -1 for unbounded cells),
 *         - per-particle ConvexPolytope (empty for unbounded cells).
 */
template <typename T, typename Vec3>
std::tuple<
    AdjacencyList<T>,
    std::unordered_map<T, float>,
    std::unordered_map<T, ConvexPolytope>
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
    // 3. Per-particle Voronoi volume + ConvexPolytope
    // ------------------------------------------------------------------
    std::unordered_map<T, float>          volumeMap;
    std::unordered_map<T, ConvexPolytope> voronoiPolytopes;
    volumeMap.reserve(numParticles);
    voronoiPolytopes.reserve(numParticles);

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
                volumeMap[id]        = -1.0f;
                voronoiPolytopes[id] = ConvexPolytope{};
                continue;
            }
        }

        std::vector<glm::vec3> mergedVerts;
        std::map<std::tuple<float, float, float>, int> vertIdx;
        std::vector<std::array<int, 3>> triangles;

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
                int i0 = faceIdx[0];
                int i1 = faceIdx[j];
                int i2 = faceIdx[j + 1];

                const glm::vec3& p0 = mergedVerts[i0];
                const glm::vec3& p1 = mergedVerts[i1];
                const glm::vec3& p2 = mergedVerts[i2];
                glm::vec3 n = glm::cross(p1 - p0, p2 - p0);
                if (glm::dot(n, p0 - genPt) < 0.0f)
                    std::swap(i1, i2);

                triangles.push_back({i0, i1, i2});
                totalVolume += detail::signedTetVolume(genPt, p0, p1, p2);
            }
        }

        volumeMap[id]        = std::abs(totalVolume);
        voronoiPolytopes[id] = detail::buildConvexPolytope(mergedVerts, triangles);
    }

    return {adjList, volumeMap, voronoiPolytopes};
}

} // namespace DynamicFoam::Sim2D
