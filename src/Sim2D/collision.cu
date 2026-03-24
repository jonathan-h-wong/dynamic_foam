// =============================================================================
// collision.cu
// Broadphase, narrowphase dual BVH traversal, and exact GJK+EPA Voronoi-cell
// collision detection for dynamic foam simulation.
//
// Detection pipeline:
//   1. Broadphase  – world-space AABB pair overlap (O(N²) foam pairs).
//   2. Narrowphase – dual BVH traversal in local space.  One foam's BVH boxes
//                    are transformed into the other foam's local space at each
//                    node so the traversal operates without an explicit world
//                    space expansion.
//   3. Exact       – GJK to test penetration; EPA to extract contact normal,
//                    depth, and contact point for every confirmed collision.
// =============================================================================

#define NOMINMAX
#include "dynamic_foam/Sim2D/collision.h"
#include "dynamic_foam/Sim2D/components.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <optional>
#include <stack>
#include <vector>
#include <utility>

namespace DynamicFoam::Sim2D {

// =============================================================================
// Section 1 – AABB helpers
// =============================================================================

namespace {

inline bool aabbOverlap(const AABB& a, const AABB& b) noexcept {
    return (a.max_pt.x >= b.min_pt.x) & (a.min_pt.x <= b.max_pt.x) &
           (a.max_pt.y >= b.min_pt.y) & (a.min_pt.y <= b.max_pt.y) &
           (a.max_pt.z >= b.min_pt.z) & (a.min_pt.z <= b.max_pt.z);
}

/**
 * Conservative AABB of `box` after applying transform `m`.
 * Transforms all 8 corners and computes the tight enclosing AABB.
 */
inline AABB transformAABB(const AABB& box, const glm::mat4& m) noexcept {
    const glm::vec3 mn = box.min_pt;
    const glm::vec3 mx = box.max_pt;

    glm::vec3 corners[8] = {
        glm::vec3(m * glm::vec4(mn.x, mn.y, mn.z, 1.f)),
        glm::vec3(m * glm::vec4(mx.x, mn.y, mn.z, 1.f)),
        glm::vec3(m * glm::vec4(mn.x, mx.y, mn.z, 1.f)),
        glm::vec3(m * glm::vec4(mx.x, mx.y, mn.z, 1.f)),
        glm::vec3(m * glm::vec4(mn.x, mn.y, mx.z, 1.f)),
        glm::vec3(m * glm::vec4(mx.x, mn.y, mx.z, 1.f)),
        glm::vec3(m * glm::vec4(mn.x, mx.y, mx.z, 1.f)),
        glm::vec3(m * glm::vec4(mx.x, mx.y, mx.z, 1.f)),
    };

    AABB result(corners[0], corners[0]);
    for (int i = 1; i < 8; ++i) {
        result.min_pt = glm::min(result.min_pt, corners[i]);
        result.max_pt = glm::max(result.max_pt, corners[i]);
    }
    return result;
}

// =============================================================================
// Section 2 – BVH host copy
// =============================================================================

/**
 * Copy `2*n - 1` BVH nodes from the GPU into a host vector.
 * The BVH stores (n-1) internal nodes followed by n leaf nodes.
 */
std::vector<BVHNode> bvhToHost(const BVH& bvh) {
    const int n = bvh.num_primitives();
    if (n <= 0) return {};
    const int total = 2 * n - 1;
    std::vector<BVHNode> nodes(total);
    cudaMemcpy(nodes.data(), bvh.export_nodes(),
               total * sizeof(BVHNode), cudaMemcpyDeviceToHost);
    return nodes;
}

// =============================================================================
// Section 3 – Dual BVH traversal (CPU)
// =============================================================================

/**
 * Traverse two host-side BVH trees simultaneously to collect all pairs of
 * leaf primitives whose bounding boxes overlap.
 *
 * @param nodesA  Host nodes for foam A (local-A space).
 * @param nodesB  Host nodes for foam B (local-B space).
 * @param bToA    Transform that converts a point from B-local → A-local:
 *                  bToA = inverse(transformA) * transformB
 * @param pairs   Output: (primIdxA, primIdxB) pairs that survived the cull.
 */
void dualBVHTraversal(
    const std::vector<BVHNode>& nodesA,
    const std::vector<BVHNode>& nodesB,
    const glm::mat4&            bToA,
    std::vector<std::pair<int,int>>& pairs)
{
    if (nodesA.empty() || nodesB.empty()) return;

    struct NodePair { int a, b; };
    std::stack<NodePair> stack;
    stack.push({0, 0});

    while (!stack.empty()) {
        auto [na, nb] = stack.top();
        stack.pop();

        const BVHNode& nodeA = nodesA[na];
        const BVHNode& nodeB = nodesB[nb];

        // Transform nodeB's bounding box into A-local space and check overlap.
        const AABB bboxBinA = transformAABB(nodeB.bbox, bToA);
        if (!aabbOverlap(nodeA.bbox, bboxBinA)) continue;

        const bool leafA = nodeA.prim_idx >= 0;
        const bool leafB = nodeB.prim_idx >= 0;

        if (leafA && leafB) {
            pairs.emplace_back(nodeA.prim_idx, nodeB.prim_idx);
        } else if (leafA) {
            // Descend only B.
            if (nodeB.left  >= 0) stack.push({na, nodeB.left});
            if (nodeB.right >= 0) stack.push({na, nodeB.right});
        } else if (leafB) {
            // Descend only A.
            if (nodeA.left  >= 0) stack.push({nodeA.left,  nb});
            if (nodeA.right >= 0) stack.push({nodeA.right, nb});
        } else {
            // Both internal: descend both (4 child pairs).
            if (nodeA.left  >= 0 && nodeB.left  >= 0) stack.push({nodeA.left,  nodeB.left});
            if (nodeA.left  >= 0 && nodeB.right >= 0) stack.push({nodeA.left,  nodeB.right});
            if (nodeA.right >= 0 && nodeB.left  >= 0) stack.push({nodeA.right, nodeB.left});
            if (nodeA.right >= 0 && nodeB.right >= 0) stack.push({nodeA.right, nodeB.right});
        }
    }
}

// =============================================================================
// Section 4 – GJK (Gilbert–Johnson–Keerthi, 3D)
// =============================================================================

/**
 * Support function: furthest vertex of shape `verts` along direction `dir`.
 */
inline glm::vec3 supportPoint(
    const std::vector<glm::vec3>& verts,
    const glm::vec3& dir) noexcept
{
    float best = -FLT_MAX;
    glm::vec3 pt{};
    for (const auto& v : verts) {
        float d = glm::dot(v, dir);
        if (d > best) { best = d; pt = v; }
    }
    return pt;
}

/**
 * Minkowski-difference support: support_A(dir) - support_B(-dir).
 */
inline glm::vec3 minkSupport(
    const std::vector<glm::vec3>& vertsA,
    const std::vector<glm::vec3>& vertsB,
    const glm::vec3& dir) noexcept
{
    return supportPoint(vertsA, dir) - supportPoint(vertsB, -dir);
}

// -----------------------------------------------------------------------
// Simplex and direction-update routines
// Convention: pts[0] is always the most recently added point (called 'A').
// -----------------------------------------------------------------------

struct Simplex {
    std::array<glm::vec3, 4> pts{};
    int size = 0;

    void push(const glm::vec3& p) noexcept {
        // Shift down, inject new point at front.
        switch (size) {
            case 3: pts[3] = pts[2]; [[fallthrough]];
            case 2: pts[2] = pts[1]; [[fallthrough]];
            case 1: pts[1] = pts[0]; [[fallthrough]];
            default: break;
        }
        pts[0] = p;
        size = std::min(size + 1, 4);
    }
};

inline bool sameDir(const glm::vec3& a, const glm::vec3& b) noexcept {
    return glm::dot(a, b) > 0.f;
}

// Returns true if the simplex contains the origin, updates `dir` otherwise.
bool doLineSimplex(Simplex& s, glm::vec3& dir) noexcept {
    const glm::vec3& A = s.pts[0];
    const glm::vec3& B = s.pts[1];
    const glm::vec3 AB = B - A;
    const glm::vec3 AO = -A;

    if (sameDir(AB, AO)) {
        // Origin is between A and B projected onto the line.
        dir = glm::cross(glm::cross(AB, AO), AB);
    } else {
        // Closest to A alone.
        s.size = 1;
        dir = AO;
    }
    return false;
}

bool doTriangleSimplex(Simplex& s, glm::vec3& dir) noexcept {
    const glm::vec3& A = s.pts[0];
    const glm::vec3& B = s.pts[1];
    const glm::vec3& C = s.pts[2];

    const glm::vec3 AB  = B - A;
    const glm::vec3 AC  = C - A;
    const glm::vec3 AO  = -A;
    const glm::vec3 ABC = glm::cross(AB, AC);  // Face normal

    if (sameDir(glm::cross(ABC, AC), AO)) {
        // Origin is outside the AC edge region.
        if (sameDir(AC, AO)) {
            s.pts[1] = s.pts[2];   // keep A and C
            s.size   = 2;
            dir      = glm::cross(glm::cross(AC, AO), AC);
        } else {
            s.size = 2;             // keep A and B only
            return doLineSimplex(s, dir);
        }
    } else if (sameDir(glm::cross(AB, ABC), AO)) {
        // Origin is outside the AB edge region.
        s.size = 2;                 // keep A and B only
        return doLineSimplex(s, dir);
    } else {
        // Origin is above or below the triangle face.
        if (sameDir(ABC, AO)) {
            dir = ABC;
        } else {
            // Flip to keep CCW winding when viewed from origin.
            std::swap(s.pts[1], s.pts[2]);
            dir = -ABC;
        }
    }
    return false;
}

bool doTetrahedronSimplex(Simplex& s, glm::vec3& dir) noexcept {
    const glm::vec3& A = s.pts[0];
    const glm::vec3& B = s.pts[1];
    const glm::vec3& C = s.pts[2];
    const glm::vec3& D = s.pts[3];

    const glm::vec3 AB = B - A;
    const glm::vec3 AC = C - A;
    const glm::vec3 AD = D - A;
    const glm::vec3 AO = -A;

    // Normals of the three faces that include A (outward from tetrahedron).
    const glm::vec3 nABC = glm::cross(AB, AC);
    const glm::vec3 nACD = glm::cross(AC, AD);
    const glm::vec3 nADB = glm::cross(AD, AB);

    // Correct outward orientation: normal must point away from the 4th vertex.
    const glm::vec3 nABC_out = sameDir(nABC, AD) ? -nABC : nABC;
    const glm::vec3 nACD_out = sameDir(nACD, AB) ? -nACD : nACD;
    const glm::vec3 nADB_out = sameDir(nADB, AC) ? -nADB : nADB;

    if (sameDir(nABC_out, AO)) {
        // Origin is outside face ABC – reduce to triangle ABC.
        s.size = 3;  // pts[0]=A, pts[1]=B, pts[2]=C
        // Ensure the triangle winding has nABC_out as its face normal.
        if (!sameDir(glm::cross(B - A, C - A), nABC_out)) std::swap(s.pts[1], s.pts[2]);
        return doTriangleSimplex(s, dir);
    }
    if (sameDir(nACD_out, AO)) {
        // Origin is outside face ACD – reduce to triangle ACD.
        s.pts[1] = s.pts[2];  // B <- C
        s.pts[2] = s.pts[3];  // C <- D
        s.size   = 3;
        if (!sameDir(glm::cross(s.pts[1]-A, s.pts[2]-A), nACD_out)) std::swap(s.pts[1], s.pts[2]);
        return doTriangleSimplex(s, dir);
    }
    if (sameDir(nADB_out, AO)) {
        // Origin is outside face ADB – reduce to triangle ADB.
        s.pts[2] = s.pts[1];  // C <- B
        s.pts[1] = s.pts[3];  // B <- D
        s.size   = 3;
        if (!sameDir(glm::cross(s.pts[1]-A, s.pts[2]-A), nADB_out)) std::swap(s.pts[1], s.pts[2]);
        return doTriangleSimplex(s, dir);
    }

    // Origin is inside the tetrahedron.
    return true;
}

bool doSimplex(Simplex& s, glm::vec3& dir) noexcept {
    switch (s.size) {
        case 2:  return doLineSimplex(s, dir);
        case 3:  return doTriangleSimplex(s, dir);
        case 4:  return doTetrahedronSimplex(s, dir);
        default: return false;
    }
}

constexpr int   GJK_MAX_ITER = 64;
constexpr float GJK_EPSILON  = 1e-6f;

/**
 * Run GJK between two convex shapes given as world-space vertex lists.
 *
 * @param vertsA   World-space vertices of shape A.
 * @param vertsB   World-space vertices of shape B.
 * @param simplex  Output: the terminating simplex (a tetrahedron if
 *                 penetrating, for subsequent EPA use).
 * @return  true if the two shapes are penetrating (origin ∈ Minkowski diff).
 */
bool gjk(
    const std::vector<glm::vec3>& vertsA,
    const std::vector<glm::vec3>& vertsB,
    Simplex& simplex)
{
    // Initial direction: centroid difference.
    glm::vec3 dirA{}, dirB{};
    for (const auto& v : vertsA) dirA += v;
    for (const auto& v : vertsB) dirB += v;
    dirA /= static_cast<float>(vertsA.size());
    dirB /= static_cast<float>(vertsB.size());

    glm::vec3 dir = dirA - dirB;
    if (glm::length(dir) < GJK_EPSILON) dir = glm::vec3(1.f, 0.f, 0.f);

    simplex = Simplex{};
    simplex.push(minkSupport(vertsA, vertsB, dir));

    dir = -simplex.pts[0];  // toward origin

    for (int iter = 0; iter < GJK_MAX_ITER; ++iter) {
        if (glm::length(dir) < GJK_EPSILON) return true;  // origin on boundary

        glm::vec3 A = minkSupport(vertsA, vertsB, dir);
        if (glm::dot(A, dir) < 0.f) return false;  // no intersection

        simplex.push(A);

        if (doSimplex(simplex, dir)) return true;
    }
    return false;
}

// =============================================================================
// Section 5 – EPA (Expanding Polytope Algorithm, 3D)
// =============================================================================

/**
 * Ensure the tetrahedron in `simplex` is non-degenerate so EPA can start.
 * If the GJK simplex has fewer than 4 points (degenerate penetration case),
 * we expand it by probing along axis-aligned directions until we have a valid
 * initial polytope.
 */
void expandToTetrahedron(
    Simplex& s,
    const std::vector<glm::vec3>& vertsA,
    const std::vector<glm::vec3>& vertsB)
{
    const glm::vec3 axes[6] = {
        { 1.f,0.f,0.f},{-1.f,0.f,0.f},{ 0.f,1.f,0.f},{ 0.f,-1.f,0.f},{ 0.f,0.f,1.f},{ 0.f,0.f,-1.f}
    };

    while (s.size < 4) {
        for (const auto& ax : axes) {
            glm::vec3 p = minkSupport(vertsA, vertsB, ax);
            // Only add if it's not a duplicate of an existing simplex point.
            bool dup = false;
            for (int k = 0; k < s.size; ++k) {
                if (glm::length(p - s.pts[k]) < GJK_EPSILON) { dup = true; break; }
            }
            if (!dup) { s.push(p); break; }
        }
    }
}

struct EPAFace {
    glm::vec3 a, b, c;     ///< Vertices (CCW when viewed from outside).
    glm::vec3 normal;      ///< Outward unit normal.
    float     distance;    ///< Signed distance from origin along normal (>= 0).
};

inline EPAFace makeFace(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) noexcept {
    EPAFace f;
    f.a = a; f.b = b; f.c = c;
    glm::vec3 n = glm::cross(b - a, c - a);
    float len   = glm::length(n);
    f.normal    = (len > 1e-9f) ? (n / len) : glm::vec3(0.f, 0.f, 1.f);
    f.distance  = glm::dot(f.normal, a);
    // Origin may be on either side for degenerate cases; take the positive side.
    if (f.distance < 0.f) {
        f.normal    = -f.normal;
        f.distance  = -f.distance;
        std::swap(f.b, f.c);   // Flip winding to match corrected normal.
    }
    return f;
}

constexpr int   EPA_MAX_ITER = 64;
constexpr float EPA_EPSILON  = 1e-4f;

/**
 * Run EPA starting from the terminating GJK tetrahedron simplex.
 *
 * @param simplex  4-point penetrating GJK simplex.
 * @param vertsA   World-space vertices of shape A.
 * @param vertsB   World-space vertices of shape B.
 * @param normal   Output: anti-penetration direction (B → A), unit length.
 * @param depth    Output: penetration depth (positive).
 * @return false if EPA failed to converge (degenerate geometry).
 */
bool epa(
    Simplex&                      simplex,
    const std::vector<glm::vec3>& vertsA,
    const std::vector<glm::vec3>& vertsB,
    glm::vec3&                    normal,
    float&                        depth)
{
    // Ensure we have a full tetrahedron.
    expandToTetrahedron(simplex, vertsA, vertsB);

    // Build initial polytope from the 4 tetrahedron faces.
    // Winding: each face's normal must point outward (away from origin).
    const glm::vec3& A = simplex.pts[0];
    const glm::vec3& B = simplex.pts[1];
    const glm::vec3& C = simplex.pts[2];
    const glm::vec3& D = simplex.pts[3];

    std::vector<EPAFace> faces;
    faces.reserve(64);
    faces.push_back(makeFace(A, B, C));
    faces.push_back(makeFace(A, C, D));
    faces.push_back(makeFace(A, D, B));
    faces.push_back(makeFace(B, D, C));   // Base face (opposite to A)

    for (int iter = 0; iter < EPA_MAX_ITER; ++iter) {
        // Find the face closest to the origin.
        int   minIdx  = 0;
        float minDist = FLT_MAX;
        for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
            if (faces[i].distance < minDist) {
                minDist = faces[i].distance;
                minIdx  = i;
            }
        }
        const EPAFace& closest = faces[minIdx];

        // Support point in the direction of the closest face normal.
        glm::vec3 sup = minkSupport(vertsA, vertsB, closest.normal);
        const float expansion = glm::dot(closest.normal, sup) - closest.distance;

        if (expansion < EPA_EPSILON) {
            // Converged: this face is the minimum-depth boundary.
            normal = closest.normal;
            depth  = closest.distance;
            return true;
        }

        // Find all faces visible from `sup` (those whose outward normal
        // points toward `sup`).  Collect silhouette edges (edges shared
        // between a visible and an invisible face).
        struct Edge { glm::vec3 a, b; };
        std::vector<Edge> silhouette;

        std::vector<EPAFace> newFaces;
        newFaces.reserve(faces.size());

        for (const auto& f : faces) {
            if (glm::dot(f.normal, sup - f.a) > 0.f) {
                // Face is visible: find its edges adjacent to invisible faces.
                // Since we remove all visible faces, we just collect every edge
                // and then keep only those that appear exactly once (boundary).
                silhouette.push_back({f.a, f.b});
                silhouette.push_back({f.b, f.c});
                silhouette.push_back({f.c, f.a});
            } else {
                newFaces.push_back(f);
            }
        }

        if (silhouette.empty()) {
            // All faces visible – degenerate case, fall back to best face.
            normal = closest.normal;
            depth  = closest.distance;
            return true;
        }

        // Remove duplicate / interior edges (edges shared between two visible
        // faces).  An edge e=(a,b) is a silhouette edge if and only if the
        // reverse edge (b,a) does NOT appear in the collected list.
        auto edgeEq = [](const glm::vec3& p, const glm::vec3& q) {
            return glm::length(p - q) < 1e-7f;
        };

        for (int i = 0; i < static_cast<int>(silhouette.size()); ++i) {
            bool isInterior = false;
            for (int j = 0; j < static_cast<int>(silhouette.size()); ++j) {
                if (i == j) continue;
                if (edgeEq(silhouette[i].a, silhouette[j].b) &&
                    edgeEq(silhouette[i].b, silhouette[j].a)) {
                    isInterior = true; break;
                }
            }
            if (!isInterior) {
                // Silhouette edge: build a new face connecting the edge to sup.
                newFaces.push_back(makeFace(silhouette[i].a, silhouette[i].b, sup));
            }
        }

        faces = std::move(newFaces);
        if (faces.empty()) {
            // Should not happen in a valid polytope.
            return false;
        }
    }

    // Did not converge within iteration budget – return best approximation.
    int   minIdx  = 0;
    float minDist = FLT_MAX;
    for (int i = 0; i < static_cast<int>(faces.size()); ++i) {
        if (faces[i].distance < minDist) {
            minDist = faces[i].distance;
            minIdx  = i;
        }
    }
    normal = faces[minIdx].normal;
    depth  = faces[minIdx].distance;
    return true;
}

// =============================================================================
// Section 6 – Per-pair exact test
// =============================================================================

/**
 * Given two local-space vertex buffers and the foam world transforms, run
 * GJK + EPA and fill in a FoamCollision if penetrating.
 *
 * Contact-point convention: midpoint of the deepest supporting points on each
 * shape in the penetration-normal direction.
 */
std::optional<FoamCollision> testParticlePair(
    int          foamIdA,
    int          foamIdB,
    entt::entity particleA,
    entt::entity particleB,
    const std::vector<glm::vec3>& vertsA,  // local-A space
    const std::vector<glm::vec3>& vertsB,  // local-B space
    const glm::mat4& transformA,
    const glm::mat4& transformB)
{
    if (vertsA.empty() || vertsB.empty()) return std::nullopt;

    // Lift vertices into world space.
    auto toWorld = [](const std::vector<glm::vec3>& locals,
                      const glm::mat4&               xform)
    {
        std::vector<glm::vec3> world;
        world.reserve(locals.size());
        for (const auto& v : locals)
            world.push_back(glm::vec3(xform * glm::vec4(v, 1.f)));
        return world;
    };

    const auto worldA = toWorld(vertsA, transformA);
    const auto worldB = toWorld(vertsB, transformB);

    Simplex simplex;
    if (!gjk(worldA, worldB, simplex)) return std::nullopt;

    glm::vec3 normal;
    float     depth;
    if (!epa(simplex, worldA, worldB, normal, depth)) return std::nullopt;

    if (depth < 0.f) return std::nullopt;

    // Contact point: midpoint between the two deepest support points.
    const glm::vec3 ptOnA = supportPoint(worldA,  normal);
    const glm::vec3 ptOnB = supportPoint(worldB, -normal);
    const glm::vec3 contact = 0.5f * (ptOnA + ptOnB);

    FoamCollision col;
    col.foamIdA          = foamIdA;
    col.foamIdB          = foamIdB;
    col.particleA        = particleA;
    col.particleB        = particleB;
    col.normal           = normal;          // world-space, B → A
    col.penetrationDepth = depth;
    col.contactPoint     = contact;
    return col;
}

} // anonymous namespace

// =============================================================================
// Section 7 – Public entry point
// =============================================================================

std::vector<FoamCollision> detectCollisions(
    const std::unordered_map<int, AABB>&                         foamAABBs,
    const std::unordered_map<int, BVH>&                          foamBVHs,
    const std::unordered_map<int, glm::mat4>&                    foamTransforms,
    const std::unordered_map<int, AdjacencyList<entt::entity>>&  foamAdjacencyLists,
    const entt::registry&                                        particleRegistry)
{
    std::vector<FoamCollision> results;

    // Collect foam IDs that have all necessary data.
    std::vector<int> foamIds;
    foamIds.reserve(foamAABBs.size());
    for (const auto& [id, _] : foamAABBs) {
        if (foamBVHs.count(id) &&
            foamTransforms.count(id) &&
            foamAdjacencyLists.count(id) &&
            foamBVHs.at(id).num_primitives() > 0)
        {
            foamIds.push_back(id);
        }
    }

    const int N = static_cast<int>(foamIds.size());

    // Cache host-side BVH node arrays (one GPU download per foam per step).
    std::unordered_map<int, std::vector<BVHNode>> hostBVHs;
    hostBVHs.reserve(N);
    for (int id : foamIds)
        hostBVHs[id] = bvhToHost(foamBVHs.at(id));

    // Cache ordered particle lists (used to resolve prim_idx → entity).
    std::unordered_map<int, std::vector<entt::entity>> orderedParticles;
    orderedParticles.reserve(N);
    for (int id : foamIds)
        orderedParticles[id] = foamAdjacencyLists.at(id).getOrderedNodeIds();

    // -------------------------------------------------------------------------
    // Phase 1: Broadphase – world-space AABB overlap for every foam pair.
    // -------------------------------------------------------------------------
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            const int idA = foamIds[i];
            const int idB = foamIds[j];

            if (!aabbOverlap(foamAABBs.at(idA), foamAABBs.at(idB))) continue;

            // -----------------------------------------------------------------
            // Phase 2: Narrowphase – dual BVH traversal in local space.
            // Transform B-local boxes into A-local space.
            // -----------------------------------------------------------------
            const glm::mat4& txA = foamTransforms.at(idA);
            const glm::mat4& txB = foamTransforms.at(idB);
            const glm::mat4  bToA = glm::inverse(txA) * txB;

            std::vector<std::pair<int,int>> leafPairs;
            dualBVHTraversal(hostBVHs.at(idA), hostBVHs.at(idB), bToA, leafPairs);

            // -----------------------------------------------------------------
            // Phase 3: Exact – GJK + EPA per candidate leaf pair.
            // -----------------------------------------------------------------
            const auto& ordA = orderedParticles.at(idA);
            const auto& ordB = orderedParticles.at(idB);

            for (const auto& [primA, primB] : leafPairs) {
                if (primA < 0 || primA >= static_cast<int>(ordA.size())) continue;
                if (primB < 0 || primB >= static_cast<int>(ordB.size())) continue;

                const entt::entity eA = ordA[primA];
                const entt::entity eB = ordB[primB];

                const auto* cpA = particleRegistry.try_get<ParticleVertices>(eA);
                const auto* cpB = particleRegistry.try_get<ParticleVertices>(eB);
                if (!cpA || !cpB) continue;

                auto col = testParticlePair(
                    idA, idB, eA, eB,
                    cpA->vertices, cpB->vertices,
                    txA, txB);

                if (col) results.push_back(std::move(*col));
            }
        }
    }

    return results;
}

} // namespace DynamicFoam::Sim2D
