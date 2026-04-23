// =============================================================================
// collision.cu
// GPU broadphase and narrowphase collision detection.
//
// See collision.cuh for a description of each phase.
// =============================================================================

#define NOMINMAX
#include "dynamic_foam/Sim2D/collision.cuh"

#include <glm/gtc/matrix_inverse.hpp>

#include <algorithm>
#include <array>
#include <cfloat>
#include <optional>
#include <stack>
#include <stdexcept>

namespace DynamicFoam::Sim2D {

// =============================================================================
// Phase 1 — transform local-space foam AABBs to world space
// =============================================================================

__global__ void k_col_transform_foam_aabbs(
    const AABB*      local_aabbs,
    const glm::mat4* transforms,
    AABB*            world_aabbs_out,
    int              num_foams)
{
    const int fid = blockIdx.x * blockDim.x + threadIdx.x;
    if (fid >= num_foams) return;

    const AABB&      local = local_aabbs[fid];
    const glm::mat4& tx    = transforms[fid];
    const glm::vec3  mn    = local.min_pt;
    const glm::vec3  mx    = local.max_pt;

    // Transform all 8 corners and compute the tight enclosing AABB.
    glm::vec3 c0   = glm::vec3(tx * glm::vec4(mn, 1.f));
    glm::vec3 wmin = c0, wmax = c0;
    for (int j = 1; j < 8; ++j) {
        glm::vec3 c(j & 1 ? mx.x : mn.x,
                    j & 2 ? mx.y : mn.y,
                    j & 4 ? mx.z : mn.z);
        c    = glm::vec3(tx * glm::vec4(c, 1.f));
        wmin = glm::min(wmin, c);
        wmax = glm::max(wmax, c);
    }
    world_aabbs_out[fid] = AABB(wmin, wmax);
}

// =============================================================================
// Phase 2 — broadphase: world-space AABB pair overlap
// =============================================================================

__global__ void k_col_broadphase_foam_pairs(
    const AABB* world_aabbs,
    const int*  foam_ids,
    FoamPair*   pairs_out,
    int*        pair_counter,
    int         num_foams)
{
    // Map the flat thread index to the upper-triangular pair (i, j), i < j.
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_pairs = num_foams * (num_foams - 1) / 2;
    if (tid >= num_pairs) return;

    // Recover (i, j) from the flat upper-triangular index.
    // Row i starts at index i*(2*N - i - 1)/2 for the N×N upper triangle.
    // Simple O(N) decode is fine — num_foams is tiny.
    int i = 0, j = 0, acc = 0;
    for (int row = 0; row < num_foams - 1; ++row) {
        const int row_len = num_foams - row - 1;
        if (acc + row_len > tid) {
            i = row;
            j = row + 1 + (tid - acc);
            break;
        }
        acc += row_len;
    }

    const int fid_a = foam_ids[i];
    const int fid_b = foam_ids[j];

    const AABB& a = world_aabbs[fid_a];
    const AABB& b = world_aabbs[fid_b];

    // Separating-axis AABB overlap test.
    if ((a.max_pt.x < b.min_pt.x) | (a.min_pt.x > b.max_pt.x) |
        (a.max_pt.y < b.min_pt.y) | (a.min_pt.y > b.max_pt.y) |
        (a.max_pt.z < b.min_pt.z) | (a.min_pt.z > b.max_pt.z))
        return;

    const int out_idx = atomicAdd(pair_counter, 1);
    pairs_out[out_idx] = { fid_a, fid_b };
}

// =============================================================================
// Phase 3 — narrowphase: dual BVH traversal in A-local space
// =============================================================================

// Stack capacity per thread — leans on register/local memory.
// A balanced binary BVH of depth d has at most 2*d nodes on the stack.
// With 2^15 = 32 K particles the depth is at most 15; 128 is very conservative.
static constexpr int kDualStackDepth = 128;

// Encodes a (nodeA, nodeB) pair onto a single 64-bit word for the GPU stack.
__device__ __forceinline__ uint64_t encodePair(int na, int nb) {
    return ((uint64_t)(unsigned)na << 32) | (unsigned)nb;
}
__device__ __forceinline__ void decodePair(uint64_t v, int& na, int& nb) {
    na = (int)(v >> 32);
    nb = (int)(v & 0xFFFFFFFFu);
}

// Transform an AABB by matrix m using the 8-corner method.
__device__ AABB d_transformAABB(const AABB& box, const glm::mat4& m) {
    const glm::vec3 mn = box.min_pt, mx = box.max_pt;
    glm::vec3 c0   = glm::vec3(m * glm::vec4(mn, 1.f));
    glm::vec3 wmin = c0, wmax = c0;
    for (int j = 1; j < 8; ++j) {
        glm::vec3 c(j & 1 ? mx.x : mn.x,
                    j & 2 ? mx.y : mn.y,
                    j & 4 ? mx.z : mn.z);
        c    = glm::vec3(m * glm::vec4(c, 1.f));
        wmin = glm::min(wmin, c);
        wmax = glm::max(wmax, c);
    }
    return AABB(wmin, wmax);
}

// AABB vs AABB overlap (device).
__device__ __forceinline__ bool d_aabbOverlap(const AABB& a, const AABB& b) {
    return (a.max_pt.x >= b.min_pt.x) & (a.min_pt.x <= b.max_pt.x) &
           (a.max_pt.y >= b.min_pt.y) & (a.min_pt.y <= b.max_pt.y) &
           (a.max_pt.z >= b.min_pt.z) & (a.min_pt.z <= b.max_pt.z);
}

__global__ void k_col_narrowphase_dual_bvh(
    const FoamPair*      broadphase_pairs,
    int                  num_broadphase_pairs,
    const BVHNode*       bvh_nodes,
    const int*           bvh_offsets,
    const int*           foam_particle_counts,
    const glm::mat4*     transforms,
    const glm::mat4*     inv_transforms,
    CollisionCandidate*  candidates_out,
    int*                 candidate_counter,
    int                  max_candidates)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_broadphase_pairs) return;

    const FoamPair pair   = broadphase_pairs[tid];
    const int      fid_a  = pair.foam_id_a;
    const int      fid_b  = pair.foam_id_b;

    const int n_a = foam_particle_counts[fid_a];
    const int n_b = foam_particle_counts[fid_b];
    if (n_a <= 0 || n_b <= 0) return;

    // BVH roots are stored starting at bvh_offsets[fid].
    // Internal nodes: [0, n-2], leaf nodes: [n-1, 2n-2] (flat layout per Karras).
    const BVHNode* bvhA = bvh_nodes + bvh_offsets[fid_a];
    const BVHNode* bvhB = bvh_nodes + bvh_offsets[fid_b];

    // bToA: transforms a point from B-local → A-local.
    //   bToA = inv(txA) * txB
    const glm::mat4 bToA = inv_transforms[fid_a] * transforms[fid_b];

    // Iterative dual traversal using a local stack of (nodeA, nodeB) pairs.
    uint64_t stack[kDualStackDepth];
    int      stack_ptr = 0;
    stack[stack_ptr++] = encodePair(0, 0);  // both trees start at root (index 0)

    while (stack_ptr > 0) {
        int na, nb;
        decodePair(stack[--stack_ptr], na, nb);

        const BVHNode& nodeA = bvhA[na];
        const BVHNode& nodeB = bvhB[nb];

        // Transform B's box into A-local space for the overlap test.
        const AABB bboxBinA = d_transformAABB(nodeB.bbox, bToA);
        if (!d_aabbOverlap(nodeA.bbox, bboxBinA)) continue;

        const bool leafA = (nodeA.prim_idx >= 0);
        const bool leafB = (nodeB.prim_idx >= 0);

        if (leafA && leafB) {
            // Emit a candidate (bounds-check against allocated capacity).
            const int out_idx = atomicAdd(candidate_counter, 1);
            if (out_idx < max_candidates) {
                candidates_out[out_idx] = {
                    fid_a, fid_b,
                    nodeA.prim_idx,
                    nodeB.prim_idx
                };
            }
        } else if (leafA) {
            // A is a leaf — descend B.
            if (nodeB.left  >= 0 && stack_ptr < kDualStackDepth - 1)
                stack[stack_ptr++] = encodePair(na, nodeB.left);
            if (nodeB.right >= 0 && stack_ptr < kDualStackDepth - 1)
                stack[stack_ptr++] = encodePair(na, nodeB.right);
        } else if (leafB) {
            // B is a leaf — descend A.
            if (nodeA.left  >= 0 && stack_ptr < kDualStackDepth - 1)
                stack[stack_ptr++] = encodePair(nodeA.left,  nb);
            if (nodeA.right >= 0 && stack_ptr < kDualStackDepth - 1)
                stack[stack_ptr++] = encodePair(nodeA.right, nb);
        } else {
            // Both internal — push all 4 child combinations.
            if (nodeA.left  >= 0 && nodeB.left  >= 0 && stack_ptr < kDualStackDepth - 1)
                stack[stack_ptr++] = encodePair(nodeA.left,  nodeB.left);
            if (nodeA.left  >= 0 && nodeB.right >= 0 && stack_ptr < kDualStackDepth - 1)
                stack[stack_ptr++] = encodePair(nodeA.left,  nodeB.right);
            if (nodeA.right >= 0 && nodeB.left  >= 0 && stack_ptr < kDualStackDepth - 1)
                stack[stack_ptr++] = encodePair(nodeA.right, nodeB.left);
            if (nodeA.right >= 0 && nodeB.right >= 0 && stack_ptr < kDualStackDepth - 1)
                stack[stack_ptr++] = encodePair(nodeA.right, nodeB.right);
        }
    }
}

// =============================================================================
// Phase 3b — entity resolve
// One thread per candidate: gather entity IDs from d_active_ids in place.
// =============================================================================

__global__ void k_col_resolve_entity_ids(
    CollisionCandidate* candidates,
    int                 num_candidates,
    const uint32_t*     d_active_ids,
    const int*          d_foam_active_start)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_candidates) return;

    CollisionCandidate& c = candidates[i];
    const int base_a = d_foam_active_start[c.foam_id_a];
    const int base_b = d_foam_active_start[c.foam_id_b];
    c.entity_id_a = d_active_ids[base_a + c.prim_idx_a];
    c.entity_id_b = d_active_ids[base_b + c.prim_idx_b];
}

// =============================================================================
// Host entry point — detectCandidates
// =============================================================================

std::vector<CollisionCandidate> detectCandidates(
    const GpuSlabAllocator&                   gpuSlab,
    const std::unordered_map<int, glm::mat4>& foamTransforms,
    const std::vector<int>&                   foamIds,
    int                                       maxCandidates)
{
    const int N = static_cast<int>(foamIds.size());
    if (N < 2) return {};

    // -------------------------------------------------------------------------
    // Upload per-foam transforms and inverse transforms to the GPU.
    // These are indexed directly by foam_id, so we allocate a contiguous array
    // of size max(foam_id)+1 matching the slab's own indexing convention.
    // -------------------------------------------------------------------------
    const int maxFoamId = *std::max_element(foamIds.begin(), foamIds.end());
    const int tableSize = maxFoamId + 1;

    std::vector<glm::mat4> h_transforms(tableSize, glm::mat4(1.f));
    std::vector<glm::mat4> h_inv_transforms(tableSize, glm::mat4(1.f));
    std::vector<int>       h_particle_counts(tableSize, 0);

    for (int fid : foamIds) {
        const auto& tx      = foamTransforms.at(fid);
        h_transforms[fid]     = tx;
        h_inv_transforms[fid] = glm::inverse(tx);
        h_particle_counts[fid] = gpuSlab.slots.at(fid).active_count;
    }

    glm::mat4* d_transforms     = nullptr;
    glm::mat4* d_inv_transforms = nullptr;
    int*       d_particle_counts= nullptr;
    CUDA_CHECK(cudaMalloc(&d_transforms,      tableSize * sizeof(glm::mat4)));
    CUDA_CHECK(cudaMalloc(&d_inv_transforms,  tableSize * sizeof(glm::mat4)));
    CUDA_CHECK(cudaMalloc(&d_particle_counts, tableSize * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_transforms,      h_transforms.data(),      tableSize * sizeof(glm::mat4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_inv_transforms,  h_inv_transforms.data(),  tableSize * sizeof(glm::mat4), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_particle_counts, h_particle_counts.data(), tableSize * sizeof(int),       cudaMemcpyHostToDevice));

    // Upload active foam ID list.
    int* d_foam_ids = nullptr;
    CUDA_CHECK(cudaMalloc(&d_foam_ids, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_foam_ids, foamIds.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    // -------------------------------------------------------------------------
    // Phase 1 — transform local-space foam AABBs to world space.
    // d_world_aabbs is indexed by foam_id, same as the slab's d_foam_aabbs.
    // -------------------------------------------------------------------------
    AABB* d_world_aabbs = nullptr;
    CUDA_CHECK(cudaMalloc(&d_world_aabbs, tableSize * sizeof(AABB)));

    {
        const int threads = 128;
        const int blocks  = (tableSize + threads - 1) / threads;
        k_col_transform_foam_aabbs<<<blocks, threads>>>(
            gpuSlab.d_foam_aabbs,   // local-space, indexed by foam_id
            d_transforms,
            d_world_aabbs,
            tableSize);
        CUDA_CHECK(cudaGetLastError());
    }

    // -------------------------------------------------------------------------
    // Phase 2 — broadphase: one thread per (i < j) foam pair.
    // -------------------------------------------------------------------------
    const int numPairs = N * (N - 1) / 2;

    FoamPair* d_broadphase_pairs = nullptr;
    int*      d_pair_counter     = nullptr;
    CUDA_CHECK(cudaMalloc(&d_broadphase_pairs, numPairs * sizeof(FoamPair)));
    CUDA_CHECK(cudaMalloc(&d_pair_counter,     sizeof(int)));
    CUDA_CHECK(cudaMemset(d_pair_counter, 0, sizeof(int)));

    {
        const int threads = 128;
        const int blocks  = (numPairs + threads - 1) / threads;
        k_col_broadphase_foam_pairs<<<blocks, threads>>>(
            d_world_aabbs,
            d_foam_ids,
            d_broadphase_pairs,
            d_pair_counter,
            N);
        CUDA_CHECK(cudaGetLastError());
    }

    int h_num_broadphase = 0;
    CUDA_CHECK(cudaMemcpy(&h_num_broadphase, d_pair_counter, sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<CollisionCandidate> results;
    if (h_num_broadphase == 0) goto cleanup;

    // -------------------------------------------------------------------------
    // Phase 3 — narrowphase: dual BVH traversal, one thread per surviving pair.
    // -------------------------------------------------------------------------
    {
        CollisionCandidate* d_candidates       = nullptr;
        int*                d_candidate_counter= nullptr;
        CUDA_CHECK(cudaMalloc(&d_candidates,        maxCandidates * sizeof(CollisionCandidate)));
        CUDA_CHECK(cudaMalloc(&d_candidate_counter, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_candidate_counter, 0, sizeof(int)));

        const int threads = 128;
        const int blocks  = (h_num_broadphase + threads - 1) / threads;
        k_col_narrowphase_dual_bvh<<<blocks, threads>>>(
            d_broadphase_pairs,
            h_num_broadphase,
            gpuSlab.d_bvh_nodes,
            gpuSlab.d_foam_bvh_start,
            d_particle_counts,
            d_transforms,
            d_inv_transforms,
            d_candidates,
            d_candidate_counter,
            maxCandidates);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());  // must complete before reading counter

        int h_num_candidates = 0;
        CUDA_CHECK(cudaMemcpy(&h_num_candidates, d_candidate_counter, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_num_candidates > maxCandidates)
            h_num_candidates = maxCandidates;  // clamped; caller should resize if this fires

        // Phase 3b: resolve prim_idx → entity_id on the GPU, in-place.
        // Launched with the exact candidate count so no thread reads past the
        // live region of d_candidates, avoiding any out-of-bounds d_active_ids access.
        if (h_num_candidates > 0) {
            const int rthreads = 128;
            const int rblocks  = (h_num_candidates + rthreads - 1) / rthreads;
            k_col_resolve_entity_ids<<<rblocks, rthreads>>>(
                d_candidates,
                h_num_candidates,
                gpuSlab.d_active_ids,
                gpuSlab.d_foam_active_start);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            results.resize(h_num_candidates);
            CUDA_CHECK(cudaMemcpy(results.data(), d_candidates,
                                  h_num_candidates * sizeof(CollisionCandidate),
                                  cudaMemcpyDeviceToHost));
        }

        CUDA_CHECK(cudaFree(d_candidates));
        CUDA_CHECK(cudaFree(d_candidate_counter));
    }

cleanup:
    CUDA_CHECK(cudaFree(d_world_aabbs));
    CUDA_CHECK(cudaFree(d_broadphase_pairs));
    CUDA_CHECK(cudaFree(d_pair_counter));
    CUDA_CHECK(cudaFree(d_foam_ids));
    CUDA_CHECK(cudaFree(d_transforms));
    CUDA_CHECK(cudaFree(d_inv_transforms));
    CUDA_CHECK(cudaFree(d_particle_counts));

    return results;
}

// =============================================================================
// Phase 4 — GJK + EPA (CPU)
// All code below is host-only and operates on world-space vertex buffers.
// =============================================================================

namespace {

// ---------------------------------------------------------------------------
// GJK support helpers
// ---------------------------------------------------------------------------

inline glm::vec3 supportPoint(
    const std::vector<glm::vec3>& verts, const glm::vec3& dir) noexcept
{
    float best = -FLT_MAX;
    glm::vec3 pt{};
    for (const auto& v : verts) {
        float d = glm::dot(v, dir);
        if (d > best) { best = d; pt = v; }
    }
    return pt;
}

inline glm::vec3 minkSupport(
    const std::vector<glm::vec3>& vertsA,
    const std::vector<glm::vec3>& vertsB,
    const glm::vec3& dir) noexcept
{
    return supportPoint(vertsA, dir) - supportPoint(vertsB, -dir);
}

// ---------------------------------------------------------------------------
// GJK simplex
// Convention: pts[0] is always the most recently added point.
// ---------------------------------------------------------------------------

struct Simplex {
    std::array<glm::vec3, 4> pts{};
    int size = 0;

    void push(const glm::vec3& p) noexcept {
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

bool doLineSimplex(Simplex& s, glm::vec3& dir) noexcept {
    const glm::vec3& A = s.pts[0];
    const glm::vec3& B = s.pts[1];
    const glm::vec3  AB = B - A, AO = -A;
    if (sameDir(AB, AO))
        dir = glm::cross(glm::cross(AB, AO), AB);
    else { s.size = 1; dir = AO; }
    return false;
}

bool doTriangleSimplex(Simplex& s, glm::vec3& dir) noexcept {
    const glm::vec3& A = s.pts[0];
    const glm::vec3& B = s.pts[1];
    const glm::vec3& C = s.pts[2];
    const glm::vec3  AB = B - A, AC = C - A, AO = -A;
    const glm::vec3  ABC = glm::cross(AB, AC);
    if (sameDir(glm::cross(ABC, AC), AO)) {
        if (sameDir(AC, AO)) { s.pts[1] = s.pts[2]; s.size = 2; dir = glm::cross(glm::cross(AC, AO), AC); }
        else { s.size = 2; return doLineSimplex(s, dir); }
    } else if (sameDir(glm::cross(AB, ABC), AO)) {
        s.size = 2; return doLineSimplex(s, dir);
    } else {
        if (sameDir(ABC, AO)) dir = ABC;
        else { std::swap(s.pts[1], s.pts[2]); dir = -ABC; }
    }
    return false;
}

bool doTetrahedronSimplex(Simplex& s, glm::vec3& dir) noexcept {
    const glm::vec3& A = s.pts[0], &B = s.pts[1], &C = s.pts[2], &D = s.pts[3];
    const glm::vec3  AB = B-A, AC = C-A, AD = D-A, AO = -A;
    const glm::vec3  nABC = glm::cross(AB, AC);
    const glm::vec3  nACD = glm::cross(AC, AD);
    const glm::vec3  nADB = glm::cross(AD, AB);
    const glm::vec3  nABC_out = sameDir(nABC, AD) ? -nABC : nABC;
    const glm::vec3  nACD_out = sameDir(nACD, AB) ? -nACD : nACD;
    const glm::vec3  nADB_out = sameDir(nADB, AC) ? -nADB : nADB;
    if (sameDir(nABC_out, AO)) {
        s.size = 3;
        if (!sameDir(glm::cross(B-A, C-A), nABC_out)) std::swap(s.pts[1], s.pts[2]);
        return doTriangleSimplex(s, dir);
    }
    if (sameDir(nACD_out, AO)) {
        s.pts[1] = s.pts[2]; s.pts[2] = s.pts[3]; s.size = 3;
        if (!sameDir(glm::cross(s.pts[1]-A, s.pts[2]-A), nACD_out)) std::swap(s.pts[1], s.pts[2]);
        return doTriangleSimplex(s, dir);
    }
    if (sameDir(nADB_out, AO)) {
        s.pts[2] = s.pts[1]; s.pts[1] = s.pts[3]; s.size = 3;
        if (!sameDir(glm::cross(s.pts[1]-A, s.pts[2]-A), nADB_out)) std::swap(s.pts[1], s.pts[2]);
        return doTriangleSimplex(s, dir);
    }
    return true;
}

bool doSimplex(Simplex& s, glm::vec3& dir) noexcept {
    switch (s.size) {
        case 2: return doLineSimplex(s, dir);
        case 3: return doTriangleSimplex(s, dir);
        case 4: return doTetrahedronSimplex(s, dir);
        default: return false;
    }
}

constexpr int   GJK_MAX_ITER = 64;
constexpr float GJK_EPSILON  = 1e-6f;

bool gjk(const std::vector<glm::vec3>& vertsA,
         const std::vector<glm::vec3>& vertsB,
         Simplex& simplex)
{
    glm::vec3 dirA{}, dirB{};
    for (const auto& v : vertsA) dirA += v;
    for (const auto& v : vertsB) dirB += v;
    dirA /= float(vertsA.size());
    dirB /= float(vertsB.size());
    glm::vec3 dir = dirA - dirB;
    if (glm::length(dir) < GJK_EPSILON) dir = glm::vec3(1.f, 0.f, 0.f);

    simplex = Simplex{};
    simplex.push(minkSupport(vertsA, vertsB, dir));
    dir = -simplex.pts[0];

    for (int iter = 0; iter < GJK_MAX_ITER; ++iter) {
        if (glm::length(dir) < GJK_EPSILON) return true;
        glm::vec3 A = minkSupport(vertsA, vertsB, dir);
        if (glm::dot(A, dir) < 0.f) return false;
        simplex.push(A);
        if (doSimplex(simplex, dir)) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// EPA
// ---------------------------------------------------------------------------

void expandToTetrahedron(Simplex& s,
                         const std::vector<glm::vec3>& vertsA,
                         const std::vector<glm::vec3>& vertsB)
{
    const glm::vec3 axes[6] = {
        { 1,0,0},{-1,0,0},{0, 1,0},{0,-1,0},{0,0, 1},{0,0,-1}
    };
    while (s.size < 4) {
        for (const auto& ax : axes) {
            glm::vec3 p = minkSupport(vertsA, vertsB, ax);
            bool dup = false;
            for (int k = 0; k < s.size; ++k)
                if (glm::length(p - s.pts[k]) < GJK_EPSILON) { dup = true; break; }
            if (!dup) { s.push(p); break; }
        }
    }
}

struct EPAFace { glm::vec3 a, b, c, normal; float distance; };

inline EPAFace makeFace(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c) noexcept {
    EPAFace f; f.a = a; f.b = b; f.c = c;
    glm::vec3 n = glm::cross(b - a, c - a);
    float len   = glm::length(n);
    f.normal    = (len > 1e-9f) ? (n / len) : glm::vec3(0.f, 0.f, 1.f);
    f.distance  = glm::dot(f.normal, a);
    if (f.distance < 0.f) { f.normal = -f.normal; f.distance = -f.distance; std::swap(f.b, f.c); }
    return f;
}

constexpr int   EPA_MAX_ITER = 64;
constexpr float EPA_EPSILON  = 1e-4f;

bool epa(Simplex& simplex,
         const std::vector<glm::vec3>& vertsA,
         const std::vector<glm::vec3>& vertsB,
         glm::vec3& normal, float& depth)
{
    expandToTetrahedron(simplex, vertsA, vertsB);
    const glm::vec3& A = simplex.pts[0], &B = simplex.pts[1],
                   & C = simplex.pts[2], &D = simplex.pts[3];

    std::vector<EPAFace> faces;
    faces.reserve(64);
    faces.push_back(makeFace(A, B, C));
    faces.push_back(makeFace(A, C, D));
    faces.push_back(makeFace(A, D, B));
    faces.push_back(makeFace(B, D, C));

    for (int iter = 0; iter < EPA_MAX_ITER; ++iter) {
        int minIdx = 0; float minDist = FLT_MAX;
        for (int i = 0; i < int(faces.size()); ++i)
            if (faces[i].distance < minDist) { minDist = faces[i].distance; minIdx = i; }

        const EPAFace& closest = faces[minIdx];
        glm::vec3 sup = minkSupport(vertsA, vertsB, closest.normal);
        if (glm::dot(closest.normal, sup) - closest.distance < EPA_EPSILON) {
            normal = closest.normal; depth = closest.distance; return true;
        }

        struct Edge { glm::vec3 a, b; };
        std::vector<Edge> silhouette;
        std::vector<EPAFace> newFaces;
        newFaces.reserve(faces.size());
        for (const auto& f : faces) {
            if (glm::dot(f.normal, sup - f.a) > 0.f) {
                silhouette.push_back({f.a, f.b});
                silhouette.push_back({f.b, f.c});
                silhouette.push_back({f.c, f.a});
            } else {
                newFaces.push_back(f);
            }
        }
        if (silhouette.empty()) { normal = closest.normal; depth = closest.distance; return true; }

        auto edgeEq = [](const glm::vec3& p, const glm::vec3& q) {
            return glm::length(p - q) < 1e-7f;
        };
        for (int i = 0; i < int(silhouette.size()); ++i) {
            bool interior = false;
            for (int j = 0; j < int(silhouette.size()); ++j) {
                if (i == j) continue;
                if (edgeEq(silhouette[i].a, silhouette[j].b) &&
                    edgeEq(silhouette[i].b, silhouette[j].a)) { interior = true; break; }
            }
            if (!interior)
                newFaces.push_back(makeFace(silhouette[i].a, silhouette[i].b, sup));
        }
        faces = std::move(newFaces);
        if (faces.empty()) return false;
    }

    int minIdx = 0; float minDist = FLT_MAX;
    for (int i = 0; i < int(faces.size()); ++i)
        if (faces[i].distance < minDist) { minDist = faces[i].distance; minIdx = i; }
    normal = faces[minIdx].normal; depth = faces[minIdx].distance;
    return true;
}

// ---------------------------------------------------------------------------
// Per-candidate exact test: lift vertices to world space, run GJK+EPA.
// ---------------------------------------------------------------------------

std::optional<FoamCollision> testParticlePair(
    int foamIdA, int foamIdB,
    entt::entity particleA, entt::entity particleB,
    const std::vector<glm::vec3>& vertsA,   // local-A space
    const std::vector<glm::vec3>& vertsB,   // local-B space
    const glm::mat4& transformA,
    const glm::mat4& transformB)
{
    if (vertsA.empty() || vertsB.empty()) return std::nullopt;

    auto toWorld = [](const std::vector<glm::vec3>& locals, const glm::mat4& xform) {
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

    glm::vec3 normal; float depth;
    if (!epa(simplex, worldA, worldB, normal, depth)) return std::nullopt;
    if (depth < 0.f) return std::nullopt;

    const glm::vec3 ptOnA   = supportPoint(worldA,  normal);
    const glm::vec3 ptOnB   = supportPoint(worldB, -normal);
    const glm::vec3 contact = 0.5f * (ptOnA + ptOnB);

    FoamCollision col;
    col.foamIdA          = foamIdA;
    col.foamIdB          = foamIdB;
    col.particleA        = particleA;
    col.particleB        = particleB;
    col.normal           = normal;
    col.penetrationDepth = depth;
    col.contactPoint     = contact;
    return col;
}

} // anonymous namespace

// =============================================================================
// Public: detectCollisions — full pipeline (phases 1–4)
// =============================================================================

std::vector<FoamCollision> detectCollisions(
    const GpuSlabAllocator&                        gpuSlab,
    const std::unordered_map<int, glm::mat4>&      foamTransforms,
    const entt::registry&                          particleRegistry,
    const std::vector<int>&                        foamIds,
    int                                            maxCandidates)
{
    // -------------------------------------------------------------------------
    // Phases 1–3: GPU broadphase + narrowphase + entity resolve.
    // Each CollisionCandidate already carries entity_id_a/b, resolved on the
    // GPU by k_col_resolve_entity_ids before the D2H transfer.  No active_ids
    // slice download is needed.
    // -------------------------------------------------------------------------
    const std::vector<CollisionCandidate> candidates =
        detectCandidates(gpuSlab, foamTransforms, foamIds, maxCandidates);

    if (candidates.empty()) return {};

    // -------------------------------------------------------------------------
    // Phase 4: CPU GJK + EPA for each surviving candidate.
    // -------------------------------------------------------------------------
    std::vector<FoamCollision> results;
    results.reserve(candidates.size());

    for (const CollisionCandidate& c : candidates) {
        const entt::entity eA = static_cast<entt::entity>(c.entity_id_a);
        const entt::entity eB = static_cast<entt::entity>(c.entity_id_b);

        const auto* cpA = particleRegistry.try_get<ParticleVertices>(eA);
        const auto* cpB = particleRegistry.try_get<ParticleVertices>(eB);
        if (!cpA || !cpB) continue;

        auto col = testParticlePair(
            c.foam_id_a, c.foam_id_b, eA, eB,
            cpA->vertices, cpB->vertices,
            foamTransforms.at(c.foam_id_a),
            foamTransforms.at(c.foam_id_b));

        if (col) results.push_back(std::move(*col));
    }

    return results;
}

} // namespace DynamicFoam::Sim2D
