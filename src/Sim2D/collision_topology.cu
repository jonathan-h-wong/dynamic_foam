// =============================================================================
// collision_topology.cu
// Point-in-cell containment via Voronoi half-plane tests.
//
// For each (point P, cell C) pair:
//   For each Voronoi neighbor N_i of C:
//     sdf_i = dot(P - C, N_i - C) - 0.5 * |N_i - C|^2
//   P is inside C iff all sdf_i <= 0.
//   The least-negative sdf gives the contact normal and depth.
// =============================================================================

#define NOMINMAX
#include "dynamic_foam/Sim2D/collision_topology.cuh"

#include <glm/gtc/matrix_transform.hpp>

#include <cfloat>
#include <unordered_map>
#include <vector>

namespace DynamicFoam::Sim2D {

// =============================================================================
// Section 1 -- Shared Voronoi half-plane test (host + device).
// =============================================================================

// Returns true if P is inside the Voronoi cell centered at C.
// Voronoi neighbors are in neighbors[0..numNeighbors).
// outNormal: unit vector pointing toward the nearest boundary (eject direction).
// outDepth:  positive penetration distance.
__host__ __device__ inline bool voronoiContainmentNorm(
    const glm::vec3& P,
    const glm::vec3& C,
    const glm::vec3* neighbors,
    int              numNeighbors,
    glm::vec3&       outNormal,
    float&           outDepth) noexcept
{
    float     maxNormSdf = -FLT_MAX; // least-negative normalised sdf
    glm::vec3 bestNormal{};

    for (int i = 0; i < numNeighbors; ++i) {
        glm::vec3 edge = neighbors[i] - C;
        float     len  = sqrtf(glm::dot(edge, edge));
        if (len < 1e-6f) continue;

        glm::vec3 n        = edge / len;
        float     halfDist = 0.5f * len;
        // Normalised signed distance: positive = outside.
        float normSdf = glm::dot(P - C, n) - halfDist;

        if (normSdf > 0.0f) return false;

        if (normSdf > maxNormSdf) {
            maxNormSdf = normSdf;
            bestNormal = n;
        }
    }

    if (numNeighbors == 0) return false;

    outNormal = bestNormal;
    outDepth  = -maxNormSdf; // positive penetration depth
    return true;
}

// =============================================================================
// Section 2 -- GPU kernel.
// =============================================================================

__global__ void k_pcc_containment(
    const PointCellTest*  tests,
    int                   num_tests,
    const glm::vec3*      neighbor_centers,
    PointCellContact*     contacts_out,
    int*                  contact_counter,
    int                   max_contacts)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_tests) return;

    const PointCellTest& t = tests[tid];

    glm::vec3 normal{};
    float     depth{};
    bool hit = voronoiContainmentNorm(
        t.pointWorld, t.cellCenter,
        neighbor_centers + t.neighborOffset, t.neighborCount,
        normal, depth);

    if (!hit) return;

    int slot = atomicAdd(contact_counter, 1);
    if (slot >= max_contacts) return;

    PointCellContact& c = contacts_out[slot];
    c.pointEntity  = static_cast<entt::entity>(t.pointEntity);
    c.cellEntity   = static_cast<entt::entity>(t.cellEntity);
    c.foamIdPoint  = t.foamIdPoint;
    c.foamIdCell   = t.foamIdCell;
    c.normal       = normal;
    c.depth        = depth;
}

// =============================================================================
// Section 3 -- CPU/GPU shared flattening helper.
//
// Builds:
//   tests           -- one PointCellTest per (neighbor-of-A, cellB) and
//                      (neighbor-of-B, cellA) pair.
//   allNeighbors    -- flat buffer of neighbor world-space centers, referenced
//                      by PointCellTest::neighborOffset / neighborCount.
//
// For each cell we cache its neighbor list (world-space) keyed by entity so
// cells referenced by multiple collision pairs are not duplicated.
// =============================================================================

struct FlattenResult {
    std::vector<PointCellTest> tests;
    std::vector<glm::vec3>     allNeighbors; // flat neighbor-center buffer
};

static glm::vec3 toWorld(const glm::mat4& tx, const glm::vec3& local) {
    return glm::vec3(tx * glm::vec4(local, 1.0f));
}

static FlattenResult flattenTests(
    const std::vector<FoamCollision>&              collisions,
    const std::unordered_map<int, AdjacencyList>&  foamAdj,
    const std::unordered_map<int, glm::mat4>&      foamTx,
    const entt::registry&                          reg)
{
    FlattenResult result;

    // Cache: entity -> (neighborOffset, neighborCount) in allNeighbors.
    struct CellCache {
        int offset;
        int count;
        glm::vec3 center;
    };
    std::unordered_map<uint32_t, CellCache> cache;

    // Helper: ensure cell C's neighbor list is in allNeighbors, return cache entry.
    auto ensureCell = [&](entt::entity cellEnt, int foamId) -> const CellCache& {
        uint32_t key = static_cast<uint32_t>(cellEnt);
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;

        const auto& tx     = foamTx.at(foamId);
        const auto& adjIt  = foamAdj.find(foamId);

        glm::vec3 center{};
        if (reg.all_of<ParticleLocalPosition>(cellEnt))
            center = toWorld(tx, reg.get<ParticleLocalPosition>(cellEnt).value);

        int offset = static_cast<int>(result.allNeighbors.size());
        int count  = 0;

        if (adjIt != foamAdj.end()) {
            adjIt->second.forEachNeighbor(static_cast<uint32_t>(cellEnt), [&](uint32_t nbId) {
                entt::entity nbEnt = static_cast<entt::entity>(nbId);
                if (reg.all_of<ParticleLocalPosition>(nbEnt)) {
                    result.allNeighbors.push_back(
                        toWorld(tx, reg.get<ParticleLocalPosition>(nbEnt).value));
                    ++count;
                }
            });
        }

        CellCache entry{offset, count, center};
        cache.emplace(key, entry);
        return cache.at(key);
    };

    // Helper: add one (point, cell) test.
    auto addTest = [&](
        entt::entity pointEnt, int foamIdPoint,
        entt::entity cellEnt,  int foamIdCell)
    {
        const auto& cellCache = ensureCell(cellEnt, foamIdCell);
        const auto& ptTx      = foamTx.at(foamIdPoint);

        glm::vec3 ptWorld{};
        if (reg.all_of<ParticleLocalPosition>(pointEnt))
            ptWorld = toWorld(ptTx, reg.get<ParticleLocalPosition>(pointEnt).value);

        PointCellTest t{};
        t.pointWorld     = ptWorld;
        t.cellCenter     = cellCache.center;
        t.pointEntity    = static_cast<uint32_t>(pointEnt);
        t.cellEntity     = static_cast<uint32_t>(cellEnt);
        t.foamIdPoint    = foamIdPoint;
        t.foamIdCell     = foamIdCell;
        t.neighborOffset = cellCache.offset;
        t.neighborCount  = cellCache.count;
        result.tests.push_back(t);
    };

    for (const auto& col : collisions) {
        entt::entity eA = col.particleA;
        entt::entity eB = col.particleB;
        int fA = col.foamIdA;
        int fB = col.foamIdB;

        if (foamAdj.count(fA)) {
            foamAdj.at(fA).forEachNeighbor(static_cast<uint32_t>(eA), [&](uint32_t nbId) {
                addTest(static_cast<entt::entity>(nbId), fA, eB, fB);
            });
        }
        if (foamAdj.count(fB)) {
            foamAdj.at(fB).forEachNeighbor(static_cast<uint32_t>(eB), [&](uint32_t nbId) {
                addTest(static_cast<entt::entity>(nbId), fB, eA, fA);
            });
        }
    }

    return result;
}

// =============================================================================
// Section 4 -- CPU path.
// =============================================================================

static std::vector<PointCellContact> detectPointCellCPU(
    const std::vector<FoamCollision>&              collisions,
    const std::unordered_map<int, AdjacencyList>&  foamAdj,
    const std::unordered_map<int, glm::mat4>&      foamTx,
    const entt::registry&                          reg)
{
    auto [tests, allNeighbors] = flattenTests(collisions, foamAdj, foamTx, reg);

    std::vector<PointCellContact> contacts;
    for (const auto& t : tests) {
        glm::vec3 normal{};
        float     depth{};
        bool hit = voronoiContainmentNorm(
            t.pointWorld, t.cellCenter,
            allNeighbors.data() + t.neighborOffset, t.neighborCount,
            normal, depth);
        if (!hit) continue;

        PointCellContact c{};
        c.pointEntity = static_cast<entt::entity>(t.pointEntity);
        c.cellEntity  = static_cast<entt::entity>(t.cellEntity);
        c.foamIdPoint = t.foamIdPoint;
        c.foamIdCell  = t.foamIdCell;
        c.normal      = normal;
        c.depth       = depth;
        contacts.push_back(c);
    }
    return contacts;
}

// =============================================================================
// Section 5 -- GPU path.
// =============================================================================

static std::vector<PointCellContact> detectPointCellGPU(
    const std::vector<FoamCollision>&              collisions,
    const std::unordered_map<int, AdjacencyList>&  foamAdj,
    const std::unordered_map<int, glm::mat4>&      foamTx,
    const entt::registry&                          reg,
    int                                            maxContacts)
{
    auto [tests, allNeighbors] = flattenTests(collisions, foamAdj, foamTx, reg);
    if (tests.empty()) return {};

    // --- Upload ---
    PointCellTest* d_tests{};
    glm::vec3*     d_neighbors{};
    PointCellContact* d_contacts{};
    int*           d_counter{};

    CUDA_CHECK(cudaMalloc(&d_tests,     tests.size()        * sizeof(PointCellTest)));
    CUDA_CHECK(cudaMalloc(&d_neighbors, allNeighbors.size() * sizeof(glm::vec3)));
    CUDA_CHECK(cudaMalloc(&d_contacts,  maxContacts         * sizeof(PointCellContact)));
    CUDA_CHECK(cudaMalloc(&d_counter,   sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_tests,     tests.data(),        tests.size()        * sizeof(PointCellTest),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors, allNeighbors.data(), allNeighbors.size() * sizeof(glm::vec3),      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

    // --- Kernel ---
    int numTests = static_cast<int>(tests.size());
    int blocks   = (numTests + 255) / 256;
    k_pcc_containment<<<blocks, 256>>>(
        d_tests, numTests, d_neighbors,
        d_contacts, d_counter, maxContacts);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Download ---
    int h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost));
    h_count = std::min(h_count, maxContacts);

    std::vector<PointCellContact> contacts(h_count);
    if (h_count > 0)
        CUDA_CHECK(cudaMemcpy(contacts.data(), d_contacts, h_count * sizeof(PointCellContact), cudaMemcpyDeviceToHost));

    cudaFree(d_tests);
    cudaFree(d_neighbors);
    cudaFree(d_contacts);
    cudaFree(d_counter);

    return contacts;
}

// =============================================================================
// Section 6 -- Public entry point.
// =============================================================================

std::vector<PointCellContact> detectPointCellContainment(
    const std::vector<FoamCollision>&              collisions,
    const std::unordered_map<int, AdjacencyList>&  foamAdjacencyLists,
    const std::unordered_map<int, glm::mat4>&      foamTransforms,
    const entt::registry&                          particleRegistry,
    int                                            maxContacts,
    bool                                           forceGpu)
{
    if (collisions.empty()) return {};

    bool useGpu = forceGpu ||
                  (static_cast<int>(collisions.size()) >= kPointCellGpuThreshold);

    if (useGpu)
        return detectPointCellGPU(collisions, foamAdjacencyLists, foamTransforms,
                                  particleRegistry, maxContacts);
    else
        return detectPointCellCPU(collisions, foamAdjacencyLists, foamTransforms,
                                  particleRegistry);
}

} // namespace DynamicFoam::Sim2D
