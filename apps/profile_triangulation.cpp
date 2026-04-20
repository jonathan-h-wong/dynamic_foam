/**
 * profile_triangulation.cpp
 *
 * Benchmark for triangulateWithMetadata() in utils.h.
 * Breaks timing down into three phases:
 *   1. dt.insert            – CGAL Delaunay insertion
 *   2. Adjacency build      – vertex→index mapping + edge walking
 *   3. Voronoi metadata     – volume + Voronoi-vertex per particle
 *
 * Build target: profile_triangulation  (added to apps/CMakeLists.txt)
 * Run:          ./profile_triangulation
 */

#include <chrono>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <future>
#include <thread>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

#define NOMINMAX
#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>

#include "dynamic_foam/Sim2D/adjacency.cuh"

// Pull in CGAL directly so we can time individual phases.
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Delaunay_triangulation_cell_base_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_data_structure_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>

using K       = CGAL::Exact_predicates_inexact_constructions_kernel;
using Vb      = CGAL::Triangulation_vertex_base_with_info_3<size_t, K>;
using Cb      = CGAL::Delaunay_triangulation_cell_base_3<K>;
using Tds     = CGAL::Triangulation_data_structure_3<Vb, Cb>;
using Delaunay_3 = CGAL::Delaunay_triangulation_3<K, Tds>;
using Point_3    = K::Point_3;
using namespace DynamicFoam::Sim2D;
using Clock = std::chrono::high_resolution_clock;
using ms    = std::chrono::duration<double, std::milli>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void generateCloud(size_t N, std::vector<glm::vec3>& pos,
                          std::vector<uint32_t>& ids, std::mt19937& rng)
{
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    pos.resize(N); ids.resize(N);
    for (size_t i = 0; i < N; ++i) {
        pos[i] = { d(rng), d(rng), d(rng) };
        ids[i] = static_cast<uint32_t>(i);
    }
}

// ---------------------------------------------------------------------------
// Per-phase timing struct
//
// Phases:
//   insert_ms      – dt.insert (CGAL filtered-predicate Delaunay build)
//   adj_edge_ms    – finite_edges iterator walk only (CGAL side)
//   adj_map_ms     – AdjacencyList::addEdge calls (nested unordered_map/set)
//   cache_ms       – one dt.dual() per finite cell into ccCache
//   vor_verts_ms   – incident_cells loop per vertex (reads cache)
// ---------------------------------------------------------------------------
struct PhaseTimes {
    double insert_ms;
    double adj_edge_ms;   // CGAL edge walk
    double adj_map_ms;    // AdjacencyList hash-map insertions
    double cache_ms;
    double vor_verts_ms;  // Voronoi vertex collection
};

// ---------------------------------------------------------------------------
// Generate a tight spatial cluster of N points around a random centre.
// radius controls how clustered the points are.
// ---------------------------------------------------------------------------
static void generateCluster(size_t N, glm::vec3 centre, float radius,
                             std::vector<glm::vec3>& pos, std::mt19937& rng)
{
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    pos.resize(N);
    for (size_t i = 0; i < N; ++i) {
        glm::vec3 offset(d(rng), d(rng), d(rng));
        pos[i] = centre + offset * radius;
    }
}

// ---------------------------------------------------------------------------
// Benchmark B: incremental hint-chain insert of a cluster into an existing DT.
//
// Models the real use-case: foam already exists (baseDt), cursor spawns K new
// particles near a known vertex.  We find the nearest existing vertex to the
// cluster centre, then insert the cluster one point at a time propagating the
// hint.  Only the insert phase is timed; adjacency stages are identical to
// the batch path and are not re-measured here.
//
// Returns just the insert_ms for the K new points.
// ---------------------------------------------------------------------------
static double runIncrementalInsert(const Delaunay_3& baseDt,
                                   const std::vector<glm::vec3>& clusterPos,
                                   glm::vec3 clusterCentre)
{
    // Copy the base DT so repeated trials are independent.
    Delaunay_3 dt(baseDt);

    // Find the existing vertex closest to the cluster centre to use as hint.
    // In production this would be a cheap spatial lookup; here we do a linear
    // scan once at setup time (not timed).
    Delaunay_3::Vertex_handle hint;
    double bestDist = std::numeric_limits<double>::max();
    for (auto vit = dt.finite_vertices_begin();
              vit != dt.finite_vertices_end(); ++vit) {
        double dx = vit->point().x() - clusterCentre.x;
        double dy = vit->point().y() - clusterCentre.y;
        double dz = vit->point().z() - clusterCentre.z;
        double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < bestDist) { bestDist = d2; hint = vit; }
    }

    // Time only the sequential hint-chain inserts.
    auto t0 = Clock::now();
    size_t idx = 1'000'000; // info indices won't collide with base
    for (const auto& p : clusterPos) {
        auto vh = dt.insert(Point_3(p.x, p.y, p.z), hint->cell());
        vh->info() = idx++;
        hint = vh;
    }
    auto t1 = Clock::now();

    return ms(t1 - t0).count();
}

static PhaseTimes runTrial(size_t N, std::mt19937& rng)
{
    std::vector<glm::vec3> positions;
    std::vector<uint32_t>  ids;
    generateCloud(N, positions, ids, rng);

    // ---- Phase 1: CGAL insertion (with embedded index) -----------------
    auto t0 = Clock::now();

    Delaunay_3 dt;
    {
        std::vector<std::pair<Point_3, size_t>> indexed_points;
        indexed_points.reserve(N);
        for (size_t i = 0; i < N; ++i)
            indexed_points.emplace_back(
                Point_3(positions[i].x, positions[i].y, positions[i].z), i);
        dt.insert(indexed_points.begin(), indexed_points.end());
    }

    auto t1 = Clock::now();

    // ---- Phase 2a: CGAL edge walk only (collect edge pairs) ------------
    struct EdgePair { uint32_t a, b; };
    std::vector<EdgePair> edges;
    edges.reserve(static_cast<size_t>(dt.number_of_finite_edges()));
    for (auto eit = dt.finite_edges_begin(); eit != dt.finite_edges_end(); ++eit) {
        auto v1 = eit->first->vertex(eit->second);
        auto v2 = eit->first->vertex(eit->third);
        edges.push_back({ ids[v1->info()], ids[v2->info()] });
    }

    auto t2 = Clock::now();

    // ---- Phase 2b: AdjacencyList hash-map insertions -------------------
    AdjacencyList adjList(ids);
    adjList.reserveEdges(edges.size());
    for (const auto& e : edges)
        adjList.addEdge(e.a, e.b);

    auto t3 = Clock::now();

    // ---- Phase 3: Circumcenter cache (one dt.dual() per finite cell) ----
    std::unordered_map<Delaunay_3::Cell_handle, glm::vec3> ccCache;
    ccCache.reserve(static_cast<size_t>(dt.number_of_finite_cells()));
    for (auto cit = dt.finite_cells_begin(); cit != dt.finite_cells_end(); ++cit) {
        Point_3 cc = dt.dual(cit);
        ccCache[cit] = glm::vec3(
            static_cast<float>(cc.x()),
            static_cast<float>(cc.y()),
            static_cast<float>(cc.z()));
    }

    auto t4 = Clock::now();

    // ---- Phase 4a: Voronoi vertex collection (reads ccCache) -----------
    std::unordered_map<uint32_t, std::vector<glm::vec3>> voronoiVerts;
    voronoiVerts.reserve(N);
    for (auto vit = dt.finite_vertices_begin(); vit != dt.finite_vertices_end(); ++vit) {
        size_t   i  = vit->info();
        uint32_t id = ids[i];
        if (dt.is_infinite(vit)) { voronoiVerts[id] = {}; continue; }

        std::vector<Delaunay_3::Cell_handle> incident_cells;
        dt.incident_cells(vit, std::back_inserter(incident_cells));
        std::vector<glm::vec3> v_verts;
        v_verts.reserve(incident_cells.size());
        for (const auto& ch : incident_cells) {
            auto it = ccCache.find(ch);
            if (it != ccCache.end()) v_verts.push_back(it->second);
        }
        voronoiVerts[id] = std::move(v_verts);
    }

    auto t5 = Clock::now();

    (void)adjList; (void)voronoiVerts;
    return {
        ms(t1-t0).count(),  // insert
        ms(t2-t1).count(),  // adj_edge
        ms(t3-t2).count(),  // adj_map
        ms(t4-t3).count(),  // cache
        ms(t5-t4).count()   // vor_verts
    };
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int /*argc*/, char* argv[])
{
    constexpr int WARMUP   = 1;
    constexpr int TRIALS   = 3;
    constexpr double SKIP_MS = 10'000.0;

    const std::vector<size_t> sizes = {
        100, 250, 500, 1'000, 2'500, 5'000, 10'000, 25'000, 50'000
    };

    std::mt19937 rng(42);

    // Build timestamped output filename next to the executable.
    std::time_t now = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    // Always write next to the executable in a profiles/ subdirectory.
    fs::path profileDir = fs::path(argv[0]).parent_path() / "profiles";
    fs::create_directories(profileDir);
    std::ostringstream fname;
    fname << "triangulation_profile_"
          << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".csv";
    fs::path csvPath = profileDir / fname.str();
    std::ofstream csv(csvPath);

    // CSV header -- one row per individual trial for full variance data.
    csv << "timestamp,points,trial,insert_ms,adj_edge_ms,adj_map_ms,cache_ms,vor_verts_ms,total_ms\n";
    std::string timestamp;
    {
        std::ostringstream ts;
        ts << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        timestamp = ts.str();
    }

    const int W = 12;
    std::cout << "\nCGAL 3-D Delaunay triangulation benchmark -- phase breakdown\n";
    std::cout << std::string(W*7, '-') << "\n";
    std::cout << std::setw(W) << "Points"
              << std::setw(W) << "Insert"
              << std::setw(W) << "AdjEdge"
              << std::setw(W) << "AdjMap"
              << std::setw(W) << "Cache"
              << std::setw(W) << "VorVerts"
              << std::setw(W) << "Total"
              << "\n";
    std::cout << std::string(W*7, '-') << "\n";

    for (size_t N : sizes) {
        for (int w = 0; w < WARMUP; ++w) runTrial(N, rng);

        // Per-trial rows go to CSV; averages go to console.
        PhaseTimes sum{};
        for (int t = 0; t < TRIALS; ++t) {
            auto p = runTrial(N, rng);
            double tot = p.insert_ms + p.adj_edge_ms + p.adj_map_ms
                       + p.cache_ms + p.vor_verts_ms;
            csv << timestamp << ","
                << N << ","
                << (t + 1) << ","
                << std::fixed << std::setprecision(4)
                << p.insert_ms    << ","
                << p.adj_edge_ms  << ","
                << p.adj_map_ms   << ","
                << p.cache_ms     << ","
                << p.vor_verts_ms << ","
                << tot << "\n";
            sum.insert_ms    += p.insert_ms;
            sum.adj_edge_ms  += p.adj_edge_ms;
            sum.adj_map_ms   += p.adj_map_ms;
            sum.cache_ms     += p.cache_ms;
            sum.vor_verts_ms += p.vor_verts_ms;
        }
        double ins  = sum.insert_ms    / TRIALS;
        double aedg = sum.adj_edge_ms  / TRIALS;
        double amap = sum.adj_map_ms   / TRIALS;
        double cch  = sum.cache_ms     / TRIALS;
        double vvrt = sum.vor_verts_ms / TRIALS;
        double tot  = ins + aedg + amap + cch + vvrt;

        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(W) << N
                  << std::setw(W) << ins
                  << std::setw(W) << aedg
                  << std::setw(W) << amap
                  << std::setw(W) << cch
                  << std::setw(W) << vvrt
                  << std::setw(W) << tot
                  << "\n";

        if (tot > SKIP_MS) {
            std::cout << "  (skipping larger sizes -- avg > 10 s)\n";
            break;
        }
    }

    std::cout << std::string(W*7, '-') << "\n";
    csv.flush();
    std::cout << "\nResults written to: " << csvPath.string() << "\n";

    // =========================================================================
    // Benchmark B: incremental hint-chain vs cold batch insert
    //
    // Simulates the real cursor use-case:
    //   - A base foam of BASE_N particles already has a triangulation.
    //   - A cursor spawns CLUSTER_N new points in a tight spatial region.
    //   - We compare:
    //       "batch cold"  – destroy & rebuild a fresh DT from CLUSTER_N pts
    //       "incremental" – insert CLUSTER_N pts one-at-a-time with hint
    //                       into the existing base DT
    //
    // The cluster radius is set so points are tightly packed (~0.05 units),
    // maximising hint-chain locality.
    // =========================================================================
    constexpr size_t BASE_N        = 1000;   // existing foam size
    constexpr float  CLUSTER_RADIUS = 0.05f; // tight spatial cluster

    const std::vector<size_t> clusterSizes = { 10, 25, 50, 100, 250, 500, 1000 };

    std::cout << "\nBenchmark B: incremental hint-chain vs cold batch insert\n";
    std::cout << "  Base DT: " << BASE_N << " existing particles\n";
    std::cout << "  Cluster radius: " << CLUSTER_RADIUS << "\n\n";

    const int W2 = 13;
    std::cout << std::setw(W2) << "ClusterN"
              << std::setw(W2) << "BatchCold"
              << std::setw(W2) << "Incremental"
              << std::setw(W2) << "Speedup"
              << "\n";
    std::cout << std::string(W2*4, '-') << "\n";

    csv << "\nbenchmark_b,cluster_n,trial,batch_cold_ms,incremental_ms\n";

    // Build the base DT once (not timed – represents the existing foam).
    std::vector<glm::vec3> basePos;
    std::vector<uint32_t>  baseIds;
    generateCloud(BASE_N, basePos, baseIds, rng);
    Delaunay_3 baseDt;
    {
        std::vector<std::pair<Point_3, size_t>> basePts;
        basePts.reserve(BASE_N);
        for (size_t i = 0; i < BASE_N; ++i)
            basePts.emplace_back(Point_3(basePos[i].x, basePos[i].y, basePos[i].z), i);
        baseDt.insert(basePts.begin(), basePts.end());
    }

    for (size_t CN : clusterSizes) {
        // Pick a random cluster centre inside the base DT's [0,1]^3 domain.
        std::uniform_real_distribution<float> cd(0.1f, 0.9f);
        glm::vec3 centre(cd(rng), cd(rng), cd(rng));

        double sumBatch = 0.0, sumIncr = 0.0;

        for (int w = 0; w < WARMUP; ++w) {
            // warmup: batch cold
            std::vector<glm::vec3> cp; generateCluster(CN, centre, CLUSTER_RADIUS, cp, rng);
            Delaunay_3 wdt;
            std::vector<std::pair<Point_3, size_t>> wpts;
            wpts.reserve(CN);
            for (size_t i = 0; i < CN; ++i)
                wpts.emplace_back(Point_3(cp[i].x, cp[i].y, cp[i].z), i);
            wdt.insert(wpts.begin(), wpts.end());
            // warmup: incremental
            runIncrementalInsert(baseDt, cp, centre);
        }

        for (int t = 0; t < TRIALS; ++t) {
            std::vector<glm::vec3> clusterPos;
            generateCluster(CN, centre, CLUSTER_RADIUS, clusterPos, rng);

            // -- Batch cold: fresh DT from CLUSTER_N points --
            auto tb0 = Clock::now();
            {
                Delaunay_3 bdt;
                std::vector<std::pair<Point_3, size_t>> bpts;
                bpts.reserve(CN);
                for (size_t i = 0; i < CN; ++i)
                    bpts.emplace_back(Point_3(clusterPos[i].x, clusterPos[i].y,
                                              clusterPos[i].z), i);
                bdt.insert(bpts.begin(), bpts.end());
                (void)bdt;
            }
            double batchMs = ms(Clock::now() - tb0).count();

            // -- Incremental hint-chain into base DT --
            double incrMs = runIncrementalInsert(baseDt, clusterPos, centre);

            csv << "b," << CN << "," << (t+1) << ","
                << std::fixed << std::setprecision(4)
                << batchMs << "," << incrMs << "\n";

            sumBatch += batchMs;
            sumIncr  += incrMs;
        }

        double avgBatch = sumBatch / TRIALS;
        double avgIncr  = sumIncr  / TRIALS;
        double speedup  = avgBatch / std::max(avgIncr, 0.001);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(W2) << CN
                  << std::setw(W2) << avgBatch
                  << std::setw(W2) << avgIncr
                  << std::setw(W2-2) << speedup << "x"
                  << "\n";
    }

    std::cout << std::string(W2*4, '-') << "\n";
    csv.flush();
    std::cout << "\nResults written to: " << csvPath.string() << "\n";

    // =========================================================
    // BENCHMARK C – K parallel cold builds vs single bulk build
    // =========================================================
    // Each of K async tasks independently builds a fresh DT of
    // CLUSTER_C_N points.  We measure wall-clock from dispatch
    // to last .get(), then compare against a single bulk cold
    // build of (K * CLUSTER_C_N) points.
    // =========================================================
    std::cout << "\n=== Benchmark C: K parallel cold builds vs bulk ==="
              << "\n";

    constexpr int    CLUSTER_C_N  = 100;   // pts per parallel cluster
    constexpr int    C_TRIALS     = 10;
    const std::vector<int> K_VALUES = {1, 2, 4, 8, 16, 32};
    const float      CLUSTER_C_R  = 0.08f;

    // CSV header
    csv << "\nbench,K,ClusterN,TotalN,trial,seqMs,parMs,bulkMs\n";

    const int WC = 12;
    std::cout << std::setw(WC) << "K"
              << std::setw(WC) << "TotalN"
              << std::setw(WC) << "Seq(ms)"
              << std::setw(WC) << "Par(ms)"
              << std::setw(WC) << "Bulk(ms)"
              << std::setw(WC) << "ParSpeedup"
              << "\n"
              << std::string(WC * 6, '-') << "\n";

    // Lambda: build one cold DT from a pre-generated cluster
    auto coldBuild = [](std::vector<glm::vec3> pts) -> double {
        auto t0 = Clock::now();
        Delaunay_3 dt;
        std::vector<std::pair<Point_3, size_t>> p;
        p.reserve(pts.size());
        for (size_t i = 0; i < pts.size(); ++i)
            p.emplace_back(Point_3(pts[i].x, pts[i].y, pts[i].z), i);
        dt.insert(p.begin(), p.end());
        (void)dt;
        return ms(Clock::now() - t0).count();
    };

    std::uniform_real_distribution<float> ud(0.1f, 0.9f);

    for (int K : K_VALUES) {
        int totalN = K * CLUSTER_C_N;
        double sumSeq = 0, sumPar = 0, sumBulk = 0;

        // Warmup
        {
            std::vector<std::future<double>> wfuts;
            for (int k = 0; k < K; ++k) {
                glm::vec3 wctr(ud(rng), ud(rng), ud(rng));
                std::vector<glm::vec3> wcp;
                generateCluster(CLUSTER_C_N, wctr, CLUSTER_C_R, wcp, rng);
                wfuts.push_back(std::async(std::launch::async, coldBuild, std::move(wcp)));
            }
            for (auto& f : wfuts) f.get();
        }

        for (int t = 0; t < C_TRIALS; ++t) {
            // Generate K disjoint cluster centres and point sets
            std::vector<std::vector<glm::vec3>> clusters(K);
            for (int k = 0; k < K; ++k) {
                glm::vec3 ctr(ud(rng), ud(rng), ud(rng));
                generateCluster(CLUSTER_C_N, ctr, CLUSTER_C_R, clusters[k], rng);
            }

            // --- Sequential: K cold builds one after another ---
            auto ts0 = Clock::now();
            for (int k = 0; k < K; ++k)
                coldBuild(clusters[k]);  // discard return
            double seqMs = ms(Clock::now() - ts0).count();

            // --- Parallel: K concurrent async cold builds ---
            auto tp0 = Clock::now();
            {
                std::vector<std::future<double>> futs;
                futs.reserve(K);
                for (int k = 0; k < K; ++k)
                    futs.push_back(std::async(std::launch::async, coldBuild, clusters[k]));
                for (auto& f : futs) f.get();
            }
            double parMs = ms(Clock::now() - tp0).count();

            // --- Bulk: single DT of all K*CLUSTER_C_N points ---
            auto tb0_c = Clock::now();
            {
                Delaunay_3 bdt;
                std::vector<std::pair<Point_3, size_t>> bpts;
                bpts.reserve(totalN);
                size_t idx = 0;
                for (int k = 0; k < K; ++k)
                    for (auto& p : clusters[k])
                        bpts.emplace_back(Point_3(p.x, p.y, p.z), idx++);
                bdt.insert(bpts.begin(), bpts.end());
                (void)bdt;
            }
            double bulkMs = ms(Clock::now() - tb0_c).count();

            csv << "c," << K << "," << CLUSTER_C_N << "," << totalN
                << "," << (t+1) << ","
                << std::fixed << std::setprecision(4)
                << seqMs << "," << parMs << "," << bulkMs << "\n";

            sumSeq  += seqMs;
            sumPar  += parMs;
            sumBulk += bulkMs;
        }

        double avgSeq  = sumSeq  / C_TRIALS;
        double avgPar  = sumPar  / C_TRIALS;
        double avgBulk = sumBulk / C_TRIALS;
        double parSpeedup = avgSeq / std::max(avgPar, 0.001);

        std::cout << std::fixed << std::setprecision(3)
                  << std::setw(WC) << K
                  << std::setw(WC) << totalN
                  << std::setw(WC) << avgSeq
                  << std::setw(WC) << avgPar
                  << std::setw(WC) << avgBulk
                  << std::setw(WC-2) << parSpeedup << "x"
                  << "\n";
    }

    std::cout << std::string(WC * 6, '-') << "\n";
    csv.flush();
    std::cout << "\nBenchmark C results appended to: " << csvPath.string() << "\n";
    return 0;
}
