// =============================================================================
// gpu_slab.cuh
// GpuSlabAllocator — flat GPU buffer manager for BVH nodes, CSR adjacency
// data, and per-particle render data.
//
// Each foam receives a pre-assigned contiguous slice in every flat buffer,
// so BVH and adjacency construction kernels can write results directly into
// the renderer's working memory via cudaMemcpyDeviceToDevice, eliminating the
// GPU→CPU→GPU round-trips of the per-foam fragmented allocation model.
//
// Overflow handling
// -----------------
//   resize(foam_id, ...)  Tombstones the old slot and appends a larger one at
//                         the watermark.  No GPU data movement needed for the
//                         old slot — the caller rebuilds into the new slice.
//   compact()             D→D packs all live slots to the slab front.  Call
//                         when the watermark approaches slab capacity.
//   Slab grow             When watermark + new_request exceeds total capacity,
//                         a new larger buffer is allocated, existing data is
//                         copied D→D, and the old buffer is freed.  This is
//                         the only site that calls cudaMalloc at runtime, and
//                         only occurs when the total live particle count
//                         genuinely exceeds the initial estimate.
// =============================================================================

#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "dynamic_foam/Sim2D/bvh.cuh"
#include "dynamic_foam/Sim2D/cuda_utils.cuh"

namespace DynamicFoam::Sim2D {



// =============================================================================
// FoamSlot — per-foam layout within each slab buffer.
// =============================================================================
struct FoamSlot {
    // BVH nodes: (2*n_particles - 1) BVHNode entries.
    int bvh_offset   = 0;
    int bvh_capacity = 0;

    // CSR node_offsets: (n_particles + 1) uint32 entries.
    int csr_node_offset   = 0;
    int csr_node_capacity = 0;

    // CSR nbrs: num_directed_edges uint32 entries.
    int csr_edge_offset   = 0;
    int csr_edge_capacity = 0;

    // Particle data (positions, colors, surface mask): n_particles entries.
    int particle_offset   = 0;
    int particle_capacity = 0;

    // Active IDs (surface + 1-hop neighbours): Morton-sorted positions within the
    // already-Morton-sorted particle arrays.  Count written by bulkMortonSort.
    int active_offset   = 0;
    int active_capacity = 0;
    int active_count    = 0;

    // COO edge list: (coo_count) pairs of (src, dst) uint32 entries each.
    // Populated by stageCOOData; consumed by GPU adjacency-list construction
    // kernels so that no full H2D COO transfer is needed at build time.
    int coo_offset   = 0;
    int coo_capacity = 0; ///< Per-buffer element capacity (same for src and dst).
    int coo_count    = 0; ///< Actual directed-edge count staged into the buffer.

    bool dead = false; // Tombstoned by resize(); ignored by kernels and compact().
};

// =============================================================================
// FoamUpdate — describes a batch of particle insertions and deletions for one
// foam.  All insertion buffers must be the same length; this is enforced by
// the constructor.
// =============================================================================
struct FoamUpdate {
    // Insertion buffers (one entry per new particle):
    std::vector<glm::vec3> particle_position_ins;
    std::vector<glm::vec4> particle_color_ins;
    std::vector<uint8_t>   particle_surface_mask_ins;
    std::vector<AABB>      particle_aabb_ins;
    std::vector<uint32_t>  particle_active_ids_ins;

    // Deletion buffer — entity IDs of particles to remove:
    std::vector<uint32_t> particle_id_dels;

    // Edge insertion buffers — directed COO edges to append after deletions.
    // An edge (coo_src_ins[i], coo_dst_ins[i]) is appended to the slab COO
    // buffers.  Both vectors must be the same length.
    std::vector<uint32_t> coo_src_ins;
    std::vector<uint32_t> coo_dst_ins;

    FoamUpdate() = default;

    FoamUpdate(
        std::vector<glm::vec3> positions,
        std::vector<glm::vec4> colors,
        std::vector<uint8_t>   surface_masks,
        std::vector<AABB>      aabbs,
        std::vector<uint32_t>  active_ids,
        std::vector<uint32_t>  del_ids)
        : particle_position_ins    (std::move(positions))
        , particle_color_ins       (std::move(colors))
        , particle_surface_mask_ins(std::move(surface_masks))
        , particle_aabb_ins        (std::move(aabbs))
        , particle_active_ids_ins  (std::move(active_ids))
        , particle_id_dels         (std::move(del_ids))
    {
        const size_t n = particle_position_ins.size();
        if (particle_color_ins.size()         != n ||
            particle_surface_mask_ins.size()  != n ||
            particle_aabb_ins.size()          != n ||
            particle_active_ids_ins.size()    != n)
        {
            throw std::invalid_argument(
                "FoamUpdate: all insertion buffers must be the same length");
        }
    }
};

// =============================================================================
// GpuSlabAllocator
// =============================================================================
class GpuSlabAllocator {
public:
    GpuSlabAllocator()  = default;
    ~GpuSlabAllocator() { free_all(); }

    // Non-copyable; device memory ownership is exclusive.
    GpuSlabAllocator(const GpuSlabAllocator&)             = delete;
    GpuSlabAllocator& operator=(const GpuSlabAllocator&) = delete;

    // -------------------------------------------------------------------------
    // init — allocate the underlying flat GPU buffers once at startup.
    // Arguments are lower-bound estimates; each buffer grows with a 2× strategy
    // if a subsequent allocate() call would overflow it.
    //
    //   total_bvh_nodes  — sum of (2*n_i − 1) across all initial foams.
    //   total_csr_nodes  — sum of (n_i + 1) across all initial foams.
    //   total_csr_edges  — sum of directed edge counts across all initial foams.
    //   total_particles  — sum of particle counts across all initial foams.
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // init — allocate the underlying flat GPU buffers once at startup.
    // Arguments are lower-bound estimates; each buffer grows with a 2× strategy
    // if a subsequent allocate() call would overflow it.
    //
    //   total_bvh_nodes  — sum of (2*n_i − 1) across all initial foams.
    //   total_csr_nodes  — sum of (n_i + 1) across all initial foams.
    //   total_csr_edges  — sum of directed edge counts across all initial foams.
    //   total_particles  — sum of particle counts across all initial foams.
    //   total_coo_edges  — sum of directed COO edge counts across all initial foams
    //                      (can be the same estimate as total_csr_edges).
    // -------------------------------------------------------------------------
    void init(int total_bvh_nodes, int total_csr_nodes,
              int total_csr_edges, int total_particles,
              int total_coo_edges = -1) {
        grow_bvh(total_bvh_nodes);
        grow_csr_nodes(total_csr_nodes);
        grow_csr_edges(total_csr_edges);
        grow_particles(total_particles);
        grow_active(total_particles);  // worst case: all particles active
        grow_coo(total_coo_edges >= 0 ? total_coo_edges : total_csr_edges);
        grow_foam_aabbs(8);            // grows with num_foams

        // Initial per-foam slab start tables (grown on demand).
        CUDA_CHECK(cudaMalloc(&d_foam_bvh_start,      8 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_foam_rowptr_start,   8 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_foam_particle_start, 8 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_foam_colidx_start,   8 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_foam_active_start,   8 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_foam_coo_start,      8 * sizeof(int)));
        offset_table_cap = 8;
        h_foam_bvh_start.assign(8, 0);
        h_foam_rowptr_start.assign(8, 0);
        h_foam_particle_start.assign(8, 0);
        h_foam_colidx_start.assign(8, 0);
        h_foam_active_start.assign(8, 0);
        h_foam_coo_start.assign(8, 0);
    }

    // -------------------------------------------------------------------------
    // allocate — reserve a 2× overcommitted slice for foam_id.
    // Returns a stable reference to the new FoamSlot.
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // allocate — reserve a 2× overcommitted slice for foam_id.
    // n_coo_edges is the directed COO edge count; pass -1 to match n_csr_edges.
    // Returns a stable reference to the new FoamSlot.
    // -------------------------------------------------------------------------
    FoamSlot& allocate(int foam_id,
                       int n_bvh_nodes, int n_csr_nodes,
                       int n_csr_edges, int n_particles,
                       int n_coo_edges = -1) {
        const int bvh_need  = n_bvh_nodes  * 2;
        const int cn_need   = n_csr_nodes  * 2;
        const int ce_need   = n_csr_edges  * 2;
        const int part_need = n_particles  * 2;
        const int coo_need  = (n_coo_edges >= 0 ? n_coo_edges : n_csr_edges) * 2;

        if (bvh_watermark       + bvh_need  > (int)bvh_cap)      grow_bvh(bvh_watermark       + bvh_need);
        if (csr_node_watermark  + cn_need   > (int)csr_node_cap)  grow_csr_nodes(csr_node_watermark  + cn_need);
        if (csr_edge_watermark  + ce_need   > (int)csr_edge_cap)  grow_csr_edges(csr_edge_watermark  + ce_need);
        if (particle_watermark  + part_need > (int)particle_cap)  grow_particles(particle_watermark  + part_need);
        if (active_watermark    + part_need > (int)active_cap)    grow_active(active_watermark    + part_need);
        if (coo_watermark       + coo_need  > (int)coo_cap)       grow_coo(coo_watermark       + coo_need);

        FoamSlot s;
        s.bvh_offset        = bvh_watermark;        s.bvh_capacity      = bvh_need;
        s.csr_node_offset   = csr_node_watermark;   s.csr_node_capacity = cn_need;
        s.csr_edge_offset   = csr_edge_watermark;   s.csr_edge_capacity = ce_need;
        s.particle_offset   = particle_watermark;   s.particle_capacity = part_need;
        s.active_offset     = active_watermark;     s.active_capacity   = part_need;
        s.coo_offset        = coo_watermark;        s.coo_capacity      = coo_need;
        s.coo_count         = 0;
        s.dead              = false;

        bvh_watermark       += bvh_need;
        csr_node_watermark  += cn_need;
        csr_edge_watermark  += ce_need;
        particle_watermark  += part_need;
        active_watermark    += part_need;
        coo_watermark       += coo_need;

        slots[foam_id] = s;
        num_foams = std::max(num_foams, foam_id + 1);
        grow_foam_aabbs(num_foams);  // ensure d_foam_aabbs[foam_id] is valid
        rebuild_and_upload_offset_tables();
        return slots.at(foam_id);
    }

    // -------------------------------------------------------------------------
    // resize — tombstone the old slot and allocate a larger one at the
    // watermark.  The caller is responsible for rebuilding BVH/CSR into the
    // new slice via buildBVH / buildGPUAdjacencyList.
    // -------------------------------------------------------------------------
    void resize(int foam_id,
                int new_bvh_nodes, int new_csr_nodes,
                int new_csr_edges, int new_particles,
                int new_coo_edges = -1) {
        slots.at(foam_id).dead = true;
        allocate(foam_id, new_bvh_nodes, new_csr_nodes, new_csr_edges, new_particles, new_coo_edges);
    }

    // -------------------------------------------------------------------------
    // needsResize — returns true if any of the actual counts exceed the
    // capacity of the foam's current slab slot.  Pass the exact current counts
    // (not pre-doubled); the method compares against the already-2×-padded slot
    // capacities.  Returns false when no slot exists yet (first-build path).
    // -------------------------------------------------------------------------
    bool needsResize(int foam_id, int n_bvh, int n_csr_nodes,
                      int n_csr_edges, int n_particles,
                      int n_coo_edges = -1) const {
        auto it = slots.find(foam_id);
        if (it == slots.end() || it->second.dead) return false;
        const FoamSlot& s = it->second;
        return n_bvh       > s.bvh_capacity      ||
               n_csr_nodes > s.csr_node_capacity  ||
               n_csr_edges > s.csr_edge_capacity  ||
               n_particles > s.particle_capacity  ||
               (n_coo_edges >= 0 && n_coo_edges > s.coo_capacity);
    }

    // -------------------------------------------------------------------------
    // needsCompaction — returns true when the wasted fraction of the BVH slab
    // (dead regions left behind by prior resizes) exceeds the given threshold.
    // Default: 50% waste triggers compaction.
    //
    // Call after updateTopology(); if true, call compact() followed by
    // rebuildAllSlabCsr() to fix the CSR node_offsets, which are invalidated
    // by the change in csr_edge_offset that compaction produces.
    // -------------------------------------------------------------------------
    bool needsCompaction(float wasted_fraction_threshold = 0.5f) const {
        if (bvh_watermark == 0) return false;
        int live_bvh = 0;
        for (const auto& [id, s] : slots)
            if (!s.dead) live_bvh += s.bvh_capacity;
        const float wasted =
            1.f - static_cast<float>(live_bvh) / static_cast<float>(bvh_watermark);
        return wasted > wasted_fraction_threshold;
    }

    // -------------------------------------------------------------------------
    // compact — D→D pack all live slots contiguously.
    // Resets watermarks and re-uploads offset tables.
    //
    // IMPORTANT: compact() moves the raw CSR node_offsets values, which were
    // biased by each foam's old csr_edge_offset.  After compaction those offsets
    // are stale (the edge blocks moved to new positions).  Call
    // rebuildAllSlabCsr() immediately after compact() to rewrite them.
    // -------------------------------------------------------------------------
    void compact() {
        // Sort live entries in address order (by BVH offset as proxy).
        std::vector<std::pair<int, int>> ordered; // {bvh_offset, foam_id}
        for (auto& [id, s] : slots)
            if (!s.dead) ordered.push_back({s.bvh_offset, id});
        std::sort(ordered.begin(), ordered.end());

        int bw = 0, cnw = 0, cew = 0, pw = 0, aw = 0, coow = 0;
        for (auto& [_, id] : ordered) {
            FoamSlot& s = slots[id];
            if (s.bvh_offset != bw)
                CUDA_CHECK(cudaMemcpy(d_bvh_nodes + bw,
                    d_bvh_nodes + s.bvh_offset,
                    s.bvh_capacity * sizeof(BVHNode), cudaMemcpyDeviceToDevice));
            s.bvh_offset = bw;  bw += s.bvh_capacity;

            if (s.csr_node_offset != cnw)
                CUDA_CHECK(cudaMemcpy(d_csr_rowptr + cnw,
                    d_csr_rowptr + s.csr_node_offset,
                    s.csr_node_capacity * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            s.csr_node_offset = cnw;  cnw += s.csr_node_capacity;

            if (s.csr_edge_offset != cew)
                CUDA_CHECK(cudaMemcpy(d_csr_colidx + cew,
                    d_csr_colidx + s.csr_edge_offset,
                    s.csr_edge_capacity * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            s.csr_edge_offset = cew;  cew += s.csr_edge_capacity;

            if (s.particle_offset != pw) {
                CUDA_CHECK(cudaMemcpy(d_particle_positions + pw,
                    d_particle_positions + s.particle_offset,
                    s.particle_capacity * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_particle_colors + pw,
                    d_particle_colors + s.particle_offset,
                    s.particle_capacity * sizeof(glm::vec4), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_particle_surface_mask + pw,
                    d_particle_surface_mask + s.particle_offset,
                    s.particle_capacity * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_particle_aabbs + pw,
                    d_particle_aabbs + s.particle_offset,
                    s.particle_capacity * sizeof(AABB), cudaMemcpyDeviceToDevice));
            }
            s.particle_offset = pw;  pw += s.particle_capacity;

            if (s.active_offset != aw)
                CUDA_CHECK(cudaMemcpy(d_active_ids + aw,
                    d_active_ids + s.active_offset,
                    s.active_capacity * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            s.active_offset = aw;  aw += s.active_capacity;

            if (s.coo_offset != coow) {
                CUDA_CHECK(cudaMemcpy(d_coo_src + coow,
                    d_coo_src + s.coo_offset,
                    s.coo_capacity * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
                CUDA_CHECK(cudaMemcpy(d_coo_dst + coow,
                    d_coo_dst + s.coo_offset,
                    s.coo_capacity * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            }
            s.coo_offset = coow;  coow += s.coo_capacity;
        }
        bvh_watermark      = bw;   csr_node_watermark = cnw;
        csr_edge_watermark = cew;  particle_watermark  = pw;
        active_watermark   = aw;   coo_watermark       = coow;

        rebuild_and_upload_offset_tables();
    }

    // -------------------------------------------------------------------------
    // biasCsrOffsets — after buildGPUAdjacencyList writes 0-based node_offsets
    // into a slab slice, this kernel adds the foam's csr_edge_offset so the
    // values are valid global indices into d_csr_colidx.
    // Must be called once per foam immediately after buildGPUAdjacencyList.
    // -------------------------------------------------------------------------
    void biasCsrOffsets(int foam_id);

    // -------------------------------------------------------------------------
    // Bulk upload of all per-particle data arrays for one foam in a single
    // call.  All five buffers must always be uploaded together to guarantee
    // that every slab array is consistent before bulkMortonSort runs.
    //
    //   h_aabbs     — local-space per-particle AABBs (BVH primitives / Morton centroids).
    //   h_colors    — per-particle RGBA color.
    //   h_positions — per-particle world-space position.
    //   h_mask      — per-particle surface mask (1 = surface, 0 = interior).
    //   h_ids       — per-particle entity IDs; reordered by bulkMortonSort so
    //                 d_active_ids[i] is the entity at Morton position i.
    //   n           — particle count (must match the slot's particle_capacity).
    // -------------------------------------------------------------------------
    void stageParticleData(int foam_id,
                           const AABB*      h_aabbs,
                           const glm::vec4* h_colors,
                           const glm::vec3* h_positions,
                           const uint8_t*   h_mask,
                           const uint32_t*  h_ids,
                           int n) {
        const FoamSlot& s = slots.at(foam_id);
        assert(n <= s.active_capacity && "stageParticleData: count exceeds slab capacity");
        CUDA_CHECK(cudaMemcpy(d_particle_aabbs     + s.particle_offset, h_aabbs,     n * sizeof(AABB),       cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_particle_colors    + s.particle_offset, h_colors,    n * sizeof(glm::vec4),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_particle_positions + s.particle_offset, h_positions, n * sizeof(glm::vec3),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_particle_surface_mask + s.particle_offset, h_mask,      n * sizeof(uint8_t),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_active_ids         + s.active_offset,   h_ids,       n * sizeof(uint32_t),   cudaMemcpyHostToDevice));
    }

    // -------------------------------------------------------------------------
    // stageCOOData — upload directed COO edge pairs for one foam in a single H2D
    // transfer.
    //   h_src      — host array of source node IDs for each directed edge.
    //   h_dst      — host array of destination node IDs for each directed edge.
    //   n_edges    — number of directed edges (must be <= slot.coo_capacity).
    // -------------------------------------------------------------------------
    void stageCOOData(int foam_id,
                      const uint32_t* h_src,
                      const uint32_t* h_dst,
                      int n_edges) {
        FoamSlot& s = slots.at(foam_id);
        assert(n_edges <= s.coo_capacity && "stageCOOData: n_edges exceeds slab COO capacity");
        CUDA_CHECK(cudaMemcpy(d_coo_src + s.coo_offset, h_src, n_edges * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_coo_dst + s.coo_offset, h_dst, n_edges * sizeof(uint32_t), cudaMemcpyHostToDevice));
        s.coo_count = n_edges;
    }

    // -------------------------------------------------------------------------
    // updateFoamData — applies a FoamUpdate to the particle slab buffers for
    // foam_id.  Deletions are resolved by searching d_active_ids for matching
    // entity IDs and compacting all five particle arrays.  Insertions are
    // appended after the surviving particles; if the post-update count would
    // exceed the slot's particle_capacity the slot is tombstoned and a larger
    // one is allocated at the watermark before insertion.
    //
    // Note: this method only updates the raw particle data arrays.  The caller
    // is responsible for rebuilding the BVH and CSR adjacency afterwards if
    // particle counts changed.
    // Implemented in gpu_slab.cu (requires nvcc).
    // -------------------------------------------------------------------------
    void updateFoamData(int foam_id, const FoamUpdate& update);

    // Bulk Morton sort — the single entry point for all foam data ordering.
    //
    // Requires that d_particle_aabbs, d_particle_colors, d_particle_positions,
    // d_particle_surface_mask, and d_active_ids have already been staged via
    // stageParticleData.
    //
    // This routine:
    //   1. Computes Morton codes for all n_particles from their local-space AABB
    //      centroids, normalized against the foam's own AABB.
    //   2. Gathers all per-particle slab buffers (AABBs, colors, positions,
    //      surface mask, active IDs) into Morton order via a single permutation.
    //   3. Sets slot.active_count = n_particles.
    //
    // Implemented in gpu_slab.cu (requires nvcc).
    void bulkMortonSort(int foam_id, int n_particles);

    // Reduce all n per-particle local-space AABBs for foam_id into a single
    // foam AABB, write it into d_foam_aabbs[foam_id] (device), and also copy
    // it to out_bbox (host).  Called automatically by bulkMortonSort.
    // Implemented in gpu_slab.cu (requires nvcc).
    void computeFoamAABB(int foam_id, int n, AABB& out_bbox);

    // =========================================================================
    // Flat device buffers — owned by the allocator, lifetime = simulation.
    // -------------------------------------------------------------------------
    BVHNode*   d_bvh_nodes  = nullptr; ///< All foams' BVH nodes packed end-to-end.
    uint32_t*  d_csr_rowptr = nullptr; ///< All foams' CSR row pointers (per-particle neighbor-list start) packed end-to-end.
    uint32_t*  d_csr_colidx = nullptr; ///< All foams' CSR column indices (neighbor particle ids) packed end-to-end.
    glm::vec3* d_particle_positions = nullptr; ///< Flat particle local-space positions.
    glm::vec4* d_particle_colors    = nullptr; ///< Flat particle RGBA colors.
    uint8_t*   d_particle_surface_mask = nullptr; ///< 1 = surface particle, 0 = interior.
    AABB*      d_particle_aabbs     = nullptr; ///< Local-space per-particle AABB (BVH primitives). Indexed by particle_offset.
    uint32_t*  d_active_ids         = nullptr; ///< Morton-sorted entity IDs. d_active_ids[slot.active_offset + i] = entity at Morton position i.
    AABB*      d_foam_aabbs         = nullptr; ///< Local-space AABB for each foam, indexed directly by foam_id (size = num_foams).
    uint32_t*  d_coo_src            = nullptr; ///< Flat COO source node IDs — all foams packed end-to-end.
    uint32_t*  d_coo_dst            = nullptr; ///< Flat COO destination node IDs — all foams packed end-to-end.

    // Per-foam slab start tables — one int per foam, indexed by foam_id.
    // d_foam_bvh_start[fid]      = slot.bvh_offset       (start in d_bvh_nodes)
    // d_foam_rowptr_start[fid]   = slot.csr_node_offset  (start in d_csr_rowptr)
    // d_foam_colidx_start[fid]   = slot.csr_edge_offset  (start in d_csr_colidx — not currently used by kernels but available)
    // d_foam_particle_start[fid] = slot.particle_offset  (start in d_particle_positions etc.)
    // d_foam_coo_start[fid]      = slot.coo_offset        (start in d_coo_src / d_coo_dst)
    int* d_foam_bvh_start      = nullptr;
    int* d_foam_rowptr_start   = nullptr;
    int* d_foam_colidx_start   = nullptr;
    int* d_foam_particle_start = nullptr;
    int* d_foam_active_start   = nullptr; ///< Active-IDs slice start per foam (indexed by foam_id).
    int* d_foam_coo_start      = nullptr; ///< COO slice start per foam (indexed by foam_id).

    int num_foams = 0; ///< Max foam_id + 1 (offset table stride).

    // Host-side slot registry (foam_id → FoamSlot).
    std::unordered_map<int, FoamSlot> slots;

private:
    // Watermarks — next free element in each slab.
    int bvh_watermark      = 0;
    int csr_node_watermark = 0;
    int csr_edge_watermark = 0;
    int particle_watermark = 0;
    int active_watermark   = 0;
    int coo_watermark      = 0;

    // Slab capacities (element counts).
    size_t bvh_cap        = 0;
    size_t csr_node_cap   = 0;
    size_t csr_edge_cap   = 0;
    size_t particle_cap   = 0;
    size_t active_cap     = 0;
    size_t coo_cap        = 0;
    size_t foam_aabb_cap_ = 0;

    // Host mirrors of the per-foam slab start tables.
    std::vector<int> h_foam_bvh_start;
    std::vector<int> h_foam_rowptr_start;
    std::vector<int> h_foam_colidx_start;
    std::vector<int> h_foam_particle_start;
    std::vector<int> h_foam_active_start;
    std::vector<int> h_foam_coo_start;
    size_t offset_table_cap = 0;

    // -------------------------------------------------------------------------
    // Slab grow helpers — allocate a larger buffer, D→D copy live data, free old.
    // -------------------------------------------------------------------------
    void grow_bvh(int required) {
        const size_t nc = std::max<size_t>(required * 2, 64);
        BVHNode* d_new = nullptr;
        CUDA_CHECK(cudaMalloc(&d_new, nc * sizeof(BVHNode)));
        if (d_bvh_nodes && bvh_watermark > 0)
            CUDA_CHECK(cudaMemcpy(d_new, d_bvh_nodes,
                bvh_watermark * sizeof(BVHNode), cudaMemcpyDeviceToDevice));
        if (d_bvh_nodes) CUDA_CHECK(cudaFree(d_bvh_nodes));
        d_bvh_nodes = d_new;  bvh_cap = nc;
    }

    void grow_csr_nodes(int required) {
        const size_t nc = std::max<size_t>(required * 2, 64);
        uint32_t* d_new = nullptr;
        CUDA_CHECK(cudaMalloc(&d_new, nc * sizeof(uint32_t)));
        if (d_csr_rowptr && csr_node_watermark > 0)
            CUDA_CHECK(cudaMemcpy(d_new, d_csr_rowptr,
                csr_node_watermark * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        if (d_csr_rowptr) CUDA_CHECK(cudaFree(d_csr_rowptr));
        d_csr_rowptr = d_new;  csr_node_cap = nc;
    }

    void grow_csr_edges(int required) {
        const size_t nc = std::max<size_t>(required * 2, 64);
        uint32_t* d_new = nullptr;
        CUDA_CHECK(cudaMalloc(&d_new, nc * sizeof(uint32_t)));
        if (d_csr_colidx && csr_edge_watermark > 0)
            CUDA_CHECK(cudaMemcpy(d_new, d_csr_colidx,
                csr_edge_watermark * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        if (d_csr_colidx) CUDA_CHECK(cudaFree(d_csr_colidx));
        d_csr_colidx = d_new;  csr_edge_cap = nc;
    }

    void grow_particles(int required) {
        const size_t nc = std::max<size_t>(required * 2, 64);
        glm::vec3* dp = nullptr;
        glm::vec4* dc = nullptr;
        uint8_t*   ds = nullptr;
        AABB*      da = nullptr;
        CUDA_CHECK(cudaMalloc(&dp, nc * sizeof(glm::vec3)));
        CUDA_CHECK(cudaMalloc(&dc, nc * sizeof(glm::vec4)));
        CUDA_CHECK(cudaMalloc(&ds, nc * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&da, nc * sizeof(AABB)));
        if (d_particle_positions && particle_watermark > 0) {
            CUDA_CHECK(cudaMemcpy(dp, d_particle_positions,
                particle_watermark * sizeof(glm::vec3), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(dc, d_particle_colors,
                particle_watermark * sizeof(glm::vec4), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(ds, d_particle_surface_mask,
                particle_watermark * sizeof(uint8_t), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(da, d_particle_aabbs,
                particle_watermark * sizeof(AABB), cudaMemcpyDeviceToDevice));
        }
        if (d_particle_positions) CUDA_CHECK(cudaFree(d_particle_positions));
        if (d_particle_colors)    CUDA_CHECK(cudaFree(d_particle_colors));
        if (d_particle_surface_mask) CUDA_CHECK(cudaFree(d_particle_surface_mask));
        if (d_particle_aabbs)     CUDA_CHECK(cudaFree(d_particle_aabbs));
        d_particle_positions = dp;
        d_particle_colors    = dc;
        d_particle_surface_mask = ds;
        d_particle_aabbs     = da;
        particle_cap = nc;
    }

    void grow_active(int required) {
        const size_t nc = std::max<size_t>(required * 2, 64);
        uint32_t* da = nullptr;
        CUDA_CHECK(cudaMalloc(&da, nc * sizeof(uint32_t)));
        if (d_active_ids && active_watermark > 0)
            CUDA_CHECK(cudaMemcpy(da, d_active_ids,
                active_watermark * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        if (d_active_ids) CUDA_CHECK(cudaFree(d_active_ids));
        d_active_ids = da;
        active_cap = nc;
    }

    void grow_coo(int required) {
        const size_t nc = std::max<size_t>(required * 2, 64);
        uint32_t* ds = nullptr;
        uint32_t* dd = nullptr;
        CUDA_CHECK(cudaMalloc(&ds, nc * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dd, nc * sizeof(uint32_t)));
        if (d_coo_src && coo_watermark > 0) {
            CUDA_CHECK(cudaMemcpy(ds, d_coo_src,
                coo_watermark * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(dd, d_coo_dst,
                coo_watermark * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
        }
        if (d_coo_src) CUDA_CHECK(cudaFree(d_coo_src));
        if (d_coo_dst) CUDA_CHECK(cudaFree(d_coo_dst));
        d_coo_src = ds;
        d_coo_dst = dd;
        coo_cap = nc;
    }

    void grow_foam_aabbs(int required) {
        if ((size_t)required <= foam_aabb_cap_) return;
        const size_t nc = std::max<size_t>(required * 2, 8);
        AABB* d_new = nullptr;
        CUDA_CHECK(cudaMalloc(&d_new, nc * sizeof(AABB)));
        // Zero-init so un-used slots have a valid empty AABB.
        CUDA_CHECK(cudaMemset(d_new, 0, nc * sizeof(AABB)));
        if (d_foam_aabbs && foam_aabb_cap_ > 0)
            CUDA_CHECK(cudaMemcpy(d_new, d_foam_aabbs,
                foam_aabb_cap_ * sizeof(AABB), cudaMemcpyDeviceToDevice));
        if (d_foam_aabbs) CUDA_CHECK(cudaFree(d_foam_aabbs));
        d_foam_aabbs  = d_new;
        foam_aabb_cap_ = nc;
    }

    void grow_offset_tables(int required) {
        if ((size_t)required <= offset_table_cap) return;
        const size_t nc = std::max<size_t>(required * 2, 8);
        auto realloc_ints = [&](int*& ptr) {
            int* d_new = nullptr;
            CUDA_CHECK(cudaMalloc(&d_new, nc * sizeof(int)));
            if (ptr) CUDA_CHECK(cudaFree(ptr));
            ptr = d_new;
        };
        realloc_ints(d_foam_bvh_start);
        realloc_ints(d_foam_rowptr_start);
        realloc_ints(d_foam_colidx_start);
        realloc_ints(d_foam_particle_start);
        realloc_ints(d_foam_active_start);
        realloc_ints(d_foam_coo_start);
        h_foam_bvh_start.resize(nc, 0);
        h_foam_rowptr_start.resize(nc, 0);
        h_foam_colidx_start.resize(nc, 0);
        h_foam_particle_start.resize(nc, 0);
        h_foam_active_start.resize(nc, 0);
        h_foam_coo_start.resize(nc, 0);
        offset_table_cap = nc;
    }

    void rebuild_and_upload_offset_tables() {
        grow_offset_tables(num_foams);
        for (auto& [id, s] : slots) {
            if (s.dead) continue;
            h_foam_bvh_start[id]      = s.bvh_offset;
            h_foam_rowptr_start[id]   = s.csr_node_offset;
            h_foam_colidx_start[id]   = s.csr_edge_offset;
            h_foam_particle_start[id] = s.particle_offset;
            h_foam_active_start[id]   = s.active_offset;
            h_foam_coo_start[id]      = s.coo_offset;
        }
        CUDA_CHECK(cudaMemcpy(d_foam_bvh_start, h_foam_bvh_start.data(),
            num_foams * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_foam_rowptr_start, h_foam_rowptr_start.data(),
            num_foams * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_foam_colidx_start, h_foam_colidx_start.data(),
            num_foams * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_foam_particle_start, h_foam_particle_start.data(),
            num_foams * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_foam_active_start, h_foam_active_start.data(),
            num_foams * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_foam_coo_start, h_foam_coo_start.data(),
            num_foams * sizeof(int), cudaMemcpyHostToDevice));
    }

    void free_all() {
        auto f = [](void* p) { if (p) cudaFree(p); };
        f(d_bvh_nodes);           d_bvh_nodes           = nullptr;
        f(d_csr_rowptr);          d_csr_rowptr          = nullptr;
        f(d_csr_colidx);          d_csr_colidx          = nullptr;
        f(d_particle_positions);  d_particle_positions  = nullptr;
        f(d_particle_colors);     d_particle_colors     = nullptr;
        f(d_particle_surface_mask); d_particle_surface_mask = nullptr;
        f(d_particle_aabbs);      d_particle_aabbs      = nullptr;
        f(d_active_ids);          d_active_ids          = nullptr;
        f(d_foam_aabbs);          d_foam_aabbs          = nullptr;
        f(d_coo_src);             d_coo_src             = nullptr;
        f(d_coo_dst);             d_coo_dst             = nullptr;
        f(d_foam_bvh_start);      d_foam_bvh_start      = nullptr;
        f(d_foam_rowptr_start);   d_foam_rowptr_start   = nullptr;
        f(d_foam_colidx_start);   d_foam_colidx_start   = nullptr;
        f(d_foam_particle_start); d_foam_particle_start = nullptr;
        f(d_foam_active_start);   d_foam_active_start   = nullptr;
        f(d_foam_coo_start);      d_foam_coo_start      = nullptr;
        bvh_cap = csr_node_cap = csr_edge_cap = particle_cap = active_cap = coo_cap = foam_aabb_cap_ = offset_table_cap = 0;
    }
};

} // namespace DynamicFoam::Sim2D
