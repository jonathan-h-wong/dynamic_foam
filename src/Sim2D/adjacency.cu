// =============================================================================
// adjacency.cu
// nvcc-compiled implementation of the buildGPUAdjacencyList free function.
//
// Separated from adjacency.cuh so the declaration is visible to
// MSVC-compiled translation units (simulation.cpp) without __CUDACC__ guards.
// =============================================================================

#include <cub/cub.cuh>
#include <thrust/scatter.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "dynamic_foam/Sim2D/adjacency.cuh"
#include "dynamic_foam/Sim2D/cuda_utils.cuh"

namespace DynamicFoam::Sim2D {

namespace {

// Remap each entry in coo_ids from raw entity-ID space to its Morton-sorted
// position, using a binary search over the entity-sorted lookup table.
//
//   coo_ids            — raw entity IDs from the COO buffer (one per edge end)
//   remapped           — output Morton-sorted positions
//   sorted_entity_ids  — N entity IDs sorted ascending by entity value
//                        (d_sorted_entity_ids + slot.active_offset)
//   entity_sorted_pos  — corresponding Morton positions
//                        (d_entity_sorted_pos + slot.active_offset)
//   N                  — search-space size (particle count for this foam)
//   E                  — number of COO entries to process
__global__ void remapCOOBinarySearch(
    const uint32_t* __restrict__ coo_ids,
    uint32_t*       __restrict__ remapped,
    const uint32_t* __restrict__ sorted_entity_ids,
    const uint32_t* __restrict__ entity_sorted_pos,
    uint32_t N,
    uint32_t E)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= E) return;
    const uint32_t target = coo_ids[i];
    // Lower-bound binary search in sorted_entity_ids[0..N).
    uint32_t lo = 0, hi = N;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (sorted_entity_ids[mid] < target) lo = mid + 1;
        else                                  hi = mid;
    }
    remapped[i] = entity_sorted_pos[lo];
}

// Fills arr[i] = i for i in [0, n).
__global__ void k_iota(uint32_t* arr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) arr[i] = static_cast<uint32_t>(i);
}

} // anonymous namespace

// =============================================================================
// buildGPUAdjacencyList — fully GPU-resident CSR builder.
//
// Uses pre-staged device buffers (Morton-sorted IDs and COO edge pairs) to
// build the CSR adjacency without any H2D transfers.  The only D2H traffic
// is one scalar D2H read (run count from CUB RunLengthEncode).
// COO remapping uses O(N) binary search instead of an O(max_entity_id) inverse map.
// =============================================================================
void buildGPUAdjacencyList(
    AdjacencyListGPU& out,
    const uint32_t*   d_sorted_ids,
    uint32_t          N,
    const uint32_t*   d_coo_src,
    const uint32_t*   d_coo_dst,
    uint32_t          E,
    uint32_t*         d_nbrs_slice,
    size_t            nbrs_cap,
    uint32_t*         d_node_offsets_slice,
    size_t            node_offsets_cap,
    cudaStream_t      stream)
{
    if (N == 0 || E == 0) return;

    // Wire slab slices into out (non-owned).
    if (out.nbrs         && out.nbrs_owned)         { cudaFree(out.nbrs);         }
    if (out.node_offsets && out.node_offsets_owned) { cudaFree(out.node_offsets); }

    out.nbrs                  = d_nbrs_slice;
    out.nbrs_capacity         = nbrs_cap;
    out.nbrs_owned            = false;
    out.node_offsets          = d_node_offsets_slice;
    out.node_offsets_capacity = node_offsets_cap;
    out.node_offsets_owned    = false;
    out.num_nodes             = N;
    out.num_edges             = E;

    // D2D copy sorted IDs into out.nodes (owned buffer for caller queries).
    cuda_realloc_if_needed(&out.nodes, &out.nodes_capacity, N);
    CUDA_CHECK(cudaMemcpyAsync(out.nodes, d_sorted_ids,
        N * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));

    // ------------------------------------------------------------------
    // Step 2 — build entity-ID → Morton-position lookup as local scratch.
    //
    // d_sorted_ids is Morton-sorted (entity IDs in spatial order). Sort a
    // copy by entity-ID value paired with an iota of Morton positions so
    // that remapCOOBinarySearch can binary-search for any raw entity ID and
    // retrieve its Morton position in O(log N) with no slab allocation.
    // ------------------------------------------------------------------
    uint32_t* d_entity_ids_sorted = nullptr;  // entity IDs sorted ascending
    uint32_t* d_entity_morton_idx = nullptr;  // corresponding Morton positions
    uint32_t* d_morton_pos_iota   = nullptr;  // identity [0 .. N-1]
    CUDA_CHECK(cudaMalloc(&d_entity_ids_sorted, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_entity_morton_idx, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_morton_pos_iota,   N * sizeof(uint32_t)));

    k_iota<<<grid_size(N), 256, 0, stream>>>(d_morton_pos_iota, N);
    CUDA_CHECK(cudaGetLastError());

    // Sort (entity_id → Morton position) pairs by entity_id ascending.
    CUB_CALL(cub::DeviceRadixSort::SortPairs,
        d_sorted_ids,      d_entity_ids_sorted,
        d_morton_pos_iota, d_entity_morton_idx,
        static_cast<int>(N), 0, sizeof(uint32_t) * 8, stream);

    // ------------------------------------------------------------------
    // Step 3 — remap COO entity IDs to Morton-sorted positions via
    //           binary search in the entity-sorted lookup table.
    // ------------------------------------------------------------------
    uint32_t* d_remapped_src = nullptr;
    uint32_t* d_remapped_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_remapped_src, E * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_remapped_dst, E * sizeof(uint32_t)));

    remapCOOBinarySearch<<<grid_size(E), 256, 0, stream>>>(
        d_coo_src, d_remapped_src, d_entity_ids_sorted, d_entity_morton_idx, N, E);
    remapCOOBinarySearch<<<grid_size(E), 256, 0, stream>>>(
        d_coo_dst, d_remapped_dst, d_entity_ids_sorted, d_entity_morton_idx, N, E);

    // ------------------------------------------------------------------
    // Step 4 — radix sort COO by source
    // ------------------------------------------------------------------
    uint32_t* d_coo_src_sorted = nullptr;
    uint32_t* d_coo_dst_sorted = nullptr;
    CUDA_CHECK(cudaMalloc(&d_coo_src_sorted, E * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_coo_dst_sorted, E * sizeof(uint32_t)));

    CUB_CALL(cub::DeviceRadixSort::SortPairs,
        d_remapped_src, d_coo_src_sorted,
        d_remapped_dst, d_coo_dst_sorted,
        static_cast<int>(E), 0, sizeof(uint32_t) * 8, stream);

    CUDA_CHECK(cudaMemcpyAsync(out.nbrs, d_coo_dst_sorted,
        E * sizeof(uint32_t), cudaMemcpyDeviceToDevice, stream));

    // ------------------------------------------------------------------
    // Step 5 — run-length encode sorted src -> per-node degrees
    // ------------------------------------------------------------------
    uint32_t* d_unique_nodes = nullptr;
    uint32_t* d_run_lengths  = nullptr;
    uint32_t* d_num_runs_dev = nullptr;
    CUDA_CHECK(cudaMalloc(&d_unique_nodes, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_run_lengths,  N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_num_runs_dev, sizeof(uint32_t)));

    CUB_CALL(cub::DeviceRunLengthEncode::Encode,
        d_coo_src_sorted,
        d_unique_nodes,
        d_run_lengths,
        d_num_runs_dev,
        static_cast<int>(E), stream);

    uint32_t h_num_runs = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_num_runs, d_num_runs_dev,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    uint32_t* d_degrees = nullptr;
    CUDA_CHECK(cudaMalloc(&d_degrees, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemsetAsync(d_degrees, 0, N * sizeof(uint32_t), stream));

    thrust::scatter(
        thrust::cuda::par.on(stream),
        thrust::device_ptr<uint32_t>(d_run_lengths),
        thrust::device_ptr<uint32_t>(d_run_lengths + h_num_runs),
        thrust::device_ptr<uint32_t>(d_unique_nodes),
        thrust::device_ptr<uint32_t>(d_degrees));

    // ------------------------------------------------------------------
    // Step 6 — exclusive scan over degrees -> node_offsets + sentinel
    // ------------------------------------------------------------------
    CUB_CALL(cub::DeviceScan::ExclusiveSum,
        d_degrees, out.node_offsets, static_cast<int>(N), stream);

    CUDA_CHECK(cudaMemcpyAsync(out.node_offsets + N, &E,
        sizeof(uint32_t), cudaMemcpyHostToDevice, stream));

    // ------------------------------------------------------------------
    // Cleanup temporaries
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_entity_ids_sorted));
    CUDA_CHECK(cudaFree(d_entity_morton_idx));
    CUDA_CHECK(cudaFree(d_morton_pos_iota));
    CUDA_CHECK(cudaFree(d_remapped_src));
    CUDA_CHECK(cudaFree(d_remapped_dst));
    CUDA_CHECK(cudaFree(d_coo_src_sorted));
    CUDA_CHECK(cudaFree(d_coo_dst_sorted));
    CUDA_CHECK(cudaFree(d_unique_nodes));
    CUDA_CHECK(cudaFree(d_run_lengths));
    CUDA_CHECK(cudaFree(d_num_runs_dev));
    CUDA_CHECK(cudaFree(d_degrees));
}

} // namespace DynamicFoam::Sim2D
