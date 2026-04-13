// =============================================================================
// adjacency.cu
// nvcc-compiled implementations of AdjacencyList GPU methods.
//
// Separated from adjacency.cuh so these bodies are visible to MSVC-compiled
// translation units (simulation.cpp) via a plain declaration in the header —
// without requiring __CUDACC__ guards.
// =============================================================================

#include <cub/cub.cuh>
#include <thrust/scatter.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

#include "dynamic_foam/Sim2D/adjacency.cuh"
#include "dynamic_foam/Sim2D/cuda_utils.cuh"

namespace DynamicFoam::Sim2D {

namespace {

// Remap raw node IDs in a COO buffer to their sorted positions.
// inverse_map[original_id] = sorted_position
__global__ void remapCOOKernel(
    const uint32_t* __restrict__ raw,
    uint32_t*       __restrict__ remapped,
    const uint32_t* __restrict__ inverse_map,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) remapped[i] = inverse_map[raw[i]];
}

// Build inverse map from sorted index array.
// scatter: inverse_map[sorted_ids[i]] = i
__global__ void buildInverseMapKernel(
    const uint32_t* __restrict__ sorted_ids,
    uint32_t*       __restrict__ inverse_map,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) inverse_map[sorted_ids[i]] = i;
}

} // anonymous namespace

void AdjacencyList::buildGPUAdjacencyListImpl(
    AdjacencyListGPU&            out,
    const std::vector<uint32_t>& sorted_ids,
    cudaStream_t                 stream) const
{
    const bool     use_subset = !sorted_ids.empty();
    const uint32_t N = use_subset
                       ? static_cast<uint32_t>(sorted_ids.size())
                       : static_cast<uint32_t>(adj.size());

    // Ensure the COO cache is current before we read it.
    if (coo_dirty) rebuildCOO();

    // Filter edges on the CPU — only intra-subset edges are uploaded.
    // For the full-graph path the internal coo_src/coo_dst are used as-is.
    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> subset_coo;
    if (use_subset)
        subset_coo = getSubsetCOO(sorted_ids);

    const std::vector<uint32_t>& src_buf = use_subset ? subset_coo.first  : coo_src;
    const std::vector<uint32_t>& dst_buf = use_subset ? subset_coo.second : coo_dst;
    const uint32_t E = static_cast<uint32_t>(src_buf.size());

    if (N == 0 || E == 0) return;

    out.num_nodes = N;
    out.num_edges = E;

    // ------------------------------------------------------------------
    // Grow persistent output buffers as needed.
    // Non-owned buffers (slab slices) must already have enough capacity;
    // we assert rather than reallocate them.
    // ------------------------------------------------------------------
    cuda_realloc_if_needed(&out.nodes, &out.nodes_capacity, N);
    if (out.nbrs_owned) {
        cuda_realloc_if_needed(&out.nbrs, &out.nbrs_capacity, E);
    } else {
        assert(out.nbrs_capacity >= E &&
               "slab nbrs slice is too small for this foam");
    }
    if (out.node_offsets_owned) {
        cuda_realloc_if_needed(&out.node_offsets, &out.node_offsets_capacity, N + 1);
    } else {
        assert(out.node_offsets_capacity >= N + 1 &&
               "slab node_offsets slice is too small for this foam");
    }

    // ------------------------------------------------------------------
    // Step 1 — upload node ordering (pure H2D, no D2H anywhere)
    // ------------------------------------------------------------------
    if (use_subset) {
        CUDA_CHECK(cudaMemcpy(
            out.nodes, sorted_ids.data(),
            N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    } else {
        std::vector<uint32_t> h_nodes;
        h_nodes.reserve(N);
        for (const auto& [node, _] : adj)
            h_nodes.push_back(node);
        CUDA_CHECK(cudaMemcpy(
            out.nodes, h_nodes.data(),
            N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // ------------------------------------------------------------------
    // Step 2 — build inverse map: node_id -> sorted position
    //
    // Entity IDs are assigned globally by entt (not per-foam), so the max
    // ID can be >> N. Allocate the map to max_entity_id + 1 and zero-fill
    // unused slots to avoid out-of-bounds writes from buildInverseMapKernel.
    // When using a subset the COO only contains subset IDs, so sizing to
    // the subset max is sufficient. We still scan adj as a safe
    // upper bound that covers both paths.
    // ------------------------------------------------------------------
    uint32_t max_entity_id = 0;
    for (const auto& [node, _] : adj)
        max_entity_id = std::max(max_entity_id, static_cast<uint32_t>(node));

    uint32_t* d_inverse_map = nullptr;
    const uint32_t inv_map_size = max_entity_id + 1;
    CUDA_CHECK(cudaMalloc(&d_inverse_map, inv_map_size * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_inverse_map, 0, inv_map_size * sizeof(uint32_t)));

    buildInverseMapKernel<<<grid_size(N), 256, 0, stream>>>(
        out.nodes,
        d_inverse_map, N);

    // ------------------------------------------------------------------
    // Step 3 — upload COO and remap both ends to sorted positions
    //
    // src_buf/dst_buf are either the full graph COO or a caller-supplied
    // pre-filtered subset COO built by getSubsetCOO() on the CPU.
    // ------------------------------------------------------------------
    uint32_t* d_coo_src = nullptr;
    uint32_t* d_coo_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_coo_src, E * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_coo_dst, E * sizeof(uint32_t)));

    CUDA_CHECK(cudaMemcpy(d_coo_src, src_buf.data(),
        E * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coo_dst, dst_buf.data(),
        E * sizeof(uint32_t), cudaMemcpyHostToDevice));

    remapCOOKernel<<<grid_size(E), 256, 0, stream>>>(
        d_coo_src, d_coo_src, d_inverse_map, E);
    remapCOOKernel<<<grid_size(E), 256, 0, stream>>>(
        d_coo_dst, d_coo_dst, d_inverse_map, E);

    // ------------------------------------------------------------------
    // Step 4 — radix sort COO by source
    //
    // After sorting, d_coo_dst_sorted is already the final nbrs buffer
    // (neighbor indices grouped by node, in sorted-position space).
    // ------------------------------------------------------------------
    uint32_t* d_coo_src_sorted = nullptr;
    uint32_t* d_coo_dst_sorted = nullptr;
    CUDA_CHECK(cudaMalloc(&d_coo_src_sorted, E * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_coo_dst_sorted, E * sizeof(uint32_t)));

    CUB_CALL(cub::DeviceRadixSort::SortPairs,
        d_coo_src, d_coo_src_sorted,
        d_coo_dst, d_coo_dst_sorted,
        E, 0, sizeof(uint32_t) * 8, stream);

    CUDA_CHECK(cudaMemcpy(out.nbrs, d_coo_dst_sorted,
        E * sizeof(uint32_t), cudaMemcpyDeviceToDevice));

    // ------------------------------------------------------------------
    // Step 5 — run-length encode sorted src -> per-node degrees
    //
    // Because src is sorted, RLE gives exact degrees with no atomics.
    // Isolated nodes (degree 0) won't appear in the output -- we handle
    // them by zeroing d_degrees first and scattering only the live runs.
    // ------------------------------------------------------------------
    uint32_t* d_unique_nodes = nullptr;
    uint32_t* d_run_lengths  = nullptr;
    uint32_t* d_num_runs     = nullptr;
    CUDA_CHECK(cudaMalloc(&d_unique_nodes, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_run_lengths,  N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_num_runs,     sizeof(uint32_t)));

    CUB_CALL(cub::DeviceRunLengthEncode::Encode,
        d_coo_src_sorted,
        d_unique_nodes,
        d_run_lengths,
        d_num_runs,
        E, stream);

    uint32_t h_num_runs = 0;
    CUDA_CHECK(cudaMemcpy(&h_num_runs, d_num_runs,
        sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint32_t* d_degrees = nullptr;
    CUDA_CHECK(cudaMalloc(&d_degrees, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_degrees, 0, N * sizeof(uint32_t)));

    // Scatter run lengths into degree array at their sorted positions.
    // d_unique_nodes[i] is already a sorted-position index (post-remap).
    thrust::scatter(
        thrust::cuda::par.on(stream),
        thrust::device_ptr<uint32_t>(d_run_lengths),
        thrust::device_ptr<uint32_t>(d_run_lengths + h_num_runs),
        thrust::device_ptr<uint32_t>(d_unique_nodes),
        thrust::device_ptr<uint32_t>(d_degrees));

    // ------------------------------------------------------------------
    // Step 6 — exclusive scan over degrees -> node_offsets
    // ------------------------------------------------------------------
    CUB_CALL(cub::DeviceScan::ExclusiveSum,
        d_degrees, out.node_offsets, N, stream);

    // Write sentinel: node_offsets[N] = total edge count
    CUDA_CHECK(cudaMemcpy(out.node_offsets + N, &E,
        sizeof(uint32_t), cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------
    // Cleanup temporaries
    // ------------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_inverse_map));
    CUDA_CHECK(cudaFree(d_coo_src));
    CUDA_CHECK(cudaFree(d_coo_dst));
    CUDA_CHECK(cudaFree(d_coo_src_sorted));
    CUDA_CHECK(cudaFree(d_coo_dst_sorted));
    CUDA_CHECK(cudaFree(d_unique_nodes));
    CUDA_CHECK(cudaFree(d_run_lengths));
    CUDA_CHECK(cudaFree(d_num_runs));
    CUDA_CHECK(cudaFree(d_degrees));
}

void AdjacencyList::buildGPUAdjacencyList(
    AdjacencyListGPU&            out,
    uint32_t*                    d_nbrs_slice,
    size_t                       nbrs_cap,
    uint32_t*                    d_node_offsets_slice,
    size_t                       node_offsets_cap,
    const std::vector<uint32_t>& sorted_ids,
    cudaStream_t                 stream) const
{
    // Free any previously owned nbrs / node_offsets buffers, then adopt
    // the slices without taking ownership.
    if (out.nbrs         && out.nbrs_owned)         { cudaFree(out.nbrs);         }
    if (out.node_offsets && out.node_offsets_owned) { cudaFree(out.node_offsets); }

    out.nbrs                  = d_nbrs_slice;
    out.nbrs_capacity         = nbrs_cap;
    out.nbrs_owned            = false;

    out.node_offsets          = d_node_offsets_slice;
    out.node_offsets_capacity = node_offsets_cap;
    out.node_offsets_owned    = false;

    buildGPUAdjacencyListImpl(out, sorted_ids, stream);
}

} // namespace DynamicFoam::Sim2D
