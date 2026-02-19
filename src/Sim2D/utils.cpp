// TODO: 
// Add position + quaternion => 4x4 matrix conversion utilities

#pragma once
#include <random>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <glm/mat3x3.hpp>

#include "dynamic_foam/sim2d/utils.h"
#include "dynamic_foam/sim2d/adjacency_list.h"

namespace DynamicFoam::Sim2D {

// CUDA kernel for Monte Carlo area integration
__global__ void monteCarloAreaKernel(
    const float* positions,           // [x0, y0, x1, y1, ...]
    const float* bboxes,              // [minX, minY, maxX, maxY, ...] per particle
    const int* neighborOffsets,       // Offset into neighbors array for each particle
    const int* neighborCounts,        // Number of neighbors for each particle
    const int* neighbors,             // Flattened neighbor indices
    float* areas,                     // Output areas
    int numParticles,
    int samplesPerCell,
    unsigned int seed
) {
    int particleIdx = blockIdx.x;
    if (particleIdx >= numParticles) return;
    
    // Get particle generator position
    float genX = positions[2 * particleIdx];
    float genY = positions[2 * particleIdx + 1];
    
    // Get bounding box
    float minX = bboxes[4 * particleIdx];
    float minY = bboxes[4 * particleIdx + 1];
    float maxX = bboxes[4 * particleIdx + 2];
    float maxY = bboxes[4 * particleIdx + 3];
    float bboxArea = (maxX - minX) * (maxY - minY);
    
    // Get neighbor info
    int neighborOffset = neighborOffsets[particleIdx];
    int neighborCount = neighborCounts[particleIdx];
    
    // Thread-local RNG (one per thread in block)
    curandState state;
    int tid = threadIdx.x;
    curand_init(seed + particleIdx * 1024 + tid, 0, 0, &state);
    
    // Monte Carlo sampling (distributed across threads in block)
    int insideCount = 0;
    int samplesPerThread = (samplesPerCell + blockDim.x - 1) / blockDim.x;
    
    for (int s = 0; s < samplesPerThread; ++s) {
        // Generate random point in bounding box
        float x = minX + curand_uniform(&state) * (maxX - minX);
        float y = minY + curand_uniform(&state) * (maxY - minY);
        
        // Check if point is in Voronoi cell
        float dx = x - genX;
        float dy = y - genY;
        float minDist2 = dx*dx + dy*dy;
        
        bool isInside = true;
        for (int n = 0; n < neighborCount; ++n) {
            int neighborIdx = neighbors[neighborOffset + n];
            float nX = positions[2 * neighborIdx];
            float nY = positions[2 * neighborIdx + 1];
            
            float ndx = x - nX;
            float ndy = y - nY;
            float dist2 = ndx*ndx + ndy*ndy;
            
            if (dist2 < minDist2) {
                isInside = false;
                break;
            }
        }
        
        if (isInside) {
            insideCount++;
        }
    }
    
    // Reduce across threads in block
    __shared__ int blockCounts[256]; // Max 256 threads per block
    blockCounts[tid] = insideCount;
    __syncthreads();
    
    // Simple reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            blockCounts[tid] += blockCounts[tid + stride];
        }
        __syncthreads();
    }
    
    // Thread 0 writes result
    if (tid == 0) {
        int totalInside = blockCounts[0];
        int totalSamples = samplesPerThread * blockDim.x;
        areas[particleIdx] = bboxArea * (float(totalInside) / float(totalSamples));
    }
}

// Helper function to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

/**
 * Triangulate a set of 2D particles using Delaunay triangulation
 * 
 * @param positions: Vector of particle positions (vec2 or similar with .x, .y)
 * @param particleIds: Vector of particle IDs
 * @return AdjacencyList representing vertex connectivity
 */
template <typename T, typename Vec2>
AdjacencyList<T> triangulate(
    const std::vector<Vec2>& positions,
    const std::vector<T>& particleIds
) {
    size_t numParticles = particleIds.size();
    
    // Build Delaunay triangulation
    Delaunay_2 dt;
    std::vector<Point_2> points;
    points.reserve(numParticles);
    
    for (size_t i = 0; i < numParticles; ++i) {
        points.emplace_back(positions[i].x, positions[i].y);
    }
    
    // Insert points and track vertex handles
    std::unordered_map<Delaunay_2::Vertex_handle, T> vertexToId;
    for (size_t i = 0; i < numParticles; ++i) {
        auto vh = dt.insert(points[i]);
        vertexToId[vh] = particleIds[i];
    }
    
    // Build adjacency list
    AdjacencyList<T> adjList(particleIds);
    
    for (auto eit = dt.finite_edges_begin(); eit != dt.finite_edges_end(); ++eit) {
        auto face = eit->first;
        int idx = eit->second;
        
        auto v1 = face->vertex((idx + 1) % 3);
        auto v2 = face->vertex((idx + 2) % 3);
        
        T id1 = vertexToId[v1];
        T id2 = vertexToId[v2];
        
        adjList.addNodeEdges(id1, {id2});
    }
    
    return adjList;
}

/**
 * Triangulate with GPU-accelerated Monte Carlo area integration
 * 
 * @param positions: Vector of particle positions (vec2 or similar with .x, .y)
 * @param particleIds: Vector of particle IDs
 * @param samplesPerCell: Number of Monte Carlo samples per cell
 * @param seed: Random seed for reproducibility
 * @return Pair of (AdjacencyList, area map)
 */
template <typename T, typename Vec2>
std::pair<AdjacencyList<T>, std::unordered_map<T, float>> 
triangulateWithAreaIntegration(
    const std::vector<Vec2>& positions,
    const std::vector<T>& particleIds,
    int samplesPerCell = 10000,
    unsigned int seed = 42
) {
    size_t numParticles = particleIds.size();

    // Perform Delaunay triangulation first
    auto triangulationResult = triangulate(positions, particleIds);
    AdjacencyList<T> adjList = triangulationResult.first;
    Delaunay_2 dt = triangulationResult.second;
    std::unordered_map<typename Delaunay_2::Vertex_handle, T> vertexToId = triangulationResult.third;

    // Create a reverse map from ID to vertex handle for convenience
    std::unordered_map<T, typename Delaunay_2::Vertex_handle> idToVertex;
    std::unordered_map<typename Delaunay_2::Vertex_handle, size_t> vertexToIndex;
    std::unordered_map<T, size_t> idToIndex;
    for(size_t i = 0; i < particleIds.size(); ++i) {
        idToIndex[particleIds[i]] = i;
    }

    for(const auto& pair : vertexToId) {
        idToVertex[pair.second] = pair.first;
        vertexToIndex[pair.first] = idToIndex[pair.second];
    }

    // Build neighbor lists for GPU from adjacency list
    std::vector<std::vector<int>> neighborLists(numParticles);
    for (size_t i = 0; i < numParticles; ++i) {
        T pId = particleIds[i];
        const auto& neighbors = adjList.getNeighbors(pId);
        for (const T& neighborId : neighbors) {
            neighborLists[i].push_back(idToIndex[neighborId]);
        }
    }
    
    // Prepare data for GPU
    std::vector<float> flatPositions(numParticles * 2);
    std::vector<float> bboxes(numParticles * 4);
    std::vector<int> neighborOffsets(numParticles);
    std::vector<int> neighborCounts(numParticles);
    std::vector<int> flatNeighbors;
    
    int currentOffset = 0;
    for (size_t i = 0; i < numParticles; ++i) {
        T id = particleIds[i];
        auto vh = idToVertex[id];
        
        flatPositions[2*i] = positions[i].x;
        flatPositions[2*i + 1] = positions[i].y;
        
        // Compute AABB from Voronoi vertices (circumcenters)
        std::vector<Point_2> voronoiVertices;
        auto fc = dt.incident_faces(vh);
        
        if (fc != nullptr) {
            auto done = fc;
            do {
                if (!dt.is_infinite(fc)) {
                    voronoiVertices.push_back(dt.circumcenter(fc));
                }
                ++fc;
            } while (fc != done);
        }
        
        if (!voronoiVertices.empty()) {
            float minX = voronoiVertices[0].x();
            float maxX = voronoiVertices[0].x();
            float minY = voronoiVertices[0].y();
            float maxY = voronoiVertices[0].y();
            
            for (const auto& v : voronoiVertices) {
                minX = std::min(minX, static_cast<float>(v.x()));
                maxX = std::max(maxX, static_cast<float>(v.x()));
                minY = std::min(minY, static_cast<float>(v.y()));
                maxY = std::max(maxY, static_cast<float>(v.y()));
            }
            
            bboxes[4*i] = minX;
            bboxes[4*i + 1] = minY;
            bboxes[4*i + 2] = maxX;
            bboxes[4*i + 3] = maxY;
        } else {
            // Fallback: small box around point
            bboxes[4*i] = positions[i].x - 0.1f;
            bboxes[4*i + 1] = positions[i].y - 0.1f;
            bboxes[4*i + 2] = positions[i].x + 0.1f;
            bboxes[4*i + 3] = positions[i].y + 0.1f;
        }
        
        // Flatten neighbor list
        neighborOffsets[i] = currentOffset;
        neighborCounts[i] = neighborLists[i].size();
        for (int neighbor : neighborLists[i]) {
            flatNeighbors.push_back(neighbor);
        }
        currentOffset += neighborLists[i].size();
    }
    
    // Allocate GPU memory
    float *d_positions, *d_bboxes, *d_areas;
    int *d_neighborOffsets, *d_neighborCounts, *d_neighbors;
    
    CUDA_CHECK(cudaMalloc(&d_positions, flatPositions.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bboxes, bboxes.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_areas, numParticles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_neighborOffsets, neighborOffsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighborCounts, neighborCounts.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_neighbors, flatNeighbors.size() * sizeof(int)));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_positions, flatPositions.data(), 
                          flatPositions.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bboxes, bboxes.data(), 
                          bboxes.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighborOffsets, neighborOffsets.data(), 
                          neighborOffsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighborCounts, neighborCounts.data(), 
                          neighborCounts.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors, flatNeighbors.data(), 
                          flatNeighbors.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel (one block per particle, 256 threads per block)
    int threadsPerBlock = 256;
    monteCarloAreaKernel<<<numParticles, threadsPerBlock>>>(
        d_positions,
        d_bboxes,
        d_neighborOffsets,
        d_neighborCounts,
        d_neighbors,
        d_areas,
        numParticles,
        samplesPerCell,
        seed
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back
    std::vector<float> areas(numParticles);
    CUDA_CHECK(cudaMemcpy(areas.data(), d_areas, 
                          numParticles * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_positions));
    CUDA_CHECK(cudaFree(d_bboxes));
    CUDA_CHECK(cudaFree(d_areas));
    CUDA_CHECK(cudaFree(d_neighborOffsets));
    CUDA_CHECK(cudaFree(d_neighborCounts));
    CUDA_CHECK(cudaFree(d_neighbors));
    
    // Build area map
    std::unordered_map<T, float> areaMap;
    for (size_t i = 0; i < numParticles; ++i) {
        areaMap[particleIds[i]] = areas[i];
    }
    
    return {adjList, areaMap};
}

// Find leaf nodes in the adjacency list
template <typename T>
std::unordered_map<T, int> findConnectedComponents(
    const AdjacencyList<T>& adjList,
    const std::unordered_map<T, float>& opacityMap
) {
    std::unordered_map<T, int> componentLabels;
    std::unordered_set<T> visited;
    int componentId = 0;

    for (const auto& pair : opacityMap) {
        const T& startNode = pair.first;
        float opacity = pair.second;

        if (opacity > 0.0f && visited.find(startNode) == visited.end()) {
            // Start of a new component
            componentId++;
            std::vector<T> stack;

            stack.push_back(startNode);
            visited.insert(startNode);
            componentLabels[startNode] = componentId;

            while (!stack.empty()) {
                T currentNode = stack.back();
                stack.pop_back();

                for (const auto& neighbor : adjList.getNeighbors(currentNode)) {
                    auto it = opacityMap.find(neighbor);
                    if (it != opacityMap.end() && it->second > 0.0f && visited.find(neighbor) == visited.end()) {
                        visited.insert(neighbor);
                        componentLabels[neighbor] = componentId;
                        stack.push_back(neighbor);
                    }
                }
            }
        }
    }

    return componentLabels;
}

// Find surface cells
template <typename T>
std::vector<T> findSurfaceCells(
    const AdjacencyList<T>& adjList,
    const std::unordered_map<T, float>& opacityMap
) {
    std::vector<T> surfaceCells;
    for (const auto& pair : opacityMap) {
        const T& nodeId = pair.first;
        float opacity = pair.second;

        if (opacity > 0.0f) {
            bool isSurfaceCell = false;
            for (const auto& neighborId : adjList.getNeighbors(nodeId)) {
                auto it = opacityMap.find(neighborId);
                if (it == opacityMap.end() || it->second == 0.0f) {
                    isSurfaceCell = true;
                    break;
                }
            }
            if (isSurfaceCell) {
                surfaceCells.push_back(nodeId);
            }
        }
    }
    return surfaceCells;
}

// Calculate center of mass
template <typename T>
glm::vec3 calculateCenterOfMass(
    const std::unordered_map<T, glm::vec3>& positions,
    const std::unordered_map<T, float>& masses
) {
    glm::vec3 com(0.0f);
    float totalMass = 0.0f;

    for (const auto& pair : positions) {
        const T& id = pair.first;
        const glm::vec3& pos = pair.second;
        
        auto massIt = masses.find(id);
        if (massIt != masses.end()) {
            float mass = massIt->second;
            com += pos * mass;
            totalMass += mass;
        }
    }

    if (totalMass > 0.0f) {
        com /= totalMass;
    }

    return com;
}

// Calculate inertia tensor
template <typename T>
glm::mat3 calculateInertiaTensor(
    const std::unordered_map<T, glm::vec3>& localPositions,
    const std::unordered_map<T, float>& masses
) {
    glm::mat3 inertiaTensor(0.0f);

    for (const auto& pair : localPositions) {
        const T& id = pair.first;
        const glm::vec3& r = pair.second; // Local position
        
        auto massIt = masses.find(id);
        if (massIt != masses.end()) {
            float mass = massIt->second;
            
            // Diagonal elements
            inertiaTensor[0][0] += mass * (r.y * r.y + r.z * r.z);
            inertiaTensor[1][1] += mass * (r.x * r.x + r.z * r.z);
            inertiaTensor[2][2] += mass * (r.x * r.x + r.y * r.y);
            
            // Off-diagonal elements
            inertiaTensor[0][1] -= mass * r.x * r.y;
            inertiaTensor[0][2] -= mass * r.x * r.z;
            inertiaTensor[1][2] -= mass * r.y * r.z;
        }
    }

    // Symmetrize the matrix
    inertiaTensor[1][0] = inertiaTensor[0][1];
    inertiaTensor[2][0] = inertiaTensor[0][2];
    inertiaTensor[2][1] = inertiaTensor[1][2];

    return inertiaTensor;
}

// Calculate AABB
std::pair<glm::vec3, glm::vec3> calculateAABB(
    const std::vector<glm::vec3>& positions
) {
    if (positions.empty()) {
        return {glm::vec3(0.0f), glm::vec3(0.0f)};
    }

    glm::vec3 min = positions[0];
    glm::vec3 max = positions[0];

    for (size_t i = 1; i < positions.size(); ++i) {
        const glm::vec3& pos = positions[i];
        min.x = std::min(min.x, pos.x);
        min.y = std::min(min.y, pos.y);
        min.z = std::min(min.z, pos.z);
        max.x = std::max(max.x, pos.x);
        max.y = std::max(max.y, pos.y);
        max.z = std::max(max.z, pos.z);
    }

    return {min, max};
}

} // namespace DynamicFoam::Sim2D