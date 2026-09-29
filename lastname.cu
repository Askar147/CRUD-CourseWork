#include <cuda_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iostream>

#define call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)          \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";      \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

constexpr int N = 1 << 6;  // 64

// Fill one matrix. Launch A and B on different streams.
__global__ void fillMatrixKernel(int* matrix, int totalElements, int seed) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < totalElements;
         i += blockDim.x * gridDim.x) {
        // Small deterministic integer values.
        matrix[i] = ((i + seed) % 10) + 1;
    }
}

// Parallel matrix multiplication with a grid-stride loop over C elements.
__global__ void matrixMulKernel(const int* A, const int* B, int* C, int n) {
    const int totalElements = n * n;

    for (int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < totalElements;
         index += blockDim.x * gridDim.x) {

        const int row = index / n;
        const int col = index % n;

        int sum = 0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[index] = sum;
    }
}

// Sequential CPU multiplication.
void matrixMulHost(const int* A, const int* B, int* C, int n) {
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < n; ++col) {
            int sum = 0;
            for (int k = 0; k < n; ++k) {
                sum += A[row * n + k] * B[k * n + col];
            }
            C[row * n + col] = sum;
        }
    }
}

// Verify CPU and GPU results.
bool verifyResult(const int* hostResult, const int* gpuResult, int totalElements) {
    for (int i = 0; i < totalElements; ++i) {
        if (hostResult[i] != gpuResult[i]) {
            std::cerr << "Mismatch at element " << i
                      << ": host=" << hostResult[i]
                      << ", gpu=" << gpuResult[i] << '\n';
            return false;
        }
    }
    return true;
}

int main() {
    const int totalElements = N * N;
    const size_t bytes = static_cast<size_t>(totalElements) * sizeof(int);

    // Read GPU properties so the launch follows the exam requirement.
    int device = 0;
    cudaGetDevice(&device));

    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device));

    // Requirement: number of blocks = warp size * number of SMs.
    const int numBlocks = prop.warpSize * prop.multiProcessorCount;

    // Use a warp-aligned block size, normally 256 threads.
    int threadsPerBlock = prop.warpSize * 8;
    if (threadsPerBlock > prop.maxThreadsPerBlock) {
        threadsPerBlock =
            (prop.maxThreadsPerBlock / prop.warpSize) * prop.warpSize;
    }

    std::cout << "GPU: " << prop.name << '\n';
    std::cout << "N: " << N << " x " << N << '\n';
    std::cout << "SMs: " << prop.multiProcessorCount << '\n';
    std::cout << "Warp size: " << prop.warpSize << '\n';
    std::cout << "Blocks: " << numBlocks << '\n';
    std::cout << "Threads/block: " << threadsPerBlock << "\n\n";

    // Pinned host memory allows asynchronous transfers.
    int* h_A = nullptr;
    int* h_B = nullptr;
    int* h_C_host = nullptr;
    int* h_C_gpu = nullptr;

    cudaMallocHost(reinterpret_cast<void**>(&h_A), bytes));
    cudaMallocHost(reinterpret_cast<void**>(&h_B), bytes));
    cudaMallocHost(reinterpret_cast<void**>(&h_C_host), bytes));
    cudaMallocHost(reinterpret_cast<void**>(&h_C_gpu), bytes));

    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;

    cudaMalloc(reinterpret_cast<void**>(&d_A), bytes));
    cudaMalloc(reinterpret_cast<void**>(&d_B), bytes));
    cudaMalloc(reinterpret_cast<void**>(&d_C), bytes));

    cudaStream_t streamA{};
    cudaStream_t streamB{};
    cudaStreamCreate(&streamA));
    cudaStreamCreate(&streamB));

    // Initialize A and B independently on two CUDA streams.
    fillMatrixKernel<<<numBlocks, threadsPerBlock, 0, streamA>>>(
        d_A, totalElements, 1);
    cudaGetLastError());

    fillMatrixKernel<<<numBlocks, threadsPerBlock, 0, streamB>>>(
        d_B, totalElements, 7);
    cudaGetLastError());

    // Copy inputs back once for the CPU reference calculation.
    // Because h_A/h_B are pinned, these transfers can be asynchronous.
    cudaMemcpyAsync(
        h_A, d_A, bytes, cudaMemcpyDeviceToHost, streamA));
    cudaMemcpyAsync(
        h_B, d_B, bytes, cudaMemcpyDeviceToHost, streamB));

    cudaStreamSynchronize(streamA));
    cudaStreamSynchronize(streamB));

    // -------- GPU multiplication timing: kernel only --------
    cudaEvent_t gpuStart{};
    cudaEvent_t gpuStop{};
    cudaEventCreate(&gpuStart));
    cudaEventCreate(&gpuStop));

    cudaEventRecord(gpuStart));

    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaGetLastError());

    cudaEventRecord(gpuStop));
    cudaEventSynchronize(gpuStop));

    float gpuMilliseconds = 0.0f;
    cudaEventElapsedTime(&gpuMilliseconds, gpuStart, gpuStop));

    // -------- CPU multiplication timing --------
    const auto cpuStart = std::chrono::high_resolution_clock::now();
    matrixMulHost(h_A, h_B, h_C_host, N);
    const auto cpuStop = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::milli> cpuMilliseconds =
        cpuStop - cpuStart;

    // One final D2H transfer for verification.
    cudaMemcpyAsync(
        h_C_gpu, d_C, bytes, cudaMemcpyDeviceToHost, streamA));
    cudaStreamSynchronize(streamA));

    const bool correct = verifyResult(h_C_host, h_C_gpu, totalElements);

    std::cout << "GPU multiplication kernel time: "
              << gpuMilliseconds << " ms\n";
    std::cout << "CPU multiplication time: "
              << cpuMilliseconds.count() << " ms\n";
    std::cout << "Verification: " << (correct ? "PASSED" : "FAILED") << '\n';

    // Cleanup.
    cudaEventDestroy(gpuStart));
    cudaEventDestroy(gpuStop));

    cudaStreamDestroy(streamA));
    cudaStreamDestroy(streamB));

    cudaFree(d_A));
    cudaFree(d_B));
    cudaFree(d_C));

    cudaFreeHost(h_A));
    cudaFreeHost(h_B));
    cudaFreeHost(h_C_host));
    cudaFreeHost(h_C_gpu));

    cudaDeviceReset());

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
