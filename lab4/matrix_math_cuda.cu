#include "matrix_math_cuda.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << error << " (" << cudaGetErrorString(error) << ")\n"; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void matmulKernel(const float* A, const float* B, float* C,
                              int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < M; ++i) {
            sum += A[row * M + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

void printDeviceInfo() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " CUDA device(s)\n";
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "\nDevice " << i << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "  Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    }
}

std::string getDeviceName() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    return std::string(prop.name);
}

void matmulCUDA(const float* A, const float* B, float* C,
                int N, int M, int K,
                int blockSizeX, int blockSizeY) {
    
    dim3 blockSize(blockSizeX, blockSizeY);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * K * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, M * K * sizeof(float), cudaMemcpyHostToDevice));
    
    matmulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(C, d_C, N * K * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void benchmarkCUDA(const float* A, const float* B, float* C,
                   int N, int M, int K,
                   int blockSizeX, int blockSizeY,
                   double& timeMs) {
    
    dim3 blockSize(blockSizeX, blockSizeY);
    dim3 gridSize((K + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);
    
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, N * M * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, N * K * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_A, A, N * M * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, M * K * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    matmulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M, K);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    matmulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N, M, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&timeMs, start, stop);
    
    CUDA_CHECK(cudaMemcpy(C, d_C, N * K * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void runCudaExperiments() {
    std::vector<int> sizes = {200, 400, 800, 1200, 1600, 2000};
    
    std::vector<std::pair<int, int>> blockConfigs = {
        {8, 8}, {16, 16}, {32, 32}, {16, 8}, {8, 16}, {32, 16}
    };
    
    std::cout << "\n=== CUDA Matrix Multiplication Experiments ===\n";
    printDeviceInfo();
    std::cout << "\nStarting benchmarks...\n\n";
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "Size\t";
    for (auto& cfg : blockConfigs) {
        std::cout << cfg.first << "x" << cfg.second << "\t";
    }
    std::cout << "\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (int size : sizes) {
        std::cout << size << "\t";
        
        std::vector<float> A(size * size);
        std::vector<float> B(size * size);
        std::vector<float> C(size * size);
        
        for (int i = 0; i < size * size; ++i) {
            A[i] = static_cast<float>(rand()) / RAND_MAX;
            B[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        
        for (auto& config : blockConfigs) {
            double timeMs;
            try {
                benchmarkCUDA(A.data(), B.data(), C.data(),
                             size, size, size,
                             config.first, config.second, timeMs);
                std::cout << timeMs * 1000 << "\t";
            } catch (const std::exception& e) {
                std::cout << "ERROR\t";
            }
            std::cout.flush();
        }
        std::cout << "\n";
    }
    
    std::cout << "\n=== Experiments completed ===\n";
}