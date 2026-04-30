#ifndef MATRIX_MATH_CUDA_H
#define MATRIX_MATH_CUDA_H

#include <iostream>
#include <string>

void printDeviceInfo();
std::string getDeviceName();
void matmulCUDA(const float* A, const float* B, float* C, 
                int N, int M, int K,
                int blockSizeX, int blockSizeY);
void benchmarkCUDA(const float* A, const float* B, float* C,
                   int N, int M, int K,
                   int blockSizeX, int blockSizeY,
                   double& timeMs);
void runCudaExperiments();

#endif