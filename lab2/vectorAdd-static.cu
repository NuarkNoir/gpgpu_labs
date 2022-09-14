#include <iostream>
#include <fstream>
#include "../commonUtils.cpp"

__global__ void vectorAdd(const float *__restrict a, const float *__restrict b, float *__restrict c, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) c[tid] = a[tid] + b[tid];
}


int main() {
    size_t size = 0;
    std::ifstream data("data.txt", std::ios::in);
    data >> size;
    
    const size_t sizeInBytes = sizeof(float) * size;
    const int32_t THREADS_PER_BLOCK = 1024;
    const int32_t BLOCKS = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    auto hA = emptyVec(size);
    auto hB = emptyVec(size);
    auto hSumE = emptyVec(size);

    for (size_t i = 0; i < size; i++) {
        data >> hA[i] >> hB[i] >> hSumE[i];
    }

    auto hdSum = emptyVec(size);

    float *dA, *dB, *dSum;
    cudaMalloc(&dA, sizeInBytes);
    cudaMalloc(&dB, sizeInBytes);
    cudaMalloc(&dSum, sizeInBytes);

    cudaMemcpy(dA, hA, sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeInBytes, cudaMemcpyHostToDevice);

    vectorAdd<<<BLOCKS, THREADS_PER_BLOCK>>>(dA, dB, dSum, size);

    cudaMemcpy(hdSum, dSum, sizeInBytes, cudaMemcpyDeviceToHost);

    bool veq = compareVecs(hSumE, hdSum, size);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dSum);

    if (veq) {
        std::cout << "Success!" << std::endl;
        return 0;
    }
    
    std::cout << "Failure!" << std::endl;
    return 1;
}