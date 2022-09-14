#include <iostream>
#include <vector>
#include "../commonUtils.cpp"

__global__ void vectorAdd(const float *a, const float *b, float *c, int N) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid < N) c[tid] = a[tid] + b[tid];
}

int main(int argc, char** argv) {
    size_t size = 0;
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <size>" << std::endl;
        return 1;
    }

    size = std::atoll(argv[1]);
    
    const int32_t THREADS_PER_BLOCK = 1024;
    const int32_t BLOCKS = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    const size_t sizeInBytes = sizeof(float) * size;
    auto hA = generateVec(size, 0, 10);
    auto hB = generateVec(size, 0, 10);
    auto hdSum = emptyVec(size);

    float *dA, *dB, *dSum;
    cudaMalloc(&dA, sizeInBytes);
    cudaMalloc(&dB, sizeInBytes);
    cudaMalloc(&dSum, sizeInBytes);

    cudaMemcpy(dA, hA, sizeInBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, sizeInBytes, cudaMemcpyHostToDevice);

    vectorAdd<<<BLOCKS, THREADS_PER_BLOCK>>>(dA, dB, dSum, size);

    cudaMemcpy(hdSum, dSum, sizeInBytes, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dSum);

    return 0;
}