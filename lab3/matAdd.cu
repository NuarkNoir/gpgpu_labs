#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int N = 16384;
constexpr int SIZE = N*N;
constexpr int BLOCK_SIZE = 16;

__global__ void matMul(const float *a, const float *b, float *c, int N) {
	int x = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	int y = threadIdx.y + blockIdx.y * BLOCK_SIZE;

  c[x * N + y] = 0;
  for (int i = 0; i < N; i++) {
    c[x * N + y] += a[x * N + i] * b[i * N + y];
  }
}

__global__ void matMulShared(float* a, float* b, float* c, size_t N) {
	float sum = 0.0;

	int aBegin = N * BLOCK_SIZE * blockIdx.y;
	int aEnd = aBegin + N - 1;
  
	int aStep = BLOCK_SIZE;

	int bBegin = BLOCK_SIZE * blockIdx.x;
	int bStep = BLOCK_SIZE * N;

	for (int ia = aBegin, ib = bBegin; ia <= aEnd; ia += aStep, ib += bStep) {
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		As[threadIdx.y][threadIdx.x] = a[ia + N * threadIdx.y + threadIdx.x];
		Bs[threadIdx.y][threadIdx.x] = b[ib + N * threadIdx.y + threadIdx.x];

		__syncthreads();

		for (int k = 0; k < BLOCK_SIZE; k++)
		{
			sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
		}

		__syncthreads();
	}

	int ic = N * BLOCK_SIZE * blockIdx.y + BLOCK_SIZE * blockIdx.x;
	c[ic + N * threadIdx.y + threadIdx.x] = sum;
}

void printDeviceInfo();
std::string formatBytes(size_t bytes);
void printMatrix(float *matrix, int N);
bool matrixEquals(float *a, float *b, int N);

int main() {
  printDeviceInfo();

  float *h_A = new float[SIZE];
  float *h_B = new float[SIZE];
  float *hd_C = new float[SIZE];
  float *hds_C = new float[SIZE];

  for (int i = 0; i < SIZE; i++) {
    h_A[i] = (rand() % 1000) / 1000.0f;
    h_B[i] = (rand() % 1000) / 1000.0f;
  }

  float *d_A, *d_B, *ds_C, *d_C;
  cudaMalloc(&d_A, SIZE * sizeof(float));
  cudaMalloc(&d_B, SIZE * sizeof(float));
  cudaMalloc(&d_C, SIZE * sizeof(float));
  cudaMalloc(&ds_C, SIZE * sizeof(float));

  cudaMemcpy(d_A, h_A, SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, SIZE * sizeof(float), cudaMemcpyHostToDevice);

	constexpr int blockCount = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 blockConf(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridConf(blockCount, blockCount);

  float cudaMills = 0;
  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matMul<<<gridConf, blockConf>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaMemcpy(hd_C, d_C, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cudaMills, start, stop);
  }

  float cudaMillsShared = 0;
  {
    cudaFuncSetCacheConfig(matMulShared, cudaFuncCachePreferShared);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    matMulShared<<<gridConf, blockConf>>>(d_A, d_B, ds_C, N);
    cudaEventRecord(stop);

    cudaMemcpy(hds_C, ds_C, SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cudaMillsShared, start, stop);
  }

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  std::cout << "Matrix equals: " << matrixEquals(hd_C, hds_C, N) << std::endl;
  std::cout << "Kernel execution time: " << cudaMills << " ms" << std::endl;
  std::cout << "Kernel shared mem execution time: " << cudaMillsShared << " ms" << std::endl;

  return 0;
}

void printDeviceInfo() {
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, 0);
  std::cout << "Device Name: " << devProp.name << std::endl;
  std::cout << "Total Global Memory: " << formatBytes(devProp.totalGlobalMem) << std::endl;
  std::cout << "Shared Memory per Block: " << formatBytes(devProp.sharedMemPerBlock) << std::endl;
  std::cout << "Registers per Block: " << devProp.regsPerBlock << std::endl;
  std::cout << "Total Constant Memory: " << formatBytes(devProp.totalConstMem) << std::endl;
  std::cout << "L2 Cache Size (bytes): " << formatBytes(devProp.l2CacheSize) << std::endl;
}

std::string formatBytes(size_t bytes) {
  std::string suffix = "b";
  double value = bytes;
  if (bytes >= 1024) {
    suffix = "KB";
    value = bytes / 1024.0;
  }
  if (bytes >= 1024 * 1024) {
    suffix = "MB";
    value = bytes / (1024.0 * 1024.0);
  }
  if (bytes >= 1024 * 1024 * 1024) {
    suffix = "GB";
    value = bytes / (1024.0 * 1024.0 * 1024.0);
  }
  return std::to_string(value) + suffix;
}

void printMatrix(float *matrix, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      std::cout << matrix[i * N + j] << " ";
    }
    std::cout << std::endl;
  }
}

bool matrixEquals(float *a, float *b, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (fabs(a[i * N + j] - b[i * N + j]) >= 0.01f) {
        std::cout << "a[" << i << "][" << j << "] = " << a[i * N + j] << " != b[" << i << "][" << j << "] = " << b[i * N + j] << std::endl;
        return false;
      }
    }
  }
  return true;
}