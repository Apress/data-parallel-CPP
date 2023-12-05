// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>

#include <iostream>
#include <numeric>
#include <vector>

constexpr size_t size = 1024 * 1024;

// BEGIN CODE SNIP
__shared__ int scratch[256];
__global__ void Reverse(int* ptr, size_t size) {
  auto gid = blockIdx.x * blockDim.x + threadIdx.x;
  auto lid = threadIdx.x;

  scratch[lid] = ptr[gid];
  __syncthreads();
  ptr[gid] = scratch[256 - lid - 1];
}

int main() {
  std::vector<int> data(size);
  std::iota(data.begin(), data.end(), 0);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  std::cout << "Running on device: " << deviceProp.name << "\n";

  int* ptr = nullptr;
  cudaMalloc(&ptr, size * sizeof(int));
  cudaMemcpy(ptr, data.data(), size * sizeof(int),
             cudaMemcpyDefault);
  Reverse<<<size / 256, 256>>>(ptr, size);
  cudaError_t result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cout << "An error occurred!\n";
  }
  // ...
// END CODE SNIP

  cudaMemcpy(data.data(), ptr, size * sizeof(int),
             cudaMemcpyDefault);

  for (size_t s = 0; s < size; s += 256) {
    for (size_t i = 0; i < 256; i++) {
      auto got = data[s + i];
      auto want = s + 256 - i - 1;
      if (got != want) {
        std::cout << "Mismatch at index " << s + i << ", got "
                  << got << ", wanted " << want << "\n";
        return -1;
      }
    }
  }

  cudaFree(ptr);
  std::cout << "Success.\n";
  return 0;
}
