// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>

#include <iostream>

constexpr int count = 1024 * 1024;

// BEGIN CODE SNIP
// The CUDA kernel is a separate function
__global__ void TestKernel(int* dst) {
  auto id = blockIdx.x * blockDim.x + threadIdx.x;
  dst[id] = id;
}

int main() {
  // CUDA uses device zero by default
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  std::cout << "Running on device: " << deviceProp.name << "\n";

  int* buffer = nullptr;
  cudaMallocHost(&buffer, count * sizeof(int));
  cudaMemset(buffer, 0, count * sizeof(int));

  TestKernel<<<count / 256, 256>>>(buffer);
  cudaDeviceSynchronize();
  // ...
// END CODE SNIP

  int mismatches = 0;
  for (int i = 0; i < count; i++) {
    if (buffer[i] != i) {
      mismatches++;
    }
  }
  if (mismatches) {
    std::cout << "Found " << mismatches << " mismatches out of "
              << count << " elements.\n";
  } else {
    std::cout << "Success.\n";
  }

  cudaFreeHost(buffer);
  return 0;
}
