// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

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
