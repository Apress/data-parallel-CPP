// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

__global__ void Reverse(int* ptr, size_t size) {
  auto gid = blockIdx.x * blockDim.x + threadIdx.x;
  auto lid = threadIdx.x;

  scratch[lid] = ptr[gid];
  __syncthreads();
  ptr[gid] = scratch[256 - lid - 1];
}

int main() {
  std::array<int, size> data;
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
