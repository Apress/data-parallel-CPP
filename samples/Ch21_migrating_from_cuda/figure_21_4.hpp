// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

__global__ void ExchangeKernel(int* dst) {
  auto index = get_global_linear_id(); // helper function
  auto fastest = threadIdx.x;
  auto neighbor = __shfl_xor_sync(0xFFFFFFFF, fastest, 1);
  dst[index] = neighbor;
}
...
  dim3 threadsPerBlock(16, 2);
  ExchangeKernel<<<1, threadsPerBlock>>>(buffer);
  cudaDeviceSynchronize();
