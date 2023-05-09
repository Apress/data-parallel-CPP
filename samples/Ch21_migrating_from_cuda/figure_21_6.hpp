// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.


__global__ void ExchangeKernelCoopGroups(int* dst) {
  namespace cg = cooperative_groups;
  auto index = cg::this_grid().thread_rank();
  auto fastest = threadIdx.x;
  auto warp = cg::tiled_partition<32>(cg::this_thread_block());
  auto neighbor = warp.shfl_xor(fastest, 1);
  dst[index] = neighbor;
}
