// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue q;

  // Allocate N floats

  // C-style
  float *f1 = static_cast<float *>(
      malloc_shared(N * sizeof(float), q));

  // C++-style
  float *f2 = malloc_shared<float>(N, q);

  // C++-allocator-style
  usm_allocator<float, usm::alloc::shared> alloc(q);
  float *f3 = alloc.allocate(N);

  // Free our allocations
  free(f1, q.get_context());
  free(f2, q);
  alloc.deallocate(f3, N);

  return 0;
}
