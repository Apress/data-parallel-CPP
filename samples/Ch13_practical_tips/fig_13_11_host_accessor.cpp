// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <algorithm>
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // BEGIN CODE SNIP

  constexpr size_t N = 1024;

  // Set up queue on any available device
  queue q;

  // Create host containers to initialize on the host
  std::vector<int> in_vec(N), out_vec(N);

  // Initialize input and output vectors
  for (int i = 0; i < N; i++) in_vec[i] = i;
  std::fill(out_vec.begin(), out_vec.end(), 0);

  // Create buffers using host allocations (vector in this
  // case)
  buffer in_buf{in_vec}, out_buf{out_vec};

  // Submit the kernel to the queue
  q.submit([&](handler& h) {
    accessor in{in_buf, h};
    accessor out{out_buf, h};

    h.parallel_for(range{N},
                   [=](id<1> idx) { out[idx] = in[idx]; });
  });

  // Check that all outputs match expected value
  // Use host accessor!  Buffer is still in scope / alive
  host_accessor A{out_buf};

  for (int i = 0; i < N; i++)
    std::cout << "A[" << i << "]=" << A[i] << "\n";

  // END CODE SNIP
  return 0;
}
