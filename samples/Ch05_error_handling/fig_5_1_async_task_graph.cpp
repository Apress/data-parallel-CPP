// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  constexpr int size = 16;
  buffer<int> b{range{size}};

  // Create queue on any available device
  queue q;

  q.submit([&](handler& h) {
    accessor a{b, h};

    h.parallel_for(size, [=](auto& idx) { a[idx] = idx; });
  });

  // Obtain access to buffer on the host
  // Will wait for device kernel to execute to generate data
  host_accessor a{b};
  for (int i = 0; i < size; i++)
    std::cout << "data[" << i << "] = " << A[i] << "\n";

  return 0;
}
