// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// BEGIN CODE SNIP
#include <array>
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  constexpr int size = 16;
  std::array<int, size> data;

  // Create queue on implementation-chosen default device
  queue q;

  // Create buffer using host allocated "data" array
  buffer B{data};

  q.submit([&](handler& h) {
    accessor A{B, h};
    h.parallel_for(size, [=](auto& idx) { A[idx] = idx; });
  });

  // Obtain access to buffer on the host
  // Will wait for device kernel to execute to generate data
  host_accessor A{B};
  for (int i = 0; i < size; i++)
    std::cout << "data[" << i << "] = " << A[i] << "\n";

  return 0;
}
// END CODE SNIP
