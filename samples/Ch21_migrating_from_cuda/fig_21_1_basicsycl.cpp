// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

constexpr int count = 1024 * 1024;

int main() {
  // BEGIN CODE SNIP
  // Declare an in-order SYCL queue for the default device
  queue Q{property::queue::in_order()};
  std::cout << "Running on device: "
            << Q.get_device().get_info<info::device::name>()
            << "\n";

  int* buffer = malloc_host<int>(count, Q);
  Q.fill(buffer, 0, count);

  Q.parallel_for(count, [=](auto id) {
     buffer[id] = id;
   }).wait();
  // END CODE SNIP

  int mismatches = 0;
  for (int i = 0; i < count; i++) {
    if (buffer[i] != i) {
      mismatches++;
    }
  }
  if (mismatches) {
    std::cout << "Found " << mismatches
              << " mismatches out of " << count
              << " elements.\n";
  } else {
    std::cout << "Success.\n";
  }

  free(buffer, Q);
  return 0;
}
