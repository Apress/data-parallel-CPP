// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

constexpr int count = 10;

int main() {
  queue Q{property::queue::in_order()};
  std::cout << "Running on device: "
            << Q.get_device().get_info<info::device::name>()
            << "\n";

  int* buffer = malloc_host<int>(count, Q);
  Q.fill(buffer, 0, count);

  // BEGIN CODE SNIP
  std::cout << "WARNING: May deadlock on some devices!\n";
  Q.parallel_for(nd_range<1>{64, 64}, [=](auto item) {
     int id = item.get_global_id(0);
     if (id >= count) {
        return; // early exit
     }
     group_barrier(item.get_group());
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
    std::cout << "Found " << mismatches << " mismatches out of "
              << count << " elements.\n";
  } else {
    std::cout << "Success.\n";
  }

  free(buffer, Q);
  return 0;
}
