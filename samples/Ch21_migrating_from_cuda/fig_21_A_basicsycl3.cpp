// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

constexpr int count = 1024 * 1024;

int main() {
  // Declare a SYCL queue for the default device
  queue Q{property::queue::in_order()};
  std::cout << "Running on device: "
            << Q.get_device().get_info<info::device::name>()
            << "\n";

  int* buffer = malloc_host<int>(count, Q);
  Q.fill(buffer, 0, count);

  // This is OK:
  Q.parallel_for(nd_range<1>{{count}, {256}}, [=](nd_item<1> it) {
    int i = it.get_global_linear_id();
     buffer[i] = i;
  }).wait();
  // This is OK too::
  Q.parallel_for(nd_range<1>{{count}, {256}}, [=](item<1> i) {
     buffer[i] = i;
  }).wait();
  // This is OK too:
  Q.parallel_for(nd_range<1>{{count}, {256}}, [=](id<1> i) {
     buffer[i] = i;
  }).wait();
  // This is OK too?
  Q.parallel_for(nd_range<1>{{count}, {256}}, [=](int i) {
     buffer[i] = i;
  }).wait();

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
