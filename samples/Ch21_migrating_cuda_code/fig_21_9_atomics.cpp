// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

constexpr int count = 1024 * 1024;

int main() {
  queue q{property::queue::in_order()};
  std::cout << "Running on device: "
            << q.get_device().get_info<info::device::name>()
            << "\n";

  int* buffer = malloc_device<int>(1, q);
  q.fill(buffer, 0, 1);

  // BEGIN CODE SNIP
  q.parallel_for(count, [=](auto id) {
    // The SYCL atomic_ref must specify the default order
    // and default scope as part of the atomic_ref type. To
    // match the behavior of the CUDA atomicAdd we want a
    // relaxed atomic with device scope:
    atomic_ref<int, memory_order::relaxed,
               memory_scope::device>
        aref(*buffer);

    // When no memory order is specified, the defaults are
    // used:
    aref.fetch_add(1);

    // We can also specify the memory order and scope as
    // part of the atomic operation:
    aref.fetch_add(1, memory_order::relaxed,
                   memory_scope::device);
  });
  // END CODE SNIP

  int test = -1;
  q.copy(buffer, &test, 1).wait();

  if (test != 2 * count) {
    std::cout << "Found " << test << ", wanted "
              << 2 * count << ".\n";
  } else {
    std::cout << "Success.\n";
  }

  free(buffer, q);
  return 0;
}
