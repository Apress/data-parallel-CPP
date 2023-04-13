// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

constexpr int count = 1024 * 1024;

int main() {
  queue Q;
  std::cout << "Running on device: "
            << Q.get_device().get_info<info::device::name>()
            << "\n";

  int* buffer = malloc_host<int>(1, Q);
  buffer[0] = 0;

  // BEGIN CODE SNIP
  Q.parallel_for(count, [=](auto id) {
     // The SYCL atomic_ref must specify the default order and
     // default scope as part of the atomic_ref type.  To match
     // the behavior of the CUDA atomicAdd we want a relaxed
     // atomic with device scope:
     atomic_ref<int, memory_order::relaxed, memory_scope::device>
         aref(*buffer);

     // When no memory order is specified, the defaults are used:
     aref.fetch_add(1);

     // We can also specify the memory order and scope as part
     // of the atomic operation:
     aref.fetch_add(1, memory_order::relaxed,
                    memory_scope::device);
   }).wait();
  // END CODE SNIP

  if (buffer[0] != 2 * count) {
    std::cout << "Found " << buffer[0] << ", wanted "
              << 2 * count << ".\n";
  } else {
    std::cout << "Success.\n";
  }

  free(buffer, Q);
  return 0;
}
