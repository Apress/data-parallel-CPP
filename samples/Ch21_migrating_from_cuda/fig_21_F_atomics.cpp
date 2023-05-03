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

  device d = Q.get_device();

  auto orders = d.get_info<
      info::device::atomic_memory_order_capabilities>();
  if (std::find(std::begin(orders), std::end(orders),
                memory_order::seq_cst) ==
      std::end(orders)) {
    std::cout << "This device does not support "
                 "memory_order::seq_cst.\n";
    return 0;
  }

  auto scopes = d.get_info<
      info::device::atomic_memory_scope_capabilities>();
  if (std::find(std::begin(scopes), std::end(scopes),
                memory_scope::system) == std::end(scopes)) {
    std::cout << "This device does not support "
                 "memory_scope::system.\n";
    return 0;
  }

  int* buffer = malloc_host<int>(1, Q);
  buffer[0] = 0;

  Q.parallel_for(count, [=](auto id) {
     // The scope is optional for a CUDA atomic_ref and
     // defaults to the entire system if unspecified.
     // Additionally, the default CUDA atomic order is
     // sequentially consistent. So, this is the SYCL
     // equivalent of:
     //   cuda::atomic_ref<int> aref(*buffer)
     atomic_ref<int, memory_order::seq_cst,
                memory_scope::system>
         aref(*buffer);

     // When no memory order is specified, the default CUDA
     // atomic_ref atomic order is sequentially consistent.
     aref.fetch_add(1);

     // The CUDA atomic_ref can also provide a specific
     // atomic order but cannot change the scope:
     aref.fetch_add(1, memory_order::relaxed);
   }).wait();

  if (buffer[0] != 2 * count) {
    std::cout << "Found " << buffer[0] << ", wanted "
              << 2 * count << ".\n";
  } else {
    std::cout << "Success.\n";
  }

  free(buffer, Q);
  return 0;
}
