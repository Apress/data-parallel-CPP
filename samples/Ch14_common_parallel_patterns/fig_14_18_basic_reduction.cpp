// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr size_t N = 16;

  queue q;
  int* data = malloc_shared<int>(N, q);
  int* sum = malloc_shared<int>(1, q);
  std::iota(data, data + N, 1);
  *sum = 0;

  // BEGIN CODE SNIP
  q.parallel_for(N, [=](id<1> i) {
     atomic_ref<int, memory_order::relaxed,
                memory_scope::system,
                access::address_space::global_space>(
         *sum) += data[i];
   }).wait();
  // END CODE SNIP

  std::cout << "sum = " << *sum << "\n";
  bool passed = (*sum == ((N * (N + 1)) / 2));
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(sum, q);
  free(data, q);
  return (passed) ? 0 : 1;
}
