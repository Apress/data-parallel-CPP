// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// -------------------------------------------------------
// Changed from Book:
//   dropped 'using namespace sycl::ONEAPI'
//   this allows reduction to use the sycl::reduction
// -------------------------------------------------------

#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr size_t N = 16;

  queue Q;
  int* data = malloc_shared<int>(N, Q);
  int* sum = malloc_shared<int>(1, Q);
  std::iota(data, data + N, 1);
  *sum = 0;

  Q.submit([&](handler& h) {
     // BEGIN CODE SNIP
     h.parallel_for(
         range<1>{N}, reduction(sum, plus<>()),
         [=](id<1> i, auto& sum) { sum += data[i]; });
     // END CODE SNIP
   }).wait();

  std::cout << "sum = " << *sum << "\n";
  bool passed = (*sum == ((N * (N + 1)) / 2));
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(sum, Q);
  free(data, Q);
  return (passed) ? 0 : 1;
}
