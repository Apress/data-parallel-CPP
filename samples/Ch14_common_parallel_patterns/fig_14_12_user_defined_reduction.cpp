// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

// -------------------------------------------------------
// Changed from Book:
//   dropped 'using namespace sycl::ONEAPI'
//   this allows reduction to use the sycl::reduction
// -------------------------------------------------------

#include <sycl/sycl.hpp>
#include <iostream>
#include <random>

using namespace sycl;

template <typename T, typename I>
using minloc = minimum<std::pair<T, I>>;

int main() {
  constexpr size_t N = 16;

  queue Q;
  float* data = malloc_shared<float>(N, Q);
  std::pair<float, int>* res = malloc_shared<std::pair<float, int>>(1, Q);
  std::generate(data, data + N, std::mt19937{});

  std::pair<float, int> identity = {
      std::numeric_limits<float>::max(), std::numeric_limits<int>::min()};
  *res = identity;

  auto red = sycl::reduction(res, identity, minloc<float, int>());

  Q.submit([&](handler& h) {
     h.parallel_for(range<1>{N}, red, [=](id<1> i, auto& res) {
       std::pair<float, int> partial = {data[i], i};
       res.combine(partial);
     });
   }).wait();

  std::cout << "minimum value = " << res->first << " at " << res->second << "\n";

  std::pair<float, int> gold = identity;
  for (int i = 0; i < N; ++i) {
    if (data[i] <= gold.first || (data[i] == gold.first && i < gold.second)) {
      gold.first = data[i];
      gold.second = i;
    }
  }
  bool passed = (res->first == gold.first) && (res->second == gold.second);
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(res, Q);
  free(data, Q);
  return (passed) ? 0 : 1;
}
