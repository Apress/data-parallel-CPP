// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue q{property::queue::in_order()};

  int *data = malloc_shared<int>(N, q);

  q.parallel_for(N, [=](id<1> i) { data[i] = 1; });

  q.single_task([=]() {
    for (int i = 1; i < N; i++) data[0] += data[i];
  });
  q.wait();

  assert(data[0] == N);
  return 0;
}
