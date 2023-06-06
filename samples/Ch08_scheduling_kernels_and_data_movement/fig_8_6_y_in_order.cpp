// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue q{property::queue::in_order()};

  int *data1 = malloc_shared<int>(N, q);
  int *data2 = malloc_shared<int>(N, q);

  q.parallel_for(N, [=](id<1> i) { data1[i] = 1; });

  q.parallel_for(N, [=](id<1> i) { data2[i] = 2; });

  q.parallel_for(N, [=](id<1> i) { data1[i] += data2[i]; });

  q.single_task([=]() {
    for (int i = 1; i < N; i++) data1[0] += data1[i];

    data1[0] /= 3;
  });
  q.wait();

  assert(data1[0] == N);
  return 0;
}
