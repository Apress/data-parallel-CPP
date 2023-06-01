// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue q;

  int *data1 = malloc_shared<int>(N, q);
  int *data2 = malloc_shared<int>(N, q);

  auto e1 =
      q.parallel_for(N, [=](id<1> i) { data1[i] = 1; });

  auto e2 =
      q.parallel_for(N, [=](id<1> i) { data2[i] = 2; });

  auto e3 = q.parallel_for(
      range{N}, {e1, e2},
      [=](id<1> i) { data1[i] += data2[i]; });

  q.single_task(e3, [=]() {
    for (int i = 1; i < N; i++) data1[0] += data1[i];

    data1[0] /= 3;
  });
  q.wait();

  assert(data1[0] == N);
  return 0;
}
