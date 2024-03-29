// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue q;

  buffer<int> data{range{N}};

  q.submit([&](handler &h) {
    accessor a{data, h};
    h.parallel_for(N, [=](id<1> i) { a[i] = 1; });
  });

  q.submit([&](handler &h) {
    accessor a{data, h};
    h.single_task([=]() {
      for (int i = 1; i < N; i++) a[0] += a[i];
    });
  });

  host_accessor h_a{data};
  assert(h_a[0] == N);
  return 0;
}
