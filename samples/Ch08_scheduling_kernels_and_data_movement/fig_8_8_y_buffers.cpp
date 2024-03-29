// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue q;

  buffer<int> data1{range{N}};
  buffer<int> data2{range{N}};

  q.submit([&](handler &h) {
    accessor a{data1, h};
    h.parallel_for(N, [=](id<1> i) { a[i] = 1; });
  });

  q.submit([&](handler &h) {
    accessor b{data2, h};
    h.parallel_for(N, [=](id<1> i) { b[i] = 2; });
  });

  q.submit([&](handler &h) {
    accessor a{data1, h};
    accessor b{data2, h, read_only};
    h.parallel_for(N, [=](id<1> i) { a[i] += b[i]; });
  });

  q.submit([&](handler &h) {
    accessor a{data1, h};
    h.single_task([=]() {
      for (int i = 1; i < N; i++) a[0] += a[i];

      a[0] /= 3;
    });
  });

  host_accessor h_a{data1};
  assert(h_a[0] == N);
  return 0;
}
