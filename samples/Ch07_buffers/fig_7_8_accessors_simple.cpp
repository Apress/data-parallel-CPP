// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <cassert>
#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue q;
  // Create 3 buffers of 42 ints
  buffer<int> a_buf{range{N}};
  buffer<int> b_buf{range{N}};
  buffer<int> c_buf{range{N}};
  accessor pc{c_buf};

  q.submit([&](handler &h) {
    accessor a{a_buf, h};
    accessor b{b_buf, h};
    accessor c{c_buf, h};
    h.parallel_for(N, [=](id<1> i) {
      a[i] = 1;
      b[i] = 40;
      c[i] = 0;
    });
  });
  q.submit([&](handler &h) {
    accessor a{a_buf, h};
    accessor b{b_buf, h};
    accessor c{c_buf, h};
    h.parallel_for(
        N, [=](id<1> i) { c[i] += a[i] + b[i]; });
  });
  q.submit([&](handler &h) {
    h.require(pc);
    h.parallel_for(N, [=](id<1> i) { pc[i]++; });
  });

  host_accessor result{c_buf};
  for (int i = 0; i < N; i++) {
    assert(result[i] == N);
  }
  return 0;
}
