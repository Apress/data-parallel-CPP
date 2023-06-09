// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// BEGIN CODE SNIP
#include <cassert>
#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue q;

  // Create 3 buffers of 42 ints
  buffer<int> buf_a{range{N}};
  buffer<int> buf_b{range{N}};
  buffer<int> buf_c{range{N}};

  accessor pc{buf_c};

  q.submit([&](handler &h) {
    accessor a{buf_a, h, write_only, no_init};
    accessor b{buf_b, h, write_only, no_init};
    accessor c{buf_c, h, write_only, no_init};
    h.parallel_for(N, [=](id<1> i) {
      a[i] = 1;
      b[i] = 40;
      c[i] = 0;
    });
  });
  q.submit([&](handler &h) {
    accessor a{buf_a, h, read_only};
    accessor b{buf_b, h, read_only};
    accessor c{buf_c, h, read_write};
    h.parallel_for(N,
                   [=](id<1> i) { c[i] += a[i] + b[i]; });
  });
  q.submit([&](handler &h) {
    h.require(pc);
    h.parallel_for(N, [=](id<1> i) { pc[i]++; });
  });

  host_accessor result{buf_c, read_only};

  for (int i = 0; i < N; i++) {
    assert(result[i] == N);
  }
  return 0;
}
// END CODE SNIP
