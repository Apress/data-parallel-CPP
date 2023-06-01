// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <array>
#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  std::array<int, N> a, b, c;
  for (int i = 0; i < N; i++) {
    a[i] = b[i] = c[i] = 0;
  }

  queue q;

  // We will learn how to simplify this example later
  buffer a_buf{a};
  buffer b_buf{b};
  buffer c_buf{c};

  q.submit([&](handler &h) {
    accessor a(a_buf, h, read_only);
    accessor b(b_buf, h, write_only);
    h.parallel_for(  // computeB
        N, [=](id<1> i) { b[i] = a[i] + 1; });
  });

  q.submit([&](handler &h) {
    accessor a(a_buf, h, read_only);
    h.parallel_for(  // readA
        N, [=](id<1> i) {
          // Useful only as an example
          int data = a[i];
        });
  });

  q.submit([&](handler &h) {
    // RAW of buffer B
    accessor b(b_buf, h, read_only);
    accessor c(c_buf, h, write_only);
    h.parallel_for(  // computeC
        N, [=](id<1> i) { c[i] = b[i] + 2; });
  });

  // read C on host
  host_accessor host_acc_c(c_buf, read_only);
  for (int i = 0; i < N; i++) {
    std::cout << host_acc_c[i] << " ";
  }
  std::cout << "\n";
  return 0;
}
