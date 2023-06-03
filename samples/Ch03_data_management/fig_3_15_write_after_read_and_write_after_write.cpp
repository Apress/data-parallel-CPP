// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <array>
#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  std::array<int, N> a, b;
  for (int i = 0; i < N; i++) {
    a[i] = b[i] = 0;
  }

  queue q;
  buffer a_buf{a};
  buffer b_buf{b};

  q.submit([&](handler &h) {
    accessor a(a_buf, h, read_only);
    accessor b(b_buf, h, write_only);
    h.parallel_for(  // computeB
        N, [=](id<1> i) { b[i] = a[i] + 1; });
  });

  q.submit([&](handler &h) {
    // WAR of buffer A
    accessor a(a_buf, h, write_only);
    h.parallel_for(  // rewriteA
        N, [=](id<1> i) { a[i] = 21 + 21; });
  });

  q.submit([&](handler &h) {
    // WAW of buffer B
    accessor b(b_buf, h, write_only);
    h.parallel_for(  // rewriteB
        N, [=](id<1> i) { b[i] = 30 + 12; });
  });

  host_accessor host_acc_a(a_buf, read_only);
  host_accessor host_acc_b(b_buf, read_only);
  for (int i = 0; i < N; i++) {
    std::cout << host_acc_a[i] << " " << host_acc_b[i]
              << " ";
  }
  std::cout << "\n";
  return 0;
}
