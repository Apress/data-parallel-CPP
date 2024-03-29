// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 4;

int main() {
  queue q{property::queue::in_order()};

  q.submit([&](handler& h) {
    h.parallel_for(N, [=](id<1> i) { /*...*/ });  // Task A
  });
  q.submit([&](handler& h) {
    h.parallel_for(N, [=](id<1> i) { /*...*/ });  // Task B
  });
  q.submit([&](handler& h) {
    h.parallel_for(N, [=](id<1> i) { /*...*/ });  // Task C
  });

  return 0;
}
