// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 4;

int main() {
  queue q;

  auto eA = q.submit([&](handler &h) {
    h.parallel_for(N, [=](id<1> i) { /*...*/ });  // Task A
  });
  eA.wait();
  auto eB = q.submit([&](handler &h) {
    h.parallel_for(N, [=](id<1> i) { /*...*/ });  // Task B
  });
  auto eC = q.submit([&](handler &h) {
    h.depends_on(eB);
    h.parallel_for(N, [=](id<1> i) { /*...*/ });  // Task C
  });
  auto eD = q.submit([&](handler &h) {
    h.depends_on({eB, eC});
    h.parallel_for(N, [=](id<1> i) { /*...*/ });  // Task D
  });

  return 0;
}
