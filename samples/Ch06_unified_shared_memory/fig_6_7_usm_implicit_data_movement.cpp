// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue q;

  int* host_array = malloc_host<int>(N, q);
  int* shared_array = malloc_shared<int>(N, q);
  for (int i = 0; i < N; i++) host_array[i] = i;

  q.submit([&](handler& h) {
    h.parallel_for(N, [=](id<1> i) {
      // access shared_array and host_array on device
      shared_array[i] = host_array[i] + 1;
    });
  });
  q.wait();

  free(shared_array, q);
  free(host_array, q);
  return 0;
}
