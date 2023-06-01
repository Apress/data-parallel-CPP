// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue q;
  int *host_array = malloc_host<int>(N, q);
  int *shared_array = malloc_shared<int>(N, q);

  for (int i = 0; i < N; i++) {
    // Initialize host_array on host
    host_array[i] = i;
  }

  // We will learn how to simplify this example later
  q.submit([&](handler &h) {
    h.parallel_for(N, [=](id<1> i) {
      // access shared_array and host_array on device
      shared_array[i] = host_array[i] + 1;
    });
  });
  q.wait();

  for (int i = 0; i < N; i++) {
    // access shared_array on host
    host_array[i] = shared_array[i];
  }

  free(shared_array, q);
  free(host_array, q);
  return 0;
}
