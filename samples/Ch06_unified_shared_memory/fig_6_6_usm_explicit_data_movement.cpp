// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <array>
#include <sycl/sycl.hpp>
using namespace sycl;
constexpr int N = 42;

int main() {
  queue q;

  std::array<int, N> host_array;
  int* device_array = malloc_device<int>(N, q);
  for (int i = 0; i < N; i++) host_array[i] = N;

  q.submit([&](handler& h) {
    // copy host_array to device_array
    h.memcpy(device_array, &host_array[0], N * sizeof(int));
  });
  q.wait();  // needed for now (we learn a better way later)

  q.submit([&](handler& h) {
    h.parallel_for(N, [=](id<1> i) { device_array[i]++; });
  });
  q.wait();  // needed for now (we learn a better way later)

  q.submit([&](handler& h) {
    // copy device_array back to host_array
    h.memcpy(&host_array[0], device_array, N * sizeof(int));
  });
  q.wait();  // needed for now (we learn a better way later)

  free(device_array, q);
  return 0;
}
