// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  const int n = 10;
  int* h_head = sycl::malloc_host<int>(n, q);
  int* d_head = sycl::malloc_device<int>(n, q);
  std::fill(oneapi::dpl::execution::make_device_policy(q),
            d_head, d_head + n, 78);
  q.wait();

  q.memcpy(h_head, d_head, n * sizeof(int));
  q.wait();

  if (h_head[8] == 78)
    std::cout << "passed" << std::endl;
  else
    std::cout << "failed" << std::endl;

  sycl::free(h_head, q);
  sycl::free(d_head, q);
  return 0;
}
