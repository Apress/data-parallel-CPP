// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  const int n = 10;
  int * h_head = sycl::malloc_host<int>(n, Q);
  int * d_head = sycl::malloc_device<int>(n, Q);
  std::fill(oneapi::dpl::execution::make_device_policy(Q),
            d_head, d_head + n, 78);
  Q.wait();

  Q.memcpy(h_head, d_head, n*sizeof(int));
  Q.wait();

  if (h_head[8] == 78)
    std::cout << "passed" << std::endl;
  else
    std::cout << "failed" << std::endl;

  sycl::free(h_head, Q);
  sycl::free(d_head, Q);
  return 0;
}
