// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  const int n = 10;
  int * d_head = static_cast<int *>(
    sycl::malloc_device(n * sizeof(int),
                        Q.get_device(),
                        Q.get_context()));

  std::fill(oneapi::dpl::execution::make_device_policy(Q),
            d_head, d_head + n, 78);
  Q.wait();

  sycl::free(d_head, Q.get_context());
  return 0;
}
