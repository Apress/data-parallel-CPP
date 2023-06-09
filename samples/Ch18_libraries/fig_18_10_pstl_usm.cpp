// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

int main() {
  sycl::queue q;
  const int n = 10;
  sycl::usm_allocator<int, sycl::usm::alloc::shared> alloc(
      q);
  std::vector<int, decltype(alloc)> vec(n, alloc);

  std::fill(oneapi::dpl::execution::make_device_policy(q),
            vec.begin(), vec.end(), 78);
  q.wait();

  return 0;
}
