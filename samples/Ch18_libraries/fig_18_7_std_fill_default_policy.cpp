// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#include <oneapi/dpl/iterator>
#include <sycl/sycl.hpp>

int main() {
  std::vector<int> v(100000);
  std::fill(oneapi::dpl::execution::dpcpp_default,
            v.begin(), v.end(), 42);

  if (v[788] == 42)
    std::cout << "passed" << std::endl;
  else
    std::cout << "failed" << std::endl;

  return 0;
}
