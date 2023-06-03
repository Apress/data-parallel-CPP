// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // BEGIN CODE SNIP
  queue q;

  std::cout << "By default, we are running on "
            << q.get_device().get_info<info::device::name>()
            << "\n";
  // END CODE SNIP

  return 0;
}
