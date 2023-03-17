// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // BEGIN CODE SNIP
  queue Q;

  std::cout << "By default, we are running on "
            << Q.get_device().get_info<info::device::name>()
            << "\n";
  // END CODE SNIP

  return 0;
}
