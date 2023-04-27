// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {

// BEGIN CODE SNIP
  queue MyQ;

  std::cout << "By default, we are running on "
    << MyQ.get_device().get_info<info::device::name>() << "\n";
// END CODE SNIP

  return 0;
}

