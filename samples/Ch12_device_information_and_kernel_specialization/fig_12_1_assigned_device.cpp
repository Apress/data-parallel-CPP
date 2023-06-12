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

// Example Outputs (one line per run – depends on system):
// By default, we are running on NVIDIA GeForce RTX 3060
// By default, we are running on AMD Radeon RX 5700 XT
// By default, we are running on Intel(R) UHD Graphics 770
// By default, we are running on Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
// By default, we are running on Intel(R) Data Center GPU Max 1100
  
