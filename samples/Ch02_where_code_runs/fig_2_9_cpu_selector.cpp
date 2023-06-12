// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Create queue to use the CPU device explicitly
  queue q{cpu_selector_v};

  std::cout << "Selected device: "
            << q.get_device().get_info<info::device::name>()
            << "\n";
  std::cout
      << " -> Device vendor: "
      << q.get_device().get_info<info::device::vendor>()
      << "\n";

  return 0;
}

// Example Output:
// Selected device: Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz
//   -> Device vendor: Intel(R) Corporation
  
