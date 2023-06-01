// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Create queue bound to an available GPU device
  queue q{gpu_selector_v};

  std::cout << "Selected device: "
            << q.get_device().get_info<info::device::name>()
            << "\n";
  std::cout
      << " -> Device vendor: "
      << q.get_device().get_info<info::device::vendor>()
      << "\n";

  return 0;
}
