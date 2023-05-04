// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Create queue bound to an available GPU device
  queue Q{gpu_selector_v};

  std::cout << "Selected device: "
            << Q.get_device().get_info<info::device::name>()
            << "\n";
  std::cout
      << " -> Device vendor: "
      << Q.get_device().get_info<info::device::vendor>()
      << "\n";

  return 0;
}
