// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Create queue on whatever default device that the
  // implementation chooses. Implicit use of
  // default_selector_v
  queue q;

  std::cout << "Selected device: "
            << q.get_device().get_info<info::device::name>()
            << "\n";

  return 0;
}

// Sample Outputs (one line per run depending on system):
// Selected device: NVIDIA GeForce RTX 3060
// Selected device: AMD Radeon RX 5700 XT
// Selected device: Intel(R) Data Center GPU Max 1100
// Selected device: Intel(R) FPGA Emulation Device
// Selected device: AMD Ryzen 5 3600 6-Core Processor
// Selected device: Intel(R) UHD Graphics 770
// Selected device: Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz
// Selected device: 11th Gen Intel(R) Core(TM) i9-11900KB @ 3.30GHz
// many more possible… these are only examples
