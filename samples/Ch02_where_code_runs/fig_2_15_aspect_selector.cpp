// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // In the aspect_selector form taking a comma seperated
  // group of aspects, all aspects must be present for a
  // device to be selected.
  queue q1{aspect_selector(aspect::fp16, aspect::gpu)};

  // In the aspect_selector form that takes two vectors, the
  // first vector contains aspects that a device must
  // exhibit, and the second contains aspects that must NOT
  // be exhibited.
  queue q2{aspect_selector(
      std::vector{aspect::fp64, aspect::fp16},
      std::vector{aspect::gpu, aspect::accelerator})};

  std::cout
      << "First selected device is: "
      << q1.get_device().get_info<info::device::name>()
      << "\n";

  std::cout
      << "Second selected device is: "
      << q2.get_device().get_info<info::device::name>()
      << "\n";

  return 0;
}

// Example Output:
// First selected device is: Intel(R) UHD Graphics [0x9a60]
//   Second selected device is: 11th Gen Intel(R) Core(TM) i9-11900KB @ 3.30GHz
