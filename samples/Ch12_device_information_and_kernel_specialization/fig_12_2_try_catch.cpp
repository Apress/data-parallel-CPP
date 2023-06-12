// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // BEGIN CODE SNIP
  auto GPU_is_available = false;

  try {
    device testForGPU(gpu_selector_v);
    GPU_is_available = true;
  } catch (exception const& ex) {
    std::cout << "Caught this SYCL exception: " << ex.what()
              << std::endl;
  }

  auto q = GPU_is_available ? queue(gpu_selector_v)
                            : queue(default_selector_v);

  std::cout
      << "After checking for a GPU, we are running on:\n "
      << q.get_device().get_info<info::device::name>()
      << "\n";

  // END CODE SNIP
  return 0;
}

// After checking for a GPU, we are running on:
//  AMD Radeon RX 5700 XT
// After checking for a GPU, we are running on:
//  Intel(R) Data Center GPU Max 1100
// After checking for a GPU, we are running on:
//  NVIDIA GeForce RTX 3060
// After checking for a GPU, we are running on:
//  Intel(R) UHD Graphics 770
// 
// Example Output (using a system without GPU):
// Caught this SYCL exception: No device of
// requested type 'info::device_type::gpu' available.
// ...(PI_ERROR_DEVICE_NOT_FOUND)
// After checking for a GPU, we are running on:
//  AMD Ryzen 5 3600 6-Core Processor
