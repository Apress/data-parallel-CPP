// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // BEGIN CODE SNIP
  // Loop through available platforms
  for (auto const& this_platform :
       platform::get_platforms()) {
    std::cout
        << "Found platform: "
        << this_platform.get_info<info::platform::name>()
        << "\n";

    // Loop through available devices in this platform
    for (auto const& this_device :
         this_platform.get_devices()) {
      std::cout
          << " Device: "
          << this_device.get_info<info::device::name>()
          << "\n";
    }
    std::cout << "\n";
  }
  // END CODE SNIP

  return 0;
}


// % clang++ -fsycl fig_12_5_curious.cpp -o curious
// 
// % ./curious
// Found platform: NVIDIA CUDA BACKEND
//  Device: NVIDIA GeForce RTX 3060
// 
// Found platform: AMD HIP BACKEND
//  Device: AMD Radeon RX 5700 XT
// 
// Found platform: Intel(R) OpenCL
//  Device: Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz
// 
// Found platform: Intel(R) OpenCL HD Graphics
//  Device: Intel(R) UHD Graphics P630 [0x3e96]
// 
// Found platform: Intel(R) Level-Zero
//  Device: Intel(R) UHD Graphics P630 [0x3e96]
// 
// Found platform: Intel(R) FPGA Emulation Platform for OpenCL(TM)
//  Device: Intel(R) FPGA Emulation Device
 
