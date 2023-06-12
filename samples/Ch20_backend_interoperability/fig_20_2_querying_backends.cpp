// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  for (auto& p : platform::get_platforms()) {
    std::cout << "SYCL Platform: "
              << p.get_info<info::platform::name>()
              << " is associated with SYCL Backend: "
              << p.get_backend() << std::endl;
  }
  return 0;
}

// Example Output:
// SYCL Platform: Portable Computing Language is associated with SYCL Backend: opencl
// SYCL Platform: Intel(R) OpenCL HD Graphics is associated with SYCL Backend: opencl
// SYCL Platform: Intel(R) OpenCL is associated with SYCL Backend: opencl
// SYCL Platform: Intel(R) FPGA Emulation Platform for OpenCL(TM) is associated with SYCL Backend: opencl
// SYCL Platform: Intel(R) Level-Zero is associated with SYCL Backend: ext_oneapi_level_zero
// SYCL Platform: NVIDIA CUDA BACKEND is associated with SYCL Backend: ext_oneapi_cuda
// SYCL Platform: AMD HIP BACKEND is associated with SYCL Backend: ext_oneapi_hip
