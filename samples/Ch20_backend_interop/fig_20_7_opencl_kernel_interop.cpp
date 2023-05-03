// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <iostream>
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>
#include <vector>
using namespace sycl;

std::vector<platform> getOpenCLPlatforms() {
  std::vector<platform> platforms;
  for (auto& p : platform::get_platforms()) {
    if (p.get_backend() == backend::opencl) {
      platforms.push_back(p);
    }
  }
  return platforms;
}

int main(int argc, char* argv[]) {
  int platformIndex = 0;
  int deviceIndex = 0;

  if (argc > 1) {
    platformIndex = std::stoi(argv[1]);
  }
  if (argc > 2) {
    deviceIndex = std::stoi(argv[2]);
  }
  if (argc <= 1) {
    std::cout
        << "Run as ./<progname> <OpenCL platform index> "
           "<OpenCL device index>\n";
    std::cout
        << "Defaulting to the first OpenCL platform and "
           "device.\n";
  }

  std::vector<platform> openclPlatforms =
      getOpenCLPlatforms();
  if (openclPlatforms.size() == 0) {
    std::cout
        << "Could not find any SYCL platforms associated "
           "with an OpenCL backend!\n";
    return 0;
  }
  if (platformIndex >= openclPlatforms.size()) {
    std::cout
        << "Platform index " << platformIndex
        << " exceeds the number of platforms associated "
           "with an OpenCL backend!\n";
    return -1;
  }

  platform p = openclPlatforms[platformIndex];
  if (deviceIndex >= p.get_devices().size()) {
    std::cout << "Device index " << deviceIndex
              << " exceeds the number of devices in the "
                 "platform!\n";
  }

  constexpr size_t size = 16;
  std::array<int, size> data;

  for (int i = 0; i < size; i++) {
    data[i] = i;
  }

  device d = p.get_devices()[deviceIndex];
  context c = context{d};
  queue Q{c, d};

  std::cout << "Running on device: "
            << d.get_info<info::device::name>() << "\n";

  {
    buffer data_buf{data};

    // BEGIN CODE SNIP
    // Get the native OpenCL context from the SYCL context:
    auto openclContext = get_native<backend::opencl>(c);
    const char* kernelSource =
        R"CLC(
            kernel void add(global int* data) {
                int index = get_global_id(0);
                data[index] = data[index] + 1;
            }
        )CLC";
    // Create an OpenCL kernel using this context:
    cl_program p = clCreateProgramWithSource(
        openclContext, 1, &kernelSource, nullptr, nullptr);
    clBuildProgram(p, 0, nullptr, nullptr, nullptr,
                   nullptr);
    cl_kernel k = clCreateKernel(p, "add", nullptr);

    // Create a SYCL kernel from the OpenCL kernel:
    auto sk = make_kernel<backend::opencl>(k, c);

    // Use the OpenCL kernel with a SYCL queue:
    Q.submit([&](handler& h) {
      accessor data_acc{data_buf, h};

      h.set_args(data_acc);
      h.parallel_for(size, sk);
    });

    // Clean up OpenCL objects when done:
    clReleaseContext(openclContext);
    clReleaseProgram(p);
    clReleaseKernel(k);
    // END CODE SNIP
  }

  for (int i = 0; i < size; i++) {
    if (data[i] != i + 1) {
      std::cout << "Results did not validate at index " << i
                << "!\n";
      return -1;
    }
  }

  std::cout << "Success!\n";
  return 0;
}
