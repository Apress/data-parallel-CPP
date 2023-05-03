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
    std::cout << "Run as ./<progname> <OpenCL platform index> <OpenCL device index>\n";
    std::cout << "Defaulting to the first OpenCL platform and device.\n";
  }

  std::vector<platform> openclPlatforms = getOpenCLPlatforms();
  if (openclPlatforms.size() == 0) {
    std::cout << "Could not find any SYCL platforms associated with an OpenCL backend!\n";
    return 0;
  }
  if (platformIndex >= openclPlatforms.size()) {
    std::cout << "Platform index " << platformIndex
              << " exceeds the number of platforms associated with an OpenCL backend!\n";
    return -1;
  }

  platform p = openclPlatforms[platformIndex];
  if (deviceIndex >= p.get_devices().size()) {
    std::cout << "Device index " << deviceIndex
              << " exceeds the number of devices in the platform!\n";
  }

  device d = p.get_devices()[deviceIndex];
  context c = context{d};

  // BEGIN CODE SNIP
  cl_device_id openclDevice = get_native<backend::opencl>(d);
  cl_context openclContext = get_native<backend::opencl>(c);

  // Query the device name from OpenCL:
  size_t sz = 0;
  clGetDeviceInfo(openclDevice, CL_DEVICE_NAME, 0, nullptr, &sz);
  std::string openclDeviceName(sz, ' ');
  clGetDeviceInfo(openclDevice, CL_DEVICE_NAME, sz, &openclDeviceName[0], nullptr);
  std::cout << "Device name from OpenCL is: " << openclDeviceName << "\n";

  // Allocate some memory from OpenCL:
  cl_mem openclBuffer = clCreateBuffer(openclContext, 0, sizeof(int), nullptr, nullptr);

  // Clean up OpenCL objects when done:
  clReleaseDevice(openclDevice);
  clReleaseContext(openclContext);
  clReleaseMemObject(openclBuffer);
  // END CODE SNIP

  return 0;
}
