// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include <iostream>
#include <sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>
#include <vector>
using namespace sycl;

int main(int argc, char* argv[]) {
  int openclPlatformIndex = 0;
  int openclDeviceIndex = 0;

  if (argc > 1) {
    openclPlatformIndex = std::stoi(argv[1]);
  }
  if (argc > 2) {
    openclDeviceIndex = std::stoi(argv[2]);
  }
  if (argc <= 1) {
    std::cout << "Run as ./<progname> <OpenCL platform index> <OpenCL device index>\n";
    std::cout << "Defaulting to the first OpenCL platform and device.\n";
  }

  constexpr size_t size = 16;
  std::array<int, size> data;

  for (int i = 0; i < size; i++) {
    data[i] = i;
  }

  // Create an OpenCL context and some OpenCL memory objects:

  cl_uint openclNumPlatforms = 0;
  clGetPlatformIDs(0, nullptr, &openclNumPlatforms);

  if (openclNumPlatforms == 0) {
    std::cout << "Could not find any OpenCL platforms!\n";
    return 0;
  }
  if (openclPlatformIndex >= openclNumPlatforms) {
    std::cout << "Could not find OpenCL platform " << openclPlatformIndex
              << "!\n";
    return -1;
  }

  std::vector<cl_platform_id> openclPlatforms(openclNumPlatforms);
  clGetPlatformIDs(openclNumPlatforms, openclPlatforms.data(), nullptr);

  cl_platform_id openclPlatform = openclPlatforms[openclPlatformIndex];
  cl_uint openclNumDevices = 0;
  clGetDeviceIDs(openclPlatform, CL_DEVICE_TYPE_ALL, 0, nullptr, &openclNumDevices);

  if (openclDeviceIndex >= openclNumDevices) {
    std::cout << "Could not find OpenCL device " << openclDeviceIndex << "!\n";
    return -1;
  }

  std::vector<cl_device_id> openclDevices(openclNumDevices);
  clGetDeviceIDs(
      openclPlatform,
      CL_DEVICE_TYPE_ALL,
      openclNumDevices,
      openclDevices.data(),
      nullptr);

  cl_device_id openclDevice = openclDevices[openclDeviceIndex];
  cl_context openclContext = clCreateContext(nullptr, 1, &openclDevice, nullptr, nullptr, nullptr);
  cl_command_queue openclQueue = clCreateCommandQueue(openclContext, openclDevice, 0, nullptr);
  cl_mem openclBuffer = clCreateBuffer(
      openclContext,
      CL_MEM_USE_HOST_PTR,
      size * sizeof(int),
      data.data(),
      nullptr);

  {
    // BEGIN CODE SNIP
    // Create SYCL objects from the native backend objects.
    context c = make_context<backend::opencl>(openclContext);
    device d = make_device<backend::opencl>(openclDevice);
    buffer data_buf = make_buffer<backend::opencl, int>(openclBuffer, c);

    // Problem #1:
    // Queue cannot be constructed with the given context and device since the device is not a member of the
    // context (descendants of devices from the context are not supported on OpenCL yet).
    // -33 (PI_ERROR_INVALID_DEVICE)

    // Now use the SYCL objects to create a queue and submit a kernel.
    queue Q{c, d};
    // queue Q = make_queue<backend::opencl>(openclQueue, c);

    Q.submit([&](handler& h) {
       accessor data_acc{data_buf, h};
       h.parallel_for(size, [=](id<1> i) {
         data_acc[i] = data_acc[i] + 1;
       });
     }).wait();
    // END CODE SNIP
  }

  clReleaseContext(openclContext);
  clReleaseCommandQueue(openclQueue);
  clReleaseMemObject(openclBuffer);

  for (int i = 0; i < size; i++) {
    if (data[i] != i + 1) {
      std::cout << "Results did not validate at index " << i << "!\n";
      return -1;
    }
  }

  std::cout << "Success!\n";
  return 0;
}
