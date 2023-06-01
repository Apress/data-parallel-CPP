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
    std::cout << "Run as ./<progname> <OpenCL platform "
                 "index> <OpenCL device index>\n";
    std::cout << "Defaulting to the first OpenCL platform "
                 "and device.\n";
  }

  std::vector<platform> openclPlatforms =
      getOpenCLPlatforms();
  if (openclPlatforms.size() == 0) {
    std::cout << "Could not find any SYCL platforms "
                 "associated with an OpenCL backend!\n";
    return 0;
  }
  if (platformIndex >= openclPlatforms.size()) {
    std::cout << "Platform index " << platformIndex
              << " exceeds the number of platforms "
                 "associated with an OpenCL backend!\n";
    return -1;
  }

  platform p = openclPlatforms[platformIndex];
  if (deviceIndex >= p.get_devices().size()) {
    std::cout << "Device index " << deviceIndex
              << " exceeds the number of devices in the "
                 "platform!\n";
  }

  device d = p.get_devices()[deviceIndex];
  std::cout << "Running on device: "
            << d.get_info<info::device::name>() << "\n";

  buffer<int> b{16};
  queue q{d};

  // BEGIN CODE SNIP
  q.submit([&](handler& h) {
    accessor a{b, h};
    h.host_task([=](interop_handle ih) {
      // Get the OpenCL queue from the interop handle:
      auto nq = ih.get_native_queue<backend::opencl>();

      // Query device name from the OpenCL queue:
      cl_device_id nd;
      clGetCommandQueueInfo(nq, CL_QUEUE_DEVICE, sizeof(nd),
                            &nd, nullptr);

      size_t sz = 0;
      clGetDeviceInfo(nd, CL_DEVICE_NAME, 0, nullptr, &sz);
      std::string openclDeviceName(sz, ' ');
      clGetDeviceInfo(nd, CL_DEVICE_NAME, sz,
                      &openclDeviceName[0], nullptr);
      std::cout << "Queue device name from OpenCL is: "
                << openclDeviceName << "\n";

      // Get the OpenCL buffer from the interop handle:
      auto nmem = ih.get_native_mem<backend::opencl>(a)[0];

      // Query the size of the OpenCL buffer:
      clGetMemObjectInfo(nmem, CL_MEM_SIZE, sizeof(sz), &sz,
                         nullptr);
      std::cout << "Buffer size from OpenCL is: " << sz
                << " bytes\n";
    });
  });
  // END CODE SNIP

  return 0;
}
