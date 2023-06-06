// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <level_zero/ze_api.h>

#include <iostream>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <vector>
using namespace sycl;

std::vector<platform> getLevelZeroPlatforms() {
  std::vector<platform> platforms;
  for (auto& p : platform::get_platforms()) {
    if (p.get_backend() == backend::ext_oneapi_level_zero) {
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
    std::cout << "Run as ./<progname> <Level Zero platform "
                 "index> <Level Zero device index>\n";
    std::cout << "Defaulting to the first Level Zero "
                 "platform and device.\n";
  }

  std::vector<platform> level0Platforms =
      getLevelZeroPlatforms();
  if (level0Platforms.size() == 0) {
    std::cout << "Could not find any SYCL platforms "
                 "associated with a Level Zero backend!\n";
    return 0;
  }
  if (platformIndex >= level0Platforms.size()) {
    std::cout << "Platform index " << platformIndex
              << " exceeds the number of platforms "
                 "associated with a Level Zero backend!\n";
    return -1;
  }

  platform p = level0Platforms[platformIndex];
  if (deviceIndex >= p.get_devices().size()) {
    std::cout << "Device index " << deviceIndex
              << " exceeds the number of devices in the "
                 "platform!\n";
  }

  device d = p.get_devices()[deviceIndex];
  context c = context{d};

  std::cout << "Running on device: "
            << d.get_info<info::device::name>() << "\n";

  // BEGIN CODE SNIP
  ze_device_handle_t level0Device =
      get_native<backend::ext_oneapi_level_zero>(d);
  ze_context_handle_t level0Context =
      get_native<backend::ext_oneapi_level_zero>(c);

  // Query the device name from Level Zero:
  ze_device_properties_t level0DeviceProps = {};
  level0DeviceProps.stype =
      ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;

  zeDeviceGetProperties(level0Device, &level0DeviceProps);

  std::cout << "Device name from SYCL is: "
            << d.get_info<info::device::name>() << "\n";
  std::cout << "Device name from Level Zero is: "
            << level0DeviceProps.name << "\n";

  // Allocate some memory from Level Zero:
  void* level0Ptr = nullptr;
  ze_host_mem_alloc_desc_t level0HostAllocDesc = {};
  level0HostAllocDesc.stype =
      ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  zeMemAllocHost(level0Context, &level0HostAllocDesc,
                 sizeof(int), 0, &level0Ptr);

  // Clean up Level Zero objects when done:
  zeMemFree(level0Context, level0Ptr);
  // END CODE SNIP

  return 0;
}
