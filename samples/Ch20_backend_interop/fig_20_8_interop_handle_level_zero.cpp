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
  std::cout << "Running on device: "
            << d.get_info<info::device::name>() << "\n";

  buffer<int> b{16};
  queue q{d};

  // BEGIN CODE SNIP
  q.submit([&](handler& h) {
    accessor a{b, h};
    h.host_task([=](interop_handle ih) {
      // Get the Level Zero device from the interop handle:
      auto level0Device = ih.get_native_device<
          backend::ext_oneapi_level_zero>();

      // Query the device name from Level Zero:
      ze_device_properties_t level0DeviceProps = {};
      level0DeviceProps.stype =
          ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES;
      zeDeviceGetProperties(level0Device,
                            &level0DeviceProps);
      std::cout << "Device name from Level Zero is: "
                << level0DeviceProps.name << "\n";

      // Get the Level Zero context and memory allocation
      // from the interop handle:
      auto level0Context = ih.get_native_context<
          backend::ext_oneapi_level_zero>();
      auto ptr =
          ih.get_native_mem<backend::ext_oneapi_level_zero>(
              a);

      // Query the size of the memory allocation:
      size_t sz = 0;
      zeMemGetAddressRange(level0Context, ptr, nullptr,
                           &sz);
      std::cout << "Buffer size from Level Zero is: " << sz
                << " bytes\n";
    });
  });
  // END CODE SNIP

  return 0;
}
