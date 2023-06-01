// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <level_zero/ze_api.h>

#include <iostream>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <vector>
using namespace sycl;

#define CHECK_CALL(_call)                          \
  do {                                             \
    ze_result_t result = _call;                    \
    if (result != ZE_RESULT_SUCCESS) {             \
      printf("%s returned %u!\n", #_call, result); \
    }                                              \
  } while (0)

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

  std::vector<platform> l0Platforms =
      getLevelZeroPlatforms();
  if (l0Platforms.size() == 0) {
    std::cout << "Could not find any SYCL platforms "
                 "associated with a Level Zero backend!\n";
    return 0;
  }
  if (platformIndex >= l0Platforms.size()) {
    std::cout << "Platform index " << platformIndex
              << " exceeds the number of platforms "
                 "associated with a Level Zero backend!\n";
    return -1;
  }

  platform p = l0Platforms[platformIndex];
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
  auto l0Context =
      get_native<backend::ext_oneapi_level_zero>(
          q.get_context());

  // BEGIN CODE SNIP
  q.submit([&](handler& h) {
    accessor a{b, h};
    h.host_task([=](interop_handle ih) {
      // Get the Level Zero memory allocation from the
      // interop handle:
      auto ptr =
          ih.get_native_mem<backend::ext_oneapi_level_zero>(
              a);

      // Query the size of the memory allocation:
      size_t sz = 0;
      CHECK_CALL(zeMemGetAddressRange(l0Context, ptr,
                                      nullptr, &sz));
      std::cout << "Buffer size from Level Zero is: " << sz
                << " bytes\n";
    });
  });
  // END CODE SNIP

  return 0;
}
