// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <level_zero/ze_api.h>

#include <iostream>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <vector>
using namespace sycl;

int main(int argc, char* argv[]) {
  int level0DriverIndex = 0;
  int level0DeviceIndex = 0;

  if (argc > 1) {
    level0DriverIndex = std::stoi(argv[1]);
  }
  if (argc > 2) {
    level0DeviceIndex = std::stoi(argv[2]);
  }
  if (argc <= 1) {
    std::cout << "Run as ./<progname> <Level Zero driver "
                 "index> <Level Zero device index>\n";
    std::cout << "Defaulting to the first OpenCL Level "
                 "Zero driver and device.\n";
  }

  constexpr size_t size = 16;
  std::array<int, size> data;

  for (int i = 0; i < size; i++) {
    data[i] = i;
  }

  zeInit(0);

  // Create an Level Zero context and some Level Zero memory
  // allocations:

  uint32_t level0NumDrivers = 0;
  zeDriverGet(&level0NumDrivers, nullptr);

  if (level0NumDrivers == 0) {
    std::cout << "Could not find any Level Zero drivers!\n";
    return 0;
  }
  if (level0DriverIndex >= level0NumDrivers) {
    std::cout << "Could not find Level Zero driver "
              << level0DriverIndex << "!\n";
    return -1;
  }

  std::vector<ze_driver_handle_t> level0Drivers(
      level0NumDrivers);
  zeDriverGet(&level0NumDrivers, level0Drivers.data());

  ze_driver_handle_t level0Driver =
      level0Drivers[level0DriverIndex];
  uint32_t level0NumDevices = 0;
  zeDeviceGet(level0Driver, &level0NumDevices, nullptr);

  if (level0DeviceIndex >= level0NumDevices) {
    std::cout << "Could not find Level Zero device "
              << level0DeviceIndex << "!\n";
    return -1;
  }

  std::vector<ze_device_handle_t> level0Devices(
      level0NumDevices);
  zeDeviceGet(level0Driver, &level0NumDevices,
              level0Devices.data());

  ze_device_handle_t level0Device =
      level0Devices[level0DeviceIndex];
  ze_context_handle_t level0Context = nullptr;
  ze_context_desc_t level0ContextDesc = {};
  level0ContextDesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  zeContextCreateEx(level0Driver, &level0ContextDesc, 1,
                    &level0Device, &level0Context);

  void* level0Ptr = nullptr;
  ze_host_mem_alloc_desc_t level0HostAllocDesc = {};
  level0HostAllocDesc.stype =
      ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  zeMemAllocHost(level0Context, &level0HostAllocDesc,
                 size * sizeof(int), 0, &level0Ptr);

  std::memcpy(level0Ptr, data.data(), size * sizeof(int));

  {
    // BEGIN CODE SNIP
    // Create SYCL objects from the native backend objects.
    device d = make_device<backend::ext_oneapi_level_zero>(
        level0Device);
    context c =
        make_context<backend::ext_oneapi_level_zero>(
            {level0Context,
             {d},
             ext::oneapi::level_zero::ownership::keep});
    buffer data_buf =
        make_buffer<backend::ext_oneapi_level_zero, int>(
            {level0Ptr,
             ext::oneapi::level_zero::ownership::keep},
            c);

    // Now use the SYCL objects to create a queue and submit
    // a kernel.
    queue q{c, d};

    q.submit([&](handler& h) {
       accessor data_acc{data_buf, h};
       h.parallel_for(size, [=](id<1> i) {
         data_acc[i] = data_acc[i] + 1;
       });
     }).wait();
    // END CODE SNIP
  }

  std::memcpy(data.data(), level0Ptr, size * sizeof(int));

  zeMemFree(level0Context, level0Ptr);
  zeContextDestroy(level0Context);

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
