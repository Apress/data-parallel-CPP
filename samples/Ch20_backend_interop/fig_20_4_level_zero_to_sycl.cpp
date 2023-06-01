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

int main(int argc, char* argv[]) {
  int l0DriverIndex = 0;
  int l0DeviceIndex = 0;

  if (argc > 1) {
    l0DriverIndex = std::stoi(argv[1]);
  }
  if (argc > 2) {
    l0DeviceIndex = std::stoi(argv[2]);
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

  CHECK_CALL(zeInit(0));

  // Create an Level Zero context and some Level Zero memory
  // allocations:

  uint32_t l0NumDrivers = 0;
  CHECK_CALL(zeDriverGet(&l0NumDrivers, nullptr));

  if (l0NumDrivers == 0) {
    std::cout << "Could not find any Level Zero drivers!\n";
    return 0;
  }
  if (l0DriverIndex >= l0NumDrivers) {
    std::cout << "Could not find Level Zero driver "
              << l0DriverIndex << "!\n";
    return -1;
  }

  std::vector<ze_driver_handle_t> l0Drivers(l0NumDrivers);
  CHECK_CALL(zeDriverGet(&l0NumDrivers, l0Drivers.data()));

  ze_driver_handle_t l0Driver = l0Drivers[l0DriverIndex];
  uint32_t l0NumDevices = 0;
  CHECK_CALL(zeDeviceGet(l0Driver, &l0NumDevices, nullptr));

  if (l0DeviceIndex >= l0NumDevices) {
    std::cout << "Could not find Level Zero device "
              << l0DeviceIndex << "!\n";
    return -1;
  }

  std::vector<ze_device_handle_t> l0Devices(l0NumDevices);
  CHECK_CALL(zeDeviceGet(l0Driver, &l0NumDevices,
                         l0Devices.data()));

  ze_device_handle_t l0Device = l0Devices[l0DeviceIndex];
  ze_context_handle_t l0Context = nullptr;
  ze_context_desc_t l0ContextDesc = {};
  l0ContextDesc.stype = ZE_STRUCTURE_TYPE_CONTEXT_DESC;
  CHECK_CALL(zeContextCreateEx(l0Driver, &l0ContextDesc, 1,
                               &l0Device, &l0Context));

  void* l0Ptr = nullptr;
  ze_host_mem_alloc_desc_t l0HostAllocDesc = {};
  l0HostAllocDesc.stype =
      ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC;
  CHECK_CALL(zeMemAllocHost(l0Context, &l0HostAllocDesc,
                            size * sizeof(int), 0, &l0Ptr));

  std::memcpy(l0Ptr, data.data(), size * sizeof(int));

  {
    // BEGIN CODE SNIP
    // Create SYCL objects from the native backend objects.
    device d = make_device<backend::ext_oneapi_level_zero>(
        l0Device);
    context c =
        make_context<backend::ext_oneapi_level_zero>(
            {l0Context,
             {d},
             ext::oneapi::level_zero::ownership::keep});
    buffer data_buf =
        make_buffer<backend::ext_oneapi_level_zero, int>(
            {l0Ptr,
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

  std::memcpy(data.data(), l0Ptr, size * sizeof(int));

  CHECK_CALL(zeMemFree(l0Context, l0Ptr));
  CHECK_CALL(zeContextDestroy(l0Context));

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
