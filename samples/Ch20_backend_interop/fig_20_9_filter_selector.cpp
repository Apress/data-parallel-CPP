// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  auto find_device = [](backend b, info::device_type t =
                                       info::device_type::all) {
    for (auto d : device::get_devices(t)) {
      if (d.get_backend() == b) {
        return d;
      }
    }
    throw sycl::exception(
        errc::runtime,
        "Could not find a device with the requested backend!");
  };

  try {
    device d{find_device(backend::opencl)};
    std::cout << "Found an OpenCL SYCL device: "
              << d.get_info<info::device::name>() << "\n";
  } catch (const sycl::exception &e) {
    std::cout << "No OpenCL SYCL devices were found.\n";
  }

  try {
    device d{find_device(backend::ext_oneapi_level_zero)};
    std::cout << "Found a Level Zero SYCL device: "
              << d.get_info<info::device::name>() << "\n";
  } catch (const sycl::exception &e) {
    std::cout << "No Level Zero SYCL devices were found.\n";
  }

  return 0;
}
