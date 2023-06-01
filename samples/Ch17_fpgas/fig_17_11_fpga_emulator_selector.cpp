// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/ext/intel/fpga_extensions.hpp>  // For fpga_selector_v
#include <sycl/sycl.hpp>
using namespace sycl;

void say_device(const queue& q) {
  std::cout << "Device : "
            << q.get_device().get_info<info::device::name>()
            << "\n";
}

int main() {
  queue q{ext::intel::fpga_emulator_selector_v};
  say_device(q);

  q.submit([&](handler& h) {
    h.parallel_for(1024, [=](auto idx) {
      // ...
    });
  });

  return 0;
}
