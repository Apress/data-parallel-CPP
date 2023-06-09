// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // BEGIN CODE SNIP
  queue q;
  device dev = q.get_device();

  std::cout << "We are running on:\n"
            << dev.get_info<info::device::name>() << "\n";

  // Query results like the following can be used to
  // calculate how large your kernel invocations can be.
  auto maxWG =
      dev.get_info<info::device::max_work_group_size>();
  auto maxGmem =
      dev.get_info<info::device::global_mem_size>();
  auto maxLmem =
      dev.get_info<info::device::local_mem_size>();

  std::cout << "Max WG size is " << maxWG
            << "\nGlobal memory size is " << maxGmem
            << "\nLocal memory size is " << maxLmem << "\n";

  // END CODE SNIP
  return 0;
}
