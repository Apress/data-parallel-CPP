// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <array>
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  constexpr int size = 16;
  std::array<int, size> data;
  buffer B{data};

  queue q{};  // Select any device for this queue

  std::cout << "Selected device is: "
            << q.get_device().get_info<info::device::name>()
            << "\n";

  // BEGIN CODE SNIP

  q.submit([&](handler& h) {
    accessor acc{B, h};

    h.parallel_for(size,
                   [=](auto& idx) { acc[idx] = idx; });
  });

  // END CODE SNIP

  return 0;
}
