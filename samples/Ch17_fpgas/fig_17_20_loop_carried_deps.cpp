// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/ext/intel/fpga_extensions.hpp>  // For fpga_emulator_selector_v
#include <sycl/sycl.hpp>
using namespace sycl;

int generate_random_number(const int& state) {
  return 0;  // Useless non-RNG generator as proxy!
};

int main() {
  constexpr int size = 64;
  queue q{ext::intel::fpga_emulator_selector_v};

  buffer<int> b{range{size}};

  q.submit([&](handler& h) {
    accessor output(b, h);

    h.single_task([=]() {
      // BEGIN CODE SNIP
      int a = 0;
      for (int i = 0; i < size; i++) {
        a = a + i;
      }
      // END CODE SNIP
    });
  });

  return 0;
}
