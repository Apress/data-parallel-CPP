// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/ext/intel/fpga_extensions.hpp>  // For fpga_emulator_selector_v
#include <sycl/sycl.hpp>
using namespace sycl;

int generate_random_number_from_ID(const id<3>& I) {
  return 0;  // Useless non-RNG generator as proxy!
};

int main() {
  queue q{ext::intel::fpga_emulator_selector_v};

  buffer<int, 3> B{range{16, 16, 16}};

  q.submit([&](handler& h) {
    accessor output(B, h);
    // BEGIN CODE SNIP
    h.parallel_for({16, 16, 16}, [=](auto I) {
      output[I] = generate_random_number_from_ID(I);
    });
    // END CODE SNIP
  });

  return 0;
}
