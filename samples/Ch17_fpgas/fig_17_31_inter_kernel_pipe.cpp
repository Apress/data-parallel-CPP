// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <array>
#include <sycl/ext/intel/fpga_extensions.hpp>  // For fpga_emulator_selector_v
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  constexpr int count = 1024;
  std::array<int, count> in_array;

  // Initialize input array
  for (int i = 0; i < count; i++) {
    in_array[i] = i;
  }

  // Buffer initialized from in_array (std::array)
  buffer<int> b_in{in_array};

  // Uninitialized buffer with count elements
  buffer<int> b_out{range{count}};

  // Acquire queue to emulated FPGA device
  queue q{ext::intel::fpga_emulator_selector_v};

  // BEGIN CODE SNIP
  // Create alias for pipe type to be consistent across uses
  using my_pipe = ext::intel::pipe<class some_pipe, int>;

  // ND-range kernel
  q.submit([&](handler& h) {
    auto a = accessor(b_in, h);

    h.parallel_for(
        count, [=](auto idx) { my_pipe::write(a[idx]); });
  });

  // Single_task kernel
  q.submit([&](handler& h) {
    auto a = accessor(b_out, h);

    h.single_task([=]() {
      for (int i = 0; i < count; i++) {
        a[i] = my_pipe::read();
      }
    });
  });

  // END CODE SNIP

  auto a = host_accessor(b_out);
  for (int i = 0; i < count; i++) {
    if (a[i] != i) {
      std::cout << "Failure on element " << i << "\n";
      return 1;
    }
  }
  std::cout << "Passed!\n";
  return 0;
}
