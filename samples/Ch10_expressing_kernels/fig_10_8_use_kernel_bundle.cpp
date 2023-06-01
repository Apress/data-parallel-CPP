// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <array>
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  constexpr size_t size = 16;
  std::array<int, size> data;

  for (int i = 0; i < size; i++) {
    data[i] = i;
  }

  {
    buffer data_buf{data};

    queue q;
    std::cout
        << "Running on device: "
        << q.get_device().get_info<info::device::name>()
        << "\n";

    // BEGIN CODE SNIP
    auto kb = get_kernel_bundle<bundle_state::executable>(
        q.get_context());

    std::cout
        << "All kernel compilation should be done now.\n";

    q.submit([&](handler& h) {
      // Use the pre-compiled kernel from the kernel bundle.
      h.use_kernel_bundle(kb);

      accessor data_acc{data_buf, h};
      h.parallel_for(range{size}, [=](id<1> i) {
        data_acc[i] = data_acc[i] + 1;
      });
    });
    // END CODE SNIP
  }

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
