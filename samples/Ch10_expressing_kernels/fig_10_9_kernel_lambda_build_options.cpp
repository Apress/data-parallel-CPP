// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <CL/sycl.hpp>
#include <array>
#include <iostream>
using namespace sycl;

class Add;

int main() {
  constexpr size_t size = 16;
  std::array<int, size> data;

  for (int i = 0; i < size; i++) {
    data[i] = i;
  }

  {
    buffer data_buf{data};

    queue Q{ cpu_selector{} };
    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>() << "\n";

// BEGIN CODE SNIP
    kernel_id AddID = get_kernel_id<class Add>();
    auto KernelBundle =
        get_kernel_bundle<bundle_state::executable>(Q.get_context(), {AddID});
    kernel Kernel = KernelBundle.get_kernel(AddID);

    Q.submit([&](handler& h) {
      accessor data_acc {data_buf, h};

      h.parallel_for<class Add>(
          // This uses the previously compiled kernel.
          Kernel, range{size}, [=](id<1> i) { data_acc[i] = data_acc[i] + 1; });
    });
// END CODE SNIP
  }

  for (int i = 0; i < size; i++) {
    if (data[i] != i + 1) {
      std::cout << "Results did not validate at index " << i << "!\n";
      return -1;
    }
  }

  std::cout << "Success!\n";
  return 0;
}
