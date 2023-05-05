// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <array>
#include <iostream>
#include <sycl/sycl.hpp>
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

    queue Q;
    std::cout
        << "Running on device: "
        << Q.get_device().get_info<info::device::name>()
        << "\n";

    // BEGIN CODE SNIP
    auto kid = get_kernel_id<class Add>();
    auto kb = get_kernel_bundle<bundle_state::executable>(
        Q.get_context(), {Q.get_device()}, {kid});
    auto kernel = kb.get_kernel(kid);

    std::cout
        << "The maximum work-group size for the kernel and "
           "this device is: "
        << kernel.get_info<info::kernel_device_specific::
                               work_group_size>(
               Q.get_device())
        << "\n";

    std::cout
        << "The preferred work-group size multiple for the "
           "kernel and this device is: "
        << kernel.get_info<
               info::kernel_device_specific::
                   preferred_work_group_size_multiple>(
               Q.get_device())
        << "\n";
    // END CODE SNIP

    Q.submit([&](handler& h) {
      accessor data_acc{data_buf, h};
      h.parallel_for<class Add>(range{size}, [=](id<1> i) {
        data_acc[i] = data_acc[i] + 1;
      });
    });
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
