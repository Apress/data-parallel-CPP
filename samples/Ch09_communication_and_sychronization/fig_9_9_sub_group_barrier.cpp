// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <array>
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  constexpr size_t size = 16;
  std::array<int, size> data;

  for (int i = 0; i < size; i++) data[i] = i;

  {
    buffer data_buf{data};

    queue q;
    std::cout
        << "Running on device: "
        << q.get_device().get_info<info::device::name>()
        << "\n";

    q.submit([&](handler& h) {
      accessor data_acc{data_buf, h};

      // BEGIN CODE SNIP
      h.parallel_for(
          nd_range{{size}, {16}}, [=](nd_item<1> item) {
            auto sg = item.get_sub_group();
            group_barrier(sg);
            // ...
            auto index = item.get_global_id();
            data_acc[index] = data_acc[index] + 1;
          });
      // END CODE SNIP
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
