// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

constexpr int count = 16 * 2;

int main() {
  queue Q{property::queue::in_order()};
  std::cout << "Running on device: "
            << Q.get_device().get_info<info::device::name>()
            << "\n";

  int* buffer = malloc_host<int>(count, Q);
  Q.fill(buffer, 0, count);

  // BEGIN CODE SNIP
  Q.parallel_for(nd_range<2>{{2, 16}, {2, 16}},
                 [=](auto item) {
                   auto index = item.get_global_linear_id();
                   auto fastest = item.get_local_id(1);
                   auto sg = item.get_sub_group();
                   auto neighbor =
                       permute_group_by_xor(sg, fastest, 1);
                   buffer[index] = neighbor;
                 })
      .wait();
  // END CODE SNIP

  int unexpected = 0;
  for (int i = 0; i < count; i += 2) {
    if (buffer[i] == buffer[i + 1]) {
      unexpected++;
    }
  }
  if (unexpected) {
    std::cout << "Error, found " << unexpected
              << " matching pairs.\n";
  } else {
    std::cout << "Success.\n";
  }

  free(buffer, Q);
  return 0;
}
