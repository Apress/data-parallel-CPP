// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  queue q{property::queue::in_order()};
  std::cout << "Running on device: "
            << q.get_device().get_info<info::device::name>()
            << "\n";

  // BEGIN CODE SNIP
  q.parallel_for(nd_range<1>{16, 16}, [=](auto item) {
     // Equivalent of __syncthreads, or
     // this_thread_block().sync():
     group_barrier(item.get_group());

     // Equivalent of __syncwarp, or
     // tiled_partition<32>(this_thread_block()).sync():
     group_barrier(item.get_sub_group());
   }).wait();
  // END CODE SNIP

  std::cout << "Success.\n";
  return 0;
}
