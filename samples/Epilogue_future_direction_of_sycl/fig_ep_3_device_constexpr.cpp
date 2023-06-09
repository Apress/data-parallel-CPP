// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;

  q.submit([&](handler& h) {
     stream out(9, 9, h);
     // BEGIN CODE SNIP
     h.parallel_for(range{1}, [=](id<1> idx) {
       if_device_has<aspect::cpu>([&]() {
         /* Code specialized for CPUs */
         out << "On a CPU!" << endl;
       }).else_if_device_has<aspect::gpu>([&]() {
         /* Code specialized for GPUs */
         out << "On a GPU!" << endl;
       });
     });
     // END CODE SNIP
   }).wait();
}
