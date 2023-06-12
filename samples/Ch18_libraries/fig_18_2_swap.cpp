// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <array>
#include <iostream>
#include <sycl/sycl.hpp>
#include <utility>
using namespace sycl;

int main() {
  std::array<int, 2> arr{8, 9};
  buffer<int> buf{arr};

  {
    host_accessor host_A(buf);
    std::cout << "Before: " << host_A[0] << ", "
              << host_A[1] << "\n";
  }  // End scope of host_A so that upcoming kernel can
     // operate on buf

  queue q;
  q.submit([&](handler &h) {
    accessor a{buf, h};
    h.single_task([=]() {
      // Call std::swap!
      std::swap(a[0], a[1]);
    });
  });

  host_accessor host_B(buf);
  std::cout << "After:  " << host_B[0] << ", " << host_B[1]
            << "\n";
  return 0;
}

// Sample output:
// 8, 9
// 9, 8
	       
