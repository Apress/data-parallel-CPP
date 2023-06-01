// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#define SYCL_SIMPLE_SWIZZLES
#include <array>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // BEGIN CODE SNIP
  constexpr int size = 16;

  std::array<float4, size> input;
  for (int i = 0; i < size; i++) {
    input[i] = float4(8.0f, 6.0f, 2.0f, i);
  }

  buffer b(input);

  queue q;
  q.submit([&](handler& h) {
    accessor a{b, h};

    //  We can access the individual elements of a vector by
    //  using the functions x(), y(), z(), w() and so on.
    //
    //  "Swizzles" can be used by calling a vector member
    //  equivalent to the swizzle order that we need, for
    //  example zyx() or any combination of the elements.
    //  The swizzle need not be the same size as the
    //  original vector
    h.parallel_for(size, [=](id<1> idx) {
      auto e = a[idx];
      float w = e.w();
      float4 sw = e.xyzw();
      sw = e.xyzw() * sw.wzyx();
      sw = sw + w;
      a[idx] = sw.xyzw();
    });
  });
  // END CODE SNIP

  host_accessor hostAcc(b);

  for (int i = 0; i < size; i++) {
    if (hostAcc[i].y() != 12.0f + i) {
      std::cout << "Failed\n";
      return -1;
    }
  }

  std::cout << "Passed\n";
  return 0;
}
