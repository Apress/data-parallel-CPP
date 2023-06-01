// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  buffer<int> b{range{16}};

  // ERROR: Create sub-buffer larger than size of parent
  // buffer An exception is thrown from within the buffer
  // constructor
  buffer<int> b2(b, id{8}, range{16});

  return 0;
}
