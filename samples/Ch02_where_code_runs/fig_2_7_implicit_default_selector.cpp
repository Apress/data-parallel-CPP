// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  // Create queue on whatever default device that the
  // implementation chooses. Implicit use of
  // default_selector_v
  queue Q;

  std::cout << "Selected device: "
            << Q.get_device().get_info<info::device::name>()
            << "\n";

  return 0;
}
