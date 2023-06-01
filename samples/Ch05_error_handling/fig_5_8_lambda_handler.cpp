// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;

// BEGIN CODE SNIP

// Our simple asynchronous handler function
auto handle_async_error = [](exception_list elist) {
  for (auto& e : elist) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception& e) {
      std::cout << "ASYNC EXCEPTION!!\n";
      std::cout << e.what() << "\n";
    }
  }
};

// END CODE SNIP

void say_device(const queue& q) {
  std::cout << "Device : "
            << q.get_device().get_info<info::device::name>()
            << "\n";
}

int main() {
  queue q1{gpu_selector_v, handle_async_error};
  queue q2{cpu_selector_v, handle_async_error};
  say_device(q1);
  say_device(q2);

  try {
    q1.submit(
        [&](handler& h) {
          // Empty command group is illegal and generates an
          // error
        },
        q2);  // Secondary/backup queue!
  } catch (...) {
  }  // Discard regular C++ exceptions for this example
  return 0;
}
