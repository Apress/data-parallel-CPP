// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <mutex>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  queue q;
  int my_ints[42];

  // Create a buffer of 42 ints
  buffer<int> b{range(42)};

  // Create a buffer of 42 ints, initialize with a host
  // pointer, and add the use_host_pointer property
  buffer b1{my_ints,
            range(42),
            {property::buffer::use_host_ptr{}}};

  // Create a buffer of 42 ints, initialize with a host
  // pointer, and add the use_mutex property
  std::mutex myMutex;
  buffer b2{my_ints,
            range(42),
            {property::buffer::use_mutex{myMutex}}};
  // Retrieve a pointer to the mutex used by this buffer
  auto mutexPtr =
      b2.get_property<property::buffer::use_mutex>()
          .get_mutex_ptr();
  // Lock the mutex until we exit scope
  std::lock_guard<std::mutex> guard{*mutexPtr};

  // Create a context-bound buffer of 42 ints, initialized
  // from a host pointer
  buffer b3{
      my_ints,
      range(42),
      {property::buffer::context_bound{q.get_context()}}};

  return 0;
}
