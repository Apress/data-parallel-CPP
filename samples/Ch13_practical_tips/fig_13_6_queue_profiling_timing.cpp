// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

// Array type and data size for this example.
constexpr size_t array_size = (1 << 16);
typedef std::array<int, array_size> IntArray;

void VectorAdd(queue &q, const IntArray &a,
               const IntArray &b, IntArray &sum) {
  range<1> num_items{a.size()};
  buffer a_buf(a), b_buf(b);
  buffer sum_buf(sum.data(), num_items);
  auto t1 =
      std::chrono::steady_clock::now();  // Start timing

  event e = q.submit([&](handler &h) {
    auto a_acc = a_buf.get_access<access::mode::read>(h);
    auto b_acc = b_buf.get_access<access::mode::read>(h);
    auto sum_acc =
        sum_buf.get_access<access::mode::write>(h);

    h.parallel_for(num_items, [=](id<1> i) {
      sum_acc[i] = a_acc[i] + b_acc[i];
    });
  });
  q.wait();

  double timeA =
      (e.template get_profiling_info<
           info::event_profiling::command_end>() -
       e.template get_profiling_info<
           info::event_profiling::command_start>());

  auto t2 =
      std::chrono::steady_clock::now();  // Stop timing

  double timeB = (std::chrono::duration_cast<
                      std::chrono::microseconds>(t2 - t1)
                      .count());

  std::cout
      << "profiling: Vector add completed on device in "
      << timeA << " nanoseconds\n";
  std::cout << "chrono: Vector add completed on device in "
            << timeB * 1000 << " nanoseconds\n";
  std::cout << "chrono more than profiling by "
            << (timeB * 1000 - timeA) << " nanoseconds\n";
}

void InitializeArray(IntArray &a) {
  for (size_t i = 0; i < a.size(); i++) a[i] = i;
}

int main() {
  IntArray a, b, sum;
  InitializeArray(a);
  InitializeArray(b);

  queue q(property::queue::enable_profiling{});

  std::cout << "Vector size: " << a.size()
            << "\nRunning on device: "
            << q.get_device().get_info<info::device::name>()
            << "\n";

  VectorAdd(q, a, b, sum);

  return 0;
}

// Vector size: 65536
// Running on device: Intel(R) UHD Graphics P630 [0x3e96]
// profiling: Vector add completed on device in 57602 nanoseconds
// chrono: Vector add completed on device in 2.85489e+08 nanoseconds
// chrono more than profiling by 2.85431e+08 nanoseconds
// 
// Vector size: 65536
// Running on device: NVIDIA GeForce RTX 3060
// profiling: Vector add completed on device in 17410 nanoseconds
// chrono: Vector add completed on device in 3.6071e+07 nanoseconds
// chrono more than profiling by 3.60536e+07 nanoseconds
// 
// Vector size: 65536
// Running on device: Intel(R) Data Center GPU Max 1100
// profiling: Vector add completed on device in 9440 nanoseconds
// chrono: Vector add completed on device in 5.6976e+07 nanoseconds
// chrono more than profiling by 5.69666e+07 nanoseconds
