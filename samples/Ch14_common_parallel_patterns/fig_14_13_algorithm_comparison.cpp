// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr size_t number_of_reductions = 16;
  constexpr size_t elements_per_reduction = 4;

  queue q;
  int* input = malloc_shared<int>(
      number_of_reductions * elements_per_reduction, q);
  int* output1 =
      malloc_shared<int>(number_of_reductions, q);
  int* output2 =
      malloc_shared<int>(number_of_reductions, q);
  int* output3 =
      malloc_shared<int>(number_of_reductions, q);
  std::iota(
      input,
      input + number_of_reductions * elements_per_reduction,
      1);

  // BEGIN CODE SNIP
  // std::reduce
  // Each work-item reduces over a given input range
  q.parallel_for(number_of_reductions, [=](size_t i) {
     output1[i] = std::reduce(
         input + i * elements_per_reduction,
         input + (i + 1) * elements_per_reduction);
   }).wait();

  // sycl::joint_reduce
  // Each work-group reduces over a given input range
  // The elements are automatically distributed over
  // work-items in the group
  q.parallel_for(nd_range<1>{number_of_reductions *
                                 elements_per_reduction,
                             elements_per_reduction},
                 [=](nd_item<1> it) {
                   auto g = it.get_group();
                   int sum = joint_reduce(
                       g,
                       input + g.get_group_id() *
                                   elements_per_reduction,
                       input + (g.get_group_id() + 1) *
                                   elements_per_reduction,
                       plus<>());
                   if (g.leader()) {
                     output2[g.get_group_id()] = sum;
                   }
                 })
      .wait();

  // sycl::reduce_over_group
  // Each work-group reduces over data held in work-item
  // private memory Each work-item is responsible for
  // loading and contributing one value
  q.parallel_for(
       nd_range<1>{
           number_of_reductions * elements_per_reduction,
           elements_per_reduction},
       [=](nd_item<1> it) {
         auto g = it.get_group();
         int x = input[g.get_group_id() *
                           elements_per_reduction +
                       g.get_local_id()];
         int sum = reduce_over_group(g, x, plus<>());
         if (g.leader()) {
           output3[g.get_group_id()] = sum;
         }
       })
      .wait();
  // END CODE SNIP

  std::vector<int> expected(number_of_reductions);
  for (int r = 0; r < number_of_reductions; ++r) {
    int start = r * elements_per_reduction;
    int end = (r + 1) * elements_per_reduction;
    int sum_1_to_end = (end * (end + 1)) / 2;
    int sum_1_to_start = (start * (start + 1)) / 2;
    expected[r] = sum_1_to_end - sum_1_to_start;
  }

  bool passed = true;

  std::cout << "std::reduce:" << std::endl;
  for (int r = 0; r < number_of_reductions; ++r) {
    std::cout << "output[" << r << "]: " << output1[r]
              << std::endl;
    passed &= (output1[r] == expected[r]);
  }
  std::cout << std::endl;

  std::cout << "sycl::joint_reduce:" << std::endl;
  for (int r = 0; r < number_of_reductions; ++r) {
    std::cout << "output[" << r << "]: " << output2[r]
              << std::endl;
    passed &= (output2[r] == expected[r]);
  }
  std::cout << std::endl;

  std::cout << "sycl::reduce_over_group:" << std::endl;
  for (int r = 0; r < number_of_reductions; ++r) {
    std::cout << "output[" << r << "]: " << output3[r]
              << std::endl;
    passed &= (output1[r] == expected[r]);
  }
  std::cout << std::endl;

  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << "\n";

  free(output3, q);
  free(output2, q);
  free(output1, q);
  free(input, q);
  return (passed) ? 0 : 1;
}
