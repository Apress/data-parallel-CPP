// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <algorithm>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;

  const uint32_t N = 32;
  const uint32_t M = 4;

  int* data = malloc_shared<int>(N, q);
  std::fill(data, data + N, 0);

  // Launch exactly one work-group
  // Number of work-groups = global / local
  range<1> global{N};
  range<1> local{N};

  q.parallel_for(nd_range<1>{global, local},
                 [=](nd_item<1> it) {
                   int i = it.get_global_id(0);
                   int j = i % M;
                   for (int round = 0; round < N; ++round) {
                     // Allow exactly one work-item update
                     // per round
                     if (i == round) {
                       data[j] += 1;
                     }
                     group_barrier(it.get_group());
                   }
                 })
      .wait();

  for (int i = 0; i < N; ++i) {
    std::cout << "data [" << i << "] = " << data[i] << "\n";
  }

  bool passed = true;
  int* gold = (int*)malloc(N * sizeof(int));
  std::fill(gold, gold + N, 0);
  for (int i = 0; i < N; ++i) {
    int j = i % M;
    gold[j] += 1;
  }
  for (int i = 0; i < N; ++i) {
    if (data[i] != gold[i]) {
      passed = false;
    }
  }
  std::cout << ((passed) ? "SUCCESS\n" : "FAILURE\n");
  free(gold);
  free(data, q);
  return (passed) ? 0 : 1;
}
