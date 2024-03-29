// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <algorithm>
#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue q;

  const size_t N = 32;
  const size_t M = 4;

  // BEGIN CODE SNIP
  int* data = malloc_shared<int>(N, q);
  std::fill(data, data + N, 0);

  q.parallel_for(N, [=](id<1> i) {
     int j = i % M;
     data[j] += 1;
   }).wait();

  for (int i = 0; i < N; ++i) {
    std::cout << "data [" << i << "] = " << data[i] << "\n";
  }
  // END CODE SNIP

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


// N = 2, M = 2:
// data [0] = 1
// data [1] = 1
// 
// N = 2, M = 1:
// data [0] = 1
// data [1] = 0
	       
