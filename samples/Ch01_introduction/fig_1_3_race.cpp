// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

const std::string secret{
    "Ifmmp-!xpsme\"\012J(n!tpssz-!Ebwf/!"
    "J(n!bgsbje!J!dbo(u!ep!uibu/!.!IBM\01"};

const auto sz = secret.size();

int main() {
  queue q;

  // BEGIN CODE SNIP
  // ...we are changing one line from Figure 1-1
  char* result = malloc_shared<char>(sz, q);

  // Introduce potential data race!  We don't define a
  // dependence to ensure correct ordering with later
  // operations.
  q.memcpy(result, secret.data(), sz);

  q.parallel_for(sz, [=](auto& i) {
     result[i] -= 1;
   }).wait();

  // ...
  // END CODE SNIP
  std::cout << result << "\n";
  free(result, q);
  return 0;
}
