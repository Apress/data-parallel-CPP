// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <cassert>
#include <sycl/sycl.hpp>
#define M 32

using namespace sycl;

template <typename T, size_t N>
bool checkEqual(marray<T, N> A, marray<T, N> B) {
  for (int i = 0; i < N; i++) {
    if (A[i] != B[i]) {
      return false;
    }
  }
  return true;
}

int main() {
  // BEGIN CODE SNIP
  queue q;
  marray<float, 4> input{1.0004f, 1e-4f, 1.4f, 14.0f};
  marray<float, 4> res[M];
  for (int i = 0; i < M; i++)
    res[i] = {-(i + 1), -(i + 1), -(i + 1), -(i + 1)};
  {
    buffer in_buf(&input, range{1});
    buffer re_buf(res, range{M});

    q.submit([&](handler &cgh) {
      accessor re_acc{re_buf, cgh, read_write};
      accessor in_acc{in_buf, cgh, read_only};

      cgh.parallel_for(range<1>(M), [=](id<1> idx) {
        int i = idx[0];
        re_acc[i] = cos(in_acc[0]);
      });
    });
  }
  // END CODE SNIP

  if (checkEqual(res[0], res[M / 2]))
    std::cout << "passed\n";
  else
    std::cout << "failed\n";
  return 0;
}
