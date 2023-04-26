// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <cassert>
#include <sycl/sycl.hpp>

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
  queue q;
  marray<float, 4> input{1.0004f, 1e-4f, 1.4f, 14.0f};
  marray<float, 4> res[2] = {{-1, -1, -1, -1}, {-2, -2, -2, -2}};
  {
    buffer<marray<float, 4>, 1> in_buf(&input, 1);
    buffer<marray<float, 4>, 1> re_buf(&res[0], sycl::range<1>(2));

    q.submit([&](handler &cgh) {
      accessor<marray<float, 4>, 1, access_mode::write, target::device> re_acc(re_buf, cgh);
      accessor<marray<float, 4>, 1, access_mode::read,  target::device> in_acc(in_buf, cgh);

      cgh.parallel_for(range<1>(2), [=](id<1> idx) {
        int i = idx[0];
	if (i == 0) 
          re_acc[i] = native::cos(in_acc[0]);
        else
          re_acc[i] = cos(in_acc[0]);
      });
    });
  }
  if (checkEqual(res[0], res[1]))
    std::cout << "passed\n";
  else 
    std::cout << "failed\n";
  return 0;
}
