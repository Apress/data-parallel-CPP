// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

// Function expects one vector argument (x) and one scalar argument (n)
simd<float, 8> scale(simd<float, 8> x, float n)
{
  return x * n;
}

q.parallel_for(..., sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(8)]]
{
  // In SPMD code, each work-item has its own x and n variables
  float x = ...;
  float n = ...;

  // Invoke SIMD function (scale) using work-items in the sub-group
  // x values from each work-item are combined into a simd<float, 8>
  // n values are defined to be the same (uniform) across all work-items
  // Returned simd<float, 8> is unpacked
  sycl::sub_group sg = it.get_sub_group();
  float y = invoke_simd(sg, scale, x, uniform(n));
});
