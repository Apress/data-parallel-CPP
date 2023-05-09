// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

q.submit([&](sycl::handler &h) { // assume sub group size is 8
  ...
  h.parallel_for<class vec_compute>(range<1>(8), [=](id <1> i) {
    ... 
    float4 y4 = b[i];            // i=0, 1, 2, ...  
    ...  
    float  x = dowork(&y4);      // the “dowork” expects y4, 
                                 // i.e., vec_y[8][4] layout 
});
