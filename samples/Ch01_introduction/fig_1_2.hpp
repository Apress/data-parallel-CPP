// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snippets
// that are not set up to be compiled as is.

! Fortran loop
do i = 1, n
  z(i) = alpha * x(i) + y(i)
end do

// C++ loop
for (int i=0;i<n;i++) {
  z[i] = alpha * x[i] + y[i];
}

// SYCL kernel
q.parallel_for(range{n},[=](id<1> i) {
  z[i] = alpha * x[i] + y[i];
}).wait();
