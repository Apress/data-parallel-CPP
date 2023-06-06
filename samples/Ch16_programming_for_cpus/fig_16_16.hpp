// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snippets that are not set up to be compiled as is.

cgh.parallel_for<class aos<T>>(numOfItems,[=](id<1> wi) {
  x[wi] =
      a.x[wi];  // lead to unit-stride vector load x[0:4]
  y[wi] =
      a.y[wi];  // lead to unit-stride vector load y[0:4]
  z[wi] =
      a.z[wi];  // lead to unit-stride vector load z[0:4]
  w[wi] =
      a.w[wi];  // lead to unit-stride vector load w[0:4]
});
