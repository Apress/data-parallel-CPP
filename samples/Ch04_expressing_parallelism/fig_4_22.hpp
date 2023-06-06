// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snippets that are not set up to be compiled as is.

size_t N = ...;  // amount of work
size_t W = ...;  // number of workers
h.parallel_for(range{W}, [=](item<1> it) {
  for (int i = it.get_id()[0]; i < N;
       i += it.get_range()[0]) {
    output[i] = function(input[i]);
  }
});
