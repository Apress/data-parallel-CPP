// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

void body(group& g);

h.parallel_for(nd_range{global, local}, [=](nd_item<1> it) {
  group<1> g = it.get_group();
  range<1> r = g.get_local_range();
  ...
  body(g);
});
