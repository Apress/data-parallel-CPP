// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snipets that are not set up to be compiled as is.

launch N kernel instances {
  int id =
      get_instance_id();  // unique identifier in [0, N)
  c[id] = a[id] + b[id];
}
