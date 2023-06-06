// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snippets that are not set up to be compiled as is.

template <int Dimensions = 1>
class nd_range {
 public:
  // Construct an nd_range from global and work-group local
  // ranges
  nd_range(range<Dimensions> global,
           range<Dimensions> local);

  // Return the global and work-group local ranges
  range<Dimensions> get_global_range() const;
  range<Dimensions> get_local_range() const;

  // Return the number of work-groups in the global range
  range<Dimensions> get_group_range() const;
};
