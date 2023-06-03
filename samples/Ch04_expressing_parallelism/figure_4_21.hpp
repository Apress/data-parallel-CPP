// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snipets that are not set up to be compiled as is.

class sub_group {
 public:
  // Return the index of the sub-group
  id<1> get_group_id() const;

  // Return the number of sub-groups in this item's parent
  // work-group
  range<1> get_group_range() const;

  // Return the index of the work-item in this sub-group
  id<1> get_local_id() const;

  // Return the number of work-items in this sub-group
  range<1> get_local_range() const;

  // Return the maximum number of work-items in any
  // sub-group in this item's parent work-group
  range<1> get_max_local_range() const;
};
