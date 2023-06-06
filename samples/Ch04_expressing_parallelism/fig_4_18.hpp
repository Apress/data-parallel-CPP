// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snippets that are not set up to be compiled as is.

template <int Dimensions = 1>
class nd_item {
 public:
  // Return the index of this item in the kernel's execution
  // range
  id<Dimensions> get_global_id() const;
  size_t get_global_id(int dimension) const;
  size_t get_global_linear_id() const;

  // Return the execution range of the kernel executed by
  // this item
  range<Dimensions> get_global_range() const;
  size_t get_global_range(int dimension) const;

  // Return the index of this item within its parent
  // work-group
  id<Dimensions> get_local_id() const;
  size_t get_local_id(int dimension) const;
  size_t get_local_linear_id() const;

  // Return the execution range of this item's parent
  // work-group
  range<Dimensions> get_local_range() const;
  size_t get_local_range(int dimension) const;

  // Return a handle to the work-group
  // or sub-group containing this item
  group<Dimensions> get_group() const;
  sub_group get_sub_group() const;
};
