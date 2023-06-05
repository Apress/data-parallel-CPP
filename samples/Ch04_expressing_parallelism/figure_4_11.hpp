// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snipets that are not set up to be compiled as is.

template <int Dimensions = 1, bool WithOffset = true>
class item {
 public:
  // Return the index of this item in the kernel's execution
  // range
  id<Dimensions> get_id() const;
  size_t get_id(int dimension) const;
  size_t operator[](int dimension) const;

  // Return the execution range of the kernel executed by
  // this item
  range<Dimensions> get_range() const;
  size_t get_range(int dimension) const;

  // Return the offset of this item (if WithOffset == true)
  id<Dimensions> get_offset() const;

  // Return the linear index of this item
  // e.g. id(0) * range(1) * range(2) + id(1) * range(2) +
  // id(2)
  size_t get_linear_id() const;
};
