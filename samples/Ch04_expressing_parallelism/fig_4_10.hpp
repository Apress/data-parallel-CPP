// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snippets that are not set up to be compiled as is.

template <int Dimensions = 1>
class id {
 public:
  // Construct an id with one, two or three dimensions
  id(size_t dim0);
  id(size_t dim0, size_t dim1);
  id(size_t dim0, size_t dim1, size_t dim2);

  // Return the component of the id in a specific dimension
  size_t get(int dimension) const;
  size_t &operator[](int dimension);
  size_t operator[](int dimension) const;

  // Arithmetic operations on ids are also supported
};
