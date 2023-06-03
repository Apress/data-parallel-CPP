// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snipets that are not set up to be compiled as is.

template <int Dimensions = 1>
class range {
 public:
  // Construct a range with one, two or three dimensions
  range(size_t dim0);
  range(size_t dim0, size_t dim1);
  range(size_t dim0, size_t dim1, size_t dim2);

  // Return the size of the range in a specific dimension
  size_t get(int dimension) const;
  size_t &operator[](int dimension);
  size_t operator[](int dimension) const;

  // Return the product of the size of each dimension
  size_t size() const;

  // Arithmetic operations on ranges are also supported
};
