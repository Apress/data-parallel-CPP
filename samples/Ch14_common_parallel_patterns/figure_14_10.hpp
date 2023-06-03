// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snipets that are not set up to be compiled as is.

template <typename T, typename BinaryOperation,
          /* implementation-defined */>
class reducer {
  // Combine partial result with reducer's value
  void combine(const T& partial);
};

// Other operators are available for standard binary
// operations
template <typename T>
auto& operator+=(reducer<T, plus::<T>>&, const T&);
