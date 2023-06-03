// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snipets that are not set up to be compiled as is.

vec Class
    declaration template <typename dataT, int numElements>
    class vec;
vec Class Members using element_type = dataT;
vec();
explicit vec(const dataT &arg);
template <typename … argTN>
vec(const argTN &... args);
vec(const vec<dataT, numElements> &rhs);

#ifdef __SYCL_DEVICE_ONLY__  // available on device only
vec(vector_t openclVector);
operator vector_t() const;
#endif

operator dataT()
    const;  // Available only if numElements == 1
size_t get_count() const;
size_t get_size() const;

template <typename convertT, rounding_mode roundingMode>
vec<convertT, numElements> convert() const;
template <typename asT>
asT as() const;
