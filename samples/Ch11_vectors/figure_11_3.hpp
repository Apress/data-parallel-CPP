// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

__swizzled_vec__ lo() const; 
__swizzled_vec__ hi() const; 
__swizzled_vec__ odd() const; 
__swizzled_vec__ even() const; 

template <access::address_space addressSpace> 
  void load(size_t offset, multi_ptr ptr<dataT, addressSpace> ptr); 
template <access::address_space addressSpace>
  void store(size_t offset, multi_ptr ptr<dataT, addressSpace> ptr) const; 

vec<dataT, numElements> &operator=(const vec<dataT, numElements>  &rhs); 
vec<dataT, numElements> &operator=(const dataT &rhs); 
vec<RET, numElements>  operator!(); 

// Not available for floating point types:
vec<dataT, numElements> operator~(); 
