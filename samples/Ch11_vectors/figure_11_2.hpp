// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

template<int… swizzleindexes>
__swizzled_vec__ swizzle() const; 
__swizzled_vec__ XYZW_ACCESS() const; 
__swizzled_vec__ RGBA_ACCESS() const; 
__swizzled_vec__ INDEX_ACCESS() const; 

#ifdef SYCL_SIMPLE_SWIZZLES
// Available only when numElements <= 4 
// XYZW_SWIZZLE is all permutations with repetition of: 
// x, y, z, w, subject to numElements
__swizzled_vec__ XYZW_SWIZZLE() const;
   
// Available only when numElements == 4 
// RGBA_SWIZZLE is all permutations with repetition of: r, g, b, a. 
__swizzled_vec__ RGBA_SWIZZLE() const; 
#endif 
