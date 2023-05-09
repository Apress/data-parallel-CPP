// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

// C++ STREAM Triad workload
// __restrict is used to denote no memory aliasing among arguments  
template <typename T>
double triad(T* __restrict VA, T* __restrict VB, 
             T* __restrict VC, size_t array_size, const T scalar) {
   double ts = timer_start()
   for (size_t id = 0; id < array_size; id++) {
      VC[id] = VA[id] + scalar * VB[id];
   }
   double te = timer_end();
   return (te – ts); 
}
