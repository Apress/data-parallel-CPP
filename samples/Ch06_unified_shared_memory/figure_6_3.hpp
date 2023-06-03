// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snipets that are not set up to be compiled as is.

// Named Functions
template <typename T>
T *malloc_device(size_t Count, const device &Dev,
                 const context &Ctxt);
template <typename T>
T *malloc_device(size_t Count, const queue &Q);
template <typename T>
T *aligned_alloc_device(size_t Alignment, size_t Count,
                        const device &Dev,
                        const context &Ctxt);
template <typename T>
T *aligned_alloc_device(size_t Alignment, size_t Count,
                        const queue &Q);

template <typename T>
T *malloc_host(size_t Count, const context &Ctxt);
template <typename T>
T *malloc_host(size_t Count, const queue &Q);
template <typename T>
T *aligned_alloc_host(size_t Alignment, size_t Count,
                      const context &Ctxt);
template <typename T>
T *aligned_alloc_host(size_t Alignment, size_t Count,
                      const queue &Q);

template <typename T>
T *malloc_shared(size_t Count, const device &Dev,
                 const context &Ctxt);
template <typename T>
T *malloc_shared(size_t Count, const queue &Q);
template <typename T>
T *aligned_alloc_shared(size_t Alignment, size_t Count,
                        const device &Dev,
                        const context &Ctxt);
template <typename T>
T *aligned_alloc_shared(size_t Alignment, size_t Count,
                        const queue &Q);

// Single Function
template <typename T>
T *malloc(size_t Count, const device &Dev,
          const context &Ctxt, usm::alloc Kind);
template <typename T>
T *malloc(size_t Count, const queue &Q, usm::alloc Kind);
template <typename T>
T *aligned_alloc(size_t Alignment, size_t Count,
                 const device &Dev, const context &Ctxt,
                 usm::alloc Kind);
template <typename T>
T *aligned_alloc(size_t Alignment, size_t Count,
                 const queue &Q, usm::alloc Kind);
