// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snippets that are not set up to be compiled as is.

// Named Functions
void *malloc_device(size_t size, const device &dev,
                    const context &ctxt);
void *malloc_device(size_t size, const queue &q);
void *aligned_alloc_device(size_t alignment, size_t size,
                           const device &dev,
                           const context &ctxt);
void *aligned_alloc_device(size_t alignment, size_t size,
                           const queue &q);

void *malloc_host(size_t size, const context &ctxt);
void *malloc_host(size_t size, const queue &q);
void *aligned_alloc_host(size_t alignment, size_t size,
                         const context &ctxt);
void *aligned_alloc_host(size_t alignment, size_t size,
                         const queue &q);

void *malloc_shared(size_t size, const device &dev,
                    const context &ctxt);
void *malloc_shared(size_t size, const queue &q);
void *aligned_alloc_shared(size_t alignment, size_t size,
                           const device &dev,
                           const context &ctxt);
void *aligned_alloc_shared(size_t alignment, size_t size,
                           const queue &q);

// Single Function
void *malloc(size_t size, const device &dev,
             const context &ctxt, usm::alloc kind);
void *malloc(size_t size, const queue &q, usm::alloc kind);
void *aligned_alloc(size_t alignment, size_t size,
                    const device &dev, const context &ctxt,
                    usm::alloc kind);
void *aligned_alloc(size_t alignment, size_t size,
                    const queue &q, usm::alloc kind);
