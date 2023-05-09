// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

h.parallel_for(range(1024),[=](id<1> k) {
    z[k] = x[k] + y[k];
});
