// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

cgh.parallel_for<class aos<T>>(numOfItems,[=](id<1> wi) {
    x[wi] = a[wi].x;  // lead to gather x0, x1, x2, x3
    y[wi] = a[wi].y;  // lead to gather y0, y1, y2, y3
    z[wi] = a[wi].z;  // lead to gather z0, z1, z2, z3 
    w[wi] = a[wi].w;  // lead to gather w0, w1, w2, w3
});
