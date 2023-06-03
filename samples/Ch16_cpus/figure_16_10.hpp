// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snipets that are not set up to be compiled as is.

template <typename T>
void init(queue& deviceQueue, T* VA, T* VB, T* VC,
          size_t array_size) {
  range<1> numOfItems{array_size};

  buffer<T, 1> bufferA(VA, numOfItems);
  buffer<T, 1> bufferB(VB, numOfItems);
  buffer<T, 1> bufferC(VC, numOfItems);

  auto queue_event = deviceQueue.submit([&](handler& cgh) {
    auto aA = bufA.template get_access<sycl_write>(cgh);
    auto aB = bufB.template get_access<sycl_write>(cgh);
    auto aC = bufC.template get_access<sycl_write>(cgh);

    cgh.parallel_for<class Init<T>>(numOfItems, [=](id<1> wi) {
      aA[wi] = 2.0;
      aB[wi] = 1.0;
      aC[wi] = 0.0;
    });
  });

  queue_event.wait();
}
