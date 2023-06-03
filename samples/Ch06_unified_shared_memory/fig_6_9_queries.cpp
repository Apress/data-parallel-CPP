// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
using namespace sycl;
namespace dinfo = info::device;
constexpr int N = 42;

template <typename T>
void foo(T data, id<1> i) {
  data[i] = N;
}

int main() {
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();
  bool usm_shared = dev.has(aspect::usm_shared_allocations);
  bool usm_device = dev.has(aspect::usm_device_allocations);
  bool use_USM = usm_shared || usm_device;

  if (use_USM) {
    int *data;
    if (usm_shared) {
      data = malloc_shared<int>(N, q);
    } else /* use device allocations */ {
      data = malloc_device<int>(N, q);
    }
    std::cout << "Using USM with "
              << ((get_pointer_type(data, ctxt) ==
                   usm::alloc::shared)
                      ? "shared"
                      : "device")
              << " allocations on "
              << get_pointer_device(data, ctxt)
                     .get_info<dinfo::name>()
              << "\n";
    q.parallel_for(N, [=](id<1> i) { foo(data, i); });
    q.wait();
    free(data, q);
  } else /* use buffers */ {
    buffer<int, 1> data{range{N}};
    q.submit([&](handler &h) {
      accessor a(data, h);
      h.parallel_for(N, [=](id<1> i) { foo(a, i); });
    });
    q.wait();
  }
  return 0;
}
