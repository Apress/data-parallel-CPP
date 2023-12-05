// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <dpct/dpct.hpp>
#include <iostream>
#include <numeric>
#include <sycl/sycl.hpp>
#include <vector>

constexpr size_t size = 1024 * 1024;

// BEGIN CODE SNIP
void Reverse(int *ptr, size_t size,
             const sycl::nd_item<3> &item_ct1,
             int *scratch) {
  auto gid =
      item_ct1.get_group(2) * item_ct1.get_local_range(2) +
      item_ct1.get_local_id(2);
  auto lid = item_ct1.get_local_id(2);

  scratch[lid] = ptr[gid];
  item_ct1.barrier(sycl::access::fence_space::local_space);
  ptr[gid] = scratch[256 - lid - 1];
}
// END CODE SNIP

int main() try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  sycl::queue &q_ct1 = dev_ct1.default_queue();
  std::vector<int> data(size);
  std::iota(data.begin(), data.end(), 0);

  // BEGIN CODE SNIP
  dpct::device_info deviceProp;
  dpct::dev_mgr::instance().get_device(0).get_device_info(
      deviceProp);
  std::cout << "Running on device: "
            << deviceProp.get_name() << "\n";
  // END CODE SNIP

  int *ptr = nullptr;
  ptr = sycl::malloc_device<int>(size, q_ct1);
  q_ct1.memcpy(ptr, data.data(), size * sizeof(int)).wait();
  q_ct1.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<int, 1> scratch_acc_ct1(
        sycl::range<1>(256), cgh);

    auto size_ct1 = size;

    cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, size / 256) *
                              sycl::range<3>(1, 1, 256),
                          sycl::range<3>(1, 1, 256)),
        [=](sycl::nd_item<3> item_ct1) {
          Reverse(ptr, size_ct1, item_ct1,
                  scratch_acc_ct1.get_pointer());
        });
  });
  /*
  DPCT1003:2: Migrated API does not return error code.
  (*,0) is inserted. You may need to rewrite this code.
  */
  dpct::err0 result = (dev_ct1.queues_wait_and_throw(), 0);
  /*
  DPCT1000:1: Error handling if-stmt was detected but could
  not be rewritten.
  */
  if (result != 0) {
    /*
    DPCT1001:0: The statement could not be removed.
    */
    std::cout << "An error occurred!\n";
  }

  q_ct1.memcpy(data.data(), ptr, size * sizeof(int)).wait();

  for (size_t s = 0; s < size; s += 256) {
    for (size_t i = 0; i < 256; i++) {
      auto got = data[s + i];
      auto want = s + 256 - i - 1;
      if (got != want) {
        std::cout << "Mismatch at index " << s + i
                  << ", got " << got << ", wanted " << want
                  << "\n";
        return -1;
      }
    }
  }

  sycl::free(ptr, q_ct1);
  std::cout << "Success.\n";
  return 0;
} catch (sycl::exception const &exc) {
  std::cerr << exc.what()
            << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
