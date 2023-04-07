// Copyright (C) 2020 Intel Corporation

// SPDX-License-Identifier: MIT

#include <array>
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

// BEGIN CODE SNIP
class AddWithAttribute {
 public:
  AddWithAttribute(accessor<int> acc) : data_acc(acc) {}
  [[sycl::reqd_work_group_size(8)]] void operator()(
      id<1> i) const {
    data_acc[i] = data_acc[i] + 1;
  }

 private:
  accessor<int> data_acc;
};

class MulWithAttribute {
 public:
  MulWithAttribute(accessor<int> acc) : data_acc(acc) {}
  void operator()
      [[sycl::reqd_work_group_size(8)]] (id<1> i) const {
    data_acc[i] = data_acc[i] * 2;
  }

 private:
  accessor<int> data_acc;
};
// END CODE SNIP

int main() {
  constexpr size_t size = 16;
  std::array<int, size> data;

  for (int i = 0; i < size; i++) {
    data[i] = i;
  }

  {
    buffer data_buf{data};

    queue Q;
    std::cout << "Running on device: "
              << Q.get_device().get_info<info::device::name>()
              << "\n";

    Q.submit([&](handler& h) {
      accessor data_acc{data_buf, h};
      h.parallel_for(nd_range{{size}, {8}},
                     AddWithAttribute(data_acc));
    });

    Q.submit([&](handler& h) {
      accessor data_acc{data_buf, h};
      h.parallel_for(nd_range{{size}, {8}},
                     MulWithAttribute(data_acc));
    });
  }

  for (int i = 0; i < size; i++) {
    if (data[i] != (i + 1) * 2) {
      std::cout << "Results did not validate at index " << i
                << "!\n";
      return -1;
    }
  }

  std::cout << "Success!\n";
  return 0;
}
