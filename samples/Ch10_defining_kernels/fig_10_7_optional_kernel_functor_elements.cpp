// Copyright (C) 2023 Intel Corporation

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
      nd_item<1> item) const {
    auto i = item.get_global_id();
    data_acc[i] = data_acc[i] + 1;
  }

 private:
  accessor<int> data_acc;
};

class MulWithAttribute {
 public:
  MulWithAttribute(accessor<int> acc) : data_acc(acc) {}
  void operator() [[sycl::reqd_work_group_size(8)]] (
      nd_item<1> item) const {
    auto i = item.get_global_id();
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

    queue q;
    std::cout
        << "Running on device: "
        << q.get_device().get_info<info::device::name>()
        << "\n";

    q.submit([&](handler& h) {
      accessor data_acc{data_buf, h};
      h.parallel_for(nd_range{{size}, {8}},
                     AddWithAttribute(data_acc));
    });

    q.submit([&](handler& h) {
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
