// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <array>
#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

#define TEMPORARY_FIX

// BEGIN CODE SNIP
class AddWithAttribute {
 public:
  AddWithAttribute(accessor<int> acc) : data_acc(acc) {}
#ifdef TEMPORARY_FIX
  // TEMPORARY FIX: for a bug with a 1D
  // reqd_work_group_size.
  [[sycl::reqd_work_group_size(1, 1, 8)]] void operator()(
      id<3> _i) const {
    auto i = _i[2];
    data_acc[i] = data_acc[i] + 1;
  }
#else
  [[sycl::reqd_work_group_size(8)]] void operator()(
      id<1> i) const {
    data_acc[i] = data_acc[i] + 1;
  }
#endif

 private:
  accessor<int> data_acc;
};

class MulWithAttribute {
 public:
  MulWithAttribute(accessor<int> acc) : data_acc(acc) {}
#ifdef TEMPORARY_FIX
  // TEMPORARY FIX: for a bug with a 1D
  // reqd_work_group_size.
  void operator()
      [[sycl::reqd_work_group_size(1, 1,
                                   8)]] (id<3> _i) const {
    auto i = _i[2];
    data_acc[i] = data_acc[i] * 2;
  }
#else
  void operator()
      [[sycl::reqd_work_group_size(8)]] (id<1> i) const {
    data_acc[i] = data_acc[i] * 2;
  }
#endif

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
#ifdef TEMPORARY_FIX
      h.parallel_for(nd_range<3>{{1, 1, size}, {1, 1, 8}},
                     AddWithAttribute(data_acc));
    });
#else
      h.parallel_for(nd_range{{size}, {8}},
                     AddWithAttribute(data_acc));
    });
#endif

    q.submit([&](handler& h) {
      accessor data_acc{data_buf, h};
#ifdef TEMPORARY_FIX
      h.parallel_for(nd_range<3>{{1, 1, size}, {1, 1, 8}},
                     MulWithAttribute(data_acc));
    });
#else
      h.parallel_for(nd_range{{size}, {8}},
                     MulWithAttribute(data_acc));
    });
#endif
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
