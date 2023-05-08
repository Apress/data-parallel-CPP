#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

int main() {
  queue MyQ;

  constexpr int size = 16;
  std::array<double, size> data;

  // Using "sycl::device_has()" as an attribute does not
  // affect the device we select. Therefore, our host code
  // should check the device's aspects before submitting a
  // kernel which does require that attribute.
  if (MyQ.get_device().has(aspect::fp64)) {
    buffer B{data};
    MyQ.submit([&](handler& h) {
      accessor A{B, h};
      // the attributes here say that the kernel is allowed
      // to require fp64 support any attribute(s) from
      // Figure 12-3 could be specified note that namespace
      // stmt above (for C++) does not affect attributes (a
      // C++ quirk) so sycl:: is needed here
      h.parallel_for(
          size, [=](auto& idx)
                    [[sycl::device_has(aspect::fp64)]] {
                      A[idx] = idx * 2.0;
                    });
    });
    std::cout << "doubles were used\n";
  } else {
    // here we use an alternate method (not needing double
    // math support on the device) to help our code be
    // flexible and hence more portable
    std::array<float, size> fdata;
    {
      buffer B{fdata};
      MyQ.submit([&](handler& h) {
        accessor A{B, h};
        h.parallel_for(
            size, [=](auto& idx) { A[idx] = idx * 2.0f; });
      });
    }

    for (int i = 0; i < size; i++) data[i] = fdata[i];

    std::cout << "no doubles used\n";
  }
  for (int i = 0; i < size; i++)
    std::cout << "data[" << i << "] = " << data[i] << "\n";
  return 0;
}
