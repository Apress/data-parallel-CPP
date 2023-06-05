// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snipets that are not set up to be compiled as is.

constexpr int num_runs = 10;
constexpr size_t scalar = 3;

double triad(const std::vector<double>& vecA,
             const std::vector<double>& vecB,
             std::vector<double>& vecC) {
  assert(vecA.size() == vecB.size() == vecC.size());
  const size_t array_size = vecA.size();
  double min_time_ns = std::numeric_limits<double>::max();

  queue Q{property::queue::enable_profiling{}};
  std::cout << "Running on device: "
            << Q.get_device().get_info<info::device::name>()
            << "\n";

  buffer<double> bufA(vecA);
  buffer<double> bufB(vecB);
  buffer<double> bufC(vecC);

  for (int i = 0; i < num_runs; i++) {
    auto Q_event = Q.submit([&](handler& h) {
      accessor A{bufA, h};
      accessor B{bufB, h};
      accessor C{bufC, h};

      h.parallel_for(array_size, [=](id<1> idx) {
        C[idx] = A[idx] + B[idx] * scalar;
      });
    });

    double exec_time_ns =
        Q_event.get_profiling_info<
            info::event_profiling::command_end>() -
        Q_event.get_profiling_info<
            info::event_profiling::command_start>();

    std::cout << "Execution time (iteration " << i
              << ") [sec]: "
              << (double)exec_time_ns * 1.0E-9 << "\n";
    min_time_ns = std::min(min_time_ns, exec_time_ns);
  }

  return min_time_ns;
}
