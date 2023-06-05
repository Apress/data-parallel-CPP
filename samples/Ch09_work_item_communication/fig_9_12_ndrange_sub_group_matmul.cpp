// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <chrono>
#include <sycl/sycl.hpp>
using namespace sycl;

extern const int matrixSize = 128;
static const int iterations = 16;

// T is the type of data stored in the matrix
template <typename T>
double run_sycl(const std::vector<T>& vecA,
                const std::vector<T>& vecB,
                std::vector<T>& vecC) {
  using ns = std::chrono::nanoseconds;
  ns::rep best_time = std::numeric_limits<ns::rep>::max();

  const int M = matrixSize;
  const int N = matrixSize;
  const int K = matrixSize;

  std::fill(vecC.begin(), vecC.end(), (T)0);

  buffer<T, 2> bufA{vecA.data(), range<2>{M, K}};
  buffer<T, 2> bufB{vecB.data(), range<2>{K, N}};
  buffer<T, 2> bufC{vecC.data(), range<2>{M, N}};

  queue q;
  std::cout << "Running on device: "
            << q.get_device().get_info<info::device::name>()
            << "\n";

  for (int i = 0; i < iterations; ++i) {
    auto start = std::chrono::steady_clock::now();

    q.submit([&](handler& h) {
      accessor matrixA{bufA, h};
      accessor matrixB{bufB, h};
      accessor matrixC{bufC, h};

      // BEGIN CODE SNIP
      // Note: This example assumes that the sub-group size
      // is greater than or equal to the tile size!
      constexpr int tile_size = 4;
      h.parallel_for(
          nd_range<2>{{M, N}, {1, tile_size}},
          [=](nd_item<2> item) {
            auto sg = item.get_sub_group();

            // Indices in the global index space:
            int m = item.get_global_id()[0];
            int n = item.get_global_id()[1];

            // Index in the local index space:
            int i = item.get_local_id()[1];

            // Template type T is the type of data stored
            // in the matrix
            T sum = 0;
            for (int kk = 0; kk < K; kk += tile_size) {
              // Load the matrix tile from matrix A.
              T tileA = matrixA[m][kk + i];

              // Perform computation by broadcasting from
              // the matrix tile and loading from matrix B
              // in global memory.  The loop variable k
              // describes which work-item in the sub-group
              // to broadcast data from.
              for (int k = 0; k < tile_size; k++) {
                sum += group_broadcast(sg, tileA, k) *
                       matrixB[kk + k][n];
              }
            }

            // Write the final result to global memory.
            matrixC[m][n] = sum;
          });
      // END CODE SNIP
    });
    q.wait();

    auto duration =
        std::chrono::steady_clock::now() - start;
    auto time =
        std::chrono::duration_cast<ns>(duration).count();

    best_time = std::min(time, best_time);
  }

  double best_seconds = (double)best_time / 1e9;

  return best_seconds;
}

template double run_sycl<float>(
    const std::vector<float>& vecA,
    const std::vector<float>& vecB,
    std::vector<float>& vecC);
