// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snippets that are not set up to be compiled as is.

class queue {
 public:
  // Submit a memset operation writing to the specified
  // pointer. Return an event representing this operation.
  event memset(void* ptr, int value, size_t count);

  // Submit a memcpy operation copying from src to dest.
  // Return an event representing this operation.
  event memcpy(void* dest, const void* src, size_t count);

  // Submit different forms of kernel for execution.
  // Return an event representing the kernel operation.
  template <typename KernelName, typename KernelType>
  event single_task(KernelType kernel);

  template <typename KernelName, typename KernelType,
            int Dims>
  event parallel_for(range<Dims> num_work_items,
                     KernelType kernel);

  template <typename KernelName, typename KernelType,
            int Dims>
  event parallel_for(nd_range<Dims> execution_range,
                     KernelType kernel);

  // Submit different forms of kernel for execution.
  // Wait for the specified event(s) to complete
  // before executing the kernel.
  // Return an event representing the kernel operation.
  template <typename KernelName, typename KernelType>
  event single_task(const std::vector<event>& events,
                    KernelType kernel);

  template <typename KernelName, typename KernelType,
            int Dims>
  event parallel_for(range<Dims> num_work_items,
                     const std::vector<event>& events,
                     KernelType kernel);

  template <typename KernelName, typename KernelType,
            int Dims>
  event parallel_for(nd_range<Dims> execution_range,
                     const std::vector<event>& events,
                     KernelType kernel);
};
