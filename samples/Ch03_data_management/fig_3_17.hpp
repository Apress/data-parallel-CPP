// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snippets that are not set up to be compiled as is.

class handler {
  ...
      // Specifies event(s) that must be complete before the
      // action defined in this command group executes.
      void depends_on({event / std::vector<event> & });

  // Enqueues a memcpy from Src to Dest.
  // Count bytes are copied.
  void memcpy(void* Dest, const void* Src, size_t Count);

  // Enqueues a memcpy from Src to Dest.
  // Count elements are copied.
  template <typename T>
  void copy(const T* Src, T* Dest, size_t Count);

  // Enqueues a memset operation on the specified pointer.
  // Writes the first byte of Value into Count bytes.
  void memset(void* Ptr, int Value, size_t Count)

      // Enques a fill operation on the specified pointer.
      // Fills Pattern into Ptr Count times.
      template <typename T>
      void fill(void* Ptr, const T& Pattern, size_t Count);

  // Submits a kernel of one work-item for execution.
  template <typename KernelName, typename KernelType>
  void single_task(KernelType KernelFunc);

  // Submits a kernel with NumWork-items work-items for
  // execution.
  template <typename KernelName, typename KernelType,
            int Dims>
  void parallel_for(range<Dims> NumWork - items,
                    KernelType KernelFunc);

  // Submits a kernel for execution over the supplied
  // nd_range.
  template <typename KernelName, typename KernelType,
            int Dims>
  void parallel_for(nd_range<Dims> ExecutionRange,
                    KernelType KernelFunc);
  ...
};
