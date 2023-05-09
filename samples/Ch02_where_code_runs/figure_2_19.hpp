// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

class handler { 
public:
  // Specify event(s) that must be complete before the action 
  // defined in this command group executes.
  void depends_on(std::vector<event>& events);

  // Guarantee that the memory object accessed by the accessor
  // is updated on the host after this action executes.
  template <typename AccessorT>
    void update_host(AccessorT acc);

  // Submit a memset operation writing to the specified pointer.
  // Return an event representing this operation.
  event memset(void *ptr, int value, size_t count); 

  // Submit a memcpy operation copying from src to dest.
  // Return an event representing this operation.
  event memcpy(void *dest, const void *src, size_t count);

  // Copy to/from an accessor and host memory.
  // Accessors are required to have appropriate correct permissions.
  // Pointer can be a raw pointer or shared_ptr.
  template <typename SrcAccessorT, typename DestPointerT>
    void copy(SrcAccessorT src, DestPointerT dest);

  template <typename SrcPointerT, typename DestAccessorT>
    void copy(SrcPointerT src, DestAccessorT dest);

  // Copy between accessors.
  // Accessors are required to have appropriate correct permissions.
  template <typename SrcAccessorT, typename DestAccessorT>
    void copy(SrcAccessorT src, DestAccessorT dest);

  // Submit different forms of kernel for execution.
  template <typename KernelName, typename KernelType>
    void single_task(KernelType kernel);

  template <typename KernelName, typename KernelType, int Dims>
    void parallel_for(range<Dims> num_work_items, 
                      KernelType kernel);  

  template <typename KernelName, typename KernelType, int Dims>
    void parallel_for(nd_range<Dims> execution_range, 
                      KernelType kernel);

  template <typename KernelName, typename KernelType, int Dims>
    void parallel_for_work_group(range<Dims> num_groups, 
                                 KernelType kernel);

  template <typename KernelName, typename KernelType, int Dims>
    void parallel_for_work_group(range<Dims> num_groups,
                                 range<Dims> group_size, 
                                 KernelType kernel);
};
