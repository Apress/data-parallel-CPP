// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

template <typename T, usm::alloc AllocKind, size_t Alignment = 0>
class usm_allocator {
public:
  using value_type = T;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;

public:
  template <typename U> struct rebind {
    typedef usm_allocator<U, AllocKind, Alignment> other;
  };

  usm_allocator() = delete;
  usm_allocator(const context& syclContext,
                const device& syclDevice,
                const property_list& propList = {});
  usm_allocator(const queue& syclQueue,
                const property_list& propList = {});
  usm_allocator(const usm_allocator& other);
  usm_allocator(usm_allocator&&) noexcept;
  usm_allocator& operator=(const usm_allocator&);
  usm_allocator& operator=(usm_allocator&&);

  template <class U>
  usm_allocator(usm_allocator<U, AllocKind, Alignment> const&) noexcept;

  /// Allocate memory
  T* allocate(size_t count);

  /// Deallocate memory
  void deallocate(T* Ptr, size_t count);

  /// Equality Comparison
  ///
  /// Allocators only compare equal if they are of the same USM kind, alignment,
  /// context, and device
  template <class U, usm::alloc AllocKindU, size_t AlignmentU>
  friend bool operator==(const usm_allocator<T, AllocKind, Alignment>&,
                         const usm_allocator<U, AllocKindU, AlignmentU>&);

  /// Inequality Comparison
  /// Allocators only compare unequal if they are not of the same USM kind, alignment,
  /// context, or device
  template <class U, usm::alloc AllocKindU, size_t AlignmentU>
  friend bool operator!=(const usm_allocator<T, AllocKind, Alignment>&,
                         const usm_allocator<U, AllocKindU, AlignmentU>&);
};
