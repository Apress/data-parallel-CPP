// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

template <typename T,
          memory_order DefaultOrder,
          memory_scope DefaultScope,           
          access::address_space AddressSpace>
class atomic_ref {
 public:
  using value_type = T;
  static constexpr size_t required_alignment =
    /* implementation-defined */;
  static constexpr bool is_always_lock_free =
    /* implementation-defined */;
  static constexpr memory_order default_read_order =
    memory_order_traits<DefaultOrder>::read_order;
  static constexpr memory_order default_write_order =
    memory_order_traits<DefaultOrder>::write_order;
  static constexpr memory_order default_read_modify_write_order =
    DefaultOrder;
  static constexpr memory_scope default_scope = DefaultScope;

  explicit atomic_ref(T& obj);
  atomic_ref(const atomic_ref& ref) noexcept;
};
