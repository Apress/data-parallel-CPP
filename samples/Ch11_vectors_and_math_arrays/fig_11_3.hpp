// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snippets that are not set up to be compiled as is.

template <access::address_space AddressSpace, access::decorated IsDecorated> 
  void load(size_t offset, multi_ptr<DataT, AddressSpace, IsDecorated> ptr);

template <access::address_space addressSpace, access::decorated IsDecorated>
  void store(size_t offset, multi_ptr<DataT, AddressSpace, IsDecorated> ptr) const;
