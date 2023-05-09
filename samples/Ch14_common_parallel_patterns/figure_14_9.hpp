// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are snipets
// that are not set up to be compiled as is.

template <typename BufferT, typename BinaryOperation>
unspecified reduction(BufferT variable, handler& h,
                      BinaryOperation combiner,
                      const property_list& properties = {});

template <typename BufferT, typename BinaryOperation>
unspecified reduction(BufferT variable, handler& h,
                      const BufferT::value_type& identity, BinaryOperation combiner,
                      const property_list& properties = {});
          

template <typename T, typename BinaryOperation>
unspecified reduction(T* variable,
                      BinaryOperation combiner,
                      const property_list& properties = {});

template <typename T, typename BinaryOperation>
unspecified reduction(T* variable,
                      const T& identity, BinaryOperation combiner,
                      const property_list& properties = {});                          


template <typename T, typename Extent, typename BinaryOperation>
unspecified reduction(span<T, Extent> variables,
                      BinaryOperation combiner,
                      const property_list& properties = {});

template <typename T, typename Extent, typename BinaryOperation>
unspecified reduction(span<T, Extent> variables,
                      const T& identity, BinaryOperation combiner,
                      const property_list& properties = {});
