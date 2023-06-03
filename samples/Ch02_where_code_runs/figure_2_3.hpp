// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snipets that are not set up to be compiled as is.

class queue {
 public:
  // Create a queue associated with a default
  // (implementation chosen) device.
  queue(const property_list & = {});

  queue(const async_handler &, const property_list & = {});

  // Create a queue using a DeviceSelector.  A
  // DeviceSelector is a callable that ranks devices
  // numerically.  There are a few SYCL-defined device
  // selectors available such as cpu_selector_v and
  // gpu_selector_v.
  template <typename DeviceSelector>
  explicit queue(const DeviceSelector &deviceSelector,
                 const property_list &propList = {});

  // Create a queue associated with an explicit device to
  // which the program already holds a reference.
  queue(const device &, const property_list & = {});

  // Create a queue associated with a device in a specific
  // SYCL context A device selector may be used in place of
  // a device.
  queue(const context &, const device &,
        const property_list & = {});
};
