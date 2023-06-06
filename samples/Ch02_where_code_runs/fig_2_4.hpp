// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snippets that are not set up to be compiled as is.

class queue {
 public:
  // Submit a command group to this queue.
  // The command group may be a lambda expression or
  // function object. Returns an event reflecting the status
  // of the action performed in the command group.
  template <typename T>
  event submit(T);

  // Wait for all previously submitted actions to finish
  // executing.
  void wait();

  // Wait for all previously submitted actions to finish
  // executing. Pass asynchronous exceptions to an
  // async_handler function.
  void wait_and_throw();
};
