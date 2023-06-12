// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>

class something_went_wrong {};

int main() {
  std::cout << "Hello\n";

  throw(something_went_wrong{});
}

// Example output:
// Hello
// terminate called after throwing an instance of 'something_went_wrong'
// Aborted
