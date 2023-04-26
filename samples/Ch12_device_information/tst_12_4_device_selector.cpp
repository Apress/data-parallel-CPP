// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <sycl/sycl.hpp>
#include <iostream>
using namespace std;

int my_selector(int isgpu, string foo) {
  int score = -1;

  // We prefer non-Martian GPUs, especially ACME GPUs
  if (isgpu) {
    if (foo.find("ACME") != std::string::npos)
      score += 25;

    if (foo.find("Martian") ==
        std::string::npos)
      score += 800;
  }

  // If there is no GPU on the system all devices will be given a negative score
  // and the selector will not select a device. This will cause an exception.
  return score;
}

int main() {
  string foo;
  foo="Intel GPU";
  cout << "NOTGPU" << foo << my_selector(0,foo) << '\n';
  cout << "YESGPU" << foo << my_selector(1,foo) << '\n';
  foo="Intel ACME GPU";
  cout << "NOTGPU" << foo << my_selector(0,foo) << '\n';
  cout << "YESGPU" << foo << my_selector(1,foo) << '\n';
  foo="Intel GPU Martian";
  cout << "NOTGPU" << foo << my_selector(0,foo) << '\n';
  cout << "YESGPU" << foo << my_selector(1,foo) << '\n';
  foo="Intel Martian ACME GPU";
  cout << "NOTGPU" << foo << my_selector(0,foo) << '\n';
  cout << "YESGPU" << foo << my_selector(1,foo) << '\n';
  foo="ACME";
  cout << "NOTGPU" << foo << my_selector(0,foo) << '\n';
  cout << "YESGPU" << foo << my_selector(1,foo) << '\n';
  foo="MartianACME";
  cout << "NOTGPU" << foo << my_selector(0,foo) << '\n';
  cout << "YESGPU" << foo << my_selector(1,foo) << '\n';
  foo="Martian";
  cout << "NOTGPU" << foo << my_selector(0,foo) << '\n';
  cout << "YESGPU" << foo << my_selector(1,foo) << '\n';
  return 0;
}

