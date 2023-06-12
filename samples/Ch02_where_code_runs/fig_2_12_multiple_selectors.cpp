// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <string>
#include <sycl/ext/intel/fpga_extensions.hpp>  // For fpga_selector_v
#include <sycl/sycl.hpp>
using namespace sycl;

void output_dev_info(const device& dev,
                     const std::string& selector_name) {
  std::cout << selector_name << ": Selected device: "
            << dev.get_info<info::device::name>() << "\n";
  std::cout << "                  -> Device vendor: "
            << dev.get_info<info::device::vendor>() << "\n";
}

int main() {
  output_dev_info(device{default_selector_v},
                  "default_selector_v");
  output_dev_info(device{cpu_selector_v}, "cpu_selector_v");
  output_dev_info(device{gpu_selector_v}, "gpu_selector_v");
  output_dev_info(device{accelerator_selector_v},
                  "accelerator_selector_v");
  output_dev_info(device{ext::intel::fpga_selector_v},
                  "fpga_selector_v");

  return 0;
}

// Example Output:
// default_selector_v: Selected device: Intel(R) UHD Graphics [0x9a60]
//   -> Device vendor: Intel(R) Corporation
//   cpu_selector_v: Selected device: 11th Gen Intel(R) Core(TM) i9-11900KB @ 3.30GHz
//   -> Device vendor: Intel(R) Corporation
//   gpu_selector_v: Selected device: Intel(R) UHD Graphics [0x9a60]
//   -> Device vendor: Intel(R) Corporation
//   accelerator_selector_v: Selected device: Intel(R) FPGA Emulation Device
//   -> Device vendor: Intel(R) Corporation
//   fpga_selector_v: Selected device: pac_a10 : Intel PAC Platform (pac_ee00000)
// -> Device vendor: Intel Corp
