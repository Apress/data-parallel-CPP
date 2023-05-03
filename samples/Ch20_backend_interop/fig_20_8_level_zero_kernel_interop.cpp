// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <level_zero/ze_api.h>

#include <iostream>
#include <sycl/ext/intel/experimental/online_compiler.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>
#include <vector>
using namespace sycl;

#define CHECK_CALL(_call)                          \
  do {                                             \
    ze_result_t result = _call;                    \
    if (result != ZE_RESULT_SUCCESS) {             \
      printf("%s returned %u!\n", #_call, result); \
    }                                              \
  } while (0)

std::vector<platform> getLevelZeroPlatforms() {
  std::vector<platform> platforms;
  for (auto& p : platform::get_platforms()) {
    if (p.get_backend() == backend::ext_oneapi_level_zero) {
      platforms.push_back(p);
    }
  }
  return platforms;
}

int main(int argc, char* argv[]) {
  int platformIndex = 0;
  int deviceIndex = 0;

  if (argc > 1) {
    platformIndex = std::stoi(argv[1]);
  }
  if (argc > 2) {
    deviceIndex = std::stoi(argv[2]);
  }
  if (argc <= 1) {
    std::cout << "Run as ./<progname> <Level Zero platform "
                 "index> "
                 "<Level Zero device index>\n";
    std::cout << "Defaulting to the first Level Zero "
                 "platform and "
                 "device.\n";
  }

  std::vector<platform> l0Platforms =
      getLevelZeroPlatforms();
  if (l0Platforms.size() == 0) {
    std::cout << "Could not find any SYCL platforms "
                 "associated with "
                 "a Level Zero backend!\n";
    return 0;
  }
  if (platformIndex >= l0Platforms.size()) {
    std::cout << "Platform index " << platformIndex
              << " exceeds the number of platforms "
                 "associated with a "
                 "Level Zero backend!\n";
    return -1;
  }

  platform p = l0Platforms[platformIndex];
  if (deviceIndex >= p.get_devices().size()) {
    std::cout << "Device index " << deviceIndex
              << " exceeds the number of devices in the "
                 "platform!\n";
  }

  constexpr size_t size = 16;
  std::array<int, size> data;

  for (int i = 0; i < size; i++) {
    data[i] = i;
  }

  device d = p.get_devices()[deviceIndex];
  context c = context{d};
  queue Q{c, d};

  std::cout << "Running on device: "
            << d.get_info<info::device::name>() << "\n";

  {
    buffer data_buf{data};

    // The online compiler is currently in the
    // sycl::ext::intel::experimental namespace.
    using namespace sycl::ext::intel::experimental;

    // BEGIN CODE SNIP
    // Compile OpenCL C kernel source to SPIR-V intermediate
    // representation using the online compiler:
    const char* kernelSource =
        R"CLC(
            kernel void add(global int* data) {
                int index = get_global_id(0);
                data[index] = data[index] + 1;
            }
        )CLC";
    online_compiler<source_language::opencl_c> compiler(d);
    std::vector<byte> spirv =
        compiler.compile(kernelSource);

    // Get the native Level Zero context and device:
    auto l0Context =
        get_native<backend::ext_oneapi_level_zero>(c);
    auto l0Device =
        get_native<backend::ext_oneapi_level_zero>(d);

    // Create a Level Zero kernel using this context:
    ze_module_handle_t l0Module = nullptr;
    ze_module_desc_t moduleDesc = {};
    moduleDesc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
    moduleDesc.format = ZE_MODULE_FORMAT_IL_SPIRV;
    moduleDesc.inputSize = spirv.size();
    moduleDesc.pInputModule = spirv.data();
    CHECK_CALL(zeModuleCreate(l0Context, l0Device,
                              &moduleDesc, &l0Module,
                              nullptr));

    ze_kernel_handle_t l0Kernel = nullptr;
    ze_kernel_desc_t kernelDesc = {};
    kernelDesc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    kernelDesc.pKernelName = "add";
    CHECK_CALL(
        zeKernelCreate(l0Module, &kernelDesc, &l0Kernel));

    // Create a SYCL kernel from the Level Zero kernel:
    auto skb =
        make_kernel_bundle<backend::ext_oneapi_level_zero,
                           bundle_state::executable>(
            {l0Module}, c);
    auto sk = make_kernel<backend::ext_oneapi_level_zero>(
        {skb, l0Kernel}, c);

    // Use the Level Zero kernel with a SYCL queue:
    Q.submit([&](handler& h) {
      accessor data_acc{data_buf, h};

      h.set_args(data_acc);
      h.parallel_for(size, sk);
    });
    // END CODE SNIP

    // Note: We transferred ownership so no additional
    // cleanup is needed.
  }

  for (int i = 0; i < size; i++) {
    if (data[i] != i + 1) {
      std::cout << "Results did not validate at index " << i
                << "!\n";
      return -1;
    }
  }

  std::cout << "Success!\n";
  return 0;
}
