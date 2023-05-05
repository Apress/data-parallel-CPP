// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

#include <iostream>
#include <sycl/sycl.hpp>
using namespace sycl;

template <typename queryT, typename T>
void do_query(const T& obj_to_query,
              const std::string& name, int indent = 4) {
  std::cout << std::string(indent, ' ') << name << " is '"
            << obj_to_query.template get_info<queryT>()
            << "'\n";
}

int main() {
  // Loop through the available platforms
  for (auto const& this_platform :
       platform::get_platforms()) {
    std::cout << "Found Platform:\n";
    do_query<info::platform::name>(this_platform,
                                   "info::platform::name");
    do_query<info::platform::vendor>(
        this_platform, "info::platform::vendor");
    do_query<info::platform::version>(
        this_platform, "info::platform::version");
    do_query<info::platform::profile>(
        this_platform, "info::platform::profile");

    // Loop through the devices available in this plaform
    for (auto& dev : this_platform.get_devices()) {
      std::cout << "  Device: "
                << dev.get_info<info::device::name>()
                << "\n";
      // is_cpu() == has(aspect::cpu)
      std::cout << "    is_cpu(): "
                << (dev.is_cpu() ? "Yes" : "No") << "\n";
      // is_cpu() == has(aspect::gpu)
      std::cout << "    is_gpu(): "
                << (dev.is_gpu() ? "Yes" : "No") << "\n";
      // is_cpu() == has(aspect::accelerator)
      std::cout << "    is_accelerator(): "
                << (dev.is_accelerator() ? "Yes" : "No")
                << "\n";

      std::cout << "    has(emulated): "
                << (dev.has(aspect::emulated)) ? "Yes" : "No") << "\n";
      std::cout << "    has(fp16): "
                << (dev.has(aspect::fp16)) ? "Yes" : "No") << "\n";
      std::cout << "    has(fp64): "
                << (dev.has(aspect::fp64)) ? "Yes" : "No") << "\n";
      std::cout << "    has(host_debuggable): "
                << (dev.has(aspect::host_debuggable)) ? "Yes" : "No") << "\n";
      std::cout << "    has(atomic64): "
                << (dev.has(aspect::atomic64)) ? "Yes" : "No") << "\n";
      // see Chapter 13
      std::cout << "    has(queue_profiling): "
                << (dev.has(aspect::queue_profiling)) ? "Yes" : "No") << "\n";
      std::cout << "    has(emulated): "
                << (dev.has(aspect::emulated)) ? "Yes" : "No") << "\n";
      std::cout << "    has(usm_device_allocations): "
                << (dev.has(aspect::usm_device_allocations)) ? "Yes" : "No") << "\n";
      std::cout << "    has(usm_host_allocations): "
                << (dev.has(aspect::usm_host_allocations)) ? "Yes" : "No") << "\n";
      std::cout << "    has(usm_atomic_host_allocations): "
                << (dev.has(aspect::usm_atomic_host_allocations)) ? "Yes" : "No") << "\n";
      std::cout << "    has(usm_shared_allocations): "
                << (dev.has(aspect::usm_shared_allocations)) ? "Yes" : "No") << "\n";
      std::cout << "    has(usm_atomic_shared_allocations): "
                << (dev.has(aspect::usm_atomic_shared_allocations)) ? "Yes" : "No") << "\n";
      std::cout << "    has(usm_system_allocations): "
                << (dev.has(aspect::usm_system_allocations)) ? "Yes" : "No") << "\n";

      do_query<info::device::vendor>(
          dev, "info::device::vendor");
      do_query<info::device::driver_version>(
          dev, "info::device::driver_version");
      do_query<info::device::max_work_item_dimensions>(
          dev, "info::device::max_work_item_dimensions");
      do_query<info::device::max_work_group_size>(
          dev, "info::device::max_work_group_size");
      do_query<info::device::mem_base_addr_align>(
          dev, "info::device::mem_base_addr_align");
      do_query<info::device::partition_max_sub_devices>(
          dev, "info::device::partition_max_sub_devices");

      std::cout << "    Many more queries are available "
                   "than shown here!\n";
    }
    std::cout << "\n";
  }
  return 0;
}
