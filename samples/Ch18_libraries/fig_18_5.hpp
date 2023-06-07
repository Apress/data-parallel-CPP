// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snippets that are not set up to be compiled as is.

auto policy_b = device_policy<parallel_unsequenced_policy,
                              class PolicyB>{
    sycl::device{sycl::gpu_selector{}}};
std::for_each(policy_b, …);
auto policy_c =
    device_policy<parallel_unsequenced_policy,
                  class PolicyС>{sycl::default_selector{}};
std::for_each(policy_c, …);
auto policy_d =
    make_device_policy<class PolicyD>(default_policy);
std::for_each(policy_d, …);
auto policy_e =
    make_device_policy<class PolicyE>(sycl::queue{});
std::for_each(policy_e, …);
