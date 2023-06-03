// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snipets that are not set up to be compiled as is.

auto fpga_policy_a =
    fpga_device_policy<class FPGAPolicyA>{};

auto fpga_policy_b =
    make_fpga_policy(queue{intel::fpga_selector{}});

constexpr auto unroll_factor = 8;
auto fpga_policy_c =
    make_fpga_policy<class FPGAPolicyC, unroll_factor>(
        fpga_policy);
