# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: MIT

add_book_sample(
    TEST
    TARGET fig_20_2_querying_backends
    SOURCES fig_20_2_querying_backends.cpp)

add_book_sample(
    #TEST   # disable temporarily due to bug
    TARGET fig_20_3_opencl_to_sycl
    SOURCES fig_20_3_opencl_to_sycl.cpp
    LIBS OpenCL)

if(NOT NOL0)
add_book_sample(
    #TEST   # disable temporarily due to bug
    TARGET fig_20_4_level_zero_to_sycl
    SOURCES fig_20_4_level_zero_to_sycl.cpp
    LIBS ze_loader)
endif()

add_book_sample(
    TEST
    TARGET fig_20_5_sycl_to_opencl
    SOURCES fig_20_5_sycl_to_opencl.cpp
    LIBS OpenCL)

if(NOT NOL0)
add_book_sample(
    TEST
    TARGET fig_20_6_sycl_to_level_zero
    SOURCES fig_20_6_sycl_to_level_zero.cpp
    LIBS ze_loader)
endif()

add_book_sample(
    TEST
    TARGET fig_20_7_interop_handle_opencl
    SOURCES fig_20_7_interop_handle_opencl.cpp
    LIBS OpenCL)

if(NOT NOl0)
add_book_sample(
    TEST
    TARGET fig_20_8_interop_handle_level_zero
    SOURCES fig_20_8_interop_handle_level_zero.cpp
    LIBS ze_loader)
endif()

add_book_sample(
    TEST
    TARGET fig_20_9_opencl_kernel_interop
    SOURCES fig_20_9_opencl_kernel_interop.cpp
    LIBS OpenCL)

if(NOT NOL0)
add_book_sample(
    TEST
    TARGET fig_20_10_level_zero_kernel_interop
    SOURCES fig_20_10_level_zero_kernel_interop.cpp
    LIBS ze_loader)
endif()

add_book_sample(
    TEST
    TARGET fig_20_11_filter_selector
    SOURCES fig_20_11_filter_selector.cpp)
