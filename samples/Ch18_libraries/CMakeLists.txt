# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: MIT

if(NOT WITHCUDA)
# TEMPORARILY DISABLE: doesn't work with CUDA backend.
add_book_sample(
    TEST
    TARGET fig_18_1_builtin
    SOURCES fig_18_1_builtin.cpp)
endif()

add_book_sample(
    TEST
    TARGET fig_18_2_swap
    SOURCES fig_18_2_swap.cpp)

if(NOT NODPL)
add_book_sample(
    TEST
    TARGET fig_18_6_std_fill
    SOURCES fig_18_6_std_fill.cpp)
endif()

if(NOT NODPL)
add_book_sample(
    TEST
    TARGET fig_18_7_std_fill_default_policy
    SOURCES fig_18_7_std_fill_default_policy.cpp)
endif()

if(NOT NODPL)
add_book_sample(
    TEST
    TARGET fig_18_8_binary_search
    SOURCES fig_18_8_binary_search.cpp)
endif()

if(NOT NODPL)
add_book_sample(
    TEST
    TARGET fig_18_9_pstl_usm_device
    SOURCES fig_18_9_pstl_usm_device.cpp)
endif()

if(NOT NODPL)
add_book_sample(
    TEST
    TARGET fig_18_10_pstl_usm
    SOURCES fig_18_10_pstl_usm.cpp)
endif()

