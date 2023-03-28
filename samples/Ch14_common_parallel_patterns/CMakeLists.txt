# Copyright (C) 2020 Intel Corporation

# SPDX-License-Identifier: MIT

add_book_sample(
    TEST
    TARGET fig_14_8_one_reduction
    SOURCES fig_14_8_one_reduction.cpp)

add_book_sample(
    TEST
    TARGET fig_14_11_array_reduction
    SOURCES fig_14_11_array_reduction.cpp)

add_book_sample(
    TEST
    TARGET fig_14_12_user_defined_reduction
    SOURCES fig_14_12_user_defined_reduction.cpp)

add_book_sample(
    TEST
    TARGET fig_14_13_algorithm_comparison
    SOURCES fig_14_13_algorithm_comparison.cpp)

add_book_sample(
    TEST
    TARGET fig_14_15_map
    SOURCES fig_14_15_map.cpp
    ADDITIONAL_COMPILE_OPTIONS -fno-fast-math)

add_book_sample(
    TEST
    TARGET fig_14_16_stencil
    SOURCES fig_14_16_stencil.cpp)

add_book_sample(
    TEST
    TARGET fig_14_17_local_stencil
    SOURCES fig_14_17_local_stencil.cpp)

add_book_sample(
    TEST
    TARGET fig_14_18_basic_reduction
    SOURCES fig_14_18_basic_reduction.cpp)

add_book_sample(
    TEST
    TARGET fig_14_19_nd_range_reduction
    SOURCES fig_14_19_nd_range_reduction.cpp)

add_book_sample(
    TEST
    TARGET fig_14_20-22_inclusive_scan
    SOURCES fig_14_20-22_inclusive_scan.cpp)

add_book_sample(
    TEST
    TARGET fig_14_24_local_pack
    SOURCES fig_14_24_local_pack.cpp)

add_book_sample(
    TEST
    TARGET fig_14_26_local_unpack
    SOURCES fig_14_26_local_unpack.cpp)
