# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: MIT

add_book_sample(
    TEST
    TARGET fig_21_1_basicsycl
    SOURCES fig_21_1_basicsycl.cpp)

add_book_sample(
    TEST
    TARGET fig_21_5_walkorder
    SOURCES fig_21_5_walkorder.cpp)

add_book_sample(
    TARGET fig_21_7_possible_deadlock
    SOURCES fig_21_7_possible_deadlock.cpp)

add_book_sample(
    TEST
    TARGET fig_21_8_barriers
    SOURCES fig_21_8_barriers.cpp)

add_book_sample(
    TEST
    TARGET fig_21_9_atomics
    SOURCES fig_21_9_atomics.cpp)

if(NOT NODPCT)
add_book_sample(
    TEST
    TARGET fig_21_13-14_reverse_migrated
    SOURCES fig_21_13-14_reverse_migrated.cpp)
endif()

# Note: we do not currently build these pure CUDA samples:
# fig_21_2_basiccuda.cu
# fig_21_4-6_walkorder.cu
# fig_21_10_reverse.cu
