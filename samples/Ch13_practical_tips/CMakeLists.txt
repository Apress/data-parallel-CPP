# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: MIT

add_book_sample(
    TEST
    TARGET fig_13_4_stream
    SOURCES fig_13_4_stream.cpp)

add_book_sample(
    TEST
    TARGET fig_13_6_queue_profiling_timing
    SOURCES fig_13_6_queue_profiling_timing.cpp)

add_book_sample(
    TEST
    TARGET fig_13_9_common_buffer_pattern
    SOURCES fig_13_9_common_buffer_pattern.cpp)

add_book_sample(
    TEST
    TARGET fig_13_10_common_pattern_bug
    SOURCES fig_13_10_common_pattern_bug.cpp)

add_book_sample(
    TEST
    TARGET fig_13_11_host_accessor
    SOURCES fig_13_11_host_accessor.cpp)

add_book_sample(
    TEST
    TARGET fig_13_12_host_accessor_for_init
    SOURCES fig_13_12_host_accessor_for_init.cpp)

add_book_sample(
    TARGET fig_13_13_host_accessor_deadlock
    SOURCES fig_13_13_host_accessor_deadlock.cpp)

