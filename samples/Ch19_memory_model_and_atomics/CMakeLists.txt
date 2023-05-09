# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: MIT

add_book_sample(
    TARGET fig_19_3_data_race
    SOURCES fig_19_3_data_race.cpp)

add_book_sample(
    TEST
    TARGET fig_19_6_avoid_data_race_with_barrier
    SOURCES fig_19_6_avoid_data_race_with_barrier.cpp)

add_book_sample(
    TEST
    TARGET fig_19_7_avoid_data_race_with_atomics
    SOURCES fig_19_7_avoid_data_race_with_atomics.cpp)

add_book_sample(
    TEST
    TARGET fig_19_15_buffer_and_atomic_ref
    SOURCES fig_19_15_buffer_and_atomic_ref.cpp)

add_book_sample(
    TEST
    TARGET fig_19_16_usm_and_atomic_ref
    SOURCES fig_19_16_usm_and_atomic_ref.cpp)

add_book_sample(
    TEST
    TARGET fig_19_17_histogram
    SOURCES fig_19_17_histogram.cpp)

add_book_sample(
    TEST
    TARGET fig_19_18-19_device_latch
    SOURCES fig_19_18-19_device_latch.cpp)
