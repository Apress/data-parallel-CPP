# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: MIT

if(NOT WITHCUDA)
# TEMPORARILY DISABLE: doesn't work with CUDA backend.
add_book_sample(
    TEST
    TARGET fig_8_3_linear_dependence_in_order
    SOURCES fig_8_3_linear_dependence_in_order.cpp)
endif()

if(NOT WITHCUDA)
# TEMPORARILY DISABLE: doesn't work with CUDA backend.
add_book_sample(
    TEST
    TARGET fig_8_4_linear_dependence_events
    SOURCES fig_8_4_linear_dependence_events.cpp)
endif()

add_book_sample(
    TEST
    TARGET fig_8_5_linear_dependence_buffers
    SOURCES fig_8_5_linear_dependence_buffers.cpp)

if(NOT WITHCUDA)
# TEMPORARILY DISABLE: doesn't work with CUDA backend.
add_book_sample(
    TEST
    TARGET fig_8_6_y_in_order
    SOURCES fig_8_6_y_in_order.cpp)
endif()

if(NOT WITHCUDA)
# TEMPORARILY DISABLE: doesn't work with CUDA backend.
add_book_sample(
    TEST
    TARGET fig_8_7_y_events
    SOURCES fig_8_7_y_events.cpp)
endif()

add_book_sample(
    TEST
    TARGET fig_8_8_y_buffers
    SOURCES fig_8_8_y_buffers.cpp)


