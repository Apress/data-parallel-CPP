# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: MIT

add_book_sample(
    TARGET fig_17_9_fpga_selector
    SOURCES fig_17_9_fpga_selector.cpp)

add_book_sample(
    TARGET fig_17_11_fpga_emulator_selector
    SOURCES fig_17_11_fpga_emulator_selector.cpp)

add_book_sample(
    TARGET fig_17_17_ndrange_func
    SOURCES fig_17_17_ndrange_func.cpp)

add_book_sample(
    TARGET fig_17_18_loop_func
    SOURCES fig_17_18_loop_func.cpp)

add_book_sample(
    TARGET fig_17_20_loop_carried_deps
    SOURCES fig_17_20_loop_carried_deps.cpp)

add_book_sample(
    TARGET fig_17_22_loop_carried_state
    SOURCES fig_17_22_loop_carried_state.cpp)

if(NOT WITHCUDA AND NOT WITHROCM)
# TEMPORARILY DISABLE: doesn't work with CUDA or ROCm backend.
add_book_sample(
    TARGET fig_17_31_inter_kernel_pipe
    SOURCES fig_17_31_inter_kernel_pipe.cpp)
endif()
