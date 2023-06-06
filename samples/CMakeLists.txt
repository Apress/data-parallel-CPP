# Copyright (C) 2023 Intel Corporation

# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.10)
project(DPCPPSamples)

function(add_book_sample)

    set(options TEST)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES LIBS ADDITIONAL_COMPILE_OPTIONS)
    cmake_parse_arguments(BOOK_SAMPLE
        "${options}" "${one_value_args}" "${multi_value_args}"
        ${ARGN}
    )

    add_executable(${BOOK_SAMPLE_TARGET} ${BOOK_SAMPLE_SOURCES})

    if(WITHCUDA)
        set(BOOK_SAMPLE_ADDITIONAL_COMPILE_OPTIONS ${BOOK_SAMPLE_ADDITIONAL_COMPILE_OPTIONS} -fsycl-targets=spir64,nvptx64-nvidia-cuda)
        set(BOOK_SAMPLE_LIBS ${BOOK_SAMPLE_LIBS} -fsycl-targets=spir64,nvptx64-nvidia-cuda)
    endif()

    target_compile_options(${BOOK_SAMPLE_TARGET} PRIVATE -fsycl -fsycl-unnamed-lambda -ferror-limit=1 -Wall -Wpedantic ${BOOK_SAMPLE_ADDITIONAL_COMPILE_OPTIONS})

    target_link_libraries(${BOOK_SAMPLE_TARGET} PRIVATE ${BOOK_SAMPLE_LIBS} -fsycl)

    if(CMAKE_CONFIGURATION_TYPES)
        set(BOOK_SAMPLE_CONFIGS ${CMAKE_CONFIGURATION_TYPES})
    else()
        set(BOOK_SAMPLE_CONFIGS ${CMAKE_BUILD_TYPE})
    endif()
    foreach(CONFIG ${BOOK_SAMPLE_CONFIGS})
        install(TARGETS ${BOOK_SAMPLE_TARGET} CONFIGURATIONS ${CONFIG} DESTINATION ${CONFIG})
    endforeach()
    if(BOOK_SAMPLE_TEST)
        add_test(NAME ${BOOK_SAMPLE_TARGET} COMMAND ${BOOK_SAMPLE_TARGET})
    endif()
endfunction()

add_subdirectory( Ch01_introduction )
add_subdirectory( Ch02_where_code_runs )
add_subdirectory( Ch03_data_management )
add_subdirectory( Ch04_expressing_parallelism )
add_subdirectory( Ch05_error_handling )
add_subdirectory( Ch06_unified_shared_memory )
add_subdirectory( Ch07_buffers )
add_subdirectory( Ch08_scheduling_kernels_and_data_movement )
add_subdirectory( Ch09_communication_and_sychronization )
add_subdirectory( Ch10_defining_kernels )
add_subdirectory( Ch11_vectors_and_math_arrays )
add_subdirectory( Ch12_device_information_and_kernel_specialization )
add_subdirectory( Ch13_practical_tips )
add_subdirectory( Ch14_common_parallel_patterns )
add_subdirectory( Ch15_programming_for_gpus )
add_subdirectory( Ch16_programming_for_cpus )
add_subdirectory( Ch17_programming_for_fpgas )
add_subdirectory( Ch18_libraries )
add_subdirectory( Ch19_memory_model_and_atomics )
add_subdirectory( Ch20_backend_interoperability )
add_subdirectory( Ch21_migrating_cuda_code )
#add_subdirectory( Epilogue_future_direction_of_sycl )
