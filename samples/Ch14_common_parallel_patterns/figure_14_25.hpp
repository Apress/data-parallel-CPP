// Copyright (C) 2023 Intel Corporation

// SPDX-License-Identifier: MIT

// These ".hpp" files are text from the book that are
// snipets that are not set up to be compiled as is.

uint32_t index =
    exclusive_scan(sg, (uint32_t)predicate, plus<>());
return (predicate) ? new_value[index] : original_value;
