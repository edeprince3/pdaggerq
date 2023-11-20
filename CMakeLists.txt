#
# pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
# Filename: CMakeLists.txt
# Copyright (C) 2020 A. Eugene DePrince III
#
# Author: A. Eugene DePrince III <adeprince@fsu.edu>
# Maintainer: DePrince group
#
# This file is part of the pdaggerq package.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

cmake_minimum_required(VERSION 3.11)
project(pdaggerq)
set(CMAKE_CXX_STANDARD 17)

# find python and set variables for:
#     the interpreter, include dirs, libraries, and version
find_package(Python3 REQUIRED)
set(PYTHON_EXECUTABLE   ${Python3_EXECUTABLE})
set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
set(PYTHON_LIBRARIES    ${Python3_LIBRARIES})
set(PYTHON_VERSION      ${Python3_VERSION})

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG        v2.5.0
)

FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

set(PYBIND11_CPP_STANDARD -std=c++17)

# find openmp and set variables
find_package(OpenMP REQUIRED) #TODO: make this optional
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()



# we are adding a private module _pdaggerq so we don't have a naming collision
pybind11_add_module( 
    _pdaggerq
        pdaggerq/pq_tensor.cc
        pdaggerq/pq_string.cc 
        pdaggerq/pq_utils.cc
        pdaggerq/pq_swap_operators.cc
        pdaggerq/pq_add_spin_labels.cc
        pdaggerq/pq_add_label_ranges.cc
        pdaggerq/pq_helper.cc

        pdaggerq/ta_builder/src/vertex.cc
        pdaggerq/ta_builder/src/linkage.cc
        pdaggerq/ta_builder/src/term.cc
        pdaggerq/ta_builder/src/equation.cc
        pdaggerq/ta_builder/src/tabuilder.cc
        pdaggerq/ta_builder/src/consolidate.cc
        pdaggerq/ta_builder/src/term_perm_helper.cc
        pdaggerq/ta_builder/src/timer.cc
        pdaggerq/ta_builder/include/line.hpp
        pdaggerq/ta_builder/include/collections/scaling_map.hpp
        pdaggerq/ta_builder/include/collections/linkage_set.hpp

        )
