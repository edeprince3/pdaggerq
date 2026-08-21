//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: line.cc
// Copyright (C) 2020 A. Eugene DePrince III
//
// Author: A. Eugene DePrince III <adeprince@fsu.edu>
// Maintainer: DePrince group
//
// This file is part of the pdaggerq package.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

#include "../include/line.hpp"

// Single out-of-line definition of Line::subscript_map_ (declared thread_local in
// line.hpp, not inline -- see the comment there for why).
namespace pdaggerq {
    thread_local std::unordered_map<std::string, char> Line::subscript_map_{};
}
