//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_helper.cc
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

#ifndef PQ_UTILS_H
#define PQ_UTILS_H

#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include<algorithm>
#include<cstring>
#include<math.h>
#include<sstream>

#include "tensor.h"

namespace pdaggerq {

/// is a label classified as occupied?
bool is_occ(std::string idx);

/// is a label classified as virtual?
bool is_vir(std::string idx);

// how many times does an index appear deltas?
int index_in_deltas(std::string idx, std::vector<delta_functions> deltas);

// how many times does an index appear integrals?
int index_in_integrals(std::string idx, std::vector<integrals> ints);

// how many times does an index appear amplitudes?
int index_in_amplitudes(std::string idx, std::vector<amplitudes> amps);

// how many times does an index appear amplitudes, deltas, and integrals?
int index_in_anywhere(std::shared_ptr<StringData> data, std::string idx);

/// replace one label with another (in delta functions)
void replace_index_in_deltas(std::string old_idx, std::string new_idx, std::vector<delta_functions> &deltas);

/// replace one label with another (in a given set of integrals)
void replace_index_in_integrals(std::string old_idx, std::string new_idx, std::vector<integrals> &ints);

/// replace one label with another (in a given set of amplitudes)
void replace_index_in_amplitudes(std::string old_idx, std::string new_idx, std::vector<amplitudes> &amps);

/// concatinate a list of operators (a list of strings) into a single list
std::vector<std::string> concatinate_operators(std::vector<std::vector<std::string>> ops);

/// remove "*" from std::string
void removeStar(std::string &x);

/// remove "(" and ")" from std::string
void removeParentheses(std::string &x);

}

#endif 
