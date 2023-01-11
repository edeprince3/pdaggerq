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

#ifndef PQ_EXTRAS_H
#define PQ_EXTRAS_H

#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include<algorithm>
#include<cstring>
#include<math.h>
#include<sstream>

#include "tensor.h"
#include "data.h"

namespace pdaggerq {

/// consolidate terms that differ by three summed labels plus permutations
void consolidate_permutations_plus_three_swaps(
    std::vector<std::shared_ptr<StringData> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2,
    std::vector<std::string> labels_3);

/// consolidate terms that differ by four summed labels plus permutations
void consolidate_permutations_plus_four_swaps(
    std::vector<std::shared_ptr<StringData> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2,
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4);

/// consolidate terms that differ by five summed labels plus permutations
void consolidate_permutations_plus_five_swaps(
    std::vector<std::shared_ptr<StringData> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2,
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4,
    std::vector<std::string> labels_5);

/// consolidate terms that differ by six summed labels plus permutations
void consolidate_permutations_plus_six_swaps(
    std::vector<std::shared_ptr<StringData> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2,
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4,
    std::vector<std::string> labels_5,
    std::vector<std::string> labels_6);

/// consolidate terms that differ by seven summed labels plus permutations
void consolidate_permutations_plus_seven_swaps(
    std::vector<std::shared_ptr<StringData> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2,
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4,
    std::vector<std::string> labels_5,
    std::vector<std::string> labels_6,
    std::vector<std::string> labels_7);

/// consolidate terms that differ by eight summed labels plus permutations
void consolidate_permutations_plus_eight_swaps(
    std::vector<std::shared_ptr<StringData> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2,
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4,
    std::vector<std::string> labels_5,
    std::vector<std::string> labels_6,
    std::vector<std::string> labels_7,
    std::vector<std::string> labels_8);

}

#endif 
