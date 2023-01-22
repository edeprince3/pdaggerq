//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_utils.h
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

#ifndef PQ_ADD_LABEL_RANGES_H
#define PQ_ADD_LABEL_RANGES_H

#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include<algorithm>
#include<cstring>
#include<math.h>
#include<sstream>

#include "pq_tensor.h"
#include "pq_string.h"

namespace pdaggerq {

/// expand sums to account for different orbital ranges and zero terms where appropriate
void add_label_ranges(std::shared_ptr<pq_string> in, std::vector<std::shared_ptr<pq_string> > &range_blocked, std::map<std::string, std::vector<std::string> > label_ranges);

/// add label ranges to a string
bool add_ranges_to_string(std::shared_ptr<pq_string> in, std::vector<std::shared_ptr<pq_string> > &list);

/// do ranges in two strings differ?
bool do_ranges_differ(std::string portion, std::string range, std::vector<std::string> in1, std::vector<std::string> in2);

// reorder two ranges ... only one case to consider: ba -> ab
void reorder_two_ranges(tensor & tens, int i1, int i2, int & sign);

/// reorder three ranges ... cases to consider: aba/baa -> aab; bba/bab -> abb
void reorder_three_ranges(amplitudes & amps, int i1, int i2, int i3, int & sign);

/// reorder four label ranges ... cases to consider: aaba/abaa/baaa -> aaab; baab/abba/baba/bbaa/abab -> aabb; babb/bbab/bbba -> abbb
void reorder_four_ranges(amplitudes & amps, int i1, int i2, int i3, int i4, int & sign);

}

#endif 
