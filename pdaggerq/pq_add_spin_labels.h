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

#ifndef PQ_ADD_SPIN_LABELS_H
#define PQ_ADD_SPIN_LABELS_H

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

/// add spin labels to a string
bool add_spins(std::shared_ptr<pq_string> in, std::vector<std::shared_ptr<pq_string> > &list);

/// expand sums to include spin and zero terms where appropriate
void spin_blocking(std::shared_ptr<pq_string> in, std::vector<std::shared_ptr<pq_string> > &spin_blocked, std::map<std::string, std::string> spin_map);

/// reorder three spin labels as aab or abb
void reorder_three_spins(amplitudes & amps, int i1, int i2, int i3, int & sign);

/// reorder four spin labels as aaab, aabb, or abbb
void reorder_four_spins(amplitudes & amps, int i1, int i2, int i3, int i4, int & sign);

}

#endif 
