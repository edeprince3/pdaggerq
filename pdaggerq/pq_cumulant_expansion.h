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

#ifndef PQ_CUMULANT_EXPANSION_H
#define PQ_CUMULANT_EXPANSION_H

#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include<algorithm>
#include<cstring>
#include<cmath>
#include<sstream>

#include "pq_string.h"

namespace pdaggerq {

/// replace rdms with cumulant expansion, ignoring the n-body cumulant
void cumulant_expansion(std::vector<std::shared_ptr<pq_string> > &ordered, std::vector<int> ignore_cumulant_rdms);

/// expand rdms in an input string using cumulant expansion, ignoring the n-body cumulant
bool expand_rdms(const std::shared_ptr<pq_string>& in, std::vector<std::shared_ptr<pq_string> > &list, int order);

}

#endif 
