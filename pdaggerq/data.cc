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

#include "data.h"
#include "tensor.h"

namespace pdaggerq {

// constructor
StringData::StringData(){
}

// descructor
StringData::~StringData(){
}

// sort amplitude, integral, and delta function labels
void StringData::sort_labels() {

    for (size_t i = 0; i < integral_types.size(); i++) {
        std::string type = integral_types[i];
        for (size_t j = 0; j < ints[type].size(); j++) {
            ints[type][j].sort();
        }
    }
    for (size_t i = 0; i < amplitude_types.size(); i++) {
        char type = amplitude_types[i];
        for (size_t j = 0; j < amps[type].size(); j++) {
            amps[type][j].sort();
        }
    }
    for (size_t i = 0; i < deltas.size(); i++) {
        deltas[i].sort();
    }

}

}
