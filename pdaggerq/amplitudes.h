//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: data.h
// Copyright (C) 2020 A. Eugene DePrince III
//
// Author: A. Eugene DePrince III <adeprince@fsu.edu>
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
//  limitations under the License./>.
//

#ifndef AMPLITUDES_H
#define AMPLITUDES_H

#include<vector>

namespace pdaggerq {

class amplitudes {

  public:

    /// constructor
    amplitudes(){};

    /// destructor
    ~amplitudes(){};

    /// amplitude labels (human readable)
    std::vector<std::string> labels;

    /// amplitude numerical labels
    std::vector<int> numerical_labels;

    /// number of permutations required to sort amplitude labels
    int permutations = 0;

    /// sort amplitudes, keep track of permutations, assign total numerical value
    void sort();

    /// comparison between two amplitudes
    bool operator==(const amplitudes& rhs);

    /// copy amplitudes
    amplitudes operator=(const amplitudes& rhs);

};


}

#endif
