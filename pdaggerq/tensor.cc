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

#include"tensor.h"

#include<string>

namespace pdaggerq {

/// sort amplitude labels
void amplitudes::sort() {

    numerical_labels.clear();

    // convert labels to numerical labels
    for (size_t i = 0; i < labels.size(); i++) {
        int numerical_label = 0;
        int factor = 1;
        for (size_t j = 0; j < labels[i].size(); j++) {
            numerical_label += factor * labels[i][j];
            factor *= 128;
        }
        numerical_labels.push_back(numerical_label);
    }

    permutations = 0;

    // sort labels and accumulate permutations
    for (size_t step = 1; step < numerical_labels.size(); step++) {
        
        bool swapped = false;
        for (size_t i = 0; i < numerical_labels.size() - step; i++) {
    
            // compare elements
            if (numerical_labels[i] > numerical_labels[i + 1]) {
    
              // swap
              int temp = numerical_labels[i];
              numerical_labels[i] = numerical_labels[i + 1];
              numerical_labels[i + 1] = temp;

              // accumulate permutations
              permutations++;

              swapped = true;

            }
        }
        if ( !swapped ) {
            break;
        }
    }

    return;
}

/// copy amplitudes
amplitudes amplitudes::operator=(const amplitudes& rhs) {

    amplitudes amps;

    amps.labels.clear();
    amps.numerical_labels.clear();

    for (size_t i = 0; i < rhs.labels.size(); i++) {
        amps.labels.push_back(rhs.labels[i]);
    }

    amps.is_reference = rhs.is_reference;

    //amps.sort();

    return amps;
}

void amplitudes::print(char symbol) {

    if ( labels.size() > 0 ) {

        size_t size  = labels.size();
        size_t order = labels.size() / 2;
        if ( 2*order != size ) {
            order++;
        }
        printf("%c",symbol);
        printf("%zu",order);
        printf("(");
        for (size_t j = 0; j < size-1; j++) {
            printf("%s",labels[j].c_str());
            printf(",");
        }
        printf("%s",labels[size-1].c_str());
        printf(")");
        printf(" ");

    }else if ( is_reference ) {
        printf("%c0", symbol);
        printf(" ");
    }
}

std::string amplitudes::to_string(char symbol) {

    std::string val;

    std::string symbol_s(1, symbol);

    if ( labels.size() > 0 ) {

        size_t size  = labels.size();
        size_t order = labels.size() / 2;
        if ( 2*order != size ) {
            order++;
        }
        val = symbol_s + std::to_string(order) + "(";
        for (int j = 0; j < size-1; j++) {
            val += labels[j] + ",";
        }
        val += labels[size-1] + ")";

    }

    if ( is_reference ) {
        val = symbol_s + "0";
    }

    return val;
}




/// sort integrals labels
void integrals::sort() {

    numerical_labels.clear();

    permutations = 0;

    return;
}


/// copy integrals
integrals integrals::operator=(const integrals& rhs) {

    integrals ints;

    ints.labels.clear();
    ints.numerical_labels.clear();

    for (size_t i = 0; i < rhs.labels.size(); i++) {
        ints.labels.push_back(rhs.labels[i]);
    }

    return ints;
}

}

