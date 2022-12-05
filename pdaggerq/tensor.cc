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
    amps.spin_labels.clear();

    for (size_t i = 0; i < rhs.labels.size(); i++) {
        amps.labels.push_back(rhs.labels[i]);
    }
    for (size_t i = 0; i < rhs.spin_labels.size(); i++) {
        amps.spin_labels.push_back(rhs.spin_labels[i]);
    }

    amps.order = rhs.order;

    //amps.sort();

    return amps;
}

/// print amplitudes 
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

    }else if ( order == 0 ) {
        printf("%c0", symbol);
        printf(" ");
    }
}

/// print amplitudes to string 
std::string amplitudes::to_string(char symbol) {

    std::string val;

    std::string symbol_s(1, symbol);

    if ( labels.size() > 0 ) {

        size_t size  = labels.size();
        //size_t order = labels.size() / 2;
        //if ( 2*order != size ) {
        //    order++;
        //}
        val = symbol_s + std::to_string(order) + "(";
        for (int j = 0; j < size-1; j++) {
            val += labels[j] + ",";
        }
        val += labels[size-1] + ")";

    }

    if ( order == 0 ) {
        val = symbol_s + "0";
    }

    return val;
}

/// print amplitudes to string with spin labels
std::string amplitudes::to_string_with_spin(char symbol) {

    std::string val;

    std::string symbol_s(1, symbol);

    std::string spin = "_";
    for (size_t k = 0; k < spin_labels.size(); k++) {
        spin += spin_labels[k];
    }

    if ( labels.size() > 0 ) {

        size_t size  = labels.size();
        //size_t order = labels.size() / 2;
        //if ( 2*order != size ) {
        //    order++;
        //}
        val = symbol_s + std::to_string(order) + spin + "(";
        for (int j = 0; j < size-1; j++) {
            val += labels[j] + ",";
        }
        val += labels[size-1] + ")";

    }

    if ( order == 0 ) {
        val = symbol_s + "0";
    }

    return val;
}

/// sort integrals labels
void integrals::sort() {

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

    if ( numerical_labels.size() == 4 ) {

        // bra
        if ( numerical_labels[0] > numerical_labels[1] ) {
            int tmp = numerical_labels[0];
            numerical_labels[0] = numerical_labels[1];
            numerical_labels[1] = tmp;
            permutations++;
        }

        // ket
        if ( numerical_labels[2] > numerical_labels[3] ) {
            int tmp = numerical_labels[2];
            numerical_labels[2] = numerical_labels[3];
            numerical_labels[3] = tmp;
            permutations++;
        }

    }

    return;
}

/// copy integrals
integrals integrals::operator=(const integrals& rhs) {

    integrals ints;

    ints.labels.clear();
    ints.numerical_labels.clear();
    ints.spin_labels.clear();

    for (size_t i = 0; i < rhs.labels.size(); i++) {
        ints.labels.push_back(rhs.labels[i]);
    }
    for (size_t i = 0; i < rhs.spin_labels.size(); i++) {
        ints.spin_labels.push_back(rhs.spin_labels[i]);
    }

    return ints;
}

/// print integrals 
void integrals::print(std::string symbol) {

    if ( symbol == "two_body") {
        printf("g(");
        printf("%s",labels[0].c_str());
        printf(",");
        printf("%s",labels[1].c_str());
        printf(",");
        printf("%s",labels[2].c_str());
        printf(",");
        printf("%s",labels[3].c_str());
        printf(")");
        printf(" ");
    }else if (symbol == "eri" ) {
        printf("<");
        printf("%s",labels[0].c_str());
        printf(",");
        printf("%s",labels[1].c_str());
        printf("||");
        printf("%s",labels[2].c_str());
        printf(",");
        printf("%s",labels[3].c_str());
        printf(">");
        printf(" ");
    }else if ( symbol == "core") {
        printf("h(");
        printf("%s",labels[0].c_str());
        printf(",");
        printf("%s",labels[1].c_str());
        printf(")");
        printf(" ");
    }else if ( symbol == "fock") {
        printf("f(");
        printf("%s",labels[0].c_str());
        printf(",");
        printf("%s",labels[1].c_str());
        printf(")");
        printf(" ");
    }else if ( symbol == "d+") {
        printf("d+(");
        printf("%s",labels[0].c_str());
        printf(",");
        printf("%s",labels[1].c_str());
        printf(")");
        printf(" ");
    }else if ( symbol == "d-") {
        printf("d-(");
        printf("%s",labels[0].c_str());
        printf(",");
        printf("%s",labels[1].c_str());
        printf(")");
        printf(" ");
    }else {
        printf("\n");
        printf("    unknown integral type\n");
        printf("\n");
        exit(1);
    }

}

/// print integrals to string
std::string integrals::to_string(std::string symbol) {

    std::string val;

    if ( symbol == "two_body") {
        val = "g("
            + labels[0]
            + ","
            + labels[1]
            + ","
            + labels[2]
            + ","
            + labels[3]
            + ")";
    }else if ( symbol == "eri" ) {
        val = "<"
            + labels[0]
            + ","
            + labels[1]
            + "||"
            + labels[2]
            + ","
            + labels[3]
            + ">";
    }else if ( symbol == "core") {
        val = "h("
            + labels[0]
            + ","
            + labels[1]
            + ")";
    }else if ( symbol == "fock") {
        val = "f("
            + labels[0]
            + ","
            + labels[1]
            + ")";
    }else if ( symbol == "d+") {
        val = "d+("
            + labels[0]
            + ","
            + labels[1]
            + ")";
    }else if ( symbol == "d-") {
        val = "d-("
            + labels[0]
            + ","
            + labels[1]
            + ")";
    }

    return val;
}

/// print integrals to string with spin labels
std::string integrals::to_string_with_spin(std::string symbol) {

    std::string val;

    std::string spin = "_";
    for (size_t k = 0; k < spin_labels.size(); k++) {
        spin += spin_labels[k];
    }

    if ( symbol == "two_body") {
        val = "g" + spin + "("
            + labels[0]
            + ","
            + labels[1]
            + ","
            + labels[2]
            + ","
            + labels[3]
            + ")";
    }else if ( symbol == "eri" ) {
        val = "<"
            + labels[0]
            + ","
            + labels[1]
            + "||"
            + labels[2]
            + ","
            + labels[3]
            + ">" + spin;
    }else if ( symbol == "core") {
        val = "h" + spin + "("
            + labels[0]
            + ","
            + labels[1]
            + ")";
    }else if ( symbol == "fock") {
        val = "f" + spin + "("
            + labels[0]
            + ","
            + labels[1]
            + ")";
    }else if ( symbol == "d+") {
        val = "d+" + spin + "("
            + labels[0]
            + ","
            + labels[1]
            + ")";
    }else if ( symbol == "d-") {
        val = "d-" + spin + "("
            + labels[0]
            + ","
            + labels[1]
            + ")";
    }

    return val;
}


/// sort deltas labels
void delta_functions::sort() {

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

    return;
}

/// copy deltas
delta_functions delta_functions::operator=(const delta_functions& rhs) {

    delta_functions d;

    d.labels.clear();
    d.numerical_labels.clear();
    d.spin_labels.clear();

    for (size_t i = 0; i < rhs.labels.size(); i++) {
        d.labels.push_back(rhs.labels[i]);
    }
    for (size_t i = 0; i < rhs.spin_labels.size(); i++) {
        d.spin_labels.push_back(rhs.spin_labels[i]);
    }

    return d;
}

/// print deltas 
void delta_functions::print() {

    printf("d(");
    printf("%s",labels[0].c_str());
    printf(",");
    printf("%s",labels[1].c_str());
    printf(")");
    printf(" ");
}

/// print deltas to string
std::string delta_functions::to_string() {

    std::string val;

    val = "d("
        + labels[0]
        + ","
        + labels[1]
        + ")";

    return val;
}

/// print deltas to string with spin labels
std::string delta_functions::to_string_with_spin() {

    std::string val;

    std::string spin = "_";
    for (size_t k = 0; k < spin_labels.size(); k++) {
        spin += spin_labels[k];
    }

    val = "d" + spin + "("
        + labels[0]
        + ","
        + labels[1]
        + ")";

    return val;
}

}

