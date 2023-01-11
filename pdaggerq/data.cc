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
#include "pq.h"

#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include<algorithm>
#include<cstring>
#include<math.h>
#include<sstream>

namespace pdaggerq {

// work-around for finite precision of std::to_string
template <typename T> std::string to_string_with_precision(const T a_value, const int n = 14) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

// constructor
StringData::StringData(std::string vacuum_type){

    vacuum = vacuum_type;
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

// is fermion part of string in normal order?
bool StringData::is_normal_order() {

    // don't bother bringing to normal order if we're going to skip this string
    if (skip) return true;

    // fermions
    if ( vacuum == "TRUE" ) {
        for (int i = 0; i < (int)symbol.size()-1; i++) {
            if ( !is_dagger[i] && is_dagger[i+1] ) {
                return false;
            }
        }
    }else {
        for (int i = 0; i < (int)symbol.size()-1; i++) {
            // check if stings should be zero or not
            bool is_dagger_right = is_dagger_fermi[symbol.size()-1];
            bool is_dagger_left  = is_dagger_fermi[0];
            if ( !is_dagger_right || is_dagger_left ) {
                skip = true; // added 5/28/21
                return true;
            }
            if ( !is_dagger_fermi[i] && is_dagger_fermi[i+1] ) {
                return false;
            }
        }
    }

    // bosons
    if ( !is_boson_normal_order() ) {
        return false;
    }

    return true;
}

// is boson part of string in normal order?
bool StringData::is_boson_normal_order() {

    if ( is_boson_dagger.size() == 1 ) {
        bool is_dagger_right = is_boson_dagger[0];
        bool is_dagger_left  = is_boson_dagger[0];
        if ( !is_dagger_right || is_dagger_left ) {
            skip = true;
            return true;
        }
    }
    for (int i = 0; i < (int)is_boson_dagger.size()-1; i++) {

        // check if stings should be zero or not ... added 5/28/21
        bool is_dagger_right = is_boson_dagger[is_boson_dagger.size()-1];
        bool is_dagger_left  = is_boson_dagger[0];
        if ( !is_dagger_right || is_dagger_left ) {
            skip = true;
            return true;
        }

        if ( !is_boson_dagger[i] && is_boson_dagger[i+1] ) {
            return false;
        }
    }
    return true;
}

// print string information
void StringData::print() {

    if ( skip ) return;

    if ( vacuum == "FERMI" && symbol.size() > 0 ) {
        // check if stings should be zero or not
        bool is_dagger_right = is_dagger_fermi[symbol.size()-1];
        bool is_dagger_left  = is_dagger_fermi[0];
        if ( !is_dagger_right || is_dagger_left ) {
            //return;
        }
    }

    printf("    ");
    printf("//     ");
    printf("%c", sign > 0 ? '+' : '-');
    printf(" ");
    printf("%20.14lf", fabs(factor));
    printf(" ");

    if ( permutations.size() > 0 ) {
        // should have an even number of symbols...how many pairs?
        size_t n = permutations.size() / 2;
        int count = 0;
        for (int i = 0; i < n; i++) {
            printf("P(");
            printf("%s",permutations[count++].c_str());
            printf(",");
            printf("%s",permutations[count++].c_str());
            printf(")");
            printf(" ");
        }
    }

    for (size_t i = 0; i < symbol.size(); i++) {
        printf("%s",symbol[i].c_str());
        if ( is_dagger[i] ) {
            printf("%c",'*');
        }
        printf(" ");
    }

    // print deltas
    for (size_t i = 0; i < deltas.size(); i++) {
        deltas[i].print();
    }

    // print integrals
    for (size_t i = 0; i < integral_types.size(); i++) {
        std::string type = integral_types[i];
        for (size_t j = 0; j < ints[type].size(); j++) {
            ints[type][j].print(type);
        }
    }

    // print amplitudes
    for (size_t i = 0; i < amplitude_types.size(); i++) {
        char type = amplitude_types[i];
        for (size_t j = 0; j < amps[type].size(); j++) {
            amps[type][j].print(type);
        }
    }

    // bosons:
    for (size_t i = 0; i < is_boson_dagger.size(); i++) {
        if ( is_boson_dagger[i] ) {
            printf("B* ");
        }else {
            printf("B ");
        }
    }
    if ( has_w0 ) {
        printf("w0");
        printf(" ");
    }

    printf("\n");
}

// return string information (with spin)
std::vector<std::string> StringData::get_string_with_spin() {
    
    std::vector<std::string> my_string;
        
    if ( skip ) return my_string;
        
    if ( vacuum == "FERMI" && symbol.size() > 0 ) {
        // check if stings should be zero or not
        bool is_dagger_right = is_dagger_fermi[symbol.size()-1];
        bool is_dagger_left  = is_dagger_fermi[0];
        if ( !is_dagger_right || is_dagger_left ) {
            //return;
        }
    }
    
    std::string tmp;
    if ( sign > 0 ) {
        tmp = "+";
    }else {
        tmp = "-"; 
    }   
    //my_string.push_back(tmp + std::to_string(fabs(factor)));
    my_string.push_back(tmp + to_string_with_precision(fabs(factor), 14));
            
    if ( permutations.size() > 0 ) {
        // should have an even number of symbols...how many pairs?
        size_t n = permutations.size() / 2;
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            tmp  = "P(";
            tmp += permutations[count++];
            tmp += ",";
            tmp += permutations[count++];
            tmp += ")";
            my_string.push_back(tmp);
        }
    }   
    
    for (size_t i = 0; i < symbol.size(); i++) {
        std::string tmp = symbol[i];
        if ( is_dagger[i] ) {
            tmp += "*";
        }
        my_string.push_back(tmp);
    }
    
    // deltas
    for (size_t i = 0; i < deltas.size(); i++) {
        my_string.push_back( deltas[i].to_string_with_spin() );
    }   
    
    // integrals
    for (size_t i = 0; i < integral_types.size(); i++) {
        std::string type = integral_types[i];
        for (size_t j = 0; j < ints[type].size(); j++) {
            my_string.push_back( ints[type][j].to_string_with_spin(type) );
        }   
    }

    // amplitudes
    for (size_t i = 0; i < amplitude_types.size(); i++) {
        char type = amplitude_types[i];
        for (size_t j = 0; j < amps[type].size(); j++) {
            my_string.push_back( amps[type][j].to_string_with_spin(type));
        }
    }

    // bosons:
    for (size_t i = 0; i < is_boson_dagger.size(); i++) {
        if ( is_boson_dagger[i] ) {
            my_string.push_back("B*");
        }else {
            my_string.push_back("B");
        }
    }
    if ( has_w0 ) {
        my_string.push_back("w0");
    }

    return my_string;
}

// return string information
std::vector<std::string> StringData::get_string() {

    std::vector<std::string> my_string;

    if ( skip ) return my_string;

    if ( vacuum == "FERMI" && symbol.size() > 0 ) {
        // check if stings should be zero or not
        bool is_dagger_right = is_dagger_fermi[symbol.size()-1];
        bool is_dagger_left  = is_dagger_fermi[0];
        if ( !is_dagger_right || is_dagger_left ) {
            //return;
        }
    }

    std::string tmp;
    if ( sign > 0 ) {
        tmp = "+";
    }else {
        tmp = "-";
    }
    //my_string.push_back(tmp + std::to_string(fabs(factor)));
    my_string.push_back(tmp + to_string_with_precision(fabs(factor), 14));

    if ( permutations.size() > 0 ) {
        // should have an even number of symbols...how many pairs?
        size_t n = permutations.size() / 2;
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            tmp  = "P(";
            tmp += permutations[count++];
            tmp += ",";
            tmp += permutations[count++];
            tmp += ")";
            my_string.push_back(tmp);
        }
    }

    for (size_t i = 0; i < symbol.size(); i++) {
        std::string tmp = symbol[i];
        if ( is_dagger[i] ) {
            tmp += "*";
        }
        my_string.push_back(tmp);
    }

    // deltas
    for (size_t i = 0; i < deltas.size(); i++) {
        my_string.push_back( deltas[i].to_string() );
    }

    // integrals
    for (size_t i = 0; i < integral_types.size(); i++) {
        std::string type = integral_types[i];
        for (size_t j = 0; j < ints[type].size(); j++) {
            my_string.push_back( ints[type][j].to_string(type) );
        }
    }

    // amplitudes
    for (size_t i = 0; i < amplitude_types.size(); i++) {
        char type = amplitude_types[i];
        for (size_t j = 0; j < amps[type].size(); j++) {
            my_string.push_back( amps[type][j].to_string(type) );
        }
    }

    // bosons:
    for (size_t i = 0; i < is_boson_dagger.size(); i++) {
        if ( is_boson_dagger[i] ) {
            my_string.push_back("B*");
        }else {
            my_string.push_back("B");
        }
    }
    if ( has_w0 ) {
        my_string.push_back("w0");
    }

    return my_string;
}

// copy all data, except symbols and daggers. 

// TODO: should probably make sure all of the std::vectors
//       (ints, amplitudes, deltas) have been cleared.
void StringData::shallow_copy(void * copy_me) {

    pq * in = reinterpret_cast<pq * >(copy_me);

    // skip string?
    skip   = in->data->skip;

    // sign
    sign   = in->data->sign;

    // factor
    factor = in->data->factor;

    // deltas
    for (size_t i = 0; i < in->data->deltas.size(); i++) {
        deltas.push_back(in->data->deltas[i]);
    }

    // integrals
    for (size_t i = 0; i < integral_types.size(); i++) {
        std::string type = integral_types[i];
        for (size_t j = 0; j < in->data->ints[type].size(); j++) {
            ints[type].push_back( in->data->ints[type][j] );
        }
    }

    // amplitudes
    for (size_t i = 0; i < amplitude_types.size(); i++) {
        char type = amplitude_types[i];
        for (size_t j = 0; j < in->data->amps[type].size(); j++) {
            amps[type].push_back( in->data->amps[type][j] );
        }
    }

    // w0 
    has_w0 = in->data->has_w0;

    // non-summed spin labels
    non_summed_spin_labels = in->data->non_summed_spin_labels;
}

// copy all data, including symbols and daggers
void StringData::copy(void * copy_me) {

    shallow_copy(copy_me);

    pq * in = reinterpret_cast<pq * >(copy_me);

    // operators
    for (size_t j = 0; j < in->data->symbol.size(); j++) {
        symbol.push_back(in->data->symbol[j]);

        // dagger?
        is_dagger.push_back(in->data->is_dagger[j]);

        // dagger (relative to fermi vacuum)?
        if ( vacuum == "FERMI" ) {
            is_dagger_fermi.push_back(in->data->is_dagger_fermi[j]);
        }
    }

    // boson daggers
    for (size_t i = 0; i < in->data->is_boson_dagger.size(); i++) {
        is_boson_dagger.push_back(in->data->is_boson_dagger[i]);
    }

    // permutations
    for (size_t i = 0; i < in->data->permutations.size(); i++) {
        permutations.push_back(in->data->permutations[i]);
    }
}

void StringData::set_spin_everywhere(std::string target, std::string spin) {

    // integrals
    for (size_t i = 0; i < integral_types.size(); i++) {
        std::string type = integral_types[i];
        for (size_t j = 0; j < ints[type].size(); j++) {
            for (size_t k = 0; k < ints[type][j].labels.size(); k++) {
                if ( ints[type][j].labels[k] == target ) {
                    ints[type][j].spin_labels[k] = spin;
                }
            }
        }
    }
    // amplitudes
    for (size_t i = 0; i < amplitude_types.size(); i++) {
        char type = amplitude_types[i];
        for (size_t j = 0; j < amps[type].size(); j++) {
            for (size_t k = 0; k < amps[type][j].labels.size(); k++) {
                if ( amps[type][j].labels[k] == target ) {
                    amps[type][j].spin_labels[k] = spin;
                }
            }
        }
    }
    // deltas
    for (size_t i = 0; i < deltas.size(); i++) {
        for (size_t j = 0; j < deltas[i].labels.size(); j++) {
            if ( deltas[i].labels[j] == target ) {
                deltas[i].spin_labels[j] = spin;
            }
        }
    }
}

}
