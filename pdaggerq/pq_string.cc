//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_string.cc
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

#include "pq_string.h"
#include "pq_tensor.h"

#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include<algorithm>
#include<cstring>
#include<cmath>
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
pq_string::pq_string(const std::string &vacuum_type){

    vacuum = vacuum_type;
}

// sort amplitude, integral, and delta function labels
void pq_string::sort_labels() {

    for (auto &ints_pair : ints) {
        std::string type = ints_pair.first;
        std::vector<integrals> &ints_vec = ints_pair.second;
        for (integrals & integral : ints_vec) {
            integral.sort();
        }
    }
    for (auto &amps_pair : amps) {
        char type = amps_pair.first;
        std::vector<amplitudes> &amps_vec = amps_pair.second;
        for (amplitudes & amp : amps_vec) {
            amp.sort();
        }
    }
    for (delta_functions & delta : deltas) {
        delta.sort();
    }
}

// is string in normal order? both fermion and boson parts
bool pq_string::is_normal_order() {

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
bool pq_string::is_boson_normal_order() {

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
void pq_string::print() {

    if ( skip ) return;

    if ( vacuum == "FERMI" && !symbol.empty() ) {
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

    if ( !permutations.empty() ) {
        // should have an even number of symbols...how many pairs?
        size_t n = permutations.size() / 2;
        int count = 0;
        for (int i = 0; i < n; i++) {
            printf("P(");
            printf("%s", permutations[count++].c_str());
            printf(",");
            printf("%s", permutations[count++].c_str());
            printf(")");
            printf(" ");
        }
    }
    if ( !paired_permutations_2.empty() ) {
        // should have an number of symbols divisible by 4
        size_t n = paired_permutations_2.size() / 4;
        int count = 0;
        for (int i = 0; i < n; i++) {
            printf("PP2(");
            printf("%s",paired_permutations_2[count++].c_str());
            printf(",");
            printf("%s",paired_permutations_2[count++].c_str());
            printf(",");
            printf("%s",paired_permutations_2[count++].c_str());
            printf(",");
            printf("%s",paired_permutations_2[count++].c_str());
            printf(")");
            printf(" ");
        }
    }
    if ( !paired_permutations_6.empty() ) {
        // should have an number of symbols divisible by 6
        size_t n = paired_permutations_6.size() / 6;
        int count = 0;
        for (int i = 0; i < n; i++) {
            printf("PP6(");
            printf("%s",paired_permutations_6[count++].c_str());
            printf(",");
            printf("%s",paired_permutations_6[count++].c_str());
            printf(",");
            printf("%s",paired_permutations_6[count++].c_str());
            printf(",");
            printf("%s",paired_permutations_6[count++].c_str());
            printf(",");
            printf("%s",paired_permutations_6[count++].c_str());
            printf(",");
            printf("%s",paired_permutations_6[count++].c_str());
            printf(")");
            printf(" ");
        }
    }
    if ( !paired_permutations_3.empty() ) {
        // should have an number of symbols divisible by 6
        size_t n = paired_permutations_3.size() / 6;
        int count = 0;
        for (int i = 0; i < n; i++) {
            printf("PP3(");
            printf("%s",paired_permutations_3[count++].c_str());
            printf(",");
            printf("%s",paired_permutations_3[count++].c_str());
            printf(",");
            printf("%s",paired_permutations_3[count++].c_str());
            printf(",");
            printf("%s",paired_permutations_3[count++].c_str());
            printf(",");
            printf("%s",paired_permutations_3[count++].c_str());
            printf(",");
            printf("%s",paired_permutations_3[count++].c_str());
            printf(")");
            printf(" ");
        }
    }

    for (size_t i = 0; i < symbol.size(); i++) {
        printf("%s", symbol[i].c_str());
        if ( is_dagger[i] ) {
            printf("%c", '*');
        }
        printf(" ");
    }

    // print deltas
    for (const delta_functions & delta : deltas) {
        delta.print();
    }

    // print integrals
    for (auto &ints_pair : ints) {
        std::string type = ints_pair.first;
        std::vector<integrals> &ints_vec = ints_pair.second;
        for (integrals & integral : ints_vec) {
            integral.print(type);
        }
    }

    // print amplitudes
    for (auto &amps_pair : amps) {
        char type = amps_pair.first;
        std::vector<amplitudes> &amps_vec = amps_pair.second;
        for (amplitudes & amp : amps_vec) {
            amp.print(type);
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
std::vector<std::string> pq_string::get_string_with_spin() {
    
    std::vector<std::string> my_string;
        
    if ( skip ) return my_string;
        
    if ( vacuum == "FERMI" && !symbol.empty() ) {
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
            
    if ( !permutations.empty() ) {
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

    if ( !paired_permutations_2.empty() ) {
        // should have a number of symbols divisible by 4
        size_t n = paired_permutations_2.size() / 4;
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            tmp  = "PP2(";
            tmp += paired_permutations_2[count++];
            tmp += ",";
            tmp += paired_permutations_2[count++];
            tmp += ",";
            tmp += paired_permutations_2[count++];
            tmp += ",";
            tmp += paired_permutations_2[count++];
            tmp += ")";
            my_string.push_back(tmp);
        }
    }

    if ( !paired_permutations_6.empty() ) {
        // should have a number of symbols divisible by 6
        size_t n = paired_permutations_6.size() / 6;
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            tmp  = "PP6(";
            tmp += paired_permutations_6[count++];
            tmp += ",";
            tmp += paired_permutations_6[count++];
            tmp += ",";
            tmp += paired_permutations_6[count++];
            tmp += ",";
            tmp += paired_permutations_6[count++];
            tmp += ",";
            tmp += paired_permutations_6[count++];
            tmp += ",";
            tmp += paired_permutations_6[count++];
            tmp += ")";
            my_string.push_back(tmp);
        }
    }

    if ( !paired_permutations_3.empty() ) {
        // should have a number of symbols divisible by 6
        size_t n = paired_permutations_3.size() / 6;
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            tmp  = "PP3(";
            tmp += paired_permutations_3[count++];
            tmp += ",";
            tmp += paired_permutations_3[count++];
            tmp += ",";
            tmp += paired_permutations_3[count++];
            tmp += ",";
            tmp += paired_permutations_3[count++];
            tmp += ",";
            tmp += paired_permutations_3[count++];
            tmp += ",";
            tmp += paired_permutations_3[count++];
            tmp += ")";
            my_string.push_back(tmp);
        }
    }
    
    for (size_t i = 0; i < symbol.size(); i++) {
        std::string tmp_symbol = symbol[i];
        if ( is_dagger[i] ) {
            tmp_symbol += "*";
        }
        my_string.push_back(tmp_symbol);
    }
    
    // deltas
    for (const delta_functions & delta : deltas) {
        my_string.push_back( delta.to_string_with_spin() );
    }   
    
    // integrals
    for (auto &ints_pair : ints) {
        std::string type = ints_pair.first;
        std::vector<integrals> &ints_vec = ints_pair.second;
        for (integrals & integral : ints_vec) {
            my_string.push_back( integral.to_string_with_spin(type) );
        }   
    }

    // amplitudes
    for (auto &amps_pair : amps) {
        char type = amps_pair.first;
        std::vector<amplitudes> &amps_vec = amps_pair.second;
        for (amplitudes & amp : amps_vec) {
            my_string.push_back( amp.to_string_with_spin(type));
        }
    }

    // bosons:
    for (size_t i = 0; i < is_boson_dagger.size(); i++) {
        if ( is_boson_dagger[i] ) {
            my_string.emplace_back("B*");
        }else {
            my_string.emplace_back("B");
        }
    }
    if ( has_w0 ) {
        my_string.emplace_back("w0");
    }

    return my_string;
}

// return string information
std::vector<std::string> pq_string::get_string() {

    std::vector<std::string> my_string;

    if ( skip ) return my_string;

    if ( vacuum == "FERMI" && !symbol.empty() ) {
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

    if ( !permutations.empty() ) {
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

    if ( !paired_permutations_2.empty() ) {
        // should have a number of symbols divisible by 4
        size_t n = paired_permutations_2.size() / 4;
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            tmp  = "PP2(";
            tmp += paired_permutations_2[count++];
            tmp += ",";
            tmp += paired_permutations_2[count++];
            tmp += ",";
            tmp += paired_permutations_2[count++];
            tmp += ",";
            tmp += paired_permutations_2[count++];
            tmp += ")";
            my_string.push_back(tmp);
        }
    }

    if ( !paired_permutations_6.empty() ) {
        // should have a number of symbols divisible by 6
        size_t n = paired_permutations_6.size() / 6;
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            tmp  = "PP6(";
            tmp += paired_permutations_6[count++];
            tmp += ",";
            tmp += paired_permutations_6[count++];
            tmp += ",";
            tmp += paired_permutations_6[count++];
            tmp += ",";
            tmp += paired_permutations_6[count++];
            tmp += ",";
            tmp += paired_permutations_6[count++];
            tmp += ",";
            tmp += paired_permutations_6[count++];
            tmp += ")";
            my_string.push_back(tmp);
        }
    }

    if ( !paired_permutations_3.empty() ) {
        // should have a number of symbols divisible by 6
        size_t n = paired_permutations_3.size() / 6;
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            tmp  = "PP3(";
            tmp += paired_permutations_3[count++];
            tmp += ",";
            tmp += paired_permutations_3[count++];
            tmp += ",";
            tmp += paired_permutations_3[count++];
            tmp += ",";
            tmp += paired_permutations_3[count++];
            tmp += ",";
            tmp += paired_permutations_3[count++];
            tmp += ",";
            tmp += paired_permutations_3[count++];
            tmp += ")";
            my_string.push_back(tmp);
        }
    }

    for (size_t i = 0; i < symbol.size(); i++) {
        std::string tmp_symbol = symbol[i];
        if ( is_dagger[i] ) {
            tmp_symbol += "*";
        }
        my_string.push_back(tmp_symbol);
    }

    // deltas
    for (const delta_functions & delta : deltas) {
        my_string.push_back( delta.to_string() );
    }

    // integrals
    for (auto &ints_pair : ints) {
        std::string type = ints_pair.first;
        std::vector<integrals> &ints_vec = ints_pair.second;
        for (integrals & integral : ints_vec) {
            my_string.push_back( integral.to_string(type) );
        }
    }

    // amplitudes
    for (auto &amps_pair : amps) {
        char type = amps_pair.first;
        std::vector<amplitudes> &amps_vec = amps_pair.second;
        for (amplitudes & amp : amps_vec) {
            my_string.push_back( amp.to_string(type) );
        }
    }

    // bosons:
    for (size_t i = 0; i < is_boson_dagger.size(); i++) {
        if ( is_boson_dagger[i] ) {
            my_string.emplace_back("B*");
        }else {
            my_string.emplace_back("B");
        }
    }
    if ( has_w0 ) {
        my_string.emplace_back("w0");
    }

    return my_string;
}


// TODO: should probably make sure all of the std::vectors
//       (ints, amplitudes, deltas) have been cleared.

// copy string data, possibly excluding symbols and daggers
void pq_string::copy(void * copy_me, bool copy_daggers_and_symbols) {

    auto * in = reinterpret_cast<pq_string * >(copy_me);

    // skip string?
    skip   = in->skip;

    // sign
    sign   = in->sign;

    // factor
    factor = in->factor;

    // deltas
    deltas = in->deltas;

    // integrals
    ints = in->ints;

    // amplitudes
    amps = in->amps;

    // w0
    has_w0 = in->has_w0;

    // non-summed spin labels
    non_summed_spin_labels = in->non_summed_spin_labels;

    // permutations
    permutations = in->permutations;

    // paired permutations (2)
    paired_permutations_2 = in->paired_permutations_2;

    // paired permutations (3)
    paired_permutations_3 = in->paired_permutations_3;

    // paired permutations (6)
    paired_permutations_6 = in->paired_permutations_6;

    if ( copy_daggers_and_symbols ) {

        // fermion operator symbols
        symbol = in->symbol;

        // fermion daggers
        is_dagger = in->is_dagger;

        // fermion daggers with respect to fermi vacuum
        if ( vacuum == "FERMI" ) {
            is_dagger_fermi = in->is_dagger_fermi;
        }

        // boson daggers
        is_boson_dagger = in->is_boson_dagger;
    }
}

void pq_string::set_spin_everywhere(const std::string &target, const std::string &spin) {

    // integrals
    for (auto &ints_pair : ints) {
        std::string type = ints_pair.first;
        std::vector<integrals> &ints_vec = ints_pair.second;
        for (integrals & integral : ints_vec) {
            for (size_t k = 0; k < integral.labels.size(); k++) {
                if ( integral.labels[k] == target ) {
                    integral.spin_labels[k] = spin;
                }
            }
        }
    }
    // amplitudes
    for (auto &amps_pair : amps) {
        char type = amps_pair.first;
        std::vector<amplitudes> &amps_vec = amps_pair.second;
        for (amplitudes & amp : amps_vec) {
            for (size_t k = 0; k < amp.labels.size(); k++) {
                if ( amp.labels[k] == target ) {
                     amp.spin_labels[k] = spin;
                }
            }
        }
    }
    // deltas
    for (delta_functions & delta : deltas) {
        for (size_t j = 0; j < delta.labels.size(); j++) {
            if ( delta.labels[j] == target ) {
                delta.spin_labels[j] = spin;
            }
        }
    }
}

// reset spin labels
void pq_string::reset_spin_labels() {

    // amplitudes
    for (auto &amps_pair : amps) {
        char type = amps_pair.first;
        std::vector<amplitudes> &amps_vec = amps_pair.second;
        for (amplitudes & amp : amps_vec) {
            amp.spin_labels.clear();
            for (size_t k = 0; k < amp.labels.size(); k++) {
                amp.spin_labels.emplace_back("");
            }
        }
    }
    // integrals
    for (auto &ints_pair : ints) {
        std::string type = ints_pair.first;
        std::vector<integrals> &ints_vec = ints_pair.second;
        for (integrals & integral : ints_vec) {
            integral.spin_labels.clear();
            for (size_t k = 0; k < integral.labels.size(); k++) {
                integral.spin_labels.emplace_back("");
            }
        }
    }
    // deltas
    for (delta_functions & delta : deltas) {
        delta.spin_labels.clear();
        for (size_t j = 0; j < delta.labels.size(); j++) {
            delta.spin_labels.emplace_back("");
        }
    }

    std::vector<std::string> occ_labels { "i", "j", "k", "l", "m", "n", "o" };
    std::vector<std::string> vir_labels { "a", "b", "c", "d", "e", "f", "g" };

    // set spins for occupied non-summed labels
    for (const std::string & occ_label : occ_labels) {
        std::string spin = non_summed_spin_labels[occ_label];
        if ( spin == "a" || spin == "b" ) {
            // amplitudes
            for (auto &amps_pair : amps) {
                char type = amps_pair.first;
                std::vector<amplitudes> &amps_vec = amps_pair.second;
                for (amplitudes & amp : amps_vec) {
                    for (size_t k = 0; k < amp.labels.size(); k++) {
                        if ( amp.labels[k] == occ_label ) {
                            amp.spin_labels[k] = spin;
                        }
                    }
                }
            }
            // integrals
            for (auto &ints_pair : ints) {
                std::string type = ints_pair.first;
                std::vector<integrals> &ints_vec = ints_pair.second;
                for (integrals & integral : ints_vec) {
                    for (size_t k = 0; k < integral.labels.size(); k++) {
                        if ( integral.labels[k] == occ_label ) {
                            integral.spin_labels[k] = spin;
                        }
                    }
                }
            }
            // deltas
            for (delta_functions & delta : deltas) {
                for (size_t j = 0; j < delta.labels.size(); j++) {
                    if ( delta.labels[j] == occ_label ) {
                        delta.spin_labels[j] = spin;
                    }
                }
            }
        }
    }

    // set spins for virtual non-summed labels
    for (const auto & vir_label : vir_labels) {
        std::string spin = non_summed_spin_labels[vir_label];
        if ( spin == "a" || spin == "b" ) {
            // amplitudes
            for (auto &amps_pair : amps) {
                char type = amps_pair.first;
                std::vector<amplitudes> &amps_vec = amps_pair.second;
                for (amplitudes & amp : amps_vec) {
                    for (size_t k = 0; k < amp.labels.size(); k++) {
                        if ( amp.labels[k] == vir_label ) {
                            amp.spin_labels[k] = spin;
                        }
                    }
                }
            }
            // integrals
            for (auto &ints_pair : ints) {
                std::string type = ints_pair.first;
                std::vector<integrals> &ints_vec = ints_pair.second;
                for (integrals & integral : ints_vec) {
                    for (size_t k = 0; k < integral.labels.size(); k++) {
                        if ( integral.labels[k] == vir_label ) {
                            integral.spin_labels[k] = spin;
                        }
                    }
                }
            }
            // deltas
            for (delta_functions & delta : deltas) {
                for (size_t j = 0; j < delta.labels.size(); j++) {
                    if ( delta.labels[j] == vir_label ) {
                        delta.spin_labels[j] = spin;
                    }
                }
            }
        }
    }
}

// set labels for integrals
void pq_string::set_integrals(const std::string &type, const std::vector<std::string> &in) {
    integrals new_ints;
    new_ints.labels.assign(in.begin(), in.end());
    new_ints.sort();
    ints[type].push_back(new_ints);
}

// set labels for amplitudes
void pq_string::set_amplitudes(char type, int order, const std::vector<std::string> &in) {
    amplitudes new_amps;
    new_amps.labels.assign(in.begin(), in.end());
    new_amps.order = order;
    new_amps.sort();
    amps[type].push_back(new_amps);
}

}
