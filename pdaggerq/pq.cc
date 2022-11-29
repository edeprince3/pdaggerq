//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq.cc
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

#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include<algorithm>
#include<cstring>
#include <math.h>
#include<sstream>

#include "pq.h"

// work-around for finite precision of std::to_string
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 14)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

namespace pdaggerq {

pq::pq(std::string vacuum_type) {

  vacuum = vacuum_type;
  skip = false;
  data = (std::shared_ptr<StringData>)(new StringData());

}

pq::~pq() {
}

bool pq::is_occ(std::string idx) {
    if ( idx == "I" || idx == "i") {
        return true;
    }else if ( idx == "J" || idx == "j") {
        return true;
    }else if ( idx == "K" || idx == "k") {
        return true;
    }else if ( idx == "L" || idx == "l") {
        return true;
    }else if ( idx == "M" || idx == "m") {
        return true;
    }else if ( idx == "N" || idx == "n") {
        return true;
    }else if ( idx == "N" || idx == "o") {
        return true;
    }else if ( idx.at(0) == 'O' || idx.at(0) == 'o') {
        return true;
    }else if ( idx.at(0) == 'I' || idx.at(0) == 'i') {
        return true;
    }
    return false;
}

bool pq::is_vir(std::string idx) {
    if ( idx == "A" || idx == "a") {
        return true;
    }else if ( idx == "B" || idx == "b") {
        return true;
    }else if ( idx == "C" || idx == "c") {
        return true;
    }else if ( idx == "D" || idx == "d") {
        return true;
    }else if ( idx == "E" || idx == "e") {
        return true;
    }else if ( idx == "F" || idx == "f") {
        return true;
    }else if ( idx == "F" || idx == "g") {
        return true;
    }else if ( idx.at(0) == 'V' || idx.at(0) == 'v') {
        return true;
    }else if ( idx.at(0) == 'A' || idx.at(0) == 'a') {
        return true;
    }
    return false;
}

void pq::print() {

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
    printf("%20.14lf", fabs(data->factor));
    printf(" ");

    if ( data->permutations.size() > 0 ) {
        // should have an even number of symbols...how many pairs?
        size_t n = data->permutations.size() / 2;
        int count = 0;
        for (int i = 0; i < n; i++) {
            printf("P(");
            printf("%s",data->permutations[count++].c_str());
            printf(",");
            printf("%s",data->permutations[count++].c_str());
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
    for (size_t i = 0; i < data->deltas.size(); i++) {
        data->deltas[i].print();
    }

    // print integrals
    for (size_t i = 0; i < data->integral_types.size(); i++) { 
        std::string type = data->integral_types[i];
        for (size_t j = 0; j < data->ints[type].size(); j++) { 
            data->ints[type][j].print(type);
        }
    }

    // print amplitudes
    for (size_t i = 0; i < data->amplitude_types.size(); i++) { 
        char type = data->amplitude_types[i];
        for (size_t j = 0; j < data->amps[type].size(); j++) { 
            data->amps[type][j].print(type);
        }
    }

    // bosons:
    for (size_t i = 0; i < data->is_boson_dagger.size(); i++) {
        if ( data->is_boson_dagger[i] ) {
            printf("B* ");
        }else {
            printf("B ");
        }
    }
    if ( data->has_w0 ) {
        printf("w0");
        printf(" ");
    }

    printf("\n");
}

std::vector<std::string> pq::get_string_with_spin() {

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
    //my_string.push_back(tmp + std::to_string(fabs(data->factor)));
    my_string.push_back(tmp + to_string_with_precision(fabs(data->factor),14));

    if ( data->permutations.size() > 0 ) {
        // should have an even number of symbols...how many pairs?
        size_t n = data->permutations.size() / 2;
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            tmp  = "P(";
            tmp += data->permutations[count++];
            tmp += ",";
            tmp += data->permutations[count++];
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
    for (size_t i = 0; i < data->deltas.size(); i++) {
        my_string.push_back( data->deltas[i].to_string_with_spin() );
    }

    // integrals
    for (size_t i = 0; i < data->integral_types.size(); i++) { 
        std::string type = data->integral_types[i];
        for (size_t j = 0; j < data->ints[type].size(); j++) { 
            my_string.push_back( data->ints[type][j].to_string_with_spin(type) );
        }
    }

    // amplitudes
    for (size_t i = 0; i < data->amplitude_types.size(); i++) { 
        char type = data->amplitude_types[i];
        for (size_t j = 0; j < data->amps[type].size(); j++) { 
            my_string.push_back( data->amps[type][j].to_string_with_spin(type));
        }
    }

    // bosons:
    for (size_t i = 0; i < data->is_boson_dagger.size(); i++) {
        if ( data->is_boson_dagger[i] ) {
            my_string.push_back("B*");
        }else {
            my_string.push_back("B");
        }
    }
    if ( data->has_w0 ) {
        my_string.push_back("w0");
    }

    return my_string;
}
std::vector<std::string> pq::get_string() {

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
    //my_string.push_back(tmp + std::to_string(fabs(data->factor)));
    my_string.push_back(tmp + to_string_with_precision(fabs(data->factor),14));

    if ( data->permutations.size() > 0 ) {
        // should have an even number of symbols...how many pairs?
        size_t n = data->permutations.size() / 2;
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            tmp  = "P(";
            tmp += data->permutations[count++];
            tmp += ",";
            tmp += data->permutations[count++];
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
    for (size_t i = 0; i < data->deltas.size(); i++) {
        my_string.push_back( data->deltas[i].to_string() );
    }

    // integrals
    for (size_t i = 0; i < data->integral_types.size(); i++) { 
        std::string type = data->integral_types[i];
        for (size_t j = 0; j < data->ints[type].size(); j++) { 
            my_string.push_back( data->ints[type][j].to_string(type) );
        }
    }

    // amplitudes
    for (size_t i = 0; i < data->amplitude_types.size(); i++) { 
        char type = data->amplitude_types[i];
        for (size_t j = 0; j < data->amps[type].size(); j++) { 
            my_string.push_back( data->amps[type][j].to_string(type) );
        }
    }

    // bosons:
    for (size_t i = 0; i < data->is_boson_dagger.size(); i++) {
        if ( data->is_boson_dagger[i] ) {
            my_string.push_back("B*");
        }else {
            my_string.push_back("B");
        }
    }
    if ( data->has_w0 ) {
        my_string.push_back("w0");
    }

    return my_string;
}

bool pq::is_normal_order() {

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

bool pq::is_boson_normal_order() {

    if ( data->is_boson_dagger.size() == 1 ) {
        bool is_dagger_right = data->is_boson_dagger[0];
        bool is_dagger_left  = data->is_boson_dagger[0];
        if ( !is_dagger_right || is_dagger_left ) {
            skip = true; 
            return true;
        }
    }
    for (int i = 0; i < (int)data->is_boson_dagger.size()-1; i++) {

        // check if stings should be zero or not ... added 5/28/21
        bool is_dagger_right = data->is_boson_dagger[data->is_boson_dagger.size()-1];
        bool is_dagger_left  = data->is_boson_dagger[0];
        if ( !is_dagger_right || is_dagger_left ) {
            skip = true; 
            return true;
        }

        if ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] ) {
            return false;
        }
    }
    return true;

}

// in order to compare strings, the creation and annihilation 
// operators should be ordered in some consistent way.
// alphabetically seems reasonable enough
void pq::alphabetize(std::vector<std::shared_ptr<pq> > &ordered) {

    // alphabetize string
    for (size_t i = 0; i < ordered.size(); i++) {

        // creation
        bool not_alphabetized = false;
        do {
            not_alphabetized = false;
            int ndagger = 0;
            for (size_t j = 0; j < ordered[i]->symbol.size(); j++) {
                if ( ordered[i]->is_dagger[j] ) ndagger++;
            }
            for (int j = 0; j < ndagger-1; j++) {
                int val1 = ordered[i]->symbol[j].c_str()[0];
                int val2 = ordered[i]->symbol[j+1].c_str()[0];
                if ( val2 < val1 ) {
                    std::string dum = ordered[i]->symbol[j];
                    ordered[i]->symbol[j] = ordered[i]->symbol[j+1];
                    ordered[i]->symbol[j+1] = dum;
                    ordered[i]->sign = -ordered[i]->sign;
                    not_alphabetized = true;
                    j = ordered[i]->symbol.size() + 1;
                    not_alphabetized = true;
                }
            }
        }while(not_alphabetized);
        // annihilation
        not_alphabetized = false;
        do {
            not_alphabetized = false;
            int ndagger = 0;
            for (size_t j = 0; j < ordered[i]->symbol.size(); j++) {
                if ( ordered[i]->is_dagger[j] ) ndagger++;
            }
            for (int j = ndagger; j < (int)ordered[i]->symbol.size()-1; j++) {
                int val1 = ordered[i]->symbol[j].c_str()[0];
                int val2 = ordered[i]->symbol[j+1].c_str()[0];
                if ( val2 < val1 ) {
                    std::string dum = ordered[i]->symbol[j];
                    ordered[i]->symbol[j] = ordered[i]->symbol[j+1];
                    ordered[i]->symbol[j+1] = dum;
                    ordered[i]->sign = -ordered[i]->sign;
                    not_alphabetized = true;
                    j = ordered[i]->symbol.size() + 1;
                    not_alphabetized = true;
                }
            }
        }while(not_alphabetized);
    }

    // alphabetize deltas
    for (size_t i = 0; i < ordered.size(); i++) {
        for (size_t j = 0; j < ordered[i]->data->deltas.size(); j++) {
            int val1 = ordered[i]->data->deltas[j].labels[0].c_str()[0];
            int val2 = ordered[i]->data->deltas[j].labels[1].c_str()[0];
            if ( val2 < val1 ) {
                std::string dum = ordered[i]->data->deltas[j].labels[0];
                ordered[i]->data->deltas[j].labels[0] = ordered[i]->data->deltas[j].labels[1];
                ordered[i]->data->deltas[j].labels[1] = dum;
            }
        }
    }
}

void pq::swap_two_labels(std::string label1, std::string label2) {

    replace_index_everywhere(label1,"x");
    replace_index_everywhere(label2,label1);
    replace_index_everywhere("x",label2);

}

void pq::reorder_t_amplitudes() {

    size_t dim = data->amps['t'].size();

    if ( dim == 0 ) return;

    bool* nope = (bool*)malloc(dim * sizeof(bool));
    memset((void*)nope,'\0',dim * sizeof(bool));

    std::vector<std::vector<std::string> > tmp;

    std::vector<amplitudes> tmp_new;

    for (size_t order = 1; order < 7; order++) {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if ( nope[j] ) continue;

                if ( data->amps['t'][j].labels.size() == 2 * order ) {
                    tmp_new.push_back(data->amps['t'][j]);
                    nope[j] = true;
                    break;
                }

            }
        }
    }

    if ( dim != tmp_new.size() ) { 
        printf("\n");
        printf("    something went very wrong in reorder_t_amplitudes()\n");
        printf("    this function breaks for t6 and higher. why would\n");
        printf("    you want that, anyway?\n");
        printf("\n");
        exit(1);
    }

    for (int i = 0; i < dim; i++) {
        data->amps['t'][i].labels.clear();
        data->amps['t'][i].numerical_labels.clear();
    }
    data->amps['t'].clear();
    for (size_t i = 0; i < tmp_new.size(); i++) {
        data->amps['t'].push_back(tmp_new[i]);
    }

    free(nope);
    
}

// sort amplitude and integral labels
void pq::sort_labels() {

    for (size_t i = 0; i < data->integral_types.size(); i++) { 
        std::string type = data->integral_types[i];
        for (size_t j = 0; j < data->ints[type].size(); j++) {
            data->ints[type][j].sort();
        }
    }
    for (size_t i = 0; i < data->amplitude_types.size(); i++) { 
        char type = data->amplitude_types[i];
        for (size_t j = 0; j < data->amps[type].size(); j++) {
            data->amps[type][j].sort();
        }
    }
    for (size_t i = 0; i < data->deltas.size(); i++) { 
        data->deltas[i].sort();
    }

}

// reset spin labels
void pq::reset_spin_labels() {
 
    // amplitudes
    for (size_t i = 0; i < data->amplitude_types.size(); i++) {
        char type = data->amplitude_types[i];
        for (size_t j = 0; j < data->amps[type].size(); j++) {
            data->amps[type][j].spin_labels.clear();
            for (size_t k = 0; k < data->amps[type][j].labels.size(); k++) {
                data->amps[type][j].spin_labels.push_back("");
            }
        }
    }
    // integrals
    for (size_t i = 0; i < data->integral_types.size(); i++) {
        std::string type = data->integral_types[i];
        for (size_t j = 0; j < data->ints[type].size(); j++) {
            data->ints[type][j].spin_labels.clear();
            for (size_t k = 0; k < data->ints[type][j].labels.size(); k++) {
                data->ints[type][j].spin_labels.push_back("");
            }
        }
    }
    // deltas
    for (size_t i = 0; i < data->deltas.size(); i++) {
        data->deltas[i].spin_labels.clear();
        for (size_t j = 0; j < data->deltas[i].labels.size(); j++) {
            data->deltas[i].spin_labels.push_back("");
        }
    }

    std::vector<std::string> occ_labels { "i", "j", "k", "l", "m", "n", "o" };
    std::vector<std::string> vir_labels { "a", "b", "c", "d", "e", "f", "g" };

    // set spins for occupied non-summed labels
    for (size_t label = 0; label < occ_labels.size(); label++) {
        std::string spin = data->non_summed_spin_labels[occ_labels[label]];
        if ( spin == "a" || spin == "b" ) {
            // amplitudes
            for (size_t i = 0; i < data->amplitude_types.size(); i++) {
                char type = data->amplitude_types[i];
                for (size_t j = 0; j < data->amps[type].size(); j++) {
                    for (size_t k = 0; k < data->amps[type][j].labels.size(); k++) {
                        if ( data->amps[type][j].labels[k] == occ_labels[label] ) {
                            data->amps[type][j].spin_labels[k] = spin;
                        }
                    }
                }
            }
            // integrals
            for (size_t i = 0; i < data->integral_types.size(); i++) {
                std::string type = data->integral_types[i];
                for (size_t j = 0; j < data->ints[type].size(); j++) {
                    for (size_t k = 0; k < data->ints[type][j].labels.size(); k++) {
                        if ( data->ints[type][j].labels[k] == occ_labels[label] ) {
                            data->ints[type][j].spin_labels[k] = spin;
                        }
                    }
                }
            }
            // deltas
            for (size_t i = 0; i < data->deltas.size(); i++) {
                for (size_t j = 0; j < data->deltas[i].labels.size(); j++) {
                    if ( data->deltas[i].labels[j] == occ_labels[label] ) {
                        data->deltas[i].spin_labels[j] = spin;
                    }
                }
            }
        }
    }

    // set spins for virtual non-summed labels
    for (size_t label = 0; label < vir_labels.size(); label++) {
        std::string spin = data->non_summed_spin_labels[vir_labels[label]];
        if ( spin == "a" || spin == "b" ) {
            // amplitudes
            for (size_t i = 0; i < data->amplitude_types.size(); i++) {
                char type = data->amplitude_types[i];
                for (size_t j = 0; j < data->amps[type].size(); j++) {
                    for (size_t k = 0; k < data->amps[type][j].labels.size(); k++) {
                        if ( data->amps[type][j].labels[k] == vir_labels[label] ) {
                            data->amps[type][j].spin_labels[k] = spin;
                        }
                    }
                }
            }
            // integrals
            for (size_t i = 0; i < data->integral_types.size(); i++) {
                std::string type = data->integral_types[i];
                for (size_t j = 0; j < data->ints[type].size(); j++) {
                    for (size_t k = 0; k < data->ints[type][j].labels.size(); k++) {
                        if ( data->ints[type][j].labels[k] == vir_labels[label] ) {
                            data->ints[type][j].spin_labels[k] = spin;
                        }
                    }
                }
            }
            // deltas
            for (size_t i = 0; i < data->deltas.size(); i++) {
                for (size_t j = 0; j < data->deltas[i].labels.size(); j++) {
                    if ( data->deltas[i].labels[j] == vir_labels[label] ) {
                        data->deltas[i].spin_labels[j] = spin;
                    }   
                }   
            }
        }
    }
}

// expand sums to include spin and zero terms where appropriate
void pq::spin_blocking(std::vector<std::shared_ptr<pq> > &spin_blocked, std::map<std::string, std::string> spin_map) {

    // check that non-summed spin labels match those specified
    std::vector<std::string> occ_labels { "i", "j", "k", "l", "m", "n", "o" };
    std::vector<std::string> vir_labels { "a", "b", "c", "d", "e", "f", "g" };

    std::map<std::string, bool> found_labels;
    
    // ok, what non-summed labels do we have in the occupied space? 
    for (size_t j = 0; j < occ_labels.size(); j++) {
        int found = index_in_anywhere(occ_labels[j]);
        if ( found == 1 ) {
            found_labels[occ_labels[j]] = true;
        }else{
            found_labels[occ_labels[j]] = false;
        }
    }
    
    // ok, what non-summed labels do we have in the virtual space? 
    for (size_t j = 0; j < vir_labels.size(); j++) {
        int found = index_in_anywhere(vir_labels[j]);
        if ( found == 1 ) {
            found_labels[vir_labels[j]] = true;
        }else{
            found_labels[vir_labels[j]] = false;
        }
    }

    for (size_t j = 0; j < occ_labels.size(); j++) {
        if ( found_labels[occ_labels[j]] ) {
            if ( spin_map[occ_labels[j]] != "a" && spin_map[occ_labels[j]] != "b" ) {
                printf("\n");
                printf("    error: spin label for non-summed index %s is invalid\n", occ_labels[j].c_str());
                printf("\n");
                exit(1);
            }
        }
    }
    for (size_t j = 0; j < vir_labels.size(); j++) {
        if ( found_labels[vir_labels[j]] ) {
            if ( spin_map[vir_labels[j]] != "a" && spin_map[vir_labels[j]] != "b" ) {
                printf("\n");
                printf("    error: spin label for non-summed index %s is invalid\n", vir_labels[j].c_str());
                printf("\n");
                exit(1);
            }
        }
    }

    // non-summed spin labels
    data->non_summed_spin_labels = spin_map;

    // copy this term and zero spins

    std::shared_ptr<pq> newguy (new pq(vacuum));
    newguy->copy((void*)this);

    newguy->reset_spin_labels();

    // list of expanded sums
    std::vector< std::shared_ptr<pq> > tmp;
    tmp.push_back(newguy);

    for (size_t i = 0; i < tmp.size(); i++) {

        // but first expand permutations where spins don't match 
        size_t n = tmp[i]->data->permutations.size() / 2;

        for (size_t j = 0; j < n; j++) {

            std::string idx1 = tmp[i]->data->permutations[2*j];
            std::string idx2 = tmp[i]->data->permutations[2*j+1];

            // spin 1
            std::string spin1 = "";
            spin1 = tmp[i]->data->non_summed_spin_labels[idx1];

            // spin 2
            std::string spin2 = "";
            spin2 = tmp[i]->data->non_summed_spin_labels[idx2];

            // if spins are not the same, then the permutation needs to be expanded explicitly and allowed spins redetermined
            if ( spin1 != spin2 ) {

                // first guy is just a copy
                std::shared_ptr<pq> newguy1 (new pq(vacuum));
                newguy1->copy((void*)tmp[i].get());

                // second guy is a copy with permuted labels and change in sign
                std::shared_ptr<pq> newguy2 (new pq(vacuum));
                newguy2->copy((void*)tmp[i].get());
                newguy2->swap_two_labels(idx1, idx2);
                newguy2->sign *= -1;

                // reset non-summed spins for this guy
                newguy2->reset_spin_labels();

                // both guys need to have permutation lists adjusted
                newguy1->data->permutations.clear();
                newguy2->data->permutations.clear();

                for (size_t k = 0; k < n; k++) {

                    // skip jth permutation, which is the one we expanded
                    if ( j == k ) continue;

                    newguy1->data->permutations.push_back(tmp[i]->data->permutations[2*k]);
                    newguy1->data->permutations.push_back(tmp[i]->data->permutations[2*k+1]);

                    newguy2->data->permutations.push_back(tmp[i]->data->permutations[2*k]);
                    newguy2->data->permutations.push_back(tmp[i]->data->permutations[2*k+1]);
                }

                tmp[i]->skip = true;
                tmp.push_back(newguy1);
                tmp.push_back(newguy2);

                // break loop over permutations because this above logic only works on one permutation at a time
                break;
            }
        }
    }

    // now, expand sums 

    bool done_adding_spins = false;
    do {
        std::vector< std::shared_ptr<pq> > list;
        done_adding_spins = true;
        for (size_t i = 0; i < tmp.size(); i++) {
            bool am_i_done = tmp[i]->add_spins(list);
            if ( !am_i_done ) done_adding_spins = false;
        }
        if ( !done_adding_spins ) {
            tmp.clear();
            for (size_t i = 0; i < list.size(); i++) {
                if ( !list[i]->skip ) {
                    tmp.push_back(list[i]);
                }
            }
        }
    }while(!done_adding_spins);


    // kill terms that have mismatched spin 
    for (size_t i = 0; i < tmp.size(); i++) {

        if ( tmp[i]->skip ) continue;

        bool killit = false;

        // amplitudes
        // TODO: this logic only works for particle-conserving amplitudes
        for (size_t j = 0; j < data->amplitude_types.size(); j++) {
            char type = data->amplitude_types[j];
            for (size_t k = 0; k < tmp[i]->data->amps[type].size(); k++) {

                size_t order = tmp[i]->data->amps[type][k].labels.size()/2;

                int left_a = 0;
                int left_b = 0;
                int right_a = 0;
                int right_b = 0;
                for (size_t l = 0; l < order; l++) {
                    if ( tmp[i]->data->amps[type][k].spin_labels[l] == "a" ) {
                        left_a++;
                    }else {
                        left_b++;
                    }
                    if ( tmp[i]->data->amps[type][k].spin_labels[l+order] == "a" ) {
                        right_a++;
                    }else {
                        right_b++;
                    }
                }
                if (left_a != right_a || left_b != right_b ) {
                    killit = true;
                    break;
                }

            }
            if ( killit ) break;
        }
        if ( killit ) {
            tmp[i]->skip = true;
            continue;
        }

        killit = false;

        // integrals
        for (size_t j = 0; j < data->integral_types.size(); j++) {
            std::string type = data->integral_types[j];
            for (size_t k = 0; k < tmp[i]->data->ints[type].size(); k++) {
                size_t order = tmp[i]->data->ints[type][k].labels.size()/2;

                int left_a = 0;
                int left_b = 0;
                int right_a = 0;
                int right_b = 0;
                for (size_t l = 0; l < order; l++) {
                    if ( tmp[i]->data->ints[type][k].spin_labels[l] == "a" ) {
                        left_a++;
                    }else {
                        left_b++;
                    }
                    if ( tmp[i]->data->ints[type][k].spin_labels[l+order] == "a" ) {
                        right_a++;
                    }else {
                        right_b++;
                    }
                }
                if (left_a != right_a || left_b != right_b ) {
                    killit = true;
                    break;
                }

            }
            if ( killit ) break;
        }
        if ( killit ) {
            tmp[i]->skip = true;
            continue;
        }

        killit = false;

        // delta functions 
        for (size_t j = 0; j < data->deltas.size(); j++) {
            if ( tmp[i]->data->deltas[j].spin_labels[0] != tmp[i]->data->deltas[j].spin_labels[1] ) {
                killit = true;
                break;
            }
        }

        if ( killit ) {
            tmp[i]->skip = true;
            continue;
        }
    }

    
    // rearrange terms so that they have standard spin order (abba -> -abab, etc.)
    for (size_t p = 0; p < tmp.size(); p++) {

        if ( tmp[p]->skip ) continue;

        // amplitudes
        for (size_t i = 0; i < data->amplitude_types.size(); i++) {
            char type = data->amplitude_types[i];
            for (size_t j = 0; j < tmp[p]->data->amps[type].size(); j++) {
                size_t order = tmp[p]->data->amps[type][j].labels.size()/2;
                if ( order > 4 ) {
                    printf("\n");
                    printf("    error: spin tracing doesn't work for higher than quadruples yet\n");
                    printf("\n");
                    exit(1);
                }
                if ( order == 2 ) {

                    // three cases that require attention: ab;ba, ba;ab, and ba;ba

                    if (       tmp[p]->data->amps[type][j].spin_labels[0] == "a"
                            && tmp[p]->data->amps[type][j].spin_labels[1] == "b"
                            && tmp[p]->data->amps[type][j].spin_labels[2] == "b"
                            && tmp[p]->data->amps[type][j].spin_labels[3] == "a" ) {

                            std::string tmp_label = tmp[p]->data->amps[type][j].labels[2];
                            tmp[p]->data->amps[type][j].labels[2] = tmp[p]->data->amps[type][j].labels[3];
                            tmp[p]->data->amps[type][j].labels[3] = tmp_label;

                            tmp[p]->data->amps[type][j].spin_labels[2] = "a";
                            tmp[p]->data->amps[type][j].spin_labels[3] = "b";

                            tmp[p]->sign *= -1;

                    }else if ( tmp[p]->data->amps[type][j].spin_labels[0] == "b"
                            && tmp[p]->data->amps[type][j].spin_labels[1] == "a"
                            && tmp[p]->data->amps[type][j].spin_labels[2] == "a"
                            && tmp[p]->data->amps[type][j].spin_labels[3] == "b" ) {

                            std::string tmp_label = tmp[p]->data->amps[type][j].labels[0];
                            tmp[p]->data->amps[type][j].labels[0] = tmp[p]->data->amps[type][j].labels[1];
                            tmp[p]->data->amps[type][j].labels[1] = tmp_label;

                            tmp[p]->data->amps[type][j].spin_labels[0] = "a";
                            tmp[p]->data->amps[type][j].spin_labels[1] = "b";

                            tmp[p]->sign *= -1;


                    }else if ( tmp[p]->data->amps[type][j].spin_labels[0] == "b"
                            && tmp[p]->data->amps[type][j].spin_labels[1] == "a"
                            && tmp[p]->data->amps[type][j].spin_labels[2] == "b"
                            && tmp[p]->data->amps[type][j].spin_labels[3] == "a" ) {

                            std::string tmp_label = tmp[p]->data->amps[type][j].labels[0];
                            tmp[p]->data->amps[type][j].labels[0] = tmp[p]->data->amps[type][j].labels[1];
                            tmp[p]->data->amps[type][j].labels[1] = tmp_label;

                            tmp[p]->data->amps[type][j].spin_labels[0] = "a";
                            tmp[p]->data->amps[type][j].spin_labels[1] = "b";

                            tmp_label = tmp[p]->data->amps[type][j].labels[2];
                            tmp[p]->data->amps[type][j].labels[2] = tmp[p]->data->amps[type][j].labels[3];
                            tmp[p]->data->amps[type][j].labels[3] = tmp_label;

                            tmp[p]->data->amps[type][j].spin_labels[2] = "a";
                            tmp[p]->data->amps[type][j].spin_labels[3] = "b";

                    }
                }else if ( order == 3 ) {

                    // target order: aaa, aab, abb, bbb
                    int sign = 1;
                    reorder_three_spins(tmp[p]->data->amps[type][j], 0, 1, 2, sign);
                    reorder_three_spins(tmp[p]->data->amps[type][j], 3, 4, 5, sign);
                    tmp[p]->sign *= sign;

                }else if ( order == 4 ) {

                    // target order: aaaa, aaab, aabb, abbb, bbbb
                    int sign = 1;
                    reorder_four_spins(tmp[p]->data->amps[type][j], 0, 1, 2, 3, sign);
                    reorder_four_spins(tmp[p]->data->amps[type][j], 4, 5, 6, 7, sign);
                    tmp[p]->sign *= sign;

                }
            }
        }

        // integrals
        for (size_t i = 0; i < data->integral_types.size(); i++) {
            std::string type = data->integral_types[i];
            for (size_t j = 0; j < tmp[p]->data->ints[type].size(); j++) {

                size_t order = tmp[p]->data->ints[type][j].labels.size()/2;

                if ( order != 2 ) continue;

                // three cases that require attention: ab;ba, ba;ab, and ba;ba

                // integrals
                if (       tmp[p]->data->ints[type][j].spin_labels[0] == "a"
                        && tmp[p]->data->ints[type][j].spin_labels[1] == "b"
                        && tmp[p]->data->ints[type][j].spin_labels[2] == "b"
                        && tmp[p]->data->ints[type][j].spin_labels[3] == "a" ) {

                        std::string tmp_label = tmp[p]->data->ints[type][j].labels[2];
                        tmp[p]->data->ints[type][j].labels[2] = tmp[p]->data->ints[type][j].labels[3];
                        tmp[p]->data->ints[type][j].labels[3] = tmp_label;

                        tmp[p]->data->ints[type][j].spin_labels[2] = "a";
                        tmp[p]->data->ints[type][j].spin_labels[3] = "b";

                        tmp[p]->sign *= -1;

                }else if ( tmp[p]->data->ints[type][j].spin_labels[0] == "b"
                        && tmp[p]->data->ints[type][j].spin_labels[1] == "a"
                        && tmp[p]->data->ints[type][j].spin_labels[2] == "a"
                        && tmp[p]->data->ints[type][j].spin_labels[3] == "b" ) {

                        std::string tmp_label = tmp[p]->data->ints[type][j].labels[0];
                        tmp[p]->data->ints[type][j].labels[0] = tmp[p]->data->ints[type][j].labels[1];
                        tmp[p]->data->ints[type][j].labels[1] = tmp_label;

                        tmp[p]->data->ints[type][j].spin_labels[0] = "a";
                        tmp[p]->data->ints[type][j].spin_labels[1] = "b";

                        tmp[p]->sign *= -1;


                }else if ( tmp[p]->data->ints[type][j].spin_labels[0] == "b"
                        && tmp[p]->data->ints[type][j].spin_labels[1] == "a"
                        && tmp[p]->data->ints[type][j].spin_labels[2] == "b"
                        && tmp[p]->data->ints[type][j].spin_labels[3] == "a" ) {

                        std::string tmp_label = tmp[p]->data->ints[type][j].labels[0];
                        tmp[p]->data->ints[type][j].labels[0] = tmp[p]->data->ints[type][j].labels[1];
                        tmp[p]->data->ints[type][j].labels[1] = tmp_label;

                        tmp[p]->data->ints[type][j].spin_labels[0] = "a";
                        tmp[p]->data->ints[type][j].spin_labels[1] = "b";

                        tmp_label = tmp[p]->data->ints[type][j].labels[2];
                        tmp[p]->data->ints[type][j].labels[2] = tmp[p]->data->ints[type][j].labels[3];
                        tmp[p]->data->ints[type][j].labels[3] = tmp_label;

                        tmp[p]->data->ints[type][j].spin_labels[2] = "a";
                        tmp[p]->data->ints[type][j].spin_labels[3] = "b";

                }
            }
        }
    }

    // 
    for (size_t i = 0; i < tmp.size(); i++) {
        if ( tmp[i]->skip ) continue;
        spin_blocked.push_back(tmp[i]);
    }

    tmp.clear();

}

// reorder four spins ... cases to consider: aaba/abaa/baaa -> aaab; baab/abba/baba/bbaa/abab -> aabb; babb/bbab/bbba -> abbb

void pq::reorder_four_spins(amplitudes & amps, int i1, int i2, int i3, int i4, int & sign) {

    // aaba/abaa/baaa -> aaab
    if (       amps.spin_labels[i1] == "a"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "b" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i3];
            amps.labels[i3] = tmp_label;

            amps.spin_labels[i3] = "a";
            amps.spin_labels[i4] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "a"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "a" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.spin_labels[i2] = "a";
            amps.spin_labels[i4] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "a" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i4] = "b";

            sign *= -1;

    // baab/abba/baba/bbaa/abab -> aabb
    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "a" 
            && amps.spin_labels[i4] == "b" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i3] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "a"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "b" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.spin_labels[i2] = "a";
            amps.spin_labels[i4] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "b" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i4] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "a" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.spin_labels[i2] = "a";
            amps.spin_labels[i4] = "b";

            tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i3] = "b";

    }else if ( amps.spin_labels[i1] == "a"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "a"
            && amps.spin_labels[i4] == "b" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.spin_labels[i2] = "a";
            amps.spin_labels[i3] = "b";

            sign *= -1;

    // babb/bbab/bbba -> abbb
    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "b" 
            && amps.spin_labels[i4] == "b" ) {

            std::string tmp_label = amps.labels[i2];

            amps.labels[i2] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i2] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "a" 
            && amps.spin_labels[i4] == "b" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i3] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "b" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i4] = "b";

            sign *= -1;

    }

}

// reorder three spins ... cases to consider: aba/baa -> aab; bba/bab -> abb

void pq::reorder_three_spins(amplitudes & amps, int i1, int i2, int i3, int & sign) {

    if (       amps.spin_labels[i1] == "a"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "a" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.spin_labels[i2] = "a";
            amps.spin_labels[i3] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "a" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i3] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "a" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i3] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "b" ) {

            std::string tmp_label = amps.labels[i2];

            amps.labels[i2] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i2] = "b";

            sign *= -1;

    }

}


bool pq::add_spins(std::vector<std::shared_ptr<pq> > &list) {

    if ( skip ) return true;

    bool all_spins_added = false;

    // amplitudes
    for (size_t i = 0; i < data->amplitude_types.size(); i++) {
        char type = data->amplitude_types[i];
        for (size_t j = 0; j < data->amps[type].size(); j++) {
            for (size_t k = 0; k < data->amps[type][j].labels.size(); k++) {
                if ( data->amps[type][j].spin_labels[k] == "" ) {

                    std::shared_ptr<pq> sa (new pq(vacuum));
                    std::shared_ptr<pq> sb (new pq(vacuum));

                    sa->copy((void*)this);
                    sb->copy((void*)this);

                    sa->set_spin_everywhere(data->amps[type][j].labels[k], "a");
                    sb->set_spin_everywhere(data->amps[type][j].labels[k], "b");

                    //sa->data->amps[type][j].spin_labels[k] = "a";
                    //sb->data->amps[type][j].spin_labels[k] = "b";

                    list.push_back(sa);
                    list.push_back(sb);
                    return false;

                }
            }
        }
    }

    // integrals
    for (size_t i = 0; i < data->integral_types.size(); i++) {
        std::string type = data->integral_types[i];
        for (size_t j = 0; j < data->ints[type].size(); j++) {
            for (size_t k = 0; k < data->ints[type][j].labels.size(); k++) {
                if ( data->ints[type][j].spin_labels[k] == "" ) {

                    std::shared_ptr<pq> sa (new pq(vacuum));
                    std::shared_ptr<pq> sb (new pq(vacuum));

                    sa->copy((void*)this);
                    sb->copy((void*)this);

                    sa->set_spin_everywhere(data->ints[type][j].labels[k], "a");
                    sb->set_spin_everywhere(data->ints[type][j].labels[k], "b");

                    //sa->data->ints[type][j].spin_labels[k] = "a";
                    //sb->data->ints[type][j].spin_labels[k] = "b";

                    list.push_back(sa);
                    list.push_back(sb);
                    return false;

                }
            }
        }
    }

    // must be done.
    return true;

}

// compare strings and remove terms that cancel

void pq::cleanup(std::vector<std::shared_ptr<pq> > &ordered) {


    for (size_t i = 0; i < ordered.size(); i++) {

        // order amplitudes such that they're ordered t1, t2, t3, etc.
        ordered[i]->reorder_t_amplitudes();

        // sort amplitude labels
        ordered[i]->sort_labels();

    }

    // prune list so it only contains non-skipped ones
    std::vector< std::shared_ptr<pq> > pruned;
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]-> skip ) continue;

        // for normal order relative to fermi vacuum, i doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming
        if ( vacuum == "FERMI" ) {
            if ( ordered[i]->symbol.size() != 0 ) continue;
            if ( ordered[i]->data->is_boson_dagger.size() != 0 ) continue;
        }

        pruned.push_back(ordered[i]);
    }
    ordered.clear();
    for (size_t i = 0; i < pruned.size(); i++) {
        ordered.push_back(pruned[i]);
    }
    pruned.clear();

    //printf("starting string comparisons\n");fflush(stdout);

    std::vector<std::string> occ_labels { "i", "j", "k", "l", "m", "n", "o" };
    std::vector<std::string> vir_labels { "a", "b", "c", "d", "e", "f", "g" };

    consolidate_permutations(ordered);

    consolidate_permutations_plus_swap(ordered,occ_labels);
    consolidate_permutations_plus_swap(ordered,vir_labels);

    consolidate_permutations_plus_two_swaps(ordered,occ_labels,occ_labels);
    consolidate_permutations_plus_two_swaps(ordered,vir_labels,vir_labels);
    consolidate_permutations_plus_two_swaps(ordered,occ_labels,vir_labels);

    // these don't seem to be necessary for test cases up to ccsdtq
/*
    consolidate_permutations_plus_three_swaps(ordered,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_three_swaps(ordered,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_three_swaps(ordered,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_three_swaps(ordered,vir_labels,vir_labels,vir_labels);

    consolidate_permutations_plus_four_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_four_swaps(ordered,occ_labels,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_four_swaps(ordered,occ_labels,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_four_swaps(ordered,occ_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_four_swaps(ordered,vir_labels,vir_labels,vir_labels,vir_labels);

    consolidate_permutations_plus_five_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_five_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_five_swaps(ordered,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_five_swaps(ordered,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_five_swaps(ordered,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_five_swaps(ordered,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);

    consolidate_permutations_plus_six_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_six_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_six_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_six_swaps(ordered,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_six_swaps(ordered,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_six_swaps(ordered,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_six_swaps(ordered,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);

    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_seven_swaps(ordered,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);

    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
*/

    // probably only relevant for vacuum = fermi
    if ( vacuum != "FERMI" ) return;

    consolidate_permutations_non_summed(ordered,occ_labels);
    consolidate_permutations_non_summed(ordered,vir_labels);

    // re-prune
    pruned.clear();
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        // for normal order relative to fermi vacuum, i doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming
        if ( vacuum == "FERMI" ) {
            if ( ordered[i]->symbol.size() != 0 ) continue;
            if ( ordered[i]->data->is_boson_dagger.size() != 0 ) continue;
        }

        pruned.push_back(ordered[i]);
    }
    ordered.clear();
    for (size_t i = 0; i < pruned.size(); i++) {
        ordered.push_back(pruned[i]);
    }
    pruned.clear();

}

// consolidate terms that differ by permutations of non-summed labels
void pq::consolidate_permutations_non_summed(
    std::vector<std::shared_ptr<pq> > &ordered,
    std::vector<std::string> labels) {


    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_idx;

        // ok, what labels do we have? 
        for (size_t j = 0; j < labels.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels[j]);
            // this is buggy when existing permutation labels belong to 
            // the same space as the labels we're permuting ... so skip those for now.
            bool same_space = false;
            bool is_occ1 = is_occ(labels[j]);
            for (size_t k = 0; k < ordered[i]->data->permutations.size(); k++) {
                bool is_occ2 = is_occ(ordered[i]->data->permutations[k]);
                if ( is_occ1 && is_occ2 ) {
                    same_space = true;
                    break;
                }else if ( !is_occ1 && !is_occ2 ) {
                    same_space = true;
                    break;
                }
            }

            if ( !same_space ) {
                find_idx.push_back(found);
            }else{
                find_idx.push_back(0);
            }
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            std::string permutation_1 = "";
            std::string permutation_2 = "";

            // try swapping non-summed labels
            for (size_t id1 = 0; id1 < labels.size(); id1++) {
                if ( find_idx[id1] != 1 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels.size(); id2++) {
                    if ( find_idx[id2] != 1 ) continue;

                    std::shared_ptr<pq> newguy (new pq(vacuum));
                    newguy->copy((void*)(ordered[i].get()));
                    newguy->swap_two_labels(labels[id1],labels[id2]);

                    strings_same = compare_strings(ordered[j],newguy,n_permute);

                    if ( strings_same ) {

                        permutation_1 = labels[id1];
                        permutation_2 = labels[id2];
                        break;

                    }
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            // it is possible to have made it through the previous logic without 
            // assigning permutation labels, if strings are identical but 
            // permutation operators differ
            //if ( permutation_1 == "" || permutation_2 == "" ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, then this is a permutation
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->data->permutations.push_back(permutation_1);
                ordered[i]->data->permutations.push_back(permutation_2);
                ordered[j]->skip = true;
                break;
            }

            // otherwise, something has gone wrong in the previous consolidation step...
        }
    }
}


// consolidate terms that differ by eight summed labels plus permutations
void pq::consolidate_permutations_plus_eight_swaps(
    std::vector<std::shared_ptr<pq> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2, 
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4,
    std::vector<std::string> labels_5,
    std::vector<std::string> labels_6,
    std::vector<std::string> labels_7,
    std::vector<std::string> labels_8) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;
        std::vector<int> find_3;
        std::vector<int> find_4;
        std::vector<int> find_5;
        std::vector<int> find_6;
        std::vector<int> find_7;
        std::vector<int> find_8;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_3[j]);
            find_3.push_back(found);
        }

        // ok, what labels do we have? list 4
        for (size_t j = 0; j < labels_4.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_4[j]);
            find_4.push_back(found);
        }

        // ok, what labels do we have? list 5
        for (size_t j = 0; j < labels_5.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_5[j]);
            find_5.push_back(found);
        }

        // ok, what labels do we have? list 6
        for (size_t j = 0; j < labels_6.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_6[j]);
            find_6.push_back(found);
        }

        // ok, what labels do we have? list 7
        for (size_t j = 0; j < labels_7.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_7[j]);
            find_7.push_back(found);
        }

        // ok, what labels do we have? list 8
        for (size_t j = 0; j < labels_8.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_8[j]);
            find_8.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels 1
            for (size_t id1 = 0; id1 < labels_1.size(); id1++) {
                if ( find_1[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels_1.size(); id2++) {
                    if ( find_1[id2] != 2 ) continue;

                    // try swapping non-summed labels 2
                    for (size_t id3 = 0; id3 < labels_2.size(); id3++) {
                        if ( find_2[id3] != 2 ) continue;
                        for (size_t id4 = id3 + 1; id4 < labels_2.size(); id4++) {
                            if ( find_2[id4] != 2 ) continue;

                            // try swapping non-summed labels 3
                            for (size_t id5 = 0; id5 < labels_3.size(); id5++) {
                                if ( find_3[id5] != 2 ) continue;
                                for (size_t id6 = id5 + 1; id6 < labels_3.size(); id6++) {
                                    if ( find_3[id6] != 2 ) continue;

                                    // try swapping non-summed labels 4
                                    for (size_t id7 = 0; id7 < labels_4.size(); id7++) {
                                        if ( find_4[id7] != 2 ) continue;
                                        for (size_t id8 = id7 + 1; id8 < labels_4.size(); id8++) {
                                            if ( find_4[id8] != 2 ) continue;

                                            // try swapping non-summed labels 5
                                            for (size_t id9 = 0; id9 < labels_5.size(); id9++) {
                                                if ( find_5[id9] != 2 ) continue;
                                                for (size_t id10 = id9 + 1; id10 < labels_5.size(); id10++) {
                                                    if ( find_5[id10] != 2 ) continue;

                                                    // try swapping non-summed labels 6
                                                    for (size_t id11 = 0; id11 < labels_6.size(); id11++) {
                                                        if ( find_6[id11] != 2 ) continue;
                                                        for (size_t id12 = id11 + 1; id12 < labels_6.size(); id12++) {
                                                            if ( find_6[id12] != 2 ) continue;

                                                            // try swapping non-summed labels 7
                                                            for (size_t id13 = 0; id13 < labels_7.size(); id13++) {
                                                                if ( find_7[id13] != 2 ) continue;
                                                                for (size_t id14 = id13 + 1; id14 < labels_7.size(); id14++) {
                                                                    if ( find_7[id14] != 2 ) continue;

                                                                    // try swapping non-summed labels 8
                                                                    for (size_t id15 = 0; id15 < labels_8.size(); id15++) {
                                                                        if ( find_8[id15] != 2 ) continue;
                                                                        for (size_t id16 = id15 + 1; id16 < labels_8.size(); id16++) {
                                                                            if ( find_8[id16] != 2 ) continue;

                                                                            std::shared_ptr<pq> newguy (new pq(vacuum));
                                                                            newguy->copy((void*)(ordered[i].get()));
                                                                            newguy->swap_two_labels(labels_1[id1],labels_1[id2]);
                                                                            newguy->swap_two_labels(labels_2[id3],labels_2[id4]);
                                                                            newguy->swap_two_labels(labels_3[id5],labels_3[id6]);
                                                                            newguy->swap_two_labels(labels_4[id7],labels_4[id8]);
                                                                            newguy->swap_two_labels(labels_5[id9],labels_5[id10]);
                                                                            newguy->swap_two_labels(labels_6[id11],labels_6[id12]);
                                                                            newguy->swap_two_labels(labels_7[id13],labels_7[id14]);
                                                                            newguy->swap_two_labels(labels_8[id15],labels_8[id16]);
                                                                            newguy->sort_labels();
                                                                            strings_same = compare_strings(ordered[j],newguy,n_permute);

                                                                            if ( strings_same ) break;
                                                                        }
                                                                        if ( strings_same ) break;
                                                                    }
                                                                    if ( strings_same ) break;
                                                                }
                                                                if ( strings_same ) break;
                                                            }
                                                            if ( strings_same ) break;
                                                        }
                                                        if ( strings_same ) break;
                                                    }
                                                    if ( strings_same ) break;
                                                }
                                                if ( strings_same ) break;
                                            }
                                            if ( strings_same ) break;
                                        }
                                        if ( strings_same ) break;
                                    }
                                    if ( strings_same ) break;
                                }
                                if ( strings_same ) break;
                            }
                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

// consolidate terms that differ by seven summed labels plus permutations
void pq::consolidate_permutations_plus_seven_swaps(
    std::vector<std::shared_ptr<pq> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2, 
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4,
    std::vector<std::string> labels_5,
    std::vector<std::string> labels_6,
    std::vector<std::string> labels_7) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;
        std::vector<int> find_3;
        std::vector<int> find_4;
        std::vector<int> find_5;
        std::vector<int> find_6;
        std::vector<int> find_7;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_3[j]);
            find_3.push_back(found);
        }

        // ok, what labels do we have? list 4
        for (size_t j = 0; j < labels_4.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_4[j]);
            find_4.push_back(found);
        }

        // ok, what labels do we have? list 5
        for (size_t j = 0; j < labels_5.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_5[j]);
            find_5.push_back(found);
        }

        // ok, what labels do we have? list 6
        for (size_t j = 0; j < labels_6.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_6[j]);
            find_6.push_back(found);
        }

        // ok, what labels do we have? list 7
        for (size_t j = 0; j < labels_7.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_7[j]);
            find_7.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels 1
            for (size_t id1 = 0; id1 < labels_1.size(); id1++) {
                if ( find_1[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels_1.size(); id2++) {
                    if ( find_1[id2] != 2 ) continue;

                    // try swapping non-summed labels 2
                    for (size_t id3 = 0; id3 < labels_2.size(); id3++) {
                        if ( find_2[id3] != 2 ) continue;
                        for (size_t id4 = id3 + 1; id4 < labels_2.size(); id4++) {
                            if ( find_2[id4] != 2 ) continue;

                            // try swapping non-summed labels 3
                            for (size_t id5 = 0; id5 < labels_3.size(); id5++) {
                                if ( find_3[id5] != 2 ) continue;
                                for (size_t id6 = id5 + 1; id6 < labels_3.size(); id6++) {
                                    if ( find_3[id6] != 2 ) continue;

                                    // try swapping non-summed labels 4
                                    for (size_t id7 = 0; id7 < labels_4.size(); id7++) {
                                        if ( find_4[id7] != 2 ) continue;
                                        for (size_t id8 = id7 + 1; id8 < labels_4.size(); id8++) {
                                            if ( find_4[id8] != 2 ) continue;

                                            // try swapping non-summed labels 5
                                            for (size_t id9 = 0; id9 < labels_5.size(); id9++) {
                                                if ( find_5[id9] != 2 ) continue;
                                                for (size_t id10 = id9 + 1; id10 < labels_5.size(); id10++) {
                                                    if ( find_5[id10] != 2 ) continue;

                                                    // try swapping non-summed labels 6
                                                    for (size_t id11 = 0; id11 < labels_6.size(); id11++) {
                                                        if ( find_6[id11] != 2 ) continue;
                                                        for (size_t id12 = id11 + 1; id12 < labels_6.size(); id12++) {
                                                            if ( find_6[id12] != 2 ) continue;

                                                            // try swapping non-summed labels 7
                                                            for (size_t id13 = 0; id13 < labels_7.size(); id13++) {
                                                                if ( find_7[id13] != 2 ) continue;
                                                                for (size_t id14 = id13 + 1; id14 < labels_7.size(); id14++) {
                                                                    if ( find_7[id14] != 2 ) continue;

                                                                    std::shared_ptr<pq> newguy (new pq(vacuum));
                                                                    newguy->copy((void*)(ordered[i].get()));
                                                                    newguy->swap_two_labels(labels_1[id1],labels_1[id2]);
                                                                    newguy->swap_two_labels(labels_2[id3],labels_2[id4]);
                                                                    newguy->swap_two_labels(labels_3[id5],labels_3[id6]);
                                                                    newguy->swap_two_labels(labels_4[id7],labels_4[id8]);
                                                                    newguy->swap_two_labels(labels_5[id9],labels_5[id10]);
                                                                    newguy->swap_two_labels(labels_6[id11],labels_6[id12]);
                                                                    newguy->swap_two_labels(labels_7[id13],labels_7[id14]);
                                                                    newguy->sort_labels();
                                                                    strings_same = compare_strings(ordered[j],newguy,n_permute);

                                                                    if ( strings_same ) break;
                                                                }
                                                                if ( strings_same ) break;
                                                            }
                                                            if ( strings_same ) break;
                                                        }
                                                        if ( strings_same ) break;
                                                    }
                                                    if ( strings_same ) break;
                                                }
                                                if ( strings_same ) break;
                                            }
                                            if ( strings_same ) break;
                                        }
                                        if ( strings_same ) break;
                                    }
                                    if ( strings_same ) break;
                                }
                                if ( strings_same ) break;
                            }
                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

// consolidate terms that differ by six summed labels plus permutations
void pq::consolidate_permutations_plus_six_swaps(
    std::vector<std::shared_ptr<pq> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2, 
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4,
    std::vector<std::string> labels_5,
    std::vector<std::string> labels_6) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;
        std::vector<int> find_3;
        std::vector<int> find_4;
        std::vector<int> find_5;
        std::vector<int> find_6;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_3[j]);
            find_3.push_back(found);
        }

        // ok, what labels do we have? list 4
        for (size_t j = 0; j < labels_4.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_4[j]);
            find_4.push_back(found);
        }

        // ok, what labels do we have? list 5
        for (size_t j = 0; j < labels_5.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_5[j]);
            find_5.push_back(found);
        }

        // ok, what labels do we have? list 6
        for (size_t j = 0; j < labels_6.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_6[j]);
            find_6.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels 1
            for (size_t id1 = 0; id1 < labels_1.size(); id1++) {
                if ( find_1[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels_1.size(); id2++) {
                    if ( find_1[id2] != 2 ) continue;

                    // try swapping non-summed labels 2
                    for (size_t id3 = 0; id3 < labels_2.size(); id3++) {
                        if ( find_2[id3] != 2 ) continue;
                        for (size_t id4 = id3 + 1; id4 < labels_2.size(); id4++) {
                            if ( find_2[id4] != 2 ) continue;

                            // try swapping non-summed labels 3
                            for (size_t id5 = 0; id5 < labels_3.size(); id5++) {
                                if ( find_3[id5] != 2 ) continue;
                                for (size_t id6 = id5 + 1; id6 < labels_3.size(); id6++) {
                                    if ( find_3[id6] != 2 ) continue;

                                    // try swapping non-summed labels 4
                                    for (size_t id7 = 0; id7 < labels_4.size(); id7++) {
                                        if ( find_4[id7] != 2 ) continue;
                                        for (size_t id8 = id7 + 1; id8 < labels_4.size(); id8++) {
                                            if ( find_4[id8] != 2 ) continue;

                                            // try swapping non-summed labels 5
                                            for (size_t id9 = 0; id9 < labels_5.size(); id9++) {
                                                if ( find_5[id9] != 2 ) continue;
                                                for (size_t id10 = id9 + 1; id10 < labels_5.size(); id10++) {
                                                    if ( find_5[id10] != 2 ) continue;

                                                    // try swapping non-summed labels 6
                                                    for (size_t id11 = 0; id11 < labels_6.size(); id11++) {
                                                        if ( find_6[id11] != 2 ) continue;
                                                        for (size_t id12 = id11 + 1; id12 < labels_6.size(); id12++) {
                                                            if ( find_6[id12] != 2 ) continue;

                                                            std::shared_ptr<pq> newguy (new pq(vacuum));
                                                            newguy->copy((void*)(ordered[i].get()));
                                                            newguy->swap_two_labels(labels_1[id1],labels_1[id2]);
                                                            newguy->swap_two_labels(labels_2[id3],labels_2[id4]);
                                                            newguy->swap_two_labels(labels_3[id5],labels_3[id6]);
                                                            newguy->swap_two_labels(labels_4[id7],labels_4[id8]);
                                                            newguy->swap_two_labels(labels_5[id9],labels_5[id10]);
                                                            newguy->swap_two_labels(labels_6[id11],labels_6[id12]);
                                                            newguy->sort_labels();
                                                            strings_same = compare_strings(ordered[j],newguy,n_permute);

                                                            if ( strings_same ) break;
                                                        }
                                                        if ( strings_same ) break;
                                                    }
                                                    if ( strings_same ) break;
                                                }
                                                if ( strings_same ) break;
                                            }
                                            if ( strings_same ) break;
                                        }
                                        if ( strings_same ) break;
                                    }
                                    if ( strings_same ) break;
                                }
                                if ( strings_same ) break;
                            }
                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

// consolidate terms that differ by five summed labels plus permutations
void pq::consolidate_permutations_plus_five_swaps(
    std::vector<std::shared_ptr<pq> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2, 
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4,
    std::vector<std::string> labels_5) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;
        std::vector<int> find_3;
        std::vector<int> find_4;
        std::vector<int> find_5;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_3[j]);
            find_3.push_back(found);
        }

        // ok, what labels do we have? list 4
        for (size_t j = 0; j < labels_4.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_4[j]);
            find_4.push_back(found);
        }

        // ok, what labels do we have? list 5
        for (size_t j = 0; j < labels_5.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_5[j]);
            find_5.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels 1
            for (size_t id1 = 0; id1 < labels_1.size(); id1++) {
                if ( find_1[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels_1.size(); id2++) {
                    if ( find_1[id2] != 2 ) continue;

                    // try swapping non-summed labels 2
                    for (size_t id3 = 0; id3 < labels_2.size(); id3++) {
                        if ( find_2[id3] != 2 ) continue;
                        for (size_t id4 = id3 + 1; id4 < labels_2.size(); id4++) {
                            if ( find_2[id4] != 2 ) continue;

                            // try swapping non-summed labels 3
                            for (size_t id5 = 0; id5 < labels_3.size(); id5++) {
                                if ( find_3[id5] != 2 ) continue;
                                for (size_t id6 = id5 + 1; id6 < labels_3.size(); id6++) {
                                    if ( find_3[id6] != 2 ) continue;

                                    // try swapping non-summed labels 4
                                    for (size_t id7 = 0; id7 < labels_4.size(); id7++) {
                                        if ( find_4[id7] != 2 ) continue;
                                        for (size_t id8 = id7 + 1; id8 < labels_4.size(); id8++) {
                                            if ( find_4[id8] != 2 ) continue;

                                            // try swapping non-summed labels 5
                                            for (size_t id9 = 0; id9 < labels_5.size(); id9++) {
                                                if ( find_5[id9] != 2 ) continue;
                                                for (size_t id10 = id9 + 1; id10 < labels_5.size(); id10++) {
                                                    if ( find_5[id10] != 2 ) continue;

                                                    std::shared_ptr<pq> newguy (new pq(vacuum));
                                                    newguy->copy((void*)(ordered[i].get()));
                                                    newguy->swap_two_labels(labels_1[id1],labels_1[id2]);
                                                    newguy->swap_two_labels(labels_2[id3],labels_2[id4]);
                                                    newguy->swap_two_labels(labels_3[id5],labels_3[id6]);
                                                    newguy->swap_two_labels(labels_4[id7],labels_4[id8]);
                                                    newguy->swap_two_labels(labels_5[id9],labels_5[id10]);
                                                    newguy->sort_labels();
                                                    strings_same = compare_strings(ordered[j],newguy,n_permute);

                                                    if ( strings_same ) break;
                                                }
                                                if ( strings_same ) break;
                                            }
                                            if ( strings_same ) break;
                                        }
                                        if ( strings_same ) break;
                                    }
                                    if ( strings_same ) break;
                                }
                                if ( strings_same ) break;
                            }
                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

// consolidate terms that differ by four summed labels plus permutations
void pq::consolidate_permutations_plus_four_swaps(
    std::vector<std::shared_ptr<pq> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2, 
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;
        std::vector<int> find_3;
        std::vector<int> find_4;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_3[j]);
            find_3.push_back(found);
        }

        // ok, what labels do we have? list 4
        for (size_t j = 0; j < labels_4.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_4[j]);
            find_4.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels 1
            for (size_t id1 = 0; id1 < labels_1.size(); id1++) {
                if ( find_1[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels_1.size(); id2++) {
                    if ( find_1[id2] != 2 ) continue;

                    // try swapping non-summed labels 2
                    for (size_t id3 = 0; id3 < labels_2.size(); id3++) {
                        if ( find_2[id3] != 2 ) continue;
                        for (size_t id4 = id3 + 1; id4 < labels_2.size(); id4++) {
                            if ( find_2[id4] != 2 ) continue;

                            // try swapping non-summed labels 3
                            for (size_t id5 = 0; id5 < labels_3.size(); id5++) {
                                if ( find_3[id5] != 2 ) continue;
                                for (size_t id6 = id5 + 1; id6 < labels_3.size(); id6++) {
                                    if ( find_3[id6] != 2 ) continue;

                                    // try swapping non-summed labels 4
                                    for (size_t id7 = 0; id7 < labels_4.size(); id7++) {
                                        if ( find_4[id7] != 2 ) continue;
                                        for (size_t id8 = id7 + 1; id8 < labels_4.size(); id8++) {
                                            if ( find_4[id8] != 2 ) continue;

                                            std::shared_ptr<pq> newguy (new pq(vacuum));
                                            newguy->copy((void*)(ordered[i].get()));
                                            newguy->swap_two_labels(labels_1[id1],labels_1[id2]);
                                            newguy->swap_two_labels(labels_2[id3],labels_2[id4]);
                                            newguy->swap_two_labels(labels_3[id5],labels_3[id6]);
                                            newguy->swap_two_labels(labels_4[id7],labels_4[id8]);
                                            newguy->sort_labels();
                                            strings_same = compare_strings(ordered[j],newguy,n_permute);

                                            if ( strings_same ) break;
                                        }
                                        if ( strings_same ) break;
                                    }
                                    if ( strings_same ) break;
                                }
                                if ( strings_same ) break;
                            }
                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

// consolidate terms that differ by three summed labels plus permutations
void pq::consolidate_permutations_plus_three_swaps(
    std::vector<std::shared_ptr<pq> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2, 
    std::vector<std::string> labels_3) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;
        std::vector<int> find_3;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_3[j]);
            find_3.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels 1
            for (size_t id1 = 0; id1 < labels_1.size(); id1++) {
                if ( find_1[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels_1.size(); id2++) {
                    if ( find_1[id2] != 2 ) continue;

                    // try swapping non-summed labels 2
                    for (size_t id3 = 0; id3 < labels_2.size(); id3++) {
                        if ( find_2[id3] != 2 ) continue;
                        for (size_t id4 = id3 + 1; id4 < labels_2.size(); id4++) {
                            if ( find_2[id4] != 2 ) continue;

                            // try swapping non-summed labels 3
                            for (size_t id5 = 0; id5 < labels_3.size(); id5++) {
                                if ( find_3[id5] != 2 ) continue;
                                for (size_t id6 = id5 + 1; id6 < labels_3.size(); id6++) {
                                    if ( find_3[id6] != 2 ) continue;

                                    std::shared_ptr<pq> newguy (new pq(vacuum));
                                    newguy->copy((void*)(ordered[i].get()));
                                    newguy->swap_two_labels(labels_1[id1],labels_1[id2]);
                                    newguy->swap_two_labels(labels_2[id3],labels_2[id4]);
                                    newguy->swap_two_labels(labels_3[id5],labels_3[id6]);
                                    newguy->sort_labels();
                                    strings_same = compare_strings(ordered[j],newguy,n_permute);

                                    if ( strings_same ) break;
                                }
                                if ( strings_same ) break;
                            }
                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

// consolidate terms that differ by two summed labels plus permutations
void pq::consolidate_permutations_plus_two_swaps(
    std::vector<std::shared_ptr<pq> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_2[j]);
            find_2.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels 1
            for (size_t id1 = 0; id1 < labels_1.size(); id1++) {
                if ( find_1[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels_1.size(); id2++) {
                    if ( find_1[id2] != 2 ) continue;

                    // try swapping non-summed labels 2
                    for (size_t id3 = 0; id3 < labels_2.size(); id3++) {
                        if ( find_2[id3] != 2 ) continue;
                        for (size_t id4 = id3 + 1; id4 < labels_2.size(); id4++) {
                            if ( find_2[id4] != 2 ) continue;

                            std::shared_ptr<pq> newguy (new pq(vacuum));
                            newguy->copy((void*)(ordered[i].get()));
                            newguy->swap_two_labels(labels_1[id1],labels_1[id2]);
                            newguy->swap_two_labels(labels_2[id3],labels_2[id4]);
                            newguy->sort_labels();

                            strings_same = compare_strings(ordered[j],newguy,n_permute);

                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}


// consolidate terms that differ by summed labels plus permutations
void pq::consolidate_permutations_plus_swap(std::vector<std::shared_ptr<pq> > &ordered,
                                            std::vector<std::string> labels) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_idx;

        // ok, what labels do we have?
        for (size_t j = 0; j < labels.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels[j]);
            find_idx.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels
            for (size_t id1 = 0; id1 < labels.size(); id1++) {
                if ( find_idx[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels.size(); id2++) {
                    if ( find_idx[id2] != 2 ) continue;

                    std::shared_ptr<pq> newguy (new pq(vacuum));
                    newguy->copy((void*)(ordered[i].get()));
                    newguy->swap_two_labels(labels[id1],labels[id2]);
                    newguy->sort_labels();

                    strings_same = compare_strings(ordered[j],newguy,n_permute);

                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

// consolidate terms that differ by permutations
void pq::consolidate_permutations(std::vector<std::shared_ptr<pq> > &ordered) {

    // consolidate terms that differ by permutations
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;
        }
    }
}

bool pq::compare_strings(std::shared_ptr<pq> ordered_1, std::shared_ptr<pq> ordered_2, int & n_permute) {

    // don't forget w0
    if ( ordered_1->data->has_w0 != ordered_2->data->has_w0 ) {
        return false;
    }

    // are strings same?
    if ( ordered_1->symbol.size() != ordered_2->symbol.size() ) return false;
    int nsame_s = 0;
    for (size_t k = 0; k < ordered_1->symbol.size(); k++) {
        if ( ordered_1->symbol[k] == ordered_2->symbol[k] ) {
            nsame_s++;
        }
    }
    if ( nsame_s != ordered_1->symbol.size() ) return false;

    // same delta functions (recall these aren't sorted in any way)
    int nsame_d = 0;
    for (size_t k = 0; k < ordered_1->data->deltas.size(); k++) {
        for (size_t l = 0; l < ordered_2->data->deltas.size(); l++) {
            if ( ordered_1->data->deltas[k].labels[0] == ordered_2->data->deltas[l].labels[0] 
              && ordered_1->data->deltas[k].labels[1] == ordered_2->data->deltas[l].labels[1] ) {
                nsame_d++;
                //break;
            }else if ( ordered_1->data->deltas[k].labels[0] == ordered_2->data->deltas[l].labels[1] 
                    && ordered_1->data->deltas[k].labels[1] == ordered_2->data->deltas[l].labels[0] ) {
                nsame_d++;
                //break;
            }
        }
    }
    if ( nsame_d != ordered_1->data->deltas.size() ) return false;

    // amplitude comparisons, with permutations
    n_permute = 0;

    bool same_string = false;
    for (size_t i = 0; i < data->amplitude_types.size(); i++) {
        char type = data->amplitude_types[i];
        same_string = compare_amplitudes( ordered_1->data->amps[type], ordered_2->data->amps[type], n_permute);
        if ( !same_string ) return false;
    }

    // integral comparisons, with permutations
    for (size_t i = 0; i < data->integral_types.size(); i++) {
        std::string type = data->integral_types[i];
        same_string = compare_integrals( ordered_1->data->ints[type], ordered_2->data->ints[type], n_permute);
        if ( !same_string ) return false;
    }

    // permutations should be the same, too wtf
    // also need to check if the permutations are the same...
    // otherwise, we shouldn't be combining these terms
    if ( ordered_1->data->permutations.size() != ordered_2->data->permutations.size() ) {
        return false;
    }

    int nsame_permutations = 0;
    // remember, permutations come in pairs
    size_t n = data->permutations.size() / 2;
    int count = 0;
    for (int i = 0; i < n; i++) {

        if ( ordered_1->data->permutations[count] == ordered_2->data->permutations[count] ) {
            nsame_permutations++;
        }else if (  ordered_1->data->permutations[count]   == ordered_2->data->permutations[count+1] ) {
            nsame_permutations++;
        }else if (  ordered_1->data->permutations[count+1] == ordered_2->data->permutations[count]   ) {
            nsame_permutations++;
        }else if (  ordered_1->data->permutations[count+1] == ordered_2->data->permutations[count+1] ) {
            nsame_permutations++;
        }
        count += 2;

    }
    if ( nsame_permutations != n ) {
        return false;
    }

    return true;
}

/// compare two lists of integrals
bool pq::compare_integrals( std::vector<integrals> ints1,
                            std::vector<integrals> ints2,
                            int & n_permute ) {

    if ( ints1.size() != ints2.size() ) return false;
    
    size_t nsame_ints = 0;
    for (size_t i = 0; i < ints1.size(); i++) {
        for (size_t j = 0; j < ints2.size(); j++) {

            if ( ints1[i] == ints2[j] ) {

                n_permute += ints1[i].permutations + ints2[j].permutations;

                nsame_ints++;
                break;
            }

        }
    }

    if ( nsame_ints != ints1.size() ) return false;

    return true;
}

/// compare two lists of amplitudes
bool pq::compare_amplitudes( std::vector<amplitudes> amps1,
                             std::vector<amplitudes> amps2,
                             int & n_permute ) {

    if ( amps1.size() != amps2.size() ) return false;
    
    size_t nsame_amps = 0;
    for (size_t i = 0; i < amps1.size(); i++) {
        for (size_t j = 0; j < amps2.size(); j++) {

            if ( amps1[i] == amps2[j] ) {

                n_permute += amps1[i].permutations + amps2[j].permutations;

                nsame_amps++;
                break;
            }

        }
    }

    if ( nsame_amps != amps1.size() ) return false;

    return true;
}


// copy all data, except symbols and daggers. 

// TODO: should probably make sure all of the std::vectors
//       (ints, amplitudes, deltas) have been cleared.
void pq::shallow_copy(void * copy_me) {

    pq * in = reinterpret_cast<pq * >(copy_me);

    // skip string?
    skip   = in->skip;
    
    // sign
    sign   = in->sign;
    
    // factor
    data->factor = in->data->factor;

    // deltas
    for (size_t i = 0; i < in->data->deltas.size(); i++) {
        data->deltas.push_back(in->data->deltas[i]);
    }

    // integrals
    for (size_t i = 0; i < data->integral_types.size(); i++) {
        std::string type = data->integral_types[i];
        for (size_t j = 0; j < in->data->ints[type].size(); j++) {
            data->ints[type].push_back( in->data->ints[type][j] );
        }
    }

    // amplitudes
    for (size_t i = 0; i < data->amplitude_types.size(); i++) {
        char type = data->amplitude_types[i];
        for (size_t j = 0; j < in->data->amps[type].size(); j++) {
            data->amps[type].push_back( in->data->amps[type][j] );
        }
    }

    // w0 
    data->has_w0 = in->data->has_w0;

    // non-summed spin labels
    data->non_summed_spin_labels = in->data->non_summed_spin_labels;

}

// how many times does an index appear amplitudes, deltas, and integrals?
int pq::index_in_anywhere(std::string idx) {

    int n = 0;

    n += index_in_deltas(idx);
    for (size_t i = 0; i < data->integral_types.size(); i++) {
        std::string type = data->integral_types[i];
        n += index_in_integrals(idx, data->ints[type]);
    }
    for (size_t i = 0; i < data->amplitude_types.size(); i++) {
        char type = data->amplitude_types[i];
        n += index_in_amplitudes(idx, data->amps[type]);
    }

    return n;
}

// how many times does an index appear deltas?
int pq::index_in_deltas(std::string idx) {

    int n = 0;
    for (size_t i = 0; i < data->deltas.size(); i++) {
        if ( data->deltas[i].labels[0] == idx ) {
            n++;
        }
        if ( data->deltas[i].labels[1] == idx ) {
            n++;
        }
    }
    return n;
}

// how many times does an index appear integrals?
int pq::index_in_integrals(std::string idx, std::vector<integrals> ints) {

    int n = 0;
    for (size_t i = 0; i < ints.size(); i++) {
        for (size_t j = 0; j < ints[i].labels.size(); j++) {
            if ( ints[i].labels[j] == idx ) {
                n++;
            }
           
        }
    }
    return n;

}
int pq::index_in_amplitudes(std::string idx, std::vector<amplitudes> amps) {

    int n = 0;
    for (size_t i = 0; i < amps.size(); i++) {
        for (size_t j = 0; j < amps[i].labels.size(); j++) {
            if ( amps[i].labels[j] == idx ) {
                n++;
            }
           
        }
    }
    return n;

}

void pq::replace_index_everywhere(std::string old_idx, std::string new_idx) {

    //replace_index_in_deltas(old_idx,new_idx);
    for (size_t i = 0; i < data->integral_types.size(); i++) {
        std::string type = data->integral_types[i];
        replace_index_in_integrals(old_idx, new_idx, data->ints[type]);
    }
    for (size_t i = 0; i < data->amplitude_types.size(); i++) {
        char type = data->amplitude_types[i];
        replace_index_in_amplitudes(old_idx, new_idx, data->amps[type]);
    }
    sort_labels();

}

void pq::set_spin_everywhere(std::string target, std::string spin) {

    // integrals
    for (size_t i = 0; i < data->integral_types.size(); i++) {
        std::string type = data->integral_types[i];
        for (size_t j = 0; j < data->ints[type].size(); j++) {
            for (size_t k = 0; k < data->ints[type][j].labels.size(); k++) {
                if ( data->ints[type][j].labels[k] == target ) {
                    data->ints[type][j].spin_labels[k] = spin;
                }
            }
        }
    }
    // amplitudes
    for (size_t i = 0; i < data->amplitude_types.size(); i++) {
        char type = data->amplitude_types[i];
        for (size_t j = 0; j < data->amps[type].size(); j++) {
            for (size_t k = 0; k < data->amps[type][j].labels.size(); k++) {
                if ( data->amps[type][j].labels[k] == target ) {
                    data->amps[type][j].spin_labels[k] = spin;
                }
            }
        }
    }
    // deltas
    for (size_t i = 0; i < data->deltas.size(); i++) {
        for (size_t j = 0; j < data->deltas[i].labels.size(); j++) {
            if ( data->deltas[i].labels[j] == target ) {
                data->deltas[i].spin_labels[j] = spin;
            }
        }
    }

}

void pq::replace_index_in_deltas(std::string old_idx, std::string new_idx) {

    for (size_t i = 0; i < data->deltas.size(); i++) {
        if ( data->deltas[i].labels[0] == old_idx ) {
            data->deltas[i].labels[0] = new_idx;
            // dont' return because indices may be repeated in two-electron integrals
            //return;
        }
    }
    for (size_t i = 0; i < data->deltas.size(); i++) {
        if ( data->deltas[i].labels[1] == old_idx ) {
            data->deltas[i].labels[1] = new_idx;
            // dont' return because indices may be repeated in two-electron integrals
            //return;
        }
    }

}

void pq::replace_index_in_amplitudes(std::string old_idx, std::string new_idx, std::vector<amplitudes> &amps) {

    for (size_t i = 0; i < amps.size(); i++) {
        for (size_t j = 0; j < amps[i].labels.size(); j++) {
            if ( amps[i].labels[j] == old_idx ) {
                amps[i].labels[j] = new_idx;
            }
        }
    }
}

void pq::replace_index_in_integrals(std::string old_idx, std::string new_idx, std::vector<integrals> &ints) {

    for (size_t i = 0; i < ints.size(); i++) {
        for (size_t j = 0; j < ints[i].labels.size(); j++) {
            if ( ints[i].labels[j] == old_idx ) {
                ints[i].labels[j] = new_idx;
            }
        }
    }
}

// find and replace any funny labels in integrals with conventional ones. i.e., o1 -> i ,v1 -> a
void pq::use_conventional_labels() {

    // occupied first:
    std::vector<std::string> occ_in{"o0","o1","o2","o3","o4","o5","o6","o7","o8","o9",
                                    "o10","o11","o12","o13","o14","o15","o16","o17","o18","o19",
                                    "o20","o21","o22","o23","o24","o25","o26","o27","o28","o29"};
    std::vector<std::string> occ_out{"i","j","k","l","m","n","o","t",
                                     "i0","i1","i2","i3","i4","i5","i6","i7","i8","i9",
                                     "i10","i11","i12","i13","i14","i15","i16","i17","i18","i19"};

    for (size_t i = 0; i < occ_in.size(); i++) {

        if ( index_in_anywhere(occ_in[i]) > 0 ) {

            for (size_t j = 0; j < occ_out.size(); j++) {

                if ( index_in_anywhere(occ_out[j]) == 0 ) {

                    replace_index_everywhere(occ_in[i],occ_out[j]);
                    break;
                }
            }
        }
    }

    // now virtual
    std::vector<std::string> vir_in{"v0","v1","v2","v3","v4","v5","v6","v7","v8","v9",
                                    "v10","v11","v12","v13","v14","v15","v16","v17","v18","v19",
                                    "v20","v21","v22","v23","v24","v25","v26","v27","v28","v29"};
    std::vector<std::string> vir_out{"a","b","c","d","e","f","g","h",
                                     "a0","a1","a2","a3","a4","a5","a6","a7","a8","a9",
                                     "a10","a11","a12","a13","a14","a15","a16","a17","a18","a19"};

    for (size_t i = 0; i < vir_in.size(); i++) {

        if ( index_in_anywhere(vir_in[i]) > 0 ) {

            for (size_t j = 0; j < vir_out.size(); j++) {

                if ( index_in_anywhere(vir_out[j]) == 0 ) {

                    replace_index_everywhere(vir_in[i],vir_out[j]);
                    break;
                }
            }
        }
    }
}

void pq::gobble_deltas() {

    std::vector<std::string> tmp_delta1;
    std::vector<std::string> tmp_delta2;

    // create list of summation labels. only consider internally-created labels

    std::vector<std::string> occ_labels{"o0","o1","o2","o3","o4","o5","o6","o7","o8","o9",
                                    "o10","o11","o12","o13","o14","o15","o16","o17","o18","o19",
                                    "o20","o21","o22","o23","o24","o25","o26","o27","o28","o29"};
    std::vector<std::string> vir_labels{"v0","v1","v2","v3","v4","v5","v6","v7","v8","v9",
                                    "v10","v11","v12","v13","v14","v15","v16","v17","v18","v19",
                                    "v20","v21","v22","v23","v24","v25","v26","v27","v28","v29"};

    std::vector<std::string> sum_labels;
    for (size_t i = 0; i < occ_labels.size(); i++) {
        if ( index_in_anywhere( occ_labels[i] ) == 2 ) {
            sum_labels.push_back( occ_labels[i] );
        }
    }
    for (size_t i = 0; i < vir_labels.size(); i++) {
        if ( index_in_anywhere( vir_labels[i] ) == 2 ) {
            sum_labels.push_back( vir_labels[i] );
        }
    }

    for (size_t i = 0; i < data->deltas.size(); i++) {

        // is delta label 1 in list of summation labels?
        bool have_delta1 = false;
        for (size_t j = 0; j < sum_labels.size(); j++) {
            if ( data->deltas[i].labels[0] == sum_labels[j] ) {
                have_delta1 = true;
                break;
            }
        }
        // is delta label 2 in list of summation labels?
        bool have_delta2 = false;
        for (size_t j = 0; j < sum_labels.size(); j++) {
            if ( data->deltas[i].labels[1] == sum_labels[j] ) {
                have_delta2 = true;
                break;
            }
        }

/*
        // this logic is obviously cleaner than that below, but 
        // for some reason the code has a harder time collecting 
        // like terms this way. requires swapping up to four 
        // labels.
        if ( have_delta1 ) {
            replace_index_everywhere( data->deltas[i].labels[0], data->deltas[i].labels[1] );
            continue;
        }else if ( have_delta2 ) {
            replace_index_everywhere( data->deltas[i].labels[1], data->deltas[i].labels[0] );
            continue;
        }
*/

        bool do_continue = false;
        for (size_t j = 0; j < data->integral_types.size(); j++) { 
            std::string type = data->integral_types[j];
            if ( have_delta1 && index_in_integrals( data->deltas[i].labels[0], data->ints[type] ) > 0 ) {
               replace_index_in_integrals( data->deltas[i].labels[0], data->deltas[i].labels[1], data->ints[type] );
               do_continue = true;
               break;
            }else if ( have_delta2 && index_in_integrals( data->deltas[i].labels[1], data->ints[type] ) > 0 ) {
               replace_index_in_integrals( data->deltas[i].labels[1], data->deltas[i].labels[0], data->ints[type] );
               do_continue = true;
               break;
            }
        }
        if ( do_continue ) continue;

        // TODO: note that the code only efficiently collects terms when the amplitude
        // list is ordered as {'t', 'l', 'r', 'u', 'm', 's'} ... i don't know why, but
        // i do know that this is the problematic part of the code
        do_continue = false;
        std::vector<char> types = {'t', 'l', 'r', 'u', 'm', 's'};
        for (size_t j = 0; j < types.size(); j++) { 
            char type = types[j];
            if ( have_delta1 && index_in_amplitudes( data->deltas[i].labels[0], data->amps[type] ) > 0 ) {
               replace_index_in_amplitudes( data->deltas[i].labels[0], data->deltas[i].labels[1], data->amps[type] );
               do_continue = true;
               break;
            }else if ( have_delta2 && index_in_amplitudes( data->deltas[i].labels[1], data->amps[type] ) > 0 ) {
               replace_index_in_amplitudes( data->deltas[i].labels[1], data->deltas[i].labels[0], data->amps[type] );
               do_continue = true;
               break;
            }
        }
        if ( do_continue ) continue;

        // at this point, it is safe to assume the delta function must remain
        tmp_delta1.push_back(data->deltas[i].labels[0]);
        tmp_delta2.push_back(data->deltas[i].labels[1]);

    }

    data->deltas.clear();

    for (size_t i = 0; i < tmp_delta1.size(); i++) {

        delta_functions deltas;
        deltas.labels.push_back(tmp_delta1[i]);
        deltas.labels.push_back(tmp_delta2[i]);
        deltas.sort();
        data->deltas.push_back(deltas);

    }

}

// copy all data, including symbols and daggers
void pq::copy(void * copy_me) { 

    shallow_copy(copy_me);

    pq * in = reinterpret_cast<pq * >(copy_me);

    // operators
    for (size_t j = 0; j < in->symbol.size(); j++) {
        symbol.push_back(in->symbol[j]);

        // dagger?
        is_dagger.push_back(in->is_dagger[j]);

        // dagger (relative to fermi vacuum)?
        if ( vacuum == "FERMI" ) {
            is_dagger_fermi.push_back(in->is_dagger_fermi[j]);
        }
    }

    // boson daggers
    for (size_t i = 0; i < in->data->is_boson_dagger.size(); i++) {
        data->is_boson_dagger.push_back(in->data->is_boson_dagger[i]);
    }
    
    // permutations
    for (size_t i = 0; i < in->data->permutations.size(); i++) {
        data->permutations.push_back(in->data->permutations[i]);
    }

}

bool pq::normal_order_true_vacuum(std::vector<std::shared_ptr<pq> > &ordered) {

    if ( skip ) return true;

    if ( is_normal_order() ) {

        // push current ordered operator onto running list
        std::shared_ptr<pq> newguy (new pq(vacuum));

        newguy->copy((void*)this);

        ordered.push_back(newguy);

        return true;
    }

    // new strings
    std::shared_ptr<pq> s1 ( new pq(vacuum) );
    std::shared_ptr<pq> s2 ( new pq(vacuum) );

    // copy data common to both new strings
    s1->shallow_copy((void*)this);
    s2->shallow_copy((void*)this);

    // rearrange operators
    for (int i = 0; i < (int)symbol.size()-1; i++) {

        bool swap = ( !is_dagger[i] && is_dagger[i+1] );

        if ( swap ) {

            std::vector<std::string> labels;
            delta_functions deltas;
            deltas.labels.push_back(symbol[i]);
            deltas.labels.push_back(symbol[i+1]);
            deltas.sort();
            s1->data->deltas.push_back(deltas);

            s2->sign = -s2->sign;
            s2->symbol.push_back(symbol[i+1]);
            s2->symbol.push_back(symbol[i]);
            s2->is_dagger.push_back(is_dagger[i+1]);
            s2->is_dagger.push_back(is_dagger[i]);

            for (size_t j = i+2; j < symbol.size(); j++) {

                s1->symbol.push_back(symbol[j]);
                s2->symbol.push_back(symbol[j]);

                s1->is_dagger.push_back(is_dagger[j]);
                s2->is_dagger.push_back(is_dagger[j]);

            }
            break;

        }else {

            s1->symbol.push_back(symbol[i]);
            s2->symbol.push_back(symbol[i]);

            s1->is_dagger.push_back(is_dagger[i]);
            s2->is_dagger.push_back(is_dagger[i]);

        }
    }

    // now, s1 and s2 are closer to normal order in the fermion space
    // we should more toward normal order in the boson space, too

    if ( is_boson_normal_order() ) {

        // copy boson daggers 
        for (size_t i = 0; i < data->is_boson_dagger.size(); i++) {
            s1->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
            s2->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
        }
        //s1->normal_order_true_vacuum(ordered);
        //s2->normal_order_true_vacuum(ordered);
        ordered.push_back(s1);
        ordered.push_back(s2);
        return false;

    }else {

        // new strings
        std::shared_ptr<pq> s1a ( new pq(vacuum) );
        std::shared_ptr<pq> s1b ( new pq(vacuum) );
        std::shared_ptr<pq> s2a ( new pq(vacuum) );
        std::shared_ptr<pq> s2b ( new pq(vacuum) );

        // copy data common to new strings
        s1a->copy((void*)s1.get());
        s1b->copy((void*)s1.get());

        // ensure boson daggers are clear (they should be anyway)
        s1a->data->is_boson_dagger.clear();
        s1b->data->is_boson_dagger.clear();

        for (int i = 0; i < (int)data->is_boson_dagger.size()-1; i++) {

            // swap operators?
            bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

            if ( swap ) {

                // nothing happens to s1a. add swapped operators to s1b
                s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                // push remaining operators onto s1a and s1b
                for (size_t j = i+2; j < data->is_boson_dagger.size(); j++) {

                    s1a->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);

                }
                break;

            }else {

                s1a->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

            }
        }

        // copy data common to new strings
        s2a->copy((void*)s2.get());
        s2b->copy((void*)s2.get());

        // ensure boson daggers are clear (they should be anyway)
        s2a->data->is_boson_dagger.clear();
        s2b->data->is_boson_dagger.clear();

        for (int i = 0; i < (int)data->is_boson_dagger.size()-1; i++) {

            // swap operators?
            bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

            if ( swap ) {

                // nothing happens to s2a. add swapped operators to s2b
                s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                // push remaining operators onto s2a and s2b
                for (size_t j = i+2; j < data->is_boson_dagger.size(); j++) {

                    s2a->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);
                    s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);

                }
                break;

            }else {

                s2a->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

            }
        }

        //s1a->normal_order_true_vacuum(ordered);
        //s1b->normal_order_true_vacuum(ordered);
        //s2a->normal_order_true_vacuum(ordered);
        //s2b->normal_order_true_vacuum(ordered);
        ordered.push_back(s1a);
        ordered.push_back(s1b);
        ordered.push_back(s2a);
        ordered.push_back(s2b);
        return false;

    }

    return false;
}

bool pq::normal_order_fermi_vacuum(std::vector<std::shared_ptr<pq> > &ordered) {

    if ( skip ) return true;

    if ( is_normal_order() ) {

        // push current ordered operator onto running list
        std::shared_ptr<pq> newguy (new pq(vacuum));

        newguy->copy((void*)this);

        ordered.push_back(newguy);

        return true;
    }

    // new strings
    std::shared_ptr<pq> s1 ( new pq(vacuum) );
    std::shared_ptr<pq> s2 ( new pq(vacuum) );

    // copy data common to both new strings
    s1->shallow_copy((void*)this);
    s2->shallow_copy((void*)this);

    // rearrange operators

    int n_new_strings = 1;

    for (int i = 0; i < (int)symbol.size()-1; i++) {

        bool swap = ( !is_dagger_fermi[i] && is_dagger_fermi[i+1] );

        // four cases: **, --, *-, -*
        // **, --: change sign, swap labels
        // *-, -*: standard swap

        bool daggers_differ = ( is_dagger[i] != is_dagger[i+1] );

        if ( swap && daggers_differ ) {

            // we're going to have two new strings
            n_new_strings = 2;

            // delta function
            std::vector<std::string> labels;
            delta_functions deltas;
            deltas.labels.push_back(symbol[i]);
            deltas.labels.push_back(symbol[i+1]);
            deltas.sort();
            s1->data->deltas.push_back(deltas);

            s2->sign = -s2->sign;
            s2->symbol.push_back(symbol[i+1]);
            s2->symbol.push_back(symbol[i]);
            s2->is_dagger.push_back(is_dagger[i+1]);
            s2->is_dagger.push_back(is_dagger[i]);
            s2->is_dagger_fermi.push_back(is_dagger_fermi[i+1]);
            s2->is_dagger_fermi.push_back(is_dagger_fermi[i]);

            for (size_t j = i+2; j < symbol.size(); j++) {

                s1->symbol.push_back(symbol[j]);
                s2->symbol.push_back(symbol[j]);

                s1->is_dagger.push_back(is_dagger[j]);
                s2->is_dagger.push_back(is_dagger[j]);

                s1->is_dagger_fermi.push_back(is_dagger_fermi[j]);
                s2->is_dagger_fermi.push_back(is_dagger_fermi[j]);

            }
            break;

        }else if ( swap && !daggers_differ )  {

            // we're only going to have one new string, with a different sign
            n_new_strings = 1;

            s1->sign = -s1->sign;
            s1->symbol.push_back(symbol[i+1]);
            s1->symbol.push_back(symbol[i]);
            s1->is_dagger.push_back(is_dagger[i+1]);
            s1->is_dagger.push_back(is_dagger[i]);
            s1->is_dagger_fermi.push_back(is_dagger_fermi[i+1]);
            s1->is_dagger_fermi.push_back(is_dagger_fermi[i]);

            for (size_t j = i+2; j < symbol.size(); j++) {

                s1->symbol.push_back(symbol[j]);

                s1->is_dagger.push_back(is_dagger[j]);

                s1->is_dagger_fermi.push_back(is_dagger_fermi[j]);

            }
            break;

        }else {

            s1->symbol.push_back(symbol[i]);
            s2->symbol.push_back(symbol[i]);

            s1->is_dagger.push_back(is_dagger[i]);
            s2->is_dagger.push_back(is_dagger[i]);

            s1->is_dagger_fermi.push_back(is_dagger_fermi[i]);
            s2->is_dagger_fermi.push_back(is_dagger_fermi[i]);

        }
    }

    // now, s1 (and s2) are closer to normal order in the fermion space
    // we should more toward normal order in the boson space, too

    if ( n_new_strings == 1 ) {

        if ( is_boson_normal_order() ) {
            if ( !skip ) {
                // copy boson daggers
                for (size_t i = 0; i < data->is_boson_dagger.size(); i++) {
                    s1->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                }
                //s1->normal_order_fermi_vacuum(ordered);
                ordered.push_back(s1);
                return false;
            }
        }else {

            // new strings
            std::shared_ptr<pq> s1a ( new pq(vacuum) );
            std::shared_ptr<pq> s1b ( new pq(vacuum) );

            // copy data common to both new strings
            s1a->copy((void*)s1.get());
            s1b->copy((void*)s1.get());

            // ensure boson daggers are clear (they should be anyway)
            s1a->data->is_boson_dagger.clear();
            s1b->data->is_boson_dagger.clear();

            for (int i = 0; i < (int)data->is_boson_dagger.size()-1; i++) {

                // swap operators?
                bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

                if ( swap ) {

                    // nothing happens to s1a. add swapped operators to s1b
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                    // push remaining operators onto s1a and s1b
                    for (size_t j = i+2; j < data->is_boson_dagger.size(); j++) {
        
                        s1a->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);
                        s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);
        
                    }
                    break;

                }else {

                    s1a->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                }
            }
            //s1a->normal_order_fermi_vacuum(ordered);
            //s1b->normal_order_fermi_vacuum(ordered);
            ordered.push_back(s1a);
            ordered.push_back(s1b);
            return false;
        }

    }else if ( n_new_strings == 2 ) {

        if ( is_boson_normal_order() ) {
            if ( !skip ) {
                // copy boson daggers
                for (size_t i = 0; i < data->is_boson_dagger.size(); i++) {
                    s1->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                    s2->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                }
                //s1->normal_order_fermi_vacuum(ordered);
                //s2->normal_order_fermi_vacuum(ordered);
                ordered.push_back(s1);
                ordered.push_back(s2);
                return false;
            }
        }else {

            // new strings
            std::shared_ptr<pq> s1a ( new pq(vacuum) );
            std::shared_ptr<pq> s1b ( new pq(vacuum) );
            std::shared_ptr<pq> s2a ( new pq(vacuum) );
            std::shared_ptr<pq> s2b ( new pq(vacuum) );

            // copy data common to new strings
            s1a->copy((void*)s1.get());
            s1b->copy((void*)s1.get());

            // ensure boson daggers are clear (they should be anyway)
            s1a->data->is_boson_dagger.clear();
            s1b->data->is_boson_dagger.clear();

            for (int i = 0; i < (int)data->is_boson_dagger.size()-1; i++) {

                // swap operators?
                bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

                if ( swap ) {

                    // nothing happens to s1a. add swapped operators to s1b
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                    // push remaining operators onto s1a and s1b
                    for (size_t j = i+2; j < data->is_boson_dagger.size(); j++) {

                        s1a->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);
                        s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);

                    }
                    break;

                }else {

                    s1a->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                }
            }

            // copy data common to new strings
            s2a->copy((void*)s2.get());
            s2b->copy((void*)s2.get());

            // ensure boson daggers are clear (they should be anyway)
            s2a->data->is_boson_dagger.clear();
            s2b->data->is_boson_dagger.clear();

            for (int i = 0; i < data->is_boson_dagger.size()-1; i++) {

                // swap operators?
                bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

                if ( swap ) {

                    // nothing happens to s2a. add swapped operators to s2b
                    s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                    s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                    // push remaining operators onto s2a and s2b
                    for (size_t j = i+2; j < data->is_boson_dagger.size(); j++) {

                        s2a->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);
                        s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);

                    }
                    break;

                }else {

                    s2a->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                    s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                }
            }

            //s1a->normal_order_fermi_vacuum(ordered);
            //s1b->normal_order_fermi_vacuum(ordered);
            //s2a->normal_order_fermi_vacuum(ordered);
            //s2b->normal_order_fermi_vacuum(ordered);
            ordered.push_back(s1a);
            ordered.push_back(s1b);
            ordered.push_back(s2a);
            ordered.push_back(s2b);
            return false;

        }

    }
    return false;

}

bool pq::normal_order(std::vector<std::shared_ptr<pq> > &ordered) {
    if ( vacuum == "TRUE" ) {
        return normal_order_true_vacuum(ordered);
    }else {
        return normal_order_fermi_vacuum(ordered);
    }
}

// re-classify fluctuation potential terms
void pq::reclassify_integrals() {

    if ( data->ints["occ_repulsion"].size() > 1 ) {
       printf("\n");
       printf("only support for one integral type object per string\n");
       printf("\n");
       exit(1);
    }

    if ( data->ints["occ_repulsion"].size() > 0 ) {

        // pick summation label not included in string already
        std::vector<std::string> occ_out{"i","j","k","l","m","n","o","t","i0","i1","i2","i3","i4","i5","i6","i7","i8","i9"};
        std::string idx;

        int skip = -999;

        for (size_t i = 0; i < occ_out.size(); i++) {
            if ( index_in_anywhere(occ_out[i]) == 0 ) {
                idx = occ_out[i];
                skip = i;
                break;
            }
        }
        if ( skip == -999 ) {
            printf("\n");
            printf("    uh oh. no suitable summation index could be found.\n");
            printf("\n");
            exit(1);
        }

        std::string idx1 = data->ints["occ_repulsion"][0].labels[0];
        std::string idx2 = data->ints["occ_repulsion"][0].labels[1];

        data->ints["occ_repulsion"].clear();

        integrals ints;

        ints.labels.clear();
        ints.numerical_labels.clear();

        ints.labels.push_back(idx1);
        ints.labels.push_back(idx);
        ints.labels.push_back(idx2);
        ints.labels.push_back(idx);

        ints.sort();

        if ( data->ints["eri"].size() > 0 ) {
           printf("\n");
           printf("only support for one integral type object per string\n");
           printf("\n");
           exit(1);
        }
        data->ints["eri"].clear();
        data->ints["eri"].push_back(ints);

    }

}

} // End namespaces

