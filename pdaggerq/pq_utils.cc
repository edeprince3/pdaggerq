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

#include "pq_utils.h"

namespace pdaggerq {

/// concatinate a list of operators (a list of strings) into a single list
std::vector<std::string> concatinate_operators(std::vector<std::vector<std::string>> ops) {

    std::vector<std::string> ret;
    size_t size = 0;
    for (size_t i = 0; i < ops.size(); i++) {
        size += ops[i].size();
    }
    ret.reserve(size);
    for (size_t i = 0; i < ops.size(); i++) {
        ret.insert(ret.end(), ops[i].begin(), ops[i].end());
    }
    return ret;

}

/// remove "*" from std::string
void removeStar(std::string &x) {

  auto it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == '*');});
  x.erase(it, std::end(x));
}

/// remove "(" and ")" from std::string
void removeParentheses(std::string &x) {

  auto it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == '(');});
  x.erase(it, std::end(x));

  it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == ')');});
  x.erase(it, std::end(x));

}

// is a label classified as occupied?
bool is_occ(std::string idx) {
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

// is a label classified as virtual?
bool is_vir(std::string idx) {
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

// how many times does an index appear deltas?
int index_in_deltas(std::string idx, std::vector<delta_functions> deltas) {

    int n = 0;
    for (size_t i = 0; i < deltas.size(); i++) {
        if ( deltas[i].labels[0] == idx ) {
            n++;
        }
        if ( deltas[i].labels[1] == idx ) {
            n++;
        }
    }
    return n;
}

// how many times does an index appear integrals?
int index_in_integrals(std::string idx, std::vector<integrals> ints) {

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

// how many times does an index appear in amplitudes?
int index_in_amplitudes(std::string idx, std::vector<amplitudes> amps) {

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

/// replace one label with another (in a given set of deltas)
void replace_index_in_deltas(std::string old_idx, std::string new_idx, std::vector<delta_functions> &deltas) {

    for (size_t i = 0; i < deltas.size(); i++) {
        if ( deltas[i].labels[0] == old_idx ) {
            deltas[i].labels[0] = new_idx;
        }
    }
    for (size_t i = 0; i < deltas.size(); i++) {
        if ( deltas[i].labels[1] == old_idx ) {
            deltas[i].labels[1] = new_idx;
        }
    }
}

/// replace one label with another (in a given set of amplitudes)
void replace_index_in_amplitudes(std::string old_idx, std::string new_idx, std::vector<amplitudes> &amps) {

    for (size_t i = 0; i < amps.size(); i++) {
        for (size_t j = 0; j < amps[i].labels.size(); j++) {
            if ( amps[i].labels[j] == old_idx ) {
                amps[i].labels[j] = new_idx;
            }
        }
    }
}

/// replace one label with another (in a given set of integrals)
void replace_index_in_integrals(std::string old_idx, std::string new_idx, std::vector<integrals> &ints) {

    for (size_t i = 0; i < ints.size(); i++) {
        for (size_t j = 0; j < ints[i].labels.size(); j++) {
            if ( ints[i].labels[j] == old_idx ) {
                ints[i].labels[j] = new_idx;
            }
        }
    }
}

}
