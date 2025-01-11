//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_utils.cc
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
#include "pq_utils.h"
#include "pq_bernoulli.h"

namespace pdaggerq {

// determine the operator type for the part of an input string corresponding to a target portion (for bernoulli)
std::string bernoulli_type(std::shared_ptr<pq_string> &in, std::string target_portion, size_t portion_number, int bernoulli_excitation_level) {

    std::shared_ptr<pq_string> newguy (new pq_string(in->vacuum));

    // portions in integrals
    bool has_ints = false;
    for (const auto & int_pair : in->ints) {
        const std::string &type = int_pair.first;
        const std::vector<integrals> &ints = int_pair.second;
        for (const integrals & integral : ints) {
            std::string portion = integral.op_portions[portion_number];
            if ( portion != target_portion ) {
                continue;
            }
            has_ints = true;
            newguy->set_integrals(type, integral.labels);
        }
    }

    // portions in amplitudes
    bool has_amps = false;
    for (const auto & amp_pair : in->amps) {
        const char &type = amp_pair.first;
        if ( type != 't' ) {
            continue;
        }
        const std::vector<amplitudes> &amps = amp_pair.second;
        for (const amplitudes & amp : amps) {
            std::string portion = amp.op_portions[portion_number];
            if ( portion != target_portion ) {
                continue;
            }
            has_amps = true;
            newguy->set_amplitudes(type, amp.n_create, amp.n_annihilate, amp.n_ph, amp.labels);
        }
    }
    if ( !has_amps && !has_ints ) {
        return "";
    }

    // now, count the number of occupied / virtual labels in the bra
    // and ket, excluding those labels that are repeated

    int no_bra = 0;
    int nv_bra = 0;
    int nt_bra = 0;

    int no_ket = 0;
    int nv_ket = 0;
    int nt_ket = 0;

    // portions in integrals
    for (const auto & int_pair : newguy->ints) {
        const std::string &type = int_pair.first;
        const std::vector<integrals> &ints = int_pair.second;

        int n_create = 1;
        int n_annihilate = 1;
        if ( type == "eri" || type == "two_body" ) {
            n_create = 2;
            n_annihilate = 2;
        }

        for (const integrals & integral : ints) {

            for (int j = 0; j < n_create; j++) {
                std::string label = integral.labels[j];

                // skip repeated labels
                int found = newguy->index_in_anywhere(label);
                if ( found != 1 ) {
                    continue;
                }

                nt_bra++;
                if (is_occ(label)) {
                    no_bra++;
                }else {
                    nv_bra++;
                }
            }

            for (int j = n_create; j < n_create + n_annihilate; j++) {
                std::string label = integral.labels[j];

                // skip repeated labels
                int found = newguy->index_in_anywhere(label);
                if ( found != 1 ) {
                    continue;
                }

                nt_ket++;
                if (is_occ(label)) {
                    no_ket++;
                }else {
                    nv_ket++;
                }
            }
        }
    }

    // portions in amplitudes
    for (const auto & amp_pair : newguy->amps) {
        const char &type = amp_pair.first;
        if ( type != 't' ) {
            continue;
        }
        const std::vector<amplitudes> &amps = amp_pair.second;
        for (const amplitudes & amp : amps) {

            for (int j = 0; j < amp.n_create; j++) {
                std::string label = amp.labels[j];

                // skip repeated labels
                int found = newguy->index_in_anywhere(label);
                if ( found == 2 ) {
                    continue;
                }

                nt_bra++;
                if (is_occ(label)) {
                    no_bra++;
                }else {
                    nv_bra++;
                }
            }

            for (int j = amp.n_create; j < amp.n_create + amp.n_annihilate; j++) {
                std::string label = amp.labels[j];

                // skip repeated labels
                int found = newguy->index_in_anywhere(label);
                if ( found == 2 ) {
                    continue;
                }

                nt_ket++;
                if (is_occ(label)) {
                    no_ket++;
                }else {
                    nv_ket++;
                }
            }
        }
    }

    // return the portion type
    if ( nt_bra > bernoulli_excitation_level ) { 
        return "R"; 
    }
    if ( no_bra == nt_bra && nv_ket == nt_ket ) {
        return "N"; 
    }else if ( no_ket == nt_ket && nv_bra == nt_bra ) {
        return "N";
    }
    return "R";
}

// eliminate terms based on operator portions (for bernoulli)
void eliminate_operator_portions(std::shared_ptr<pq_string> &in, int bernoulli_excitation_level){

    // first, ensure the list of portions is of the same size for all amplitudes and integrals

    std::unordered_map<size_t, size_t> len_map;

    // portions in integrals
    for (const auto & int_pair : in->ints) {
        const std::string &type = int_pair.first;
        const std::vector<integrals> &ints = int_pair.second;
        for (const integrals & integral : ints) {
            size_t len = integral.op_portions.size();
            len_map[len] = 0;
        }
    }

    // portions in amplitudes
    for (const auto & amp_pair : in->amps) {
        const char &type = amp_pair.first;
        if ( type != 't' ) { 
            continue;
        }
        const std::vector<amplitudes> &amps = amp_pair.second;
        for (const amplitudes & amp : amps) {
            size_t len = amp.op_portions.size();
            len_map[len] = 0;
        }
    }

    if ( len_map.size() > 1 ) {
        printf("\n");
        printf("    error: string components have inconsistent operator portions.\n");
        printf("\n");
        exit(1);
    }

    // now, eliminate pieces that should not exist. we can do this by
    // counting how many occupied and virtual labels there are in 
    // the bra and ket parts of the operators of "N" or "R" type, but
    // we must be careful to ignore repeated labels within the relevant
    // parts of the string. we should just build a new string that
    // contains only the relevant pieces

    size_t n_op_portions = 0;
    for (const auto & [ key, value ] : len_map) {
        n_op_portions = key; 
    }

    // all components of "N" type
    for (size_t i = 0; i < n_op_portions; i++) {

        if ( bernoulli_type(in, "N", i, bernoulli_excitation_level) == "R" ) {
            in->skip = true;
            return;
        }

        if ( bernoulli_type(in, "R", i, bernoulli_excitation_level) == "N" ) {
            in->skip = true;
            return;
        }
    }
}

// bernoulli expansion involves operator portions. strip these off 
// and return them as a string
std::string get_operator_portions_as_string(const std::string& op) {
    std::string ret;
    size_t start = op.find('{');
    size_t end = op.find('}');
    if ( start != std::string::npos && end == std::string::npos ) { 
        printf("\n");
        printf("    something is wrong with the bernoulli operator portions definition.\n");
        printf("\n");
        exit(1);
    }else if ( start == std::string::npos && end != std::string::npos ) { 
        printf("\n");
        printf("    something is wrong with the bernoulli operator portions definition.\n");
        printf("\n");
        exit(1);
    }else if ( start != std::string::npos && end != std::string::npos ) { 
        ret = op.substr(start + 1, end - start - 1);
    }
    return ret;
}

// bernoulli expansion involves operator portions. strip these off 
// and return them as a vector
std::vector<std::string> get_operator_portions_as_vector(const std::string& op) {
    std::string s = get_operator_portions_as_string(op);
    std::vector<std::string> tokens;
    if ( s == "" ) {
        return tokens;
    }
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(",")) != std::string::npos) {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + 1);
    }
    tokens.push_back(s);

    return tokens;
}

// bernoulli expansion involves operator portions. strip these off 
// and return the base operator name
std::string get_operator_base_name(std::string op) {
    std::string ret;
    size_t start = op.find('{');
    size_t end = op.find('}');
    if ( start != std::string::npos && end == std::string::npos ) { 
        printf("\n");
        printf("    something is wrong with the bernoulli operator portions definition.\n");
        printf("\n");
        exit(1);
    }else if ( start == std::string::npos && end != std::string::npos ) { 
        printf("\n");
        printf("    something is wrong with the bernoulli operator portions definition.\n");
        printf("\n");
        exit(1);
    }else if ( start != std::string::npos && end != std::string::npos ) { 
        ret = op.substr(0, start);
    }else {
        ret = op;
    }
    return ret;
}

} // End namespaces
