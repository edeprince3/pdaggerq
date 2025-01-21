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

// generate lists of operator partitions, e.g., { {'N', 'A'}, {'A', 'A'} }
std::vector<std::string> get_partitions_list(std::vector<std::string> in ) {
    std::vector<std::string> partitions_list;
    for (size_t i = 0; i < in.size(); i++) {
        std::string partitions = "{";
        for (size_t j = 0; j < i; j++) {
            partitions += "A,";
        }
        for (size_t j = i; j < in.size(); j++) {
            partitions += in[j];
            if ( j < in.size() - 1 ) {
                partitions += ",";
            }
        }
        partitions += "}";
        partitions_list.push_back(partitions);
    }
    return partitions_list;
}

// first-order bernoulli terms: 1/2 [v, sigma] + 1/2 [v_R, sigma]
std::vector<pq_operator_terms> get_bernoulli_operator_terms_1(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops) {

    std::vector<pq_operator_terms> bernoulli_terms;

    std::vector<std::string> targets_partitions;
    std::vector<std::string> b_ops_partitions;
    std::vector<double> bernoulli_factors;

    // 1/2 [v, sigma]
    targets_partitions.push_back("{A,A}");
    b_ops_partitions.push_back("{A,A}");
    bernoulli_factors.push_back(0.5);

    // 1/2 [v_R, sigma]
    targets_partitions.push_back("{R,A}");
    b_ops_partitions.push_back("{A,A}");
    bernoulli_factors.push_back(0.5);

    int dim = (int)ops.size();

    pq_helper pq("");

    for (size_t p = 0; p < targets_partitions.size(); p++){

        // mutable copies of targets and ops
        std::vector<std::string> b_targets;
        std::vector<std::string> b_ops;

        for (auto target: targets){
            b_targets.push_back(target + targets_partitions[p]);
        }

        for (auto op: ops){
            b_ops.push_back(op + b_ops_partitions[p]);
        }

        for (int i = 0; i < dim; i++) {
            std::vector<pq_operator_terms> tmp = pq.get_commutator_terms(bernoulli_factors[p] * factor, b_targets, {b_ops[i]});
            bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        }
    }

    return bernoulli_terms;
}

// second-order bernoulli terms: 1/12 [[V_N, sigma], sigma] + 1/4 [[V, sigma]_R, sigma] + 1/4 [[V_R, sigma]_R, sigma]
std::vector<pq_operator_terms> get_bernoulli_operator_terms_2(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops) {

    std::vector<pq_operator_terms> bernoulli_terms;

    std::vector<std::vector<std::string> > partitions_lists;
    std::vector<double> bernoulli_factors;

    // 1/12 [[V_N, sigma], sigma]
    partitions_lists.push_back(get_partitions_list({"N", "A", "A"}));
    bernoulli_factors.push_back(1.0/12.0);

    // 1/4 [[V, sigma]_R, sigma]
    partitions_lists.push_back(get_partitions_list({"A", "R", "A"}));
    bernoulli_factors.push_back(1.0/4.0);

    // 1/4 [[V_R, sigma]_R, sigma]
    partitions_lists.push_back(get_partitions_list({"R", "R", "A"}));
    bernoulli_factors.push_back(1.0/4.0);

    int dim = (int)ops.size();

    pq_helper pq("");

    for (size_t p = 0; p < partitions_lists.size(); p++){

        // mutable copies of targets and ops
        std::vector<std::string> b_targets;
        std::vector<std::string> b_ops1;
        std::vector<std::string> b_ops2;

        for (auto target: targets){
            b_targets.push_back(target + partitions_lists[p][0]);
        }

        for (auto op: ops){
            b_ops1.push_back(op + partitions_lists[p][1]);
            b_ops2.push_back(op + partitions_lists[p][2]);
        }

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                std::vector<pq_operator_terms> tmp = pq.get_double_commutator_terms(bernoulli_factors[p] * factor, b_targets, {b_ops1[i]}, {b_ops2[j]});
                bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
            }
        }
    }

    return bernoulli_terms;
}

// third-order bernoulli terms
std::vector<pq_operator_terms> get_bernoulli_operator_terms_3(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops) {

    std::vector<pq_operator_terms> bernoulli_terms;

    std::vector<std::vector<std::string> > partitions_lists;
    std::vector<double> bernoulli_factors;

    // 1/24 [[[V_N, sigma], sigma]_R, sigma]
    partitions_lists.push_back(get_partitions_list({"N", "A", "R", "A"}));
    bernoulli_factors.push_back(1.0/24.0);

    // 1/8 [[[V_R, sigma]_R, sigma]_R, sigma]
    partitions_lists.push_back(get_partitions_list({"R", "R", "R", "A"}));
    bernoulli_factors.push_back(1.0/8.0);

    // 1/8 [[[V, sigma]_R, sigma]_R, sigma]
    partitions_lists.push_back(get_partitions_list({"A", "R", "R", "A"}));
    bernoulli_factors.push_back(1.0/8.0);

    // -1/24 [[[V, sigma]_R, sigma], sigma]
    partitions_lists.push_back(get_partitions_list({"A", "R", "A", "A"}));
    bernoulli_factors.push_back(-1.0/24.0);

    // -1/24 [[[V_R, sigma]_R, sigma], sigma]
    partitions_lists.push_back(get_partitions_list({"R", "R", "A", "A"}));
    bernoulli_factors.push_back(-1.0/24.0);

    int dim = (int)ops.size();

    pq_helper pq("");

    for (size_t p = 0; p < partitions_lists.size(); p++){

        // mutable copies of targets and ops
        std::vector<std::string> b_targets;
        std::vector<std::string> b_ops1;
        std::vector<std::string> b_ops2;
        std::vector<std::string> b_ops3;

        for (auto target: targets){
            b_targets.push_back(target + partitions_lists[p][0]);
        }

        for (auto op: ops){
            b_ops1.push_back(op + partitions_lists[p][1]);
            b_ops2.push_back(op + partitions_lists[p][2]);
            b_ops3.push_back(op + partitions_lists[p][3]);
        }

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    std::vector<pq_operator_terms> tmp = pq.get_triple_commutator_terms(bernoulli_factors[p] * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]});
                    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
                }
            }
        }
    }

    return bernoulli_terms;
}

// fourth-order bernoulli terms
std::vector<pq_operator_terms> get_bernoulli_operator_terms_4(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops) {
    
    std::vector<pq_operator_terms> bernoulli_terms;

    std::vector<std::vector<std::string> > partitions_lists;
    std::vector<double> bernoulli_factors;
    
    // 1/16 [[[[V_R, sigma]_R, sigma]_R, sigma]_R, sigma]
    partitions_lists.push_back(get_partitions_list({"R", "R", "R", "R", "A"}));
    bernoulli_factors.push_back(1.0/16.0);
    
    // 1/16 [[[[V, sigma]_R, sigma]_R, sigma]_R, sigma]
    partitions_lists.push_back(get_partitions_list({"A", "R", "R", "R", "A"}));
    bernoulli_factors.push_back(1.0/16.0);
    
    // 1/48 [[[[V_N, sigma], sigma]_R, sigma]_R, sigma]
    partitions_lists.push_back(get_partitions_list({"N", "A", "R", "R", "A"}));
    bernoulli_factors.push_back(1.0/48.0);
    
    // -1/48 [[[[V, sigma]_R, sigma], sigma]_R, sigma]
    partitions_lists.push_back(get_partitions_list({"A", "R", "A", "R", "A"}));
    bernoulli_factors.push_back(-1.0/48.0);
    
    // -1/48 [[[[V_R, sigma]_R, sigma], sigma]_R, sigma]
    partitions_lists.push_back(get_partitions_list({"R", "R", "A", "R", "A"}));
    bernoulli_factors.push_back(-1.0/48.0);

    // -1/144 [[[[V_N, sigma], sigma]_R, sigma], sigma]
    partitions_lists.push_back(get_partitions_list({"N", "A", "R", "A", "A"}));
    bernoulli_factors.push_back(-1.0/144.0);
        
    // -1/48 [[[[V, sigma]_R, sigma]_R, sigma], sigma]
    partitions_lists.push_back(get_partitions_list({"A", "R", "R", "A", "A"}));
    bernoulli_factors.push_back(-1.0/48.0);
            
    // -1/48 [[[[V_R, sigma]_R, sigma]_R, sigma], sigma]
    partitions_lists.push_back(get_partitions_list({"R", "R", "R", "A", "A"}));
    bernoulli_factors.push_back(-1.0/48.0);
            
    // -1/720 [[[[V_N, sigma], sigma], sigma], sigma]
    partitions_lists.push_back(get_partitions_list({"N", "A", "A", "A", "A"}));
    bernoulli_factors.push_back(-1.0/720.0);

    int dim = (int)ops.size();

    pq_helper pq("");

    for (size_t p = 0; p < partitions_lists.size(); p++){
                
        // mutable copies of targets and ops
        std::vector<std::string> b_targets;
        std::vector<std::string> b_ops1;
        std::vector<std::string> b_ops2;
        std::vector<std::string> b_ops3;
        std::vector<std::string> b_ops4;

        for (auto target: targets){
            b_targets.push_back(target + partitions_lists[p][0]);
        }

        for (auto op: ops){
            b_ops1.push_back(op + partitions_lists[p][1]);
            b_ops2.push_back(op + partitions_lists[p][2]);
            b_ops3.push_back(op + partitions_lists[p][3]);
            b_ops4.push_back(op + partitions_lists[p][4]);
        }

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = pq.get_quadruple_commutator_terms(bernoulli_factors[p] * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
                    }
                }
            }
        }
    }

    return bernoulli_terms;
}

// fifth-order bernoulli terms
std::vector<pq_operator_terms> get_bernoulli_operator_terms_5(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops) {
    
    std::vector<pq_operator_terms> bernoulli_terms;
    
    std::vector<std::vector<std::string> > partitions_lists;
    std::vector<double> bernoulli_factors;
    
    //  1/32   [[[[[V_A, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "R", "R", "R", "A"}));
    bernoulli_factors.push_back(1.0/32.0);
    
    //  1/32   [[[[[V_R, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "R", "R", "R", "A"}));
    bernoulli_factors.push_back(1.0/32.0);

    // -1/96   [[[[[V_A, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "A", "R", "R", "A"}));
    bernoulli_factors.push_back(-1.0/96.0);

    // -1/96   [[[[[V_R, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "A", "R", "R", "A"}));
    bernoulli_factors.push_back(-1.0/96.0);

    // -1/96   [[[[[V_A, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A 
    partitions_lists.push_back(get_partitions_list({"A", "R", "R", "A", "R", "A"}));
    bernoulli_factors.push_back(-1.0/96.0);

    // -1/96   [[[[[V_R, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A 
    partitions_lists.push_back(get_partitions_list({"R", "R", "R", "A", "R", "A"}));
    bernoulli_factors.push_back(-1.0/96.0);

    // -1/96   [[[[[V_A, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "R", "R", "A", "A"})); 
    bernoulli_factors.push_back(-1.0/96.0);
        
    // -1/96   [[[[[V_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "R", "R", "A", "A"}));
    bernoulli_factors.push_back(-1.0/96.0);

    //  1/288  [[[[[V_A, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "A", "R", "A", "A"}));
    bernoulli_factors.push_back(1.0/288.0);

    //  1/288  [[[[[V_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "A", "R", "A", "A"}));
    bernoulli_factors.push_back(1.0/288.0);

    //  1/1440 [[[[[V_A, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "A", "A", "A", "A"}));
    bernoulli_factors.push_back(1.0/1440.0);

    //  1/1440 [[[[[V_R, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "A", "A", "A", "A"}));
    bernoulli_factors.push_back(1.0/1440.0);

    // -1/1440 [[[[[V_N, sigma]_A, sigma]_A, sigma]_A, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"N", "A", "A", "A", "R", "A"}));
    bernoulli_factors.push_back(-1.0/1440.0);

    //  1/96   [[[[[V_N, sigma]_A, sigma]_R, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"N", "A", "R", "R", "R", "A"}));
    bernoulli_factors.push_back(1.0/96.0);

    // -1/288  [[[[[V_N, sigma]_A, sigma]_R, sigma]_A, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"N", "A", "R", "A", "R", "A"}));
    bernoulli_factors.push_back(-1.0/288.0);

    // -1/288  [[[[[V_N, sigma]_A, sigma]_R, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"N", "A", "R", "R", "A", "A"}));
    bernoulli_factors.push_back(-1.0/288.0);

    int dim = (int)ops.size();

    pq_helper pq("");

    for (size_t p = 0; p < partitions_lists.size(); p++){

        // mutable copies of targets and ops
        std::vector<std::string> b_targets;
        std::vector<std::string> b_ops1;
        std::vector<std::string> b_ops2;
        std::vector<std::string> b_ops3;
        std::vector<std::string> b_ops4;
        std::vector<std::string> b_ops5;

        for (auto target: targets){
            b_targets.push_back(target + partitions_lists[p][0]);
        }

        for (auto op: ops){
            b_ops1.push_back(op + partitions_lists[p][1]);
            b_ops2.push_back(op + partitions_lists[p][2]);
            b_ops3.push_back(op + partitions_lists[p][3]);
            b_ops4.push_back(op + partitions_lists[p][4]);
            b_ops5.push_back(op + partitions_lists[p][5]);
        }

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    for (int l = 0; l < dim; l++) {
                        for (int m = 0; m < dim; m++) {
                            std::vector<pq_operator_terms> tmp = pq.get_quintuple_commutator_terms(bernoulli_factors[p] * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[m]});
                            bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
                        }
                    }
                }
            }
        }
    }

    return bernoulli_terms;
}

// sixth-order bernoulli terms
std::vector<pq_operator_terms> get_bernoulli_operator_terms_6(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops) {

    std::vector<pq_operator_terms> bernoulli_terms;

    std::vector<std::vector<std::string> > partitions_lists;
    std::vector<double> bernoulli_factors;

    //     1/64    [[[[[[V_A, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "R", "R", "R", "R", "A"}));
    bernoulli_factors.push_back(1.0/64.0);

    //     1/64    [[[[[[V_R, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "R", "R", "R", "R", "A"}));
    bernoulli_factors.push_back(1.0/64.0);

    //    -1/192   [[[[[[V_A, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "A", "R", "R", "R", "A"}));
    bernoulli_factors.push_back(-1.0/192.0);

    //    -1/192   [[[[[[V_R, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "A", "R", "R", "R", "A"}));
    bernoulli_factors.push_back(-1.0/192.0);

    //    -1/192   [[[[[[V_A, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "R", "A", "R", "R", "A"}));
    bernoulli_factors.push_back(-1.0/192.0);

    //    -1/192   [[[[[[V_R, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "R", "A", "R", "R", "A"}));
    bernoulli_factors.push_back(-1.0/192.0);

    //    -1/192   [[[[[[V_A, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "R", "R", "A", "R", "A"}));
    bernoulli_factors.push_back(-1.0/192.0);

    //    -1/192   [[[[[[V_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "R", "R", "A", "R", "A"}));
    bernoulli_factors.push_back(-1.0/192.0);

    //     1/576   [[[[[[V_A, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "A", "R", "A", "R", "A"}));
    bernoulli_factors.push_back(1.0/576.0);

    //     1/576   [[[[[[V_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "A", "R", "A", "R", "A"}));
    bernoulli_factors.push_back(1.0/576.0);

    //     1/2880  [[[[[[V_A, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "A", "A", "A", "R", "A"}));
    bernoulli_factors.push_back(1.0/2880.0);

    //     1/2880  [[[[[[V_R, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "A", "A", "A", "R", "A"}));
    bernoulli_factors.push_back(1.0/2880.0);

    //    -1/192   [[[[[[V_A, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "R", "R", "R", "A", "A"}));
    bernoulli_factors.push_back(-1.0/192.0);

    //    -1/192   [[[[[[V_R, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "R", "R", "R", "A", "A"}));
    bernoulli_factors.push_back(-1.0/192.0);

    //     1/576   [[[[[[V_A, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "A", "R", "R", "A", "A"}));
    bernoulli_factors.push_back(1.0/576.0);

    //     1/576   [[[[[[V_R, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "A", "R", "R", "A", "A"}));
    bernoulli_factors.push_back(1.0/576.0);

    //     1/576   [[[[[[V_A, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "R", "A", "R", "A", "A"}));
    bernoulli_factors.push_back(1.0/576.0);

    //     1/576   [[[[[[V_R, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "R", "A", "R", "A", "A"}));
    bernoulli_factors.push_back(1.0/576.0);

    //     1/2880  [[[[[[V_A, sigma]_R, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"A", "R", "R", "A", "A", "A", "A"}));
    bernoulli_factors.push_back(1.0/2880.0);

    //     1/2880  [[[[[[V_R, sigma]_R, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"R", "R", "R", "A", "A", "A", "A"}));
    bernoulli_factors.push_back(1.0/2880.0);

    //     1/30240 [[[[[[V_N, sigma]_A, sigma]_A, sigma]_A, sigma]_A, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"N", "A", "A", "A", "A", "A", "A"}));
    bernoulli_factors.push_back(1.0/30240.0);

    //    -1/2880  [[[[[[V_N, sigma]_A, sigma]_A, sigma]_A, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"N", "A", "A", "A", "R", "R", "A"}));
    bernoulli_factors.push_back(-1.0/2880.0);

    //     1/192   [[[[[[V_N, sigma]_A, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"N" ,"A", "R", "R", "R", "R", "A"}));
    bernoulli_factors.push_back(1.0/192.0);

    //    -1/576   [[[[[[V_N, sigma]_A, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"N", "A", "R", "A", "R", "R", "A"}));
    bernoulli_factors.push_back(-1.0/576.0);

    //    -1/576   [[[[[[V_N, sigma]_A, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A
    partitions_lists.push_back(get_partitions_list({"N", "A", "R", "R", "A", "R", "A"}));
    bernoulli_factors.push_back(-1.0/576.0);

    //     1/8640  [[[[[[V_N, sigma]_A, sigma]_A, sigma]_A, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"N", "A", "A", "A", "R", "A", "A"}));
    bernoulli_factors.push_back(1.0/8640.0);

    //    -1/576   [[[[[[V_N, sigma]_A, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"N", "A", "R", "R", "R", "A", "A"}));
    bernoulli_factors.push_back(-1.0/576.0);

    //     1/1728  [[[[[[V_N, sigma]_A, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"N", "A", "R", "A", "R", "A", "A"}));
    bernoulli_factors.push_back(1.0/1728.0);

    //     1/8640  [[[[[[V_N, sigma]_A, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_A
    partitions_lists.push_back(get_partitions_list({"N", "A", "R", "A", "A", "A", "A"}));
    bernoulli_factors.push_back(1.0/8640.0);

    int dim = (int)ops.size();

    pq_helper pq("");

    for (size_t p = 0; p < partitions_lists.size(); p++){

        // mutable copies of targets and ops
        std::vector<std::string> b_targets;
        std::vector<std::string> b_ops1;
        std::vector<std::string> b_ops2;
        std::vector<std::string> b_ops3;
        std::vector<std::string> b_ops4;
        std::vector<std::string> b_ops5;
        std::vector<std::string> b_ops6;

        for (auto target: targets){
            b_targets.push_back(target + partitions_lists[p][0]);
        }

        for (auto op: ops){
            b_ops1.push_back(op + partitions_lists[p][1]);
            b_ops2.push_back(op + partitions_lists[p][2]);
            b_ops3.push_back(op + partitions_lists[p][3]);
            b_ops4.push_back(op + partitions_lists[p][4]);
            b_ops5.push_back(op + partitions_lists[p][5]);
            b_ops6.push_back(op + partitions_lists[p][6]);
        }

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    for (int l = 0; l < dim; l++) {
                        for (int m = 0; m < dim; m++) {
                            for (int n = 0; n < dim; n++) {
                                std::vector<pq_operator_terms> tmp = pq.get_hextuple_commutator_terms(bernoulli_factors[p] * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[m]}, {b_ops6[n]});
                                bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
                            }
                        }
                    }
                }
            }
        }
    }

    return bernoulli_terms;
}

} // End namespaces
