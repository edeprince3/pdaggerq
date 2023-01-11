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

#ifndef PQ_UTILS_H
#define PQ_UTILS_H

#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include<algorithm>
#include<cstring>
#include<math.h>
#include<sstream>

#include "tensor.h"
#include "pq.h"

namespace pdaggerq {

/// is a label classified as occupied?
bool is_occ(std::string idx);

/// is a label classified as virtual?
bool is_vir(std::string idx);

/// how many times does an index appear deltas?
int index_in_deltas(std::string idx, std::vector<delta_functions> deltas);

/// how many times does an index appear integrals?
int index_in_integrals(std::string idx, std::vector<integrals> ints);

/// how many times does an index appear amplitudes?
int index_in_amplitudes(std::string idx, std::vector<amplitudes> amps);

/// how many times does an index appear amplitudes, deltas, and integrals?
int index_in_anywhere(std::shared_ptr<StringData> data, std::string idx);

/// replace one label with another (in delta functions)
void replace_index_in_deltas(std::string old_idx, std::string new_idx, std::vector<delta_functions> &deltas);

/// replace one label with another (in a given set of integrals)
void replace_index_in_integrals(std::string old_idx, std::string new_idx, std::vector<integrals> &ints);

/// replace one label with another (in a given set of amplitudes)
void replace_index_in_amplitudes(std::string old_idx, std::string new_idx, std::vector<amplitudes> &amps);

/// replace one label with another (in integrals and amplitudes)
void replace_index_everywhere(std::shared_ptr<StringData> data, std::string old_idx, std::string new_idx);

/// swap two labels
void swap_two_labels(std::shared_ptr<StringData> data, std::string label1, std::string label2);

/// compare two strings
bool compare_strings(std::shared_ptr<pq> ordered_1, std::shared_ptr<pq> ordered_2, int & n_permute);

/// compare two lists of amplitudes
bool compare_amplitudes( std::vector<amplitudes> amps1,
                         std::vector<amplitudes> amps2,
                         int & n_permute );

/// compare two lists of integrals
bool compare_integrals( std::vector<integrals> ints1,
                        std::vector<integrals> ints2,
                        int & n_permute );

/// consolidate terms that differ by permutations
void consolidate_permutations(std::vector<std::shared_ptr<pq> > &ordered);

/// consolidate terms that differ by summed labels plus permutations
void consolidate_permutations_plus_swap(std::vector<std::shared_ptr<pq> > &ordered,
                                        std::vector<std::string> labels);

/// consolidate terms that differ by two summed labels plus permutations
void consolidate_permutations_plus_two_swaps(
    std::vector<std::shared_ptr<pq> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2);

/// alphabetize operators to simplify string comparisons (for true vacuum only)
void alphabetize(std::vector<std::shared_ptr<pq> > &ordered);

/// cancel terms where appropriate
void cleanup(std::vector<std::shared_ptr<pq> > &ordered);

/// reorder t amplitudes as t1, t2, t3, t4
void reorder_t_amplitudes(std::shared_ptr<pq> in);

/// reorder three spin labels as aab or abb
void reorder_three_spins(amplitudes & amps, int i1, int i2, int i3, int & sign);

/// reorder four spin labels as aaab, aabb, or abbb
void reorder_four_spins(amplitudes & amps, int i1, int i2, int i3, int i4, int & sign);

/// re-classify fluctuation potential terms
void reclassify_integrals(std::shared_ptr<pq> in);

/// apply delta functions to amplitude and integral labels
void gobble_deltas(std::shared_ptr<pq> in);

/// replace internal labels with conventional ones (o1 -> i, etc.)
void use_conventional_labels(std::shared_ptr<pq> in);

/// concatinate a list of operators (a list of strings) into a single list
std::vector<std::string> concatinate_operators(std::vector<std::vector<std::string>> ops);

/// remove "*" from std::string
void removeStar(std::string &x);

/// remove "(" and ")" from std::string
void removeParentheses(std::string &x);

}

#endif 
