//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_utils.h
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
#include<cmath>
#include<sstream>

#include "pq_tensor.h"
#include "pq_string.h"

namespace pdaggerq {

/// is a label classified as occupied?
bool is_occ(const std::string &idx);

/// is a label classified as virtual?
bool is_vir(const std::string &idx);

/// how many times does an index appear in deltas?
int index_in_deltas(const std::string &idx, const std::vector<delta_functions> &deltas);

/// how many times does an index appear in integrals?
int index_in_integrals(const std::string &idx, const std::vector<integrals> &ints);

/// how many times does an index appear in amplitudes?
int index_in_amplitudes(const std::string &idx, const std::vector<amplitudes> &amps);

/// how many times does an index appear in operators (symbol)?
int index_in_operators(const std::string &idx, const std::vector<std::string> &ops);

/// how many times does an index appear amplitudes, deltas, and integrals?
int index_in_anywhere(const std::shared_ptr<pq_string> &in, const std::string &idx);

/// replace one label with another (in delta functions)
void replace_index_in_deltas(const std::string &old_idx, const std::string &new_idx, std::vector<delta_functions> &deltas);

/// replace one label with another (in a given set of integrals)
void replace_index_in_integrals(const std::string &old_idx, const std::string &new_idx, std::vector<integrals> &ints);

/// replace one label with another (in a given set of amplitudes)
void replace_index_in_amplitudes(const std::string &old_idx, const std::string &new_idx, std::vector<amplitudes> &amps);

/// replace one label with another (in a given set of operators (symbol))
void replace_index_in_operators(const std::string &old_idx, const std::string &new_idx, std::vector<std::string> &ops);

/// replace one label with another (in integrals and amplitudes)
void replace_index_everywhere(std::shared_ptr<pq_string> &in, const std::string &old_idx, const std::string &new_idx);

/// swap two labels
void swap_two_labels(std::shared_ptr<pq_string> &in, const std::string &label1, const std::string &label2);

/// compare two strings
bool compare_strings(const std::shared_ptr<pq_string> &ordered_1, const std::shared_ptr<pq_string> &ordered_2, int & n_permute);

/// compare two lists of amplitudes
bool compare_amplitudes( const std::vector<amplitudes> &amps1,
                         const std::vector<amplitudes> &amps2,
                         int & n_permute );

/// compare two lists of integrals
bool compare_integrals( const std::vector<integrals> &ints1,
                        const std::vector<integrals> &ints2,
                        int & n_permute );

// consolidate terms that differ may differ by permutations of summed labels
void consolidate_permutations_plus_swaps(std::vector<std::shared_ptr<pq_string> > &ordered,
                                          const std::vector<std::vector<std::string> > &labels);

// consolidate terms that differ by permutations of non-summed labels
void consolidate_permutations_non_summed(
    std::vector<std::shared_ptr<pq_string> > &ordered,
    const std::vector<std::string> &labels);

// look for paired permutations of non-summed labels
// a) P3a(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + R(ikj;acb) + R(jik;bac) + R(jki;bca) + R(kij;cab) + R(kji;cba)
// b) P3b(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + (jik;bac) + R(kji;cba)
void consolidate_paired_permutations_non_summed(
    std::vector<std::shared_ptr<pq_string> > &ordered,
    const std::vector<std::string> &occ_labels,
    const std::vector<std::string> &vir_labels,
    int n_fold);

/// compare two strings when swapping (multiple) summed labels
void compare_strings_with_swapped_summed_labels(const std::vector<std::vector<std::string> > &labels,
                                                size_t iter,
                                                const std::shared_ptr<pq_string> &in1,
                                                const std::shared_ptr<pq_string> &in2,
                                                int & n_permute,
                                                bool & strings_same);

/// compare two strings when swapping (multiple) summed labels and ov pairs of nonsumed labels
void compare_strings_with_swapped_summed_and_nonsummed_labels(
    const std::vector<std::vector<std::string> > &labels,
    const std::vector<std::vector<std::string>> &pairs,
    size_t iter,
    const std::shared_ptr<pq_string> &in1,
    const std::shared_ptr<pq_string> &in2,
    size_t in2_id,
    std::vector<size_t> &my_permutations,
    std::vector<bool> &permutation_types,
    int n_permutation_type,
    int & n_permute,
    bool & strings_same,
    bool & found_paired_permutation);

/// alphabetize operators to simplify string comparisons (for true vacuum only)
void alphabetize(std::vector<std::shared_ptr<pq_string> > &ordered);

/// cancel terms where appropriate
void cleanup(std::vector<std::shared_ptr<pq_string> > &ordered, bool find_paired_permutations);

/// reorder t amplitudes as t1, t2, t3, t4
void reorder_t_amplitudes(std::shared_ptr<pq_string> &in);

/// re-classify fluctuation potential terms
void reclassify_integrals(std::shared_ptr<pq_string> &in);

/// apply delta functions to amplitude and integral labels
void gobble_deltas(std::shared_ptr<pq_string> &in);

/// replace internal labels with conventional ones (o1 -> i, etc.)
void use_conventional_labels(std::shared_ptr<pq_string> &in);

// bring a new string to normal order and add to list of normal ordered strings (fermi vacuum)
void add_new_string_true_vacuum(const std::shared_ptr<pq_string> &in, std::vector<std::shared_ptr<pq_string> > &ordered, int print_level, bool find_paired_permutations);

// bring a new string to normal order and add to list of normal ordered strings (fermi vacuum)
void add_new_string_fermi_vacuum(const std::shared_ptr<pq_string> &in, std::vector<std::shared_ptr<pq_string> > &ordered, int print_level, bool find_paired_permutations, int occ_label_count, int vir_label_count);

/// concatinate a list of operators (a list of strings) into a single list
std::vector<std::string> concatinate_operators(const std::vector<std::vector<std::string>> &ops);

/// remove "*" from std::string
void removeStar(std::string &x);

/// remove "(" and ")" from std::string
void removeParentheses(std::string &x);

/// remove " " from std::string
void removeSpaces(std::string &x);

/// expand general labels, p -> o,v
bool expand_general_labels(const std::shared_ptr<pq_string> & in, std::vector<std::shared_ptr<pq_string> > & list, int occ_label_count, int vir_label_count);

}

#endif 
