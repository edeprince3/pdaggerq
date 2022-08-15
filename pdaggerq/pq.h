//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq.h
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

#ifndef SQE_H
#define SQE_H

#include "data.h"

namespace pdaggerq {

class pq {

  private:

    /// is the entire string (fermions+bosons) in normal order?
    bool is_normal_order();

    /// are bosonic operators in normal order?
    bool is_boson_normal_order();

    /// how many times does label "idx" appear in any term?
    int index_in_anywhere(std::string idx);

    /// how many times does label "idx" appear in delta terms?
    int index_in_deltas(std::string idx);

    /// how many times does label "idx" appear in a given set of integrals
    int index_in_integrals(std::string idx, std::vector<integrals> ints);

    /// how many times does label "idx" appear in a given set of amplitudes
    int index_in_amplitudes(std::string idx, std::vector<amplitudes> amps);

    /// replace one label with another (everywhere)
    void replace_index_everywhere(std::string old_idx, std::string new_idx);

    /// replace one label with another (in delta functions)
    void replace_index_in_deltas(std::string old_idx, std::string new_idx);

    /// replace one label with another (in a given set of integrals)
    void replace_index_in_integrals(std::string old_idx, std::string new_idx, std::vector<integrals> &ints);

    /// replace one label with another (in a given set of amplitudes)
    void replace_index_in_amplitudes(std::string old_idx, std::string new_idx, std::vector<amplitudes> &amps);

    /// are two strings the same? if so, how many permutations to relate them?
    bool compare_strings(std::shared_ptr<pq> ordered_1, std::shared_ptr<pq> ordered_2, int & n_permute);

    /// swap to labels
    void swap_two_labels(std::string label1, std::string label2);

    /// compare two lists of amplitudes 
    bool compare_amplitudes( std::vector<amplitudes> amps1, 
                             std::vector<amplitudes> amps2, 
                             int & n_permute );

    /// compare two lists of integrals 
    bool compare_integrals( std::vector<integrals> ints1, 
                            std::vector<integrals> ints2, 
                            int & n_permute );

  public:

    /// constructor
    pq(std::string vacuum_type);

    /// destructor
    ~pq();

    /// vacuum type (fermi, true)
    std::string vacuum;

    /// do skip because will evaluate to zero?
    bool skip     = false;

    /// sign
    int sign      = 1;

    /// copy all data, except symbols and daggers. 
    void shallow_copy(void * copy_me);

    /// copy all data, including symbols and daggers. 
    void copy(void * copy_me);

    /// list: symbols for fermionic creation / annihilation operators
    std::vector<std::string> symbol;

    /// list: is fermionic operator creator or annihilator (relative to true vacuum)?
    std::vector<bool> is_dagger;

    /// list: is fermionic operator creator or annihilator (relative to fermi vacuum)?
    std::vector<bool> is_dagger_fermi;

    /// list of delta functions (index 1)
    std::vector<std::string> delta1;

    /// list of delta functions (index 2)
    std::vector<std::string> delta2;

    /// detailed information about string (t, R, L, amplitudes, bosons, etc.)
    std::shared_ptr<StringData> data;

    /// print string information
    void print();

    /// get string information
    std::vector<std::string> get_string();

    /// apply delta functions to amplitude and integral labels
    void gobble_deltas();

    /// replace internal labels with conventional ones (o1 -> i, etc.)
    void use_conventional_labels();

    /// bring string to normal order (relative to either vacuum)
    bool normal_order(std::vector<std::shared_ptr<pq> > &ordered);

    /// bring string to normal order relative to fermi vacuum
    bool normal_order_fermi_vacuum(std::vector<std::shared_ptr<pq> > &ordered);

    /// bring string to normal order relative to true vacuum
    bool normal_order_true_vacuum(std::vector<std::shared_ptr<pq> > &ordered);

    /// alphabetize operators to simplify string comparisons
    void alphabetize(std::vector<std::shared_ptr<pq> > &ordered);

    /// cancel terms where appropriate
    void cleanup(std::vector<std::shared_ptr<pq> > &ordered);

    /// consolidate terms that differ by summed labels plus permutations
    void consolidate_permutations_plus_swap(
        std::vector<std::shared_ptr<pq> > &ordered,
        std::vector<std::string> labels);

    /// consolidate terms that differ by two summed labels plus permutations
    void consolidate_permutations_plus_two_swaps(
        std::vector<std::shared_ptr<pq> > &ordered,
        std::vector<std::string> labels_1,
        std::vector<std::string> labels_2);

    /// consolidate terms that differ by three summed labels plus permutations
    void consolidate_permutations_plus_three_swaps(
        std::vector<std::shared_ptr<pq> > &ordered,
        std::vector<std::string> labels_1,
        std::vector<std::string> labels_2,
        std::vector<std::string> labels_3);

    /// consolidate terms that differ by four summed labels plus permutations
    void consolidate_permutations_plus_four_swaps(
        std::vector<std::shared_ptr<pq> > &ordered,
        std::vector<std::string> labels_1,
        std::vector<std::string> labels_2,
        std::vector<std::string> labels_3,
        std::vector<std::string> labels_4);

    /// consolidate terms that differ by five summed labels plus permutations
    void consolidate_permutations_plus_five_swaps(
        std::vector<std::shared_ptr<pq> > &ordered,
        std::vector<std::string> labels_1,
        std::vector<std::string> labels_2,
        std::vector<std::string> labels_3,
        std::vector<std::string> labels_4, 
        std::vector<std::string> labels_5);

    /// consolidate terms that differ by six summed labels plus permutations
    void consolidate_permutations_plus_six_swaps(
        std::vector<std::shared_ptr<pq> > &ordered,
        std::vector<std::string> labels_1,
        std::vector<std::string> labels_2,
        std::vector<std::string> labels_3,
        std::vector<std::string> labels_4, 
        std::vector<std::string> labels_5,
        std::vector<std::string> labels_6);

    /// consolidate terms that differ by seven summed labels plus permutations
    void consolidate_permutations_plus_seven_swaps(
        std::vector<std::shared_ptr<pq> > &ordered,
        std::vector<std::string> labels_1,
        std::vector<std::string> labels_2,
        std::vector<std::string> labels_3,
        std::vector<std::string> labels_4, 
        std::vector<std::string> labels_5,
        std::vector<std::string> labels_6,
        std::vector<std::string> labels_7);

    /// consolidate terms that differ by eight summed labels plus permutations
    void consolidate_permutations_plus_eight_swaps(
        std::vector<std::shared_ptr<pq> > &ordered,
        std::vector<std::string> labels_1,
        std::vector<std::string> labels_2,
        std::vector<std::string> labels_3,
        std::vector<std::string> labels_4, 
        std::vector<std::string> labels_5,
        std::vector<std::string> labels_6,
        std::vector<std::string> labels_7,
        std::vector<std::string> labels_8);

    /// consolidate terms that differ by permutations of non-summed labels

    /// consolidate terms that differ by permutations of non-summed labels
    void consolidate_permutations_non_summed(
        std::vector<std::shared_ptr<pq> > &ordered,
        std::vector<std::string> labels);


    /// consolidate terms that differ by permutations
    void consolidate_permutations(std::vector<std::shared_ptr<pq> > &ordered);

    /// sort amplitude and integral labels
    void sort_labels();

    /// reorder t amplitudes as t1, t2, t3
    void reorder_t_amplitudes();

    /// does label correspond to occupied orbital?
    bool is_occ(std::string idx);

    /// does label correspond to virtual orbital?
    bool is_vir(std::string idx);

    /// re-classify fluctuation potential terms
    void reclassify_integrals();
};

}

#endif
