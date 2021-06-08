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

    /// is label "idx" present in tensor term?
    bool index_in_tensor(std::string idx);

    /// is label "idx" present in t-amplitudes?
    bool index_in_t_amplitudes(std::string idx);

    /// is label "idx" present in u-amplitudes?
    bool index_in_u_amplitudes(std::string idx);

    /// is label "idx" present in m-amplitudes?
    bool index_in_m_amplitudes(std::string idx);

    /// is label "idx" present in s-amplitudes?
    bool index_in_s_amplitudes(std::string idx);

    /// is label "idx" present in left-hand amplitudes?
    bool index_in_left_amplitudes(std::string idx);

    /// is label "idx" present in right-hand amplitudes?
    bool index_in_right_amplitudes(std::string idx);

    /// replace one label with another (everywhere)
    void replace_index_everywhere(std::string old_idx, std::string new_idx);

    /// replace one label with another (in tensor)
    void replace_index_in_tensor(std::string old_idx, std::string new_idx);

    /// replace one label with another (in t-amplitudes)
    void replace_index_in_t_amplitudes(std::string old_idx, std::string new_idx);

    /// replace one label with another (in u-amplitudes)
    void replace_index_in_u_amplitudes(std::string old_idx, std::string new_idx);

    /// replace one label with another (in m-amplitudes)
    void replace_index_in_m_amplitudes(std::string old_idx, std::string new_idx);

    /// replace one label with another (in s-amplitudes)
    void replace_index_in_s_amplitudes(std::string old_idx, std::string new_idx);

    /// replace one label with another (in left-hand amplitudes)
    void replace_index_in_left_amplitudes(std::string old_idx, std::string new_idx);

    /// replace one label with another (in right-hand amplitudes)
    void replace_index_in_right_amplitudes(std::string old_idx, std::string new_idx);

    /// are two strings the same? if so, how many permutations to relate them?
    bool compare_strings(std::shared_ptr<pq> ordered_1, std::shared_ptr<pq> ordered_2, int & n_permute);

    /// prioritize summation labels as i > j > k > l and a > b > c > d.
    void update_summation_labels();

    //// move bra lables the right t_amplitudes and tensor
    void update_bra_labels();

    /// swap to labels
    void swap_two_labels(std::string label1, std::string label2);

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

    /// check if string should be zero by spin symmetry (no longer supported)
    void check_spin();

    /// check if string should be zero by o/v labels in delta function
    void check_occ_vir();

    /// apply delta functions to string / tensor labels
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

    /// reorder t amplitudes as t1, t2, t3
    void reorder_t_amplitudes();

    /// does label correspond to occupied orbital?
    bool is_occ(std::string idx);

    /// does label correspond to virtual orbital?
    bool is_vir(std::string idx);

    /// re-classify fluctuation potential terms
    void reclassify_tensors();
};

}

#endif
