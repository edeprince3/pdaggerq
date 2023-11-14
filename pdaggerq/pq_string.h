//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_string.h
// Copyright (C) 2020 A. Eugene DePrince III
//
// Author: A. Eugene DePrince III <adeprince@fsu.edu>
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
//  limitations under the License./>.
//

#ifndef PQ_STRING_H
#define PQ_STRING_H

#include "pq_tensor.h"
#include <map>

namespace pdaggerq {

class pq_string {

  private:

  public:

    /**
     *
     * constructor
     *
     * @param vacuum_type: normal order is defined with respect to the TRUE vacuum or the FERMI vacuum
     *
     */
    explicit pq_string(const std::string &vacuum_type);

    /**
     *
     * destructor
     *
     */
    ~pq_string();

    /**
     *
     * the vacuum type ("TRUE", "FERMI")
     *
     */
    std::string vacuum;

    /**
     *
     * the sign associated with string
     *
     */
    int sign = 1;

    /**
     *
     * should we skip this term when moving toward normal order and printing?
     *
     */
    bool skip = false;

    /**
     *
     * sort amplitude, integral, and delta function labels (useful when comparing strings)
     *
     */
    void sort_labels();

    /**
     *
     * a numerical factor associated with the string
     *
     */
    double factor = 1.0;

    /**
     *
     * a list of labels for fermionic creation / annihilation operators 
     *
     */
    std::vector<std::string> string;

    /**
     *
     * supported integral types
     *
     */
    static inline
    std::vector<std::string> integral_types = {"fock", "core", "two_body", "eri", "d+", "d-", "occ_repulsion"};

    /**
     *
     * map integral_types onto lists of integrals
     *
     */
    std::map<std::string, std::vector<integrals> > ints;

    /**
     *
     * supported amplitude types
     *
     */
    static inline
    std::vector<char> amplitude_types = {'l', 'r', 't', 'u', 'm', 's'};

    /**
     *
     * map amplitude_types onto lists of amplitudes
     *
     */
    std::map<char, std::vector<amplitudes> > amps;

    /**
     *
     * non-summed spin labels
     *
     */
    std::map<std::string, std::string> non_summed_spin_labels;

    /**
     *
     * a list of delta functions
     *
     */
    std::vector<delta_functions> deltas;

    /**
     *
     * a list of permutation operators: P(i,j) R(ijab) = R(ijab) - R(jiab)
     *
     */
    std::vector<std::string> permutations;

    /**
     *
     * a list of permutation operators: PP6(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + R(ikj;acb) + R(jik;bac) + R(jki;bca) + R(kij;cab) + R(kji;cba)
     *
     */
    std::vector<std::string> paired_permutations_6;

    /**
     *
     * a list of permutation operators: PP3(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + (jik;bac) + R(kji;cba)
     *
     */
    std::vector<std::string> paired_permutations_3;

    /**
     *
     * a list of permutation operators: PP2(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + (jik;bac)
     *
     */
    std::vector<std::string> paired_permutations_2;

    /**
     *
     * should we account for w0?
     *
     */
    bool has_w0 = false;

    /**
     *
     * a list indicating if bosonic operators are creators or annihilators
     *
     */
    std::vector<bool> is_boson_dagger;

    /**
     *
     * a list indicating if fermionic operators are creators or annihilators (relative to the true vacuum)
     *
     */
    std::vector<bool> is_dagger;

    /**
     *
     * a list indicating if fermionic operators are creators or annihilators (relative to the fermi vacuum)
     *
     */
    std::vector<bool> is_dagger_fermi;

    /**
     *
     * a list of symbols indicating the labels on fermionic creation / annihilation operators
     *
     */
    std::vector<std::string> symbol;

    /**
     *
     * is the string in normal order? checks both fermion and boson parts
     *
     */
    bool is_normal_order();

    /**
     *
     * is the bosonic part of the string in normal order?
     *
     */
    bool is_boson_normal_order();

    /**
     *
     * print string information to stdout
     *
     */
    void print();

    /**
     *
     * return string information as list of std::string
     *
     */
    std::vector<std::string> get_string();

    /**
     *
     * return string information as list of std::string (includes spin labels)
     *
     */
    std::vector<std::string> get_string_with_spin();

    /**
     *
     * return string information as list of std::string (includes label ranges)
     *
     */
    std::vector<std::string> get_string_with_label_ranges();

    /**
     *
     * copy string data, possibly excluding symbols and daggers. 
     *
     * @param copy_me: pointer to pq_string to be copied
     * @param copy_daggers_and_symbols: copy the dagers and symbols?
     */
    void copy(void * copy_me, bool copy_daggers_and_symbols = true);

    /**
     *
     * set spin labels in the integrals and amplitudes
     *
     * @param target: a target label in the integrals or amplitudes
     * @param spin: the spin label to be added to target
     */
    void set_spin_everywhere(const std::string &target, const std::string &spin);

    /**
     *
     * reset spin labels (so only non-summed labels are set)
     *
     */
    void reset_spin_labels();

    /**
     *
     * set label range in the integrals and amplitudes
     *
     * @param target: a target label in the integrals or amplitudes
     * @param range: the range to be added to target
     */
    void set_range_everywhere(std::string target, std::string range);

    /**
     *
     * reset label ranges (so only non-summed labels are set)
     *
     */
    void reset_label_ranges(std::map<std::string, std::vector<std::string> > label_ranges);

    /**
     *
     * set labels for integrals
     *
     * @param type: the integrals_type
     * @param in: the list of labels for the integrals
     */
    void set_integrals(const std::string &type, const std::vector<std::string> &in);

    /**
     *
     * set labels for amplitudes
     *
     * @param type: the amplitudes_type
     * @param n_create: the number of labels corresponding to creation operators
     * @param n_annihilate: the number of labels corresponding to annihilation operators
     * @param in: the list of labels for the amplitudes
     */
    void set_amplitudes(char type, int n_create, int n_annihilate, std::vector<std::string> in);

};

}

#endif
