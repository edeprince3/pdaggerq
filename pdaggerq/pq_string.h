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

    /// constructor
    pq_string(std::string vacuum_type);

    /// descructor
    ~pq_string();

    // vacuum type ("TRUE", "FERMI")
    std::string vacuum;

    // sign associated with string
    int sign = 1;

    // skip this term when moving toward normal order and printing
    bool skip = false;

    // sort amplitude, integral, and delta function labels
    void sort_labels();

    /// factor
    double factor = 1.0;

    /// list: labels for fermionic creation / annihilation operators 
    std::vector<std::string> string;

    /// integral types
    std::vector<std::string> integral_types = {"fock", "core", "two_body", "eri", "d+", "d-", "occ_repulsion"};

    /// integrals
    std::map<std::string, std::vector<integrals> > ints;

    /// amplitude types
    std::vector<char> amplitude_types = {'l', 'r', 't', 'u', 'm', 's'};

    /// amplitudes
    std::map<char, std::vector<amplitudes> > amps;

    /// non-summed spin labels
    std::map<std::string, std::string> non_summed_spin_labels;

    /// delta functions
    std::vector<delta_functions> deltas;

    /// list: labels permutation operators
    std::vector<std::string> permutations;

    /// should we account for w0?
    bool has_w0 = false;

    /// list: is bosonic operator creator or annihilator?
    std::vector<bool> is_boson_dagger;

    /// list: is fermionic operator creator or annihilator (relative to true vacuum)?
    std::vector<bool> is_dagger;

    /// list: is fermionic operator creator or annihilator (relative to fermi vacuum)?
    std::vector<bool> is_dagger_fermi;

    /// list: symbols for fermionic creation / annihilation operators
    std::vector<std::string> symbol;

    /// is fermion part of string in normal order?
    bool is_normal_order();

    /// is boson part of string in normal order?
    bool is_boson_normal_order();

    /// print string information
    void print();

    /// return string information
    std::vector<std::string> get_string();

    /// return string information (with spin)
    std::vector<std::string> get_string_with_spin();

    // copy all data, except symbols and daggers. 
    void shallow_copy(void * copy_me);

    /// copy all data, including symbols and daggers. 
    void copy(void * copy_me);

    /// set spin labels in integrals and amplitudes
    void set_spin_everywhere(std::string target, std::string spin);

    /// reset spin labels (so only non-summed labels are set)
    void reset_spin_labels();

    /// set labels for integrals
    void set_integrals(std::string type, std::vector<std::string> in);

    /// set labels for amplitudes
    void set_amplitudes(char type, int order, std::vector<std::string> in);

};

}

#endif