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
#include<cmath>
#include<sstream>
#include <map>
#include <memory>
#include <unordered_map>

namespace pdaggerq {

// work-around for finite precision of std::to_string
template <typename T> std::string to_string_with_precision(const T a_value, const int n = 14) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

// determine minimum precision needed
template <typename T> int minimum_precision(T factor) {
    constexpr double epsilon = 1.0e-10;
    double factor_abs = std::fabs((double)factor);

    if (fabs(factor_abs - epsilon) < epsilon)
        return 0;

    // represent factor as a string of a fixed precision
    std::stringstream ss;
    ss << std::fixed << 10*factor_abs;
    std::string str = ss.str();

    int precision = 0;
    bool decimal_point_encountered = false;
    bool is_repeated = false;
    char last_digit = ' ';
    int  repeat_count = 0;

    for (char digit : str) {
        is_repeated = (digit == last_digit);
        last_digit = digit;
        if (digit == '.') {
            decimal_point_encountered = true;
        } else if (decimal_point_encountered && is_repeated) {
            if (++repeat_count >= 12) break; // keep at most 12 repeating digits
        }

        if (!is_repeated) repeat_count = 0; // reset count
        if (decimal_point_encountered) precision++; // increment precision

    }

    // if the last repeating digit is zero, we can reduce the precision
    if (precision >= repeat_count && last_digit == '0')
        precision -= repeat_count;

    // we should always have at least two digits
    if (precision < 2)
        precision = 2;

    return precision;
}

class pq_string 
{

  private:

  public:

    static inline bool is_spin_blocked = false;
    static inline bool is_range_blocked = false;

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
     * copy constructor
     *
     */
    pq_string(const pq_string &copy_me) = default;

    /**
     *
     * copy constructor without copying symbols and daggers
     *
     */
    pq_string(pq_string* copy_me, bool copy_daggers_and_symbols) {
        copy(copy_me, copy_daggers_and_symbols);
    }

    /**
     *
     * assignment operator
     *
     */
    pq_string &operator=(const pq_string &copy_me) = default;

    /**
     *
     * move constructor
     *
     */
    pq_string(pq_string &&move_me) = default;

    /**
     *
     * move assignment operator
     *
     */
    pq_string &operator=(pq_string &&move_me) = default;

    /**
     *
     * destructor
     *
     */
    ~pq_string() = default;

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
     * sort amplitude, integral,and delta function labels and define key (for comparing strings)
     *
     */
    void sort();

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
    std::string integral_types[] {"fock", "core", "two_body", "eri", "d+", "d-", "occ_repulsion"};

    /**
     *
     * map integral_types onto lists of integrals
     *
     */
    std::unordered_map<std::string, std::vector<integrals> > ints;

    /**
     *
     * supported amplitude and rdm types
     *
     */
    static inline
    char amplitude_types[] {'l', 'r', 't', 'D'};

    /**
     *
     * map amplitude_types onto lists of amplitudes
     *
     */
    std::unordered_map<char, std::vector<amplitudes> > amps;

    /**
     *
     * non-summed spin labels
     *
     */
    std::unordered_map<std::string, std::string> non_summed_spin_labels;

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
    void print() const;

    /**
     *
     * serialize string information to a buffer
     * @param buffer: the buffer to send the string information to
     *
     */
    void serialize(std::ofstream &buffer) const;

    /**
     *
     * deserialize string information from a buffer
     * @param buffer: the buffer to read the string information from
     *
     */
    void deserialize(std::ifstream &buffer);

    /**
     *
     * return string information as list of std::string
     *
     */
    std::vector<std::string> get_string();

    /**
     *
     * return string identifier as std::string
     *
     */
    std::string get_key();

    /**
     *
     * string identifier as std::string
     *
     */
    std::string key;

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
    void set_range_everywhere(const std::string& target, const std::string& range);

    /**
     *
     * reset label ranges (so only non-summed labels are set)
     *
     */
    void reset_label_ranges(const std::unordered_map<std::string, std::vector<std::string>> &label_ranges);

    /**
     *
     * set labels for integrals
     *
     * @param type: the integrals_type
     * @param in: the list of labels for the integrals
     * @param op_portions: {"A", "N", "R", ...}, "A" = "N" + "R" (used for Bernoulli expansion)
     */
    void set_integrals(const std::string &type, const std::vector<std::string> &in, std::vector<std::string> op_portions = {});

    /**
     *
     * set labels for amplitudes
     *
     * @param type: the amplitudes_type
     * @param n_create: the number of labels corresponding to creation operators
     * @param n_annihilate: the number of labels corresponding to annihilation operators
     * @param n_ph: the number of photons
     * @param in: the list of labels for the amplitudes
     * @param op_portions: {"A", "N", "R", ...}, "A" = "N" + "R" (used for Bernoulli expansion)
     */
    void set_amplitudes(char type, int n_create, int n_annihilate, int n_ph, const std::vector<std::string> &in, std::vector<std::string> op_portions = {});

    /** 
     *
     * how many times does an index appear amplitudes, deltas, and integrals?
     *
     * @param idx: the index
     * @return: the number of times idx appears in the string
     *
     */
    int index_in_anywhere(const std::string &idx);

};

}

#endif
