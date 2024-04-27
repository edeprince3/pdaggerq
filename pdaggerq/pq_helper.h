//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_helper.h
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

#ifndef PQ_HELPER_H
#define PQ_HELPER_H

#include "pq_string.h"

namespace pdaggerq {

class pq_helper {

  public:

    /**
     *
     * constructor
     *
     * @param vacuum_type: normal order is defined with respect to the TRUE vacuum or the FERMI vacuum
     *
     */
    explicit pq_helper(const std::string &vacuum_type);

    /**
     *
     * destructor
     *
     */
    ~pq_helper() = default;

    /**
     *
     * set operators to apply to the left of any operator products we add
     *
     * @param in: strings indicating a sum (outer list) of products (inner lists) of operators that define the bra state
     *
     */
    void set_left_operators(const std::vector<std::vector<std::string>> &in);

    /**
     *
     * set operators to apply to the right of any operator products we add
     *
     * @param in: strings indicating a sum (outer list) of products (inner lists) of operators that define the ket state
     *
     */
    void set_right_operators(const std::vector<std::vector<std::string>> &in);

    /**
     *
     * set right-hand operators type
     *
     * @param type: a string specifying the type of operators that define the ket state ("EE", "IP", "EA", "DEA", "DIP")
     *
     */
    void set_right_operators_type(const std::string& type);

    /**
     *
     * get right-hand operators type
     *
     * @return type: a string specifying the type of operators that define the ket state ("EE", "IP", "EA", "DEA", "DIP")
     *
     */
    std::string get_right_operators_type(){return right_operators_type;}

    /**
     *
     * set left-hand operators type
     *
     * @param type: a string specifying the type of operators that define the bra state ("EE", "IP", "EA", "DEA", "DIP")
     *
     */
    void set_left_operators_type(const std::string& type);

    /**
     *
     * get left-hand operators type
     *
     * @return type: a string specifying the type of operators that define the ket state ("EE", "IP", "EA", "DEA", "DIP")
     *
     */
    std::string get_left_operators_type(){return left_operators_type;}

    /**
     *
     * set whether operators entering similarity transformation commute
     *
     * @param do_cluster_operators_commute: true/false
     *
     */
    void set_cluster_operators_commute(bool do_cluster_operators_commute);

    /**
     *
     * set whether we should search for paired ov permutations that arise in ccsdt
     *
     * @param do_find_paired_permutations: true/false
     *
     */
    void set_find_paired_permutations(bool do_find_paired_permutations);

    /**
     *
     * set print level 
     *
     * @param level: an integer. any value greater than zero will cause the code to print starting strings
     *
     */
    void set_print_level(int level);

    /**
     *
     * set whether final strings contain bare creation / annihilation operators or their expectation value (rdms)?
     *
     * @param do_use_rdms
     *
     */
    void set_use_rdms(bool do_use_rdms, std::vector<int> ignore_cumulant);

    /**
     *
     * add a product of operators (i.e., {'h','t1'} )
     *
     * @param in: a list of strings defining the operator product
     *
     */
    void add_operator_product(double factor, std::vector<std::string> in);

    /**
     *
     * add a similarity-transformed operator using the BCH expansion and four nested commutators
     * exp(-T) f exp(T) = f + [f, T] + 1/2 [[f, T], T] + 1/6 [[[f, T], T], T] + 1/24 [[[[f, T], T], T], T]
     *
     * @param targets: a list of strings defining the operator product to be transformed (here, f)
     * @param ops: a list of strings defining a sum of operators that define the transformation (here, T)
     *
     */
    void add_st_operator(double factor, const std::vector<std::string> &targets,
                                        const std::vector<std::string> &ops);

    /**
     *
     * add a anticommutator of two operators, {op0, op1}
     *
     * @param op0: a list of strings defining an operator product
     * @param op1: a list of strings defining an operator product
     *
     */
    void add_anticommutator(double factor, const std::vector<std::string> &op0,
                                           const std::vector<std::string> &op1);

    /**
     *
     * add a commutator of two operators, [op0, op1]
     *
     * @param op0: a list of strings defining an operator product
     * @param op1: a list of strings defining an operator product
     *
     */
    void add_commutator(double factor, const std::vector<std::string> &op0,
                                       const std::vector<std::string> &op1);

    /**
     *
     * add a double commutator involving three operators, [[op0, op1], op2]
     *
     * @param op0: a list of strings defining an operator product
     * @param op1: a list of strings defining an operator product
     * @param op2: a list of strings defining an operator product
     *
     */
    void add_double_commutator(double factor, const std::vector<std::string> &op0,
                                              const std::vector<std::string> &op1,
                                              const std::vector<std::string> &op2);

    /**
     *
     * add a triple commutator involving four operators, [[[op0, op1], op2], op3]
     *
     * @param op0: a list of strings defining an operator product
     * @param op1: a list of strings defining an operator product
     * @param op2: a list of strings defining an operator product
     * @param op3: a list of strings defining an operator product
     *
     */
    void add_triple_commutator(double factor, const std::vector<std::string> &op0,
                                              const std::vector<std::string> &op1,
                                              const std::vector<std::string> &op2,
                                              const std::vector<std::string> &op3);

    /**
     *
     * add a quadrupole commutator involving five operators, [[[[op0, op1], op2], op3], op4]
     *
     * @param op0: a list of strings defining an operator product
     * @param op1: a list of strings defining an operator product
     * @param op2: a list of strings defining an operator product
     * @param op3: a list of strings defining an operator product
     * @param op4: a list of strings defining an operator product
     *
     */
    void add_quadruple_commutator(double factor, const std::vector<std::string> &op0,
                                                 const std::vector<std::string> &op1,
                                                 const std::vector<std::string> &op2,
                                                 const std::vector<std::string> &op3,
                                                 const std::vector<std::string> &op4);

    /**
     *
     * cancel terms, if possible, and identify permutations of non-summed labels
     *
     */
    void simplify();

    /**
     *
     * clear the current list of strings. note that the right- and left-hand operators
     * set using set_left/right_operators will not be cleared. if you want to change 
     * these, you must call the relevant functions again.
     *
     */
    void clear();

    /**
     *
     * get a list of all strings 
     *
     */
    std::vector<std::vector<std::string> > strings() const;

    /**
     *
     * get list of fully-contracted strings
     *
     */
    std::vector<std::vector<std::string> > fully_contracted_strings() const;

    /**
     *
     * get list of fully-contracted strings, after spin tracing
     *
     * @param spin_labels: a map/dictionary mapping non-summed labels onto spins ("a" or "b")
     */
    [[deprecated("use pq_helper::block_by_spin(spin_labels) instead")]]
    std::vector<std::vector<std::string> > fully_contracted_strings_with_spin(const std::unordered_map<std::string, std::string> &spin_labels);

    /**
     *
     * this function is used to block strings by spin
     *
     */
    void block_by_spin(const std::unordered_map<std::string, std::string> &spin_labels);

    /**
     *
     * get list of fully-contracted strings, after assigning ranges to the labels
     *
     * @param label_ranges: a map/dictionary mapping non-summed labels onto ranges ("all", "active", or "external")
     */
    [[deprecated("use pq_helper::block_by_range(label_ranges) instead")]]
    std::vector<std::vector<std::string> > fully_contracted_strings_with_ranges(
            const std::unordered_map<std::string, std::vector<std::string>> &label_ranges);

    /**
     *
     * this function is used to block strings by label ranges
     *
     */
    void block_by_range(const std::unordered_map<std::string, std::vector<std::string>> &label_ranges);

    /**
     *
     * print strings to stdout
     *
     * @param string_type: a string specifying which strings to print ("all", "one-body", "two-body", "fully-contracted").
     *                     on the python side, the default value is "fully-contracted"
     */
    void print(const std::string &string_type) const;

    /**
     *
     * the number of threads
     *
     */
    static inline int nthreads = 1;

private:

    /**
     *
     * a list of strings of operators/amplitudes/integrals/deltas
     *
     */
    std::vector< std::shared_ptr<pq_string> > ordered;
    std::vector< std::shared_ptr<pq_string> > ordered_blocked;

    /**
     *
     * the vacuum type ("TRUE" or "FERMI")
     *
     */
    std::string vacuum;

    /**
     *
     * the print level
     *
     */
    int print_level;

    /**
     *
     * should final strings contain bare creation / annihilation operators or their expectation value (rdms)?
     *
     */
    bool use_rdms;

    /**
     *
     * if final string contains rdms, which n-body cumulants should we ignore
     *
     */
    std::vector<int> ignore_cumulant_rdms = {};

    /**
     *
     * sum (outer list) of products (inner list) defining the bra state
     *
     */
    std::vector<std::vector<std::string> > left_operators;

    /**
     *
     * sum (outer list) of products (inner list) defining the ket state
     *
     */
    std::vector<std::vector<std::string> > right_operators;

    /**
     *
     * opertor type for operators defining the ket state
     *
     */
    std::string right_operators_type;

    /**
     *
     * opertor type for operators defining the bra state
     *
     */
    std::string left_operators_type;

    /**
     *
     * do the operators entering a similarity transformation commute?
     *
     */
    bool cluster_operators_commute;

    /**
     *
     * should we look for paired ov permutations that arise in ccsdt?
     *
     */
    bool find_paired_permutations;

};

}

#endif
