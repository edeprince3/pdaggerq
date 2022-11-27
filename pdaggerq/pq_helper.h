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

#include "pq.h"
#include "data.h"

namespace pdaggerq {

class pq_helper {

  private:

    /// list of strings of operators
    std::vector< std::shared_ptr<pq> > ordered;

    /// strings, amplitudes, integrals, etc.
    std::shared_ptr<StringData> data;

    /// vacuum (fermi or true)
    std::string vacuum;

    /// print level
    int print_level;

    /// operators to apply to the left of any operator products we add
    std::vector<std::vector<std::string> > left_operators;

    /// operators to apply to the right of any operator products we add
    std::vector<std::vector<std::string> > right_operators;

    /// do operators entering a similarity transformation commute?
    bool cluster_operators_commute_;

    /// right-hand operators type (EE, IP, EA)
    std::string right_operators_type;

    /// left-hand operators type (EE, IP, EA)
    std::string left_operators_type;

  public:

    /// constructor
    pq_helper(std::string vacuum_type);

    /// destructor
    ~pq_helper();

    /// set operators to apply to the left of any operator products we add
    void set_left_operators(std::vector<std::vector<std::string> > in);

    /// set operators to apply to the right of any operator products we add
    void set_right_operators(std::vector<std::vector<std::string> >in);

    /// set print level (default zero)
    void set_print_level(int level);

    /// set a string of creation / annihilation operators
    void set_string(std::vector<std::string> in);

    /// set labels for integrals
    void set_integrals(std::string type, std::vector<std::string> in);

    /// set labels for amplitudes
    void set_amplitudes(char type, std::vector<std::string> in, bool is_reference);

    /// set a numerical factor
    void set_factor(double in);

    /// add new completed string / integrals / amplitudes / factor
    void add_new_string(std::vector<std::vector<std::string>> spin_labels);

    /// add new completed string / integrals / amplitudes / factor (assuming normal order is definied relative to the true vacuum
    void add_new_string_true_vacuum(std::vector<std::vector<std::string>> spin_labels);

    /// add new completed string / integrals / amplitudes / factor (assuming normal order is definied relative to the fermi vacuum
    void add_new_string_fermi_vacuum(std::vector<std::vector<std::string>> spin_labels);

    /// add new complete string as a product of operators (i.e., {'h(pq)','t1(ai)'} )
    void add_operator_product(double factor, std::vector<std::string> in, std::vector<std::vector<std::string> > spin_labels = { {}, {} });

    /// add similarity-transformed operator expansion of an operator
    void add_st_operator(double factor, std::vector<std::string> targets, 
                                        std::vector<std::string> ops, 
                                        std::vector<std::vector<std::string>> spin_labels = { {}, {} });

    /// add commutator of two operators
    void add_commutator(double factor, std::vector<std::string> op0,
                                       std::vector<std::string> op1, 
                                       std::vector<std::vector<std::string>> spin_labels = { {}, {} });

    /// add double commutator involving three operators
    void add_double_commutator(double factor, std::vector<std::string> op0,
                                              std::vector<std::string> op1,
                                              std::vector<std::string> op2, 
                                              std::vector<std::vector<std::string>> spin_labels = { {}, {} });

    /// add triple commutator involving four operators
    void add_triple_commutator(double factor, std::vector<std::string> op0,
                                              std::vector<std::string> op1,
                                              std::vector<std::string> op2,
                                              std::vector<std::string> op3, 
                                              std::vector<std::vector<std::string>> spin_labels = { {}, {} });

    /// add quadruple commutator involving five operators
    void add_quadruple_commutator(double factor, std::vector<std::string> op0,
                                                 std::vector<std::string> op1,
                                                 std::vector<std::string> op2,
                                                 std::vector<std::string> op3,
                                                 std::vector<std::string> op4, 
                                                 std::vector<std::vector<std::string>> spin_labels = { {}, {} });

    /// cancel terms, if possible
    void simplify();

    /// clear strings
    void clear();

    /// print strings
    void print();

    /// get list of strings 
    std::vector<std::vector<std::string> > strings();

    /// print fully-contracted strings
    void print_fully_contracted();

    /// get list of fully-contracted strings
    std::vector<std::vector<std::string> > fully_contracted_strings();

    /// get list of fully-contracted strings, after spin tracing
    std::vector<std::vector<std::string> > fully_contracted_strings_with_spin(std::map<std::string, std::string> spin_labels);

    /// print one-body strings
    void print_one_body();

    /// print two-body strings
    void print_two_body();

    /// do operators entering similarity transformation commute? default true
    void set_cluster_operators_commute(bool cluster_operators_commute);

    /// set right-hand operators type (EE, IP, EA)
    void set_right_operators_type(std::string type);

    /// set right-hand operators type (EE, IP, EA)
    void set_left_operators_type(std::string type);

};

}

#endif
