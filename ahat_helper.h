//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: ahat_helper.h
// Copyright (C) 2020 A. Eugene DePrince III
//
// Author: A. Eugene DePrince III <adeprince@fsu.edu>
// Maintainer: DePrince group
//
// This file is part of the pdaggerq package.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef AHAT_HELPER_H
#define AHAT_HELPER_H

#include "ahat.h"
#include "data.h"

namespace pdaggerq {

class ahat_helper {

  private:

    /// list of strings of operators
    std::vector< std::shared_ptr<ahat> > ordered;

    /// strings, tensors, etc.
    std::shared_ptr<StringData> data;

    /// vacuum (fermi or true)
    std::string vacuum;

    /// bra (vacuum, singles, or doubles)
    std::string bra;

    /// ket (vacuum, singles, or doubles)
    std::string ket;

    /// print level
    int print_level;

  public:

    /// constructor
    ahat_helper(std::string vacuum_type);

    /// destructor
    ~ahat_helper();

    /// when bringing to normal order, does the bra involve any operators?
    void set_bra(std::string bra_type);

    /// when bringing to normal order, does the ket involve any operators?
    void set_ket(std::string ket_type);

    /// set print level (default zero)
    void set_print_level(int level);

    /// set a string of creation / annihilation operators
    void set_string(std::vector<std::string> in);

    /// set labels for a one- or two-body tensor
    void set_tensor(std::vector<std::string> in, std::string tensor_type);

    /// set labels for t1 or t2 amplitudes
    void set_t_amplitudes(std::vector<std::string> in);

    /// set labels for u1 or u2 amplitudes
    void set_u_amplitudes(std::vector<std::string> in);

    /// set labels for l1 or l2 amplitudes
    void set_left_amplitudes(std::vector<std::string> in);

    /// set labels for r1 or r2 amplitudes
    void set_right_amplitudes(std::vector<std::string> in);

    /// set a numerical factor
    void set_factor(double in);

    /// add new completed string / tensor / amplitudes / factor
    void add_new_string();

    /// add new completed string / tensor / amplitudes / factor (assuming normal order is definied relative to the true vacuum
    void add_new_string_true_vacuum();

    /// add new completed string / tensor / amplitudes / factor (assuming normal order is definied relative to the fermi vacuum
    void add_new_string_fermi_vacuum();

    /// add new complete string as a product of operators (i.e., {'h(pq)','t1(ai)'} )
    void add_operator_product(double factor, std::vector<std::string> in);

    /// add commutator of two operators
    void add_commutator(double factor, std::vector<std::string> in);

    /// add double commutator involving three operators
    void add_double_commutator(double factor, std::vector<std::string> in);

    /// add triple commutator involving four operators
    void add_triple_commutator(double factor, std::vector<std::string> in);

    /// add quadruple commutator involving five operators
    void add_quadruple_commutator(double factor, std::vector<std::string> in);

    /// cancel terms, if possible
    void simplify();

    /// clear strings
    void clear();

    /// print strings
    void print();

    /// print fully-contracted strings
    void print_fully_contracted();

    /// print one-body strings
    void print_one_body();

    /// print two-body strings
    void print_two_body();

};

}

#endif
