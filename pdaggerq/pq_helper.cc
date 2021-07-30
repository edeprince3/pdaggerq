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


#ifndef _python_api2_h_
#define _python_api2_h_

#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include <cctype>
#include<algorithm>

#include "data.h"
#include "pq.h"
#include "pq_helper.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;

namespace pdaggerq {

void export_pq_helper(py::module& m) {
    py::class_<pdaggerq::pq_helper, std::shared_ptr<pdaggerq::pq_helper> >(m, "pq_helper")
        .def(py::init< std::string >())
        .def("set_print_level", &pq_helper::set_print_level)
        .def("set_bra", &pq_helper::set_bra)
        .def("set_ket", &pq_helper::set_ket)
        .def("set_string", &pq_helper::set_string)
        .def("set_tensor", &pq_helper::set_tensor)
        .def("set_t_amplitudes", &pq_helper::set_t_amplitudes)
        .def("set_u_amplitudes", &pq_helper::set_u_amplitudes)
        .def("set_m_amplitudes", &pq_helper::set_m_amplitudes)
        .def("set_s_amplitudes", &pq_helper::set_s_amplitudes)
        .def("set_left_amplitudes", &pq_helper::set_left_amplitudes)
        .def("set_right_amplitudes", &pq_helper::set_right_amplitudes)
        .def("set_left_operators", &pq_helper::set_left_operators)
        .def("set_right_operators", &pq_helper::set_right_operators)
        .def("set_factor", &pq_helper::set_factor)
        .def("add_new_string", &pq_helper::add_new_string)
        .def("add_operator_product", &pq_helper::add_operator_product)
        .def("add_st_operator", &pq_helper::add_st_operator)
        .def("add_commutator", &pq_helper::add_commutator)
        .def("add_double_commutator", &pq_helper::add_double_commutator)
        .def("add_triple_commutator", &pq_helper::add_triple_commutator)
        .def("add_quadruple_commutator", &pq_helper::add_quadruple_commutator)
        .def("simplify", &pq_helper::simplify)
        .def("clear", &pq_helper::clear)
        .def("print", &pq_helper::print)
        .def("strings", &pq_helper::strings)
        .def("fully_contracted_strings", &pq_helper::fully_contracted_strings)
        .def("print_fully_contracted", &pq_helper::print_fully_contracted)
        .def("print_one_body", &pq_helper::print_one_body)
        .def("print_two_body", &pq_helper::print_two_body);
}

PYBIND11_MODULE(_pdaggerq, m) {
    m.doc() = "Python API of pdaggerq: A code for bringing strings of creation / annihilation operators to normal order.";
    export_pq_helper(m);
}

void removeStar(std::string &x)
{ 
  auto it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == '*');});
  x.erase(it, std::end(x));
}

void removeParentheses(std::string &x)
{ 
  auto it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == '(');});
  x.erase(it, std::end(x));

  it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == ')');});
  x.erase(it, std::end(x));

}


pq_helper::pq_helper(std::string vacuum_type)
{

    if ( vacuum_type == "" ) {
        vacuum = "TRUE";
    }else if ( vacuum_type == "TRUE" || vacuum_type == "true" ) {
        vacuum = "TRUE";
    }else if ( vacuum_type == "FERMI" || vacuum_type == "fermi" ) {
        vacuum = "FERMI";
    }else {
        printf("\n");
        printf("    error: invalid vacuum type (%s)\n",vacuum_type.c_str());
        printf("\n");
        exit(1);
    }

    data = (std::shared_ptr<StringData>)(new StringData());

    bra = "VACUUM";
    ket = "VACUUM";

    print_level = 0;

}

pq_helper::~pq_helper()
{
}

void pq_helper::set_print_level(int level) {
    print_level = level;
}

void pq_helper::set_left_operators(std::vector<std::string> in) {

    left_operators.clear();
    for (int i = 0; i < (int)in.size(); i++) {
        left_operators.push_back(in[i]);
    }

}

void pq_helper::set_right_operators(std::vector<std::string> in) {

    right_operators.clear();
    for (int i = 0; i < (int)in.size(); i++) {
        right_operators.push_back(in[i]);
    }

}

void pq_helper::set_bra(std::string bra_type){

    if ( bra_type == "" ) {
        bra = "VACUUM";
    }else if ( bra_type == "SINGLES" || bra_type == "singles" ) {
        bra = "SINGLES";
    }else if ( bra_type == "SINGLES_1" || bra_type == "singles_1" ) {
        bra = "SINGLES_1";
    }else if ( bra_type == "DOUBLES" || bra_type == "doubles" ) {
        bra = "DOUBLES";
    }else if ( bra_type == "DOUBLES_1" || bra_type == "doubles_1" ) {
        bra = "DOUBLES_1";
    }else if ( bra_type == "TRIPLES" || bra_type == "triples" ) {
        bra = "TRIPLES";
    }else if ( bra_type == "VACUUM" || bra_type == "vacuum" ) {
        bra = "VACUUM";
    }else if ( bra_type == "VACUUM_1" || bra_type == "vacuum_1" ) {
        bra = "VACUUM_1";
    }else {
        printf("\n");
        printf("    error: invalid bra type (%s)\n",bra_type.c_str());
        printf("\n");
        exit(1);
    }
}

void pq_helper::set_ket(std::string ket_type){

    if ( ket_type == "" ) {
        ket = "VACUUM";
    }else if ( ket_type == "SINGLES" || ket_type == "singles" ) {
        ket = "SINGLES";
    }else if ( ket_type == "SINGLES_1" || ket_type == "singles_1" ) {
        ket = "SINGLES_1";
    }else if ( ket_type == "DOUBLES" || ket_type == "doubles" ) {
        ket = "DOUBLES";
    }else if ( ket_type == "DOUBLES_1" || ket_type == "doubles_1" ) {
        ket = "DOUBLES_1";
    }else if ( ket_type == "VACUUM" || ket_type == "vacuum" ) {
        ket = "VACUUM";
    }else if ( ket_type == "VACUUM_1" || ket_type == "vacuum_1" ) {
        ket = "VACUUM_1";
    }else {
        printf("\n");
        printf("    error: invalid ket type (%s)\n",ket_type.c_str());
        printf("\n");
        exit(1);
    }
}

void pq_helper::add_commutator(double factor,
                                 std::vector<std::string> op0,
                                 std::vector<std::string> op1) {

    // op0 op1
    std::vector<std::string> tmp;
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

    // op1 op0
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

}

void pq_helper::add_double_commutator(double factor,
                                        std::vector<std::string> op0, 
                                        std::vector<std::string> op1, 
                                        std::vector<std::string> op2) {

    std::vector<std::string> tmp;

    //   op0 op1 op2
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

    // - op1 op0 op2
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

    // - op2 op0 op1
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

    //   op2 op1 op0
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

}

void pq_helper::add_triple_commutator(double factor,
                                        std::vector<std::string> op0,
                                        std::vector<std::string> op1,
                                        std::vector<std::string> op2,
                                        std::vector<std::string> op3) {

    std::vector<std::string> tmp;

    //    op0 op1 op2 op3
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

    //  - op1 op0 op2 op3
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

    //  - op2 op0 op1 op3
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

    //    op2 op1 op0 op3
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

    //  - op3 op0 op1 op2
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

    //    op3 op1 op0 op2
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

    //    op3 op2 op0 op1
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

    //  - op3 op2 op1 op0
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

}

void pq_helper::add_quadruple_commutator(double factor,
                                           std::vector<std::string> op0,
                                           std::vector<std::string> op1,
                                           std::vector<std::string> op2,
                                           std::vector<std::string> op3,
                                           std::vector<std::string> op4) {

    std::vector<std::string> tmp;

    //  op0 op1 op2 op3 op4
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

    // -op1 op0 op2 op3 op4
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

    // -op2 op0 op1 op3 op4
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

    //  op2 op1 op0 op3 op4
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

    // -op3 op0 op1 op2 op4
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

    //  op3 op1 op0 op2 op4
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

    //  op3 op2 op0 op1 op4
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

    // -op3 op2 op1 op0 op4
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

    // -op4 op0 op1 op2 op3
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

    //  op4 op1 op0 op2 op3
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

    //  op4 op2 op0 op1 op3
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

    // -op4 op2 op1 op0 op3
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

    //  op4 op3 op0 op1 op2
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

    // -op4 op3 op1 op0 op2
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

    // -op4 op3 op2 op0 op1
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    add_operator_product(-factor, tmp );
    tmp.clear();

    //  op4 op3 op2 op1 op0
    for (int i = 0; i < (int)op4.size(); i++) tmp.push_back(op4[i]);
    for (int i = 0; i < (int)op3.size(); i++) tmp.push_back(op3[i]);
    for (int i = 0; i < (int)op2.size(); i++) tmp.push_back(op2[i]);
    for (int i = 0; i < (int)op1.size(); i++) tmp.push_back(op1[i]);
    for (int i = 0; i < (int)op0.size(); i++) tmp.push_back(op0[i]);
    add_operator_product( factor, tmp );
    tmp.clear();

}

// add a string of operators

void pq_helper::add_operator_product(double factor, std::vector<std::string>  in){

    // first check if there is a fluctuation potential operator 
    // that needs to be split into multiple terms

    std::vector<std::string> tmp;

    // left operators
    for (int i = 0; i < (int)left_operators.size(); i++) {
        if ( left_operators[i] == "v" ) {
            tmp.push_back("j1");
            tmp.push_back("j2");
        }else {
            tmp.push_back(left_operators[i]);
        }
    }
    left_operators.clear();
    for (int i = 0; i < (int)tmp.size(); i++) {
        left_operators.push_back(tmp[i]);
    }
    tmp.clear();
    
    // right operators
    for (int i = 0; i < (int)right_operators.size(); i++) {
        if ( right_operators[i] == "v" ) {
            tmp.push_back("j1");
            tmp.push_back("j2");
        }else {
            tmp.push_back(right_operators[i]);
        }
    }
    right_operators.clear();
    for (int i = 0; i < (int)tmp.size(); i++) {
        right_operators.push_back(tmp[i]);
    }
    tmp.clear();
    

    int count = 0;
    bool found_v = false;
    for (int i = 0; i < (int)in.size(); i++) {
        if ( in[i] == "v" ) {
            found_v = true;
            break;
        }else {
            tmp.push_back(in[i]);
            count++;
        }
    }
    if ( found_v ) {

        // term 1
        tmp.push_back("j1");
        for (int i = count+1; i < (int)in.size(); i++) {
            tmp.push_back(in[i]);
        }
        in.clear();
        for (int i = 0; i < (int)tmp.size(); i++) {
            in.push_back(tmp[i]);
        }
        add_operator_product(factor,in);

        // term 2
        in.clear();
        for (int i = 0; i < count; i++) {
            in.push_back(tmp[i]);
        }
        in.push_back("j2");
        for (int i = count + 1; i < (int)tmp.size(); i++) {
            in.push_back(tmp[i]);
        }
        add_operator_product(factor,in);
        
        return;

    }


    // apply any extra operators on left or right:
    std::vector<std::string> save;
    for (int i = 0; i < (int)in.size(); i++) {
        save.push_back(in[i]);
    }

    if ( (int)left_operators.size() == 0 ) {
        left_operators.push_back("1");
    }
    if ( (int)right_operators.size() == 0 ) {
        right_operators.push_back("1");
    }

    double original_factor = factor;

    for (int left = 0; left < (int)left_operators.size(); left++) {

        for (int right = 0; right < (int)right_operators.size(); right++) {

            factor = original_factor;

            std::vector<std::string> tmp_string;

            if ( bra == "SINGLES" ) {

                // for singles equations: <me| = <0|m*e
                tmp_string.push_back("m*");
                tmp_string.push_back("e");

            }else if ( bra == "DOUBLES" ) {

                // for doubles equations: <mnef| = <0|m*n*fe
                tmp_string.push_back("m*");
                tmp_string.push_back("n*");
                tmp_string.push_back("f");
                tmp_string.push_back("e");

            }else if ( bra == "TRIPLES" ) {

                // for triples equations: <mnoefg| = <0|m*n*o*gfe
                tmp_string.push_back("m*");
                tmp_string.push_back("n*");
                tmp_string.push_back("o*");
                tmp_string.push_back("g");
                tmp_string.push_back("f");
                tmp_string.push_back("e");

            }else if ( bra == "SINGLES_1" ) {

                // for singles equations: <me,1| = <0|m*e B
                tmp_string.push_back("m*");
                tmp_string.push_back("e");

                data->is_boson_dagger.push_back(false);

            }else if ( bra == "DOUBLES_1" ) {

                // for doubles equations: <mnef,1| = <0|m*n*fe B
                tmp_string.push_back("m*");
                tmp_string.push_back("n*");
                tmp_string.push_back("f");
                tmp_string.push_back("e");

                data->is_boson_dagger.push_back(false);
            }else if ( bra == "VACUUM_1" ) {

                data->is_boson_dagger.push_back(false);

            }

            bool has_l0       = false;
            bool has_r0       = false;
            bool has_u0       = false;
            bool has_m0       = false;
            bool has_s0       = false;
            bool has_w0       = false;
            //bool has_b        = false;
            //bool has_b_dagger = false;

            // stupid design choice ... o1-o4 and v1-v4 are already used
            int occ_label_count = 5;
            int vir_label_count = 5;
            int gen_label_count = 0;


            // apply any extra operators on left or right:
            std::vector<std::string> tmp;
            tmp.push_back(left_operators[left]);
            for (int i = 0; i < (int)save.size(); i++) {
                tmp.push_back(save[i]);
            }
            tmp.push_back(right_operators[right]);
            in.clear();
            for (int i = 0; i < (int)tmp.size(); i++) {
                in.push_back(tmp[i]);
            }
            tmp.clear();


            for (int i = 0; i < (int)in.size(); i++) {

                // blank string
                if ( in[i].size() == 0 ) continue;

                // lowercase indices
                std::transform(in[i].begin(), in[i].end(), in[i].begin(), [](unsigned char c){ return std::tolower(c); });

                // remove parentheses
                removeParentheses(in[i]);

                if ( in[i].substr(0,1) == "h" ) { // one-electron operator

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // tensor
                    set_tensor({idx1,idx2},"CORE");

                }else if ( in[i].substr(0,1) == "f" ) { // fock operator

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // tensor
                    set_tensor({idx1,idx2},"FOCK");

                }else if ( in[i].substr(0,2) == "d+" ) { // one-electron operator (dipole + boson creator)

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // tensor
                    set_tensor({idx1,idx2},"D+");

                    // boson operator
                    data->is_boson_dagger.push_back(true);

                }else if ( in[i].substr(0,2) == "d-" ) { // one-electron operator (dipole + boson annihilator)

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // tensor
                    set_tensor({idx1,idx2},"D-");

                    // boson operator
                    data->is_boson_dagger.push_back(false);

                }else if ( in[i].substr(0,1) == "g" ) { // general two-electron operator

                    //factor *= 0.25;

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);
                    std::string idx3 = "p" + std::to_string(gen_label_count++);
                    std::string idx4 = "p" + std::to_string(gen_label_count++);

                    tmp_string.push_back(idx1+"*");
                    tmp_string.push_back(idx2+"*");
                    tmp_string.push_back(idx3);
                    tmp_string.push_back(idx4);

                    set_tensor({idx1,idx2,idx4,idx3},"TWO_BODY");

                }else if ( in[i].substr(0,1) == "j" ) { // fluctuation potential

                    if ( in[i].substr(1,1) == "1" ){

                        factor *= -1.0;

                        std::string idx1 = "p" + std::to_string(gen_label_count++);
                        std::string idx2 = "p" + std::to_string(gen_label_count++);

                        // index 1
                        tmp_string.push_back(idx1+"*");

                        // index 2
                        tmp_string.push_back(idx2);

                        // tensor
                        set_tensor({idx1,idx2},"OCC_REPULSION");

                    }else if ( in[i].substr(1,1) == "2" ){

                        factor *= 0.25;

                        std::string idx1 = "p" + std::to_string(gen_label_count++);
                        std::string idx2 = "p" + std::to_string(gen_label_count++);
                        std::string idx3 = "p" + std::to_string(gen_label_count++);
                        std::string idx4 = "p" + std::to_string(gen_label_count++);

                        tmp_string.push_back(idx1+"*");
                        tmp_string.push_back(idx2+"*");
                        tmp_string.push_back(idx3);
                        tmp_string.push_back(idx4);

                        set_tensor({idx1,idx2,idx4,idx3},"ERI");

                    }

                }else if ( in[i].substr(0,1) == "t" ){

                    int n = std::stoi(in[i].substr(1));

                    std::vector<std::string> op_left;
                    std::vector<std::string> op_right;
                    std::vector<std::string> label_left;
                    std::vector<std::string> label_right;
                    for (int id = 0; id < n; id++) {

                        std::string idx1 = "v" + std::to_string(vir_label_count++);
                        std::string idx2 = "o" + std::to_string(occ_label_count++);

                        op_left.push_back(idx1+"*");
                        op_right.push_back(idx2);

                        label_left.push_back(idx1);
                        label_right.push_back(idx2);
                    }
                    // a*b*...
                    for (int id = 0; id < n; id++) {
                        tmp_string.push_back(op_left[id]);
                    }
                    // i*j*...
                    for (int id = 0; id < n; id++) {
                        tmp_string.push_back(op_right[id]);
                    }
                    std::vector<std::string> labels;
                    // tn(ab...
                    for (int id = 0; id < n; id++) {
                        labels.push_back(label_left[id]);
                    }
                    // tn(ab......ji)
                    for (int id = n-1; id >= 0; id--) {
                        labels.push_back(label_right[id]);
                    }
                    set_t_amplitudes(labels);

                    // factor = 1/(n!)^2
                    double my_factor = 1.0;
                    for (int id = 0; id < n; id++) {
                        my_factor *= (id+1);
                    }
                    factor *= 1.0 / my_factor / my_factor;

                }else if ( in[i].substr(0,1) == "w" ){ // w0 B*B

                    if ( in[i].substr(1,1) == "0" ){

                        has_w0 = true;

                        data->is_boson_dagger.push_back(true);
                        data->is_boson_dagger.push_back(false);

                    }else {
                        printf("\n");
                        printf("    error: only w0 is supported\n");
                        printf("\n");
                        exit(1);
                    }

                }else if ( in[i].substr(0,2) == "b+" ){ // B*

                        //has_b_dagger = true;

                        data->is_boson_dagger.push_back(true);

                }else if ( in[i].substr(0,2) == "b-" ){ // B

                        //has_b = true;

                        data->is_boson_dagger.push_back(false);

                }else if ( in[i].substr(0,1) == "u" ){ // t-amplitudes + boson creator

                    int n = std::stoi(in[i].substr(1));

                    if ( n == 0 ){

                        has_u0 = true;

                        data->is_boson_dagger.push_back(true);

                    }else {

                        std::vector<std::string> op_left;
                        std::vector<std::string> op_right;
                        std::vector<std::string> label_left;
                        std::vector<std::string> label_right;
                        for (int id = 0; id < n; id++) {
                            
                            std::string idx1 = "v" + std::to_string(vir_label_count++);
                            std::string idx2 = "o" + std::to_string(occ_label_count++);
                            
                            op_left.push_back(idx1+"*");
                            op_right.push_back(idx2);
                            
                            label_left.push_back(idx1);
                            label_right.push_back(idx2);
                        }
                        // a*b*...
                        for (int id = 0; id < n; id++) {
                            tmp_string.push_back(op_left[id]);
                        }
                        // i*j*...
                        for (int id = 0; id < n; id++) {
                            tmp_string.push_back(op_right[id]);
                        }
                        std::vector<std::string> labels;
                        // tn(ab... 
                        for (int id = 0; id < n; id++) {
                            labels.push_back(label_left[id]);
                        }
                        // tn(ab......ji)
                        for (int id = n-1; id >= 0; id--) {
                            labels.push_back(label_right[id]);
                        }
                        set_u_amplitudes(labels);
                        
                        // factor = 1/(n!)^2
                        double my_factor = 1.0;
                        for (int id = 0; id < n; id++) {
                            my_factor *= (id+1);
                        }
                        factor *= 1.0 / my_factor / my_factor;

                        data->is_boson_dagger.push_back(true);
                    }

                }else if ( in[i].substr(0,1) == "r" ){


                    int n = std::stoi(in[i].substr(1));

                    if ( n == 0 ){

                        has_r0 = true;

                    }else {

                        std::vector<std::string> op_left;
                        std::vector<std::string> op_right;
                        std::vector<std::string> label_left;
                        std::vector<std::string> label_right;
                        for (int id = 0; id < n; id++) {

                            std::string idx1 = "v" + std::to_string(vir_label_count++);
                            std::string idx2 = "o" + std::to_string(occ_label_count++);

                            op_left.push_back(idx1+"*");
                            op_right.push_back(idx2);

                            label_left.push_back(idx1);
                            label_right.push_back(idx2);
                        }
                        // a*b*...
                        for (int id = 0; id < n; id++) {
                            tmp_string.push_back(op_left[id]);
                        }
                        // i*j*...
                        for (int id = 0; id < n; id++) {
                            tmp_string.push_back(op_right[id]);
                        }
                        std::vector<std::string> labels;
                        // tn(ab...
                        for (int id = 0; id < n; id++) {
                            labels.push_back(label_left[id]);
                        }
                        // tn(ab......ji)
                        for (int id = n-1; id >= 0; id--) {
                            labels.push_back(label_right[id]);
                        }
                        set_right_amplitudes(labels);

                        // factor = 1/(n!)^2
                        double my_factor = 1.0;
                        for (int id = 0; id < n; id++) {
                            my_factor *= (id+1);
                        }
                        factor *= 1.0 / my_factor / my_factor;

                    }

                }else if ( in[i].substr(0,1) == "s" ){ // r amplitudes + boson creator

                    int n = std::stoi(in[i].substr(1));

                    if ( n == 0 ){

                        has_s0 = true;

                        data->is_boson_dagger.push_back(true);

                    }else {
                        
                        std::vector<std::string> op_left;
                        std::vector<std::string> op_right;
                        std::vector<std::string> label_left;
                        std::vector<std::string> label_right;
                        for (int id = 0; id < n; id++) {
                            
                            std::string idx1 = "v" + std::to_string(vir_label_count++);
                            std::string idx2 = "o" + std::to_string(occ_label_count++);
                            
                            op_left.push_back(idx1+"*");
                            op_right.push_back(idx2);
                            
                            label_left.push_back(idx1);
                            label_right.push_back(idx2);
                        }
                        // a*b*...
                        for (int id = 0; id < n; id++) {
                            tmp_string.push_back(op_left[id]);
                        }
                        // i*j*...
                        for (int id = 0; id < n; id++) {
                            tmp_string.push_back(op_right[id]);
                        }
                        std::vector<std::string> labels;
                        // tn(ab... 
                        for (int id = 0; id < n; id++) {
                            labels.push_back(label_left[id]);
                        }
                        // tn(ab......ji)
                        for (int id = n-1; id >= 0; id--) {
                            labels.push_back(label_right[id]);
                        }
                        set_s_amplitudes(labels);
                        
                        // factor = 1/(n!)^2
                        double my_factor = 1.0;
                        for (int id = 0; id < n; id++) {
                            my_factor *= (id+1);
                        }
                        factor *= 1.0 / my_factor / my_factor;
                    
                        data->is_boson_dagger.push_back(true);

                    }

                }else if ( in[i].substr(0,1) == "l" ){

                    int n = std::stoi(in[i].substr(1));

                    if ( n == 0 ){

                        has_l0 = true;

                    }else {
                        
                        std::vector<std::string> op_left;
                        std::vector<std::string> op_right;
                        std::vector<std::string> label_left;
                        std::vector<std::string> label_right;
                        for (int id = 0; id < n; id++) {
                            
                            std::string idx1 = "o" + std::to_string(occ_label_count++);
                            std::string idx2 = "v" + std::to_string(vir_label_count++);
                            
                            op_left.push_back(idx1+"*");
                            op_right.push_back(idx2);
                            
                            label_left.push_back(idx1);
                            label_right.push_back(idx2);
                        }
                        // i*j*...
                        for (int id = 0; id < n; id++) {
                            tmp_string.push_back(op_left[id]);
                        }
                        // a*b*...
                        for (int id = 0; id < n; id++) {
                            tmp_string.push_back(op_right[id]);
                        }
                        std::vector<std::string> labels;
                        // tn(ab... 
                        for (int id = 0; id < n; id++) {
                            labels.push_back(label_left[id]);
                        }
                        // tn(ab......ji)
                        for (int id = n-1; id >= 0; id--) {
                            labels.push_back(label_right[id]);
                        }
                        set_left_amplitudes(labels);
                        
                        // factor = 1/(n!)^2
                        double my_factor = 1.0;
                        for (int id = 0; id < n; id++) {
                            my_factor *= (id+1);
                        }
                        factor *= 1.0 / my_factor / my_factor;
                    
                    }

                }else if ( in[i].substr(0,1) == "m" ){ // l amplitudes plus boson annihilator

                    int n = std::stoi(in[i].substr(1));

                    if ( n == 0 ){

                        has_m0 = true;

                        data->is_boson_dagger.push_back(false);

                    }else {

                        std::vector<std::string> op_left;
                        std::vector<std::string> op_right;
                        std::vector<std::string> label_left;
                        std::vector<std::string> label_right;
                        for (int id = 0; id < n; id++) {

                            std::string idx1 = "o" + std::to_string(occ_label_count++);
                            std::string idx2 = "v" + std::to_string(vir_label_count++);

                            op_left.push_back(idx1+"*");
                            op_right.push_back(idx2);

                            label_left.push_back(idx1);
                            label_right.push_back(idx2);
                        }
                        // i*j*...
                        for (int id = 0; id < n; id++) {
                            tmp_string.push_back(op_left[id]);
                        }
                        // a*b*...
                        for (int id = 0; id < n; id++) {
                            tmp_string.push_back(op_right[id]);
                        }
                        std::vector<std::string> labels;
                        // tn(ab... 
                        for (int id = 0; id < n; id++) {
                            labels.push_back(label_left[id]);
                        }
                        // tn(ab......ji)
                        for (int id = n-1; id >= 0; id--) {
                            labels.push_back(label_right[id]);
                        }
                        set_m_amplitudes(labels);

                        // factor = 1/(n!)^2
                        double my_factor = 1.0;
                        for (int id = 0; id < n; id++) {
                            my_factor *= (id+1);
                        }
                        factor *= 1.0 / my_factor / my_factor;

                        data->is_boson_dagger.push_back(false);

                    }

                }else if ( in[i].substr(0,2) == "2p" ){ // particle-particle 

                    // find comma
                    size_t pos = in[i].find(",");
                    if ( pos == std::string::npos ) {
                        printf("\n");
                        printf("    error in particle-particle operator definition\n");
                        printf("\n");
                        exit(1);
                    }
                    size_t len = pos - 2; 

                    // index 1
                    tmp_string.push_back(in[i].substr(2,len)+"*");

                    // index 2
                    tmp_string.push_back(in[i].substr(pos+1)+"*");

                }else if ( in[i].substr(0,2) == "2h" ){ // hole-hole 

                    // find comma
                    size_t pos = in[i].find(",");
                    if ( pos == std::string::npos ) {
                        printf("\n");
                        printf("    error in particle-particle operator definition\n");
                        printf("\n");
                        exit(1);
                    }
                    size_t len = pos - 2; 

                    // index 1
                    tmp_string.push_back(in[i].substr(2,len));

                    // index 2
                    tmp_string.push_back(in[i].substr(pos+1));

                }else if ( in[i].substr(0,1) == "e" ){


                    if ( in[i].substr(1,1) == "1" ){

                        // find comma
                        size_t pos = in[i].find(",");
                        if ( pos == std::string::npos ) {
                            printf("\n");
                            printf("    error in e1 operator definition\n");
                            printf("\n");
                            exit(1);
                        }
                        size_t len = pos - 2; 

                        // index 1
                        tmp_string.push_back(in[i].substr(2,len)+"*");

                        // index 2
                        tmp_string.push_back(in[i].substr(pos+1));

                    }else if ( in[i].substr(1,1) == "2" ){

                        //printf("\n");
                        //printf("    error: e2 operator not yet implemented.\n");
                        //printf("\n");
                        //exit(1);

                        // count indices
                        size_t pos = 0;
                        int ncomma = 0;
                        std::vector<size_t> commas;
                        pos = in[i].find(",", pos + 1);
                        commas.push_back(pos);
                        while( pos != std::string::npos){
                            pos = in[i].find(",", pos + 1);
                            commas.push_back(pos);
                            ncomma++;
                        }

                        if ( ncomma != 3 ) {
                            printf("\n");
                            printf("    error in e2 definition\n");
                            printf("\n");
                            exit(1);
                        }

                        tmp_string.push_back(in[i].substr(2,commas[0]-2)+"*");
                        tmp_string.push_back(in[i].substr(commas[0]+1,commas[1]-commas[0]-1)+"*");
                        tmp_string.push_back(in[i].substr(commas[1]+1,commas[2]-commas[1]-1));
                        tmp_string.push_back(in[i].substr(commas[2]+1));

                    }else if ( in[i].substr(1,1) == "3" ){

                        // count indices
                        size_t pos = 0;
                        int ncomma = 0;
                        std::vector<size_t> commas;
                        pos = in[i].find(",", pos + 1);
                        commas.push_back(pos);
                        while( pos != std::string::npos){
                            pos = in[i].find(",", pos + 1);
                            commas.push_back(pos);
                            ncomma++;
                        }

                        if ( ncomma != 5 ) {
                            printf("\n");
                            printf("    error in e3 definition\n");
                            printf("\n");
                            exit(1);
                        }

                        tmp_string.push_back(in[i].substr(2,commas[0]-2)+"*");
                        tmp_string.push_back(in[i].substr(commas[0]+1,commas[1]-commas[0]-1)+"*");
                        tmp_string.push_back(in[i].substr(commas[1]+1,commas[2]-commas[1]-1)+"*");
                        tmp_string.push_back(in[i].substr(commas[2]+1,commas[3]-commas[2]-1));
                        tmp_string.push_back(in[i].substr(commas[3]+1,commas[4]-commas[3]-1));
                        tmp_string.push_back(in[i].substr(commas[4]+1));

                    }else if ( in[i].substr(1,1) == "4" ){

                        // count indices
                        size_t pos = 0;
                        int ncomma = 0;
                        std::vector<size_t> commas;
                        pos = in[i].find(",", pos + 1);
                        commas.push_back(pos);
                        while( pos != std::string::npos){
                            pos = in[i].find(",", pos + 1);
                            commas.push_back(pos);
                            ncomma++;
                        }

                        if ( ncomma != 7 ) {
                            printf("\n");
                            printf("    error in e4 definition\n");
                            printf("\n");
                            exit(1);
                        }

                        tmp_string.push_back(in[i].substr(2,commas[0]-2)+"*");
                        tmp_string.push_back(in[i].substr(commas[0]+1,commas[1]-commas[0]-1)+"*");
                        tmp_string.push_back(in[i].substr(commas[1]+1,commas[2]-commas[1]-1)+"*");
                        tmp_string.push_back(in[i].substr(commas[2]+1,commas[3]-commas[2]-1)+"*");
                        tmp_string.push_back(in[i].substr(commas[3]+1,commas[4]-commas[3]-1));
                        tmp_string.push_back(in[i].substr(commas[4]+1,commas[5]-commas[4]-1));
                        tmp_string.push_back(in[i].substr(commas[5]+1,commas[6]-commas[5]-1));
                        tmp_string.push_back(in[i].substr(commas[6]+1));

                    }else {
                        printf("\n");
                        printf("    error: only e1, e2, e3, and e4 operators are supported\n");
                        printf("\n");
                        exit(1);
                    }

                }else if ( in[i].substr(0,1) == "1" ) { // unit operator ... do nothing

                }else {
                        printf("\n");
                        printf("    error: undefined string\n");
                        printf("\n");
                        exit(1);
                }
                
            }

            set_factor(factor);

            if ( ket == "SINGLES" ) {

                // for singles equations: |em> = e*m|0>
                tmp_string.push_back("e*");
                tmp_string.push_back("m");

            }else if ( ket == "SINGLES_1" ) {

                // for singles equations: |em,1> = e*m B*|0>
                tmp_string.push_back("e*");
                tmp_string.push_back("m");

                data->is_boson_dagger.push_back(true);

            }else if ( ket == "DOUBLES" ) {

                // for doubles equations: |efmn> = e*f*mn|0>
                tmp_string.push_back("e*");
                tmp_string.push_back("f*");
                tmp_string.push_back("n");
                tmp_string.push_back("m");

            }else if ( ket == "DOUBLES_1" ) {

                // for doubles equations: |efmn,1> = e*f*mn B*|0>
                tmp_string.push_back("e*");
                tmp_string.push_back("f*");
                tmp_string.push_back("n");
                tmp_string.push_back("m");

                data->is_boson_dagger.push_back(true);

            }else if ( ket == "VACUUM_1" ) {

                data->is_boson_dagger.push_back(true);

            }

            set_string(tmp_string);

            data->has_r0       = has_r0;
            data->has_l0       = has_l0;
            data->has_u0       = has_u0;
            data->has_m0       = has_m0;
            data->has_s0       = has_s0;
            data->has_w0       = has_w0;
            //data->has_b        = has_b;
            //data->has_b_dagger = has_b_dagger;

            add_new_string();

        }
    }


}

void pq_helper::set_string(std::vector<std::string> in) {
    for (int i = 0; i < (int)in.size(); i++) {
        data->string.push_back(in[i]);
    }
}

void pq_helper::set_tensor(std::vector<std::string> in, std::string tensor_type) {
    for (int i = 0; i < (int)in.size(); i++) {
        data->tensor.push_back(in[i]);
    }
    data->tensor_type = tensor_type;
}

void pq_helper::set_t_amplitudes(std::vector<std::string> in) {
    std::vector<std::string> tmp;
    for (int i = 0; i < (int)in.size(); i++) {
        tmp.push_back(in[i]);
    }
    data->t_amplitudes.push_back(tmp);
}

void pq_helper::set_u_amplitudes(std::vector<std::string> in) {
    std::vector<std::string> tmp;
    for (int i = 0; i < (int)in.size(); i++) {
        tmp.push_back(in[i]);
    }
    data->u_amplitudes.push_back(tmp);
}

void pq_helper::set_m_amplitudes(std::vector<std::string> in) {
    std::vector<std::string> tmp;
    for (int i = 0; i < (int)in.size(); i++) {
        tmp.push_back(in[i]);
    }
    data->m_amplitudes.push_back(tmp);
}

void pq_helper::set_s_amplitudes(std::vector<std::string> in) {
    std::vector<std::string> tmp;
    for (int i = 0; i < (int)in.size(); i++) {
        tmp.push_back(in[i]);
    }
    data->s_amplitudes.push_back(tmp);
}

void pq_helper::set_left_amplitudes(std::vector<std::string> in) {
    std::vector<std::string> tmp;
    for (int i = 0; i < (int)in.size(); i++) {
        tmp.push_back(in[i]);
    }
    data->left_amplitudes.push_back(tmp);
}

void pq_helper::set_right_amplitudes(std::vector<std::string> in) {
    std::vector<std::string> tmp;
    for (int i = 0; i < (int)in.size(); i++) {
        tmp.push_back(in[i]);
    }
    data->right_amplitudes.push_back(tmp);
}

void pq_helper::set_factor(double in) {
    data->factor = in;
}

void pq_helper::add_new_string_true_vacuum(){

    std::shared_ptr<pq> mystring (new pq(vacuum));

    if ( data->factor > 0.0 ) {
        mystring->sign = 1;
        mystring->data->factor = fabs(data->factor);
    }else {
        mystring->sign = -1;
        mystring->data->factor = fabs(data->factor);
    }

    mystring->data->has_r0       = data->has_r0;
    mystring->data->has_l0       = data->has_l0;
    mystring->data->has_u0       = data->has_u0;
    mystring->data->has_m0       = data->has_m0;
    mystring->data->has_s0       = data->has_s0;
    mystring->data->has_w0       = data->has_w0;
    //mystring->data->has_b        = data->has_b;
    //mystring->data->has_b_dagger = data->has_b_dagger;

    for (int i = 0; i < (int)data->string.size(); i++) {
        std::string me = data->string[i];
        if ( me.find("*") != std::string::npos ) {
            removeStar(me);
            mystring->is_dagger.push_back(true);
        }else {
            mystring->is_dagger.push_back(false);
        }
        mystring->symbol.push_back(me);
    }

    for (int i = 0; i < (int)data->tensor.size(); i++) {
        mystring->data->tensor.push_back(data->tensor[i]);
    }
    mystring->data->tensor_type = data->tensor_type;

    for (int i = 0; i < (int)data->t_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)data->t_amplitudes[i].size(); j++) {
            tmp.push_back(data->t_amplitudes[i][j]);
        }
        mystring->data->t_amplitudes.push_back(tmp);
    }
    for (int i = 0; i < (int)data->u_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)data->u_amplitudes[i].size(); j++) {
            tmp.push_back(data->u_amplitudes[i][j]);
        }
        mystring->data->u_amplitudes.push_back(tmp);
    }

    for (int i = 0; i < (int)data->m_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)data->m_amplitudes[i].size(); j++) {
            tmp.push_back(data->m_amplitudes[i][j]);
        }
        mystring->data->m_amplitudes.push_back(tmp);
    }

    for (int i = 0; i < (int)data->s_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)data->s_amplitudes[i].size(); j++) {
            tmp.push_back(data->s_amplitudes[i][j]);
        }
        mystring->data->s_amplitudes.push_back(tmp);
    }

    for (int i = 0; i < (int)data->left_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)data->left_amplitudes[i].size(); j++) {
            tmp.push_back(data->left_amplitudes[i][j]);
        }
        mystring->data->left_amplitudes.push_back(tmp);
    }
    for (int i = 0; i < (int)data->right_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)data->right_amplitudes[i].size(); j++) {
            tmp.push_back(data->right_amplitudes[i][j]);
        }
        mystring->data->right_amplitudes.push_back(tmp);
    }
    for (int i = 0; i < (int)data->is_boson_dagger.size(); i++) {
        mystring->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
    }

    if ( print_level > 0 ) {
        printf("\n");
        printf("    ");
        printf("// starting string:\n");
        mystring->print();
    }

    // rearrange strings
    //mystring->normal_order(ordered);

    std::vector< std::shared_ptr<pq> > tmp;
    tmp.push_back(mystring);

    bool done_rearranging = false;
    do {  
        std::vector< std::shared_ptr<pq> > list;
        done_rearranging = true;
        for (int i = 0; i < (int)tmp.size(); i++) {
            bool am_i_done = tmp[i]->normal_order(list);
            if ( !am_i_done ) done_rearranging = false;
        }
        tmp.clear();
        for (int i = 0; i < (int)list.size(); i++) {
            tmp.push_back(list[i]);
        }
    }while(!done_rearranging);

    //ordered.clear();
    for (int i = 0; i < (int)tmp.size(); i++) {
        ordered.push_back(tmp[i]);
    }
    tmp.clear();


    // alphabetize
    mystring->alphabetize(ordered);

    // cancel terms
    mystring->cleanup(ordered);

    // reset data object
    data.reset();
    data = (std::shared_ptr<StringData>)(new StringData());

}

void pq_helper::add_new_string() {

    if ( vacuum == "TRUE" ) {
        add_new_string_true_vacuum();
    }else {
        add_new_string_fermi_vacuum();
    }

}

void pq_helper::add_new_string_fermi_vacuum(){

    std::vector<std::shared_ptr<pq> > mystrings;
    mystrings.push_back( (std::shared_ptr<pq>)(new pq(vacuum)) );

    // if normal order is defined with respect to the fermi vacuum, we must
    // check here if the input string contains any general-index operators
    // (h, g). If it does, then the string must be split to account explicitly
    // for sums over 
    int n_gen_idx = 0;
    for (int i = 0; i < (int)data->string.size(); i++) {
        std::string me = data->string[i];
        std::string me_nostar = me;
        if (me_nostar.find("*") != std::string::npos ){
            removeStar(me_nostar);
        }
         
        if ( !mystrings[0]->is_vir(me_nostar) && !mystrings[0]->is_occ(me_nostar) ) {
            n_gen_idx++;
        }

    }
    //printf("number of general indices: %5i\n",n_gen_idx);
    // need number of strings to be square of number of general indices 
    if ( n_gen_idx > 0 ) {
        mystrings.clear();
        for (int i = 0; i < n_gen_idx * n_gen_idx; i++) {
            mystrings.push_back( (std::shared_ptr<pq>)(new pq(vacuum)) );
        }
    }

    // TODO: this function only works correctly if you go through the
    // add_operator_product function (or some function that calls that one
    // one). should generalize so set_tensor, etc. can be used directly.

    if ( n_gen_idx == 0 ) {
        n_gen_idx = 1;
    }

    //printf("current list size: %zu\n",ordered.size());
    for (int string_num = 0; string_num < n_gen_idx * n_gen_idx; string_num++) {

        // factors:
        if ( data->factor > 0.0 ) {
            mystrings[string_num]->sign = 1;
            mystrings[string_num]->data->factor = fabs(data->factor);
        }else {
            mystrings[string_num]->sign = -1;
            mystrings[string_num]->data->factor = fabs(data->factor);
        }

        mystrings[string_num]->data->has_r0       = data->has_r0;
        mystrings[string_num]->data->has_l0       = data->has_l0;
        mystrings[string_num]->data->has_u0       = data->has_u0;
        mystrings[string_num]->data->has_m0       = data->has_m0;
        mystrings[string_num]->data->has_s0       = data->has_s0;
        mystrings[string_num]->data->has_w0       = data->has_w0;
        //mystrings[string_num]->data->has_b        = data->has_b;
        //mystrings[string_num]->data->has_b_dagger = data->has_b_dagger;

        // tensor type
        mystrings[string_num]->data->tensor_type = data->tensor_type;

        int my_gen_idx = 0;
        for (int i = 0; i < (int)data->string.size(); i++) {
            std::string me = data->string[i];


            std::string me_nostar = me;
            if (me_nostar.find("*") != std::string::npos ){
                removeStar(me_nostar);
            }

            // fermi vacuum 
            if ( mystrings[string_num]->is_vir(me_nostar) ) {
                if (me.find("*") != std::string::npos ){
                    mystrings[string_num]->is_dagger.push_back(true);
                    mystrings[string_num]->is_dagger_fermi.push_back(true);
                }else {
                    mystrings[string_num]->is_dagger.push_back(false);
                    mystrings[string_num]->is_dagger_fermi.push_back(false);
                }
                mystrings[string_num]->symbol.push_back(me_nostar);
            }else if ( mystrings[string_num]->is_occ(me_nostar) ) {
                if (me.find("*") != std::string::npos ){
                    mystrings[string_num]->is_dagger.push_back(true);
                    mystrings[string_num]->is_dagger_fermi.push_back(false);
                }else {
                    mystrings[string_num]->is_dagger.push_back(false);
                    mystrings[string_num]->is_dagger_fermi.push_back(true);
                }
                mystrings[string_num]->symbol.push_back(me_nostar);
            }else {

                //two-index tensor
                // 00, 01, 10, 11
                if ( n_gen_idx == 2 ) {
                    if ( my_gen_idx == 0 ) {
                        if ( string_num == 0 || string_num == 1 ) {
                            // first index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }
                            mystrings[string_num]->data->tensor.push_back("o1");
                            mystrings[string_num]->symbol.push_back("o1");
                        }else {
                            // first index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }else {
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                                mystrings[string_num]->is_dagger.push_back(false);
                            }
                            mystrings[string_num]->data->tensor.push_back("v1");
                            mystrings[string_num]->symbol.push_back("v1");
                        }
                    }else {
                        if ( string_num == 0 || string_num == 2 ) {
                            // second index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }
                            mystrings[string_num]->data->tensor.push_back("o2");
                            mystrings[string_num]->symbol.push_back("o2");
                        }else {
                            // second index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }
                            mystrings[string_num]->data->tensor.push_back("v2");
                            mystrings[string_num]->symbol.push_back("v2");
                        }
                    }
                }

                //four-index tensor

                // managing these labels is so very confusing:
                // p*q*sr (pr|qs) -> o*t*uv (ov|tu), etc.
                // p*q*sr (pr|qs) -> w*x*yz (wz|xy), etc.

                if ( n_gen_idx == 4 ) {
                    if ( my_gen_idx == 0 ) {
                        //    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
                        // 0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111
                        if ( string_num == 0 || 
                             string_num == 1 ||
                             string_num == 2 ||
                             string_num == 3 ||
                             string_num == 4 ||
                             string_num == 5 ||
                             string_num == 6 ||
                             string_num == 7 ) {

                            // first index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }
                            mystrings[string_num]->data->tensor.push_back("o1");
                            mystrings[string_num]->symbol.push_back("o1");
                        }else {
                            // first index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }
                            mystrings[string_num]->data->tensor.push_back("v1");
                            mystrings[string_num]->symbol.push_back("v1");
                        }
                    }else if ( my_gen_idx == 1 ) {
                        //    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
                        // 0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111
                        if ( string_num ==  0 || 
                             string_num ==  1 ||
                             string_num ==  2 ||
                             string_num ==  3 ||
                             string_num ==  8 ||
                             string_num ==  9 ||
                             string_num == 10 ||
                             string_num == 11 ) {
                            // second index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }
                            mystrings[string_num]->data->tensor.push_back("o2");
                            mystrings[string_num]->symbol.push_back("o2");
                        }else {
                            // second index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }
                            mystrings[string_num]->data->tensor.push_back("v2");
                            mystrings[string_num]->symbol.push_back("v2");
                        }
                    }else if ( my_gen_idx == 2 ) {
                        //    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
                        // 0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111
                        if ( string_num ==  0 || 
                             string_num ==  1 ||
                             string_num ==  4 ||
                             string_num ==  5 ||
                             string_num ==  8 ||
                             string_num ==  9 ||
                             string_num == 12 ||
                             string_num == 13 ) {
                            // third index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }
                            mystrings[string_num]->data->tensor.push_back("o3");
                            mystrings[string_num]->symbol.push_back("o3");
                        }else {
                            // third index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }
                            mystrings[string_num]->data->tensor.push_back("v3");
                            mystrings[string_num]->symbol.push_back("v3");
                        }
                    }else {
                        if ( string_num ==  0 || 
                             string_num ==  2 ||
                             string_num ==  4 ||
                             string_num ==  6 ||
                             string_num ==  8 ||
                             string_num == 10 ||
                             string_num == 12 ||
                             string_num == 14 ) {
                            // fourth index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }
                            mystrings[string_num]->data->tensor.push_back("o4");
                            mystrings[string_num]->symbol.push_back("o4");
                        }else {
                            // fourth index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystrings[string_num]->is_dagger.push_back(true);
                                mystrings[string_num]->is_dagger_fermi.push_back(true);
                            }else {
                                mystrings[string_num]->is_dagger.push_back(false);
                                mystrings[string_num]->is_dagger_fermi.push_back(false);
                            }
                            mystrings[string_num]->data->tensor.push_back("v4");
                            mystrings[string_num]->symbol.push_back("v4");
                        }
                    }
                }

                my_gen_idx++;
            }

        }

        for (int i = 0; i < (int)data->t_amplitudes.size(); i++) {
            std::vector<std::string> tmp;
            for (int j = 0; j < (int)data->t_amplitudes[i].size(); j++) {
                tmp.push_back(data->t_amplitudes[i][j]);
            }
            mystrings[string_num]->data->t_amplitudes.push_back(tmp);
        }

        for (int i = 0; i < (int)data->u_amplitudes.size(); i++) {
            std::vector<std::string> tmp;
            for (int j = 0; j < (int)data->u_amplitudes[i].size(); j++) {
                tmp.push_back(data->u_amplitudes[i][j]);
            }
            mystrings[string_num]->data->u_amplitudes.push_back(tmp);
        }

        for (int i = 0; i < (int)data->m_amplitudes.size(); i++) {
            std::vector<std::string> tmp;
            for (int j = 0; j < (int)data->m_amplitudes[i].size(); j++) {
                tmp.push_back(data->m_amplitudes[i][j]);
            }
            mystrings[string_num]->data->m_amplitudes.push_back(tmp);
        }

        for (int i = 0; i < (int)data->s_amplitudes.size(); i++) {
            std::vector<std::string> tmp;
            for (int j = 0; j < (int)data->s_amplitudes[i].size(); j++) {
                tmp.push_back(data->s_amplitudes[i][j]);
            }
            mystrings[string_num]->data->s_amplitudes.push_back(tmp);
        }

        for (int i = 0; i < (int)data->left_amplitudes.size(); i++) {
            std::vector<std::string> tmp;
            for (int j = 0; j < (int)data->left_amplitudes[i].size(); j++) {
                tmp.push_back(data->left_amplitudes[i][j]);
            }
            mystrings[string_num]->data->left_amplitudes.push_back(tmp);
        }

        for (int i = 0; i < (int)data->right_amplitudes.size(); i++) {
            std::vector<std::string> tmp;
            for (int j = 0; j < (int)data->right_amplitudes[i].size(); j++) {
                tmp.push_back(data->right_amplitudes[i][j]);
            }
            mystrings[string_num]->data->right_amplitudes.push_back(tmp);
        }

        // now, string is complete, but labels in four-index tensors need to be reordered p*q*sr(pq|sr) -> (pr|qs)
        if ( (int)mystrings[string_num]->data->tensor.size() == 4 ) {


            // mulliken notation: g(prqs) p*q*sr
            //std::vector<std::string> tmp;
            //tmp.push_back(mystrings[string_num]->data->tensor[0]);
            //tmp.push_back(mystrings[string_num]->data->tensor[3]);
            //tmp.push_back(mystrings[string_num]->data->tensor[1]);
            //tmp.push_back(mystrings[string_num]->data->tensor[2]);

            // dirac notation: g(pqrs) p*q*sr
            std::vector<std::string> tmp;
            tmp.push_back(mystrings[string_num]->data->tensor[0]);
            tmp.push_back(mystrings[string_num]->data->tensor[1]);
            tmp.push_back(mystrings[string_num]->data->tensor[3]);
            tmp.push_back(mystrings[string_num]->data->tensor[2]);

            mystrings[string_num]->data->tensor.clear();
            mystrings[string_num]->data->tensor.push_back(tmp[0]);
            mystrings[string_num]->data->tensor.push_back(tmp[1]);
            mystrings[string_num]->data->tensor.push_back(tmp[2]);
            mystrings[string_num]->data->tensor.push_back(tmp[3]);

        }
        for (int i = 0; i < (int)data->is_boson_dagger.size(); i++) {
            mystrings[string_num]->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
        }

        if ( print_level > 0 ) {
            printf("\n");
            printf("    ");
            printf("// starting string:\n");
            mystrings[string_num]->print();
        }

        // rearrange strings
        //mystrings[string_num]->normal_order(ordered);
        std::vector< std::shared_ptr<pq> > tmp;
        tmp.push_back(mystrings[string_num]);

        bool done_rearranging = false;
        do { 
            std::vector< std::shared_ptr<pq> > list;
            done_rearranging = true;
            for (int i = 0; i < (int)tmp.size(); i++) {
                bool am_i_done = tmp[i]->normal_order(list);
                if ( !am_i_done ) done_rearranging = false;
            }
            tmp.clear();
            for (int i = 0; i < (int)list.size(); i++) {
                if ( !list[i]->skip ) {
                    tmp.push_back(list[i]);
                }
            }
        }while(!done_rearranging);

        //ordered.clear();
        for (int i = 0; i < (int)tmp.size(); i++) {
            ordered.push_back(tmp[i]);
        }
        //printf("current list size: %zu\n",ordered.size());
        tmp.clear();

    }

    //for (int n_ordered = 0; n_ordered < (int)ordered.size(); n_ordered++) {
    //    ordered[n_ordered]->check_occ_vir();
    //}

    // TODO: this only seems to work with normal ordering relative to the true vacuum
    // alphabetize
    //mystring->alphabetize(ordered);

    // TODO: moved cleanup to final simplify function?
    // cancel terms. i think the work is actually done on "ordered" so only need to call once 
    //mystrings[0]->cleanup(ordered);

    // reset data object
    data.reset();
    data = (std::shared_ptr<StringData>)(new StringData());
 
}

void pq_helper::simplify() {

    std::shared_ptr<pq> mystring (new pq(vacuum));

    // eliminate strings based on delta functions and use delta functions to alter tensor / amplitude labels
    for (int i = 0; i < (int)ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        // check for occ/vir pairs in delta functions ... i think this is handled by the normal order procedure
        //ordered[i]->check_occ_vir();

        // apply delta functions
        ordered[i]->gobble_deltas();

        // re-classify fluctuation potential terms
        ordered[i]->reclassify_tensors();

        // replace any funny labels that were added with conventional ones (fermi vacumm only)
        if ( vacuum == "FERMI" ) {
            ordered[i]->use_conventional_labels();
        }
    }

    // try to cancel similar terms
    mystring->cleanup(ordered);

}

void pq_helper::print_two_body() {

    printf("\n");
    printf("    ");
    printf("// two-body strings:\n");
    for (int i = 0; i < (int)ordered.size(); i++) {
        if ( ordered[i]->symbol.size() != 4 ) continue;
        ordered[i]->print();
    }
    printf("\n");

}

void pq_helper::print_fully_contracted() {

    printf("\n");
    printf("    ");
    printf("// fully-contracted strings:\n");
    for (int i = 0; i < (int)ordered.size(); i++) {
        if ( ordered[i]->symbol.size() != 0 ) continue;
        if ( ordered[i]->data->is_boson_dagger.size() != 0 ) continue;
        ordered[i]->print();
    }
    printf("\n");

}

std::vector<std::vector<std::string> > pq_helper::fully_contracted_strings() {

    std::vector<std::vector<std::string> > list;
    for (int i = 0; i < (int)ordered.size(); i++) {
        if ( ordered[i]->symbol.size() != 0 ) continue;
        if ( ordered[i]->data->is_boson_dagger.size() != 0 ) continue;
        std::vector<std::string> my_string = ordered[i]->get_string();
        if ( (int)my_string.size() > 0 ) {
            list.push_back(my_string);
        }
    }

    return list;

}

std::vector<std::vector<std::string> > pq_helper::strings() {

    std::vector<std::vector<std::string> > list;
    for (int i = 0; i < (int)ordered.size(); i++) {
        std::vector<std::string> my_string = ordered[i]->get_string();
        if ( (int)my_string.size() > 0 ) {
            list.push_back(my_string);
        }
    }

    return list;

}

void pq_helper::print_one_body() {

    printf("\n");
    printf("    ");
    printf("// one-body strings:\n");
    for (int i = 0; i < (int)ordered.size(); i++) {
        if ( ordered[i]->symbol.size() != 2 ) continue;
        ordered[i]->print();
    }
    printf("\n");

}

void pq_helper::print() {

    printf("\n");
    printf("    ");
    printf("// normal-ordered strings:\n");
    for (int i = 0; i < (int)ordered.size(); i++) {
        ordered[i]->print();
    }
    printf("\n");

}

void pq_helper::clear() {

    ordered.clear();

}

void pq_helper::add_st_operator(double factor, std::vector<std::string> targets, std::vector<std::string> ops) {

    int dim = (int)ops.size();

    add_operator_product( factor, targets);
    simplify();

    for (int i = 0; i < dim; i++) {
        add_commutator( factor, targets, {ops[i]});
    }
    simplify();

    // for higher than single commutators, if operators commute, then
    // we only need to consider unique pairs/triples/quadruplets of
    // operators. need to add logic to handle cases where the operators
    // do not commute.

    for (int i = 0; i < dim; i++) {
        for (int j = i + 1; j < dim; j++) {
            add_double_commutator(factor, targets, {ops[i]}, {ops[j]});
        }
    }
    simplify();
    for (int i = 0; i < dim; i++) {
        add_double_commutator(0.5 * factor, targets, {ops[i]}, {ops[i]});
    }
    simplify();

    // ijk
    for (int i = 0; i < dim; i++) {
        for (int j = i + 1; j < dim; j++) {
            for (int k = j + 1; k < dim; k++) {
                add_triple_commutator( factor, targets, {ops[i]}, {ops[j]}, {ops[k]});
            }
        }
    }
    simplify();

    // ijj
    for (int i = 0; i < dim; i++) {
        for (int j = i + 1; j < dim; j++) {
            add_triple_commutator( 0.5 * factor, targets, {ops[i]}, {ops[j]}, {ops[j]});
            add_triple_commutator( 0.5 * factor, targets, {ops[i]}, {ops[i]}, {ops[j]});
        }
    }
    simplify();

     // iii
    for (int i = 0; i < dim; i++) {
        add_triple_commutator( 1.0 / 6.0 * factor, targets, {ops[i]}, {ops[i]}, {ops[i]});
    }
    simplify();

    // ijkl
    for (int i = 0; i < dim; i++) {
        for (int j = i + 1; j < dim; j++) {
            for (int k = j + 1; k < dim; k++) {
                for (int l = k + 1; l < dim; l++) {
                    add_quadruple_commutator( factor, targets, {ops[i]}, {ops[j]}, {ops[k]}, {ops[l]});
                }
            }
        }
    }
    simplify();

    // ijkk
    for (int i = 0; i < dim; i++) {
        for (int j = i + 1; j < dim; j++) {
            for (int k = j + 1; k < dim; k++) {
                add_quadruple_commutator( 0.5 * factor, targets, {ops[i]}, {ops[j]}, {ops[k]}, {ops[k]});
                add_quadruple_commutator( 0.5 * factor, targets, {ops[i]}, {ops[j]}, {ops[j]}, {ops[k]});
                add_quadruple_commutator( 0.5 * factor, targets, {ops[i]}, {ops[i]}, {ops[j]}, {ops[k]});
            }
        }
    }
    simplify();

    // iijj
    for (int i = 0; i < dim; i++) {
        for (int j = i + 1; j < dim; j++) {
            add_quadruple_commutator( 0.25 * factor, targets, {ops[i]}, {ops[i]}, {ops[j]}, {ops[j]});
        }
    }
    simplify();

    // iiij
    for (int i = 0; i < dim; i++) {
        for (int j = i + 1; j < dim; j++) {
            add_quadruple_commutator( 1.0 / 6.0 * factor, targets, {ops[i]}, {ops[i]}, {ops[i]}, {ops[j]});
            add_quadruple_commutator( 1.0 / 6.0 * factor, targets, {ops[i]}, {ops[j]}, {ops[j]}, {ops[j]});
        }
    }
    simplify();

    // iiii
    for (int i = 0; i < dim; i++) {
        add_quadruple_commutator( 1.0 / 24.0 * factor, targets, {ops[i]}, {ops[i]}, {ops[i]}, {ops[i]});
    }
    simplify();
}

} // End namespaces

#endif
