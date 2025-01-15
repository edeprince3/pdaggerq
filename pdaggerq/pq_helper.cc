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

#include <memory>
#include <vector>
#include <iostream>
#include <string>
#include <cctype>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pq_helper.h"
#include "pq_utils.h"
#include "pq_bernoulli.h"
#include "pq_string.h"
#include "pq_add_label_ranges.h"
#include "pq_add_spin_labels.h"
#include "pq_cumulant_expansion.h"
#include "../pq_graph/include/pq_graph.h"

namespace py = pybind11;
using namespace pybind11::literals;

namespace pdaggerq {

std::vector<int> empty_list = {};

void export_pq_helper(py::module& m) {
    py::class_<pdaggerq::pq_helper, std::shared_ptr<pdaggerq::pq_helper> >(m, "pq_helper")
        .def(py::init< std::string >())
        .def("set_print_level", &pq_helper::set_print_level)
        .def("set_unitary_cc", &pq_helper::set_unitary_cc)
        .def("set_bernoulli_excitation_level", &pq_helper::set_bernoulli_excitation_level)
        .def("set_left_operators", &pq_helper::set_left_operators)
        .def("set_right_operators", &pq_helper::set_right_operators)
        .def("set_left_operators_type", &pq_helper::set_left_operators_type)
        .def("set_right_operators_type", &pq_helper::set_right_operators_type)
        .def("get_right_operators_type", &pq_helper::get_right_operators_type)
        .def("get_left_operators_type", &pq_helper::get_left_operators_type)
        .def("set_find_paired_permutations", &pq_helper::set_find_paired_permutations)
        .def("simplify", &pq_helper::simplify)
        .def("clear", &pq_helper::clear)
        .def("clone", &pq_helper::clone)
        .def("save", &pq_helper::serialize)
        .def("load", &pq_helper::deserialize)
        .def("set_use_rdms",
            [](pq_helper& self, const bool & do_use_rdms, const std::vector<int> & ignore_cumulant) {
                return self.set_use_rdms(do_use_rdms, ignore_cumulant);
            },
            py::arg("do_use_rdms"), py::arg("ignore_cumulant") = empty_list )
        .def("strings", 
            [](pq_helper& self, 
                const std::unordered_map<std::string, std::string> &spin_labels, 
                const std::unordered_map<std::string, std::vector<std::string> > &label_ranges) {

                bool has_spin_labels = spin_labels.find("DUMMY") == spin_labels.end();
                bool has_label_ranges = label_ranges.find("DUMMY") == label_ranges.end();

                if ( has_spin_labels && has_label_ranges ) {
                    printf("\n");
                    printf("    error: cannot simultaneously block by spin and by range\n");
                    printf("\n");
                    exit(1);
                }
                
                if ( has_spin_labels ) {
                
                    // spin blocking 
                    self.block_by_spin(spin_labels);
                    return self.strings();
                
                } else if ( has_label_ranges ) {
                
                    // range labels
                   self.block_by_range(label_ranges);
                   return self.strings();

                   // no blocking
                } else return self.strings();

            },
            py::arg("spin_labels") = std::unordered_map<std::string, std::string>{{"DUMMY",""}},
            py::arg("label_ranges") = std::unordered_map<std::string, std::vector<std::string>>{{"DUMMY",{""}}} )
        .def("block_by_spin",
            [](pq_helper& self, const std::unordered_map<std::string, std::string> &spin_labels) {
                self.block_by_spin(spin_labels);
            },
            py::arg("spin_labels") = std::unordered_map<std::string, std::string>() )
        .def("block_by_range",
            [](pq_helper& self, const std::unordered_map<std::string, std::vector<std::string> > &label_ranges) {
                self.block_by_range(label_ranges);
            },
            py::arg("spin_labels") = std::unordered_map<std::string, std::string>() )
        .def("add_st_operator",
            [](pq_helper& self, double factor, 
                                const std::vector<std::string> &targets, 
                                const std::vector<std::string> &ops, 
                                bool do_operators_commute) {
                return self.add_st_operator(factor, targets, ops, do_operators_commute);
            },
            py::arg("factor"), py::arg("targets"), py::arg("ops"), py::arg("do_operators_commute") = true )
        .def("get_st_operator_terms", &pq_helper::get_st_operator_terms)
        .def("add_bernoulli_operator", &pq_helper::add_bernoulli_operator)
        .def("add_anticommutator", &pq_helper::add_anticommutator)
        .def("add_commutator", &pq_helper::add_commutator)
        .def("add_double_commutator", &pq_helper::add_double_commutator)
        .def("add_triple_commutator", &pq_helper::add_triple_commutator)
        .def("add_quadruple_commutator", &pq_helper::add_quadruple_commutator)
        .def("add_quintuple_commutator", &pq_helper::add_quintuple_commutator)
        .def("add_hextuple_commutator", &pq_helper::add_hextuple_commutator)
        .def("add_operator_product", &pq_helper::add_operator_product);

    //py::class_<pdaggerq::pq_operator_terms, std::shared_ptr<pdaggerq::pq_operator_terms> >(m, "pq_operator_terms")
    //    .def(py::init< double, std::vector<std::string> >())
    //    .def("factor", &pq_operator_terms::factor)
    //    .def("operators", &pq_operator_terms::operators);
    py::class_<pdaggerq::pq_operator_terms>(m, "pq_operator_terms")
        .def(py::init<double, std::vector<std::string>>())
        .def("factor", &pq_operator_terms::get_factor)
        .def("operators", &pq_operator_terms::get_operators);

    // add pq graph class for optimizing, visualizing, and generating code from pq_helper
    PQGraph::export_pq_graph(m);
}

PYBIND11_MODULE(_pdaggerq, m) {
    m.doc() = "Python API of pdaggerq: A code for bringing strings of creation / annihilation operators to normal order.";
    export_pq_helper(m);
}

pq_helper::pq_helper(const std::string &vacuum_type)
{

    if ( vacuum_type.empty() ) {
        vacuum = "TRUE";
    }else if ( vacuum_type == "TRUE" || vacuum_type == "true" ) {
        vacuum = "TRUE";
    }else if ( vacuum_type == "FERMI" || vacuum_type == "fermi" ) {
        vacuum = "FERMI";
    }else {
        printf("\n");
        printf("    error: invalid vacuum type (%s)\n", vacuum_type.c_str());
        printf("\n");
        exit(1);
    }

    use_rdms = false;

    print_level = 0;

    // assume cluster operator is not antihermitian by default
    is_unitary_cc = false;

    // default maximum excitation order for "N" type operators in Bernoulli expansion for UCC
    bernoulli_excitation_level = 2;

    // by default, do not look for paired permutations (until parsers catch up)
    find_paired_permutations = false;

    /// right operators type (EE, IP, EA)
    right_operators_type = "EE";

    /// left operators type (EE, IP, EA)
    left_operators_type = "EE";

}

pq_helper::pq_helper(const pq_helper &other) {

    // copy data
    this->vacuum                    = other.vacuum;
    this->print_level               = other.print_level;
    this->use_rdms                  = other.use_rdms;
    this->ignore_cumulant_rdms      = other.ignore_cumulant_rdms;
    this->left_operators            = other.left_operators;
    this->right_operators           = other.right_operators;
    this->right_operators_type      = other.right_operators_type;
    this->left_operators_type       = other.left_operators_type;
    this->find_paired_permutations  = other.find_paired_permutations;

    // deep copy pointers to pq_strings
    ordered.clear();
    ordered.reserve(other.ordered.size());
    for (const std::shared_ptr<pq_string> & pq_str : other.ordered) {
        this->ordered.push_back(std::make_shared<pq_string>(*pq_str));
    }

    ordered_blocked.clear();
    ordered_blocked.reserve(other.ordered_blocked.size());
    for (const std::shared_ptr<pq_string> & pq_str : other.ordered_blocked) {
        this->ordered_blocked.push_back(std::make_shared<pq_string>(*pq_str));
    }
}

pq_helper &pq_helper::operator=(const pq_helper &other) {
    // check for self-assignment
    if (this == &other) {
        return *this;
    }

    // construct deep copy and move to this
    *this = std::move(pq_helper(other));
    return *this;

}


void pq_helper::set_find_paired_permutations(bool do_find_paired_permutations) {
    find_paired_permutations = do_find_paired_permutations;
}

void pq_helper::set_print_level(int level) {
    print_level = level;
}
void pq_helper::set_use_rdms(bool do_use_rdms, std::vector<int> ignore_cumulant = {}) {
    use_rdms = do_use_rdms;
    ignore_cumulant_rdms = ignore_cumulant;
}


void pq_helper::set_right_operators(const std::vector<std::vector<std::string>> &in) {

    right_operators.clear();
    for (const std::vector<std::string> & ops : in) {
        right_operators.push_back(ops);
    }
}

void pq_helper::set_left_operators(const std::vector<std::vector<std::string>> &in) {

    left_operators.clear();
    for (const std::vector<std::string> & ops : in) {
        left_operators.push_back(ops);
    }
}

void pq_helper::set_left_operators_type(const  std::string &type) {
    if ( type == "EE" || type == "IP" || type == "EA" || type == "DIP" || type == "DEA" ) {
        left_operators_type = type;
    }else {
        printf("\n");
        printf("    error: invalid left-hand operator type (%s)\n", type.c_str());
        printf("\n");
        exit(1);
    }
}

void pq_helper::set_right_operators_type(const std::string &type) {
    if ( type == "EE" || type == "IP" || type == "EA" || type == "DIP" || type == "DEA" ) {
        right_operators_type = type;
    }else {
        printf("\n");
        printf("    error: invalid right-hand operator type (%s)\n", type.c_str());
        printf("\n");
        exit(1);
    }
}

// is the cluster operator antihermitian for ucc? default false
void pq_helper::set_unitary_cc(bool is_unitary) {
    is_unitary_cc = is_unitary;
}

// is the cluster operator antihermitian for ucc? default false
void pq_helper::set_bernoulli_excitation_level(int excitation_level) {
    bernoulli_excitation_level = excitation_level;
}

void pq_helper::add_anticommutator(double factor,
                                   const std::vector<std::string> &op0,
                                   const std::vector<std::string> &op1){

    add_operator_product(factor, concatinate_operators({op0, op1}) );
    add_operator_product(factor, concatinate_operators({op1, op0}) );

}

void pq_helper::add_commutator(double factor,
                               const std::vector<std::string> &op0,
                               const std::vector<std::string> &op1){

    std::vector<pq_operator_terms> ops = get_commutator_terms(factor, op0, op1);
    for (auto op : ops){
        add_operator_product(op.factor, op.operators);
    }
}

std::vector<pq_operator_terms> pq_helper::get_commutator_terms(double factor,
                                                               const std::vector<std::string> &op0,
                                                               const std::vector<std::string> &op1){

    std::vector<pq_operator_terms> ops;

    ops.push_back(pq_operator_terms( factor, concatinate_operators({op0, op1})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op1, op0})));

    return ops;  
}

void pq_helper::add_double_commutator(double factor,
                                      const std::vector<std::string> &op0,
                                      const std::vector<std::string> &op1,
                                      const std::vector<std::string> &op2){

    std::vector<pq_operator_terms> ops = get_double_commutator_terms(factor, op0, op1, op2);
    for (auto op : ops){
        add_operator_product(op.factor, op.operators);
    }
}

std::vector<pq_operator_terms> pq_helper::get_double_commutator_terms(double factor,
                                                                      const std::vector<std::string> &op0,
                                                                      const std::vector<std::string> &op1,
                                                                      const std::vector<std::string> &op2){

    std::vector<pq_operator_terms> ops;

    ops.push_back(pq_operator_terms( factor, concatinate_operators({op0, op1, op2})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op1, op0, op2})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op2, op0, op1})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op2, op1, op0})));

    return ops;  
}

void pq_helper::add_triple_commutator(double factor,
                                        const std::vector<std::string> &op0,
                                        const std::vector<std::string> &op1,
                                        const std::vector<std::string> &op2,
                                        const std::vector<std::string> &op3){

    std::vector<pq_operator_terms> ops = get_triple_commutator_terms(factor, op0, op1, op2, op3);
    for (auto op : ops){
        add_operator_product(op.factor, op.operators);
    }
}

std::vector<pq_operator_terms> pq_helper::get_triple_commutator_terms(double factor,
                                                                      const std::vector<std::string> &op0,
                                                                      const std::vector<std::string> &op1,
                                                                      const std::vector<std::string> &op2,
                                                                      const std::vector<std::string> &op3){

    std::vector<pq_operator_terms> ops;

    ops.push_back(pq_operator_terms( factor, concatinate_operators({op0, op1, op2, op3})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op1, op0, op2, op3})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op2, op0, op1, op3})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op2, op1, op0, op3})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op3, op0, op1, op2})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op3, op1, op0, op2})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op3, op2, op0, op1})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op3, op2, op1, op0})));

    return ops;  
}

void pq_helper::add_quadruple_commutator(double factor,
                                         const std::vector<std::string> &op0,
                                         const std::vector<std::string> &op1,
                                         const std::vector<std::string> &op2,
                                         const std::vector<std::string> &op3,
                                         const std::vector<std::string> &op4){

    std::vector<pq_operator_terms> ops = get_quadruple_commutator_terms(factor, op0, op1, op2, op3, op4);
    for (auto op : ops){
        add_operator_product(op.factor, op.operators);
    }
}

std::vector<pq_operator_terms> pq_helper::get_quadruple_commutator_terms(double factor,
                                                                         const std::vector<std::string> &op0,
                                                                         const std::vector<std::string> &op1,
                                                                         const std::vector<std::string> &op2,
                                                                         const std::vector<std::string> &op3,
                                                                         const std::vector<std::string> &op4){

    std::vector<pq_operator_terms> ops;

    ops.push_back(pq_operator_terms( factor, concatinate_operators({op0, op1, op2, op3, op4})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op1, op0, op2, op3, op4})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op2, op0, op1, op3, op4})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op2, op1, op0, op3, op4})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op3, op0, op1, op2, op4})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op3, op1, op0, op2, op4})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op3, op2, op0, op1, op4})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op3, op2, op1, op0, op4})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op4, op0, op1, op2, op3})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op4, op1, op0, op2, op3})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op4, op2, op0, op1, op3})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op4, op2, op1, op0, op3})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op4, op3, op0, op1, op2})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op4, op3, op1, op0, op2})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op4, op3, op2, op0, op1})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op4, op3, op2, op1, op0})));

    return ops;  
}

std::vector<pq_operator_terms> pq_helper::get_quintuple_commutator_terms(double factor,
                                                                         const std::vector<std::string> &op0,
                                                                         const std::vector<std::string> &op1,
                                                                         const std::vector<std::string> &op2,
                                                                         const std::vector<std::string> &op3,
                                                                         const std::vector<std::string> &op4,
                                                                         const std::vector<std::string> &op5){

    std::vector<pq_operator_terms> ops;

    ops.push_back(pq_operator_terms( factor, concatinate_operators({op0, op1, op2, op3, op4, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op1, op0, op2, op3, op4, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op2, op0, op1, op3, op4, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op2, op1, op0, op3, op4, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op3, op0, op1, op2, op4, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op3, op1, op0, op2, op4, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op3, op2, op0, op1, op4, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op3, op2, op1, op0, op4, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op4, op0, op1, op2, op3, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op4, op1, op0, op2, op3, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op4, op2, op0, op1, op3, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op4, op2, op1, op0, op3, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op4, op3, op0, op1, op2, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op4, op3, op1, op0, op2, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op4, op3, op2, op0, op1, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op4, op3, op2, op1, op0, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op0, op1, op2, op3, op4})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op1, op0, op2, op3, op4})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op2, op0, op1, op3, op4})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op2, op1, op0, op3, op4})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op3, op0, op1, op2, op4})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op3, op1, op0, op2, op4})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op3, op2, op0, op1, op4})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op3, op2, op1, op0, op4})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op4, op0, op1, op2, op3})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op4, op1, op0, op2, op3})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op4, op2, op0, op1, op3})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op4, op2, op1, op0, op3})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op4, op3, op0, op1, op2})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op4, op3, op1, op0, op2})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op4, op3, op2, op0, op1})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op4, op3, op2, op1, op0})));

    return ops;  
}

void pq_helper::add_quintuple_commutator(double factor,
                                         const std::vector<std::string> &op0,
                                         const std::vector<std::string> &op1,
                                         const std::vector<std::string> &op2,
                                         const std::vector<std::string> &op3,
                                         const std::vector<std::string> &op4,
                                         const std::vector<std::string> &op5){

    std::vector<pq_operator_terms> ops = get_quintuple_commutator_terms(factor, op0, op1, op2, op3, op4, op5);
    for (auto op : ops){
        add_operator_product(op.factor, op.operators);
    }
}

std::vector<pq_operator_terms> pq_helper::get_hextuple_commutator_terms(double factor,
                                                                        const std::vector<std::string> &op0,
                                                                        const std::vector<std::string> &op1,
                                                                        const std::vector<std::string> &op2,
                                                                        const std::vector<std::string> &op3,
                                                                        const std::vector<std::string> &op4,
                                                                        const std::vector<std::string> &op5,
                                                                        const std::vector<std::string> &op6){

    std::vector<pq_operator_terms> ops;

    ops.push_back(pq_operator_terms( factor, concatinate_operators({op0, op1, op2, op3, op4, op5, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op1, op0, op2, op3, op4, op5, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op2, op0, op1, op3, op4, op5, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op2, op1, op0, op3, op4, op5, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op3, op0, op1, op2, op4, op5, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op3, op1, op0, op2, op4, op5, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op3, op2, op0, op1, op4, op5, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op3, op2, op1, op0, op4, op5, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op4, op0, op1, op2, op3, op5, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op4, op1, op0, op2, op3, op5, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op4, op2, op0, op1, op3, op5, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op4, op2, op1, op0, op3, op5, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op4, op3, op0, op1, op2, op5, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op4, op3, op1, op0, op2, op5, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op4, op3, op2, op0, op1, op5, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op4, op3, op2, op1, op0, op5, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op0, op1, op2, op3, op4, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op1, op0, op2, op3, op4, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op2, op0, op1, op3, op4, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op2, op1, op0, op3, op4, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op3, op0, op1, op2, op4, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op3, op1, op0, op2, op4, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op3, op2, op0, op1, op4, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op3, op2, op1, op0, op4, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op4, op0, op1, op2, op3, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op4, op1, op0, op2, op3, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op4, op2, op0, op1, op3, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op4, op2, op1, op0, op3, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op4, op3, op0, op1, op2, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op4, op3, op1, op0, op2, op6})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op5, op4, op3, op2, op0, op1, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op5, op4, op3, op2, op1, op0, op6})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op0, op1, op2, op3, op4, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op1, op0, op2, op3, op4, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op2, op0, op1, op3, op4, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op2, op1, op0, op3, op4, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op3, op0, op1, op2, op4, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op3, op1, op0, op2, op4, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op3, op2, op0, op1, op4, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op3, op2, op1, op0, op4, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op4, op0, op1, op2, op3, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op4, op1, op0, op2, op3, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op4, op2, op0, op1, op3, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op4, op2, op1, op0, op3, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op4, op3, op0, op1, op2, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op4, op3, op1, op0, op2, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op4, op3, op2, op0, op1, op5})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op4, op3, op2, op1, op0, op5})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op5, op0, op1, op2, op3, op4})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op5, op1, op0, op2, op3, op4})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op5, op2, op0, op1, op3, op4})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op5, op2, op1, op0, op3, op4})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op5, op3, op0, op1, op2, op4})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op5, op3, op1, op0, op2, op4})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op5, op3, op2, op0, op1, op4})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op5, op3, op2, op1, op0, op4})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op5, op4, op0, op1, op2, op3})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op5, op4, op1, op0, op2, op3})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op5, op4, op2, op0, op1, op3})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op5, op4, op2, op1, op0, op3})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op5, op4, op3, op0, op1, op2})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op5, op4, op3, op1, op0, op2})));
    ops.push_back(pq_operator_terms(-factor, concatinate_operators({op6, op5, op4, op3, op2, op0, op1})));
    ops.push_back(pq_operator_terms( factor, concatinate_operators({op6, op5, op4, op3, op2, op1, op0})));

    return ops;  
}

void pq_helper::add_hextuple_commutator(double factor,
                                        const std::vector<std::string> &op0,
                                        const std::vector<std::string> &op1,
                                        const std::vector<std::string> &op2,
                                        const std::vector<std::string> &op3,
                                        const std::vector<std::string> &op4,
                                        const std::vector<std::string> &op5,
                                        const std::vector<std::string> &op6){

    std::vector<pq_operator_terms> ops = get_hextuple_commutator_terms(factor, op0, op1, op2, op3, op4, op5, op6);
    for (auto op : ops){
        add_operator_product(op.factor, op.operators);
    }
}

// add a string of operators
void pq_helper::add_operator_product(double factor, std::vector<std::string>  in){

    // check if there is a fluctuation potential operator 
    // that needs to be split into multiple terms

    // left operators 
    // this is not handled correctly now that left operators can be sums of products of operators ... just exit with an error
    for (std::vector<std::string> & left_operator : left_operators) {
        std::vector<std::string> tmp;
        for (const std::string & op : left_operator) {
            if (op == "v" || op == "V" || op.substr(0, 2) == "v{" || op.substr(0, 2) == "V{") {

                printf("\n");
                printf("    error: the fluctuation potential cannot appear in operators defining the bra state\n");
                printf("\n");
                exit(1);

            }else {
                tmp.push_back(op);
            }
        }
        left_operator.clear();
        for (const auto & op : tmp) {
            left_operator.push_back(op);
        }
        tmp.clear();
    }
    
    // right operators 
    // this is not handled correectly now that right operators can be sums of products of operators ... just exit with an error
    for (std::vector<std::string> & right_operator : right_operators) {
        std::vector<std::string> tmp;
        for (const std::string & op : right_operator) {
            if (op == "v" || op == "V" || op.substr(0, 2) == "v{" || op.substr(0, 2) == "V{") {

                printf("\n");
                printf("    error: the fluctuation potential cannot appear in operators defining the ket state\n");
                printf("\n");
                exit(1);

            }else {
                tmp.push_back(op);
            }
        }
        right_operator.clear();
        for (const std::string & op : tmp) {
            right_operator.push_back(op);
        }
        tmp.clear();
    }

    int count = 0;
    bool found_v = false;
    std::vector<std::string> tmp_in;
    for (const std::string & op : in) {
        if (op == "v" || op == "V" || op.substr(0, 2) == "v{" || op.substr(0, 2) == "V{") {
            found_v = true;
            break;
        }else {
            tmp_in.push_back(op);
            count++;
        }
    }
    if ( found_v ) {

        // get bernoulli operator portions
        std::string op_portions = get_operator_portions_as_string(in[count]);

        // term 1
        std::string v_type = "j1";
        if ( op_portions.length() > 0 ) { 
            v_type += "{" + op_portions + "}";
        }
        tmp_in.emplace_back(v_type);
        for (int i = count+1; i < (int)in.size(); i++) {
            tmp_in.push_back(in[i]);
        }
        in.clear();
        for (const auto & op : tmp_in) {
            in.push_back(op);
        }
        add_operator_product(factor, in);

        // term 2
        in.clear();
        for (int i = 0; i < count; i++) {
            in.push_back(tmp_in[i]);
        }
        v_type[1] = '2';
        in.emplace_back(v_type);
        for (int i = count + 1; i < (int)tmp_in.size(); i++) {
            in.push_back(tmp_in[i]);
        }
        add_operator_product(factor, in);
        
        return;
    }

    // now check for t and add de-excitation operators if doing unitary cc
    // first, if unitary cc, t can't show up in right or left operator lists (yet)
    if ( is_unitary_cc ) {
        for (size_t i = 0; i < left_operators.size(); i++) {
            for (size_t j = 0; j < left_operators[i].size(); j++) {
                if ( left_operators[i][j].substr(0,1) == "t" || left_operators[i][j].substr(0,1) == "T" ||
                     left_operators[i][j].substr(0,2) == "t{" || left_operators[i][j].substr(0,2) == "T{" ){

                    printf("\n");
                    printf("    error: unitary cluster operators cannot appear in the bra state\n");
                    printf("\n");
                    exit(1);

                }
            }
        }
        for (size_t i = 0; i < right_operators.size(); i++) {
            for (size_t j = 0; j < right_operators[i].size(); j++) {
                if ( right_operators[i][j].substr(0,1) == "t" || right_operators[i][j].substr(0,1) == "T" ||
                     right_operators[i][j].substr(0,2) == "t{" || right_operators[i][j].substr(0,2) == "T{" ){

                    printf("\n");
                    printf("    error: unitary cluster operators cannot appear in the ket state\n");
                    printf("\n");
                    exit(1);

                }
            }
        }
    }

    // now either rename cluster operators or split them into two, depending on whether we're unitary or not
    count = 0;
    bool found_t = false;
    for (size_t i = 0; i < in.size(); i++) {
        if ( in[i].substr(0,1) == "t" || in[i].substr(0,1) == "T" ) {
            if ( in[i].substr(0,2) != "te" && in[i].substr(0,2) != "td" ) {
                found_t = true;
                break;
            }else {
                count++;
            }
        }else {
            count++;
        }
    }

    if ( found_t ) {

        // term 1 (excitation)

        in[count].insert(1, "e", 1);
        add_operator_product(factor, in);

        // term 2 (de-excitation)
        if ( is_unitary_cc ) {

            in[count][1] = 'd';
            add_operator_product(-factor, in);
        }
        return;
    }

    // apply any extra operators on left or right:
    std::vector<std::string> save;
    for (const std::string & op : in) {
        save.push_back(op);
    }

    if ( (int)left_operators.size() == 0 ) {
        std::vector<std::string> junk;
        junk.emplace_back("1");
        left_operators.push_back(junk);
    }
    if ( (int)right_operators.size() == 0 ) {
        std::vector<std::string> junk;
        junk.emplace_back("1");
        right_operators.push_back(junk);
    }

    // build strings
    double original_factor = factor;

    for (std::vector<std::string> & left_operator : left_operators) {
        for (std::vector<std::string> & right_operator : right_operators) {

            std::shared_ptr<pq_string> newguy (new pq_string(vacuum));

            factor = original_factor;

            std::vector<std::string> tmp_string;

            bool has_w0       = false;

            int occ_label_count = 0;
            int vir_label_count = 0;
            int gen_label_count = 0;

            // apply any extra operators on left or right:
            std::vector<std::string> tmp = left_operator;
            for (const std::string & op : save) {
                tmp.push_back(op);
            }
            for (const std::string & op : right_operator) {
                tmp.push_back(op);
            }

            for (std::string & op_including_portions : tmp) {

                // bernoulli expansion requires operator portion specification. split into base name and portion
                std::string op = get_operator_base_name(op_including_portions);
                std::vector<std::string> op_portions = get_operator_portions_as_vector(op_including_portions);
                
                // blank string
                if ( op.empty() ) continue;

                // Stephen: removed so that we can distinguish lower- and uppercase indices
                // lowercase indices
                // std::transform(op.begin(), op.end(), op.begin(), [](unsigned char c){ return std::tolower(c); });

                // remove spaces
                removeSpaces(op);

                // remove parentheses
                removeParentheses(op);

                if (op.substr(0, 1) == "h" || op.substr(0, 1) == "H") { // one-electron operator

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // integrals
                    newguy->set_integrals("core", {idx1, idx2}, op_portions);

                }else if (op.substr(0, 1) == "f" || op.substr(0, 1) == "F") { // fock operator

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // integrals
                    newguy->set_integrals("fock", {idx1, idx2}, op_portions);

                }else if (op.substr(0, 2) == "d+" || op.substr(0, 2) == "D+") { // one-electron operator (dipole + boson creator)

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // integrals
                    newguy->set_integrals("d+", {idx1, idx2}, op_portions);

                    // boson operator
                    newguy->is_boson_dagger.push_back(true);

                }else if (op.substr(0, 2) == "d-" || op.substr(0, 2) == "D-") { // one-electron operator (dipole + boson annihilator)

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // integrals
                    newguy->set_integrals("d-", {idx1, idx2}, op_portions);

                    // boson operator
                    newguy->is_boson_dagger.push_back(false);

                }else if (op.substr(0, 1) == "g" || op.substr(0, 1) == "G") { // general two-electron operator

                    //factor *= 0.25;

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);
                    std::string idx3 = "p" + std::to_string(gen_label_count++);
                    std::string idx4 = "p" + std::to_string(gen_label_count++);

                    tmp_string.push_back(idx1+"*");
                    tmp_string.push_back(idx2+"*");
                    tmp_string.push_back(idx3);
                    tmp_string.push_back(idx4);

                    newguy->set_integrals("two_body", {idx1, idx2, idx4, idx3}, op_portions);

                }else if (op.substr(0, 1) == "j" || op.substr(0, 1) == "J") { // fluctuation potential

                    if (op.substr(1, 1) == "1" ){

                        factor *= -1.0;

                        std::string idx1 = "p" + std::to_string(gen_label_count++);
                        std::string idx2 = "p" + std::to_string(gen_label_count++);

                        // index 1
                        tmp_string.push_back(idx1+"*");

                        // index 2
                        tmp_string.push_back(idx2);

                        // integrals
                        newguy->set_integrals("occ_repulsion", {idx1, idx2}, op_portions);

                    }else if (op.substr(1, 1) == "2" ){

                        factor *= 0.25;

                        std::string idx1 = "p" + std::to_string(gen_label_count++);
                        std::string idx2 = "p" + std::to_string(gen_label_count++);
                        std::string idx3 = "p" + std::to_string(gen_label_count++);
                        std::string idx4 = "p" + std::to_string(gen_label_count++);

                        tmp_string.push_back(idx1+"*");
                        tmp_string.push_back(idx2+"*");
                        tmp_string.push_back(idx3);
                        tmp_string.push_back(idx4);

                        newguy->set_integrals("eri", {idx1, idx2, idx4, idx3}, op_portions);

                    }

                }else if (op.substr(0, 1) == "t"){

                    int n = std::stoi(op.substr(2));
                    std::vector<std::string> labels;

                    if ( n == 0 ){

                        // nothing to do

                    }else {

                        std::vector<std::string> op_left;
                        std::vector<std::string> op_right;
                        std::vector<std::string> label_left;
                        std::vector<std::string> label_right;
 
                        // excitation:
                        if ( op.substr(0,2) == "te" ) {

                            for (int id = 0; id < n; id++) {

                                std::string idx1 = "v" + std::to_string(vir_label_count++);
                                std::string idx2 = "o" + std::to_string(occ_label_count++);

                                op_left.push_back(idx1+"*");
                                op_right.push_back(idx2);

                                label_left.push_back(idx1);
                                label_right.push_back(idx2);
                            }
                        }else if ( op.substr(0,2) == "td" ) {

                            // de-excitation:
                            for (int id = 0; id < n; id++) {

                                std::string idx1 = "v" + std::to_string(vir_label_count++);
                                std::string idx2 = "o" + std::to_string(occ_label_count++);

                                op_left.push_back(idx2+"*");
                                op_right.push_back(idx1);

                                // do not transpose de-excitation amplitude labels
                                //label_left.push_back(idx1);
                                //label_right.push_back(idx2);
                                // transpose de-excitation amplitude labels
                                label_left.push_back(idx2);
                                label_right.push_back(idx1);
                            }
                        }else {
                            printf("\n");
                            printf("    invalid operator type: %s\n", op.c_str());
                            printf("\n");
                            exit(1);
                        }

                        // a*b*...
                        for (int id = 0; id < n; id++) {
                            tmp_string.push_back(op_left[id]);
                        }
                        // op*j*...
                        for (int id = 0; id < n; id++) {
                            tmp_string.push_back(op_right[id]);
                        }

                        // tn(ab...
                        for (int id = 0; id < n; id++) {
                            labels.push_back(label_left[id]);
                        }
                        // tn(ab......ji)
                        for (int id = n-1; id >= 0; id--) {
                            labels.push_back(label_right[id]);
                        }

                        // factor = 1/(n!)^2
                        double my_factor = 1.0;
                        for (int id = 0; id < n; id++) {
                            my_factor *= (id+1);
                        }
                        factor *= 1.0 / my_factor / my_factor;
                    }

                    int n_ph = 0;
                    if (op.size() > 3 ) {
                        if ( op.substr(3,1) == ",") {
                            n_ph = std::stoi(op.substr(4));
                            if ( op.substr(0,2) == "te" ) {
                                // excitation
                                for (int ph = 0; ph < n_ph; ph++) {
                                    newguy->is_boson_dagger.push_back(true);
                                }
                            }else if ( op.substr(0,2) == "td" ) {
                                // de-excitation
                                for (int ph = 0; ph < n_ph; ph++) {
                                    newguy->is_boson_dagger.push_back(false);
                                }
                            }
                        }
                    }
                    newguy->set_amplitudes('t', n, n, n_ph, labels, op_portions);

                }else if (op.substr(0, 1) == "w" || op.substr(0, 1) == "W"){ // w0 B*B

                    if (op.substr(1, 1) == "0" ){

                        has_w0 = true;

                        newguy->is_boson_dagger.push_back(true);
                        newguy->is_boson_dagger.push_back(false);

                    }else {
                        printf("\n");
                        printf("    error: only w0 is supported\n");
                        printf("\n");
                        exit(1);
                    }

                }else if (op.substr(0, 2) == "b+" || op.substr(0, 2) == "B+"){ // B*

                        newguy->is_boson_dagger.push_back(true);

                }else if (op.substr(0, 2) == "b-" || op.substr(0, 2) == "B-"){ // B

                        newguy->is_boson_dagger.push_back(false);

                }else if (op.substr(0, 1) == "r" || op.substr(0, 1) == "R"){


                    int n = std::stoi(op.substr(1));
                    int n_annihilate = n;
                    int n_create     = n;
                    std::vector<std::string> labels;

                    if ( n == 0 ){

                        // nothing to do

                    }else {

                        if ( right_operators_type == "IP" ) n_create--;
                        if ( right_operators_type == "DIP" ) n_create -= 2;
                        if ( right_operators_type == "EA" ) n_annihilate--;
                        if ( right_operators_type == "DEA" ) n_annihilate -= 2;

                        std::vector<std::string> op_left;
                        std::vector<std::string> op_right;
                        std::vector<std::string> label_left;
                        std::vector<std::string> label_right;
                        for (int id = 0; id < n_create; id++) {
                            std::string idx1 = "v" + std::to_string(vir_label_count++);
                            op_left.push_back(idx1+"*");
                            label_left.push_back(idx1);
                        }
                        for (int id = 0; id < n_annihilate; id++) {
                            std::string idx2 = "o" + std::to_string(occ_label_count++);
                            op_right.push_back(idx2);
                            label_right.push_back(idx2);
                        }
                        // a*b*...
                        for (int id = 0; id < n_create; id++) {
                            tmp_string.push_back(op_left[id]);
                        }
                        // ij...
                        for (int id = 0; id < n_annihilate; id++) {
                            tmp_string.push_back(op_right[id]);
                        }

                        // tn(ab...
                        for (int id = 0; id < n_create; id++) {
                            labels.push_back(label_left[id]);
                        }
                        // tn(ab......ji)
                        for (int id = n_annihilate-1; id >= 0; id--) {
                            labels.push_back(label_right[id]);
                        }

                        // factor = 1/(n!)^2
                        double my_factor_create = 1.0;
                        double my_factor_annihilate = 1.0;
                        for (int id = 0; id < n_create; id++) {
                            my_factor_create *= (id+1);
                        }
                        for (int id = 0; id < n_annihilate; id++) {
                            my_factor_annihilate *= (id+1);
                        }
                        factor *= 1.0 / my_factor_create / my_factor_annihilate;
                    }

                    int n_ph = 0;
                    if (op.size() > 2 ) {
                        if ( op.substr(2,1) == ",") {
                            n_ph = std::stoi(op.substr(3));
                            for (int ph = 0; ph < n_ph; ph++) {
                                newguy->is_boson_dagger.push_back(true);
                            }
                        }
                    }
                    newguy->set_amplitudes('r', n_create, n_annihilate, n_ph, labels, op_portions);

                }else if (op.substr(0, 1) == "l" || op.substr(0, 1) == "L"){

                    int n = std::stoi(op.substr(1));
                    int n_annihilate = n;
                    int n_create     = n;
                    std::vector<std::string> labels;

                    if ( n == 0 ){

                        // nothing to do

                    }else {
                        
                        if ( left_operators_type == "IP" ) n_annihilate--;
                        if ( left_operators_type == "DIP" ) n_annihilate -= 2;
                        if ( left_operators_type == "EA" ) n_create--;
                        if ( left_operators_type == "DEA" ) n_create -= 2;

                        std::vector<std::string> op_left;
                        std::vector<std::string> op_right;
                        std::vector<std::string> label_left;
                        std::vector<std::string> label_right;
                        for (int id = 0; id < n_create; id++) {
                            std::string idx1 = "o" + std::to_string(occ_label_count++);
                            op_left.push_back(idx1+"*");
                            label_left.push_back(idx1);
                        }
                        for (int id = 0; id < n_annihilate; id++) {
                            std::string idx2 = "v" + std::to_string(vir_label_count++);
                            op_right.push_back(idx2);
                            label_right.push_back(idx2);
                        }
                        // op*j*...
                        for (int id = 0; id < n_create; id++) {
                            tmp_string.push_back(op_left[id]);
                        }
                        // ab...
                        for (int id = 0; id < n_annihilate; id++) {
                            tmp_string.push_back(op_right[id]);
                        }

                        // tn(ij... 
                        for (int id = 0; id < n_create; id++) {
                            labels.push_back(label_left[id]);
                        }
                        // tn(ij......ba)
                        for (int id = n_annihilate-1; id >= 0; id--) {
                            labels.push_back(label_right[id]);
                        }
                        
                        // factor = 1/(n!)^2
                        double my_factor_create = 1.0;
                        double my_factor_annihilate = 1.0;
                        for (int id = 0; id < n_create; id++) {
                            my_factor_create *= (id+1);
                        }
                        for (int id = 0; id < n_annihilate; id++) {
                            my_factor_annihilate *= (id+1);
                        }
                        factor *= 1.0 / my_factor_create / my_factor_annihilate;
                    
                    }

                    int n_ph = 0;
                    if (op.size() > 2 ) {
                        if ( op.substr(2,1) == ",") {
                            n_ph = std::stoi(op.substr(3));
                            for (int ph = 0; ph < n_ph; ph++) {
                                newguy->is_boson_dagger.push_back(false);
                            }
                        }
                    }
                    newguy->set_amplitudes('l', n_create, n_annihilate, n_ph, labels, op_portions);

                }else if (op.substr(0, 1) == "e" || op.substr(0, 1) == "E"){


                    if (op.substr(1, 1) == "1" ){

                        // find comma
                        size_t pos = op.find(',');
                        if ( pos == std::string::npos ) {
                            printf("\n");
                            printf("    error in e1 operator definition\n");
                            printf("\n");
                            exit(1);
                        }
                        size_t len = pos - 2; 

                        // index 1
                        tmp_string.push_back(op.substr(2, len) + "*");

                        // index 2
                        tmp_string.push_back(op.substr(pos + 1));

                    }else if (op.substr(1, 1) == "2" ){

                        // count indices
                        size_t pos = 0;
                        int ncomma = 0;
                        std::vector<size_t> commas;
                        pos = op.find(',', pos + 1);
                        commas.push_back(pos);
                        while( pos != std::string::npos){
                            pos = op.find(',', pos + 1);
                            commas.push_back(pos);
                            ncomma++;
                        }

                        if ( ncomma != 3 ) {
                            printf("\n");
                            printf("    error in e2 definition\n");
                            printf("\n");
                            exit(1);
                        }

                        tmp_string.push_back(op.substr(2, commas[0] - 2) + "*");
                        tmp_string.push_back(op.substr(commas[0] + 1, commas[1] - commas[0] - 1) + "*");
                        tmp_string.push_back(op.substr(commas[1] + 1, commas[2] - commas[1] - 1));
                        tmp_string.push_back(op.substr(commas[2] + 1));

                    }else if (op.substr(1, 1) == "3" ){

                        // count indices
                        size_t pos = 0;
                        int ncomma = 0;
                        std::vector<size_t> commas;
                        pos = op.find(',', pos + 1);
                        commas.push_back(pos);
                        while( pos != std::string::npos){
                            pos = op.find(',', pos + 1);
                            commas.push_back(pos);
                            ncomma++;
                        }

                        if ( ncomma != 5 ) {
                            printf("\n");
                            printf("    error in e3 definition\n");
                            printf("\n");
                            exit(1);
                        }

                        tmp_string.push_back(op.substr(2, commas[0] - 2) + "*");
                        tmp_string.push_back(op.substr(commas[0] + 1, commas[1] - commas[0] - 1) + "*");
                        tmp_string.push_back(op.substr(commas[1] + 1, commas[2] - commas[1] - 1) + "*");
                        tmp_string.push_back(op.substr(commas[2] + 1, commas[3] - commas[2] - 1));
                        tmp_string.push_back(op.substr(commas[3] + 1, commas[4] - commas[3] - 1));
                        tmp_string.push_back(op.substr(commas[4] + 1));

                    }else if (op.substr(1, 1) == "4" ){

                        // count indices
                        size_t pos = 0;
                        int ncomma = 0;
                        std::vector<size_t> commas;
                        pos = op.find(',', pos + 1);
                        commas.push_back(pos);
                        while( pos != std::string::npos){
                            pos = op.find(',', pos + 1);
                            commas.push_back(pos);
                            ncomma++;
                        }

                        if ( ncomma != 7 ) {
                            printf("\n");
                            printf("    error in e4 definition\n");
                            printf("\n");
                            exit(1);
                        }

                        tmp_string.push_back(op.substr(2, commas[0] - 2) + "*");
                        tmp_string.push_back(op.substr(commas[0] + 1, commas[1] - commas[0] - 1) + "*");
                        tmp_string.push_back(op.substr(commas[1] + 1, commas[2] - commas[1] - 1) + "*");
                        tmp_string.push_back(op.substr(commas[2] + 1, commas[3] - commas[2] - 1) + "*");
                        tmp_string.push_back(op.substr(commas[3] + 1, commas[4] - commas[3] - 1));
                        tmp_string.push_back(op.substr(commas[4] + 1, commas[5] - commas[4] - 1));
                        tmp_string.push_back(op.substr(commas[5] + 1, commas[6] - commas[5] - 1));
                        tmp_string.push_back(op.substr(commas[6] + 1));

                    }else {
                        printf("\n");
                        printf("    error: only e1, e2, e3, and e4 operators are supported\n");
                        printf("\n");
                        exit(1);
                    }

                }else if (op.substr(0, 1) == "1" ) { // unit operator ... do nothing

                }else if (op.substr(0, 1) == "a" || op.substr(0, 1) == "A"){ // single creator / annihilator


                    if (op.substr(1, 1) == "*" ){ // creator

                        tmp_string.push_back(op.substr(1) + "*");

                    }else { // annihilator

                        tmp_string.push_back(op.substr(1));

                    }

                }else {
                        printf("\n");
                        printf("    error: undefined string\n");
                        printf("\n");
                        exit(1);
                }
            }

            newguy->factor = factor;

            for (const std::string & op : tmp_string) {
                newguy->string.push_back(op);
            }

            newguy->has_w0 = has_w0;

            // make sure factor > 0
            if ( newguy->factor < 0.0 ) {
                newguy->factor = fabs(newguy->factor);
                newguy->sign *= -1;
            }

            if (vacuum == "TRUE") {
                add_new_string_true_vacuum(newguy, ordered, print_level, find_paired_permutations);
            } else {
                add_new_string_fermi_vacuum(newguy, ordered, print_level, find_paired_permutations, occ_label_count, vir_label_count);
            }
        }
    }
}

void pq_helper::simplify() {

    // eliminate strings based on delta functions and use delta functions to alter integral / amplitude labels
    for (std::shared_ptr<pq_string> & pq_str : ordered) {

        if ( pq_str->skip ) continue;

        // apply delta functions
        gobble_deltas(pq_str);

        // re-classify fluctuation potential terms
        reclassify_integrals(pq_str);

        // replace any funny labels that were added with conventional ones
        use_conventional_labels(pq_str);

        // eliminate terms based on operator portions (for bernoulli)
        eliminate_operator_portions(pq_str, bernoulli_excitation_level);

        // if UCC de-excitation amplitudes were transposed, transpose them back
        if ( is_unitary_cc ) {
            // relabel amplitudes t(i, a) -> t(a, i)
            for (size_t j = 0; j < pq_str->amps['t'].size(); j++) {
                // check if first label is occupied or not. if so, reverse order and flip sign
                if ( pq_str->amps['t'][j].labels.size() == 0 ) continue;
                if ( is_occ(pq_str->amps['t'][j].labels[0]) ) {
                    std::reverse(pq_str->amps['t'][j].labels.begin(), pq_str->amps['t'][j].labels.end());
                }
            }
        }

        // replace creation / annihilation operators with rdms
        if ( use_rdms ) {

            size_t n = pq_str->symbol.size();
            size_t n_create = 0;
            size_t n_annihilate = 0;
            for (size_t i = 0; i < n; i++) {
                if ( pq_str->is_dagger[i] ) n_create++;
                else                        n_annihilate++;
            }

            if ( n_create != n_annihilate ) {
                printf("\n");
                printf("    error: rdms not defined for this case\n");
                printf("\n");
                exit(1);
            }

            std::vector<std::string> rdm_labels;
            for (size_t i = 0; i < n_create; i++) {
                rdm_labels.push_back(pq_str->symbol[i]);
            }
            for (size_t i = 0; i < n_annihilate; i++) {
                rdm_labels.push_back(pq_str->symbol[n - i - 1]);
            }

            // TODO: we're assuming no photons ... 
            // TODO: would there ever be a use case where we'd want to specify operator portions here?
            pq_str->set_amplitudes('D', n_create, n_annihilate, 0, rdm_labels);
            pq_str->symbol.clear();
        }
    }

    // replace rdms with cumulant expansion, ignoring the n-body cumulant
    cumulant_expansion(ordered, ignore_cumulant_rdms);

    // try to cancel similar terms
    cleanup(ordered, find_paired_permutations);

}

// block labels by orbital spaces
void pq_helper::block_by_range(const std::unordered_map<std::string, std::vector<std::string>> &label_ranges) {
    ordered_blocked.clear();

    // add ranges to labels
    pq_string::is_range_blocked = true;
    if ( pq_string::is_spin_blocked ) {
        printf("\n");
        printf("    error: cannot simultaneously block by spin and by range\n");
        printf("\n");
        exit(1);
    }

    std::vector< std::shared_ptr<pq_string> > range_blocked;

    for (auto & pq_str : ordered) {
        if ( !pq_str->symbol.empty() ) continue;
        if ( !pq_str->is_boson_dagger.empty() ) continue;
        std::vector< std::shared_ptr<pq_string> > tmp;
        add_label_ranges(pq_str, tmp, label_ranges);
        for (const auto & op : tmp) {
            ordered_blocked.push_back(op);
        }
    }
}

// block labels by spin
void pq_helper::block_by_spin(const std::unordered_map<std::string, std::string> &spin_labels) {
    ordered_blocked.clear();

    // perform spin tracing
    pq_string::is_spin_blocked = true;
    if ( pq_string::is_range_blocked ) {
        printf("\n");
        printf("    error: cannot simultaneously block by spin and by range\n");
        printf("\n");
        exit(1);
    }

    for (std::shared_ptr<pq_string> & pq_str : ordered) {
        if (!pq_str->symbol.empty()) continue;
        if (!pq_str->is_boson_dagger.empty()) continue;
        std::vector<std::shared_ptr<pq_string> > tmp_ordered;
        spin_blocking(pq_str, tmp_ordered, spin_labels);
        for (const std::shared_ptr<pq_string> & tmp_pq_str : tmp_ordered) {
            ordered_blocked.push_back(tmp_pq_str);
        }
    }
}

std::vector<std::vector<std::string> > pq_helper::strings() const {

    bool is_blocked = pq_string::is_spin_blocked || pq_string::is_range_blocked;
    const auto &reference = is_blocked ? ordered_blocked : ordered;

    std::vector<std::vector<std::string> > list;
    for (const std::shared_ptr<pq_string> & pq_str : reference) {
        std::vector<std::string> my_string = pq_str->get_string();
        if ( (int)my_string.size() > 0 ) {
            list.push_back(my_string);
        }
    }

    return list;

}

void pq_helper::clear() {
    ordered.clear();
    ordered_blocked.clear();
    pq_string::is_spin_blocked = false;
    pq_string::is_range_blocked = false;
}

void pq_helper::add_st_operator(double factor, 
                                const std::vector<std::string> &targets,
                                const std::vector<std::string> &ops,
                                bool do_operators_commute = true){

    std::vector<pq_operator_terms> st_terms = get_st_operator_terms(factor, targets, ops, do_operators_commute);
    for (auto term : st_terms){
        add_operator_product(term.factor, term.operators);
    }
}

std::vector<pq_operator_terms> pq_helper::get_st_operator_terms(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops, bool do_operators_commute = true){

    int dim = (int)ops.size();

    std::vector<pq_operator_terms> st_terms;
    st_terms.push_back(pq_operator_terms(factor, targets));

    for (int i = 0; i < dim; i++) {
        std::vector<pq_operator_terms> tmp = get_commutator_terms(factor, targets, {ops[i]});
        st_terms.insert(std::end(st_terms), std::begin(tmp), std::end(tmp));
    }

    if ( do_operators_commute ) {

        for (int i = 0; i < dim; i++) {
            for (int j = i + 1; j < dim; j++) {
                std::vector<pq_operator_terms> tmp = get_double_commutator_terms(factor, targets, {ops[i]}, {ops[j]});
                st_terms.insert(std::end(st_terms), std::begin(tmp), std::end(tmp));
            }
        }
        for (int i = 0; i < dim; i++) {
            std::vector<pq_operator_terms> tmp = get_double_commutator_terms(0.5 * factor, targets, {ops[i]}, {ops[i]});
            st_terms.insert(std::end(st_terms), std::begin(tmp), std::end(tmp));
        }

        // ijk
        for (int i = 0; i < dim; i++) {
            for (int j = i + 1; j < dim; j++) {
                for (int k = j + 1; k < dim; k++) {
                    std::vector<pq_operator_terms> tmp = get_triple_commutator_terms(factor, targets, {ops[i]}, {ops[j]}, {ops[k]});
                    st_terms.insert(std::end(st_terms), std::begin(tmp), std::end(tmp));
                }
            }
        }

        // ijj
        for (int i = 0; i < dim; i++) {
            for (int j = i + 1; j < dim; j++) {
                std::vector<pq_operator_terms> tmp1 = get_triple_commutator_terms(0.5 * factor, targets, {ops[i]}, {ops[j]}, {ops[j]});
                std::vector<pq_operator_terms> tmp2 = get_triple_commutator_terms(0.5 * factor, targets, {ops[i]}, {ops[i]}, {ops[j]});

                st_terms.insert(std::end(st_terms), std::begin(tmp1), std::end(tmp1));
                st_terms.insert(std::end(st_terms), std::begin(tmp2), std::end(tmp2));
            }
        }

         // iii
        for (int i = 0; i < dim; i++) {
            std::vector<pq_operator_terms> tmp = get_triple_commutator_terms(1.0 / 6.0 * factor, targets, {ops[i]}, {ops[i]}, {ops[i]});
            st_terms.insert(std::end(st_terms), std::begin(tmp), std::end(tmp));
        }

        // ijkl
        for (int i = 0; i < dim; i++) {
            for (int j = i + 1; j < dim; j++) {
                for (int k = j + 1; k < dim; k++) {
                    for (int l = k + 1; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quadruple_commutator_terms(factor, targets, {ops[i]}, {ops[j]}, {ops[k]}, {ops[l]});
                        st_terms.insert(std::end(st_terms), std::begin(tmp), std::end(tmp));
                    }
                }
            }
        }

        // ijkk
        for (int i = 0; i < dim; i++) {
            for (int j = i + 1; j < dim; j++) {
                for (int k = j + 1; k < dim; k++) {
                    std::vector<pq_operator_terms> tmp1 = get_quadruple_commutator_terms(0.5 * factor, targets, {ops[i]}, {ops[j]}, {ops[k]}, {ops[k]});
                    std::vector<pq_operator_terms> tmp2 = get_quadruple_commutator_terms(0.5 * factor, targets, {ops[i]}, {ops[j]}, {ops[j]}, {ops[k]});
                    std::vector<pq_operator_terms> tmp3 = get_quadruple_commutator_terms(0.5 * factor, targets, {ops[i]}, {ops[i]}, {ops[j]}, {ops[k]});

                    st_terms.insert(std::end(st_terms), std::begin(tmp1), std::end(tmp1));
                    st_terms.insert(std::end(st_terms), std::begin(tmp2), std::end(tmp2));
                    st_terms.insert(std::end(st_terms), std::begin(tmp3), std::end(tmp3));
                }
            }
        }

        // iijj
        for (int i = 0; i < dim; i++) {
            for (int j = i + 1; j < dim; j++) {
                std::vector<pq_operator_terms> tmp = get_quadruple_commutator_terms(0.25 * factor, targets, {ops[i]}, {ops[i]}, {ops[j]}, {ops[j]});
                st_terms.insert(std::end(st_terms), std::begin(tmp), std::end(tmp));
            }
        }

        // iiij
        for (int i = 0; i < dim; i++) {
            for (int j = i + 1; j < dim; j++) {
                std::vector<pq_operator_terms> tmp1 = get_quadruple_commutator_terms(1.0 / 6.0 * factor, targets, {ops[i]}, {ops[i]}, {ops[i]}, {ops[j]});
                std::vector<pq_operator_terms> tmp2 = get_quadruple_commutator_terms(1.0 / 6.0 * factor, targets, {ops[i]}, {ops[j]}, {ops[j]}, {ops[j]});

                st_terms.insert(std::end(st_terms), std::begin(tmp1), std::end(tmp1));
                st_terms.insert(std::end(st_terms), std::begin(tmp2), std::end(tmp2));
            }
        }

        // iiii
        for (int i = 0; i < dim; i++) {
            std::vector<pq_operator_terms> tmp = get_quadruple_commutator_terms(1.0 / 24.0 * factor, targets, {ops[i]}, {ops[i]}, {ops[i]}, {ops[i]});
            st_terms.insert(std::end(st_terms), std::begin(tmp), std::end(tmp));
        }

    }else {

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                std::vector<pq_operator_terms> tmp = get_double_commutator_terms(0.5 * factor, targets, {ops[i]}, {ops[j]});
                st_terms.insert(std::end(st_terms), std::begin(tmp), std::end(tmp));
            }
        }

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    std::vector<pq_operator_terms> tmp = get_triple_commutator_terms(1.0 / 6.0 * factor, targets, {ops[i]}, {ops[j]}, {ops[k]});
                    st_terms.insert(std::end(st_terms), std::begin(tmp), std::end(tmp));
                }
            }
        }

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quadruple_commutator_terms(1.0 / 24.0 * factor, targets, {ops[i]}, {ops[j]}, {ops[k]}, {ops[l]});
                        st_terms.insert(std::end(st_terms), std::begin(tmp), std::end(tmp));
                    }
                }
            }
        }
    }

    return st_terms;
}

void pq_helper::add_bernoulli_operator(double factor,
                                       const std::vector<std::string> &targets,
                                       const std::vector<std::string> &ops,
                                       const int max_order) {

    std::vector<pq_operator_terms> bernoulli_terms = get_bernoulli_operator_terms(factor, targets, ops, max_order);
    for (auto term : bernoulli_terms){
        add_operator_product(term.factor, term.operators);
    }
}

std::vector<pq_operator_terms> pq_helper::get_bernoulli_operator_terms(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops, const int max_order) {

    if ( max_order > 5 ) {
        printf("\n");
        printf("    error: Bernoulli terms beyond 5th-order are not yet implemented.\n");
        printf("\n");
        exit(1);
    }

    int dim = (int)ops.size();

    std::vector<pq_operator_terms> bernoulli_terms;

    // zeroth-order terms: v
    bernoulli_terms.push_back(pq_operator_terms(factor, targets));

    if ( max_order == 0 ) {
        return bernoulli_terms;
    }

    // first-order terms: 1/2 [v, sigma] + 1/2 [v_R, sigma]

    std::vector<pq_operator_terms> tmp = get_bernoulli_operator_terms_1(factor, targets, ops);
    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));

    if ( max_order == 1 ) {
        return bernoulli_terms;
    }

    // second-order terms: 1/12 [[V_N, sigma], sigma] + 1/4 [[V, sigma]_R, sigma] + 1/4 [[V_R, sigma]_R, sigma]
    tmp.clear();
    tmp = get_bernoulli_operator_terms_2(factor, targets, ops);
    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));

    if ( max_order == 2 ) {
        return bernoulli_terms;
    }

    // third-order terms: 
    tmp.clear();
    tmp = get_bernoulli_operator_terms_3(factor, targets, ops);
    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));

    if ( max_order == 3 ) {
        return bernoulli_terms;
    }

    // fourth-order terms: 
    tmp.clear();
    tmp = get_bernoulli_operator_terms_4(factor, targets, ops);
    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));


    if ( max_order == 4 ) {
        return bernoulli_terms;
    }

    // fifth-order terms: 
    tmp.clear();
    tmp = get_bernoulli_operator_terms_5(factor, targets, ops);
    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));

    if ( max_order == 5 ) {
        return bernoulli_terms;
    }

    // sixth-order terms: 
    tmp.clear();
    tmp = get_bernoulli_operator_terms_6(factor, targets, ops);
    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));

    return bernoulli_terms;
}

// first-order bernoulli terms: 1/2 [v, sigma] + 1/2 [v_R, sigma]
std::vector<pq_operator_terms> pq_helper::get_bernoulli_operator_terms_1(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops) {

    std::vector<pq_operator_terms> bernoulli_terms;

    // mutable copies of targets and ops
    std::vector<std::string> b_targets;
    std::vector<std::string> b_ops;

    for (auto target: targets){
        b_targets.push_back(target + "{A,A}");
    }

    for (auto op: ops){
        b_ops.push_back(op + "{A,A}");
    }

    int dim = (int)ops.size();

    for (int i = 0; i < dim; i++) {
        std::vector<pq_operator_terms> tmp = get_commutator_terms(0.5 * factor, b_targets, {b_ops[i]});
        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
    }

    b_targets.clear();
    for (auto target: targets){
        b_targets.push_back(target + "{R,A}");
    }

    b_ops.clear();
    for (auto op: ops){
        b_ops.push_back(op + "{A,A}");
    }

    for (int i = 0; i < dim; i++) {
        std::vector<pq_operator_terms> tmp = get_commutator_terms(0.5 * factor, b_targets, {b_ops[i]});
        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
    }

    return bernoulli_terms;
}

// second-order bernoulli terms: 1/12 [[V_N, sigma], sigma] + 1/4 [[V, sigma]_R, sigma] + 1/4 [[V_R, sigma]_R, sigma]
std::vector<pq_operator_terms> pq_helper::get_bernoulli_operator_terms_2(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops) {

    std::vector<pq_operator_terms> bernoulli_terms;

    // mutable copies of targets and ops
    std::vector<std::string> b_targets;
    std::vector<std::string> b_ops1;
    std::vector<std::string> b_ops2;

    // 1/12 [[V_N, sigma], sigma]
    for (auto target: targets){
        b_targets.push_back(target + "{N,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,A}");
        b_ops2.push_back(op + "{A,A,A}");
    }

    int dim = (int)ops.size();

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            std::vector<pq_operator_terms> tmp = get_double_commutator_terms(1.0 / 12.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]});
            bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        }
    }

    // 1/4 [[V, sigma]_R, sigma]
    b_targets.clear();
    for (auto target: targets){
        b_targets.push_back(target + "{A,R,A}");
    }

    b_ops1.clear();
    b_ops2.clear();
    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A}");
        b_ops2.push_back(op + "{A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            std::vector<pq_operator_terms> tmp = get_double_commutator_terms(1.0 / 4.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]});
            bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        }
    }

    // 1/4 [[V_R, sigma]_R, sigma]
    b_targets.clear();
    for (auto target: targets){
        b_targets.push_back(target + "{R,R,A}");
    }

    b_ops1.clear();
    b_ops2.clear();
    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A}");
        b_ops2.push_back(op + "{A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            std::vector<pq_operator_terms> tmp = get_double_commutator_terms(1.0 / 4.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]});
            bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        }
    }

    return bernoulli_terms;
}

// third-order bernoulli terms
std::vector<pq_operator_terms> pq_helper::get_bernoulli_operator_terms_3(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops) {

    std::vector<pq_operator_terms> bernoulli_terms;

    // mutable copies of targets and ops
    std::vector<std::string> b_targets;
    std::vector<std::string> b_ops1;
    std::vector<std::string> b_ops2;
    std::vector<std::string> b_ops3;

    // 1/24 [[[V_N, sigma], sigma]_R, sigma]

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,R,A}");
        b_ops2.push_back(op + "{A,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,A}");
    }

    int dim = (int)ops.size();

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                std::vector<pq_operator_terms> tmp = get_triple_commutator_terms(1.0 / 24.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]});
                bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
            }
        }
    }

    // 1/8 [[[V_R, sigma]_R, sigma]_R, sigma]

    b_targets.clear();
    for (auto target: targets){
        b_targets.push_back(target + "{R,R,R,A}");
    }

    b_ops1.clear();
    b_ops2.clear();
    b_ops3.clear();
    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                std::vector<pq_operator_terms> tmp = get_triple_commutator_terms(1.0 / 8.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]});
                bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
            }
        }
    }

    // 1/8 [[[V, sigma]_R, sigma]_R, sigma]

    b_targets.clear();
    for (auto target: targets){
        b_targets.push_back(target + "{A,R,R,A}");
    }

    b_ops1.clear();
    b_ops2.clear();
    b_ops3.clear();
    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                std::vector<pq_operator_terms> tmp = get_triple_commutator_terms(1.0 / 8.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]});
                bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
            }
        }
    }

    // -1/24 [[[V, sigma]_R, sigma], sigma]

    b_targets.clear();
    for (auto target: targets){
        b_targets.push_back(target + "{A,R,A,A}");
    }

    b_ops1.clear();
    b_ops2.clear();
    b_ops3.clear();
    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,A}");
        b_ops2.push_back(op + "{A,A,A,A}");
        b_ops3.push_back(op + "{A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                std::vector<pq_operator_terms> tmp = get_triple_commutator_terms(-1.0 / 24.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]});
                bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
            }
        }
    }

    // -1/24 [[[V_R, sigma]_R, sigma], sigma]

    b_targets.clear();
    for (auto target: targets){
        b_targets.push_back(target + "{R,R,A,A}");
    }

    b_ops1.clear();
    b_ops2.clear();
    b_ops3.clear();
    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,A}");
        b_ops2.push_back(op + "{A,A,A,A}");
        b_ops3.push_back(op + "{A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                std::vector<pq_operator_terms> tmp = get_triple_commutator_terms(-1.0 / 24.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]});
                bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
            }
        }
    }

    return bernoulli_terms;
}

// fourth-order bernoulli terms
std::vector<pq_operator_terms> pq_helper::get_bernoulli_operator_terms_4(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops) {

    std::vector<pq_operator_terms> bernoulli_terms;

    // mutable copies of targets and ops
    std::vector<std::string> b_targets;
    std::vector<std::string> b_ops1;
    std::vector<std::string> b_ops2;
    std::vector<std::string> b_ops3;
    std::vector<std::string> b_ops4;


    // 1/16 [[[[V_R, sigma]_R, sigma]_R, sigma]_R, sigma]

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,A}");
    }

    int dim = (int)ops.size();

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    std::vector<pq_operator_terms> tmp = get_quadruple_commutator_terms(1.0 / 16.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]});
                    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
                }
            }
        }
    }

    // 1/16 [[[[V, sigma]_R, sigma]_R, sigma]_R, sigma]

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    std::vector<pq_operator_terms> tmp = get_quadruple_commutator_terms(1.0 / 16.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]});
                    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
                }
            }
        }
    }

    // 1/48 [[[[V_N, sigma], sigma]_R, sigma]_R, sigma]

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    std::vector<pq_operator_terms> tmp = get_quadruple_commutator_terms(1.0 / 48.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]});
                    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
                }
            }
        }
    }


    // -1/48 [[[[V, sigma]_R, sigma]_R, sigma]_R, sigma]

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    std::vector<pq_operator_terms> tmp = get_quadruple_commutator_terms(-1.0 / 48.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]});
                    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
                }
            }
        }
    }


    // -1/48 [[[[V_R, sigma]_R, sigma], sigma]_R, sigma]

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,A,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,R,A}");
        b_ops2.push_back(op + "{A,A,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    std::vector<pq_operator_terms> tmp = get_quadruple_commutator_terms(-1.0 / 48.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]});
                    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
                }
            }
        }
    }


    // -1/144 [[[[V_N, sigma], sigma]_R, sigma], sigma]

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,R,A,A}");
        b_ops2.push_back(op + "{A,A,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    std::vector<pq_operator_terms> tmp = get_quadruple_commutator_terms(-1.0 / 144.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]});
                    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
                }
            }
        }
    }


    // -1/48 [[[[V, sigma]_R, sigma]_R, sigma], sigma]

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,A,A}");
        b_ops2.push_back(op + "{A,A,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    std::vector<pq_operator_terms> tmp = get_quadruple_commutator_terms(-1.0 / 48.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]});
                    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
                }
            }
        }
    }


    // -1/48 [[[[V_R, sigma]_R, sigma]_R, sigma], sigma]

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,A,A}");
        b_ops2.push_back(op + "{A,A,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    std::vector<pq_operator_terms> tmp = get_quadruple_commutator_terms(-1.0 / 48.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]});
                    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
                }
            }
        }
    }


    // -1/720 [[[[V_N, sigma], sigma], sigma], sigma]

    for (auto target: targets){
        b_targets.push_back(target + "{A,A,A,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,A,A,A}");
        b_ops2.push_back(op + "{A,A,A,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    std::vector<pq_operator_terms> tmp = get_quadruple_commutator_terms(-1.0 / 720.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]});
                    bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
                }
            }
        }
    }

    return bernoulli_terms;
}

// fifth-order bernoulli terms
std::vector<pq_operator_terms> pq_helper::get_bernoulli_operator_terms_5(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops) {

    std::vector<pq_operator_terms> bernoulli_terms;

    // mutable copies of targets and ops
    std::vector<std::string> b_targets;
    std::vector<std::string> b_ops1;
    std::vector<std::string> b_ops2;
    std::vector<std::string> b_ops3;
    std::vector<std::string> b_ops4;
    std::vector<std::string> b_ops5;

    //  1/32   [[[[[V_A, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,R,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    int dim = (int)ops.size();

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(1.0 / 32.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }


    //  1/32   [[[[[V_R, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,R,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(1.0 / 32.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }


    // -1/96   [[[[[V_A, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,A,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,R,R,A}");
        b_ops2.push_back(op + "{A,A,A,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(-1.0 / 96.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }



    // -1/96   [[[[[V_R, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,A,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,R,R,A}");
        b_ops2.push_back(op + "{A,A,A,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(-1.0 / 96.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }



    // -1/96   [[[[[V_A, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A 

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,R,A,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,A,R,A}");
        b_ops2.push_back(op + "{A,A,R,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(-1.0 / 96.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }



    // -1/96   [[[[[V_R, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A 

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,R,A,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,A,R,A}");
        b_ops2.push_back(op + "{A,A,R,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(-1.0 / 96.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }



    // -1/96   [[[[[V_A, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,R,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,R,A,A}");
        b_ops2.push_back(op + "{A,A,R,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,R,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(-1.0 / 96.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }



    // -1/96   [[[[[V_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,R,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(-1.0 / 96.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }



    //  1/288  [[[[[V_A, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,A,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,R,A,A}");
        b_ops2.push_back(op + "{A,A,A,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,R,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(1.0 / 288.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }



    //  1/288  [[[[[V_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,A,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,R,A,A}");
        b_ops2.push_back(op + "{A,A,A,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,R,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(1.0 / 288.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }



    //  1/1440 [[[[[V_A, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,A,A,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,A,A,A}");
        b_ops2.push_back(op + "{A,A,A,A,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(1.0 / 1440.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }



    //  1/1440 [[[[[V_R, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,A,A,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,A,A,A}");
        b_ops2.push_back(op + "{A,A,A,A,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(1.0 / 1440.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }



    // -1/1440 [[[[[V_N, sigma]_A, sigma]_A, sigma]_A, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,A,A,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,A,A,R,A}");
        b_ops2.push_back(op + "{A,A,A,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(-1.0 / 1440.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }



    //  1/96   [[[[[V_N, sigma]_A, sigma]_R, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,R,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,R,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(1.0 / 96.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }



    // -1/288  [[[[[V_N, sigma]_A, sigma]_R, sigma]_A, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,R,A,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,R,A,R,A}");
        b_ops2.push_back(op + "{A,A,R,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(-1.0 / 288.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }



    // -1/288  [[[[[V_N, sigma]_A, sigma]_R, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,R,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,R,R,A,A}");
        b_ops2.push_back(op + "{A,A,R,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,R,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_quintuple_commutator_terms(-1.0 / 288.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
		    }
                }
            }
        }
    }




    return bernoulli_terms;
}

// sixth-order bernoulli terms
std::vector<pq_operator_terms> pq_helper::get_bernoulli_operator_terms_6(double factor, const std::vector<std::string> &targets,const std::vector<std::string> &ops) {

    std::vector<pq_operator_terms> bernoulli_terms;

    // mutable copies of targets and ops
    std::vector<std::string> b_targets;
    std::vector<std::string> b_ops1;
    std::vector<std::string> b_ops2;
    std::vector<std::string> b_ops3;
    std::vector<std::string> b_ops4;
    std::vector<std::string> b_ops5;
    std::vector<std::string> b_ops6;


    //     1/64    [[[[[[V_A, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,R,R,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,R,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    int dim = (int)ops.size();

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 64.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/64    [[[[[[V_R, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,R,R,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,R,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 64.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //    -1/192   [[[[[[V_A, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,A,R,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,R,R,R,A}");
        b_ops2.push_back(op + "{A,A,A,R,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(-1.0 / 192.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //    -1/192   [[[[[[V_R, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,A,R,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,R,R,R,A}");
        b_ops2.push_back(op + "{A,A,A,R,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(-1.0 / 192.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //    -1/192   [[[[[[V_A, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,R,A,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,A,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,A,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(-1.0 / 192.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //    -1/192   [[[[[[V_R, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,R,A,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,A,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,A,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(-1.0 / 192.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //    -1/192   [[[[[[V_A, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,R,R,A,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,R,A,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(-1.0 / 192.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //    -1/192   [[[[[[V_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,R,R,A,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,R,A,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(-1.0 / 192.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/576   [[[[[[V_A, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,A,R,A,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,R,A,R,A}");
        b_ops2.push_back(op + "{A,A,A,R,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 576.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/576   [[[[[[V_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,A,R,A,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,R,A,R,A}");
        b_ops2.push_back(op + "{A,A,A,R,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 576.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/2880  [[[[[[V_A, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,A,A,A,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,A,A,R,A}");
        b_ops2.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 2880.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/2880  [[[[[[V_R, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,A,A,A,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,A,A,R,A}");
        b_ops2.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 2880.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //    -1/192   [[[[[[V_A, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,R,R,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,R,R,A,A}");
        b_ops2.push_back(op + "{A,A,R,R,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(-1.0 / 192.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //    -1/192   [[[[[[V_R, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,R,R,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,R,R,A,A}");
        b_ops2.push_back(op + "{A,A,R,R,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(-1.0 / 192.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/576   [[[[[[V_A, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,A,R,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,R,R,A,A}");
        b_ops2.push_back(op + "{A,A,A,R,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 576.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/576   [[[[[[V_R, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,A,R,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,A,R,R,A,A}");
        b_ops2.push_back(op + "{A,A,A,R,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 576.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/576   [[[[[[V_A, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,R,A,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,A,R,A,A}");
        b_ops2.push_back(op + "{A,A,R,A,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,R,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 576.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/576   [[[[[[V_R, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,R,A,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,A,R,A,A}");
        b_ops2.push_back(op + "{A,A,R,A,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,R,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 576.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/2880  [[[[[[V_A, sigma]_R, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{A,R,R,A,A,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,A,A,A,A}");
        b_ops2.push_back(op + "{A,A,R,A,A,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 2880.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/2880  [[[[[[V_R, sigma]_R, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{R,R,R,A,A,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,R,R,A,A,A,A}");
        b_ops2.push_back(op + "{A,A,R,A,A,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 2880.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/30240 [[[[[[V_N, sigma]_A, sigma]_A, sigma]_A, sigma]_A, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,A,A,A,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops2.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 30240.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //    -1/2880  [[[[[[V_N, sigma]_A, sigma]_A, sigma]_A, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,A,A,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops2.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 2880.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/192   [[[[[[V_N, sigma]_A, sigma]_R, sigma]_R, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,R,R,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,R,R,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 192.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //    -1/576   [[[[[[V_N, sigma]_A, sigma]_R, sigma]_A, sigma]_R, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,R,A,R,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,R,A,R,R,A}");
        b_ops2.push_back(op + "{A,A,R,A,R,R,A}");
        b_ops3.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(-1.0 / 576.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //    -1/576   [[[[[[V_N, sigma]_A, sigma]_R, sigma]_R, sigma]_A, sigma]_R, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,R,R,A,R,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,R,R,A,R,A}");
        b_ops2.push_back(op + "{A,A,R,R,A,R,A}");
        b_ops3.push_back(op + "{A,A,A,R,A,R,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,R,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(-1.0 / 576.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/8640  [[[[[[V_N, sigma]_A, sigma]_A, sigma]_A, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,A,A,A,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops2.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 8640.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //    -1/576   [[[[[[V_N, sigma]_A, sigma]_R, sigma]_R, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,R,R,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,R,R,R,A,A}");
        b_ops2.push_back(op + "{A,A,R,R,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,R,R,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(-1.0 / 576.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/1728  [[[[[[V_N, sigma]_A, sigma]_R, sigma]_A, sigma]_R, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,R,A,R,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,R,A,R,A,A}");
        b_ops2.push_back(op + "{A,A,R,A,R,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,R,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,R,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 1728.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }


    //     1/8640  [[[[[[V_N, sigma]_A, sigma]_R, sigma]_A, sigma]_A, sigma]_A, sigma]_A

    for (auto target: targets){
        b_targets.push_back(target + "{N,A,R,A,A,A,A}");
    }

    for (auto op: ops){
        b_ops1.push_back(op + "{A,A,R,A,A,A,A}");
        b_ops2.push_back(op + "{A,A,R,A,A,A,A}");
        b_ops3.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops4.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops5.push_back(op + "{A,A,A,A,A,A,A}");
        b_ops6.push_back(op + "{A,A,A,A,A,A,A}");
    }

    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            for (int k = 0; k < dim; k++) {
                for (int l = 0; l < dim; l++) {
                    for (int l = 0; l < dim; l++) {
                        std::vector<pq_operator_terms> tmp = get_hextuple_commutator_terms(1.0 / 8640.0 * factor, b_targets, {b_ops1[i]}, {b_ops2[j]}, {b_ops3[k]}, {b_ops4[l]}, {b_ops5[l]}, {b_ops6[l]});
                        bernoulli_terms.insert(std::end(bernoulli_terms), std::begin(tmp), std::end(tmp));
        	    }
                }
            }
        }
    }



    return bernoulli_terms;
}

} // End namespaces

#endif
