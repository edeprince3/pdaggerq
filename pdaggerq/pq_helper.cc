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


#include "pq_helper.h"
#include "pq_utils.h"
#include "pq_string.h"
#include "pq_tensor.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <omp.h>

std::map<std::string, std::string> empty_spin_labels(){
    std::map<std::string, std::string> ret;
    return ret;
}

namespace py = pybind11;
using namespace pybind11::literals;

namespace pdaggerq {

void export_pq_helper(py::module& m) {
    py::class_<pdaggerq::pq_helper, std::shared_ptr<pdaggerq::pq_helper> >(m, "pq_helper")
        .def(py::init< std::string >())
        .def("set_print_level", &pq_helper::set_print_level)
        .def("set_nthreads", [](pq_helper& self, int nthreads) {
                self.nthreads = nthreads;
            })
        .def("set_left_operators", &pq_helper::set_left_operators)
        .def("set_right_operators", &pq_helper::set_right_operators)
        .def("set_left_operators_type", &pq_helper::set_left_operators_type)
        .def("set_right_operators_type", &pq_helper::set_right_operators_type)
        .def("set_cluster_operators_commute", &pq_helper::set_cluster_operators_commute)
        .def("set_find_paired_permutations", &pq_helper::set_find_paired_permutations)
        .def("simplify", &pq_helper::simplify)
        .def("clear", &pq_helper::clear)
        .def("print",
             [](pq_helper& self, const std::string& string_type) {
                 return self.print(string_type);
             },
             py::arg("string_type") = "fully-contracted" )
        .def("strings", &pq_helper::strings)
        .def("fully_contracted_strings", &pq_helper::fully_contracted_strings)
        .def("fully_contracted_strings_with_spin",
             [](pq_helper& self, std::unordered_map<std::string, std::string> &spin_labels) {
                 return self.fully_contracted_strings_with_spin(spin_labels);
             },
             py::arg("spin_labels") = empty_spin_labels() )

        .def("add_st_operator", &pq_helper::add_st_operator)
        .def("add_commutator", &pq_helper::add_commutator)
        .def("add_double_commutator", &pq_helper::add_double_commutator)
        .def("add_triple_commutator", &pq_helper::add_triple_commutator)
        .def("add_quadruple_commutator", &pq_helper::add_quadruple_commutator)
        .def("add_operator_product", &pq_helper::add_operator_product);
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

    print_level = 0;

    // assume operators entering a similarity transformation
    // commute. only relevant for the add_st_operator() function
    cluster_operators_commute = true;

    // by default, do not look for paired permutations (until parsers catch up)
    find_paired_permutations = false;

    /// right operators type (EE, IP, EA)
    right_operators_type = "EE";

    /// left operators type (EE, IP, EA)
    left_operators_type = "EE";

}

void pq_helper::set_find_paired_permutations(bool do_find_paired_permutations) {
    find_paired_permutations = do_find_paired_permutations;
}

void pq_helper::set_print_level(int level) {
    print_level = level;
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

// do operators entering similarity transformation commute? default true
void pq_helper::set_cluster_operators_commute(bool do_cluster_operators_commute) {
    cluster_operators_commute = do_cluster_operators_commute;
}

void pq_helper::add_commutator(double factor,
                               const std::vector<std::string> &op0,
                               const std::vector<std::string> &op1){

    add_operator_product( factor, concatinate_operators({op0, op1}) );
    add_operator_product(-factor, concatinate_operators({op1, op0}) );

}

void pq_helper::add_double_commutator(double factor,
                                        const std::vector<std::string> &op0,
                                        const std::vector<std::string> &op1,
                                        const std::vector<std::string> &op2){

    add_operator_product( factor, concatinate_operators({op0, op1, op2}) );
    add_operator_product(-factor, concatinate_operators({op1, op0, op2}) );
    add_operator_product(-factor, concatinate_operators({op2, op0, op1}) );
    add_operator_product( factor, concatinate_operators({op2, op1, op0}) );

}

void pq_helper::add_triple_commutator(double factor,
                                        const std::vector<std::string> &op0,
                                        const std::vector<std::string> &op1,
                                        const std::vector<std::string> &op2,
                                        const std::vector<std::string> &op3){

    add_operator_product( factor, concatinate_operators({op0, op1, op2, op3}) );
    add_operator_product(-factor, concatinate_operators({op1, op0, op2, op3}) );
    add_operator_product(-factor, concatinate_operators({op2, op0, op1, op3}) );
    add_operator_product( factor, concatinate_operators({op2, op1, op0, op3}) );
    add_operator_product(-factor, concatinate_operators({op3, op0, op1, op2}) );
    add_operator_product( factor, concatinate_operators({op3, op1, op0, op2}) );
    add_operator_product( factor, concatinate_operators({op3, op2, op0, op1}) );
    add_operator_product(-factor, concatinate_operators({op3, op2, op1, op0}) );

}

void pq_helper::add_quadruple_commutator(double factor,
                                           const std::vector<std::string> &op0,
                                           const std::vector<std::string> &op1,
                                           const std::vector<std::string> &op2,
                                           const std::vector<std::string> &op3,
                                           const std::vector<std::string> &op4){


    add_operator_product( factor, concatinate_operators({op0, op1, op2, op3, op4}) );
    add_operator_product(-factor, concatinate_operators({op1, op0, op2, op3, op4}) );
    add_operator_product(-factor, concatinate_operators({op2, op0, op1, op3, op4}) );
    add_operator_product( factor, concatinate_operators({op2, op1, op0, op3, op4}) );
    add_operator_product(-factor, concatinate_operators({op3, op0, op1, op2, op4}) );
    add_operator_product( factor, concatinate_operators({op3, op1, op0, op2, op4}) );
    add_operator_product( factor, concatinate_operators({op3, op2, op0, op1, op4}) );
    add_operator_product(-factor, concatinate_operators({op3, op2, op1, op0, op4}) );
    add_operator_product(-factor, concatinate_operators({op4, op0, op1, op2, op3}) );
    add_operator_product( factor, concatinate_operators({op4, op1, op0, op2, op3}) );
    add_operator_product( factor, concatinate_operators({op4, op2, op0, op1, op3}) );
    add_operator_product(-factor, concatinate_operators({op4, op2, op1, op0, op3}) );
    add_operator_product( factor, concatinate_operators({op4, op3, op0, op1, op2}) );
    add_operator_product(-factor, concatinate_operators({op4, op3, op1, op0, op2}) );
    add_operator_product(-factor, concatinate_operators({op4, op3, op2, op0, op1}) );
    add_operator_product( factor, concatinate_operators({op4, op3, op2, op1, op0}) );

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
            if (op == "v" ) {

                printf("\n");
                printf("    error: the fluctuation potential cannot appear in operators defining the bra state\n");
                printf("\n");
                exit(1);

                tmp.emplace_back("j1");
                tmp.emplace_back("j2");
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
            if (op == "v" ) {

                printf("\n");
                printf("    error: the fluctuation potential cannot appear in operators defining the ket state\n");
                printf("\n");
                exit(1);

                tmp.emplace_back("j1");
                tmp.emplace_back("j2");
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
        if (op == "v" ) {
            found_v = true;
            break;
        }else {
            tmp_in.push_back(op);
            count++;
        }
    }
    if ( found_v ) {

        // term 1
        tmp_in.emplace_back("j1");
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
        in.emplace_back("j2");
        for (int i = count + 1; i < (int)tmp_in.size(); i++) {
            in.push_back(tmp_in[i]);
        }
        add_operator_product(factor, in);
        
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

    // set number of threads to use
    omp_set_num_threads(nthreads);
#pragma omp parallel for schedule(dynamic,1) default(none) \
shared(original_factor, factor, save, left_operators, right_operators, vacuum) firstprivate(in)
    for (std::vector<std::string> & left_operator : left_operators) {
        for (std::vector<std::string> & right_operator : right_operators) {

            std::shared_ptr<pq_string> newguy (new pq_string(vacuum));

            factor = original_factor;

            std::vector<std::string> tmp_string;

            bool has_w0       = false;

            // stupid design choice ... o1-o4 and v1-v4 are already used
            int occ_label_count = 5;
            int vir_label_count = 5;
            int gen_label_count = 0;

            // apply any extra operators on left or right:
            std::vector<std::string> tmp = left_operator;
            for (const std::string & op : save) {
                tmp.push_back(op);
            }
            for (const std::string & op : right_operator) {
                tmp.push_back(op);
            }

            for (std::string & op : tmp) {

                // blank string
                if ( op.empty() ) continue;

                // lowercase indices
                std::transform(op.begin(), op.end(), op.begin(), [](unsigned char c){ return std::tolower(c); });

                // remove parentheses
                removeParentheses(op);

                if (op.substr(0, 1) == "h" ) { // one-electron operator

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // integrals
                    newguy->set_integrals("core", {idx1, idx2});

                }else if (op.substr(0, 1) == "f" ) { // fock operator

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // integrals
                    newguy->set_integrals("fock", {idx1, idx2});

                }else if (op.substr(0, 2) == "d+" ) { // one-electron operator (dipole + boson creator)

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // integrals
                    newguy->set_integrals("d+", {idx1, idx2});

                    // boson operator
                    newguy->is_boson_dagger.push_back(true);

                }else if (op.substr(0, 2) == "d-" ) { // one-electron operator (dipole + boson annihilator)

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // integrals
                    newguy->set_integrals("d-", {idx1, idx2});

                    // boson operator
                    newguy->is_boson_dagger.push_back(false);

                }else if (op.substr(0, 1) == "g" ) { // general two-electron operator

                    //factor *= 0.25;

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);
                    std::string idx3 = "p" + std::to_string(gen_label_count++);
                    std::string idx4 = "p" + std::to_string(gen_label_count++);

                    tmp_string.push_back(idx1+"*");
                    tmp_string.push_back(idx2+"*");
                    tmp_string.push_back(idx3);
                    tmp_string.push_back(idx4);

                    newguy->set_integrals("two_body", {idx1, idx2, idx4, idx3});

                }else if (op.substr(0, 1) == "j" ) { // fluctuation potential

                    if (op.substr(1, 1) == "1" ){

                        factor *= -1.0;

                        std::string idx1 = "p" + std::to_string(gen_label_count++);
                        std::string idx2 = "p" + std::to_string(gen_label_count++);

                        // index 1
                        tmp_string.push_back(idx1+"*");

                        // index 2
                        tmp_string.push_back(idx2);

                        // integrals
                        newguy->set_integrals("occ_repulsion", {idx1, idx2});

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

                        newguy->set_integrals("eri", {idx1, idx2, idx4, idx3});

                    }

                }else if (op.substr(0, 1) == "t" ){

                    int n = std::stoi(op.substr(1));

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
                    // op*j*...
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
                    newguy->set_amplitudes('t', n, labels);

                    // factor = 1/(n!)^2
                    double my_factor = 1.0;
                    for (int id = 0; id < n; id++) {
                        my_factor *= (id+1);
                    }
                    factor *= 1.0 / my_factor / my_factor;

                }else if (op.substr(0, 1) == "w" ){ // w0 B*B

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

                }else if (op.substr(0, 2) == "b+" ){ // B*

                        newguy->is_boson_dagger.push_back(true);

                }else if (op.substr(0, 2) == "b-" ){ // B

                        newguy->is_boson_dagger.push_back(false);

                }else if (op.substr(0, 1) == "u" ){ // t-amplitudes + boson creator

                    int n = std::stoi(op.substr(1));

                    if ( n == 0 ){

                        std::vector<std::string> labels;
                        newguy->set_amplitudes('u', n, labels);

                        newguy->is_boson_dagger.push_back(true);

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
                        // op*j*...
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
                        newguy->set_amplitudes('u', n, labels);
                        
                        // factor = 1/(n!)^2
                        double my_factor = 1.0;
                        for (int id = 0; id < n; id++) {
                            my_factor *= (id+1);
                        }
                        factor *= 1.0 / my_factor / my_factor;

                        newguy->is_boson_dagger.push_back(true);
                    }

                }else if (op.substr(0, 1) == "r" ){


                    int n = std::stoi(op.substr(1));

                    if ( n == 0 ){

                        std::vector<std::string> labels;
                        newguy->set_amplitudes('r', n, labels);

                    }else {

                        int n_annihilate = n;
                        int n_create     = n;
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
                        std::vector<std::string> labels;
                        // tn(ab...
                        for (int id = 0; id < n_create; id++) {
                            labels.push_back(label_left[id]);
                        }
                        // tn(ab......ji)
                        for (int id = n_annihilate-1; id >= 0; id--) {
                            labels.push_back(label_right[id]);
                        }
                        newguy->set_amplitudes('r', n, labels);

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

                }else if (op.substr(0, 1) == "s" ){ // r amplitudes + boson creator

                    int n = std::stoi(op.substr(1));

                    if ( n == 0 ){

                        std::vector<std::string> labels;
                        newguy->set_amplitudes('s', n, labels);

                        newguy->is_boson_dagger.push_back(true);

                    }else {
                       
                        int n_annihilate = n;
                        int n_create     = n;
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
                        std::vector<std::string> labels;
                        // tn(ab... 
                        for (int id = 0; id < n_create; id++) {
                            labels.push_back(label_left[id]);
                        }
                        // tn(ab......ji)
                        for (int id = n_annihilate-1; id >= 0; id--) {
                            labels.push_back(label_right[id]);
                        } 
                        newguy->set_amplitudes('s', n, labels);
                        
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
                    
                        newguy->is_boson_dagger.push_back(true);

                    }

                }else if (op.substr(0, 1) == "l" ){

                    int n = std::stoi(op.substr(1));

                    if ( n == 0 ){

                        std::vector<std::string> labels;
                        newguy->set_amplitudes('l', n, labels);

                    }else {
                        
                        int n_annihilate = n;
                        int n_create     = n;
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
                        std::vector<std::string> labels;
                        // tn(ij... 
                        for (int id = 0; id < n_create; id++) {
                            labels.push_back(label_left[id]);
                        }
                        // tn(ij......ba)
                        for (int id = n_annihilate-1; id >= 0; id--) {
                            labels.push_back(label_right[id]);
                        }
                        newguy->set_amplitudes('l', n, labels);
                        
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

                }else if (op.substr(0, 1) == "m" ){ // l amplitudes plus boson annihilator

                    int n = std::stoi(op.substr(1));

                    if ( n == 0 ){

                        std::vector<std::string> labels;
                        newguy->set_amplitudes('m', n, labels);

                        newguy->is_boson_dagger.push_back(false);

                    }else {

                        int n_annihilate = n;
                        int n_create     = n;
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
                        std::vector<std::string> labels;
                        // tn(ij... 
                        for (int id = 0; id < n_create; id++) {
                            labels.push_back(label_left[id]);
                        }
                        // tn(ij......ba)
                        for (int id = n_annihilate-1; id >= 0; id--) {
                            labels.push_back(label_right[id]);
                        }
                        newguy->set_amplitudes('m', n, labels);

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

                        newguy->is_boson_dagger.push_back(false);

                    }

                }else if (op.substr(0, 1) == "e" ){


                    if (op.substr(1, 1) == "1" ){

                        // find comma
                        size_t pos = op.find(",");
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
                        pos = op.find(",", pos + 1);
                        commas.push_back(pos);
                        while( pos != std::string::npos){
                            pos = op.find(",", pos + 1);
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
                        pos = op.find(",", pos + 1);
                        commas.push_back(pos);
                        while( pos != std::string::npos){
                            pos = op.find(",", pos + 1);
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
                        pos = op.find(",", pos + 1);
                        commas.push_back(pos);
                        while( pos != std::string::npos){
                            pos = op.find(",", pos + 1);
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

                }else if (op.substr(0, 1) == "a" ){ // single creator / annihilator


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

#pragma omp critical
{
                if (vacuum == "TRUE") {
                    add_new_string_true_vacuum(newguy, ordered, print_level, find_paired_permutations);
                } else {
                    add_new_string_fermi_vacuum(newguy, ordered, print_level, find_paired_permutations);
                }
}
        }
    }

    // reset number of threads for other parts of code
    omp_set_num_threads(1);
}

void pq_helper::simplify() {

    // eliminate strings based on delta functions and use delta functions to alter integral / amplitude labels
    for (std::shared_ptr<pq_string> & pq_str : ordered) {

        if ( pq_str->skip ) continue;

        // apply delta functions
        gobble_deltas(pq_str);

        // re-classify fluctuation potential terms
        reclassify_integrals(pq_str);

        // replace any funny labels that were added with conventional ones (fermi vacumm only)
        if ( vacuum == "FERMI" ) {
            use_conventional_labels(pq_str);
        }
    }

    // try to cancel similar terms
    cleanup(ordered, find_paired_permutations);

}

void pq_helper::print(const std::string &string_type) const {

    printf("\n");
    printf("    ");

    int n = 0;

    if ( string_type == "all" ) {

        printf("// normal-ordered strings:\n");
        for (const std::shared_ptr<pq_string> & pq_str : ordered) {
            pq_str->print();
        }
        printf("\n");
        return;

    }else if ( string_type == "one-body" ) {
        printf("// one-body strings:\n");
        n = 1;
    }else if ( string_type == "two-body" ) {
        n = 2;
        printf("// two-body strings:\n");
    }else if ( string_type == "fully-contracted" ) {
        printf("// fully-contracted strings:\n");
        n = 0;
    }

    for (const std::shared_ptr<pq_string> & pq_str : ordered) {
        // number of fermion + boson operators
        int my_n = pq_str->symbol.size() / 2 + pq_str->is_boson_dagger.size();
        if ( my_n != n ) continue;
        pq_str->print();
    }
    printf("\n");

}

// get list of fully-contracted strings, after spin tracing
std::vector<std::vector<std::string> > pq_helper::fully_contracted_strings_with_spin(std::unordered_map<std::string, std::string> &spin_labels) {

    // perform spin tracing

    std::vector< std::shared_ptr<pq_string> > spin_blocked;

    for (std::shared_ptr<pq_string> & pq_str : ordered) {
        if ( !pq_str->symbol.empty() ) continue;
        if ( !pq_str->is_boson_dagger.empty() ) continue;
        std::vector< std::shared_ptr<pq_string> > tmp;
        spin_blocking(pq_str, tmp, spin_labels);
        for (const std::shared_ptr<pq_string> & op : tmp) {
            spin_blocked.push_back(op);
        }
    }

    std::vector<std::vector<std::string> > list;
    for (auto & spin_str : spin_blocked) {
        if ( !spin_str->symbol.empty() ) continue;
        if ( !spin_str->is_boson_dagger.empty() ) continue;
        std::vector<std::string> my_string = spin_str->get_string_with_spin();
        if ( !my_string.empty() ) {
            list.push_back(my_string);
        }
    }

    return list;

}

std::vector<std::vector<std::string> > pq_helper::fully_contracted_strings() const {

    std::vector<std::vector<std::string> > list;
    for (const std::shared_ptr<pq_string> & pq_str : ordered) {
        if ( !pq_str->symbol.empty() ) continue;
        if ( !pq_str->is_boson_dagger.empty() ) continue;
        std::vector<std::string> my_string = pq_str->get_string();
        if ( (int)my_string.size() > 0 ) {
            list.push_back(my_string);
        }
    }

    return list;

}

std::vector<std::vector<std::string> > pq_helper::strings() const {

    std::vector<std::vector<std::string> > list;
    for (const std::shared_ptr<pq_string> & pq_str : ordered) {
        std::vector<std::string> my_string = pq_str->get_string();
        if ( (int)my_string.size() > 0 ) {
            list.push_back(my_string);
        }
    }

    return list;

}

void pq_helper::clear() {
    ordered.clear();
}

void pq_helper::add_st_operator(double factor, const std::vector<std::string> &targets,
                                               const std::vector<std::string> &ops){

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
    if ( cluster_operators_commute ) {

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
    }else {

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                add_double_commutator( 0.5 * factor, targets, {ops[i]}, {ops[j]});
            }
        }
        simplify();

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    add_triple_commutator( 1.0 / 6.0 * factor, targets, {ops[i]}, {ops[j]}, {ops[k]});
                }
            }
        }
        simplify();

        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                for (int k = 0; k < dim; k++) {
                    for (int l = 0; l < dim; l++) {
                        add_quadruple_commutator( 1.0 / 24.0 * factor, targets, {ops[i]}, {ops[j]}, {ops[k]}, {ops[l]});
                    }
                }
            }
        }
        simplify();

    }
}

} // End namespaces

#endif
