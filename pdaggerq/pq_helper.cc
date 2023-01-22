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
#include "pq_add_label_ranges.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// empty map of spin labels for fully_contracted_strings_with_spin
std::map<std::string, std::string> empty_spin_labels(){
    std::map<std::string, std::string> ret;
    return ret;
}

// empty map of spin labels for fully_contracted_strings_with_ranges
std::map<std::string, std::vector<std::string> > empty_label_ranges(){
    std::map<std::string, std::vector<std::string> > ret;
    return ret;
}

namespace py = pybind11;
using namespace pybind11::literals;

namespace pdaggerq {

void export_pq_helper(py::module& m) {
    py::class_<pdaggerq::pq_helper, std::shared_ptr<pdaggerq::pq_helper> >(m, "pq_helper")
        .def(py::init< std::string >())
        .def("set_print_level", &pq_helper::set_print_level)
        .def("set_left_operators", &pq_helper::set_left_operators)
        .def("set_right_operators", &pq_helper::set_right_operators)
        .def("set_left_operators_type", &pq_helper::set_left_operators_type)
        .def("set_right_operators_type", &pq_helper::set_right_operators_type)
        .def("set_cluster_operators_commute", &pq_helper::set_cluster_operators_commute)
        .def("set_find_paired_permutations", &pq_helper::set_find_paired_permutations)
        .def("simplify", &pq_helper::simplify)
        .def("clear", &pq_helper::clear)
        .def("print",
             [](pq_helper& self, std::string string_type) {
                 return self.print(string_type);
             },
             py::arg("string_type") = "fully-contracted" )
        .def("strings", &pq_helper::strings)
        .def("fully_contracted_strings", &pq_helper::fully_contracted_strings)
        .def("fully_contracted_strings_with_spin",
             [](pq_helper& self, std::map<std::string, std::string> spin_labels) {
                 return self.fully_contracted_strings_with_spin(spin_labels);
             },
             py::arg("spin_labels") = empty_spin_labels() )
        .def("fully_contracted_strings_with_ranges",
             [](pq_helper& self, std::map<std::string, std::vector<std::string> > label_ranges) {
                 return self.fully_contracted_strings_with_ranges(label_ranges);
             },
             py::arg("label_ranges") = empty_label_ranges() )
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

pq_helper::~pq_helper()
{
}

void pq_helper::set_find_paired_permutations(bool do_find_paired_permutations) {
    find_paired_permutations = do_find_paired_permutations;
}

void pq_helper::set_print_level(int level) {
    print_level = level;
}

void pq_helper::set_right_operators(std::vector<std::vector<std::string> >in) {

    right_operators.clear();
    for (int i = 0; i < (int)in.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)in[i].size(); j++) {
            tmp.push_back(in[i][j]);
        }
        right_operators.push_back(tmp);
    }
}

void pq_helper::set_left_operators(std::vector<std::vector<std::string> >in) {

    left_operators.clear();
    for (int i = 0; i < (int)in.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)in[i].size(); j++) {
            tmp.push_back(in[i][j]);
        }
        left_operators.push_back(tmp);
    }
}

void pq_helper::set_left_operators_type(std::string type) {
    if ( type == "EE" || type == "IP" || type == "EA" || type == "DIP" || type == "DEA" ) {
        left_operators_type = type;
    }else {
        printf("\n");
        printf("    error: invalid left-hand operator type (%s)\n", type.c_str());
        printf("\n");
        exit(1);
    }
}

void pq_helper::set_right_operators_type(std::string type) {
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
                               std::vector<std::string> op0,
                               std::vector<std::string> op1){

    add_operator_product( factor, concatinate_operators({op0, op1}) );
    add_operator_product(-factor, concatinate_operators({op1, op0}) );

}

void pq_helper::add_double_commutator(double factor,
                                        std::vector<std::string> op0, 
                                        std::vector<std::string> op1, 
                                        std::vector<std::string> op2){

    add_operator_product( factor, concatinate_operators({op0, op1, op2}) );
    add_operator_product(-factor, concatinate_operators({op1, op0, op2}) );
    add_operator_product(-factor, concatinate_operators({op2, op0, op1}) );
    add_operator_product( factor, concatinate_operators({op2, op1, op0}) );

}

void pq_helper::add_triple_commutator(double factor,
                                        std::vector<std::string> op0,
                                        std::vector<std::string> op1,
                                        std::vector<std::string> op2,
                                        std::vector<std::string> op3){

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
                                           std::vector<std::string> op0,
                                           std::vector<std::string> op1,
                                           std::vector<std::string> op2,
                                           std::vector<std::string> op3,
                                           std::vector<std::string> op4){


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

    std::vector<std::string> tmp;

    // left operators 
    // this is not handled correctly now that left operators can be sums of products of operators ... just exit with an error
    for (int i = 0; i < (int)left_operators.size(); i++) {
        tmp.clear();
        for (int j = 0; j < (int)left_operators[i].size(); j++) {
            if ( left_operators[i][j] == "v" ) {

                printf("\n");
                printf("    error: the fluctuation potential cannot appear in operators defining the bra state\n");
                printf("\n");
                exit(1);

                tmp.push_back("j1");
                tmp.push_back("j2");
            }else {
                tmp.push_back(left_operators[i][j]);
            }
        }
        left_operators[i].clear();
        for (int j = 0; j < (int)tmp.size(); j++) {
            left_operators[i].push_back(tmp[j]);
        }
        tmp.clear();
    }
    
    // right operators 
    // this is not handled correectly now that right operators can be sums of products of operators ... just exit with an error
    for (int i = 0; i < (int)right_operators.size(); i++) {
        tmp.clear();
        for (int j = 0; j < (int)right_operators[i].size(); j++) {
            if ( right_operators[i][j] == "v" ) {

                printf("\n");
                printf("    error: the fluctuation potential cannot appear in operators defining the ket state\n");
                printf("\n");
                exit(1);

                tmp.push_back("j1");
                tmp.push_back("j2");
            }else {
                tmp.push_back(right_operators[i][j]);
            }
        }
        right_operators[i].clear();
        for (int j = 0; j < (int)tmp.size(); j++) {
            right_operators[i].push_back(tmp[j]);
        }
        tmp.clear();
    }

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
        add_operator_product(factor, in);

        // term 2
        in.clear();
        for (int i = 0; i < count; i++) {
            in.push_back(tmp[i]);
        }
        in.push_back("j2");
        for (int i = count + 1; i < (int)tmp.size(); i++) {
            in.push_back(tmp[i]);
        }
        add_operator_product(factor, in);
        
        return;
    }

    // apply any extra operators on left or right:
    std::vector<std::string> save;
    for (int i = 0; i < (int)in.size(); i++) {
        save.push_back(in[i]);
    }

    if ( (int)left_operators.size() == 0 ) {
        std::vector<std::string> junk;
        junk.push_back("1");
        left_operators.push_back(junk);
    }
    if ( (int)right_operators.size() == 0 ) {
        std::vector<std::string> junk;
        junk.push_back("1");
        right_operators.push_back(junk);
    }

    // build strings
    double original_factor = factor;

    for (int left = 0; left < (int)left_operators.size(); left++) {

        for (int right = 0; right < (int)right_operators.size(); right++) {

            std::shared_ptr<pq_string> newguy (new pq_string(vacuum));

            factor = original_factor;

            std::vector<std::string> tmp_string;

            bool has_w0       = false;

            // stupid design choice ... o1-o4 and v1-v4 are already used
            int occ_label_count = 5;
            int vir_label_count = 5;
            int gen_label_count = 0;

            // apply any extra operators on left or right:
            std::vector<std::string> tmp;
            for (int i = 0; i < (int)left_operators[left].size(); i++) {
                tmp.push_back(left_operators[left][i]);
            }
            for (int i = 0; i < (int)save.size(); i++) {
                tmp.push_back(save[i]);
            }
            for (int i = 0; i < (int)right_operators[right].size(); i++) {
                tmp.push_back(right_operators[right][i]);
            }
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

                if ( in[i].substr(0, 1) == "h" ) { // one-electron operator

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // integrals
                    newguy->set_integrals("core", {idx1, idx2});

                }else if ( in[i].substr(0, 1) == "f" ) { // fock operator

                    std::string idx1 = "p" + std::to_string(gen_label_count++);
                    std::string idx2 = "p" + std::to_string(gen_label_count++);

                    // index 1
                    tmp_string.push_back(idx1+"*");

                    // index 2
                    tmp_string.push_back(idx2);

                    // integrals
                    newguy->set_integrals("fock", {idx1, idx2});

                }else if ( in[i].substr(0, 2) == "d+" ) { // one-electron operator (dipole + boson creator)

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

                }else if ( in[i].substr(0, 2) == "d-" ) { // one-electron operator (dipole + boson annihilator)

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

                }else if ( in[i].substr(0, 1) == "g" ) { // general two-electron operator

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

                }else if ( in[i].substr(0, 1) == "j" ) { // fluctuation potential

                    if ( in[i].substr(1, 1) == "1" ){

                        factor *= -1.0;

                        std::string idx1 = "p" + std::to_string(gen_label_count++);
                        std::string idx2 = "p" + std::to_string(gen_label_count++);

                        // index 1
                        tmp_string.push_back(idx1+"*");

                        // index 2
                        tmp_string.push_back(idx2);

                        // integrals
                        newguy->set_integrals("occ_repulsion", {idx1, idx2});

                    }else if ( in[i].substr(1, 1) == "2" ){

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

                }else if ( in[i].substr(0, 1) == "t" ){

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
                    newguy->set_amplitudes('t', n, labels);

                    // factor = 1/(n!)^2
                    double my_factor = 1.0;
                    for (int id = 0; id < n; id++) {
                        my_factor *= (id+1);
                    }
                    factor *= 1.0 / my_factor / my_factor;

                }else if ( in[i].substr(0, 1) == "w" ){ // w0 B*B

                    if ( in[i].substr(1, 1) == "0" ){

                        has_w0 = true;

                        newguy->is_boson_dagger.push_back(true);
                        newguy->is_boson_dagger.push_back(false);

                    }else {
                        printf("\n");
                        printf("    error: only w0 is supported\n");
                        printf("\n");
                        exit(1);
                    }

                }else if ( in[i].substr(0, 2) == "b+" ){ // B*

                        newguy->is_boson_dagger.push_back(true);

                }else if ( in[i].substr(0, 2) == "b-" ){ // B

                        newguy->is_boson_dagger.push_back(false);

                }else if ( in[i].substr(0, 1) == "u" ){ // t-amplitudes + boson creator

                    int n = std::stoi(in[i].substr(1));

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
                        newguy->set_amplitudes('u', n, labels);
                        
                        // factor = 1/(n!)^2
                        double my_factor = 1.0;
                        for (int id = 0; id < n; id++) {
                            my_factor *= (id+1);
                        }
                        factor *= 1.0 / my_factor / my_factor;

                        newguy->is_boson_dagger.push_back(true);
                    }

                }else if ( in[i].substr(0, 1) == "r" ){


                    int n = std::stoi(in[i].substr(1));

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

                }else if ( in[i].substr(0, 1) == "s" ){ // r amplitudes + boson creator

                    int n = std::stoi(in[i].substr(1));

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

                }else if ( in[i].substr(0, 1) == "l" ){

                    int n = std::stoi(in[i].substr(1));

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
                        // i*j*...
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

                }else if ( in[i].substr(0, 1) == "m" ){ // l amplitudes plus boson annihilator

                    int n = std::stoi(in[i].substr(1));

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
                        // i*j*...
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

                }else if ( in[i].substr(0, 1) == "e" ){


                    if ( in[i].substr(1, 1) == "1" ){

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
                        tmp_string.push_back(in[i].substr(2, len)+"*");

                        // index 2
                        tmp_string.push_back(in[i].substr(pos+1));

                    }else if ( in[i].substr(1, 1) == "2" ){

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

                        tmp_string.push_back(in[i].substr(2, commas[0]-2)+"*");
                        tmp_string.push_back(in[i].substr(commas[0]+1, commas[1]-commas[0]-1)+"*");
                        tmp_string.push_back(in[i].substr(commas[1]+1, commas[2]-commas[1]-1));
                        tmp_string.push_back(in[i].substr(commas[2]+1));

                    }else if ( in[i].substr(1, 1) == "3" ){

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

                        tmp_string.push_back(in[i].substr(2, commas[0]-2)+"*");
                        tmp_string.push_back(in[i].substr(commas[0]+1, commas[1]-commas[0]-1)+"*");
                        tmp_string.push_back(in[i].substr(commas[1]+1, commas[2]-commas[1]-1)+"*");
                        tmp_string.push_back(in[i].substr(commas[2]+1, commas[3]-commas[2]-1));
                        tmp_string.push_back(in[i].substr(commas[3]+1, commas[4]-commas[3]-1));
                        tmp_string.push_back(in[i].substr(commas[4]+1));

                    }else if ( in[i].substr(1, 1) == "4" ){

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

                        tmp_string.push_back(in[i].substr(2, commas[0]-2)+"*");
                        tmp_string.push_back(in[i].substr(commas[0]+1, commas[1]-commas[0]-1)+"*");
                        tmp_string.push_back(in[i].substr(commas[1]+1, commas[2]-commas[1]-1)+"*");
                        tmp_string.push_back(in[i].substr(commas[2]+1, commas[3]-commas[2]-1)+"*");
                        tmp_string.push_back(in[i].substr(commas[3]+1, commas[4]-commas[3]-1));
                        tmp_string.push_back(in[i].substr(commas[4]+1, commas[5]-commas[4]-1));
                        tmp_string.push_back(in[i].substr(commas[5]+1, commas[6]-commas[5]-1));
                        tmp_string.push_back(in[i].substr(commas[6]+1));

                    }else {
                        printf("\n");
                        printf("    error: only e1, e2, e3, and e4 operators are supported\n");
                        printf("\n");
                        exit(1);
                    }

                }else if ( in[i].substr(0, 1) == "1" ) { // unit operator ... do nothing

                }else if ( in[i].substr(0, 1) == "a" ){ // single creator / annihilator


                    if ( in[i].substr(1, 1) == "*" ){ // creator

                        tmp_string.push_back(in[i].substr(1)+"*");

                    }else { // annihilator

                        tmp_string.push_back(in[i].substr(1));

                    }

                }else {
                        printf("\n");
                        printf("    error: undefined string\n");
                        printf("\n");
                        exit(1);
                }
            }

            newguy->factor = factor;

            for (int i = 0; i < (int)tmp_string.size(); i++) {
                newguy->string.push_back(tmp_string[i]);
            }

            newguy->has_w0 = has_w0;

            if ( vacuum == "TRUE" ) {
                add_new_string_true_vacuum(newguy, ordered, print_level, find_paired_permutations);
            }else {
                add_new_string_fermi_vacuum(newguy, ordered, print_level, find_paired_permutations);
            }
        }
    }
}

void pq_helper::simplify() {

    // eliminate strings based on delta functions and use delta functions to alter integral / amplitude labels
    for (int i = 0; i < (int)ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        // apply delta functions
        gobble_deltas(ordered[i]);

        // re-classify fluctuation potential terms
        reclassify_integrals(ordered[i]);

        // replace any funny labels that were added with conventional ones (fermi vacumm only)
        if ( vacuum == "FERMI" ) {
            use_conventional_labels(ordered[i]);
        }
    }

    // try to cancel similar terms
    cleanup(ordered, find_paired_permutations);

}

void pq_helper::print(std::string string_type) {

    printf("\n");
    printf("    ");

    int n = 0;

    if ( string_type == "all" ) {

        printf("// normal-ordered strings:\n");
        for (int i = 0; i < (int)ordered.size(); i++) {
            ordered[i]->print();
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

    for (int i = 0; i < (int)ordered.size(); i++) {
        // number of fermion + boson operators
        int my_n = ordered[i]->symbol.size()/2 + ordered[i]->is_boson_dagger.size();
        if ( my_n != n ) continue;
        ordered[i]->print();
    }
    printf("\n");

}

// get list of fully-contracted strings, after assigning ranges to the labels
std::vector<std::vector<std::string> > pq_helper::fully_contracted_strings_with_ranges(std::map<std::string, std::vector<std::string> > label_ranges) {

    // add ranges to labels

    std::vector< std::shared_ptr<pq_string> > range_blocked;

    for (size_t i = 0; i < ordered.size(); i++) {
        if ( ordered[i]->symbol.size() != 0 ) continue;
        if ( ordered[i]->is_boson_dagger.size() != 0 ) continue;
        std::vector< std::shared_ptr<pq_string> > tmp;
        add_label_ranges(ordered[i], tmp, label_ranges);
        for (size_t j = 0; j < tmp.size(); j++) {
            range_blocked.push_back(tmp[j]);
        }
    }

    std::vector<std::vector<std::string> > list;
    for (size_t i = 0; i < range_blocked.size(); i++) {
        if ( range_blocked[i]->symbol.size() != 0 ) continue;
        if ( range_blocked[i]->is_boson_dagger.size() != 0 ) continue;
        std::vector<std::string> my_string = range_blocked[i]->get_string_with_label_ranges();
        //std::vector<std::string> my_string = range_blocked[i]->get_string();
        if ( my_string.size() > 0 ) {
            list.push_back(my_string);
        }
    }

    return list;

}

/*
    // add ranges to labels
    for (int i = 0; i < ordered.size(); i++) {
        if ( ordered[i]->symbol.size() != 0 ) continue;
        if ( ordered[i]->is_boson_dagger.size() != 0 ) continue;

        // get ranges for amplitudes
        for (size_t j = 0; j < ordered[i]->amplitude_types.size(); j++) {
            char type = ordered[i]->amplitude_types[j];
            for (size_t k = 0; k < ordered[i]->amps[type].size(); k++) {

                // amplitude type+order (ie 't' + '2' = "t2")
                std::string amp;
                amp.push_back(type);
                amp += std::to_string(ordered[i]->amps[type][k].order);
                
                // is this amplitude in the map?
                if ( label_ranges.find(amp) == label_ranges.end() ) {
                    printf("\n");
                    printf("    error: label range map does not contain %s\n", amp.c_str());
                    printf("\n");
                    exit(1);
                }

                // clear label ranges in case this function is called more than once
                ordered[i]->amps[type][k].label_ranges.clear();

                // add label ranges
                for (size_t m = 0; m < label_ranges[amp].size(); m++) {
                    ordered[i]->amps[type][k].label_ranges.push_back(label_ranges[amp][m]);
                }
            }
        }

        // now, consider amplitudes again, but make ranges consistent. if two labels have 
        // different ranges, change one to be the more restrictive case (e.g., "all" v "act" 
        // -> both "act" ). also, zero terms where ranges are inconsistent ("act" v "ext")
        bool kill_term = false;
        for (size_t j = 0; j < ordered[i]->amplitude_types.size(); j++) {
            char type = ordered[i]->amplitude_types[j];
            for (size_t k = 0; k < ordered[i]->amps[type].size(); k++) {
                for (size_t l = k+1; l < ordered[i]->amps[type].size(); l++) {
                    // check labels from amp1 vs labels in amp2. if they are common, check ranges
                    for (size_t kk = 0; kk < ordered[i]->amps[type][k].labels.size(); kk++) {
                        for (size_t ll = 0; ll < ordered[i]->amps[type][l].labels.size(); ll++) {
                            // common labels?
                            if ( ordered[i]->amps[type][k].labels[kk] == ordered[i]->amps[type][l].labels[ll] ) {

                                // six cases that require action:
                                if ( ordered[i]->amps[type][k].label_ranges[kk] == "all" && ordered[i]->amps[type][l].label_ranges[ll] == "ext" ) {
                                    ordered[i]->amps[type][k].label_ranges[kk] = "ext";
                                }else if ( ordered[i]->amps[type][k].label_ranges[kk] == "all" && ordered[i]->amps[type][l].label_ranges[ll] == "act" ) {
                                    ordered[i]->amps[type][k].label_ranges[kk] = "act";
                                }else if ( ordered[i]->amps[type][k].label_ranges[kk] == "ext" && ordered[i]->amps[type][l].label_ranges[ll] == "all" ) {
                                    ordered[i]->amps[type][l].label_ranges[ll] = "ext";
                                }else if ( ordered[i]->amps[type][k].label_ranges[kk] == "act" && ordered[i]->amps[type][l].label_ranges[ll] == "all" ) {
                                    ordered[i]->amps[type][l].label_ranges[ll] = "act";
                                }else if ( ordered[i]->amps[type][k].label_ranges[kk] == "ext" && ordered[i]->amps[type][l].label_ranges[ll] == "act" ) {
                                    kill_term = true;
                                    break;
                                }else if ( ordered[i]->amps[type][k].label_ranges[kk] == "act" && ordered[i]->amps[type][l].label_ranges[ll] == "ext" ) {
                                    kill_term = true;
                                    break;
                                }
                            }
                            if ( kill_term ) break;
                        }
                        if ( kill_term ) break;
                    }
                    if ( kill_term ) break;
                }
                if ( kill_term ) break;
            }
            if ( kill_term ) break;
        }
        if ( kill_term ) {
            ordered[i]->skip = true;
            continue;
        }

        // consider amplitudes yet again, but now make sure ranges are consistent with those
        // for non-summed labels. as above, change one to be the more restrictive case, and
        // zero terms where ranges are inconsistent
        kill_term = false;
        for (size_t j = 0; j < ordered[i]->amplitude_types.size(); j++) {
            char type = ordered[i]->amplitude_types[j];
            for (size_t k = 0; k < ordered[i]->amps[type].size(); k++) {
                for (size_t kk = 0; kk < ordered[i]->amps[type][k].labels.size(); kk++) {

                    // is this label a non-summed label contained in the map? if not, continue
                    if ( label_ranges.find(ordered[i]->amps[type][k].labels[kk]) == label_ranges.end() ) continue;

                    std::vector<std::string> my_label_range = label_ranges[ordered[i]->amps[type][k].labels[kk]];
                    if ( my_label_range.size() > 1 ) {
                        printf("\n");
                        printf("    error: non-summed label range lists should contain only one entry\n");
                        printf("\n");
                        exit(1);
                    }

                    // six cases that require action:
                    if ( ordered[i]->amps[type][k].label_ranges[kk] == "all" && my_label_range[0] == "ext" ) {
                        ordered[i]->amps[type][k].label_ranges[kk] = "ext";
                    }else if ( ordered[i]->amps[type][k].label_ranges[kk] == "all" && my_label_range[0] == "act" ) {
                        ordered[i]->amps[type][k].label_ranges[kk] = "act";
                    }else if ( ordered[i]->amps[type][k].label_ranges[kk] == "ext" && my_label_range[0] == "all" ) {
                        // do nothing?
                    }else if ( ordered[i]->amps[type][k].label_ranges[kk] == "act" && my_label_range[0] == "all" ) {
                        // do nothing?
                    }else if ( ordered[i]->amps[type][k].label_ranges[kk] == "ext" && my_label_range[0] == "act" ) {
                        kill_term = true;
                        break;
                    }else if ( ordered[i]->amps[type][k].label_ranges[kk] == "act" && my_label_range[0] == "ext" ) {
                        kill_term = true;
                        break;
                    }
                }
                if ( kill_term ) break;
            }
            if ( kill_term ) break;
        }
        if ( kill_term ) {
            ordered[i]->skip = true;
            continue;
        }

        // make integrals ranges consistent with those of amplitudes. first clear integral label ranges
        for (size_t l = 0; l < ordered[i]->integral_types.size(); l++) {
            std::string ints_type = ordered[i]->integral_types[l];
            for (size_t m = 0; m < ordered[i]->ints[ints_type].size(); m++) {

                // clear 
                ordered[i]->ints[ints_type][m].label_ranges.clear();
                for (size_t mm = 0; mm < ordered[i]->ints[ints_type][m].labels.size(); mm++) {
                    ordered[i]->ints[ints_type][m].label_ranges.push_back("all");
                }
            }
        }

        // now, make consistent with amplitudes
        for (size_t j = 0; j < ordered[i]->amplitude_types.size(); j++) {
            char amps_type = ordered[i]->amplitude_types[j];
            for (size_t k = 0; k < ordered[i]->amps[amps_type].size(); k++) {

                for (size_t l = 0; l < ordered[i]->integral_types.size(); l++) {
                    std::string ints_type = ordered[i]->integral_types[l];
                    for (size_t m = 0; m < ordered[i]->ints[ints_type].size(); m++) {

                        // check labels from amp1 vs labels in amp2. if they are common, check ranges
                        for (size_t kk = 0; kk < ordered[i]->amps[amps_type][k].labels.size(); kk++) {
                            for (size_t mm = 0; mm < ordered[i]->ints[ints_type][m].labels.size(); mm++) {
                                // common labels? if so, set ints ranges to match amps ranges
                                if ( ordered[i]->amps[amps_type][k].labels[kk] == ordered[i]->ints[ints_type][m].labels[mm] ) {
                                    ordered[i]->ints[ints_type][m].label_ranges[mm] = ordered[i]->amps[amps_type][k].label_ranges[kk];
                                }
                            }
                        }
                    }
                }
            }
        }

        // consider integrals again, but now make sure ranges are consistent with those
        // for non-summed labels. as above, change one to be the more restrictive case, and
        // zero terms where ranges are inconsistent
        kill_term = false;
        for (size_t j = 0; j < ordered[i]->integral_types.size(); j++) {
            std::string type = ordered[i]->integral_types[j];
            for (size_t k = 0; k < ordered[i]->ints[type].size(); k++) {
                for (size_t kk = 0; kk < ordered[i]->ints[type][k].labels.size(); kk++) {

                    // is this label a non-summed label contained in the map? if not, continue
                    if ( label_ranges.find(ordered[i]->ints[type][k].labels[kk]) == label_ranges.end() ) continue;

                    std::vector<std::string> my_label_range = label_ranges[ordered[i]->ints[type][k].labels[kk]];
                    if ( my_label_range.size() > 1 ) {
                        printf("\n");
                        printf("    error: non-summed label range lists should contain only one entry\n");
                        printf("\n");
                        exit(1);
                    }

                    // six cases that require action:
                    if ( ordered[i]->ints[type][k].label_ranges[kk] == "all" && my_label_range[0] == "ext" ) {
                        ordered[i]->ints[type][k].label_ranges[kk] = "ext";
                    }else if ( ordered[i]->ints[type][k].label_ranges[kk] == "all" && my_label_range[0] == "act" ) {
                        ordered[i]->ints[type][k].label_ranges[kk] = "act";
                    }else if ( ordered[i]->ints[type][k].label_ranges[kk] == "ext" && my_label_range[0] == "all" ) {
                        // do nothing?
                    }else if ( ordered[i]->ints[type][k].label_ranges[kk] == "act" && my_label_range[0] == "all" ) {
                        // do nothing?
                    }else if ( ordered[i]->ints[type][k].label_ranges[kk] == "ext" && my_label_range[0] == "act" ) {
                        kill_term = true;
                        break;
                    }else if ( ordered[i]->ints[type][k].label_ranges[kk] == "act" && my_label_range[0] == "ext" ) {
                        kill_term = true;
                        break;
                    }
                }
                if ( kill_term ) break;
            }
            if ( kill_term ) break;
        }
        if ( kill_term ) {
            ordered[i]->skip = true;
            continue;
        }

        // now permutation operators: 
        // 1. default to "all"
        // 2. set to whatever value shows up in the map
        // 3. if ranges within permutation operator are not consistent, expand the term explicitly
        //    like when doing spin tracing, this should actually be done before assigning any spaces
        //    to be sure we get all terms. for now, we'll ignore this situation, in which case
        //    we can just adjust permutaion labels below

    }

    // get strings. rather than implementing a new get_string_with_ranges
    // function, just create a new list of strings with modified labels
    std::vector< std::shared_ptr<pq_string> > new_ordered;
    for (int i = 0; i < ordered.size(); i++) {
        if ( ordered[i]->symbol.size() != 0 ) continue;
        if ( ordered[i]->is_boson_dagger.size() != 0 ) continue;
        if ( ordered[i]->skip ) continue;

        std::shared_ptr<pq_string> tmp (new pq_string(ordered[i]->vacuum));
        tmp->copy(ordered[i].get());

        for (size_t j = 0; j < tmp->amplitude_types.size(); j++) {
            char type = tmp->amplitude_types[j];
            for (size_t k = 0; k < tmp->amps[type].size(); k++) {
                for (size_t kk = 0; kk < tmp->amps[type][k].labels.size(); kk++) {
                    tmp->amps[type][k].labels[kk] += "_" + ordered[i]->amps[type][k].label_ranges[kk];
                }
            }
        }

        for (size_t j = 0; j < tmp->integral_types.size(); j++) {
            std::string type = tmp->integral_types[j];
            for (size_t k = 0; k < tmp->ints[type].size(); k++) {
                for (size_t kk = 0; kk < tmp->ints[type][k].labels.size(); kk++) {
                    tmp->ints[type][k].labels[kk] += "_" + ordered[i]->ints[type][k].label_ranges[kk];
                }
            }
        }

        // now permutation operators: 
        // 1. default to "all"
        // 2. set to whatever value shows up in the map
        for (size_t j = 0; j < tmp->permutations.size(); j++) {

            // is this label a non-summed label contained in the map? if not, set as "all"
            if ( label_ranges.find(tmp->permutations[j]) == label_ranges.end() ) {
                tmp->permutations[j] += "_all";
                continue;
            }

            std::vector<std::string> my_label_range = label_ranges[tmp->permutations[j]];
            if ( my_label_range.size() > 1 ) {
                printf("\n");
                printf("    error: non-summed label range lists should contain only one entry\n");
                printf("\n");
                exit(1);
            }

            tmp->permutations[j] += "_" + my_label_range[0];
        }

        new_ordered.push_back(tmp);
    }

    std::vector<std::vector<std::string> > list;
    for (size_t i = 0; i < new_ordered.size(); i++) {
        if ( new_ordered[i]->symbol.size() != 0 ) continue;
        if ( new_ordered[i]->is_boson_dagger.size() != 0 ) continue;
        if ( new_ordered[i]->skip ) continue;
        std::vector<std::string> my_string = new_ordered[i]->get_string();
        if ( my_string.size() > 0 ) {
            list.push_back(my_string);
        }
    }
    return list;
*/

// get list of fully-contracted strings, after spin tracing
std::vector<std::vector<std::string> > pq_helper::fully_contracted_strings_with_spin(std::map<std::string, std::string> spin_labels) {

    // perform spin tracing

    std::vector< std::shared_ptr<pq_string> > spin_blocked;

    for (size_t i = 0; i < ordered.size(); i++) {
        if ( ordered[i]->symbol.size() != 0 ) continue;
        if ( ordered[i]->is_boson_dagger.size() != 0 ) continue;
        std::vector< std::shared_ptr<pq_string> > tmp;
        spin_blocking(ordered[i], tmp, spin_labels);
        for (size_t j = 0; j < tmp.size(); j++) {
            spin_blocked.push_back(tmp[j]);
        }
    }

    std::vector<std::vector<std::string> > list;
    for (size_t i = 0; i < spin_blocked.size(); i++) {
        if ( spin_blocked[i]->symbol.size() != 0 ) continue;
        if ( spin_blocked[i]->is_boson_dagger.size() != 0 ) continue;
        std::vector<std::string> my_string = spin_blocked[i]->get_string_with_spin();
        if ( my_string.size() > 0 ) {
            list.push_back(my_string);
        }
    }

    return list;

}

std::vector<std::vector<std::string> > pq_helper::fully_contracted_strings() {

    std::vector<std::vector<std::string> > list;
    for (int i = 0; i < (int)ordered.size(); i++) {
        if ( ordered[i]->symbol.size() != 0 ) continue;
        if ( ordered[i]->is_boson_dagger.size() != 0 ) continue;
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

void pq_helper::clear() {
    ordered.clear();
}

void pq_helper::add_st_operator(double factor, std::vector<std::string> targets, std::vector<std::string> ops){

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
