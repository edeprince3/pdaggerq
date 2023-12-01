//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: term_perm_helper.cc
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

#include <algorithm>
#include <map>
#include <cmath>
#include <iostream>
#include <stack>
#include "../include/term.h"

using std::logic_error;

namespace pdaggerq {

    void Term::set_perm(const string & perm_string) {// extract permutation indices
        VertexPtr perm_op = make_shared<Vertex>(perm_string); // create permutation vertex
        perm_op->sort(); // sort lines in permutation vertex

        // check if permutation is a P, PP2, PP3, or PP6
        size_t perm_rank = perm_op->rank(); // get rank of permutation (number of indices in permutation)

        if (perm_rank == 2) { // single index permutation
            perm_type_ = 1;
            term_perms_.emplace_back((*perm_op)[0].label_, (*perm_op)[1].label_); // add permutation indices to vector
        } else if (perm_rank == 4){ // PP2 permutation
            perm_type_ = 2;
            term_perms_.emplace_back((*perm_op)[0].label_, (*perm_op)[1].label_);
            term_perms_.emplace_back((*perm_op)[2].label_, (*perm_op)[3].label_);
        } else if (perm_rank == 6){
            // check if PP3 or PP6 (same ranks)
            if (perm_string[2] == '3'){ // PP3
                perm_type_ = 3;
                term_perms_.emplace_back((*perm_op)[0].label_, (*perm_op)[1].label_);
                term_perms_.emplace_back((*perm_op)[2].label_, (*perm_op)[3].label_);
                term_perms_.emplace_back((*perm_op)[4].label_, (*perm_op)[5].label_);
            } else if (perm_string[2] == '6'){ // PP6
                perm_type_ = 6;
                term_perms_.emplace_back((*perm_op)[0].label_, (*perm_op)[1].label_);
                term_perms_.emplace_back((*perm_op)[2].label_, (*perm_op)[3].label_);
                term_perms_.emplace_back((*perm_op)[4].label_, (*perm_op)[5].label_);
            } else throw logic_error("Invalid permutation vertex: " + perm_string);
        } else throw logic_error("Invalid permutation vertex: " + perm_string);
    }

    /**
     * permute terms with a given set of permutations
     * @param perms list of permutations
     * @param perm_type type of permutation
     * @return vector of permuted terms (including original term as first element)
     */
    vector<Term> Term::permute(const perm_list &perms, size_t perm_type) const{

        // return original term if no permutations are given
        if (perm_type == 0) return {*this};

        // initialize vector for the permuted terms
        // add original term
        vector<Term> perm_terms{*this};
        perm_terms.front().reset_perm();


        if (perm_type == 1) { // single index permutations

            // get all combinations of single index permutations
            vector<perm_list> perm_combos;

            // magic bit manipulation to get all 0->n combinations of n indices
            // https://www.geeksforgeeks.org/print-subsets-given-size-set/
            size_t n = perms.size();
            for (size_t i = 1; i < (1 << n); i++) {

                // build subset indices
                vector<size_t> subset;
                subset.reserve(n);

                for (size_t j = 0; j < n; j++) {
                    if (i & (1 << j)) subset.push_back(j);
                }

                // build subset
                perm_list add_perm;
                add_perm.reserve(subset.size());
                for (unsigned long idx : subset) {
                    add_perm.push_back(perms[idx]);
                }
                perm_combos.push_back(add_perm);
            }

            // permute vertices

            // create copy of the term
            Term perm_term = *this; // copy term
            perm_term.reset_perm(); // reset permutation indices
            perm_term.is_assignment_ = false; // set to false since we are permuting the term

            for (const auto &perm_combo: perm_combos) {

                // create deep copy of rhs vertices
                vector<VertexPtr> perm_vertices;
                for (const auto &vertex: rhs_)
                    perm_vertices.push_back(vertex->deep_copy_ptr());
                perm_term.rhs_ = perm_vertices; // set vertices in term
                perm_term.coefficient_ = coefficient_; // set vertices in term

                // set sign of permutation
                if (perm_combo.size() % 2 == 1)
                    perm_term.coefficient_ = -coefficient_;

                // single index permutations
                for (const auto &perm: perm_combo) {
                    for (VertexPtr &vertex: perm_vertices) {
                        for (Line &line: vertex->lines()) {
                            if (line.label_ == perm.first) line.label_ = perm.second;
                            else if (line.label_ == perm.second) line.label_ = perm.first;
                        }
                    }
                }

                // recomputes scaling
                perm_term.compute_scaling(true);
                perm_term.reset_comments();

                // add permuted term to vector
                perm_terms.push_back(perm_term);
            }


            return perm_terms;
        }

        // if not single index permutation, we are working with paired permutations that are not recursive

        // create deep copy of the term
        Term perm_term = *this; // copy term
        vector<VertexPtr> perm_vertices;
        for (const auto &vertex: rhs_)
            perm_vertices.push_back(vertex->deep_copy_ptr());
        perm_term.rhs_ = perm_vertices; // set vertices in term
        perm_term.reset_perm(); // reset permutation indices
        perm_term.is_assignment_ = false; // set to false since we are permuting the term

        // paired permutations
        if (perm_type == 2) {

            if (perms.size() != 2)
                throw logic_error("Invalid number of permutations for PP2 permutation");

            pair<string, string> perm_pair1 = perms[0];
            pair<string, string> perm_pair2 = perms[1];

            string perm_line1_1 = perm_pair1.first;
            string perm_line1_2 = perm_pair1.second;
            string perm_line2_1 = perm_pair2.first;
            string perm_line2_2 = perm_pair2.second;

            // swap line pairs
            for (VertexPtr & vertex : perm_vertices) {
                for (Line & line : vertex->lines()) {
                    if (line.label_ == perm_line1_1) line.label_ = perm_line2_1;
                    else if (line.label_ == perm_line2_1) line.label_ = perm_line1_1;
                    else if (line.label_ == perm_line1_2) line.label_ = perm_line2_2;
                    else if (line.label_ == perm_line2_2) line.label_ = perm_line1_2;
                }
            }

            // recomputes scaling
            perm_term.compute_scaling(true);
            perm_term.reset_comments();

            // add permuted term to vector
            perm_terms.push_back(perm_term);

            return perm_terms;
        }

        if (perm_type == 3) {
            if (perms.size() != 3)
                throw logic_error("Invalid number of permutations for PP3 permutation");

            pair<string, string> perm_pair1 = perms[0];
            pair<string, string> perm_pair2 = perms[1];
            pair<string, string> perm_pair3 = perms[2];

            string perm_line1_1 = perm_pair1.first;
            string perm_line1_2 = perm_pair1.second;
            string perm_line2_1 = perm_pair2.first;
            string perm_line2_2 = perm_pair2.second;
            string perm_line3_1 = perm_pair3.first;
            string perm_line3_2 = perm_pair3.second;

            // first pair permutation
            for (VertexPtr &vertex: perm_vertices) {
                for (Line &line: vertex->lines()) {
                    if (line.label_ == perm_line1_1) line.label_ = perm_line2_1;
                    else if (line.label_ == perm_line2_1) line.label_ = perm_line1_1;
                    else if (line.label_ == perm_line1_2) line.label_ = perm_line2_2;
                    else if (line.label_ == perm_line2_2) line.label_ = perm_line1_2;
                }
            }

            // add first permutation to vector
            // recomputes scaling
            perm_term.compute_scaling(true);
            perm_term.reset_comments();

            // add permuted term to vector
            perm_terms.push_back(perm_term);

            // make another deep copy of the term
            perm_vertices.clear();
            for (const auto &vertex: rhs_)
                perm_vertices.push_back(vertex->deep_copy_ptr());
            perm_term.rhs_ = perm_vertices;

            // second pair permutation
            for (VertexPtr &vertex: perm_vertices) {
                for (Line &line: vertex->lines()) {
                    if (line.label_ == perm_line1_1) line.label_ = perm_line3_1;
                    else if (line.label_ == perm_line3_1) line.label_ = perm_line1_1;
                    else if (line.label_ == perm_line1_2) line.label_ = perm_line3_2;
                    else if (line.label_ == perm_line3_2) line.label_ = perm_line1_2;
                }
            }

            // add second permutation to vector
            // recomputes scaling
            perm_term.compute_scaling(true);
            perm_term.reset_comments();

            // add permuted term to vector
            perm_terms.push_back(perm_term);
            return perm_terms;

        }

        if (perm_type == 6) {
            if (perms.size() != 3)
                throw logic_error("Invalid number of permutations for PP6 permutation");

            pair<string, string> perm_pair1 = perms[0];
            pair<string, string> perm_pair2 = perms[1];
            pair<string, string> perm_pair3 = perms[2];

            string perm_line1_1 = perm_pair1.first;
            string perm_line1_2 = perm_pair1.second;
            string perm_line2_1 = perm_pair2.first;
            string perm_line2_2 = perm_pair2.second;
            string perm_line3_1 = perm_pair3.first;
            string perm_line3_2 = perm_pair3.second;

            // reference (abc;ijk)

            // pair permutation (acb;ikj)
            for (VertexPtr &vertex: perm_vertices) {
                for (Line &line: vertex->lines()) {
                    if (line.label_ == perm_line1_1) line.label_ = perm_line2_1;
                    else if (line.label_ == perm_line2_1) line.label_ = perm_line1_1;
                    else if (line.label_ == perm_line1_2) line.label_ = perm_line2_2;
                    else if (line.label_ == perm_line2_2) line.label_ = perm_line1_2;
                }
            }
            // recomputes scaling
            perm_term.compute_scaling(true);
            perm_term.reset_comments();

            // add permuted term to vector
            perm_terms.push_back(perm_term);

            // make another deep copy of the term
            perm_vertices.clear();
            for (const auto &vertex: rhs_)
                perm_vertices.push_back(vertex->deep_copy_ptr());
            perm_term.rhs_ = perm_vertices;

            // pair permutation (bac;jik)
            for (VertexPtr &vertex: perm_vertices) {
                for (Line &line: vertex->lines()) {
                    if (line.label_ == perm_line1_1) line.label_ = perm_line2_1;
                    else if (line.label_ == perm_line2_1) line.label_ = perm_line1_1;
                    else if (line.label_ == perm_line1_2) line.label_ = perm_line2_2;
                    else if (line.label_ == perm_line2_2) line.label_ = perm_line1_2;
                }
            }

            // recomputes scaling
            perm_term.compute_scaling(true);
            perm_term.reset_comments();

            // add permuted term to vector
            perm_terms.push_back(perm_term);

            // pair permutation (cab;kij)
            for (VertexPtr &vertex: perm_vertices) {
                for (Line &line: vertex->lines()) {
                    if (line.label_ == perm_line1_1) line.label_ = perm_line3_1;
                    else if (line.label_ == perm_line3_1) line.label_ = perm_line1_1;
                    else if (line.label_ == perm_line1_2) line.label_ = perm_line3_2;
                    else if (line.label_ == perm_line3_2) line.label_ = perm_line1_2;
                }
            }
            // recomputes scaling
            perm_term.compute_scaling(true);
            perm_term.reset_comments();

            // add permuted term to vector
            perm_terms.push_back(perm_term);

            // pair permutation (cba;kji)
            for (VertexPtr &vertex: perm_vertices) {
                for (Line &line: vertex->lines()) {
                    if (line.label_ == perm_line2_1) line.label_ = perm_line3_1;
                    else if (line.label_ == perm_line3_1) line.label_ = perm_line2_1;
                    else if (line.label_ == perm_line2_2) line.label_ = perm_line3_2;
                    else if (line.label_ == perm_line3_2) line.label_ = perm_line2_2;
                }
            }
            // recomputes scaling
            perm_term.compute_scaling(true);
            perm_term.reset_comments();

            // add permuted term to vector
            perm_terms.push_back(perm_term);

            // pair permutation (bca;jki)
            for (VertexPtr &vertex: perm_vertices) {
                for (Line &line: vertex->lines()) {
                    if (line.label_ == perm_line1_1) line.label_ = perm_line2_1;
                    else if (line.label_ == perm_line2_1) line.label_ = perm_line1_1;
                    else if (line.label_ == perm_line1_2) line.label_ = perm_line2_2;
                    else if (line.label_ == perm_line2_2) line.label_ = perm_line1_2;
                }
            }
            // recomputes scaling
            perm_term.compute_scaling(true);
            perm_term.reset_comments();

            // add permuted term to vector
            perm_terms.push_back(perm_term);
            return perm_terms;
        } else throw logic_error("Invalid permutation type: " + std::to_string(perm_type));
    }

} // pdaggerq
