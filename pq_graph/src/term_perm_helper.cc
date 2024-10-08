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

#include <iostream>
#include <stack>

#include "../include/term.h"

using std::logic_error;

namespace pdaggerq {

    void Term::set_perm(const string & perm_string) { // extract permutation indices
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
            term_perms_.emplace_back((*perm_op)[0].label_, (*perm_op)[1].label_);
            term_perms_.emplace_back((*perm_op)[2].label_, (*perm_op)[3].label_);
            term_perms_.emplace_back((*perm_op)[4].label_, (*perm_op)[5].label_);

            // check if PP3 or PP6 (same ranks)
            if (perm_string[2] == '3') // PP3
                perm_type_ = 3;
            else if (perm_string[2] == '6') // PP6
                perm_type_ = 6;
            else throw logic_error("Invalid permutation vertex: " + perm_string);

        } else throw logic_error("Invalid permutation vertex: " + perm_string);
    }

    vector<Term> Term::permute(const perm_list &perms, size_t perm_type) const{

        // return original term if no permutations are given
        if (perm_type == 0) return {*this};

        // initialize vector for the permuted terms
        // add original term (assuming rank 6 term as 'worst' case scenario)
        Term abcijk = this->clone();
        vector<Term> perm_terms{abcijk};
        perm_terms.front().reset_perm();

        // helper functions to swap lines in vertices
        static auto swap2lines = [](const ConstVertexPtr &reference_vertex, const string &p1_line, const string &p2_line){
            std::unordered_map<Line, Line, LineHash> line_map;
            VertexPtr vertex = reference_vertex->clone();
            for (Line &line: vertex->lines()) {
                if (line.label_ == p1_line){
                    Line new_line(line); new_line.label_ = p2_line; line_map[line] = new_line;
                } else if (line.label_ == p2_line) {
                    Line new_line(line); new_line.label_ = p1_line; line_map[line] = new_line;
                }
                else line_map[line] = line;
            }
            vertex->replace_lines(line_map);
            return vertex;
        };
        static auto apply_swaps = [](const Term& reference_term, const vector<pair<string,string>> &perm_pairs){
            Term perm_term = reference_term.clone(); // deep copy term
            vertex_vector perm_vertices;
            for (const auto &vertex: reference_term.rhs_) {
                VertexPtr perm_vertex = vertex->clone();
                for (const auto &perm_pair: perm_pairs) {
                    perm_vertex = swap2lines(perm_vertex, perm_pair.first, perm_pair.second);
                }
                perm_vertices.push_back(perm_vertex);
            }

            perm_term.rhs_ = perm_vertices;   // set vertices in term
            perm_term.reset_perm();           // reset permutation indices
            perm_term.is_assignment_ = false; // set to false since we are permuting the term
            perm_term.compute_scaling(true);  // recomputes scaling
            perm_term.reset_comments();       // reset comments

            return perm_term;
        };

        // single index permutations
        if (perm_type == 1) {

            // get all combinations of single index permutations
            size_t n = perms.size();
            for (size_t i = 1; i < (1 << n); i++) {
                // build indices for combination of permutations
                vector<size_t> comb_idxs;
                for (size_t j = 0; j < n; j++) {
                    // magic bit manipulation to get all 0->n combinations of n indices
                    if (i & (1 << j)) comb_idxs.push_back(j);
                }
                if (comb_idxs.empty()) continue;

                // build current combination of permutations to apply
                perm_list subperm;
                for (unsigned long idx : comb_idxs)
                    subperm.push_back(perms[idx]);

                // permute vertices

                /* EXAMPLE:
                 * // rt2(a,b,i,j) = 1.0 P(a,b) P(i,j) perm(a,b,i,j)
                 * rt2("a,b,i,j") += perm("a,b,i,j");
                 * rt2("a,b,i,j") -= perm("b,a,i,j");
                 * rt2("a,b,i,j") -= perm("a,b,j,i");
                 * rt2("a,b,i,j") += perm("b,a,j,i");
                 *
                 * // so we have 2^2 = 4 permutations: the original, swap a/b, swap i/j, and swap a/b and i/j
                 *
                 * // P(a,b) P(i,j) P(k,l) would have 2^3 = 8 permutations:
                 * //    the original (1)
                 * //    swap a/b, swap i/j, swap k/l (3)
                 * //    swap a/b and i/j, swap a/b and k/l, swap i/j and k/l (3)
                 * //    swap a/b, i/j, and k/l (1)
                 * */

                // single index permutations
                Term perm_term = apply_swaps(abcijk, subperm);

                // invert sign if odd number of permutations
                if (subperm.size() % 2 == 1)
                    perm_term.coefficient_ = -coefficient_;

                // add permuted term to vector
                perm_terms.push_back(perm_term);
            }

        }

        // paired permutations
        else if (perm_type == 2) {

            if (perms.size() != 2)
                throw logic_error("Invalid number of permutations for PP2 permutation");


            /*  EXAMPLE:
             * // rt3 = 1.0 PP2(i,a,j,b) perm(a,b,c,i,j,k)
                rt3("a,b,c,i,j,k") += perm("a,b,c,i,j,k");
                rt3("a,b,c,i,j,k") += perm("b,a,c,j,i,k");

                // we have 2 permutations: the original; swap a/b and i/j
             */

            const auto &[i_line, a_line] = perms[0];
            const auto &[j_line, b_line] = perms[1];

            // swap a/b and i/j
            Term bacjik = apply_swaps(abcijk, {{a_line, b_line}, {i_line, j_line}});

            // add permuted terms to vector
            perm_terms.push_back(bacjik);
        }
        else if (perm_type == 3 || perm_type == 6) {
            if (perms.size() != 3)
                throw logic_error("Invalid number of permutations for PP3 or PP6 permutation");

            /*  EXAMPLE:
             * // rt3 = 1.0 PP3(i,a,j,b,k,c) perm(a,b,c,i,j,k)
                rt3("a,b,c,i,j,k") += perm("a,b,c,i,j,k");
                rt3("a,b,c,i,j,k") += perm("b,a,c,j,i,k");
                rt3("a,b,c,i,j,k") += perm("c,b,a,k,j,i");

                // we have 3 permutations: the original; swap a/b and i/j; swap a/c and i/k
             */

            const auto &[i_line, a_line] = perms[0];
            const auto &[j_line, b_line] = perms[1];
            const auto &[k_line, c_line] = perms[2];

            // a/b and i/j swap
            Term bacjik = apply_swaps(abcijk, {{a_line, b_line}, {i_line, j_line}});
            // a/c and i/k swap
            Term cbakji = apply_swaps(abcijk, {{a_line, c_line}, {i_line, k_line}});

            // add permuted terms to vector
            perm_terms.push_back(bacjik);
            perm_terms.push_back(cbakji);

            if (perm_type == 6) {
                /*  EXAMPLE:
                 * // rt3 = 1.0 PP6(i,a,j,b,k,c) perm(a,b,c,i,j,k)
                    rt3("a,b,c,i,j,k") += perm("a,b,c,i,j,k");
                    rt3("a,b,c,i,j,k") += perm("b,a,c,j,i,k");
                    rt3("a,b,c,i,j,k") += perm("c,b,a,k,j,i");

                    rt3("a,b,c,i,j,k") += perm("a,c,b,i,k,j");
                    rt3("a,b,c,i,j,k") += perm("c,a,b,k,i,j");
                    rt3("a,b,c,i,j,k") += perm("b,c,a,j,k,i");

                    // we have 6 permutations total.
                    // It looks like a PP3 permutation of abc;ijk with a b/c and j/k swap of each PP3 permutation
                 */

                // swap b/c and j/k in each permuted term
                Term acbikj = apply_swaps(abcijk, {{b_line, c_line}, {j_line, k_line}});
                Term cabkij = apply_swaps(bacjik, {{b_line, c_line}, {j_line, k_line}});
                Term bcajki = apply_swaps(cbakji, {{b_line, c_line}, {j_line, k_line}});

                // add permuted terms to vector
                perm_terms.push_back(acbikj);
                perm_terms.push_back(cabkij);
                perm_terms.push_back(bcajki);
            }

        } else // invalid permutation type
            throw logic_error("Invalid permutation type: " + std::to_string(perm_type));

        return perm_terms;
    }

} // pdaggerq
