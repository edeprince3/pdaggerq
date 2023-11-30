//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: term.cc
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
#include <cstring>
#include <memory>
#include "../include/term.h"

using std::next_permutation;
using std::string;
using std::vector;
using std::map;
using std::pair;
using std::make_shared;
using std::shared_ptr;
using std::to_string;
using std::cout;
using std::endl;
using std::max;

namespace pdaggerq {

    Term::Term(const string &name, const shared_ptr<pq_string>& pq_str) {

        // check if term should be skipped (this should already be done before the term is constructed)
        if ( pq_str->skip ) return;

        // set coefficient
        coefficient_ = pq_str->sign * fabs(pq_str->factor);

        // check the permutation type
        perm_type_ = 0; // assume no permutations
        vector<std::string> perm_list;
        if (!pq_str->permutations.empty()) {
            perm_type_ = 1; // single permutations
            perm_list = pq_str->permutations;
        } else if (!pq_str->paired_permutations_2.empty()) {
            perm_type_ = 2; // double paired permutations
            perm_list = pq_str->paired_permutations_2;
        } else if (!pq_str->paired_permutations_3.empty()){
            perm_type_ = 3; // triple paired permutations
            perm_list = pq_str->paired_permutations_3;
        } else if (!pq_str->paired_permutations_6.empty()) {
            perm_type_ = 6; // sextuple paired permutations
            perm_list = pq_str->paired_permutations_6;
        }

        // set permutation indices (if any)
        if (perm_type_ != 0) {
            size_t n = perm_list.size();
            for (size_t i = 0; i < n; i += 2) {
                string perm1 = perm_list[i];
                string perm2 = perm_list[i + 1];

                // enforce alphabetical ordering
                if (perm1 > perm2)
                    std::swap(perm1, perm2);

                term_perms_.emplace_back(perm1, perm2);
            }
        }

        // add fermion operators
        for (size_t i = 0; i < pq_str->symbol.size(); i++) {
            string tmp = pq_str->symbol[i];
            if ( pq_str->is_dagger[i] )
                tmp += "*";
            rhs_.emplace_back(make_shared<Vertex>(tmp));
        }
        // add boson operators
        for (size_t i = 0; i < pq_str->is_boson_dagger.size(); i++) {
            string tmp = "B";
            if (pq_str->is_boson_dagger[i])
                tmp += "*";
            rhs_.emplace_back(make_shared<Vertex>(tmp));
        }
        if ( pq_str->has_w0 )
            rhs_.emplace_back(make_shared<Vertex>("w0"));

        // add lhs vertex
        lhs_ = make_shared<Vertex>(name);
        eq_ = copy_vert(lhs_);

        // create rhs vertices
        for (const auto & delta : pq_str->deltas) // add delta functions
            rhs_.push_back(make_shared<Vertex>(delta));
        for (const auto & int_pair : pq_str->ints) { // add integrals
            const string &type = int_pair.first;
            for (auto & integral : int_pair.second) {
                VertexPtr int_vert = make_shared<Vertex>(integral, type);
                if (type == "eri") { // permute eri to proper form
                    // swap sign if eri is permuted with sign change
                    if (int_vert->permute_eri()) swap_sign();
                }
                rhs_.push_back(int_vert);
            }
        }
        for (const auto & amp_pair : pq_str->amps) { // add amplitudes
            char type = amp_pair.first;
            for (auto & amp : amp_pair.second)
                rhs_.push_back(make_shared<Vertex>(amp, type));
        }

        // compute flop and memory scaling of the term
        compute_scaling();

        // set bottleneck flop and memory scaling
        bottleneck_flop_ = flop_map_.begin()->first;
        bottleneck_mem_ = mem_map_.begin()->first;

        // set comments
        comments_.push_back(to_string(coefficient_)); // add coefficient to vertex strings
        for (const auto &op : rhs_)
            comments_.push_back(op->str());

    }

    Term::Term(const string &name, const vector<string> &vertex_strings)
    : lhs_(make_shared<Vertex>(name)), comments_(vertex_strings){ // create lhs vertex

        // extract coefficient (first element in string)
        coefficient_ = stod(vertex_strings[0]); // convert string to double

        // assume no permutations in term
        perm_type_ = 0;

        /// construct rhs
        rhs_.reserve(vertex_strings.size() - 1); // reserve space for rhs
        for (int i = 1; i < vertex_strings.size(); i++) { // iterate over rhs
            const string& op_string = vertex_strings[i]; // get vertex string

            // check if vertex is a permutation (has a 'P' as first character)
            if (op_string[0] == 'P') {
                set_perm(op_string); // set permutation
            } else {
                // add vertex to vector
                VertexPtr op = make_shared<Vertex>(op_string); // create vertex
                if (op->name().find("eri") != string::npos && op->name().find("_X_") == string::npos) {
                    // check if vertex is an eri and not a linkage.
                    if (op->permute_eri()) swap_sign(); // swap sign if eri is permuted with sign change
                }
                rhs_.push_back(op); // add vertex to vector
            }
        }

        if (rhs_.empty()) return; // if constant, no need to construct linkage

//        make_generic(); // set rhs to generic form
        compute_scaling(); // compute flop and memory scaling of the term

        // set bottleneck flop and memory scaling
        bottleneck_flop_ = flop_map_.begin()->first;
        bottleneck_mem_ = mem_map_.begin()->first;

    }

    Term::Term(const VertexPtr &lhs_vertex, const vector<VertexPtr> &vertices, double coefficient) {

        lhs_ = lhs_vertex; // set lhs vertex
        rhs_ = vertices; // set rhs
        coefficient_ = coefficient; // set coefficient

        // check sign of coefficient if term has an eri vertex
        for (auto & op : rhs_) {
            // check if eri is in name
            if (op->base_name() =="eri") {
                if (op->permute_eri()) swap_sign(); // swap sign if eri is permuted with sign change
            }
        }

        compute_scaling(); // compute flop and memory scaling of the term

        // set bottleneck flop and memory scaling
        bottleneck_flop_ = flop_map_.begin()->first;
        bottleneck_mem_ = mem_map_.begin()->first;

        // set vertex strings
        comments_.push_back(to_string(coefficient_)); // add coefficient to vertex strings
        for (const auto &op : rhs_) comments_.push_back(op->str());
    }

    Term::Term(const LinkagePtr &linkage, double coeff) {

        is_assignment_ = true;

        // initialize coefficient as 1
        coefficient_ = coeff;

        // initialize lhs vertex
        lhs_ = linkage;
        rhs_ = {linkage->left_, linkage->right_};

        // set permutation indices as empty
        term_perms_ = {};
        perm_type_ = 0;

        // initialize bottleneck flop and memory scaling
        bottleneck_flop_ = linkage->flop_scale();
        bottleneck_mem_ = linkage->mem_scale();

        // reorder rhs
        reorder();
        term_linkage_->id_ = linkage->id_;

        // make labels generic
        *this = genericize();

        // set vertex strings


        string link_string = linkage->tot_str(true); // get linkage string with full expressions
        comments_.push_back(to_string(coefficient_)); // add linkage string to vertex strings
        comments_.emplace_back(link_string); // add linkage string to vertex strings

    }

    void Term::compute_scaling(const vector<VertexPtr>& arrangement, bool recompute) {

        if (!needs_update_ && !recompute) return; // if term does not need updating, return

        // reset flop and memory scaling maps
        flop_map_.clear(); // clear flop scaling map
        mem_map_.clear(); // clear memory scaling map

        /// add scaling from lhs

        shape lhs_shape = lhs_->shape_;

        // add scaling from permutation
        if (perm_type_ == 0) {
            flop_map_[lhs_shape]++;
            mem_map_[lhs_shape]++;
        }
        else if (perm_type_ == 1) {
            long long int num_perms = (1 << term_perms_.size()); // number of permutations
            flop_map_[lhs_shape] += num_perms;
            mem_map_[lhs_shape]  += num_perms;
        }
        else if (perm_type_ == 2) {
            flop_map_[lhs_shape] += 2;
            mem_map_[lhs_shape]  += 2;
        }
        else if (perm_type_ == 3) {
            flop_map_[lhs_shape] += 3;
            mem_map_[lhs_shape]  += 3;
        }
        else if (perm_type_ == 6) {
            flop_map_[lhs_shape] += 6;
            mem_map_[lhs_shape]  += 6;
        }
        else throw std::runtime_error("Invalid permutation type: " + std::to_string(perm_type_));

        // check if number of rhs is <= 1
        if (arrangement.size() <= 1) {

            bottleneck_flop_ = lhs_shape;
            bottleneck_mem_ = lhs_shape;

            if (arrangement.size() == 1) {
                term_linkage_ = as_link(std::make_shared<Vertex>("") * arrangement[0]);
            }

            return;
        }

        /// add scaling from rhs

        // get the total linkage of the term with its flop and memory scalings
        auto [term_linkage, flop_scales, mem_scales] = Linkage::link_and_scale(arrangement);
        term_linkage_ = term_linkage;


        // populate flop and memory scaling maps; get bottleneck scaling
        shape this_bottleneck_flop_;
        shape this_bottleneck_mem_;
        for (auto flop_scale : flop_scales) {
            flop_map_[flop_scale]++;
            if (flop_scale > this_bottleneck_flop_)
                this_bottleneck_flop_ = flop_scale;
        }
        for (auto mem_scale : mem_scales) {
            mem_map_[mem_scale]++;
            if (mem_scale > this_bottleneck_mem_)
                this_bottleneck_mem_ = mem_scale;
        }

        // indicate that term no longer needs updating
        needs_update_ = false;

    }

    void Term::reorder(bool recompute) { // reorder rhs in term

        /// Reorder by taking every permutation of vertex ordering and compute the scaling of the linkages.
        /// Keep permutation that minimizes the floating point cost of each linkage.

        // get number of rhs
        size_t n_vertices = rhs_.size();

        if (is_optimal_ && !recompute) return; // if term is already optimal return
        if (n_vertices <= 2) { // check if term has only one linkage or only one vertex; if so, compute scaling and return
            compute_scaling(recompute);
            return;
        }

        size_t initial_permutation[n_vertices]; // array to store initial permutation
        size_t current_permutation[n_vertices]; // initialize index for current permutation (initially 0, 1, 2, ...)
        size_t best_permutation[n_vertices];    // initialize index for best permutation (initially 0, 1, 2, ...)
        for (size_t i = 0; i < n_vertices; i++) {
            initial_permutation[i] = i;
            current_permutation[i] = i;
            best_permutation[i] = i;
        }

        // store best scaling as current scaling (scaling is performed in compute_scaling and called in constructor)
        scaling_map best_flop_map = flop_map_; // initialize the best flop scaling map
        scaling_map best_mem_map = mem_map_; // initialize the best memory scaling map

        // iterate over all permutations of the rhs
        while (next_permutation(current_permutation, current_permutation + n_vertices)) { // get next permutation

            // clear flop and memory scaling maps
            flop_map_.clear(); // clear flop scaling map
            mem_map_.clear(); // clear memory scaling map

            // create new arrangement
            std::vector<VertexPtr> new_arrangement;
            for (size_t i = 0; i < n_vertices; i++) {
                new_arrangement.push_back(rhs_[current_permutation[i]]);
            }

            // compute scaling for current permutation (populates flop and memory scaling maps)
            needs_update_ = true; // set flag to update scaling
            compute_scaling(new_arrangement, recompute);

            int scaling_check = flop_map_.compare(best_flop_map); // check if current permutation is better than best permutation

            bool is_better = scaling_check == scaling_map::this_better; // check if current permutation is better than best permutation
            if (scaling_check == 0) { // if scaling is equal, check memory scaling
                // check if current permutation is better than the best permutation in terms of memory scaling
                is_better = mem_map_.compare(best_mem_map) == scaling_map::this_better; // check if current permutation is better than best permutation
            }

            if (is_better) { // if current permutation is better than the best permutation
                best_flop_map = flop_map_; // set best scaling to current permutation
                best_mem_map = mem_map_; // set best scaling to current permutation
                for (size_t i = 0; i < n_vertices; i++) { // copy current permutation to best permutation
                    best_permutation[i] = current_permutation[i];
                }
            } // else, current permutation is worse than the best permutation and does not need to be saved

        }

        // clear best flop and memory scaling maps (no longer needed since best permutation is saved)
        best_flop_map.clear(); // clear best flop scaling map
        best_mem_map.clear(); // clear best memory scaling map

        // reorder rhs
        vector<VertexPtr> reordered_vertices; // initialize vector to store reordered rhs
        reordered_vertices.reserve(n_vertices); // reserve space for reordered rhs
        for (size_t i = 0; i < n_vertices; i++) { // iterate over rhs
            reordered_vertices.push_back(rhs_[best_permutation[i]]); // add vertex to reordered rhs
        }
        rhs_ = reordered_vertices; // set reordered rhs

        // remove any empty vertices
        rhs_.erase(std::remove_if(rhs_.begin(), rhs_.end(), [](const VertexPtr &vertex) {
            return vertex->base_name().empty(); }), rhs_.end()
        );

        // re-populate flop and memory scaling maps/bottlenecks and linkages

        needs_update_ = true; // set needs update to true
        compute_scaling(recompute); // compute scaling for reordered rhs

        is_optimal_ = true; // indicate that the term is optimal
    }

    string Term::str() const {

        string output;

        bool no_permutations = term_perms_.empty() || perm_type_ == 0;
        if ( no_permutations ) { // if no permutations
            if (Term::make_einsum)
                return einsum_str();

            // get lhs vertex string
            output = lhs_->str();

            // get sign of coefficient
            bool is_negative = coefficient_ < 0;
            if (is_assignment_) output += "  = ";
            else if (is_negative) output += " -= ";
            else output += " += ";

            // get absolute value of coefficient
            double abs_coeff = stod(to_string(fabs(coefficient_)));

            // if the coefficient is not 1, add it to the string
            bool added_coeff = false;
            bool needs_coeff = fabs(abs_coeff - 1) >= 1e-8 || rhs_.empty();

            // assignments of terms with negative coefficients need it to be added
            needs_coeff = (is_assignment_ && is_negative) || needs_coeff;

            if (needs_coeff) {
                // add coefficient to string
                added_coeff = true;
                if (is_assignment_ && is_negative)
                     output += "-";

//                output += to_string(abs_coeff);
                auto [numerator, denominator] = as_fraction(abs_coeff);
                if (denominator == 1)
                     output += to_string(numerator) + ".0";
                else output += to_string(numerator) + ".0/" + to_string(denominator) + ".0";


                // add multiplication sign if there are rhs vertices
                if (!rhs_.empty())
                    output += " * ";
            }

            // check if lhs vertex rank is zero
            bool lhs_zero_rank = lhs_->rank() == 0;

            bool format_dot = false;
            size_t vertex_count = rhs_.size();
            if (lhs_zero_rank && vertex_count > 1){
                // if there is more than one vertex with a rank greater than zero, format for a dot product
                size_t num_high_rank_ = 0;
                for (const auto & vertex : rhs_) {
                    if (vertex->rank() > 0) num_high_rank_++;
                    if (num_high_rank_ > 1) {
                        format_dot = true;
                        break;
                    }
                }
            }

            if (format_dot){
                // if lhs vertex rank is zero but has more than one vertex, format for dot product
                if (!added_coeff) {
                    auto [numerator, denominator] = as_fraction(abs_coeff);
                    if (denominator == 1)
                         output +=  to_string(numerator) + ".0";
                    else output += to_string(numerator) + ".0/" + to_string(denominator) + ".0";
                    output += " * ";
                }
                output += "dot(";
                // add rhs
                for (size_t i = 0; i < vertex_count; i++) {
                    output += rhs_[i]->str();

                    if (i < vertex_count - 2) output += " * ";
                    else if (i == vertex_count - 2) output += ", ";
                    else output += ");";
                }
            } else {
                // add rhs
                for (size_t i = 0; i < vertex_count; i++) {
                    output += rhs_[i]->str();
                    if (i != rhs_.size() - 1) output += " * ";
                    else output += ";";
                }
            }
        } else { // if there are permutations

            // make intermediate vertex for the permutation
            VertexPtr perm_vertex;

            bool make_perm_tmp = rhs_.size() == 1;
            if (make_perm_tmp) perm_vertex = rhs_[0]; // no need to create intermediate vertex if there is only one
            else { // else, create the intermediate vertex and its assignment term
                perm_vertex = copy_vert(lhs_);
                string perm_name = "perm_tmps";
                perm_name += "_" + perm_vertex->dimstring();
                perm_vertex->set_name(perm_name); // set name of permutation vertex
                perm_vertex->set_base_name("perm_tmps"); // set base name of permutation vertex
                perm_vertex->sort(); // sort permutation vertex

                // initialize initial permutation term
                Term perm_term = *this; // copy term
                perm_term.lhs_ = perm_vertex; // set lhs to permutation vertex
                perm_term.reset_perm();
                perm_term.is_assignment_ = true; // set term as assignment
                perm_term.coefficient_ = fabs(coefficient_); // set coefficient to absolute value of coefficient

                // add string to output
                output += perm_term.str();
                output += "\n";

            } // if only one vertex, use that vertex directly

            // initialize term to permute
            Term perm_term = *this; // copy term
            perm_term.rhs_ = {perm_vertex};

            // remove comments from term
            perm_term.comments_.clear();

            // if more than one vertex, set coefficient to 1 or -1
            if (!make_perm_tmp)
                perm_term.coefficient_ = coefficient_ > 0 ? 1 : -1;

            // get permuted terms
            vector<Term> perm_terms = perm_term.expand_perms();

            // add permuted terms to output
            for (auto & permuted_term : perm_terms) {
                output += permuted_term.str();
                output += "\n";
            }
            output.pop_back(); // remove last newline character
        }

        // ensure the last character is a semicolon (might not be there if no rhs vertices)
        if (output.back() != ';')
            output += ";";

        return output;
    }

    pair<int, int> Term::as_fraction(double coeff, double threshold) {
        /*
         * Represent the coefficient as a fraction
         * @param coeff coefficient to represent
         * @return string representation of the coefficient (i.e. 0.5 -> 1/2)
         */

        // check if coefficient is an integer and return it as a string if it is
        if (coeff == (int) coeff) return {coeff, 1};

        // assume the coefficient is 1 (it's not since we checked if it was an integer)
        int numerator = 1, denominator = 1;

        // get the error of the assumed fraction
        double error = fabs(coeff - (int) coeff);

        // store the best fraction
        pair<int, int> best_fraction = {1, 1};

        size_t maxiter = 1000, iter = 0;
        while (error > threshold && iter < maxiter) {

            // guess the next fraction
            double guess = numerator / (double) denominator;

            // check if the guess is better than the current best fraction
            double new_error = fabs(guess - coeff);
            if (new_error < error) {
                best_fraction = {numerator, denominator};
                error = new_error;
            }

            // increment the numerator or denominator
            if (coeff > guess) numerator++;
            else denominator++;
        }

        // return the best fraction
        return best_fraction;

    }

    string Term::einsum_str() const {
        string output;

        // get left hand side vertex name
        if (lhs_->is_linked())
             output = as_link(lhs_)->str(true, false);
        else output = lhs_->name();

        // get sign of coefficient
        bool is_negative = coefficient_ < 0;
        if (is_assignment_) output += " = ";
        else if (is_negative) output += " -= ";
        else output += " += ";

        // get absolute value of coefficient
        double abs_coeff = stod(to_string(fabs(coefficient_)));

        // if the coefficient is not 1, add it to the string
        bool added_coeff = false;
        bool needs_coeff = fabs(abs_coeff - 1) >= 1e-8 || rhs_.empty();

        // assignments of terms with negative coefficients need it to be added
        needs_coeff = (is_assignment_ && is_negative) || needs_coeff;

        if (needs_coeff) {
            // add coefficient to string
            added_coeff = true;
            if (is_assignment_ && is_negative)
                output += "-";

//            output += to_string(abs_coeff);
            auto [numerator, denominator] = as_fraction(abs_coeff);
            if (denominator == 1)
                output += to_string(numerator);
            else output += to_string(numerator) + ".0/" + to_string(denominator) + ".0";

            // add multiplication sign if there are rhs vertices
            if (!rhs_.empty())
                output += " * ";
        }

        // separate scalars and tensors in rhs vertices
        vector<VertexPtr> scalars;
        vector<VertexPtr> tensors;

        for (const VertexPtr &vertex : rhs_) {
            if (vertex->rank() == 0) scalars.push_back(vertex);
            else tensors.push_back(vertex);
        }

        bool has_tensors = !tensors.empty();

        // add scalars first
        for (size_t i = 0; i < scalars.size(); i++) {
            if (scalars[i]->is_linked())
                 output += as_link(scalars[i])->str(true, false);
            else output += scalars[i]->name();

            if (i != scalars.size() - 1 || has_tensors) output += " * ";
        }
        if (!has_tensors) return output;

        // make vector of line strings for each tensor
        vector<string> rhs_strings;
        for (const VertexPtr &vertex : tensors) {
            vector<Line> vertex_lines = vertex->lines();
            string line_string;
            for (auto & vertex_line : vertex_lines)
                line_string += vertex_line.label_;

            rhs_strings.push_back(line_string);
        }

        // get string of lines
        vector<Line> link_lines;
        string link_string;
        if (!tensors.empty()) {
            // get string of lines from lhs vertex
            for (auto & line : lhs_->lines())
                link_string += line.label_;
        }

        // make einsum string
        string einsum_string = "np.einsum('";
        for (const auto & rhs_string : rhs_strings){
            einsum_string += rhs_string;
            if (rhs_string != rhs_strings.back()) einsum_string += ",";
        }

        einsum_string += "->" + link_string + "', ";

        // add tensor names to einsum string
        for (size_t i = 0; i < tensors.size(); i++) {
            if (tensors[i]->is_linked())
                 einsum_string += as_link(tensors[i])->str(true, false);
            else einsum_string += tensors[i]->name();

            if (i != tensors.size() - 1) einsum_string += ", ";
            else einsum_string += ")";
        }

        output += einsum_string;
        return output;
    }

    string Term::make_comments(bool only_flop, bool only_comment) const {
        if (comments_.empty()) return "";

        string comment;
        if (!only_flop) {

            const vector<VertexPtr> &term_operators = term_linkage_->to_vector();
            for (const auto & vertex : term_operators) {
                if (vertex->base_name_.empty())
                    continue;
                comment += vertex->base_name_;
                if (vertex->has_blks()){
                    comment += "_" + vertex->blk_string();
                }
                comment += vertex->line_str();
                if (vertex != term_operators.back())
                    comment += " ";
            }

            // add permutations to comment if there are any
            if (!term_perms_.empty()){
                string perm_str;
                int count = 0;
                switch (perm_type_){
                    case 0: break;
                    case 1:
                        for (const auto & perm : term_perms_)
                            perm_str += "P(" + perm.first + "," + perm.second + ") ";
                        break;
                    case 2:
                        count = 0;
                        for (const auto & perm : term_perms_) {
                            if (count++ % 2 == 0)
                                perm_str += "PP2(" + perm.first + "," + perm.second;
                            else
                                perm_str += ";" + perm.first + "," + perm.second + ") ";
                        }
                        break;
                    case 3:
                        count = 0;
                        for (const auto & perm : term_perms_) {
                            if (count % 3 == 0)
                                perm_str += "PP3(" + perm.first + "," + perm.second;
                            else
                                perm_str += ";" + perm.first + "," + perm.second;
                            if (count++ % 3 == 2)
                                perm_str += ") ";
                        }
                        break;
                    case 6:
                        count = 0;
                        for (const auto & perm : term_perms_) {
                            if (count % 3 == 0)
                                perm_str += "PP6(" + perm.first + "," + perm.second;
                            else
                                perm_str += ";" + perm.first + "," + perm.second;
                            if (count++ % 3 == 2)
                                perm_str += ") ";
                        }
                        break;
                }

                comment = perm_str + comment;
            }

            // get coefficient
            double coeff = stod(comments_[0]);
            bool is_negative = coeff < 0;

            string assign_str = is_assignment_ ? " = " : " += ";

            auto [numerator, denominator] = as_fraction(fabs(coeff));
            string frac_coeff;
            if (denominator == 1 && numerator == 1 && !is_negative)
                 frac_coeff = "";
            else if (denominator == 1)
                 frac_coeff = to_string(numerator) + ".0 ";
            else frac_coeff = to_string(numerator) + ".0/" + to_string(denominator) + ".0 ";
            if (is_negative) frac_coeff = "-" + frac_coeff;

            // add lhs to comment
            comment = "// " + lhs_->str() + assign_str + frac_coeff + comment;
        }

        if (Term::make_einsum) // turn '//' into '#'
            std::replace(comment.begin(), comment.end(), '/', '#');

        if (only_comment) return comment;
        if (only_flop) comment += "\n";

        // add comment with flop and memory scaling
        if (rhs_.size() <= 1) {
            comment += " // flops: " + lhs_->dim().str();
            comment += " | mem: " + lhs_->dim().str();
            if (Term::make_einsum) // turn '//' into '#'
                std::replace(comment.begin(), comment.end(), '/', '#');
            return comment; // if there is only one vertex, return comment (no scaling to add)
        }

        string assign_str = " <- ";
        if (coefficient_ < 0 ) assign_str += "-";

        auto [flop_scales, mem_scales] = term_linkage_->scale_list(rhs_);
        if (flop_scales.empty() && mem_scales.empty()) { // no scaling to add as an additional comment
            // remove all quotes from comment
            comment.erase(std::remove(comment.begin(), comment.end(), '\"'), comment.end());
            if (Term::make_einsum) // turn '//' into '#'
                std::replace(comment.begin(), comment.end(), '/', '#');
            return comment;
        }

        comment += " // flops: " + lhs_->dim().str() + assign_str;
        for (const auto & flop : flop_scales)
            comment += flop.str() + " -> ";

        if (!flop_scales.empty()) {
            // remove last arrow (too lazy right now to do this elegantly)
            comment.pop_back(); comment.pop_back(); comment.pop_back(); comment.pop_back();
        }

        comment += " | mem: " + lhs_->dim().str() + assign_str;
        for (const auto & mem : mem_scales)
            comment += mem.str() + " -> ";
        if (!mem_scales.empty()) {
            // remove last arrow (too lazy right now to do this elegantly)
            comment.pop_back(); comment.pop_back(); comment.pop_back(); comment.pop_back();
        }

        // remove all quotes from comment
        comment.erase(std::remove(comment.begin(), comment.end(), '\"'), comment.end());

        if (Term::make_einsum) // turn '//' into '#'
            std::replace(comment.begin(), comment.end(), '/', '#');
        return comment;
    }

    void Term::apply_self_links() {
        if (rhs_.empty()) return; // if constant, exit

        // iterate over all rhs and convert traces to dot products with delta functions
        vector<VertexPtr> new_rhs; new_rhs.reserve(rhs_.size());
        for (auto & op : rhs_) {
            // check if vertex is a trace
            // get self-contracted lines
            VertexPtr copy = copy_vert(op);
            map<Line, uint_fast8_t> self_links = copy->self_links();

            bool has_self_link = false;
            for (const auto & [line, freq] : self_links) {
                if (freq > 1) {
                    has_self_link = true; break;
                }
            }
            if (!has_self_link) {
                new_rhs.push_back(copy); continue;
            }

            vector<VertexPtr> deltas = copy->make_self_linkages(self_links);

            // skip if no self links (this should never happen at this point)
            if (deltas.empty()) {
                new_rhs.push_back(op); continue;
            }

            // add delta functions to new rhs
            deltas.push_back(copy);

            // add to new rhs at beginning
            new_rhs.insert(new_rhs.begin(), deltas.begin(), deltas.end());

        }

        // reassign rhs
        rhs_ = new_rhs;

        // recompute the flop and memory cost of the term
        compute_scaling(true); // force recomputation of scaling
    }

    bool Term::equivalent(const Term &term1, const Term &term2) {

        // check if terms have the same number of rhs vertices
        if (term1.size() != term2.size()) return false;

        // do the terms have the same kind of permutation?
        bool same_permutation = term1.perm_type() == term2.perm_type(); // same permutation type?
        if (same_permutation) {
            if (term1.term_perms() != term2.term_perms())
               return false; // same permutation pairs?
        } else return false;

        // do the terms have similar rhs?
        bool similar_vertices = term1 == term2; // check if terms have similar rhs

        if (term1.size() > 1 && !similar_vertices) {

            // check that the vertex representation of the final rhs linkage is the same
            if (term1.term_linkage_->Vertex::operator!=(*term2.term_linkage_))
                return false;

            // above is redundant, but I'm keeping it here for now until I'm sure it's not needed

            // test if linkages are the same
            vector<LinkagePtr> term1_linkages = Linkage::links(term1.rhs_);
            vector<LinkagePtr> term2_linkages = Linkage::links(term2.rhs_);

            // if the terms do not have the same number of linkages, return false (they should)
            if (term1_linkages.size() != term2_linkages.size()) return false;

            // check if linkages are equivalent
            similar_vertices = true; // assume linkages are equivalent
            for (int i = 0; i < term1_linkages.size(); ++i) {
                if (*term1_linkages[i] != *term2_linkages[i]) {
                    similar_vertices = false;
                    break;
                }
            }
        }

        return similar_vertices;
    }

    pair<bool, bool> Term::same_permutation(const Term &ref_term, const Term &compare_term) {

        // check if terms have the same number of rhs vertices
        if (ref_term.size() != compare_term.size())
            return {false, false};

        // do the terms have the same kind of permutation?
        bool same_permutation = ref_term.perm_type() == compare_term.perm_type(); // same permutation type?
        if (same_permutation) {
            if (ref_term.term_perms() != compare_term.term_perms())
               return {false, false}; // same permutation pairs?
        } else return {false, false};

        // do the terms have similar rhs?
        bool similar_vertices = equivalent(ref_term, compare_term); // check if terms have similar rhs
        if (similar_vertices)
            return {true, false};

        // now we check if the terms are equivalent up to a permutation
        if (ref_term.size() > 1)
             return ref_term.term_linkage_->permuted_equals(*compare_term.term_linkage_);
        else return {false, false};
    }

    Term Term::genericize() const {
        // map unqiue lines to generic lines (i.e. a, b, c, ...)
        static std::string vir_lines[] {"a", "b", "c", "d", "e", "f", "g", "h", "v",
                                        "A", "B", "C", "D", "E", "F", "G", "H", "V"};
        static std::string occ_lines[] {"i", "j", "k", "l", "m", "n", "o",
                                        "I", "J", "K", "L", "M", "N", "O"};
        static std::string sig_lines[] {"X", "Y", "Z"};
        static std::string den_lines[] {"Q", "U"};

        size_t c_occ = 0;
        size_t c_vir = 0;
        size_t c_den = 0;
        size_t c_sig = 0;

        thread_local unordered_map<Line, Line, LineHash> line_map(256);
        line_map.clear();

        auto assign_generic_label = [ &c_occ, &c_vir, &c_den, &c_sig](const Line &line, const std::string &label) {
            // line does not exist in map, add it
            if (line.sig_) {
                if (c_sig >= 3) throw std::runtime_error("Too many sigma lines in genericize");
                line_map[line].label_ = sig_lines[c_sig++];
            } else if (line.den_) {
                if (c_den >= 2) throw std::runtime_error("Too many density lines in genericize");
                line_map[line].label_ = den_lines[c_den++];
            } else {
                if (line.o_) {
                    if (c_occ >= 14) throw std::runtime_error("Too many occupied lines in genericize");
                    line_map[line].label_ = occ_lines[c_occ++];
                } else {
                    if (c_vir >= 18) throw std::runtime_error("Too many virtual lines in genericize");
                    line_map[line].label_ = vir_lines[c_vir++];
                }
            }
        };

        /// map lines in term to generic lines

        for (const auto & line : lhs_->lines()) {
            size_t count = line_map.count(line);
            if (count == 0) {
                // line does not exist in map, add it
                line_map[line] = line;
                assign_generic_label(line, line.label_);
            }
        }
        if (eq_ != nullptr) {
            for (const auto &line: eq_->lines()) {
                size_t count = line_map.count(line);
                if (count == 0) {
                    // line does not exist in map, add it
                    line_map[line] = line;
                    assign_generic_label(line, line.label_);
                }
            }
        }
        for (const auto & vertex : rhs_) {
            for (const auto &line: vertex->lines()) {
                size_t count = line_map.count(line);
                if (count == 0) {
                    // line does not exist in map, add it
                    line_map[line] = line;
                    assign_generic_label(line, line.label_);
                }
            }
        }


        /// make a copy of the term but replace all lines with generic lines from map

        std::vector<VertexPtr> new_rhs;
        new_rhs.reserve(rhs_.size());
        for (const auto & vertex : rhs_) {
            VertexPtr new_vertex = copy_vert(vertex);
            std::vector<Line> new_lines = new_vertex->lines();
            for (Line & line : new_lines) {
                line = line_map[line];
            }
            new_vertex->update_lines(new_lines);
            new_rhs.push_back(new_vertex);
        }

        // make eq vertex generic
        VertexPtr new_eq = copy_vert((eq_ != nullptr) ? eq_ : lhs_);
        std::vector<Line> new_lines = new_eq->lines();
        for (Line & line : new_lines) {
            line = line_map[line];
        }
        new_eq->update_lines(new_lines);

        // make lhs generic
        VertexPtr new_lhs = copy_vert(lhs_);
        new_lines = new_lhs->lines();
        for (Line & line : new_lines) {
            line = line_map[line];
        }
        new_lhs->update_lines(new_lines);

        // make permutation generic
        perm_list new_perms;
        for (const auto & perm : term_perms_) {
            auto pos1 = std::find_if(line_map.begin(), line_map.end(), [&](const auto & pair) {
                return pair.first.label_ == perm.first;
            });
            auto pos2 = std::find_if(line_map.begin(), line_map.end(), [&](const auto & pair) {
                return pair.first.label_ == perm.second;
            });

            new_perms.emplace_back(pos1->first.label_, pos2->first.label_);
        }

        // make a copy of the term with the generic rhs
        Term new_term = *this;
        new_term.eq_ = lhs_;
        new_term.lhs_ = new_lhs;
        new_term.rhs_ = new_rhs;
        new_term.term_perms_ = new_perms;
        new_term.compute_scaling(true);
        return new_term;
    }

    void Term::request_update() {
        is_optimal_ = false; // set term to not optimal (for now)
        needs_update_ = true; // set term to be updated
        generated_linkages_ = false; // set term to not have generated linkages
    }

    void Term::reset_comments() {
        // set comments
        comments_.push_back(to_string(coefficient_)); // add coefficient to vertex strings
        for (const auto &op : rhs_)
            comments_.push_back(op->str());
    }

} // pdaggerq
