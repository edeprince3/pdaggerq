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
        eq_ = lhs_->safe_clone();

        // create rhs vertices
        for (const auto & delta : pq_str->deltas) // add delta functions
            rhs_.push_back(make_shared<Vertex>(delta));
        for (const auto & [type, integrals] : pq_str->ints) { // add integrals
            for (auto & integral : integrals) {
                VertexPtr int_vert = make_shared<Vertex>(integral, type);
                if (type == "eri") { // permute eri to proper form
                    // swap sign if eri is permuted with sign change
                    if (int_vert->permute_eri())
                        swap_sign();
                }
                rhs_.push_back(int_vert);
            }
        }
        for (const auto & [type, amp_vec] : pq_str->amps) { // add amplitudes
            for (auto & amp : amp_vec)
                rhs_.push_back(make_shared<Vertex>(amp, type));
        }

        // compute flop and memory scaling of the term
        compute_scaling();

        // set comments
        comments_.push_back(to_string(coefficient_)); // add coefficient to vertex strings
        for (const auto &op : rhs_)
            comments_.push_back(op->str());

        for (const std::string & str : pq_str->get_string())
            original_pq_ += str + ' ';

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
                if (op->name().find("eri") != string::npos && op->name().find('\t') == string::npos) {
                    // check if vertex is an eri and not a linkage.
                    if (op->permute_eri()) swap_sign(); // swap sign if eri is permuted with sign change
                }
                rhs_.push_back(op); // add vertex to vector
            }
        }

        if (rhs_.empty()) return; // if constant, no need to construct linkage

        compute_scaling(); // compute flop and memory scaling of the term

    }

    Term::Term(const ConstVertexPtr &lhs_vertex, const vector<ConstVertexPtr> &vertices, double coefficient) {

        lhs_ = lhs_vertex; // set lhs vertex
        rhs_ = vertices; // set rhs
        coefficient_ = coefficient; // set coefficient

        // check sign of coefficient if term has an eri vertex
        for (auto & op : rhs_) {
            // check if eri is in name
            if (op->base_name() =="eri") {
                VertexPtr new_eri = op->clone();
                if (new_eri->permute_eri()) swap_sign(); // swap sign if eri is permuted with sign change
                op = new_eri;
            }
        }

        compute_scaling(); // compute flop and memory scaling of the term

        // set vertex strings
        comments_.push_back(to_string(coefficient_)); // add coefficient to vertex strings
        for (const auto &op : rhs_) comments_.push_back(op->str());
    }

    Term::Term(const ConstLinkagePtr &linkage, double coeff) {

        is_assignment_ = true;

        // initialize coefficient as 1
        coefficient_ = coeff;

        // initialize lhs vertex
        lhs_ = linkage;

        rhs_ = linkage->link_vector();

        // set permutation indices as empty
        term_perms_ = {};
        perm_type_ = 0;

        // make labels generic (performs a deep copy)
//        *this = genericize();

        // compute flop and memory scaling of the term
        request_update();
        compute_scaling();

        // set vertex strings

        string link_string = linkage->tot_str(true); // get linkage string with full expressions
        comments_.push_back(to_string(coefficient_)); // add linkage string to vertex strings
        comments_.emplace_back(link_string); // add linkage string to vertex strings

    }

    Term::Term(const string &print_override) {
        // call default constructor
        *this = Term();

        // set print override
        print_override_ = print_override;

    }

    tuple<scaling_map, scaling_map, LinkagePtr> Term::compute_scaling(const vector<ConstVertexPtr>& arrangement, bool recompute) {

        // reset flop and memory scaling maps
        scaling_map flop_map; // clear flop scaling map
        scaling_map mem_map; // clear memory scaling map

        // helper function to add scaling with consideration of permutations
        auto add_scaling = [this, &flop_map, &mem_map](shape new_shape) {
            // add scaling from permutation
            if (perm_type_ == 0) {
                flop_map[new_shape]++;
                mem_map[new_shape]++;
            } else if (perm_type_ == 1) {
                long long int num_perms = (1 << term_perms_.size()); // number of permutations
                flop_map[new_shape] += num_perms;
                mem_map[new_shape] += num_perms;
            } else if (perm_type_ == 2) {
                flop_map[new_shape] += 2;
                mem_map[new_shape] += 2;
            } else if (perm_type_ == 3) {
                flop_map[new_shape] += 3;
                mem_map[new_shape] += 3;
            } else if (perm_type_ == 6) {
                flop_map[new_shape] += 6;
                mem_map[new_shape] += 6;
            } else throw std::runtime_error("Invalid permutation type: " + std::to_string(perm_type_));
        };

        /// add scaling from lhs

        shape lhs_shape = lhs_->shape_;
        add_scaling(lhs_shape);

        // check if number of rhs is <= 1
        if (arrangement.size() == 1) {
            add_scaling(arrangement[0]->dim());
            return {flop_map, mem_map, term_linkage_};
        } else if (arrangement.empty()) {
            return {flop_map, mem_map, term_linkage_};
        }

        /// add scaling from rhs

        // get the total linkage of the term with its flop and memory scalings
        auto [term_linkage, flop_scales, mem_scales] = Linkage::link_and_scale(arrangement);

        // populate flop and memory scaling maps; get bottleneck scaling
        for (shape flop_scale : flop_scales)
            flop_map[flop_scale]++;
//            add_scaling(flop_scale);
        for (shape mem_scale : mem_scales)
            mem_map[mem_scale]++;
//            add_scaling(mem_scale);

        return {flop_map, mem_map, term_linkage_};

    }

    size_t Term::count_idx_perm(const line_vector& ref_lines, const vector<ConstVertexPtr>& arrangement) {
        line_vector lines; lines.reserve(2*ref_lines.size());

        for (const auto & vertex : arrangement)
            for (const auto & line : vertex->lines())
                if (std::find(ref_lines.begin(), ref_lines.end(), line) != ref_lines.end())
                    lines.push_back(line);
        lines.erase(std::unique(lines.begin(), lines.end()), lines.end());

        size_t perms = 0;
        do {
            if (lines == ref_lines) break;
            perms++;
        } while (std::next_permutation(lines.begin(), lines.end()));

        return perms;
    }

    void Term::reorder(bool recompute) { // reorder rhs in term

        if (recompute) {
            is_optimal_ = false;
            needs_update_ = true;
        }

        if (is_optimal_ && !needs_update_) return; // if term is already optimal return

        // recompute initial scaling
        compute_scaling();

        if (is_optimal_) return; // if term is optimal return

        /// Reorder by taking every permutation of vertex ordering and compute the scaling of the linkages.
        /// Keep permutation that minimizes the floating point cost of each linkage.

        // get number of rhs
        size_t n_vertices = rhs_.size();

        if (n_vertices < 2) { return; } // not enough vertices to reorder

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
        vector<ConstVertexPtr> best_arrangement = rhs_; // initialize best arrangement
        line_vector left_lines = lhs_->lines(); // get lines of lhs
        bool found_better = false;

        // iterate over all permutations of the rhs
        while (next_permutation(current_permutation, current_permutation + n_vertices)) { // get next permutation

            // create new arrangement
            std::vector<ConstVertexPtr> new_arrangement;
            new_arrangement.reserve(n_vertices); // reserve space for new arrangement
            for (size_t i = 0; i < n_vertices; i++) {
                new_arrangement.push_back(rhs_[current_permutation[i]]);
            }

            // compute scaling for current permutation (populates flop and memory scaling maps)
            auto [flop_map, mem_map, linkage] = compute_scaling(new_arrangement);

            int scaling_check = flop_map.compare(best_flop_map); // check if current permutation is better than best permutation

            bool is_better = scaling_check == scaling_map::this_better; // check if current permutation is better than best permutation
            if (scaling_check == scaling_map::is_same) { // if scaling is equal, check memory scaling
                // check if current permutation is better than the best permutation in terms of memory scaling
                is_better = mem_map.compare(best_mem_map) == scaling_map::this_better; // check if current permutation is better than best permutation

                // if still equal, prefer linkage with the closest indices to the lhs (requires less index permutations)
                if (!is_better) {
                    size_t current_perm_count = count_idx_perm(left_lines, new_arrangement);
                    size_t best_perm_count = count_idx_perm(left_lines, best_arrangement);

                    // check if current permutation is better than the best permutation
                    is_better = current_perm_count < best_perm_count;
                }
            }

            if (is_better) { // if current permutation is better than the best permutation
                best_flop_map = flop_map; // set best scaling to current permutation
                best_mem_map = mem_map; // set best scaling to current permutation
                for (size_t i = 0; i < n_vertices; i++) { // copy current permutation to best permutation
                    best_permutation[i] = current_permutation[i];
                }
                best_arrangement = new_arrangement; // set best arrangement to current permutation
                found_better = true;
            } // else, current permutation is worse than the best permutation and does not need to be saved
        }

        if (!found_better)
            return;

        // reorder rhs
        vector<ConstVertexPtr> reordered_vertices; // initialize vector to store reordered rhs
        reordered_vertices.reserve(n_vertices); // reserve space for reordered rhs
        for (size_t i = 0; i < n_vertices; i++) { // iterate over rhs
            reordered_vertices.push_back(rhs_[best_permutation[i]]); // add vertex to reordered rhs
        }
        rhs_ = reordered_vertices; // set reordered rhs

        // remove any empty vertices
        rhs_.erase(std::remove_if(rhs_.begin(), rhs_.end(), [](const ConstVertexPtr &vertex) {
            return vertex->empty(); }), rhs_.end()
        );

        // re-populate flop and memory scaling maps/bottlenecks and linkages
        compute_scaling(true);
        is_optimal_ = true; // indicate that the term is optimal
    }

    string Term::str() const {

        if (!print_override_.empty())
            // return print override if it exists for custom printing
            return print_override_;

        string output;

        bool no_permutations = term_perms_.empty() || perm_type_ == 0;
        if ( no_permutations ) { // if no permutations
            if (make_einsum)
                return einsum_str();

            // get lhs vertex string
            output = lhs_->str();

            // get sign of coefficient
            bool is_negative = coefficient_ < 0;
            if (is_assignment_) output += "  = ";
            else if (is_negative) output += " -= ";
            else output += " += ";

            // get absolute value of coefficient
            double abs_coeff = fabs(coefficient_);

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

                int precision = minimum_precision(abs_coeff);
                output += to_string_with_precision(abs_coeff, precision);

                // add multiplication sign if there are rhs vertices
                if (!rhs_.empty())
                    output += " * ";
            }

            // check if lhs vertex rank is zero
            bool lhs_zero_rank = lhs_->rank() == 0;

            // seperate scalars and tensors in rhs vertices
            vector<ConstVertexPtr> scalars;
            vector<ConstVertexPtr> tensors;
            for (const ConstVertexPtr &vertex : rhs_) {
                if (vertex->rank() == 0)
                     scalars.push_back(vertex);
                else tensors.push_back(vertex);
            }

            bool format_dot = false;
            format_dot = lhs_zero_rank && tensors.size() > 1;

            if (format_dot){
                // if lhs vertex rank is zero but has more than one vertex, format for dot product
                if (!added_coeff && scalars.empty()) {
                    int precision = minimum_precision(abs_coeff);
                    output += to_string_with_precision(abs_coeff, precision);
                    output += " * ";
                }

                // first add scalars
                for (size_t i = 0; i < scalars.size(); i++) {
                    output += scalars[i]->str();
                    if (i != scalars.size() - 1 || !tensors.empty()) output += " * ";
                }

                // now add tensors with dot product
                output += "dot(";
                for (size_t i = 0; i < tensors.size(); i++) {
                    output += tensors[i]->str();

                    if (i < tensors.size() - 2) output += " * ";
                    else if (i == tensors.size() - 2) output += ", ";
                    else output += ");";
                }

            } else {
                // add rhs
                for (size_t i = 0; i < rhs_.size(); i++) {
                    output += rhs_[i]->str();
                    if (i != rhs_.size() - 1) output += " * ";
                    else output += ";";
                }
            }
        } else { // if there are permutations

            // make intermediate vertex for the permutation
            VertexPtr perm_vertex;

            bool make_perm_tmp = rhs_.size() == 1;
            if (make_perm_tmp) perm_vertex = rhs_[0]->clone(); // no need to create intermediate vertex if there is only one
            else { // else, create the intermediate vertex and its assignment term
                perm_vertex = lhs_->clone();
                string perm_name = "perm_tmps";
                perm_vertex->format_map_ = true; // format permutation vertex to print as map
                perm_vertex->sort(); // sort permutation vertex
                perm_vertex->update_name("perm_tmps"); // set name of permutation vertex

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
                output += '\n';
            }
            output.pop_back(); // remove last newline character
        }

        if (make_einsum)
            return output;

        // ensure the last character is a semicolon (might not be there if no rhs vertices)
        if (output.back() != ';')
            output += ';';

        return output;
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
        double abs_coeff = fabs(coefficient_);

        // if the coefficient is not 1, add it to the string
        bool added_coeff = false;
        bool needs_coeff = fabs(abs_coeff - 1) >= 1e-8 || rhs_.empty();

        // assignments of terms with negative coefficients need it to be added
        needs_coeff = (is_assignment_ && is_negative) || needs_coeff;

        // this is weird einsum craziness:
        // if there is only one operator on the right-hand side, and you DO NOT multiply by a scalar,
        // einsum will make a shallow copy of the operator and not a deep copy. This will then overwrite your
        // original operator (if it is a tensor). To avoid this, we always multiply by a scalar in this scenario.
        if (rhs_.size() == 1) {
            if (is_assignment_ && !rhs_[0]->is_scalar())
                needs_coeff = true;
        }


        if (needs_coeff) {
            // add coefficient to string
            added_coeff = true;
            if (is_assignment_ && is_negative)
                output += "-";

            int precision = minimum_precision(abs_coeff);
            output += to_string_with_precision(abs_coeff, precision);

            // add multiplication sign if there are rhs vertices
            if (!rhs_.empty())
                output += " * ";
        }

        // separate scalars and tensors in rhs vertices
        vector<ConstVertexPtr> scalars;
        vector<ConstVertexPtr> tensors;

        for (const ConstVertexPtr &vertex : rhs_) {
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
        for (const ConstVertexPtr &vertex : tensors) {
            line_vector vertex_lines = vertex->lines();
            string line_string;
            for (auto & vertex_line : vertex_lines)
                // einsum can only handle single character labels
                line_string += vertex_line.label_.front();

            rhs_strings.push_back(line_string);
        }

        // get string of lines
        line_vector link_lines;
        string link_string;
        if (!tensors.empty()) {
            // get string of lines from lhs vertex
            for (auto & line : lhs_->lines())
                link_string += line.label_.front();
        }

        // make einsum string
        string einsum_string = "einsum('";
        for (const auto & rhs_string : rhs_strings){
            einsum_string += rhs_string;
            if (rhs_string != rhs_strings.back()) einsum_string += ",";
        }

        einsum_string += "->" + link_string + "', ";

        // add tensor names to einsum string
        for (auto & tensor : tensors) {
            if (tensor->is_linked())
                 einsum_string += as_link(tensor)->str(true, false);
            else einsum_string += tensor->name();

            einsum_string += ", ";
        }

        if (tensors.size() > 2) {
            einsum_string += "optimize=['einsum_path',";
            for (size_t i = 0; i < tensors.size()-1; i++) {
                einsum_string += "(0,1),";
            }
            einsum_string.pop_back();
            einsum_string += "]";
        } else if (!tensors.empty()){
            einsum_string.pop_back();
            einsum_string.pop_back();
        }

        einsum_string += ')';
        output += einsum_string;
        return output;
    }

    string Term::make_comments(bool only_flop, bool only_comment) const {
        if (comments_.empty())
            return "";

        string comment;
        for (const auto &vertex: rhs_) {
            if (vertex->is_linked())
                comment += as_link(vertex)->tot_str(true);
            else
                comment += vertex->str();
            if (vertex != rhs_.back())
                comment += " * ";
        }

        // add permutations to comment if there are any
        if (!term_perms_.empty()) {
            string perm_str;
            int count = 0;
            switch (perm_type_) {
                case 0:
                    break;
                case 1:
                    for (const auto &perm: term_perms_)
                        perm_str += "P(" + perm.first + "," + perm.second + ") ";
                    break;
                case 2:
                    count = 0;
                    for (const auto &perm: term_perms_) {
                        if (count++ % 2 == 0)
                            perm_str += "PP2(" + perm.first + "," + perm.second;
                        else
                            perm_str += ";" + perm.first + "," + perm.second + ") ";
                    }
                    break;
                case 3:
                    count = 0;
                    for (const auto &perm: term_perms_) {
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
                    for (const auto &perm: term_perms_) {
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
        double coeff = coefficient_;
        bool is_negative = coeff < 0;

        string assign_str = is_assignment_ ? " = " : " += ";

        int precision = minimum_precision(coefficient_);
        string coeff_str = to_string_with_precision(fabs(coefficient_), precision);
        if (is_negative) coeff_str.insert(coeff_str.begin(), '-');
        coeff_str += ' ';

        if (original_pq_.empty()) {
            // add lhs to comment
            comment = "// " + lhs_->str() + assign_str + coeff_str + comment;
        } else {
            comment = "// " + lhs_->name() + assign_str + original_pq_;
        }

        if (only_flop) // clear comment if only flop is requested
            comment.clear();

        // remove all quotes from comment
        comment.erase(std::remove(comment.begin(), comment.end(), '\"'), comment.end());

        // format comment for python if needed
        if (make_einsum){
            // turn '//' into '#'
            size_t pos = comment.find("//");
            while (pos != std::string::npos) {
                comment.replace(pos, 2, "#");
                pos = comment.find("//", pos + 2);
            }
        }

        if (only_comment) return comment;
        if (only_flop) comment += "\n";

        // add comment with flop and memory scaling
        if (term_linkage_->depth() <= 1 || rhs_.empty()) {
            comment += " // flops: " + lhs_->dim().str();
            comment += " | mem: " + lhs_->dim().str();
            return comment; // if there is only one vertex, return comment (no scaling to add)
        }

        auto [term_linkage, flop_scales, mem_scales] = Linkage::link_and_scale(term_linkage_->link_vector());
        if (flop_scales.empty() && mem_scales.empty()) { // no scaling to add as an additional comment
            return comment;
        }

        comment += " // flops: " + lhs_->dim().str() + assign_str;
        for (const auto & flop : flop_scales)
            comment += flop.str() + " ";

        // remove last space
        if (!flop_scales.empty())
            comment.pop_back();


        comment += " | mem: " + lhs_->dim().str() + assign_str;
        for (const auto & mem : mem_scales)
            comment += mem.str() + " ";

        // remove last space
        if (!mem_scales.empty())
            comment.pop_back();

        return comment;
    }

    bool Term::apply_self_links() {
        if (rhs_.empty()) return false; // if constant, exit
        bool has_any_self_link = false;

        // iterate over all rhs and convert traces to dot products with delta functions
        vector<ConstVertexPtr> new_rhs; new_rhs.reserve(rhs_.size());
        for (auto & op : rhs_) {
            // check if vertex is a trace
            // get self-contracted lines
            VertexPtr copy = op->clone();
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

            has_any_self_link = true;

            vector<ConstVertexPtr> deltas = copy->make_self_linkages(self_links);

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

        return has_any_self_link;
    }

    bool Term::equivalent(const Term &term1, const Term &term2) {

        // make sure both terms have exactly the same lhs
        if (term1.lhs_->Vertex::operator!=(*term2.lhs_)) return false;

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

            // check that the linkages are equivalent
            auto term1_link = term1.lhs_ + term1.term_linkage_;
            auto term2_link = term2.lhs_ + term2.term_linkage_;

            if (*term1_link != *term2_link) return false;
            return true;

            // if so, ensure that the lines are exactly the same
//            return term1_link->lines() == term2_link->lines();
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

        auto assign_generic_label = [ &c_occ, &c_vir, &c_den, &c_sig](const Line &line, const string &label) {
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

        auto add_lines = [&assign_generic_label](const ConstVertexPtr &vertex){

            // check if vertex is a linkage
            if (vertex->is_linked()) {
                // expand all operators in linkage
                for (const auto & op : as_link(vertex)->vertices()) {
                    for (const auto & line : op->lines()) {
                        size_t count = line_map.count(line);
                        if (count == 0) {
                            // line does not exist in map, add it
                            line_map[line] = line;
                            assign_generic_label(line, line.label_);
                        }
                    }
                }
            } else {
                for (const auto &line: vertex->lines()) {
                    size_t count = line_map.count(line);
                    if (count == 0) {
                        // line does not exist in map, add it
                        line_map[line] = line;
                        assign_generic_label(line, line.label_);
                    }
                }
            }
        };

        /// map lines in term to generic lines

        // add lhs lines
        add_lines(lhs_);

        // add eq lines
        if (eq_ != nullptr)
            add_lines(eq_);

        // add rhs lines
        for (const auto & vertex : rhs_) {
            add_lines(vertex);
        }


        /// make a copy of the term but replace all lines with generic lines from map

        std::vector<ConstVertexPtr> new_rhs;
        new_rhs.reserve(rhs_.size());
        for (const auto & vertex : rhs_) {
            VertexPtr new_vertex = vertex->clone();
            new_vertex->replace_lines(line_map);
            new_rhs.push_back(new_vertex);
        }

        // make eq vertex generic
        VertexPtr new_eq = (eq_ != nullptr) ? eq_->clone() : lhs_->clone();
        new_eq->replace_lines(line_map);

        // make lhs generic
        VertexPtr new_lhs = lhs_->clone();
        new_lhs->replace_lines(line_map);

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
        new_term.reset_comments();
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

    vector<Term> Term::density_fitting() {
        // find every "eri" vertex and split it into two vertices and two terms using density fitting
        // so <pq|rs> becomes (Q|pq)(Q|rs) - (Q|ps)(Q|qr)

        vector<Term> new_terms; //
        new_terms.reserve(rhs_.size()+1);

        // iterate over all rhs and every time we see a vertex that is an eri,
        // split it into two vertices and two terms using density fitting
        if (rhs_.empty()) return {*this}; // if constant, return itself

        for (int i = 0; i < rhs_.size(); i++) {
            auto & op = rhs_[i];

            // check if vertex is an eri
            if (op->base_name() == "eri") {
                // term with eri looks like <pq||rs>
                // to do density fitting, we need to replace it with a product of two density fitting vertices within
                // two terms, so we need to create two new vertices and two new terms
                // <pq||rs> = <pq|rs> - <pq|sr> = (pr|qs) - (ps|qr) = (Q|pr)(Q|qs) - (Q|ps)(Q|qr)

                // grab the lines from the eri
                const line_vector &lines = op->lines();

                // create lines for the density fitting vertices
                Line den_line = Line("Q");

                line_vector B1_lines{den_line, lines[0], lines[2]};
                line_vector B2_lines{den_line, lines[1], lines[3]};
                line_vector B3_lines{den_line, lines[0], lines[3]};
                line_vector B4_lines{den_line, lines[1], lines[2]};

                // create vertices
                ConstVertexPtr B1 = make_shared<const Vertex>("B", B1_lines);
                ConstVertexPtr B2 = make_shared<const Vertex>("B", B2_lines);
                ConstVertexPtr B3 = make_shared<const Vertex>("B", B3_lines);
                ConstVertexPtr B4 = make_shared<const Vertex>("B", B4_lines);

                // create two new terms replacing the eri with the two new vertices
                Term new_term1 = *this, new_term2 = *this;

                // set new rhs of term1
                new_term1.rhs_[i] = B1;
                new_term1.rhs_.insert(new_term1.rhs_.begin() + (i+1), B2);

                // set new rhs of term2
                new_term2.rhs_[i] = B3;
                new_term2.rhs_.insert(new_term2.rhs_.begin() + (i+1), B4);
                new_term2.coefficient_ *= -1; // change sign of term2


                // add new terms to vector
                new_terms.push_back(new_term1);
                new_terms.push_back(new_term2);
            }
        }

        if (new_terms.empty()) return {*this}; // if no eris, return itself
        return new_terms;
    }

    Term Term::clone() const {
        Term new_term = *this;

        // make deep copies of all vertices
        new_term.lhs_ = lhs_ ? lhs_->clone() : nullptr;
        new_term.eq_  =  eq_ ? eq_->clone() : nullptr;
        new_term.rhs_.clear();
        for (const auto & vertex : rhs_)
            new_term.rhs_.push_back(vertex->clone());
        new_term.term_linkage_ = as_link(term_linkage_->clone());

        return new_term;
    }

    void Term::compute_scaling(bool recompute) {
        if (!needs_update_ && !recompute)
            return; // if term does not need updating, return

        auto [flop_map, mem_map, linkage] = compute_scaling(rhs_, recompute); // compute scaling of current rhs

        flop_map_ = flop_map;
        mem_map_  = mem_map;
        if (rhs_.size() > 1)
            term_linkage_ = linkage;
        else if (!rhs_.empty()) term_linkage_ = as_link(make_shared<Vertex>() * rhs_[0]);
        else term_linkage_ = as_link(make_shared<Vertex>() * make_shared<Vertex>());

        // indicate that term no longer needs updating
        needs_update_ = false;
    }

    set<string> Term::conditions() const {

        // TODO: use map instead of set to group similar conditions together

        ConstLinkagePtr term_linkage = term_linkage_; // get linkage representation of term
        set<string> conditions{}; // set to store conditions

        if (!term_linkage) {
            // return current conditions if no linkage
            if (rhs_.empty())
                return conditions;

            // if rhs is not empty, create a new term and get its linkage
            Term new_term = *this;
            new_term.compute_scaling(true); // force recomputation of scaling
            term_linkage = new_term.term_linkage_; // get linkage
        }

        if (!term_linkage)
            return conditions; // return current conditions if no linkage

        // map that stores conditions to their related operators
        const map<string, vector<string>> &mapped_conditions = mapped_conditions_;

        // create a set of operator basenames
        vector<ConstVertexPtr> vertices = term_linkage->vertices();
        for (const auto & vertex : vertices) {
            // loop over named conditions
            for (const auto & [condition, restrict_ops] : mapped_conditions) {
                // check if vertex is in the list of operators
                if (std::find(restrict_ops.begin(), restrict_ops.end(), vertex->base_name()) != restrict_ops.end())
                    conditions.insert(condition); // if so, add named condition to set
            }
        }

        // return set of operator basenames that have conditions
        return conditions;
    }

} // pdaggerq
