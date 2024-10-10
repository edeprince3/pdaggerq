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
        for (auto && is_boson_creator : pq_str->is_boson_dagger) {
            string tmp = "B";
            if (is_boson_creator)
                tmp += "*";
            rhs_.emplace_back(make_shared<Vertex>(tmp));
        }
        if ( pq_str->has_w0 )
            rhs_.emplace_back(make_shared<Vertex>("w0"));

        // add lhs vertex
        lhs_ = make_shared<Vertex>(name);
        eq_ = lhs_->clone();

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

        // shallow copy of lhs
        eq_ = lhs_->clone();

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

    Term::Term(const ConstVertexPtr &lhs_vertex, const vertex_vector &vertices, double coefficient) {

        lhs_ = lhs_vertex; // set lhs vertex
        eq_ = lhs_->clone(); // shallow copy of lhs for equation
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
        eq_ = lhs_->clone(); // shallow copy of lhs for equation

        LinkagePtr link = as_link(linkage->clone());
        link->id() = -1; // set id to -1 to allow it to be expanded

        expand_rhs(link); // expand rhs into vector

        // set permutation indices as empty
        term_perms_ = {};
        perm_type_ = 0;

        // make labels generic (performs a deep copy)
//        *this = genericize();

        // compute flop and memory scaling of the term
        request_update();
        reorder();

        // unset comments
        comments_.emplace_back("");
    }

    Term::Term(const string &print_override) {
        // call default constructor
        *this = Term();

        // set print override
        print_override_ = print_override;

    }

    tuple<scaling_map, scaling_map, LinkagePtr> Term::compute_scaling(const ConstVertexPtr &lhs, const vertex_vector &arrangement) {

        /// add scaling from rhs

        // get the total linkage of the term with its flop and memory scalings
        LinkagePtr linkage = Linkage::link(arrangement);
        auto [flop_map, mem_map] = linkage->netscales();

        /// add scaling from lhs
        flop_map[lhs->shape_]++;
        mem_map[lhs->shape_]++;

        return {flop_map, mem_map, linkage};

    }

    void Term::expand_rhs(const ConstVertexPtr &term_link) {

        // expand linkage into vector of vertices
        if (term_link->is_expandable()) {
            rhs_ = term_link->link_vector();
        } else if (term_link->is_linked() && !term_link->is_temp() && !term_link->is_addition()) {
            rhs_ = {as_link(term_link)->left(), as_link(term_link)->right()};
        } else {
            rhs_ = {term_link};
        }

        // find constants in rhs and merge them into the coefficient. skip empty vertices
        double merged_factor = coefficient_;

        vertex_vector new_rhs; new_rhs.reserve(rhs_.size());
        bool found_constant = false;
        for (ConstVertexPtr & op : rhs_) {
            if (op->empty()) continue; // skip empty vertices

            // determine if name is convertible to a double
            if (op->is_constant()) {
                merged_factor *= stod(op->name());
                found_constant = true;
            } else {
                new_rhs.push_back(op);
            }
        }

        // update coefficient
        if (found_constant)
            coefficient_ = merged_factor;

        // update rhs
        rhs_ = new_rhs;

        request_update();
        compute_scaling(true);
    }

    void Term::reorder(bool recompute) { // reorder rhs in term

        if (recompute) {
            request_update(); // request update if recompute is true
        }

        if (is_optimal_ && !needs_update_) return; // if term is already optimal return

        // recompute initial scaling
        compute_scaling(recompute);

        if (is_optimal_) return; // if term is already optimal, return

        /// Reorder by taking every permutation of the term and finding the best one.
        /// We use the linkage to determine the best permutation

        // generate every permutation and return the best one
        ConstLinkagePtr best_linkage = term_linkage()->best_permutation();

        // replace the rhs with the best linkage (if it is a temp or addition, we should not expand into a vector)
        expand_rhs(best_linkage);

        // recompute scaling
        compute_scaling(true);
        is_optimal_ = true;
    }

    bool Term::apply_self_links() {
        if (rhs_.empty()) return false; // if constant, exit
        bool has_any_self_link = false;

        // iterate over all rhs and convert traces to dot products with delta functions
        vertex_vector new_rhs; new_rhs.reserve(rhs_.size());
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

            vertex_vector deltas = copy->make_self_linkages(self_links);

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

    void Term::request_update() {
        is_optimal_ = false; // set term to not optimal (for now)
        needs_update_ = true; // set term to be updated
        generated_linkages_ = false; // set term to not have generated linkages
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
        Term new_term(*this);

        // make deep copies of all vertices
        new_term.lhs_ = lhs_ ? lhs_->clone() : nullptr;
        new_term.eq_  =  eq_ ? eq_->clone() : nullptr;
        new_term.rhs_.clear();
        for (const auto & vertex : rhs_)
            new_term.rhs_.push_back(vertex->clone());
        new_term.term_linkage() = as_link(term_linkage()->clone());

        return new_term;
    }

    Term Term::shallow() const {
        Term new_term(*this);

        // make deep copies of all vertices
        new_term.lhs_ = lhs_ ? lhs_->shallow() : nullptr;
        new_term.eq_  =  eq_ ? eq_->shallow() : nullptr;
        new_term.rhs_.clear();
        for (const auto & vertex : rhs_)
            new_term.rhs_.push_back(vertex->shallow());
        new_term.term_linkage() = as_link(term_linkage()->shallow());

        return new_term;
    }

    void Term::compute_scaling(bool recompute) {
        if (!needs_update_ && !recompute)
            return; // if term does not need updating, return

        // update coefficient if needed
        auto [flop_map, mem_map, linkage] = compute_scaling(lhs_, rhs_); // compute scaling of current rhs

        // needs to request update if the term_linkage is not the same as the computed linkage
        if (*term_linkage() != *linkage)
            request_update();

        flop_map_ = flop_map;
        mem_map_  = mem_map;
        term_linkage() = linkage;

        // indicate that term no longer needs updating
        needs_update_ = false;
    }

    idset Term::term_ids(const string &type) const {

        idset lhs_ids, rhs_ids, total_ids;

        rhs_ids = term_linkage()->get_ids(type);
        if (lhs()->is_temp()) lhs_ids = as_link(lhs())->get_ids(type);

        total_ids = rhs_ids;
        total_ids.insert(lhs_ids.begin(), lhs_ids.end());

        return total_ids;
    }

    long Term::max_id(const string &type) const {
        idset ids = term_ids(type);
        if (ids.empty()) return -1;

        return *std::max_element(ids.begin(), ids.end());
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
            int count;
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

        string assign_str = is_assignment_ ? "  = " : " += ";

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
        if (Vertex::print_type_ == "python"){
            // turn '//' into '#'
            size_t pos = comment.find("//");
            while (pos != std::string::npos) {
                comment.replace(pos, 2, "#");
                pos = comment.find("//", pos + 2);
            }
        }

        const auto &[flop_scales, mem_scales] = term_linkage(true)->scales();
        only_comment |= flop_scales.empty() && mem_scales.empty();
        if (only_comment) return comment;

        if (!comment.empty()) comment += "\n    ";
        comment += "// flops: " + lhs_->dim().str() + assign_str;
        for (const auto & flop : flop_scales)
            comment += flop.str() + " ";

        // remove last space
        if (!flop_scales.empty())
            comment.pop_back();


        comment += "\n    //  mems: " + lhs_->dim().str() + assign_str;
        for (const auto & mem : mem_scales)
            comment += mem.str() + " ";

        // remove last space
        if (!mem_scales.empty())
            comment.pop_back();

        // format comment for python if needed
        if (Vertex::print_type_ == "python"){
            // turn '//' into '#'
            size_t pos = comment.find("//");
            while (pos != std::string::npos) {
                comment.replace(pos, 2, "#");
                pos = comment.find("//", pos + 2);
            }
        }

        return comment;
    }

    void Term::reset_comments() {
        // set comments
        comments_.push_back(to_string(coefficient_)); // add coefficient to vertex strings
        for (const auto &op : rhs_)
            comments_.push_back(op->str());
    }

    std::set<string> Term::conditions() const {

        // map that stores conditions to their related operators
        const std::map<string, std::vector<string>> &mapped_conditions = mapped_conditions_;
        if (mapped_conditions.empty()) return {}; // return empty set if no conditions

        ConstLinkagePtr linkage = term_linkage(); // get linkage representation of term
        std::set<string> conditions{}; // set to store conditions

        if (!linkage) {
            // return current conditions if no linkage
            if (rhs_.empty())
                return conditions;

            // if rhs is not empty, create a new term and get its linkage
            Term new_term = *this;
            new_term.compute_scaling(true); // force recomputation of scaling
            linkage = new_term.term_linkage(); // get linkage
        }

        if (!linkage)
            return conditions; // return current conditions if no linkage

        // create a set of operator basenames
        vertex_vector vertices = linkage->vertices();

        // include lhs vertex in conditions (if it exists)
        if (lhs_) vertices.push_back(lhs_);
        for (const auto & vertex : vertices) {
            if (!vertex) continue; // skip if vertex is null

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
