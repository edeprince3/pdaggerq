//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: equation.cc
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

#include "../include/equation.h"
#include <omp.h>
#include <iostream>

namespace pdaggerq {

    Equation::Equation(const string &name, const vector<vector<string>> &term_strings) {

        // set name of equation
        name_ = name;

        /// construct terms
        terms_.reserve(term_strings.size()); // reserve space for terms
        for (const auto & term_string : term_strings) {
            if (term_string.empty()) throw invalid_argument("Empty term string for Equation: " + name);
            terms_.emplace_back(name, term_string); // create terms
        }

        // set assignment vertex
        assignment_vertex_ = terms_.front().lhs();

        // set equation vertex for all terms
        for (auto & term : terms_)
            term.eq() = assignment_vertex_;

        // collect scaling of equations
        collect_scaling();

    }

    Equation::Equation(const string &name, const vector<Term> &terms) {

        // set name of equation
        name_ = name;

        // set assignment vertex
        assignment_vertex_ = terms.back().lhs();

        // set terms
        terms_ = terms;

        // set equation vertex for all terms if applicable
        for (auto &term: terms_)
            term.eq()  = assignment_vertex_;


        // remove all terms that have 't1' in the base name of any rhs vertex
        if (remove_t1) {
            for (auto term_it = terms_.end() - 1; term_it >= terms_.begin(); --term_it) {
                bool found_t1 = false;
                for (auto &op : *term_it) {
                    found_t1 = op->base_name_ == "t1";
                    if (found_t1) break;
                }
                if (found_t1) terms_.erase(term_it);
            }
        }

        // collect scaling of equations
        collect_scaling();

    }

    Equation::Equation(const ConstVertexPtr &assignment, const vector<Term> &terms) {

        // set name of equation
        name_ = assignment->name();

        // set assignment vertex
        assignment_vertex_ = assignment;

        // set terms
        terms_ = terms;

        // set equation vertex for all terms if applicable
        for (auto &term: terms_)
            term.eq()  = assignment_vertex_;


        // remove all terms that have 't1' in the base name of any rhs vertex
        if (remove_t1) {
            for (auto term_it = terms_.end() - 1; term_it >= terms_.begin(); --term_it) {
                bool found_t1 = false;
                for (auto &op : *term_it) {
                    found_t1 = op->base_name_ == "t1";
                    if (found_t1) break;
                }
                if (found_t1) terms_.erase(term_it);
            }
        }

        // collect scaling of equations
        collect_scaling();

    }

    bool Equation::empty() {
        return terms_.empty();
    }

    vector<string> Equation::to_strings() const {

        vector<string> output;
        bool is_declaration = is_temp_equation_; // whether this is a declaration

        // set of conditions already found. Used to avoid printing duplicate conditions
        set<string> current_conditions = {"initialize"}; // initialize with arbitrary string
        if (Term::make_einsum)
            current_conditions = {}; // ignore conditions if make_einsum is true
        bool closed_condition = true; // whether the current condition has been closed
        for (const auto & term : terms_) { // iterate over terms

            // check if condition is already printed
            set<string> conditions = {}; //term.which_conditions();
            if (Term::make_einsum)
                conditions = {}; // ignore conditions if make_einsum is true
            if (conditions != current_conditions) { // if conditions are different, print new condition

                if (!closed_condition) {
                    // if the previous condition was not closed, close it
                    output.emplace_back("}");
                    closed_condition = true;
                }

                if (conditions.empty()) {
                    // if there are no conditions, print '{'
                    output.emplace_back("\n{");
                } else {

                    string if_block = "\nif (";
                    for (const string &condition : conditions) if_block += "include_" + condition + "_ && ";


                    if_block.pop_back();
                    if_block.pop_back();
                    if_block.pop_back();
                    if_block.pop_back();
                    if_block += ") {";
                    output.push_back(if_block);
                }

                // set closed condition to false
                closed_condition = false;

                // set current conditions
                current_conditions = conditions;

            } // else do nothing

            // add comments
            string comment = term.make_comments();
            if (!comment.empty()) {

                if (is_declaration) comment = term.make_comments(true);
                else comment.insert(0, "\n"); // add newline to the beginning of the comment

                // replace all '\"' with '' in comment
                size_t pos = 0;
                while ((pos = comment.find('\"', pos)) != string::npos) {
                    comment.replace(pos, 1, "");
                    pos += 1;
                }

                // replace all "\n" with "\n    " in comment
                pos = 0;
                while ((pos = comment.find('\n', pos)) != string::npos) {
                    comment.replace(pos, 1, "\n    ");
                    pos += 5;
                }

                output.push_back(comment); // add comment
            }

            // get string representation of term
            string term_string = "    ";
            term_string += term.str();

            // replace all "\n" with "\n    " in term_string
            size_t pos = 0;
            while ((pos = term_string.find('\n', pos)) != string::npos) {
                term_string.replace(pos, 1, "\n    ");
                pos += 5;
            }

            output.push_back(term_string);
        }

        if (!closed_condition) {
            // if the final condition was not closed, close it
            output.emplace_back("}");
            closed_condition = true;
        }

        return output;
    }

    void Equation::collect_scaling(bool regenerate) {

        if (terms_.empty()) return; // if no terms, return (nothing to do)

        // reset scaling maps
        flop_map_.clear(); // clear flop scaling map
        mem_map_.clear(); // clear memory scaling map

        // reset bottleneck scaling
        bottleneck_flop_ = terms_.front().bottleneck_flop(); // initialize bottleneck flop scaling

        for (auto & term : terms_) { // iterate over terms
            // compute scaling from term
            term.compute_scaling(regenerate);

            // collect scaling of terms
            scaling_map term_flop_map = term.flop_map(),
                        term_mem_map  = term.mem_map();

            flop_map_ += term_flop_map; // add flop scaling map
            mem_map_  += term_mem_map; // add memory scaling map

            // find bottlenecks
            const shape & term_bottleneck_flop = term.bottleneck_flop();

            if (term_bottleneck_flop > bottleneck_flop_)
                bottleneck_flop_ = term_bottleneck_flop;

        }
    }

    void Equation::reorder(bool recompute) {
        // reorder rhs in term
        for (auto & term : terms_)
            term.reorder(recompute);

        // collect scaling of terms
        collect_scaling();
    }

    size_t Equation::substitute(const ConstLinkagePtr &linkage, bool allow_equality) {

        if (name_ == "scalars") // if scalars, return
            return 0;

        // check if linkage is more expensive than current bottleneck
        if (linkage->worst_flop() > bottleneck_flop_) return 0;

        /// iterate over terms and substitute
        size_t num_terms = terms_.size();
        size_t num_subs = 0; // number of substitutions
        for (int i = 0; i < num_terms; i++) {
            Term &term = terms_[i]; // get term

            // check if linkage is compatible with term
            if (!term.is_compatible(linkage)) continue; // skip term if linkage is not compatible

            /// substitute linkage in term
            bool madeSub;
            madeSub = term.substitute(linkage, allow_equality);

            /// increment number of substitutions if substitution was successful
            if (madeSub) {
                ++num_subs;
                term.request_update(); // set term to be updated
            }
        } // substitute linkage in term

        return num_subs;
    }

    size_t Equation::test_substitute(const LinkagePtr &linkage, scaling_map &test_flop_map, bool allow_equality) {

        if (name_ == "scalars") { // if tmps, return
            test_flop_map += flop_map_; // add flop scaling map for whole equation
            return 0;
        } else if (name_ == "reuse_tmps")
            return 0;

        // check if linkage is more expensive than current bottleneck
        if (linkage->worst_flop() > bottleneck_flop_) {
            test_flop_map += flop_map_; // add flop scaling map for whole equation
            return 0; // return 0 substitutions
        }

        /// iterate over terms and substitute
        size_t num_terms = terms_.size();
        size_t num_subs = 0; // number of substitutions
        scaling_map eq_flop_map = flop_map_; // test memory scaling map
        for (int i = 0; i < num_terms; i++) {
            // skip term if linkage is not compatible
            if (!terms_[i].is_compatible(linkage)) continue;

            // get term copy
            Term term = terms_[i];

            // substitute linkage in term copy
            bool madeSub;
            madeSub = term.substitute(linkage, allow_equality);

            // increment number of substitutions if substitution was successful
            if (madeSub) {
                ++num_subs; // increment number of substitutions

                // update flop scaling map. It's faster to subtract the old scaling and add the new scaling than
                // to recompute the scaling map from scratch
                eq_flop_map -= terms_[i].flop_map();
                eq_flop_map += term.flop_map();
            }
        } // substitute linkage in term copy

        test_flop_map += eq_flop_map; // add flop scaling map for whole equation

        return num_subs;
    }

    linkage_set Equation::generate_linkages(bool compute_all) {

        linkage_set all_linkages(2048); // all possible linkages in the equations (start with large bucket n_ops)

        omp_set_num_threads((int)nthreads_);
        #pragma omp parallel for schedule(guided) shared(terms_, all_linkages) default(none) firstprivate(compute_all)
        for (auto & term : terms_) { // iterate over terms

            // skip term if it is optimal, and we are not computing all linkages
            if (!compute_all && term.generated_linkages_)
                continue;
            if (!term.is_optimal_ || term.needs_update_)
                term.reorder(); // reorder term if it is not optimal

            linkage_set term_linkages = term.generate_linkages(); // generate linkages in term

            #pragma omp critical
            {
                all_linkages += term_linkages; // add linkages to the set of all linkages
            }

            term.generated_linkages_ = true; // set term to have generated linkages

        } // iterate over terms
        omp_set_num_threads(1);

        return all_linkages;
    }

    void Equation::insert(const Term& term, int index) {
        if (index < 0) index = (int)terms_.size() + index + 1; // convert negative index to positive index from end
        terms_.insert(terms_.begin() + index, term); // add term to index of terms
    }

    void Equation::form_dot_products(linkage_set &scalars, size_t &n_temps) {

        // iterate over terms
        for (auto &term: terms_) {

            Term term_copy = term;
            LinkagePtr dot_product = term_copy.make_dot_products(0); // make dot product

            // skip if no dot product
            if (dot_product == nullptr) continue;
            if (dot_product->empty()) continue;

            // check if all_scalars contains dot_product
            bool found = scalars.contains(dot_product);
            if (found) {
                size_t idx = scalars[dot_product]->id_;
                dot_product = term.make_dot_products(idx); // apply dot product
            } else {
                dot_product = term.make_dot_products(n_temps++); // apply dot product
                scalars.insert(dot_product); // add linkage to set
            }

            bool has_more = true;
            while (has_more) { // iterate until no more dot products
                dot_product = term_copy.make_dot_products(0); // find next dot product

                // break if null pointer
                if (dot_product == nullptr) break;


                // if the dot product is empty, there are no more dot products
                has_more = !dot_product->empty();
                if (has_more) {
                    found = scalars.contains(dot_product);
                    if (found) {
                        size_t idx = scalars[dot_product]->id_;
                        dot_product = term.make_dot_products(idx); // apply dot product
                    } else {
                        dot_product = term.make_dot_products(n_temps++); // apply dot product
                        scalars.insert(dot_product); // add linkage to set
                    }
                }

            }
        }
    }

    void Equation::apply_self_links() {
        // iterate over terms
        for (auto& term : terms_) {
            term.apply_self_links(); // make trace
        }
    }

    size_t Equation::merge_terms() {
        if (is_temp_equation_) return 0; // don't merge temporary equations

        // map to store term counts, comments, and merged coefficients using a hash of the term
        merge_map_type merge_terms_map;
        size_t terms_size = terms_.size();

        // iterate over terms and accumulate similar terms
        for (int i = 0; i < terms_size; ++i) {
            Term term = terms_[i]; // get term
            // see if term is in map
            bool term_in_map = merge_terms_map.find(term) != merge_terms_map.end();

            // if term is not in map, check if a permuted version is in map
            if (!term_in_map)// && permuted_merge_) // if permuted merge is enabled
                merge_permuted_term(merge_terms_map, term, term_in_map);

            // add term to map
            if (!term_in_map)
                merge_terms_map[term] = make_pair(1, make_pair(term.comments(), term.coefficient_)); // initialize term count_ and coefficient
            else { // term is in map; update term count_ and coefficient
                // get iterator to term in map
                merge_terms_map[term].first++; // increment term count_
                auto &pair = merge_terms_map[term].second; // get a pair of vertex strings and coefficient

                pair.second += term.coefficient_; // accumulate coefficient
                auto &op_strings = pair.first; // get vertex strings
                auto &test_strings = term.comments(); // get vertex strings of test term

                // update vertex strings
                string pad = string(assignment_vertex_->name().size() + 2, ' ');
                op_strings.emplace_back("+");
                op_strings.push_back(pad + to_string(stod(test_strings[0]))); // add coefficient of test term to new term

                for (size_t j = 1; j < test_strings.size(); ++j) {
                    const string &op_string = test_strings[j]; // get vertex string
                    op_strings.push_back(op_string); // add vertex string to new term
                }
            }
        }

        size_t num_merged = 0;
        vector<Term> new_terms; // new terms
        // iterate over the map, adding each term to the new_terms vector
        for (auto &unique_term_pair: merge_terms_map) {
            Term new_term = unique_term_pair.first; // get term

            double new_coefficient = unique_term_pair.second.second.second; // get coefficient
            new_term.coefficient_ = new_coefficient; // set coefficient

            if (fabs(new_term.coefficient_) <= 1e-12) continue; // skip terms with zero coefficients

            vector<string> &new_op_strings = unique_term_pair.second.second.first; // get vertex strings
            new_term.comments() = new_op_strings; // set vertex strings

            size_t term_count = unique_term_pair.second.first; // get term count_
            num_merged += term_count - 1; // increment number of terms merged

            // add term to new_terms
            new_terms.push_back(new_term);
        }

        terms_ = new_terms;
        collect_scaling(true);

        return num_merged;
    }

    void Equation::merge_permuted_term(merge_map_type &merge_terms_map, Term &term, bool &term_in_map) {
        // test all possible permutations of each vertex.
        // If no more permutations are possible, an empty vertex is returned.
        // In this case, reset the vertex and move to the next vertex.
        // Do this for all possible permutations of all rhs.

        Term permuted_term = term;
        size_t num_ops = permuted_term.size(); // number of operators in term
        size_t op_index = 0; // index of vertex to permute
        size_t perm_idx = 0; // the n'th permutation of the vertex
        bool vertex_signs[permuted_term.size()]; // if true, the i'th vertex has an odd parity
        for (int j = 0; j < permuted_term.size(); ++j) vertex_signs[j] = false; // initialize vertex signs

        while (op_index < permuted_term.size()) {
            bool is_temp = term[op_index]->is_linked(); // is vertex an intermediate? (cannot permute)

            bool vertex_swap_sign = false; // if true, this vertex has an odd number of permutations
            VertexPtr op = make_shared<Vertex>(term[op_index]->permute(perm_idx, vertex_swap_sign)); // permute vertex
            if (op->empty() || is_temp) { // if no more permutations are possible, or vertex is temporary
                // reset vertex and move to next vertex
                permuted_term[op_index] = term[op_index];
                vertex_signs[op_index] = false;
                op_index++;
                perm_idx = 0;
                if (op_index == permuted_term.size()) break; // if no more rhs, break
            } else { // if vertex is not empty, set vertex and increment permutation index
                permuted_term[op_index] = op;
                perm_idx++;
            }

            // update vertex sign
            vertex_signs[op_index] = vertex_swap_sign;

            // now test every permutation of the other rhs
            size_t other_op_index = op_index + 1;
            size_t other_perm_idx = 0;
            while (other_op_index < permuted_term.size()) {
                bool other_is_temp = term[other_op_index]->is_linked(); // is vertex an intermediate? (cannot permute)

                bool other_vertex_swap_sign = false; // if true, this vertex has an odd number of permutations
                VertexPtr other_op = make_shared<Vertex>(term[other_op_index]->permute(other_perm_idx, other_vertex_swap_sign)); // permute vertex
                if (other_op->empty() || other_is_temp) { // if no more permutations, or vertex is temporary
                    // reset vertex and move to next vertex
                    permuted_term[other_op_index] = term[other_op_index];
                    vertex_signs[other_op_index] = false;
                    other_op_index++;
                    other_perm_idx = 0;
                    if (other_op_index == permuted_term.size()) break; // if no more rhs, break
                } else { // if vertex is not empty, set vertex and increment permutation index
                    permuted_term[other_op_index] = other_op;
                    other_perm_idx++;
                }

                // update vertex sign
                vertex_signs[other_op_index] = other_vertex_swap_sign;

                // there are more permutations of the vertex, so test if term is now in map
                term_in_map = merge_terms_map.find(permuted_term) != merge_terms_map.end();
                if (term_in_map) {
                    // if term is in map, determine if the permutation has an odd number of swaps
                    bool term_swap_sign = false;
                    for (int j = 0; j < permuted_term.size(); ++j) {
                        if (vertex_signs[j]) term_swap_sign = !term_swap_sign;
                    }

                    // change coefficient sign if necessary
                    permuted_term.coefficient_ = term.coefficient_ * (term_swap_sign ? -1 : 1);

                    // set term to permuted term
                    term = permuted_term;

                    term_in_map = true; // term is now in map

                    break; // break out of while linkage
                }
            } // end while iterate over other rhs

            // if term is not in map, reset other rhs
            if (term_in_map) break; // break out of while iterate over rhs if term is in map
            else{ // term is not in map, so reset other rhs
                for (size_t j = op_index + 1; j < permuted_term.size(); ++j) {
                    permuted_term[j] = term[j];
                    vertex_signs[j] = false;
                }
            }

            // there may be more permutations of the first vertex, so test if term is now in map
            term_in_map = merge_terms_map.find(permuted_term) != merge_terms_map.end();
            if (term_in_map) {
                // if term is in map, determine if the permutation has an odd number of swaps
                bool term_swap_sign = false;
                for (int j = 0; j < permuted_term.size(); ++j) {
                    if (vertex_signs[j]) term_swap_sign = !term_swap_sign;
                }

                // change coefficient sign if necessary
                permuted_term.coefficient_ = term.coefficient_ * (term_swap_sign ? -1 : 1);

                // set term to permuted term
                term = permuted_term;

                term_in_map = true; // term is now in map

                break; // break out of while linkage
            }
        } // end while iterate over rhs
    }

    void Equation::merge_permutations() {
        if (is_temp_equation_) return; // if tmps, return

        // make a map permutation types with their associated terms
        map<pair<perm_list, size_t>, vector<Term>> perm_type_to_terms;
        for (auto &term : terms_) perm_type_to_terms[{term.term_perms(), term.perm_type()}].push_back(term);

        vector<Term> new_terms;
        for (auto &perm_type_to_term : perm_type_to_terms){
            // get terms
            vector<Term> &terms = perm_type_to_term.second;

            // get permutation type
            const perm_list &perm = perm_type_to_term.first.first;
            size_t perm_type = perm_type_to_term.first.second;

            // if no permutation, just add terms
            if (perm_type == 0){
                for (auto &term : terms) new_terms.push_back(term);
                continue;
            }

            // make string representation of permutation
            string perm_str;
            for (auto &p : perm) perm_str += p.first + p.second + "_";
            perm_str += to_string(perm_type) + "_";

            // append "perm" to the name of the lhs vertex
            const ConstVertexPtr lhs_op = terms[0].lhs();
            VertexPtr new_lhs_op = lhs_op->deep_copy_ptr();
            new_lhs_op->set_base_name("perm_tmps_" + perm_str + new_lhs_op->base_name());
            new_lhs_op->update_lines(new_lhs_op->lines());

            bool first_assignment = false;
            for (Term term : terms){

                // reassigning the lhs vertex
                term.set_lhs(new_lhs_op);

                // reset permutation type
                term.set_perm(perm_list(), 0);

                // set term to be updated
                term.request_update();
                term.reorder();

                if (!first_assignment){
                    term.is_assignment_ = true;
                    first_assignment = true;
                }

                // add term to new terms
                new_terms.push_back(term);
            }

            // create new term that permutes the container
            Term perm_term = Term(lhs_op, {new_lhs_op}, 1.0);
            perm_term.set_perm(perm, perm_type);
            perm_term.eq() = assignment_vertex_;
            perm_term.request_update();
            perm_term.reorder();

            // add term to new terms
            new_terms.push_back(perm_term);
        }

        // set terms to new terms
        terms_ = std::move(new_terms);

        // reorder and collect scaling
        reorder();
        collect_scaling();
    }

    void Equation::expand_permutations() {
        vector<Term> new_terms;
        for (auto &term : terms_) {

            // only expand terms with one operator on the rhs
//            if (term.size() >= 2) {
//                new_terms.push_back(term);
//                continue;
//            }

            vector<Term> expanded_terms = term.expand_perms();
            for (size_t i = 0; i < expanded_terms.size(); ++i) {
                Term &expanded_term = expanded_terms[i];
                if (i != 0) // clear comments for permuted terms
                    expanded_term.comments() = {};

                new_terms.push_back(expanded_term);
            }
        }
        terms_ = new_terms;

        // reorder and collect scaling
        reorder(true);
    }

    vector<Term>::iterator Equation::insert_term(const Term &term, int index) {
        if (index < 0)
            index = (int)terms_.size() + index + 1; // convert negative index to positive index from the end
        return terms_.insert(terms_.begin() + index, term); // add term to index of terms
    }

    vector<Term *> Equation::get_temp_terms(const LinkagePtr& contraction) {
        // for every term, check if this contraction is within the term (do not check lhs)
        vector<Term *> temp_terms;
        for (auto &term : terms_) {
            for (auto &op : term) {
                if (op->is_linked()) {
                    const ConstLinkagePtr &linkage = as_link(op);
                    if (linkage->id_ == contraction->id_ && *linkage == *contraction) {
                        temp_terms.push_back(&term);
                        break;
                    }
                }
            }
        }
        return temp_terms;
    }

} // pdaggerq