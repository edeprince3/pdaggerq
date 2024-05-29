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

        if (terms.empty()) // throw error if no terms
            throw invalid_argument("Empty terms for Equation: " + name);

        // set assignment vertex
        assignment_vertex_ = terms.back().lhs();

        // set terms
        terms_ = terms;

        // set equation vertex for all terms if applicable
        for (auto &term: terms_)
            term.eq()  = assignment_vertex_;

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
        set<string> current_conditions = {};
        bool closed_condition = true; // whether the current condition has been closed
        for (const auto & term : terms_) { // iterate over terms


            // check if condition is already printed
            set<string> conditions = term.conditions();
            if (conditions != current_conditions) { // if conditions are different, print new condition

                bool has_condition = !conditions.empty();
                bool had_condition = !current_conditions.empty();

                if (had_condition && !closed_condition) {
                    // if the previous condition was not closed, close it
                    if (!Term::make_einsum)
                        output.emplace_back("}");
                    closed_condition = true; // indicate that the condition is closed
                }

                if (has_condition) {
                    closed_condition = false; // set condition to be closed

                    string if_block = condition_string(conditions);
                    output.push_back(if_block);
                }

                // set current conditions
                current_conditions = conditions;

            } // else do nothing

            // if override is set, print override
            bool override = !term.print_override_.empty();
            if (override) {
                string padding = !conditions.empty() ? "    " : "";
                output.push_back(padding += term.print_override_);
                continue;
            }


            // add comments
            string comment = term.make_comments();
            if (!comment.empty() && !override) {

                if (is_declaration) comment = term.make_comments(true);
                else comment.insert(0, "\n"); // add newline to the beginning of the comment

                // replace all '\"' with '' in comment
                size_t pos = 0;
                while ((pos = comment.find('\"', pos)) != string::npos) {
                    comment.replace(pos, 1, "");
                    pos += 1;
                }

                // replace all "\n" with "\n    " in comment
                if (!conditions.empty()) {
                    pos = 0;
                    while ((pos = comment.find('\n', pos)) != string::npos) {
                        comment.replace(pos, 1, "\n    ");
                        pos += 5;
                    }
                }

                output.push_back(comment); // add comment
            }

            // get string representation of term

            string term_string;
            if(!conditions.empty())
                term_string += "    ";

            term_string += term.str();


            // replace all "\n" with "\n    " in term_string
            size_t pos = 0;
            while ((pos = term_string.find('\n', pos)) != string::npos) {
                term_string.replace(pos, 1, "\n    ");
                pos += 5;
            }

            output.push_back(term_string);
        }

        if (!closed_condition && !Term::make_einsum && !current_conditions.empty()) {
            // if the final condition was not closed, close it
            output.emplace_back("}");
        }

        return output;
    }

    string Equation::condition_string(set<string> &conditions) {

        // if no conditions, return empty string
        if (conditions.empty()) return "";

        string if_block;
        if (!Term::make_einsum) {
            if_block = "\nif (";
            for (const string &condition: conditions)
                if_block += "includes_[\"" + condition + "\"] && ";
            if_block.resize(if_block.size() - 4);
            if_block += ") {";
        } else {
            if_block = "\nif ";
            for (const string &condition: conditions)
                if_block += "includes_[\"" + condition + "\"] and ";
            if_block.resize(if_block.size() - 5);
            if_block += ":";
        }
        return if_block;
    }

    void Equation::collect_scaling(bool regenerate) {

        if (terms_.empty()) return; // if no terms, return (nothing to do)

        // reset scaling maps
        flop_map_.clear(); // clear flop scaling map
        mem_map_.clear(); // clear memory scaling map

        for (auto & term : terms_) { // iterate over terms
            // compute scaling from term
            term.compute_scaling(regenerate);

            // collect scaling of terms
            scaling_map term_flop_map = term.flop_map(),
                        term_mem_map  = term.mem_map();

            flop_map_ += term_flop_map; // add flop scaling map
            mem_map_  += term_mem_map; // add memory scaling map
        }

        vector<ConstVertexPtr> all_term_linkages;
        for (auto & term : terms_) {
            if (term.term_linkage())
                all_term_linkages.push_back(term.term_linkage());
        }

        // if there are no terms with linkages, return
        if (all_term_linkages.empty()) return;

        // if only one term with a linkage, set eq_linkage_ to that linkage
        if (all_term_linkages.size() == 1) {
            eq_linkage_ = assignment_vertex_ * all_term_linkages.front();
            return;
        }

        // if there are multiple terms with linkages, set eq_linkage_ to the sum of all linkages
        eq_linkage_ = all_term_linkages.front();

        for (size_t i = 1; i < all_term_linkages.size(); ++i) {
            eq_linkage_ = eq_linkage_ + all_term_linkages[i];
        }

        // add assignment vertex to eq_linkage_
        eq_linkage_ = assignment_vertex_ * eq_linkage_;


    }

    void Equation::reorder(bool recompute) {
        // reorder rhs in term
        for (auto & term : terms_)
            term.reorder(recompute);

        // collect scaling of terms
        collect_scaling();
    }

    size_t Equation::substitute(const ConstLinkagePtr &linkage, bool allow_equality) {

        // check if linkage is more expensive than current bottleneck
        if (linkage->flop_scale() > worst_flop()) return 0;

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

        // check if linkage is more expensive than the current bottleneck
        if (linkage->flop_scale() > worst_flop()) {
            test_flop_map += flop_map_; // add flop scaling map for whole equation
            return 0; // return 0 substitutions
        }

        /// iterate over terms and substitute
        size_t num_terms = terms_.size();
        size_t num_subs = 0; // number of substitutions
        test_flop_map += flop_map_; // test memory scaling map
        for (int i = 0; i < num_terms; i++) {
            // skip term if linkage is not compatible
            if (!terms_[i].is_compatible(linkage)) continue;

            // get term copy
            Term term = terms_[i];

            // It's faster to subtract the old scaling and add the new scaling than
            // to recompute the scaling map from scratch
            test_flop_map -= term.flop_map(); // subtract flop scaling map for term

            // substitute linkage in term copy
            bool madeSub = term.substitute(linkage, allow_equality);

            // update test flop scaling map.
            if (!linkage->is_reused_ && !linkage->is_scalar()) {
                // reused intermediates will be pulled out of the term,
                // so we don't need to add the scaling
                test_flop_map += term.flop_map();
            } else {
                // if the linkage is reused, we need to ensure no negative scalings
                for (auto & [key, value] : term.flop_map()) {
                    if (value < 0) test_flop_map[key] = 0l;
                }
            }

            // increment number of substitutions if substitution was successful
            if (madeSub) ++num_subs; // increment number of substitutions

        } // substitute linkage in term copy

        return num_subs;
    }

    linkage_set Equation::make_all_links(bool compute_all) {

        linkage_set all_linkages(2048); // all possible linkages in the equations (start with large bucket n_ops)

        #pragma omp parallel for schedule(guided) shared(terms_, all_linkages) default(none) firstprivate(compute_all)
        for (auto & term : terms_) { // iterate over terms

            // skip term if it is optimal, and we are not computing all linkages
            if (!compute_all && term.generated_linkages_)
                continue;
            if (!term.is_optimal_ || term.needs_update_)
                term.reorder(); // reorder term if it is not optimal

            linkage_set &&term_linkages = term.make_all_links(); // generate linkages in term TODO: lazy evaluate
            all_linkages += term_linkages; // add linkages to the set of all linkages

            term.generated_linkages_ = true; // set term to have generated linkages

        } // iterate over terms

        return all_linkages;
    }

    void Equation::insert(const Term& term, int index) {
        if (index < 0) index = (int)terms_.size() + index + 1; // convert negative index to positive index from end
        terms_.insert(terms_.begin() + index, term); // add term to index of terms
    }

    void Equation::make_scalars(linkage_set &scalars, size_t &n_temps) {
        
        // term ids with scalars
        set<size_t> scalar_term_idxs;

        // iterate over terms
        for (size_t i = 0; i < terms_.size(); ++i) {
            Term &term = terms_[i]; // get term
            if (term.size() <= 2) continue; // skip terms with two or fewer operators

            // make scalars in term
            bool made_scalar = true;

            // current size of scalars
            size_t cur_size = scalars.size();

            while (made_scalar) {

                // make scalars in term
                made_scalar = term.make_scalar(scalars, n_temps);

                // add term to scalar_terms if a scalar was made
                if (made_scalar)
                    scalar_term_idxs.insert(i);

                // increment number of temps if new scalars were made
                size_t new_size = scalars.size();
                n_temps += new_size - cur_size;
                cur_size = new_size;
            } // eventually no more scalars will be made
        }
    }

    size_t Equation::merge_terms() {
        if (is_temp_equation_) return 0; // don't merge temporary equations

        size_t terms_size = terms_.size(); // number of terms before merging

        // split terms into groups by permutation type
        map<pair<perm_list, size_t>, vector<Term*>> perm_type_to_terms;
        for (auto &term : terms_) {
            perm_type_to_terms[{term.term_perms(), term.perm_type()}].push_back(&term);
        }

        // iterate over terms and accumulate similar terms
        vector<Term> new_terms; // new terms
        for (auto& [perm_type, terms] : perm_type_to_terms) {

            std::unordered_map<ConstLinkagePtr, pair<Term, double>, LinkageHash, LinkageEqual>
                    linkage_count; // map of term linkages to their associated terms and counts

            for (auto &term : terms) {
                // if term does not have a linkage, skip it
                if (!term->term_linkage()) {
                    new_terms.push_back(*term);
                    continue;
                }

                LinkagePtr term_link = as_link(term->term_linkage()->clone()); // get linkage of term
                term_link->tree_sort(); // sort linkage
                term_link = as_link(term->lhs() + term_link); // add lhs to linkage

                const auto & pos = linkage_count.find(term_link);
                if (pos == linkage_count.end()) {
                    linkage_count[term_link] = make_pair(*term, term->coefficient_);
                } else {
                    pos->second.second += term->coefficient_;
                    pos->second.first.original_pq_ = pos->second.first.original_pq_ + "@ " + term->original_pq_;
                }
            }

            // iterate over the map, adding each term to the new_terms vector
            for (auto &[linkage, unique_term_pair] : linkage_count) {
                auto &[new_term, new_coeff] = unique_term_pair; // get term

                new_term.coefficient_ = new_coeff; // set coefficient
                if (fabs(new_term.coefficient_) <= 1e-12) continue; // skip terms with zero coefficients

                // add term to new_terms
                new_terms.push_back(new_term);
            }

        }

        terms_ = new_terms;
        collect_scaling(true);

        return terms_size - terms_.size();
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
            bool is_temp = term[op_index]->is_temp(); // is vertex an intermediate? (cannot permute)

            bool vertex_swap_sign = false; // if true, this vertex has an odd number of permutations
            VertexPtr op = make_shared<Vertex>(term[op_index]->permute(perm_idx, vertex_swap_sign)); // permute vertex
            if (op->empty() || is_temp) { // if no more permutations are possible, or a vertex is temporary
                // reset the vertex and move to the next
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
                bool other_is_temp = term[other_op_index]->is_temp(); // is vertex an intermediate? (cannot permute)

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
            VertexPtr new_lhs_op = lhs_op->clone();
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

        // collect scaling
        collect_scaling(true);
    }

    vector<Term>::iterator Equation::insert_term(const Term &term, int index) {
        if (index < 0)
            index = (int)terms_.size() + index + 1; // convert negative index to positive index from the end
        return terms_.insert(terms_.begin() + index, term); // add term to index of terms
    }

    vector<Term *> Equation::get_temp_terms(const ConstLinkagePtr& contraction) {
        // for every term, check if this contraction is within the term (do not check lhs)
        vector<Term *> temp_terms;
        for (auto &term : terms_) {
            for (auto &op : term) {
                if (op->is_temp()) {
                    const ConstLinkagePtr &linkage = as_link(op);
                    if (linkage->id_ == contraction->id_ && linkage->is_reused_ == contraction->is_reused_
                        && *linkage == *contraction) {
                        temp_terms.push_back(&term);
                        break;
                    }
                }
            }
            if (term.lhs()->is_temp()) {
                const ConstLinkagePtr &linkage = as_link(term.lhs());
                if (linkage->id_ == contraction->id_ && linkage->is_reused_ == contraction->is_reused_
                    && *linkage == *contraction) {
                    temp_terms.push_back(&term);
                }
            }
        }
        return temp_terms;
    }

    Equation Equation::clone() const {
        Equation copy = *this;

        // deep copy terms
        copy.terms_.clear();

        for (const auto &term : terms_) {
            copy.terms_.push_back(term.clone());
        }

        return copy;
    }

    void Equation::sort_tmp_type(Equation &equation, char type) {

        // no terms, return
        if ( equation.terms().empty() ) return;

        // to sort the tmps while keeping the order of terms without tmps, we need to
        // make a map of the equation terms and their index in the equation and sort that (so annoying)
        std::vector<pair<Term*, size_t>> indexed_terms;
        size_t eq_size = equation.terms().size();
        indexed_terms.reserve(eq_size);
        for (size_t i = 0; i < eq_size; ++i)
            indexed_terms.emplace_back(&equation.terms()[i], i);

        // sort the terms by the maximum id of the tmps in the term, then by the index of the term

        auto is_in_order = [type](const pair<Term*, size_t> &a, const pair<Term*, size_t> &b) {

            const Term &a_term = *a.first;
            const Term &b_term = *b.first;

            size_t a_idx = a.second;
            size_t b_idx = b.second;

            const ConstVertexPtr &a_lhs = a_term.lhs();
            const ConstVertexPtr &b_lhs = b_term.lhs();

            typedef std::set<long, std::less<>> idset;

            // recursive function to get nested temp ids from a vertex
            std::function<idset(const ConstVertexPtr&)> test_vertex;
            test_vertex = [&test_vertex, type](const ConstVertexPtr &op) {

                idset ids;
                if (op->is_temp()) {
                    ConstLinkagePtr link = as_link(op);
                    long link_id = link->id_;

                    bool insert_id;
                    insert_id  = type == 't' && !link->is_scalar() && !link->is_reused_; // only non-scalar temps
                    insert_id |= type == 'r' &&  link->is_reused_; // only reuse tmps
                    insert_id |= type == 's' &&  (link->is_reused_ || link->is_scalar()); // only scalars

                    if (insert_id)
                        ids.insert(link_id);

                    // recurse into nested temps
                    for (const auto &nested_op: link->link_vector()) {
                        idset sub_ids = test_vertex(nested_op);
                        ids.insert(sub_ids.begin(), sub_ids.end());
                    }
                }

                return ids;
            };

            // get min id of temps from lhs
            auto get_lhs_id = [&test_vertex](const Term &term) {
                return test_vertex(term.lhs());
            };

            // get min id of temps from rhs
            auto get_rhs_id = [&test_vertex](const Term &term) {

                idset ids;
                for (const auto &op: term.rhs()) {
                    idset sub_ids = test_vertex(op);
                    ids.insert(sub_ids.begin(), sub_ids.end());
                }
                return ids;
            };

            // get all ids from lhs and rhs
            idset a_lhs_ids = get_lhs_id(a_term), a_rhs_ids = get_rhs_id(a_term);
            idset b_lhs_ids = get_lhs_id(b_term), b_rhs_ids = get_rhs_id(b_term);

            // get total ids
            idset a_total_ids = a_lhs_ids, b_total_ids = b_lhs_ids;
            a_total_ids.insert(a_rhs_ids.begin(), a_rhs_ids.end());
            b_total_ids.insert(b_rhs_ids.begin(), b_rhs_ids.end());

            // get number of ids
            bool a_has_temp = !a_lhs_ids.empty() || !a_rhs_ids.empty();
            bool b_has_temp = !b_lhs_ids.empty() || !b_rhs_ids.empty();

            // keep terms without temps first and if both have no temps, keep order
            if (a_has_temp ^ b_has_temp) return !a_has_temp;
            else if (!a_has_temp)        return a_idx < b_idx;

            // keep in lexicographical order of ids
            if (a_total_ids != b_total_ids)
                return a_total_ids < b_total_ids;

            // if lhs ids are empty, ignore assignment
            if (a_lhs_ids.empty() && b_lhs_ids.empty())
                return a_idx < b_idx;

            // if ids are the same, ensure assignment is first
            if (a_term.is_assignment_ ^ b_term.is_assignment_)
                return a_term.is_assignment_;

            // keep in order of lhs ids
            if (a_lhs_ids != b_lhs_ids)
                return a_lhs_ids < b_lhs_ids;

            // keep in order of rhs ids
            if (a_rhs_ids != b_rhs_ids)
                return a_rhs_ids < b_rhs_ids;

            // preserve order if all else is equal
            return a_idx < b_idx;
        };

        stable_sort(indexed_terms.begin(), indexed_terms.end(), is_in_order);

        // replace the terms in the equation with the sorted terms
        std::vector<Term> sorted_terms;
        sorted_terms.reserve(indexed_terms.size());
        for (const auto &indexed_term : indexed_terms) {
            sorted_terms.push_back(*indexed_term.first);
        }

        equation.terms() = sorted_terms;
    }

    void Equation::rearrange(char type) {

        // sort by conditionals, then by permutation type, then by number of operators, then by cost.
        // This is a stable sort, so the order of terms with the same cost will be preserved.
        std::stable_sort(terms_.begin(), terms_.end(), [](const Term &a, const Term &b) {

            if (a.conditions() != b.conditions())
                return a.conditions() < b.conditions();

            // sort by permutation type
            if (a.perm_type() != b.perm_type())
                return a.perm_type() < b.perm_type();

            // sort by number of permutations
            if (a.perm_type() != 0) {
                return a.term_perms() < b.term_perms();
            }

            // sort by number of operators
            if (a.size() != b.size())
                return a.size() < b.size();

            // sort by cost
            return *a.term_linkage() < *b.term_linkage();

        });

        // lastly, sort the terms by the maximum id of the tmps type in the term, then by the index of the term
        sort_tmp_type(*this, type);
    }


} // pdaggerq
