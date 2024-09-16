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

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#endif
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

    void Equation::insert(const Term& term, int index) {
        if (index < 0) index = (int)terms_.size() + index + 1; // convert negative index to positive index from end
        terms_.insert(terms_.begin() + index, term); // add term to index of terms
    }

    struct TermHash { // hash functor for finding similar terms
        size_t operator()(const Term& term) const {
            constexpr LinkageHash link_hasher;
            ConstVertexPtr total_representation = term.lhs() + term.term_linkage();

            return link_hasher(as_link(total_representation));
        }
    };

    struct TermEqual { // predicate functor for finding similar terms
        bool operator()(const Term& term1, const Term& term2) const {
            // determine if permutation is the same
            if (term1.perm_type()  != term2.perm_type()) return false;
            if (term1.term_perms() != term2.term_perms()) return false;

            // compare term linkages
            ConstLinkagePtr total_representation1 = as_link(term1.lhs() + term1.term_linkage(true));
            ConstLinkagePtr total_representation2 = as_link(term2.lhs() + term2.term_linkage(true));

            return *total_representation1 == *total_representation2;
        }
    };

    size_t Equation::merge_terms() {
        if (is_temp_equation_) return 0; // don't merge temporary equations

        size_t terms_size = terms_.size(); // number of terms before merging
        vector<Term> new_terms; // new terms


        // iterate over the map, adding each term to the new_terms vector
        // map of terms to their associated coefficients
        std::unordered_map<Term, double, TermHash, TermEqual> term_count;
        for (auto &term : terms_) {
            string term_str = term.str(); // get term string
            // check if term is in map
            auto it = term_count.find(term);
            if (it != term_count.end()) {
                string unique_term_str = it->first.str(); // get unique term string
                // if term is in map, increment coefficient
                it->second += term.coefficient_;

                // add original pq to unique term
                it->first.original_pq_ += Term::make_einsum ? "\n    # " : "\n    // ";
                it->first.original_pq_ += string(term.lhs()->name().size(), ' ');
                it->first.original_pq_ += " += " + term.original_pq_;
            } else {
                // if term is not in map, add term to map
                term_count[term] = term.coefficient_;
            }
        }


        for (auto &[unique_term, coeff] : term_count) {

            Term new_term = unique_term; // copy term
            new_term.coefficient_ = coeff; // set coefficient

            // skip terms with zero coefficients
            if (fabs(new_term.coefficient_) <= 1e-12)
                continue;

            // add term to new_terms
            new_terms.push_back(new_term);
        }


        terms_ = new_terms;
        collect_scaling(true);

        return terms_size - terms_.size();
    }

    set<Term *> Equation::get_temp_terms(const ConstLinkagePtr& linkage) {
        // for every term, check if this linkage is within the term (do not check lhs)
        set<Term *> temp_terms;
        for (auto &term : terms_) {
            bool found = false;
            for (auto &op : term) {
                // check if this term has the temp
                found = op->has_temp(linkage, false);
                if (found) { temp_terms.insert(&term); break; }
            }

            // check if the lhs is the same temp if it was not found in the rhs
            if (!found && term.lhs()->same_temp(linkage))
                temp_terms.insert(&term);
        }

        // return the terms that contain the temp
        return temp_terms;
    }

    Equation Equation::clone() const {
        Equation copy = *this;
        if (assignment_vertex_) copy.assignment_vertex_ = assignment_vertex_->clone(); // deep copy assignment vertex
        if (eq_linkage_) copy.eq_linkage_ = eq_linkage_->clone(); // deep copy eq_linkage

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

            typedef std::set<long> idset;

            auto [a_lhs_ids, a_rhs_ids, a_total_ids] = a_term.term_ids(type);
            auto [b_lhs_ids, b_rhs_ids, b_total_ids] = b_term.term_ids(type);


            // get number of ids
            bool a_has_temp = !a_lhs_ids.empty() || !a_rhs_ids.empty();
            bool b_has_temp = !b_lhs_ids.empty() || !b_rhs_ids.empty();

            // keep terms without temps first and if both have no temps, keep order
            if (a_has_temp != b_has_temp) return !a_has_temp;
            else if (!a_has_temp)        return a_idx < b_idx;

            // remove ids shared between a and b
            idset shared_ids;
            std::set_intersection(a_total_ids.begin(), a_total_ids.end(),
                                  b_total_ids.begin(), b_total_ids.end(),
                                  std::inserter(shared_ids, shared_ids.begin()));
            shared_ids.insert(-1); // add -1 to ignore unlinked vertices

            for (const auto &id: shared_ids) {
                a_total_ids.erase(id);
                b_total_ids.erase(id);
                a_lhs_ids.erase(id);
                b_lhs_ids.erase(id);
                b_lhs_ids.erase(id);
                a_rhs_ids.erase(id);
                b_rhs_ids.erase(id);
            }


            // if ids are the same, ensure assignment is first
            bool same_ids = a_total_ids == b_total_ids;
            if (same_ids && a_term.is_assignment_ != b_term.is_assignment_)
                return a_term.is_assignment_;
            else if (same_ids) {
                return a_idx < b_idx; // keep order if ids are the same
            } else if (a_term.is_assignment_ == b_term.is_assignment_ && a_term.is_assignment_) {
                // ensure that no rhs id in b is larger than the lhs id in a
                if (!a_rhs_ids.empty()) {
                    auto a_max_rhs = *a_rhs_ids.rbegin();
                    auto b_min_lhs = *b_lhs_ids.begin();
                    if (a_max_rhs > b_min_lhs) return false;
                }
                if (!b_rhs_ids.empty()) {
                    auto b_max_rhs = *b_rhs_ids.rbegin();
                    auto a_min_lhs = *a_lhs_ids.begin();
                    if (b_max_rhs > a_min_lhs) return true;
                }
            }

            // keep in lexicographical order of ids
            return a_total_ids < b_total_ids;

        };

        stable_sort(indexed_terms.begin(), indexed_terms.end(), is_in_order);

        // initialize map of lhs names
        std::set<std::string> lhs_name_map;

        // replace the terms in the equation with the sorted terms
        std::vector<Term> sorted_terms;
        sorted_terms.reserve(indexed_terms.size());
        for (const auto &indexed_term : indexed_terms) {
            // check if lhs is in the map
            ConstVertexPtr lhs = indexed_term.first->lhs();
            bool lhs_seen = lhs_name_map.find(lhs->name()) != lhs_name_map.end() && !lhs->is_temp();
            if (!lhs_seen) {
                lhs_name_map.insert(lhs->name());
                indexed_term.first->is_assignment_ = true;
            } else if (!lhs->is_temp()) {
                // lhs has already been used, so this is not an assignment
                indexed_term.first->is_assignment_ = false;
            } // else ignore temp assignments

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

            // sort by name of lhs
            if (a.lhs()->name() != b.lhs()->name())
                return a.lhs()->name() < b.lhs()->name();

            // sort by number of operators
            if (a.size() != b.size())
                return a.size() < b.size();

            // sort by cost
            return a.flop_map() < b.flop_map();

        });

        // lastly, sort the terms by the maximum id of the tmps type in the term, then by the index of the term
        sort_tmp_type(*this, type);
    }

} // pdaggerq
