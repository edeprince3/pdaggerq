//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: consolidate.cc
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

#include <memory>

#include "../include/pq_graph.h"
#include "iostream"

// include omp only if defined
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_set_num_threads(n) 1
#endif

using std::ostream, std::string, std::vector, std::map, std::unordered_map, std::shared_ptr, std::make_shared,
        std::set, std::unordered_set, std::pair, std::make_pair, std::to_string, std::invalid_argument,
        std::stringstream, std::cout, std::endl, std::flush, std::max, std::min;

using namespace pdaggerq;


size_t PQGraph::prune(bool keep_single_use) {

    if (opt_level_< 5)
        return 0; // do not remove unused temps if pruning is disabled

    print_guard guard;
    if (print_level_ < 2) {
        guard.lock();
    }

    // remove unused contractions (only used in one term and its assignment)

    // get all temps in the equations
    linkage_set all_temp_set; all_temp_set.reserve(10*(saved_linkages_["temp"].size()+1));
    for (auto & [name, eq] : equations_) {
        for (auto &term: eq.terms()) {
            vertex_vector term_temps = (term.lhs() + term.term_linkage())->get_temps();
            all_temp_set.insert(term_temps.begin(), term_temps.end());
        }
    }

    // get all matching terms for each temp in saved_linkages
    linkage_map<pair<set<Term*>,set<Term*>>> matching_terms;
    matching_terms.reserve(all_temp_set.size());
    for (const auto &linkage : all_temp_set) {

        if (linkage->id() == -1) continue; // skip if id is -1
        auto [tmp_terms, tmp_decls] = get_matching_terms(linkage);
        if (tmp_terms.empty() && tmp_decls.empty()) continue; // occurs nowhere in the equations; skip

        matching_terms[linkage] = {tmp_decls, tmp_terms};
    }

    // remove temps that are used in only one term or are not used at all

    size_t num_removed = 0;
    linkage_set to_remove;
    for (const auto & [temp, terms_pair] : matching_terms) {

        auto [tmp_decl_terms, terms] = terms_pair;

        // remove (regardless of use) if never declared
        if (!tmp_decl_terms.empty()) {

            // count number of occurrences of the temp in the terms
            size_t num_occurrences = 0;
            for (auto &term: terms) {
                if (term->lhs() == nullptr) continue; // skip if term has no lhs (will be removed later)
                for (auto &vertex: term->rhs()) {
                    num_occurrences += as_link(vertex)->count(temp, false);
                }
            }

            // skip if temp is used at least once
            if (num_occurrences > 1) continue;
            else if (num_occurrences == 1) {
                // skip if temp is used only once and we want to keep single use temps
                if (keep_single_use) continue;

                // we always keep scalars
                if (temp->is_scalar()) continue;

                // we keep temps that are additions or have additions in them
                if (temp->is_addition() || temp->left()->is_addition() || temp->right()->is_addition()) continue;

                // we keep reused temps if it is used in an equation
                Term *used_term = *terms.begin();
                if (temp->is_reused() && !used_term->lhs()->is_temp()) continue;

            }
        }

        num_removed++;

        // set lhs to a null pointer to mark for removal
        if (!tmp_decl_terms.empty()) {
            for (auto &term: tmp_decl_terms) {
                term->lhs() = nullptr;
            }
        }

        // add to the list of temps to remove
        to_remove.insert(temp);
    }

    // remove all terms with lhs set to nullptr if any are found
    for (auto &[name, eq]: equations_) {
        vector<Term> new_terms;
        for (auto &term: eq.terms()) {
            if (term.lhs() != nullptr) {
                term.reorder(true);
                new_terms.push_back(term);
            }
        }
        eq.terms() = new_terms;
    }

    // get all terms in the equations
    vector<Term*> all_terms; all_terms.reserve(10*equations_.size());
    for (auto &[name, eq]: equations_) {
        for (auto &term: eq.terms()) {
            all_terms.push_back(&term);
        }
    }

    if (num_removed > 0) {

        // sort to_remove by decreasing id
        linkage_vector sorted_to_remove;
        sorted_to_remove.reserve(to_remove.size());
        sorted_to_remove.insert(sorted_to_remove.begin(), to_remove.begin(), to_remove.end());
        std::sort(sorted_to_remove.begin(), sorted_to_remove.end(), [](const LinkagePtr &a, const LinkagePtr &b) {
            // if types are different, sort by type
            if (a->type() != b->type()) return a->type() > b->type();
                // else sort by id for the same type
            else return a->get_ids(a->type()) > b->get_ids(b->type());
        });

        auto remove_unused = [&sorted_to_remove](VertexPtr vertex){
            // TODO: this lambda does not work with temps that have additions in them
            bool made_replacement = false;
            if (vertex->is_linked()) {
                for (auto &temp: sorted_to_remove) {
                    auto [new_vertex, replaced] = as_link(vertex)->replace_id(temp, -1);
                    if (replaced) {
                        vertex = new_vertex;
                        made_replacement = true;
                    }
                }
            }
            return make_pair(vertex, made_replacement);
        };

        cout << "Removing unused temps:" << endl;
        for (auto & temp : sorted_to_remove) {
            cout << "    " << temp->str() << endl;
        }

        // unset the temp in saved_linkages
        map<string, linkage_set> new_saved_linkages;
        for (auto &[type, linkages]: saved_linkages_) {
            for (const auto &link: linkages) {
                if (link->id() == -1) continue; // skip if id is -1

                auto [new_link, replaced] = remove_unused(link);
                if (new_link->id() != -1)
                    new_saved_linkages[type].insert(as_link(new_link));
            }
        }
        saved_linkages_ = new_saved_linkages;

        // unset the temp in all the terms
#pragma omp parallel for schedule(guided) shared(all_terms, remove_unused, sorted_to_remove) default(none)
        for (auto &term_ptr: all_terms) {
            Term &term = *term_ptr;
            bool made_replacement = false;

            // remove temps from the lhs
            if (term.lhs() != nullptr && term.lhs()->is_temp()) {
                auto [new_lhs, replaced] = remove_unused(term.lhs());

                // replace only if found and the temp is not removed
                if (replaced && new_lhs->is_temp()) {
                    term.lhs() = new_lhs;
                    made_replacement = true;
                }
            }

            // remove temps from the eq
            if (term.eq() != nullptr && term.eq()->is_temp()) {
                auto [new_eq, replaced] = remove_unused(term.eq());
                if (replaced && new_eq->is_temp()) {
                    term.eq() = new_eq;
                    made_replacement = true;
                }
            }

            // // remove temps from the rhs
            for (auto &op: term.rhs()) {
                if (op != nullptr && op->is_linked()) {
                    auto [new_op, replaced] = remove_unused(op);
                    if (replaced) {
                        op = new_op;
                        made_replacement = true;
                    }
                }
            }

            if (made_replacement) {
                term.request_update();
                term.reorder();
            }
        }

        // overwrite saved_linkages
        cout << endl; // print newline after all removals
    }

    if (opt_level_ >= 6) {
#pragma omp parallel for schedule(guided) default(none) shared(all_terms)
        for (Term *term_ptr: all_terms) {
            Term &term = *term_ptr;
            // factor the term linkage
            MutableLinkagePtr term_link = as_link(term.term_linkage()->shallow());
            if (!term_link->is_temp()) continue;
            else term_link->factor();

            term.expand_rhs(term_link);
            term.reorder(true);
        }
    }

    size_t num_removed_total = num_removed;
    while (num_removed > 0) {
        num_removed = prune(keep_single_use); // recursively prune until no more temps are removed
        num_removed_total += num_removed;
    }

    return num_removed_total;
}

pair<set<Term *>, set<Term*>> PQGraph::get_matching_terms(const LinkagePtr &intermediate) {
    // grab all terms with this tmp

    // initialize vector of term pointers
    set<Term*> tmp_terms;

    vector<string> eq_keys = get_equation_keys();
#pragma omp parallel for schedule(guided) default(none) shared(equations_, eq_keys, tmp_terms, intermediate)
    for (const auto& eq_name : eq_keys) { // iterate over equations in parallel
        // get equation
        Equation &equation = equations_[eq_name]; // get equation

        // get all terms with this tmp
        set<Term*> tmp_terms_local = equation.get_temp_terms(intermediate);
#pragma omp critical(InsertTmpTerms) // ensure thread-safe insertion into tmp_terms
        {
            // add terms to tmp_terms
            tmp_terms.insert(tmp_terms_local.begin(), tmp_terms_local.end());
        }

    }

    set<Term*> tmp_decl_terms;
    set<Term*> pruned_tmp_terms;
    for (auto &term : tmp_terms) {
        if (term->lhs()->same_temp(intermediate) && term->is_assignment_)
            tmp_decl_terms.insert(term);
        else pruned_tmp_terms.insert(term);
    }

    return {pruned_tmp_terms, tmp_decl_terms};
}

size_t PQGraph::merge_terms() {

    if (opt_level_< 5)
        return 0; // do not merge terms if not allowed

    print_guard guard;
    if (print_level_ < 2) {
        guard.lock();
    }

    // iterate over equations and merge terms
    size_t num_merged = 0;
    vector<string> eq_keys = get_equation_keys();
#pragma omp parallel for reduction(+:num_merged) default(none) shared(equations_, eq_keys)
    for (const auto &key: eq_keys) {
        Equation &eq = equations_[key];
        if (eq.is_temp_equation_) continue; // skip tmps equation

        num_merged += eq.merge_terms(); // merge terms with same rhs up to a permutation
    }
    collect_scaling(); // collect new scalings

    if (num_merged > 0) cout << "Merged " << num_merged << " terms" << endl;

    return num_merged;
}

Term& PQGraph::add_tmp(const LinkagePtr& precon, Equation &equation, double coeff) {
    // make term with tmp
    equation.terms().insert(equation.end(), Term(precon, coeff));
    return equation.terms().back();
}

void PQGraph::forget() {
    // forget linkage history of all linkages
    for (auto &linkage: all_links_)
        linkage->forget(true);

    // forget all linkages in equations
    for (auto & [name, eq] : equations_) {
        for (auto &term: eq.terms()) {
            if (term.lhs()->is_linked()) as_link(term.lhs())->forget(true);
            for (auto &op: term.rhs())
                if (op->is_linked()) as_link(op)->forget(true);
        }
    }

    // forget all saved linkages
    for (auto & [type, linkages] : saved_linkages_) {
        for (auto &linkage: linkages)
            linkage->forget(true);
    }

}

PQGraph PQGraph::clone() const {
    // make initial copy
    PQGraph copy = *this;

    // copy equations and make deep copies of terms
    map<string, Equation> copy_equations;
    for (auto & [name, eq] : equations_) {
        copy_equations[name] = eq.clone();
    }
    copy.equations_ = copy_equations;

    // copy all linkages
    map<string, linkage_set> copy_saved_linkages;
    for (const auto & [type, linkages] : saved_linkages_) {
        linkage_set new_linkages;
        for (const auto & linkage : linkages) {
            LinkagePtr link = as_link(linkage->clone());
            new_linkages.insert(link) ;
        }
        copy_saved_linkages[type] = new_linkages;
    }
    copy.saved_linkages_ = copy_saved_linkages;

    return copy;
}

void PQGraph::reindex() {

    print_guard guard;
    if (print_level_ <= 1)
        guard.lock();

    // reset saved linkages and temp counts
    saved_linkages_.clear();
    temp_counts_.clear();

    // search for all intermediates and update temp counts
    linkage_map<long> temp_map; // map to track found linkages to their current counts

    auto reindex_vertex = [this, &temp_map](VertexPtr &vertex) {
        if (vertex != nullptr && vertex->is_linked()) {
            // get temps and sort by id
            auto nested_temps = as_link(vertex)->get_temps();
            std::sort(nested_temps.begin(), nested_temps.end(), [](const VertexPtr &a, const VertexPtr &b) {
                return a->id() < b->id();
            });
            for (auto &temp : nested_temps) {
                // find temp in temp map
                auto it = temp_map.find(as_link(temp));
                long &temp_id = temp_map[as_link(temp)];
                if (it == temp_map.end()) {
                    temp_id = ++temp_counts_[temp->type()];
                }

                // replace temp id in lhs
                vertex = as_link(vertex)->replace_id(temp, temp_id).first;
            }
        }
    };

    // extract every term in every equation
    vector<Term*> term_ptrs = every_term();
    vector<Term> all_terms; all_terms.reserve(term_ptrs.size());
    for (auto &term_ptr : term_ptrs) {
        all_terms.emplace_back(*term_ptr);
    }

    // sort terms by type
    Equation::sort_tmp_type(all_terms, "scalar");
    Equation::sort_tmp_type(all_terms, "reused");
    Equation::sort_tmp_type(all_terms, "temp");

    // find last usage of each temp
    linkage_map<size_t> last_usage;
    size_t loc = 0;
    for (auto &term : all_terms) {
        auto found_temps = term.lhs()->get_temps(false);
        for (auto &op : term.rhs()) {
            auto rhs_temps = op->get_temps();
            found_temps.insert(found_temps.end(), rhs_temps.begin(), rhs_temps.end());
        }
        for (auto &temp : found_temps) {
            last_usage[as_link(temp)] = loc;
        }
        loc++;
    }

    // sort by last usage
    vector<pair<LinkagePtr, size_t>> last_usage_vec;
    for (auto & [link, pos] : last_usage) {
        last_usage_vec.emplace_back(link, pos);
    }
    std::sort(last_usage_vec.begin(), last_usage_vec.end(), [](const pair<LinkagePtr, size_t> &a, const pair<LinkagePtr, size_t> &b) {
        if (a.second != b.second) return a.second < b.second;
        return a.first->id() < b.first->id();
    });

    // reindex all temps
    for (auto & [link, _] : last_usage_vec) {
        VertexPtr new_temp = link->clone();
        reindex_vertex(new_temp);
    }

    // loop over all vertices in all equations and terms
    for (auto & [name, eq] : equations_) {
        eq.collect_scaling(true);

        // reindex all rhs first
        for (auto &term : eq.terms()) {
            for (auto &op : term.rhs())
                reindex_vertex(op);
        }

        // reindex all lhs
        for (auto &term : eq.terms()) {
            reindex_vertex(term.lhs());
        }

        // reindex all eq
        for (auto &term : eq.terms())
            reindex_vertex(term.eq());

        // recollect scaling
        eq.collect_scaling(true);
        eq.rearrange();
    }

    // add all temps in temp map to saved linkages
    vector<pair<long, string>> print_map;
    for (auto & [temp, id] : temp_map) {
        MutableLinkagePtr new_temp = as_link(temp->clone());
        new_temp->id() = id;
        saved_linkages_[new_temp->type()].insert(new_temp);
        stringstream ss;
        ss << temp->str() << " -> " << new_temp->str();

        print_map.emplace_back(id, ss.str());
    }

    // sort print map by id
    std::sort(print_map.begin(), print_map.end(), [](const pair<long, string> &a, const pair<long, string> &b) {
        return a.first < b.first;
    });

    // print reindexing
    if (!print_map.empty()) {
        cout << "Reindexed temps:" << endl;
        for (auto &[id, str]: print_map) {
            cout << "        " << str << endl;
        }
    }

    // reindex again for good measure
    static int reindex_count = 0;
    reindex_count = ++reindex_count % 3;
    if (reindex_count != 0)
        reindex();
}

size_t PQGraph::get_num_terms() const {
    size_t num_terms = 0;
    for (const auto & [name, eq] : equations_) {
        num_terms += eq.size();
    }
    return num_terms;
}
