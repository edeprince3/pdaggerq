//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: substitute.cc
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
#include <iostream>
#include <memory>
#include "../include/pq_graph.h"

using std::next_permutation;
using std::string;
using std::vector;
using std::map;
using std::pair;
using std::make_shared;
using std::shared_ptr;
using std::to_string;
using std::cout;
using std::flush;
using std::endl;
using std::max;

using namespace pdaggerq;

linkage_set Equation::make_all_links(bool compute_all) {

    linkage_set all_linkages(2048); // all possible linkages in the equations (start with large bucket n_ops)

#pragma omp parallel for schedule(guided) shared(terms_, all_linkages) default(none) firstprivate(compute_all)
    for (auto & term : terms_) { // iterate over terms

        // skip term if it is optimal, and we are not computing all linkages
        if (!compute_all && term.generated_linkages_)
            continue;

        term.reorder(); // reorder term (only if necessary)
        all_linkages += term.make_all_links(); // nerate linkages in term and add to the set of all linkages

        term.generated_linkages_ = true; // set term to have generated linkages

    } // iterate over terms

    return all_linkages;
}

linkage_set Term::make_all_links() const {

    if (rhs_.empty())
        return {}; // if constant, return an empty set of linkages

    // initialize vector of linkages
    linkage_set linkages;

    if (term_linkage()->is_temp()) return {}; // the term_linkage is already a temp, no need to test it.

    // generate all subgraphs of the term
    auto subgraphs = term_linkage()->subgraphs(Term::max_depth_);

    // insert all subgraphs of a given deoth into the set of linkages
    for (const auto &subgraph : subgraphs) {
        if (subgraph->shape_ > Term::max_shape_) continue; // skip if subgraph shape is too large
        if (subgraph->empty()) continue; // skip if subgraph is empty
        if (subgraph->is_temp()) continue; // the subgraph is already a temp, no need to test it.

        ConstLinkagePtr best_perm = subgraph->best_permutation(); // get best permutation of subgraph
        subgraph->forget();
        best_perm->forget(); // clear the history of the best permutation

        // insert the best subperm into the set of linkages
        linkages.insert(best_perm);
    }

    return linkages;
}

size_t Equation::substitute(const ConstLinkagePtr &linkage, bool allow_equality) {

    /// iterate over terms and substitute
    size_t num_terms = terms_.size();
    size_t num_subs = 0; // number of substitutions

    #pragma omp parallel for schedule(guided) shared(terms_, linkage) firstprivate(num_terms, allow_equality) \
                             reduction(+:num_subs) default(none)
    for (int i = 0; i < num_terms; i++) {
        Term &term = terms_[i]; // get term

        // check if linkage is compatible with term
        if (!term.is_compatible(linkage)) continue; // skip term if linkage is not compatible

        /// substitute linkage in term
        bool madeSub;
        madeSub = term.substitute(linkage);

        /// increment number of substitutions if substitution was successful
        if (madeSub) {
            ++num_subs;
            term.request_update(); // set term to be updated
        }
    } // substitute linkage in term

    return num_subs;
}

size_t Equation::test_substitute(const LinkagePtr &linkage, scaling_map &test_flop_map, bool allow_equality) {

    /// iterate over terms and substitute
    size_t num_terms = terms_.size();
    size_t num_subs = 0; // number of substitutions
    test_flop_map += flop_map_; // test memory scaling map
    for (int i = 0; i < num_terms; i++) {
        // skip term if linkage is not compatible
        if (!terms_[i].is_compatible(linkage)) continue;

        // get term copy
        Term term = terms_[i];
        term.term_linkage() = as_link(term.term_linkage()->shallow()); // deep copy of term linkage

        // It's faster to subtract the old scaling and add the new scaling than
        // to recompute the scaling map from scratch
        test_flop_map -= term.flop_map(); // subtract flop scaling map for term

        // substitute linkage in term copy
        bool madeSub = term.substitute(linkage);
        term.term_linkage()->forget(); // clear the linkage history for lazy evaluation
        test_flop_map += term.flop_map(); // add new flop scaling map for term

        // increment number of substitutions if substitution was successful
        if (madeSub) ++num_subs; // increment number of substitutions

    } // substitute linkage in term copy

    return num_subs;
}

bool Term::is_compatible(const ConstLinkagePtr &linkage) const {

    // if no possible linkages, return false
    if (rhs_.empty()) return false;

    if (lhs_->is_temp()){

        // do not allow substitution to intermediates with smaller ids unless they are different types
        if(lhs_->id() <= linkage->id()) {
            if (lhs_->type() == linkage->type()) return false;
        }

        // do not allow substitution of reused intermediates with non-reused intermediates
        if (lhs_->type() != "temp" && linkage->type() == "temp") return false;
    }

    // get total vector of linkage vertices (without expanding nested linkages)
    vector<ConstVertexPtr> link_list = linkage->link_vector();
    vector<ConstVertexPtr> term_list = term_linkage()->link_vector();

    // sort lists by name
    sort(link_list.begin(), link_list.end(), [](const ConstVertexPtr &a, const ConstVertexPtr &b) {
        return a->name_ < b->name_;
    });
    sort(term_list.begin(), term_list.end(), [](const ConstVertexPtr &a, const ConstVertexPtr &b) {
        return a->name_ < b->name_;
    });

    // check if all vertex names are found in the term
    bool all_found = std::includes(term_list.begin(), term_list.end(), link_list.begin(), link_list.end(),
                                  [](const ConstVertexPtr &a, const ConstVertexPtr &b) {
                                      return a->name_ < b->name_;
                                  });

    // return true if all linkages are found in the term
    return all_found;

}

bool Term::substitute(const ConstLinkagePtr &linkage) {

    if (rhs_.empty())
        return false;

    // recompute the flop and memory cost of the term if necessary
    compute_scaling();

    // break out of loops if a substitution was made
    bool madeSub = false; // initialize boolean to track if substitution was made

    // generate every permutation of the term
    const vector<ConstLinkagePtr> &graph_perms = term_linkage()->permutations();

    // iterate over all possible orderings of vertex subsets
    ConstLinkagePtr best_linkage = as_link(term_linkage()->shallow());
    for (const auto &graph_perm : graph_perms) {
        // substitute the linkage in the permutation (if possible)
        graph_perm->forget();
        auto matching_linkages = graph_perm->find_links(linkage);
        if (matching_linkages.empty()) continue; // skip if linkage is not found in permutation

        // otherwise, make the substitution for each matching linkage
        ConstLinkagePtr new_term_linkage = graph_perm;
        for (const auto &found_linkage : matching_linkages) {
            VertexPtr new_link = found_linkage->shallow();
            as_link(new_link)->copy_misc(linkage);
            new_term_linkage = as_link(new_term_linkage->replace(found_linkage, new_link).first);
        }

        new_term_linkage = as_link(new_term_linkage)->best_permutation();
        if (new_term_linkage->netscales().first > best_linkage->netscales().first) continue;

        // create the best permutation of the substitution and break
        best_linkage = new_term_linkage;
        madeSub = true;
        break;
    }

    // if a substitution was made, replace the linkage in the term
    if (madeSub) {
        // replace the rhs with the best linkage (if it is a temp or addition, we should not expand into a vector)
        expand_rhs(best_linkage);
        request_update(); // set flags for optimization
        compute_scaling(true); // recompute the flop and memory cost of the term
    }

    return madeSub;

}

void PQGraph::make_scalars() {

    cout << "Finding scalars..." << flush;
    // find scalars in all equations and substitute them
    linkage_set scalars = saved_linkages_["scalar"];
    for (auto &[name, eq]: equations_) {
        // do not make scalars in scalar equation
        if (name == "scalar") continue;
        eq.make_scalars(scalars, temp_counts_["scalar"]);
    }
    cout << " Done" << endl;

    // create new equation for scalars if it does not exist
    if (equations_.find("scalar") == equations_.end()) {
        equations_.emplace("scalar", Equation(make_shared<Vertex>("scalar"), {}));
        equations_["scalar"].is_temp_equation_ = true;
    }

    if (Equation::no_scalars_) {

        cout << "Removing scalars from equations..." << endl;

        // remove scalar equation
        equations_.erase("scalar");

        // remove scalars from all equations
        vector<string> to_remove;
        for (auto &[name, eq]: equations_) {
            vector<Term> new_terms;
            for (auto &term: eq.terms()) {
                bool has_scalar = false;
                for (auto &op: term.rhs()) {
                    if (op->is_linked() && op->is_scalar()) {
                        has_scalar = true;
                        break;
                    }
                }

                if (!has_scalar)
                    new_terms.push_back(term);
            }
            // if no terms left, remove equation
            if (new_terms.empty())
                to_remove.push_back(name);
            else
                eq.terms() = new_terms;
        }

        // remove equations
        for (const auto &name: to_remove) {
            cout << "Removing equation: " << name << " (no terms left after removing scalars)" << endl;
            equations_.erase(name);
        }

        // remove scalars from saved linkages
        scalars.clear();
    }


    vector<ConstLinkagePtr> scalars_vec(scalars.begin(), scalars.end());
    // sort by the id of the scalars
    sort(scalars_vec.begin(), scalars_vec.end(), [](const ConstLinkagePtr &a, const ConstLinkagePtr &b) {
        return a->id() < b->id();
    });

    // add new scalars to all linkages and equations
    for (const auto &scalar: scalars_vec) {
        // add term to scalars equation
        add_tmp(scalar, equations_["scalar"]);
        saved_linkages_["scalar"].insert(scalar);

        // print scalar
        cout << scalar->str() << " = " << *scalar << endl;
    }

    // remove comments from scalars
    for (Term &term: equations_["scalar"].terms())
        term.comments() = {}; // comments should be self-explanatory

    cout << endl;

    // collect scaling
    collect_scaling(true);
    is_assembled_ = true;
}

void Equation::make_scalars(linkage_set &scalars, long &n_temps) {

    // iterate over terms
    for (auto & term : terms_) {

        // make scalars in term
        bool made_scalar = true;
        while (made_scalar) {
            // make scalars in term
            made_scalar = term.make_scalars(scalars, n_temps);

        } // eventually no more scalars will be made
    }
}

bool Term::make_scalars(linkage_set &scalars, long &id) {

    if (rhs_.empty())
        return false; // do nothing if term is empty

    // break out of loops if a substitution was made
    bool made_scalar = false; // initialize boolean to track if substitution was made

    const vector<ConstLinkagePtr> &graph_perms = term_linkage()->permutations();
    linkage_map<linkage_set> term_scalars;
    for (const auto &graph_perm : graph_perms) {
        const auto perm_scalars = graph_perm->find_scalars();
        auto &perm_entry = term_scalars[graph_perm];
        for (const auto &scalar : perm_scalars) {
            if (!scalar->is_scalar()) continue; // skip if scalar is not actually a scalar (should not happen)
            if (scalar->is_temp()) continue;    // skip if scalar is already a temp
            if (!scalar->is_linked()) continue; // skip if scalar is not linked
            perm_entry.insert(scalar);
        }
    }
    if (term_scalars.empty()) return false; // do nothing if no scalars are found

    ConstLinkagePtr new_linkage = as_link(term_linkage()->shallow());
    for (const auto& [perm_linkage, perm_scalars] : term_scalars) {
        for (const auto &scalar : perm_scalars) {

            // reorder scalar for the best permutation
            ConstLinkagePtr scalar_link = as_link(scalar)->best_permutation();
            LinkagePtr new_scalar = as_link(scalar_link->shallow());

            // check if scalar is already in set of scalars for setting the id
            long new_id = id + 1;
            auto scalar_pos = scalars.find(new_scalar);
            if (scalar_pos != scalars.end())
                new_id = scalar_pos->get()->id(); // if scalar is already in set of scalars, change the id
            else ++id; // if scalar is not in set of scalars, increment the id for the next scalar

            new_scalar->id_ = new_id;

            // replace scalar in the term linkage
            auto [subbed_linkage, replaced] = as_link(perm_linkage)->replace(scalar, new_scalar);
            if (replaced) {
                new_linkage = as_link(subbed_linkage);
                scalars.insert(new_scalar); // insert scalar into set of scalars
                made_scalar = true;
                break;
            }
        }
        if (made_scalar) break;
    }

    // if a substitution was made, replace the linkage in the term
    if (made_scalar) {
        // replace the rhs with the best linkage (if it is a temp, we should not expand into a vector)
        expand_rhs(new_linkage);
        request_update(); // set flags for optimization
        compute_scaling(true); // recompute the flop and memory cost of the term
        return made_scalar;
    } else {
        return made_scalar;
    }
}