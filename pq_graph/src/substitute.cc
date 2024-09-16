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

#include "../include/term.h"
#include "../include/equation.h"

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

    /// iterate over terms and substitute
    size_t num_terms = terms_.size();
    size_t num_subs = 0; // number of substitutions
    test_flop_map += flop_map_; // test memory scaling map
    for (int i = 0; i < num_terms; i++) {
        // skip term if linkage is not compatible
        if (!terms_[i].is_compatible(linkage)) continue;

        // get term copy
        Term term = terms_[i];
        term.term_linkage() = as_link(term.term_linkage()->clone()); // deep copy of term linkage

        // It's faster to subtract the old scaling and add the new scaling than
        // to recompute the scaling map from scratch
        test_flop_map -= term.flop_map(); // subtract flop scaling map for term

        // substitute linkage in term copy
        bool madeSub = term.substitute(linkage, allow_equality);
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
        const auto &lhs_link = as_link(lhs_);

        // do not allow substitution to intermediates with smaller ids unless they are different types
        if(lhs_link->id_ <= linkage->id_) {
            if (lhs_link->is_reused() == linkage->is_reused() &&
                lhs_link->is_scalar() == linkage->is_scalar()) return false;
        }

        // do not allow substitution of scalar intermediates with non-scalar linkages
        if (lhs_link->is_scalar() && (!linkage->is_scalar() || !linkage->is_reused())) return false;
    }

    // get total vector of linkage vertices (without expanding nested linkages)
    vector<ConstVertexPtr> link_list = linkage->link_vector();
    vector<ConstVertexPtr> term_list = rhs_;

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
                                      return a->base_name_ < b->base_name_;
                                  });

    // return true if all linkages are found in the term
    return all_found;

}

bool Term::substitute(const ConstLinkagePtr &linkage, bool allow_equality) {

    if (rhs_.empty())
        return false;

    // recompute the flop and memory cost of the term if necessary
    compute_scaling();

    // break out of loops if a substitution was made
    bool madeSub = false; // initialize boolean to track if substitution was made

    // generate every permutation of the term
    const vector<ConstLinkagePtr> &graph_perms = term_linkage()->permutations();

    // iterate over all possible orderings of vertex subsets
    ConstLinkagePtr best_linkage = term_linkage();
    for (const auto &graph_perm : graph_perms) {
        // substitute the linkage in the permutation (if possible)
        graph_perm->forget();
        auto [found_linkage, found] = graph_perm->find_link(linkage);
        if (!found) continue; // skip if linkage is not found in permutation
        else madeSub = true;

        // otherwise, make the substitution
        VertexPtr new_link = found_linkage->shallow();
        as_link(new_link)->copy_misc(linkage);
        auto [new_term_linkage, replaced] = graph_perm->replace(found_linkage, new_link);

        // create the best permutation of the substitution and break
        if (replaced) {
            best_linkage = as_link(new_term_linkage)->best_permutation();
            break;
        }
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

void Equation::make_scalars(linkage_set &scalars, long &n_temps) {

    // iterate over terms
    for (auto & term : terms_) {

        // make scalars in term
        bool made_any_scalar = true;
        while (made_any_scalar) {

            // make scalars in term
            auto [new_scalar, made_scalar] = term.make_scalar(scalars, n_temps);
            made_any_scalar = made_scalar;

            // add term to scalar_terms if a scalar was made
            if (made_scalar)
                scalars.insert(new_scalar);
        } // eventually no more scalars will be made
    }
}

pair<ConstLinkagePtr,bool> Term::make_scalar(linkage_set &scalars, long &id) {

    if (rhs_.empty())
        return {nullptr, false}; // do nothing if term is empty

    // break out of loops if a substitution was made
    bool made_scalar = false; // initialize boolean to track if substitution was made

    vector<ConstVertexPtr> term_scalars = term_linkage()->find_scalars();
    if (term_scalars.empty()) return {nullptr, false}; // do nothing if no scalars are found

    ConstLinkagePtr new_linkage = as_link(term_linkage()->shallow());
    LinkagePtr new_scalar = nullptr;
    for (const auto& scalar : term_scalars) {
        if (scalar->is_temp()) continue; // skip if scalar is already a temp
        if (!scalar->is_linked()) continue; // skip if scalar is not linked

        // reorder scalar for the best permutation
        ConstLinkagePtr scalar_link = as_link(scalar)->best_permutation();
        new_scalar = as_link(scalar_link->shallow());

        // check if scalar is already in set of scalars for setting the id
        long new_id = id;
        auto scalar_pos = scalars.find(new_scalar);
        if (scalar_pos != scalars.end())
             new_id = scalar_pos->get()->id(); // if scalar is already in set of scalars, change the id
        else ++id; // if scalar is not in set of scalars, increment the id for the next scalar

        new_scalar->id_ = new_id;

        // replace scalar in the term linkage
        auto [subbed_linkage, replaced] = as_link(new_linkage)->replace(scalar, new_scalar);
        if (replaced) {
            new_linkage = as_link(subbed_linkage);
            made_scalar = true;
            break;
        }
    }

    // if a substitution was made, replace the linkage in the term
    if (made_scalar) {
        // replace the rhs with the best linkage (if it is a temp, we should not expand into a vector)
        expand_rhs(new_linkage);
        request_update(); // set flags for optimization
        compute_scaling(true); // recompute the flop and memory cost of the term
        return {new_scalar, made_scalar};
    } else {
        return {nullptr, made_scalar};
    }
}