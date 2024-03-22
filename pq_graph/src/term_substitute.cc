//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: term_substitute.cc
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

using namespace pdaggerq;

// function that iterates over all possible orderings of vertex subsets and executes a lambda function on each ordering
void Term::operate_subsets(
        size_t n, // size of the set
        const std::function<void(const vector<size_t>&)>& op, // operation to perform on each subset
        const std::function<bool(const vector<size_t>&)>& valid_op, // operation to check if subset is valid
        const std::function<bool(const vector<size_t>&)>& break_perm_op, // operation to check if permutation should be broken
        const std::function<bool(const vector<size_t>&)>& terminate_op // operation to check if lambda should be terminated
) {

    // magic bit manipulation to get all 0->n permutations of n indices
    // https://www.geeksforgeeks.org/print-subsets-given-size-set/
    for (size_t i = 1; i < (1 << n); i++) {

        // build subset indices
        vector<size_t> subset;
        subset.reserve(n);

        for (size_t j = 0; j < n; j++) {
            if (i & (1 << j)) subset.push_back(j);
        }

        if (valid_op != nullptr) {
            if (!valid_op(subset)) continue; // skip invalid subsets based on user-defined function
        }

        // execute operation on each permutation of the current subset
        do {
            op(subset); // execute operation on current subset arrangement
            if (break_perm_op != nullptr) {
                if (break_perm_op(subset)) break; // break permutation loop if user-defined function returns true
            }
        } while (next_permutation(subset.begin(), subset.end()));

        if (terminate_op != nullptr) {
            if (terminate_op(subset)) return; // break subset loop if user-defined function returns true
        }
    }
}

linkage_set Term::make_all_links() const {

    if (rhs_.empty())
        return {}; // if constant, return an empty set of linkages

    // initialize vector of linkages
    linkage_set linkages;

    auto max_link = Term::max_depth_;
    const auto valid_op = [max_link](const  vector<size_t> &subset) {
        return subset.size() <= max_link && subset.size() > 1;
    };

    // iterate over all subsets
    shape bottleneck_flop = worst_flop();
    auto &rhs = rhs_;
    bool allow_nested = allow_nesting_;
    const auto op = [&allow_nested, &linkages, &bottleneck_flop, &rhs](const vector<size_t> &subset) {

        // extract subset vertices
        vector<ConstVertexPtr> subset_vec;
        subset_vec.reserve(subset.size());
        for (size_t j: subset) {
            ConstVertexPtr vertex = rhs[j];
            if (vertex->is_temp() && !allow_nested)
                return; // do not consider nested linkages
            else subset_vec.push_back(vertex);
        }

        // make linkages from subset vertices
        LinkagePtr this_linkage = Linkage::link(subset_vec);
        this_linkage->tree_sort(); // sort the linkage tree

        shape link_shape = this_linkage->shape_;
        size_t link_occ = (size_t) link_shape.oa_ + (size_t) link_shape.ob_;
        size_t link_vir = (size_t) link_shape.va_ + (size_t) link_shape.vb_;

        size_t max_occ = (size_t) max_shape_.oa_ + (size_t) max_shape_.ob_;
        size_t max_vir = (size_t) max_shape_.va_ + (size_t) max_shape_.vb_;

        if (max_occ + max_vir > 0) { // user has defined a maximum size
            if (link_occ > max_occ || link_vir > max_vir)
                return; // skip linkages that are too large for the user-defined maximum
        }

        // add linkage to set if it has a scaling less than or equal to the bottleneck
        linkages.insert(this_linkage);

    };

    // fill the set with linkages
    operate_subsets(rhs_.size(), op, valid_op);

    // return the set of linkages
    return linkages;
}

bool Term::is_compatible(const ConstLinkagePtr &linkage) const {

    // if no possible linkages, return false
    if (size() <= 1) return false;


    if (lhs_->is_temp()){
        const auto &lhs_link = as_link(lhs_);

        // do not allow substitution to intermediates with smaller ids unless they are different types
        if(lhs_link->id_ <= linkage->id_) {
            if (lhs_link->is_reused_ == linkage->is_reused_ &&
                lhs_link->is_scalar() == linkage->is_scalar()) return false;
        }

        // do not allow substitution of scalar intermediates with non-scalar linkages
        if (lhs_link->is_scalar() && (!linkage->is_scalar() || !linkage->is_reused_)) return false;
    }


    // get total vector of linkage vertices (without expanding nested linkages)
    vector<ConstVertexPtr> link_list = linkage->link_vector();
    vector<ConstVertexPtr> term_list = term_linkage_->link_vector();

    // sort lists by name
    sort(link_list.begin(), link_list.end(), [](const ConstVertexPtr &a, const ConstVertexPtr &b) {
        return a->name_ < b->name_;
    });
    sort(term_list.begin(), term_list.end(), [](const ConstVertexPtr &a, const ConstVertexPtr &b) {
        return a->name_ < b->name_;
    });

    // check if all linkages are found in the term
    bool all_found = std::includes(term_list.begin(), term_list.end(), link_list.begin(), link_list.end(),
                                  [](const ConstVertexPtr &a, const ConstVertexPtr &b) {
                                      return a->name_ < b->name_;
                                  });

    // return true if all linkages are found in the term
    return all_found;

}

bool Term::substitute(const ConstLinkagePtr &linkage, bool allow_equality) {

    // no substitution possible for constant terms
    if (rhs_.empty())
        return false;

    // recompute the flop and memory cost of the term if necessary
    compute_scaling();

    // break out of loops if a substitution was made
    bool madeSub = false; // initialize boolean to track if substitution was made

    auto break_perm_op = [&madeSub](const vector<size_t> &subset) { return madeSub; };
    auto break_subset_op = break_perm_op;

    // initialize best flop scaling and memory scaling with rhs
    scaling_map best_flop_map = flop_map_;
    scaling_map best_mem_map = mem_map_;
    bool best_is_odd = false;
    vector<ConstVertexPtr> best_vertices = rhs_;

    vector<ConstVertexPtr> link_vec = linkage->link_vector();
    size_t num_link = link_vec.size(); // get number of linkages in rhs

    // sort link_vec by name
    sort(link_vec.begin(), link_vec.end(), [](const ConstVertexPtr &a, const ConstVertexPtr &b) {
        return a->name_ < b->name_;
    });

    /// determine if a linkage could possibly be substituted
    auto valid_op = [this, &link_vec, num_link](const vector<size_t> &subset) {

        size_t num_set = subset.size();

        // skip subsets with invalid sizes
        if (num_set != num_link || num_set > max_depth_ || num_set <= 1)
            return false;

        // build subset vertices
        vector<ConstVertexPtr> subset_vec;
        subset_vec.reserve(num_set);
        for (size_t i : subset) {
            // do not allow subsets with scalars
            if (rhs_[i]->is_scalar())
                return false;

            // insert by name order
            subset_vec.insert(
                    std::lower_bound(subset_vec.begin(), subset_vec.end(), rhs_[i],
                                     [](const ConstVertexPtr &a, const ConstVertexPtr &b) {
                                         return a->name_ < b->name_;
                                     }),
                    rhs_[i] // insert rhs vertex at correct position
            );
        }

        // sort subset_vec by name
        sort(subset_vec.begin(), subset_vec.end(), [](const ConstVertexPtr &a, const ConstVertexPtr &b) {
            return a->name_ < b->name_;
        });

        // ensure sorted subset vertices are equivalent to sorted linkage vertices
        for (size_t i = 0; i < num_set; i++) {
            if (subset_vec[i]->name_ != link_vec[i]->name_) {
                return false;
            }
        }

        // all tests passed
        return true;
    };

    // make copy of term to store new rhs and test scaling
    Term new_term(*this);

    // iterate over all possible orderings of vertex subsets
    auto op = [this, &new_term, &best_flop_map, &best_mem_map, &best_vertices, &best_is_odd,
               &madeSub, &linkage, allow_equality](const vector<size_t> &subset) {

        // build rhs from subset indices
        vector<ConstVertexPtr> subset_vec;
        subset_vec.reserve(subset.size());
        for (size_t j: subset)
            subset_vec.push_back(rhs_[j]);

        // make linkage from rhs with subset indices
        LinkagePtr this_linkage = Linkage::link(subset_vec);

        // skip if linkage is not equivalent to input linkage, up to permutation
//        auto [is_equiv, odd_parity] = this_linkage->permuted_equals(*linkage);
//        if (!is_equiv) return;

        bool odd_parity = false;
        if (*linkage != *this_linkage)
            return;

        // remove subset vertices from rhs
        vector<ConstVertexPtr> new_rhs;
        bool added_linkage = false;
        for (size_t i = 0; i < rhs_.size(); i++) {
            auto sub_pos = find(subset.begin(), subset.end(), i);
            if (sub_pos == subset.end()) {
                // add vertex to new rhs if it is not in the subset
                new_rhs.push_back(rhs_[i]);
            } else if (i == subset.front()) {
                // add linkage to new rhs if this is the first vertex in the subset
                this_linkage->copy_misc(linkage);
                new_rhs.push_back(this_linkage);
                added_linkage = true;
            } // else do not add vertex to new rhs
        }

        // skip if linkage was not added to new rhs
        if (!added_linkage) return;

        // create a copy of the term with the new rhs
        new_term.rhs_ = new_rhs;
        new_term.compute_scaling(true);
        scaling_map &new_flop = new_term.flop_map_;
        scaling_map &new_mem = new_term.mem_map_;

        // check if flop and memory scaling are better than the best scaling
        int scaling_comparison = new_flop.compare(best_flop_map);
        bool set_best = scaling_comparison == scaling_map::this_better;

        // check if we allow for substitutions with equal scaling
        if (allow_equality && !set_best) {
            set_best = scaling_comparison == scaling_map::is_same;
        }

        if (set_best) { // flop scaling is better
            // set the best scaling and rhs
            best_flop_map = new_flop;
            best_mem_map = new_mem;
            best_vertices = new_rhs;
            best_is_odd = odd_parity;

            madeSub = true;
        }
    };

    // iterate over all subsets of rhs and return the best flop scaling and rhs
    operate_subsets(size(), op, valid_op, break_perm_op, break_subset_op);

    // set rhs to the best rhs if a substitution was made
    if (madeSub) {
        rhs_ = best_vertices;
        request_update(); // set flags for optimization
        reorder();
        if (best_is_odd) coefficient_ *= -1;
    }

    // return a boolean indicating if a substitution was made
    return madeSub;
}

bool Term::make_scalar(linkage_set &scalars, size_t id) {

    if (rhs_.empty())
        return false; // do nothing if term is empty

    // break out of loops if a substitution was made
    bool made_scalar = false; // initialize boolean to track if substitution was made

    /// determine if a linkage could possibly be substituted
    auto valid_op = [this](const vector<size_t> &subset) {

        // skip subsets with invalid sizes
        if (subset.size() <= 1) return false;

        // make the linkage of the subset
        vector<ConstVertexPtr> subset_vec;
        subset_vec.reserve(subset.size());
        for (size_t i : subset)
            subset_vec.push_back(rhs_[i]);

        LinkagePtr this_linkage = Linkage::link(subset_vec);

        // subset is valid if the linkage is a scalar for the first permutation (any permutation will do)
        bool is_scalar = this_linkage->is_scalar();
        return is_scalar;
    };

    // initialize best flop scaling and memory scaling with rhs
    scaling_map best_flop_map = flop_map_;
    scaling_map best_mem_map = mem_map_;
    LinkagePtr  best_scalar;
    vector<ConstVertexPtr> best_vertices = rhs_;

    // make copy of term to store new rhs and test scaling
    Term new_term(*this);

    auto op = [this, &scalars, &made_scalar, &new_term, &best_scalar, &id, &best_flop_map, &best_mem_map, &best_vertices]
            (const vector<size_t> &subset) {

        // make the linkage of the subset
        vector<ConstVertexPtr> subset_vec;
        subset_vec.reserve(subset.size());
        for (size_t i : subset)
            subset_vec.push_back(rhs_[i]);
        LinkagePtr this_linkage = Linkage::link(subset_vec);

        // check if the linkage is a scalar
        bool is_scalar = this_linkage->is_scalar();

        vector<ConstVertexPtr> new_rhs;
        if (is_scalar) {

            // check if scalar is already in set of scalars
            auto scalar_pos = scalars.find(this_linkage);
            long new_id = (long) id;
            if (scalar_pos != scalars.end()) {
                // if scalar is already in set of scalars, change the id
                new_id = scalar_pos->get()->id_;
            }

            // get the id of the linkage
            this_linkage->id_ = (long) new_id;

            // remove subset vertices from rhs
            bool added_linkage = false;
            for (size_t i = 0; i < rhs_.size(); i++) {
                auto sub_pos = find(subset.begin(), subset.end(), i);
                if (sub_pos == subset.end()) {
                    // add vertex to new rhs if it is not in the subset
                    new_rhs.push_back(rhs_[i]);
                } else if (i == subset.front()) {
                    // add linkage to new rhs if this is the first vertex in the subset
                    // add the dot product to the front of the rhs
                    new_rhs.insert(new_rhs.begin(), this_linkage);
                    added_linkage = true;
                } // else do not add vertex to new rhs
            }
            if (!added_linkage) return; // skip if linkage was not added to new rhs
        } else return;  // if linkage does not match, continue

        new_term.rhs_ = new_rhs; // replace rhs with new rhs
        new_term.compute_scaling(true);

        // check if flop and memory scaling are better than best scaling
        if (new_term.flop_map_ <= best_flop_map || !made_scalar) { // flop scaling is better
            // set the best scaling and rhs
            best_flop_map = new_term.flop_map_;
            best_mem_map  = new_term.mem_map_;
            best_vertices = new_term.rhs_;
            best_scalar = this_linkage; // set scalar
            made_scalar = true;
        }
    };

    operate_subsets(rhs_.size(), op, valid_op);

    // reorder rhs
    if (made_scalar) {

        // check if scalar is already in set of scalars
        auto scalar_pos = scalars.find(best_scalar);
        long new_id = (long)id;
        if (scalar_pos != scalars.end()) {
            // if scalar is already in set of scalars, change the id
            new_id = scalar_pos->get()->id_;
        } else {
            // if scalar is not in set of scalars, add it
            scalars.insert(best_scalar);
        }

        // change the best vertices to the new id
        for (auto &v : best_vertices) {
            if (v->is_temp() && as_link(v)->id_ == id && v->is_scalar()) {
                // change id of scalar if it is a temp
                VertexPtr link = as_link(v)->clone();
                as_link(link)->id_ = new_id;
                v = link;
            }
        }

        // set new rhs
        rhs_ = best_vertices;
        request_update(); // set flags for optimization
        reorder(); // reorder the term
    }

    return made_scalar;
}