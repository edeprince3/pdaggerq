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

using namespace pdaggerq;

// function that iterates over all possible orderings of vertex subsets and executes a lambda function on each ordering
void Term::operate_subsets(
        size_t n, // size of the set
        const std::function<void(const vector<size_t>&)>& op, // operation to perform on each subset
        const std::function<bool(const vector<size_t>&)>& valid_op, // operation to check if subset is valid
        const std::function<bool(const vector<size_t>&)>& break_perm_op, // operation to check if permutation should be broken
        const std::function<bool(const vector<size_t>&)>& break_subset_op // operation to check if subset should be broken
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

        if (break_subset_op != nullptr) {
            if (break_subset_op(subset)) break; // break subset loop if user-defined function returns true
        }
    }
}

linkage_set Term::generate_linkages() const {

    if (rhs_.empty())
        return {}; // if constant, return an empty set of linkages

    // initialize vector of linkages
    linkage_set linkages;

    auto max_link = Term::max_depth;
    const auto valid_op = [max_link](const  vector<size_t> &subset) {
        return subset.size() <= max_link && subset.size() > 1;
    };

    // iterate over all subsets
    shape bottleneck_flop = worst_flop();
    auto &rhs = rhs_;
    bool allow_nested = allow_nesting_; //false;
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

        shape link_shape = this_linkage->shape_;
        size_t link_occ = (size_t) link_shape.o_.first + (size_t) link_shape.o_.second;
        size_t link_vir = (size_t) link_shape.v_.first + (size_t) link_shape.v_.second;

        size_t max_occ = (size_t) max_shape_.o_.first + (size_t) max_shape_.o_.second;
        size_t max_vir = (size_t) max_shape_.v_.first + (size_t) max_shape_.v_.second;

        if (max_occ + max_vir > 0) { // user has defined a maximum size
            if (link_occ > max_occ || link_vir > max_vir)
                return; // skip linkages that are too large for the user-defined maximum
        }

        // add linkage to set if it has a scaling less than or equal to the bottleneck
//        if (this_linkage->flop_scale() <= bottleneck_flop) {
            linkages.insert(this_linkage);
//        }

    };

    // fill the set with linkages
    operate_subsets(rhs_.size(), op, valid_op);

    // return the set of linkages
    return linkages;
}

bool Term::is_compatible(const ConstLinkagePtr &linkage) const {

    // if no possible linkages, return false
    if (size() <= 1) return false;

    // do not allow self substitution of intermediate linkages
    if (lhs_->is_temp() && as_link(lhs_)->id_ == linkage->id_)
        return false;

    // check if linkage is more expensive than the current bottleneck
//    if (linkage->flop_scale() > worst_flop()) return false;

//    TODO: nested temps do not work for some reason.


    // get total vector of linkage vertices (without expanding nested linkages)
    const list<ConstVertexPtr> &link_list = linkage->get_vertices(false, false);
    const list<ConstVertexPtr> &term_list = term_linkage_->get_vertices(false, false);

    size_t num_ops = term_list.size(); // get number of rhs
    size_t num_link = link_list.size(); // get number of linkages in rhs

    if (term_list.size() < link_list.size()) return false; // skip if number of vertices in rhs is less than linkage

    // create hash set for vertices
//    unordered_set<ConstVertexPtr, SimilarVertexPtrHash, SimilarVertexPtrEqual>
//            vert_set(num_ops + num_link);
//
//    // add rhs to hash set
//    for (const ConstVertexPtr &v : term_list)
//        vert_set.insert(v);
//
//    // add linkage vertices to hash set; if no equivalent vertex is found, return false
//    for (const ConstVertexPtr &v : link_list) {
//        if (vert_set.find(v) == vert_set.end()) {
//            return false;
//        }
//    }

    // if not all rhs were found, return false
    return true;

}

bool Term::substitute(const ConstLinkagePtr &linkage, bool allow_equality) {

    // no substitution possible for constant terms
    if (rhs_.empty())
        return false;

    // recompute the flop and memory cost of the term if necessary
    compute_scaling();

    // break out of loops if a substitution was made
    bool madeSub = false; // initialize boolean to track if substitution was made

    auto break_perm_op = [&](const vector<size_t> &subset) { return madeSub && !allow_equality; };
    auto break_subset_op = break_perm_op;

    // get vector of vertices involved in the linkage (without expanding nested linkages)
//    const list<ConstVertexPtr> &link_list = linkage->get_vertices(false, true);

    // initialize best flop scaling and memory scaling with rhs
    scaling_map best_flop_map = flop_map_;
    scaling_map best_mem_map = mem_map_;
    vector<ConstVertexPtr> best_vertices = rhs_;

    const list<ConstVertexPtr> &link_list = linkage->get_vertices(false, false);
    size_t num_link = link_list.size(); // get number of linkages in rhs

    /// determine if a linkage could possibly be substituted
    auto valid_op = [&](const vector<size_t> &subset) {

        // skip subsets with invalid sizes
        if (subset.size() > max_depth || subset.size() <= 1)
            return false;

        // make a linkage of the first permutation of the subset
        list<ConstVertexPtr> subset_list;
        for (size_t i : subset) {
            subset_list.push_back(rhs_[i]);
        }

        // get size of subset list
        size_t num_set = subset_list.size(); // get number of vertices in subset

        if (num_set != num_link) return false; // skip if number of vertices in subset is less than linkage

        //TODO: figure out why below doesn't work!

//        // check if each vertex in the subset has an equivalent vertex in the linkage
//        // create hash set for vertices
//        unordered_set<ConstVertexPtr, SimilarVertexPtrHash, SimilarVertexPtrEqual>
//            vert_set(num_set + num_link);
//
//        // add subset_list to hash set (return false if scalar is found)
//        for (const ConstVertexPtr &v : subset_list) {
//            vert_set.insert(v);
//        }
//
//        // add linkage vertices to hash set; if no equivalent vertex is found, return false
//        for (const ConstVertexPtr &v : subset_list) {
//            if (vert_set.find(v) == vert_set.end()) {
//                return false;
//            }
//        }

        // all tests passed
        return true;
    };

    // make copy of term to store new rhs and test scaling
    Term new_term(*this);

    // initialize new rhs
    vector<ConstVertexPtr> new_vertices;

    // iterate over all possible orderings of vertex subsets
    auto op = [&](const vector<size_t> &subset) {
        // build rhs from subset indices
        // TODO: test permutations of the lines in each vertex too
        new_vertices.clear();
        for (size_t j: subset)
            new_vertices.push_back(rhs_[j]);

        // make linkage from rhs with subset indices
        LinkagePtr this_linkage = Linkage::link(new_vertices);

        // skip if linkage is not equivalent to input linkage
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

        // check if flop and memory scaling are better than the best scaling
        int scaling_comparison = new_term.flop_map_.compare(best_flop_map);
        bool set_best = scaling_comparison == scaling_map::this_better;

        // check if we allow for substitutions with equal scaling
        if (allow_equality && !set_best) {
            set_best = scaling_comparison == scaling_map::is_same;
        }

        if (set_best) { // flop scaling is better
            // set the best scaling and rhs
            best_flop_map = new_term.flop_map_;
            best_mem_map = new_term.mem_map_;
            best_vertices = new_term.rhs_;
            madeSub = true;
        }
    };

    // iterate over all subsets of rhs and return the best flop scaling and rhs
    operate_subsets(size(), op, valid_op, break_perm_op, break_subset_op);

    // set rhs to the best rhs if a substitution was made
    if (madeSub) {
        rhs_ = best_vertices;
        request_update(); // set flags for optimization
        compute_scaling(true);
    }

    // return a boolean indicating if a substitution was made
    return madeSub;
}

LinkagePtr Term::make_dot_products(size_t id) {

    if (rhs_.empty()) return {}; // if constant, return empty linkage

    // break out of loops if a substitution was made
    bool madeSub = false; // initialize boolean to track if substitution was made
    LinkagePtr scalar; // initialize scalar to track if a scalar was found

    // break out of permutation loop if a scalar was found
    const auto break_perm_op = [&madeSub](const vector<size_t> &subset) {
        return madeSub;
    };

    // break out of subset loop if a scalar was found
    auto break_subset_op = [&scalar](const vector<size_t> &subset) {
        if (scalar == nullptr)
            return false;
        else return !scalar->empty();
    };

    /// determine if a linkage could possibly be substituted
    auto valid_op = [&](const vector<size_t> &subset) {

        // skip subsets with invalid sizes
        if (subset.size() <= 1) return false;
        else return true;

    };

    // initialize best flop scaling and memory scaling with rhs
    scaling_map best_flop_map = flop_map_;
    scaling_map best_mem_map = mem_map_;
    vector<ConstVertexPtr> best_vertices = rhs_;

    // make copy of term to store new rhs and test scaling
    Term new_term(*this);

    auto op = [&](const vector<size_t> &subset) {
        // extract subset vertices
        vector<ConstVertexPtr> subset_vec;
        subset_vec.reserve(subset.size());
        for (size_t i : subset)
            subset_vec.push_back(rhs_[i]);

        // make the linkage of the subset
        LinkagePtr this_linkage = Linkage::link(subset_vec);

        // check if the linkage is a scalar
        bool is_scalar = this_linkage->is_scalar();

        vector<ConstVertexPtr> new_rhs;
        if (is_scalar) {

            // set the id of the linkage
            this_linkage->id_ = (long) id;

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
        if (new_term.flop_map_ <= best_flop_map) { // flop scaling is better
            // set the best scaling and rhs
            best_flop_map = new_term.flop_map_;
            best_mem_map  = new_term.mem_map_;
            best_vertices = new_term.rhs_;
            scalar = this_linkage; // set scalar
            madeSub = true;
        }

    };

    operate_subsets(rhs_.size(), op, valid_op, break_perm_op, break_subset_op);

    // set new rhs
    rhs_ = best_vertices;

    // reorder rhs
    if (madeSub) {
        request_update(); // set flags for optimization
        reorder();
    }

    return scalar; // return scalar
}