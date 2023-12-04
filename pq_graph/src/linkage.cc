//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: linkage.cc
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
#include <iostream>
#include <memory>
#include <utility>
#include <cstring>
#include "../include/linkage.h"

namespace pdaggerq {

    /********** Constructors **********/

    inline Linkage::Linkage(const ConstVertexPtr &left, const ConstVertexPtr &right, bool is_addition) : Vertex(),
                            left_(left), right_(right) {


        if (!left->is_linked() && !right->is_linked()) {
            // a binary linkage of pure vertices is associative (left and right are interchangeable)
            // sort left and right vertices by name to prevent duplicates
            if (left->name() > right->name())
                std::swap(left_, right_);
        }

        // determine the depth of the linkage
        depth_  =  left_->depth();
        depth_ += right_->depth();

        // populate the flop/mem history
        flop_history_.reserve(depth_ + 1);
        mem_history_.reserve(depth_ + 1);
        all_vert_.reserve(depth_ + 1);
        partial_vert_.reserve(depth_ + 1);

        if (left_->is_linked()){

            ConstLinkagePtr left_link = as_link(left_);

            // set the worst scale from left
            worst_flop_ = left_link->worst_flop_;
            
            // insert flop/mem history from left
            flop_history_.insert(flop_history_.end(),
                                 left_link->flop_history_.begin(), left_link->flop_history_.end());
            mem_history_.insert(mem_history_.end(),
                                 left_link->mem_history_.begin(), left_link->mem_history_.end());

            // copy all_vert
            all_vert_.insert(all_vert_.end(), left_link->all_vert_.begin(), left_link->all_vert_.end());

            // if left is not a tmp, copy partial_vert
            if (!left_link->is_temp())
                partial_vert_.insert(partial_vert_.end(), left_link->partial_vert_.begin(), left_link->partial_vert_.end());
            // else just add left
            else partial_vert_.push_back(left_);

        } else {
            // left is a pure vertex
            // add left to all_vert and partial_vert
            all_vert_.push_back(left_);
            partial_vert_.push_back(left_);
        }


        if (right_->is_linked()){

            ConstLinkagePtr right_link = as_link(right_);

            // set the worst scale from right
            if (right_link->worst_flop_ > worst_flop_)
                worst_flop_ = right_link->worst_flop_;

            // insert flop/mem history from right
            flop_history_.insert(flop_history_.end(),
                                 right_link->flop_history_.begin(), right_link->flop_history_.end());
            mem_history_.insert(mem_history_.end(),
                                 right_link->mem_history_.begin(), right_link->mem_history_.end());

            // copy all_vert
            all_vert_.insert(all_vert_.end(), right_link->all_vert_.begin(), right_link->all_vert_.end());

            // if right is not a tmp, copy partial_vert
            if (!right_link->is_temp())
                partial_vert_.insert(partial_vert_.end(), right_link->partial_vert_.begin(), right_link->partial_vert_.end());
            // else just add right
            else partial_vert_.push_back(right_);

        } else {
            // right is a pure vertex
            // add right to all_vert and partial_vert
            all_vert_.push_back(right_);
            partial_vert_.push_back(right_);
        }

        is_addition_ = is_addition;

        // create hash for the name (should be unique and faster for comparisons)
        base_name_ = left_->name_ + "\t" + right_->name_;
        name_ = base_name_;

        // build internal and external lines with their index mapping
        set_links();
        
        // update worst scale
        if (flop_scale_ > worst_flop_)
            worst_flop_ = flop_scale_;
        
        // add flop/mem scale to history
        flop_history_.push_back(flop_scale_);
        mem_history_.push_back(mem_scale_);
        
    }

    /**
     * @class line_map
     * @brief A class representing a line population map.
     *
     * The line_map class is used to store line pointers and their corresponding
     * frequencies of occurrence in a table-like structure. It uses linear
     * probing to handle collisions.
     */
    struct line_map {

        // closest prime number to max no. line
        static constexpr uint_fast16_t nbins_ = 521;

        const Line *line_table[nbins_];
        uint_fast8_t idx_table[nbins_];
        bool       found_table[nbins_];

        // refer to line.hpp
        static inline LinePtrHash lineptr_hasher;
        static inline LinePtrEqual lineptr_equal;

        line_map() {
            // initialize occupieds
            memset(found_table, false, nbins_);
        }

        inline pair<bool, uint_fast8_t> operator()(const Line *lineptr,
                                                   uint_fast8_t line_idx) {

            // hasher returns 13 bit number so this is safe from overflow
            alignas(16) uint_fast16_t index = lineptr_hasher(lineptr) % nbins_;

            // if there is a collision, check if the lineptr is the same
            while (found_table[index] &&
                   !lineptr_equal(line_table[index], lineptr)) {

                // use linear probing to find the next available index
                index = 2 * (index + 1) % nbins_;
            }

            // create a new entry if the index is not occupied
            uint_fast8_t ret_idx;
            bool found = found_table[index];
            if (!found_table[index]) {
                line_table[index] = lineptr;
                idx_table[index] = line_idx;
                ret_idx = line_idx;
                found_table[index] = true;
            } else {
                ret_idx = idx_table[index];
                idx_table[index] = line_idx;
            }

            // return whether there was a match and the returned index
            return {found, ret_idx};
        }
    };

    inline void Linkage::set_links() {

        // clear internal and external lines and connections
        lines_.clear();
        connec_.clear();
        disconnec_.clear();

        // grab data from left and right vertices
        const uint_fast8_t left_size = left_->size();
        const uint_fast8_t right_size = right_->size();
        const uint_fast8_t total_size = left_size + right_size;

        const auto &left_lines = left_->lines();
        const auto &right_lines = right_->lines();

        // handle scalars

        if (left_size == 0 && right_size == 0) {
            // both vertices are scalars (no lines)
            set_properties();
            return;
        }
        if (left_size == 0) {
            // if left is a scalar, just use right_lines as linkage
            mem_scale_ = right_->shape_;
            flop_scale_ = right_->shape_;

            // update vertex members
            lines_ = right_lines;
            set_properties();
            return;
        }
        if (right_size == 0) {
            // if right is a scalar, just use left_lines as linkage
            mem_scale_ = left_->shape_;
            flop_scale_ = left_->shape_;
            lines_  = left_lines;

            // update vertex members
            set_properties();
            return;
        }

        // create a map of lines to their frequency and whether they have been
        // visited
        line_map line_populations;
        lines_.reserve(total_size);
        disconnec_.reserve(total_size);
        connec_.reserve(std::max(left_size, right_size));

        // populate right lines
        bool skip[right_size]; memset(skip, 0, right_size);
        for (uint_fast8_t right_idx = 0; right_idx < right_size; right_idx++) {
            auto [has_match, index] = line_populations(&right_lines[right_idx], right_idx);

            // get line
            const Line &right_line = right_lines[right_idx];
            if (has_match) {
                // this is a self-contraction; add to connection map and skip
                pair<uint_fast8_t, uint_fast8_t>
                    connec{left_size + index, left_size + right_idx};

                connec_.insert(std::lower_bound(
                                   connec_.begin(), connec_.end(),
                                   connec, std::less<>()), connec);

                // track index for building right external lines later
                skip[right_idx] = true;
                skip[index] = true;
            }
        }
        
        // populate left lines
        for (uint_fast8_t left_idx = 0; left_idx < left_size; left_idx++) {
            auto [is_match, index]
                = line_populations(&left_lines[left_idx], left_idx);

            // get the line
            const Line &left_line = left_lines[left_idx];

            // check if left line is internal if it is already in the map
            if (is_match) {
                // add to connections
                pair<uint_fast8_t, uint_fast8_t>
                    connec{left_idx, left_size + index};

                connec_.insert(std::lower_bound(
                                   connec_.begin(), connec_.end(),
                                   connec, std::less<>()), connec);
                
                skip[index] = true;
            } else {
                // this is an external line to be added to the linkage
                lines_.insert(
                    std::lower_bound( lines_.begin(), lines_.end(),
                                      left_line, line_compare()),
                                      left_line);

                // update mem scale
                mem_scale_ += left_line;

                // check if external line is a sigma or density fitting index
                if (left_line.sig_)    is_sigma_ = true;
                else if (left_line.den_) is_den_ = true;

                // add to left external indices
                disconnec_.insert(
                        std::lower_bound( disconnec_.begin(), disconnec_.end(),
                                          left_idx, std::less<>()), left_idx);
            }

            // update flop scale
            flop_scale_ += left_line;
        }

        // find external right lines
        for (uint_fast8_t right_idx = 0; right_idx < right_size; right_idx++) {
            if (skip[right_idx])
                continue;

            // this is an external line if not skipped
            const Line &right_line = right_lines[right_idx];

            // add to external lines
            // this is an external line to be added to the linkage
            lines_.insert(
                std::lower_bound( lines_.begin(), lines_.end(),
                                 right_line, line_compare()),
                right_line);

            // update flop/mem scale
            mem_scale_   += right_line;
            flop_scale_  += right_line;

            // check if external line is a sigma or density fitting index
            if (right_line.sig_)
                is_sigma_ = true;
            else if (right_line.den_)
                is_den_ = true;

            //  add to right external indices
            uint_fast8_t rid = right_idx + left_size;
            disconnec_.insert(
                std::lower_bound( disconnec_.begin(), disconnec_.end(),
                                 rid, std::less<>()), rid);
        }

        // update vertex members
        set_properties();
    }

    void Linkage::set_properties() {
        // set properties
        rank_  = lines_.size();
        shape_ = mem_scale_;
        has_blk_ = left_->has_blk_ || right_->has_blk_;
        is_sigma_ = left_->is_sigma_ || right_->is_sigma_ || shape_.L_ > 0;
        is_den_ = left_->is_den_ || right_->is_den_ || shape_.Q_ > 0;
    }

    set<Line> Linkage::int_lines() const {
        set<Line> int_lines;

        // add left and right to set
        for (const auto &line :  left_->lines()) int_lines.insert(line);
        for (const auto &line : right_->lines()) int_lines.insert(line);

        return int_lines;
    }

    LinkagePtr Linkage::link(const vector<ConstVertexPtr> &op_vec) {
        uint_fast8_t op_vec_size = op_vec.size();

        if (op_vec_size == 0) {
            throw invalid_argument("Linkage::link(): op_vec must have at least two elements");
        } else if (op_vec_size == 1) {
            // this is a hack to allow for the creation of a LinkagePtr from a single VertexPtr
            // TODO: find a better way to do this
            return as_link(make_shared<Vertex>("") * op_vec[0]);
        }

        VertexPtr linkage = op_vec[0] * op_vec[1];
        for (uint_fast8_t i = 2; i < op_vec_size; i++)
            linkage = linkage * op_vec[i];

        return as_link(linkage);
    }

    vector<LinkagePtr> Linkage::links(const vector<ConstVertexPtr> &op_vec){
        uint_fast8_t op_vec_size = op_vec.size();
        if (op_vec_size <= 1) {
            throw invalid_argument("Linkage::link(): op_vec must have at least two elements");
        }

        vector<LinkagePtr> linkages(op_vec_size - 1);

        VertexPtr linkage = op_vec[0] * op_vec[1];
        linkages[0] = as_link(linkage);
        for (uint_fast8_t i = 2; i < op_vec_size; i++) {
            linkage = linkage * op_vec[i];
            linkages[i - 1] = as_link(linkage);
        }

        return linkages;
    }

    Linkage::Linkage() : Vertex(), left_(nullptr), right_(nullptr) {
        id_ = -1;
        flop_scale_ = shape();
        mem_scale_ = shape();
    }

    /****** operator overloads ******/

    bool Linkage::operator==(const Linkage &other) const {

        // check if both linkage are empty or not
        if (empty()) return other.empty();

        // check if linkage type is the same
        if (is_addition_ != other.is_addition_) return false;

        // check the depth of the linkage
        if (depth_ != other.depth_) return false;

        // check if left and right vertices are linked in the same way
        if ( left_->is_linked() !=  other.left_->is_linked()) return false;
        if (right_->is_linked() != other.right_->is_linked()) return false;

        // check that scales are equal
        if (flop_history_ != other.flop_history_) return false;
        if (mem_history_  !=  other.mem_history_) return false;

        // check linkage maps
        if (disconnec_ !=  other.disconnec_) return false;
        if (connec_    !=  other.connec_)    return false;

        // recursively check if left linkages are equivalent
        if (left_->is_linked()) {
            if (*as_link(left_) != *as_link(other.left_)) return false;
        } else {
            if ( !left_->equivalent( *other.left_)) return false;
        }

        // check if right linkages are equivalent
        if (right_->is_linked()) {
            if (*as_link(right_) != *as_link(other.right_))
                return false;
        } else {
            if ( !right_->equivalent( *other.right_))
                return false;
        }

        // if all tests pass, return true
        return true;
    }

    // repeat code from == operator, but invert the logic to end recursion early if possible
    bool Linkage::operator!=(const Linkage &other) const {
        return !(*this == other);
    }

    pair<bool, bool> Linkage::permuted_equals(const Linkage &other) const {
        // first test if the linkages are equal
        if (*this == other) return {true, false};

        // test if the linkages have the same number of vertices
        if (depth_ != other.depth_) return {false, false};

        // extract total vector of vertices
        const vector<ConstVertexPtr> &this_vert = get_vertices();
        const vector<ConstVertexPtr> &other_vert = other.get_vertices();

        // check if the vertices are isomorphic and keep track of the number of permutations
        bool swap_sign = false;
        for (size_t i = 0; i < depth_; i++) {
            bool odd_perm = false;
            bool same_to_perm = is_isomorphic(*this_vert[i], *other_vert[i], odd_perm);
            if (!same_to_perm) return {false, false};
            if (odd_perm) swap_sign = !swap_sign;
        }

        // if the linkages are isomorphic, return true and if the permutation is odd
        return {true, swap_sign};

    }


    string Linkage::str(bool make_generic, bool include_lines) const {

        if (!is_temp()) { // TODO: this might be annoying if we want to reuse a tmp. We will see when we get there...
            // this is not an intermediate vertex (generic linkage).
            // return the str of the left and right vertices
            return tot_str(false, true);
        }

        if (!make_generic) return str();

        // prepare output string as a map of tmps (or reuse_tmps) to a generic name
        string generic_str = is_reused_ ? "reuse_tmps_[\"" : "tmps_[\"";

        // add dimension string
        generic_str += dimstring();
        generic_str += "_";

        // format for scalars
        if (is_scalar())
            generic_str = "scalars_[\"";


        // use id_ to create a generic name
        if (id_ >= 0)
            generic_str += to_string(id_);

        generic_str += "\"]";

        if (include_lines) // if lines are included, add them to the generic name (default)
            generic_str += line_str();

        // create a generic vertex that has the same lines as this linkage.
        // this adds the spin and type strings to name
        // return its string representation
        return generic_str;
    }

    string Linkage::tot_str(bool expand, bool make_dot) const {

        if (empty()) return "";

        // do not expand linkages that are not intermediates
        if (!is_temp()) expand = false;

        // prepare output string
        string output, left_string, right_string;

        // build left string representation recursively
        if (left_->is_linked() && expand) left_string = as_link(left_)->tot_str(expand, make_dot);
        else left_string = left_->str();

        // build right string representation recursively
        if (right_->is_linked() && expand) right_string = as_link(right_)->tot_str(expand, make_dot);
        else right_string = right_->str();
        
        
        if (!is_addition_) output = left_string + " * " + right_string;
        else { output = "(" + left_string + " + " + right_string + ")"; }


        // if rank == 0, all lines are internal; requires dot() function call
        if (rank() == 0 && !is_addition_ && make_dot) {
            // add 'dot(' after '='
            output = "dot(" + output;

            // find the last star in output
            size_t last_star = output.rfind(" * ");

            // find last ' * '; replace with ', '
            output.replace(last_star, 3, ", ");

            // add closing parenthesis
            output += ")";
        }

        return output;
    }

    vector<ConstVertexPtr> Linkage::get_vertices(bool regenerate, bool full_expand) const {
        return full_expand ? all_vert_ : partial_vert_;
    }

    void Linkage::clone_link(const Linkage &other) {
        // Lock the mutex for the scope of the function
        std::lock_guard<std::mutex> lock(mtx_);

        // call base class copy constructor
        this->Vertex::operator=(other);

        // fill linkage data (shallow copy, but should not be modified either way)
        left_  = other.left_;
        right_ = other.right_;

//        left_parent_ = other.left_parent_;
//        right_parent_ = other.right_parent_;

        id_ = other.id_;
        depth_ = other.depth_;

        connec_ = other.connec_;
        disconnec_ = other.disconnec_;

        worst_flop_   = other.worst_flop_;
        flop_scale_   = other.flop_scale_;
        flop_history_ = other.flop_history_;
        mem_scale_    = other.mem_scale_;
        mem_history_  = other.mem_history_;

        is_addition_ = other.is_addition_;
        is_reused_ = other.is_reused_;

        all_vert_ = other.all_vert_;
        partial_vert_ = other.partial_vert_;
    }

    Linkage::Linkage(const Linkage &other) {
        clone_link(other);
    }

    VertexPtr Linkage::deep_copy_ptr() const {
        LinkagePtr link_copy = make_shared<Linkage>(left_->deep_copy_ptr(), right_->deep_copy_ptr(), is_addition_);
        link_copy->copy_misc(*this);
        return link_copy;
    }

    Vertex Linkage::deep_copy() const {
        return *deep_copy_ptr();
    }

    Linkage &Linkage::operator=(const Linkage &other) {
        // check for self-assignment
        if (this == &other) return *this;
        else clone_link(other);

        return *this;
    }

    void Linkage::move_link(Linkage &&other) {
        // Lock the mutex for the scope of the function
        std::lock_guard<std::mutex> lock(mtx_);

        // call base class move constructor
        this->Vertex::operator=(other);

        // move linkage data
        left_ = std::move(other.left_);
        right_ = std::move(other.right_);

//        left_parent_ = std::move(other.left_parent_);
//        right_parent_ = std::move(other.right_parent_);

        id_ = other.id_;
        depth_ = other.depth_;

        connec_ = std::move(other.connec_);
        disconnec_ = std::move(other.disconnec_);

        worst_flop_   = std::move(other.worst_flop_);
        flop_scale_   = std::move(other.flop_scale_);
        flop_history_ = std::move(other.flop_history_);
        mem_scale_    = std::move(other.mem_scale_);
        mem_history_  = std::move(other.mem_history_);


        is_addition_ = other.is_addition_;
        is_reused_ = other.is_reused_;

        all_vert_ = std::move(other.all_vert_);
        partial_vert_ = std::move(other.partial_vert_);
    }

    Linkage::Linkage(Linkage &&other) noexcept {

        // call move constructor
        move_link(std::move(other));
    }

    Linkage &Linkage::operator=(Linkage &&other) noexcept {
        // check for self-assignment
        if (this == &other) return *this;
        else move_link(std::move(other));

        return *this;
    }

    extern VertexPtr operator*(const ConstVertexPtr &left, const ConstVertexPtr &right){
        return make_shared<Linkage>(left, right, false);
    }

    extern VertexPtr operator+(const ConstVertexPtr &left, const ConstVertexPtr &right){
        return make_shared<Linkage>(left, right, true);
    }

} // pdaggerq
