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

    inline Linkage::Linkage(const ConstVertexPtr &left, const ConstVertexPtr &right, bool is_addition) : Vertex() {

        // set inputs
        left_ = left;
        right_ = right;

        if (!left->is_linked() && !right->is_linked()) {
            // a binary linkage of pure vertices is associative (left and right are interchangeable)
            // sort left and right vertices by name to prevent duplicates
            if (left->name() > right->name())
                std::swap(left_, right_);
        }

        // count_ the left and right vertices
        depth_ = 0;

        // determine the depth of the linkage
        if (left_->is_linked()){
            // add the number of vertices in left
            depth_ += as_link(left_)->depth_;
        } else depth_++;
        if (right_->is_linked()){
            // add the number of vertices in right
            depth_ += as_link(right_)->depth_;
        } else depth_++;

        is_addition_ = is_addition;

        // create hash for the name (should be unique and faster for comparisons)
        base_name_ = left_->name_;
        base_name_ += '\t';
        base_name_ += right_->name_;
        name_ = base_name_;

        // build internal and external lines with their index mapping
        set_links();

    }

    inline void Linkage::set_links() {

        // grab data from left and right vertices
        uint_fast8_t left_size = left_->size();
        uint_fast8_t right_size = right_->size();
        uint_fast8_t total_size = left_size + right_size;

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
            r_ext_idx_.reserve(right_size);
            for (uint_fast8_t i = 0; i < right_size; i++)
                r_ext_idx_.push_back(i);
            set_properties();
            return;
        }
        if (right_size == 0) {
            // if right is a scalar, just use left_lines as linkage
            mem_scale_ = left_->shape_;
            flop_scale_ = left_->shape_;
            lines_  = left_lines;

            l_ext_idx_.reserve(left_size);
            for (uint_fast8_t i = 0; i < left_size; i++)
                l_ext_idx_.push_back(i);

            // update vertex members
            set_properties();
            return;
        }

        // reserve lines for vertex
        lines_.reserve(total_size);
        int_connec_.reserve(total_size);
        l_ext_idx_.reserve(left_size);
        r_ext_idx_.reserve(right_size);

        // create a map of lines to their corresponding indicies
        unordered_map<const Line*, pair<char, char>, LinePtrHash, LinePtrEqual>
                line_populations(total_size);

        // populate left lines
        for (uint_fast8_t i = 0; i < left_size; i++) {
            auto &[left_id, right_id] = line_populations[&left_lines[i]];
            left_id = (char) i; // add index to left_id
            right_id = -1; // indicates that the line is not in the right
        }

        // populate right lines and track index
        for (uint_fast8_t i = 0; i < right_size; i++) {
            auto pos = line_populations.find(&right_lines[i]);

            // if we found right line from left, add to internal lines
            if (pos != line_populations.end()) {
                // line is in both left and right
                auto &[left_id, right_id] = pos->second;
                right_id = (char) i; // add index to right_id

            } else {
                // line is only in right
                auto &[left_id, right_id] = line_populations[&right_lines[i]];
                right_id = (char) i; // add index to right_id
                left_id = -1; // indicates that the line is not in the left
            }
        }

        // now we have a map of lines to their corresponding indices
        // populate data
        for (auto &[line_ptr, indices] : line_populations) {
            bool is_internal = indices.first >= 0 && indices.second >= 0;
            if (is_internal) {
                // add to int_connec_ in order
                pair<uint_fast8_t, uint_fast8_t> connec(indices.first, indices.second);
                int_connec_.insert(
                        std::upper_bound( int_connec_.begin(), int_connec_.end(), connec), connec);
            } else {
                // add to external lines
                if (indices.first >= 0) {
                    // add to left external lines (from upper bound)
                    lines_.insert(
                            std::upper_bound( lines_.begin(), lines_.end(), *line_ptr, line_compare()), *line_ptr);

                    l_ext_idx_.insert(
                            std::upper_bound( l_ext_idx_.begin(), l_ext_idx_.end(), indices.first), indices.first);
                } else if (indices.second >= 0) {
                    // add to right external lines (from lower bound)
                    lines_.insert(
                            std::lower_bound( lines_.begin(), lines_.end(), *line_ptr, line_compare()), *line_ptr);

                    r_ext_idx_.insert(
                            std::upper_bound( r_ext_idx_.begin(), r_ext_idx_.end(), indices.second), indices.second);
                } else {
                    // this should never happen
                    throw runtime_error("Linkage::set_links(): line not found in left or right");
                }

                // update mem scaling
                mem_scale_ += *line_ptr;
            }

            // update flop scaling
            flop_scale_ += *line_ptr;
        }

        // update vertex members
        set_properties();
    }

    void Linkage::set_properties() {
        // set properties
        rank_  = lines_.size();
        shape_ = shape(lines_);
        has_blk_ = left_->has_blk_ || right_->has_blk_;
        is_sigma_ = left_->is_sigma_ || right_->is_sigma_ || shape_.L_ > 0;
        is_den_ = left_->is_den_ || right_->is_den_ || shape_.Q_ > 0;
    }


    vector<Line> Linkage::int_lines() const {
        vector<Line> int_lines;
        size_t left_size = left_->size();
        size_t right_size = right_->size();

        // if both left and right are scalars, there are no internal lines
        if (left_size == 0 && right_size == 0)
            return int_lines;

        int_lines.reserve(left_size + right_size - lines_.size());

        // every internal line shows up in both the left and right vertices
        // so let's use the smaller, nonzero set of lines
        bool use_left = left_size != 0 && left_size <= right_size;

        const vector<Line> &ref_lines = use_left ? left_->lines() : right_->lines();

        // use int_connec_ to grab the internal lines
        for (const auto &[left_idx, right_idx] : int_connec_) {
            if (use_left) int_lines.push_back(ref_lines[left_idx]);
            else int_lines.push_back(ref_lines[right_idx]);
        }

        return int_lines;
    }

    LinkagePtr Linkage::link(const vector<ConstVertexPtr> &op_vec) {
        uint_fast8_t op_vec_size = op_vec.size();

        // cannot link less than two vertices
        if (op_vec_size <= 1)
            throw invalid_argument("Linkage::link(): op_vec must have at least two elements");


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

    tuple<ConstLinkagePtr, vector<shape>, vector<shape>> Linkage::link_and_scale(const vector<ConstVertexPtr> &op_vec) {
        uint_fast8_t op_vec_size = op_vec.size();
        if (op_vec_size == 0) {
            throw invalid_argument("link(): op_vec must have at least two elements");
        } else if (op_vec_size == 1) {
            ConstLinkagePtr linkage = as_link(make_shared<Vertex>() * op_vec[0]);
            return {linkage, {linkage->flop_scale_}, {linkage->mem_scale_}};
        }


        vector<shape> flop_list, mem_list;
        flop_list.reserve(op_vec_size - 1);
        mem_list.reserve(op_vec_size - 1);

        LinkagePtr linkage = as_link(op_vec[0] * op_vec[1]);
        flop_list.push_back(linkage->flop_scale_);
        mem_list.push_back(linkage->mem_scale_);

        for (uint_fast8_t i = 2; i < op_vec_size; i++) {
            linkage = as_link(linkage * op_vec[i]);
            flop_list.push_back(linkage->flop_scale_);
            mem_list.push_back(linkage->mem_scale_);
        }

        return {linkage, flop_list, mem_list};
    }

    Linkage::Linkage() {
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
        if ( left_->is_linked() ^  other.left_->is_linked()) return false;
        if (right_->is_linked() ^ other.right_->is_linked()) return false;

        // check that scales are equal
        if (flop_scale_ != other.flop_scale_) return false;
        if (mem_scale_  !=  other.mem_scale_) return false;

        // check linkage maps
        if (l_ext_idx_  !=  other.l_ext_idx_) return false;
        if (r_ext_idx_  !=  other.r_ext_idx_) return false;
        if (int_connec_ != other.int_connec_) return false;

        // recursively check if left linkages are equivalent
        if (left_->is_linked()) {
            if (*as_link(left_) != *as_link(other.left_)) return false;
        } else {
            if ( !left_->equivalent( *other.left_)) return false;
        }

        // check if right linkages are equivalent
        if (right_->is_linked()) {
            if (*as_link(right_) != *as_link(other.right_)) return false;
        } else {
            if ( !right_->equivalent( *other.right_)) return false;
        }

        if (!Vertex::equivalent(other))
            return false;

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
        const vector<ConstVertexPtr> &this_vert = vertices();
        const vector<ConstVertexPtr> &other_vert = other.vertices();

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

    inline void Linkage::to_vector(vector<ConstVertexPtr> &result, size_t &i, bool regenerate, bool full_expand) const {

        if (empty()) return;

        std::function<void(const ConstVertexPtr&, vector<ConstVertexPtr>&, size_t&)> expand_vertex;

        expand_vertex = [regenerate, full_expand, &expand_vertex]
                (const ConstVertexPtr& vertex, vector<ConstVertexPtr> &result, size_t &i) {

            if (vertex->base_name_.empty()) return;

            if (vertex->is_linked()) {
                const ConstLinkagePtr link = as_link(vertex);

                // check if left linkage is a tmp
                if (!full_expand && link->is_temp()) {
                    // if this is a tmp and we are not expanding, add it to the result and return
                    result[i++] = link;
                } else {

                    // compute the left vertices recursively and save them
                    for (const ConstVertexPtr &link_vertex: link->to_vector(regenerate, full_expand))
                        expand_vertex(link_vertex, result, i);
                }

            } else result[i++] = vertex;
        };

        // get the left vertices
        expand_vertex(left_, result, i);

        // get the right vertices
        expand_vertex(right_, result, i);
    }

    const vector<ConstVertexPtr> &Linkage::to_vector(bool regenerate, bool full_expand) const {

        // Lock the mutex for the scope of the function
        std::lock_guard<std::mutex> lock(mtx_);

        // if full_expand is false, we only need to expand the vertices that are
        // not tmps
        if (!full_expand) {
            // the vertices are not known
            if (link_vector_.empty() || regenerate) {
                // compute the vertices recursively and store the vertices in
                // all_vert_ for next query

                size_t i = 0;
                auto result = std::vector<ConstVertexPtr>(depth_);
                to_vector(result, i, regenerate, full_expand);
                if (i != depth_)
                  result.resize(i);

                link_vector_ = result;
                return link_vector_;
            } else {
                return link_vector_;
            }
        }

        // the vertices are not known
        if (all_vert_.empty() || regenerate) {

            size_t i = 0;
            auto result = std::vector<ConstVertexPtr>(depth_);
            to_vector(result, i, regenerate, full_expand);
            if (i != depth_)
                result.resize(i);
            all_vert_ = result;
            return all_vert_;
        } else {
            return all_vert_;
        }
    }

    const vector<ConstVertexPtr> &Linkage::vertices(bool regenerate) const {
        return to_vector(regenerate, true);
    }

    void Linkage::clone_link(const Linkage &other) {
        // Lock the mutex for the scope of the function
        std::lock_guard<std::mutex> lock(mtx_);

        // call base class copy constructor
        Vertex::operator=(other);

        // fill linkage data (shallow copy, but should not be modified either way)
        left_  = other.left_;
        right_ = other.right_;

        id_ = other.id_;
        depth_ = other.depth_;

        int_connec_ = other.int_connec_;
        l_ext_idx_ = other.l_ext_idx_;
        r_ext_idx_ = other.r_ext_idx_;

        flop_scale_ = other.flop_scale_;
        mem_scale_ = other.mem_scale_;

        is_addition_ = other.is_addition_;
        is_reused_ = other.is_reused_;

        all_vert_ = other.all_vert_;
        link_vector_ = other.link_vector_;
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

        id_ = other.id_;
        depth_ = other.depth_;

        int_connec_ = std::move(other.int_connec_);
        l_ext_idx_ = std::move(other.l_ext_idx_);
        r_ext_idx_ = std::move(other.r_ext_idx_);

        flop_scale_ = other.flop_scale_;
        mem_scale_ = other.mem_scale_;

        is_addition_ = other.is_addition_;
        is_reused_ = other.is_reused_;

        all_vert_ = std::move(other.all_vert_);
        link_vector_ = std::move(other.link_vector_);
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
