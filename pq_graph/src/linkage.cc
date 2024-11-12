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
#include <stack>
#include <queue>
#include "../include/linkage.h"
#include "../include/linkage_set.hpp"
#include "../../pdaggerq/pq_string.h"

namespace pdaggerq {

    /********** Constructors **********/

    Linkage::Linkage(VertexPtr  left, VertexPtr  right, bool is_addition) :
                            left_(std::move(left)), right_(std::move(right)), addition_(is_addition) {

        // build internal and external lines with their index mapping
        build_connections();
    }

    void Linkage::build_connections() {

        lines_.clear();
        flop_scale_ = {};
        mem_scale_ = {};
        connec_map_.clear();
        forget();

        // replace null pointers with empty vertices
        if ( left_ == nullptr)  left_ = std::make_shared<const Vertex>();
        if (right_ == nullptr) right_ = std::make_shared<const Vertex>();

        // determine if left and right vertices are valid
        bool left_valid  =  !left_->empty() && (fabs( left_->value() - 1.0) > 1e-8);
        bool right_valid = !right_->empty() && (fabs(right_->value() - 1.0) > 1e-8);

        // if one is invalid and the other is linked, set this linkage to the linked vertex
        if ( !left_valid && right_->is_linked()) { *this = *as_link(right_->shallow()); return; }
        if (!right_valid &&  left_->is_linked()) { *this = *as_link( left_->shallow()); return; }

        // if both are invalid, set this linkage to an empty vertex
        if ( !left_valid && !right_valid) { *this = Linkage(); return; }

        // if one is invalid and the other is not, keep empty vertex on the left
        if ( left_valid && !right_valid) { std::swap(left_, right_); }

        // grab data from left and right vertices
        uint_fast8_t left_size = left_->size();
        uint_fast8_t right_size = right_->size();

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
            if (!left_->empty()) {
                mem_scale_ = right_->shape_;
                flop_scale_ = right_->shape_;
            }

            // update vertex members
            lines_ = right_lines;

            // add right lines to connec_map_
            connec_map_.reserve(right_size);
            for (uint_fast8_t i = 0; i < right_size; i++)
                connec_map_.push_back({-1, (int_fast8_t) i});

            set_properties();
            return;
        }
        if (right_size == 0) {
            // if right is a scalar, just use left_lines as linkage
            if (!right_->empty()) {
                mem_scale_ = left_->shape_;
                flop_scale_ = left_->shape_;
            }
            lines_  = left_lines;

            // add left lines to connec_map_
            connec_map_.reserve(left_size);
            for (uint_fast8_t i = 0; i < left_size; i++)
                connec_map_.push_back({(int_fast8_t) i, -1});

            // update vertex members
            set_properties();
            return;
        }

        // create a map of lines to their corresponding indicies
        unordered_map<const Line*, std::array<int_fast8_t, 2>, LineHash, LineEqual>
                line_populations;

        // populate left lines
        for (uint_fast8_t i = 0; i < left_size; i++) {
            auto &[left_id, right_id] = line_populations[&left_lines[i]];
            left_id = static_cast<int_fast8_t>(i);
            right_id = -1;
        }

        // populate right lines and track index
        for (uint_fast8_t i = 0; i < right_size; i++) {

            // attempt to insert right line into map
            auto [pos, inserted] = line_populations.try_emplace(&right_lines[i], std::array<int_fast8_t, 2>{-1, static_cast<int_fast8_t>(i)});

            if (!inserted) {
                // if line already exists, update right_id
                auto &[left_id, right_id] = pos->second;
                right_id = (int_fast8_t) i; // add index to right_id
            }
        }

        // now we have a map of lines to their corresponding indices
        // determine which lines are internal and external and store their indices
        bool left_ext_idx[left_size], right_ext_idx[right_size];
        memset( left_ext_idx, '\0',  left_size);
        memset(right_ext_idx, '\0', right_size);

        // reserve lines for indices
        connec_map_.reserve(line_populations.size());

        // populate connec_map_, rank, memory and flop scaling
        for (auto &[line_ptr, line_connec] : line_populations) {

            // get indices
            const auto &[left_idx, right_idx] = line_connec;

            // add to connection map
            connec_map_.push_back(line_connec);

            // get line
            const Line &line = *line_ptr;

            // check if line is external and should be added
            bool left_external  = right_idx < 0;
            bool right_external =  left_idx < 0;

            // keep track of external indicies
            left_ext_idx[  left_idx] =  left_external;
            right_ext_idx[right_idx] = right_external;

            // update flop scaling
            flop_scale_ += line;
        }

        // make external lines
        lines_.reserve(mem_scale_.n_);
        line_vector sig_lines;
        line_vector den_lines;

        // sort the connections
        std::sort(connec_map_.begin(), connec_map_.end());

        line_vector &lines = lines_;
        auto add_line = [&lines, &sig_lines, &den_lines](const Line &line) {
            if (!line.sig_ & !line.den_)
                lines.push_back(line);
            else if (line.sig_)
                sig_lines.push_back(line);
            else
                den_lines.push_back(line);
        };


        // left half
        for (uint_fast8_t i = 0; i < left_size; ++i) {
            // skip internal lines, and keep all lines if addition
            if (!addition_ & !left_ext_idx[i]) continue;
            add_line(left_lines[i]);
            mem_scale_ += left_lines[i];
        }

        // right half
        for (uint_fast8_t i = 0; i < right_size; ++i) {
            // skip internal lines, and keep all lines if addition
            if (!right_ext_idx[i]) continue;
            add_line(right_lines[i]);
            mem_scale_ += right_lines[i];
        }

        // add sigma lines to the beginning of lines_
        if (!sig_lines.empty())
            lines_.insert(lines_.begin(), sig_lines.begin(), sig_lines.end());

        // add density lines to the beginning of lines_
        if (!den_lines.empty())
            lines_.insert(lines_.begin(), den_lines.begin(), den_lines.end());

//        std::sort(lines_.begin(), lines_.end(), line_compare());

        // update vertex members
        set_properties();
    }

    VertexPtr Linkage::relabel() const {

        // create a deep copy of the linkage
        MutableLinkagePtr new_link = as_link(clone());

        // get lines from vertices
        line_vector lines; lines.reserve(2*lines_.size()+1);
        VertexPtr cur_vert = shared_from_this();

        // use queue to recursively traverse the linkage to get lines in order
        std::queue<VertexPtr> vert_queue;
        vert_queue.push(cur_vert);
        while (!vert_queue.empty()) {

            // get the current vertex
            cur_vert = vert_queue.front(); vert_queue.pop();

            // add lines to the list
            lines.insert(lines.end(), cur_vert->lines().begin(), cur_vert->lines().end());

            // if the vertex is expandable, add the left and right vertices to the stack
            if (cur_vert->is_linked() && !cur_vert->is_temp()) {
                vert_queue.push(as_link(cur_vert)->left_);
                vert_queue.push(as_link(cur_vert)->right_);
            }
        }

        // begin relabeling the lines

        size_t occ_idx = 0, virt_idx = 0, sig_idx = 0, den_idx = 0;

        // map lines to their first appearance
        unordered_map<Line, Line, LineHash> line_map;
        for (const auto &line : lines) {
            // first check if line is already in map; skip if it is
            if (line_map.find(line) != line_map.end())
                continue;

            // adjust index based on line type
            string new_label;
            switch (line.type()) {
                case 'o': new_label = Line::occ_labels_[occ_idx++]; break;
                case 'v': new_label = Line::virt_labels_[virt_idx++]; break;
                default: new_label  = line.label_; break;
            }

            // add line to map
            Line new_line = line;
            new_line.label_ = new_label;
            line_map[line] = new_line;
        }

        // replace lines in vertices
        new_link->replace_lines(line_map, true);
        new_link->build_connections();

        // return the new linkage
        return new_link;

    }


    void Linkage::set_properties() {
        // determine the depth of the linkage
        depth_ = 1;
        size_t left_depth = left_->depth(), right_depth = right_->depth();
        depth_ += std::max(left_depth, right_depth);

        // create the name of the linkage
        base_name_.reserve(left_->name_.size() + right_->name_.size() + 1);
        base_name_ = left_->name_;
        if (addition_)
             base_name_ += '+';
        else base_name_ += '*';
        base_name_ += right_->name_;
        base_name_ += ' ';

        // add connection map to base name
        for (const auto &[leftidx, rightidx] : connec_map_) {
            base_name_ += (char)leftidx + '1'; // convert to int
            base_name_ += '>';
            base_name_ += (char)rightidx + '1'; // convert to int
        }

        // add hashes of the lines
        base_name_ += ' ';
        for (const auto &line : left_->lines_) {
            base_name_ += line.type() + line.block() - 'a';
        }
        base_name_ += ' ';
        for (const auto &line : right_->lines_) {
            base_name_ += line.type() + line.block() - 'a';
        }
        base_name_ += ' ';
        for (const auto &line : lines_) {
            base_name_ += line.type() + line.block() - 'a';
        }

        // base name is complete
        name_ = base_name_;

        // set descriptors
        rank_  = lines_.size();
        shape_ = shape(lines_);
        has_blk_  = left_->has_blk_  || right_->has_blk_;
        is_sigma_ = left_->is_sigma_ || right_->is_sigma_ || left_->shape_.L_ > 0 || right_->shape_.L_ > 0;
        is_den_   = left_->is_den_   || right_->is_den_   || left_->shape_.Q_ > 0 || right_->shape_.Q_ > 0;
    }

    vector<Line> Linkage::internal_lines() const {
        vector<Line> int_lines;
        size_t left_size = left_->size();
        size_t right_size = right_->size();

        // if both left and right are scalars, there are no internal lines
        if (left_size == 0 && right_size == 0)
            return int_lines;

        int_lines.reserve(left_size + right_size - lines_.size());

        // use connec_map_ to grab the internal lines
        const auto & left_lines = left_->lines();
        for (const auto &[left_idx, right_idx] : connec_map_) {
            if (left_idx >= 0 && right_idx >= 0) {
                // add to internal lines (just use left lines since the line is in both)
                int_lines.push_back(left_lines[left_idx]);
            }
        }

        return int_lines;
    }

    Linkage::Linkage() {
        addition_ = false;

        // initialize left and right vertices with empty vertices
        left_ = std::make_shared<const Vertex>();
        right_ = std::make_shared<const Vertex>();
    }

    /****** operator overloads ******/

    bool Linkage::similar_root(const Linkage &other) const{
        // check if both linkage are empty or not
        if (empty() != other.empty()) return false;

        // check if linkage type is the same
        if (addition_ != other.addition_) return false;

        // check the depth of the linkage
        if (depth_ != other.depth_) return false;

        // note, we do NOT check the id of the linkage

        // check if left and right vertices are linked in the same way
        if ( left_->is_linked() !=  other.left_->is_linked()) return false;
        if (right_->is_linked() != other.right_->is_linked()) return false;

        // check that scales are equal
        if (flop_scale_ != other.flop_scale_) return false;
        if (mem_scale_  !=  other.mem_scale_) return false;

        // check that the connectivity of the linkages is the same (it is sorted, so compare by element)
        return connec_map_ == other.connec_map_;

        // all checks pass
    }
    bool Linkage::operator==(const Linkage &other) const {

        // the roots of the linkages are not equivalent
        if (!similar_root(other))
            return false;

        // recursively check if left linkages are equivalent
        if (left_->is_linked()) {
            if (*left_ != *other.left_) return false;
        } else if ( !left_->equivalent( *other.left_)) return false;

        // check if right linkages are equivalent
        if (right_->is_linked()) {
            if (*right_ != *other.right_) return false;
        } else if ( !right_->equivalent( *other.right_)) return false;

        // ensure root vertices are equivalent
        return Vertex::equivalent(other);
    }

    bool Linkage::operator!=(const Linkage &other) const {

        // repeat code from == operator, but invert the logic to end recursion early if possible
        if (!similar_root(other))
            return true;

        // recursively check if left linkages are equivalent
        bool left_same, right_same;
        if (left_->is_linked())
             left_same = *as_link(left_) == *as_link(other.left_);
        else left_same = left_->equivalent( *other.left_);

        // left is not equivalent; therefore, the linkages are not equivalent
        if (!left_same) return true;

        // check if right linkages are equivalent
        if (right_->is_linked())
             right_same = *as_link(right_) == *as_link(other.right_);
        else right_same = right_->equivalent( *other.right_);

        // right is not equivalent; therefore, the linkages are not equivalent
        if (!right_same) return true;

        // ensure root vertices are not equivalent
        return !Vertex::equivalent(other);
    }

    bool Linkage::same_temp(const VertexPtr &other) const {
        if (!this->is_temp() || !other->is_temp())
            return false; // neither is a temp

        // whether the linkage corresponds to the same intermediate contraction as another vertex
        bool same_type = id_ == other->id() && type() == other->type();
        if (!same_type) return false;

        return  *this == *as_link(other);
    }

    void Linkage::copy_link(const Linkage &other) {
        // Lock the mutex for the scope of the function
//        std::lock_guard<std::mutex> lock(mtx_);

        // call base class copy constructor
        Vertex::operator=(other);

        // fill linkage data (shallow copy, but vertex cannot be modified)
        left_  = other.left_;
        right_ = other.right_;
        depth_ = other.depth_;

        // copy vectors that keep track of the graph structure
        all_vert_     = other.all_vert_;
        link_vector_  = other.link_vector_;
        permutations_ = other.permutations_;

        // copy root linkage connectivity and scales
        connec_map_ = other.connec_map_;
        flop_scale_ = other.flop_scale_;
        mem_scale_  = other.mem_scale_;

        // copy misc properties
        copy_misc(other);
    }

    void Linkage::forget(bool forget_all) const {
        // clears all vectors that track the graph structure of the linkage (allows for rebuilding)
        all_vert_.clear();
        link_vector_.clear();
        permutations_.clear();

        if (forget_all) {
            // clear subgraphs
            if (left_ && left_->is_linked() && !left_->empty())
                as_link(left_)->forget(true);
            if (right_ && right_->is_linked() && !right_->empty())
                as_link(right_)->forget(true);
        }
    }

    MutableVertexPtr Linkage::clone() const {
        MutableLinkagePtr clone(new Linkage(*this));

        // recursively clone left and right vertices
        clone->left_ = left_->clone();
        clone->right_ = right_->clone();

        // vertex pointers within vectors are not cloned
        // Must clear to rebuild when needed
        clone->forget(false);

        return clone;
    }

    MutableVertexPtr Linkage::shallow() const {
        return make_shared<Linkage>(*this);
    }

    Linkage::Linkage(const Linkage &other) {
        copy_link(other);
    }

    Linkage &Linkage::operator=(const Linkage &other) {
        // check for self-assignment
        if (this == &other) return *this;
        else copy_link(other);

        return *this;
    }

    void Linkage::move_link(Linkage &&other) {
        // Lock the mutex for the scope of the function
//        std::lock_guard<std::mutex> lock(mtx_);

        // call base class move constructor
        this->Vertex::operator=(other);

        // move linkage data
        left_  = std::move(other.left_);
        right_ = std::move(other.right_);
        depth_ = other.depth_;

        // move vectors that keep track of the graph structure
        all_vert_     = std::move(other.all_vert_);
        link_vector_  = std::move(other.link_vector_);
        permutations_ = std::move(other.permutations_);

        // move root linkage connectivity and scales
        connec_map_ = std::move(other.connec_map_);
        flop_scale_ = other.flop_scale_;
        mem_scale_  = other.mem_scale_;

        // copy misc properties
        copy_misc(other);
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

    void Linkage::update_lines(const line_vector &lines, bool update_name) {

        // map lines to the new lines if compatible
        bool compatible = true;
        for (size_t i = 0; i < lines_.size(); i++) {
            if (!lines_[i].equivalent(lines[i])) {
                compatible = false; break;
            }
        }
        if (compatible) {
            unordered_map<Line, Line, LineHash> line_map = LineHash::map_lines(lines_, lines);
            this->replace_lines(line_map, update_name);
        }

        if (update_name)
            this->update_name();
    }

    extern MutableVertexPtr operator*(const VertexPtr &left, const VertexPtr &right){
        bool left_valid = left && !left->empty(), right_valid = right && !right->empty();
        if ( left_valid && !right_valid) return  left->shallow();
        if (!left_valid &&  right_valid) return right->shallow();
        if (!left_valid && !right_valid) return make_shared<Vertex>();

        return make_shared<Linkage>(left, right, false);
    }
    extern MutableVertexPtr operator*(const MutableVertexPtr &left, const MutableVertexPtr &right){
        bool left_valid = left && !left->empty(), right_valid = right && !right->empty();
        if ( left_valid && !right_valid) return  left->shallow();
        if (!left_valid &&  right_valid) return right->shallow();
        if (!left_valid && !right_valid) return make_shared<Vertex>();

        return make_shared<Linkage>(left, right, false);
    }
    extern MutableVertexPtr operator*(double factor, const VertexPtr &right){
        int min_precision = minimum_precision(factor);
        string factor_str = to_string_with_precision(factor, min_precision);
        MutableVertexPtr left = make_shared<Vertex>(factor_str);

        if (!right || right->empty())
            return left;

        return make_shared<Linkage>(left, right, false);
    }
    extern MutableVertexPtr operator*(const VertexPtr &left, double factor){
        int min_precision = minimum_precision(factor);
        string factor_str = to_string_with_precision(factor, min_precision);
        MutableVertexPtr right = make_shared<Vertex>(factor_str);

        if (!left || left->empty())
            return right;

        return make_shared<Linkage>(left, right, false);
    }

    extern MutableVertexPtr operator+(const VertexPtr &left, const VertexPtr &right){
        bool left_valid = left && !left->empty(), right_valid = right && !right->empty();
        if ( left_valid && !right_valid) return  left->shallow();
        if (!left_valid &&  right_valid) return right->shallow();
        if (!left_valid && !right_valid) return make_shared<Vertex>();

        return make_shared<Linkage>(left, right, true);
    }
    extern MutableVertexPtr operator+(const MutableVertexPtr &left, const MutableVertexPtr &right){
        bool left_valid = left && !left->empty(), right_valid = right && !right->empty();
        if ( left_valid && !right_valid) return  left->shallow();
        if (!left_valid &&  right_valid) return right->shallow();
        if (!left_valid && !right_valid) return make_shared<Vertex>();

        return make_shared<Linkage>(left, right, true);
    }
    extern MutableVertexPtr operator+(double factor, const VertexPtr &right){
        int min_precision = minimum_precision(factor);
        string factor_str = to_string_with_precision(factor, min_precision);
        MutableVertexPtr left = make_shared<Vertex>(factor_str);

        if (!right || right->empty())
            return left;

        return make_shared<Linkage>(left, right, true);
    }
    extern MutableVertexPtr operator+(const VertexPtr &left, double factor){
        int min_precision = minimum_precision(factor);
        string factor_str = to_string_with_precision(factor, min_precision);
        MutableVertexPtr right = make_shared<Vertex>(factor_str);

        if (!left || left->empty())
            return right;

        return make_shared<Linkage>(left, right, true);
    }

} // pdaggerq
