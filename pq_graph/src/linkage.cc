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
#include <numeric>
#include "../include/linkage.h"

namespace pdaggerq {

    /********** Constructors **********/

    inline Linkage::Linkage(const ConstVertexPtr &left, const ConstVertexPtr &right, bool is_addition) : Vertex() {

        // set inputs
        left_ = left;
        right_ = right;

        is_addition_ = is_addition;

        if (!left->is_linked() && !right->is_linked() && !is_addition_) {
            // a binary multiplication of pure vertices is associative (left and right are interchangeable)
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

            // add right lines to connec_map_
            connec_map_.reserve(right_size);
            for (uint_fast8_t i = 0; i < right_size; i++)
                connec_map_.push_back({-1, (int_fast8_t) i});

            set_properties();
            return;
        }
        if (right_size == 0) {
            // if right is a scalar, just use left_lines as linkage
            mem_scale_ = left_->shape_;
            flop_scale_ = left_->shape_;
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
        bool left_first = left_size <= right_size;
        line_vector sig_lines;
        line_vector den_lines;

        // sort the connections
        std::sort(connec_map_.begin(), connec_map_.end());

        auto add_line = [this, &sig_lines, &den_lines](const Line &line) {
            if (!line.sig_ & !line.den_)
                lines_.push_back(line);
            else if (line.sig_)
                sig_lines.push_back(line);
            else
                den_lines.push_back(line);
        };


        // left half
        for (uint_fast8_t i = 0; i < left_size; ++i) {
            // skip internal lines, and keep all lines if addition
            if (!is_addition_ & !left_ext_idx[i]) continue;
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

    LinkagePtr Linkage::link(const vector<ConstVertexPtr> &op_vec) {
        size_t op_vec_size = op_vec.size();

        // cannot link less than two vertices
        if (op_vec_size <= 1)
            throw invalid_argument("Linkage::link(): op_vec must have at least two elements");


        LinkagePtr linkage = as_link(op_vec[0] * op_vec[1]);
        for (size_t i = 2; i < op_vec_size; i++){
            LinkagePtr link = as_link(linkage * op_vec[i]);
            linkage = link;
        }

        return linkage;
    }

    tuple<ConstLinkagePtr, vector<shape>, vector<shape>> Linkage::link_and_scale(const vector<ConstVertexPtr> &op_vec) {
        size_t op_vec_size = op_vec.size();
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

        for (size_t i = 2; i < op_vec_size; i++) {
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

    bool Linkage::similar_root(const Linkage &other) const{
        // check if both linkage are empty or not
        if (empty() == other.empty()) return false;

        // check if linkage type is the same
        if (is_addition_ == other.is_addition_) return false;

        // check the depth of the linkage
        if (depth_ != other.depth_) return false;

        // check if left and right vertices are linked in the same way
        bool left_linked = left_->is_linked(), right_linked = right_->is_linked();

        // check if both left and right vertices are linked or not
        if ( left_linked ==  other.left_->is_linked()) return false;
        if (right_linked == other.right_->is_linked()) return false;

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

        // ensure root vertices are equivalent
        return Vertex::equivalent(other);
    }

    bool Linkage::operator!=(const Linkage &other) const {

        // repeat code from == operator, but invert the logic to end recursion early if possible
        if (!similar_root(other))
            return true;

        // recursively check if left linkages are equivalent
        bool left_same = false, right_same = false;
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

    bool Linkage::same_temp(const ConstVertexPtr &other) const {
        if (!this->is_temp() || !other->is_temp())
            return false; // neither is a temp

        // whether the linkage corresponds to the same intermediate contraction as another vertex
        bool same_id = id_ == other->id() && is_reused_ == other->is_reused();
        if (!same_id) return false;

        // check if the shapes of the lines are the same
        bool equivalent_root = Vertex::equivalent(*other);
        if (!equivalent_root) return false;

        // replace lines of the other vertex with the lines of this vertex
        VertexPtr other_clone = other->clone();
        other_clone->replace_lines(LineHash::map_lines(lines_, other_clone->lines_));

        return  *this == *as_link(other_clone);
    }

    void Linkage::replace_lines(const unordered_map<Line, Line, LineHash> &line_map) {

        // replace the lines of the root vertex with the new lines
        Vertex::replace_lines(line_map);
        name_ = base_name_;

        // recursively replace the lines of the left and right vertices
        VertexPtr  left = left_->clone();
        VertexPtr right = right_->clone();

        // determine new lines of left/right from the line map (if they do not exist, they are not replaced)
        auto make_new_lines = [&line_map](const VertexPtr &vertex) {
            line_vector new_lines;
            for (const auto &line : vertex->lines()) {
                // find the line in the map
                auto it = line_map.find(line);
                if (it != line_map.end())
                    new_lines.push_back(it->second);
                else {
                    // check if the line is already an entry in the new_lines
                    if (std::find(new_lines.begin(), new_lines.end(), line) == new_lines.end())
                        new_lines.push_back(line);
                    else {
                        // if the line is already in the new_lines, use opposite mapping
                        for (const auto &[key, value] : line_map) {
                            if (value == line) {
                                new_lines.push_back(key);
                                break;
                            }
                        }
                    }
                }
            }
            return new_lines;
        };

        line_vector new_left_lines  = make_new_lines(left);
        line_vector new_right_lines = make_new_lines(right);

        // replace the lines of the left and right vertices
        left->replace_lines(line_map);
        right->replace_lines(line_map);

        // set new left and right vertices
        left_  = left;
        right_ = right;

    }

    pair<bool, bool> Linkage::permuted_equals(const Linkage &other) const {

        // check if linkages are exactly the same
        bool same_linkage = *this == other;
        if (same_linkage) return {true, false};
        else if (is_temp() || other.is_temp())
            return {false, false}; // cannot permute intermediates

        // check if both linkage are empty or not
        if (empty() == other.empty()) return {false, false};

        // check if linkage type is the same
        if (is_addition_ == other.is_addition_) return {false, false};

        // check the depth of the linkage
        if (depth_ != other.depth_) return {false, false};

        // check if left and right vertices are linked in the same way
        bool left_linked = left_->is_linked(), right_linked = right_->is_linked();

        // check if both left and right vertices are linked or not
        if ( left_linked ==  other.left_->is_linked()) return {false, false};
        if (right_linked == other.right_->is_linked()) return {false, false};

        // check that scales are equal
        if (flop_scale_ != other.flop_scale_) return {false, false};
        if (mem_scale_  !=  other.mem_scale_) return {false, false};


        bool is_equiv = false, odd_parities = false;

        // expand the vertices
        const vector<ConstVertexPtr> &this_verts = link_vector(true);
        const vector<ConstVertexPtr> &other_verts = other.link_vector(true);

        // check that link vectors are the same size
        if (this_verts.size() != other_verts.size())
            return {is_equiv, odd_parities};

        // permute each vertex in other_verts to match this_verts
        vector<ConstVertexPtr> permuted_verts;
        for (size_t i = 0; i < this_verts.size(); i++) {

            // check if the vertex base names are the same
            if (this_verts[i]->base_name() != other_verts[i]->base_name())
                return {false, false};

            // if the vertex is an intermediate, do not permute and test equality (cannot permute intermediates)
            if (this_verts[i]->is_temp()) {

                // check if the linkages are the same
                if (this_verts[i]->same_temp(other_verts[i]))
                    return {false, false};

                // replace the lines of the other vertex with the lines of this vertex
                VertexPtr other_clone = other_verts[i]->clone();
                other_clone->replace_lines(this_verts[i]->lines_);

                // add the vertex to the permuted list
                permuted_verts.push_back(other_clone);
                continue;
            }

            // permute the vertex to match this_verts
            bool odd_parity = false;
            auto [permuted, success] = other_verts[i]->permute_like(*this_verts[i], odd_parity);

            // if not found, return false
            if (!success) return {false, false};

            // check if the permutation is odd
            if (odd_parity)
                odd_parities = !odd_parities;

            // add the permuted vertex to the list
            permuted_verts.push_back(permuted.shared_from_this());

        }

        // create a new linkage from the permuted vertices
        ConstLinkagePtr permuted_link = link(permuted_verts);

        // check if the permuted linkage is the same as this linkage
        is_equiv = *permuted_link == *this;
        if (!is_equiv) odd_parities = false;

        // return the result
        return {is_equiv, odd_parities};
    }


    string Linkage::str(bool make_generic, bool include_lines) const {

        if (!is_temp()) {
            // this is not an intermediate vertex (generic linkage).
            // return the str of the left and right vertices
            return tot_str(false, true);
        }

        if (!make_generic) return Vertex::str();

        // prepare output string as a map of tmps, scalars, or reuse_tmps to a generic name
        string generic_str;
        if (is_scalar())
             generic_str = "scalars_[\"";
        else if (is_reused_)
             generic_str = "reuse_tmps_[\"";
        else generic_str = "tmps_[\"";

        // use id_ to create a generic name
        string dimstring = this->dimstring();
        if (id_ >= 0)
            generic_str += to_string(id_);

        if (!dimstring.empty())
            generic_str += "_" + dimstring;

        generic_str += "\"]";

        if (include_lines) // if lines are included, add them to the generic name (default)
            generic_str += line_str(); // sorts print order

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

    inline void Linkage::link_vector(vector<ConstVertexPtr> &result, size_t &i, bool regenerate, bool full_expand) const {

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
                    for (const ConstVertexPtr &link_vertex: link->link_vector(regenerate, full_expand))
                        expand_vertex(link_vertex, result, i);
                }

            } else result[i++] = vertex;
        };

        // get the left vertices
        expand_vertex(left_, result, i);

        // get the right vertices
        expand_vertex(right_, result, i);
    }

    const vector<ConstVertexPtr> &Linkage::link_vector(bool regenerate, bool full_expand) const {

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
                link_vector(result, i, regenerate, full_expand);
                if (i != depth_)
                  result.resize(i);

                // remove any empty vertices
                result.erase(std::remove_if(result.begin(), result.end(), [](const ConstVertexPtr &vertex) {
                    return vertex->empty(); }), result.end()
                );

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
            link_vector(result, i, regenerate, full_expand);
            if (i != depth_)
                result.resize(i);

            // remove any empty vertices
            result.erase(std::remove_if(result.begin(), result.end(), [](const ConstVertexPtr &vertex) {
                return vertex->empty(); }), result.end()
            );

            all_vert_ = result;
            return all_vert_;
        } else {
            return all_vert_;
        }
    }

    const vector<ConstVertexPtr> &Linkage::vertices(bool regenerate) const {
        return link_vector(regenerate, true);
    }

    void Linkage::copy_link(const Linkage &other) {
        // Lock the mutex for the scope of the function
//        std::lock_guard<std::mutex> lock(mtx_);

        // call base class copy constructor
        Vertex::operator=(other);

        // fill linkage data (shallow copy, but should not be modified either way)
        left_  = other.left_;
        right_ = other.right_;

        id_ = other.id_;
        depth_ = other.depth_;

        connec_map_ = other.connec_map_;

        flop_scale_ = other.flop_scale_;
        mem_scale_ = other.mem_scale_;

        is_addition_ = other.is_addition_;
        is_reused_ = other.is_reused_;

        all_vert_ = other.all_vert_;
        link_vector_ = other.link_vector_;
    }

    Linkage::Linkage(const Linkage &other) {
        copy_link(other);
    }

    ConstVertexPtr Linkage::shallow() const {
        return shared_from_this();
    }

    Linkage &Linkage::operator=(const Linkage &other) {
        // check for self-assignment
        if (this == &other) return *this;
        else copy_link(other);

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

        connec_map_ = std::move(other.connec_map_);

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

    ConstVertexPtr Linkage::tree_sort() const {
        return tree_sort(shared_from_this());
    }

    ConstVertexPtr Linkage::tree_sort(const ConstVertexPtr & root) {

        return root; // ignore tree sorting until fixed

        // this is a vertex; nothing to do
        if (!root->is_linked()) return root;
        // else it is a linkage of vertices

        // do not modify intermediates (they cannot change once made)
        if (root->is_temp()) return root;
        // else we can modify the connectivity of the vertices

        ConstLinkagePtr root_link = as_link(root);

        // check if left or right vertices are linkages
        bool left_linked = root_link->left()->is_linked();
        bool right_linked = root_link->right()->is_linked();

        // nothing should be done if both left and right are vertices
        if (!left_linked && !right_linked) return root;

        // clone the left/right vertices
        ConstVertexPtr left  = root_link->left()->tree_sort();
        ConstVertexPtr right = root_link->right()->tree_sort();

        // swap the right operator of the left vertex with the left operator of the right vertex
        ConstVertexPtr LL, LR, RL, RR;

        if (left_linked) {
            LL = as_link(left)->left()->tree_sort();
            LR = as_link(left)->right()->tree_sort();
        }
        if (right_linked) {
            RL = as_link(right)->left()->tree_sort();
            RR = as_link(right)->right()->tree_sort();
        }

        // try all pairwise combinations of LL, LR with RL, RR
        vector<ConstVertexPtr> permutation;
        if (left_linked && right_linked) {
            permutation = {LL, LR, RL, RR};
        } else if (left_linked) {
            permutation = {LL, LR, right};
        } else { // right must be linked at this point
            permutation = {left, RL, RR};
        }
        size_t n = permutation.size();

        // create the best linkage as initial linkage
        ConstLinkagePtr best_link = as_link(root->shallow());
        string best_name = best_link->str();
        string best_flop_str;
        scaling_map best_flop_map, best_mem_map;

        do {


            // create the new linkage
            auto [new_link, new_flop_vec, new_mem_vec] = Linkage::link_and_scale(permutation);

            // build scaling maps
            scaling_map new_flop_map(new_flop_vec);
            scaling_map new_mem_map(new_mem_vec);

            string new_name = new_link->str();
            string new_flop_str = new_flop_map.str();

            if (best_flop_map.empty()) {
                best_flop_map = new_flop_map;
                best_mem_map  = new_mem_map;
                best_link = new_link;
                best_name = new_name;
                best_flop_str = new_flop_str;
                continue;
            }

            // check if the new linkage is better than the current best and update if so
            bool update = new_flop_map < best_flop_map;
            if (!update) // if flop maps are equal, check memory maps
                update = best_flop_map == new_flop_map && best_mem_map < new_mem_map;

            if (update) {
                best_link = new_link;
                best_flop_map = new_flop_map;
                best_mem_map  = new_mem_map;
                best_name = new_name;
                best_flop_str = new_flop_str;
            }
        } while (std::next_permutation(permutation.begin(), permutation.end()));

        // return the best linkage
        return best_link;
    }

    VertexPtr Linkage::clone() const {
        VertexPtr left_clone = left_->clone();
        VertexPtr right_clone = right_->clone();

        LinkagePtr clone = make_shared<Linkage>(Linkage(*this));
        clone->left_ = left_clone;
        clone->right_ = right_clone;

        clone->copy_misc(*this);

        return clone;
    }

    ConstVertexPtr Linkage::find_link(const ConstVertexPtr &target_vertex) const {
        if (same_temp(target_vertex)) return this->shared_from_this();
        if (left_->is_linked()) {
            const auto &left = as_link(left_)->find_link(target_vertex);
            if (left) return left;
        }
        if (right_->is_linked()) {
            const auto &right = as_link(right_)->find_link(target_vertex);
            if (right) return right;
        }
        return nullptr;
    }

    void Linkage::replace_link(const ConstVertexPtr &target_vertex, const ConstVertexPtr &new_vertex) {
        if (same_temp(target_vertex)) {
            *this = *as_link(new_vertex->clone());
            return;
        }
        VertexPtr left = left_->clone(), right = right_->clone();
        if (left->is_linked())
            as_link(left)->replace_link(target_vertex, new_vertex);
        if (right->is_linked())
            as_link(right)->replace_link(target_vertex, new_vertex);

        left_ = left;
        right_ = right;
    }

    bool Linkage::has_temp(const ConstVertexPtr &temp) const {
        if (!temp) return false;
        if (same_temp(temp)) return true;
        if (left_->is_linked() && as_link(left_)->has_temp(temp)) return true;
        if (right_->is_linked() && as_link(right_)->has_temp(temp)) return true;
        return false;
    }

    ConstVertexPtr Linkage::expand_to_temp(const ConstLinkagePtr &temp) const {
        if (same_temp(temp)) // if the temp is the same, return the linkage expanded
            return temp->left_ * temp->right_;
        auto left = left_->clone(), right = right_->clone();
        if (left_->is_linked())
            left = as_link(left)->expand_to_temp(temp)->clone();
        if (right_->is_linked())
            right = as_link(right)->expand_to_temp(temp)->clone();
        VertexPtr result = left * right;
        as_link(result)->copy_misc(*this);
        return result;
    }

    extern VertexPtr operator*(const ConstVertexPtr &left, const ConstVertexPtr &right){
        if (left && !right)
            return left->clone();
        if (!left && right)
            return right->clone();
        if (!left && !right)
            return make_shared<Vertex>();

        LinkagePtr &&linkage = make_shared<Linkage>(left, right, false);
        return linkage;
    }
    extern VertexPtr operator*(const VertexPtr &left, const VertexPtr &right){
        if (left && !right)
            return left->clone();
        if (!left && right)
            return right->clone();
        if (!left && !right)
            return make_shared<Vertex>();

        LinkagePtr &&linkage = make_shared<Linkage>(left, right, false);
        return linkage;
    }

    extern VertexPtr operator+(const ConstVertexPtr &left, const ConstVertexPtr &right){
        if (left && !right)
            return left->clone();
        if (!left && right)
            return right->clone();
        if (!left && !right)
            return make_shared<Vertex>();

        LinkagePtr &&linkage = make_shared<Linkage>(left, right, true);
        return linkage;
    }

} // pdaggerq
