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
        unordered_map<const Line*, std::array<int_fast8_t, 2>, LineHash, LinePtrEqual>
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
            // skip internal lines, and keep all left lines if addition
            if (!is_addition_ & !left_ext_idx[i]) continue;
            add_line(left_lines[i]);
            mem_scale_ += left_lines[i];
        }

        // right half
        for (uint_fast8_t i = 0; i < right_size; ++i) {
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
        uint_fast8_t op_vec_size = op_vec.size();

        // cannot link less than two vertices
        if (op_vec_size <= 1)
            throw invalid_argument("Linkage::link(): op_vec must have at least two elements");


        LinkagePtr linkage = as_link(op_vec[0] * op_vec[1]);
        for (uint_fast8_t i = 2; i < op_vec_size; i++){
            LinkagePtr link = as_link(linkage * op_vec[i]);
            linkage = link;
        }

        return linkage;
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
        if (empty() ^ other.empty()) return false;

        // check if linkage type is the same
        if (is_addition_ ^ other.is_addition_) return false;

        // check the depth of the linkage
        if (depth_ != other.depth_) return false;

        // check if left and right vertices are linked in the same way
        bool left_linked = left_->is_linked(), right_linked = right_->is_linked();

        if ( left_linked ^  other.left_->is_linked()) return false;
        if (right_linked ^ other.right_->is_linked()) return false;

        // check that scales are equal
        if (flop_scale_ != other.flop_scale_) return false;
        if (mem_scale_  !=  other.mem_scale_) return false;

        // check linkage maps
        if (connec_map_.size() != other.connec_map_.size()) return false;

        // check that every element of this connec_map_ is in other.connec_map_, in any order
        for (const auto &connec : connec_map_) {
            if (std::find(other.connec_map_.begin(),
                          other.connec_map_.end(), connec) == other.connec_map_.end())
                return false;
        }

        // recursively check if left linkages are equivalent
        if (left_linked) {
            if (*as_link(left_) != *as_link(other.left_)) return false;
        } else {
            if ( !left_->equivalent( *other.left_)) return false;
        }

        // check if right linkages are equivalent
        if (right_linked) {
            if (*as_link(right_) != *as_link(other.right_)) return false;
        } else {
            if ( !right_->equivalent( *other.right_)) return false;
        }

        // lastly, check if the vertex representations are equivalent
        return Vertex::equivalent(other);
    }

    // repeat code from == operator, but invert the logic to end recursion early if possible
    bool Linkage::operator!=(const Linkage &other) const {
        return !(*this == other);
    }

    /**
     * Tests if two linkages are equivalent up to permutation of the external lines
     * @param other linkage to compare
     * @return pair of bools:
     *      1) true if equivalent up to permutation
     *      2) true if the parity of the permutation is odd
     */
    pair<bool, bool> Linkage::permuted_equals(const Linkage &other) const {

        throw std::runtime_error("Linkage::permuted_equals() is not operational");

        // check if the linkages are equivalent
        if (*this == other) return {true, false};

        // check if the names of the linkages are the same (indicates same vertices are being linked)
        if (this->name() != other.name()) return {false, false};

        // ensure the same number of lines
        if (this->rank() != other.rank())
            return {false, false};

        // ensure same number of line types (the sum of the similarLineHashes should be the same for both linkages)
        constexpr SimilarLineHash similarLineHash;
        size_t this_line_sum = 0, other_line_sum = 0;
        for (const Line &line : this->lines_) this_line_sum  += similarLineHash(line);
        for (const Line &line : other.lines_) other_line_sum += similarLineHash(line);

        if (this_line_sum != other_line_sum)
            return {false, false};

        // create a new linkage of this for every permutation of the external lines
        size_t rank = this->rank();
        size_t perm_vec[rank];
        std::iota(perm_vec, perm_vec + this->rank(), 0);

        // initialize map of lines to their replacement lines
        unordered_map<Line, Line, LineHash> replacement_map;

        // recursively replace the lines of the vertices with the permuted lines
        vector<ConstVertexPtr> this_vertices = this->vertices();
        vector<ConstVertexPtr> other_vertices = other.vertices();

        // remake the other linkage from its vertices (in case the tree structure is different, but equivalent)
        LinkagePtr other_linkage = link(other_vertices);

        bool is_odd = false;
        while (std::next_permutation(perm_vec, perm_vec + rank)) {

            // update parity of permutation
            is_odd = !is_odd;

            // map the lines to their replacement lines
            replacement_map.clear();
            for (size_t i = 0; i < rank; i++) {
                replacement_map[this->lines_[i]] = other.lines_[perm_vec[i]];
            }

            // generate permuted vertices
            vector<ConstVertexPtr> permuted_vertices;
            permuted_vertices.reserve(this_vertices.size());
            for (const ConstVertexPtr &vertex : this_vertices){
                VertexPtr permuted_vertex = vertex->clone();
                permuted_vertex->replace_lines(replacement_map);
                permuted_vertices.push_back(permuted_vertex);
            }

            // create linkage from permuted vertices and
            LinkagePtr permuted_linkage = link(permuted_vertices);

            // return whether the permuted linkage is equivalent to the other linkage
            // and the parity of the permutation
            if (*permuted_linkage == *other_linkage)
                return {true, is_odd};

        }

        // if no permutation is equivalent, return false
        return {false, false};

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

    ConstVertexPtr Linkage::safe_clone() const {
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
        ConstVertexPtr left  = root_link->left()->safe_clone();
        ConstVertexPtr right = root_link->right()->safe_clone();

        // sort the left and right vertices
        if (left_linked)  left  = tree_sort(left);
        if (right_linked) right = tree_sort(right);

        // swap the right operator of the left vertex with the left operator of the right vertex
        ConstVertexPtr LL, LR, RL, RR;

        if (left_linked) {
            LL  = as_link(left)->left()->safe_clone();
            LR  = as_link(left)->right()->safe_clone();
        }
        if (right_linked) {
            RL  = as_link(right)->left()->safe_clone();
            RR = as_link(right)->right()->safe_clone();
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
        ConstLinkagePtr best_link = as_link(root->safe_clone());

        // set the best flop and memory scales
        shape best_flop_scale = best_link->flop_scale_;
        shape best_mem_scale = best_link->mem_scale_;

        while (std::next_permutation(permutation.begin(), permutation.end())) {
            // create the new linkage
            ConstLinkagePtr new_link = link(permutation);

            // check if the new linkage is better than the current best and update if so
            shape new_flop_scale = new_link->flop_scale_;
            shape new_mem_scale = new_link->mem_scale_;

            // check if the new linkage is better than the current best and update if so
            bool update = new_flop_scale < best_flop_scale;
            if (!update) // if flop scales are equal, check memory scales
                update = best_flop_scale == new_flop_scale && best_mem_scale < new_mem_scale;

            if (update) {
                best_link = new_link;
                best_flop_scale = new_flop_scale;
                best_mem_scale = new_mem_scale;
            }
        }

        // return the best linkage
        return best_link;
    }

    void Linkage::replace_lines(const unordered_map<Line, Line, LineHash> &line_map) {
        // replace the lines of the vertices
        left_->clone()->replace_lines(line_map);
        right_->clone()->replace_lines(line_map);


        // rebuild the linkage
        long id = id_;
        bool is_reused = is_reused_;
        *this = Linkage(left_, right_, is_addition_);
        id_ = id;
        is_reused_ = is_reused;

    }

    VertexPtr Linkage::clone() const {
        VertexPtr left_clone = left_->clone();
        VertexPtr right_clone = right_->clone();

        LinkagePtr clone = make_shared<Linkage>(*this);
        clone->left_ = left_clone;
        clone->right_ = right_clone;

        return clone;
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
