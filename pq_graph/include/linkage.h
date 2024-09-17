//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: linkage.h
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

#ifndef PDAGGERQ_linkage_H
#define PDAGGERQ_linkage_H

#include <set>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <utility>

#include "vertex.h"
#include "scaling_map.hpp"

using std::ostream;
using std::string;
using std::vector;
using std::map;
using std::unordered_map;
using std::shared_ptr;
using std::make_shared;
using std::set;
using std::unordered_set;
using std::pair;
using std::make_pair;
using std::to_string;
using std::invalid_argument;
using std::tuple;
using std::static_pointer_cast;
using std::dynamic_pointer_cast;

namespace pdaggerq {

    // define cast function from Vertex pointers to Linkage pointers  and vice versa

    static LinkagePtr as_link(const VertexPtr &vertex)  {
        if (vertex->is_linked()) return static_pointer_cast<Linkage>(vertex);
        else throw invalid_argument("Cannot cast Vertex to Linkage");
    }
    static ConstLinkagePtr as_link(const ConstVertexPtr &vertex)  {
        if (vertex->is_linked()) return static_pointer_cast<const Linkage>(vertex);
        else throw invalid_argument("Cannot cast Vertex to Linkage");
    }

    /**
     * Perform linkage of two vertices by overload of * operator
     * @param other vertex to contract with
     * @return linkage of the two vertices
     * @note this is an extern function to allow for operator overloading outside of the namespace
     */
    extern VertexPtr operator*(const ConstVertexPtr &left, const ConstVertexPtr &right);
    extern VertexPtr operator*(const VertexPtr &left, const VertexPtr &right);
    extern VertexPtr operator*(double factor, const ConstVertexPtr &right);
    extern VertexPtr operator*(const ConstVertexPtr &left, double factor);

    /**
     * Perform linkage of two vertices by overload of + operator
     * @param other vertex to add
     * @return linkage of linkage
     * @note this is an extern function to allow for operator overloading outside of the namespace
     */
    extern VertexPtr operator+(const ConstVertexPtr &left, const ConstVertexPtr &right);
    extern VertexPtr operator+(const VertexPtr &left, const VertexPtr &right);
    extern VertexPtr operator+(double factor, const ConstVertexPtr &right);
    extern VertexPtr operator+(const ConstVertexPtr &left, double factor);

    /**
     * Class to represent contractions of a single vertex with a set of other vertices
     * The contraction itself is also a vertex and is defined by a left and right vertex
     */
    class Linkage;

    class Linkage : public Vertex {

        /// vertices in the linkage
        ConstVertexPtr left_, right_; // the left and right vertices of the linkage

        /// cost of linkage (flops and memory) as pair of vir and occ counts
        shape flop_scale_{}; // flops
        shape mem_scale_{}; // memory

        mutable std::mutex mtx_; // mutex for thread safety
        mutable vector<ConstVertexPtr> all_vert_; // all vertices from linkages (mutable to allow for lazy evaluation)
        mutable vector<ConstVertexPtr> link_vector_; // all non-intermediate vertices from linkages
        mutable vector<ConstLinkagePtr> permutations_; // all permutations of the linkage

    public:
        long id_ = -1; // id of the linkage (default to -1 if not set)
        size_t depth_{}; // number of vertices in the linkage
        bool addition_ = false; // whether the linkage is an addition; else it is a contraction
        bool reused_ = false; // whether the linkage is a shared operator (can be extracted)

        bool is_addition() const override { return addition_; } // whether the linkage is an addition
        bool &is_addition() override { return addition_; } // whether the linkage is an addition
        bool is_reused() const override { return reused_; } // whether the linkage is reused
        bool &is_reused() override { return reused_; } // whether the linkage is reused

        bool is_temp() const override { return id_ != -1; } // || is_reused_; } // whether the linkage corresponds to an intermediate contraction
        bool same_temp(const ConstVertexPtr &other) const override;
        string type() const override {
            // get the type of the linkage
            if (id_ == -1) return "link";
            if (is_scalar()) return "scalar";
            if (is_reused()) return "reused";
            return "temp";
        }

        bool is_expandable() const override {
            return !is_temp() && !is_addition() && !empty() && !is_scalar();
        }

        bool is_linked() const override { return true; } // indicates the vertex is linked to another vertex
        bool is_leaf(bool expand) const override {
            // check if the vertex is a leaf within the linkage
            if (!is_linked()) return true;
            if (!expand && is_temp()) return true;
            return false;
        }
        long &id() override { return id_; } // get the id of the linkage
        long id() const override { return id_; } // get the id of the linkage

        /// vertices in the linkage

        const ConstVertexPtr &left() const { return left_; }
        const ConstVertexPtr &right() const { return right_; }

        /// map of connec_map between lines
        std::vector<std::array<int_fast8_t, 2>> connec_map_; // connec_map between lines

        /********** Constructors **********/

        Linkage();

        /**
         * Constructor
         * @param left vertex to contract with
         * @param right vertex to contract with
         */
        Linkage(ConstVertexPtr left, ConstVertexPtr right, bool is_addition);

        /**
         * Connects the lines of the linkage, sets the flop and memory scaling, and sets the name
         * this function will populate the Vertex base class with the result of the contraction
         */
        void set_links();

        /**
        * Sets propeties of the vertex data members
        */
        void set_properties();

        /**
         * return vector of internal lines using the internal connection map
         * @return vector of internal lines
         */
        vector<Line> internal_lines() const;
        
        /**
         * fuses together similar sublinkages
         */
        void fuse();

        /**
         * Merges constants together in linkage
         */
         void merge_constants();

        /**
         * Destructor
         */
        ~Linkage() override = default;

        /**
         * Copy constructor
         * @param other linkage to copy
         */
        Linkage(const Linkage &other);

        /**
         * Return a deep copy of the linkage where all nested linkages are also copied
         * @return deep copy of the linkage
         */
        VertexPtr shallow() const override;
        VertexPtr clone() const override;

        /**
         * clears vectors that are populated by lazy evaluation:
         * all_vert_, link_vector_, permutations_, subgraphs_
         * @param forget_all whether to forget all subgraphs
         */
        void forget(bool forget_all = false) const;

        /**
         * Move constructor
         * @param other linkage to move
         */
        Linkage(Linkage &&other) noexcept;

        /**
         * helper to move only the linkage data
         * @param other linkage to move
         */
        void move_link(Linkage &&other);

        /**
         * helper to clone only the linkage data
         * @param other linkage to clone
         */
        void copy_link(const Linkage &other);

        /****** operator overloads ******/

        /**
         * Copy assignment operator
         * @param other linkage to copy
         * @return reference to this
         */
        Linkage &operator=(const Linkage &other);

        /**
         * Move assignment operator
         * @param other linkage to move
         * @return reference to this
         */
        Linkage &operator=(Linkage &&other) noexcept;

        /**
         * Equality operator
         * @param other linkage to compare
         * @return true if equal, false otherwise
         */
        bool operator==(const Linkage &other) const;
        bool operator==(const Vertex &other) const override {
            if (!other.is_linked()) return false;
            const auto &other_link = dynamic_cast<const Linkage&>(other);
            return *this == other_link;
        }
        /**
         * Inequality operator
         * @param other linkage to compare
         * @return true if not equal, false otherwise
         */
        bool operator!=(const Linkage &other) const;
        bool operator!=(const Vertex &other) const override {
            if (!other.is_linked()) return true;
            const auto &other_link = dynamic_cast<const Linkage&>(other);
            return *this != other_link;
        }

        bool similar_root(const Linkage &other) const; // compare the root of the linkage

        /**
         * Overload of Vertex::equivalent operator to compare two linkages (same as ==)
         * @param other linkage to compare
         * @return true if equivalent, false otherwise
         */
        bool equivalent(const Vertex &other) const override {
            if (!other.is_linked())
                return false;

            auto otherPtr = dynamic_cast<const Linkage*>(&other);
            return *this == *otherPtr;
        }

        /**
         * Reccursively update the lines of the linkage
         * @param lines new lines
         * @param update_name whether to update the name of the linkage
         */
        void update_lines(const line_vector &lines, bool update_name) override;

        /**
         * Recursively update the lines of the linkage using a map
         * @param line_map map of old lines to new lines
         */
        void replace_lines(const unordered_map<Line, Line, LineHash> &line_map) override;

        /**
         * Less than operator
         * @note compares flop scaling
         */
        bool operator<(const Linkage &other) const{
            return flop_scale_ < other.flop_scale_;
        }
        bool operator<(const Vertex &other) const override {
            if (!other.is_linked()) return false;
            const auto &other_link = dynamic_cast<const Linkage&>(other);
            return *this < other_link;
        }

        /**
         * Greater than operator
         * @note compares flop scaling
         */
        bool operator>(const Linkage &other) const{
            return flop_scale_ > other.flop_scale_;
        }
        bool operator>(const Vertex &other) const override {
            if (!other.is_linked()) return true;
            const auto &other_link = dynamic_cast<const Linkage&>(other);
            return *this > other_link;
        }

        /**
         * Less than or equal to operator
         * @note compares flop scaling
         */
        bool operator<=(const Linkage &other) const{
            return flop_scale_ <= other.flop_scale_;
        }
        bool operator<=(const Vertex &other) const override {
            if (!other.is_linked()) return false;
            const auto &other_link = dynamic_cast<const Linkage&>(other);
            return *this <= other_link;
        }

        /**
         * Greater than or equal to operator
         * @note compares flop scaling
         */
        bool operator>=(const Linkage &other) const{
            return flop_scale_ >= other.flop_scale_;
        }
        bool operator>=(const Vertex &other) const override {
            if (!other.is_linked()) return true;
            const auto &other_link = dynamic_cast<const Linkage&>(other);
            return *this >= other_link;
        }

        /**
         * convert the linkage to a const vector of vertices
         * @return vector of vertices
         * @note this function is recursive
         */
        vector<ConstVertexPtr> link_vector(bool regenerate = false, bool fully_expand = false) const override;

        /**
         * return a vector of vertices in order
         * @param regenerate whether to regenerate the vertices (deprecated; no-op)
         * @param full_expand whether to fully expand nested intermediates
         */
         vector<ConstVertexPtr> vertices(bool regenerate = false) const;

        /**
         * Return all permutations of the linkage
         * for example, given a graph A->B->C, then other graphs could be:
                B->A->C, B->C->A, A->C->B, C->A->B, C->B->A
         * @param regenerate whether to regenerate the permutations
         * @return vector of permutations
         */
         static inline bool low_memory_ = false; // whether to store permutations in memory for lazy evaluation
        vector<ConstLinkagePtr> permutations(bool regenerate = false) const;

        /**
         * Return the best permutation of the linkage that minimizes the number of contractions and memory
         * @return best permutation of the linkage
         */
        ConstLinkagePtr best_permutation() const;

        /**
         * Return all subgraphs of the linkage
         * @param max_depth maximum depth of subgraphs returned
         * @param with_permutations whether to include permutations of the subgraphs
         * @return vector of subgraphs
         */
        vector<ConstLinkagePtr> subgraphs(size_t max_depth, bool with_permutations = false) const;

        /**
         * find all linked scalars within the linkage
         * @return vector of all linked scalars within the linkage
         */
        vector<ConstVertexPtr> find_scalars() const;

        /**
         * Get connec_map, the map of connections between lines
         * @return connec_map
         */
        const std::vector<std::array<int_fast8_t, 2>> &connec_map() const { return connec_map_; }

        /**
         * Make a series of linkages from vertices into a single linkage
         * @param op_vec list of vertices
         */
        static LinkagePtr link(const vector<ConstVertexPtr> &op_vec);

        /**
         * copy just the members of the linkage that do not depend on the vertices
         * @param to_copy
         */
        void copy_misc(const Linkage& to_copy) {
            id_ = to_copy.id_;
            reused_ = to_copy.reused_;
            addition_ = to_copy.addition_;
        }
        void copy_misc(const ConstLinkagePtr& to_copy) { copy_misc(*to_copy); }

        /**
         * get the sequential scaling of the linkage (flops and memory), excluding intermediates
         * @param fully_expand whether to fully expand nested intermediates
         * @return tuple of flop and memory scaling
         */
        pair<vector<shape>, vector<shape>> scales(bool fully_expand = false) const;

        /**
         * get the total scaling of the linkage (flops and memory), excluding intermediates
         * @param fully_expand whether to fully expand nested intermediates
         * @return tuple of flop and memory scaling maps
         */
         pair<scaling_map, scaling_map> netscales(bool fully_expand = false) const {
            auto [flops, mems] = scales(fully_expand);
            return {scaling_map(flops), scaling_map(mems)};
         }

        static inline string print_type_ = "c++"; // default to python print type

        /**
         * Create generic string representation of linkage
         * @param format_temp if true, make generic string representation
         * @return generic string representation of linkage
         */
        string str(bool format_temp, bool include_lines = true) const;

        string name() const override {
            return str(true, false);
        }

        string str() const override {
            // default to generic string representation when not specified
            return str(true);
        }

        friend ostream &operator<<(ostream &os, const Linkage &linkage) {
            os << linkage.tot_str(true);
            return os;
        }

        /**
        * Get string of contractions and additions
        * @param fully_expand if true, fully_expand contractions recursively
        * @return linkage string
        */
        string tot_str(bool fully_expand = false, bool make_dot = true) const;

        /**
         * goes down the tree and replaces the id of any intermediate vertices to a new value
         * @param target_vertex the vertex to replace
         * @param new_vertex the new vertex to replace with
         * @param only_temps whether to check only for temps
         * @return the copy of linkage with the replaced vertex and a bool indicating if the vertex was replaced
         */
        pair<ConstVertexPtr, bool> replace(const ConstVertexPtr &target_vertex, const ConstVertexPtr &new_vertex, bool only_temps = false) const;

        /**
         * goes down the tree and replaces the id of any intermediate vertices that matched the target to a new value
         * @param target_vertex the vertex to replace
         * @param new_id the new id to replace with
         * @param only_temps whether to check only for temps
         * @return true if the id was replaced, false otherwise
         */
        pair<ConstVertexPtr, bool> replace_id(const ConstVertexPtr &target_vertex, long new_id, bool only_temps = true) const;

        /**
         * goes down the tree and finds all occurences of the target vertex
         * @param target_vertex the vertex to find
         * @param only_temps whether to check only for temps
         * @return vector of all vertices that match the target
         */
        vector<ConstVertexPtr> find_links(const ConstVertexPtr &target_vertex, long search_depth = -1) const;

        /**
         * goes down the tree and returns true if any intermediate vertices have the target ids
         * @param target_ids the id to find
         */
        bool has_temp(const ConstVertexPtr &temp, bool enter_temps = true, long depth = -1) const override;
        bool has_any_temp() const override; // whether the linkage has any intermediate vertices
        vector<ConstVertexPtr> get_temps() const override;
        set<long> get_ids(bool enter_temps = true) const;

        /**
         * Write DOT representation of linkage to file stream (to visualize linkage in graphviz)
         * @param os output stream
         * @param linkage linkage to write
         * @return output stream
         */
        ostream &write_dot(ostream &os, size_t &term_id, size_t &dummy_count, const std::string &color = "black") const;

        /**
         * check if linkage is empty
         * @return true if empty, false otherwise
         */
        bool empty() const override {
            if (left_->empty() && right_->empty()) return true;
            else return Vertex::empty();
        }

        /**
         * Get depth of linkage
         * @return depth of linkage
         */
        size_t depth() const { return depth_; }


    }; // class linkage

    // define cast function from Vertex pointers to Linkage pointers
    static Linkage *as_link(Vertex *vertex) { return dynamic_cast<Linkage *>(vertex); }

} // pdaggerq

#endif //PDAGGERQ_linkage_H
