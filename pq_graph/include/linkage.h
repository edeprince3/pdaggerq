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
#include "vertex.h"
#include "scaling_map.hpp"
#include "timer.h"
#include <memory>
#include <mutex>
#include <utility>


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

    static LinkagePtr as_link(const VertexPtr &vertex)  { return static_pointer_cast<Linkage>(vertex); }
    static VertexPtr as_vert(const LinkagePtr &linkage) { return static_pointer_cast<Vertex>(linkage); }
    static ConstLinkagePtr as_link(const ConstVertexPtr &vertex)  { return static_pointer_cast<const Linkage>(vertex); }
    static ConstVertexPtr as_vert(const ConstLinkagePtr &linkage) { return static_pointer_cast<const Vertex>(linkage); }

    /**
     * Perform linkage of two vertices by overload of * operator
     * @param other vertex to contract with
     * @return linkage of the two vertices
     * @note this is an extern function to allow for operator overloading outside of the namespace
     */
    extern VertexPtr operator*(const ConstVertexPtr &left, const ConstVertexPtr &right);
    extern VertexPtr operator*(const VertexPtr &left, const VertexPtr &right);

    /**
     * Perform linkage of two vertices by overload of + operator
     * @param other vertex to add
     * @return linkage of linkage
     * @note this is an extern function to allow for operator overloading outside of the namespace
     */
    extern VertexPtr operator+(const ConstVertexPtr &left, const ConstVertexPtr &right);

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

    public:
        long id_ = -1; // id of the linkage (default to -1 if not set)
        size_t depth_{}; // number of vertices in the linkage
        bool is_addition_ = false; // whether the linkage is an addition; else it is a contraction
        bool is_reused_ = false; // whether the linkage is a shared operator (can be extracted)


        bool is_temp() const override { return id_ != -1 || is_reused_; } // whether the linkage corresponds to an intermediate contraction
        bool is_linked() const override { return true; } // indicates the vertex is linked to another vertex
        long id() const override { return id_; } // get the id of the linkage
        bool is_reused() const override { return is_reused_; } // whether the linkage is reused


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
        Linkage(const ConstVertexPtr &left, const ConstVertexPtr &right, bool is_addition);

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
        vector<Line> int_lines() const;

        /**
         * Replace the lines of the linkage
         * @param lines new lines
         * @note this function will recursively replace the lines of the vertices
         */
        void replace_lines(const unordered_map<Line, Line, LineHash> &line_map) override;

        /**
         * This function will rebalance the linkage by sorting the left and right vertices
         * @note this function is recursive
         * @note this function will not modify intermediates
         * @return the rebalanced linkage
         */
        static ConstVertexPtr tree_sort(const ConstVertexPtr &root);
        ConstVertexPtr tree_sort() const override;

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
        ConstVertexPtr safe_clone() const override;
        VertexPtr clone() const override;


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
         * Tests if two linkages are equivalent up to permutation of the external lines
         * @param other linkage to compare
         * @return pair of bools:
         *      1) true if equivalent up to permutation
         *      2) true if the parity of the permutation is odd
         */
        pair<bool, bool> permuted_equals(const Linkage &other) const;


        /**
         * Inequality operator
         * @param other linkage to compare
         * @return true if not equal, false otherwise
         */
        bool operator!=(const Linkage &other) const;

        /**
         * Less than operator
         * @note compares flop scaling
         */
        bool operator<(const Linkage &other) const{
            return flop_scale_ < other.flop_scale_;
        }

        /**
         * Greater than operator
         * @note compares flop scaling
         */
        bool operator>(const Linkage &other) const{
            return flop_scale_ > other.flop_scale_;
        }

        /**
         * Less than or equal to operator
         * @note compares flop scaling
         */
        bool operator<=(const Linkage &other) const{
            return flop_scale_ <= other.flop_scale_;
        }

        /**
         * Greater than or equal to operator
         * @note compares flop scaling
         */
        bool operator>=(const Linkage &other) const{
            return flop_scale_ >= other.flop_scale_;
        }

        /**
        * convert the linkage to a vector of vertices in order
        * @param result vector of vertices
        * @note this function is recursive
        */
        void link_vector(vector<ConstVertexPtr> &result, size_t &i, bool regenerate, bool full_expand) const;

        /**
         * convert the linkage to a vector of vertices in order
         * @param regenerate whether to regenerate the vertices (deprecated; no-op)
         * @param full_expand whether to fully expand nested intermediates
         */


        /**
         * convert the linkage to a const vector of vertices
         * @return vector of vertices
         * @note this function is recursive
         */
        const vector<ConstVertexPtr> &link_vector(bool regenerate = false, bool full_expand = false) const;

        /**
         * return a vector of vertices in order
         * @param regenerate whether to regenerate the vertices (deprecated; no-op)
         * @param full_expand whether to fully expand nested intermediates
         */
         const vector<ConstVertexPtr> &vertices(bool regenerate = false) const;

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
            is_reused_ = to_copy.is_reused_;
            is_addition_ = to_copy.is_addition_;
        }
        void copy_misc(const ConstLinkagePtr& to_copy) { copy_misc(*to_copy); }

        /**
         * Returns a list of all flop and mem scales within a series of linkages and the linkage
         * @param op_vec list of vertices
         * @return the resulting linkage with the list of all flop and mem scales
         */
        static tuple<ConstLinkagePtr, vector<shape>, vector<shape>> link_and_scale(const vector<ConstVertexPtr> &op_vec);

        /**
         * Get flop cost of linkage
         */
        const shape &flop_scale() const { return flop_scale_; }

        /**
         * Get memory cost of linkage
         */
        const shape &mem_scale() const { return mem_scale_; }

        /**
         * Create generic string representation of linkage
         * @param make_generic if true, make generic string representation
         * @return generic string representation of linkage
         */
        string str(bool make_generic, bool include_lines = true) const;

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
        * @param expand if true, expand contractions recursively
        * @return linkage string
        */
        string tot_str(bool expand = false, bool make_dot = true) const;

        /**
         * Write DOT representation of linkage to file stream (to visualize linkage in graphviz)
         * @param os output stream
         * @param linkage linkage to write
         * @return output stream
         */
        ostream &write_dot(ostream &os, const std::string &color = "black", bool reset = false) const;

        /**
         * check if linkage is empty
         * @return true if empty, false otherwise
         */
        bool empty() const override {
            if (depth_ == 0) return true;
            else return Vertex::empty();
        }

        /**
         * Get depth of linkage
         * @return depth of linkage
         */
        size_t depth() const override { return depth_; }

    }; // class linkage

    // define cast function from Vertex pointers to Linkage pointers  and vice versa

    static Linkage *as_link(Vertex *vertex) { return dynamic_cast<Linkage *>(vertex); }

    static Vertex *as_vert(Linkage *linkage) { return dynamic_cast<Vertex *>(linkage); }

} // pdaggerq

#endif //PDAGGERQ_linkage_H
