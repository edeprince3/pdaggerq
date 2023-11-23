//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: vertex.h
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

#ifndef PDAGGERQ_VERTEX_H
#define PDAGGERQ_VERTEX_H
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include "line.hpp"
#include "scaling_map.hpp"
#include "../../pdaggerq/pq_tensor.h"
#include <memory>

using std::ostream;
using std::string;
using std::vector;
using std::map;
using std::unordered_map;
using std::shared_ptr;

namespace pdaggerq {

    /**
     * Vertex class
     * Represents a vertex in the form of a string with the vertex name and its indices
     * e.g. "t2(a,b,i,j)" or "eri(a,b,i,j)"
     *
     * The indices are stored in a map of lines to their occupied/virtual and/or block type
     * e.g. { "a" : 0, "b" : 1, "i" : 2, "j" : 3 } for a t2 vertex
     * e.g. { "a" : true, "b" : false, "i" : true, "j" : false } for a t2_abij vertex
     */
    struct Vertex {

        string name_{}; // name of the vertex
        string base_name_{}; // name of vertex without index markup

        // uint_fast8_t is sufficient for up to 255 line indices and is more efficient than size_t, which is 64 bits
        // 255 indices is more than enough for any reasonable vertex.
        // It is important to keep the Vertex class as small as possible because it is constructed many, many times.
        vector<Line> lines_{}; // vector of lines for the vertex
        uint_fast8_t rank_{}; // rank of the vertex
        shape shape_{}; // shape of the vertex

        // boolean identifiers

        // indicates the vertex is not linked to another vertex
        virtual bool is_linked() const { return false; }
        virtual bool is_temp() const { return false; }

        bool has_blk_ = false; // whether the vertex is blocked by spin, range, etc (assumed false by default)
        bool is_sigma_ = false; // whether the vertex is an excited state vertex
        bool is_den_ = false; // whether the vertex is a density-fitting vertex

        // indicates whether the vertex is allowed to be permuted
        static inline bool allow_permute_ = false;

        /****** Constructors ******/

        Vertex();

        /**
         * Constructor
         * @param vertex_string string representation of the vertex
         */
        explicit Vertex(const string &vertex_string);

        /// Constructors from different pq tensors
        explicit Vertex(const delta_functions &delta);
        explicit Vertex(const integrals &integral, const string &type);
        explicit Vertex(const amplitudes &amplitude, char type);

        /**
         * Constructor
         * @param name name of the vertex without index markup
         * @param vir_map map of virtual lines to their index in the vertex
         * @param occ_map map of occupied lines to their index in the vertex
         */
        Vertex(string base_name, const vector<Line>& lines);

        /**
         * Destructor
         */
        virtual ~Vertex() = default;

        /**
         * Copy constructor
         * @param other vertex to copy
         */
        Vertex(const Vertex &other) = default;

        /**
         * move constructor
         * @param other vertex to move
         */
        Vertex(Vertex &&other) noexcept = default;

        /**
         * get lines of vertex from vir_map and occ_map
         */
        const vector<Line> &lines() const { return lines_; }

        /**
         * sets parameters of the vertex from the lines of the vertex
         * @param lines vector of lines
         * @param update_name boolean indicating whether the name of the vertex should be updated (default true)
         */
        void update_lines(const vector<Line> &lines, bool update_name = true);

        /**
         * sets parameters of the vertex from the string representation of the vertex
         * @param lines vector of lines
         * @param blk_string string representation of the blocks in this vertex
         */
        void set_lines(const vector<string> &lines, const string &blk_string);

        /**
         * This function is used to update the name of the vertex from the lines of the vertex
         * @param ovstring string representation of the vertex
         * @param new_blk_string string representation of the blocks in this vertex
         */
        void format_name(const string &ovstring, const string &new_blk_string);

        /**
         * Sorts lines such that virtual lines come first; if the vertex is blocked, then
         * the blocked lines (alpha/active) come first, followed by the full lines (full/beta) for the same virtual/occupied block
         */
        void sort();
        static void sort(vector<Line> &lines); // static version of sort

        /**
         * get the ovstring of from lines
         * @param lines vector of lines
         */
        static string ovstring(const vector<Line> &lines);

        /**
         * get the ov_string and blk_string of this vertex from its lines
         * @return string {blk_string}_{ov_string}
         */
        string dimstring() const;


        /**
         * get the ovstring of this vertex from its lines
         * @param lines vector of lines
         */
        string ovstring() const { return ovstring(lines_); }

        /**
         * permute the vertex to i'th representation
         * @param perm_id index of unique permutation
         * @return boolean reference if sign change is needed
         * @return the permuted vertex
         */
        Vertex permute(size_t perm_id, bool &swap_sign) const;

        /**
         * permutes this vertex to have the same structure as the other vertex (if possible)
         * @param other vertex to match
         * @param swap_sign boolean indicating whether a sign change is needed
         * @return permuted vertex
         */
        Vertex permute_like(const Vertex &other, bool &swap_sign) const;

        /**
         * utilize symmetry of eri vertex to reduce the number of unique indices
         * @return boolean if sign change is needed
         */
        bool permute_eri();

        /**
         * Determines if two rhs are equal up to a permutation of the indices
         * @param left first vertex
         * @param right second vertex
         * @param swap_signs boolean reference indicating whether a sign change is needed
         * @return boolean indicating whether the rhs are equal up to a permutation
         */
        friend bool is_isomorphic(const Vertex &left, const Vertex &right, bool &swap_signs);

        /**
         * Determines if other vertex is equal up to a permutation of the indices
         * @param other vertex to compare to
         * @return boolean indicating whether the rhs are equal up to a permutation
         */
        bool isomorphic(const Vertex &other) const;


        /****** Vertex overloads ******/

        /**
         * Assignment operator
         * @param other vertex to copy
         * @return reference to this vertex
         */
        Vertex &operator=(const Vertex &other) = default;

        /**
         * Move assignment operator
         * @param other vertex to move
         * @return reference to this vertex
         */
        Vertex &operator=(Vertex &&other) noexcept = default;

        /**
         * Compare two rhs for equality
         * @return boolean indicating whether the rhs are equal
         */
        bool operator==(const Vertex &other) const;

        /**
         * Compare two rhs for inequality
         * @return boolean indicating whether the rhs are not equal
         */
        bool operator!=(const Vertex &other) const;

        /**
         * Compare the n_ops of two rhs via < operator
         * @return boolean indicating whether the size of the lhs is less than the n_ops of the rhs_vertex
         */
        bool operator<(const Vertex &other) const;

        /**
         * Compare the n_ops of two rhs via > operator
         * @return boolean indicating whether the size of the lhs is greater than the n_ops of the rhs_vertex
         */
        bool operator>(const Vertex &other) const{ return other < *this; }

        /****** Getters ******/

        /**
         * Get the name of the vertex
         * @return string of the vertex name
         */
        const string &name() const { return name_; }

        /**
         * Get the base name of the vertex
         * @return string of the vertex base name
         */
        const string &base_name() const { return base_name_; }

        /**
         * set the name of the vertex
         * @param name string of the vertex name
         */
        void set_name(const string &name) { name_ = name; }

        /**
         * set the base name of the vertex
         * @param name string of the vertex base name
         * TODO: use this function to also update the name of the vertex
         */
        void set_base_name(const string &name) { base_name_ = name; }

        /**
         * returns the rank of the vertex
         * @return rank of the vertex
         */
        uint_fast8_t rank() const { return rank_; }

        /**
         * represents the dimensions of the lines of the vertex (occ, vir, block-type, sigma, den)
         * @return shape object
         */
        shape dim() const { return {lines_}; }

        /**
         * turns blk_map into a string corresponding to index of lines
         * @return string representation of block
         */
        string blk_string() const;

        /**
         * get whether the vertex is blocked by spin, range, or other
         * @return boolean indicating whether the vertex has blocks
         */
        bool has_blks() const { return has_blk_; }

        /**
         * string representation of the vertex
         * @return string representation of the vertex
         */
        virtual string str() const;
        string line_str() const;
        string operator+(const string &other) const { return str() + other; }
        friend string operator+(const string &other, const Vertex &op) { return other + op.str(); }

        friend ostream &operator<<(ostream &os, const Vertex &op){
            os << op.str();
            return os;
        }

        /**
         * check if vertex is a trace vertex
         * @return a map of the labels of the self-contractions of the vertex
         *         and a pair of the line with the frequency of the label
         */
        map<Line, uint_fast8_t> self_links() const;

        /**
         *
         * @param internal_lines
         * @return
         */
        vector<shared_ptr<Vertex>> make_self_linkages(map<Line, uint_fast8_t> &self_links);

        /**
         * return n_ops of vertex (number of lines)
         * @return n_ops of vertex
         */
        uint_fast8_t size() const { return lines_.size(); }

        /**
         * check if vertex is initialized
         * @return boolean indicating whether the vertex is initialized
         */
        virtual bool empty() const { return name_ == "Empty"; }

        /**
         * check if linkage is a scalar
         * @return true if scalar, false otherwise
         * @note a scalar is a linkage with no lines
         */
        bool is_scalar() const {
            return lines_.empty();
        }

        /**
         * return begin iterator of lines
         * @return begin iterator of lines
         */
        vector<Line>::const_iterator begin() const { return lines_.begin(); }

        virtual /**
         * return end iterator of lines
         * @return end iterator of lines
         */
        vector<Line>::const_iterator end() const { return lines_.end(); }

        /**
         * overload [] operator
         * @return reference to line at index i
         */
        Line& operator[](uint_fast8_t i) { return lines_[i]; }

        /**
        * const overload [] operator
        * @return const reference to line at index i
        */
        const Line& operator[](uint_fast8_t i) const { return lines_[i]; }

        /**
         * checks if lines in other vertex are in this vertex and that the blocks are the same
         * @param other other vertex to compare lines
         * @return boolean indicating whether the lines are the same
         */
        bool same_lines(const Vertex &other) const;

        /**
         * modifies the vertex to have generic indices
         */
        void genericize();

        /**
         * returns a list of lines with generic indices
         * @param lines reference to list of lines
         * @return same lines with generic indices
         */
        static vector<Line> general_lines(const vector<Line>& lines);

        /**
         * returns a new vertex with generic indices
         * @return new vertex with generic indices
         */
        Vertex generic() const;

        /**
         * checks if the labels between two rhs are equivalent in terms of occupation and blocks
         * @param op1 first vertex
         * @param other second vertex
         * @return boolean indicating whether the labels are equivalent
         */
        bool equivalent(const Vertex &other) const;

        /**
         * Designates the vertex as an excited state vertex
         */
        void make_sigma();

        /**
         * Removes the sigma designation from the vertex
         */
        void remove_sigma();
    }; // end Vertex class

    // typedef for shared pointer to Vertex
    typedef shared_ptr<Vertex> VertexPtr;



} // pdaggerq

#endif //PDAGGERQ_VERTEX_H
