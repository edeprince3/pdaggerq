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
#include <unordered_set>
#include <memory>

#include "../../pdaggerq/pq_tensor.h"
#include "shape.hpp"

using std::ostream;
using std::string;
using std::vector;
using std::map;
using std::unordered_map;
using std::shared_ptr;

namespace pdaggerq {

    // forward declaration of Vertex and Linkage
    struct Vertex;
    class Linkage;

    // typedef for shared pointer to Vertex and Linkage
    typedef shared_ptr<Vertex> VertexPtr;
    typedef shared_ptr<Linkage> LinkagePtr;
    typedef shared_ptr<const Vertex>  ConstVertexPtr;
    typedef shared_ptr<const Linkage> ConstLinkagePtr;

    typedef std::vector<ConstVertexPtr> vertex_vector;
    typedef std::vector<ConstLinkagePtr> linkage_vector;

    /**
     * Vertex class
     * Represents a vertex in the form of a string with the vertex name and its indices
     * e.g. "t2(a,b,i,j)" or "eri(a,b,i,j)"
     *
     * The indices are stored in a map of lines to their occupied/virtual and/or block type
     * e.g. { "a" : 0, "b" : 1, "i" : 2, "j" : 3 } for a t2 vertex
     * e.g. { "a" : true, "b" : false, "i" : true, "j" : false } for a t2_abij vertex
     */
    struct Vertex : public std::enable_shared_from_this<Vertex> {

        string name_{}; // name of the vertex
        string base_name_{}; // name of vertex without index markup

        // uint_fast8_t is sufficient for up to 255 line indices and is more efficient than size_t, which is 64 bits
        // 255 indices is more than enough for any reasonable vertex.
        // It is important to keep the Vertex class as small as possible because it is constructed many, many times.
        line_vector lines_{}; // vector of lines for the vertex
        uint_fast8_t rank_{}; // rank of the vertex
        shape shape_{}; // shape of the vertex

        // gives the type of the vertex:
        // 'a' for amplitude, 'p' for permutation, 'v' for arbitrary vertex (default)
        char vertex_type_ = 'v';

        // boolean identifiers
        bool has_blk_ = false; // whether the vertex is blocked by spin, range, etc (assumed false by default)
        bool is_sigma_ = false; // whether the vertex is an excited state vertex
        bool is_den_ = false; // whether the vertex is a density-fitting vertex

        // indicates whether the vertex is allowed to be permuted
        static inline bool allow_permute_ = true;
        static inline bool use_trial_index = false;
        static inline bool permute_eri_ = true;
        static inline string print_type_ = "c++"; // default print type is c++

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
        Vertex(string base_name, const line_vector& lines);

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
         * deep copy of vertex returned as a pointer
         * @return pointer to deep copy of vertex
         */
        virtual VertexPtr clone() const {
            return std::make_shared<Vertex>(*this);
        }

        /**
         * copy of vertex returned as a pointer (linkage will have shallow copy of left and right)
         * @return pointer to shallow copy of vertex
         */
        virtual VertexPtr shallow() const {
            return std::make_shared<Vertex>(*this);
        }

        /**
         * move constructor
         * @param other vertex to move
         */
        Vertex(Vertex &&other) noexcept = default;

        /**
         * get lines of vertex from vir_map and occ_map
         */
        const line_vector &lines() const { return lines_; }
        line_vector &lines() { return lines_; }

        /**
         * sets parameters of the vertex from the lines of the vertex
         * @param lines vector of lines
         * @param update_name boolean indicating whether the name of the vertex should be updated (default true)
         */
        virtual void update_lines(const line_vector &lines, bool update_name = true);


        /**
         * replaces the lines of the vertex with the lines in the argument
         * @param lines vector of lines
         */
        virtual void replace_lines(const unordered_map<Line, Line, LineHash> &line_map);

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
        void format_name();
        void update_name(const string &base_name = "") {
            if (!base_name.empty())
              base_name_ = base_name;
            format_name();
        }

        /**
         * Sorts lines such that virtual lines come first; if the vertex is blocked, then
         * the blocked lines (alpha/active) come first, followed by the full lines (full/beta) for the same virtual/occupied block
         */
        void sort();
        static void sort(line_vector &lines); // static version of sort

        /**
         * get the ovstring of from lines
         * @param lines vector of lines
         */
        static string ovstring(const line_vector &lines);

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
         * bring vertex to a canonical form and determine if a sign change is needed
         * @return boolean if sign change is needed
         */
        bool permute_eri();


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
        virtual bool operator==(const Vertex &other) const;

        /**
         * Compare two rhs for inequality
         * @return boolean indicating whether the rhs are not equal
         */
        virtual bool operator!=(const Vertex &other) const;

        /**
         * Compare the n_ops of two rhs via < operator
         * @return boolean indicating whether the size of the lhs is less than the n_ops of the rhs_vertex
         */
        virtual bool operator<(const Vertex &other) const;
        virtual bool operator<=(const Vertex &other) const;

        /**
         * Compare the n_ops of two rhs via > operator
         * @return boolean indicating whether the size of the lhs is greater than the n_ops of the rhs_vertex
         */
        virtual bool operator>(const Vertex &other) const{ return other < *this; }
        virtual bool operator>=(const Vertex &other) const{ return other <= *this; }

        /****** Getters ******/

        /**
         * Get the name of the vertex
         * @return string of the vertex name
         */
        virtual string name() const { return name_; }

        /**
         * Get the base name of the vertex
         * @return string of the vertex base name
         */
        const string &base_name() const { return base_name_; }

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
        string line_str(bool sort = false) const;
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
        vertex_vector make_self_linkages(map<Line, uint_fast8_t> &self_links);

        /**
         * return n_ops of vertex (number of lines)
         * @return n_ops of vertex
         */
        uint_fast8_t size() const { return lines_.size(); }

        /**
         * check if vertex is initialized
         * @return boolean indicating whether the vertex is initialized
         */
        virtual bool empty() const { return name_.empty() || base_name_.empty(); }

        /**
         * check if vertex is a scalar
         * @return true if scalar, false otherwise
         * @note a scalar is a vertex with no lines
         */
        bool is_scalar() const {
            return lines_.empty();
        }

        /**
         * check if vertex is a constant
         */
        bool is_constant() const;

        /**
         * return value of the constant vertex. if the vertex is not a constant, returns 0.0
         */
        double value() const;

        /**
         * return begin iterator of lines
         * @return begin iterator of lines
         */
        line_vector::const_iterator begin() const { return lines_.begin(); }

        virtual /**
         * return end iterator of lines
         * @return end iterator of lines
         */
        line_vector::const_iterator end() const { return lines_.end(); }

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
         * checks if the labels between two rhs are equivalent in terms of occupation and blocks
         * @param op1 first vertex
         * @param other second vertex
         * @return boolean indicating whether the labels are equivalent
         */
        virtual bool equivalent(const Vertex &other) const;


        /// virtual functions for inherited Linkage class (refer to linkage.h for more information

        virtual bool is_linked() const { return false; }
        virtual bool is_temp() const { return false; }
        virtual bool is_addition() const { return false; } // whether the linkage is an addition
        virtual bool is_expandable() const { return false; } // whether the linkage is expandable
        virtual vertex_vector link_vector(bool regenerate = false, bool fully_expand = false) const { return {shared_from_this()}; }
        virtual bool is_reused() const { return false; } // whether the linkage is reused
        virtual size_t depth() const { return 0; }
        virtual bool &is_addition() { static bool null = false; null = false; return null; } // whether the linkage is an addition
        virtual bool &is_reused() { static bool null = false; null = false; return null; } // whether the linkage is reused
        virtual bool is_leaf(bool expand = true) const { return true; }
        virtual long id() const { return false; }
        virtual long &id() { static long null = -1; null = -1; return null; }
        virtual string type() const { return "vertex"; }
        virtual bool has_temp(const ConstVertexPtr &other, bool enter_temp = true, long depth = -1) const { return false; }
        virtual bool same_temp(const ConstVertexPtr &other) const { return false; }
        virtual bool has_any_temp() const { return false; }
        virtual vertex_vector get_temps() const { return {}; }

    }; // end Vertex class

} // pdaggerq

#endif //PDAGGERQ_VERTEX_H
