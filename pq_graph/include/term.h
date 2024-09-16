//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: term.h
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

#ifndef PDAGGERQ_TERM_H
#define PDAGGERQ_TERM_H

#include <string>
#include <vector>
#include <map>
#include <cmath>

#include "../../pdaggerq/pq_string.h"
#include "scaling_map.hpp"
#include "linkage_set.hpp"


namespace pdaggerq {

    typedef vector<pair<string, string>> perm_list;

    /**
     * Term class
     * A term is a product of rhs and a coefficient
     * Each vertex is a string with the vertex name and its indices
     * Term is optimized for floating point operations
     */
    class Term {

    protected:

        ConstVertexPtr lhs_; // vertex on the left hand side of the term
        ConstVertexPtr eq_; // vertex of the equation this term is in (usually the same as lhs_)
        vector<ConstVertexPtr> rhs_; // rhs of the term
        mutable ConstLinkagePtr term_linkage_ = nullptr; // linkage of the term
        mutable vector<string> comments_; // string representation of the original rhs

        /// scaling of the term (stored as a pair of integers, (num virtual, num occupied))
        scaling_map flop_map_; // map of flop scaling with linkage occurrence in term
        scaling_map mem_map_; // map of memory scaling with linkage occurrence in term

        /// list of permutation indices (should generalize to arbitrary number of indices)

        // perm_type_ = 0: no permutation
        // perm_type_ = 1: P(i,j) R(ij;ab) = R(ij;ab) - R(ji;ab)
        // perm_type_ = 2: PP2(i,a;j,b) R(ijk;abc) = R(ijk;abc) + (jik,bac)
        // perm_type_ = 3: PP3(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + (jik;bac) + R(kji;cba)
        // perm_type_ = 6: PP6(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + R(jik;bac) + R(jki;bca) + R(ikj;acb) + R(kji;cba) + R(kij;cab)
        perm_list term_perms_; // list of permutation indices
        size_t perm_type_ = 0; // default is no permutation

    public:

        bool is_optimal_ = false; // flag for if term has optimal linkages (default is false)
        bool needs_update_ = true; // flag for if term needs to be updated (default is true)
        bool generated_linkages_ = false; // flag for if term has generated linkages (default is false)
        bool is_assignment_ = false; // true if the term is an assignment (default is false, using +=)
        string print_override_; // string to override print function

        static inline size_t max_depth_ = -1; // maximum number of rhs in a linkage (no limit by default)
        static inline shape max_shape_; // maximum shape of a linkage
        static inline bool make_einsum = false;


        typedef map<string, vector<string>> condition_map;
        static inline condition_map mapped_conditions_{}; // map of conditionals to their relevant operators
        mutable string original_pq_; // the original pq string representation

        /******** Constructors ********/

        Term() = default;
        ~Term() = default;

        double coefficient_{}; // coefficient of the term

        /**
         * expand rhs of term using a linkage
         * @param term_link linkage to expand into rhs
         */
        void expand_rhs(const ConstVertexPtr &term_link); // expand rhs of term using a linkage
        void expand_rhs(){ expand_rhs(term_linkage()); } // uses term_linkage

        /**
         * Constructor
         * @param name name of the assignment vertex
         * @param pq_str representation of term from pq_helper
         */
        Term(const string &name, const shared_ptr<pq_string>& pq_str);

        /**
         * Constructor
         * @param name name of the assignment vertex
         * @param vertex_strings vector of vertex string representations
         */
        Term(const string &name, const vector<string> &vertex_strings);

        /**
         * Constructor
         * @param lhs_vertex assignment vertex
         * @param vertices vector of rhs
         * @param coefficient coefficient of the term
         */
        Term(const ConstVertexPtr &lhs_vertex, const vector<ConstVertexPtr> &vertices, double coefficient);

        /**
         * Constructor to build assignment of a linkage
         * @param linkage linkage to assign
         */
        explicit Term(const ConstLinkagePtr &linkage, double coeff = 1.0);

        /**
         * Constructor that takes in a single string and overrides printing
         */
        explicit Term(const string &print_override);

        /**
         * Copy constructor
         * @param other term to copy
         */
        Term(const Term &other) = default;

        /**
         * Move constructor
         * @param other term to move
         */
        Term(Term &&other) noexcept = default;

        /******** operator overloads ********/

        /**
         * Assignment operator
         * @param other term to copy
         * @return reference to this term
         */
        Term &operator=(const Term &other) = default;

        /**
         * Deep copy of term where all vertices are cloned
         * @return cloned term
         */
        Term clone() const;

        /**
         * Move assignment operator
         * @param other term to move
         * @return reference to this term
         */
        Term &operator=(Term &&other) noexcept = default;

        /**
         * return reference to vertex at index
         * @param index index of vertex
         * @return reference to vertex at index
         */
        ConstVertexPtr &operator[](size_t index){ return rhs_[index]; }

        /**
         * return const reference to vertex at index
         * @param index index of vertex
         * @return const reference to vertex at index
         */
        const ConstVertexPtr &operator[](size_t index) const{ return rhs_[index]; }


        /**
         * Formats permutation rhs for term
         * @param perm_string string representation of permutation vertex
         */
        void set_perm(const string & perm_string);

        /**
         * Formats permutation rhs for term
         * @param perm_pairs pairs of permutations
         * @param perm_type the type of permutation
         */
        void set_perm(const perm_list& perm_pairs, const size_t perm_type){
            term_perms_ = perm_pairs;
            perm_type_ = perm_type;
        }

        /**
         * reset permutation rhs for term
         */
        void reset_perm(){
            term_perms_.clear();
            perm_type_ = 0;
        }

        /**
         * Get left hand side vertex
         * @return left hand side vertex
         */
        const ConstVertexPtr &lhs() const { return lhs_; }
        ConstVertexPtr &lhs() { return lhs_; }

        /**
        * Get rhs and allow modification
        * @return vector of rhs
        */
        vector<ConstVertexPtr> &rhs() {
            return rhs_;
        }

        /**
         * Get const rhs
         */
        const vector<ConstVertexPtr> &rhs() const {
            return rhs_;
        }

        /**
         * Get reference vertex for the equation
         * @return vertex for the equation
         */
        const ConstVertexPtr &eq() const { return eq_; }
        ConstVertexPtr &eq() { return eq_; }

        /**
         * get all vertices in the term (lhs + rhs)
         */
        vector<ConstVertexPtr> vertices() const {
            vector<ConstVertexPtr> vertices = {lhs_};
            vertices.insert(vertices.end(), rhs_.begin(), rhs_.end());
            return vertices;
        }

        /**
         * permutation indices
         * @return permutation indices
         */
        perm_list &term_perms() { return term_perms_; }
        const perm_list &term_perms() const { return term_perms_; }

        /**
         * Get type of permutation
         * @return type of permutation
         */
        size_t &perm_type() { return perm_type_; }
        const size_t &perm_type() const { return perm_type_; }

        /**
         * Get number of rhs
         */
         size_t size() const { return rhs_.size(); }

        /**
         * begin iterator
         */
        vector<ConstVertexPtr>::iterator begin() { return rhs_.begin(); }

        /**
         * end iterator
         */
        vector<ConstVertexPtr>::iterator end() { return rhs_.end(); }

        /**
         * begin const iterator
         */
        vector<ConstVertexPtr>::const_iterator begin() const { return rhs_.begin(); }

        /**
         * end const iterator
         */
        vector<ConstVertexPtr>::const_iterator end() const { return rhs_.end(); }

        /**
         * Get mutable reference to vertex strings
         * @return vector of vertex strings
         */
        vector<string> &comments() {
            return comments_;
        }

        /**
         * Get const reference to comments
         * @return const vector reference of comments
         */
        const vector<string> &comments() const {
            return comments_;
        }

        /**
         * generate string for comment
         * @param only_flop if true, only flop scaling is included in comment
         * @return string for comment
         */
        string make_comments(bool only_flop = false, bool only_comment=false) const;

        /**
         * set comments variable
         */
        void reset_comments();

        /**
         * Get flop scaling
         * @return map of flop scaling
         */
        const scaling_map &flop_map() const { return flop_map_; }

        /**
         * Get memory scaling
         * @return map of memory scaling
         */
        const scaling_map &mem_map() const { return mem_map_; }

        /**
         * Get worst flop scaling
         */
        shape worst_flop() const { return flop_map_.worst(); }

        /******** Functions ********/

        /**
         * Reorder term for optimal floating point operations and store scaling
         */
        void reorder(bool recompute = false);

         /**
          * Populate flop and memory scaling maps
          * @param perm permutation of the rhs
          * @return a tuple of the flop_scale, mem_scale, and term linkage
          */
         static tuple<scaling_map, scaling_map, LinkagePtr> compute_scaling(const ConstVertexPtr &lhs, const vector<ConstVertexPtr> &arrangement);

         /**
          * Populate flop and memory scaling maps with identity permutation
          */
        void compute_scaling(bool recompute = false);

        /**
         * Determine which mapped_conditions_ are needed for the term
         * @return a map of mapped_conditions_ to their condition (T/F) in this term
         */
        set<string> conditions() const;

        /**
        * Update booleans for optimization
        */
        void request_update();

        /**
         * Compare flop scaling of this term to another term by overloading <
         * @param other term to compare to
         */
        bool operator<(const Term &other) const{
            return flop_map_ < other.flop_map_;
        }

        /**
         * Compare flop scaling of this term to another term by overloading >
         * @param other term to compare to
         */
        bool operator>(const Term &other) const{
            return flop_map_ > other.flop_map_;
        }

        /**
         * Compare rhs of this term to another term by overloading ==
         * @param other term to compare to
         * @note only compares flop scaling.
         *       DOES NOT compare coefficient
         */
        bool operator==(const Term &other) const {

            // check if terms have the same number of rhs vertices
            if (size() != other.size()) return false;

            // do the terms have the same kind of permutation?
            bool same_permutation = perm_type_ == other.perm_type_; // same permutation type?
            if (same_permutation) {
                if (term_perms_ != other.term_perms_)
                    return false; // same permutation pairs?
            } else return false;

            // make sure both terms have exactly the same lhs
            if (lhs_->Vertex::operator!=(*other.lhs_)) return false;

            // exactly the same rhs
            for (size_t i = 0; i < rhs_.size(); i++) {
                if (rhs_[i]->Vertex::operator!=(*other.rhs_[i]))
                    return false;
            }

            // they are the same! (we ignore the coefficient)
            return true;
        }

        /**
         * Compare rhs of this term to another term by overloading !=
         * @param other term to compare to
         * @note only compares flop scaling.
         *       DOES NOT compare coefficient
         */
        bool operator!=(const Term &other) const{
            return !(*this == other);
        }

        /**
         * Create string representation of the term
         * @return string representation of the term
         */
        string str() const;
        string einsum_str() const;

        string operator+(const string &other) const{ return str() + other; }
        friend string operator+(const string &other, const Term &term){ return other + term.str(); }
        friend ostream &operator<<(ostream &os, const Term &term){
            os << term.str();
            return os;
        }

        /**
         * Represent the coefficient as a fraction
         * @param coeff coefficient to represent
         * @param threshold threshold for error of representation
         * @return pair of numerator and denominator
         */
//            static pair<int,int> as_fraction(double coeff, double threshold = 1e-6);

        /**
         * permute terms with a given set of permutations
         * @param perm_list list of permutations
         * @param perm_type type of permutation
         * @return vector of permuted terms (including original term as first element)
         */
        vector<Term> permute(const perm_list &perm_list, size_t perm_type) const;
        vector<Term> expand_perms() const{ return permute(term_perms_, perm_type_); }

        /**
         * Substitute linkage into the term
         * @param linkage linkage to substitute
         * @param allow_equality allow equality of scaling
         * @return boolean indicating if substitution was successful
         */
        bool substitute(const ConstLinkagePtr &linkage, bool allow_equality = false);

        /**
         * collect all possible linkages from all equations
         */
        linkage_set make_all_links() const;

        /**
         * Get the ids of all intermediate vertices within the term
         * @param type type of intermediate ids to get
         * @return tuple of sets of intermediate vertex ids. the lhs ids, rhs ids, and all ids
         */
         tuple<set<long>, set<long>, set<long>> term_ids(char type) const;

        /**
         * get the term linkage
         */
        ConstLinkagePtr &term_linkage(bool recompute = false) const {
            if (term_linkage_ == nullptr || recompute)
                term_linkage_ = Linkage::link(rhs_);
            return term_linkage_;
        }

        /**
         * find best scalar linkage for a given term
         * @param scalars set of scalars to add to
         * @param id id of the ter
         * @return pair of the best scalar linkage and a boolean indicating if a scalar was made
         */
        pair<ConstLinkagePtr,bool> make_scalar(linkage_set &scalars, long &id);

         /**
          * find vertices with self-contractions and format with delta functions
          * @return boolean indicating if self-links were applied
          */
         bool apply_self_links();

        /**
         * insert vertex into term
         */
        void insert(size_t pos, const VertexPtr &op) { rhs_.insert(rhs_.begin() + (int)pos, op); }

        /**
         * check if term includes rhs of the linkage
         * @param linkage linkage to check
         * @return boolean indicating if term includes rhs of the linkage
         */
        bool is_compatible(const ConstLinkagePtr &linkage) const;

        /**
         * swaps the sign of the term
         */
        void swap_sign() { coefficient_ *= -1; }

        /**
         * Replace all two-body operators with density fitting operators
         * @return vector of terms with density fitting operators
         */
        vector<Term> density_fitting();

    }; // end Term class

} // pdaggerq

#endif //PDAGGERQ_TERM_H
