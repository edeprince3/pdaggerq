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
#include "scaling_map.hpp"
#include "vertex.h"
#include "linkage.h"
#include "linkage_set.hpp"
#include "../../pdaggerq/pq_string.h"


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

            mutable LinkagePtr term_linkage_; // linkage of the term

    public:

            bool is_optimal_ = false; // flag for if term has optimal linkages (default is false)
            bool needs_update_ = true; // flag for if term needs to be updated (default is true)
            bool generated_linkages_ = false; // flag for if term has generated linkages (default is false)
            bool is_assignment_ = false; // true if the term is an assignment (default is false, using +=)
            string print_override_; // string to override print function

            static inline size_t max_depth_ = -1; // maximum number of rhs in a linkage (no limit by default)
            static inline shape max_shape_; // maximum shape of a linkage
            static inline bool allow_nesting_ = true;
            static inline bool permute_vertices_ = false;
            static inline bool make_einsum = false;
            static inline size_t depth_ = 0; // depth of nested tmps


            typedef map<string, vector<string>> condition_map;
            static inline condition_map mapped_conditions_{}; // map of conditionals to their relevant operators
            string original_pq_; // the original pq string representation

            /******** Constructors ********/

            Term() = default;
            ~Term() = default;

            double coefficient_{}; // coefficient of the term

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
            explicit Term(const ConstLinkagePtr &linkage, double coeff);

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
             * Get vertex for the equation
             * @return vertex for the equation
             */
            const ConstVertexPtr &eq() const { return eq_; }
            ConstVertexPtr &eq() { return eq_; }

            /**
             * Set left hand side vertex
             */
            void set_lhs(const VertexPtr &lhs) { lhs_ = lhs; }

            /**
             * permutation indices
             * @return permutation indices
             */
            const perm_list &term_perms() const { return term_perms_; }

            /**
             * Get type of permutation
             * @return type of permutation
             */
            size_t perm_type() const { return perm_type_; }

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
             * Get rhs and allow modification
             * @return vector of rhs
             */
            vector<ConstVertexPtr> &rhs() {
                return rhs_;
            }

            /**
             * Set rhs
             * @param vertices vector of rhs
             */
            void set_rhs(const vector<ConstVertexPtr> &rhs) {
                rhs_ = rhs;
            }

            /**
             * Get const rhs
             */
            const vector<ConstVertexPtr> &rhs() const {
                return rhs_;
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
             * function to count permutations of lines in arrangment to match those in the reference lines
             * @param ref_lines the reference lines (usually the lhs of the term)
             * @param arrangement the arrangement of lines to count permutations of
             * @return the number of permutations
             */
            static size_t count_idx_perm(const line_vector &ref_lines, const vector<ConstVertexPtr> &arrangement);


            //TODO: make this a static function when called with a term as an argument

             /**
              * Populate flop and memory scaling maps
              * @param perm permutation of the rhs
              * @return a tuple of the flop_scale, mem_scale, and term linkage
              */
             tuple<scaling_map, scaling_map, LinkagePtr> compute_scaling(const vector<ConstVertexPtr> &arrangement, bool recompute = true);

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
             * Execute an operation of all combinations of subterms
             */
            static void operate_subsets(
                    size_t n, // number of subterms
                    const std::function<void(const vector<size_t>&)>& op, // operation to perform on each subset
                    const std::function<bool(const vector<size_t>&)>& valid_op = nullptr, // operation to check if subset is valid
                    const std::function<bool(const vector<size_t>&)>& break_perm_op = nullptr, // operation to check if permutation should be broken
                    const std::function<bool(const vector<size_t>&)>& terminate_op = nullptr // operation to check if subset should be broken
            );

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
             * get the term linkage
             * @param recompute whether to recompute the linkage
             */
            ConstLinkagePtr term_linkage(bool recompute = true) const {
                if (!recompute && term_linkage_) return term_linkage_;

                if (rhs_.size() > 1)
                    term_linkage_ = Linkage::link(rhs_);
                else if (!rhs_.empty()) term_linkage_ = as_link(make_shared<Vertex>() * rhs_[0]);
                else term_linkage_ = as_link(make_shared<Vertex>() * make_shared<Vertex>());

                // return a clone of the term linkage
                return as_link(term_linkage_->clone());
            }

            /**
             * find best scalar linkage for a given term
             * @param scalars set of scalars to add to
             * @param id id of the term
             */
            bool make_scalar(linkage_set &scalars, size_t id);

             /**
              * find vertices with self-contractions and format with delta functions
              * @return boolean indicating if self-links were applied
              */
             bool apply_self_links();

            /**
             * pop_back vertex from term
             */
            void pop_back() { rhs_.pop_back(); }

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
             * check if term is equivalent to another term
             * @param term1 the first term
             * @param term2 the second term
             * @return boolean indicating if term1 is equivalent to term2
             */
            static bool equivalent(const Term &term1, const Term &term2);


            /**
             * check if term is equivalent to another term up to a permutation (keeping track of the permutation)
             * @param ref_term the first term
             * @param compare_term the second term
             * @return pair of booleans indicating if ref_term is equivalent to compare_term and if the permutation is odd
             */
            static pair<bool, bool> same_permutation(const Term &ref_term, const Term &compare_term);

            /**
             * Replace lines in term with generic lines
             * @return term with generic lines
             */
            Term genericize() const;

            vector<Term> density_fitting();

    }; // end Term class

    struct TermHash { // hash functor for finding similar terms
        size_t operator()(const Term& term) const {
            constexpr LinkageHash link_hasher;
            ConstVertexPtr total_representation = term.lhs() + term.term_linkage();

            return link_hasher(as_link(total_representation));
        }
    };

    struct TermEqual { // predicate functor for finding similar terms
        bool operator()(const Term& term1, const Term& term2) const {
            // determine if permutation is the same
            if (term1.perm_type()  != term2.perm_type()) return false;
            if (term1.term_perms() != term2.term_perms()) return false;

            // get vertices of terms
            vector<ConstVertexPtr> term1_vertices = term1.term_linkage()->link_vector(true);
            vector<ConstVertexPtr> term2_vertices = term2.term_linkage()->link_vector(true);

            // sort by vertex name
            sort(term1_vertices.begin(), term1_vertices.end(), [](const ConstVertexPtr &a, const ConstVertexPtr &b){
                return a->name() < b->name();
            });
            sort(term2_vertices.begin(), term2_vertices.end(), [](const ConstVertexPtr &a, const ConstVertexPtr &b){
                return a->name() < b->name();
            });

            // create linkage of vertices
            ConstLinkagePtr term1_linkage = Linkage::link(term1_vertices);
            ConstLinkagePtr term2_linkage = Linkage::link(term2_vertices);

            // create total representation of terms
            ConstLinkagePtr total_repr1 = as_link(term1.lhs() + term1_linkage);
            ConstLinkagePtr total_repr2 = as_link(term2.lhs() + term2_linkage);

            bool same_terms = *total_repr1 == *total_repr2;

            // ensure sorted linkage representations are the same
            return same_terms;
        }
    };

} // pdaggerq

#endif //PDAGGERQ_TERM_H
