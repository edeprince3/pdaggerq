//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: equation.h
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

#ifndef PDAGGERQ_EQUATION_H
#define PDAGGERQ_EQUATION_H

#include <string>
#include <vector>
#include <iostream>

#include "term.h"

using std::cout;
using std::endl;

namespace pdaggerq {

    struct TermHash;
    struct TermEqual;

    /**
     * Equation class
     * Represents an equation in the form of a vector of terms
     * Each term is a vector of rhs
     * Each vertex is a string with the vertex name and its indices
     */
    class Equation {

        string name_; // name of the equation
        VertexPtr assignment_vertex_; // assignment vertex
        vector<Term> terms_; // terms in the equation

        /// scaling of the equation
        scaling_map flop_map_; // map of flop scaling with linkage occurrence in equation
        scaling_map mem_map_; // map of memory scaling with linkage occurrence in equation

    public:
        static inline size_t nthreads_ = 1; // number of threads to use when substituting
        static inline bool permuted_merge_ = false; // whether to merge terms with permutations
        static inline bool separate_conditions_ = true; // whether to separate terms into their conditions
        static inline bool no_scalars_ = false; // whether to remove scalar terms
        bool is_temp_equation_ = false; // whether this is an equation to hold intermediates
        bool allow_substitution_ = true; // whether to allow substitution of linkages

        // default constructor
        Equation() = default;

        // default constructor with name
        explicit Equation(const string &name) {
            *this = Equation();
            name_ = name;
            assignment_vertex_ = make_shared<Vertex>(name);
            //terms_.emplace_back();
        }

        /**
         * Constructor
         * @param name name of the equation
         * @param term_strings vector of term string representations
         */
        Equation(const string &name, const vector<vector<string>> &term_strings);

        /**
         * Constructor
         * @param name name of the equation
         * terms vector of terms
         */
        Equation(const string &name, const vector<Term> &terms);

        /**
         * Constructor
         * @param vertex vertex of the equation
         * terms vector of terms
         */
        Equation(const VertexPtr &assignment, const vector<Term> &terms);

        /**
         * Copy constructor
         * @param other equation to copy
         */
        Equation(const Equation &other) = default;

        /**
         * move constructor
         * @param other equation to move
         */
        Equation(Equation &&other) noexcept = default;

        /**
         * Assignment operator
         * @param other equation to copy
         * @return this
         */
        Equation &operator=(const Equation &other) = default;

        /**
         * move assignment operator
         * @param other equation to move
         * @return this
         */
        Equation &operator=(Equation &&other) noexcept = default;

        /**
         * Destructor
         */
        ~Equation() = default;

        /**
         * Reorder rhs in each term
         */
        void reorder(bool recompute = false);

        /**
         * Collect flop and memory scaling of the equation from each term
         * @param regenerate whether to regenerate the scaling for terms
         */
        void collect_scaling(bool regenerate = true);

        /**
         * Get the name of the equation
         * @return name of the equation
         */
        const string &name() const { return name_; }

        /**
         * Get the assignment vertex
         * @return assignment vertex
         */
        const VertexPtr &assignment_vertex() const { return assignment_vertex_; }

        /**
         * Get the terms in the equation
         * @return terms in the equation
         */
        const vector<Term> &terms() const { return terms_; }
        vector<Term> &terms() { return terms_; }

        /**
         * Get begin iterator of terms
         * @return begin iterator of terms
         */
        vector<Term>::const_iterator begin() const { return terms_.begin(); }
        vector<Term>::iterator begin() { return terms_.begin(); }

        /**
         * Get end iterator of terms
         * @return end iterator of terms
         */
        vector<Term>::const_iterator end() const { return terms_.end(); }
        vector<Term>::iterator end() { return terms_.end(); }

        /**
         * Get number of terms
         * @return number of terms
         */
         size_t size() const { return terms_.size(); }

         /**
          * clear the equation
          */
         void clear() { terms_.clear(); }

         /**
          * check if the equation is empty
          * @return true if empty, false otherwise
          */
         bool empty();

        /**
         * overload [] to get reference to term
         * @param i index of term
         */
         Term &operator[](size_t i) { return terms_[i]; }

        /**
        * const overload [] to get reference to term
        * @param i index of term
        */
        const Term &operator[](size_t i) const { return terms_[i]; }

        /**
         * Get the flop scaling map
         * @return flop scaling map
         */
        const scaling_map &flop_map() const { return flop_map_; }

        /**
         * Get the memory scaling map
         * @return memory scaling map
         */
        const scaling_map &mem_map() const { return mem_map_; }

        /**
         * Get the worst flop scaling
         * @return worst flop scaling
         */
        shape worst_flop() const { return flop_map_.worst(); }


        /**
         * Get the string representation of each term in the equation
         * @return string representation of each term in the equation
         */
        vector<string> to_strings() const;

        friend ostream &operator<<(ostream &os, const Equation &eq){
            for (const string &s : eq.to_strings()) os << "    " << s << endl;
            return os;
        }

        /**
         * Write DOT representation of terms to file stream (to visualize linkage in graphviz)
         * @param os output stream
         * @param color color of the vertices and edges
         * @return output stream
         */
        ostream &write_dot(ostream &os, size_t& term_count, size_t& dummy_count, const string &color = "black");

        /**
         * substitute a linkage into the equation
         * @param linkage linkage to substitute
         * @param allow_equality allow equality of scaling
         * @return number of substitutions
         */
        size_t substitute(const LinkagePtr &linkage, bool allow_equality = false);

        /**
         * test a linkage substituted into the equation
         * @param linkage linkage to substitute
         * @param test_flop_map reference to flop scaling map that collects the flop scaling of the substitution
         * @return number of substitutions
         */
        size_t test_substitute(const MutableLinkagePtr &linkage, scaling_map &test_flop_map, bool allow_equality = false);

        /**
         * collect all possible linkages from all terms
         */
        linkage_set make_all_links(bool compute_all = true);

        /**
         * substitute scalars into the equation
         * @param scalars set of scalars to add
         * @param n_temps reference to number of tmps
         * @return linkage set of scalars
         */
        void make_scalars(linkage_set &scalars, long &n_temps);

        /**
         * add a term to the equation
         * @param term term to add
         * @param index index to add term at (default: 0)
         */
        void insert(const Term& term, int index = 0);

        /**
         * This finds terms that have the same rhs and merges them
         */
        size_t merge_terms();

        // data structure that maps terms to their count_, comments, and net coefficient using a hash table
        typedef unordered_map<Term, pair<size_t, pair<string, double>>, TermHash, TermEqual> merge_map_type;

        /**
         * Gets pointer to terms that contain a given linkage
         * @param linkage linkage to search for
         * @return pointer to terms that contain a given linkage
         */
        set<Term *> get_temp_terms(const LinkagePtr& linkage);

        /**
         * Make a deep copy of the equation, making new shared pointers for vertices and linkages
         * @return deep copy of the equation
         */
        Equation clone() const;

        /**
         * Sort terms in the equation
         * group together terms by
         *      1) their conditions
         *      2) the scaling of their linkages
         *      3) their permutation container
         *      4) the rhs of the terms
         * @return sorted equation
         */
        void rearrange(const string &type = "all");

        /**
         * Sorts tmps by the maximum id of the rhs in the linkage and the tmp itself
         * @param terms vector of terms to sort
         * @param type type of tmps to sort (t for tmps, s for scalars, r for reuse)
         */
        static void sort_tmp_type(vector<Term> &terms, const string &type = "temp");

        /**
         * get 'if' block string formatting
         * @param conditions set of conditions
         * @return string of 'if' block
         */
        static string condition_string(set<string> &conditions);
    }; // class Equation

} // pdaggerq

#endif //PDAGGERQ_EQUATION_H
