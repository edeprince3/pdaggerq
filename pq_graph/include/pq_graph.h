//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_graph.h
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

#ifndef PDAGGERQ_PQ_GRAPH_H
#define PDAGGERQ_PQ_GRAPH_H

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    #include "pybind11/pybind11.h" // surpresses warnings from pybind11
#pragma GCC diagnostic pop


#include <string>
#include <vector>
#include <fstream>
#include "equation.h"
#include "term.h"
#include "vertex.h"
#include "linkage.h"
#include "scaling_map.hpp"
#include "linkage_set.hpp"
#include "timer.h"
#include "../../pdaggerq/pq_string.h"
#include "../../pdaggerq/pq_helper.h"

using std::ofstream;

namespace pdaggerq {

    class PQGraph {
        map<string, Equation> equations_; // equations to be optimized
        map<string, linkage_set> all_linkages_ = { // all intermediate linkages
                                                    {"scalars", linkage_set(256)},
                                                    {"tmps", linkage_set(256)},
                                                    {"reuse_tmps", linkage_set(256)}
                                                 };

        // counts of tmps and scalars
        map<string, size_t> temp_counts_ = {{"scalars", 0}, {"tmps", 0}, {"reuse_tmps", 0}};
        linkage_set tmp_candidates_; // all possible linkages in the equations

        bool is_reordered_ = false; // whether the equations have been reordered
        bool is_reused_ = false; // whether the equations have been reused
        bool has_perms_merged_ = false; // whether the equations have merged permutations
        bool reuse_permutations_ = true; // whether to reuse permutations during the reusing step
        bool use_as_array_ = true; // whether to use the array version of the tmps

        Timer build_timer; // timer for construction of the equations
        Timer reorder_timer; // timer for the reorder function
        Timer substitute_timer; // timer for the substitute function

        Timer update_timer; // timer for updating equations

        /// scaling of the equations
        scaling_map flop_map_; // map of flop scaling with linkage occurrence in all equations

        scaling_map mem_map_; // map of memory scaling with linkage occurrence in all equations
        scaling_map flop_map_init_; // map of flop scaling before reordering

        scaling_map mem_map_init_; // map of memory scaling before reordering
        scaling_map flop_map_pre_; // map of flop scaling before reordering or before subexpression elimination

        scaling_map mem_map_pre_; // map of memory scaling before reordering or before subexpression elimination
        shape bottleneck_flop_; // bottleneck flop scaling of all equation

        shape bottleneck_mem_; // bottleneck memory scaling of all equation

        /// options for the builder
        size_t max_temps_ = -1; // maximum number of temporary rhs (-1 for no limit by overflow)
        bool make_scalars_ = true; // whether to format dot products and traces as scalars
        bool batched_ = false; // whether to use batched substitution
        int num_threads_ = 1; // number of threads to use
        bool verbose = true; // whether to print verbose output
        bool allow_merge_ = false; // whether to merge terms

        /// options for building sigma vectors
        //bool format_eom_ = false; // whether to format equations for the sigma build
        bool has_sigma_vecs_ = false;
        bool store_trials_ = true; // whether to store the sigma vectors in the builder


    public:

        // default constructor
        PQGraph() = default;

        explicit PQGraph(const pybind11::dict& options){
            set_options(options);
        }

        /**
         * Set options for PQ GRAPH
         * @param options dictionary of options
         */
        void set_options(const pybind11::dict& options);

        /**
         * add an equation to the builder from a pq_helper object
         * @param pq pq_helper object of the equation
         * @param equation_name name of the equation (optional)
         */
        void add(const pq_helper &pq, const string &equation_name = "");

        /**
         * clears everything in the builder
         */
        void clear() {
            equations_.clear();
            all_linkages_.clear();

            flop_map_.clear(); flop_map_init_.clear(); flop_map_pre_.clear();
            mem_map_.clear(); mem_map_init_.clear(); mem_map_pre_.clear();

            temp_counts_ = {{"scalars", 0}, {"tmps", 0}, {"reuse_tmps", 0}};
            tmp_candidates_.clear();
            is_reordered_ = false;
            is_reused_ = false;
            has_perms_merged_ = false;
        }

        /**
         * export tabuilder to python
         * @param m module
         */
        static void export_pq_graph(pybind11::module &m);

        /**
         * string representation of the equations
         */
        vector<string> to_strings();

        /**
         * keys of the equations
         * @return vector of equation keys
         */
        vector<string> get_equation_keys();

        /**
         * Reorder terms in each equation
         */
        void reorder();

        /**
         * Fully optimize equations by reordering, substituting, merging, and reusing intermediates.
         * @note this is a shortcut for calling reorder, substitute, merge_terms, and reuse on the python side
         */
        void optimize();

        /**
         * collect scaling of all equations
         * @param regenerate whether to regenerate the scaling for terms
         */
        void collect_scaling(bool recompute = true);

        /**
         * report summary of scaling for all equations
         */
        void analysis() const;

        /**
         * Substitute common linkages in each equation
         */
        void substitute();

        /**
         * make set of linkages to test for subexpression elimination
         * @return set of linkages to test
         */
        linkage_set make_test_set();

        /**
         * separate terms with permutations in each equation into separate terms
         */
        void expand_permutations();

        /**
         * collect all possible linkages from all equations (remove none)
         * @param recompute whether to recompute all linkages or just the ones in modified terms
         */
        void generate_linkages(bool recompute = true);

        /**
         * Print all terms in each equation
         */
        void print();

        /**
         * turn tabuilder into a string
         */
        string str();

        /**
         * Write DOT representation of equations to file stream (to visualize linkage in graphviz)
         * @param os output stream
         * @param color color of the vertices and edges
         * @return output stream
         */
        void write_dot(std::string &filepath);

        /**
         * adds a tmp to all_linkages_ and adds to equations
         * @param precon tmp to add
         * @note recomputes scaling after adding tmps
         */
        static void add_tmp(const LinkagePtr& precon, Equation &equation);

        /**
         * Sorts tmps by the maximum id of the rhs in the linkage and the tmp itself
         * @param equation equation to sort tmps for
         */
        static void sort_tmps(Equation &equation);

        /**
         * Find the common coefficient of a set of terms
         * @param terms terms to find common coefficient of
         * @return common coefficient
         */
        static double common_coefficient(vector<Term> &terms);

        /**
         * print the scaling of the current equations and difference from previous scalings
         * @param original_map The original scaling map
         * @param previous_map the previous scaling map
         * @param current_map the current scaling map
         */
        static void print_new_scaling(const scaling_map &original_map, const scaling_map &previous_map, const scaling_map &current_map) ;

    }; // PQGraph

} // pdaggerq

#endif //PDAGGERQ_PQ_GRAPH_H
