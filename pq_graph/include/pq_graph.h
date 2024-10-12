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
#include <fcntl.h>

#include "../../pdaggerq/pq_helper.h"
#include "equation.h"
#include "timer.h"

using std::ofstream;

namespace pdaggerq {

    class PQGraph; // forward declaration
    class PQGraph {
        map<string, Equation> equations_; // equations to be optimized

        // TODO: merge temp tracking variables into a class with methods for adding, removing, and updating temps
        map<string, linkage_set> saved_linkages_ = {
                                                      // all stored intermediate linkages with their types
                                                      {"vertex", linkage_set(256)},
                                                      {"link", linkage_set(256)},
                                                      {"scalar", linkage_set(256)},
                                                      {"temp", linkage_set(256)},
                                                      {"reused", linkage_set(256)}
                                                   };
        // counts of tmps and scalars
        map<string, long> temp_counts_ = {{"vertex", 0}, {"link", 0}, {"scalar", 0}, {"temp", 0}, {"reused", 0}};


        linkage_set all_links_; // all possible linkages in the equations

        bool is_assembled_ = false; // whether equations have been assembled for printing
        bool is_reordered_ = false; // whether the equations have been reordered
        bool is_optimized_ = false; // whether the equations have been optimized

        Timer total_timer; // timer for the total time of the builder
        Timer build_timer; // timer for construction of the equations
        Timer reorder_timer; // timer for the reorder function
        Timer substitute_timer; // timer for the substitute function

        Timer update_timer; // timer for updating equations

        /// scaling of the equations
        scaling_map flop_map_; // map of flop scaling with linkage occurrence in all equations
        scaling_map mem_map_; // map of memory scaling with linkage occurrence in all equations

        // TODO: replace with pointers to copies of the PQGraph
        shared_ptr<PQGraph> original_; // original pq_graph before optimization
        shared_ptr<PQGraph> reordered_; // pq_graph after reordering
        shared_ptr<PQGraph> optimized_; // pq_graph after optimization (should be this pq_graph)

        scaling_map flop_map_init_; // map of flop scaling before reordering
        scaling_map mem_map_init_; // map of memory scaling before reordering

        scaling_map flop_map_pre_; // map of flop scaling before reordering or before subexpression elimination
        scaling_map mem_map_pre_; // map of memory scaling before reordering or before subexpression elimination

        /// options for the builder

        /**
         * print level for the builder
         *     0: no printing of optimization steps (default)
         *     1: print optimization steps without fusion or merging
         *     2: print optimization steps with fusion and merging
         */
        size_t print_level_ = 0;

        /**
         * optimization level for the builder
         *    0: no optimization
         *    1: reordering only
         *    2: reordering and substitution
         *    3: reordering, substitution, and separation of reusable intermediates (for sigma vectors)
         *    4: reordering, substitution, and separation; unused intermediates are removed (pruning)
         *    5: reordering, substitution, separation, and merging of equivalent terms with pruning
         *    6: reordering, substitution, separation, merging, and fusion of intermediates with pruning (default)
         */
        size_t opt_level_ = 6;

        /// number of threads to use
        int nthreads_ = 1;

        /**
         * whether to use batched substitution
         *   true (default): candidate substitutions are applied in batches rather than one at a time.
         *                   Generally faster, but may not yield optimal results compared to single substitution.
         */
        bool batched_ = false;
        size_t batch_size_ = 10; // number of substitutions to apply in each batch

        /// maximum number of temporary rhs (-1 for no limit by overflow)
        size_t max_temps_ = static_cast<size_t>(-1l);

        /// whether to use density fitted integrals
        bool use_density_fitting_ = false;

        /// whether the equations have any sigma vectors
        bool has_sigma_vecs_ = false;

    public:

        // default constructor
        PQGraph() = default;
        ~PQGraph() = default;

        explicit PQGraph(const pybind11::dict& options){
            substitute_timer.precision_ = 2;
            reorder_timer.precision_    = 2;
            build_timer.precision_      = 2;
            update_timer.precision_     = 2;

            set_options(options);
        }

        /**
         * Get equations
         */
        map<string, Equation> &equations() { return equations_; }

        /**
         * Get saved linkages
         */
        map<string, linkage_set> &saved_linkages() { return saved_linkages_; }

        /**
         * get temp counts
         */
        map<string, long> &temp_counts() { return temp_counts_; }

        /**
         * Get all linkages
         */
        linkage_set &all_links() { return all_links_; }

        /**
         * Get pointers to all terms from all equations
         */
        vector<Term*> every_term();

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
        void add(const pq_helper &pq, const std::string &equation_name = "", vector<std::string> label_order = vector<string>());

        /**
         * assemble the equations into a pq_graph
         */
        void assemble();

        /**
         * clears everything in the builder
         */
        void clear() {
            *this = PQGraph();   // reset the builder
        }

        /**
         * export tabuilder to python
         * @param m module
         */
        static void export_pq_graph(pybind11::module &m);

        /**
         * string representation of the equations
         * @param print_type type of print (c++ or python)
         */
        vector<string> to_strings(const string &print_type) const;

        /**
         * keys of the equations
         * @return vector of equation keys
         */
        vector<string> get_equation_keys();

        /**
         * Reorder terms in each equation
         */
        void reorder(bool regenerate = false);

        /**
         * merge terms in each equation
         * @return number of terms merged
         */
        size_t merge_terms();

        /**
         * fuse intermediate terms with the same rhs
         */
        size_t merge_intermediates();

        /**
         * Fully optimize equations by reordering, substituting, merging, and reusing intermediates.
         * @note this is a shortcut for calling reorder, substitute, merge_terms, and reuse on the python side
         */
        void optimize();

        /**
         * collect scaling of all equations
         * @param regenerate whether to regenerate the scaling for terms
         */
        void collect_scaling(bool recompute = true, bool include_reuse = false);

        /**
         * report summary of scaling for all equations
         */
        void analysis() const;

        /**
         * Substitute common linkages in each equation
         * @param format_sigma whether to only substitute intermediates without sigma vectors
         * @param only_scalars whether to only substitute scalar intermediates
         */
        void substitute(bool format_sigma, bool only_scalars);


        /**
         * collect all possible linkages from all equations (remove none)
         * @param recompute whether to recompute all linkages or just the ones in modified terms
         */
        void make_all_links(bool recompute);

        /**
         * Forget the linkage history within all linkages to free memory from lazy evaluation
         */
        void forget();

        /**
         * Sync all pointers so that the same vertices have the same memory address
         */
        void sync_pointers();

        /**
         * Print all terms in each equation
         * @param print_type type of print (c++ or python)
         */
        void print(const string & print_type) const;

        /**
         * turn pq_graph into a string
         * @param print_type type of print (c++ or python)
         * @return string representation of the pq_graph
         */
        string str(const string &print_type) const;

        /**
         * Write DOT representation of equations to file stream (to visualize linkage in graphviz)
         * @param os output stream
         * @param color color of the vertices and edges
         * @return output stream
         */
        void write_dot(std::string &filepath);

        /**
         * adds a tmp to saved_linkages_ and adds to equations
         * @param precon tmp to add
         * @note recomputes scaling after adding tmps
         */
        static Term &add_tmp(const ConstLinkagePtr& precon, Equation &equation, double coeff = 1.0);

        /**
         * Find the common coefficient of a set of terms
         * @param terms terms to find common coefficient of
         * @return common coefficient
         */
        static double common_coefficient(set<Term*> &terms);

        /**
         * print the scaling of the current equations and difference from previous scalings
         * @param original_map The original scaling map
         * @param previous_map the previous scaling map
         * @param current_map the current scaling map
         */
        static void print_new_scaling(const scaling_map &original_map, const scaling_map &previous_map, const scaling_map &current_map) ;

        /**
         * get all terms that contain a given intermediate
         * @param intermediate intermediate to find
         * @return 1) vector of terms that contain the intermediate, 2) declaration terms for the intermediate
         */
        pair<set<Term *>, set<Term*>> get_matching_terms(const ConstLinkagePtr &intermediate);

        /**
         * remove redundant temps that only appear once
         * @param keep_single_use whether to keep temps that only appear once
         */
        size_t prune(bool keep_single_use = true);

        /**
         * reindex the intermediates in the equations
         */
        void reindex();

        /**
         * deep copy of the pq_graph
         * @return deep copy of the pq_graph
         */
        PQGraph clone() const;

        /**
         * generate all scalar contractions
         */
        void make_scalars();

        /**
         * remove all scalar contractions
         */
        void remove_scalars();

    }; // PQGraph

    /**
     * struct to disable cout stream for printing within a scope
     */
    struct print_guard {

        bool prior_locked;
        int bak_out, null_out;

        // Constructor to lock cout
        print_guard() {
            prior_locked = std::cout.fail();
            if (!prior_locked) {
                fflush(stdout);
                bak_out = dup(fileno(stdout));
            }
        }

        // Restore cout
        void restore() const {
            std::cout.clear();
            fflush(stdout);
            dup2(bak_out, fileno(stdout));
            close(bak_out);
        }

        // Redirect cout to null stream
        void lock() {
            if (!prior_locked) {
                std::cout.setstate(std::ios::failbit);
                null_out = open("/dev/null",  O_WRONLY);
                dup2(null_out, fileno(stdout));
                close(null_out);
            }
        }

        // Destructor to restore the original buffer
        ~print_guard() {
            if (!prior_locked)
                restore();
        }
    };

} // pdaggerq

#endif //PDAGGERQ_PQ_GRAPH_H
