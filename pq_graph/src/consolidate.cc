//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: consolidate.cc
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

#include "../include/pq_graph.h"
#include "iostream"
#include <omp.h>

using std::logic_error;

namespace pdaggerq {

    size_t PQGraph::merge_terms() {

        if (verbose) cout << "Merging similar terms:" << endl;

        // iterate over equations and merge terms
        size_t num_fuse = 0;
        omp_set_num_threads(num_threads_);
        vector<string> eq_keys = get_equation_keys();
        #pragma omp parallel for reduction(+:num_fuse) default(none) shared(equations_, eq_keys)
        for (const auto &key: eq_keys) {
            Equation &eq = equations_[key];
            if (eq.name() == "tmps") continue; // skip tmps equation
            if (eq.assignment_vertex()->rank() == 0) continue; // skip if lhs vertex is scalar
            num_fuse += eq.merge_terms(); // merge terms with same rhs up to a permutation
        }
        omp_set_num_threads(1);
        collect_scaling(); // collect new scalings

        if (verbose) cout << "Done (" << num_fuse << " terms merged)" << endl << endl;

        return num_fuse;
    }

    void PQGraph::merge_permutations() {

        /*
         * This function merges the permutation containers of the equations in the tabuilder.
         * This way, we can just add terms to the same type of permutation and permute at the very end.
         */
        if (is_reused_) throw logic_error("Cannot merge permutation containers of reused equations.");

        // merge permutation containers for each equation
        omp_set_num_threads(num_threads_);
        vector<string> eq_keys = get_equation_keys();
        #pragma omp parallel for default(none) shared(equations_, eq_keys)
        for (const auto &key: eq_keys) {
            Equation &eq = equations_[key];
            eq.merge_permutations();
        }
        omp_set_num_threads(1);

        has_perms_merged_ = true;

    }

    double PQGraph::common_coefficient(vector<Term> &terms) {

        // make a count_ of the reciprocal of the coefficients of the terms
        map<size_t, size_t> reciprocal_counts;
        for (const Term &term: terms) {
            auto reciprocal = static_cast<size_t>(round(1.0 / fabs(term.coefficient_)));
            reciprocal_counts[reciprocal]++;
        }

        // find the most common reciprocal
        size_t most_common_reciprocal = 1; // default to 1
        size_t most_common_reciprocal_count = 1;
        for (const auto &reciprocal_count: reciprocal_counts) {
            if (reciprocal_count.first <= 0) continue; // skip 0 values (generally doesn't happen)
            if (reciprocal_count.second > most_common_reciprocal_count) {
                most_common_reciprocal = reciprocal_count.first;
                most_common_reciprocal_count = reciprocal_count.second;
            }
        }
        double common_coefficient = 1.0 / static_cast<double>(most_common_reciprocal);
        return common_coefficient;
    }
}
