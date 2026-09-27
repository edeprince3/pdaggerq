//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: passes.h
// Copyright (C) 2020 A. Eugene DePrince III
//
// Author: A. Eugene DePrince III <adeprince@fsu.edu>
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

// the passes that turn pdaggerq output into a Program

#ifndef PQ_OPT_PASSES_H
#define PQ_OPT_PASSES_H

#include "ir.h"

namespace pdaggerq {
class pq_helper;
}

namespace pdaggerq::opt {

/// ingest.cc: pq_helper strings -> Equation

struct IngestOptions {
    bool permute_eri = true;      // bring eri blocks to the canonical set (oovv, vovo, ...)
    bool symmetric_eri = false;   // eri has bra/ket symmetry (<pq|rs> = <rs|pq>)
    bool use_trial_index = false; // give r/l amplitudes a leading trial-vector index
};

/**
 * build an equation from the (possibly blocked) strings held by a pq_helper
 * @param pq the pq_helper
 * @param equation_name lhs name, e.g. "rt2", or with its index order, e.g. "rt2(a,b,i,j)"
 * @param label_order the order of the lhs indices: every external index exactly once, except
 *        the trial index, which is always first, and boson indices, which are optional
 *        (all or none) and otherwise last.
 *        may instead be given in the name
 * @param options ingest options
 */
Equation ingest(const pq_helper &pq, const std::string &equation_name,
                const std::vector<std::string> &label_order, const IngestOptions &options);

/// order.cc: contraction trees and their cost

/// number of elements spanned by a set of indices
double extent(const Indices &idx, const Sizes &sizes);

/// flops of the contraction at this node plus all below it
double cost(const ExprPtr &expr, const Sizes &sizes);

/// the costliest single contraction in the tree, as counts per index space (e.g. o2v4)
std::map<char, int> scaling(const ExprPtr &expr);

/// one n-ary node over all tensors, in the order pdaggerq gave them
ExprPtr flat_expr(const Term &term, const Indices &out);

/// the binary contraction tree with the fewest flops (exhaustive DP over subsets);
/// the flat tree when there are fewer than 3 or more than 16 tensors
ExprPtr optimal_expr(const Term &term, const Indices &out, const Sizes &sizes);

/// intermediates.cc: shared intermediates (opt_level >= 2)

/**
 * replace products that occur in more than one term by shared intermediates
 *
 * a two-tensor node of a term's optimal contraction tree that occurs in several terms
 * (of any equations) is computed once, as tmps_["0001_ov"], and read everywhere.
 * greedy: the product that saves the most flops first, until none saves any
 *
 * @param eqs the equations; their terms are rewritten to use the intermediates
 * @param sizes index extents for the cost model
 * @param max_temps the most intermediates to create (-1: no limit)
 * @param use_antisymmetry also match products that differ by a permutation of an antisymmetric
 *        tensor's indices (only tensors pdaggerq marks antisymmetric); false: identical only
 * @return the intermediates' definitions, one term each, in the order they must be computed
 */
std::vector<Equation> extract_intermediates(std::vector<Equation> &eqs, const Sizes &sizes, long max_temps,
                                           bool use_antisymmetry);

/// build_program.cc: equations -> statements

/**
 * turn each term into one statement, and free each intermediate (tmps_) after its last use
 * @param eqs the equations
 * @param reorder use optimal_expr (true) or flat_expr (false)
 * @param sizes index extents for the cost model
 */
Program build_program(const std::vector<Equation> &eqs, bool reorder, const Sizes &sizes);

} // namespace pdaggerq::opt

#endif
