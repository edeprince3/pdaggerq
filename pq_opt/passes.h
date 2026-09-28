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

#include <set>

namespace pdaggerq {
class pq_helper;
}

namespace pdaggerq::opt {

/// ingest.cc: pq_helper strings -> Equation

struct IngestOptions {
    bool permute_eri = true;      // bring eri blocks to the canonical set (oovv, vovo, ...)
    bool symmetric_eri = false;   // eri has bra/ket symmetry (<pq|rs> = <rs|pq>)
    bool use_trial_index = false; // give r/l amplitudes a leading trial-vector index
    // the tensors that change between calls of the generated code (TensorRef::varies), by
    // name or by a prefix followed by a digit ("r" matches r1, r2, r0_1p). everything else is
    // taken to be fixed; e.g. response theory also varies "x" amplitudes and the perturbation "h"
    std::set<std::string> varying = {"r", "l"};
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

/**
 * the binary contraction tree with the fewest flops (exhaustive DP over subsets);
 * the flat tree when there are fewer than 3 or more than 16 tensors
 * @param calls when > 0, hoist: the code is expected to be called this many times with new
 *        values of the tensors that vary (TensorRef::varies), and a subtree of the others is
 *        computed once and stored if it can be (no intermediate in it larger than hoist_cap)
 *        and if that pays off: the flops minimized are those per call plus the stored
 *        subtrees' flops divided by calls. see hoisted_subtrees
 */
ExprPtr optimal_expr(const Term &term, const Indices &out, const Sizes &sizes, double calls = 0.0);

/// the largest tensor of a term that does not vary between calls: the most a hoisted
/// intermediate may hold
double hoist_cap(const Term &term, const Sizes &sizes);

/// does the subtree hold only tensors that do not vary between calls?
bool is_fixed(const ExprPtr &expr);

/// the subtrees of a tree from optimal_expr(..., calls > 0) that are computed once: the
/// fixed internal children of varying nodes with no intermediate larger than cap
std::vector<ExprPtr> hoisted_subtrees(const ExprPtr &root, double cap, const Sizes &sizes);

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

/**
 * compute once what does not change between calls (opt_level >= 3)
 *
 * when some tensor varies between calls (TensorRef::varies: r and l amplitudes), each term
 * gets the tree with the fewest flops per call; its fixed subtrees that can be stored are
 * defined once as reused_["0001_ov"] (one product of two tensors each, identical products
 * shared), and the terms with no varying tensor are summed into one reused_ per equation.
 * does nothing when no tensor varies
 *
 * @param eqs the equations; their terms are rewritten to read the reused_ tensors
 * @param sizes index extents for the cost model
 * @param calls the number of calls the code is expected to serve (see optimal_expr)
 * @param use_antisymmetry as for extract_intermediates
 * @return the reused_ definitions, in the order they must be computed
 */
std::vector<Equation> hoist_invariants(std::vector<Equation> &eqs, const Sizes &sizes, double calls,
                                       bool use_antisymmetry);

/**
 * merge terms that differ only in their fixed tensor (opt_level >= 4, after hoist_invariants)
 *
 * the terms of an equation that are (varying tensors) x (one fixed tensor), with the same
 * varying tensors contracted the same way (up to antisymmetry) and the same permutation
 * operator, c_k P[V F_k], become one term P[V G], where G = sum_k c_k F_k is a reused_
 * built once. one contraction per call replaces one per term
 *
 * @param eqs the equations; merged terms are replaced
 * @param first_id the id of the first reused_ made here (after hoist_invariants' ones)
 * @param use_antisymmetry as for extract_intermediates
 * @return the definitions of the sums, to be computed after hoist_invariants' definitions
 */
std::vector<Equation> merge_terms(std::vector<Equation> &eqs, size_t first_id, bool use_antisymmetry);

/**
 * merge terms that differ only in what they multiply one fixed tensor by (opt_level >= 4,
 * after merge_terms)
 *
 * the terms of an equation that have varying tensors and the same largest fixed tensor F,
 * contracted the same way (up to antisymmetry), with the same permutation operator,
 * c_k P[F R_k], become one term P[F S], where S = sum_k c_k R_k is a tmps_ built on every
 * call. a term takes part only if building its R_k first costs less than its own best order,
 * and a group only if the flops saved exceed the one contraction of F with S
 *
 * @param eqs the equations; merged terms are replaced
 * @param sizes index extents for the cost model
 * @param use_antisymmetry as for extract_intermediates
 * @return the definitions of the sums, to be computed before the equations that read them
 */
std::vector<Equation> merge_varying(std::vector<Equation> &eqs, const Sizes &sizes, bool use_antisymmetry);

/// build_program.cc: equations -> statements

/**
 * turn each term into one statement, and free each intermediate (tmps_) after its last use
 * @param eqs the equations
 * @param reorder use optimal_expr (true) or flat_expr (false)
 * @param sizes index extents for the cost model
 * @param free_reused the keys of reused_ intermediates to free after their last use too
 *        (those that are only needed to build other reused_ ones)
 */
Program build_program(const std::vector<Equation> &eqs, bool reorder, const Sizes &sizes,
                      const std::set<std::string> &free_reused = {});

} // namespace pdaggerq::opt

#endif
