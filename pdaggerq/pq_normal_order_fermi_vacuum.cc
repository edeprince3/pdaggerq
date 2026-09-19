//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_swap_operators.h
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

#include "pq_string.h"
#include "pq_tensor.h"
#include "pq_utils.h"

#include <cstdint>
#include <memory>
#include <vector>
#include <utility>

namespace pdaggerq {

namespace {

// Normal ordering by adjacent swaps. Used for keep_operators == true and for
// strings longer than wick_max_ops; everything else goes through the direct
// Wick contraction below.
void normal_order_by_swaps(const std::shared_ptr<pq_string> &in,
                           std::vector<std::shared_ptr<pq_string>> &ordered,
                           bool keep_operators) {

    // work stack: (string still being sorted, position to resume scanning from)
    std::vector<std::pair<std::shared_ptr<pq_string>, size_t>> work;
    work.reserve(64); // cheap guard against repeated small reallocations; grows as needed
    work.emplace_back(in, 0);

    while (!work.empty()) {

        auto [cur, scan_from] = std::move(work.back());
        work.pop_back();

        if (cur->skip) continue;

        const std::vector<bool> &isdf = cur->is_dagger_fermi;
        const std::vector<bool> &isd  = cur->is_dagger;
        const std::vector<std::string> &sym = cur->symbol;
        size_t n = sym.size();

        // Cheap boundary/doom check, done FIRST, on every popped item --
        // not just once the string happens to be fully sorted.
        //
        // is_dagger_fermi[0] and is_dagger_fermi[n-1] are permanently fixed
        // the moment a string is created: a swap/contraction only ever fires
        // at a position i with isdf[i]==false and isdf[i+1]==true, so
        // position 0 can never be touched if isdf[0]==true (it can't be the
        // left side of a trigger, and there's no i=-1 for it to be the right
        // side of one), and symmetrically for the last position if
        // isdf[last]==false. So if this condition holds, EVERY descendant of
        // this branch is guaranteed trivially zero -- checking it here, once,
        // prunes the entire subtree in O(1) instead of after fully expanding
        // it. This mirrors is_normal_order()'s boundary check, which the
        // original swap-based code effectively re-runs on every branch, every
        // pass (since it's called at the top of swap_operators_fermi_vacuum).
        if (n > 0) {
            bool is_dagger_right = isdf[n - 1];
            bool is_dagger_left  = isdf[0];
            if (!is_dagger_right || is_dagger_left) {
                if (!keep_operators) {
                    cur->skip = true;
                }
                if (!cur->skip) {
                    ordered.push_back(cur);
                }
                continue;
            }
        }

        // advance from scan_from (not 0) to the first remaining inversion
        size_t i = scan_from;
        while (i + 1 < n && !(!isdf[i] && isdf[i + 1])) i++;

        if (i + 1 >= n) {
            // fully sorted (or fully contracted, n==0) and boundary already
            // validated above -- this branch survives.
            ordered.push_back(cur);
            continue;
        }

        bool daggers_differ = (isd[i] != isd[i + 1]);

        // NEO-CC: operators of different species (e.g. electron vs nuclear)
        // live in disjoint orbital spaces -- they never contract, and in the
        // commuting convention for distinct particle types they swap without
        // a sign change. Only same-species creator/annihilator pairs
        // contract. Mirrors the same_species/can_contract logic added to
        // swap_operators_fermi_vacuum() in pq_swap_operators.cc.
        bool same_species = (is_nuclear(sym[i]) == is_nuclear(sym[i + 1]));
        bool can_contract = daggers_differ && same_species;

        size_t resume_from = (i > 0) ? i - 1 : 0;

        if (can_contract) {
            // contraction branch: {A_i, A_{i+1}} = delta(sym[i], sym[i+1])
            auto contracted = std::make_shared<pq_string>(cur.get(), /*copy_daggers_and_symbols=*/false);
            // deltas/ints/amps/sign/factor/skip/permutations already copied
            // by the constructor (verified against pq_string::copy()).

            delta_functions d;
            d.labels.push_back(sym[i]);
            d.labels.push_back(sym[i + 1]);
            d.sort();
            contracted->deltas.push_back(d);

            contracted->symbol.reserve(n - 2);
            contracted->is_dagger.reserve(n - 2);
            contracted->is_dagger_fermi.reserve(n - 2);
            for (size_t k = 0; k < n; k++) {
                if (k == i || k == i + 1) continue;
                contracted->symbol.push_back(sym[k]);
                contracted->is_dagger.push_back(isd[k]);
                contracted->is_dagger_fermi.push_back(isdf[k]);
            }
            contracted->is_boson_dagger = cur->is_boson_dagger; // not copied by copy(..., false)

            work.emplace_back(std::move(contracted), resume_from);
        }

        // swap branch: always happens. Same-species pairs anticommute
        // (sign flips: A_i A_{i+1} -> -A_{i+1} A_i); distinct-species pairs
        // commute (no sign change: A_i A_{i+1} -> A_{i+1} A_i).
        auto swapped = std::make_shared<pq_string>(cur.get(), /*copy_daggers_and_symbols=*/false);
        swapped->sign = same_species ? -cur->sign : cur->sign;

        swapped->symbol.reserve(n);
        swapped->is_dagger.reserve(n);
        swapped->is_dagger_fermi.reserve(n);
        for (size_t k = 0; k < n; k++) {
            size_t src = k;
            if (k == i)         src = i + 1;
            else if (k == i + 1) src = i;
            swapped->symbol.push_back(sym[src]);
            swapped->is_dagger.push_back(isd[src]);
            swapped->is_dagger_fermi.push_back(isdf[src]);
        }
        swapped->is_boson_dagger = cur->is_boson_dagger;

        work.emplace_back(std::move(swapped), resume_from);
    }
}

// ---------------------------------------------------------------------------
// Direct Wick contraction (keep_operators == false)
//
// Relative to the Fermi vacuum, a contraction is nonzero only for a
// quasi-annihilator (QA) standing to the left of a quasi-creator (QC) of the
// same sector: same species (electron / nuclear) and same space (occupied /
// virtual; an operator is occupied iff is_dagger != is_dagger_fermi). This is
// exactly the daggers_differ && same_species condition of the swap algorithm.
//
// Full contractions are enumerated by always resolving the leftmost
// remaining QC, branching over its QA partners, and recursing. Every full
// contraction pairs that QC with exactly one QA, so the branches partition
// the set of full contractions: nothing is missed or counted twice.
//
// The sign of contracting positions p < q is (-1)^m, with m the number of
// still-present same-species operators strictly between them (distinct
// species commute).
//
// Pruning: within each sector, scan left to right counting +1 per QA and -1
// per QC. A full contraction exists iff that running balance never goes
// negative and ends at zero (balanced parentheses, QA = '(' and QC = ')').
// A QC in the leftmost slot or a QA in the rightmost slot are special cases.
// This is checked once, up front. Contracting (p, q) lowers the balance by
// one on [p, q) and leaves it unchanged elsewhere; with q the leftmost QC the
// balance on [p, q) is just the number of sector QAs seen so far, which is at
// least one, so every same-sector QA left of q is an admissible partner and
// the search never visits a dead end.
//
// Order: QCs resolved left to right, partners taken farthest-left first, and
// deltas recorded in that order. This is the order in which the adjacent-swap
// algorithm produces the same terms, and downstream simplification (dummy
// relabeling, permutation detection) depends on it, so keeping it makes the
// final expressions identical to the swap algorithm's, not just equivalent.
// ---------------------------------------------------------------------------

constexpr int wick_max_ops = 128;

struct mask128 {
    uint64_t lo = 0;
    uint64_t hi = 0;
};

inline mask128 operator&(mask128 a, mask128 b) { return {a.lo & b.lo, a.hi & b.hi}; }

inline bool is_empty(mask128 m) { return (m.lo | m.hi) == 0; }

inline void set_bit(mask128 &m, int i) {
    if (i < 64) m.lo |= uint64_t(1) << i;
    else        m.hi |= uint64_t(1) << (i - 64);
}

inline mask128 without(mask128 m, int i) {
    if (i < 64) m.lo &= ~(uint64_t(1) << i);
    else        m.hi &= ~(uint64_t(1) << (i - 64));
    return m;
}

// bits [0, k), 0 <= k <= 128
inline mask128 below(int k) {
    mask128 m;
    m.lo = k >= 64 ? ~uint64_t(0) : (uint64_t(1) << k) - 1;
    m.hi = k <= 64 ? 0 : (k >= 128 ? ~uint64_t(0) : (uint64_t(1) << (k - 64)) - 1);
    return m;
}

// bits [lo, hi), lo <= hi
inline mask128 between(int lo, int hi) {
    mask128 a = below(hi), b = below(lo);
    return {a.lo & ~b.lo, a.hi & ~b.hi};
}

#if defined(__GNUC__) || defined(__clang__)
inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }
inline int lowest64(uint64_t x)   { return __builtin_ctzll(x); }
#else
inline int popcount64(uint64_t x) { int c = 0; while (x) { x &= x - 1; c++; } return c; }
inline int lowest64(uint64_t x)   { int i = 0; while (!((x >> i) & 1)) i++; return i; }
#endif

inline int popcount(mask128 m) { return popcount64(m.lo) + popcount64(m.hi); }

// m must be nonempty
inline int lowest(mask128 m) { return m.lo ? lowest64(m.lo) : 64 + lowest64(m.hi); }

struct wick_context {
    pq_string *in;
    const std::vector<std::string> *symbol;
    mask128 qc;              // quasi-creators
    mask128 qa;              // quasi-annihilators
    mask128 species[2];      // electron, nuclear
    mask128 sector[4];       // (nuclear ? 2 : 0) + (occupied ? 1 : 0)
    int sector_of[wick_max_ops];
    int species_of[wick_max_ops];
    std::vector<std::pair<int, int>> pairs; // (QA position, QC position), current branch
    std::vector<std::shared_ptr<pq_string>> *ordered;
};

void emit_full_contraction(wick_context &ctx, int sign) {
    auto out = std::make_shared<pq_string>(ctx.in, /*copy_daggers_and_symbols=*/false);
    out->deltas.reserve(out->deltas.size() + ctx.pairs.size());
    for (const auto &pq : ctx.pairs) {
        delta_functions d;
        d.labels.push_back((*ctx.symbol)[pq.first]);
        d.labels.push_back((*ctx.symbol)[pq.second]);
        d.sort();
        out->deltas.push_back(d);
    }
    out->sign = sign;
    out->is_boson_dagger = ctx.in->is_boson_dagger; // not copied by copy(..., false)
    ctx.ordered->push_back(std::move(out));
}

//
//
// find fully-contracted terms via wick's theorem. 
// 
// @param wick_context ctx: string details: which positions are QC/QA, their
//     species and sector, pairs contracted so far
// @param mask128 remaining: positions not yet contracted
// @param int sign: sign accumulated so far
//
// as an example, let's consider
// 
// a(a) a(b) a*(c) a*(d)
// 
// so, on entry,
// 
// ctx.qc = {2,3}
// remaining = {0, 1, 2, 3}
// sign = 1
// 
//
void wick_contract(wick_context &ctx, mask128 remaining, int sign) {

    // qcs is set of quasi-creators still present, e.g., {0,1,2,3} ∩ {2,3} = {2,3}
    mask128 qcs = remaining & ctx.qc;

    // if no QCs remain, everything is contracted
    if (is_empty(qcs)) {
        // the balance invariant guarantees no QAs are left either
        emit_full_contraction(ctx, sign);
        return;
    }

    // pick the left-most QC
    int q = lowest(qcs);

    // check which of the remaining positions have the same species as q
    mask128 same_species = remaining & ctx.species[ctx.species_of[q]];

    // loop over all partners, m, with which q could contract. m is:
    // 
    // 1. present in remaining
    // 2. in the same sector as q (e.g., both occ), 
    // 3. a quasi-annihilator
    // 4. to the left of q
    // 
    // the loop runs until m is empty
    // 
    // so, in the first pass, m = {0, 1}
    // 
    for (mask128 m = remaining & ctx.sector[ctx.sector_of[q]] & ctx.qa & below(q); !is_empty(m); ) {

        // p is the left-most remaining partner of q. remove it from m for the next iteration
        int p = lowest(m);
        m = without(m, p);

        // how many operators of the same species does q need to hop over to get to p
        int hops = popcount(same_species & between(p + 1, q));

        // record the contracted pair
        ctx.pairs.emplace_back(p, q);

        // contract next pair in string without p and q and with sign determined by hops
        wick_contract(ctx, without(without(remaining, p), q), (hops & 1) ? -sign : sign);

        // unrecord the contracted pair before advancing m
        ctx.pairs.pop_back();
    }
}

void normal_order_by_wick(const std::shared_ptr<pq_string> &in,
                          std::vector<std::shared_ptr<pq_string>> &ordered) {

    const std::vector<std::string> &sym = in->symbol;
    const std::vector<bool> &isd  = in->is_dagger;
    const std::vector<bool> &isdf = in->is_dagger_fermi;
    int n = (int) sym.size();

    if (n == 0) {
        ordered.push_back(in);
        return;
    }

    wick_context ctx;
    ctx.in = in.get();
    ctx.symbol = &sym;
    ctx.ordered = &ordered;

    // counters on electron(vir), electron(occ), nuclear(vir), nuclear(occ)
    int balance[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; i++) {
        int nuclear = is_nuclear(sym[i]) ? 1 : 0;
        int occupied = (isd[i] != isdf[i]) ? 1 : 0;
        int s = 2 * nuclear + occupied;

        ctx.sector_of[i] = s;
        ctx.species_of[i] = nuclear;
        set_bit(ctx.species[nuclear], i);
        set_bit(ctx.sector[s], i);

        if (isdf[i]) {
            set_bit(ctx.qc, i);
            if (--balance[s] < 0) return; // this QC has no QA to its left
        } else {
            set_bit(ctx.qa, i);
            ++balance[s];
        }
    }
    for (int b : balance) {
        if (b != 0) return; // unmatched operators survive every contraction
    }

    ctx.pairs.reserve(n / 2);
    wick_contract(ctx, below(n), in->sign);
}

} // namespace

// Bring 'in' to normal order with respect to the Fermi vacuum. Fully
// contracted terms (keep_operators == false, the only mode pq_helper uses)
// are generated directly by Wick's theorem; otherwise, or for strings with
// more than wick_max_ops fermion operators, fall back to adjacent swaps.
void fermion_normal_order_fermi_vacuum(const std::shared_ptr<pq_string> &in,
                                       std::vector<std::shared_ptr<pq_string>> &ordered,
                                       bool keep_operators) {

    if (in->skip) return;

    if (keep_operators || in->symbol.size() > (size_t) wick_max_ops) {
        normal_order_by_swaps(in, ordered, keep_operators);
        return;
    }

    normal_order_by_wick(in, ordered);
}

// Iterative replacement for the boson-handling blocks embedded in
// swap_operators_fermi_vacuum() (the s1a/s1b/s2a/s2b construction in
// pq_swap_operators.cc). Resolves the boson part of one already
// fermion-normal-ordered pq_string to its complete list of boson-normal-
// ordered output strings.
//
// Two things differ from the fermion function, both taken directly from the
// original boson-handling blocks:
//   1. branches are built via the FULL default copy constructor (*cur), not
//      the (cur.get(), false) constructor -- fermion state (symbol/is_dagger/
//      is_dagger_fermi/deltas/etc.) must be preserved as-is (it can be
//      non-empty leftover operators when keep_operators == true); only
//      is_boson_dagger is cleared and rebuilt.
//   2. bosons carry no label/index at all (is_boson_dagger is just a vector
//      of creator/annihilator flags), so a "contraction" is a plain commutator
//      constant (1): both operators are simply dropped, no delta_functions
//      entry, and -- since bosons commute -- no sign change on the swap
//      branch either (contrast the fermion function's conditional sign flip).
void boson_normal_order(const std::shared_ptr<pq_string> &in,
                        std::vector<std::shared_ptr<pq_string>> &ordered,
                        bool keep_operators) {

    std::vector<std::pair<std::shared_ptr<pq_string>, size_t>> work;
    work.reserve(16);
    work.emplace_back(in, 0);

    while (!work.empty()) {

        auto [cur, scan_from] = std::move(work.back());
        work.pop_back();

        if (cur->skip) continue;

        const std::vector<bool> &isbd = cur->is_boson_dagger;
        size_t n = isbd.size();

        // Boundary/doom check, same permanence argument and same structure
        // as fermion_normal_order_fermi_vacuum's, mirroring
        // is_boson_normal_order() -- including its explicit n==1 special
        // case: with a single leftover boson operator, is_dagger_right and
        // is_dagger_left are the SAME element, so "!x || x" is a tautology
        // -- a lone boson operator is always doomed (unless keep_operators).
        if (n > 0) {
            bool is_dagger_right = isbd[n - 1];
            bool is_dagger_left  = isbd[0];
            if (!is_dagger_right || is_dagger_left) {
                if (!keep_operators) {
                    cur->skip = true;
                }
                if (!cur->skip) {
                    ordered.push_back(cur);
                }
                continue;
            }
        }

        // advance from scan_from (not 0) to the first remaining inversion
        size_t i = scan_from;
        while (i + 1 < n && !(!isbd[i] && isbd[i + 1])) i++;

        if (i + 1 >= n) {
            // fully sorted (or empty) and boundary already validated above
            ordered.push_back(cur);
            continue;
        }

        size_t resume_from = (i > 0) ? i - 1 : 0;

        // contraction branch: drop both operators (commutator constant 1,
        // no delta needed), no sign change.
        {
            auto contracted = std::make_shared<pq_string>(*cur); // full copy: preserve fermion state
            contracted->is_boson_dagger.clear();
            contracted->is_boson_dagger.reserve(n - 2);
            for (size_t k = 0; k < n; k++) {
                if (k == i || k == i + 1) continue;
                contracted->is_boson_dagger.push_back(isbd[k]);
            }
            work.emplace_back(std::move(contracted), resume_from);
        }

        // swap branch: bosons commute -- no sign change.
        {
            auto swapped = std::make_shared<pq_string>(*cur); // full copy: preserve fermion state
            swapped->is_boson_dagger.clear();
            swapped->is_boson_dagger.reserve(n);
            for (size_t k = 0; k < n; k++) {
                size_t src = k;
                if (k == i)         src = i + 1;
                else if (k == i + 1) src = i;
                swapped->is_boson_dagger.push_back(isbd[src]);
            }
            work.emplace_back(std::move(swapped), resume_from);
        }
    }
}

} // namespace pdaggerq
