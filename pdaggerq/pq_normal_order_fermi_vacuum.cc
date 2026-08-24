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

#include <memory>
#include <vector>
#include <utility>

namespace pdaggerq {

// Iterative replacement for the do-while(swap_operators_fermi_vacuum(...))
// block in add_new_string_fermi_vacuum(). Resolves 'in' to its complete list
// of normal-ordered / fully-contracted output strings.
void fermion_normal_order_fermi_vacuum(const std::shared_ptr<pq_string> &in,
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

// Iterative replacement for the boson-handling blocks embedded in
// swap_operators_fermi_vacuum() (the s1a/s1b/s2a/s2b construction in
// pq_swap_operators.cc). Resolves the boson part of one already
// fermion-normal-ordered pq_string to its complete list of boson-normal-
// ordered output strings.
//
// IMPORTANT: this must run AFTER fermion_normal_order_fermi_vacuum(), on its
// output -- not on the original unordered input. Correct call site:
//
//   std::vector< std::shared_ptr<pq_string> > fermion_ordered, tmp;
//   fermion_normal_order_fermi_vacuum(mystring, fermion_ordered, keep_operators);
//   for (const auto & pq_str : fermion_ordered) {
//       boson_normal_order(pq_str, tmp, keep_operators);
//   }
//   new_strings[k] = tmp;
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
