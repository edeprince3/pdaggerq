//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: ingest.cc
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

#include "passes.h"
#include "pdaggerq_access.h"

#include "../pdaggerq/pq_helper.h"
#include "../pdaggerq/pq_utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>

namespace pdaggerq::opt {

namespace {

[[noreturn]] void unsupported(const std::string &what) {
    throw std::runtime_error("pq_opt: " + what + " is not supported yet");
}

// 'o'/'v' for an electron label, 'O'/'V' for a nuclear (multicomponent) one such as "ni"
char space_of(const std::string &label) {
    bool nuclear = is_nuclear(label);
    if (is_occ(label)) return nuclear ? 'O' : 'o';
    if (is_vir(label)) return nuclear ? 'V' : 'v';
    throw std::runtime_error("pq_opt: index '" + label + "' is neither occupied nor virtual");
}

// spin ('a'/'b') or range ('1' active / '0' external) block of the i-th label
char block_of(const tensor &t, size_t i) {
    if (pq_string::is_spin_blocked) return spin_label(t, i) == "a" ? 'a' : 'b';
    if (pq_string::is_range_blocked) return range_label(t, i) == "act" ? '1' : '0';
    return '\0';
}

// a tensor's fermion indices, then its boson (cavity-mode) indices
Indices indices_of(const tensor &t) {
    Indices idx;
    const std::vector<std::string> &labels = fermion_labels(t);
    for (size_t i = 0; i < labels.size(); i++)
        idx.push_back({labels[i], space_of(labels[i]), block_of(t, i)});
    for (const std::string &label : boson_labels(t))
        idx.push_back({label, 'b', '\0'});
    return idx;
}

std::string ovstring(const Indices &idx) {
    std::string s;
    for (const Index &i : idx) s += i.space;
    return s;
}

std::string blkstring(const Indices &idx) {
    std::string s;
    for (const Index &i : idx)
        if (i.block) s += i.block;
    return s;
}

// map key of an integral or identity: "oovv", or "abab_oovv" when blocked
std::string operator_key(const Indices &idx) {
    std::string blk = blkstring(idx);
    return blk.empty() ? ovstring(idx) : blk + "_" + ovstring(idx);
}

// permute <p,q||r,s> into a block the caller provides (oovv, vovo, ... and, when
// blocked, aaaa/bbbb/abab if possible). returns the sign of the permutation.
int permute_eri(Indices &idx, bool symmetric) {
    static const std::set<std::string> valid_ov = {"oooo", "vvvv", "oovv", "vvoo", "vovo",
                                                   "vooo", "oovo", "vovv", "vvvo"};
    static const std::set<std::string> valid_blk = {"", "aaaa", "bbbb", "abab"};

    struct Perm { int p[4]; int sign; };
    std::vector<Perm> perms = {{{0, 1, 2, 3}, 1}, {{1, 0, 2, 3}, -1}, {{0, 1, 3, 2}, -1}, {{1, 0, 3, 2}, 1}};
    if (symmetric) {
        std::vector<Perm> conj;
        for (const Perm &q : perms)
            conj.push_back({{q.p[2], q.p[3], q.p[0], q.p[1]}, q.sign});
        perms.insert(perms.end(), conj.begin(), conj.end());
    }

    auto rearranged = [&](const Perm &q) -> Indices {
        return {idx[q.p[0]], idx[q.p[1]], idx[q.p[2]], idx[q.p[3]]};
    };
    auto valid_ov_string = [&](const Perm &q) { return valid_ov.count(ovstring(rearranged(q))) > 0; };
    auto valid_block = [&](const Perm &q) { return valid_blk.count(blkstring(rearranged(q))) > 0; };

    // the perms are tried in order, identity first, so an acceptable eri is left as it is.
    // first pass: a valid ov string and a standard spin block
    for (const Perm &q : perms) {
        if (valid_ov_string(q) && valid_block(q)) {
            idx = rearranged(q);
            return q.sign;
        }
    }

    // second pass: a valid ov string with a nonstandard block (e.g. abba_oovo)
    for (const Perm &q : perms) {
        if (valid_ov_string(q)) {
            idx = rearranged(q);
            return q.sign;
        }
    }

    // no valid ov string: leave the eri unchanged
    return 1;
}

// the antisymmetric position groups of a tensor: each group of positions (e.g. the bra or
// ket of <p,q||r,s>) split into the parts whose indices share a space and block, since only
// those can be permuted without changing the tensor's block; parts of one are dropped
std::vector<std::vector<size_t>> antisymmetric_groups(const Indices &idx,
                                                      const std::vector<std::vector<size_t>> &groups) {
    std::vector<std::vector<size_t>> parts_found;
    for (const std::vector<size_t> &group : groups) {
        std::map<std::pair<char, char>, std::vector<size_t>> parts;
        for (size_t p : group) parts[{idx[p].space, idx[p].block}].push_back(p);
        for (auto &[kind, part] : parts)
            if (part.size() > 1) parts_found.push_back(part);
    }
    return parts_found;
}

// name of an amplitude, as pdaggerq's amplitudes::to_string writes it: t2, l1, r2; with
// nuclear indices t2_n (all nuclear) or t2_ep (mixed; from rank 3 the electron/nuclear
// split is spelled out, t3_ep21); with photons t1_1p, t0_2p (no fermion indices), ...
std::string amplitude_name(char type, const amplitudes &amp) {
    int order = std::max(creation_count(amp), annihilation_count(amp));
    std::string name = std::string(1, type) + std::to_string(order);

    size_t n_nuclear = 0;
    for (const std::string &label : fermion_labels(amp)) n_nuclear += is_nuclear(label);
    if (n_nuclear > 0 && n_nuclear == fermion_labels(amp).size()) {
        name += "_n";
    } else if (n_nuclear > 0) {
        name += "_ep";
        int n_proton = static_cast<int>(n_nuclear / 2);
        if (order >= 3) name += std::to_string(order - n_proton) + std::to_string(n_proton);
    }

    if (amp.n_ph > 0) name += "_" + std::to_string(amp.n_ph) + "p";
    return name;
}

/**
 * the simplest fraction h/k close to x, as a double; x itself if there is none
 *
 * pdaggerq builds coefficients in floating point (products of 1/n!, Bernoulli numbers, ...),
 * so a coefficient that is exactly 1 can arrive as 0.9999999999999996. the exact values are
 * fractions, so this recovers them. no list of denominators is assumed: the convergents h/k
 * of x's continued fraction are its best rational approximations, in order of increasing
 * denominator, and the first one within a relative 1e-11 of x is taken.
 *
 * it is accepted only if h*k <= 1e8. a true coefficient is a simple fraction, while almost
 * any double has *some* fraction within 1e-11 with h*k near 1e11, so the cap keeps chance
 * matches out (about 0.07% of random doubles get through, and those move by < 1e-11).
 * two fractions within the band differ by at least 1/(k*k'), so under the cap the answer
 * is unambiguous (and, by Legendre's theorem, a convergent).
 *
 * measured on QUCCSD (Bernoulli order 4): 572 coefficients, all fractions, denominators up
 * to 5760 (e.g. -169/2880), roundoff up to 1.8e-13; all recovered, even at 10x that error.
 * B10/10! = 1/47900160 is recovered; fractions with h*k > 1e8 (e.g. B12/12!) are left as
 * pdaggerq computed them, which is still correct to its roundoff.
 */
double snap_to_fraction(double x) {
    constexpr double tolerance = 1e-11;
    constexpr double max_hk = 1e8;
    if (x == 0.0 || !std::isfinite(x)) return x;

    const double a = std::fabs(x);

    // convergents h/k, from h_{n} = c_n h_{n-1} + h_{n-2} (same for k), with c_n the
    // continued-fraction terms of a; the remainder r holds what is left to expand
    double h_prev = 1.0, h = std::floor(a);
    double k_prev = 0.0, k = 1.0;
    double r = a - h;

    for (int n = 0; n < 64; n++) {
        if (std::fabs(a - h / k) <= tolerance * a)
            return h * k <= max_hk ? std::copysign(h / k, x) : x;
        if (r <= 0.0 || k > 1e15) break; // expansion ended, or past exact integer range

        r = 1.0 / r;
        double c = std::floor(r);
        r -= c;

        double h_next = c * h + h_prev, k_next = c * k + k_prev;
        h_prev = h; h = h_next;
        k_prev = k; k = k_next;
    }
    return x;
}

std::string integral_name(const std::string &type) {
    if (type == "fock") return "f";
    if (type == "core") return "h";
    if (type == "two_body") return "g";
    if (type == "eri") return "eri";
    if (type == "d+" || type == "d-") return "dp"; // dipole (bilinear coupling), one boson index
    if (type == "n+" || type == "n-") return "N0"; // nuclear dipole coupling, boson index only
    if (type == "w0") return "w0";                  // cavity frequency, boson index only
    unsupported("integral type '" + type + "'");
}

// is a tensor named in the varying option, by its name or a prefix followed by a digit?
bool is_varying(const std::string &name, const std::set<std::string> &varying) {
    for (const std::string &v : varying) {
        if (name == v) return true;
        if (name.size() > v.size() && name.compare(0, v.size(), v) == 0 && std::isdigit(name[v.size()])) return true;
    }
    return false;
}

// the expansion of pdaggerq's permutation operators as a sum of label swaps
std::vector<PermTerm> expand_perms(const pq_string &s) {
    using Swaps = std::vector<std::pair<std::string, std::string>>;

    // product of two sums of permutations (their labels are disjoint)
    auto product = [](const std::vector<PermTerm> &x, const std::vector<PermTerm> &y) {
        if (x.empty()) return y;
        std::vector<PermTerm> xy;
        for (const PermTerm &a : x) {
            for (const PermTerm &b : y) {
                PermTerm ab{a.sign * b.sign, a.swaps};
                ab.swaps.insert(ab.swaps.end(), b.swaps.begin(), b.swaps.end());
                xy.push_back(ab);
            }
        }
        return xy;
    };

    std::vector<PermTerm> perms;

    // P(p,q) = 1 - (p q)
    for (size_t i = 0; i + 1 < s.permutations.size(); i += 2)
        perms = product(perms, {{1, {}}, {-1, {{s.permutations[i], s.permutations[i + 1]}}}});

    // PP2(p0,q0;p1,q1) = 1 + (p0 p1)(q0 q1)
    const auto &pp2 = s.paired_permutations_2;
    for (size_t i = 0; i + 3 < pp2.size(); i += 4) {
        Swaps s01 = {{pp2[i], pp2[i + 2]}, {pp2[i + 1], pp2[i + 3]}};
        perms = product(perms, {{1, {}}, {1, s01}});
    }

    // PP3(p0,q0;p1,q1;p2,q2) = 1 + (01) + (02); PP6 adds (12), (01)(12), (02)(12)
    auto paired3 = [&](const std::vector<std::string> &pp, size_t i, bool six) {
        auto swap_pair = [&](int x, int y) -> Swaps {
            return {{pp[i + 2 * x], pp[i + 2 * y]}, {pp[i + 2 * x + 1], pp[i + 2 * y + 1]}};
        };
        Swaps s01 = swap_pair(0, 1), s02 = swap_pair(0, 2), s12 = swap_pair(1, 2);
        std::vector<PermTerm> sum = {{1, {}}, {1, s01}, {1, s02}};
        if (six) {
            Swaps s01_12 = s01, s02_12 = s02;
            s01_12.insert(s01_12.end(), s12.begin(), s12.end());
            s02_12.insert(s02_12.end(), s12.begin(), s12.end());
            sum.push_back({1, s12});
            sum.push_back({1, s01_12});
            sum.push_back({1, s02_12});
        }
        return sum;
    };
    for (size_t i = 0; i + 5 < s.paired_permutations_3.size(); i += 6)
        perms = product(perms, paired3(s.paired_permutations_3, i, false));
    for (size_t i = 0; i + 5 < s.paired_permutations_6.size(); i += 6)
        perms = product(perms, paired3(s.paired_permutations_6, i, true));

    return perms;
}

// an unused label in the given space, for splitting traces, e.g., replacing
// x(..p..p..) by x(..p..q..) Id(p,q)
std::string fresh_label(char space, const std::set<std::string> &used) {
    // a boson label must stay a canonical dummy, so that it is still summed
    if (space == 'b') {
        for (const std::string &l : summed_boson_labels())
            if (!used.count(l)) return l;
        unsupported("a boson trace with every canonical boson label in use");
    }
    static const std::vector<std::string> occ = {"i", "j", "k", "l", "m", "n", "I", "J", "K", "L", "M", "N"};
    static const std::vector<std::string> vir = {"a", "b", "c", "d", "e", "f", "A", "B", "C", "D", "E", "F"};

    // a nuclear label carries the species prefix, e.g. "nk"
    const bool nuclear = space == 'O' || space == 'V';
    const std::string prefix = nuclear ? std::string(1, nuclear_prefix) : "";
    const bool occupied = space == 'o' || space == 'O';
    for (const std::string &l : occupied ? occ : vir)
        if (!used.count(prefix + l)) return prefix + l;
    for (int n = 0;; n++) {
        std::string l = prefix + std::string(1, occupied ? 'o' : 'v') + "_" + std::to_string(n);
        if (!used.count(l)) return l;
    }
}

// replace each trace x(..p..p..) by x(..p..q..) Id(p,q)
void split_traces(std::vector<TensorRef> &tensors) {
    std::set<std::string> used;
    for (const TensorRef &t : tensors)
        for (const Index &i : t.idx) used.insert(i.label);

    std::vector<TensorRef> ids;
    for (TensorRef &t : tensors) {
        for (size_t p = 0; p < t.idx.size(); p++) {
            for (size_t q = p + 1; q < t.idx.size(); q++) {
                if (t.idx[p] != t.idx[q]) continue;
                // a repeated free mode label is a diagonal over modes, not a sum
                if (t.idx[p].space == 'b' && !is_summed_boson_label(t.idx[p].label))
                    unsupported("a repeated free boson label '" + t.idx[p].label + "' in " + t.name);
                Index copy = t.idx[q];                      // copy label, space, block
                copy.label = fresh_label(copy.space, used); // get fresh label
                used.insert(copy.label);                    // add fresh label to used
                t.idx[q] = copy;                            // copy back to tensor with fresh label
                TensorRef id{"Id", "", {t.idx[p], copy}};   // create delta with appropriate labels
                id.key = copy.space == 'b' ? "boson" : operator_key(id.idx); // e.g., aa_oo, or boson
                ids.push_back(id);                          // add delta to list of TensorRefs
            }
        }
    }
    tensors.insert(tensors.end(), ids.begin(), ids.end());
}

/**
 * the indices of an equation's lhs, in the order they will have in the generated code
 *
 * a trial-vector index (present when use_trial_index is set) is always the first
 * axis and must not be listed. every fermion external index must appear in label_order
 * exactly once. boson (cavity-mode) indices may be listed too, all of them or none; if
 * none are, they are the last axes, in label order, as the QED solvers expect.
 *
 * @param external_indices the external indices of one term of the equation, sorted by label
 * @param label_order the caller's order of the external labels
 * @param lhs_name the name of the lhs, for error messages
 * @return the trial index (if any), then the indices in label_order's order, then any
 *         boson indices label_order does not list
 * @throws std::invalid_argument if label_order lists the trial index, misses or repeats a
 *         fermion label, lists some but not all boson labels, or lists a label that is not
 *         external
 */
Indices lhs_indices(const Indices &external_indices, const std::vector<std::string> &label_order,
                    const std::string &lhs_name) {
    auto join = [](const auto &labels) {
        std::string s;
        for (const std::string &l : labels) s += (s.empty() ? "" : ",") + l;
        return s;
    };

    Indices ordered_indices, boson_indices;
    std::set<std::string> expected_labels, mode_labels;
    for (const Index &i : external_indices) {
        if (i.space == 'L') ordered_indices.push_back(i);
        else if (i.space == 'b') { boson_indices.push_back(i); mode_labels.insert(i.label); }
        else expected_labels.insert(i.label);
    }

    std::string usage_error = "pq_opt: " + lhs_name + " has external indices {" + join(expected_labels)
                            + "}; pass their order to add() as label_order (or as a name like " + lhs_name
                            + "(" + join(expected_labels) + ")), listing each exactly once (got ["
                            + join(label_order) + "])";
    if (!mode_labels.empty())
        usage_error += "; its boson indices {" + join(mode_labels) + "} may be listed too, all or none";

    std::set<std::string> listed_labels, listed_bosons;
    for (const std::string &label : label_order) {
        auto match = std::find_if(external_indices.begin(), external_indices.end(),
                                  [&](const Index &i) { return i.label == label; });
        if (match != external_indices.end() && match->space == 'L')
            throw std::invalid_argument("pq_opt: the trial index '" + label + "' is always the first axis of "
                                        + lhs_name + "; leave it out of the index order");
        if (match == external_indices.end()) throw std::invalid_argument(usage_error);
        auto &listed = match->space == 'b' ? listed_bosons : listed_labels;
        if (!listed.insert(label).second) throw std::invalid_argument(usage_error);
        ordered_indices.push_back(*match);
    }
    if (listed_labels != expected_labels) throw std::invalid_argument(usage_error);
    if (!listed_bosons.empty() && listed_bosons != mode_labels) throw std::invalid_argument(usage_error);

    // boson indices not listed are the last axes, in label order
    if (listed_bosons.empty())
        ordered_indices.insert(ordered_indices.end(), boson_indices.begin(), boson_indices.end());
    return ordered_indices;
}

} // namespace

/**
 * build one equation from the strings held by a pq_helper
 *
 * each string becomes a Term: its coefficient, its tensors (deltas as Id, integrals,
 * amplitudes), and the expansion of its permutation operators. along the way
 *   - eri blocks are permuted into the set the generated code provides (oovv, vovo, ...),
 *     with the sign folded into the coefficient (options.permute_eri)
 *   - r/l amplitudes get a leading trial-vector index (options.use_trial_index)
 *   - boson (cavity-mode) indices follow a tensor's fermion indices
 *   - traces are split into contractions with Id, so no tensor repeats a label
 * the first term fixes the lhs indices (see lhs_indices); every later term must have the
 * same external indices.
 *
 * @param pq the pq_helper; its spin- or range-blocked strings are used if pdaggerq is blocked
 * @param equation_name the lhs name, e.g. "rt2", or with its index order, e.g. "rt2(a,b,i,j)"
 * @param label_order the order of the lhs indices, unless equation_name gives it; every
 *        external index exactly once, except the trial index, which is always first, and
 *        boson indices, which are optional (all or none) and otherwise last
 * @param options ingest options
 * @return the equation; it has no terms if the pq_helper holds no strings
 * @throws std::invalid_argument if the lhs index order is missing, incomplete, or given twice
 * @throws std::runtime_error if a term uses something pq_opt does not support yet (e.g. an
 *         integral type it does not know), breaks the label rule, or has other external
 *         indices than the lhs
 */
Equation ingest(const pq_helper &pq, const std::string &equation_name,
                const std::vector<std::string> &label_order, const IngestOptions &options) {

    Equation eq;

    // "rt2(a,b,i,j)" gives the lhs index order in the name
    std::vector<std::string> label_order_from_name;
    std::string lhs_name = equation_name;
    size_t paren = equation_name.find('(');
    if (paren != std::string::npos) {
        lhs_name = equation_name.substr(0, paren);
        std::string inner = equation_name.substr(paren + 1, equation_name.find(')') - paren - 1);
        size_t start = 0;
        while (start <= inner.size()) {
            size_t comma = inner.find(',', start);
            if (comma == std::string::npos) comma = inner.size();
            std::string label = inner.substr(start, comma - start);
            if (!label.empty()) label_order_from_name.push_back(label);
            start = comma + 1;
        }
    }
    eq.lhs.name = lhs_name;

    // are the equations blocked by spin or range?
    bool blocked = pq_string::is_spin_blocked || pq_string::is_range_blocked;

    // blocked or not-blocked ordered pq_strings
    const auto &strings = pq.get_ordered_strings(blocked);

    bool have_lhs = false;
    std::set<std::string> lhs_labels;

    for (const auto &s : strings) {
        if (s->skip) continue;
        if (has_uncontracted_operators(*s))
            unsupported("a term with uncontracted operators");

        Term term;
        term.coeff = snap_to_fraction(s->sign * s->factor);

        // the term as pdaggerq prints it (coefficient first), before any eri sign change below
        for (const std::string &word : s->get_string())
            term.comment += (term.comment.empty() ? "" : " ") + word;

        // add delta functions to term as TensorRefs
        for (const delta_functions &delta : s->deltas) {
            TensorRef t{"Id", "", indices_of(delta)};
            // a delta between boson modes is an identity over modes, Id["boson"]
            t.key = fermion_labels(delta).empty() ? "boson" : operator_key(t.idx);
            term.tensors.push_back(t);
        }

        // add integrals to term as TensorRefs
        for (const auto &[type, ints] : s->ints) {
            for (const integrals &integral : ints) {
                TensorRef t{integral_name(type), "", indices_of(integral)};
                if (type == "eri" && options.permute_eri)
                    term.coeff *= permute_eri(t.idx, options.symmetric_eri);
                t.key = operator_key(t.idx);
                t.varies = is_varying(t.name, options.varying);
                // <p,q||r,s> changes sign under p <-> q and under r <-> s
                if (type == "eri") t.antisymmetric = antisymmetric_groups(t.idx, {{0, 1}, {2, 3}});
                term.tensors.push_back(t);
            }
        }

        // in single-mode releases the cavity frequency is a flag on the string: a scalar factor
        if (has_w0_factor(*s)) term.tensors.push_back({"w0", "", {}});

        // add amplitudes to term as TensorRefs
        for (const auto &[type, amps] : s->amps) {
            for (const amplitudes &amp : amps) {
                TensorRef t{amplitude_name(type, amp), "", indices_of(amp)};
                t.key = blkstring(t.idx);
                t.varies = is_varying(t.name, options.varying);
                if (options.use_trial_index && (type == 'r' || type == 'l')) {
                    Index trial;
                    trial.label = type == 'r' ? "R" : "L";   // R for r amplitudes, L for l amplitudes
                    trial.space = 'L';                       // the trial-vector space (unblocked)
                    t.idx.insert(t.idx.begin(), trial);
                }
                // an amplitude changes sign under permutations among its creation labels and
                // among its annihilation labels (they follow a trial index, if any). skipped
                // when the counts do not describe its labels
                const int n_create = creation_count(amp), n_annihilate = annihilation_count(amp);
                bool counts_match = n_create >= 0 && n_annihilate >= 0
                    && static_cast<size_t>(n_create + n_annihilate) == fermion_labels(amp).size();
                if (amp.has_permutational_symmetry && counts_match) {
                    size_t offset = t.idx.size() > fermion_labels(amp).size() + boson_labels(amp).size() ? 1 : 0;
                    std::vector<size_t> create, annihilate;
                    for (int p = 0; p < n_create; p++) create.push_back(offset + p);
                    for (int p = 0; p < n_annihilate; p++) annihilate.push_back(offset + n_create + p);
                    t.antisymmetric = antisymmetric_groups(t.idx, {create, annihilate});
                }
                term.tensors.push_back(t);
            }
        }

        // split traces (full or partial): x(..p..p..) becomes x(..p..q..) Id(p,q), in this same term
        split_traces(term.tensors);

        // expand permutations, e.g., P(p,q) = 1 - (p q)
        term.perms = expand_perms(*s);

        // check the label rule; the labels that appear once are the external indices
        std::map<std::string, std::pair<int, Index>> label_counts;
        for (const TensorRef &t : term.tensors) {
            for (const Index &i : t.idx) {
                auto &entry = label_counts[i.label];
                entry.first++;
                entry.second = i;
            }
        }
        // a fermion label is external if it appears once (twice: summed). a boson label is
        // external if it is a free mode label, however many tensors carry it; a canonical
        // dummy is summed, even if only one tensor carries it
        Indices external_indices;
        for (const auto &[label, entry] : label_counts) {
            bool boson = entry.second.space == 'b';
            if (!boson && entry.first > 2)
                throw std::runtime_error("pq_opt: index '" + label + "' appears more than twice in "
                                         + lhs_name + " += " + term.comment);
            bool external = boson ? !is_summed_boson_label(label) : entry.first == 1;
            if (external) external_indices.push_back(entry.second);
        }

        // the first term fixes the lhs; the rest must agree with it
        if (!have_lhs) {
            if (!label_order_from_name.empty() && !label_order.empty())
                throw std::invalid_argument("pq_opt: give the index order of " + equation_name
                                            + " once, in its name or in label_order, not both");
            eq.lhs.idx = lhs_indices(external_indices,
                                     label_order_from_name.empty() ? label_order : label_order_from_name, lhs_name);
            for (const Index &i : eq.lhs.idx) lhs_labels.insert(i.label);
            have_lhs = true;
        }

        std::set<std::string> term_labels;
        for (const Index &i : external_indices) term_labels.insert(i.label);
        if (term_labels != lhs_labels)
            throw std::runtime_error("pq_opt: the external indices of " + lhs_name + " += " + term.comment
                                     + " do not match the lhs");

        eq.terms.push_back(std::move(term));
    }

    if (eq.terms.empty())
        std::cout << "WARNING: pq_opt: no terms found for equation '" << equation_name << "'" << std::endl;

    return eq;
}

} // namespace pdaggerq::opt
