//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: intermediates.cc
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

// Shared intermediates (opt_level >= 2).
//
// Every term already has an optimal contraction tree (order.cc). A two-tensor node
// of that tree, X = A B, is a contraction the term performs anyway. When the same
// product occurs in several terms (of any equations), computing X once and reading
// it everywhere saves exactly (occurrences - 1) * cost(A B) flops, and it can never
// make a term's own order worse, because X was part of the term's best tree.
//
// Occurrences are matched by a canonical key: the two tensors' names, blocks, and
// index pattern, with labels renamed in order of first appearance and marked free
// (kept in X) or summed. The key is the smallest such text over both operand orders
// and every index arrangement the tensors' antisymmetry allows (e.g. t2(a,b,i,j) =
// -t2(b,a,i,j)); the arrangement's sign goes into the term's coefficient. A tensor
// is only rearranged if pdaggerq marks it antisymmetric (TensorRef::antisymmetric),
// and use_antisymmetry = false turns this off: then only identical products match.
//
// The loop is greedy: substitute the product that saves the most, recompute only
// the trees of the terms it changed, and repeat. A substituted X is an ordinary
// tensor afterwards, so later rounds can pair it again (X C), which builds deeper
// intermediates.

#include "passes.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace pdaggerq::opt {

namespace {

// a two-tensor node of a term's optimal contraction tree
struct Candidate {
    std::string key;     // canonical form, the same for every occurrence of the product
    size_t a, b;         // the positions of the two tensors in the term
    Indices x_idx;       // the product's free indices, in canonical order
    double cost;         // flops of the contraction
    int sign;            // the product in the term = sign * (the canonical product)
    TensorRef x, y;      // the two tensors in the canonical arrangement
};

// "t2[abab](0vb+,1va.,...)"-style text of a tensor, with labels renamed by token
std::string encode(const TensorRef &t, std::map<std::string, int> &token, const Indices &free) {
    std::string s = t.name + "[" + t.key + "](";
    for (const Index &i : t.idx) {
        auto it = token.emplace(i.label, static_cast<int>(token.size())).first;
        bool is_free = std::find(free.begin(), free.end(), i) != free.end();
        s += std::to_string(it->second) + i.space + (i.block ? i.block : '-') + (is_free ? '+' : '.') + ",";
    }
    return s + ")";
}

// the parity of a permutation: +1 or -1
int parity(const std::vector<size_t> &perm) {
    int sign = 1;
    for (size_t i = 0; i < perm.size(); i++)
        for (size_t j = i + 1; j < perm.size(); j++)
            if (perm[i] > perm[j]) sign = -sign;
    return sign;
}

// every arrangement of a tensor's indices that its antisymmetry allows, with its sign:
// t(arrangement) = sign * t(original labels)
std::vector<std::pair<TensorRef, int>> arrangements(const TensorRef &t, bool use_antisymmetry) {
    std::vector<std::pair<TensorRef, int>> found = {{t, 1}};
    if (!use_antisymmetry) return found;
    for (const std::vector<size_t> &group : t.antisymmetric) {
        std::vector<std::pair<TensorRef, int>> next;
        std::vector<size_t> perm(group.size());
        for (size_t k = 0; k < perm.size(); k++) perm[k] = k;
        do {
            int sign = parity(perm);
            for (const auto &[u, s] : found) {
                TensorRef v = u;
                for (size_t k = 0; k < group.size(); k++) v.idx[group[k]] = u.idx[group[perm[k]]];
                next.push_back({v, s * sign});
            }
        } while (std::next_permutation(perm.begin(), perm.end()));
        found = std::move(next);
    }
    return found;
}

// the canonical form of the product x y with free indices `free`
struct Canonical {
    std::string key;
    Indices x_idx;   // the free indices in canonical order (the order of their tokens)
    int sign;        // x y = sign * first second
    TensorRef first, second;
};

Canonical canonical(const TensorRef &x, const TensorRef &y, const Indices &free, bool use_antisymmetry) {
    Canonical best;
    std::map<std::string, int> best_token;
    for (const auto &[xa, sx] : arrangements(x, use_antisymmetry)) {
        for (const auto &[ya, sy] : arrangements(y, use_antisymmetry)) {
            for (int order = 0; order < 2; order++) {
                const TensorRef &first = order ? ya : xa, &second = order ? xa : ya;
                std::map<std::string, int> token;
                std::string key = encode(first, token, free) + encode(second, token, free);
                if (!best.key.empty() && key >= best.key) continue;
                // t(arrangement) = sign * t(original), so x y = (sx sy) * first second
                best = {key, {}, sx * sy, first, second};
                best_token = std::move(token);
            }
        }
    }
    best.x_idx = free;
    std::sort(best.x_idx.begin(), best.x_idx.end(),
              [&](const Index &p, const Index &q) { return best_token.at(p.label) < best_token.at(q.label); });
    return best;
}

// the position in the term of a tensor from its tree, skipping positions already taken
size_t position(const Term &term, const TensorRef &t, const std::vector<bool> &taken) {
    for (size_t p = 0; p < term.tensors.size(); p++) {
        const TensorRef &u = term.tensors[p];
        if (!taken[p] && u.name == t.name && u.key == t.key && u.idx == t.idx) return p;
    }
    throw std::logic_error("pq_opt: a tensor of a contraction tree is not in its term");
}

// the two-tensor nodes of a term's optimal contraction tree
std::vector<Candidate> candidates(const Term &term, const Indices &out, const Sizes &sizes, bool use_antisymmetry) {
    std::vector<Candidate> found;
    std::vector<bool> taken(term.tensors.size(), false);

    auto visit = [&](auto &&self, const ExprPtr &e) -> void {
        if (!e || e->is_leaf()) return;
        if (e->args.size() == 2 && e->args[0]->is_leaf() && e->args[1]->is_leaf()) {
            const TensorRef &x = *e->args[0]->leaf, &y = *e->args[1]->leaf;
            size_t a = position(term, x, taken);
            taken[a] = true;
            size_t b = position(term, y, taken);
            taken[b] = true;

            Indices touched = x.idx;
            for (const Index &i : y.idx)
                if (std::find(touched.begin(), touched.end(), i) == touched.end()) touched.push_back(i);

            Canonical c = canonical(x, y, e->idx, use_antisymmetry);
            found.push_back({c.key, a, b, c.x_idx, extent(touched, sizes), c.sign, c.first, c.second});
            return;
        }
        for (const ExprPtr &arg : e->args) self(self, arg);
    };
    visit(visit, optimal_expr(term, out, sizes));
    return found;
}

// "f(k,c)"-style text of a tensor, for comments
std::string text(const TensorRef &t) {
    std::string s = t.key.empty() ? t.name : t.name + "[" + t.key + "]";
    s += "(";
    for (size_t i = 0; i < t.idx.size(); i++) s += (i ? "," : "") + t.idx[i].label;
    return s + ")";
}

// the map key of an intermediate: "0001_vvoo", with its blocks when blocked: "0001_abab_vvoo"
std::string intermediate_key(size_t id, const Indices &idx) {
    char number[16];
    std::snprintf(number, sizeof number, "%04zu", id);
    std::string spaces, blocks;
    for (const Index &i : idx) {
        spaces += i.space;
        if (i.block) blocks += i.block;
    }
    return std::string(number) + (blocks.empty() ? "" : "_" + blocks) + "_" + spaces;
}

} // namespace

std::vector<Equation> extract_intermediates(std::vector<Equation> &eqs, const Sizes &sizes, long max_temps,
                                           bool use_antisymmetry) {

    // every term, with the lhs indices its tree must produce and its current candidates
    struct Slot {
        Term *term;
        const Indices *out;
        std::vector<Candidate> candidates;
    };
    std::vector<Slot> slots;
    for (Equation &eq : eqs)
        for (Term &term : eq.terms) slots.push_back({&term, &eq.lhs.idx, {}});

    // key -> slot -> occurrences in that slot, and the cost of each key's contraction
    std::map<std::string, std::map<size_t, int>> occurrences;
    std::map<std::string, double> cost_of;

    auto index_slot = [&](size_t s) {
        slots[s].candidates = candidates(*slots[s].term, *slots[s].out, sizes, use_antisymmetry);
        for (const Candidate &c : slots[s].candidates) {
            occurrences[c.key][s]++;
            cost_of[c.key] = c.cost;
        }
    };
    auto unindex_slot = [&](size_t s) {
        for (const Candidate &c : slots[s].candidates) {
            auto &where = occurrences[c.key];
            if (--where[s] == 0) where.erase(s);
            if (where.empty()) occurrences.erase(c.key);
        }
        slots[s].candidates.clear();
    };
    for (size_t s = 0; s < slots.size(); s++) index_slot(s);

    std::vector<Equation> definitions;
    while (max_temps < 0 || static_cast<long>(definitions.size()) < max_temps) {

        // the product that saves the most; ties go to the smaller key (map order)
        std::string best;
        double best_saving = 0.0;
        for (const auto &[key, where] : occurrences) {
            int count = 0;
            for (const auto &[s, n] : where) count += n;
            double saving = (count - 1) * cost_of[key];
            if (saving > best_saving * (1.0 + 1e-12)) {
                best_saving = saving;
                best = key;
            }
        }
        if (best.empty()) break;

        // the intermediate, defined from its first occurrence
        std::map<size_t, int> where = occurrences[best];
        const Slot &first = slots[where.begin()->first];
        const Candidate &c0 = *std::find_if(first.candidates.begin(), first.candidates.end(),
                                            [&](const Candidate &c) { return c.key == best; });
        const TensorRef &x0 = c0.x, &y0 = c0.y;

        Equation def;
        def.lhs = {"tmps_", intermediate_key(definitions.size() + 1, c0.x_idx), c0.x_idx};
        Term def_term;
        def_term.tensors = {x0, y0};
        def_term.comment = text(x0) + " " + text(y0);
        def.terms.push_back(def_term);

        // replace the product by the intermediate wherever it occurs
        for (const auto &[s, n] : where) {
            Term &term = *slots[s].term;
            std::vector<bool> used(term.tensors.size(), false);
            std::vector<TensorRef> added;
            for (const Candidate &c : slots[s].candidates) {
                if (c.key != best || used[c.a] || used[c.b]) continue;
                used[c.a] = used[c.b] = true;
                added.push_back({def.lhs.name, def.lhs.key, c.x_idx});
                term.coeff *= c.sign;
            }
            unindex_slot(s);

            std::vector<TensorRef> kept;
            for (size_t p = 0; p < term.tensors.size(); p++)
                if (!used[p]) kept.push_back(term.tensors[p]);
            kept.insert(kept.end(), added.begin(), added.end());
            term.tensors = std::move(kept);
            index_slot(s);
        }

        definitions.push_back(std::move(def));
    }
    return definitions;
}

} // namespace pdaggerq::opt
