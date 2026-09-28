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
//
// Merging (opt_level 4) follows hoisting: terms of an equation that multiply the same
// varying tensors, in the same way, by one fixed tensor each are one term, reading the
// sum of their fixed tensors (built once; an element of Hbar in EOM-CC). Then terms that
// multiply the same fixed tensor, in the same way, by different per-call parts are one
// term, reading the sum of those parts (built on every call), when that saves flops. In
// code where nothing varies between calls (e.g. CC residuals), the same is done with each
// term's largest tensor.
//
// Hoisting (opt_level 3) uses the same keys. When the code will be called repeatedly
// with new r or l amplitudes (TensorRef::varies: EOM trial vectors, lambda amplitudes)
// while everything else stays fixed, each term's tree is chosen to minimize the flops
// per call, where a subtree of fixed tensors can be computed once and stored (if no
// intermediate in it is larger than the term's largest fixed tensor), its flops spread
// over the expected number of calls. Those subtrees are
// defined once, as reused_["0001_ov"], a product of two tensors per definition, and
// identical products (by key) are defined once for all terms. Terms with no varying
// tensor at all are summed once per equation into a reused_ of the lhs shape.

#include "passes.h"

#include <algorithm>
#include <functional>
#include <cstdio>
#include <map>
#include <set>
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

// the leaves of a tree, left to right
void leaves(const ExprPtr &e, std::vector<TensorRef> &found) {
    if (!e) return;
    if (e->is_leaf()) found.push_back(*e->leaf);
    for (const ExprPtr &arg : e->args) leaves(arg, found);
}

} // namespace

std::vector<Equation> hoist_invariants(std::vector<Equation> &eqs, const Sizes &sizes, double calls,
                                       bool use_antisymmetry) {

    // nothing varies between calls: the code runs once per call with new inputs throughout
    // (e.g. CC residuals, where t changes), so there is nothing to hoist
    bool any_varying = false;
    for (const Equation &eq : eqs)
        for (const Term &term : eq.terms)
            for (const TensorRef &t : term.tensors) any_varying |= t.varies;
    if (!any_varying) return {};

    std::vector<Equation> products;           // reused_ = a product of two tensors, in order
    std::map<std::string, size_t> product_of; // canonical key -> its position in products
    std::vector<Equation> constants;          // reused_ = the terms of an equation that never vary
    size_t count = 0;                         // reused_ made so far (their ids)

    // define a hoisted subtree bottom-up, one product at a time; its value is sign * ref
    auto define = [&](auto &&self, const ExprPtr &e) -> std::pair<TensorRef, int> {
        if (e->is_leaf()) return {*e->leaf, 1};
        auto [x, sx] = self(self, e->args[0]);
        auto [y, sy] = self(self, e->args[1]);
        Canonical c = canonical(x, y, e->idx, use_antisymmetry);
        auto it = product_of.find(c.key);
        if (it == product_of.end()) {
            Equation def;
            def.lhs = {"reused_", intermediate_key(++count, c.x_idx), c.x_idx};
            Term def_term;
            def_term.tensors = {c.first, c.second};
            def_term.comment = text(c.first) + " " + text(c.second);
            def.terms.push_back(def_term);
            it = product_of.emplace(c.key, products.size()).first;
            products.push_back(std::move(def));
        }
        // x y = c.sign * (the definition), with this occurrence's labels
        return {TensorRef{"reused_", products[it->second].lhs.key, c.x_idx}, sx * sy * c.sign};
    };

    for (Equation &eq : eqs) {
        Equation constant;
        std::vector<Term> kept;
        for (Term &term : eq.terms) {
            bool varies = false;
            for (const TensorRef &t : term.tensors) varies |= t.varies;
            if (!varies) {
                constant.terms.push_back(std::move(term));
                continue;
            }

            // hoist until the term's best per-call tree has no stored fixed subtree left
            while (true) {
                ExprPtr tree = optimal_expr(term, eq.lhs.idx, sizes, calls);
                std::vector<ExprPtr> hoisted = hoisted_subtrees(tree, hoist_cap(term, sizes), sizes);
                if (hoisted.empty()) break;

                std::vector<bool> taken(term.tensors.size(), false);
                std::vector<TensorRef> added;
                for (const ExprPtr &sub : hoisted) {
                    std::vector<TensorRef> inside;
                    leaves(sub, inside);
                    for (const TensorRef &t : inside) taken[position(term, t, taken)] = true;
                    auto [ref, sign] = define(define, sub);
                    added.push_back(ref);
                    term.coeff *= sign;
                }
                std::vector<TensorRef> rest;
                for (size_t p = 0; p < term.tensors.size(); p++)
                    if (!taken[p]) rest.push_back(term.tensors[p]);
                rest.insert(rest.end(), added.begin(), added.end());
                term.tensors = std::move(rest);
            }
            kept.push_back(std::move(term));
        }

        // the terms that never vary, summed once
        if (!constant.terms.empty()) {
            constant.lhs = {"reused_", intermediate_key(++count, eq.lhs.idx), eq.lhs.idx};
            Term read;
            read.tensors = {constant.lhs};
            read.comment = "the terms of " + eq.lhs.name + " that do not vary between calls";
            kept.push_back(read);
            constants.push_back(std::move(constant));
        }
        eq.terms = std::move(kept);
    }

    products.insert(products.end(), constants.begin(), constants.end());
    return products;
}

namespace {

// a term of a group of terms that share a common part
struct Member {
    size_t term;                             // position in the equation's terms
    int sign;                                // its common part = sign * the group's arrangement
    std::map<std::string, std::string> map;  // its summed labels in the common part -> "#k" tokens
    std::vector<size_t> common;              // positions of the common part's tensors in the term
};

// terms that share a common part and a permutation operator
struct Group {
    std::vector<TensorRef> common;           // the common part, as the first member writes it
    std::vector<Member> members;
};

/**
 * group an equation's terms by a common part (e.g. their varying tensors), up to the
 * antisymmetry of those tensors and the names of their summed labels, and by permutation operator
 * @param common_of the positions of a term's common tensors; empty if the term takes no part
 */
std::vector<Group> group_by_common(const Equation &eq,
                                   const std::function<std::vector<size_t>(const Term &)> &common_of,
                                   bool use_antisymmetry) {
    std::map<std::string, size_t> group_of;  // key -> position in groups
    std::vector<Group> groups;

    for (size_t n = 0; n < eq.terms.size(); n++) {
        const Term &term = eq.terms[n];
        std::vector<size_t> common = common_of(term);
        if (common.empty()) continue;

        // the smallest text of the common part over its antisymmetric arrangements; summed labels
        // are numbered in order of appearance, external labels (on the lhs) keep their names
        std::vector<std::pair<std::vector<TensorRef>, int>> choices = {{{}, 1}};
        for (size_t position : common) {
            std::vector<std::pair<std::vector<TensorRef>, int>> next;
            for (const auto &[ts, sign] : choices)
                for (const auto &[a, s] : arrangements(term.tensors[position], use_antisymmetry)) {
                    std::vector<TensorRef> more = ts;
                    more.push_back(a);
                    next.push_back({more, sign * s});
                }
            choices = std::move(next);
        }
        std::string best;
        Member member{n, 1, {}, common};
        std::vector<TensorRef> best_arrangement;
        for (const auto &[ts, sign] : choices) {
            std::map<std::string, std::string> token;
            std::string text;
            for (const TensorRef &t : ts) {
                text += t.name + "[" + t.key + "](";
                for (const Index &i : t.idx) {
                    bool external = std::find(eq.lhs.idx.begin(), eq.lhs.idx.end(), i) != eq.lhs.idx.end();
                    if (!external && !token.count(i.label)) token[i.label] = "#" + std::to_string(token.size());
                    text += (external ? i.label : token[i.label]) + ",";
                }
                text += ")";
            }
            if (!best.empty() && text >= best) continue;
            best = text;
            member.sign = sign;
            member.map = token;
            best_arrangement = ts;
        }
        for (const PermTerm &p : term.perms) {
            best += "|" + std::to_string(p.sign);
            for (const auto &[x, y] : p.swaps) best += "(" + x + "," + y + ")";
        }

        auto it = group_of.find(best);
        if (it == group_of.end()) {
            it = group_of.emplace(best, groups.size()).first;
            groups.push_back({best_arrangement, {}});
        }
        groups[it->second].members.push_back(member);
    }
    return groups;
}

// the rest of a member's term (the tensors not in its common part), with the labels it shares
// with the common part renamed to the group's (the first member's). labels summed within the
// rest keep their names unless those are in use by the group, in which case they get new ones
std::vector<TensorRef> rest_of(const Equation &eq, const Group &group, const Member &m) {
    std::map<std::string, std::string> label_of;   // "#k" -> the group's label
    for (const auto &[label, token] : group.members[0].map) label_of[token] = label;

    std::set<std::string> in_use;
    for (const TensorRef &t : group.common)
        for (const Index &i : t.idx) in_use.insert(i.label);
    for (const Index &i : eq.lhs.idx) in_use.insert(i.label);

    const Term &term = eq.terms[m.term];
    std::map<std::string, std::string> renamed;
    std::vector<TensorRef> rest;
    for (size_t p = 0; p < term.tensors.size(); p++) {
        if (std::find(m.common.begin(), m.common.end(), p) != m.common.end()) continue;
        TensorRef t = term.tensors[p];
        for (Index &i : t.idx) {
            auto shared = m.map.find(i.label);
            if (shared != m.map.end()) {
                i.label = label_of.at(shared->second);
            } else if (std::find(eq.lhs.idx.begin(), eq.lhs.idx.end(), i) == eq.lhs.idx.end() && in_use.count(i.label)) {
                auto it = renamed.find(i.label);
                if (it == renamed.end()) {
                    std::string fresh;
                    for (size_t k = 1; fresh.empty() || in_use.count(fresh); k++)
                        fresh = std::string(1, i.space) + "_s" + std::to_string(k);
                    in_use.insert(fresh);
                    it = renamed.emplace(i.label, fresh).first;
                }
                i.label = it->second;
            }
        }
        rest.push_back(t);
    }
    return rest;
}

// the indices a group's sum carries: those of the first member's rest that are external or
// shared with the common part, in order of appearance
Indices sum_indices(const Equation &eq, const Group &group, const std::vector<TensorRef> &first_rest) {
    Indices idx;
    for (const TensorRef &t : first_rest)
        for (const Index &i : t.idx) {
            bool external = std::find(eq.lhs.idx.begin(), eq.lhs.idx.end(), i) != eq.lhs.idx.end();
            bool shared = false;
            for (const TensorRef &c : group.common)
                shared |= std::find(c.idx.begin(), c.idx.end(), i) != c.idx.end();
            if ((external || shared) && std::find(idx.begin(), idx.end(), i) == idx.end()) idx.push_back(i);
        }
    return idx;
}

} // namespace

std::vector<Equation> merge_terms(std::vector<Equation> &eqs, size_t first_id, bool use_antisymmetry) {
    std::vector<Equation> sums;

    // the terms that are (varying tensors) x (one fixed tensor), grouped by the varying part and
    // the permutation operator. a group of c_k P[V F_k] becomes P[V G] with G = sum_k c_k F_k,
    // computed once
    auto varying_part = [](const Term &term) {
        std::vector<size_t> varying;
        size_t fixed = 0;
        for (size_t p = 0; p < term.tensors.size(); p++) {
            if (term.tensors[p].varies) varying.push_back(p);
            else fixed++;
        }
        return fixed == 1 ? varying : std::vector<size_t>{};
    };

    for (Equation &eq : eqs) {
        std::vector<bool> merged(eq.terms.size(), false);
        std::vector<Term> added;
        for (const Group &group : group_by_common(eq, varying_part, use_antisymmetry)) {
            if (group.members.size() < 2) continue;

            // G's indices are the first member's fixed tensor's
            Indices idx = rest_of(eq, group, group.members[0]).at(0).idx;
            Equation sum;
            sum.lhs = {"reused_", intermediate_key(first_id + sums.size(), idx), idx};
            for (const Member &m : group.members) {
                Term piece;
                piece.coeff = eq.terms[m.term].coeff * m.sign;
                piece.tensors = rest_of(eq, group, m);
                piece.comment = eq.terms[m.term].comment;
                sum.terms.push_back(piece);
                merged[m.term] = true;
            }

            // the merged term: V, as the first member writes it, times G
            Term term;
            term.tensors = group.common;
            term.tensors.push_back(sum.lhs);
            term.perms = eq.terms[group.members[0].term].perms;
            term.comment = std::to_string(group.members.size()) + " merged terms";
            added.push_back(term);
            sums.push_back(std::move(sum));
        }

        std::vector<Term> kept;
        for (size_t n = 0; n < eq.terms.size(); n++)
            if (!merged[n]) kept.push_back(std::move(eq.terms[n]));
        kept.insert(kept.end(), added.begin(), added.end());
        eq.terms = std::move(kept);
    }
    return sums;
}

std::vector<Equation> merge_by_anchor(std::vector<Equation> &eqs, const Sizes &sizes, bool use_antisymmetry) {
    std::vector<Equation> sums;

    // when some tensor varies between calls, only terms with varying tensors take part, and
    // their anchor is their largest fixed tensor (what they share must not change between
    // calls in a way the sum does not); otherwise every term takes part, anchored on its
    // largest tensor
    bool any_varying = false;
    for (const Equation &eq : eqs)
        for (const Term &term : eq.terms)
            for (const TensorRef &t : term.tensors) any_varying |= t.varies;

    // the terms grouped by their anchor F (the first, if tied) and the permutation operator. a
    // group of c_k P[F R_k] becomes P[F S] with S = sum_k c_k R_k, computed on every call.
    // terms whose rest carries boson labels are left alone: renaming a summed boson label would
    // break the label rule (ir.h)
    auto anchor = [&](const Term &term) {
        bool takes_part = !any_varying;
        long largest = -1;
        double largest_size = -1.0;
        for (size_t p = 0; p < term.tensors.size(); p++) {
            const TensorRef &t = term.tensors[p];
            if (t.varies) {
                takes_part = true;
            } else if (extent(t.idx, sizes) > largest_size) {
                largest = static_cast<long>(p);
                largest_size = extent(t.idx, sizes);
            }
        }
        if (!takes_part || largest < 0 || term.tensors.size() < 2) return std::vector<size_t>{};
        for (size_t p = 0; p < term.tensors.size(); p++)
            for (const Index &i : term.tensors[p].idx)
                if (static_cast<long>(p) != largest && i.space == 'b') return std::vector<size_t>{};
        return std::vector<size_t>{static_cast<size_t>(largest)};
    };

    for (Equation &eq : eqs) {
        std::vector<bool> merged(eq.terms.size(), false);
        std::vector<Term> added;
        for (const Group &group : group_by_common(eq, anchor, use_antisymmetry)) {
            if (group.members.size() < 2) continue;
            Indices idx = sum_indices(eq, group, rest_of(eq, group, group.members[0]));

            // contracting F with S once replaces one such contraction per term, but each term must
            // then build its rest first, which may cost more than its own best order. a term takes
            // part only if that costs less than its best order, and the group only if the flops
            // saved exceed the one contraction of F with S
            Indices touched = group.common[0].idx;
            for (const Index &i : idx)
                if (std::find(touched.begin(), touched.end(), i) == touched.end()) touched.push_back(i);
            double final_contraction = extent(touched, sizes);

            std::vector<const Member *> gaining;
            double saved = 0.0;
            for (const Member &m : group.members) {
                Term rest;
                rest.tensors = rest_of(eq, group, m);
                double own = cost(optimal_expr(eq.terms[m.term], eq.lhs.idx, sizes), sizes);
                double forced = cost(optimal_expr(rest, idx, sizes), sizes);
                if (own - forced <= 0.0) continue;
                gaining.push_back(&m);
                saved += own - forced;
            }
            if (gaining.size() < 2 || saved <= final_contraction * (1.0 + 1e-12)) continue;

            Equation sum;
            sum.lhs = {"tmps_", "s" + intermediate_key(sums.size() + 1, idx), idx};
            for (const Member *m : gaining) {
                Term piece;
                piece.coeff = eq.terms[m->term].coeff * m->sign;
                piece.tensors = rest_of(eq, group, *m);
                piece.comment = eq.terms[m->term].comment;
                sum.terms.push_back(piece);
                merged[m->term] = true;
            }

            // the merged term: F, as the first member writes it, times S
            Term term;
            term.tensors = group.common;
            term.tensors.push_back(sum.lhs);
            term.perms = eq.terms[group.members[0].term].perms;
            term.comment = std::to_string(gaining.size()) + " merged terms";
            added.push_back(term);
            sums.push_back(std::move(sum));
        }

        std::vector<Term> kept;
        for (size_t n = 0; n < eq.terms.size(); n++)
            if (!merged[n]) kept.push_back(std::move(eq.terms[n]));
        kept.insert(kept.end(), added.begin(), added.end());
        eq.terms = std::move(kept);
    }
    return sums;
}

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
