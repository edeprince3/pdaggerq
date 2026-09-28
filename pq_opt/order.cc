//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: order.cc
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

// Contraction order for a single term.
//
// The optimal binary tree is found by dynamic programming over subsets of the
// tensors (the same idea as join ordering in databases): the best way to
// contract a subset S is the cheapest split S = A + B, where A and B have
// already been solved. For n tensors that is O(3^n) work, which is trivial for
// the 2-8 tensors of a coupled-cluster term.
//
// As a running example, take
//
// f(k,c) t1(c,i) t2(a,b,j,k) with output a,b,i,j:
//
// Tensor        Bit  Mask
// f(k,c)        0    001
// t1(c,i)       1    010
// t2(a,b,j,k)   2    100
//

#include "passes.h"

#include <algorithm>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace pdaggerq::opt {

namespace {

// union of index sets, in order of first appearance
Indices merge(const Indices &x, const Indices &y) {
    Indices all = x;
    for (const Index &i : y)
        if (std::find(all.begin(), all.end(), i) == all.end()) all.push_back(i);
    return all;
}

// all indices touched by the contraction at an internal node
Indices touched(const Expr &node) {
    Indices all;
    for (const ExprPtr &arg : node.args) all = merge(all, arg->idx);
    return all;
}

ExprPtr make_leaf(const TensorRef &t) {
    auto node = std::make_shared<Expr>();
    node->leaf = std::make_shared<const TensorRef>(t);
    node->idx = t.idx;
    return node;
}

} // namespace

// the product of the sizes of each index's space. Size is looked up by the
// index's space ('o', 'v', 'O', 'V', 'L', 'b'), not its label. With the defaults (o = 20,
// v = 100, L = 10), indices (a,b,i,j) give 100 × 100 × 20 × 20 = 4×10^6.
double extent(const Indices &idx, const Sizes &sizes) {
    double n = 1.0;
    for (const Index &i : idx) {
        // a missing size would silently make the space free in the cost model
        auto it = sizes.find(i.space);
        if (it == sizes.end())
            throw std::runtime_error("pq_opt: no size for index space '" + std::string(1, i.space)
                                     + "' (index '" + i.label + "'); set it with the 'sizes' option");
        n *= it->second;
    }
    return n;
}

// cost takes any node and returns the flops for the whole subtree below
// it. Called on the root, it gives the whole term's cost.
double cost(const ExprPtr &expr, const Sizes &sizes) {
    if (!expr || expr->is_leaf()) return 0.0;

    // The cost of this node's own contraction. touched collects every index
    // that appears in the node's children, including the ones summed here,
    // and extent multiplies their sizes. A node with one child only reorders
    // a tensor's indices (e.g. f(a,i) into the left-hand side's order) and
    // counts as 0.
    double flops = expr->args.size() > 1 ? extent(touched(*expr), sizes) : 0.0;

    // Add the cost of producing each child, recursively.
    for (const ExprPtr &arg : expr->args) flops += cost(arg, sizes);

    return flops;
}

// what is the most expensive single contraction in this subtree, and how
// does it scale? The answer is symbolic, a count of indices per space such
// as {o:2, v:4} (printed o2v4). So unlike cost, it doesn't depend on sizes.
// Takes any node, like cost. Called on the root, it covers the whole term.
std::map<char, int> scaling(const ExprPtr &expr) {
    std::map<char, int> worst;

    // a pure number (no tensors) or a leaf: no contraction
    if (!expr || expr->is_leaf()) return worst;

    // compare scaling (s) by total rank, then by virtual count
    auto key = [](const std::map<char, int> &s) {
        int total = 0;
        for (const auto &[space, n] : s) total += n;
        auto v = s.find('v');
        return std::make_pair(total, v == s.end() ? 0 : v->second);
    };

    // This node's own contraction: count the indices it touches, by
    // space. For X(k,i) t2(a,b,j,k) that's a,b,i,j,k, giving {o:3, v:2}.
    // As in cost, a node with one child only reorders indices and counts nothing.
    if (expr->args.size() > 1)
        for (const Index &i : touched(*expr)) worst[i.space]++;

    // Recurse into each child and keep whichever is worse. In the
    // example, child X is o2v1, which is smaller than o3v2, so the term's
    // scaling is o3v2.
    for (const ExprPtr &arg : expr->args) {
        std::map<char, int> s = scaling(arg);
        if (key(s) > key(worst)) worst = s;
    }

    return worst;
}

// unoptimized expression
ExprPtr flat_expr(const Term &term, const Indices &out) {
    if (term.tensors.empty()) return nullptr;
    auto node = std::make_shared<Expr>();
    for (const TensorRef &t : term.tensors) node->args.push_back(make_leaf(t));
    node->idx = out;
    return node;
}

// optimize an expression. Subsets of tensors are stored as bitmasks: bit
// f is set when tensor f is in the subset.
ExprPtr optimal_expr(const Term &term, const Indices &out, const Sizes &sizes, double calls) {
    const bool hoist = calls > 0;
    const size_t n = term.tensors.size();

    // n <= 2: only one way to contract; n > 16: too many subsets to search
    if (n <= 2 || n > 16) return flat_expr(term, out);

    // now, determine which tensors carry each label; externals are kept free throughout

    // owners[label] is a bitmask of the tensors that contain that label.
    std::map<std::string, unsigned> owners;
    for (size_t f = 0; f < n; f++) {
        for (const Index &i : term.tensors[f].idx) {

            // 1u << f is bit f, and |= adds it. In our example,
            // c -> 011 (in f and t1),
            // k -> 101,
            // i -> 010, and
            // a, b, j -> 100.
            owners[i.label] |= 1u << f;
        }
    }

    // the output labels
    std::set<std::string> external;
    for (const Index &i : out) external.insert(i.label);

    // the mask for the full term, ie the mask with all n bits set: 111 in our example
    const unsigned full = (1u << n) - 1;

    // hoisting (opt_level 3): the tensors that vary between calls, as a mask. a subset
    // without them is fixed, and if it is stored (see below) it is computed once, ahead
    // of time, so it costs nothing per call
    unsigned varying = 0;
    for (size_t f = 0; f < n; f++)
        if (term.tensors[f].varies) varying |= 1u << f;
    const double cap = hoist_cap(term, sizes);

    // if we contract only the tensors in the subset, which indices
    // would remain?  these are the free indices of the subset, those that
    // are shared with the rest or external
    //
    // for example, for subset f and t1 of our term f(k,c) t1(c,i) t2(a,b,j,k)
    // take subset = 011 as input
    //
    // c appears only inside the subset, so it's summed;
    // k is also in t2, so it stays open;
    // i is an output index, so it stays open.
    //
    // The result is X(k,i).
    auto free_indices_of_subset = [&](unsigned subset) {
        Indices idx;
        for (size_t f = 0; f < n; f++) {

            // check if the term's f-th tensor is in the subset (check if bit f is set)
            if (!(subset & (1u << f))) continue;

            for (const Index &i : term.tensors[f].idx) {

                // which tensors own this label?
                unsigned who = owners[i.label];

                // an index stays open if a tensor outside the subset also carries it
                // or if it's an output index. Otherwise it's summed inside the subset.
                bool open = (who & ~subset) || external.count(i.label);

                // if index is open and not already contained in idx, add it
                if (open && std::find(idx.begin(), idx.end(), i) == idx.end()) idx.push_back(i);
            }
        }
        return idx;
    };

    // the cheapest way found to build one subset; best[] is indexed by the subset's mask
    struct Best {
        // cheapest flops to build this subset (when hoisting: per call, plus the flops of the
        // stored parts spread over the calls)
        double flops = std::numeric_limits<double>::infinity();

        // when hoisting, the flops of the fixed parts computed once, ahead of time
        double once = 0.0;

        // largest intermediate along the way, to break ties
        double peak = 0.0;

        // which part of the split won (a bit mask)
        unsigned left = 0;

        // this subset's free indices (the same whichever split wins)
        Indices idx;
    };
    std::vector<Best> best(full + 1);

    // base case: table entries for single tensors, from which every other entry is built
    for (size_t f = 0; f < n; f++) {
        Best &b = best[1u << f];     // single tensors, 001, 010, 100
        b.flops = 0.0;               // free to build a tensor that already exists
        b.idx = term.tensors[f].idx; // the free indices are the single tensor's own indices
    }

    // let's calculate costs! smallest subsets first, starting from 001
    for (unsigned s = 1; s <= full; s++) {

        // bit magic to skip single tensors, which the base case already filled in
        //
        // [100] - [001] = [011]
        // [100] & [011] = [000]
        // !0 = true ... skip!
        //
        // or
        //
        // [001] - [001] = [000]
        // [001] & [000] = [000]
        // !0 = true ... skip!
        if (!(s & (s - 1))) continue;

        // this subset's result
        Best &b = best[s];

        // free indices
        b.idx = s == full ? out : free_indices_of_subset(s);

        // number of elements in this subset's result: the product of the sizes of its free indices
        double size = extent(b.idx, sizes);

        // bit magic to isolate the lowest tensor in the subset
        //
        // -s = ~s+1 ... the bits above the lowest 1 are inverted, while the lowest 1 and
        // the 0s below it are unchanged, so s & -s keeps only that bit
        // e.g., -[110] = ~[110] + [001] = [001] + [001] = [010]
        unsigned low = s & -s;

        // split subset into a and its complement (a, s-a), where a contains the lowest tensor
        // so each is seen once

        // this loop visits every possible non-empty subset of s, except s itself.
        // as an example, let s = 101, low = 001. it is obvious the two subsets
        // are [100] and [001].
        //
        // 1. initialization:
        //
        // a = (s - 1) & s
        // - s-1 turns the lowest set bit in s to 0, and flips all bits to its right to 1s
        // - bitwise and with s then masks out any 1 bits that were not originally present in s
        //
        // a = [100] & [101] -> a = [100]
        //
        // however, a [100] does not contain low [001], so the continue statement is triggered
        // and the rest of the loop is skipped
        //
        // 2. next iteration
        //
        // a = (a - 1) & s
        // a = [011] & [101] = [001]
        //
        // a [001] contains low [001], so we get past the continue
        //
        // the complement of a, c is generated, c = s ^ a = [101] ^ [001] = [100]
        //
        // then, we check costs of a and c ... these are both evaluated already as
        // base cases
        //
        // 3. next iteration
        //
        // a = (a - 1) & s
        // a = [000] & [101] = [000] ... a is zero at the loop check, so loop terminates
        for (unsigned a = (s - 1) & s; a; a = (a - 1) & s) {

            // if a does not contain the lowest tensor, continue. We skip
            // them because each split would otherwise be counted twice. A
            // split divides s into two parts, a and its complement in s,
            // c = s ^ a

            if (!(a & low)) continue;

            // generate c: the complement of a in s
            unsigned c = s ^ a;

            // "Best" results for a and its complement
            const Best &ba = best[a], &bc = best[c];

            // a and c are smaller than s, so they were solved on an earlier pass of the
            // outer loop (for s = 101 they are the base cases [001] and [100]).
            // the total cost of building the current subset from a and c is:
            // build part a, build part c, then contract them. The last step touches
            // every index of either part. For that step merge(...) gives their union
            // and extent(merge(...)) gives the flops.
            //
            // when hoisting, a fixed part of a subset that varies is computed once and stored,
            // if no intermediate of its tree is larger than cap: its flops count once, not per
            // call. (a fixed subset itself is costed as usual; whether it is stored is up to
            // the subset that uses it.) a single tensor costs nothing either way
            auto stored = [&](unsigned p, const Best &bp) {
                return hoist && (s & varying) && !(p & varying) && (p & (p - 1)) && bp.peak <= cap;
            };
            // a stored part is used only if that pays off over `calls` calls: its flops, spread
            // over the calls, must be fewer than computing it on every call
            bool stored_a = stored(a, ba), stored_c = stored(c, bc);
            double flops = (stored_a ? ba.flops / calls : ba.flops) + (stored_c ? bc.flops / calls : bc.flops)
                         + extent(merge(ba.idx, bc.idx), sizes);
            double once = (stored_a ? ba.flops : ba.once) + (stored_c ? bc.flops : bc.once);

            // peak memory for storing an intermediate along this route,
            // the largest inside either part, or this subset's own result
            // The full set's result is the left-hand side, not an
            // intermediate, so it counts as 0 there.
            double peak = std::max({ba.peak, bc.peak, s == full ? 0.0 : size});

            // fewest flops (per call, when hoisting), then fewest flops once, then least memory
            bool fewer = flops < b.flops * (1.0 - 1e-12);
            bool same = !fewer && flops <= b.flops * (1.0 + 1e-12);
            bool fewer_once = once < b.once * (1.0 - 1e-12);
            bool same_once = !fewer_once && once <= b.once * (1.0 + 1e-12);
            bool better = fewer || (same && (fewer_once || (same_once && peak < b.peak)));

            // record flops along this path if best so far
            if (better) {
                b.flops = flops;
                b.once = once;
                b.peak = peak;
                b.left = a;
            }
        }
    }

    // rebuild the tree

    // A recursive lambda. [&] lets it read best and term from the enclosing
    // function. s is the subset to build, and it returns that subset's tree.
    // self is the lambda itself, passed in as an argument: inside its own
    // definition the lambda can't refer to the name build, because that
    // variable isn't initialized yet. Passing itself as a parameter is the
    // standard C++17 workaround.

    auto build = [&](auto &&self, unsigned s) -> ExprPtr {

        // The base case: s holds a single tensor. The while loop finds which
        // bit is set, which is the tensor's position. For s = 100:
        //
        // f = 0: 100 & 001 = 0, keep going;
        // f = 1: 100 & 010 = 0, keep going;
        // f = 2: 100 & 100 != 0, stop.
        //
        // So it returns a leaf for term.tensors[2], which is t2.
        if (!(s & (s - 1))) {
            size_t f = 0;
            while (!(s & (1u << f))) f++;
            return make_leaf(term.tensors[f]);
        }

        // Otherwise, make an internal node with two children: the winning
        // split's part a (best[s].left) and its complement s ^ a, each built
        // by a recursive call. self(self, ...) passes the lambda along so
        // the recursion can continue.
        auto node = std::make_shared<Expr>();
        node->args = {self(self, best[s].left), self(self, s ^ best[s].left)};

        // The node's free indices, computed in the DP. For the full set
        // that's out, the left-hand side's indices in their order, which is
        // what makes the root match the target.
        node->idx = best[s].idx;
        return node;
    };

    // example:
    // build(111)                        left = 011, right = 111 ^ 011 = 100
    // ├── build(011)                    left = 001, right = 011 ^ 001 = 010
    // │   ├── build(001) → leaf f(k,c)
    // │   └── build(010) → leaf t1(c,i)
    // │   idx = k,i                     (best[011].idx)
    // └── build(100) → leaf t2(a,b,j,k)
    // idx = a,b,i,j                     (best[111].idx = out)
    return build(build, full);
}

} // namespace pdaggerq::opt

namespace pdaggerq::opt {

// the largest tensor of the term that does not vary between calls; no hoisted
// intermediate may be larger, so hoisting never stores more than the inputs already do
double hoist_cap(const Term &term, const Sizes &sizes) {
    double cap = 0.0;
    for (const TensorRef &t : term.tensors)
        if (!t.varies) cap = std::max(cap, extent(t.idx, sizes));
    return cap;
}

bool is_fixed(const ExprPtr &expr) {
    if (!expr) return false;
    if (expr->is_leaf()) return !expr->leaf->varies;
    for (const ExprPtr &arg : expr->args)
        if (!is_fixed(arg)) return false;
    return true;
}

// the subtrees optimal_expr (with hoist) chose to compute once: the fixed internal
// children of varying nodes whose intermediates are all no larger than cap (exactly
// the parts optimal_expr counted as stored)
std::vector<ExprPtr> hoisted_subtrees(const ExprPtr &root, double cap, const Sizes &sizes) {
    std::vector<ExprPtr> found;

    // the largest intermediate in a subtree (its root included)
    auto peak = [&](auto &&self, const ExprPtr &e) -> double {
        if (e->is_leaf()) return 0.0;
        double p = extent(e->idx, sizes);
        for (const ExprPtr &arg : e->args) p = std::max(p, self(self, arg));
        return p;
    };
    auto visit = [&](auto &&self, const ExprPtr &e) -> void {
        if (!e || e->is_leaf()) return;
        for (const ExprPtr &arg : e->args) {
            if (!is_fixed(arg)) self(self, arg);
            else if (!arg->is_leaf() && peak(peak, arg) <= cap) found.push_back(arg);
        }
    };
    if (!is_fixed(root)) visit(visit, root);
    return found;
}

} // namespace pdaggerq::opt
