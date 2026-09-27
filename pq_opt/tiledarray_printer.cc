//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: tiledarray_printer.cc
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

// C++ / TiledArray backend: every statement is a TiledArray tensor expression.
//
//   rt2("a,b,i,j") += 0.5 * eri["vvvv"]("a,b,c,d") * t2("c,d,i,j");
//
// TiledArray evaluates a product left to right, so a binary tree is printed with
// parentheses that force the chosen order. Scalars use dot(x, y).

#include "printer.h"

#include <cmath>
#include <stdexcept>

namespace pdaggerq::opt {

namespace {

std::string labels(const Indices &idx) {
    std::string s;
    for (size_t i = 0; i < idx.size(); i++) s += (i ? "," : "") + idx[i].label;
    return "(\"" + s + "\")";
}

// the map an intermediate lives in: tensors in tmps_, numbers (no indices) in scalars_
std::string storage(const TensorRef &t) {
    return t.name == "tmps_" && t.idx.empty() ? "scalars_" : t.name;
}

std::string ref(const TensorRef &t, const Indices &idx) {
    std::string name = t.key.empty() ? storage(t) : storage(t) + "[\"" + t.key + "\"]";
    return idx.empty() ? name : name + labels(idx);
}

class TiledArrayPrinter : public Printer {
  public:
    using Printer::Printer;

  protected:
    std::string comment_prefix() const override { return "//"; }

    std::string free_statement(const TensorRef &t) const override {
        return storage(t) + ".erase(\"" + t.key + "\");";
    }

    std::vector<std::string> statement(const Stmt &stmt) const override {
        std::vector<std::string> out;
        const TensorRef &T = stmt.target;
        std::string R = ref(T, T.idx);

        if (!stmt.rhs) {
            if (!T.idx.empty()) throw std::runtime_error("pq_opt: a constant added to tensor " + T.name);
            out.push_back(R + (stmt.assign ? " = " : " += ") + format_number(stmt.coeff) + ";");
            return out;
        }

        std::string expr = T.idx.empty() ? scalar(stmt.rhs) : expression(stmt.rhs, true);

        if (stmt.perms.empty()) {
            double c = std::fabs(stmt.coeff);
            std::string scaled = c == 1.0 ? expr : format_number(c) + " * " + expr;
            std::string op = stmt.assign ? " = " : (stmt.coeff < 0 ? " -= " : " += ");
            if (stmt.assign && stmt.coeff < 0) scaled = "-" + format_number(c) + " * " + expr;
            out.push_back(R + op + scaled + ";");
            return out;
        }

        // P[x] with x computed once, in a scope that frees it
        TensorRef tmp{"perm_tmp", "", T.idx};
        out.push_back("{");
        out.push_back(options_.indent + "TArrayD perm_tmp;");
        out.push_back(options_.indent + ref(tmp, T.idx) + " = " + format_number(stmt.coeff) + " * " + expr + ";");
        bool assign = stmt.assign;
        for (const PermTerm &p : stmt.perms) {
            std::string src = ref(tmp, permuted(T.idx, p));
            if (assign) {
                out.push_back(options_.indent + R + " = " + (p.sign < 0 ? "-1.0 * " : "") + src + ";");
                assign = false;
            } else {
                out.push_back(options_.indent + R + (p.sign < 0 ? " -= " : " += ") + src + ";");
            }
        }
        out.push_back("}");
        return out;
    }

  private:
    // a product; inner nodes are parenthesized so TiledArray follows the tree.
    // a fully contracted node (e.g. a trace) is a number, so it goes through dot()
    std::string expression(const ExprPtr &e, bool top) const {
        if (e->is_leaf()) return ref(*e->leaf, e->leaf->idx);
        if (e->idx.empty()) return scalar(e);
        std::string s;
        for (size_t a = 0; a < e->args.size(); a++) s += (a ? " * " : "") + expression(e->args[a], false);
        return top || e->args.size() == 1 ? s : "(" + s + ")";
    }

    // a full contraction: dot(everything but the last operand, the last operand),
    // or a plain product when every operand is already a number
    std::string scalar(const ExprPtr &e) const {
        // a single number, e.g. a scalar intermediate
        if (e->is_leaf()) return ref(*e->leaf, e->leaf->idx);
        if (e->args.size() == 1) return scalar(e->args[0]);

        bool numbers = true;
        for (const ExprPtr &arg : e->args) numbers &= arg->idx.empty();
        if (numbers) {
            std::string s;
            for (size_t a = 0; a < e->args.size(); a++) s += (a ? " * " : "") + expression(e->args[a], false);
            return "(" + s + ")";
        }

        std::string left;
        for (size_t a = 0; a + 1 < e->args.size(); a++) left += (a ? " * " : "") + expression(e->args[a], false);
        return "dot(" + left + ", " + expression(e->args.back(), false) + ")";
    }
};

} // namespace

std::unique_ptr<Printer> make_tiledarray_printer(const PrintOptions &options) {
    return std::make_unique<TiledArrayPrinter>(options);
}

} // namespace pdaggerq::opt
