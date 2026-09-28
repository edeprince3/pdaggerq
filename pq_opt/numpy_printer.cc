//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: numpy_printer.cc
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

// Python / numpy backend: every contraction is an einsum call.
//
//   rt2 += 0.5 * einsum('abcd,cdij->abij', eri["vvvv"], t2)
//
// A binary tree becomes nested einsums; a flat (unordered) node with more than
// two operands leaves the order to numpy (optimize='optimal').

#include "printer.h"

#include <cmath>
#include <map>
#include <stdexcept>

namespace pdaggerq::opt {

namespace {

// einsum needs one letter per index; multi-character labels get spare letters
class Letters {
  public:
    explicit Letters(const Stmt &stmt) {
        std::vector<std::string> labels;
        auto add = [&](const Indices &idx) {
            for (const Index &i : idx) labels.push_back(i.label);
        };
        add(stmt.target.idx);
        collect(stmt.rhs, add);

        const std::string pool = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        for (const std::string &l : labels)
            if (l.size() == 1 && pool.find(l[0]) != std::string::npos) map_.emplace(l, l[0]);

        size_t next = 0;
        for (const std::string &l : labels) {
            if (map_.count(l)) continue;
            while (next < pool.size() && taken(pool[next])) next++;
            if (next == pool.size()) throw std::runtime_error("pq_opt: too many indices for einsum");
            map_.emplace(l, pool[next++]);
        }
    }

    std::string operator()(const Indices &idx) const {
        std::string s;
        for (const Index &i : idx) s += map_.at(i.label);
        return s;
    }

  private:
    template <class F> static void collect(const ExprPtr &e, F &add) {
        if (!e) return;
        add(e->idx);
        for (const ExprPtr &arg : e->args) collect(arg, add);
    }

    bool taken(char c) const {
        for (const auto &[l, ch] : map_)
            if (ch == c) return true;
        return false;
    }

    std::map<std::string, char> map_;
};

std::string ref(const TensorRef &t) {
    return t.key.empty() ? t.name : t.name + "[\"" + t.key + "\"]";
}

class NumpyPrinter : public Printer {
  public:
    using Printer::Printer;

  protected:
    std::string comment_prefix() const override { return "#"; }

    std::string free_statement(const TensorRef &t) const override { return "del " + ref(t); }

    std::vector<std::string> statement(const Stmt &stmt) const override {
        std::vector<std::string> out;
        const std::string R = ref(stmt.target);

        if (!stmt.rhs) {
            out.push_back(R + (stmt.assign ? " = " : " += ") + format_number(stmt.coeff));
            return out;
        }

        Letters letters(stmt);
        std::string expr = expression(stmt.rhs, letters);

        if (stmt.perms.empty()) {
            if (stmt.assign) {
                // the multiplication also copies, so R never aliases an input
                out.push_back(R + " = " + format_number(stmt.coeff) + " * " + expr);
            } else {
                double c = std::fabs(stmt.coeff);
                std::string scaled = c == 1.0 ? expr : format_number(c) + " * " + expr;
                out.push_back(R + (stmt.coeff < 0 ? " -= " : " += ") + scaled);
            }
            return out;
        }

        // P[x] with x computed once: R += x - x.transpose(...) ...
        const std::string tmp = "perm_tmp";
        out.push_back(tmp + " = " + format_number(stmt.coeff) + " * " + expr);
        std::string dst = letters(stmt.target.idx);
        bool assign = stmt.assign;
        for (const PermTerm &p : stmt.perms) {
            std::string src = letters(permuted(stmt.target.idx, p));
            std::string term = src == dst ? tmp : "einsum('" + src + "->" + dst + "', " + tmp + ")";
            if (assign) {
                // multiply so R is a copy, not a view of the temporary
                out.push_back(R + " = " + format_number(p.sign) + " * " + term);
                assign = false;
            } else {
                out.push_back(R + (p.sign < 0 ? " -= " : " += ") + term);
            }
        }
        if (options_.deallocate) out.push_back("del " + tmp);
        return out;
    }

  private:
    // code that evaluates e with its indices in the order e->idx
    std::string expression(const ExprPtr &e, const Letters &letters) const {
        if (e->is_leaf()) return ref(*e->leaf);

        // a lone tensor, possibly transposed
        if (e->args.size() == 1 && e->args[0]->is_leaf()) {
            const TensorRef &t = *e->args[0]->leaf;
            if (letters(t.idx) == letters(e->idx)) return ref(t);
            return "einsum('" + letters(t.idx) + "->" + letters(e->idx) + "', " + ref(t) + ")";
        }

        std::string subscripts, operands;
        for (size_t a = 0; a < e->args.size(); a++) {
            subscripts += (a ? "," : "") + letters(e->args[a]->idx);
            operands += ", " + expression(e->args[a], letters);
        }
        std::string call = "einsum('" + subscripts + "->" + letters(e->idx) + "'" + operands;
        if (e->args.size() > 2) call += ", optimize='optimal'";
        return call + ")";
    }
};

} // namespace

std::unique_ptr<Printer> make_numpy_printer(const PrintOptions &options) {
    return std::make_unique<NumpyPrinter>(options);
}

} // namespace pdaggerq::opt
