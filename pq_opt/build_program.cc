//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: build_program.cc
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

#include <map>

namespace pdaggerq::opt {

namespace {

// the intermediates (tmps_) a tree reads
void intermediates_read(const ExprPtr &e, std::vector<TensorRef> &found) {
    if (!e) return;
    if (e->is_leaf()) {
        if (e->leaf->name == "tmps_") found.push_back(*e->leaf);
        return;
    }
    for (const ExprPtr &arg : e->args) intermediates_read(arg, found);
}

// "tmps_[0001_ov](k,i)"-style lhs text for comments; just the name for an equation
std::string lhs_text(const TensorRef &lhs) {
    if (lhs.key.empty()) return lhs.name;
    std::string s = lhs.name + "[" + lhs.key + "](";
    for (size_t i = 0; i < lhs.idx.size(); i++) s += (i ? "," : "") + lhs.idx[i].label;
    return s + ")";
}

} // namespace

Program build_program(const std::vector<Equation> &eqs, bool reorder, const Sizes &sizes) {
    Program program;
    for (const Equation &eq : eqs) {
        bool first = true;
        for (const Term &term : eq.terms) {
            Stmt stmt;
            stmt.target = eq.lhs;
            stmt.assign = first;
            stmt.coeff = term.coeff;
            stmt.rhs = reorder ? optimal_expr(term, eq.lhs.idx, sizes) : flat_expr(term, eq.lhs.idx);
            stmt.perms = term.perms;
            stmt.comment = lhs_text(eq.lhs) + (first ? "  = " : " += ") + term.comment;
            program.stmts.push_back(std::move(stmt));
            first = false;
        }
    }

    // free each intermediate after the last statement that reads it
    std::map<std::string, std::pair<size_t, TensorRef>> last_read;
    for (size_t n = 0; n < program.stmts.size(); n++) {
        std::vector<TensorRef> read;
        intermediates_read(program.stmts[n].rhs, read);
        for (const TensorRef &t : read) last_read[t.key] = {n, t};
    }
    for (const auto &[key, entry] : last_read) program.stmts[entry.first].free_after.push_back(entry.second);

    return program;
}

} // namespace pdaggerq::opt
