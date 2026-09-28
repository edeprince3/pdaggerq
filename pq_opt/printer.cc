//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: printer.cc
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

#include "printer.h"
#include "passes.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <stdexcept>

namespace pdaggerq::opt {

std::vector<std::string> Printer::lines(const Program &program) const {
    std::vector<std::string> out;
    const std::string &indent = options_.indent;
    for (const Stmt &stmt : program.stmts) {
        out.emplace_back("");
        if (options_.comments) {
            if (!stmt.comment.empty()) out.push_back(indent + comment_prefix() + " " + stmt.comment);
            std::string scale = scaling_string(stmt);
            if (!scale.empty()) out.push_back(indent + comment_prefix() + " flops: " + scale);
        }
        for (const std::string &line : statement(stmt)) out.push_back(indent + line);
        if (options_.deallocate)
            for (const TensorRef &t : stmt.free_after) out.push_back(indent + free_statement(t));
    }
    return out;
}

std::unique_ptr<Printer> make_printer(const std::string &type, const PrintOptions &options) {
    std::string t = type;
    std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) { return std::tolower(c); });

    if (t == "python" || t == "einsum" || t == "numpy") return make_numpy_printer(options);
    if (t == "c++" || t == "cpp" || t == "tiledarray" || t == "ta") return make_tiledarray_printer(options);
    throw std::invalid_argument("pq_opt: unknown print type '" + type + "' (use 'python' or 'c++')");
}

std::string format_number(double x) {
    // the shortest text that reads back as exactly x (e.g. 0.3333333333333333 for 1/3)
    char buf[32];
    std::to_chars_result result = std::to_chars(buf, buf + sizeof buf, x);
    std::string s(buf, result.ptr);

    // "1" -> "1.0", so the number reads as a float; exponents ("1e-05"), inf, and nan are left alone
    if (s.find_first_of(".eEn") == std::string::npos) s += ".0";
    return s;
}

Indices permuted(const Indices &idx, const PermTerm &perm) {
    Indices out = idx;
    for (Index &i : out) {
        for (const auto &[p, q] : perm.swaps) {
            if (i.label == p) i.label = q;
            else if (i.label == q) i.label = p;
        }
    }
    return out;
}

std::string scaling_string(const Stmt &stmt) {
    std::string s;
    for (const auto &[space, n] : scaling(stmt.rhs)) s += space + std::to_string(n);
    return s;
}

} // namespace pdaggerq::opt
