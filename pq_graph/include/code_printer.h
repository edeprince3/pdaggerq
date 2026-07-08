//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: code_printer.h
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

#ifndef PDAGGERQ_CODE_PRINTER_H
#define PDAGGERQ_CODE_PRINTER_H

#include <algorithm>
#include <string>
#include <vector>
#include <set>

using std::string;
using std::vector;
using std::set;

namespace pdaggerq {

class Term;

// Per-operand data extracted by Linkage::tot_str() before invoking the printer.
// Only index_labels and index_types are used by EinsumPrinter; TammPrinter
// uses only str (which already contains line indices in TAMM mode).
struct TensorEntry {
    string str;           // fully printed operand: "t2(a,b,i,j)" or "t2"
    string index_labels;  // first char of each non-sigma line label: "abij"
    string index_types;   // Line::type() per non-sigma line: "vvoo"
};

// Abstract syntax backend — one instance per output language.
// All methods are stateless; concrete classes are singletons.
class CodePrinter {
public:
    virtual ~CodePrinter() = default;

    // ── Syntax constants ──────────────────────────────────────────────────

    virtual string comment_prefix()       const = 0; // "//" or "#"
    virtual string banner_h1()            const = 0; // "///////////////////" or "####################"
    virtual string banner_h2()            const = 0; // "/////" or "#####"
    virtual string decl_comment()         const = 0; // "// initialize -> " or "## initialize -> "
    virtual bool   include_line_indices() const = 0; // append "(i,j,...)" to vertex names?
    virtual string condition_closer()     const = 0; // "}" or ""

    // ── Statement generators ──────────────────────────────────────────────

    virtual string allocate(const string& name)             const = 0;
    virtual string deallocate(const string& name)           const = 0;
    virtual string perm_delete(const string& name)          const = 0;
    virtual string condition_open(const set<string>& conds) const = 0;

    // ── Expression formatters ─────────────────────────────────────────────

    virtual string format_contraction(
        const vector<string>&      scalar_strs,
        const vector<TensorEntry>& tensor_entries,
        const string& output_labels,
        const string& output_types) const = 0;

    virtual string format_addition(
        const string& left_str,    const string& right_str,
        const string& left_labels, const string& right_labels,
        const string& left_types,  const string& right_types) const = 0;

    // Format a complete term into a target language specific syntax.
    virtual string format_term(const Term& t) const = 0;
};
// The concrete printer implementations are now defined in separate headers
// (tamm_printer.h and einsum_printer.h). They provide the actual formatting logic
// for TAMM C++ code and Python einsum expressions, respectively.

} // namespace pdaggerq

#endif // PDAGGERQ_CODE_PRINTER_H
