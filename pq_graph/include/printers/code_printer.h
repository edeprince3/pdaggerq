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

#include "../term.h"

using std::string;
using std::vector;
using std::set;

namespace pdaggerq {

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

    virtual string allocate(const string& name)             const { return ""; }
    virtual string deallocate(const string& name)           const = 0;
    virtual string perm_delete(const string& name)          const = 0;
    virtual string condition_open(const set<string>& conds) const;    

    // ── Vertex formatters ─────────────────────────────────────────────

    virtual string format_name(const Vertex* vertex) const;
    virtual string format_intermediate_name(const Linkage* link, bool include_lines) const;
    virtual string format_lines(const line_vector& lines) const = 0;

    // ── Expression formatters ─────────────────────────────────────────────

    virtual string format_contraction(
        const vertex_vector& operators,
        const line_vector&   output_lines) const = 0;

    virtual string format_addition(
        const VertexPtr& left,
        const VertexPtr& right) const { return left->str() + " + " + right->str(); }

    // Format a complete term into a target language specific syntax.
    virtual string format_term(const Term& t) const;

    // ── Structural formatters ───────────────────────────────────────────

    // Emit declaration lines for a set of base names.
    virtual string format_declarations(const set<string>& names) const;

    // Emit a named section banner (major uses banner_h1, minor uses banner_h2).
    virtual string format_named_section(const string& name, bool major) const;

    // Emit the closing banner (triple h1).
    virtual string format_closing_banner() const;

    // ── Dimension names ─────────────────────────────────────────────────

    // Map index type character to symbolic dimension name (e.g., 'o' → "nocc").
    virtual string dim_name(char type) const { return ""; }

    // ── Intermediate naming ──────────────────────────────────────────────

    // Return prefix for internally-generated temporaries. type is 't' (tmps_),
    // 's' (scalars_), or 'r' (reused_).
    virtual string scratch_prefix(char type = 't') const;

    // ── Line-level formatters ────────────────────────────────────────────

    // Return indent string: level 0 → "", level 1 → "    ", level 2 → "        "
    virtual string padding(int level) const;

    // Wrap a raw comment for inline emission. Returns "" if raw_comment is empty.
    virtual string format_comment(const string& raw_comment, int indent) const;

    // Wrap a raw term string with padding and newline continuations.
    virtual string format_term_line(const string& term_str, int indent) const;
};
// The concrete printer implementations are now defined in separate headers
// (tamm_printer.h and einsum_printer.h). They provide the actual formatting logic
// for TAMM C++ code and Python einsum expressions, respectively.

} // namespace pdaggerq

#endif // PDAGGERQ_CODE_PRINTER_H
