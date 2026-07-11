//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: code_printer.cc
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

#include <cmath>
#include <algorithm>
#include <string>
#include <vector>
#include <set>
#include <memory>
#include <sstream>
#include <iomanip>

#include "../../include/vertex.h"
#include "../../include/term.h"
#include "../../include/printers/code_printer.h"
#include "../../include/printers/tamm_printer.h"
#include "../../include/printers/einsum_printer.h"
#include "../../include/printers/tiledarray_printer.h"
#include "../../include/printers/blas_printer.h"

using std::string;
using std::stringstream;
using std::vector;
using std::set;
using std::to_string;
using std::make_shared;

namespace pdaggerq {

// ── Vertex static member definitions ─────────────────────────────────────────

// Initialize to TAMM (C++) backend by default — must be set before any print call
const CodePrinter* Vertex::printer_ = &TiledArrayPrinter::instance();

void Vertex::set_printer(const string& type) {
    string t = type;
    for (auto& c : t) // convert to lowercase
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');

    // set the printer based on the type
    if (t == "python" || t == "einsum") {
        printer_ = &EinsumPrinter::instance();
        std::cout << "Setting printer to Einsum (Python) format" << std::endl;
    } else if (t == "tiledarray" || t == "c++" || t == "cpp") {
        printer_ = &TiledArrayPrinter::instance();
        std::cout << "Setting printer to TiledArray (C++) format" << std::endl;
    } else if (t == "tamm") {
        printer_ = &TammPrinter::instance();
        CodePrinter::binarize_ = true; // enable binarization for TAMM printer
        std::cout << "Setting printer to TAMM (C++) format" << std::endl;
    } else if (t == "blas" || t == "cblas") {
        printer_ = &BLASPrinter::instance();
        CodePrinter::binarize_ = true;
        std::cout << "Setting printer to BLAS (C) format" << std::endl;
    } else {
        std::cout << "Unknown printer type: " << type << std::endl;
    }
    std::cout << std::endl;
}

// ── Default virtual implementations ──────────────────────────────────────────

string CodePrinter::scratch_prefix(char type) const {
    switch (type) {
        case 's': return "scalars_";
        case 'r': return "reused_";
        default:  return "tmps_";
    }
}

string CodePrinter::binarize_term(const Term& t) const {
    if (!binarize_) return "";

    Term binarized_term = t.clone();
    bool needs_binarization = binarized_term.size() > 2;
    bool made_any_change = false;
    string output;
    int count = 1;

    auto make_interm = [&](const vector<VertexPtr> &verts, size_t erase_pos, size_t erase_count, size_t insert_pos) {
        MutableVertexPtr interm_vertex;
        if (verts.size() == 2)
            interm_vertex = make_shared<Vertex>(scratch_prefix(), (verts[0] * verts[1])->lines());
        else
            interm_vertex = make_shared<Vertex>(scratch_prefix(), verts[0]->lines());

        interm_vertex->vertex_type_ = (char)count + '0';
        interm_vertex->sort();
        interm_vertex->update_name();

        Term interm_term = binarized_term;
        interm_term.reset_perm();
        interm_term.coefficient_ = 1.0;
        interm_term.comments() = {};
        interm_term.is_assignment_ = true;

        interm_term.lhs() = interm_vertex;
        interm_term.rhs() = verts;
        interm_term.compute_scaling(true);

        output += interm_term.str();
        output += "\n";

        for (size_t e = 0; e < erase_count; ++e)
            binarized_term.rhs().erase(binarized_term.rhs().begin() + (int)erase_pos);
        binarized_term.rhs().insert(binarized_term.rhs().begin() + (int)insert_pos, interm_vertex);
        binarized_term.compute_scaling(true);

        made_any_change = true;
        ++count;
    };

    do {
        size_t n = binarized_term.size();
        needs_binarization = n > 2;

        if (needs_binarization) {
            VertexPtr &left = binarized_term[0], &right = binarized_term[1];

            VertexPtr &left_end = binarized_term[n - 2];
            VertexPtr &right_end = binarized_term[n - 1];

            bool first_smaller = (left*right)->shape_ <= (left_end*right_end)->shape_;

            if (first_smaller)
                make_interm({left, right}, 0, 2, 0);
            else
                make_interm({left_end, right_end}, n - 2, 2, n - 2);

        } else if (binarized_term.size() == 2) {
            VertexPtr &left = binarized_term[0], &right = binarized_term[1];
            bool left_is_add  = left->is_expandable(false, true);
            bool right_is_add = right->is_expandable(false, true);

            if (left_is_add)
                make_interm({left}, 0, 1, 0);

            if (right_is_add)
                make_interm({right}, 1, 1, 1);
        }
    } while (needs_binarization);

    if (made_any_change) {
        output += binarized_term.str();
        return output;
    }

    return "";
}

string CodePrinter::condition_open(const set<string>& conds) const {
    string s = "if (";
    for (const auto& c : conds)
        s += "includes_[\"" + c + "\"] && ";
    s.resize(s.size() - 4);
    s += ") {";
    return "\n    " + s;
}

string CodePrinter::format_name(const Vertex* v) const {
        // scalars have no dimension
        if (v->rank() == 0) return v->base_name();

        // format tensor block as a map if it is not an amplitude or if it has a block
        switch (v->vertex_type()) {
            case 'v':
                return v->base_name() + "[\"" + v->dimstring() + "\"]";
            case 'a':
                if (v->has_blks()) {
                    return v->base_name() + "[\"" + v->blk_string() + "\"]";
                }
                break;
            case 'p':
                return v->base_name() + "[\"perm_" + v->dimstring() + "\"]";
            default:
                if (v->vertex_type() != '\0')
                    return v->base_name() + "[\"bin" + v->vertex_type() + '_' + v->dimstring() + "\"]";
                break;
        }

        // default format name without any special indexing
        return v->base_name(); 
    }

string CodePrinter::format_intermediate_name(const Linkage* link, bool include_lines) const {
    string generic_str;
    if (link->is_scalar())
        generic_str = scratch_prefix('s');
    else if (link->is_reused())
        generic_str = scratch_prefix('r');
    else generic_str = scratch_prefix();
    generic_str += "[\"";

    string dimstring = link->dimstring();
    if (link->id() >= 0) {
        stringstream ss;
        ss << std::setfill('0') << std::setw(4) << link->id();
        generic_str += ss.str();
    }
    if (!dimstring.empty())
        generic_str += "_" + dimstring;
    generic_str += "\"]";

    if (include_lines && include_line_indices())
        generic_str += link->line_str();

    return generic_str;
}

string CodePrinter::format_term(const Term& t) const {
    string output = t.lhs()->str();
    bool is_negative = t.coefficient_ < 0;
    if (t.is_assignment_) output += "  = ";
    else if (is_negative) output += " -= ";
    else output += " += ";

    double abs_coeff = std::fabs(t.coefficient_);
    auto term_link = t.term_linkage();
    bool is_empty = t.rhs().empty() || term_link->empty();
    bool negative_assignment = (t.is_assignment_ && is_negative);
    bool needs_coeff = std::fabs(abs_coeff - 1.0) >= 1e-8 || is_empty || negative_assignment;

    if (needs_coeff) {
        if (negative_assignment) output += "-";
        int precision = minimum_precision(abs_coeff);
        output += to_string_with_precision(abs_coeff, precision);
        if (!is_empty) output += " * ";
    }

    output += term_link->str();

    if (output.back() != ';')
        output += ';';

    return output;
}

string CodePrinter::format_declarations(const set<string>& names) const {
    string out;
    for (const auto& name : names)
        out += decl_comment() + name + ";\n";
    return out;
}

string CodePrinter::format_named_section(const string& name, bool major) const {
    const string& bar = major ? banner_h1() : banner_h2();
    return bar + name + bar + "\n\n";
}

string CodePrinter::format_closing_banner() const {
    return banner_h1() + banner_h1() + banner_h1() + "\n\n";
}

string CodePrinter::padding(int level) const {
    return string(static_cast<size_t>(level) * 4, ' ');
}

string CodePrinter::format_comment(const string& raw_comment, int indent) const {
    if (raw_comment.empty()) return "";

    string comment = raw_comment;
    string pad = padding(indent);
    string extra_pad = padding(indent + 1);

    comment.insert(0, extra_pad);

    // remove quotes
    size_t pos = 0;
    while ((pos = comment.find('\"', pos)) != string::npos) {
        comment = comment.replace(pos, 1, "");
        pos += 1;
    }

    // replace newlines
    pos = 0;
    while ((pos = comment.find('\n', pos)) != string::npos) {
        comment = comment.replace(pos, 1, '\n' + pad);
        pos += 1;
    }

    return "\n" + comment;
}

string CodePrinter::format_term_line(const string& term_str, int indent) const {
    string output = padding(indent) + term_str;
    string extra_pad = padding(indent + 1);

    size_t pos = 0;
    while ((pos = output.find('\n', pos)) != string::npos) {
        output = output.replace(pos, 1, "\n" + extra_pad);
        pos += 1;
    }

    return output;
}

} // namespace pdaggerq
