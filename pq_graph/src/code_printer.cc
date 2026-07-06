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

#include <algorithm>
#include <string>
#include <vector>
#include <set>

#include "../include/code_printer.h"
#include "../include/vertex.h"

using std::string;
using std::vector;
using std::set;
using std::to_string;

namespace pdaggerq {

// ── Vertex static member definitions ─────────────────────────────────────────

// Initialize to TAMM (C++) backend by default — must be set before any print call
const CodePrinter* Vertex::printer_ = &TammPrinter::instance();

void Vertex::set_printer(const string& type) {
    string t = type;
    for (auto& c : t)
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');

    if (t == "python" || t == "einsum")
        printer_ = &EinsumPrinter::instance();
    else
        printer_ = &TammPrinter::instance();
}

// ── TammPrinter ───────────────────────────────────────────────────────────────

string TammPrinter::allocate(const string& name) const {
    return ".allocate(" + name + ")";
}

string TammPrinter::deallocate(const string& name) const {
    return ".deallocate(" + name + ")";
}

string TammPrinter::condition_open(const set<string>& conds) const {
    string s = "if (";
    for (const auto& c : conds)
        s += "includes_[\"" + c + "\"] && ";
    s.resize(s.size() - 4);
    s += ") {";
    return "\n    " + s;
}

string TammPrinter::format_contraction(
    const vector<string>&      scalar_strs,
    const vector<TensorEntry>& tensor_entries,
    const string& /*output_labels*/,
    const string& /*output_types*/) const
{
    if (scalar_strs.empty() && tensor_entries.empty()) return "1.0";

    string output;
    for (const auto& s : scalar_strs)
        output += s + " * ";

    if (tensor_entries.empty()) {
        output.pop_back(); output.pop_back(); output.pop_back(); // remove trailing " * "
        return output;
    }

    for (size_t i = 0; i < tensor_entries.size(); i++) {
        output += tensor_entries[i].str;
        if (i < tensor_entries.size() - 1)
            output += " * ";
    }
    return output;
}

string TammPrinter::format_addition(
    const string& left_str, const string& right_str,
    const string& /*left_labels*/,  const string& /*right_labels*/,
    const string& /*left_types*/,   const string& /*right_types*/) const
{
    return left_str + " + " + right_str;
}

// ── EinsumPrinter ─────────────────────────────────────────────────────────────

string EinsumPrinter::deallocate(const string& name) const {
    return "del " + name;
}

string EinsumPrinter::perm_delete(const string& name) const {
    return "del " + name + "\n";
}

string EinsumPrinter::condition_open(const set<string>& conds) const {
    string s = "if ";
    for (const auto& c : conds)
        s += "includes_[\"" + c + "\"] and ";
    s.resize(s.size() - 5);
    s += ":";
    return "\n    " + s;
}

string EinsumPrinter::format_contraction(
    const vector<string>&      scalar_strs,
    const vector<TensorEntry>& tensor_entries,
    const string& output_labels,
    const string& output_types) const
{
    string output;
    for (const auto& s : scalar_strs)
        output += s + " * ";

    if (!tensor_entries.empty()) {
        bool skip_einsum = false;
        if (tensor_entries.size() == 1) {
            string sorted_input  = tensor_entries[0].index_labels;
            string sorted_output = output_labels;
            std::sort(sorted_input.begin(),  sorted_input.end());
            std::sort(sorted_output.begin(), sorted_output.end());
            if (sorted_input != sorted_output) {
                if (tensor_entries[0].index_types == output_types)
                    skip_einsum = true;
            }
            if (tensor_entries[0].index_labels == output_labels)
                skip_einsum = true;
        }

        if (skip_einsum) {
            output += tensor_entries[0].str;
        } else {
            output += "einsum('";
            for (const auto& entry : tensor_entries)
                output += entry.index_labels + ",";
            output.pop_back();
            output += "->" + output_labels + "',";
            for (const auto& entry : tensor_entries)
                output += entry.str + ",";
            if (tensor_entries.size() > 2)
                output += "optimize='optimal'";
            else
                output.pop_back();
            output += ")";
        }
    } else {
        if (output.size() >= 3)
            output.resize(output.size() - 3); // remove trailing " * "
    }
    return output;
}

string EinsumPrinter::format_addition(
    const string& left_str,    const string& right_str,
    const string& left_labels, const string& right_labels,
    const string& left_types,  const string& right_types) const
{
    string output = left_str + " + ";

    if (left_labels != right_labels) {
        string sorted_left  = left_labels, sorted_right = right_labels;
        std::sort(sorted_left.begin(),  sorted_left.end());
        std::sort(sorted_right.begin(), sorted_right.end());
        if (sorted_left == sorted_right) {
            output += "einsum('" + right_labels + "->" + left_labels + "',";
        } else if (left_types != right_types && left_types.size() == right_types.size()) {
            output += "np.transpose(";
        }
    }

    output += right_str;

    if (left_labels != right_labels) {
        string sorted_left  = left_labels, sorted_right = right_labels;
        std::sort(sorted_left.begin(),  sorted_left.end());
        std::sort(sorted_right.begin(), sorted_right.end());
        if (sorted_left == sorted_right) {
            output += ")";
        } else if (left_types != right_types && left_types.size() == right_types.size()) {
            string perm = ", (";
            vector<bool> used(right_types.size(), false);
            for (size_t i = 0; i < left_types.size(); i++) {
                for (size_t j = 0; j < right_types.size(); j++) {
                    if (!used[j] && left_types[i] == right_types[j]) {
                        perm += to_string(j) + ",";
                        used[j] = true;
                        break;
                    }
                }
            }
            perm.pop_back();
            perm += "))";
            output += perm;
        }
    }
    return output;
}

} // namespace pdaggerq
