#include "../../include/printers/tiledarray_printer.h"
#include "../../include/term.h"
#include "../../../pdaggerq/pq_string.h"
#include "../../include/line.hpp"

using std::string;
using std::vector;
using std::set;

namespace pdaggerq {

// ── TiledArrayPrinter implementations ───────────────────────────────────────────────

string TiledArrayPrinter::deallocate(const string& name) const {
    return name + ".~TArrayD();";
}

string TiledArrayPrinter::perm_delete(const string& name) const {
    return name + ".~TArrayD();\n";
}

string TiledArrayPrinter::condition_open(const set<string>& conds) const {
    string s = "if (";
    for (const auto& c : conds)
        s += "includes_[\"" + c + "\"] && ";
    s.resize(s.size() - 4);
    s += ") {";
    return "\n    " + s;
}

string TiledArrayPrinter::format_lines(const line_vector& lines) const {
    if (lines.empty()) return ""; // if rank is 0, return empty string
    if (lines.size() == 1) {
        // do not print sigma lines if use_trial_index is false for otherwise scalar vertices
        if (lines[0].sig_ && !Vertex::use_trial_index)
            return "";
    }

    // loop over lines
    // string line_str = "(\"";
    string line_str = "(";
    for (const Line &line : lines) {
        if (!Vertex::use_trial_index && line.sig_) continue;
        line_str += line.label_;
        if (line.has_blk()) {
            line_str += line.block();
        }
        line_str += ",";
    }
    line_str.pop_back(); // remove last comma
    // line_str += "\")";
    line_str += ")";
    return line_str;
}

string TiledArrayPrinter::format_contraction(
    const vector<string>&      scalar_strs,
    const vector<TensorEntry>& tensor_entries,
    const string& output_labels,
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

    if (output_labels.empty() && !tensor_entries.empty()) {
        output = "dot(" + output + ")"; // TODO: replace last " * " with ", "
        
    }


    return output;
}

string TiledArrayPrinter::format_addition(
    const string& left_str, const string& right_str,
    const string& /*left_labels*/,  const string& /*right_labels*/,
    const string& /*left_types*/,   const string& /*right_types*/) const
{
    return left_str + " + " + right_str;
}

string TiledArrayPrinter::format_term(const Term& t) const {
    // Get lhs vertex string
    string output = t.lhs()->str();

    // Get sign of coefficient
    bool is_negative = t.coefficient_ < 0;
    if (t.is_assignment_) output += "  = ";
    else if (is_negative) output += " -= ";
    else output += " += ";

    // Get absolute value of coefficient
    double abs_coeff = std::fabs(t.coefficient_);

    // Check if we need to include the coefficient
    auto term_link = t.term_linkage();
    bool is_empty = t.rhs().empty() || term_link->empty();
    bool negative_assignment = (t.is_assignment_ && is_negative);
    bool needs_coeff = std::fabs(abs_coeff - 1.0) >= 1e-8 || is_empty || negative_assignment;

    if (needs_coeff) {
        if (negative_assignment)
            output += "-";

        int precision = minimum_precision(abs_coeff);
        output += to_string_with_precision(abs_coeff, precision);

        if (!is_empty)
            output += " * ";
    }

    output += term_link->str();

    // Add trailing semicolon if missing
    if (output.back() != ';')
        output += ';';

    return output;
}

} // namespace pdaggerq
