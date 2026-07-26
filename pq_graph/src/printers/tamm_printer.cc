#include "../../include/printers/tamm_printer.h"
#include "../../include/term.h"
#include "../../../pdaggerq/pq_string.h"

using std::string;
using std::vector;
using std::set;

namespace pdaggerq {

// ── TammPrinter implementations ───────────────────────────────────────────────

string TammPrinter::allocate(const string& name) const {
    return ".allocate(" + name + ")";
}

string TammPrinter::deallocate(const string& name) const {
    return ".deallocate(" + name + ")";
}

string TammPrinter::format_lines(const line_vector& lines) const {
    if (lines.empty()) return ""; // if rank is 0, return empty string
    if (lines.size() == 1) {
        // do not print sigma lines if use_trial_index is false for otherwise scalar vertices
        if (lines[0].sig_ && !Vertex::use_trial_index)
            return "";
    }

    // loop over lines
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
    line_str += ")";
    return line_str;
}

string TammPrinter::format_contraction(
    const vertex_vector& operators,
    const line_vector&   /*output_lines*/) const
{
    string output;
    vector<string> tensor_strs;

    for (const auto& op : operators) {
        if (op->empty()) continue;
        string s = op->str();
        if (op->is_addition() && !op->is_temp())
            s = "(" + s + ")";

        if (op->is_printed_scalar()) {
            output += s + " * ";
        } else {
            tensor_strs.push_back(std::move(s));
        }
    }

    if (tensor_strs.empty()) {
        if (output.empty()) return "1.0";
        output.pop_back(); output.pop_back(); output.pop_back(); // remove trailing " * "
        return output;
    }

    for (size_t i = 0; i < tensor_strs.size(); i++) {
        output += tensor_strs[i];
        if (i < tensor_strs.size() - 1)
            output += " * ";
    }
    return output;
}

string TammPrinter::format_term(const Term& t) const {
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

    // Remove trailing semicolon
    if (output.back() == ';')
        output.pop_back();

    return "( " + output + " )";
}

} // namespace pdaggerq
