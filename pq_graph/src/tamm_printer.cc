#include "../include/tamm_printer.h"
#include "../include/term.h"
#include "../../pdaggerq/pq_string.h"

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
