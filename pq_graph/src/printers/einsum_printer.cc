#include "../../include/printers/einsum_printer.h"
#include "../../include/term.h"
#include "../../../pdaggerq/pq_string.h"

using std::string;
using std::vector;
using std::set;

namespace pdaggerq {

// ── EinsumPrinter implementations ─────────────────────────────────────────────

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

    // Determine if we need to permute the rhs tensor via einsum to match the lhs labels
    bool requires_einsum = left_labels != right_labels;
    if (requires_einsum)
        output += "einsum('" + right_labels + "->" + left_labels + "',";

    output += right_str;

    if (requires_einsum) 
        output += ')';
    

    return output;
}

string EinsumPrinter::format_term(const Term& t) const {
    // Get left hand side vertex name
    string output;
    if (t.lhs()->is_linked())
        output = as_link(t.lhs())->str(true, false);
    else
        output = t.lhs()->name();

    // Get sign of coefficient
    bool is_negative = t.coefficient_ < 0;
    if (t.is_assignment_) output += "  = ";
    else if (is_negative) output += " -= ";
    else output += " += ";

    // Get absolute value of coefficient
    double abs_coeff = std::fabs(t.coefficient_);

    // Check if we need to include the coefficient
    bool needs_coeff = std::fabs(abs_coeff - 1.0) >= 1e-8 || t.rhs().empty() || t.is_assignment_;

    if (needs_coeff) {
        if (t.is_assignment_ && is_negative)
            output += "-";

        int precision = minimum_precision(abs_coeff);
        output += to_string_with_precision(abs_coeff, precision);

        if (!t.rhs().empty())
            output += " * ";
    }

    // Get string of lines
    string lhs_string, rhs_string;

    // Get string of lines from lhs vertex
    for (const auto& line : t.lhs()->lines())
        if (line.sig_ && !Vertex::use_trial_index) continue;
        else lhs_string += line.label_.front();

    // Get string of lines from the term linkage
    for (const auto& line : t.term_linkage(true)->lines())
        if (line.sig_ && !Vertex::use_trial_index) continue;
        else rhs_string += line.label_.front();

    // Get einsum string from term linkage
    string einsum_string = t.term_linkage(true)->str();

    // Permute tensors if needed
    if (lhs_string != rhs_string) {
        string sorted_lhs = lhs_string, sorted_rhs = rhs_string;
        std::sort(sorted_lhs.begin(), sorted_lhs.end());
        std::sort(sorted_rhs.begin(), sorted_rhs.end());

        if (sorted_lhs == sorted_rhs) {
            // Same character set, different order: valid einsum permutation
            einsum_string = "einsum('" + rhs_string + "->" + lhs_string + "', " + einsum_string + " )";
        } else {
            // Different character sets - compare line types positionally
            const auto& lhs_lines = t.lhs()->lines();
            const auto& rhs_lines = t.term_linkage(true)->lines();

            string lhs_types, rhs_types;
            for (const auto& line : lhs_lines)
                if (!(line.sig_ && !Vertex::use_trial_index)) lhs_types += line.type();
            for (const auto& line : rhs_lines)
                if (!(line.sig_ && !Vertex::use_trial_index)) rhs_types += line.type();

            if (lhs_types != rhs_types && lhs_types.size() == rhs_types.size()) {
                // Types differ positionally - need axis permutation via np.transpose
                string perm = "np.transpose(" + einsum_string + ", (";
                vector<bool> used(rhs_types.size(), false);
                for (size_t i = 0; i < lhs_types.size(); i++) {
                    for (size_t j = 0; j < rhs_types.size(); j++) {
                        if (!used[j] && lhs_types[i] == rhs_types[j]) {
                            perm += std::to_string(j) + ",";
                            used[j] = true;
                            break;
                        }
                    }
                }
                perm.pop_back();
                perm += "))";
                einsum_string = perm;
            }
        }
    }

    output += einsum_string;

    // Formatting issue needs to replace "* 1.00 *" with "*"
    size_t pos = 0;
    while (pos != string::npos) {
        pos = output.find("* 1.00 *", pos);
        if (pos != string::npos) {
            output = output.replace(pos, 8, "*");
            pos += 1;
        }
    }

    return output;
}

} // namespace pdaggerq
