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

string TiledArrayPrinter::format_lines(const line_vector& lines) const {
    if (lines.empty()) return ""; // if rank is 0, return empty string
    if (lines.size() == 1) {
        // do not print sigma lines if use_trial_index is false for otherwise scalar vertices
        if (lines[0].sig_ && !Vertex::use_trial_index)
            return "";
    }

    // loop over lines
    string line_str = "(\"";
    for (const Line &line : lines) {
        if (!Vertex::use_trial_index && line.sig_) continue;
        line_str += line.label_;
        if (line.has_blk()) {
            line_str += line.block();
        }
        line_str += ",";
    }
    line_str.pop_back(); // remove last comma
    line_str += "\")";
    return line_str;
}

string TiledArrayPrinter::format_contraction(
    const vertex_vector& operators,
    const line_vector&   output_lines) const
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

    bool format_as_dot = output_lines.empty() && tensor_strs.size() >= 2;
    for (size_t i = 0; i < tensor_strs.size(); i++) {
        output += tensor_strs[i];
        if (i < tensor_strs.size() - 1)
            output += " * ";
        if (format_as_dot && i == tensor_strs.size() - 2) {
            output.pop_back(); output.pop_back(); output.pop_back(); // remove trailing " * "
            output += ", ";
        }
    }

    if (format_as_dot)
        output = "dot(" + output + ")";

    return output;
}

} // namespace pdaggerq
