#include <sstream>
#include <iomanip>
#include <cmath>
#include <set>
#include <map>

#include "../../include/printers/loop_printer.h"
#include "../../include/vertex.h"
#include "../../include/term.h"
#include "../../include/linkage.h"
#include "../../../pdaggerq/pq_string.h"

using std::string;
using std::stringstream;
using std::vector;
using std::set;
using std::map;

namespace pdaggerq {

string LoopPrinter::dim_name(char type) const {
    switch (type) {
        case 'o': return "nocc";
        case 'v': return "nvirt";
        case 'a': return "nalpha";
        case 'b': return "nbeta";
        case 'L': return "nsigma";
        case 'Q': return "nden";
        default:  return "";
    }
}

string LoopPrinter::format_name(const Vertex* v) const {
    if (v->rank() == 0) return v->base_name();
    string result = v->base_name();
    string ds = v->dimstring();
    if (!ds.empty()) result += "_" + ds;
    char vt = v->vertex_type();
    if (vt != 'v' && vt != 'a' && vt != '\0')
        result += "_" + string(1, vt);
    return result;
}

string LoopPrinter::format_intermediate_name(const Linkage* link, bool) const {
    string name;
    if (link->is_scalar())
        name = scratch_prefix('s');
    else if (link->is_reused())
        name = scratch_prefix('r');
    else
        name = scratch_prefix();

    string ds = link->dimstring();
    if (link->id() >= 0) {
        stringstream ss;
        ss << std::setfill('0') << std::setw(4) << link->id();
        name += "_" + ss.str();
    }
    if (!ds.empty())
        name += "_" + ds;

    return name;
}

string LoopPrinter::allocate(const string& name) const {
    return name + " = (double*)calloc(" + name + "_size, sizeof(double));";
}

string LoopPrinter::deallocate(const string& name) const {
    return "free(" + name + "); " + name + " = NULL;";
}

string LoopPrinter::format_contraction(
    const vertex_vector& operators,
    const line_vector&   /*output_lines*/) const
{
    if (operators.empty()) return "";
    string result;
    for (const auto& op : operators) {
        if (op->empty()) continue;
        if (!result.empty()) result += " * ";
        result += op->str();
    }
    return result;
}

string LoopPrinter::format_lines(const line_vector& lines) const {
    if (lines.empty()) return "";
    string out;
    for (const Line &line : lines) {
        // trial (sigma) indices are not a materialized tensor dimension unless requested
        if (line.sig_ && !Vertex::use_trial_index) continue;
        if (!out.empty()) out += ",";
        if (!line.label_.empty())
            out += line.label_;
        else
            out += (line.type() == 'v' ? 'a' : line.type());
    }
    if (out.empty()) return "";
    return "(" + out + ")";
}

string LoopPrinter::format_declarations(const set<string>& names) const {
    string out;
    out += "/* Dimension variables expected:\n";
    out += " *   int nocc;   // number of occupied (o) orbitals\n";
    out += " *   int nvirt;  // number of virtual (v) orbitals\n";
    out += " *   int nalpha; // number of alpha (a) orbitals\n";
    out += " *   int nbeta;  // number of beta (b) orbitals\n";
    out += " *   int nsigma; // number of sigma (L) trial vectors\n";
    out += " *   int nden;   // number of density (Q) vectors\n";
    out += " */\n";
    out += "#include <stdlib.h>\n\n";
    for (const auto& name : names)
        out += "double *" + name + ";\n";
    out += "\n/* Intermediate size variables (calloc argument):\n";
    out += " *   work__*_size = nocc * nvirt * ...\n";
    out += " */\n";
    return out;
}

string LoopPrinter::format_term(const Term& t) const {
    const auto& rhs = t.rhs();
    const VertexPtr& C = t.lhs();

    // Scalar: no loops needed
    if (rhs.empty()) {
        double abs_coeff = std::fabs(t.coefficient_);
        int prec = minimum_precision(abs_coeff);
        string out = C->str();
        out += t.is_assignment_ ? " = " : " += ";
        if (t.is_assignment_ && t.coefficient_ < 0) out += "-";
        out += to_string_with_precision(abs_coeff, prec);
        out += ";";
        return out;
    }

    // Collect all unique index labels from LHS and RHS operands
    struct IndexInfo {
        char label;
        char type;
        string dim;
        bool in_C, in_A, in_B;
    };
    vector<IndexInfo> indices;
    auto find_or_add = [&](char label, char type) -> size_t {
        for (size_t i = 0; i < indices.size(); ++i)
            if (indices[i].label == label) return i;
        string dn = dim_name(type);
        indices.push_back({label, type, dn.empty() ? string(1, type) : dn,
                           false, false, false});
        return indices.size() - 1;
    };

    for (const auto& l : C->lines()) {
        if (!l.label_.empty()) {
            size_t idx = find_or_add(l.label_[0], l.type());
            indices[idx].in_C = true;
        }
    }

    size_t n_rhs = std::min(rhs.size(), size_t(2));
    const VertexPtr& A = (n_rhs >= 1) ? rhs[0] : nullptr;
    const VertexPtr& B = (n_rhs >= 2) ? rhs[1] : nullptr;

    if (A) {
        for (const auto& l : A->lines()) {
            if (!l.label_.empty()) {
                size_t idx = find_or_add(l.label_[0], l.type());
                indices[idx].in_A = true;
            }
        }
    }
    if (B) {
        for (const auto& l : B->lines()) {
            if (!l.label_.empty()) {
                size_t idx = find_or_add(l.label_[0], l.type());
                indices[idx].in_B = true;
            }
        }
    }

    // Separate into free (appear in C) and contracted (only in RHS)
    vector<size_t> free_idx, contr_idx;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i].in_C)
            free_idx.push_back(i);
        else
            contr_idx.push_back(i);
    }

    // Build column-major offset expression for a vertex given its lines
    auto make_offset = [&](const VertexPtr& V) -> string {
        string off;
        bool has_stride = false;
        string stride_expr;
        for (size_t p = 0; p < V->lines().size(); ++p) {
            const Line& l = V->lines()[p];
            if (l.label_.empty()) continue;
            IndexInfo* info = nullptr;
            for (auto& idx : indices) {
                if (idx.label == l.label_[0]) {
                    info = &idx;
                    break;
                }
            }
            if (!info) continue;
            string var = string(1, info->label);
            string dim = info->dim;
            if (!off.empty()) off += " + ";
            if (has_stride) {
                off += var + " * " + stride_expr;
            } else {
                off += var;
            }
            if (p + 1 < V->lines().size()) {
                if (has_stride) {
                    stride_expr += " * " + dim;
                } else {
                    stride_expr = dim;
                    has_stride = true;
                }
            }
        }
        return off.empty() ? "0" : off;
    };

    // Build comment describing the operation
    string loop_vars;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (!loop_vars.empty()) loop_vars += ", ";
        loop_vars += string(1, indices[i].label);
    }

    string comment = "// loops: " + C->str();
    comment += t.is_assignment_ ? " = " : " += ";
    double abs_coeff = std::fabs(t.coefficient_);
    bool is_neg = t.coefficient_ < 0;
    if (std::fabs(abs_coeff - 1.0) > 1e-12 || (t.is_assignment_ && is_neg)) {
        if (t.is_assignment_ && is_neg) comment += "-";
        int prec = minimum_precision(abs_coeff);
        comment += to_string_with_precision(abs_coeff, prec);
        comment += " * ";
    } else if (t.is_assignment_ && std::fabs(abs_coeff - 1.0) < 1e-12) {
        // show "1.0 * " for clarity on assignments
    }
    if (A) comment += A->str();
    if (B) comment += " * " + B->str();
    if (!loop_vars.empty()) comment += "  over (" + loop_vars + ")";

    // Generate loop nest: free indices outermost, contracted innermost
    string loops;
    string indent;

    vector<size_t> all_idx = free_idx;
    all_idx.insert(all_idx.end(), contr_idx.begin(), contr_idx.end());

    for (size_t li = 0; li < all_idx.size(); ++li) {
        const auto& idx = indices[all_idx[li]];
        loops += indent + "for (int " + string(1, idx.label) + " = 0; "
                 + string(1, idx.label) + " < " + idx.dim + "; ++"
                 + string(1, idx.label) + ") {\n";
        indent += "    ";
    }

    // Coefficient string for body
    string coeff_str;
    if (std::fabs(abs_coeff - 1.0) < 1e-12) {
        coeff_str = is_neg ? "-" : "";
    } else {
        int prec = minimum_precision(abs_coeff);
        coeff_str = to_string_with_precision(abs_coeff, prec);
        if (is_neg) coeff_str = "-" + coeff_str;
        coeff_str += " * ";
    }

    // Loop body: C[off] += coeff * A[off] [* B[off]]
    string C_off = C->str() + "[" + make_offset(C) + "]";
    string A_off = A ? A->str() + "[" + make_offset(A) + "]" : "";
    string B_off = B ? B->str() + "[" + make_offset(B) + "]" : "";

    loops += indent + C_off + " += " + coeff_str + A_off;
    if (B) loops += " * " + B_off;
    loops += ";\n";

    // Close loops
    for (size_t li = 0; li < all_idx.size(); ++li) {
        indent.erase(indent.size() - 4);
        loops += indent + "}\n";
    }

    return comment + "\n" + loops;
}

} // namespace pdaggerq
