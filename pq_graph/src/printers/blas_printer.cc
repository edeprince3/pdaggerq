#include <sstream>
#include <iomanip>
#include <cmath>
#include <set>
#include <map>

#include "../../include/printers/blas_printer.h"
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

string BLASPrinter::dim_name(char type) const {
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

string BLASPrinter::format_name(const Vertex* v) const {
    if (v->rank() == 0) return v->base_name();
    string result = v->base_name();
    string ds = v->dimstring();
    if (!ds.empty()) result += "_" + ds;
    char vt = v->vertex_type();
    if (vt != 'v' && vt != 'a' && vt != '\0')
        result += "_" + string(1, vt);
    return result;
}

string BLASPrinter::format_intermediate_name(const Linkage* link, bool) const {
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

string BLASPrinter::allocate(const string& name) const {
    return name + " = (double*)calloc(" + name + "_size, sizeof(double));";
}

string BLASPrinter::deallocate(const string& name) const {
    return "free(" + name + "); " + name + " = NULL;";
}

string BLASPrinter::format_contraction(
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

// ── Line formatting ─────────────────────────────────────────────────────

string BLASPrinter::format_lines(const line_vector& lines) const {
    if (lines.empty()) return "";
    string out = "(";
    for (size_t i = 0; i < lines.size(); ++i) {
        if (i > 0) out += ",";
        if (!lines[i].label_.empty())
            out += lines[i].label_;
        else
            out += (lines[i].type() == 'v' ? 'a' : lines[i].type());
    }
    out += ")";
    return out;
}

string BLASPrinter::format_declarations(const set<string>& names) const {
    string out;
    // Emit dimension reference comment
    out += "/* Dimension variables expected:\n";
    out += " *   int nocc;   // number of occupied (o) orbitals\n";
    out += " *   int nvirt;  // number of virtual (v) orbitals\n";
    out += " *   int nalpha; // number of alpha (a) orbitals\n";
    out += " *   int nbeta;  // number of beta (b) orbitals\n";
    out += " *   int nsigma; // number of sigma (L) trial vectors\n";
    out += " *   int nden;   // number of density (Q) vectors\n";
    out += " */\n";
    out += "#include <cblas.h>\n\n";
    for (const auto& name : names)
        out += "double *" + name + ";\n";
    // size variables for intermediates
    out += "\n/* Intermediate size variables (calloc argument):\n";
    out += " *   work__*_size = nocc * nvirt * ...\n";
    out += " */\n";
    return out;
}

// ── DGEMM generation ────────────────────────────────────────────────────

string BLASPrinter::dgemm_call(const Term& t) const {
    LinkagePtr link = as_link(t.term_linkage(true));
    const auto& ops = link->link_vector();

    if (ops.size() < 2) return "";

    // Extract tensor-only operators
    vertex_vector tensors;
    for (const auto& op : ops) {
        if (op->empty()) continue;
        if (op->rank() == 0 && op->lines().empty()) continue;
        tensors.push_back(op);
    }
    if (tensors.size() != 2) return "";

    const VertexPtr& A = tensors[0];
    const VertexPtr& B = tensors[1];

    size_t a_n = A->lines().size();
    size_t b_n = B->lines().size();

    // Build per-operand free/contracted lookup using connec_map_
    // connec_map_ entry {L, R}: both >= 0 => contracted; one -1 => free
    vector<bool> a_contr(a_n, false);
    vector<bool> b_contr(b_n, false);

    for (const auto& entry : link->connec_map_) {
        int_fast8_t left_pos  = entry[0];
        int_fast8_t right_pos = entry[1];
        if (left_pos >= 0 && right_pos >= 0) {
            a_contr[static_cast<size_t>(left_pos)] = true;
            b_contr[static_cast<size_t>(right_pos)] = true;
        }
    }

    // A's free line positions (in A's own order)
    vector<size_t> a_free_pos, a_contr_pos;
    for (size_t i = 0; i < a_n; ++i) {
        if (a_contr[i])
            a_contr_pos.push_back(i);
        else
            a_free_pos.push_back(i);
    }

    // B's free line positions (in B's own order)
    vector<size_t> b_free_pos, b_contr_pos;
    for (size_t i = 0; i < b_n; ++i) {
        if (b_contr[i])
            b_contr_pos.push_back(i);
        else
            b_free_pos.push_back(i);
    }

    // Interleaving check: in each operand, all free must come before all
    // contracted (or vice versa) for a clean DGEMM.
    auto is_clean = [](const vector<size_t>& free_pos,
                       const vector<size_t>& contr_pos) -> bool {
        if (free_pos.empty() || contr_pos.empty()) return true;
        return free_pos.back() < contr_pos.front() ||
               contr_pos.back() < free_pos.front();
    };
    if (!is_clean(a_free_pos, a_contr_pos)) return "";
    if (!is_clean(b_free_pos, b_contr_pos)) return "";

    // Determine TransA (NoTrans if free first, Trans if contr first)
    bool a_contr_first = (!a_contr_pos.empty() &&
                          (a_free_pos.empty() || a_contr_pos.front() < a_free_pos.front()));
    bool b_free_first  = (!b_free_pos.empty() &&
                          (b_contr_pos.empty() || b_free_pos.front() < b_contr_pos.front()));

    char transA = a_contr_first ? 'T' : 'N';
    char transB = b_free_first  ? 'T' : 'N';

    // Build dimension products
    auto dim_product_of_lines = [](const vector<size_t>& positions,
                                   const line_vector& lines) -> string {
        string prod;
        for (size_t pos : positions) {
            if (pos >= lines.size()) continue;
            char t = lines[pos].type();
            string dn;
            switch (t) {
                case 'o': dn = "nocc"; break;
                case 'v': dn = "nvirt"; break;
                case 'a': dn = "nalpha"; break;
                case 'b': dn = "nbeta"; break;
                default: continue;
            }
            if (dn.empty()) continue;
            if (!prod.empty()) prod += " * ";
            prod += dn;
        }
        return prod.empty() ? "1" : prod;
    };

    string M_str = dim_product_of_lines(a_free_pos, A->lines());
    string K_str = dim_product_of_lines(a_contr_pos, A->lines());
    string N_str = dim_product_of_lines(b_free_pos, B->lines());

    // Leading dimensions
    string LDA_str = (transA == 'N') ? M_str : K_str;
    string LDB_str = (transB == 'N') ? K_str : N_str;

    // Output analysis: determine which operand's free lines come first
    // in the output, to decide whether we need to swap A/B for DGEMM.
    // The output lines_ of the linkage are the free lines.
    const line_vector& out_lines = link->lines();

    // Label the output lines by which operand they came from:
    // match by line identity against A's/B's free lines.
    vector<char> out_origin(out_lines.size(), '?');
    for (size_t oi = 0; oi < out_lines.size(); ++oi) {
        for (size_t ai : a_free_pos) {
            if (ai < A->lines().size() &&
                out_lines[oi] == A->lines()[ai]) {
                out_origin[oi] = 'A';
                goto next_out;
            }
        }
        for (size_t bi : b_free_pos) {
            if (bi < B->lines().size() &&
                out_lines[oi] == B->lines()[bi]) {
                out_origin[oi] = 'B';
                goto next_out;
            }
        }
        next_out: ;
    }

    // Check interleaving in output
    bool seen_B = false, seen_A_after_B = false;
    for (char o : out_origin) {
        if (o == 'B') seen_B = true;
        else if (o == 'A' && seen_B) { seen_A_after_B = true; break; }
    }
    bool seen_A = false, seen_B_after_A = false;
    for (char o : out_origin) {
        if (o == 'A') seen_A = true;
        else if (o == 'B' && seen_A) { seen_B_after_A = true; break; }
    }

    // Decide if we need to swap operands for correct output order.
    // DGEMM computes C(M,N) where M = A_free, N = B_free.
    // If output has B before A, we swap: new A = B, new B = A.
    bool need_swap = seen_A_after_B; // B comes first in output

    // Resolve operands for DGEMM
    const VertexPtr* dA = need_swap ? &B : &A;
    const VertexPtr* dB = need_swap ? &A : &B;
    const string* dM_str = need_swap ? &N_str : &M_str;
    const string* dN_str = need_swap ? &M_str : &N_str;
    const string* dK_str = &K_str;
    char dTransA = need_swap ? transB : transA;
    char dTransB = need_swap ? transA : transB;
    string dLDA_str, dLDB_str;
    if (need_swap) {
        dLDA_str = (dTransA == 'N') ? *dM_str : *dK_str;
        dLDB_str = (dTransB == 'N') ? *dK_str : *dN_str;
    } else {
        dLDA_str = LDA_str;
        dLDB_str = LDB_str;
    }
    string dLDC_str = *dM_str;

    // Tensor names
    string A_name = (*dA)->str();
    string B_name = (*dB)->str();
    string C_name = t.lhs()->str();

    // Alpha / beta
    double abs_coeff = std::fabs(t.coefficient_);
    bool is_negative = t.coefficient_ < 0;
    string alpha_str;
    if (std::fabs(abs_coeff - 1.0) < 1e-12) {
        alpha_str = is_negative ? "-1.0" : "1.0";
    } else {
        int prec = minimum_precision(abs_coeff);
        alpha_str = to_string_with_precision(abs_coeff, prec);
        if (is_negative) alpha_str = "-" + alpha_str;
    }
    string beta_str = t.is_assignment_ ? "0.0" : "1.0";

    // Build comment
    string comment = "// ";
    comment += C_name;
    comment += (t.is_assignment_ ? " = " : " += ");
    comment += A_name + " * " + B_name;

    // Build DGEMM call
    string dgemm = "cblas_dgemm(CblasColMajor,\n";
    dgemm += "                  Cblas" + string(dTransA == 'N' ? "NoTrans" : "Trans") + ",\n";
    dgemm += "                  Cblas" + string(dTransB == 'N' ? "NoTrans" : "Trans") + ",\n";
    dgemm += "                  " + *dM_str + ", " + *dN_str + ", " + *dK_str + ",\n";
    dgemm += "                  " + alpha_str + ", " + A_name + ", " + dLDA_str + ",\n";
    dgemm += "                                " + B_name + ", " + dLDB_str + ",\n";
    dgemm += "                  " + beta_str + ", " + C_name + ", " + dLDC_str + ");";

    return comment + "\n" + dgemm;
}

string BLASPrinter::format_term(const Term& t) const {
    // Try DGEMM for binary contractions
    string dgemm = dgemm_call(t);
    if (!dgemm.empty()) return dgemm;

    // Check for single-RHS terms (common after binarization)
    const auto& rhs = t.rhs();
    if (rhs.size() == 1) {
        const VertexPtr& src = rhs[0];
        const VertexPtr& dst = t.lhs();
        double alpha = t.coefficient_;

        // Build a size expression from all lines (product of dimensions)
        auto make_size = [](const line_vector& lines) -> string {
            string prod;
            for (const auto& l : lines) {
                if (l.label_.empty()) continue;
                char t = l.type();
                string dn;
                switch (t) {
                    case 'o': dn = "nocc"; break;
                    case 'v': dn = "nvirt"; break;
                    case 'a': dn = "nalpha"; break;
                    case 'b': dn = "nbeta"; break;
                    default: continue;
                }
                if (dn.empty()) continue;
                if (!prod.empty()) prod += " * ";
                prod += dn;
            }
            return prod.empty() ? "1" : prod;
        };

        string size_str = make_size(dst->lines());

        // Determine if index ordering matches between src and dst
        bool same_order = (src->lines().size() == dst->lines().size());
        if (same_order) {
            for (size_t i = 0; i < src->lines().size(); ++i) {
                if (!(src->lines()[i] == dst->lines()[i])) {
                    same_order = false;
                    break;
                }
            }
        }

        if (same_order) {
            // Identity copy — can use daxpy / dcopy
            string alpha_str;
            if (std::fabs(std::fabs(alpha) - 1.0) < 1e-12) {
                alpha_str = (alpha < 0) ? "-1.0" : "1.0";
            } else {
                int prec = minimum_precision(std::fabs(alpha));
                alpha_str = to_string_with_precision(std::fabs(alpha), prec);
                if (alpha < 0) alpha_str = "-" + alpha_str;
            }

            string comment = "// " + dst->str();
            comment += (t.is_assignment_ ? " = " : " += ");
            comment += alpha_str + " * " + src->str();

            string blas;
            if (t.is_assignment_ && std::fabs(alpha - 1.0) < 1e-12) {
                // dcopy
                blas = "cblas_dcopy(" + size_str + ", " + src->str() + ", 1, " + dst->str() + ", 1);";
            } else {
                // daxpy
                blas = "cblas_daxpy(" + size_str + ", " + alpha_str + ", " + src->str() + ", 1, " + dst->str() + ", 1);";
            }

            return comment + "\n" + blas;
        } else {
            // Permuted copy — emit clear comment with index mapping
            string fallback = "// [perm] " + dst->str();
            fallback += (t.is_assignment_ ? " = " : " += ");
            if (std::fabs(std::fabs(alpha) - 1.0) > 1e-12) {
                int prec = minimum_precision(std::fabs(alpha));
                fallback += to_string_with_precision(std::fabs(alpha), prec);
                if (alpha < 0) fallback = fallback.substr(0, fallback.size() - 2) + "- ";
                else fallback += " * ";
            } else if (alpha < 0) {
                fallback += "-";
            }
            fallback += src->str();
            fallback += "  // dst[" + dst->line_str() + "] ";
            fallback += (t.is_assignment_ ? "=" : "+=");
            fallback += " src[" + src->line_str() + "]";
            return fallback;
        }
    }

    // Binary term where DGEMM failed — generate explicit loops
    const auto& rhs2 = t.rhs();
    if (rhs2.size() == 2) {
        auto link = as_link(t.term_linkage(true));
        const VertexPtr& A = rhs2[0];
        const VertexPtr& B = rhs2[1];
        const VertexPtr& C = t.lhs();

        // Collect ALL unique indices (both free and contracted)
        // Key: label char -> (type, dim_name)
        // We track which indices appear in which operands
        struct IndexInfo {
            char label;
            char type;
            string dim;
            bool in_A, in_B, in_C;
        };
        vector<IndexInfo> indices;
        auto find_or_add = [&](char label, char type) -> size_t {
            for (size_t i = 0; i < indices.size(); ++i)
                if (indices[i].label == label) return i;
            string dn;
            switch (type) {
                case 'o': dn = "nocc"; break;
                case 'v': dn = "nvirt"; break;
                case 'a': dn = "nalpha"; break;
                case 'b': dn = "nbeta"; break;
                default:  dn = string(1, type); break;
            }
            indices.push_back({label, type, dn, false, false, false});
            return indices.size() - 1;
        };

        for (const auto& l : A->lines()) {
            if (!l.label_.empty()) {
                size_t idx = find_or_add(l.label_[0], l.type());
                indices[idx].in_A = true;
            }
        }
        for (const auto& l : B->lines()) {
            if (!l.label_.empty()) {
                size_t idx = find_or_add(l.label_[0], l.type());
                indices[idx].in_B = true;
            }
        }
        for (const auto& l : C->lines()) {
            if (!l.label_.empty()) {
                size_t idx = find_or_add(l.label_[0], l.type());
                indices[idx].in_C = true;
            }
        }

        // Separate into free and contracted
        vector<size_t> free_idx, contr_idx;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i].in_C)
                free_idx.push_back(i);
            else
                contr_idx.push_back(i);
        }

        // Build offset expression for a vertex given its line positions
        auto make_offset = [&](const VertexPtr& V, bool add_contr) -> string {
            string off;
            bool has_stride = false;
            string stride_expr;
            for (size_t p = 0; p < V->lines().size(); ++p) {
                const Line& l = V->lines()[p];
                if (l.label_.empty()) continue;

                // Find index info
                IndexInfo* info = nullptr;
                for (auto& idx : indices) {
                    if (idx.label == l.label_[0]) {
                        info = &idx;
                        break;
                    }
                }
                if (!info) continue;

                // Skip contracted indices if !add_contr (for dim_product only)
                // For offset, we include ALL indices
                bool is_contr = !info->in_C;
                if (!add_contr && is_contr) continue;

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

        // Emit nested loops
        string loops;
        string indent = "";
        string loop_vars; // for the comment
        for (size_t i = 0; i < free_idx.size(); ++i) {
            if (!loop_vars.empty()) loop_vars += ", ";
            loop_vars += string(1, indices[free_idx[i]].label);
        }
        for (size_t i = 0; i < contr_idx.size(); ++i) {
            if (!loop_vars.empty()) loop_vars += ", ";
            loop_vars += string(1, indices[contr_idx[i]].label);
        }

        // First, all loops (contracted innermost)
        vector<size_t> all_idx = free_idx;
        all_idx.insert(all_idx.end(), contr_idx.begin(), contr_idx.end());

        for (size_t li = 0; li < all_idx.size(); ++li) {
            const auto& idx = indices[all_idx[li]];
            loops += indent + "for (int " + string(1, idx.label) + " = 0; "
                     + string(1, idx.label) + " < " + idx.dim + "; ++"
                     + string(1, idx.label) + ") {\n";
            indent += "    ";
        }

        // Loop body
        double abs_coeff = std::fabs(t.coefficient_);
        bool is_neg = t.coefficient_ < 0;
        string coeff_str;
        if (std::fabs(abs_coeff - 1.0) < 1e-12) {
            coeff_str = is_neg ? "-" : "";
        } else {
            int prec = minimum_precision(abs_coeff);
            coeff_str = to_string_with_precision(abs_coeff, prec);
            if (is_neg) coeff_str = "-" + coeff_str;
            else coeff_str = coeff_str + " * ";
        }
        string C_off = C->str() + "[" + make_offset(C, true) + "]";
        string A_off = A->str() + "[" + make_offset(A, true) + "]";
        string B_off = B->str() + "[" + make_offset(B, true) + "]";

        // Always use += in loops — output is either calloc'd (intermediates)
        // or already seeded from a previous term (accumulation).
        loops += indent + C_off + " += " + coeff_str + A_off + " * " + B_off + ";\n";

        // Close loops
        for (size_t li = 0; li < all_idx.size(); ++li) {
            indent.erase(indent.size() - 4);
            loops += indent + "}\n";
        }

        // Comment describing the loops
        string comment = "// loops: " + C->str() + " += " + A->str() + " * " + B->str();
        comment += "  over (" + loop_vars + ")";
        return comment + "\n" + loops;
    }

    // Final fallback: single-RHS case that's already handled above,
    // or residual case that shouldn't normally occur.
    return "// [fallback] " + t.lhs()->str() + " += " + t.term_linkage()->str() + ";";
}

} // namespace pdaggerq
