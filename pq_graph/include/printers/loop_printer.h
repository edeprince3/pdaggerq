#pragma once
#include "code_printer.h"

namespace pdaggerq {

class LoopPrinter final : public CodePrinter {
public:
    static const LoopPrinter& instance() {
        static LoopPrinter inst;
        return inst;
    }

    string comment_prefix()       const override { return "//"; }
    string banner_h1()            const override { return "///////////////////"; }
    string banner_h2()            const override { return "/////"; }
    string decl_comment()         const override { return "/* initialize -> "; }
    bool   include_line_indices() const override { return false; }
    string condition_closer()     const override { return ""; }

    string allocate(const string& name)             const override;
    string deallocate(const string& name)           const override;
    string perm_delete(const string& name)          const override { return deallocate(name); }
    string condition_open(const set<string>& conds) const override { return ""; }

    string format_name(const Vertex* v)              const override;
    string format_intermediate_name(const Linkage* link, bool) const override;
    string format_lines(const line_vector& lines)     const override;

    string format_contraction(
        const vertex_vector& operators,
        const line_vector&   output_lines) const override;

    string format_addition(
        const VertexPtr& left,
        const VertexPtr& right) const override { return left->str() + " + " + right->str(); }

    string format_declarations(const set<string>& names) const override;
    string format_term(const Term& t) const override;

    string dim_name(char type) const override;

    string scratch_prefix(char type = 't') const override { return "work_"; }

private:
    LoopPrinter() = default;
};

} // namespace pdaggerq
