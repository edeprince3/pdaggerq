#pragma once
#include "code_printer.h"

namespace pdaggerq {

class EinsumPrinter final : public CodePrinter {
public:
    static const EinsumPrinter& instance() {
        static EinsumPrinter inst;
        return inst;
    }

    string comment_prefix()       const override { return "#"; }
    string banner_h1()            const override { return "####################"; }
    string banner_h2()            const override { return "#####"; }
    string decl_comment()         const override { return "## initialize -> "; }
    bool   include_line_indices() const override { return false; }
    string condition_closer()     const override { return ""; }

    string allocate(const string& name)    const override { return ""; }
    string deallocate(const string& name)  const override;
    string perm_delete(const string& name) const override;
    string condition_open(const set<string>& conds) const override;

    string format_intermediate_name(const Linkage* link, bool include_lines) const override;

    string format_lines(const line_vector& lines) const override { return ""; }

    string format_contraction(
        const vertex_vector& operators,
        const line_vector&   output_lines) const override;

    string format_addition(
        const VertexPtr& left,
        const VertexPtr& right) const override;

    string binarize_term(const Term& t) const override { return ""; }

    string format_term(const Term& t) const override;

private:
    EinsumPrinter() = default;
};

} // namespace pdaggerq
