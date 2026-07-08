#pragma once
#include "code_printer.h"

namespace pdaggerq {

class TammPrinter final : public CodePrinter {
public:
    static const TammPrinter& instance() {
        static TammPrinter inst;
        return inst;
    }

    string comment_prefix()       const override { return "//"; }
    string banner_h1()            const override { return "///////////////////"; }
    string banner_h2()            const override { return "/////"; }
    string decl_comment()         const override { return "// initialize -> "; }
    bool   include_line_indices() const override { return true; }
    string condition_closer()     const override { return "}"; }

    string allocate(const string& name)    const override;
    string deallocate(const string& name)  const override;
    string perm_delete(const string& name) const override { return ""; }
    string condition_open(const set<string>& conds) const override;

    string format_contraction(
        const vector<string>&      scalar_strs,
        const vector<TensorEntry>& tensor_entries,
        const string& output_labels,
        const string& output_types) const override;

    string format_addition(
        const string& left_str,    const string& right_str,
        const string& left_labels, const string& right_labels,
        const string& left_types,  const string& right_types) const override;

    string format_term(const Term& t) const override;

private:
    TammPrinter() = default;
};

} // namespace pdaggerq
