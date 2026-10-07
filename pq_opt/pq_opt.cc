//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_opt.cc
// Copyright (C) 2020 A. Eugene DePrince III
//
// Author: A. Eugene DePrince III <adeprince@fsu.edu>
//
// This file is part of the pdaggerq package.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

#include "pq_opt.h"

#include "../pdaggerq/pq_helper.h"

#include <pybind11/stl.h>

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <set>

namespace py = pybind11;

namespace pdaggerq::opt {

PQOpt::PQOpt(const py::dict &options) { set_options(options); }

void PQOpt::set_options(const py::dict &options) {

    // pq_graph options that have no meaning here; accepted so pq_graph scripts run unchanged
    static const std::set<std::string> ignored = {
        "batched", "batch_size", "max_depth", "low_memory", "nthreads",
        "expand_permutations", "separate_sigma", "reindex_temps", "binarize",
        "cache_elements", "cache_depth", "max_shape", "dims"};

    // pq_graph options that would change the equations; pq_opt refuses them when turned on
    // rather than ignore them. (no_scalars/no_trace dropped terms containing a trace in
    // pq_graph; pq_opt keeps every term.)
    static const std::set<std::string> refused = {"decompose_eri", "density_fitting", "no_scalars", "no_trace"};

    for (const auto &[key, value] : options) {
        std::string k = py::str(key);
        if (k == "opt_level") opt_level_ = value.cast<int>();
        else if (k == "print_level") print_level_ = value.cast<int>();
        else if (k == "max_temps") max_temps_ = value.cast<long>();
        else if (k == "use_antisymmetry") use_antisymmetry_ = value.cast<bool>();
        else if (k == "calls") {
            calls_ = value.cast<double>();
            if (!(calls_ >= 1.0)) throw std::invalid_argument("pq_opt: calls must be at least 1");
        }
        else if (k == "permute_eri") ingest_.permute_eri = value.cast<bool>();
        else if (k == "has_symmetric_eri") ingest_.symmetric_eri = value.cast<bool>();
        else if (k == "use_trial_index") ingest_.use_trial_index = value.cast<bool>();
        else if (k == "varying") {
            auto names = value.cast<std::vector<std::string>>();
            ingest_.varying = std::set<std::string>(names.begin(), names.end());
        }
        else if (k == "print_comments") print_.comments = value.cast<bool>();
        else if (k == "deallocate") print_.deallocate = value.cast<bool>();
        else if (k == "nocc") { if (value.cast<int>() > 0) sizes_['o'] = value.cast<double>(); }
        else if (k == "nvirt") { if (value.cast<int>() > 0) sizes_['v'] = value.cast<double>(); }
        else if (k == "sizes") {
            for (const auto &[space, n] : value.cast<py::dict>()) {
                std::string s = py::str(space);
                if (s.size() != 1) throw std::invalid_argument("pq_opt: sizes keys are single characters ('o', 'v', 'O', 'V', 'L', 'b')");
                sizes_[s[0]] = n.cast<double>();
            }
        }
        else if (ignored.count(k)) continue;
        else if (refused.count(k)) {
            bool on = py::isinstance<py::bool_>(value) && value.cast<bool>();
            if (on) throw std::invalid_argument("pq_opt: option '" + k + "' is not supported");
        }
        else std::cout << "WARNING: pq_opt: unknown option '" << k << "' ignored" << std::endl;
    }
    optimized_ = false;
}

void PQOpt::add(const pq_helper &pq, const std::string &name, const std::vector<std::string> &label_order) {
    std::string lhs = name.empty() ? "eq_" + std::to_string(equations_.size()) : name;
    Equation eq = ingest(pq, lhs, label_order, ingest_);
    if (eq.terms.empty()) return;

    // the lhs indices in order, each as label, space, and block
    auto lhs_indices = [](const Equation &e) {
        std::vector<std::string> idx;
        for (const Index &i : e.lhs.idx) idx.push_back(i.label + ' ' + i.space + ' ' + i.block);
        return idx;
    };
    auto lhs_string = [](const Equation &e) {
        std::string s;
        for (const Index &i : e.lhs.idx) s += (s.empty() ? "" : ",") + i.label;
        return e.lhs.name + "(" + s + ")";
    };

    // adding to an existing equation appends its terms; its lhs indices and their order must match
    for (Equation &old : equations_) {
        if (old.lhs.name != eq.lhs.name) continue;
        if (lhs_indices(old) != lhs_indices(eq))
            throw std::invalid_argument("pq_opt: equation " + lhs_string(old) + " already exists; "
                                        "cannot add terms for " + lhs_string(eq));
        std::cout << "WARNING: equation '" << name << "' already exists. "
                     "The terms will be merged with the existing equation." << std::endl;
        old.terms.insert(old.terms.end(), eq.terms.begin(), eq.terms.end());
        optimized_ = false;
        return;
    }
    equations_.push_back(std::move(eq));
    optimized_ = false;
}

void PQOpt::optimize() {
    // level 4 builds blocks only when that makes the code cheaper: both versions are built, and
    // the one with fewer flops per call (plus its once-only flops spread over the calls) is kept
    auto [program, reused_program] = build(false);
    if (opt_level_ >= 4) {
        auto [block_program, block_reused_program] = build(true);
        auto flops = [&](const Program &p) {
            double total = 0.0;
            for (const Stmt &stmt : p.stmts) total += cost(stmt.rhs, sizes_);
            return total;
        };
        double without = flops(program) + flops(reused_program) / calls_;
        double with = flops(block_program) + flops(block_reused_program) / calls_;
        if (with < without * (1.0 - 1e-12)) {
            program = std::move(block_program);
            reused_program = std::move(block_reused_program);
        }
    }
    program_ = std::move(program);
    reused_program_ = std::move(reused_program);
    optimized_ = true;
}

std::pair<Program, Program> PQOpt::build(bool blocks) const {
    // level 0: terms as given; level >= 1: optimal contraction order per term;
    // level >= 2: also shared intermediates, computed first; level >= 3: first of all, what
    // does not change between calls (reused_), in a program of its own; level >= 4: terms
    // that differ only in a fixed tensor merged, reading the sum of those tensors, then terms
    // that differ only in what multiplies one tensor, reading the sum of those parts; with
    // blocks, before all of that, terms that are one varying tensor times fixed ones read
    // blocks of their fixed parts. a copy of the equations is rewritten, so optimizing again
    // starts from pdaggerq's terms
    std::vector<Equation> equations = equations_;
    std::vector<Equation> reused;
    if (blocks) reused = jacobian_blocks(equations, sizes_, calls_, use_antisymmetry_);
    if (opt_level_ >= 3) {
        std::vector<Equation> hoisted = hoist_invariants(equations, sizes_, calls_, use_antisymmetry_);
        reused.insert(reused.end(), hoisted.begin(), hoisted.end());
    }
    if (opt_level_ >= 4) {
        std::vector<Equation> sums = merge_terms(equations, reused.size() + 1, use_antisymmetry_);
        reused.insert(reused.end(), sums.begin(), sums.end());
        std::vector<Equation> per_call_sums = merge_by_anchor(equations, sizes_, use_antisymmetry_);
        equations.insert(equations.begin(), per_call_sums.begin(), per_call_sums.end());
    }
    std::vector<Equation> program_equations;
    if (opt_level_ >= 2)
        program_equations = extract_intermediates(equations, sizes_, max_temps_, use_antisymmetry_);
    program_equations.insert(program_equations.end(), equations.begin(), equations.end());
    Program program = build_program(program_equations, opt_level_ >= 1, sizes_);

    // the reused_ intermediates the per-call code does not read (e.g. the pieces of a merged
    // sum) are freed once they have been used
    std::set<std::string> read_per_call;
    for (const Equation &eq : program_equations)
        for (const Term &term : eq.terms)
            for (const TensorRef &t : term.tensors)
                if (t.name == "reused_") read_per_call.insert(t.key);
    std::set<std::string> free_reused;
    for (const Equation &eq : reused)
        if (!read_per_call.count(eq.lhs.key)) free_reused.insert(eq.lhs.key);

    // the once-only program shares intermediates too; its tmps_ keys start with "r"
    std::vector<Equation> reused_equations;
    if (opt_level_ >= 2 && !reused.empty())
        reused_equations = extract_intermediates(reused, sizes_, max_temps_, use_antisymmetry_, "r");
    reused_equations.insert(reused_equations.end(), reused.begin(), reused.end());
    return {program, build_program(reused_equations, true, sizes_, free_reused)};
}

std::vector<std::string> PQOpt::to_strings(const std::string &type, const std::string &part) {
    if (!optimized_) optimize();
    auto printer = make_printer(type, print_);
    if (part == "reused") return printer->lines(reused_program_);
    if (part == "per_call") return printer->lines(program_);
    if (part != "all")
        throw std::invalid_argument("pq_opt: part must be 'all', 'reused', or 'per_call', not '" + part + "'");
    std::vector<std::string> lines = printer->lines(reused_program_);
    for (const std::string &line : printer->lines(program_)) lines.push_back(line);
    return lines;
}

std::string PQOpt::str(const std::string &type, const std::string &part) {
    std::string s;
    for (const std::string &line : to_strings(type, part)) s += line + "\n";
    return s;
}

void PQOpt::analysis() const {
    if (!optimized_) {
        py::print("pq_opt: call optimize() before analysis()");
        return;
    }

    // order scalings by total rank, then by the number of virtual indices
    auto rank = [](const std::map<char, int> &s) {
        int total = 0;
        for (const auto &[space, n] : s) total += n;
        auto v = s.find('v');
        return std::make_pair(total, v == s.end() ? 0 : v->second);
    };
    auto name = [](const std::map<char, int> &s) {
        std::string str;
        for (const auto &[space, n] : s) str += space + std::to_string(n);
        return str.empty() ? std::string("none") : str;
    };

    std::string size_str;
    for (const auto &[space, n] : sizes_) size_str += " " + std::string(1, space) + "=" + format_number(n);

    char line[256];
    std::string out = "pq_opt analysis (sizes:" + size_str + ")\n";

    std::map<std::string, int> histogram;

    // one row per target (equations, tmps_, reused_), in program order, and a total
    auto section = [&](const Program &program, const std::string &title) {
        std::snprintf(line, sizeof line, "    %-20s %8s %14s   %s\n", title.c_str(), "terms", "flops", "worst scaling");
        out += line;
        std::vector<std::string> targets;
        for (const Stmt &stmt : program.stmts)
            if (std::find(targets.begin(), targets.end(), stmt.target.name) == targets.end())
                targets.push_back(stmt.target.name);

        double total = 0.0;
        size_t nterms = 0;
        for (const std::string &target : targets) {
            double flops = 0.0;
            size_t count = 0;
            std::map<char, int> worst;
            for (const Stmt &stmt : program.stmts) {
                if (stmt.target.name != target) continue;
                // a permuted term is contracted once; its permuted copies are additions, not counted
                flops += cost(stmt.rhs, sizes_);
                count++;
                std::map<char, int> s = scaling(stmt.rhs);
                histogram[name(s)]++;
                if (rank(s) > rank(worst)) worst = s;
            }
            std::snprintf(line, sizeof line, "    %-20s %8zu %14.4e   %s\n", target.c_str(), count, flops,
                          name(worst).c_str());
            out += line;
            total += flops;
            nterms += count;
        }
        std::snprintf(line, sizeof line, "    %-20s %8zu %14.4e\n", "total", nterms, total);
        out += line;
    };

    // with hoisting (opt_level 3), what runs once and what runs on every call
    if (!reused_program_.stmts.empty()) {
        section(reused_program_, "once (reused_)");
        double stored = 0.0;
        std::set<std::string> seen;
        for (const Stmt &stmt : reused_program_.stmts)
            if (seen.insert(stmt.target.key).second) stored += extent(stmt.target.idx, sizes_);
        std::snprintf(line, sizeof line, "    %-20s %8s %14.4e\n", "stored elements", "", stored);
        out += line;
        section(program_, "per call");
    } else {
        section(program_, "equation");
    }

    out += "    scaling histogram:";
    for (const auto &[s, n] : histogram) out += " " + s + ":" + std::to_string(n);
    py::print(out);
}

std::map<std::string, double> PQOpt::costs() {
    if (!optimized_) optimize();
    auto flops = [&](const Program &program) {
        double total = 0.0;
        for (const Stmt &stmt : program.stmts) total += cost(stmt.rhs, sizes_);
        return total;
    };
    double stored = 0.0;
    std::set<std::string> seen;
    for (const Stmt &stmt : reused_program_.stmts)
        if (seen.insert(stmt.target.key).second) stored += extent(stmt.target.idx, sizes_);
    return {{"once", flops(reused_program_)}, {"per_call", flops(program_)}, {"stored", stored}};
}

void PQOpt::clear() {
    equations_.clear();
    program_ = Program();
    reused_program_ = Program();
    optimized_ = false;
}

void PQOpt::export_pq_opt(py::module &m) {
    py::class_<PQOpt, std::shared_ptr<PQOpt>>(m, "pq_opt")
        .def(py::init<const py::dict &>(), py::arg("options") = py::dict())
        .def("set_options", &PQOpt::set_options)
        .def("add", &PQOpt::add, py::arg("pq"), py::arg("equation_name") = "",
             py::arg("label_order") = std::vector<std::string>())
        .def("optimize", &PQOpt::optimize)
        .def("print", [](PQOpt &self, const std::string &type, const std::string &part) {
                 py::print(self.str(type, part)); }, py::arg("print_type") = "python", py::arg("part") = "all")
        .def("str", &PQOpt::str, py::arg("print_type") = "python", py::arg("part") = "all")
        .def("__str__", [](PQOpt &self) { return self.str("python", "all"); })
        .def("to_strings", &PQOpt::to_strings, py::arg("print_type") = "python", py::arg("part") = "all")
        .def("analysis", &PQOpt::analysis)
        .def("costs", &PQOpt::costs)
        .def("clear", &PQOpt::clear)
        // pq_graph compatibility: these steps are part of optimize() here
        .def("assemble", [](PQOpt &) {})
        .def("write_dot", [](PQOpt &, const std::string &) {
            std::cout << "WARNING: pq_opt does not write DOT files" << std::endl;
        });
}

} // namespace pdaggerq::opt
