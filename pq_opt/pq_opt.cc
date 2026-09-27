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
        else if (k == "permute_eri") ingest_.permute_eri = value.cast<bool>();
        else if (k == "has_symmetric_eri") ingest_.symmetric_eri = value.cast<bool>();
        else if (k == "use_trial_index") ingest_.use_trial_index = value.cast<bool>();
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
    // level 0: terms as given; level >= 1: optimal contraction order per term;
    // level >= 2: also shared intermediates, computed first. a copy of the equations is
    // rewritten, so optimizing again starts from pdaggerq's terms
    std::vector<Equation> equations = equations_;
    std::vector<Equation> program_equations;
    if (opt_level_ >= 2)
        program_equations = extract_intermediates(equations, sizes_, max_temps_, use_antisymmetry_);
    program_equations.insert(program_equations.end(), equations.begin(), equations.end());
    program_ = build_program(program_equations, opt_level_ >= 1, sizes_);
    optimized_ = true;
}

std::vector<std::string> PQOpt::to_strings(const std::string &type) {
    if (!optimized_) optimize();
    return make_printer(type, print_)->lines(program_);
}

std::string PQOpt::str(const std::string &type) {
    std::string s;
    for (const std::string &line : to_strings(type)) s += line + "\n";
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
    std::snprintf(line, sizeof line, "    %-20s %8s %14s   %s\n", "equation", "terms", "flops", "worst scaling");
    out += line;

    // one row per equation, and one for all intermediates (tmps_), in program order
    std::vector<std::string> targets;
    for (const Stmt &stmt : program_.stmts)
        if (std::find(targets.begin(), targets.end(), stmt.target.name) == targets.end())
            targets.push_back(stmt.target.name);

    std::map<std::string, int> histogram;
    double total = 0.0;
    size_t nterms = 0;
    for (const std::string &target : targets) {
        double flops = 0.0;
        size_t count = 0;
        std::map<char, int> worst;
        for (const Stmt &stmt : program_.stmts) {
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

    out += "    scaling histogram:";
    for (const auto &[s, n] : histogram) out += " " + s + ":" + std::to_string(n);
    py::print(out);
}

void PQOpt::clear() {
    equations_.clear();
    program_ = Program();
    optimized_ = false;
}

void PQOpt::export_pq_opt(py::module &m) {
    py::class_<PQOpt, std::shared_ptr<PQOpt>>(m, "pq_opt")
        .def(py::init<const py::dict &>(), py::arg("options") = py::dict())
        .def("set_options", &PQOpt::set_options)
        .def("add", &PQOpt::add, py::arg("pq"), py::arg("equation_name") = "",
             py::arg("label_order") = std::vector<std::string>())
        .def("optimize", &PQOpt::optimize)
        .def("print", [](PQOpt &self, const std::string &type) { py::print(self.str(type)); },
             py::arg("print_type") = "python")
        .def("str", &PQOpt::str, py::arg("print_type") = "python")
        .def("__str__", [](PQOpt &self) { return self.str("python"); })
        .def("to_strings", &PQOpt::to_strings, py::arg("print_type") = "python")
        .def("analysis", &PQOpt::analysis)
        .def("clear", &PQOpt::clear)
        // pq_graph compatibility: these steps are part of optimize() here
        .def("assemble", [](PQOpt &) {})
        .def("write_dot", [](PQOpt &, const std::string &) {
            std::cout << "WARNING: pq_opt does not write DOT files" << std::endl;
        });
}

} // namespace pdaggerq::opt
