//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_opt.h
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

// pq_opt: optimize pdaggerq equations and print them as code.
//
// The Python class mirrors pq_graph (same options dict, add/optimize/print/str/
// analysis). Unlike pq_graph, add() requires the output index order of every
// equation, so a codegen script switches by changing the class name and passing
// label_order to each add().

#ifndef PQ_OPT_PQ_OPT_H
#define PQ_OPT_PQ_OPT_H

#include "ir.h"
#include "passes.h"
#include "printer.h"

#include <pybind11/pybind11.h>

namespace pdaggerq {
class pq_helper;
}

namespace pdaggerq::opt {

class PQOpt {
  public:
    explicit PQOpt(const pybind11::dict &options);

    void set_options(const pybind11::dict &options);

    /// queue an equation. label_order (or a name like "rt2(a,b,i,j)") gives the order of every
    /// lhs index; a trial index, if any, is always first and is not listed, and boson indices
    /// may be listed (all or none) or left to be the last axes
    void add(const pq_helper &pq, const std::string &name, const std::vector<std::string> &label_order);

    /// run the passes selected by opt_level
    void optimize();

    /// the generated code
    std::vector<std::string> to_strings(const std::string &type);
    std::string str(const std::string &type);

    /// flop counts and scaling of the optimized program
    void analysis() const;

    void clear();

    /// bind the class to python as pdaggerq.pq_opt
    static void export_pq_opt(pybind11::module &m);

  private:
    int opt_level_ = 1;
    int print_level_ = 0;
    long max_temps_ = -1;          // most intermediates to create at opt_level >= 2 (-1: no limit)
    bool use_antisymmetry_ = true; // match intermediates up to tensor antisymmetry (opt_level >= 2)
    IngestOptions ingest_;
    PrintOptions print_;
    // o/v: electron occupied/virtual, O/V: nuclear occupied/virtual, L: trial vectors, b: cavity modes
    Sizes sizes_ = {{'o', 20.0}, {'v', 100.0}, {'O', 1.0}, {'V', 20.0}, {'L', 10.0}, {'b', 1.0}};

    std::vector<Equation> equations_;
    Program program_;
    bool optimized_ = false;
};

} // namespace pdaggerq::opt

#endif
