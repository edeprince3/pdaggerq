//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: printer.h
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

// Printers turn a Program into source code. A backend only has to say how one
// statement is written; the base class handles comments and indentation.
//
// To add a backend: derive from Printer, implement comment_prefix() and
// statement(), and register it in make_printer() (printer.cc).

#ifndef PQ_OPT_PRINTER_H
#define PQ_OPT_PRINTER_H

#include "ir.h"

#include <memory>
#include <string>
#include <vector>

namespace pdaggerq::opt {

struct PrintOptions {
    bool comments = true;     // echo each pdaggerq term above its code
    bool deallocate = true;   // free temporaries after their last use
    std::string indent = "    ";
};

class Printer {
  public:
    explicit Printer(PrintOptions options) : options_(std::move(options)) {}
    virtual ~Printer() = default;

    /// the program as lines of source code
    std::vector<std::string> lines(const Program &program) const;

  protected:
    /// "#" or "//"
    virtual std::string comment_prefix() const = 0;

    /// the code for one statement, one entry per line, without indentation
    virtual std::vector<std::string> statement(const Stmt &stmt) const = 0;

    /// the code that frees an intermediate after its last use
    virtual std::string free_statement(const TensorRef &intermediate) const = 0;

    PrintOptions options_;
};

/**
 * the printer for a language/library
 * @param type "python" (numpy einsum) or "c++" (TiledArray)
 */
std::unique_ptr<Printer> make_printer(const std::string &type, const PrintOptions &options);

/// helpers shared by the backends

/// the shortest text that reads back as exactly this double, always with a decimal point:
/// 1.0, 0.25, 0.3333333333333333
std::string format_number(double x);

/// the labels of a target after applying a permutation's swaps
Indices permuted(const Indices &idx, const PermTerm &perm);

/// "o3v2"-style scaling of the costliest contraction in a statement
std::string scaling_string(const Stmt &stmt);

// backends
std::unique_ptr<Printer> make_numpy_printer(const PrintOptions &options);
std::unique_ptr<Printer> make_tiledarray_printer(const PrintOptions &options);

} // namespace pdaggerq::opt

#endif
