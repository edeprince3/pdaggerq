//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: code_printer.cc
// Copyright (C) 2020 A. Eugene DePrince III
//
// Author: A. Eugene DePrince III <adeprince@fsu.edu>
// Maintainer: DePrince group
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

#include <algorithm>
#include <string>
#include <vector>
#include <set>

#include "../../include/vertex.h"
#include "../../include/printers/code_printer.h"
#include "../../include/printers/tamm_printer.h"
#include "../../include/printers/einsum_printer.h"

using std::string;
using std::vector;
using std::set;
using std::to_string;

namespace pdaggerq {

// ── Vertex static member definitions ─────────────────────────────────────────

// Initialize to TAMM (C++) backend by default — must be set before any print call
const CodePrinter* Vertex::printer_ = &TammPrinter::instance();

void Vertex::set_printer(const string& type) {
    string t = type;
    for (auto& c : t) // convert to lowercase
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');

    // set the printer based on the type
    if (t == "python" || t == "einsum") {
        printer_ = &EinsumPrinter::instance();
        std::cout << "Setting printer to Einsum (Python) format" << std::endl;
    } else {
        printer_ = &TammPrinter::instance();
        std::cout << "Setting printer to TAMM (C++) format" << std::endl;
    }
    std::cout << std::endl;
}

} // namespace pdaggerq
