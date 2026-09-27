//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: ir.h
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

// The intermediate representation (IR) shared by every pq_opt pass and printer.
//
//   pq_string --ingest--> Equation/Term --build_program--> Program --printer--> code
//
// Invariant on a Term (enforced by ingest). A fermion label appears either once
// among the tensors (an external index, which must also be on the lhs) or exactly
// twice, in two different tensors (a summed index). A boson (cavity-mode) label may
// appear in any number of tensors: a free mode label (e.g. "t" from b(t)) is always
// external and a canonical dummy ("T".."Z") is always summed, so for bosons the kind
// of label, not the count, decides. Traces are rewritten as contractions with an
// identity (Id) tensor, so no tensor repeats a label.

#ifndef PQ_OPT_IR_H
#define PQ_OPT_IR_H

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace pdaggerq::opt {

// one tensor index
struct Index {
    std::string label;  // "a", "i", "o_3", "ni" (nuclear), "R" (trial vector), "t" (boson mode)
    char space = 'o';   // 'o'/'v' electron occupied/virtual, 'O'/'V' nuclear occupied/virtual,
                        // 'L' trial vector, 'b' boson mode
    char block = '\0';  // 'a'/'b' (spin), '1'/'0' (active/external range), '\0' unblocked

    bool operator==(const Index &o) const { return label == o.label; }
    bool operator!=(const Index &o) const { return label != o.label; }
};

using Indices = std::vector<Index>;

// a tensor with a particular labelling, e.g. eri["oovv"](i,j,a,b)
struct TensorRef {
    std::string name; // "t2", "eri", "f", "Id", "rt2", or an intermediate's name
    std::string key;  // map key used by printers ("oovv", "abab", "aa_ov"), "" for none
    Indices idx;

    // groups of positions in idx whose permutations only change the sign, e.g. {{0,1},{2,3}}
    // for <p,q||r,s> or t2(a,b,i,j). set by ingest only where pdaggerq's own algebra assumes
    // it: eri, and amplitudes with has_permutational_symmetry (not user-defined operators,
    // boson indices, or other integrals). empty when unknown, which is always safe
    std::vector<std::vector<size_t>> antisymmetric;

    size_t rank() const { return idx.size(); }
};

// one term in the expansion of a permutation operator that pdaggerq factors out of
// a term (P(i,j), PP2, PP3, PP6): sign times the term with these label swaps
// applied in order. e.g. P(i,j) = 1 - (i j) becomes two PermTerms: {+1, {}} and
// {-1, {(i,j)}}.
struct PermTerm {
    int sign = 1;
    std::vector<std::pair<std::string, std::string>> swaps; // empty = identity
};

// one term of an equation, as delivered by pdaggerq
struct Term {
    double coeff = 1.0;
    std::vector<TensorRef> tensors;  // may be empty (a pure number)
    std::vector<PermTerm> perms;     // empty = no permutation operator
    std::string comment;             // the pdaggerq string, for printing
};

// a whole equation: its left-hand side plus all its terms, as pdaggerq gives them
struct Equation {
    TensorRef lhs;
    std::vector<Term> terms;
};

// a contraction tree. a leaf holds a tensor; an internal node contracts its
// args (two after ordering, possibly more before) into the indices idx.
// exactly one of leaf and args is set. evaluation is bottom-up: the root's value
// is the entire right-hand side, so its idx are the lhs indices, in lhs order.
struct Expr {
    std::shared_ptr<const TensorRef> leaf;             // non-null for a leaf
    std::vector<std::shared_ptr<const Expr>> args;     // children of an internal node
    Indices idx;                                       // free indices of this node

    bool is_leaf() const { return leaf != nullptr; }
};
using ExprPtr = std::shared_ptr<const Expr>;

// one statement of generated code (for now, one term of an equation), printed as
// one or more lines:  target (=|+=) coeff * sum_p sign_p * P_p[ rhs ]
struct Stmt {
    TensorRef target;
    bool assign = false;           // '=' (first write to target) rather than '+='
    double coeff = 1.0;
    ExprPtr rhs;                   // null for a pure number
    std::vector<PermTerm> perms;   // empty = no permutation operator
    std::string comment;
    std::vector<TensorRef> free_after; // intermediates this statement is the last to read
};

// the ordered list of statements for every equation
struct Program {
    std::vector<Stmt> stmts;
};

// numeric extent of each index space, used by the cost model
using Sizes = std::map<char, double>;

} // namespace pdaggerq::opt

#endif
