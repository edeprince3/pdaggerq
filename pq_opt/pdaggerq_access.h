//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pdaggerq_access.h
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

// Every read of pdaggerq's data structures that differs between pdaggerq releases.
//
// ingest.cc is the only part of pq_opt that reads pq_string and its tensors, and it
// does so through these functions wherever the releases differ. This is the version
// for releases with a single, unlabeled boson (cavity) mode: tensors carry fermion
// labels only (in "labels"), and w0 is a flag on the string. The version for
// releases with labeled boson indices differs only in this file.

#ifndef PQ_OPT_PDAGGERQ_ACCESS_H
#define PQ_OPT_PDAGGERQ_ACCESS_H

#include "../pdaggerq/pq_string.h"
#include "../pdaggerq/pq_tensor.h"

#include <string>
#include <vector>

namespace pdaggerq::opt {

/// a tensor's fermion labels
inline const std::vector<std::string> &fermion_labels(const tensor &t) { return t.labels; }

/// the spin ("a"/"b") of a tensor's i-th fermion label, when spin blocked
inline const std::string &spin_label(const tensor &t, size_t i) { return t.spin_labels.at(i); }

/// the range ("act"/"ext") of a tensor's i-th fermion label, when range blocked
inline const std::string &range_label(const tensor &t, size_t i) { return t.label_ranges.at(i); }

/// a tensor's boson (cavity-mode) labels: none, with a single unlabeled mode
inline const std::vector<std::string> &boson_labels(const tensor &) {
    static const std::vector<std::string> none;
    return none;
}

/// the number of an amplitude's fermion labels that belong to creation / annihilation operators
inline int creation_count(const amplitudes &amp) { return amp.n_create; }
inline int annihilation_count(const amplitudes &amp) { return amp.n_annihilate; }

/// does the string still hold operators that were not contracted away?
inline bool has_uncontracted_operators(const pq_string &s) {
    return !s.symbol.empty() || !s.is_boson_dagger.empty();
}

/// is the cavity frequency w0 a scalar factor of the string?
inline bool has_w0_factor(const pq_string &s) { return s.has_w0; }

/// is a boson label a summed dummy? (there are no boson labels here)
inline bool is_summed_boson_label(const std::string &) { return false; }

/// the labels pdaggerq uses for summed boson dummies: none here
inline const std::vector<std::string> &summed_boson_labels() {
    static const std::vector<std::string> none;
    return none;
}

} // namespace pdaggerq::opt

#endif
