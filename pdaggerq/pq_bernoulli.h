//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_utils.h
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

#ifndef PQ_BERNOULLI_H
#define PQ_BERNOULLI_H

#include "pq_string.h"

namespace pdaggerq {

/**
 *
 * strip operator portions off of an operator and return them as a string (for bernoulli expansion)
 *
 * @param op: an operator, e.g., "t2", "v", "V{R}", etc.
 * @return ret: a string specifying the operator portions
 */
std::string get_operator_portions_as_string(const std::string& op);

/**
 *
 * strip operator portions off of an operator and return them as a vector (for bernoulli expansion)
 *
 * @param op: an operator, e.g., "t2", "v", "V{R}", etc.
 * @return ret: a vector of strings specifying the operator portions
 */
std::vector<std::string> get_operator_portions_as_vector(const std::string& op);

/**
 *
 * strip operator portions off of an operator and return the base operator name (for bernoulli expansion)
 *
 * @param op: an operator, e.g., "t2", "v", "V{R}", etc.
 * @return ret: a string specifying the operator base name
 *
 */
std::string get_operator_base_name(std::string op);

/// eliminate terms based on operator portions (for bernoulli)
void eliminate_operator_portions(std::shared_ptr<pq_string> &in, int bernoulli_excitation_level);

/// determine the operator type for the part of an input string corresponding to a target portion (for bernoulli)
std::string bernoulli_type(std::shared_ptr<pq_string> &in, std::string target_portion, size_t portion_number, int bernoulli_excitation_level);

}

#endif 
