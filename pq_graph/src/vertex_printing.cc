//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: vertex_printing.cc
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

#include <cmath>
#include <map>
#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>

#include "../include/pq_graph.h"
#include "../include/term.h"


// include omp only if defined
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_set_num_threads() 1
#endif

using std::ostream, std::string, std::vector, std::map, std::unordered_map, std::shared_ptr, std::make_shared,
        std::set, std::unordered_set, std::pair, std::make_pair, std::to_string, std::invalid_argument,
        std::stringstream, std::cout, std::endl, std::flush, std::max, std::min, std::unordered_map, std::unordered_set;

namespace pdaggerq {

    void Vertex::format_name() {
        name_ = base_name_;

        // scalars have no dimension
        if (rank_ == 0) return;

        // format tensor block as a map if it is not an amplitude or if it has a block
        if (vertex_type_ == 'v') {
            name_ += "[\"";
            name_ += dimstring();
            name_ += "\"]";
        } else if (vertex_type_ == 'a' && has_blk_) {
            name_ += "[\"";
            name_ += blk_string();
            name_ += "\"]";
        } else if (vertex_type_ == 'p') {
            name_ += "[\"";
            name_ += "perm_" + dimstring();
            name_ += "\"]";
        }

    }

    string Vertex::dimstring() const {
        string dimstring;
        if (rank_ == 0) return dimstring;

        if (has_blk_) {
            string blk_string = this->blk_string();
            if (!blk_string.empty()) {
                dimstring += blk_string;
                dimstring += "_";
            }
        }
        dimstring += ovstring();

        return dimstring;
    }

    string Vertex::blk_string() const {
        if (!has_blk_ || lines_.empty()) return "";

        string blk_string;
        blk_string.reserve(blk_string.size());
        for (const Line &line : lines_) {
            char blk = line.block();
            if (blk != '\0')
                blk_string += blk;
        }

        return blk_string;
    }

    string Vertex::ovstring(const line_vector &lines) {
        if (lines.empty()) return "";
        string ovstring; // ovstring assuming all occupied
        ovstring.reserve(lines.size());
        for (const Line &line : lines) {
            ovstring.push_back(line.type());
        }

        return ovstring;
    }

    string Vertex::line_str(bool sort) const{
        if (size() == 0) return ""; // if rank is 0, return empty string
        if (size() == 1) {
            // do not print sigma lines if use_trial_index is false for otherwise scalar vertices
            if (lines_[0].sig_ && !use_trial_index)
                return "";
        }

        // make a copy of lines that is sorted if sort is true
        line_vector lines;
        if (!sort) lines = lines_;
        else {
            lines.reserve(lines_.size());
            for (const Line &line : lines_)
                lines.insert(
                        std::lower_bound(lines.begin(), lines.end(), line, line_compare()), line);
        }

        // loop over lines
        string line_str = "(\"";
        for (const Line &line : lines) {
            if (!use_trial_index && line.sig_) continue;
            line_str += line.label_;
            line_str += ",";
        }
        line_str.pop_back(); // remove last comma
        line_str += "\")";
        return line_str;
    }

    string Vertex::str() const {
        string name = name_;
        if (print_type_ == "c++")
            name += line_str();
        return name;
    }

    string Linkage::str(bool format_temp, bool include_lines) const {

        if (!is_temp() || !format_temp) {
            // this is not an intermediate vertex (generic linkage) or we are not formatting intermediates
            // return the str of the left and right vertices
            return tot_str(false);
        }

        // prepare output string as a map of tmps, scalars, or reuse_tmps to a generic name
        string generic_str;
        if (is_scalar())
            generic_str = "scalars_";
        else if (reused_)
            generic_str = "reused_";
        else generic_str = "tmps_";
        generic_str += "[\"";

        // use id_ to create a generic name
        string dimstring = this->dimstring();
        if (id_ >= 0)
            generic_str += to_string(id_);

        if (!dimstring.empty())
            generic_str += "_" + dimstring;

        generic_str += "\"]";

        if (include_lines && print_type_ == "c++") // if lines are included, add them to the generic name (default)
            generic_str += line_str(); // sorts print order

        // create a generic vertex that has the same lines as this linkage.
        // this adds the spin and type strings to name
        // return its string representation
        return generic_str;
    }

    string Linkage::tot_str(bool fully_expand) const {

        if (empty()) return {};

        // do not fully_expand linkages that are not intermediates
        if (is_temp() && !fully_expand) {
            // return the string representation of the intermediate contraction
            return str(true, true);
        }

        if (left_->empty())  return right_->str();
        if (right_->empty()) return left_->str();

        // prepare output string
        string output, left_string, right_string;

        // get link vector
        vertex_vector link_vector = this->link_vector();

        // create new link vector without trial index
        if (!Vertex::use_trial_index) {
            vertex_vector link_vector_no_trial;
            for (const auto &op: link_vector) {
                VertexPtr new_op = op->clone();
                line_vector new_lines;
                for (const auto &line: new_op->lines())
                    if (!line.sig_) new_lines.push_back(line);
                new_op->update_lines(new_lines, false);
                link_vector_no_trial.push_back(new_op);
            }
            link_vector = link_vector_no_trial;
        }

        if (print_type_ == "c++") {

            if (is_addition()) {
                return left_->str() + " + " + right_->str();
            }

            vertex_vector scalars;
            vertex_vector tensors;
            for (const auto &op: link_vector) {
                if (op->empty()) continue;
                if (op->is_scalar())
                    scalars.push_back(op);
                else {
                    tensors.push_back(op);
                }
            }

            if (scalars.empty() && tensors.empty()) return "1.0";

            // first add scalars
            for (const auto &scalar: scalars) {
                string scalar_str = scalar->str();
                if (scalar->is_addition() && !scalar->is_temp())
                    scalar_str = "(" + scalar_str + ")";
                output += scalar_str + " * ";
            }

            if (tensors.empty()) {
                output.pop_back(); output.pop_back(); output.pop_back();
                return output;
            }

            bool format_dot = is_scalar() && tensors.size() > 1;

            // this is a scalar, so we need to format as a dot product
            if (format_dot) output += "1.00 * dot(";

            // add tensors
            for (size_t i = 0; i < tensors.size(); i++) {
                string tensor_string = tensors[i]->str();
                if (tensors[i]->is_addition() && !tensors[i]->is_temp())
                    tensor_string = "(" + tensor_string + ")";
                output += tensor_string;
                if (format_dot && i == tensors.size() - 2)
                    output += ", ";
                else if (i < tensors.size() - 1)
                    output += " * ";
            }
            if (format_dot) output += ")";

        }
        else if (print_type_ == "python") {
            if (is_addition()) {
                // we need to permute the right to match the left
                string left_labels, right_labels;
                for (const auto &line: left_->lines())
                    if (line.sig_ && !Vertex::use_trial_index) continue;
                    else left_labels += line.label_[0];
                for (const auto &line: right_->lines())
                    if (line.sig_ && !Vertex::use_trial_index) continue;
                    else right_labels += line.label_[0];

                output = left_->str() + " + ";

                if (left_labels != right_labels) {
                    // we need to permute the right to match the left
                    output += "np.einsum('";
                    output += right_labels + "->" + left_labels + "',";
                }
                output += right_->str();
                if (left_labels != right_labels)
                    output += ")";
                return output;
            }

            vector<string> indices;
            vertex_vector scalars;
            vertex_vector tensors;
            for (const auto &op: link_vector) {
                if (op->empty()) continue;
                if (op->is_scalar())
                    scalars.push_back(op);
                else {
                    tensors.push_back(op);
                    string label;
                    for (const auto &line: op->lines())
                        if (line.sig_ && !Vertex::use_trial_index) continue;
                        else label += line.label_[0];
                    indices.push_back(label);
                }
            }

            for (const auto &scalar: scalars) {
                string scalar_str = scalar->str();
                if (scalar->is_addition() && !scalar->is_temp())
                    scalar_str = "(" + scalar_str + ")";
                output += scalar_str + " * ";
            }
            if (!tensors.empty()) {
                output += "np.einsum('";
                for (const auto &index: indices)
                    output += index + ",";
                output.pop_back();
                output += "->";
                for (const auto &line: lines_)
                    if (line.sig_ && !Vertex::use_trial_index) continue;
                    else output += line.label_[0];
                output += "',";

                for (const auto &tensor: tensors) {
                    string tensor_str = tensor->str();
                    if (tensor->is_addition() && !tensor->is_temp())
                        tensor_str = "(" + tensor_str + ")";
                    output += tensor_str + ",";
                }

                if (tensors.size() > 2)
                    output += "optimize='optimal'";
                else output.pop_back();

                output += ")";

            } else {
                output.pop_back(); output.pop_back(); output.pop_back();
            }
        }

        return output;
    }

}