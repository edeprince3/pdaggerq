//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: vertex.cc
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

#include <set>
#include <unordered_set>
#include <iostream>
#include <utility>
#include "../include/vertex.h"
#include "../../pdaggerq/pq_tensor.h"
#include "../../pdaggerq/pq_string.h"

using std::ostream;
using std::string;
using std::vector;
using std::map;
using std::unordered_map;
using std::shared_ptr;
using std::make_shared;
using std::set;
using std::unordered_set;
using std::pair;
using std::make_pair;
using std::to_string;
using std::invalid_argument;

namespace pdaggerq {

    /****** Constructors ******/

    Vertex::Vertex(const delta_functions &delta) {

        // set base name
        base_name_ = "Id";

        // determine if vertex is blocked
        bool is_range_blocked = pq_string::is_range_blocked;
        bool is_spin_blocked = pq_string::is_spin_blocked;
        has_blk_ = is_range_blocked || is_spin_blocked;

        // get label types
        string blk_string; // get blk string
        if (is_spin_blocked) {
            for (const string &spin: delta.spin_labels)
                blk_string += spin;
        } else if (is_range_blocked) {
            for (const string &range: delta.label_ranges)
                blk_string += range == "act" ? "1" : "0";
        }

        // set lines
        set_lines(delta.labels, blk_string);
    }

    Vertex::Vertex(const integrals &integral, const string &type) {

        // set base name
        if (type == "two_body")  base_name_ = "g";
        else if (type == "eri")  base_name_ = "eri";
        else if (type == "core") base_name_ = "h";
        else if (type == "fock") base_name_ = "f";
        else if (type == "d+" || type == "d-" )
                                 base_name_ = "dp";
        else throw invalid_argument("Vertex::Vertex: invalid integral type: " + type);

        //determine if vertex is blocked
        bool is_range_blocked = pq_string::is_range_blocked;
        bool is_spin_blocked = pq_string::is_spin_blocked;
        has_blk_ = is_range_blocked || is_spin_blocked;

        // get label types
        string blk_string; // get blk string
        if (is_spin_blocked) {
            for (const string &spin: integral.spin_labels)
                blk_string += spin;
        } else if (is_range_blocked) {
            for (const string &range: integral.label_ranges)
                blk_string += range == "act" ? "1" : "0";
        }

        // set lines
        set_lines(integral.labels, blk_string);
    }

    Vertex::Vertex(const amplitudes &amp, char type) {

        // get order of amplitude
        size_t order = (amp.n_annihilate <= amp.n_create) ? amp.n_create : amp.n_annihilate;

        // set base name
        base_name_ = type + to_string(order);

        //determine if vertex is blocked
        bool is_range_blocked = pq_string::is_range_blocked;
        bool is_spin_blocked = pq_string::is_spin_blocked;
        has_blk_ = is_range_blocked || is_spin_blocked;

        // get label types
        string blk_string; // get blk string
        if (is_spin_blocked) {
            for (const string &spin: amp.spin_labels)
                blk_string += spin;
        } else if (is_range_blocked) {
            for (const string &range: amp.label_ranges)
                blk_string += range == "act" ? "1" : "0";
        }

        // set lines
        set_lines(amp.labels, blk_string);

        // if this is a sigma vertex, add a sigma line
        if (type == 'r' || type == 's' || type == 'l' || type == 'm') {
            is_sigma_ = true;
            Line sigma_line("X");
            sigma_line.sig_ = true;
            sigma_line.o_ = false;
            lines_.insert(lines_.begin(), sigma_line);
            update_name();
        }

    }


    Vertex::Vertex(const string &vertex_string) {
        /// separate name and line from vertex_string


        // first, check if vertex_string has '(' or '<'
        bool is_eri = vertex_string.find('<') != string::npos;
        if (vertex_string.find('(') == string::npos && !is_eri) { // if not, then vertex_string is just the name
            base_name_ = vertex_string;
            name_ = vertex_string;
            rank_ = 0;
            shape_ = shape();

        }  else { // if it has '(', then vertex_string is name(line)
            string line_string; // string of lines
            string blk; // blocking of lines
            if (is_eri){ // if it has '<', then vertex_string is <p,q||r,s> with name = eri and line = p,q,r,s
                base_name_ = "eri";


                // remove '<' and '>'
                size_t langel_idx = vertex_string.find('<');
                size_t rangel_idx = vertex_string.find('>');
                line_string = vertex_string.substr(langel_idx+1, rangel_idx-langel_idx-1);

                // append text after '>' to line_string
                if (rangel_idx+1 < vertex_string.size()) line_string += vertex_string.substr(rangel_idx+1);

                // remove '||' and replace with ','
                size_t vline_idx = line_string.find("||");
                line_string.replace(vline_idx, 2, ",");

            } else {
                size_t index = vertex_string.find('('); // index of '('
                size_t name_end = vertex_string.find('_');
                if (name_end == string::npos) name_end = index;
                else {
                    // blk is the text after the first '_' and before the '(' if its value is 'a' or 'b' or '0' or '1'
                    while (name_end != string::npos && blk.empty()) {
                        bool has_spin_ = vertex_string[name_end+1] == 'a' || vertex_string[name_end+1] == 'b';
                        bool has_range_ = vertex_string[name_end+1] == '0' || vertex_string[name_end+1] == '1';

                        has_blk_ = has_spin_ || has_range_;

                        if (has_blk_) blk = vertex_string.substr(name_end + 1, index - name_end - 1);
                        else name_end = vertex_string.find('_', name_end+1);
                    }
                    if (name_end == string::npos) name_end = index;
                }

                base_name_ = vertex_string.substr(0, name_end); // name is everything before '(' or '_'

                // line is everything between '(' and ')'
                line_string = vertex_string.substr(index + 1, vertex_string.size() - index - 2);
            }

            /// split line_string into lines, with ',' as delimiter
            vector<string> lines;
            string line_substring = line_string; // create a copy of line_string for splitting
            while (line_substring.find(',') != string::npos) { // while there are still commas
                size_t comma_idx = line_substring.find(','); // find index of next comma
                lines.push_back(line_substring.substr(0, comma_idx)); // extract line; append to lines

                line_substring = line_substring.substr(comma_idx+1, // remove everything before comma
                                                         line_substring.size()-comma_idx-1);
            }

            // check if '_' is in line_substring and
            size_t underscore_idx = line_substring.find('_');

            string last_line = line_substring;

            // if so, keep everything before '_' and set the blocks to be everything after (excluding '_'); this is for eris
            if (underscore_idx != string::npos && blk.empty()) {
                last_line = line_substring.substr(0, underscore_idx);
                blk = line_substring.substr(underscore_idx + 1, line_substring.size() - underscore_idx - 1);
            }

            // add last line
            if (!last_line.empty())
                lines.push_back(last_line);

            // set parameters from lines
            set_lines(lines, blk);
        }
    }

    Vertex::Vertex(string base_name, const vector<Line>& lines) :
        base_name_(std::move(base_name)), lines_(lines), rank_(lines.size()) {

        // set name from base_name and lines
        update_lines(lines);
    }

    void Vertex::set_lines(const vector<string> &lines, const string &blk_string) {
        /// set rank
        rank_ = lines.size();

        /// set lines
        lines_ = vector<Line>(rank_);

        // create line objects
        has_blk_ = !blk_string.empty();
        uint_fast8_t c_blk = 0; // count_ current number of characters for blocking
        string ovstring(rank_, 'o'); // ovstring assuming all occupied
        string new_blk_string(rank_, '\0'); // assumming no blocking
        for (int i = 0; i < lines.size(); i++) { // loop over lines

            lines_[i] = Line(lines[i]); // create line without a block

            // check if line is not ov
            bool is_sigma = lines_[i].sig_;
            bool is_den = lines_[i].den_;

            if (has_blk_ && !is_sigma && !is_den) { // if it blocked and is not sigma or density fitted
                lines_[i] = Line(lines[i], blk_string[c_blk++]); // create line with blocking
            }

            if (is_sigma) is_sigma_ = true; // check if vertex is sigma
            if (is_den) is_den_ = true; // check if vertex is density fitted

            ovstring[i] = lines_[i].ov(); // set ovstring

            if (has_blk_)
                new_blk_string[i] = lines_[i].blk(); // set blocking string
        }

        // create shape from lines
        shape_ = shape(lines_);

        // add ovstring and block to name
        format_name(ovstring, new_blk_string);

    }

    void Vertex::format_name(const string &ovstring, const string &new_blk_string) {
        name_ = base_name_ + "_";

        // scalars have no dimension
        if (rank_ == 0) return;

        char type = base_name_[0];
        bool is_amplitude = type == 't' || type == 'u' || type == 'r' || type == 's' || type == 'l' || type == 'm';

        // format tensor block as a map if it is not an amplitude or if it has a block
        if (!is_amplitude || has_blk_) {
            name_ += "[\"";
            name_ += has_blk_ ? new_blk_string : "";
            name_ += (!is_amplitude && has_blk_) ? "_" : "";
            name_ += !is_amplitude ? ovstring : "";
            name_ += "\"]";
        }

    }

    string Vertex::dimstring() const {
        string dimstring;
        if (rank_ == 0) return dimstring;

        if (has_blk_)
            dimstring += blk_string() + "_";
        dimstring += ovstring();
        return dimstring;
    }

    inline void Vertex::update_lines(const vector<Line> &lines, bool update_name){

        lines_ = lines; // set lines
        rank_ = lines.size(); // set rank
        shape_ = shape(lines_); // create shape from lines
        if (update_name)
            this->update_name(); // update name

    }

    string Vertex::blk_string() const {
        if (!has_blk_ || lines_.empty()) return "";
        string blk_string(rank_, '\0'); // string assuming no blocking
        uint_fast8_t line_idx = 0; // index of line
        for (const Line& line : lines_) {
            blk_string[line_idx++] = line.blk(); // string assuming no blocking
        }

        return blk_string;
    }

    Vertex::Vertex() {
        name_ = "Empty";
        base_name_ = "Empty";
        has_blk_ = false;
        rank_ = 0;
        shape_ = shape();
    }

    string Vertex::ovstring(const vector<Line> &lines) {
        if (lines.empty()) return "";
        uint_fast8_t line_size = lines.size();
        string ovstring(line_size, 'o'); // ovstring assuming all occupied
        uint_fast8_t line_idx = 0; // index of line
        for (uint_fast8_t i = 0; i < line_size; i++) {
            ovstring[line_idx++] = lines[i].ov(); // set ovstring
        }

        return ovstring;
    }

    inline Vertex Vertex::permute(size_t perm_id, bool &swap_sign) const {

        swap_sign = false; // initialize swap sign to false
        if (perm_id == 0) return *this; // if perm_id is 0, return self
        if (rank_ <= 2) return {}; // if rank is 2 or less, return empty vertex (no permutations possible)

        /// perform all permutations
        uint_fast8_t right_size = rank_/2; // right n_ops
        uint_fast8_t left_size = rank_ - right_size; // left n_ops

        uint_fast8_t left_perm[left_size];
        uint_fast8_t right_perm[right_size];

        // fill left_perm and right_perm
        for (uint_fast8_t i = 0; i < left_size; i++) left_perm[i] = i;
        for (uint_fast8_t i = 0; i < right_size; i++) right_perm[i] = i + left_size;

        constexpr size_t factorial_map[20]{
                1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200,
                1307674368000, 20922789888000, 355687428096000, 6402373705728000, 121645100408832000
        }; // factorial map


        auto factorial = [&factorial_map](size_t n) { // factorial function
            if (n < 20) return factorial_map[n]; // if n is less than 20, return from map

            // else, calculate factorial
            for (size_t i = n-1; i > 1; i--) n *= i;
            return n;
        };

        size_t total_left_perms  = factorial(left_size); // total number of left permutations
        size_t total_right_perms = factorial(right_size); // total number of right permutations

        // if perm_id is too large, return empty vertex
        if (perm_id >= total_left_perms * total_right_perms) return {};

        size_t left_perm_id = perm_id % total_left_perms;
        size_t right_perm_id = perm_id / total_left_perms;

        // permute left side
        uint_fast8_t left_swaps = 0; // number of swaps of left side
        for (uint_fast8_t i = 0; i < left_size; i++) {
            uint_fast8_t j = i + left_perm_id % (left_size - i);
            left_swaps += (size_t) j != i;
            std::swap(left_perm[i], left_perm[j]);
        }

        // permute right side
        uint_fast8_t right_swaps = 0; // number of swaps of right side
        for (uint_fast8_t i = 0; i < right_size; i++) {
            uint_fast8_t j = i + right_perm_id % (right_size - i);
            right_swaps += (size_t) j != i;
            std::swap(right_perm[i], right_perm[j]);
        }

        swap_sign = (left_swaps + right_swaps) % 2 == 1; // set swap sign

        // create new permuted vertex
        Vertex perm_op(*this); // copy this vertex

        // permute lines
        vector<Line> perm_lines = perm_op.lines_;
        uint_fast8_t line_idx = 0; // index of line
        for (uint_fast8_t i = 0; i < rank_; i++) {
            uint_fast8_t perm_idx = (i < left_size) ? left_perm[i] : right_perm[i-left_size];
            perm_lines[line_idx++] = lines_[perm_idx];
        }

        perm_op.update_lines(perm_lines); // reassign lines

        return perm_op; // return permuted vertex

    }

    Vertex Vertex::permute_like(const Vertex &other, bool &swap_sign) const {

        // reset swap sign
        swap_sign = false;

        if (rank_ == 0) return {}; // if rank is 0, return empty vertex (no lines to permute)

        // ensure that blocking and occupation of original vertex are not different
        if (!equivalent(other)) return {};
        if (!same_lines(other)) return {}; // ensure that lines are the same

        /// check permutations of this vertex to find one that matches other, or return empty vertex
        bool same_vertex; // flag to check if same structure
        bool has_next = true; // flag to check if there is a next permutation
        Vertex this_permuted;
        size_t perm_idx = 0;
        while (has_next) {
            // reset swap sign
            swap_sign = false;

            // make permuted vertex
            this_permuted = permute(perm_idx, swap_sign);

            // check if permuted vertex is the same as other
            same_vertex = this_permuted == other;

            if (same_vertex) break;

            perm_idx++;
            has_next = !this_permuted.empty();
        }

        if (!same_vertex) {
            swap_sign = false;
            return {};
        }

        return this_permuted;
    }

    bool is_isomorphic(const Vertex &left, const Vertex &right, bool &swap_signs){

        // if rhs are equal, return true
        if (left == right) return true;

        // check if base names are equal
        if (left.base_name_ != right.base_name_) return false;

        // check if permutations of left are equal to right
        bool test_swap_signs = false;
        Vertex permuted_similar = left.permute_like(right, test_swap_signs);

        if (!permuted_similar.equivalent(right)) return false;

        if (test_swap_signs) swap_signs = !swap_signs;
        return true; // if permutation was applied, rhs are isomorphic

    }

    bool Vertex::isomorphic(const Vertex &other) const {
        bool swap_signs = false;
        return is_isomorphic(*this, other, swap_signs);
    }

    bool Vertex::permute_eri() {

        // if allow_permute is false, do nothing
        if (!allow_permute_) return false;

        // valid ovstrings for eri vertex
        unordered_set<string> valid_ovstrings = {"oooo", "vvvv", "oovv", "vvoo", "vovo", "vooo", "oovo", "vovv", "vvvo"};
        string orig_blk = blk_string(); // save original blocking

        string ovstring;
        Vertex new_eri; // new eri vertex
        bool swap_sign = false; // swap sign flag

        size_t id = 0; // permutation id
        bool is_valid; // is ovstring valid?
        size_t count = 0; // number of permutations tried

        do { // test all permutations for a valid ovstring
            new_eri = permute(id++, swap_sign); // get permutation

            if (new_eri.empty()) return false; // if permutation is empty, do nothing

            // get ovstring
            ovstring = new_eri.ovstring();

            // check if ovstring is valid
            is_valid = valid_ovstrings.find(ovstring) != valid_ovstrings.end();

        // end while when valid ovstring is found or throw error when max permutations is reached
        } while (count++ < valid_ovstrings.size() && !is_valid);

        if (!is_valid)
            return false; // if ovstring is not valid, do nothing (should ideally not happen)

        // reassign vertex
        *this = new_eri;
        return swap_sign; // return sign change
    }

    void Vertex::sort(vector<Line> &lines) {
        if (lines.empty()) return; // do nothing if rank is zero

        // sort lines by occ/vir status (virs on left, occ on right); sort lines by blocks for same occ/vir (alpha on left, beta on right).
        // if all these are equal, sort by ASCII ordering of line name
        std::sort(lines.begin(), lines.end(), std::less<>());
    }

    void Vertex::sort() {
        sort(lines_);
        update_lines(lines_); // set lines
    }

    /****** operator overlaods ******/

    inline bool Vertex::operator==(const Vertex &other) const {

        // check if rank, n_occ, n_vir, n_alph, n_beta are equal
        if (rank_ != other.rank_) return false;
        if (shape_ != other.shape_) return false;

        // check if rhs have equivalent label properties
        if (lines_ == other.lines_) return false;

        // check if the vertex names are equal
        return name_ == other.name_;
    }

    bool Vertex::operator!=(const Vertex &other) const {
        return !(*this == other);
    }

    bool Vertex::operator<(const Vertex &other) const {
        return name_ < other.name_;
    }

    inline bool Vertex::same_lines(const Vertex &other) const {

        // check if lines are same n_ops
        if (lines_.size() != other.lines_.size())
            return false;
        else
            return lines_ == other.lines_;
    }

    inline bool Vertex::equivalent(const Vertex &other) const {

        // check if rank, n_occ, n_vir, n_alph, n_beta are equal
        if (rank_  !=  other.rank_) return false;
        if (shape_ != other.shape_) return false;

        // check if rhs have equivalent label properties
        for (uint_fast8_t i = 0; i < rank_; i++) {
            if ( !lines_[i].equivalent(other.lines_[i]) )
                return false;
        }

        // check if the vertex names are equal
        return name_ == other.name_;
    }

    string Vertex::line_str() const{
        if (rank_ == 0) return ""; // if rank is 0, return empty string

        // loop over lines
        string line_str = "(\"";
        for (const Line &line : lines_) {
            line_str += line.label_;
            line_str += ",";
        }
        line_str.pop_back(); // remove last comma
        line_str += "\")";
        return line_str;
    }

    string Vertex::str() const {
        return name_ + line_str();
    }

    map<Line, uint_fast8_t> Vertex::self_links() const {
        // if rank is 0 or 1, return empty vector
        if (rank_ <= 1) return {};

        // return a map of the labels of the self-contractions of the vertex
        //        and a pair of the line with the frequency of the label
        map<Line, uint_fast8_t> self_links;
        for (auto & line : this->lines()) {
            // count the number of times the line appears
            self_links[line]++;
        }

        return self_links;
    }

    vector<VertexPtr> Vertex::make_self_linkages(map<Line, uint_fast8_t> &self_links) {
        // replace repeated lines with arbitrary lines
        map<Line, uint_fast8_t> counts;
        for (auto & [line, freq] : self_links) {
            // if line is not repeated, do nothing
            if (freq == 1) continue;

            // if line is repeated, find all of them in the lines of the vertex
            auto it = std::find(lines_.begin(), lines_.end(), line);

            while (it != lines_.end()) {
                // replace the repeated lines with arbitrary lines
                it->label_ = it->label_ + to_string(counts[line]++);
                it = std::find(it+1, lines_.end(), line);
            }
        }

        // create delta functions for this vertex
        vector<VertexPtr> delta_ops;
        for (auto & [line, freq] : self_links) {
            // if line is not repeated, do nothing
            if (freq == 1) continue;

            // create generic labels for the delta function
            vector<Line> delta_lines;
            for (uint_fast8_t i = 0; i < freq; i++) {
                Line new_line = line;
                new_line.label_ = new_line.label_ + to_string(i);
                delta_lines.push_back(new_line);
            }

            // create delta function
            VertexPtr delta_op = std::make_shared<Vertex>("Id", delta_lines);

            // add delta function to vector
            delta_ops.push_back(delta_op);
        }

        return delta_ops;
    }

    void Vertex::genericize() {
        // set all occupied and virtual lines to 'ox' and 'vx', where x is the index
        update_lines(general_lines(lines_));
    }

    vector<Line> Vertex::general_lines(const vector<Line>& lines) {
        vector<Line> generic_lines = lines;
        uint_fast8_t c_occ = 0, c_vir = 0;
        for (Line &line : generic_lines) {
            if (line.o_) line.label_ = "o" + std::to_string(c_occ++);
            else line.label_ = "v" + std::to_string(c_vir++);
        }

        return generic_lines;
    }

    Vertex Vertex::generic() const {
        Vertex generic_op = *this;
        generic_op.genericize();
        return generic_op;
    }

    // create a hashmap to store vertices for quick lookup
    struct VertexHash {
        size_t operator()(const Vertex &vertex) const {
            return std::hash<string>()(vertex.str());
        }
    };

} // pdaggerq