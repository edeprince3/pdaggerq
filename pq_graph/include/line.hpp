//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: line.hpp
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

#ifndef PDAGGERQ_LINE_HPP
#define PDAGGERQ_LINE_HPP

#include <utility>
#include <stdexcept>
#include <unordered_set>

using std::string;
using std::unordered_set;
using std::runtime_error;
using std::hash;

namespace pdaggerq {

    /**
     * A line is a single index in an operator.
     * It is defined by its position in the tensor (idx_), whether it is occupied, virtual, alpha, or beta, and its name
     */
    struct Line {
        string label_; // name of the line

        bool o_ = false; // whether the line is occupied (true) or virtual (false/default)
        bool a_ = true; // whether the line is alpha/active (true) or beta/external (false)
        char blk_type_ = '\0'; // type of blocking (s: spin, r: range, '\0': none)
        bool sig_ = false; // whether the line is an excited state index
        bool den_ = false; // whether the line is for density fitting

        static inline unordered_set<char> occ_labels_ = { // names of occupied lines
            'i', 'j', 'k', 'l', 'm', 'n', 'o',
            'I', 'J', 'K', 'L', 'M', 'N', 'O'
        };
        static inline unordered_set<char> virt_labels_ = { // names of virtual lines
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'v',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'V'
        };
        static inline unordered_set<char> sig_labels_ = { // names of excited state lines
            'X', 'Y', 'Z'
        };
        static inline unordered_set<char> den_labels_ = { // names of density fitting lines
            'Q', 'U'
        };

        Line() = default;

        /**
         * Constructor
         * @param index index of the line position in the operator
         * @param name name of the line
         * @param blk whether the line has blocking
         */
        inline explicit Line(string name, char blk = '\0') : label_(std::move(name)){

            if (label_.empty()) throw runtime_error("Empty line label");

            char line_char = label_[0];
            sig_ = sig_labels_.find(line_char) != sig_labels_.end(); // default to not excited
            den_ = den_labels_.find(line_char) != den_labels_.end(); // default to not density fitting

            if (!sig_ && !den_) {
                // default to occupied for all other lines
                o_ = virt_labels_.find(line_char) == virt_labels_.end();
            }

            if (blk == 'a' || blk == 'b') blk_type_ = 's';
            else if (blk == '0' || blk == '1') blk_type_ = 'r';
            else if (blk != '\0')
                throw runtime_error("Invalid blk " + string(1, blk));

            if (blk_type_ == 's') a_ = blk == 'a';
            else if (blk_type_ == 'r') a_ = blk == '1';
            else a_ = false;

        }

        Line(const Line &other) = default; // copy constructor
        Line(Line &&other) noexcept = default; // move constructor
        Line &operator=(const Line &other) = default; // copy assignment
        Line &operator=(Line &&other) noexcept = default; // move assignment



        /// *** Comparison rhs *** ///
        /// all comparison rhs are defined in terms of name and properties. Index is not used.

        bool operator==(const Line& other) const {
            return label_ == other.label_ &&
                       o_ == other.o_     &&
                       a_ == other.a_     &&
                     sig_ == other.sig_   &&
                     den_ == other.den_;
        }



        bool equivalent(const Line& other) const {
            return   o_ == other.o_   &&
                     a_ == other.a_   &&
                   sig_ == other.sig_ &&
                   den_ == other.den_;
        }

        bool operator!=(const Line& other) const {
            return !(*this == other);
        }

        bool operator<(const Line& other) const {
            // sort by sig, den, o, a, then label
            if (sig_ ^ other.sig_) return sig_;
            if (den_ ^ other.den_) return den_;
            if (o_ ^ other.o_) return !o_;
            if (a_ ^ other.a_) return a_;
            return label_ < other.label_;
        }

        bool operator>(const Line& other) const {
            return other < *this;
        }

        bool operator<=(const Line& other) const {
            return *this < other || *this == other;
        }

        bool operator>=(const Line& other) const {
            return *this > other || *this == other;
        }

        inline char blk() const {
            if (blk_type_ == 's') return a_ ? 'a' : 'b';
            if (blk_type_ == 'r') return a_ ? '1' : '0';
            return '\0';
        }

        inline bool has_blk() const { return blk_type_ != '\0'; }

        inline char ov() const {
            if (sig_) return 'L';
            if (den_) return 'Q';
            return o_ ? 'o' : 'v';
        }

        inline bool empty() const {
            return label_.empty();
        }
    };

    // define hash function for Line
    struct LineHash {
        size_t operator()(const Line &line) const {
            string blk{
                line.o_ ? 'o' : 'v',
                line.a_ ? 'a' : 'b',
                line.sig_ ? 'L' : 'N',
                line.den_ ? 'Q' : 'N'
            };

            return hash<string>()(line.label_ + blk);
        }
    }; // struct LineHash

} // pdaggerq

#endif //PDAGGERQ_LINE_HPP
