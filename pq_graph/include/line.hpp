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
#include <array>
#include <algorithm>
#include <cstring>

using std::runtime_error;
using std::hash;
using std::array;
using std::find;
using std::string;


namespace pdaggerq {

    /**
     * A line is a single index in an operator.
     * It is defined by its position in the tensor (idx_), whether it is occupied, virtual, alpha, or beta, and its name
     */
    struct Line {
        string label_{'\0'}; // name of the line (default to null character)

        bool o_ = false; // whether the line is occupied (true) or virtual (false/default)
        bool a_ = true; // whether the line is alpha/active (true) or beta/external (false)
        char blk_type_ = '\0'; // type of blocking (s: spin, r: range, '\0': none)
        bool sig_ = false; // whether the line is an excited state index
        bool den_ = false; // whether the line is for density fitting

        // valid line names
        static inline array<char, 32> occ_labels_ = {               // names of occupied lines
                'i', 'j', 'k', 'l', 'm', 'n', 'o',
                'I', 'J', 'K', 'L', 'M', 'N', 'O'};
        static inline array<char, 32> virt_labels_ = {              // names of virtual lines
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'v',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'V'};
        static inline array<char, 32> sig_labels_ = {'X', 'Y', 'Z'}; // names of excited state lines
        static inline array<char, 32> den_labels_ = {'Q', 'U'};      // names of density fitting lines

        Line() = default;

        /**
         * Constructor
         * @param index index of the line position in the operator
         * @param name name of the line
         * @param blk whether the line has blocking
         */
        inline explicit Line(const std::string &name, char blk = '\0') : label_(name) {

            // check input
            if (name.empty()) throw runtime_error("Line label cannot be empty");

            // set properties from first character
            char line_char = label_[0];
            if (line_char == '\0')
                return;

            auto occ_it = find(occ_labels_.begin(), occ_labels_.end(), line_char);
            o_ = occ_it != occ_labels_.end();

            if (!o_) { // not found in occupied lines
                auto virt_it = find(virt_labels_.begin(), virt_labels_.end(), line_char);

                if (virt_it == virt_labels_.end()) { // not found in virtual lines
                    auto sig_it = find(sig_labels_.begin(), sig_labels_.end(), line_char);
                    sig_ = sig_it != sig_labels_.end();

                    if (!sig_) { // not found in excited lines
                        auto den_it = find(den_labels_.begin(), den_labels_.end(), line_char);
                        den_ = den_it != den_labels_.end();

                        // could not find in any lines, throw error
                        if (!den_)
                            throw runtime_error("Invalid line " + std::string(1, line_char));
                    }
                }
            }

            // determine block type (spin or range)
            switch (blk) {
                case '\0': // no block
                    blk_type_ = '\0';
                    a_ = false; break;
                case 'a': // spin block
                    blk_type_ = 's';
                    a_ = true; break;
                case 'b':
                    blk_type_ = 's';
                    a_ = false; break;
                case '1':
                    blk_type_ = 'r';
                    a_ = true; break;
                case '0':
                    blk_type_ = 'r';
                    a_ = false; break;
                default:
                    throw runtime_error("Invalid block type " + std::string(1, blk));
            }
        }

        /// *** Copy/move operators *** ///

        Line(const Line &other) = default; // copy constructor
        Line(Line &&other) noexcept = default; // move constructor
        Line &operator=(const Line &other) = default; // copy assignment
        Line &operator=(Line &&other) noexcept = default; // move assignment


        /// *** Comparisons *** ///

        bool operator==(const Line& other) const {
            return label_ == other.label_ &&
                       o_ == other.o_     &&
                       a_ == other.a_     &&
                     sig_ == other.sig_   &&
                     den_ == other.den_;
        }

        inline bool equivalent(const Line& other) const {
            return   o_ == other.o_   &&
                     a_ == other.a_   &&
                   sig_ == other.sig_ &&
                   den_ == other.den_;
        }

        bool operator!=(const Line& other) const {
            return !(*this == other);
        }

        inline bool operator<(const Line& other) const {
            // sort by sig, den, o, a, then label
            if (sig_ ^ other.sig_) return sig_;
            if (den_ ^ other.den_) return den_;
            if (o_ ^ other.o_) return !o_;
            if (a_ ^ other.a_) return a_;
            return label_ < other.label_;
        }

        inline bool in_order(const Line& other) const {
            // sort by sig, den, o, a, but not label
            if (sig_ ^ other.sig_) return sig_;
            if (den_ ^ other.den_) return den_;
            if (o_ ^ other.o_) return !o_;
            if (a_ ^ other.a_) return a_;
            return true;
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

        /// *** Getters/Setters *** ///

        inline bool has_blk() const { return blk_type_ != '\0'; }

        inline char block() const {
            switch (blk_type_) {
                case 's': return a_ ? 'a' : 'b';
                case 'r': return a_ ? '1' : '0';
                default:  return '\0';
            }
        }

        inline char type() const {
            if (sig_) return 'L';
            if (den_) return 'Q';
            return o_ ? 'o' : 'v';
        }

        inline bool empty() const {
            return label_.empty();
        }

        inline uint_fast8_t size() const {
            return label_.empty();
        }

    };

    /// *** Hash functions *** ///

    // define hash function for Line
    struct LineHash {
        uint_fast16_t operator()(const Line &line) const {

            uint_fast16_t hash = 0;

            // we can store each boolean as a bit in an integral type (4 bits)
            hash = (hash << 1) | line.o_;
            hash = (hash << 1) | line.a_;
            hash = (hash << 1) | line.sig_;
            hash = (hash << 1) | line.den_;

            // because a char is 8 bits, we shift by 8 (8 bits; 12 total)
            hash = (hash << 8) | line.label_[0];

            // if the label is longer than 1 character,
            // we only shift by 4 and consider the last character (which can be the same as the first)
            hash = (hash << 4) | line.label_.back(); // (4 bits; 16 total)

            // return the hash (16 bits total)
            return hash;
        }
    }; // struct LineHash

    typedef std::vector<Line, std::allocator<Line>>
    line_vector;

    // define hash function for Line
    struct LinePtrHash {
        size_t operator()(const Line *line) const {
            constexpr LineHash line_hash;

            // check if the pointer is null
            if (!line) return 0;

            // otherwise, return the hash of the line
            return line_hash(*line);
        }
    }; // struct LineHash

    struct LinePtrEqual {
        bool operator()(const Line *lhs, const Line *rhs) const {

            // check if either pointer is null
            if (!lhs || !rhs) return false;

            // check equality of the pointers
            return *lhs == *rhs;
        }
    }; // struct LinePtrEqual

    struct SimilarLineHash {
        uint_fast8_t operator()(const Line &line) const {

            //  We do not care about the label for this hash function.
            uint_fast8_t hash = 0;

            // store each boolean as a bit in an integral type
            hash = (hash << 1) | line.o_;
            hash = (hash << 1) | line.a_;
            hash = (hash << 1) | line.sig_;
            hash = (hash << 1) | line.den_;

            // return the hash (4 bits)
            return hash;
        }
    }; // struct LineHash

    struct SimilarLineEqual {
        bool operator()(const Line &lhs, const Line &rhs) const {
            return lhs.equivalent(rhs);
        }
    }; // struct LinePtrEqual

} // namespace pdaggerq

#endif //PDAGGERQ_LINE_HPP
