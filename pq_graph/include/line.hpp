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
#include <bitset>

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
                'I', 'J', 'K', 'M', 'N', 'O'};
        static inline array<char, 32> virt_labels_ = {              // names of virtual lines
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'v',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'V'};
        static inline array<char, 32> sig_labels_ = {'L', 'R', 'X', 'Y'}; // names of excited state lines
        static inline array<char, 32> den_labels_ = {'Q', 'U'};      // names of density fitting lines

        Line() = default;

        /**
         * Constructor
         * @param index index of the line position in the operator
         * @param name name of the line
         * @param blk whether the line has blocking
         */
        explicit Line(const std::string &name, char blk = '\0') : label_(name) {

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

        inline bool operator==(const Line& other) const {
            return (label_ == other.label_) &
                       (o_ == other.o_)     &
                       (a_ == other.a_)     &
                     (sig_ == other.sig_)   &
                     (den_ == other.den_);
        }

        constexpr bool equivalent(const Line& other) const {
            return   (o_ == other.o_)   &
                     (a_ == other.a_)   &
                   (sig_ == other.sig_) &
                   (den_ == other.den_);
        }

        inline bool operator!=(const Line& other) const {
            return !(*this == other);
        }

        constexpr bool operator<(const Line& other) const {
            // sort by sig, den, o, a, then label
            if (sig_ ^ other.sig_) return sig_;
            if (den_ ^ other.den_) return den_;
            if (o_ ^ other.o_) return !o_;
            if (a_ ^ other.a_) return a_;
            return label_ < other.label_;
        }

        constexpr bool same_kind(const Line& other) const {
            // sort by sig, den, o, a, but not label
            if (sig_ ^ other.sig_) return sig_;
            if (den_ ^ other.den_) return den_;
            if (o_ ^ other.o_) return !o_;
            if (a_ ^ other.a_) return a_;
            if (sig_ & other.sig_) return label_ <= other.label_; // L should be first
            return true;
        }

        constexpr bool operator>(const Line& other) const {
            return other < *this;
        }

        constexpr bool operator<=(const Line& other) const {
            return *this < other || *this == other;
        }

        constexpr bool operator>=(const Line& other) const {
            return *this > other || *this == other;
        }

        /// *** Getters/Setters *** ///

        constexpr bool has_blk() const { return blk_type_ != '\0'; }

        inline string block() const {
            switch (blk_type_) {
                case 's': return a_ ? "a" : "b";
                case 'r': return a_ ? "1" : "0";
                default: return "";
            }
        }

        constexpr char type() const {
            if (sig_) return 'L';
            if (den_) return 'Q';
            return o_ ? 'o' : 'v';
        }

        inline bool empty() const {
            return label_.empty();
        }

        inline uint_fast8_t size() const {
            return label_.size();
        }

    };

    /// *** Hash functions *** ///

    // struct for comparing lines while ignoring the label
    struct line_compare {
        constexpr bool operator()(const Line &left, const Line &right) const {
            return left.same_kind(right);
//            return left < right;
        }

        constexpr bool operator()(const Line *left, const Line *right) const {
            if (!left || !right) return !right;
            else return left->same_kind(*right);
//            else return left->operator<(*right);
        }
    };

    // define a vector of lines
    typedef std::vector<Line, std::allocator<Line>> line_vector;

    // define hash function for Line
    struct LineHash {
        inline uint_fast16_t operator()(const Line &line) const {

            // we can store each boolean as a bit in an integral type (4 bits)
            uint16_t hash = line.o_;
            hash |=   line.a_ << 1;
            hash |= line.sig_ << 2;
            hash |= line.den_ << 3;

            // store the first character of the label and return (12 bits total)
            return hash << 8 | line.label_[0];
        }

        constexpr size_t operator()(const Line *line) const {
            constexpr LineHash line_hash;

            // check if the pointer is null
            if (!line) return 0;

            // otherwise, return the hash of the line
            return line_hash(*line);
        }

        /**
         * Map lines from two vectors to a single map
         * @param old_lines source lines
         * @param new_lines destination lines
         * @return
         */
        static inline std::unordered_map<Line, Line, LineHash> map_lines(line_vector old_lines, line_vector new_lines) {

            std::unordered_map<Line, Line, LineHash> line_map;
            line_map.reserve(old_lines.size() + new_lines.size());

            // map every equivalent line to the reference line
            size_t old_n = old_lines.size(), new_n = new_lines.size();
            bool new_mapped[new_lines.size()], old_mapped[old_n];
            std::memset(new_mapped, 0, new_lines.size() * sizeof(bool));
            std::memset(old_mapped, 0, old_n * sizeof(bool));

            // map all old_lines to themselves
            for (size_t j = 0; j < new_lines.size(); j++) {
                line_map[old_lines[j]] = old_lines[j];
            }

            // start remapping old_lines
            for (size_t i = 0; i < new_n; i++) {
                bool found = false;
                for (size_t j = 0; j < old_n; j++) {
                    if (!old_mapped[j] && old_lines[j].equivalent(new_lines[i])) {
                        line_map[old_lines[j]] = new_lines[i];
                        old_mapped[j] = true;
                        found = true;
                        break;
                    }
                }
                new_mapped[i] = found;
            }

            // map all remaining new_lines to themselves
            for (size_t i = 0; i < old_n; i++) {
                if (!new_mapped[i]) {
                    // check if the line is already mapped (should not happen)
                    if (line_map.find(new_lines[i]) == line_map.end()) {
                        line_map[new_lines[i]] = new_lines[i];
                    }
                }
            }

            return line_map;
        }

    }; // struct LineHash

    struct LineEqual {
        inline bool operator()(const Line &lhs, const Line &rhs) const {

            // check equality of the lines
            return lhs == rhs;
        }

        inline bool operator()(const Line *lhs, const Line *rhs) const {

            // check if either pointer is null
            if (!lhs || !rhs) return rhs == lhs;

            // check equality of the pointers
            return *lhs == *rhs;
        }
    }; // struct LineEqual

    struct LinePropHash {
        constexpr uint_fast8_t operator()(const Line &line) const {

            // we can store each boolean as a bit in an integral type (4 bits)
            uint16_t hash = line.o_;
            hash |=   line.a_ << 1;
            hash |= line.sig_ << 2;
            hash |= line.den_ << 3;

            // return the hash (4 bits)
            return hash;
        }

        constexpr uint_fast8_t operator()(const Line *line) const {

            if (!line) return -1; // ignore null pointers (return -1)

            constexpr uint_fast8_t shift = 1;

            //  We do not care about the label for this hash function.
            uint_fast8_t hash = line->o_ << shift;

            // store each boolean as a bit in an integral type
            hash |= line->a_;
            hash = (hash << shift) | line->sig_;

            // return the hash (4 bits)
            return (hash << shift) | line->den_;
        }
    }; // struct LineHash

    struct LinePropEqual {
        constexpr bool operator()(const Line &lhs, const Line &rhs) const {
            return lhs.equivalent(rhs);
        }

        constexpr bool operator()(const Line *lhs, const Line *rhs) const {
            if (!lhs || !rhs) return rhs == lhs;
            return lhs->equivalent(*rhs);
        }
    }; // struct LineEqual

} // namespace pdaggerq

#endif //PDAGGERQ_LINE_HPP
