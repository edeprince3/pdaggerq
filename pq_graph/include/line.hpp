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

#include <cstdint>
#include <utility>
#include <stdexcept>
#include <array>
#include <vector>
#include <unordered_map>
#include <set>
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
        bool nuc_ = false; // whether the line belongs to the nuclear (second fermion) species

        // nuclear orbital labels carry this prefix, e.g. "ni" (occ) / "na" (vir);
        // must match pdaggerq::nuclear_prefix in the core. occupied/virtual is then
        // determined from the remaining character, within the nuclear space.
        static constexpr char nuclear_prefix_ = 'n';

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
        ~Line() = default;

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

            // nuclear (second-species) labels carry a prefix; a lone prefix char is
            // still an electron label, so only multi-character "n*" labels are nuclear.
            // occupied/virtual is then read from the next character within that space.
            if (label_.size() > 1 && line_char == nuclear_prefix_) {
                nuc_ = true;
                line_char = label_[1];
            }

            auto occ_it = find(occ_labels_.begin(), occ_labels_.end(), line_char);
            o_ = occ_it != occ_labels_.end();

            if (!o_) { // not found in occupied lines
                auto virt_it = find(virt_labels_.begin(), virt_labels_.end(), line_char);

                if (virt_it == virt_labels_.end()) { // not found in virtual lines
                    if (nuc_) {
                        // A nuclear line is occupied or virtual only -- never a
                        // sigma/DF line. The electron sigma/aux letters (L,R,X,Y)
                        // are perfectly valid *nuclear* index letters (e.g. "nL"),
                        // so classify a nuclear label case-insensitively against the
                        // occupied table and never set sig_/den_ for it; otherwise
                        // it falls through to virtual like any other letter.
                        char lc = (line_char >= 'A' && line_char <= 'Z')
                                  ? (char)(line_char + ('a' - 'A')) : line_char;
                        o_ = find(occ_labels_.begin(), occ_labels_.end(), lc) != occ_labels_.end();
                    } else {
                        auto sig_it = find(sig_labels_.begin(), sig_labels_.end(), line_char);
                        sig_ = sig_it != sig_labels_.end();

                        if (!sig_) { // not found in excited lines
                            auto den_it = find(den_labels_.begin(), den_labels_.end(), line_char);
                            den_ = den_it != den_labels_.end();

                            // could not find in any lines. defaults to virtual
                        }
                    }
                }
            }

            // determine block type (spin or range)
            switch (blk) {
                case '\0': // no block
                    blk_type_ = '\0'; break;
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
            return (label_ == other.label_) &
                       (o_ == other.o_)     &
                       (a_ == other.a_)     &
                     (sig_ == other.sig_)   &
                     (den_ == other.den_)   &
                     (nuc_ == other.nuc_);
        }

        bool equivalent(const Line& other) const {
            return   (o_ == other.o_)   &
                     (a_ == other.a_)   &
                   (sig_ == other.sig_) &
                   (den_ == other.den_) &
                   (nuc_ == other.nuc_);
        }

        bool operator!=(const Line& other) const {
            return !(*this == other);
        }

        bool operator<(const Line& other) const {
            // sort by sig, den, nuc, o, a, then label
            if (sig_ ^ other.sig_) return sig_;
            if (den_ ^ other.den_) return den_;
            if (nuc_ ^ other.nuc_) return !nuc_; // electron lines before nuclear lines
            if (o_ ^ other.o_) return !o_;
            if (a_ ^ other.a_) return a_;
            return label_ < other.label_;
        }

        bool same_kind(const Line& other) const {
            // sort by sig, den, nuc, o, a, but not label
            if (sig_ ^ other.sig_) return sig_;
            if (den_ ^ other.den_) return den_;
            if (nuc_ ^ other.nuc_) return !nuc_;
            if (o_ ^ other.o_) return !o_;
            if (a_ ^ other.a_) return a_;
            if (sig_ & other.sig_) return label_ <= other.label_; // L should be first
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

        bool has_blk() const { return blk_type_ != '\0'; }

        char block() const {
            switch (blk_type_) {
                case 's': return a_ ? 'a' : 'b';
                case 'r': return a_ ? '1' : '0';
                default: return '\0';
            }
        }

        char type() const {
            if (sig_) return 'L';
            if (den_) return 'Q';
            if (nuc_) return o_ ? 'O' : 'V'; // nuclear occupied/virtual (distinct block from electron o/v)
            return o_ ? 'o' : 'v';
        }

        // the subscript this line WANTS: its own character (electron labels are single
        // characters) or, for a nuclear label, the uppercased base character.
        //
        // This map is NOT injective, and must never be used on its own. The core's label
        // pool is case-doubled -- pq_utils.cc builds the nuclear labels as "n" + {i,j,k,l,
        // m,n,I,J,K,L,M,N}, so "ni" and "nI" are two DISTINCT nuclear indices -- while the
        // uppercasing here case-folds them onto the same 'I'. An electron line labelled "I"
        // collides with nuclear "ni" the same way. Two distinct lines sharing a subscript
        // silently emits a wrong contraction: a fused NEO intermediate came out as
        //     tmps_["20_ooOO"][i,j,I,I] = B["QOO"][Q,I,I] * B["Qoo"][Q,i,j]
        // -- defined only on its nuclear diagonal, then consumed OFF it as [i,j,J,I].
        // No crash, no warning, just a wrong Hessian.
        //
        // So printers assign subscripts per statement via assign_subscripts() below, and
        // ask for them through einsum_char(). This is the per-term pool the old comment
        // here described and did not implement.
        char natural_einsum_char() const {
            if (nuc_ && label_.size() > 1) {
                char c = label_[1];
                return (c >= 'a' && c <= 'z') ? char(c - 'a' + 'A') : c;
            }
            return label_[0];
        }

        // subscript assignment for the statement currently being printed, keyed by label.
        // keyed by LABEL rather than by Line: two lines with the same label are the same
        // index and must share a subscript (spin-blocked equations reuse a label across
        // blocks, and those must not be split apart).
        static inline thread_local std::unordered_map<std::string, char> subscript_map_{};

        char einsum_char() const {
            auto it = subscript_map_.find(label_);
            if (it != subscript_map_.end()) return it->second;
            return natural_einsum_char(); // no statement in scope (single-line/debug printing)
        }

        // Assign a distinct subscript to every distinct label in one statement. Each label
        // keeps its natural character where that character is unclaimed -- so every
        // statement that does not actually collide prints exactly as it did before -- and
        // otherwise takes a fresh one from the free pool. Deterministic: labels are
        // considered in order of first appearance.
        static void assign_subscripts(const std::vector<Line> &lines) {
            subscript_map_.clear();

            std::vector<std::pair<std::string, char>> order; // label -> natural char
            for (const auto &line : lines) {
                if (line.label_.empty()) continue;
                bool seen = false;
                for (const auto &p : order)
                    if (p.first == line.label_) { seen = true; break; }
                if (!seen) order.emplace_back(line.label_, line.natural_einsum_char());
            }

            auto is_natural_of_some_label = [&order](char c) {
                for (const auto &p : order)
                    if (p.second == c) return true;
                return false;
            };

            std::set<char> taken;
            for (const auto &[label, nat] : order) {
                if (!taken.count(nat)) { // natural char still free -- keep it (the common case)
                    subscript_map_[label] = nat;
                    taken.insert(nat);
                    continue;
                }
                // collided with an earlier label: take a character that is neither already
                // assigned nor the natural character of any OTHER label in this statement.
                static constexpr char pool[] = "pqrstuwxyzPSTWZabcdefghijklmnovABCDEFGHIJKLMNOQRUVXY";
                char fresh = '\0';
                for (const char *c = pool; *c; ++c) {
                    if (taken.count(*c) || is_natural_of_some_label(*c)) continue;
                    fresh = *c;
                    break;
                }
                if (fresh == '\0')
                    throw std::runtime_error("Line::assign_subscripts: subscript pool exhausted for '"
                                             + label + "' (statement has too many distinct indices)");
                subscript_map_[label] = fresh;
                taken.insert(fresh);
            }
        }

        // clears the assignment when the statement goes out of scope, so a stale map can
        // never leak into an unrelated print.
        struct SubscriptScope {
            explicit SubscriptScope(const std::vector<Line> &lines) { assign_subscripts(lines); }
            ~SubscriptScope() { subscript_map_.clear(); }
            SubscriptScope(const SubscriptScope &) = delete;
            SubscriptScope &operator=(const SubscriptScope &) = delete;
        };

        bool empty() const {
            return label_.empty();
        }

        uint_fast8_t size() const {
            return label_.size();
        }

    };

    /// *** Hash functions *** ///

    // struct for comparing lines while ignoring the label
    struct line_compare {
        bool operator()(const Line &left, const Line &right) const {
            return left.same_kind(right);
//            return left < right;
        }

        bool operator()(const Line *left, const Line *right) const {
            if (!left || !right) return !right;
            else return left->same_kind(*right);
//            else return left->operator<(*right);
        }
    };

    // define a vector of lines
    typedef std::vector<Line, std::allocator<Line>> line_vector;

} // namespace pdaggerq

// declare hash functions for Line class
namespace pdaggerq {
    struct LineHash {
        uint_fast16_t operator()(const Line &line) const {

            // we can store each boolean as a bit in an integral type (5 bits)
            uint16_t hash = line.o_;
            hash |= line.a_ << 1;
            hash |= line.sig_ << 2;
            hash |= line.den_ << 3;
            hash |= line.nuc_ << 4;

            // store the first character of the label and return
            return hash << 8 | line.label_[0];
        }

        size_t operator()(const Line *line) const {
            constexpr LineHash line_hash;

            // check if the pointer is null
            if (!line) return 0;

            // otherwise, return the hash of the line
            return line_hash(*line);
        }

        /**
         * maps one set of lines to another
         * @param old_lines the old lines
         * @param new_lines the new lines
         * @return a map of the old lines to the new lines
         */
        static std::unordered_map<Line, Line, LineHash> map_lines(const line_vector &old_lines,
                                                                  const line_vector &new_lines) {

            std::unordered_map<Line, Line, LineHash> line_map;
            line_map.reserve(old_lines.size() + new_lines.size());

            // we want to map the old lines to the new lines
            // so (a,b,i,j) -> (c,d,j,i) would be a map of a->c, b->d, i->j, j->i
            // (a,b) -> (c,d,i,j) would be a map of a->c, b->d and i->i, j->j
            // (a,b,i,j) -> (c,d) would be a map of a->c, b->d, i->i, j->j

            // first map all old lines to themselves
            for (const Line &line : old_lines) {
                line_map[line] = line;
            }

            // then map all new lines to themselves
            for (const Line &line : new_lines) {
                line_map[line] = line;
            }

            // then map the old lines to the new lines
            for (size_t i = 0; i < old_lines.size() && i < new_lines.size(); ++i) {
                line_map[old_lines[i]] = new_lines[i];
            }

            // return the map
            return line_map;
        }
    };
    struct LineEqual {
        bool operator()(const Line &lhs, const Line &rhs) const {

            // check equality of the lines
            return lhs == rhs;
        }

        bool operator()(const Line *lhs, const Line *rhs) const {

            // check if either pointer is null
            if (!lhs || !rhs) return rhs == lhs;

            // check equality of the pointers
            return *lhs == *rhs;
        }
    };
    struct LinePropHash {
        uint_fast8_t operator()(const Line &line) const {

            // we can store each boolean as a bit in an integral type (5 bits)
            uint16_t hash = line.o_;
            hash |= line.a_ << 1;
            hash |= line.sig_ << 2;
            hash |= line.den_ << 3;
            hash |= line.nuc_ << 4;

            // return the hash (5 bits)
            return hash;
        }

        uint_fast8_t operator()(const Line *line) const {

            if (!line) return -1; // ignore null pointers (return -1)

            constexpr uint_fast8_t shift = 1;

            //  We do not care about the label for this hash function.
            uint_fast8_t hash = line->o_ << shift;

            // store each boolean as a bit in an integral type
            hash |= line->a_;
            hash = (hash << shift) | line->sig_;
            hash = (hash << shift) | line->den_;

            // return the hash (5 bits)
            return (hash << shift) | line->nuc_;
        }
    };
    struct LinePropEqual {
        bool operator()(const Line &lhs, const Line &rhs) const {
            return lhs.equivalent(rhs);
        }

        bool operator()(const Line *lhs, const Line *rhs) const {
            if (!lhs || !rhs) return rhs == lhs;
            return lhs->equivalent(*rhs);
        }
    };
} // namespace pdaggerq

#endif //PDAGGERQ_LINE_HPP
