//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: scaling_map.hpp
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

#ifndef PDAGGERQ_SCALING_MAP_HPP
#define PDAGGERQ_SCALING_MAP_HPP

#include <algorithm>
#include <functional>
#include <map>
#include <stdexcept>
#include "line.hpp"
#include <clocale>
#include <sstream>

using std::pair;
using std::vector;
using std::string;
using std::to_string;
using std::map;
using std::hash;
using std::size_t;

namespace pdaggerq {

    struct shape {
        uint_fast8_t n_ = 0; // number of lines

        //TODO: split this into two variables (oa, ob, va, vb); use a function to get their sum.
        pair<uint_fast8_t, uint_fast8_t> o_{}; // pair of spin up/down occupied lines
        pair<uint_fast8_t, uint_fast8_t> v_{}; // pair of spin up/down virtual lines
        uint_fast8_t L_ = 0; // sigma index
        uint_fast8_t Q_ = 0; // density index

        shape() : o_({0,0}), v_({0,0}), L_(0), Q_(0) {}

        shape(const line_vector &lines) {
            for (const Line &line : lines)
                *this += line;
        }

        void operator+=(const shape & other) {
            n_ += other.n_;
            o_.first  += other.o_.first;
            o_.second += other.o_.second;
            v_.first  += other.v_.first;
            v_.second += other.v_.second;
            L_ += other.L_;
            Q_ += other.Q_;
        }

        shape(const shape &other) = default;
        shape(shape &&other) = default;
        shape &operator=(const shape &other) = default;
        shape &operator=(shape &&other) = default;
        ~shape() = default;

        bool operator==(const shape & other) const {
            return n_ == other.n_ && o_ == other.o_ && v_ == other.v_ && L_ == other.L_ && Q_ == other.Q_;
        }
        bool operator!=(const shape & other) const {
            return !(*this == other);
        }

        string str() const {
            string result;
            result.reserve(n_);

            result += 'o';
            result += to_string(o_.first + o_.second);

            result += 'v';
            result += to_string(v_.first + v_.second);

            if (L_ > 0) {
                result += 'L';
                result += to_string(L_);
            }
            if (Q_ > 0) {
                result += 'Q';
                result += to_string(Q_);
            }
            return result;
        }

        bool operator<( const shape & other) const {

            /// priority: o_ + v_ + L_, v_ + L_, L_, v_, va, oa

            size_t oa = o_.first, ob = o_.second, otot = oa + ob,
                   va = v_.first, vb = v_.second, vtot = va + vb;
            size_t other_oa = other.o_.first, other_ob = other.o_.second, other_otot = other_oa + other_ob,
                   other_va = other.v_.first, other_vb = other.v_.second, other_vtot = other_va + other_vb;

            // prioritize total scaling over individual scaling factors
            if (n_ != other.n_)
                return n_ < other.n_;

            /// if total scaling is the same, prioritize individual scaling factors
            if (Q_ + other.Q_ > 0) {
                // prioritize sum of v_ and L_ and Q_ over individual L_ and v_ and Q_
                uint_fast8_t sum = vtot + L_ + Q_;
                uint_fast8_t other_sum = other_vtot + other.L_ + other.Q_;
                if (sum != other_sum)
                    return sum < other_sum;

                // if sum of v_ and L_ and Q_ is the same, prioritize Q_ over L_ and v_
                if (Q_ != other.Q_)
                    return Q_ < other.Q_;
            }

            if (L_ + other.L_ > 0) {
                // prioritize sum of v_ and L_ over individual L_ and v_
                uint_fast8_t sum = vtot + L_;
                uint_fast8_t other_sum = other_vtot + other.L_;
                if (sum != other_sum)
                    return sum < other_sum;

                // if sum of v_ and L_ is the same, prioritize L_ over v_
                if (L_ != other.L_)
                    return L_ < other.L_;
            }

            // prioritize v_ over o_
            if (vtot != other_vtot) return vtot < other_vtot;

            // prioritize o over spin
            if (otot != other_otot) return otot < other_otot;

            // prioritize va over vb
//            if (va != other_va) return va < other_va; // messes up printing TODO: fix this

            // prioritize oa over ob
//            if (oa != other_oa) return oa < other_oa;

            // equal or greater scaling, return false
            return false; 
        }
        bool operator>( const shape & other) const {
            return other < *this;
        }
        bool operator<=(const shape & other) const {
            if (*this == other) return true;
            return *this < other;
        }
        bool operator>=(const shape & other) const {
            if (*this == other) return true;
            return other < *this;
        }

        shape operator+(const shape & other) const {
            shape result = *this;
            result += other;
            return result;
        }

        void operator-=(const shape & other) {
            o_.first  = (o_.first  < other.o_.first)  ? 0 : o_.first  - other.o_.first;
            o_.second = (o_.second < other.o_.second) ? 0 : o_.second - other.o_.second;
            v_.first  = (v_.first  < other.v_.first)  ? 0 : v_.first  - other.v_.first;
            v_.second = (v_.second < other.v_.second) ? 0 : v_.second - other.v_.second;
            L_ = (L_ < other.L_) ? 0 : L_ - other.L_;
            Q_ = (Q_ < other.Q_) ? 0 : Q_ - other.Q_;

            n_ = (o_.first + o_.second) + (v_.first + v_.second) + L_ + Q_;
        }

        shape operator-(const shape &other) const {
            shape result = *this;
            result -= other;
            return result;
        }

        void operator+=(const Line &line) {
            ++n_; // increment number of lines

            if (line.sig_) { ++L_; return; } // sigma
            if (line.den_) { ++Q_; return;  } // density

            if (line.o_) { // occupied
                if (line.a_) ++o_.first;
                else ++o_.second; // default for no-spin is beta
                return;
            } else { // virtual
                if (line.a_) ++v_.first;
                else ++v_.second; // default for no-spin is beta
                return;
            }
        }

        shape operator+(const Line &line) const {
            shape result = *this;
            result += line;
            return result;
        }

    };

    struct scale_metric {

        /**
         * Compare two costs
         * @param left_scale prior_links cost
         * @param right_scale next_link cost
         * @return true if prior_links cost is greater than next_link cost
         */
        bool operator()(const shape &left_scale, const shape &right_scale) const {
            return right_scale < left_scale;
        }

    }; // struct scale_metric

    /**
     * Class to store a sorted map of number of virtuals and occupieds in a linkage paired with occurrence
     * The map is sorted by (o + v)^2 + v
     */
    struct scaling_map {

        map<shape, long int, scale_metric> map_; // map of (v, o) to occurrence

        /**
         * Constructors
         */
        explicit scaling_map() = default;
        scaling_map(const scaling_map &other) = default;
        scaling_map(scaling_map &&other) = default;

        /**
         * Destructor
         */
        ~scaling_map() = default;

        /**
         * Default Assignments
         */
        scaling_map &operator=(const scaling_map &other) = default;
        scaling_map &operator=(scaling_map &&other) = default;

        /**
         * Get reference to a shape in the map. Add the shape if it is not in the map with 0 occurrence.
         * @param vopair pair of virtuals and occupieds
         * @return reference to the occurrence of the shape
         */
        long int &operator[](const shape &vopair) {
            return map_[vopair]; // add if not in map (default value is 0). return reference to value
        }

        /**
         * Get const reference to a shape in the map. Add the shape if it is not in the map with 0 occurrence.
         * @param vopair pair of virtuals and occupieds
         * @return reference to the occurrence of the shape
         */
        const long int &operator[](const shape &vopair) const {
            auto it = map_.find(vopair);
            if (it == map_.end()){
                static long int zero = 0;
                return zero; // return 0 if not in map
            }
            return it->second; // return value if in map
        }

        /**
         * Get begin iterator of the map
         * @return begin iterator
         */
        auto begin() const { return map_.begin(); }

        /**
         * Get end iterator of the map
         * @return end iterator
         */
        auto end() const { return map_.end(); }

        /**
         * Get const begin iterator of the map
         */
        auto cbegin() const { return map_.cbegin(); }

        /**
         * Get const end iterator of the map
         */
        auto cend() const { return map_.cend(); }

        /**
         * get worst scaling with non-zero value (usually first element in the map; sorted by descending scaling)
         */
        shape worst() const {
            // while the first element in the map has zero occurrences, increment the iterator
            if (map_.empty()) // return empty shape if map is empty
                return {};

            auto worst_pos = map_.begin();
            auto worst_end = map_.end();
            bool at_end = worst_pos == worst_end;
            while (worst_pos->second == 0 && !at_end) {
                worst_pos++;
                at_end = worst_pos == worst_end;
            }

            if (at_end)
                 return {}; // return empty shape if all scalings are zero
            else return worst_pos->first; // return first non-zero scaling
        }

        /**
         * Get the number of elements in the map
         * @return number of elements
         */
        size_t size() const { return map_.size(); }

        /**
         * Check if the map is empty
         * @return true if the map is empty, false otherwise
         */
        bool empty() const { return map_.empty(); }

        /**
         * get number of linkages
         * @return number of linkages
         */
        long int total() const {
            long int num = 0;
            for (const auto &it: map_) num += it.second;
            return num;
        }

        /**
         * clear the map
         */
        void clear() { map_.clear(); }

        /// initialize values for making comparison
        constexpr static int this_better = 1;
        constexpr static int is_same = 0;
        constexpr static int this_worse = -1;

        /**
         * Compare scaling maps to determine best scaling term
         * @param left prior_links scaling map
         * @param other_map other scaling map
         * @return 1 if this is cheaper than other
         *         0 if this is the same as other
         *        -1 if this is more expensive than other
         * @note scaling maps are assumed to be ordered by descending scaling.
         *      If the first scaling in the map is the same, the second scaling is compared and so on.
         *      The first scaling that is different determines the winner.
         *      If all scaling is the same, the maps are considered equal and false is returned.
         */
        static inline int compare_scaling(const scaling_map& this_map, const scaling_map &other_map) {

            // initialize this_map iterators
            auto this_begin = this_map.begin();
            auto this_end = this_map.end();

            // initialize other_map iterators
            auto other_begin = other_map.begin();
            auto other_end = other_map.end();

            // initialize metrics
            size_t this_metric;
            size_t other_metric;

            // iterate over scaling maps
            for (auto this_it = this_begin, other_it = other_begin; // initialize iterators
                 this_it != this_end && other_it != other_end; // check if iterators are valid
                 this_it++, other_it++) { // increment iterators

                // skip zero occurrences
                while (this_it->second == 0 && this_it != this_end) this_it++;
                while (other_it->second == 0 && other_it != other_end) other_it++;

                bool at_this_end = this_it == this_end;
                bool at_other_end = other_it == other_end;
                if (!at_this_end && at_other_end)
                    return this_better; // this is cheaper (other has additional scalings)
                else if (at_this_end && !at_other_end)
                    return this_worse; // this is more expensive (this has additional scalings)
                else if (at_this_end) // at_other_end must also be true
                    return is_same; // this is the same (both have same number of scalings)
                // else continue to next scaling

                const shape &this_size = this_it->first;
                size_t this_occurrence = this_it->second;

                const shape &other_size = other_it->first;
                size_t other_occurrence = other_it->second;

                if (this_size < other_size) return this_better;
                if (this_size > other_size) return this_worse;

                // if this_size == other_size
                if (this_occurrence < other_occurrence) return this_better;
                if (this_occurrence > other_occurrence) return this_worse;

                // if this_occurrence == other_occurrence then continue to next scaling
            }

            return is_same;
        }

        /**
         * Compare this scaling map to another scaling map
         * @param other_map other scaling map
         * @return 1 if this is cheaper than other
         *         0 if this is the same as other
         *        -1 if this is more expensive than other
         */
        int compare(const scaling_map &other_map) const {
            return compare_scaling(*this, other_map);
        }

        /**
         * overload operator < for scaling_map
         * @param other other scaling_map
         * @return true if this is scaling map is cheaper than other
         */
        bool operator<(const scaling_map &other) const {
            return compare_scaling(*this, other) == this_better;
        }

        /**
         * overload operator > for scaling_map
         * @param other other scaling_map
         * @return true if this is scaling map is more expensive than other
         */
        bool operator>(const scaling_map &other) const {
            return compare_scaling(*this, other) == this_worse;
        }

        /**
         * overload operator == for scaling_map
         * @param other other scaling_map
         * @return true if this is scaling map is the same cost as other
         */
        bool operator==(const scaling_map &other) const {
            return compare_scaling(*this, other) == is_same;
        }

        /**
         * overload operator != for scaling_map
         * @param other other scaling_map
         * @return true if this is scaling map is not the same cost as other
         */
        bool operator!=(const scaling_map &other) const {
            return compare_scaling(*this, other) != is_same;
        }

        /**
         * overload operator <= for scaling_map
         * @param other other scaling_map
         * @return true if this is scaling map is cheaper or the same cost as other
         */
        bool operator<=(const scaling_map &other) const {
            return compare_scaling(*this, other) >= is_same;
        }

        /**
         * overload operator >= for scaling_map
         * @param other other scaling_map
         * @return true if this is scaling map is more expensive or the same cost as other
         */
        bool operator>=(const scaling_map &other) const {
            return compare_scaling(*this, other) <= is_same;
        }

        /**
         * overload operator + for scaling_map
         * @param other other scaling_map
         * @return new map with the sum of the two maps
         */
        scaling_map operator+(const scaling_map &other) const {
            scaling_map result = *this;
            for (const auto & it : other.map_) result[it.first] += it.second; // add other map

            return result;
        }

        /**
         * overload operator - for scaling_map
         * @param other other scaling_map
         * @return new map with the difference of the two maps
         */
        scaling_map operator-(const scaling_map &other) const {
            scaling_map result = *this;
            for (const auto & [scale, count] : other.map_)
                result[scale] -= count; // subtract other map

            return result;
        }

        /**
         * overload operator += for scaling_map
         * @param other other scaling_map
         * @return this map with the sum of the two maps
         */
        inline scaling_map& operator+=(const scaling_map &other) {
            for (const auto & it : other.map_)  map_[it.first] += it.second; // add other map
            return *this;
        }

        /**
         * overload operator -= for scaling_map
         * @param other other scaling_map
         * @return this map with the difference of the two maps
         */
        inline scaling_map& operator-=(const scaling_map &other) {
            for (const auto & it : other.map_) map_[it.first] -= it.second; // subtract other map
            return *this;
        }

        /**
         * overload for printing keys and elements
         * @param os output stream
         */
        friend std::ostream& operator<<(std::ostream& os, const scaling_map& map) {
            os << "{ ";
            bool printed = false;
            std::stringstream output;
            for (const auto &[scale, count]: map.map_) {
                if (count == 0) continue;
                output << scale.str() << ": " << count << ", ";
                printed = true;
            }
            if (printed) {
                std::string output_str = output.str();
                output_str.pop_back(); output_str.pop_back();
                os << output_str;
            }
            os << " }";
            return os;
        }

    }; // class scaling_map
} // pdaggerq

#endif //PDAGGERQ_SCALING_MAP_HPP
