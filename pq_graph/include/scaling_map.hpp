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
#include <clocale>
#include <sstream>

#include "shape.hpp"

using std::pair;
using std::vector;
using std::string;
using std::to_string;
using std::map;
using std::hash;
using std::size_t;

namespace pdaggerq {

    struct scale_metric {

        /**
         * Compare two costs
         * @param left_scale prior_links cost
         * @param right_scale next_link cost
         * @return true if prior_links cost is greater than next_link cost
         */
        bool operator()(const shape &left_scale, const shape &right_scale) const {
            return left_scale > right_scale;
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
        explicit scaling_map(const vector<shape>& shapes) {
            for (const shape &shape : shapes) map_[shape]++;
        }

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
        constexpr static int this_same = 0;
        constexpr static int this_worse = -1;

        /**
         * Compare scaling maps to determine best scaling term
         * @param this_map prior_links scaling map
         * @param other_map other scaling map
         * @return 1 if this is cheaper than other
         *         0 if this is the same as other
         *        -1 if this is more expensive than other
         * @note scaling maps are assumed to be ordered by descending scaling.
         *      If the first scaling in the map is the same, the second scaling is compared and so on.
         *      The first scaling that is different determines the winner.
         *      If all scaling is the same, the maps are considered equal and false is returned.
         */
        static int compare_scaling(const scaling_map& this_map, const scaling_map &other_map) {

            // initialize this_map iterators
            auto this_begin = this_map.begin();
            auto this_it = this_begin;
            auto this_end = this_map.end();

            // initialize other_map iterators
            auto other_begin = other_map.begin();
            auto other_it = other_begin;
            auto other_end = other_map.end();

            // check if either map is at the end
            bool this_at_end = this_it == this_end;
            bool other_at_end = other_it == other_end;

            // iterate over scaling maps
            do {

                // skip zero occurrences
                while ( this_it !=  this_end &&  this_it->second == 0 ) this_it++;
                while (other_it != other_end && other_it->second == 0 ) other_it++;

                if ( this_at_end && !other_at_end) return this_better; // this is cheaper (other has more scalings)
                if (!this_at_end &&  other_at_end) return this_worse; // this is more expensive (this has more scalings)
                if (this_at_end  &&  other_at_end) return this_same; // this is the same (equal scalings)

                // else compare current scaling

                // get shapes and occurrences
                const auto &[ this_shape,  this_occurrence] = *this_it;
                const auto &[other_shape, other_occurrence] = *other_it;


                if (this_shape < other_shape) {
                    // if other shape is worse, check if other_occurrence is negative
                    if (other_occurrence < 0)
                         return this_worse;
                    else return this_better;
                }
                if (this_shape > other_shape) {
                    // if this shape is worst, check if this_occurrence is negative
                    if (this_occurrence < 0)
                         return this_better;
                    else return this_worse;
                }

                // if this_shape == other_shape, test occurrence
                if (this_occurrence < other_occurrence) return this_better;
                if (this_occurrence > other_occurrence) return this_worse;

                // if this_occurrence == other_occurrence then continue to next scaling
                this_it++; other_it++;
            } while (this_it != this_end || other_it != other_end);

            return this_same;
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
            return compare_scaling(*this, other) == this_same;
        }

        /**
         * overload operator != for scaling_map
         * @param other other scaling_map
         * @return true if this is scaling map is not the same cost as other
         */
        bool operator!=(const scaling_map &other) const {
            return compare_scaling(*this, other) != this_same;
        }

        /**
         * overload operator <= for scaling_map
         * @param other other scaling_map
         * @return true if this is scaling map is cheaper or the same cost as other
         */
        bool operator<=(const scaling_map &other) const {
            return compare_scaling(*this, other) >= this_same;
        }

        /**
         * overload operator >= for scaling_map
         * @param other other scaling_map
         * @return true if this is scaling map is more expensive or the same cost as other
         */
        bool operator>=(const scaling_map &other) const {
            return compare_scaling(*this, other) <= this_same;
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
         * set any negative values to zero
         */
        void all_positive() {
            for (auto & [scale, count] : map_) {
                if (count < 0) count = 0;
            }
        }

        /**
         * merge scalings with different spins into a new scaling map
         * @return new scaling map with merged spins
         */
        scaling_map merge_spins() const {
            // create copies that ignore alpha/beta differences
            scaling_map no_spin_map;
            for (const auto & key : map_) {
                shape new_shape = key.first;
                new_shape.va_ = new_shape.v_; new_shape.oa_ = new_shape.o_;
                new_shape.vb_ = 0; new_shape.ob_ = 0;

                new_shape.a_ = new_shape.n_;
                new_shape.b_ = 0;
                no_spin_map[new_shape] += key.second;
            }
            return no_spin_map;
        }

        /**
         * overload for printing keys and elements
         * @param os output stream
         */
        friend std::ostream& operator<<(std::ostream& os, const scaling_map& map) {
            scaling_map no_spin_map = map.merge_spins();
            os << "{ ";
            bool printed = false;
            std::stringstream output;
            for (const auto &[scale, count]: no_spin_map.map_) {
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

        /**
         * print the scaling map to a string
         * @return string representation of the scaling map
         */
        string str() const {
            std::stringstream output;
            output << *this;
            return output.str();
        }

    }; // class scaling_map
} // pdaggerq

#endif //PDAGGERQ_SCALING_MAP_HPP
