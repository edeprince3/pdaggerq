//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: linkage_set.hpp
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

#ifndef PDAGGERQ_LINKAGE_SET_HPP
#define PDAGGERQ_LINKAGE_SET_HPP
#include "linkage.h"
#include <functional>

using std::string;
using std::hash;

namespace pdaggerq {

    /**
    * hash function class for linkages
    */
    struct LinkageHash {
        size_t operator()(const ConstLinkagePtr &linkage_ptr) const {
            constexpr SimilarVertexPtrHash sim_vert_hash;
            constexpr std::hash<string> string_hasher;

            // hash every vertex in the linkage
            const vector<ConstVertexPtr> &vertices = linkage_ptr->vertices();

            string hash_str;

            // vertex hash
            for (const ConstVertexPtr &vertex : vertices) {
                if (vertex.get() != nullptr && !vertex->empty())
                    hash_str += to_string(sim_vert_hash(vertex));
            }

            // add connection hash
            for (const auto& [left_idx, right_idx] : linkage_ptr->connec_){
                hash_str += to_string(left_idx);
                hash_str += to_string(right_idx);
            }

            // add disconnection hash
            for (const uint_fast8_t index: linkage_ptr->disconnec_){
                hash_str += to_string(index);
            }

            // return string hash
            return string_hasher(hash_str);

        }
    }; // struct linkage_hash

    struct LinkagePred {
        bool operator()(const ConstLinkagePtr &lhs, const ConstLinkagePtr &rhs) const {
            return *lhs == *rhs;
        }
    }; // struct linkage_pred

    class linkage_set{

        mutable std::mutex mtx_; // mutex for thread safety
        std::unordered_set<ConstLinkagePtr, LinkageHash, LinkagePred> linkages_; // set of linkages

    public:
        /**
         * constructor
         */
        linkage_set() : linkages_(256) {}

        /**
         * constructor with initial bucket n_ops
         * @param size initial n_ops of the set
         */
        explicit linkage_set(size_t size) : linkages_(size) {}

        /**
         * copy constructor
         * @param other linkage set to copy
         */
        linkage_set(const linkage_set &other){
            std::lock_guard<std::mutex> lock(mtx_);
            linkages_ = other.linkages_;
        }

        /**
         * move constructor
         * @param other linkage set to move
         */
        linkage_set(linkage_set &&other) noexcept {
            linkages_ = std::move(other.linkages_);
        }

        /**
         * copy assignment operator
         * @param other linkage set to copy
         * @return reference to this
         */
        linkage_set &operator=(const linkage_set &other){
            std::lock_guard<std::mutex> lock(mtx_);
            linkages_ = other.linkages_;
            return *this;
        };

        /**
         * move assignment operator
         * @param other linkage set to move
         * @return reference to this
         */
        linkage_set &operator=(linkage_set &&other) noexcept{
            linkages_ = std::move(other.linkages_);
            return *this;
        }

        /**
         * destructor
         */
        ~linkage_set() = default;

        /**
         * insert a linkage into the set
         * @param linkage linkage to insert
         */
        auto insert(const ConstLinkagePtr &linkage) {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_.insert(linkage);
        }

        /**
         * check if a linkage is in the set
         * @param linkage linkage to check
         * @return true if linkage is in set
         */
        bool contains(const LinkagePtr &linkage) const {
            auto equal_range = linkages_.equal_range(linkage);
            return equal_range.first != equal_range.second;
        }

        /**
         * get the number of linkages in the set
         * @return number of linkages
         */
        size_t size() const { return linkages_.size(); }

        /**
         * get the set of linkages
         * @return set of linkages
         */
        const std::unordered_set<ConstLinkagePtr, LinkageHash, LinkagePred> &linkages() const { return linkages_; }

        /**
         * clear the set of linkages
         */
        void clear() { linkages_.clear(); }

        /**
         * test if the set is empty
         * @return true if the set is empty
         */
        bool empty() const { return linkages_.empty(); }

        /**
         * begin iterator for set of linkages
         */
        auto begin() const { return linkages_.begin(); }

        /**
         * end iterator for set of linkages
         */
        auto end() const { return linkages_.end(); }

        /**
         * find a linkage in the set
         * @param linkage linkage to find
         * @return iterator to linkage in set
         */
        auto find(const LinkagePtr &linkage) const { return linkages_.find(linkage); }

        /**
         * const overload [] operator
         * @param i index of linkage
         * @return const reference to linkage
         */
        const ConstLinkagePtr &operator[](size_t i) const { return *next(linkages_.begin(), (long) i); }


        /**
         * get reference to linkage in set by value from [] operator
         * @param linkage linkage to get reference to
         * @return reference to linkage
         */
        const ConstLinkagePtr &operator[](const ConstLinkagePtr &linkage) const {
            auto loc = linkages_.equal_range(linkage);
            return *loc.first;
        }


        /**
         * overload + operator
         * @param other linkage set to add
         * @return new linkage set
         */
        linkage_set operator+(const linkage_set &other) const {
            linkage_set new_set = *this; // new linkage set
            new_set.linkages_.insert(other.linkages_.begin(), other.linkages_.end());
            return new_set; // return new linkage set
        }

        /**
         * overload - operator
         * @param other linkage set to remove from this
         * @return new linkage set
         */
        linkage_set operator-(const linkage_set &other) const {
            linkage_set new_set = *this; // new linkage set
            for (auto it = new_set.linkages_.begin(); it != new_set.linkages_.end();) {
                auto equal_range = other.linkages_.equal_range(*it);
                for (auto other_it = equal_range.first; other_it != equal_range.second; other_it++){
                    it = new_set.linkages_.erase(it);
                }
            }
            return new_set; // return new linkage set
        }


        /**
         * overload += operator
         * @param other linkage set to add
         * @return reference to this
         */
        linkage_set &operator+=(const linkage_set &other) {
            std::lock_guard<std::mutex> lock(mtx_);
            linkages_.insert(other.linkages_.begin(), other.linkages_.end());
            return *this; // return this
        }

        /**
         * overload -= operator
         * @param other linkage set to remove from this
         * @return reference to this
         */
        linkage_set &operator-=(const linkage_set &other) {
            std::lock_guard<std::mutex> lock(mtx_);

            for (auto it = linkages_.begin(); it != linkages_.end();) {
                auto equal_range = other.linkages_.equal_range(*it);
                if (equal_range.first == equal_range.second) {
                    it++; continue;
                }

                for (auto other_it = equal_range.first; other_it != equal_range.second; other_it++){
                    it = linkages_.erase(it);
                }
            }

            return *this; // return this
        }

    }; // class linkage_set

} // namespace pdaggerq


#endif //PDAGGERQ_LINKAGE_SET_HPP