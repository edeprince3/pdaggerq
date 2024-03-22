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
#include <unordered_set>

using std::string;
using std::hash;

namespace pdaggerq {

    /**
    * hash function class for linkages
    */
    struct LinkageHash {
        size_t operator()(const ConstLinkagePtr &linkage_ptr) const {

            constexpr SimilarVertexPtrHash vertex_hash; // hash function for vertices
            constexpr size_t magic_golden_ratio = 0x9e3779b9; // the golden ratio of hashing; prevents collisions

            const Linkage &linkage = *linkage_ptr;

            // blend the hashes of the left and right vertices
            size_t left_vert_hash  = vertex_hash(linkage.left());
            size_t right_vert_hash = vertex_hash(linkage.right());
            size_t total_hash = (left_vert_hash ^ right_vert_hash) + magic_golden_ratio;

            size_t connection_hash = 0;
            for (const auto &[leftidx, rightidx] : linkage.connec_map()) {
                // create hash from leftidx and rightidx
                constexpr std::hash<int_fast8_t> hasher;
                size_t left_hash = hasher(leftidx);
                size_t right_hash = hasher(rightidx);

                // blend them together
                connection_hash ^= (left_hash ^ right_hash) + magic_golden_ratio + (connection_hash << 6) + (connection_hash >> 2);
            }

            total_hash ^= connection_hash + magic_golden_ratio + (total_hash << 6) + (total_hash >> 2);
            return total_hash;

        }
    }; // struct linkage_hash

    struct LinkageEqual {
        bool operator()(const ConstLinkagePtr &lhs, const ConstLinkagePtr &rhs) const {
            return *lhs == *rhs;
        }
    }; // struct linkage_pred

    struct LinkagePermutedEqual {
        bool operator()(const ConstLinkagePtr &lhs, const ConstLinkagePtr &rhs) const {
            return lhs->permuted_equals(*rhs).first;
        }
    }; // struct linkage_pred

    class linkage_set {

        mutable std::mutex mtx_; // mutex for thread safety
        typedef std::unordered_set<ConstLinkagePtr, LinkageHash, LinkageEqual> linkage_container;
        linkage_container linkages_; // set of linkages

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
        bool contains(const LinkagePtr &linkage) const { return linkages_.find(linkage) != linkages_.end(); }

        /**
         * get the number of linkages in the set
         * @return number of linkages
         */
        size_t size() const { return linkages_.size(); }

        /**
         * get the set of linkages
         * @return set of linkages
         */
        const linkage_container &linkages() const { return linkages_; }

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
        const ConstLinkagePtr &operator[](const ConstLinkagePtr &linkage) const { return *linkages_.find(linkage); }

        /**
         * overload + operator
         * @param other linkage set to add
         * @return new linkage set
         */
        linkage_set operator+(const linkage_set &other) const {
            linkage_set new_set = *this; // new linkage set
            for (const auto &linkage: other.linkages_) new_set.insert(linkage); // insert other set
            return new_set; // return new linkage set
        }

        /**
         * overload - operator
         * @param other linkage set to remove from this
         * @return new linkage set
         */
        linkage_set operator-(const linkage_set &other) const {
            linkage_set new_set = *this; // new linkage set
            for (const auto &linkage: other.linkages_) new_set.linkages_.erase(linkage); // remove other set
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
            for (const auto &linkage: other.linkages_)
                linkages_.erase(linkage); // remove other set
            return *this; // return this
        }

        /**
         * erase a linkage from the set
         * @param linkage linkage to erase
         */
        void erase(const ConstLinkagePtr &linkage) {
            std::lock_guard<std::mutex> lock(mtx_);
            linkages_.erase(linkage);
        }

        /**
         * overload == operator
         * @param other linkage set to compare
         * @return true if the sets are equal
         */
        bool operator==(const linkage_set &other) const {
            return linkages_ == other.linkages_;
        }

    }; // class linkage_set

} // namespace pdaggerq


#endif //PDAGGERQ_LINKAGE_SET_HPP