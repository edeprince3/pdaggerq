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
#include <functional>
#include <unordered_set>

#include "linkage.h"

using std::string;
using std::hash;

namespace pdaggerq {

    // TODO: make this work for vertices in addition to linkages (different hash functions)

    /**
    * hash function class for linkages
    */
    struct LinkageHash {
    private:

    public:
        LinkageHash() = default;

        size_t operator()(const ConstLinkagePtr &linkage) const {
            constexpr hash<string> str_hash;
            size_t totalhash = str_hash(string_hash(linkage));
            return totalhash;
        }

        string string_hash(const ConstLinkagePtr &linkage) const {

            // get base name of linkage
            string total_string = linkage->base_name() + ' ';
            total_string.reserve(512); // reserve ample space for string

            // get string of connection map between left and right vertices
            for (const auto &[leftidx, rightidx] : linkage->connec_map()) {
                total_string += leftidx + '0'; // convert to int
                total_string += '>';
                total_string += rightidx + '0'; // convert to int
                total_string += ' ';
            }
            total_string += ' ';

            // get hashes of the lines
            for (const auto &line : linkage->lines()) {
                total_string += line.type();
                total_string += line.block();
            }
            total_string += ' ';

            if (!linkage->left()->empty() && linkage->left()->is_linked())
                total_string += string_hash(as_link(linkage->left()));
            else {
                total_string += linkage->left()->base_name() + ' ';
                for (const auto &line : linkage->left()->lines()) {
                    total_string += line.type();
                    total_string += line.block();
                }
                total_string += ' ';
            }

            total_string += linkage->is_addition() ? '+' : '*';

            if (!linkage->right()->empty() && linkage->right()->is_linked())
                total_string += string_hash(as_link(linkage->right()));
            else {
                total_string += linkage->right()->base_name() + ' ';
                for (const auto &line : linkage->right()->lines()) {
                    total_string += line.type();
                    total_string += line.block();
                }
                total_string += ' ';
            }

            // return the total string
            return total_string;

        }
    }; // struct linkage_hash

    struct LinkageEqual {
        bool operator()(const ConstLinkagePtr &lhs, const ConstLinkagePtr &rhs) const {
            return *lhs == *rhs;
        }
    }; // struct linkage_pred

    //templated typedef for a map with linkages as keys
    template<typename T>
    using linkage_map = std::unordered_map<ConstLinkagePtr, T, LinkageHash, LinkageEqual>;
    typedef std::vector<ConstLinkagePtr> linkage_vector;

    // struct for parallel linkage set operations
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
            std::lock_guard<std::mutex> lock(mtx_);
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
            std::lock_guard<std::mutex> lock(mtx_);
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
         * use iterator to insert a linkage into the set
         * @param linkages iterator to linkages
         */
        void insert(const typename linkage_vector::const_iterator &begin, const typename linkage_vector::const_iterator &end) {
            std::lock_guard<std::mutex> lock(mtx_);
            linkages_.insert(begin, end);
        }

        size_t count(const ConstLinkagePtr &linkage) const {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_.count(linkage);
        }

        /**
         * check if a linkage is in the set
         * @param linkage linkage to check
         * @return true if linkage is in set
         */
        bool contains(const LinkagePtr &linkage) const {
            return count(linkage) > 0;
        }

        /**
         * get the number of linkages in the set
         * @return number of linkages
         */
        size_t size() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_.size();
        }

        /**
         * get the set of linkages
         * @return set of linkages
         */
        const linkage_container &linkages() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_;
        }

        /**
         * clear the set of linkages
         */
        void clear() {
            std::lock_guard<std::mutex> lock(mtx_);
            linkages_.clear();
        }

        /**
         * reserve space for n_ops linkages
         */
        void reserve(size_t n_ops) {
            std::lock_guard<std::mutex> lock(mtx_);
            linkages_.reserve(n_ops);
        }

        /**
         * test if the set is empty
         * @return true if the set is empty
         */
        bool empty() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_.empty();
        }

        /**
         * begin iterator for set of linkages
         */
        auto begin() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_.begin();
        }

        /**
         * end iterator for set of linkages
         */
        auto end() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_.end();
        }

        /**
         * find a linkage in the set
         * @param linkage linkage to find
         * @return iterator to linkage in set
         */
        auto find(const LinkagePtr &linkage) const {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_.find(linkage);
        }
        auto find(const ConstLinkagePtr &linkage) const {
            return find(as_link(linkage->shallow()));
        }

        /**
         * const overload [] operator
         * @param i index of linkage
         * @return const reference to linkage
         */
        const ConstLinkagePtr &operator[](size_t i) const {
            std::lock_guard<std::mutex> lock(mtx_);
            return *next(linkages_.begin(), (long) i);
        }


        /**
         * get reference to linkage in set by value from [] operator
         * @param linkage linkage to get reference to
         * @return reference to linkage
         */
        const ConstLinkagePtr &operator[](const ConstLinkagePtr &linkage) const {
            std::lock_guard<std::mutex> lock(mtx_);
            return *linkages_.find(linkage);
        }

        /**
         * overload + operator
         * @param other linkage set to add
         * @return new linkage set
         */
        linkage_set operator+(const linkage_set &other) const {
            std::lock_guard<std::mutex> lock(mtx_);
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
            std::lock_guard<std::mutex> lock(mtx_);
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
        size_t erase(const ConstLinkagePtr &linkage) {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_.erase(linkage);
        }

        /**
         * overload == operator
         * @param other linkage set to compare
         * @return true if the sets are equal
         */
        bool operator==(const linkage_set &other) const {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_ == other.linkages_;
        }

    }; // class linkage_set

} // namespace pdaggerq


#endif //PDAGGERQ_LINKAGE_SET_HPP