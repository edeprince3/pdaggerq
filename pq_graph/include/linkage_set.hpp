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
#include <algorithm>
#include <atomic>
#include <functional>
#include <unordered_set>

#include "linkage.h"

using std::string;
using std::hash;

namespace pdaggerq {

    /**
    * hash function class for linkages
    */
    struct LinkageHash {
    public:
        LinkageHash() = default;

        size_t operator()(const LinkagePtr &linkage) const {
            constexpr hash<string> str_hash;
            return str_hash(linkage->base_name());
        }
    }; // struct linkage_hash

    struct LinkageEqual {
        bool operator()(const LinkagePtr &lhs, const LinkagePtr &rhs) const {
            return *lhs == *rhs;
        }
    }; // struct linkage_pred

    //templated typedef for a map with linkages as keys
    template<typename T>
    using linkage_map = std::unordered_map<LinkagePtr, T, LinkageHash, LinkageEqual>;

    // struct for parallel linkage set operations with deterministic iteration order
    class linkage_set {

        mutable std::mutex mtx_; // mutex for thread-safe mutations
        typedef std::unordered_set<LinkagePtr, LinkageHash, LinkageEqual> linkage_container;
        linkage_container linkages_; // hash set for O(1) dedup/lookup
        mutable linkage_vector sorted_; // sorted cache for deterministic iteration
        mutable std::atomic<bool> dirty_{false}; // true when sorted_ needs rebuilding

        // rebuilds sorted_ from linkages_; caller must hold mtx_
        void do_sort_() const {
            sorted_.clear();
            sorted_.reserve(linkages_.size());
            for (const auto &link : linkages_)
                sorted_.push_back(link);
            std::sort(sorted_.begin(), sorted_.end(), [](const LinkagePtr &a, const LinkagePtr &b) {
                return a->base_name() < b->base_name();
            });
            dirty_.store(false, std::memory_order_release);
        }

        // acquires mtx_ and rebuilds sorted_ if dirty
        void rebuild_sorted_() const {
            std::lock_guard<std::mutex> lock(mtx_);
            if (!dirty_.load(std::memory_order_relaxed)) return;
            do_sort_();
        }

    public:
        typedef linkage_vector::const_iterator const_iterator;

        linkage_set() : linkages_(256) {}

        explicit linkage_set(size_t size) : linkages_(size) {}

        linkage_set(const linkage_set &other){
            std::lock_guard<std::mutex> lock(other.mtx_);
            linkages_ = other.linkages_;
            dirty_.store(true, std::memory_order_relaxed);
        }

        linkage_set(linkage_set &&other) noexcept {
            std::lock_guard<std::mutex> lock(other.mtx_);
            linkages_ = std::move(other.linkages_);
            sorted_ = std::move(other.sorted_);
            dirty_.store(other.dirty_.load(std::memory_order_relaxed), std::memory_order_relaxed);
        }

        linkage_set &operator=(const linkage_set &other){
            if (this == &other) return *this;
            std::scoped_lock lock(mtx_, other.mtx_);
            linkages_ = other.linkages_;
            sorted_.clear();
            dirty_.store(true, std::memory_order_relaxed);
            return *this;
        };

        linkage_set &operator=(linkage_set &&other) noexcept{
            if (this == &other) return *this;
            std::scoped_lock lock(mtx_, other.mtx_);
            linkages_ = std::move(other.linkages_);
            sorted_ = std::move(other.sorted_);
            dirty_.store(other.dirty_.load(std::memory_order_relaxed), std::memory_order_relaxed);
            return *this;
        }

        ~linkage_set() = default;

        auto insert(const VertexPtr &linkage) {
            std::lock_guard<std::mutex> lock(mtx_);
            auto result = linkages_.insert(as_link(linkage));
            if (result.second) dirty_.store(true, std::memory_order_release);
            return result;
        }

        void insert(const typename linkage_vector::const_iterator &begin, const typename linkage_vector::const_iterator &end) {
            std::lock_guard<std::mutex> lock(mtx_);
            size_t old_size = linkages_.size();
            linkages_.insert(begin, end);
            if (linkages_.size() != old_size) dirty_.store(true, std::memory_order_release);
        }
        void insert(const typename vertex_vector::const_iterator &begin, const typename vertex_vector::const_iterator &end) {
            std::lock_guard<std::mutex> lock(mtx_);
            size_t old_size = linkages_.size();
            for (auto it = begin; it != end; ++it)
                linkages_.insert(as_link(*it));
            if (linkages_.size() != old_size) dirty_.store(true, std::memory_order_release);
        }

        size_t count(const LinkagePtr &linkage) const {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_.count(linkage);
        }

        bool contains(const LinkagePtr &linkage) const {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_.count(linkage) > 0;
        }

        size_t size() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_.size();
        }

        void clear() {
            std::lock_guard<std::mutex> lock(mtx_);
            linkages_.clear();
            sorted_.clear();
            dirty_.store(false, std::memory_order_relaxed);
        }

        void reserve(size_t n_ops) {
            std::lock_guard<std::mutex> lock(mtx_);
            linkages_.reserve(n_ops);
        }

        bool empty() const {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_.empty();
        }

        const_iterator begin() const {
            if (dirty_.load(std::memory_order_acquire))
                rebuild_sorted_();
            return sorted_.cbegin();
        }

        const_iterator end() const {
            if (dirty_.load(std::memory_order_acquire))
                rebuild_sorted_();
            return sorted_.cend();
        }

        // find element in sorted iteration order; only use in serial/critical contexts
        const_iterator find(const LinkagePtr &linkage) const {
            std::lock_guard<std::mutex> lock(mtx_);
            if (dirty_.load(std::memory_order_relaxed))
                do_sort_();
            if (linkages_.count(linkage) == 0)
                return sorted_.cend();
            LinkageEqual eq;
            return std::find_if(sorted_.cbegin(), sorted_.cend(),
                [&](const LinkagePtr &elem) { return eq(elem, linkage); });
        }

        const LinkagePtr &operator[](size_t i) const {
            if (dirty_.load(std::memory_order_acquire))
                rebuild_sorted_();
            return sorted_[i];
        }

        const LinkagePtr &operator[](const LinkagePtr &linkage) const {
            if (dirty_.load(std::memory_order_acquire))
                rebuild_sorted_();
            LinkageEqual eq;
            auto it = std::find_if(sorted_.cbegin(), sorted_.cend(),
                [&](const LinkagePtr &elem) { return eq(elem, linkage); });
            return *it;
        }

        linkage_set operator+(const linkage_set &other) const {
            std::lock_guard<std::mutex> lock(mtx_);
            linkage_set new_set;
            new_set.linkages_ = linkages_;
            for (const auto &linkage: other.linkages_) new_set.linkages_.insert(linkage);
            new_set.dirty_.store(true, std::memory_order_relaxed);
            return new_set;
        }

        linkage_set operator-(const linkage_set &other) const {
            std::lock_guard<std::mutex> lock(mtx_);
            linkage_set new_set;
            new_set.linkages_ = linkages_;
            for (const auto &linkage: other.linkages_) {
                new_set.linkages_.erase(linkage);
            }
            new_set.dirty_.store(true, std::memory_order_relaxed);
            return new_set;
        }

        linkage_set &operator+=(const linkage_set &other) {
            std::lock_guard<std::mutex> lock(mtx_);
            size_t old_size = linkages_.size();
            linkages_.insert(other.linkages_.begin(), other.linkages_.end());
            if (linkages_.size() != old_size) dirty_.store(true, std::memory_order_release);
            return *this;
        }

        linkage_set &operator-=(const linkage_set &other) {
            std::lock_guard<std::mutex> lock(mtx_);
            for (const auto &linkage: other.linkages_)
                linkages_.erase(linkage);
            dirty_.store(true, std::memory_order_release);
            return *this;
        }

        size_t erase(const LinkagePtr &linkage) {
            std::lock_guard<std::mutex> lock(mtx_);
            size_t result = linkages_.erase(linkage);
            if (result > 0) dirty_.store(true, std::memory_order_release);
            return result;
        }

        bool operator==(const linkage_set &other) const {
            std::lock_guard<std::mutex> lock(mtx_);
            return linkages_ == other.linkages_;
        }

    }; // class linkage_set

} // namespace pdaggerq


#endif //PDAGGERQ_LINKAGE_SET_HPP
