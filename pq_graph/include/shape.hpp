//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: shape.hpp
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

#ifndef PDAGGERQ_SHAPE_HPP
#define PDAGGERQ_SHAPE_HPP

#include <sstream>
#include <clocale>
#include <stdexcept>
#include <map>
#include <functional>
#include <algorithm>
#include "line.hpp"

struct shape {
    uint_fast8_t n_ = 0; // number of lines

    //TODO: split this into two variables (oa, ob, va, vb); use a function to get their sum.
    uint_fast8_t oa_ = 0, ob_ = 0;
    uint_fast8_t va_ = 0, vb_ = 0;
    uint_fast8_t  o_ = 0,  v_ = 0;
    uint_fast8_t  a_ = 0,  b_ = 0;

    uint_fast8_t L_ = 0; // sigma index
    uint_fast8_t Q_ = 0; // density index


    // default constructors and assignments
    shape() = default;
    ~shape() = default;
    shape(const shape &other) = default;
    shape(shape &&other) = default;
    shape &operator=(const shape &other) = default;
    shape &operator=(shape &&other) = default;

    shape(const pdaggerq::line_vector &lines) {
        for (const pdaggerq::Line &line : lines)
            *this += line;
    }

    void operator+=(const shape & other) {
        n_  += other.n_;
        L_  += other.L_;
        Q_  += other.Q_;

        oa_ += other.oa_; ob_ += other.ob_;
        va_ += other.va_; vb_ += other.vb_;
        o_  = oa_ + ob_; v_  = va_ + vb_;
        a_  = oa_ + va_; b_  = ob_ + vb_;
    }

    void operator-=(const shape & other) {
        oa_  = (oa_  < other.oa_)  ? 0 : oa_  - other.oa_;
        ob_  = (ob_  < other.ob_)  ? 0 : ob_  - other.ob_;
        va_  = (va_  < other.va_)  ? 0 : va_  - other.va_;
        vb_  = (vb_  < other.vb_)  ? 0 : vb_  - other.vb_;

        L_ = (L_ < other.L_) ? 0 : L_ - other.L_;
        Q_ = (Q_ < other.Q_) ? 0 : Q_ - other.Q_;

        o_ = oa_ + ob_; v_ = va_ + vb_;
        a_ = oa_ + va_; b_ = ob_ + vb_;

        n_ = (o_ + v_) + L_ + Q_;
    }

    void operator+=(const pdaggerq::Line &line) {
        ++n_; // increment number of lines

        if (line.sig_) { ++L_; return; } // sigma
        if (line.den_) { ++Q_; return; } // density

        if (line.o_) { // occupied
            if (line.a_) ++oa_;
            else ++ob_; // default for no-spin is beta
        } else { // virtual
            if (line.a_) ++va_;
            else ++vb_; // default for no-spin is beta
        }
        o_ = oa_ + ob_; v_ = va_ + vb_;
        a_ = oa_ + va_; b_ = ob_ + vb_;
    }
    void operator-=(const pdaggerq::Line &line) {
        if (n_ != 0) --n_; // decrement number of lines
        else return; // do nothing if no lines

        if (line.sig_ && L_ != 0) { --L_; return; } // sigma
        if (line.den_ && Q_ != 0) { --Q_; return; } // density

        if (line.o_ && o_ != 0) { // occupied
            if (line.a_ && oa_ != 0) --oa_;
            else if (ob_ != 0) --ob_; // default for no-spin is beta
        } else if (v_ != 0) { // virtual
            if (line.a_ && va_ != 0) --va_;
            else if (vb_ != 0) --vb_; // default for no-spin is beta
        }
        o_ = oa_ + ob_; v_ = va_ + vb_;
        a_ = oa_ + va_; b_ = ob_ + vb_;
    }

    bool operator==(const shape & other) const {
        return  n_ == other.n_
            &&  a_ == other.a_  &&  b_ == other.b_
            &&  o_ == other.o_  &&  v_ == other.v_
            && oa_ == other.oa_ && ob_ == other.ob_
            && va_ == other.va_ && vb_ == other.vb_
            &&  L_ == other.L_  &&  Q_ == other.Q_;
    }
    bool operator!=(const shape & other) const {
        return !(*this == other);
    }

    string str() const {
        if (n_ == 0)
            return "0"; // scalar contraction has no lines

        string result;
        result.reserve(n_);

        result += 'o';
        result += std::to_string(o_);

        result += 'v';
        result += std::to_string(v_);

        if (L_ > 0) {
            result += 'L';
            result += std::to_string(L_);
        }
        if (Q_ > 0) {
            result += 'Q';
            result += std::to_string(Q_);
        }
        return result;
    }

    bool operator<( const shape & other) const {

        /// priority: o_ + v_ + L_, v_ + L_, L_, v_, va, oa

        // prioritize total scaling over individual scaling factors
        if (n_ != other.n_)
            return n_ < other.n_;

        /// if total scaling is the same, prioritize individual scaling factors
        if (Q_ + other.Q_ > 0) {
            // prioritize sum of v_ and L_ and Q_ over individual L_ and v_ and Q_
            uint_fast8_t sum = v_ + L_ + Q_;
            uint_fast8_t other_sum = other.v_ + other.L_ + other.Q_;
            if (sum != other_sum)
                return sum < other_sum;

            // if sum of v_ and L_ and Q_ is the same, prioritize Q_ over L_ and v_
            if (Q_ != other.Q_)
                return Q_ < other.Q_;
        }

        if (L_ + other.L_ > 0) {
            // prioritize sum of v_ and L_ over individual L_ and v_
            uint_fast8_t sum = v_ + L_;
            uint_fast8_t other_sum = other.v_ + other.L_;
            if (sum != other_sum)
                return sum < other_sum;

            // if sum of v_ and L_ is the same, prioritize L_ over v_
            if (L_ != other.L_)
                return L_ < other.L_;
        }

        // prioritize v_ over o_
        if (v_ != other.v_) return v_ < other.v_;

        // prioritize o over spin
        if (o_ != other.o_) return o_ < other.o_;

        // prioritize alpha over beta virtuals spin
        if (va_ != other.va_) return va_ < other.va_;

        // prioritize beta over alpha occupied spin
        if (ob_ != other.ob_) return ob_ < other.ob_;

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
    shape operator+(const pdaggerq::Line &line) const {
        shape result = *this;
        result += line;
        return result;
    }

    shape operator-(const shape &other) const {
        shape result = *this;
        result -= other;
        return result;
    }
    shape operator-(const pdaggerq::Line &line) const {
        shape result = *this;
        result -= line;
        return result;
    }

};

#endif //PDAGGERQ_SHAPE_HPP
