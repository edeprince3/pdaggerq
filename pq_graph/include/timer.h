//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: timer.h
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

#ifndef PQ_GRAPH_TIMER_H
#define PQ_GRAPH_TIMER_H
#include <string>
#include <sstream>

namespace pdaggerq {

    class Timer {

    private:
        /***** TIMERS *****/
        long double start_time_ = 0.0; // current start time
        long double end_time_ = 0.0; // current end time
        long double runtime_ = 0.0; // total runtime
        bool running_ = false; // whether the timer is running
        size_t count_ = 0; // number of times the timer has been started
    public:

        Timer() = default;
        ~Timer() = default;

        void start(); // start timer
        void stop(); // stop timer
        void reset(); // reset timer


        /**
         * @brief return the time as a human readable string
         * @param time time in seconds
         * @return human readable string
         */
        std::string elapsed() const;

        /**
         * @brief return the time in seconds
         * @return time in seconds
         */
        std::string get_time() const;

        /**
         * format time as string
         */
        int precision_ = 3; // precision of the timer
        static std::string format_time(long double time, int precision = 3);

        /**
         * @brief return the average time as a human readable string
         * @param time time in seconds
         * @return human readable string
         */
        std::string average_time() const;

        /**
         * @brief return the number of times the timer has been started
         * @return number of times the timer has been started
         */
        size_t count() const { return count_; }

        /**
         * Get runtime_ as double
         */
        long double get_runtime() const { return runtime_; }

        /**
         * Overload += operator
         */
        Timer& operator+=(const Timer& rhs){
            runtime_ += rhs.runtime_;
            running_ = false;
            return *this;
        }

        /**
         * Overload + operator
         */
        Timer operator+(const Timer& rhs) const {
            Timer tmp(*this);
            tmp += rhs;
            return tmp;
        }

        /**
         * Overload -= operator
         */
        Timer& operator-=(const Timer& rhs) {
            runtime_ -= rhs.runtime_;
            running_ = false;
            return *this;
        }

        /**
         * Overload - operator
         */
        Timer operator-(const Timer& rhs) const {
            Timer tmp(*this);
            tmp -= rhs;
            return tmp;
        }

        /**
         * Overload << operator
         */
        friend std::ostream& operator<<(std::ostream& os, const Timer& timer) {
            os << timer.get_time();
            return os;
        }

    };

}


#endif //PQ_GRAPH_TIMER_H
