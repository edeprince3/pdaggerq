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

#ifndef CC_CAVITY_TIMER_H
#define CC_CAVITY_TIMER_H
#include <string>

namespace pdaggerq {

    class Timer {

    private:
        /***** TIMERS *****/
        long double start_time_; // current start time
        long double end_time_; // current end time
        long double runtime_ = 0.0; // total runtime
        bool running_ = false; // whether the timer is running_
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
        static std::string format_time(long double time);

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
    };

}


#endif //CC_CAVITY_TIMER_H
