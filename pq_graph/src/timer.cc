//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: timer.cc
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

#include "../include/timer.h"
// include omp only if defined
#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#define omp_set_num_threads(n) 1
#define omp_get_wtime() (double)clock()/CLOCKS_PER_SEC
#endif
#include <iomanip>
#include <cmath>

using std::string;
using std::stringstream;

namespace pdaggerq {

    void Timer::start(){
        start_time_ = omp_get_wtime();
        running_ = true;
    }

    void Timer::stop() {
        end_time_ = omp_get_wtime();
        runtime_ += end_time_ - start_time_;
        running_ = false;
        count_++;
    }

    void Timer::reset() {
        start_time_ = 0.0;
        end_time_ = 0.0;
        runtime_ = 0.0;
        count_ = 0;
        running_ = false;
    }

    std::string Timer::format_time(long double time, int precision) {
        time = std::fabs(time); // ignore negative times
        std::string unit = "s"; // default unit is seconds
        if (time < 1) {
            unit = "ms"; time *= 1e3; // milliseconds
            if (time < 1) {
                unit = "us"; time *= 1e3; // microseconds
                if (time < 1) {
                    unit = "ns"; time *= 1e3; // nanoseconds
                    // faster than a clock cycle on modern CPUs... for now :D
                    if (time < 1) {
                        unit = "ps"; time *= 1e3; // picoseconds
                        if (time < 1) {
                            unit = "fs"; time *= 1e3; // femtoseconds
                            if (time < 1) {
                                unit = "as"; time *= 1e3; // attoseconds
                                if (time < 1) {
                                    unit = "zs"; time *= 1e3; // zeptoseconds
                                    if (time < 1) {
                                        unit = "ys"; time *= 1e3; // yoctoseconds
                                    }}}}}}}} // I know, I know...
        else if (time >= 60) {
            unit = "m"; time /= 60; // minutes
            if (time >= 60) {
                unit = "h"; time /= 60; // hours
                if (time >= 24) {
                    unit = "d"; time /= 24; // days
                    if (time >= 7) {
                        unit = "w"; time /= 7; // weeks
                        if (time >= 4) {
                            unit = "mo"; time /= 4; // months
                            if (time >= 12) { // you never know how long a computer would run this code. ¯\_(ツ)_/¯
                                unit = "y"; time /= 12; // years
                                if (time >= 10) {
                                    unit = "dec"; time /= 10; // decade
                                    if (time >= 10) { // maybe you should consider buying a new computer
                                        unit = "cen"; time /= 10; // century
                                        if (time >= 10) { // you should really consider buying a new computer
                                            unit = "mill"; time /= 10; // millennium
                                            if (time >= 1000) {
                                                unit = "bill"; time /= 1e3; // billions of years
                                                if (time >= 1000) { // have you ever heard of the big bang?
                                                    unit = "trill"; time /= 1e3; // trillions of years
                                                    if (time >= 1000) {
                                                        // have you ever heard of the heat death of the universe?
                                                        unit = "quad"; time /= 1e3; // quadrillions of years
                                                        if (time >= 1000) {
                                                            unit = "quin"; time /= 1e3; // quintillions of years
                                                            // You are most likely dead by now. probably.
                                                        }}}}}}}}}}}}} // I'm not going to write this out any further...
                                                                      // You get the idea...

        // return the formatted time
        string time_str;
        stringstream ss;
        ss << std::right << std::setfill(' ') << std::setw(8) << std::fixed << std::setprecision(precision) << time;
        ss >> time_str;

        return time_str + " " + unit;
    }

    string Timer::elapsed() const {
        return format_time(runtime_, precision_);
    }

    string Timer::average_time() const{
        return format_time(runtime_ / (double) count_, precision_);
    }

    string Timer::get_time() const {
        return format_time(end_time_ - start_time_, precision_);
    }


}

