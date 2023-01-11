//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_extras.cc
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

#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include<algorithm>
#include<cstring>
#include<math.h>
#include<sstream>

#include "pq_string.h"
#include "pq_utils.h"
#include "pq_extras.h"

namespace pdaggerq {

// consolidate terms that differ by eight summed labels plus permutations
void consolidate_permutations_plus_eight_swaps(
    std::vector<std::shared_ptr<pq_string> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2, 
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4,
    std::vector<std::string> labels_5,
    std::vector<std::string> labels_6,
    std::vector<std::string> labels_7,
    std::vector<std::string> labels_8) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;
        std::vector<int> find_3;
        std::vector<int> find_4;
        std::vector<int> find_5;
        std::vector<int> find_6;
        std::vector<int> find_7;
        std::vector<int> find_8;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_3[j]);
            find_3.push_back(found);
        }

        // ok, what labels do we have? list 4
        for (size_t j = 0; j < labels_4.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_4[j]);
            find_4.push_back(found);
        }

        // ok, what labels do we have? list 5
        for (size_t j = 0; j < labels_5.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_5[j]);
            find_5.push_back(found);
        }

        // ok, what labels do we have? list 6
        for (size_t j = 0; j < labels_6.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_6[j]);
            find_6.push_back(found);
        }

        // ok, what labels do we have? list 7
        for (size_t j = 0; j < labels_7.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_7[j]);
            find_7.push_back(found);
        }

        // ok, what labels do we have? list 8
        for (size_t j = 0; j < labels_8.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_8[j]);
            find_8.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels 1
            for (size_t id1 = 0; id1 < labels_1.size(); id1++) {
                if ( find_1[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels_1.size(); id2++) {
                    if ( find_1[id2] != 2 ) continue;

                    // try swapping non-summed labels 2
                    for (size_t id3 = 0; id3 < labels_2.size(); id3++) {
                        if ( find_2[id3] != 2 ) continue;
                        for (size_t id4 = id3 + 1; id4 < labels_2.size(); id4++) {
                            if ( find_2[id4] != 2 ) continue;

                            // try swapping non-summed labels 3
                            for (size_t id5 = 0; id5 < labels_3.size(); id5++) {
                                if ( find_3[id5] != 2 ) continue;
                                for (size_t id6 = id5 + 1; id6 < labels_3.size(); id6++) {
                                    if ( find_3[id6] != 2 ) continue;

                                    // try swapping non-summed labels 4
                                    for (size_t id7 = 0; id7 < labels_4.size(); id7++) {
                                        if ( find_4[id7] != 2 ) continue;
                                        for (size_t id8 = id7 + 1; id8 < labels_4.size(); id8++) {
                                            if ( find_4[id8] != 2 ) continue;

                                            // try swapping non-summed labels 5
                                            for (size_t id9 = 0; id9 < labels_5.size(); id9++) {
                                                if ( find_5[id9] != 2 ) continue;
                                                for (size_t id10 = id9 + 1; id10 < labels_5.size(); id10++) {
                                                    if ( find_5[id10] != 2 ) continue;

                                                    // try swapping non-summed labels 6
                                                    for (size_t id11 = 0; id11 < labels_6.size(); id11++) {
                                                        if ( find_6[id11] != 2 ) continue;
                                                        for (size_t id12 = id11 + 1; id12 < labels_6.size(); id12++) {
                                                            if ( find_6[id12] != 2 ) continue;

                                                            // try swapping non-summed labels 7
                                                            for (size_t id13 = 0; id13 < labels_7.size(); id13++) {
                                                                if ( find_7[id13] != 2 ) continue;
                                                                for (size_t id14 = id13 + 1; id14 < labels_7.size(); id14++) {
                                                                    if ( find_7[id14] != 2 ) continue;

                                                                    // try swapping non-summed labels 8
                                                                    for (size_t id15 = 0; id15 < labels_8.size(); id15++) {
                                                                        if ( find_8[id15] != 2 ) continue;
                                                                        for (size_t id16 = id15 + 1; id16 < labels_8.size(); id16++) {
                                                                            if ( find_8[id16] != 2 ) continue;

                                                                            std::shared_ptr<pq_string> newguy (new pq_string(ordered[i]->vacuum));
                                                                            newguy->copy((void*)(ordered[i].get()));
                                                                            swap_two_labels(newguy,labels_1[id1],labels_1[id2]);
                                                                            swap_two_labels(newguy,labels_2[id3],labels_2[id4]);
                                                                            swap_two_labels(newguy,labels_3[id5],labels_3[id6]);
                                                                            swap_two_labels(newguy,labels_4[id7],labels_4[id8]);
                                                                            swap_two_labels(newguy,labels_5[id9],labels_5[id10]);
                                                                            swap_two_labels(newguy,labels_6[id11],labels_6[id12]);
                                                                            swap_two_labels(newguy,labels_7[id13],labels_7[id14]);
                                                                            swap_two_labels(newguy,labels_8[id15],labels_8[id16]);
                                                                            newguy->sort_labels();
                                                                            strings_same = compare_strings(ordered[j],newguy,n_permute);

                                                                            if ( strings_same ) break;
                                                                        }
                                                                        if ( strings_same ) break;
                                                                    }
                                                                    if ( strings_same ) break;
                                                                }
                                                                if ( strings_same ) break;
                                                            }
                                                            if ( strings_same ) break;
                                                        }
                                                        if ( strings_same ) break;
                                                    }
                                                    if ( strings_same ) break;
                                                }
                                                if ( strings_same ) break;
                                            }
                                            if ( strings_same ) break;
                                        }
                                        if ( strings_same ) break;
                                    }
                                    if ( strings_same ) break;
                                }
                                if ( strings_same ) break;
                            }
                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->factor * ordered[i]->sign;
            double factor_j = ordered[j]->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

// consolidate terms that differ by seven summed labels plus permutations
void consolidate_permutations_plus_seven_swaps(
    std::vector<std::shared_ptr<pq_string> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2, 
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4,
    std::vector<std::string> labels_5,
    std::vector<std::string> labels_6,
    std::vector<std::string> labels_7) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;
        std::vector<int> find_3;
        std::vector<int> find_4;
        std::vector<int> find_5;
        std::vector<int> find_6;
        std::vector<int> find_7;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_3[j]);
            find_3.push_back(found);
        }

        // ok, what labels do we have? list 4
        for (size_t j = 0; j < labels_4.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_4[j]);
            find_4.push_back(found);
        }

        // ok, what labels do we have? list 5
        for (size_t j = 0; j < labels_5.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_5[j]);
            find_5.push_back(found);
        }

        // ok, what labels do we have? list 6
        for (size_t j = 0; j < labels_6.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_6[j]);
            find_6.push_back(found);
        }

        // ok, what labels do we have? list 7
        for (size_t j = 0; j < labels_7.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_7[j]);
            find_7.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels 1
            for (size_t id1 = 0; id1 < labels_1.size(); id1++) {
                if ( find_1[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels_1.size(); id2++) {
                    if ( find_1[id2] != 2 ) continue;

                    // try swapping non-summed labels 2
                    for (size_t id3 = 0; id3 < labels_2.size(); id3++) {
                        if ( find_2[id3] != 2 ) continue;
                        for (size_t id4 = id3 + 1; id4 < labels_2.size(); id4++) {
                            if ( find_2[id4] != 2 ) continue;

                            // try swapping non-summed labels 3
                            for (size_t id5 = 0; id5 < labels_3.size(); id5++) {
                                if ( find_3[id5] != 2 ) continue;
                                for (size_t id6 = id5 + 1; id6 < labels_3.size(); id6++) {
                                    if ( find_3[id6] != 2 ) continue;

                                    // try swapping non-summed labels 4
                                    for (size_t id7 = 0; id7 < labels_4.size(); id7++) {
                                        if ( find_4[id7] != 2 ) continue;
                                        for (size_t id8 = id7 + 1; id8 < labels_4.size(); id8++) {
                                            if ( find_4[id8] != 2 ) continue;

                                            // try swapping non-summed labels 5
                                            for (size_t id9 = 0; id9 < labels_5.size(); id9++) {
                                                if ( find_5[id9] != 2 ) continue;
                                                for (size_t id10 = id9 + 1; id10 < labels_5.size(); id10++) {
                                                    if ( find_5[id10] != 2 ) continue;

                                                    // try swapping non-summed labels 6
                                                    for (size_t id11 = 0; id11 < labels_6.size(); id11++) {
                                                        if ( find_6[id11] != 2 ) continue;
                                                        for (size_t id12 = id11 + 1; id12 < labels_6.size(); id12++) {
                                                            if ( find_6[id12] != 2 ) continue;

                                                            // try swapping non-summed labels 7
                                                            for (size_t id13 = 0; id13 < labels_7.size(); id13++) {
                                                                if ( find_7[id13] != 2 ) continue;
                                                                for (size_t id14 = id13 + 1; id14 < labels_7.size(); id14++) {
                                                                    if ( find_7[id14] != 2 ) continue;

                                                                    std::shared_ptr<pq_string> newguy (new pq_string(ordered[i]->vacuum));
                                                                    newguy->copy((void*)(ordered[i].get()));
                                                                    swap_two_labels(newguy,labels_1[id1],labels_1[id2]);
                                                                    swap_two_labels(newguy,labels_2[id3],labels_2[id4]);
                                                                    swap_two_labels(newguy,labels_3[id5],labels_3[id6]);
                                                                    swap_two_labels(newguy,labels_4[id7],labels_4[id8]);
                                                                    swap_two_labels(newguy,labels_5[id9],labels_5[id10]);
                                                                    swap_two_labels(newguy,labels_6[id11],labels_6[id12]);
                                                                    swap_two_labels(newguy,labels_7[id13],labels_7[id14]);
                                                                    newguy->sort_labels();
                                                                    strings_same = compare_strings(ordered[j],newguy,n_permute);

                                                                    if ( strings_same ) break;
                                                                }
                                                                if ( strings_same ) break;
                                                            }
                                                            if ( strings_same ) break;
                                                        }
                                                        if ( strings_same ) break;
                                                    }
                                                    if ( strings_same ) break;
                                                }
                                                if ( strings_same ) break;
                                            }
                                            if ( strings_same ) break;
                                        }
                                        if ( strings_same ) break;
                                    }
                                    if ( strings_same ) break;
                                }
                                if ( strings_same ) break;
                            }
                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->factor * ordered[i]->sign;
            double factor_j = ordered[j]->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

// consolidate terms that differ by six summed labels plus permutations
void consolidate_permutations_plus_six_swaps(
    std::vector<std::shared_ptr<pq_string> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2, 
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4,
    std::vector<std::string> labels_5,
    std::vector<std::string> labels_6) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;
        std::vector<int> find_3;
        std::vector<int> find_4;
        std::vector<int> find_5;
        std::vector<int> find_6;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_3[j]);
            find_3.push_back(found);
        }

        // ok, what labels do we have? list 4
        for (size_t j = 0; j < labels_4.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_4[j]);
            find_4.push_back(found);
        }

        // ok, what labels do we have? list 5
        for (size_t j = 0; j < labels_5.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_5[j]);
            find_5.push_back(found);
        }

        // ok, what labels do we have? list 6
        for (size_t j = 0; j < labels_6.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_6[j]);
            find_6.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels 1
            for (size_t id1 = 0; id1 < labels_1.size(); id1++) {
                if ( find_1[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels_1.size(); id2++) {
                    if ( find_1[id2] != 2 ) continue;

                    // try swapping non-summed labels 2
                    for (size_t id3 = 0; id3 < labels_2.size(); id3++) {
                        if ( find_2[id3] != 2 ) continue;
                        for (size_t id4 = id3 + 1; id4 < labels_2.size(); id4++) {
                            if ( find_2[id4] != 2 ) continue;

                            // try swapping non-summed labels 3
                            for (size_t id5 = 0; id5 < labels_3.size(); id5++) {
                                if ( find_3[id5] != 2 ) continue;
                                for (size_t id6 = id5 + 1; id6 < labels_3.size(); id6++) {
                                    if ( find_3[id6] != 2 ) continue;

                                    // try swapping non-summed labels 4
                                    for (size_t id7 = 0; id7 < labels_4.size(); id7++) {
                                        if ( find_4[id7] != 2 ) continue;
                                        for (size_t id8 = id7 + 1; id8 < labels_4.size(); id8++) {
                                            if ( find_4[id8] != 2 ) continue;

                                            // try swapping non-summed labels 5
                                            for (size_t id9 = 0; id9 < labels_5.size(); id9++) {
                                                if ( find_5[id9] != 2 ) continue;
                                                for (size_t id10 = id9 + 1; id10 < labels_5.size(); id10++) {
                                                    if ( find_5[id10] != 2 ) continue;

                                                    // try swapping non-summed labels 6
                                                    for (size_t id11 = 0; id11 < labels_6.size(); id11++) {
                                                        if ( find_6[id11] != 2 ) continue;
                                                        for (size_t id12 = id11 + 1; id12 < labels_6.size(); id12++) {
                                                            if ( find_6[id12] != 2 ) continue;

                                                            std::shared_ptr<pq_string> newguy (new pq_string(ordered[i]->vacuum));
                                                            newguy->copy((void*)(ordered[i].get()));
                                                            swap_two_labels(newguy,labels_1[id1],labels_1[id2]);
                                                            swap_two_labels(newguy,labels_2[id3],labels_2[id4]);
                                                            swap_two_labels(newguy,labels_3[id5],labels_3[id6]);
                                                            swap_two_labels(newguy,labels_4[id7],labels_4[id8]);
                                                            swap_two_labels(newguy,labels_5[id9],labels_5[id10]);
                                                            swap_two_labels(newguy,labels_6[id11],labels_6[id12]);
                                                            newguy->sort_labels();
                                                            strings_same = compare_strings(ordered[j],newguy,n_permute);

                                                            if ( strings_same ) break;
                                                        }
                                                        if ( strings_same ) break;
                                                    }
                                                    if ( strings_same ) break;
                                                }
                                                if ( strings_same ) break;
                                            }
                                            if ( strings_same ) break;
                                        }
                                        if ( strings_same ) break;
                                    }
                                    if ( strings_same ) break;
                                }
                                if ( strings_same ) break;
                            }
                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->factor * ordered[i]->sign;
            double factor_j = ordered[j]->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

// consolidate terms that differ by five summed labels plus permutations
void consolidate_permutations_plus_five_swaps(
    std::vector<std::shared_ptr<pq_string> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2, 
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4,
    std::vector<std::string> labels_5) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;
        std::vector<int> find_3;
        std::vector<int> find_4;
        std::vector<int> find_5;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_3[j]);
            find_3.push_back(found);
        }

        // ok, what labels do we have? list 4
        for (size_t j = 0; j < labels_4.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_4[j]);
            find_4.push_back(found);
        }

        // ok, what labels do we have? list 5
        for (size_t j = 0; j < labels_5.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_5[j]);
            find_5.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels 1
            for (size_t id1 = 0; id1 < labels_1.size(); id1++) {
                if ( find_1[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels_1.size(); id2++) {
                    if ( find_1[id2] != 2 ) continue;

                    // try swapping non-summed labels 2
                    for (size_t id3 = 0; id3 < labels_2.size(); id3++) {
                        if ( find_2[id3] != 2 ) continue;
                        for (size_t id4 = id3 + 1; id4 < labels_2.size(); id4++) {
                            if ( find_2[id4] != 2 ) continue;

                            // try swapping non-summed labels 3
                            for (size_t id5 = 0; id5 < labels_3.size(); id5++) {
                                if ( find_3[id5] != 2 ) continue;
                                for (size_t id6 = id5 + 1; id6 < labels_3.size(); id6++) {
                                    if ( find_3[id6] != 2 ) continue;

                                    // try swapping non-summed labels 4
                                    for (size_t id7 = 0; id7 < labels_4.size(); id7++) {
                                        if ( find_4[id7] != 2 ) continue;
                                        for (size_t id8 = id7 + 1; id8 < labels_4.size(); id8++) {
                                            if ( find_4[id8] != 2 ) continue;

                                            // try swapping non-summed labels 5
                                            for (size_t id9 = 0; id9 < labels_5.size(); id9++) {
                                                if ( find_5[id9] != 2 ) continue;
                                                for (size_t id10 = id9 + 1; id10 < labels_5.size(); id10++) {
                                                    if ( find_5[id10] != 2 ) continue;

                                                    std::shared_ptr<pq_string> newguy (new pq_string(ordered[i]->vacuum));
                                                    newguy->copy((void*)(ordered[i].get()));
                                                    swap_two_labels(newguy,labels_1[id1],labels_1[id2]);
                                                    swap_two_labels(newguy,labels_2[id3],labels_2[id4]);
                                                    swap_two_labels(newguy,labels_3[id5],labels_3[id6]);
                                                    swap_two_labels(newguy,labels_4[id7],labels_4[id8]);
                                                    swap_two_labels(newguy,labels_5[id9],labels_5[id10]);
                                                    newguy->sort_labels();
                                                    strings_same = compare_strings(ordered[j],newguy,n_permute);

                                                    if ( strings_same ) break;
                                                }
                                                if ( strings_same ) break;
                                            }
                                            if ( strings_same ) break;
                                        }
                                        if ( strings_same ) break;
                                    }
                                    if ( strings_same ) break;
                                }
                                if ( strings_same ) break;
                            }
                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->factor * ordered[i]->sign;
            double factor_j = ordered[j]->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

// consolidate terms that differ by four summed labels plus permutations
void consolidate_permutations_plus_four_swaps(
    std::vector<std::shared_ptr<pq_string> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2, 
    std::vector<std::string> labels_3,
    std::vector<std::string> labels_4) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;
        std::vector<int> find_3;
        std::vector<int> find_4;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_3[j]);
            find_3.push_back(found);
        }

        // ok, what labels do we have? list 4
        for (size_t j = 0; j < labels_4.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_4[j]);
            find_4.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels 1
            for (size_t id1 = 0; id1 < labels_1.size(); id1++) {
                if ( find_1[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels_1.size(); id2++) {
                    if ( find_1[id2] != 2 ) continue;

                    // try swapping non-summed labels 2
                    for (size_t id3 = 0; id3 < labels_2.size(); id3++) {
                        if ( find_2[id3] != 2 ) continue;
                        for (size_t id4 = id3 + 1; id4 < labels_2.size(); id4++) {
                            if ( find_2[id4] != 2 ) continue;

                            // try swapping non-summed labels 3
                            for (size_t id5 = 0; id5 < labels_3.size(); id5++) {
                                if ( find_3[id5] != 2 ) continue;
                                for (size_t id6 = id5 + 1; id6 < labels_3.size(); id6++) {
                                    if ( find_3[id6] != 2 ) continue;

                                    // try swapping non-summed labels 4
                                    for (size_t id7 = 0; id7 < labels_4.size(); id7++) {
                                        if ( find_4[id7] != 2 ) continue;
                                        for (size_t id8 = id7 + 1; id8 < labels_4.size(); id8++) {
                                            if ( find_4[id8] != 2 ) continue;

                                            std::shared_ptr<pq_string> newguy (new pq_string(ordered[i]->vacuum));
                                            newguy->copy((void*)(ordered[i].get()));
                                            swap_two_labels(newguy,labels_1[id1],labels_1[id2]);
                                            swap_two_labels(newguy,labels_2[id3],labels_2[id4]);
                                            swap_two_labels(newguy,labels_3[id5],labels_3[id6]);
                                            swap_two_labels(newguy,labels_4[id7],labels_4[id8]);
                                            newguy->sort_labels();
                                            strings_same = compare_strings(ordered[j],newguy,n_permute);

                                            if ( strings_same ) break;
                                        }
                                        if ( strings_same ) break;
                                    }
                                    if ( strings_same ) break;
                                }
                                if ( strings_same ) break;
                            }
                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->factor * ordered[i]->sign;
            double factor_j = ordered[j]->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

// consolidate terms that differ by three summed labels plus permutations
void consolidate_permutations_plus_three_swaps(
    std::vector<std::shared_ptr<pq_string> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2, 
    std::vector<std::string> labels_3) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;
        std::vector<int> find_3;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels_3[j]);
            find_3.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels 1
            for (size_t id1 = 0; id1 < labels_1.size(); id1++) {
                if ( find_1[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels_1.size(); id2++) {
                    if ( find_1[id2] != 2 ) continue;

                    // try swapping non-summed labels 2
                    for (size_t id3 = 0; id3 < labels_2.size(); id3++) {
                        if ( find_2[id3] != 2 ) continue;
                        for (size_t id4 = id3 + 1; id4 < labels_2.size(); id4++) {
                            if ( find_2[id4] != 2 ) continue;

                            // try swapping non-summed labels 3
                            for (size_t id5 = 0; id5 < labels_3.size(); id5++) {
                                if ( find_3[id5] != 2 ) continue;
                                for (size_t id6 = id5 + 1; id6 < labels_3.size(); id6++) {
                                    if ( find_3[id6] != 2 ) continue;

                                    std::shared_ptr<pq_string> newguy (new pq_string(ordered[i]->vacuum));
                                    newguy->copy((void*)(ordered[i].get()));
                                    swap_two_labels(newguy,labels_1[id1],labels_1[id2]);
                                    swap_two_labels(newguy,labels_2[id3],labels_2[id4]);
                                    swap_two_labels(newguy,labels_3[id5],labels_3[id6]);
                                    newguy->sort_labels();
                                    strings_same = compare_strings(ordered[j],newguy,n_permute);

                                    if ( strings_same ) break;
                                }
                                if ( strings_same ) break;
                            }
                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->factor * ordered[i]->sign;
            double factor_j = ordered[j]->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

} // End namespaces

