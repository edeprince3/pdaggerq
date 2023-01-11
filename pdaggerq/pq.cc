//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq.cc
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

#include "pq.h"
#include "pq_utils.h"

namespace pdaggerq {

pq::pq(std::string vacuum_type) {

  data = (std::shared_ptr<StringData>)(new StringData(vacuum_type));

}

pq::~pq() {
}

// in order to compare strings, the creation and annihilation 
// operators should be ordered in some consistent way.
// alphabetically seems reasonable enough
void pq::alphabetize(std::vector<std::shared_ptr<pq> > &ordered) {

    // alphabetize string
    for (size_t i = 0; i < ordered.size(); i++) {

        // creation
        bool not_alphabetized = false;
        do {
            not_alphabetized = false;
            int ndagger = 0;
            for (size_t j = 0; j < ordered[i]->data->symbol.size(); j++) {
                if ( ordered[i]->data->is_dagger[j] ) ndagger++;
            }
            for (int j = 0; j < ndagger-1; j++) {
                int val1 = ordered[i]->data->symbol[j].c_str()[0];
                int val2 = ordered[i]->data->symbol[j+1].c_str()[0];
                if ( val2 < val1 ) {
                    std::string dum = ordered[i]->data->symbol[j];
                    ordered[i]->data->symbol[j] = ordered[i]->data->symbol[j+1];
                    ordered[i]->data->symbol[j+1] = dum;
                    ordered[i]->data->sign = -ordered[i]->data->sign;
                    not_alphabetized = true;
                    j = ordered[i]->data->symbol.size() + 1;
                    not_alphabetized = true;
                }
            }
        }while(not_alphabetized);
        // annihilation
        not_alphabetized = false;
        do {
            not_alphabetized = false;
            int ndagger = 0;
            for (size_t j = 0; j < ordered[i]->data->symbol.size(); j++) {
                if ( ordered[i]->data->is_dagger[j] ) ndagger++;
            }
            for (int j = ndagger; j < (int)ordered[i]->data->symbol.size()-1; j++) {
                int val1 = ordered[i]->data->symbol[j].c_str()[0];
                int val2 = ordered[i]->data->symbol[j+1].c_str()[0];
                if ( val2 < val1 ) {
                    std::string dum = ordered[i]->data->symbol[j];
                    ordered[i]->data->symbol[j] = ordered[i]->data->symbol[j+1];
                    ordered[i]->data->symbol[j+1] = dum;
                    ordered[i]->data->sign = -ordered[i]->data->sign;
                    not_alphabetized = true;
                    j = ordered[i]->data->symbol.size() + 1;
                    not_alphabetized = true;
                }
            }
        }while(not_alphabetized);
    }

    // alphabetize deltas
    for (size_t i = 0; i < ordered.size(); i++) {
        for (size_t j = 0; j < ordered[i]->data->deltas.size(); j++) {
            int val1 = ordered[i]->data->deltas[j].labels[0].c_str()[0];
            int val2 = ordered[i]->data->deltas[j].labels[1].c_str()[0];
            if ( val2 < val1 ) {
                std::string dum = ordered[i]->data->deltas[j].labels[0];
                ordered[i]->data->deltas[j].labels[0] = ordered[i]->data->deltas[j].labels[1];
                ordered[i]->data->deltas[j].labels[1] = dum;
            }
        }
    }
}

void pq::reorder_t_amplitudes() {

    size_t dim = data->amps['t'].size();

    if ( dim == 0 ) return;

    bool* nope = (bool*)malloc(dim * sizeof(bool));
    memset((void*)nope,'\0',dim * sizeof(bool));

    std::vector<std::vector<std::string> > tmp;

    std::vector<amplitudes> tmp_new;

    for (size_t order = 1; order < 7; order++) {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if ( nope[j] ) continue;

                if ( data->amps['t'][j].labels.size() == 2 * order ) {
                    tmp_new.push_back(data->amps['t'][j]);
                    nope[j] = true;
                    break;
                }

            }
        }
    }

    if ( dim != tmp_new.size() ) { 
        printf("\n");
        printf("    something went very wrong in reorder_t_amplitudes()\n");
        printf("    this function breaks for t6 and higher. why would\n");
        printf("    you want that, anyway?\n");
        printf("\n");
        exit(1);
    }

    for (int i = 0; i < dim; i++) {
        data->amps['t'][i].labels.clear();
        data->amps['t'][i].numerical_labels.clear();
    }
    data->amps['t'].clear();
    for (size_t i = 0; i < tmp_new.size(); i++) {
        data->amps['t'].push_back(tmp_new[i]);
    }

    free(nope);
    
}

// expand sums to include spin and zero terms where appropriate
void pq::spin_blocking(std::vector<std::shared_ptr<pq> > &spin_blocked, std::map<std::string, std::string> spin_map) {

    // check that non-summed spin labels match those specified
    std::vector<std::string> occ_labels { "i", "j", "k", "l", "m", "n", "o" };
    std::vector<std::string> vir_labels { "a", "b", "c", "d", "e", "f", "g" };

    std::map<std::string, bool> found_labels;
    
    // ok, what non-summed labels do we have in the occupied space? 
    for (size_t j = 0; j < occ_labels.size(); j++) {
        int found = index_in_anywhere(data, occ_labels[j]);
        if ( found == 1 ) {
            found_labels[occ_labels[j]] = true;
        }else{
            found_labels[occ_labels[j]] = false;
        }
    }
    
    // ok, what non-summed labels do we have in the virtual space? 
    for (size_t j = 0; j < vir_labels.size(); j++) {
        int found = index_in_anywhere(data, vir_labels[j]);
        if ( found == 1 ) {
            found_labels[vir_labels[j]] = true;
        }else{
            found_labels[vir_labels[j]] = false;
        }
    }

    for (size_t j = 0; j < occ_labels.size(); j++) {
        if ( found_labels[occ_labels[j]] ) {
            if ( spin_map[occ_labels[j]] != "a" && spin_map[occ_labels[j]] != "b" ) {
                printf("\n");
                printf("    error: spin label for non-summed index %s is invalid\n", occ_labels[j].c_str());
                printf("\n");
                exit(1);
            }
        }
    }
    for (size_t j = 0; j < vir_labels.size(); j++) {
        if ( found_labels[vir_labels[j]] ) {
            if ( spin_map[vir_labels[j]] != "a" && spin_map[vir_labels[j]] != "b" ) {
                printf("\n");
                printf("    error: spin label for non-summed index %s is invalid\n", vir_labels[j].c_str());
                printf("\n");
                exit(1);
            }
        }
    }

    // non-summed spin labels
    data->non_summed_spin_labels = spin_map;

    // copy this term and zero spins

    std::shared_ptr<pq> newguy (new pq(data->vacuum));
    newguy->data->copy((void*)this);

    newguy->data->reset_spin_labels();

    // list of expanded sums
    std::vector< std::shared_ptr<pq> > tmp;
    tmp.push_back(newguy);

    for (size_t i = 0; i < tmp.size(); i++) {

        // but first expand permutations where spins don't match 
        size_t n = tmp[i]->data->permutations.size() / 2;

        for (size_t j = 0; j < n; j++) {

            std::string idx1 = tmp[i]->data->permutations[2*j];
            std::string idx2 = tmp[i]->data->permutations[2*j+1];

            // spin 1
            std::string spin1 = "";
            spin1 = tmp[i]->data->non_summed_spin_labels[idx1];

            // spin 2
            std::string spin2 = "";
            spin2 = tmp[i]->data->non_summed_spin_labels[idx2];

            // if spins are not the same, then the permutation needs to be expanded explicitly and allowed spins redetermined
            if ( spin1 != spin2 ) {

                // first guy is just a copy
                std::shared_ptr<pq> newguy1 (new pq(data->vacuum));
                newguy1->data->copy((void*)tmp[i].get());

                // second guy is a copy with permuted labels and change in sign
                std::shared_ptr<pq> newguy2 (new pq(data->vacuum));
                newguy2->data->copy((void*)tmp[i].get());
                swap_two_labels(newguy2->data, idx1, idx2);
                newguy2->data->sign *= -1;

                // reset non-summed spins for this guy
                newguy2->data->reset_spin_labels();

                // both guys need to have permutation lists adjusted
                newguy1->data->permutations.clear();
                newguy2->data->permutations.clear();

                for (size_t k = 0; k < n; k++) {

                    // skip jth permutation, which is the one we expanded
                    if ( j == k ) continue;

                    newguy1->data->permutations.push_back(tmp[i]->data->permutations[2*k]);
                    newguy1->data->permutations.push_back(tmp[i]->data->permutations[2*k+1]);

                    newguy2->data->permutations.push_back(tmp[i]->data->permutations[2*k]);
                    newguy2->data->permutations.push_back(tmp[i]->data->permutations[2*k+1]);
                }

                tmp[i]->data->skip = true;
                tmp.push_back(newguy1);
                tmp.push_back(newguy2);

                // break loop over permutations because this above logic only works on one permutation at a time
                break;
            }
        }
    }

    // now, expand sums 

    bool done_adding_spins = false;
    do {
        std::vector< std::shared_ptr<pq> > list;
        done_adding_spins = true;
        for (size_t i = 0; i < tmp.size(); i++) {
            bool am_i_done = tmp[i]->add_spins(list);
            if ( !am_i_done ) done_adding_spins = false;
        }
        if ( !done_adding_spins ) {
            tmp.clear();
            for (size_t i = 0; i < list.size(); i++) {
                if ( !list[i]->data->skip ) {
                    tmp.push_back(list[i]);
                }
            }
        }
    }while(!done_adding_spins);


    // kill terms that have mismatched spin 
    for (size_t i = 0; i < tmp.size(); i++) {

        if ( tmp[i]->data->skip ) continue;

        bool killit = false;

        // amplitudes
        // TODO: this logic only works for particle-conserving amplitudes
        for (size_t j = 0; j < data->amplitude_types.size(); j++) {
            char type = data->amplitude_types[j];
            for (size_t k = 0; k < tmp[i]->data->amps[type].size(); k++) {

                size_t order = tmp[i]->data->amps[type][k].labels.size()/2;

                int left_a = 0;
                int left_b = 0;
                int right_a = 0;
                int right_b = 0;
                for (size_t l = 0; l < order; l++) {
                    if ( tmp[i]->data->amps[type][k].spin_labels[l] == "a" ) {
                        left_a++;
                    }else {
                        left_b++;
                    }
                    if ( tmp[i]->data->amps[type][k].spin_labels[l+order] == "a" ) {
                        right_a++;
                    }else {
                        right_b++;
                    }
                }
                if (left_a != right_a || left_b != right_b ) {
                    killit = true;
                    break;
                }

            }
            if ( killit ) break;
        }
        if ( killit ) {
            tmp[i]->data->skip = true;
            continue;
        }

        killit = false;

        // integrals
        for (size_t j = 0; j < data->integral_types.size(); j++) {
            std::string type = data->integral_types[j];
            for (size_t k = 0; k < tmp[i]->data->ints[type].size(); k++) {
                size_t order = tmp[i]->data->ints[type][k].labels.size()/2;

                int left_a = 0;
                int left_b = 0;
                int right_a = 0;
                int right_b = 0;
                for (size_t l = 0; l < order; l++) {
                    if ( tmp[i]->data->ints[type][k].spin_labels[l] == "a" ) {
                        left_a++;
                    }else {
                        left_b++;
                    }
                    if ( tmp[i]->data->ints[type][k].spin_labels[l+order] == "a" ) {
                        right_a++;
                    }else {
                        right_b++;
                    }
                }
                if (left_a != right_a || left_b != right_b ) {
                    killit = true;
                    break;
                }

            }
            if ( killit ) break;
        }
        if ( killit ) {
            tmp[i]->data->skip = true;
            continue;
        }

        killit = false;

        // delta functions 
        for (size_t j = 0; j < data->deltas.size(); j++) {
            if ( tmp[i]->data->deltas[j].spin_labels[0] != tmp[i]->data->deltas[j].spin_labels[1] ) {
                killit = true;
                break;
            }
        }

        if ( killit ) {
            tmp[i]->data->skip = true;
            continue;
        }
    }

    
    // rearrange terms so that they have standard spin order (abba -> -abab, etc.)
    for (size_t p = 0; p < tmp.size(); p++) {

        if ( tmp[p]->data->skip ) continue;

        // amplitudes
        for (size_t i = 0; i < data->amplitude_types.size(); i++) {
            char type = data->amplitude_types[i];
            for (size_t j = 0; j < tmp[p]->data->amps[type].size(); j++) {
                size_t order = tmp[p]->data->amps[type][j].labels.size()/2;
                if ( order > 4 ) {
                    printf("\n");
                    printf("    error: spin tracing doesn't work for higher than quadruples yet\n");
                    printf("\n");
                    exit(1);
                }
                if ( order == 2 ) {

                    // three cases that require attention: ab;ba, ba;ab, and ba;ba

                    if (       tmp[p]->data->amps[type][j].spin_labels[0] == "a"
                            && tmp[p]->data->amps[type][j].spin_labels[1] == "b"
                            && tmp[p]->data->amps[type][j].spin_labels[2] == "b"
                            && tmp[p]->data->amps[type][j].spin_labels[3] == "a" ) {

                            std::string tmp_label = tmp[p]->data->amps[type][j].labels[2];
                            tmp[p]->data->amps[type][j].labels[2] = tmp[p]->data->amps[type][j].labels[3];
                            tmp[p]->data->amps[type][j].labels[3] = tmp_label;

                            tmp[p]->data->amps[type][j].spin_labels[2] = "a";
                            tmp[p]->data->amps[type][j].spin_labels[3] = "b";

                            tmp[p]->data->sign *= -1;

                    }else if ( tmp[p]->data->amps[type][j].spin_labels[0] == "b"
                            && tmp[p]->data->amps[type][j].spin_labels[1] == "a"
                            && tmp[p]->data->amps[type][j].spin_labels[2] == "a"
                            && tmp[p]->data->amps[type][j].spin_labels[3] == "b" ) {

                            std::string tmp_label = tmp[p]->data->amps[type][j].labels[0];
                            tmp[p]->data->amps[type][j].labels[0] = tmp[p]->data->amps[type][j].labels[1];
                            tmp[p]->data->amps[type][j].labels[1] = tmp_label;

                            tmp[p]->data->amps[type][j].spin_labels[0] = "a";
                            tmp[p]->data->amps[type][j].spin_labels[1] = "b";

                            tmp[p]->data->sign *= -1;


                    }else if ( tmp[p]->data->amps[type][j].spin_labels[0] == "b"
                            && tmp[p]->data->amps[type][j].spin_labels[1] == "a"
                            && tmp[p]->data->amps[type][j].spin_labels[2] == "b"
                            && tmp[p]->data->amps[type][j].spin_labels[3] == "a" ) {

                            std::string tmp_label = tmp[p]->data->amps[type][j].labels[0];
                            tmp[p]->data->amps[type][j].labels[0] = tmp[p]->data->amps[type][j].labels[1];
                            tmp[p]->data->amps[type][j].labels[1] = tmp_label;

                            tmp[p]->data->amps[type][j].spin_labels[0] = "a";
                            tmp[p]->data->amps[type][j].spin_labels[1] = "b";

                            tmp_label = tmp[p]->data->amps[type][j].labels[2];
                            tmp[p]->data->amps[type][j].labels[2] = tmp[p]->data->amps[type][j].labels[3];
                            tmp[p]->data->amps[type][j].labels[3] = tmp_label;

                            tmp[p]->data->amps[type][j].spin_labels[2] = "a";
                            tmp[p]->data->amps[type][j].spin_labels[3] = "b";

                    }
                }else if ( order == 3 ) {

                    // target order: aaa, aab, abb, bbb
                    int sign = 1;
                    reorder_three_spins(tmp[p]->data->amps[type][j], 0, 1, 2, sign);
                    reorder_three_spins(tmp[p]->data->amps[type][j], 3, 4, 5, sign);
                    tmp[p]->data->sign *= sign;

                }else if ( order == 4 ) {

                    // target order: aaaa, aaab, aabb, abbb, bbbb
                    int sign = 1;
                    reorder_four_spins(tmp[p]->data->amps[type][j], 0, 1, 2, 3, sign);
                    reorder_four_spins(tmp[p]->data->amps[type][j], 4, 5, 6, 7, sign);
                    tmp[p]->data->sign *= sign;

                }
            }
        }

        // integrals
        for (size_t i = 0; i < data->integral_types.size(); i++) {
            std::string type = data->integral_types[i];
            for (size_t j = 0; j < tmp[p]->data->ints[type].size(); j++) {

                size_t order = tmp[p]->data->ints[type][j].labels.size()/2;

                if ( order != 2 ) continue;

                // three cases that require attention: ab;ba, ba;ab, and ba;ba

                // integrals
                if (       tmp[p]->data->ints[type][j].spin_labels[0] == "a"
                        && tmp[p]->data->ints[type][j].spin_labels[1] == "b"
                        && tmp[p]->data->ints[type][j].spin_labels[2] == "b"
                        && tmp[p]->data->ints[type][j].spin_labels[3] == "a" ) {

                        std::string tmp_label = tmp[p]->data->ints[type][j].labels[2];
                        tmp[p]->data->ints[type][j].labels[2] = tmp[p]->data->ints[type][j].labels[3];
                        tmp[p]->data->ints[type][j].labels[3] = tmp_label;

                        tmp[p]->data->ints[type][j].spin_labels[2] = "a";
                        tmp[p]->data->ints[type][j].spin_labels[3] = "b";

                        tmp[p]->data->sign *= -1;

                }else if ( tmp[p]->data->ints[type][j].spin_labels[0] == "b"
                        && tmp[p]->data->ints[type][j].spin_labels[1] == "a"
                        && tmp[p]->data->ints[type][j].spin_labels[2] == "a"
                        && tmp[p]->data->ints[type][j].spin_labels[3] == "b" ) {

                        std::string tmp_label = tmp[p]->data->ints[type][j].labels[0];
                        tmp[p]->data->ints[type][j].labels[0] = tmp[p]->data->ints[type][j].labels[1];
                        tmp[p]->data->ints[type][j].labels[1] = tmp_label;

                        tmp[p]->data->ints[type][j].spin_labels[0] = "a";
                        tmp[p]->data->ints[type][j].spin_labels[1] = "b";

                        tmp[p]->data->sign *= -1;


                }else if ( tmp[p]->data->ints[type][j].spin_labels[0] == "b"
                        && tmp[p]->data->ints[type][j].spin_labels[1] == "a"
                        && tmp[p]->data->ints[type][j].spin_labels[2] == "b"
                        && tmp[p]->data->ints[type][j].spin_labels[3] == "a" ) {

                        std::string tmp_label = tmp[p]->data->ints[type][j].labels[0];
                        tmp[p]->data->ints[type][j].labels[0] = tmp[p]->data->ints[type][j].labels[1];
                        tmp[p]->data->ints[type][j].labels[1] = tmp_label;

                        tmp[p]->data->ints[type][j].spin_labels[0] = "a";
                        tmp[p]->data->ints[type][j].spin_labels[1] = "b";

                        tmp_label = tmp[p]->data->ints[type][j].labels[2];
                        tmp[p]->data->ints[type][j].labels[2] = tmp[p]->data->ints[type][j].labels[3];
                        tmp[p]->data->ints[type][j].labels[3] = tmp_label;

                        tmp[p]->data->ints[type][j].spin_labels[2] = "a";
                        tmp[p]->data->ints[type][j].spin_labels[3] = "b";

                }
            }
        }
    }

    // 
    for (size_t i = 0; i < tmp.size(); i++) {
        if ( tmp[i]->data->skip ) continue;
        spin_blocked.push_back(tmp[i]);
    }

    tmp.clear();

}

// reorder four spins ... cases to consider: aaba/abaa/baaa -> aaab; baab/abba/baba/bbaa/abab -> aabb; babb/bbab/bbba -> abbb

void pq::reorder_four_spins(amplitudes & amps, int i1, int i2, int i3, int i4, int & sign) {

    // aaba/abaa/baaa -> aaab
    if (       amps.spin_labels[i1] == "a"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "b" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i3];
            amps.labels[i3] = tmp_label;

            amps.spin_labels[i3] = "a";
            amps.spin_labels[i4] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "a"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "a" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.spin_labels[i2] = "a";
            amps.spin_labels[i4] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "a" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i4] = "b";

            sign *= -1;

    // baab/abba/baba/bbaa/abab -> aabb
    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "a" 
            && amps.spin_labels[i4] == "b" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i3] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "a"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "b" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.spin_labels[i2] = "a";
            amps.spin_labels[i4] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "b" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i4] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "a" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.spin_labels[i2] = "a";
            amps.spin_labels[i4] = "b";

            tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i3] = "b";

    }else if ( amps.spin_labels[i1] == "a"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "a"
            && amps.spin_labels[i4] == "b" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.spin_labels[i2] = "a";
            amps.spin_labels[i3] = "b";

            sign *= -1;

    // babb/bbab/bbba -> abbb
    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "b" 
            && amps.spin_labels[i4] == "b" ) {

            std::string tmp_label = amps.labels[i2];

            amps.labels[i2] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i2] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "a" 
            && amps.spin_labels[i4] == "b" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i3] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "b" 
            && amps.spin_labels[i4] == "a" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i4] = "b";

            sign *= -1;

    }

}

// reorder three spins ... cases to consider: aba/baa -> aab; bba/bab -> abb

void pq::reorder_three_spins(amplitudes & amps, int i1, int i2, int i3, int & sign) {

    if (       amps.spin_labels[i1] == "a"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "a" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.spin_labels[i2] = "a";
            amps.spin_labels[i3] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "a" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i3] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "b"
            && amps.spin_labels[i3] == "a" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i3] = "b";

            sign *= -1;

    }else if ( amps.spin_labels[i1] == "b"
            && amps.spin_labels[i2] == "a"
            && amps.spin_labels[i3] == "b" ) {

            std::string tmp_label = amps.labels[i2];

            amps.labels[i2] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.spin_labels[i1] = "a";
            amps.spin_labels[i2] = "b";

            sign *= -1;

    }

}


bool pq::add_spins(std::vector<std::shared_ptr<pq> > &list) {

    if ( data->skip ) return true;

    bool all_spins_added = false;

    // amplitudes
    for (size_t i = 0; i < data->amplitude_types.size(); i++) {
        char type = data->amplitude_types[i];
        for (size_t j = 0; j < data->amps[type].size(); j++) {
            for (size_t k = 0; k < data->amps[type][j].labels.size(); k++) {
                if ( data->amps[type][j].spin_labels[k] == "" ) {

                    std::shared_ptr<pq> sa (new pq(data->vacuum));
                    std::shared_ptr<pq> sb (new pq(data->vacuum));

                    sa->data->copy((void*)this);
                    sb->data->copy((void*)this);

                    sa->data->set_spin_everywhere(data->amps[type][j].labels[k], "a");
                    sb->data->set_spin_everywhere(data->amps[type][j].labels[k], "b");

                    //sa->data->amps[type][j].spin_labels[k] = "a";
                    //sb->data->amps[type][j].spin_labels[k] = "b";

                    list.push_back(sa);
                    list.push_back(sb);
                    return false;

                }
            }
        }
    }

    // integrals
    for (size_t i = 0; i < data->integral_types.size(); i++) {
        std::string type = data->integral_types[i];
        for (size_t j = 0; j < data->ints[type].size(); j++) {
            for (size_t k = 0; k < data->ints[type][j].labels.size(); k++) {
                if ( data->ints[type][j].spin_labels[k] == "" ) {

                    std::shared_ptr<pq> sa (new pq(data->vacuum));
                    std::shared_ptr<pq> sb (new pq(data->vacuum));

                    sa->data->copy((void*)this);
                    sb->data->copy((void*)this);

                    sa->data->set_spin_everywhere(data->ints[type][j].labels[k], "a");
                    sb->data->set_spin_everywhere(data->ints[type][j].labels[k], "b");

                    //sa->data->ints[type][j].spin_labels[k] = "a";
                    //sb->data->ints[type][j].spin_labels[k] = "b";

                    list.push_back(sa);
                    list.push_back(sb);
                    return false;

                }
            }
        }
    }

    // must be done.
    return true;

}

// compare strings and remove terms that cancel

void pq::cleanup(std::vector<std::shared_ptr<pq> > &ordered) {


    for (size_t i = 0; i < ordered.size(); i++) {

        // order amplitudes such that they're ordered t1, t2, t3, etc.
        ordered[i]->reorder_t_amplitudes();

        // sort amplitude labels
        ordered[i]->data->sort_labels();

    }

    // prune list so it only contains non-skipped ones
    std::vector< std::shared_ptr<pq> > pruned;
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->data->skip ) continue;

        // for normal order relative to fermi vacuum, i doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming
        if ( data->vacuum == "FERMI" ) {
            if ( ordered[i]->data->symbol.size() != 0 ) continue;
            if ( ordered[i]->data->is_boson_dagger.size() != 0 ) continue;
        }

        pruned.push_back(ordered[i]);
    }
    ordered.clear();
    for (size_t i = 0; i < pruned.size(); i++) {
        ordered.push_back(pruned[i]);
    }
    pruned.clear();

    //printf("starting string comparisons\n");fflush(stdout);

    std::vector<std::string> occ_labels { "i", "j", "k", "l", "m", "n", "o" };
    std::vector<std::string> vir_labels { "a", "b", "c", "d", "e", "f", "g" };

    consolidate_permutations(ordered);

    consolidate_permutations_plus_swap(ordered,occ_labels);
    consolidate_permutations_plus_swap(ordered,vir_labels);

    consolidate_permutations_plus_two_swaps(ordered,occ_labels,occ_labels);
    consolidate_permutations_plus_two_swaps(ordered,vir_labels,vir_labels);
    consolidate_permutations_plus_two_swaps(ordered,occ_labels,vir_labels);

    // these don't seem to be necessary for test cases up to ccsdtq
/*
    consolidate_permutations_plus_three_swaps(ordered,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_three_swaps(ordered,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_three_swaps(ordered,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_three_swaps(ordered,vir_labels,vir_labels,vir_labels);

    consolidate_permutations_plus_four_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_four_swaps(ordered,occ_labels,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_four_swaps(ordered,occ_labels,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_four_swaps(ordered,occ_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_four_swaps(ordered,vir_labels,vir_labels,vir_labels,vir_labels);

    consolidate_permutations_plus_five_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_five_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_five_swaps(ordered,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_five_swaps(ordered,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_five_swaps(ordered,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_five_swaps(ordered,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);

    consolidate_permutations_plus_six_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_six_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_six_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_six_swaps(ordered,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_six_swaps(ordered,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_six_swaps(ordered,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_six_swaps(ordered,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);

    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_seven_swaps(ordered,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_seven_swaps(ordered,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);

    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,occ_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_eight_swaps(ordered,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels,vir_labels);
*/

    // probably only relevant for vacuum = fermi
    if ( data->vacuum != "FERMI" ) return;

    consolidate_permutations_non_summed(ordered,occ_labels);
    consolidate_permutations_non_summed(ordered,vir_labels);

    // re-prune
    pruned.clear();
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->data->skip ) continue;

        // for normal order relative to fermi vacuum, i doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming
        if ( data->vacuum == "FERMI" ) {
            if ( ordered[i]->data->symbol.size() != 0 ) continue;
            if ( ordered[i]->data->is_boson_dagger.size() != 0 ) continue;
        }

        pruned.push_back(ordered[i]);
    }
    ordered.clear();
    for (size_t i = 0; i < pruned.size(); i++) {
        ordered.push_back(pruned[i]);
    }
    pruned.clear();

}

// consolidate terms that differ by permutations of non-summed labels
void pq::consolidate_permutations_non_summed(
    std::vector<std::shared_ptr<pq> > &ordered,
    std::vector<std::string> labels) {


    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->data->skip ) continue;

        std::vector<int> find_idx;

        // ok, what labels do we have? 
        for (size_t j = 0; j < labels.size(); j++) {
            int found = index_in_anywhere(ordered[i]->data, labels[j]);
            // this is buggy when existing permutation labels belong to 
            // the same space as the labels we're permuting ... so skip those for now.
            bool same_space = false;
            bool is_occ1 = is_occ(labels[j]);
            for (size_t k = 0; k < ordered[i]->data->permutations.size(); k++) {
                bool is_occ2 = is_occ(ordered[i]->data->permutations[k]);
                if ( is_occ1 && is_occ2 ) {
                    same_space = true;
                    break;
                }else if ( !is_occ1 && !is_occ2 ) {
                    same_space = true;
                    break;
                }
            }

            if ( !same_space ) {
                find_idx.push_back(found);
            }else{
                find_idx.push_back(0);
            }
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->data->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            std::string permutation_1 = "";
            std::string permutation_2 = "";

            // try swapping non-summed labels
            for (size_t id1 = 0; id1 < labels.size(); id1++) {
                if ( find_idx[id1] != 1 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels.size(); id2++) {
                    if ( find_idx[id2] != 1 ) continue;

                    std::shared_ptr<pq> newguy (new pq(data->vacuum));
                    newguy->data->copy((void*)(ordered[i].get()));
                    swap_two_labels(newguy->data,labels[id1],labels[id2]);

                    strings_same = compare_strings(ordered[j],newguy,n_permute);

                    if ( strings_same ) {

                        permutation_1 = labels[id1];
                        permutation_2 = labels[id2];
                        break;

                    }
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            // it is possible to have made it through the previous logic without 
            // assigning permutation labels, if strings are identical but 
            // permutation operators differ
            //if ( permutation_1 == "" || permutation_2 == "" ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->data->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->data->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, then this is a permutation
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->data->permutations.push_back(permutation_1);
                ordered[i]->data->permutations.push_back(permutation_2);
                ordered[j]->data->skip = true;
                break;
            }

            // otherwise, something has gone wrong in the previous consolidation step...
        }
    }
}

// find and replace any funny labels in integrals with conventional ones. i.e., o1 -> i ,v1 -> a
void pq::use_conventional_labels() {

    // occupied first:
    std::vector<std::string> occ_in{"o0","o1","o2","o3","o4","o5","o6","o7","o8","o9",
                                    "o10","o11","o12","o13","o14","o15","o16","o17","o18","o19",
                                    "o20","o21","o22","o23","o24","o25","o26","o27","o28","o29"};
    std::vector<std::string> occ_out{"i","j","k","l","m","n","o","t",
                                     "i0","i1","i2","i3","i4","i5","i6","i7","i8","i9",
                                     "i10","i11","i12","i13","i14","i15","i16","i17","i18","i19"};

    for (size_t i = 0; i < occ_in.size(); i++) {

        if ( index_in_anywhere(data, occ_in[i]) > 0 ) {

            for (size_t j = 0; j < occ_out.size(); j++) {

                if ( index_in_anywhere(data, occ_out[j]) == 0 ) {

                    replace_index_everywhere(data, occ_in[i], occ_out[j]);
                    break;
                }
            }
        }
    }

    // now virtual
    std::vector<std::string> vir_in{"v0","v1","v2","v3","v4","v5","v6","v7","v8","v9",
                                    "v10","v11","v12","v13","v14","v15","v16","v17","v18","v19",
                                    "v20","v21","v22","v23","v24","v25","v26","v27","v28","v29"};
    std::vector<std::string> vir_out{"a","b","c","d","e","f","g","h",
                                     "a0","a1","a2","a3","a4","a5","a6","a7","a8","a9",
                                     "a10","a11","a12","a13","a14","a15","a16","a17","a18","a19"};

    for (size_t i = 0; i < vir_in.size(); i++) {

        if ( index_in_anywhere(data, vir_in[i]) > 0 ) {

            for (size_t j = 0; j < vir_out.size(); j++) {

                if ( index_in_anywhere(data, vir_out[j]) == 0 ) {

                    replace_index_everywhere(data, vir_in[i], vir_out[j]);
                    break;
                }
            }
        }
    }
}

void pq::gobble_deltas() {

    std::vector<std::string> tmp_delta1;
    std::vector<std::string> tmp_delta2;

    // create list of summation labels. only consider internally-created labels

    std::vector<std::string> occ_labels{"o0","o1","o2","o3","o4","o5","o6","o7","o8","o9",
                                    "o10","o11","o12","o13","o14","o15","o16","o17","o18","o19",
                                    "o20","o21","o22","o23","o24","o25","o26","o27","o28","o29"};
    std::vector<std::string> vir_labels{"v0","v1","v2","v3","v4","v5","v6","v7","v8","v9",
                                    "v10","v11","v12","v13","v14","v15","v16","v17","v18","v19",
                                    "v20","v21","v22","v23","v24","v25","v26","v27","v28","v29"};

    std::vector<std::string> sum_labels;
    for (size_t i = 0; i < occ_labels.size(); i++) {
        if ( index_in_anywhere(data, occ_labels[i]) == 2 ) {
            sum_labels.push_back(occ_labels[i]);
        }
    }
    for (size_t i = 0; i < vir_labels.size(); i++) {
        if ( index_in_anywhere(data, vir_labels[i]) == 2 ) {
            sum_labels.push_back(vir_labels[i]);
        }
    }

    for (size_t i = 0; i < data->deltas.size(); i++) {

        // is delta label 1 in list of summation labels?
        bool have_delta1 = false;
        for (size_t j = 0; j < sum_labels.size(); j++) {
            if ( data->deltas[i].labels[0] == sum_labels[j] ) {
                have_delta1 = true;
                break;
            }
        }
        // is delta label 2 in list of summation labels?
        bool have_delta2 = false;
        for (size_t j = 0; j < sum_labels.size(); j++) {
            if ( data->deltas[i].labels[1] == sum_labels[j] ) {
                have_delta2 = true;
                break;
            }
        }

/*
        // this logic is obviously cleaner than that below, but 
        // for some reason the code has a harder time collecting 
        // like terms this way. requires swapping up to four 
        // labels.
        if ( have_delta1 ) {
            replace_index_everywhere( data->deltas[i].labels[0], data->deltas[i].labels[1] );
            continue;
        }else if ( have_delta2 ) {
            replace_index_everywhere( data->deltas[i].labels[1], data->deltas[i].labels[0] );
            continue;
        }
*/

        bool do_continue = false;
        for (size_t j = 0; j < data->integral_types.size(); j++) { 
            std::string type = data->integral_types[j];
            if ( have_delta1 && index_in_integrals( data->deltas[i].labels[0], data->ints[type] ) > 0 ) {
               replace_index_in_integrals( data->deltas[i].labels[0], data->deltas[i].labels[1], data->ints[type] );
               do_continue = true;
               break;
            }else if ( have_delta2 && index_in_integrals( data->deltas[i].labels[1], data->ints[type] ) > 0 ) {
               replace_index_in_integrals( data->deltas[i].labels[1], data->deltas[i].labels[0], data->ints[type] );
               do_continue = true;
               break;
            }
        }
        if ( do_continue ) continue;

        // TODO: note that the code only efficiently collects terms when the amplitude
        // list is ordered as {'t', 'l', 'r', 'u', 'm', 's'} ... i don't know why, but
        // i do know that this is the problematic part of the code
        do_continue = false;
        std::vector<char> types = {'t', 'l', 'r', 'u', 'm', 's'};
        for (size_t j = 0; j < types.size(); j++) { 
            char type = types[j];
            if ( have_delta1 && index_in_amplitudes( data->deltas[i].labels[0], data->amps[type] ) > 0 ) {
               replace_index_in_amplitudes( data->deltas[i].labels[0], data->deltas[i].labels[1], data->amps[type] );
               do_continue = true;
               break;
            }else if ( have_delta2 && index_in_amplitudes( data->deltas[i].labels[1], data->amps[type] ) > 0 ) {
               replace_index_in_amplitudes( data->deltas[i].labels[1], data->deltas[i].labels[0], data->amps[type] );
               do_continue = true;
               break;
            }
        }
        if ( do_continue ) continue;

        // at this point, it is safe to assume the delta function must remain
        tmp_delta1.push_back(data->deltas[i].labels[0]);
        tmp_delta2.push_back(data->deltas[i].labels[1]);

    }

    data->deltas.clear();

    for (size_t i = 0; i < tmp_delta1.size(); i++) {

        delta_functions deltas;
        deltas.labels.push_back(tmp_delta1[i]);
        deltas.labels.push_back(tmp_delta2[i]);
        deltas.sort();
        data->deltas.push_back(deltas);

    }

}

// re-classify fluctuation potential terms
void pq::reclassify_integrals() {

    if ( data->ints["occ_repulsion"].size() > 1 ) {
       printf("\n");
       printf("only support for one integral type object per string\n");
       printf("\n");
       exit(1);
    }

    if ( data->ints["occ_repulsion"].size() > 0 ) {

        // pick summation label not included in string already
        std::vector<std::string> occ_out{"i","j","k","l","m","n","o","t","i0","i1","i2","i3","i4","i5","i6","i7","i8","i9"};
        std::string idx;

        int do_skip = -999;

        for (size_t i = 0; i < occ_out.size(); i++) {
            if ( index_in_anywhere(data, occ_out[i]) == 0 ) {
                idx = occ_out[i];
                do_skip = i;
                break;
            }
        }
        if ( do_skip == -999 ) {
            printf("\n");
            printf("    uh oh. no suitable summation index could be found.\n");
            printf("\n");
            exit(1);
        }

        std::string idx1 = data->ints["occ_repulsion"][0].labels[0];
        std::string idx2 = data->ints["occ_repulsion"][0].labels[1];

        data->ints["occ_repulsion"].clear();

        integrals ints;

        ints.labels.clear();
        ints.numerical_labels.clear();

        ints.labels.push_back(idx1);
        ints.labels.push_back(idx);
        ints.labels.push_back(idx2);
        ints.labels.push_back(idx);

        ints.sort();

        if ( data->ints["eri"].size() > 0 ) {
           printf("\n");
           printf("only support for one integral type object per string\n");
           printf("\n");
           exit(1);
        }
        data->ints["eri"].clear();
        data->ints["eri"].push_back(ints);

    }

}

} // End namespaces

