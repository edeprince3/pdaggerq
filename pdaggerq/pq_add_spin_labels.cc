//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_utils.cc
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

#include "pq_utils.h"
#include "pq_add_spin_labels.h"

namespace pdaggerq {

// reorder three spins ... cases to consider: aba/baa -> aab; bba/bab -> abb
void reorder_three_spins(amplitudes & amps, int i1, int i2, int i3, int & sign) {
            
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

// reorder four spins ... cases to consider: aaba/abaa/baaa -> aaab; baab/abba/baba/bbaa/abab -> aabb; babb/bbab/bbba -> abbb
void reorder_four_spins(amplitudes & amps, int i1, int i2, int i3, int i4, int & sign) {

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

// add spin labels to a string
bool add_spins(const std::shared_ptr<pq_string>& in, std::vector<std::shared_ptr<pq_string> > &list) {

    if ( in->skip ) return true;

    bool all_spins_added = false;

    // amplitudes
    for (auto &amp_pair : in->amps) {
        char type = amp_pair.first;
        std::vector<amplitudes> & amps = amp_pair.second;
        for (auto & amp : amps) {
            for (size_t k = 0; k < amp.labels.size(); k++) {
                if ( amp.spin_labels[k].empty() ) {

                    std::shared_ptr<pq_string> sa (new pq_string(in->vacuum));
                    std::shared_ptr<pq_string> sb (new pq_string(in->vacuum));

                    sa->copy(in.get());
                    sb->copy(in.get());

                    sa->set_spin_everywhere(amp.labels[k], "a");
                    sb->set_spin_everywhere(amp.labels[k], "b");

                    list.push_back(sa);
                    list.push_back(sb);
                    return false;
                }
            }
        }
    }

    // integrals
    for (auto &ints_pair : in->ints) {
        const std::string &type = ints_pair.first;
        std::vector<integrals> & ints = ints_pair.second;
        for (auto & integral : ints) {
            for (size_t k = 0; k < integral.labels.size(); k++) {
                if ( integral.spin_labels[k].empty() ) {

                    std::shared_ptr<pq_string> sa (new pq_string(in->vacuum));
                    std::shared_ptr<pq_string> sb (new pq_string(in->vacuum));

                    sa->copy(in.get());
                    sb->copy(in.get());

                    sa->set_spin_everywhere(integral.labels[k], "a");
                    sb->set_spin_everywhere(integral.labels[k], "b");

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

// expand sums to include spin and zero terms where appropriate
void spin_blocking(const std::shared_ptr<pq_string>& in, std::vector<std::shared_ptr<pq_string> > &spin_blocked, std::unordered_map<std::string, std::string> spin_map) {

    // check that non-summed spin labels match those specified
    static std::vector<std::string> occ_labels { "i", "j", "k", "l", "m", "n", "o" };
    static std::vector<std::string> vir_labels { "a", "b", "c", "d", "e", "f", "g" };

    std::map<std::string, bool> found_labels;
    
    // ok, what non-summed labels do we have in the occupied space? 
    for (const std::string & occ_label : occ_labels) {
        int found = index_in_anywhere(in, occ_label);
        if ( found == 1 ) {
            found_labels[occ_label] = true;
        }else{
            found_labels[occ_label] = false;
        }
    }
    
    // ok, what non-summed labels do we have in the virtual space? 
    for (const std::string & vir_label : vir_labels) {
        int found = index_in_anywhere(in, vir_label);
        if ( found == 1 ) {
            found_labels[vir_label] = true;
        }else{
            found_labels[vir_label] = false;
        }
    }

    for (const std::string & occ_label : occ_labels) {
        if ( found_labels[occ_label] ) {
            // find spin label
            auto pos = spin_map.find(occ_label);
            std::string spin_label = pos == spin_map.end() ? "" : pos->second;

            // check if spin label is valid
            if ( spin_label != "a" && spin_label != "b" ) {
                printf("\n");
                printf("    error: spin label for non-summed index %s is invalid\n", occ_label.c_str());
                printf("\n");
                exit(1);
            }
        }
    }
    for (const std::string & vir_label : vir_labels) {
        if ( found_labels[vir_label] ) {
            // find spin label
            auto pos = spin_map.find(vir_label);
            std::string spin_label = pos == spin_map.end() ? "" : pos->second;

            // check if spin label is valid
            if ( spin_label != "a" && spin_label != "b" ) {
                printf("\n");
                printf("    error: spin label for non-summed index %s is invalid\n", vir_label.c_str());
                printf("\n");
                exit(1);
            }
        }
    }

    // non-summed spin labels
    in->non_summed_spin_labels = spin_map;

    // copy this term and zero spins

    std::shared_ptr<pq_string> newguy = std::make_shared<pq_string>(*in);
    newguy->reset_spin_labels();

    // list of expanded sums
    std::vector< std::shared_ptr<pq_string> > tmp;
    tmp.push_back(newguy);

    // but first expand single permutations where spins don't match 
    for (size_t i = 0; i < tmp.size(); i++) {

        size_t n = tmp[i]->permutations.size() / 2;

        for (size_t j = 0; j < n; j++) {

            std::string idx1 = tmp[i]->permutations[2*j];
            std::string idx2 = tmp[i]->permutations[2*j+1];

            // spin 1
            std::string spin1;
            spin1 = tmp[i]->non_summed_spin_labels[idx1];

            // spin 2
            std::string spin2;
            spin2 = tmp[i]->non_summed_spin_labels[idx2];

            // if spins are not the same, then the permutation needs to be expanded explicitly and allowed spins redetermined
            if ( spin1 != spin2 ) {

                // first guy is just a copy
                std::shared_ptr<pq_string> newguy1 = std::make_shared<pq_string>(*tmp[i]);

                // second guy is a copy with permuted labels and change in sign
                std::shared_ptr<pq_string> newguy2 = std::make_shared<pq_string>(*tmp[i]);
                swap_two_labels(newguy2, idx1, idx2);
                newguy2->sign *= -1;

                // reset non-summed spins for this guy
                newguy2->reset_spin_labels();

                // both guys need to have permutation lists adjusted
                newguy1->permutations.clear();
                newguy2->permutations.clear();

                for (size_t k = 0; k < n; k++) {

                    // skip jth permutation, which is the one we expanded
                    if ( j == k ) continue;

                    newguy1->permutations.push_back(tmp[i]->permutations[2*k]);
                    newguy1->permutations.push_back(tmp[i]->permutations[2*k+1]);

                    newguy2->permutations.push_back(tmp[i]->permutations[2*k]);
                    newguy2->permutations.push_back(tmp[i]->permutations[2*k+1]);
                }

                tmp[i]->skip = true;
                tmp.push_back(newguy1);
                tmp.push_back(newguy2);

                // break loop over permutations because this above logic only works on one permutation at a time
                break;
            }
        }
    }

    // now expand paired permutations (3) where spins don't match 
    for (size_t i = 0; i < tmp.size(); i++) {
        size_t n = tmp[i]->paired_permutations_3.size() / 6;

        for (size_t j = 0; j < n; j++) {

            std::string o1 = tmp[i]->paired_permutations_3[6*j];
            std::string v1 = tmp[i]->paired_permutations_3[6*j+1];
            std::string o2 = tmp[i]->paired_permutations_3[6*j+2];
            std::string v2 = tmp[i]->paired_permutations_3[6*j+3];
            std::string o3 = tmp[i]->paired_permutations_3[6*j+4];
            std::string v3 = tmp[i]->paired_permutations_3[6*j+5];

            // spin 1
            std::string spin1;
            spin1 = tmp[i]->non_summed_spin_labels[o1];
            if ( spin1 != tmp[i]->non_summed_spin_labels[v1] ) {
                printf("\n");
                printf("    error: spin label mismatch between occupied and virtual orbitals (1)\n");
                printf("\n");
                exit(1);
            }

            // spin 2
            std::string spin2;
            spin2 = tmp[i]->non_summed_spin_labels[o2];
            if ( spin2 != tmp[i]->non_summed_spin_labels[v2] ) {
                printf("\n");
                printf("    error: spin label mismatch between occupied and virtual orbitals (2)\n");
                printf("\n");
                exit(1);
            }

            // spin 3
            std::string spin3;
            spin3 = tmp[i]->non_summed_spin_labels[o3];
            if ( spin3 != tmp[i]->non_summed_spin_labels[v3] ) {
                printf("\n");
                printf("    error: spin label mismatch between occupied and virtual orbitals (3)\n");
                printf("\n");
                exit(1);
            }

            // if spins are not the same, then the permutation needs to be expanded explicitly and allowed spins redetermined

            // aab or bba -> PP2[ab](abc) + cba
            if ( spin1 == spin2 && spin1 != spin3 ) {

                // first guy is just a copy plus a PP2 permutation
                std::shared_ptr<pq_string> newguy1 = std::make_shared<pq_string>(*tmp[i]);

                // adjust permutations. no more PP3, but we have PP2
                newguy1->paired_permutations_3.clear();
                newguy1->paired_permutations_2.push_back(o1);
                newguy1->paired_permutations_2.push_back(v1);
                newguy1->paired_permutations_2.push_back(o2);
                newguy1->paired_permutations_2.push_back(v2);

                // second guy is a copy with permuted pair labels and no change in sign
                std::shared_ptr<pq_string> newguy2 = std::make_shared<pq_string>(*tmp[i]);
                swap_two_labels(newguy2, o1, o3);
                swap_two_labels(newguy2, v1, v3);

                // reset non-summed spins for this guy
                newguy2->reset_spin_labels();

                // no more PP3
                newguy2->paired_permutations_3.clear();

                for (size_t k = 0; k < n; k++) {

                    // skip jth permutation, which is the one we expanded
                    if ( j == k ) continue;

                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+1]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+2]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+3]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+4]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+5]);

                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+1]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+2]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+3]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+4]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+5]);

                }

                tmp[i]->skip = true;
                tmp.push_back(newguy1);
                tmp.push_back(newguy2);

                // break loop over permutations because this above logic only works on one permutation at a time
                break;
            }
            // abb or baa -> abc + bac + cba
            else if ( spin1 != spin2 && spin2 == spin3 ) {

                // first guy is just a copy 
                std::shared_ptr<pq_string> newguy1 = std::make_shared<pq_string>(*tmp[i]);

                // adjust permutations. no more PP3.
                newguy1->paired_permutations_3.clear();

                // second guy is a copy with permuted pair labels and no change in sign
                std::shared_ptr<pq_string> newguy2 = std::make_shared<pq_string>(*tmp[i]);
                swap_two_labels(newguy2, o1, o2);
                swap_two_labels(newguy2, v1, v2);

                // reset non-summed spins for this guy
                newguy2->reset_spin_labels();

                // no more PP3
                newguy2->paired_permutations_3.clear();

                // third guy is a copy with permuted pair labels and no change in sign
                std::shared_ptr<pq_string> newguy3 = std::make_shared<pq_string>(*tmp[i]);
                swap_two_labels(newguy3, o1, o3);
                swap_two_labels(newguy3, v1, v3);

                // reset non-summed spins for this guy
                newguy3->reset_spin_labels();

                // no more PP3
                newguy3->paired_permutations_3.clear();

                for (size_t k = 0; k < n; k++) {

                    // skip jth permutation, which is the one we expanded
                    if ( j == k ) continue;

                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+1]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+2]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+3]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+4]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+5]);

                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+1]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+2]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+3]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+4]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+5]);

                    newguy3->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k]);
                    newguy3->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+1]);
                    newguy3->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+2]);
                    newguy3->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+3]);
                    newguy3->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+4]);
                    newguy3->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+5]);

                }

                tmp[i]->skip = true;
                tmp.push_back(newguy1);
                tmp.push_back(newguy2);
                tmp.push_back(newguy3);

                // break loop over permutations because this above logic only works on one permutation at a time
                break;
            }
            // aba or bab -> PP2[ac](abc) + bac
            else if ( spin1 != spin2 && spin1 == spin3 ) {

                // first guy is just a copy plus a PP2 permutation
                std::shared_ptr<pq_string> newguy1 = std::make_shared<pq_string>(*tmp[i]);

                // adjust permutations. no more PP3, but we have PP2
                newguy1->paired_permutations_3.clear();
                newguy1->paired_permutations_2.push_back(o1);
                newguy1->paired_permutations_2.push_back(v1);
                newguy1->paired_permutations_2.push_back(o3);
                newguy1->paired_permutations_2.push_back(v3);

                // second guy is a copy with permuted pair labels and no change in sign
                std::shared_ptr<pq_string> newguy2 = std::make_shared<pq_string>(*tmp[i]);
                swap_two_labels(newguy2, o1, o2);
                swap_two_labels(newguy2, v1, v2);

                // reset non-summed spins for this guy
                newguy2->reset_spin_labels();

                // no more PP3
                newguy2->paired_permutations_3.clear();

                for (size_t k = 0; k < n; k++) {

                    // skip jth permutation, which is the one we expanded
                    if ( j == k ) continue;

                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+1]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+2]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+3]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+4]);
                    newguy1->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+5]);

                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+1]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+2]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+3]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+4]);
                    newguy2->paired_permutations_3.push_back(tmp[i]->paired_permutations_3[6*k+5]);

                }

                tmp[i]->skip = true;
                tmp.push_back(newguy1);
                tmp.push_back(newguy2);

                // break loop over permutations because this above logic only works on one permutation at a time
                break;
            }
        }
    }

    // now expand paired permutations (6) where spins don't match 
    for (size_t i = 0; i < tmp.size(); i++) {
        size_t n = tmp[i]->paired_permutations_6.size() / 6;

        for (size_t j = 0; j < n; j++) {

            std::string o1 = tmp[i]->paired_permutations_6[6*j];
            std::string v1 = tmp[i]->paired_permutations_6[6*j+1];
            std::string o2 = tmp[i]->paired_permutations_6[6*j+2];
            std::string v2 = tmp[i]->paired_permutations_6[6*j+3];
            std::string o3 = tmp[i]->paired_permutations_6[6*j+4];
            std::string v3 = tmp[i]->paired_permutations_6[6*j+5];

            // spin 1
            std::string spin1;
            spin1 = tmp[i]->non_summed_spin_labels[o1];
            if ( spin1 != tmp[i]->non_summed_spin_labels[v1] ) {
                printf("\n");
                printf("    error: spin label mismatch between occupied and virtual orbitals (1)\n");
                printf("\n");
                exit(1);
            }

            // spin 2
            std::string spin2;
            spin2 = tmp[i]->non_summed_spin_labels[o2];
            if ( spin2 != tmp[i]->non_summed_spin_labels[v2] ) {
                printf("\n");
                printf("    error: spin label mismatch between occupied and virtual orbitals (2)\n");
                printf("\n");
                exit(1);
            }

            // spin 3
            std::string spin3;
            spin3 = tmp[i]->non_summed_spin_labels[o3];
            if ( spin3 != tmp[i]->non_summed_spin_labels[v3] ) {
                printf("\n");
                printf("    error: spin label mismatch between occupied and virtual orbitals (3)\n");
                printf("\n");
                exit(1);
            }

            // if spins are not the same, then the permutation needs to be expanded explicitly and allowed spins redetermined

            // aab or bba -> PP2[ab](abc) + PP2[ab](acb) + PP2[ab](cab)
            if ( spin1 == spin2 && spin1 != spin3 ) {

                // first guy is just a copy plus a PP2 permutation
                std::shared_ptr<pq_string> newguy1 = std::make_shared<pq_string>(*tmp[i]);

                // adjust permutations. no more PP6, but we have PP2
                newguy1->paired_permutations_6.clear();
                newguy1->paired_permutations_2.push_back(o1);
                newguy1->paired_permutations_2.push_back(v1);
                newguy1->paired_permutations_2.push_back(o2);
                newguy1->paired_permutations_2.push_back(v2);

                // second guy is a copy with permuted pair labels, plus a PP2 permutation
                std::shared_ptr<pq_string> newguy2 = std::make_shared<pq_string>(*tmp[i]);
                swap_two_labels(newguy2, o2, o3);
                swap_two_labels(newguy2, v2, v3);

                // reset non-summed spins for this guy
                newguy2->reset_spin_labels();

                // no more PP6, but we have PP2
                newguy2->paired_permutations_6.clear();
                newguy2->paired_permutations_2.push_back(o1);
                newguy2->paired_permutations_2.push_back(v1);
                newguy2->paired_permutations_2.push_back(o2);
                newguy2->paired_permutations_2.push_back(v2);

                // third guy is another copy with permuted pair labels, plus a PP2 permutation
                std::shared_ptr<pq_string> newguy3 = std::make_shared<pq_string>(*tmp[i]);
                swap_two_labels(newguy3, o1, o2);
                swap_two_labels(newguy3, v1, v2);
                swap_two_labels(newguy3, o2, o3);
                swap_two_labels(newguy3, v2, v3);

                // reset non-summed spins for this guy
                newguy3->reset_spin_labels();

                // no more PP6, but we have PP2
                newguy3->paired_permutations_6.clear();
                newguy3->paired_permutations_2.push_back(o1);
                newguy3->paired_permutations_2.push_back(v1);
                newguy3->paired_permutations_2.push_back(o2);
                newguy3->paired_permutations_2.push_back(v2);

                for (size_t k = 0; k < n; k++) {

                    // skip jth permutation, which is the one we expanded
                    if ( j == k ) continue;

                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+1]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+2]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+3]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+4]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+5]);

                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+1]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+2]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+3]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+4]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+5]);

                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+1]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+2]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+3]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+4]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+5]);

                }

                tmp[i]->skip = true;
                tmp.push_back(newguy1);
                tmp.push_back(newguy2);
                tmp.push_back(newguy3);

                // break loop over permutations because this above logic only works on one permutation at a time
                break;
            }
            // abb or baa -> PP2[bc](abc) + PP2[bc](bac) + PP2[bc](bca)
            else if ( spin1 != spin2 && spin2 == spin3 ) {

                // first guy is just a copy plus a PP2 permutation
                std::shared_ptr<pq_string> newguy1 = std::make_shared<pq_string>(*tmp[i]);

                // adjust permutations. no more PP6, but we have PP2
                newguy1->paired_permutations_6.clear();
                newguy1->paired_permutations_2.push_back(o2);
                newguy1->paired_permutations_2.push_back(v2);
                newguy1->paired_permutations_2.push_back(o3);
                newguy1->paired_permutations_2.push_back(v3);

                // second guy is a copy with permuted pair labels, plus a PP2 permutation
                std::shared_ptr<pq_string> newguy2 = std::make_shared<pq_string>(*tmp[i]);
                swap_two_labels(newguy2, o1, o2);
                swap_two_labels(newguy2, v1, v2);

                // reset non-summed spins for this guy
                newguy2->reset_spin_labels();

                // no more PP6, but we have PP2
                newguy2->paired_permutations_6.clear();
                newguy2->paired_permutations_2.push_back(o2);
                newguy2->paired_permutations_2.push_back(v2);
                newguy2->paired_permutations_2.push_back(o3);
                newguy2->paired_permutations_2.push_back(v3);

                // third guy is another copy with permuted pair labels, plus a PP2 permutation
                std::shared_ptr<pq_string> newguy3 = std::make_shared<pq_string>(*tmp[i]);
                swap_two_labels(newguy3, o1, o2);
                swap_two_labels(newguy3, v1, v2);
                swap_two_labels(newguy3, o1, o3);
                swap_two_labels(newguy3, v1, v3);

                // reset non-summed spins for this guy
                newguy3->reset_spin_labels();

                // no more PP6, but we have PP2
                newguy3->paired_permutations_6.clear();
                newguy3->paired_permutations_2.push_back(o2);
                newguy3->paired_permutations_2.push_back(v2);
                newguy3->paired_permutations_2.push_back(o3);
                newguy3->paired_permutations_2.push_back(v3);

                for (size_t k = 0; k < n; k++) {

                    // skip jth permutation, which is the one we expanded
                    if ( j == k ) continue;

                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+1]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+2]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+3]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+4]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+5]);

                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+1]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+2]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+3]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+4]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+5]);

                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+1]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+2]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+3]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+4]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+5]);
                }

                tmp[i]->skip = true;
                tmp.push_back(newguy1);
                tmp.push_back(newguy2);
                tmp.push_back(newguy3);

                // break loop over permutations because this above logic only works on one permutation at a time
                break;
            }
            // aba or bab -> PP2[ac](abc) + PP2[ac](bac) + PP2[ac](acb)
            else if ( spin1 == spin2 && spin1 != spin3 ) {

                // first guy is just a copy plus a PP2 permutation
                std::shared_ptr<pq_string> newguy1 = std::make_shared<pq_string>(*tmp[i]);

                // adjust permutations. no more PP6, but we have PP2
                newguy1->paired_permutations_6.clear();
                newguy1->paired_permutations_2.push_back(o1);
                newguy1->paired_permutations_2.push_back(v1);
                newguy1->paired_permutations_2.push_back(o3);
                newguy1->paired_permutations_2.push_back(v3);

                // second guy is a copy with permuted pair labels, plus a PP2 permutation
                std::shared_ptr<pq_string> newguy2 = std::make_shared<pq_string>(*tmp[i]);
                swap_two_labels(newguy2, o1, o2);
                swap_two_labels(newguy2, v1, v2);

                // reset non-summed spins for this guy
                newguy2->reset_spin_labels();

                // no more PP6, but we have PP2
                newguy2->paired_permutations_6.clear();
                newguy2->paired_permutations_2.push_back(o1);
                newguy2->paired_permutations_2.push_back(v1);
                newguy2->paired_permutations_2.push_back(o3);
                newguy2->paired_permutations_2.push_back(v3);

                // third guy is another copy with permuted pair labels, plus a PP2 permutation
                std::shared_ptr<pq_string> newguy3 = std::make_shared<pq_string>(*tmp[i]);
                swap_two_labels(newguy3, o2, o3);
                swap_two_labels(newguy3, v2, v3);

                // reset non-summed spins for this guy
                newguy3->reset_spin_labels();

                // no more PP6, but we have PP2
                newguy3->paired_permutations_6.clear();
                newguy3->paired_permutations_2.push_back(o1);
                newguy3->paired_permutations_2.push_back(v1);
                newguy3->paired_permutations_2.push_back(o3);
                newguy3->paired_permutations_2.push_back(v3);

                for (size_t k = 0; k < n; k++) {

                    // skip jth permutation, which is the one we expanded
                    if ( j == k ) continue;

                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+1]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+2]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+3]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+4]);
                    newguy1->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+5]);

                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+1]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+2]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+3]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+4]);
                    newguy2->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+5]);

                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+1]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+2]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+3]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+4]);
                    newguy3->paired_permutations_6.push_back(tmp[i]->paired_permutations_6[6*k+5]);

                }

                tmp[i]->skip = true;
                tmp.push_back(newguy1);
                tmp.push_back(newguy2);
                tmp.push_back(newguy3);

                // break loop over permutations because this above logic only works on one permutation at a time
                break;
            }
        }
    }

    // now, expand sums 

    bool done_adding_spins = false;
    do {
        std::vector< std::shared_ptr<pq_string> > list;
        done_adding_spins = true;
        for (std::shared_ptr<pq_string> & pq_str : tmp) {
            bool am_i_done = add_spins(pq_str, list);
            if ( !am_i_done ) done_adding_spins = false;
        }
        if ( !done_adding_spins ) {
            tmp.clear();
            for (std::shared_ptr<pq_string> & pq_str : list) {
                if ( !pq_str->skip ) {
                    tmp.push_back(pq_str);
                }
            }
        }
    }while(!done_adding_spins);

    // kill terms that have mismatched spin 
    for (auto & pq_str : tmp) {

        if ( pq_str->skip ) continue;

        bool killit = false;

        // amplitudes
        for (auto &amp_pair : pq_str->amps) {
            char type = amp_pair.first;
            std::vector<amplitudes> & amps = amp_pair.second;
            for (auto & amp : amps) {

                size_t n_create_a = 0;
                size_t n_create_b = 0;

                for (size_t l = 0; l < amp.n_create; l++) {
                    if (amp.spin_labels[l] == "a" ) {
                        n_create_a++;
                    }else {
                        n_create_b++;
                    }
                }

                size_t n_annihilate_a = 0;
                size_t n_annihilate_b = 0;

                for (size_t l = 0; l < amp.n_annihilate; l++) {
                    if (amp.spin_labels[l + amp.n_create] == "a" ) {
                        n_annihilate_a++;
                    }else {
                        n_annihilate_b++;
                    }
                }

                // particle conserving excitations:
                if (amp.n_create == amp.n_annihilate) {
                    if (n_create_a != n_annihilate_a || n_create_b != n_annihilate_b ) {
                        killit = true;
                        break;
                    }
                // particle non-conserving excitations:
                }else if (amp.n_create > amp.n_annihilate) {
                    if (n_create_a < n_annihilate_a || n_create_b < n_annihilate_b ) {
                        killit = true;
                        break;
                    }
                // particle non-conserving excitations:
                }else if (amp.n_create < amp.n_annihilate) {
                    if (n_create_a > n_annihilate_a || n_create_b > n_annihilate_b ) {
                        killit = true;
                        break;
                    }
                }

            }
            if ( killit ) break;
        }
        if ( killit ) {
            pq_str->skip = true;
            continue;
        }

        killit = false;

        // integrals
        for (auto &ints_pair : pq_str->ints) {
            const std::string &type = ints_pair.first;
            std::vector<integrals> & ints = ints_pair.second;
            for (auto & integral : ints) {
                size_t order = integral.labels.size() / 2;

                int left_a = 0;
                int left_b = 0;
                int right_a = 0;
                int right_b = 0;
                for (size_t l = 0; l < order; l++) {
                    if (integral.spin_labels[l] == "a" ) {
                        left_a++;
                    }else {
                        left_b++;
                    }
                    if (integral.spin_labels[l + order] == "a" ) {
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
            pq_str->skip = true;
            continue;
        }

        killit = false;

        // delta functions 
        for (size_t j = 0; j < in->deltas.size(); j++) {
            if (pq_str->deltas[j].spin_labels[0] != pq_str->deltas[j].spin_labels[1] ) {
                killit = true;
                break;
            }
        }

        if ( killit ) {
            pq_str->skip = true;
            continue;
        }
    }
    
    // rearrange terms so that they have standard spin order (abba -> -abab, etc.)
    for (auto & pq_str : tmp) {

        if ( pq_str->skip ) continue;

        // amplitudes
        for (auto &amp_pair : pq_str->amps) {
            char type = amp_pair.first;
            std::vector<amplitudes> & amps = amp_pair.second;
            for (auto & amp : amps) {

                size_t n_create = amp.n_create;
                size_t n_annihilate = amp.n_annihilate;

                if ( n_create > 4 || n_annihilate > 4 ) {
                    printf("\n");
                    printf("    error: spin tracing doesn't work for higher than quadruples yet\n");
                    printf("\n");
                    exit(1);
                }

                // order creator labels
                if ( n_create == 2 ) {

                    // only one case to worry about
                    if (amp.spin_labels[0] == "b"
                        && amp.spin_labels[1] == "a" ) {

                        amp.spin_labels[0] = "a";
                        amp.spin_labels[1] = "b";

                        std::string tmp_label = amp.labels[0];
                        amp.labels[0] = amp.labels[1];
                        amp.labels[1] = tmp_label;

                        pq_str->sign *= -1;
                    }

                }else if ( n_create == 3 ) {

                    // target order: aaa, aab, abb, bbb
                    int sign = 1;
                    reorder_three_spins(amp, 0, 1, 2, sign);
                    pq_str->sign *= sign;

                }else if ( n_create == 4 ) {

                    // target order: aaaa, aaab, aabb, abbb, bbbb
                    int sign = 1;
                    reorder_four_spins(amp, 0, 1, 2, 3, sign);
                    pq_str->sign *= sign;

                }

                // order annihilator labels
                if ( n_annihilate == 2 ) {

                    // only one case to worry about
                    if (amp.spin_labels[n_create + 0] == "b"
                        && amp.spin_labels[n_create + 1] == "a" ) {

                        amp.spin_labels[n_create + 0] = "a";
                        amp.spin_labels[n_create + 1] = "b";

                        std::string tmp_label = amp.labels[n_create + 0];
                        amp.labels[n_create + 0] = amp.labels[n_create + 1];
                        amp.labels[n_create + 1] = tmp_label;

                        pq_str->sign *= -1;
                    }

                }else if ( n_annihilate == 3 ) {

                    // target order: aaa, aab, abb, bbb
                    int sign = 1;
                    reorder_three_spins(amp, n_create + 0, n_create + 1, n_create + 2, sign);
                    pq_str->sign *= sign;

                }else if ( n_annihilate == 4 ) {

                    // target order: aaaa, aaab, aabb, abbb, bbbb
                    int sign = 1;
                    reorder_four_spins(amp, n_create + 0, n_create + 1, n_create + 2, n_create + 3, sign);
                    pq_str->sign *= sign;

                }
            }
        }

        // integrals
        for (auto &ints_pair : pq_str->ints) {
            const std::string &type = ints_pair.first;
            std::vector<integrals> & ints = ints_pair.second;
            for (auto & integral : ints) {

                size_t order = integral.labels.size() / 2;

                if ( order != 2 ) continue;

                // three cases that require attention: ab;ba, ba;ab, and ba;ba

                // integrals
                if (integral.spin_labels[0] == "a"
                    && integral.spin_labels[1] == "b"
                    && integral.spin_labels[2] == "b"
                    && integral.spin_labels[3] == "a" ) {

                        std::string tmp_label = integral.labels[2];
                    integral.labels[2] = integral.labels[3];
                    integral.labels[3] = tmp_label;

                    integral.spin_labels[2] = "a";
                    integral.spin_labels[3] = "b";

                    pq_str->sign *= -1;

                }else if (integral.spin_labels[0] == "b"
                          && integral.spin_labels[1] == "a"
                          && integral.spin_labels[2] == "a"
                          && integral.spin_labels[3] == "b" ) {

                        std::string tmp_label = integral.labels[0];
                    integral.labels[0] = integral.labels[1];
                    integral.labels[1] = tmp_label;

                    integral.spin_labels[0] = "a";
                    integral.spin_labels[1] = "b";

                    pq_str->sign *= -1;

                }else if (integral.spin_labels[0] == "b"
                          && integral.spin_labels[1] == "a"
                          && integral.spin_labels[2] == "b"
                          && integral.spin_labels[3] == "a" ) {

                        std::string tmp_label = integral.labels[0];
                    integral.labels[0] = integral.labels[1];
                    integral.labels[1] = tmp_label;

                    integral.spin_labels[0] = "a";
                    integral.spin_labels[1] = "b";

                        tmp_label = integral.labels[2];
                    integral.labels[2] = integral.labels[3];
                    integral.labels[3] = tmp_label;

                    integral.spin_labels[2] = "a";
                    integral.spin_labels[3] = "b";
                }
            }
        }
    }

    for (auto & pq_str : tmp) {
        if ( pq_str->skip ) continue;
        spin_blocked.push_back(pq_str);
    }
    tmp.clear();
}

} // End namespaces
