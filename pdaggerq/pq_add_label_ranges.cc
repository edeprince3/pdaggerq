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

#include "pq_string.h"
#include "pq_utils.h"
#include "pq_add_label_ranges.h"

#include <algorithm>

namespace pdaggerq {

/// expand sums to account for different orbital ranges and zero terms where appropriate
void add_label_ranges(const std::shared_ptr<pq_string>& in, std::vector<std::shared_ptr<pq_string> > &range_blocked, const std::unordered_map<std::string, std::vector<std::string>> &label_ranges) {

    // check that label ranges are valid
    for (auto item : label_ranges) {
        
        for (auto range : item.second) {
            // non-summed index? not perfect logic ...
            int found = in->index_in_anywhere(item.first);
            if ( found == 1 ) {
                if ( range != "act" && range != "ext" ) {
                    printf("\n");
                    printf("    error: label range for non-summed label %s is invalid\n", item.first.c_str());
                    printf("\n");
                    exit(1);
                }
            }else {
                if ( range != "act" && range != "ext" && range != "all" ) {
                    printf("\n");
                    printf("    error: label range for %s is invalid\n", item.first.c_str());
                    printf("\n");
                    exit(1);
                }
            }
        }
    }

    // check if label range map is missing any of the non-summed spin 
    // labels. note that the logic here might not be perfect since the 
    // keys in the label_ranges map can be non-summed labels or 
    // amplitude types (e.g., t1, t2, ...)

    // amplitudes
    for (auto &amps_pair : in->amps) {
        char type = amps_pair.first;
        std::vector<amplitudes> &amps_vec = amps_pair.second;
        for (amplitudes & amp : amps_vec) {
            for (size_t k = 0; k < amp.labels.size(); k++) {
                int n = in->index_in_anywhere(amp.labels[k]);
                if ( n != 1 ) continue;
                int found = label_ranges.count(amp.labels[k]);
                if ( found == 0 ) {
                    printf("\n");
                    printf("    error: label range for non-summed index %s has not been set\n", amp.labels[k].c_str());
                    printf("\n");
                    exit(1);
                }
            }
        }
    }
    // integrals
    for (auto &ints_pair : in->ints) {
        std::string type = ints_pair.first;
        std::vector<integrals> &ints_vec = ints_pair.second;
        for (integrals & integral : ints_vec) {
            for (size_t k = 0; k < integral.labels.size(); k++) {
                int n = in->index_in_anywhere(integral.labels[k]);
                if ( n != 1 ) continue;
                int found = label_ranges.count(integral.labels[k]);
                if ( found == 0 ) {
                    printf("\n");
                    printf("    error: label range for non-summed index %s has not been set\n", integral.labels[k].c_str());
                    printf("\n");
                    exit(1);
                }
            }
        }
    }
    // deltas
    for (delta_functions & delta : in->deltas) {
        for (size_t j = 0; j < delta.labels.size(); j++) {
            int n = in->index_in_anywhere(delta.labels[j]);
            if ( n != 1 ) continue;
            int found = label_ranges.count(delta.labels[j]);
            if ( found == 0 ) {
                printf("\n");
                printf("    error: label range for non-summed index %s has not been set\n", delta.labels[j].c_str());
                printf("\n");
                exit(1);
            }
        }
    }

    std::shared_ptr<pq_string> newguy (new pq_string(in->vacuum));
    newguy->copy(in.get());

    newguy->reset_label_ranges(label_ranges);

    // list of expanded sums
    std::vector< std::shared_ptr<pq_string> > tmp;
    tmp.push_back(newguy);

    // but first expand single permutations where ranges don't match 
    for (size_t i = 0; i < tmp.size(); i++) {

        std::shared_ptr<pq_string> & tmp_str = tmp[i];

        size_t n = tmp_str->permutations.size() / 2;

        for (size_t j = 0; j < n; j++) {

            std::string idx1 = tmp_str->permutations[2*j];
            std::string idx2 = tmp_str->permutations[2*j+1];

            // find label ranges
            auto pos1 = label_ranges.find(idx1);
            auto pos2 = label_ranges.find(idx2);
            
            // range 1 and 2
            std::vector<std::string> range1 = pos1 == label_ranges.end() ? std::vector<std::string>() : pos1->second;
            std::vector<std::string> range2 = pos2 == label_ranges.end() ? std::vector<std::string>() : pos2->second;

            // if ranges are not the same, then the permutation needs to be expanded explicitly before allowed ranges redetermined
            if ( range1 != range2 ) {

                // first guy is just a copy
                std::shared_ptr<pq_string> newguy1 (new pq_string(tmp_str->vacuum));
                newguy1->copy(tmp_str.get());

                // second guy is a copy with permuted labels and change in sign
                std::shared_ptr<pq_string> newguy2 (new pq_string(tmp_str->vacuum));
                newguy2->copy(tmp_str.get());
                swap_two_labels(newguy2, idx1, idx2);
                newguy2->sign *= -1;

                // reset non-summed label ranges for this guy
                newguy2->reset_label_ranges(label_ranges);

                // both guys need to have permutation lists adjusted
                newguy1->permutations.clear();
                newguy2->permutations.clear();

                for (size_t k = 0; k < n; k++) {

                    // skip jth permutation, which is the one we expanded
                    if ( j == k ) continue;

                    newguy1->permutations.push_back(tmp_str->permutations[2*k]);
                    newguy1->permutations.push_back(tmp_str->permutations[2*k+1]);

                    newguy2->permutations.push_back(tmp_str->permutations[2*k]);
                    newguy2->permutations.push_back(tmp_str->permutations[2*k+1]);
                }

                tmp_str->skip = true;
                tmp.push_back(newguy1);
                tmp.push_back(newguy2);

                // break loop over permutations because this above logic only works on one permutation at a time
                break;
            }
        }
    }

    // now expand paired permutations (3) where label ranges don't match TODO
    for (size_t i = 0; i < tmp.size(); i++) {

        std::shared_ptr<pq_string> & tmp_str = tmp[i];

        size_t n = tmp_str->paired_permutations_3.size() / 6;

        for (size_t j = 0; j < n; j++) {

            std::string o1 = tmp_str->paired_permutations_3[6 * j];
            std::string v1 = tmp_str->paired_permutations_3[6 * j + 1];
            std::string o2 = tmp_str->paired_permutations_3[6 * j + 2];
            std::string v2 = tmp_str->paired_permutations_3[6 * j + 3];
            std::string o3 = tmp_str->paired_permutations_3[6 * j + 4];
            std::string v3 = tmp_str->paired_permutations_3[6 * j + 5];

            // find occupied label ranges
            auto poso1 = label_ranges.find(o1);
            auto poso2 = label_ranges.find(o2);
            auto poso3 = label_ranges.find(o3);
            std::vector<std::string> rangeo1 = poso1 == label_ranges.end() ? std::vector<std::string>() : poso1->second;
            std::vector<std::string> rangeo2 = poso2 == label_ranges.end() ? std::vector<std::string>() : poso2->second;
            std::vector<std::string> rangeo3 = poso3 == label_ranges.end() ? std::vector<std::string>() : poso3->second;


            // find virtual label ranges
            auto posv1 = label_ranges.find(o1);
            auto posv2 = label_ranges.find(o2);
            auto posv3 = label_ranges.find(o3);
            std::vector<std::string> rangev1 = poso1 == label_ranges.end() ? std::vector<std::string>() : posv1->second;
            std::vector<std::string> rangev2 = poso2 == label_ranges.end() ? std::vector<std::string>() : posv2->second;
            std::vector<std::string> rangev3 = poso3 == label_ranges.end() ? std::vector<std::string>() : posv3->second;

        }
    }

    // now expand paired permutations (6) where label ranges don't match TODO
    for (size_t i = 0; i < tmp.size(); i++) {

        std::shared_ptr<pq_string> & tmp_str = tmp[i];

        size_t n = tmp_str->paired_permutations_6.size() / 6;

        for (size_t j = 0; j < n; j++) {

            std::string o1 = tmp_str->paired_permutations_6[6 * j];
            std::string v1 = tmp_str->paired_permutations_6[6 * j + 1];
            std::string o2 = tmp_str->paired_permutations_6[6 * j + 2];
            std::string v2 = tmp_str->paired_permutations_6[6 * j + 3];
            std::string o3 = tmp_str->paired_permutations_6[6 * j + 4];
            std::string v3 = tmp_str->paired_permutations_6[6 * j + 5];

            // find occupied label ranges
            auto poso1 = label_ranges.find(o1);
            auto poso2 = label_ranges.find(o2);
            auto poso3 = label_ranges.find(o3);
            std::vector<std::string> rangeo1 = poso1 == label_ranges.end() ? std::vector<std::string>() : poso1->second;
            std::vector<std::string> rangeo2 = poso2 == label_ranges.end() ? std::vector<std::string>() : poso2->second;
            std::vector<std::string> rangeo3 = poso3 == label_ranges.end() ? std::vector<std::string>() : poso3->second;


            // find virtual label ranges
            auto posv1 = label_ranges.find(o1);
            auto posv2 = label_ranges.find(o2);
            auto posv3 = label_ranges.find(o3);
            std::vector<std::string> rangev1 = poso1 == label_ranges.end() ? std::vector<std::string>() : posv1->second;
            std::vector<std::string> rangev2 = poso2 == label_ranges.end() ? std::vector<std::string>() : posv2->second;
            std::vector<std::string> rangev3 = poso3 == label_ranges.end() ? std::vector<std::string>() : posv3->second;
        }
    }

    // now, expand sums 

    bool done_adding_ranges = false;
    do {
        std::vector< std::shared_ptr<pq_string> > list;
        done_adding_ranges = true;
        for (const std::shared_ptr<pq_string> & tmp_str : tmp) {
            bool am_i_done = add_ranges_to_string(tmp_str, list);
            if ( !am_i_done ) done_adding_ranges = false;
        }
        if ( !done_adding_ranges ) {
            tmp.clear();
            for (std::shared_ptr<pq_string> & pq_str : list) {
                if ( !pq_str->skip ) {
                    tmp.push_back(pq_str);
                }
            }
        }
    }while(!done_adding_ranges);

    // kill deltas that have mismatched ranges
    for (std::shared_ptr<pq_string> & tmp_str : tmp) {

        if ( tmp_str->skip ) continue;

        bool killit = false;

        // delta functions 
        for (size_t j = 0; j < in->deltas.size(); j++) {
            if (tmp_str->deltas[j].label_ranges[0] != tmp_str->deltas[j].label_ranges[1] ) {
                killit = true;
                break;
            }
        }

        if ( killit ) {
            tmp_str->skip = true;
            continue;
        }
    }

    // kill terms with ranges inconsistent with what is in the map
    for (std::shared_ptr<pq_string> & tmp_str : tmp) {

        if ( tmp_str->skip ) continue;

        bool killit = false;

        // get desired ranges for amplitudes from map
        for (auto &amp_pair : tmp_str->amps) {
            char type = amp_pair.first;
            std::vector<amplitudes> & amps = amp_pair.second;
            for (amplitudes & amp : amps) {

                // amplitude type+order (ie 't' + '2' = "t2")
                std::string amptype;
                amptype.push_back(type);
                int order = amp.n_create;
                if (amp.n_annihilate > order) {
                    order = amp.n_annihilate;
                }
                amptype += std::to_string(order);
                
                // is this amplitude in the map? if not, we can assume full ranges are desired
                auto amp_pos = label_ranges.find(amptype); 
                if ( amp_pos == label_ranges.end() ) continue;
                
                // get desired ranges for this amplitude from map
                std::vector<std::string> label_range = amp_pos->second;

                // are the number of ranges provided by the user correct?
                if (label_range.size() != amp.label_ranges.size() ) {
                    printf("\n");
                    printf("    error: something is wrong with the number of ranges for %s\n", amptype.c_str());
                    printf("\n");
                    exit(1);
                }

                // check label ranges
                killit = do_ranges_differ(0, amp.n_create, "act", amp.label_ranges, label_range);
                if ( killit ) break;

                killit = do_ranges_differ(0, amp.n_create, "ext", amp.label_ranges, label_range);
                if ( killit ) break;

                killit = do_ranges_differ(amp.n_create, amp.n_create + amp.n_annihilate, "act", amp.label_ranges, label_range);
                if ( killit ) break;

                killit = do_ranges_differ(amp.n_create, amp.n_create + amp.n_annihilate, "ext", amp.label_ranges, label_range);
                if ( killit ) break;
            }
            if ( killit ) break;

        }
        if ( killit ) {
            tmp_str->skip = true;
            continue;
        }
    }

    // rearrange terms so that they have standard range order ( ae;ea -> -ae;ae etc. )
    for (std::shared_ptr<pq_string> & tmp_str : tmp) {

        if ( tmp_str->skip ) continue;

        // amplitudes
        for (auto &amp_pair : tmp_str->amps) {
            char type = amp_pair.first;
            std::vector<amplitudes> & amps = amp_pair.second;
            for (amplitudes & amp : amps) {

                size_t n_create = amp.n_create;
                size_t n_annihilate = amp.n_annihilate;

                if ( n_create > 4 || n_annihilate > 4) {
                    printf("\n");
                    printf("    error: label ranges don't work for higher than quadruples yet\n");
                    printf("\n");
                    exit(1);
                }

                int sign = 1;

                // reorder creation labels
                if ( n_create == 2 ) {
                    // target order: aa, ae, ee
                    reorder_two_ranges(amp, 0, 1, sign);
                }else if ( n_create == 3 ) {
                    // target order: aaa, aae, aee, eee
                    reorder_three_ranges(amp, 0, 1, 2, sign);
                }else if ( n_create == 4 ) {
                    // target order: aaaa, aaae, aaee, aeee, eeee
                    reorder_four_ranges(amp, 0, 1, 2, 3, sign);
                }

                // signs for annihilation labels
                if ( n_annihilate == 2 ) {
                    // target order: aa, ae, ee
                    reorder_two_ranges(amp, n_create, n_create + 1, sign);
                }else if ( n_annihilate == 3 ) {
                    // target order: aaa, aae, aee, eee
                    reorder_three_ranges(amp, n_create, n_create + 1, n_create + 2, sign);
                }else if ( n_annihilate == 4 ) {
                    // target order: aaaa, aaae, aaee, aeee, eeee
                    reorder_four_ranges(amp, n_create, n_create + 1, n_create + 2, n_create + 3, sign);
                }

                tmp_str->sign *= sign;
            }
        }

        // integrals
        for (auto &int_pair : tmp_str->ints) {
            std::string type = int_pair.first;
            std::vector<integrals> & ints = int_pair.second;
            for (integrals & integral : ints) {

                size_t order = integral.labels.size() / 2;

                if ( order != 2 ) continue;

                // target order: aa, ae, ee
                int sign = 1;
                reorder_two_ranges(integral, 0, 1, sign);
                reorder_two_ranges(integral, 2, 3, sign);
                tmp_str->sign *= sign;
            }
        }
    }

    for (auto & tmp_str : tmp) {
        if ( tmp_str->skip ) continue;
        range_blocked.push_back(tmp_str);
    }
    tmp.clear();
}

// reorder two ranges ... only one case to consider: ba -> ab
void reorder_two_ranges(tensor & tens, int i1, int i2, int & sign) {

    if (       tens.label_ranges[i1] == "ext"
            && tens.label_ranges[i2] == "act" ) {

            std::string tmp_label = tens.labels[i2];

            tens.labels[i2] = tens.labels[i1];
            tens.labels[i1] = tmp_label;

            tens.label_ranges[i1] = "act";
            tens.label_ranges[i2] = "ext";

            sign *= -1;

    }
}

// reorder three ranges ... cases to consider: aba/baa -> aab; bba/bab -> abb
void reorder_three_ranges(amplitudes & amps, int i1, int i2, int i3, int & sign) {

    if (       amps.label_ranges[i1] == "act"
            && amps.label_ranges[i2] == "ext"
            && amps.label_ranges[i3] == "act" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.label_ranges[i2] = "act";
            amps.label_ranges[i3] = "ext";

            sign *= -1;

    }else if ( amps.label_ranges[i1] == "ext"
            && amps.label_ranges[i2] == "act"
            && amps.label_ranges[i3] == "act" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.label_ranges[i1] = "act";
            amps.label_ranges[i3] = "ext";

            sign *= -1;

    }else if ( amps.label_ranges[i1] == "ext"
            && amps.label_ranges[i2] == "ext"
            && amps.label_ranges[i3] == "act" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.label_ranges[i1] = "act";
            amps.label_ranges[i3] = "ext";

            sign *= -1;

    }else if ( amps.label_ranges[i1] == "ext"
            && amps.label_ranges[i2] == "act"
            && amps.label_ranges[i3] == "ext" ) {

            std::string tmp_label = amps.labels[i2];

            amps.labels[i2] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.label_ranges[i1] = "act";
            amps.label_ranges[i2] = "ext";

            sign *= -1;

    }

}

// do ranges in two strings differ? second string should be the map that could contain "all"
bool do_ranges_differ(size_t start, size_t end, const std::string& range, const std::vector<std::string> &in1, const std::vector<std::string> &in2) {

    // number of input ranges in current amplitude
    int n1 = 0;
    int n2 = 0;
    int nall = 0;
    for (size_t m = start; m < end; m++) {
        if ( in1[m] == "all" ) {
            printf("\n");
            printf("    error: first string range should not include 'all'\n");
            printf("\n");
            exit(1);
        }
        if ( in1[m] == range ) n1++;
        if ( in2[m] == range ) n2++;
        if ( in2[m] == "all" ) nall++;
    }

    if ( n1 == n2 ) return false;
    if ( n1 <= n2 + nall ) return false;
    return true;
}

// add label ranges to a string
bool add_ranges_to_string(const std::shared_ptr<pq_string>& in, std::vector<std::shared_ptr<pq_string> > &list) {

    if ( in->skip ) return true;

    bool all_ranges_added = false;

    // amplitudes
    for (auto &amp_pair : in->amps) {
        char type = amp_pair.first;
        std::vector<amplitudes> & amps = amp_pair.second;
        for (amplitudes & amp : amps) {
            for (size_t k = 0; k < amp.labels.size(); k++) {
                if ( amp.label_ranges[k].empty() ) {

                    std::shared_ptr<pq_string> act (new pq_string(in->vacuum));
                    std::shared_ptr<pq_string> ext (new pq_string(in->vacuum));

                    act->copy(in.get());
                    ext->copy(in.get());

                    act->set_range_everywhere(amp.labels[k], "act");
                    ext->set_range_everywhere(amp.labels[k], "ext");

                    list.push_back(act);
                    list.push_back(ext);
                    return false;
                }
            }
        }
    }

    // integrals
    for (auto &int_pair : in->ints) {
        std::string type = int_pair.first;
        std::vector<integrals> & ints = int_pair.second;
        for (integrals & integral : ints) {
        for (size_t k = 0; k < integral.labels.size(); k++) {
            if ( integral.label_ranges[k].empty() ) {

                std::shared_ptr<pq_string> act (new pq_string(in->vacuum));
                std::shared_ptr<pq_string> ext (new pq_string(in->vacuum));

                act->copy(in.get());
                ext->copy(in.get());

                act->set_range_everywhere(integral.labels[k], "act");
                ext->set_range_everywhere(integral.labels[k], "ext");

                list.push_back(act);
                list.push_back(ext);
                return false;
            }
        }
    }
}

    // must be done.
    return true;
}

// reorder four label ranges ... cases to consider: aaba/abaa/baaa -> aaab; baab/abba/baba/bbaa/abab -> aabb; babb/bbab/bbba -> abbb
void reorder_four_ranges(amplitudes & amps, int i1, int i2, int i3, int i4, int & sign) {

    // aaba/abaa/baaa -> aaab
    if (       amps.label_ranges[i1] == "act"
            && amps.label_ranges[i2] == "act"
            && amps.label_ranges[i3] == "ext"
            && amps.label_ranges[i4] == "act" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i3];
            amps.labels[i3] = tmp_label;

            amps.label_ranges[i3] = "act";
            amps.label_ranges[i4] = "ext";

            sign *= -1;

    }else if ( amps.label_ranges[i1] == "act"
            && amps.label_ranges[i2] == "ext"
            && amps.label_ranges[i3] == "act"
            && amps.label_ranges[i4] == "act" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.label_ranges[i2] = "act";
            amps.label_ranges[i4] = "ext";

            sign *= -1;

    }else if ( amps.label_ranges[i1] == "ext"
            && amps.label_ranges[i2] == "act"
            && amps.label_ranges[i3] == "act"
            && amps.label_ranges[i4] == "act" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.label_ranges[i1] = "act";
            amps.label_ranges[i4] = "ext";

            sign *= -1;

    // baab/abba/baba/bbaa/abab -> aabb
    }else if ( amps.label_ranges[i1] == "ext"
            && amps.label_ranges[i2] == "act"
            && amps.label_ranges[i3] == "act"
            && amps.label_ranges[i4] == "ext" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.label_ranges[i1] = "act";
            amps.label_ranges[i3] = "ext";

            sign *= -1;

    }else if ( amps.label_ranges[i1] == "act"
            && amps.label_ranges[i2] == "ext"
            && amps.label_ranges[i3] == "ext"
            && amps.label_ranges[i4] == "act" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.label_ranges[i2] = "act";
            amps.label_ranges[i4] = "ext";

            sign *= -1;

    }else if ( amps.label_ranges[i1] == "ext"
            && amps.label_ranges[i2] == "act"
            && amps.label_ranges[i3] == "ext"
            && amps.label_ranges[i4] == "act" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.label_ranges[i1] = "act";
            amps.label_ranges[i4] = "ext";

            sign *= -1;

    }else if ( amps.label_ranges[i1] == "ext"
            && amps.label_ranges[i2] == "ext"
            && amps.label_ranges[i3] == "act"
            && amps.label_ranges[i4] == "act" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.label_ranges[i2] = "act";
            amps.label_ranges[i4] = "ext";

            tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.label_ranges[i1] = "act";
            amps.label_ranges[i3] = "ext";

    }else if ( amps.label_ranges[i1] == "act"
            && amps.label_ranges[i2] == "ext"
            && amps.label_ranges[i3] == "act"
            && amps.label_ranges[i4] == "ext" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i2];
            amps.labels[i2] = tmp_label;

            amps.label_ranges[i2] = "act";
            amps.label_ranges[i3] = "ext";

            sign *= -1;

    // babb/bbab/bbba -> abbb
    }else if ( amps.label_ranges[i1] == "ext"
            && amps.label_ranges[i2] == "act"
            && amps.label_ranges[i3] == "ext"
            && amps.label_ranges[i4] == "ext" ) {

            std::string tmp_label = amps.labels[i2];

            amps.labels[i2] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.label_ranges[i1] = "act";
            amps.label_ranges[i2] = "ext";

            sign *= -1;

    }else if ( amps.label_ranges[i1] == "ext"
            && amps.label_ranges[i2] == "ext"
            && amps.label_ranges[i3] == "act"
            && amps.label_ranges[i4] == "ext" ) {

            std::string tmp_label = amps.labels[i3];

            amps.labels[i3] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.label_ranges[i1] = "act";
            amps.label_ranges[i3] = "ext";

            sign *= -1;

    }else if ( amps.label_ranges[i1] == "ext"
            && amps.label_ranges[i2] == "ext"
            && amps.label_ranges[i3] == "ext"
            && amps.label_ranges[i4] == "act" ) {

            std::string tmp_label = amps.labels[i4];

            amps.labels[i4] = amps.labels[i1];
            amps.labels[i1] = tmp_label;

            amps.label_ranges[i1] = "act";
            amps.label_ranges[i4] = "ext";

            sign *= -1;
    }
}

} // End namespaces
