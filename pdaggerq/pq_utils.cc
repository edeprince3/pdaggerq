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
#include "pq_swap_operators.h"
#include "pq_add_spin_labels.h"

#include <algorithm>
#include <numeric>

namespace pdaggerq {

// concatinate a list of operators (a list of strings) into a single list
std::vector<std::string> concatinate_operators(const std::vector<std::vector<std::string>> &ops) {

    std::vector<std::string> ret;
    // determine size to reserve when concatenating
    size_t size = std::accumulate(ops.begin(), ops.end(), 0,
                                  [](size_t sum, const std::vector<std::string> & op){
        return sum + op.size();
    });

    ret.reserve(size);
    std::for_each(ops.begin(), ops.end(), [&ret](const std::vector<std::string> & op){
        ret.insert(ret.end(), op.begin(), op.end());
    });

    return ret;
}

// remove "*" from std::string
void removeStar(std::string &x) {

  auto it = std::remove_if(std::begin(x), std::end(x), [](char c){return (c == '*');});
  x.erase(it, std::end(x));
}

// remove "(" and ")" from std::string
void removeParentheses(std::string &x) {

  auto it = std::remove_if(std::begin(x), std::end(x), [](char c){return (c == '(');});
  x.erase(it, std::end(x));

  it = std::remove_if(std::begin(x), std::end(x), [](char c){return (c == ')');});
  x.erase(it, std::end(x));
}

// remove " " from std::string
void removeSpaces(std::string &x) {

  auto it = std::remove_if(std::begin(x), std::end(x), [](char c){return (c == ' ');});
  x.erase(it, std::end(x));
}

// is a label classified as occupied?
bool is_occ(const std::string &idx) {

    // replacing above with comparison along char range
    if (idx.empty()) return false;

    // use integer comparison for speed
    char c_idx = idx.at(0);
    if ( c_idx >= 'i' && c_idx <= 'n' ) return true;
    else if ( c_idx >= 'I' && c_idx <= 'N' ) return true;
    else if ( c_idx == 'O' || c_idx == 'o' ) return true;
    return false;
}

// is a label classified as virtual?
bool is_vir(const std::string &idx) {
    if (idx.empty()) return false;

    // use integer comparison for speed
    char c_idx = idx.at(0);
    if ( c_idx >= 'a' && c_idx <= 'f' ) return true;
    else if ( c_idx >= 'A' && c_idx <= 'F' ) return true;
    else if ( c_idx == 'V' || c_idx == 'v' ) return true;
    return false;
}

// how many times does an index appear deltas?
int index_in_deltas(const std::string &idx, const std::vector<delta_functions> &deltas) {

    int n = 0;
    for (const delta_functions & delta : deltas) {
        if ( delta.labels[0] == idx ) {
            n++;
        }
        if ( delta.labels[1] == idx ) {
            n++;
        }
    }
    return n;
}

// how many times does an index appear integrals?
int index_in_integrals(const std::string &idx, const std::vector<integrals> &ints) {

    int n = 0;
    for (const integrals & integral : ints) {
        for (const std::string & label : integral.labels) {
            if (label == idx ) {
                n++;
            }
        }
    }
    return n;
}

// how many times does an index appear in amplitudes?
int index_in_amplitudes(const std::string &idx, const std::vector<amplitudes> &amps) {

    int n = 0;
    for (const amplitudes & amp : amps) {
        for (const std::string & label : amp.labels) {
            if ( label == idx ) {
                n++;
            }
        }
    }
    return n;
}

// how many times does an index appear in operators (symbol)?
int index_in_operators(const std::string &idx, const std::vector<std::string> &ops) {

    int n = 0;
    for (const std::string & op : ops) {
        if ( op == idx ) {
            n++;
        }
    }
    return n;
}

// how many times does an index appear amplitudes, deltas, integrals, and operators?
int index_in_anywhere(const std::shared_ptr<pq_string> &in, const std::string &idx) {

    // find index in deltas
    int n = index_in_deltas(idx, in->deltas);

    // find index in integrals
    for (const auto & int_pair : in->ints) {
        const std::string &type = int_pair.first;
        const std::vector<integrals> &ints = int_pair.second;
        n += index_in_integrals(idx, ints);
    }

    // find index in amplitudes
    for (const auto & amp_pair : in->amps) {
        const char &type = amp_pair.first;
        const std::vector<amplitudes> &amps = amp_pair.second;
        n += index_in_amplitudes(idx, amps);
    }

    // find index in operators
    n += index_in_operators(idx, in->symbol);

    return n;
}

/// replace one label with another (in a given set of deltas)
void replace_index_in_deltas(const std::string &old_idx, const std::string &new_idx, std::vector<delta_functions> &deltas) {

    for (delta_functions & delta : deltas) {
        if ( delta.labels[0] == old_idx ) {
            delta.labels[0] = new_idx;
        }
        if ( delta.labels[1] == old_idx ) {
            delta.labels[1] = new_idx;
        }
    }
}

/// replace one label with another (in a given set of amplitudes)
void replace_index_in_amplitudes(const std::string &old_idx, const std::string &new_idx, std::vector<amplitudes> &amps) {

    for (amplitudes & amp : amps) {
        for (std::string & label : amp.labels) {
            if ( label == old_idx ) {
                label = new_idx;
            }
        }
    }
}

/// replace one label with another (in a given set of integrals)
void replace_index_in_integrals(const std::string &old_idx, const std::string &new_idx, std::vector<integrals> &ints) {

    for (integrals & integral : ints) {
        for (std::string & label : integral.labels) {
            if (label == old_idx ) {
                label = new_idx;
            }
        }
    }
}

/// replace one label with another (in a given set of operators (symbol))
void replace_index_in_operators(const std::string &old_idx, const std::string &new_idx, std::vector<std::string> &ops) {

    for (std::string & op : ops) {
        if (op == old_idx ) {
            op = new_idx;
        }
    }
}

// swap two labels
void swap_two_labels(std::shared_ptr<pq_string> &in, const std::string &label1, const std::string &label2) {

    replace_index_everywhere(in, label1, "xyz");
    replace_index_everywhere(in, label2, label1);
    replace_index_everywhere(in, "xyz", label2);
}

// replace one label with another (in integrals, amplitudes, and operators)
void replace_index_everywhere(std::shared_ptr<pq_string> &in, const std::string &old_idx, const std::string &new_idx) {

    for (auto &int_pair : in->ints) {
        std::string type = int_pair.first;
        std::vector<integrals> &ints = int_pair.second;
        replace_index_in_integrals(old_idx, new_idx, ints);
    }

    for (auto &amp_pair : in->amps) {
        char type = amp_pair.first;
        std::vector<amplitudes> &amps = amp_pair.second;
        replace_index_in_amplitudes(old_idx, new_idx, amps);
    }

    replace_index_in_operators(old_idx, new_idx, in->symbol);

    replace_index_in_deltas(old_idx, new_idx, in->deltas);

    in->sort_labels();
}

/// compare two lists of integrals
bool compare_integrals(const std::vector<integrals> &ints1,
                       const std::vector<integrals> &ints2,
                       int & n_permute ) {

    if ( ints1.size() != ints2.size() ) return false;

    size_t nsame_ints = 0;
    for (const integrals & int1 : ints1) {
        for (const integrals & int2 : ints2) {

            if (int1 == int2 ) {

                n_permute += int1.permutations + int2.permutations;

                nsame_ints++;
                break;
            }
        }
    }

    if ( nsame_ints != ints1.size() ) return false;

    return true;
}

/// compare two lists of amplitudes
bool compare_amplitudes( const std::vector<amplitudes> &amps1,
                         const std::vector<amplitudes> &amps2,
                         int & n_permute ) {

    if ( amps1.size() != amps2.size() ) return false;
   
    size_t nsame_amps = 0;
    for (const amplitudes & amp1 : amps1) {
        for (const amplitudes & amp2 : amps2) {

            if (amp1 == amp2 ) {

                n_permute += amp1.permutations + amp2.permutations;

                nsame_amps++;
                break;
            }
        }
    }

    if ( nsame_amps != amps1.size() ) return false;

    return true;
}

// compare two strings
bool compare_strings(const std::shared_ptr<pq_string> &ordered_1, const std::shared_ptr<pq_string> &ordered_2, int & n_permute) {

    // don't forget w0
    if ( ordered_1->has_w0 != ordered_2->has_w0 ) {
        return false;
    }

    // are strings same?
    if ( ordered_1->symbol.size() != ordered_2->symbol.size() ) return false;
    int nsame_s = 0;
    for (size_t k = 0; k < ordered_1->symbol.size(); k++) {
        if ( ordered_1->symbol[k] == ordered_2->symbol[k] ) {
            nsame_s++;
        }
    }
    if ( nsame_s != ordered_1->symbol.size() ) return false;

    // same delta functions (recall these aren't sorted in any way)
    int nsame_d = 0;
    for (const delta_functions & deltas1 : ordered_1->deltas) {
        for (const delta_functions & deltas2 : ordered_2->deltas) {
            if ( deltas1.labels[0] == deltas2.labels[0]
              && deltas1.labels[1] == deltas2.labels[1] ) {
                nsame_d++;
                //break;
            }else if ( deltas1.labels[0] == deltas2.labels[1]
                    && deltas1.labels[1] == deltas2.labels[0] ) {
                nsame_d++;
                //break;
            }
        }
    }
    if ( nsame_d != ordered_1->deltas.size() ) return false;

    // amplitude comparisons, with permutations
    n_permute = 0;

    bool same_string = false;
    for (const auto &amp_pair : ordered_1->amps) {
        char type = amp_pair.first;
        const std::vector<amplitudes> &amps1 = amp_pair.second;

        // ensure that the same amplitude type exists in both strings
        if ( ordered_2->amps.find(type) == ordered_2->amps.end() ) {
            return false; // nope
        }

        // compare amplitudes
        const std::vector<amplitudes> &amps2 = ordered_2->amps.at(type);
        same_string = compare_amplitudes(amps1, amps2, n_permute);
        if ( !same_string ) return false;
    }

    // integral comparisons, with permutations
    for (const auto &int_pair : ordered_1->ints) {
        std::string type = int_pair.first;
        const std::vector<integrals> &ints1 = int_pair.second;

        // ensure that the same integral type exists in both strings
        if ( ordered_2->ints.find(type) == ordered_2->ints.end() ) {
            return false; // nope
        }

        // compare integrals
        const std::vector<integrals> &ints2 = ordered_2->ints.at(type);
        same_string = compare_integrals(ints1, ints2, n_permute);
        if ( !same_string ) return false;
    }

    // also need to check if the permutations are the same...
    // otherwise, we shouldn't be combining these terms
    if ( ordered_1->permutations.size() != ordered_2->permutations.size() ) {
        return false;
    }

    int nsame_permutations = 0;
    // remember, permutations come in pairs
    size_t n = ordered_1->permutations.size() / 2;
    int count = 0;
    for (int i = 0; i < n; i++) {

        if ( ordered_1->permutations[count] == ordered_2->permutations[count] ) {
            nsame_permutations++;
        }else if (  ordered_1->permutations[count]   == ordered_2->permutations[count+1] ) {
            nsame_permutations++;
        }else if (  ordered_1->permutations[count+1] == ordered_2->permutations[count]   ) {
            nsame_permutations++;
        }else if (  ordered_1->permutations[count+1] == ordered_2->permutations[count+1] ) {
            nsame_permutations++;
        }
        count += 2;
    }
    if ( nsame_permutations != n ) {
        return false;
    }

    return true;
}

/// compare two strings when swapping (multiple) summed labels
void compare_strings_with_swapped_summed_labels(const std::vector<std::vector<std::string> > &labels,
                                                size_t iter,
                                                const std::shared_ptr<pq_string> &in1,
                                                const std::shared_ptr<pq_string> &in2,
                                                int & n_permute, 
                                                bool & strings_same) {
 
    if ( iter == labels.size() ) {
        strings_same = compare_strings(in2, in1, n_permute);
        return;
    }

    // try swapping non-summed labels
    for (size_t id1 = 0; id1 < labels[iter].size(); id1++) {
        for (size_t id2 = id1 + 1; id2 < labels[iter].size(); id2++) {
    
            std::shared_ptr<pq_string> newguy = std::make_shared<pq_string>(*in1);
            swap_two_labels(newguy, labels[iter][id1], labels[iter][id2]);
            newguy->sort_labels();

            compare_strings_with_swapped_summed_labels(labels, iter+1, newguy, in2, n_permute, strings_same);
            if ( strings_same ) return;
        }
    }
}

// consolidate terms that differ may differ by permutations of summed labels
void consolidate_permutations_plus_swaps(std::vector<std::shared_ptr<pq_string> > &ordered,
                                     const std::vector<std::vector<std::string> > &labels) {

    //TODO:
    // Currently the implementation of this function runs in O(N^2) time complexity and is the limiting bottleneck by far.
    // This can be remedied by reformulating the logic to run in O(N) time complexity using a hash table.
    // However, that would require us to define a hash function for pq_string (doable, but not trivial).
    // For now, we'll just live with the O(N^2) time complexity.

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<std::vector<std::string> > found_labels;

        // ok, what summed / repeated labels do we have?
        for (const std::vector<std::string> & label : labels) {
            std::vector<std::string> tmp;
            tmp.reserve(label.size());
            for (const auto & index : label) {
                int found = index_in_anywhere(ordered[i], index);
                if ( found == 2 ) {
                    tmp.push_back(index);
                }
            }
            found_labels.push_back(tmp);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i], ordered[j], n_permute);

            compare_strings_with_swapped_summed_labels(found_labels, 0, ordered[i], ordered[j], n_permute, strings_same);

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->factor * ordered[i]->sign;
            double factor_j = ordered[j]->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0, n_permute);

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

// consolidate terms that differ by permutations of non-summed labels
void consolidate_permutations_non_summed(
    std::vector<std::shared_ptr<pq_string> > &ordered,
    const std::vector<std::string> &labels) {

    for (size_t i = 0; i < ordered.size(); i++) {

        // not sure if this logic works with existing permutation operators ... skip those for now
        //if ( !ordered[i]->permutations.empty() ) continue;

        if ( !ordered[i]->paired_permutations_2.empty() ) continue;
        if ( !ordered[i]->paired_permutations_3.empty() ) continue;
        if ( !ordered[i]->paired_permutations_6.empty() ) continue;
        
        if ( ordered[i]->skip ) continue;
    
        std::vector<int> find_idx;
    
        // ok, what labels do we have? 
        for (const auto & label : labels) {
            int found = index_in_anywhere(ordered[i], label);
            // this is buggy when existing permutation labels belong to 
            // the same space as the labels we're permuting ... so skip those for now.
            bool same_space = false;
            bool is_occ1 = is_occ(label);
            for (const auto & permutation : ordered[i]->permutations) {
                bool is_occ2 = is_occ(permutation);
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

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i], ordered[j], n_permute);

            // now that we've identified some permutations, it is possible for strings to be the same without swaps
            if (strings_same) {

                double factor_i = ordered[i]->factor * ordered[i]->sign;
                double factor_j = ordered[j]->factor * ordered[j]->sign;

                double combined_factor = factor_i + factor_j * pow(-1.0, n_permute);

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

            std::string permutation_1;
            std::string permutation_2;

            // try swapping non-summed labels
            for (size_t id1 = 0; id1 < labels.size(); id1++) {
                if ( find_idx[id1] != 1 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels.size(); id2++) {
                    if ( find_idx[id2] != 1 ) continue;

                    std::shared_ptr<pq_string> newguy = std::make_shared<pq_string>(*ordered[i]);
                    swap_two_labels(newguy, labels[id1], labels[id2]);

                    strings_same = compare_strings(ordered[j], newguy, n_permute);

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

            double factor_i = ordered[i]->factor * ordered[i]->sign;
            double factor_j = ordered[j]->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0, n_permute);

            // if terms exactly cancel, then this is a permutation
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->permutations.push_back(permutation_1);
                ordered[i]->permutations.push_back(permutation_2);
                ordered[j]->skip = true;
                break;
            }

            // otherwise, something has gone wrong in the previous consolidation step...
        }
    }
}

/// compare two strings when swapping (multiple) summed labels and ov pairs of nonsumed labels
void compare_strings_with_swapped_summed_and_nonsummed_labels(
    const std::vector<std::vector<std::string> > &labels,
    const std::vector<std::vector<std::string>> &pairs,
    size_t iter,
    const std::shared_ptr<pq_string> &in1,
    const std::shared_ptr<pq_string> &in2,
    size_t in2_id,
    std::vector<size_t> &my_permutations,
    std::vector<bool> &permutation_types,
    int n_permutation_type,
    int & n_permute, 
    bool & strings_same,
    bool & found_paired_permutation) {
 
    if ( iter == labels.size() ) {

        strings_same = compare_strings(in2, in1, n_permute);

        // try swapping three pairs of non-summed labels
        for (size_t pair1 = 0; pair1 < pairs.size(); pair1++) {
            std::string o1 = pairs[pair1][0];
            std::string v1 = pairs[pair1][1];
            for (size_t pair2 = pair1 + 1; pair2 < pairs.size(); pair2++) {
                std::string o2 = pairs[pair2][0];
                if ( o2 == o1 ) continue;
                std::string v2 = pairs[pair2][1];
                if ( v2 == v1 ) continue;
                for (size_t pair3 = pair2 + 1; pair3 < pairs.size(); pair3++) {
                    std::string o3 = pairs[pair3][0];
                    if ( o3 == o2 ) continue;
                    if ( o3 == o1 ) continue;
                    std::string v3 = pairs[pair3][1];
                    if ( v3 == v2 ) continue;
                    if ( v3 == v1 ) continue;

                    bool paired_permutation = false;

                    // for determining type PP3 permutations
                    int found_permutation_type = -1;

                    for (size_t permutation_type = 0; permutation_type < n_permutation_type; permutation_type++) {

                        std::shared_ptr<pq_string> newguy = std::make_shared<pq_string>(*in1);

                        if ( permutation_type == 0 ) {

                            // 1 <-> 2
                            swap_two_labels(newguy, o1, o2);
                            swap_two_labels(newguy, v1, v2);

                        }else if ( permutation_type == 1 ) {

                            // 1 <-> 3
                            swap_two_labels(newguy, o1, o3);
                            swap_two_labels(newguy, v1, v3);

                        }else if ( permutation_type == 2 ) {

                            // 2 <-> 3
                            swap_two_labels(newguy, o2, o3);
                            swap_two_labels(newguy, v2, v3);

                        }else if ( permutation_type == 3 ) {

                            // only relevant for 6-fold permutations:

                            // 1 <-> 2
                            swap_two_labels(newguy, o1, o2);
                            swap_two_labels(newguy, v1, v2);

                            // 1 <-> 3
                            swap_two_labels(newguy, o1, o3);
                            swap_two_labels(newguy, v1, v3);

                        }else if ( permutation_type == 4 ) {

                            // only relevant for 6-fold permutations:

                            // 1 <-> 2
                            swap_two_labels(newguy, o1, o2);
                            swap_two_labels(newguy, v1, v2);

                            // 2 <-> 3
                            swap_two_labels(newguy, o2, o3);
                            swap_two_labels(newguy, v2, v3);

                        }
                        newguy->sort_labels();

                        strings_same = compare_strings(in2, newguy, n_permute);

                        if ( strings_same ) {
                            paired_permutation = true;
                            found_permutation_type = (int)permutation_type;
                            break;
                        }
                    }

                    if ( !paired_permutation ) break;

                    double factor_i = in1->factor * in1->sign;
                    double factor_j = in2->factor * in2->sign;

                    double combined_factor = factor_i - factor_j * pow(-1.0,n_permute);

                    // if factors are identical, then this is a paired permutation
                    if ( fabs(combined_factor) < 1e-12 ) {
                        //ordered[j]->print();

                        // keep track of which term this is
                        my_permutations.push_back(in2_id);

                        found_paired_permutation = true;

                        // keep track of which labels were swapped (for 3-fold)
                        permutation_types[found_permutation_type] = true;
                    }
                    if ( found_paired_permutation ) break;
                }
                if ( found_paired_permutation ) break;
            }
            if ( found_paired_permutation ) break;
        }
        return;
    }

    // try swapping non-summed labels
    for (size_t id1 = 0; id1 < labels[iter].size(); id1++) {
        for (size_t id2 = id1 + 1; id2 < labels[iter].size(); id2++) {
    
            std::shared_ptr<pq_string> newguy = std::make_shared<pq_string>(*in1);
            swap_two_labels(newguy, labels[iter][id1], labels[iter][id2]);
            newguy->sort_labels();

            compare_strings_with_swapped_summed_and_nonsummed_labels(labels, 
                                                                     pairs, 
                                                                     iter+1, 
                                                                     newguy, 
                                                                     in2, 
                                                                     in2_id, 
                                                                     my_permutations, 
                                                                     permutation_types, 
                                                                     n_permutation_type, 
                                                                     n_permute, 
                                                                     strings_same, 
                                                                     found_paired_permutation);
            if ( strings_same ) return;
        }
    }
}


// look for paired permutations:
// a) PP6(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + R(ikj;acb) + R(jik;bac) + R(jki;bca) + R(kij;cab) + R(kji;cba)
// b) PP3(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + (jik;bac) + R(kji;cba)
void consolidate_paired_permutations_non_summed(
    std::vector<std::shared_ptr<pq_string> > &ordered,
    const std::vector<std::string> &occ_labels,
    const std::vector<std::string> &vir_labels,
    int n_fold) {

    if ( n_fold != 3 && n_fold !=6 ) {
        printf("\n");
        printf("    error: consolidate_paired_permutations_non_summed only searches for 3- or 6-fold paired permutations.\n");
        printf("\n");
        exit(1);
    }

    int n_permutation_type = 5;
    if ( n_fold == 3 ) {
        n_permutation_type = 3;
    }

    // look for n-fold permutations
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        // not sure if this logic works with existing permutation operators ... skip those for now
        if ( !ordered[i]->permutations.empty() ) continue;
        if ( !ordered[i]->paired_permutations_2.empty() ) continue;
        if ( !ordered[i]->paired_permutations_3.empty() ) continue;
        if ( !ordered[i]->paired_permutations_6.empty() ) continue;

        std::vector<std::string> found_occ;
        std::vector<std::string> found_vir;
        std::vector<std::string> found_summed_occ;
        std::vector<std::string> found_summed_vir;

        // ok, what non-summed occupied labels do we have? 
        for (const std::string & occ_label : occ_labels) {
            int found = index_in_anywhere(ordered[i], occ_label);
            if ( found == 1 ) {
                found_occ.push_back(occ_label);
            }
        }

        // ok, what non-summed virtual labels do we have? 
        for (const std::string & vir_label : vir_labels) {
            int found = index_in_anywhere(ordered[i], vir_label);
            if ( found == 1 ) {
                found_vir.push_back(vir_label);
            }
        }

        // ok, what summed labels (occupied and virtual) do we have? 
        for (const std::string & occ_label : occ_labels) {
            int found = index_in_anywhere(ordered[i], occ_label);
            if ( found == 2 ) {
                found_summed_occ.push_back(occ_label);
            }
        }
        for (const std::string & vir_label : vir_labels) {
            int found = index_in_anywhere(ordered[i], vir_label);
            if ( found == 2 ) {
                found_summed_vir.push_back(vir_label);
            }
        }

        // this function only works for swapping exactly three ov pairs
        if ( found_occ.size() != 3 || found_vir.size() != 3 ) continue;

        // ov pairs to swap
        std::vector<std::vector<std::string>> pairs;
        pairs.push_back({found_occ[0], found_vir[0]});
        pairs.push_back({found_occ[1], found_vir[1]});
        pairs.push_back({found_occ[2], found_vir[2]});

        // which labels are involve in the permutation?
        std::vector<size_t> my_permutations;

        // which pairs are swapped ( 12, 13, 23 ) ... this affects how we label 3-fold permutations
        std::vector<bool> permutation_types = { false, false, false };

        // loop over other strings
        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            // not sure if this logic works with existing permutation operators ... skip those for now
            if ( !ordered[j]->permutations.empty() ) continue;
            if ( !ordered[i]->paired_permutations_2.empty() ) continue;
            if ( !ordered[i]->paired_permutations_3.empty() ) continue;
            if ( !ordered[i]->paired_permutations_6.empty() ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            bool found_paired_permutation = false;
            std::vector<std::vector<std::vector<std::string> > > labels;
            labels.emplace_back();
            labels.push_back({found_summed_occ});
            labels.push_back({found_summed_vir});
            for (const std::vector<std::vector<std::string>> & label : labels) {
                compare_strings_with_swapped_summed_and_nonsummed_labels(label,
                                                                         pairs, 
                                                                         0, 
                                                                         ordered[i], 
                                                                         ordered[j], 
                                                                         j, 
                                                                         my_permutations, 
                                                                         permutation_types, 
                                                                         n_permutation_type, 
                                                                         n_permute, 
                                                                         strings_same, 
                                                                         found_paired_permutation);
                if ( found_paired_permutation ) break;
            }
        }

        if ( my_permutations.size() == 5 && n_fold == 6) {
            // 6-fold permutations
            for (unsigned long my_permutation : my_permutations) {
                ordered[my_permutation]->skip = true;
            }
            ordered[i]->paired_permutations_6.push_back(found_occ[0]);
            ordered[i]->paired_permutations_6.push_back(found_vir[0]);
            ordered[i]->paired_permutations_6.push_back(found_occ[1]);
            ordered[i]->paired_permutations_6.push_back(found_vir[1]);
            ordered[i]->paired_permutations_6.push_back(found_occ[2]);
            ordered[i]->paired_permutations_6.push_back(found_vir[2]);
        }else if ( my_permutations.size() == 2 && n_fold == 3 ) {
            // 3-fold permutations
            for (unsigned long my_permutation : my_permutations) {
                ordered[my_permutation]->skip = true;
            }
            if ( permutation_types[0] && permutation_types[1] && permutation_types[2] ) {
                printf("\n");
                printf("    something has gone terribly wrong in consolidate_paired_permutations_non_summed()\n");
                printf("\n");
                exit(1);
            }
            if ( permutation_types[0] && permutation_types[1] ) {
                ordered[i]->paired_permutations_3.push_back(found_occ[0]);
                ordered[i]->paired_permutations_3.push_back(found_vir[0]);
                ordered[i]->paired_permutations_3.push_back(found_occ[1]);
                ordered[i]->paired_permutations_3.push_back(found_vir[1]);
                ordered[i]->paired_permutations_3.push_back(found_occ[2]);
                ordered[i]->paired_permutations_3.push_back(found_vir[2]);
            }else if ( permutation_types[0] && permutation_types[2] ) {
                ordered[i]->paired_permutations_3.push_back(found_occ[1]);
                ordered[i]->paired_permutations_3.push_back(found_vir[1]);
                ordered[i]->paired_permutations_3.push_back(found_occ[0]);
                ordered[i]->paired_permutations_3.push_back(found_vir[0]);
                ordered[i]->paired_permutations_3.push_back(found_occ[2]);
                ordered[i]->paired_permutations_3.push_back(found_vir[2]);
            }else if ( permutation_types[1] && permutation_types[2] ) {
                ordered[i]->paired_permutations_3.push_back(found_occ[2]);
                ordered[i]->paired_permutations_3.push_back(found_vir[2]);
                ordered[i]->paired_permutations_3.push_back(found_occ[0]);
                ordered[i]->paired_permutations_3.push_back(found_vir[0]);
                ordered[i]->paired_permutations_3.push_back(found_occ[1]);
                ordered[i]->paired_permutations_3.push_back(found_vir[1]);
            }
        }
    }
}

/// alphabetize operators to simplify string comparisons (for true vacuum only)
void alphabetize(std::vector<std::shared_ptr<pq_string> > &ordered) {

    // alphabetize string
    for (std::shared_ptr<pq_string> & pq_str : ordered) {

        // creation
        bool not_alphabetized = false;
        do {
            not_alphabetized = false;
            int ndagger = 0;
            for (size_t j = 0; j < pq_str->symbol.size(); j++) {
                if ( pq_str->is_dagger[j] ) ndagger++;
            }
            for (int j = 0; j < ndagger-1; j++) {
                int val1 = pq_str->symbol[j].c_str()[0];
                int val2 = pq_str->symbol[j + 1].c_str()[0];
                if ( val2 < val1 ) {
                    std::string dum = pq_str->symbol[j];
                    pq_str->symbol[j] = pq_str->symbol[j + 1];
                    pq_str->symbol[j + 1] = dum;
                    pq_str->sign = -pq_str->sign;
                    not_alphabetized = true;
                    j = pq_str->symbol.size() + 1;
                    not_alphabetized = true;
                }
            }
        }while(not_alphabetized);
        // annihilation
        not_alphabetized = false;
        do {
            not_alphabetized = false;
            int ndagger = 0;
            for (size_t j = 0; j < pq_str->symbol.size(); j++) {
                if ( pq_str->is_dagger[j] ) ndagger++;
            }
            for (int j = ndagger; j < (int)pq_str->symbol.size() - 1; j++) {
                int val1 = pq_str->symbol[j].c_str()[0];
                int val2 = pq_str->symbol[j + 1].c_str()[0];
                if ( val2 < val1 ) {
                    std::string dum = pq_str->symbol[j];
                    pq_str->symbol[j] = pq_str->symbol[j + 1];
                    pq_str->symbol[j + 1] = dum;
                    pq_str->sign = -pq_str->sign;
                    not_alphabetized = true;
                    j = pq_str->symbol.size() + 1;
                    not_alphabetized = true;
                }
            }
        }while(not_alphabetized);
    }
        
    // alphabetize deltas
    for (std::shared_ptr<pq_string> & pq_str : ordered) {
        for (delta_functions & delta : pq_str->deltas) {
            int val1 = delta.labels[0].c_str()[0];
            int val2 = delta.labels[1].c_str()[0];
            if ( val2 < val1 ) {
                std::string dum = delta.labels[0];
                delta.labels[0] = delta.labels[1];
                delta.labels[1] = dum;
            }
        }
    }
}

// compare strings and remove terms that cancel
void cleanup(std::vector<std::shared_ptr<pq_string> > &ordered, bool find_paired_permutations) {

    for (std::shared_ptr<pq_string> & pq_str : ordered) {

        // order amplitudes such that they're ordered t1, t2, t3, etc.
        reorder_t_amplitudes(pq_str);

        // sort amplitude labels
        pq_str->sort_labels();

    }

    // prune list so it only contains non-skipped ones
    std::vector< std::shared_ptr<pq_string> > pruned;
    for (const std::shared_ptr<pq_string> & pq_str : ordered) {

        if ( pq_str->skip ) continue;

        // for normal order relative to fermi vacuum, i doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming
        if (pq_str->vacuum == "FERMI" ) {
            if ( !pq_str->symbol.empty() ) continue;
            if ( !pq_str->is_boson_dagger.empty() ) continue;
        }

        pruned.push_back(pq_str);
    }
    ordered.clear();
    for (const std::shared_ptr<pq_string> & pq_str : pruned) {
        ordered.push_back(pq_str);
    }
    pruned.clear();

    std::vector<std::string> occ_labels { "i", "j", "k", "l", "m", "n", "I", "J", "K", "L", "M", "N" };
    std::vector<std::string> vir_labels { "a", "b", "c", "d", "e", "f", "A", "B", "C", "D", "E", "F" };

    // swap up to two non-summed labels (more doesn't seem to be necessary for up to ccsdtq)

    consolidate_permutations_plus_swaps(ordered, {});

    consolidate_permutations_plus_swaps(ordered, {occ_labels});
    consolidate_permutations_plus_swaps(ordered, {vir_labels});

    consolidate_permutations_plus_swaps(ordered, {occ_labels, occ_labels});
    consolidate_permutations_plus_swaps(ordered, {vir_labels, vir_labels});
    consolidate_permutations_plus_swaps(ordered, {occ_labels, vir_labels});

    // probably only relevant for vacuum = fermi
    if ( ordered.empty() ) return;

    // probably only relevant for vacuum = fermi
    if ( ordered[0]->vacuum != "FERMI" ) return;

    // look for paired permutations of non-summed labels:
    if ( find_paired_permutations ) {

        // a) PP6(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + R(ikj;acb) + R(jik;bac) + R(jki;bca) + R(kij;cab) + R(kji;cba)
        consolidate_paired_permutations_non_summed(ordered, occ_labels, vir_labels, 6);

        // b) PP3(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + (jik;bac) + R(kji;cba)
        consolidate_paired_permutations_non_summed(ordered, occ_labels, vir_labels, 3);

    }

    consolidate_permutations_non_summed(ordered, occ_labels);
    consolidate_permutations_non_summed(ordered, vir_labels);

    // re-prune
    pruned.clear();
    for (const std::shared_ptr<pq_string> & pq_str : ordered) {

        if ( pq_str->skip ) continue;

        // for normal order relative to fermi vacuum, pq_str doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming
        if (pq_str->vacuum == "FERMI" ) {
            if ( !pq_str->symbol.empty() ) continue;
            if ( !pq_str->is_boson_dagger.empty() ) continue;
        }

        pruned.push_back(pq_str);
    }
    ordered.clear();
    for (std::shared_ptr<pq_string> & pq_str : pruned) {
        ordered.push_back(pq_str);
    }
    pruned.clear();
}

// reorder t amplitudes as t1, t2, t3, t4
void reorder_t_amplitudes(std::shared_ptr<pq_string> &in) {

    // get t amplitudes
    auto amp_pos = in->amps.find('t');
    if ( amp_pos == in->amps.end() ) return; // no t amplitudes
    
    std::vector<amplitudes> & t_amps = amp_pos->second;
        
    size_t dim = t_amps.size();
    if ( dim == 0 ) return;
    
    bool* nope = (bool*)malloc(dim * sizeof(bool));
    memset((void*)nope, '\0', dim * sizeof(bool));

    std::vector<std::vector<std::string> > tmp;
    std::vector<amplitudes> tmp_new;

    for (size_t order = 1; order < 7; order++) {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if ( nope[j] ) continue;

                if ( t_amps[j].labels.size() == 2 * order ) {
                    tmp_new.push_back(t_amps[j]);
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
        
        //TODO: (MDL 11/14/23) 
        // likely this was because of the implementation of the assignment operator for the amplitude class. 
        // This could be fixed now, but requires running that expensive test again
    }
    
    t_amps = tmp_new;
    free(nope);
}

// re-classify fluctuation potential terms
void reclassify_integrals(std::shared_ptr<pq_string> &in) {

    //return;
    
    // find if occ_repulsion is present
    auto occ_pos = in->ints.find("occ_repulsion");
    if ( occ_pos == in->ints.end() ) return;
    
    std::vector<integrals> & occ_repulsion = occ_pos->second;
    
    //if ( occ_repulsion.size() > 1 ) {
    //   printf("\n");
    //   printf("error: only support for one integral type object per string\n");
    //   printf("\n");
    //   exit(1);
    //}
   
    static std::vector<std::string> occ_out {"i", "j", "k", "l", "m", "n", "I", "J", "K", "L", "M", "N", 
                                             "i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9"};

    for (size_t i = 0; i < in->ints["occ_repulsion"].size(); i++) {

        // pick summation label not included in string already
        std::string idx;
        
        int do_skip = -999;
        
        for (size_t i = 0; i < occ_out.size(); i++) {
            if ( index_in_anywhere(in, occ_out[i]) == 0 ) {
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
        
        std::string idx1 = occ_repulsion[i].labels[0];
        std::string idx2 = occ_repulsion[i].labels[1];

        // new eri
        integrals ints;
        
        ints.labels.clear();
        ints.numerical_labels.clear();
        
        ints.labels.push_back(idx1);
        ints.labels.push_back(idx);
        ints.labels.push_back(idx2);
        ints.labels.push_back(idx);
        
        ints.sort();
        
        in->ints["eri"].push_back(ints);
    }
    in->ints["occ_repulsion"].clear();

}

// find and replace any funny labels in integrals with conventional ones. i.e., o1 -> i ,v1 -> a
void use_conventional_labels(std::shared_ptr<pq_string> &in) {

    // occupied first:
    static std::vector<std::string> occ_in{"o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9",
                                    "o10", "o11", "o12", "o13", "o14", "o15", "o16", "o17", "o18", "o19",
                                    "o20", "o21", "o22", "o23", "o24", "o25", "o26", "o27", "o28", "o29"};
    static std::vector<std::string> occ_out{"i", "j", "k", "l", "m", "n", "I", "J", "K", "L", "M", "N",
                                     "i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9",
                                     "i10", "i11", "i12", "i13", "i14", "i15", "i16", "i17", "i18", "i19"};

    for (const std::string & in_idx : occ_in) {

        if (index_in_anywhere(in, in_idx) > 0 ) {

            for (const std::string & out_idx : occ_out) {

                if (index_in_anywhere(in, out_idx) == 0 ) {

                    replace_index_everywhere(in, in_idx, out_idx);
                    break;
                }
            }
        }
    }

    // now virtual
    static std::vector<std::string> vir_in{"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                                    "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                                    "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29"};
    static std::vector<std::string> vir_out{"a", "b", "c", "d", "e", "f", "A", "B", "C", "D", "E", "F",
                                     "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9",
                                     "a10", "a11", "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19"};

    for (const std::string & in_idx : vir_in) {

        if (index_in_anywhere(in, in_idx) > 0 ) {

            for (const std::string & out_idx : vir_out) {

                if (index_in_anywhere(in, out_idx) == 0 ) {

                    replace_index_everywhere(in, in_idx, out_idx);
                    break;
                }
            }
        }
    }

    // now general
    static std::vector<std::string> gen_in{"p0", "p1", "p2", "p3"};
    static std::vector<std::string> gen_out{"p", "q", "r", "s"};

    for (const std::string & in_idx : gen_in) {

        if (index_in_anywhere(in, in_idx) > 0 ) {

            for (const std::string & out_idx : gen_out) {

                if (index_in_anywhere(in, out_idx) == 0 ) {

                    replace_index_everywhere(in, in_idx, out_idx);
                    break;
                }
            }
        }
    }
}

/// apply delta functions to amplitude and integral labels
void gobble_deltas(std::shared_ptr<pq_string> &in) {
    
    std::vector<std::string> tmp_delta1;
    std::vector<std::string> tmp_delta2;
    
    // create list of summation labels. only consider internally-created labels
                                     
    static std::vector<std::string> occ_labels{"o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9",
                                    "o10", "o11", "o12", "o13", "o14", "o15", "o16", "o17", "o18", "o19",
                                    "o20", "o21", "o22", "o23", "o24", "o25", "o26", "o27", "o28", "o29", "i", "j", "k", "l", "m", "n"};
    static std::vector<std::string> vir_labels{"v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                                    "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                                    "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "a", "b", "c", "d", "e", "f"};
    static std::vector<std::string> gen_labels{"p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9",
                                    "p10", "p11", "p12", "p13", "p14", "p15", "p16", "p17", "p18", "p19",
                                    "p20", "p21", "p22", "p23", "p24", "p25", "p26", "p27", "p28", "p29", "p", "q", "r", "s", "t", "u"};

    std::vector<std::string> sum_labels;
    for (const std::string & occ_label : occ_labels) {
        if ( index_in_anywhere(in, occ_label) == 2 ) {
            sum_labels.push_back(occ_label);
        }       
    }       
    for (const std::string & vir_label : vir_labels) {
        if ( index_in_anywhere(in, vir_label) == 2 ) {
            sum_labels.push_back(vir_label);
        }
    }
    for (const std::string & gen_label : gen_labels) {
        if ( index_in_anywhere(in, gen_label) == 2 ) {
            sum_labels.push_back(gen_label);
        }
    }
                                    
//    for (size_t i = 0; i < in->deltas.size(); i++) {
    for ( delta_functions & delta : in->deltas ) {
    
        // is delta label 1 in list of summation labels?
        bool have_delta1 = false;    
        for (const std::string & sum_label : sum_labels) {
            if ( delta.labels[0] == sum_label ) {
                have_delta1 = true;
                break;
            }
        }   
        // is delta label 2 in list of summation labels?
        bool have_delta2 = false;
        for (const std::string & sum_label : sum_labels) {
            if ( delta.labels[1] == sum_label ) {
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
            replace_index_everywhere( deltas[i].labels[0], deltas[i].labels[1] );
            continue;
        }else if ( have_delta2 ) {
            replace_index_everywhere( deltas[i].labels[1], deltas[i].labels[0] );
            continue;               
        }                           
*/

        bool do_continue = false;
        for (auto & int_pair : in->ints) {
            std::string type = int_pair.first;
            std::vector<integrals> & ints = int_pair.second;
            
            if ( have_delta1 && index_in_integrals( delta.labels[0], ints ) > 0 ) {
               replace_index_in_integrals( delta.labels[0], delta.labels[1], ints );
               do_continue = true;
               break;
            }else if ( have_delta2 && index_in_integrals( delta.labels[1], ints ) > 0 ) {
               replace_index_in_integrals( delta.labels[1], delta.labels[0], ints );
               do_continue = true;
               break;
            }
        }
        if ( do_continue ) continue;

        // TODO: note that the code only efficiently collects terms when the amplitude
        // list is ordered as {'t', 'l', 'r', 'u', 'm', 's'} ... i don't know why, but
        // i do know that this is the problematic part of the code

        /*TODO: The order of the amplitude types happen to coincide
         * with the order of descending number of amplitudes. This can be remedied by sorting the
         * types by number of amplitudes. an implementation of this is below, however this changes the
         * order of the indexing and cannot directly be compared with the test suite.
         * However, visual inspection of the output shows that the results are analytically identical.

                char types[] {'t', 'l', 'r', 'u', 'm', 's'};
                static int types_index[] {0, 1, 2, 3, 4, 5};

                // the amplitude type order will be set by the number of terms
                std::sort(types_index, types_index + 6, [&types, &in](int i1, int i2) {
                    return in->amps[types[i1]].size() > in->amps[types[i2]].size();
                });

                do_continue = false;
                for (auto & type_index : types_index)
                    char type = types[type_index];
                    std::vector<amplitudes> & amps = in->amps[type];
                    (... etc...)
         * */


        do_continue = false;
        static char types[] = {'t', 'l', 'r', 'u', 'm', 's', 'a', 'b', 'c', 'd', 'e', 'f', 'A', 'B', 'C', 'D', 'E', 'F', 'I', 'J', 'K', 'L', 'M', 'N'};
        //static char types[] = {'t', 'l', 'r', 'u', 'm', 's'};
        for (auto & type : types) {
            std::vector<amplitudes> & amps = in->amps[type];
            
            if ( have_delta1 && index_in_amplitudes( delta.labels[0], amps ) > 0 ) {
               replace_index_in_amplitudes( delta.labels[0], delta.labels[1], amps );
               do_continue = true;
               break;
            }else if ( have_delta2 && index_in_amplitudes( delta.labels[1], in->amps[type] ) > 0 ) {
               replace_index_in_amplitudes( delta.labels[1], delta.labels[0], amps );
               do_continue = true;
               break;
            }
        }
        if ( do_continue ) continue;

        // at this point, it is safe to assume the delta function must remain
        tmp_delta1.push_back(delta.labels[0]);
        tmp_delta2.push_back(delta.labels[1]);
    }

    in->deltas.clear();

    for (size_t i = 0; i < tmp_delta1.size(); i++) {

        delta_functions deltas;
        deltas.labels.push_back(tmp_delta1[i]);
        deltas.labels.push_back(tmp_delta2[i]);
        deltas.sort();
        in->deltas.push_back(deltas);
    }
}

// bring a new string to normal order and add to list of normal ordered strings (fermi vacuum)
void add_new_string_true_vacuum(const std::shared_ptr<pq_string> &in, std::vector<std::shared_ptr<pq_string> > &ordered, int print_level, bool find_paired_permutations){

    if ( in->factor > 0.0 ) {
        in->sign = 1;
        in->factor = fabs(in->factor);
    }else {
        in->sign = -1;
        in->factor = fabs(in->factor);
    }

    for (size_t i = 0; i < in->string.size(); i++) {
        std::string me = in->string[i];
        if ( me.find('*') != std::string::npos ) {
            removeStar(me);
            in->is_dagger.push_back(true);
        }else {
            in->is_dagger.push_back(false);
        }
        in->symbol.push_back(me);
    }

    if ( print_level > 0 ) {
        printf("\n");
        printf("    ");
        printf("// starting string:\n");
        in->print();
    }

    // rearrange strings
    std::vector< std::shared_ptr<pq_string> > tmp;
    tmp.push_back(in);

    bool done_rearranging = false;
    do { 
        std::vector< std::shared_ptr<pq_string> > list;
        done_rearranging = true;
        for (const std::shared_ptr<pq_string> & pq_str : tmp) {
            bool am_i_done = swap_operators_true_vacuum(pq_str, list);
            if ( !am_i_done ) done_rearranging = false;
        }
        tmp.clear();
        for (const std::shared_ptr<pq_string> & pq_str : list) {
            tmp.push_back(pq_str);
        }
    }while(!done_rearranging);

    for (const std::shared_ptr<pq_string> & pq_str : tmp) {
        ordered.push_back(pq_str);
    }
    tmp.clear();

    // alphabetize
    alphabetize(ordered);

    // try to cancel similar terms
    cleanup(ordered, find_paired_permutations);
}

// expand general labels, p -> o, v
bool expand_general_labels(const std::shared_ptr<pq_string> & in, std::vector<std::shared_ptr<pq_string> > & list, int occ_label_count, int vir_label_count) {

    for (size_t i = 0; i < in->string.size(); i++) {

        std::string me = in->string[i];

        std::string me_nostar = me;
	std::string maybe_a_star = "";
        if (me_nostar.find('*') != std::string::npos ){
	    maybe_a_star = "*";
            removeStar(me_nostar);
        }

        // is this a general label?
        if ( !is_occ(me_nostar) && !is_vir(me_nostar) ) {

            std::shared_ptr<pq_string> newguy_occ = std::make_shared<pq_string>(in.get(), true);
            std::shared_ptr<pq_string> newguy_vir = std::make_shared<pq_string>(in.get(), true);

	    std::string occ_label = "o" + std::to_string(occ_label_count+1);
	    std::string vir_label = "v" + std::to_string(vir_label_count+1);

            newguy_occ->string = in->string;
            newguy_vir->string = in->string;

            newguy_occ->string[i] = occ_label + maybe_a_star;
            newguy_vir->string[i] = vir_label + maybe_a_star;

            replace_index_everywhere(newguy_occ, me_nostar, occ_label);
            replace_index_everywhere(newguy_vir, me_nostar, vir_label);

            list.push_back(newguy_occ);
            list.push_back(newguy_vir);

            return false;
	}
    }
    return true;
}

// bring a new string to normal order and add to list of normal ordered strings (fermi vacuum)
void add_new_string_fermi_vacuum(const std::shared_ptr<pq_string> &in, std::vector<std::shared_ptr<pq_string> > &ordered, int print_level, bool find_paired_permutations, int occ_label_count, int vir_label_count){
        
    // if normal order is defined with respect to the fermi vacuum, we must
    // check here if the input string contains any general-index operators
    // (h, g, f, and v). If it does, then the string must be split to account 
    // explicitly for sums over occupied and virtual labels

    std::vector< std::shared_ptr<pq_string> > mystrings;
    mystrings.push_back(in);

    bool done_expanding = false;
    do {
        std::vector< std::shared_ptr<pq_string> > list;
        done_expanding = true;
        for (const std::shared_ptr<pq_string> & pq_str : mystrings) {
            bool am_i_done = expand_general_labels(pq_str, list, occ_label_count, vir_label_count);
            if ( !am_i_done ) done_expanding = false;
        }
        if (!done_expanding) {
            mystrings.clear();
            for (std::shared_ptr<pq_string> & pq_str : list) {
                mystrings.push_back(pq_str);
            }
            occ_label_count++;
            vir_label_count++;
        }
    }while(!done_expanding);

    // now, we need to convert the list "mystrings[i]->string" into symbols and daggers
    for (auto & mystring: mystrings ) {
        for (size_t i = 0; i < mystring->string.size(); i++) {
            std::string me = mystring->string[i];

            std::string me_nostar = me;
            if (me_nostar.find('*') != std::string::npos ){
                removeStar(me_nostar);
            }

            if ( is_vir(me_nostar) ) {
                if (me.find('*') != std::string::npos ){
                    mystring->is_dagger.push_back(true);
                    mystring->is_dagger_fermi.push_back(true);
                }else {
                    mystring->is_dagger.push_back(false);
                    mystring->is_dagger_fermi.push_back(false);
                }
                mystring->symbol.push_back(me_nostar);
            }else if ( is_occ(me_nostar) ) {
                if (me.find('*') != std::string::npos ){
                    mystring->is_dagger.push_back(true);
                    mystring->is_dagger_fermi.push_back(false);
                }else {
                    mystring->is_dagger.push_back(false);
                    mystring->is_dagger_fermi.push_back(true);
                }
                mystring->symbol.push_back(me_nostar);
            }
        }
    }

    // at this point, we've expanded all of the general labels
    // and are ready to bring the strings to normal order

    for (auto & mystring: mystrings ) {

        // rearrange strings
	//
        if ( print_level > 0 ) {
            printf("\n");
            printf("    ");
            printf("// starting string:\n");
            mystring->print();
        }

        std::vector< std::shared_ptr<pq_string> > tmp;
        tmp.push_back(mystring);

        bool done_rearranging = false;
        do {
            std::vector< std::shared_ptr<pq_string> > list;
            done_rearranging = true;
            for (const std::shared_ptr<pq_string> & pq_str : tmp) {
                bool am_i_done = swap_operators_fermi_vacuum(pq_str, list);
                if ( !am_i_done ) done_rearranging = false;
            }
            tmp.clear();
            for (std::shared_ptr<pq_string> & pq_str : list) {
                if ( !pq_str->skip ) {
                    tmp.push_back(pq_str);
                }
            }
        }while(!done_rearranging);

        //ordered.clear();
        for (const std::shared_ptr<pq_string> & pq_str : tmp) {
            ordered.push_back(pq_str);
        }
        tmp.clear();
    }
}

} // End namespaces
