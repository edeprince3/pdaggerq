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

#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace pdaggerq {

// Fast look-ahead structure to characterize an unmapped label's environment
struct ConnectivitySignature {
    // We sort based on: (1) Operator Type priority, (2) Size of the tensor, (3) Sorted peer pool index
    size_t operator_rank = 0;
    int operator_type_weight = 0; // T vs L vs R vs Integral
    size_t pool_position = 0;     // Index inside the sorted symmetry block of that downstream tensor
};

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
bool is_nuclear(const std::string &idx) {
    // a nuclear orbital label carries the species prefix followed by a label,
    // e.g. "pi" (occupied) or "pa" (virtual). a lone prefix char is not nuclear.
    return idx.size() > 1 && idx.at(0) == nuclear_prefix;
}

bool is_dummy(const std::string &idx) {
    // the internal summation labels the normal-ordering machinery hands out are
    // "o#" / "v#". a nuclear one carries the species prefix ("no#" / "nv#"), so the
    // classification has to be made inside the label's own species.
    const std::string base = is_nuclear(idx) ? idx.substr(1) : idx;
    return base.rfind("o", 0) == 0 || base.rfind("v", 0) == 0;
}

bool is_occ(const std::string &idx) {

    // replacing above with comparison along char range
    if (idx.empty()) return false;

    // nuclear labels carry a species prefix; classify the remaining label so
    // that occupied/virtual is determined within the label's own species space
    if ( is_nuclear(idx) ) return is_occ(idx.substr(1));

    // use integer comparison for speed
    char c_idx = idx.at(0);
    if ( c_idx >= 'i' && c_idx <= 'n' ) return true;
    else if ( c_idx >= 'I' && c_idx <= 'N' ) return true;
    else if ( c_idx == 'O' || c_idx == 'o' ) {
        // avoid categorizing a lone 'o' or 'O' as an occupied label
        if ( idx.size() > 1 ) {
            return true;
        }
    }
    return false;
}

// is a label classified as virtual?
bool is_vir(const std::string &idx) {
    if (idx.empty()) return false;

    // nuclear labels carry a species prefix; classify the remaining label
    if ( is_nuclear(idx) ) return is_vir(idx.substr(1));

    // use integer comparison for speed
    char c_idx = idx.at(0);
    if ( c_idx >= 'a' && c_idx <= 'f' ) return true;
    else if ( c_idx >= 'A' && c_idx <= 'F' ) return true;
    else if ( c_idx == 'V' || c_idx == 'v' ) {
        // avoid categorizing a lone 'v' or 'V' as an occupied label
        if ( idx.size() > 1 ) {
            return true;
        }
    }
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

std::unordered_map<std::string, int> count_labels(const std::shared_ptr<pq_string> &in, std::vector<char> ignore_amps = {}) {

    // map to store string label -> occurrence count
    std::unordered_map<std::string, int> counts;

    // helper to update the map
    auto add_to_counts = [&](const std::vector<std::string>& labels) {
        for (const auto& label : labels) {
            counts[label]++;
        }
    };
    
    // deltas
    for (const auto& delta : in->deltas) {
        add_to_counts(delta.labels);
    }
    
    // integrals
    for (const auto& int_pair : in->ints) {
        for (const auto& integral : int_pair.second) {
            add_to_counts(integral.labels);
        }
    }
    
    // amplitudes
    for (const auto& amp_pair : in->amps) {
        if (std::find(ignore_amps.begin(), ignore_amps.end(), amp_pair.first) != ignore_amps.end()) {
            continue; 
        }
        for (const auto& amp : amp_pair.second) {
            add_to_counts(amp.labels);
        }
    }
    
    // operators
    add_to_counts(in->symbol);

    return counts;
}

// does an index appear amplitudes, deltas, integrals, and operators?
bool keep_ucc_term(const std::shared_ptr<pq_string> &in) {

    // check if this term contains an eom operator (r or l)
    bool has_eom_operator = (in->amps.count('r') > 0 || in->amps.count('l') > 0);
    if (!has_eom_operator) { 
        return true;
    }

    // get all summed and non-summed labels
    std::unordered_map<std::string, int> counts = count_labels(in);

    // summed and non-summed labels
    std::unordered_set<std::string> summed_labels;
    std::unordered_set<std::string> non_summed_labels;
    
    for (const auto& [label, count] : counts) {
        if (count == 2) {
            summed_labels.insert(label);
            //printf("summed labels:     %s\n", label.c_str());
        }else if (count == 1) {
            non_summed_labels.insert(label);
            //printf("non-summed labels: %s\n", label.c_str());
        }else {
            printf("\n");
            printf("    a label appears %i times\n", count);
            printf("\n");
            exit(1);
        }
    }

    // count labels on hbar (excluding r and l)
    std::unordered_map<std::string, int> hbar_counts = count_labels(in, {'r', 'l'});

    bool hbar_has_shared_sum = false;
    bool hbar_has_external_leg = false;

    for (const auto& [label, count] : hbar_counts) {
        // summed_label appearing in hbar once must be shared with L or R
        if (count == 1 && summed_labels.count(label)) {
            hbar_has_shared_sum = true;
        }

        // non_summed_label appearing once in hbar is an index of the sigma vector
        if (count == 1 && non_summed_labels.count(label)) {
            hbar_has_external_leg = true;
        }
    }

    // apply connectivity rules

    // if hbar shares no sums with R or L, it's explicitly disconnected.
    if (!hbar_has_shared_sum) {
        //printf("explicitly disconnected\n");
        //in->print();
        return false;
    }

    // if hbar contributes no legs to sigma, it's a dangling artifact
    if (!non_summed_labels.empty() && !hbar_has_external_leg) {
        //printf("dangling artifact (no external legs)\n");
        //in->print();
        return false;
    }

    return true;
}

// does an index appear amplitudes, deltas, integrals, and operators?
bool found_index_anywhere(const std::shared_ptr<pq_string> &in, const std::string &idx) {

    // find index in deltas
    for (const delta_functions & delta : in->deltas) {
        if ( std::find(delta.labels.begin(), delta.labels.end(), idx) != delta.labels.end() ) return true;
    }

    // find index in integrals
    for (const auto & int_pair : in->ints) {
        const std::string &type = int_pair.first;
        const std::vector<integrals> &ints = int_pair.second;
        for (const integrals & integral : ints) {
            if ( std::find(integral.labels.begin(), integral.labels.end(), idx) != integral.labels.end() ) return true;
        }
    }

    // find index in amplitudes
    for (const auto & amp_pair : in->amps) {
        const char &type = amp_pair.first;
        const std::vector<amplitudes> &amps = amp_pair.second;
        for (const amplitudes & amp : amps) {
            if ( std::find(amp.labels.begin(), amp.labels.end(), idx) != amp.labels.end() ) return true;
        }
    }

    // find index in operators
    if ( std::find(in->symbol.begin(), in->symbol.end(), idx) != in->symbol.end() ) return true;

    return false;
}

/// replace one label with another (in a given set of permutations)
void replace_index_in_permutations(const std::string &old_idx, const std::string &new_idx, std::vector<std::string> &permutations) {

    for (std::string & label : permutations) {
        if ( label == old_idx ) {
            label = new_idx;
        }
    }
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
    in->sort();
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

    //replace_index_in_permutations(old_idx, new_idx, in->permutations);
    //in->sort();
}

// compare two strings
bool compare_strings(const std::shared_ptr<pq_string> &ordered_1, const std::shared_ptr<pq_string> &ordered_2, int & n_permute) {

    if ( ordered_1->key != ordered_2->key ) {
        return false;
    }

    // accumulate permutations of amplitudes
    n_permute = 0;
    for (const auto &amp_pair : ordered_1->amps) {
        char type = amp_pair.first;
        const std::vector<amplitudes> &amps1 = amp_pair.second;
        const std::vector<amplitudes> &amps2 = ordered_2->amps.at(type);
        for (size_t i = 0; i < amps1.size(); i++) {
            n_permute += amps1[i].permutations + amps2[i].permutations;
        }
    }

    // accumulate permutations of integrals
    for (const auto &int_pair : ordered_1->ints) {
        std::string type = int_pair.first;
        const std::vector<integrals> &ints1 = int_pair.second;
        const std::vector<integrals> &ints2 = ordered_2->ints.at(type);
        for (size_t i = 0; i < ints1.size(); i++) {
            n_permute += ints1[i].permutations + ints2[i].permutations;
        }
    }

    return true;
}

/// check map for strings when swapping (multiple) summed labels
std::string check_map_for_strings_with_swapped_summed_labels(
    const std::vector<std::vector<std::string> > &labels,
    size_t iter,
    const std::shared_ptr<pq_string> &in,
    const std::unordered_map<std::string, size_t> & string_map,
    std::vector<std::shared_ptr<pq_string> > &ordered,
    int & n_permute, 
    bool & string_in_map) {
 
    if ( iter == labels.size() ) {

        std::string key = in->key;

        // is string in map?
        if (string_map.find(key) != string_map.end()) {

            string_in_map = true; 
            //std::shared_ptr<pq_string> existing = std::make_shared<pq_string>(*string_map.at(key));
            size_t j = string_map.at(key);

            // accumulate permutations of amplitudes
            n_permute = 0;
            for (const auto &amp_pair : in->amps) {
                char type = amp_pair.first;
                const std::vector<amplitudes> &amps1 = amp_pair.second;
                //const std::vector<amplitudes> &amps2 = existing->amps.at(type);
                const std::vector<amplitudes> &amps2 = ordered[j]->amps.at(type);
                for (size_t i = 0; i < amps1.size(); i++) {
                    n_permute += amps1[i].permutations + amps2[i].permutations;
                }
            }

            // accumulate permutations of integrals
            for (const auto &int_pair : in->ints) {
                std::string type = int_pair.first;
                const std::vector<integrals> &ints1 = int_pair.second;
                //const std::vector<integrals> &ints2 = existing->ints.at(type);
                const std::vector<integrals> &ints2 = ordered[j]->ints.at(type);
                for (size_t i = 0; i < ints1.size(); i++) {
                    n_permute += ints1[i].permutations + ints2[i].permutations;
                }
            }

            return key;

        }else {
            string_in_map = false;
            return "blurf";
        }
    }

    // try swapping non-summed labels
    for (size_t id1 = 0; id1 < labels[iter].size(); id1++) {
        for (size_t id2 = id1 + 1; id2 < labels[iter].size(); id2++) {
    
            std::shared_ptr<pq_string> newguy = std::make_shared<pq_string>(*in);
            swap_two_labels(newguy, labels[iter][id1], labels[iter][id2]);
            //newguy->sort();

            //compare_strings_with_swapped_summed_labels(labels, iter+1, newguy, in2, n_permute, string_in_map);
            std::string res = check_map_for_strings_with_swapped_summed_labels(labels, iter+1, newguy, string_map, ordered, n_permute, string_in_map);
            if ( string_in_map ) return res; 
        }
    }
    string_in_map = false;
    return "blurf";
}

/// compare two strings when swapping (multiple) summed labels
void compare_strings_with_swapped_summed_labels(const std::vector<std::vector<std::string> > &labels,
                                                size_t iter,
                                                const std::shared_ptr<pq_string> &in1,
                                                const std::shared_ptr<pq_string> &in2,
                                                int & n_permute, 
                                                bool & strings_same) {
 
    if ( iter == labels.size() ) {
        strings_same = compare_strings(in1, in2, n_permute);
        return;
    }

    // try swapping non-summed labels
    for (size_t id1 = 0; id1 < labels[iter].size(); id1++) {
        for (size_t id2 = id1 + 1; id2 < labels[iter].size(); id2++) {
    
            std::shared_ptr<pq_string> newguy = std::make_shared<pq_string>(*in2);
            swap_two_labels(newguy, labels[iter][id1], labels[iter][id2]);
            //newguy->sort();
            compare_strings_with_swapped_summed_labels(labels, iter+1, in1, newguy, n_permute, strings_same);

            if ( strings_same ) return;
        }
    }
}

// consolidate terms that differ may differ by permutations of summed labels
void consolidate_permutations_plus_swaps(std::vector<std::shared_ptr<pq_string> > &ordered,
                                     const std::vector<std::vector<std::string> > &labels) {

    if ( ordered.size() == 0 ) {
        return;
    }

    std::unordered_map<std::string, size_t> string_map;

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        // ok, what summed / repeated labels do we have?
        std::vector<std::vector<std::string> > found_labels;
        for (const std::vector<std::string> & label : labels) {
            std::vector<std::string> tmp;
            tmp.reserve(label.size());
            for (const auto & index : label) {
                int found = ordered[i]->index_in_anywhere(index);
                if ( found == 2 ) {
                    tmp.push_back(index);
                }
            }
            found_labels.push_back(tmp);
        }

        // check if this string is in the map
        int n_permute = 0;
        bool string_in_map = false;

        std::string res = check_map_for_strings_with_swapped_summed_labels(found_labels, 0, ordered[i], string_map, ordered, n_permute, string_in_map);

        if ( !string_in_map ) {

            // new term in map
            string_map[ordered[i]->key] = i; 

        }else {

            // update factor for existing term in map

            size_t j = string_map.at(res);
            double factor_i = ordered[i]->factor * ordered[i]->sign;
            double factor_j = ordered[j]->factor * ordered[j]->sign;

            double combined_factor = factor_j + factor_i * pow(-1.0, n_permute);

            if ( fabs(combined_factor) < 1e-12 ) {
                auto it = string_map.find(res);
                string_map.erase(it);
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                continue;
            }
            ordered[i]->skip = true;

            ordered[j]->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[j]->sign =  1;
            }else {
                ordered[j]->sign = -1;
            }
        }
    }

/*
    // old O(N^2) sort
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<std::vector<std::string> > found_labels;

        // ok, what summed / repeated labels do we have?
        for (const std::vector<std::string> & label : labels) {
            std::vector<std::string> tmp;
            tmp.reserve(label.size());
            for (const auto & index : label) {
                int found = ordered[i]->index_in_anywhere(index);
                if ( found == 2 ) {
                    tmp.push_back(index);
                }
            }
            found_labels.push_back(tmp);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = false;

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
*/

}

// consolidate terms that differ by permutations of non-summed labels
void consolidate_permutations_non_summed(
    std::vector<std::shared_ptr<pq_string> > &ordered,
    const std::vector<std::string> &labels) {

    if ( ordered.size() == 0 ) {
        return;
    }

/*
    std::unordered_map<std::string, size_t> string_map;

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
            int found = ordered[i]->index_in_anywhere(label);
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

        std::string permutation_1;
        std::string permutation_2;
        int n_permute = 0;
        bool string_in_map = false;
        std::string key = "";

        // try swapping non-summed labels
        for (size_t id1 = 0; id1 < labels.size(); id1++) {
            if ( find_idx[id1] != 1 ) continue;
            for (size_t id2 = id1 + 1; id2 < labels.size(); id2++) {
                if ( find_idx[id2] != 1 ) continue;

                std::shared_ptr<pq_string> newguy = std::make_shared<pq_string>(*ordered[i]);
                swap_two_labels(newguy, labels[id1], labels[id2]);
                //newguy->sort();

                // is new string in map?
                string_in_map = false;

                if (string_map.find(newguy->key) != string_map.end()) {

                    key = newguy->key;
                    string_in_map = true;
                    size_t j = string_map.at(key);

                    // accumulate permutations of amplitudes
                    n_permute = 0;
                    for (const auto &amp_pair : newguy->amps) {
                        char type = amp_pair.first;
                        const std::vector<amplitudes> &amps1 = amp_pair.second;
                        const std::vector<amplitudes> &amps2 = ordered[j]->amps.at(type);
                        for (size_t k = 0; k < amps1.size(); k++) {
                            n_permute += amps1[k].permutations + amps2[k].permutations;
                        }
                    }

                    // accumulate permutations of integrals
                    for (const auto &int_pair : newguy->ints) {
                        std::string type = int_pair.first;
                        const std::vector<integrals> &ints1 = int_pair.second;
                        const std::vector<integrals> &ints2 = ordered[j]->ints.at(type);
                        for (size_t k = 0; k < ints1.size(); k++) {
                            n_permute += ints1[k].permutations + ints2[k].permutations;
                        }
                    }

                    permutation_1 = labels[id1];
                    permutation_2 = labels[id2];
                    break;
                }
            }
            if ( string_in_map ) break;
        }

        // if the string is not in the map already, add it
        if ( !string_in_map ) {
            string_map[ordered[i]->key] = i; 
            continue;
        }

        // if the string is in the map, 

        size_t j = string_map.at(key);

        double factor_i = ordered[i]->factor * ordered[i]->sign;
        double factor_j = ordered[j]->factor * ordered[j]->sign;

        double combined_factor = factor_j + factor_i * pow(-1.0, n_permute);

        // if terms exactly cancel, then this is a permutation
        if ( fabs(combined_factor) < 1e-12 ) {
            ordered[j]->permutations.push_back(permutation_1);
            ordered[j]->permutations.push_back(permutation_2);
            ordered[i]->skip = true;

            // don't forget to call sort labels so the permutation operator ends up on the identifier
            ordered[j]->sort();
        }else {
            // otherwise, something has gone wrong in the previous consolidation step...
            printf("somethign has gone wrong\n");fflush(stdout);
            exit(0);
        }
    }

    return;
*/

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
            int found = ordered[i]->index_in_anywhere(label);
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
                    //newguy->sort();

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

                // don't forget to call sort labels so the permutation operator ends up on the identifier
                ordered[i]->sort();
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
                        //newguy->sort();

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
            //newguy->sort();

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
            int found = ordered[i]->index_in_anywhere(occ_label);
            if ( found == 1 ) {
                found_occ.push_back(occ_label);
            }
        }

        // ok, what non-summed virtual labels do we have? 
        for (const std::string & vir_label : vir_labels) {
            int found = ordered[i]->index_in_anywhere(vir_label);
            if ( found == 1 ) {
                found_vir.push_back(vir_label);
            }
        }

        // ok, what summed labels (occupied and virtual) do we have? 
        for (const std::string & occ_label : occ_labels) {
            int found = ordered[i]->index_in_anywhere(occ_label);
            if ( found == 2 ) {
                found_summed_occ.push_back(occ_label);
            }
        }
        for (const std::string & vir_label : vir_labels) {
            int found = ordered[i]->index_in_anywhere(vir_label);
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
void cleanup(std::vector<std::shared_ptr<pq_string> > &ordered, bool find_paired_permutations, bool is_unitary_cc) {

    // sort amplitude labels, etc. define key for each string
    for (std::shared_ptr<pq_string> & pq_str : ordered) {
        pq_str->sort();
    }

    // TODO: the operator portions are not considered in the comparisons below. not sure this matters for future use cases
    consolidate_permutations_plus_swaps(ordered, {});

    // swap up to two non-summed labels (more doesn't seem to be necessary for up to ccsdtq)
    std::vector<std::string> occ_labels { "i", "j", "k", "l", "m", "n", "I", "J", "K", "L", "M", "N" };
    std::vector<std::string> vir_labels { "a", "b", "c", "d", "e", "f", "A", "B", "C", "D", "E", "F" };

    // nuclear (second-species) labels carry the species prefix. they form their own
    // spaces so that labels are only ever exchanged within a species -- an electron
    // label is never swapped with a nuclear one.
    const std::string npfx(1, nuclear_prefix);
    std::vector<std::string> nuc_occ_labels, nuc_vir_labels;
    for (const auto & s : occ_labels) nuc_occ_labels.push_back(npfx + s);
    for (const auto & s : vir_labels) nuc_vir_labels.push_back(npfx + s);
/*

    consolidate_permutations_plus_swaps(ordered, {occ_labels});
    consolidate_permutations_plus_swaps(ordered, {vir_labels});

    consolidate_permutations_plus_swaps(ordered, {occ_labels, occ_labels});
    consolidate_permutations_plus_swaps(ordered, {vir_labels, vir_labels});
    consolidate_permutations_plus_swaps(ordered, {occ_labels, vir_labels});

    consolidate_permutations_plus_swaps(ordered, {occ_labels, occ_labels, occ_labels});
    consolidate_permutations_plus_swaps(ordered, {occ_labels, vir_labels, vir_labels});
    consolidate_permutations_plus_swaps(ordered, {occ_labels, occ_labels, vir_labels});
    consolidate_permutations_plus_swaps(ordered, {vir_labels, vir_labels, vir_labels});

    consolidate_permutations_plus_swaps(ordered, {occ_labels, occ_labels, occ_labels, occ_labels});
    consolidate_permutations_plus_swaps(ordered, {occ_labels, occ_labels, vir_labels, vir_labels});
    consolidate_permutations_plus_swaps(ordered, {occ_labels, occ_labels, occ_labels, vir_labels});
    consolidate_permutations_plus_swaps(ordered, {occ_labels, vir_labels, vir_labels, vir_labels});
    consolidate_permutations_plus_swaps(ordered, {vir_labels, vir_labels, vir_labels, vir_labels});

    consolidate_permutations_plus_swaps(ordered, {occ_labels, occ_labels, occ_labels, occ_labels, occ_labels});
    consolidate_permutations_plus_swaps(ordered, {occ_labels, occ_labels, occ_labels, vir_labels, vir_labels});
    consolidate_permutations_plus_swaps(ordered, {occ_labels, occ_labels, occ_labels, occ_labels, vir_labels});
    consolidate_permutations_plus_swaps(ordered, {occ_labels, occ_labels, vir_labels, vir_labels, vir_labels});
    consolidate_permutations_plus_swaps(ordered, {occ_labels, vir_labels, vir_labels, vir_labels, vir_labels});
    consolidate_permutations_plus_swaps(ordered, {vir_labels, vir_labels, vir_labels, vir_labels, vir_labels});
*/

    if (is_unitary_cc) {
        for (const std::shared_ptr<pq_string> & pq_str : ordered) {
            if ( pq_str->skip ) continue;
            bool keep = keep_ucc_term(pq_str);
            pq_str->skip = !keep;
        }
    }

    if ( ordered.empty() ) return;

    // probably only relevant for vacuum = fermi
    if ( ordered[0]->vacuum == "FERMI" ) {

        // look for paired permutations of non-summed labels:
        if ( find_paired_permutations ) {

            // a) PP6(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + R(ikj;acb) + R(jik;bac) + R(jki;bca) + R(kij;cab) + R(kji;cba)
            consolidate_paired_permutations_non_summed(ordered, occ_labels, vir_labels, 6);

            // b) PP3(i,a;j,b;k,c) R(ijk;abc) = R(ijk;abc) + (jik;bac) + R(kji;cba)
            consolidate_paired_permutations_non_summed(ordered, occ_labels, vir_labels, 3);
        }

        consolidate_permutations_non_summed(ordered, occ_labels);
        consolidate_permutations_non_summed(ordered, vir_labels);
        consolidate_permutations_non_summed(ordered, nuc_occ_labels);
        consolidate_permutations_non_summed(ordered, nuc_vir_labels);
    }

    // prune list so it only contains non-skipped pq_strings
    std::vector<std::shared_ptr<pq_string> > pruned;
    pruned.reserve(ordered.size());
    for (std::shared_ptr<pq_string> & pq_str : ordered) {
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
    ordered = pruned;
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
   
    //static std::vector<std::string> occ_out {"i", "j", "k", "l", "m", "n", "I", "J", "K", "L", "M", "N", 
    //                                         "i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9"};
    static std::vector<std::string> occ_out{"o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9",
                                    "o10", "o11", "o12", "o13", "o14", "o15", "o16", "o17", "o18", "o19",
                                    "o20", "o21", "o22", "o23", "o24", "o25", "o26", "o27", "o28", "o29"};
                                             
    // nuclear occupied summation labels for a nuclear (proton) one-body fold
    static std::vector<std::string> nuc_occ_out;
    if ( nuc_occ_out.empty() )
        for (const auto & s : occ_out) nuc_occ_out.push_back(std::string(1, nuclear_prefix) + s);

    for (size_t i = 0; i < in->ints["occ_repulsion"].size(); i++) {

        // a nuclear fold sums over a nuclear occupied orbital, an electron fold over
        // an electron occupied orbital
        const std::vector<std::string> & out_list =
            is_nuclear(in->ints["occ_repulsion"][i].labels[0]) ? nuc_occ_out : occ_out;

        // pick summation label not included in string already
        std::string idx;

        int do_skip = -999;

        for (size_t k = 0; k < out_list.size(); k++) {
            if ( in->index_in_anywhere(out_list[k]) == 0 ) {
                idx = out_list[k];
                do_skip = k;
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
        ints.op_portions = occ_repulsion[i].op_portions;
        
        ints.sort();
        
        in->ints["eri"].push_back(ints);
    }
    in->ints["occ_repulsion"].clear();
    in->ints.erase("occ_repulsion");

}

void sort_amplitudes_topologically(std::vector<amplitudes> &amps_vec, std::shared_ptr<pq_string> &track) {

    // 1. count how many times every single label appears in this string
    std::unordered_map<std::string, int> label_frequencies;
    
    for (const auto& type_pair : track->ints) {
        for (const auto& integral : type_pair.second) {
            for (const auto& label : integral.labels) label_frequencies[label]++;
        }
    }
    for (const auto& type_pair : track->amps) {
        for (const auto& amp : type_pair.second) {
            for (const auto& label : amp.labels) label_frequencies[label]++;
        }
    }
    for (const auto& delta : track->deltas) {
        for (const auto& label : delta.labels) label_frequencies[label]++;
    }

    std::sort(amps_vec.begin(), amps_vec.end(), [&](const amplitudes &a, const amplitudes &b) {
        
        // 1. Core Structural Layer (Size / Excitation / Photons)
        if (a.labels.size() != b.labels.size()) return a.labels.size() < b.labels.size();
        if (a.n_ph != b.n_ph) return a.n_ph < b.n_ph;

        // 2. Internal Topological Layer (Connection/Contraction Weights)
        int a_weight = 0, b_weight = 0;
        for (const auto& l : a.labels) {
            if (label_frequencies.find(l) != label_frequencies.end()) a_weight += label_frequencies[l];
        }
        for (const auto& l : b.labels) {
            if (label_frequencies.find(l) != label_frequencies.end()) b_weight += label_frequencies[l];
        }
        if (a_weight != b_weight) return a_weight > b_weight; 

        // 3. External Anchor Tie-Breaker (Locks down identical dummy topologies)
        // Extract fixed external lines (e, f) and permutation anchors (m, n)
        std::string a_anchors = "";
        std::string b_anchors = "";
        
        for (const auto& l : a.labels) {
            // If it doesn't start with internal 'o' or 'v' prefixes, it's a fixed line!
            if (!is_dummy(l)) a_anchors += l;
        }
        for (const auto& l : b.labels) {
            if (!is_dummy(l)) b_anchors += l;
        }
        
        // Sort anchor profiles so that the pairing order itself doesn't cause a mismatch
        std::sort(a_anchors.begin(), a_anchors.end());
        std::sort(b_anchors.begin(), b_anchors.end());
        
        if (a_anchors != b_anchors) {
            return a_anchors < b_anchors; 
        }

        // ====================================================================
        // NEW CODE HERE: Raw Backend Label Signature Check
        // If weights and anchors match, sort their internal raw strings to 
        // create a layout-independent layout signature.
        // ====================================================================
        auto a_raw_sorted = a.labels;
        auto b_raw_sorted = b.labels;
        
        std::sort(a_raw_sorted.begin(), a_raw_sorted.end());
        std::sort(b_raw_sorted.begin(), b_raw_sorted.end());
        
        if (a_raw_sorted != b_raw_sorted) {
            return a_raw_sorted < b_raw_sorted;
        }

        // 4. Ultimate Fallback (Lexicographical internal string comparison)
        return a.labels < b.labels;
    });
}

// find and replace internally-used labels in integrals and amplitudes with conventional ones
// e.g, o1 -> i ,v1 -> a, using some rules to establish a canonical order of adding the labels
void canonicalize_labels(std::shared_ptr<pq_string> &in) {

    // sort amplitudes before canonicalizing the labels
    for (auto & type : in->amplitude_types) {
        if (in->amps.find(type) == in->amps.end()) continue;
        sort_amplitudes_topologically(in->amps[type], in);
    }

    std::unordered_map<std::string, std::string> occ_map;
    std::unordered_map<std::string, std::string> vir_map;

    std::vector<std::string> occ_pool  = {"i", "j", "k", "l", "m", "n", "I", "J", "K", "L", "M", "N"};
    std::vector<std::string> vir_pool = {"a", "b", "c", "d", "e", "f", "A", "B", "C", "D", "E", "F"};

    size_t occ_counter = 0;
    size_t vir_counter = 0;

    // Identify and skip labels already chosen or fixed
    // If 'i' or 'a' are already fixed/external indices in this string, 
    // we must burn those options from our pool so we don't clobber them.
    auto filter_pool = [&](std::vector<std::string>& pool, size_t& counter) {
        while (counter < pool.size() && found_index_anywhere(in, pool[counter])) {
            counter++; // Skip this label; it's already alive in the string
        }
    };

    // nuclear (second-species) labels canonicalize inside their own space: the same
    // conventional letters carried by the species prefix, drawn from their own pools.
    // an electron label and a nuclear label are therefore never mapped onto each other.
    const std::string npfx(1, nuclear_prefix);
    std::unordered_map<std::string, std::string> nuc_occ_map, nuc_vir_map;
    std::vector<std::string> nuc_occ_pool, nuc_vir_pool;
    for (const auto & l : occ_pool) nuc_occ_pool.push_back(npfx + l);
    for (const auto & l : vir_pool) nuc_vir_pool.push_back(npfx + l);

    size_t nuc_occ_counter = 0;
    size_t nuc_vir_counter = 0;

    // is this an internal occupied ('o') or virtual ('v') label? classify within the
    // label's own species, so that "no0" is nuclear-occupied and not a general label.
    auto raw_class = [](const std::string &label) -> char {
        const std::string base = is_nuclear(label) ? label.substr(1) : label;
        if ( base.rfind("o", 0) == 0 ) return 'o';
        if ( base.rfind("v", 0) == 0 ) return 'v';
        return '\0';
    };

    // reserve the next conventional letter of the label's own species
    auto assign_label = [&](const std::string &label, char cls) {
        const bool nuc = is_nuclear(label);
        auto &map     = cls == 'o' ? (nuc ? nuc_occ_map     : occ_map)
                                   : (nuc ? nuc_vir_map     : vir_map);
        auto &pool    = cls == 'o' ? (nuc ? nuc_occ_pool    : occ_pool)
                                   : (nuc ? nuc_vir_pool    : vir_pool);
        auto &counter = cls == 'o' ? (nuc ? nuc_occ_counter : occ_counter)
                                   : (nuc ? nuc_vir_counter : vir_counter);

        if ( map.find(label) != map.end() ) return;
        filter_pool(pool, counter);
        if ( counter < pool.size() ) map[label] = pool[counter++];
        else map[label] = (nuc ? npfx : "") + (cls == 'o' ? "o_" : "v_") + std::to_string(counter++);
    };

/*
    // Follow the exact macro-order of sorted amplitude vector
    for (auto & type : in->amplitude_types) {
        if (in->amps.find(type) == in->amps.end()) continue;
        for (auto & amp : in->amps[type]) {
            
            // Collect internal raw labels ONLY for this specific operator
            std::vector<std::string> local_raw_occ;
            std::vector<std::string> local_raw_vir;
    
            for (const auto & label : amp.labels) {
                if (label.rfind("o", 0) == 0) local_raw_occ.push_back(label);
                if (label.rfind("v", 0) == 0) local_raw_vir.push_back(label);
            }
    
            // Sort them alphabetically local to THIS operator.
            // This ensures equivalent internal slots (like k and l) are assigned deterministically!
            std::sort(local_raw_occ.begin(), local_raw_occ.end());
            std::sort(local_raw_vir.begin(), local_raw_vir.end());
    
            // Map them sequentially
            for (const auto& label : local_raw_occ) {
                if (occ_map.find(label) == occ_map.end()) {
                    filter_pool(occ_pool, occ_counter);
                    if (occ_counter < occ_pool.size()) occ_map[label] = occ_pool[occ_counter++];
                    else occ_map[label] = "o_" + std::to_string(occ_counter++);
                }
            }
            for (const auto& label : local_raw_vir) {
                if (vir_map.find(label) == vir_map.end()) {
                    filter_pool(vir_pool, vir_counter);
                    if (vir_counter < vir_pool.size()) vir_map[label] = vir_pool[vir_counter++];
                    else vir_map[label] = "v_" + std::to_string(vir_counter++);
                }
            }
        }
    }
*/

    // Macro-order map creation loop
    for (auto & type : in->amplitude_types) {
        if (in->amps.find(type) == in->amps.end()) continue;
        for (auto & amp : in->amps[type]) {
            
            std::vector<std::string> local_raw_occ;
            std::vector<std::string> local_raw_vir;
    
            for (const auto & label : amp.labels) {
                const char cls = raw_class(label);
                if (cls == 'o') local_raw_occ.push_back(label);
                if (cls == 'v') local_raw_vir.push_back(label);
            }

            // ====================================================================
            // CONNECTIVITY-BASED TIE BREAKING OVERRIDES
            // ====================================================================
            auto get_downstream_signature = [&](const std::string& target, bool is_virtual) {
                ConnectivitySignature sig{999, 999, 999};

                // Look downstream through ALL amplitudes to find where this label lands
                for (const auto& next_type : in->amplitude_types) {
                    if (in->amps.find(next_type) == in->amps.end()) continue;
                    for (const auto& downstream_amp : in->amps[next_type]) {
                        // Skip the current tensor we are currently assigning maps for
                        if (&downstream_amp == &amp) continue;

                        // Build an on-the-fly sorted symmetry pool for this downstream operator
                        std::vector<std::string_view> pool;
                        for (const auto& lbl : downstream_amp.labels) {
                            if (is_nuclear(lbl) != is_nuclear(target)) continue; // own species only
                            if (is_virtual && raw_class(lbl) == 'v') pool.push_back(lbl);
                            if (!is_virtual && raw_class(lbl) == 'o') pool.push_back(lbl);
                        }
                        std::sort(pool.begin(), pool.end());

                        // Look up the label's slot index inside the layout-independent sorted pool!
                        for (size_t p = 0; p < pool.size(); ++p) {
                            if (pool[p] == target) {
                                sig.operator_rank = downstream_amp.labels.size();
                                sig.pool_position = p;
                                sig.operator_type_weight = 1; // Prioritize Amplitudes over Integrals
                                return sig;
                            }
                        }
                    }
                }

                // If not found in Amplitudes, check Integrals downstream
                for (const auto& next_type : in->integral_types) {
                    if (in->ints.find(next_type) == in->ints.end()) continue;
                    for (const auto& downstream_int : in->ints[next_type]) {
                        std::vector<std::string_view> pool;
                        for (const auto& lbl : downstream_int.labels) {
                            if (is_nuclear(lbl) != is_nuclear(target)) continue; // own species only
                            if (is_virtual && raw_class(lbl) == 'v') pool.push_back(lbl);
                            if (!is_virtual && raw_class(lbl) == 'o') pool.push_back(lbl);
                        }
                        std::sort(pool.begin(), pool.end());

                        for (size_t p = 0; p < pool.size(); ++p) {
                            if (pool[p] == target) {
                                sig.operator_rank = downstream_int.labels.size();
                                sig.pool_position = p;
                                sig.operator_type_weight = 2; // Integrals secondary
                                return sig;
                            }
                        }
                    }
                }
                return sig;
            };

            // Instead of blind alphabetical sorting, sort based on the downstream graph pools!
            std::sort(local_raw_occ.begin(), local_raw_occ.end(), [&](const std::string& a, const std::string& b) {
                auto sig_a = get_downstream_signature(a, false);
                auto sig_b = get_downstream_signature(b, false);
                if (sig_a.operator_type_weight != sig_b.operator_type_weight)
                    return sig_a.operator_type_weight < sig_b.operator_type_weight;
                if (sig_a.operator_rank != sig_b.operator_rank)
                    return sig_a.operator_rank < sig_b.operator_rank;
                if (sig_a.pool_position != sig_b.pool_position)
                    return sig_a.pool_position < sig_b.pool_position;
                return a < b; // Alphabetical fallback if completely identical downstream environments
            });

            std::sort(local_raw_vir.begin(), local_raw_vir.end(), [&](const std::string& a, const std::string& b) {
                auto sig_a = get_downstream_signature(a, true);
                auto sig_b = get_downstream_signature(b, true);
                if (sig_a.operator_type_weight != sig_b.operator_type_weight)
                    return sig_a.operator_type_weight < sig_b.operator_type_weight;
                if (sig_a.operator_rank != sig_b.operator_rank)
                    return sig_a.operator_rank < sig_b.operator_rank;
                if (sig_a.pool_position != sig_b.pool_position)
                    return sig_a.pool_position < sig_b.pool_position;
                return a < b;
            });
            // ====================================================================

            // Map them sequentially, each label out of its own species' pool
            for (const auto& label : local_raw_occ) assign_label(label, 'o');
            for (const auto& label : local_raw_vir) assign_label(label, 'v');
        }
    }

    // Deterministic Traversal and In-Place Replacement
    auto translate_label = [&](std::string &label) {
        const char cls = raw_class(label);
        if ( cls == '\0' ) return;                 // general label: left to the pass below
        const bool nuc = is_nuclear(label);
        assign_label(label, cls);                  // reserve a letter if not seen yet
        label = cls == 'o' ? (nuc ? nuc_occ_map : occ_map)[label]
                           : (nuc ? nuc_vir_map : vir_map)[label];
    };

    // now, traverse amplitudes and integrals

    // for amplitudes, prioritize (1) amplitude order (2) amplitude type

    std::vector<size_t> target_sizes = {0, 1, 2, 3, 4, 5, 6, 7, 8}; // TODO: this covers up to quadruples
    
    //Multi-pass translation: low-order operators take precedence
    for (size_t current_size : target_sizes) {

        // process amplitudes of the current size
        for (auto & type : in->amplitude_types) {
            if (in->amps.find(type) == in->amps.end()) continue;    
            for (auto & amp : in->amps[type]) {
                // only translate if this amplitude matches the current rank pass
                if (amp.labels.size() != current_size) continue; 
                
                for (auto & label : amp.labels) {
                    translate_label(label);
                }
            }
        }
    }

    for (auto & type : in->integral_types) {
        if (in->ints.find(type) == in->ints.end()) continue;
        for (auto & integral : in->ints[type]) {
            for (auto & label : integral.labels) { 
                translate_label(label);
            }
        }
    }

    for (auto & delta : in->deltas) {
        for (auto & label : delta.labels) {
            translate_label(label);
        }
    }

/*
    // old non-deterministic way of assigning labels

    // occupied first:
    static std::vector<std::string> occ_in{"o0", "o1", "o2", "o3", "o4", "o5", "o6", "o7", "o8", "o9",
                                    "o10", "o11", "o12", "o13", "o14", "o15", "o16", "o17", "o18", "o19",
                                    "o20", "o21", "o22", "o23", "o24", "o25", "o26", "o27", "o28", "o29"};
    static std::vector<std::string> occ_out{"i", "j", "k", "l", "m", "n", "I", "J", "K", "L", "M", "N"};

    for (const std::string & in_idx : occ_in) {

        if (found_index_anywhere(in, in_idx)) {

            for (const std::string & out_idx : occ_out) {

                if (!found_index_anywhere(in, out_idx)) {

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
    static std::vector<std::string> vir_out{"a", "b", "c", "d", "e", "f", "A", "B", "C", "D", "E", "F"};

    for (const std::string & in_idx : vir_in) {

        if (found_index_anywhere(in, in_idx)) {

            for (const std::string & out_idx : vir_out) {

                if (!found_index_anywhere(in, out_idx)) {

                    replace_index_everywhere(in, in_idx, out_idx);
                    break;
                }
            }
        }
    }
*/

    // now general
    static std::vector<std::string> gen_in{"p0", "p1", "p2", "p3"};
    static std::vector<std::string> gen_out{"p", "q", "r", "s"};

    // ... and the nuclear analogues: the same letters carried by the species prefix,
    // so that a nuclear general label (np#) never lands on an electron one.
    static std::vector<std::string> nuc_gen_in, nuc_gen_out;
    if ( nuc_gen_in.empty() ) {
        const std::string np(1, nuclear_prefix);
        for (const std::string & l : gen_in)  nuc_gen_in.push_back(np + l);
        for (const std::string & l : gen_out) nuc_gen_out.push_back(np + l);
    }

    for (size_t species = 0; species < 2; species++) {

        const std::vector<std::string> &in_list  = species == 0 ? gen_in  : nuc_gen_in;
        const std::vector<std::string> &out_list = species == 0 ? gen_out : nuc_gen_out;

        for (const std::string & in_idx : in_list) {

            if (in->index_in_anywhere(in_idx) > 0 ) {

                for (const std::string & out_idx : out_list) {

                    if (in->index_in_anywhere(out_idx) == 0 ) {

                        replace_index_everywhere(in, in_idx, out_idx);
                        break;
                    }
                }
            }
        }
    }

}

/// apply delta functions to amplitude and integral labels
void gobble_deltas_slow(std::shared_ptr<pq_string> &in) {
    if (in->deltas.empty()) return;
    
    std::vector<std::string> tmp_delta1;
    std::vector<std::string> tmp_delta2;
    
    for ( delta_functions & delta : in->deltas ) {
    
        // is delta label 1 in list of summation labels?
        bool have_delta1 = false;    
        if ( in->index_in_anywhere(delta.labels[0]) == 2 ){
            have_delta1 = true;
        }
        bool have_delta2 = false;
        if ( in->index_in_anywhere(delta.labels[1]) == 2 ){
            have_delta2 = true;
        }

        // if the deltas don't contain any summation labels, we should keep them
        if (!have_delta1 && !have_delta2) {
            tmp_delta1.push_back(delta.labels[0]);
            tmp_delta2.push_back(delta.labels[1]);
            continue;
        }
   
/* 
        // this logic is obviously cleaner than that below, but 
        // for some reason the code has a harder time collecting 
        // like terms this way. requires swapping up to four 
        // labels.
        if ( have_delta1 ) { 
            replace_index_everywhere(in, delta.labels[0], delta.labels[1] );
            continue;
        }else if ( have_delta2 ) {
            replace_index_everywhere(in, delta.labels[1], delta.labels[0] );
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

        // TODO: The order of the amplitude types happen to coincide
        // with the order of descending number of amplitudes. This can be remedied by sorting the
        // types by number of amplitudes. an implementation of this is below, however this changes the
        // order of the indexing and cannot directly be compared with the test suite.
        // However, visual inspection of the output shows that the results are analytically identical.

        //      char types[] {'t', 'l', 'r', 'u', 'm', 's'};
        //      static int types_index[] {0, 1, 2, 3, 4, 5};

        //      // the amplitude type order will be set by the number of terms
        //      std::sort(types_index, types_index + 6, [&types, &in](int i1, int i2) {
        //          return in->amps[types[i1]].size() > in->amps[types[i2]].size();
        //      });

        //      do_continue = false;
        //      for (auto & type_index : types_index)
        //          char type = types[type_index];
        //          std::vector<amplitudes> & amps = in->amps[type];
        //          (... etc...)
        //

        do_continue = false;
        static std::vector<char> types = {'t', 'l', 'r', 'u', 'm', 's', 'D'};
        
        // add user-added amplitude types
        for (auto & type : in->amplitude_types) {
            auto it = std::find(types.begin(), types.end(), type);
            if (it == types.end()) {
                types.push_back(type);
            }
        }

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
        //deltas.sort();
        in->deltas.push_back(deltas);
    }

}

void gobble_deltas(std::shared_ptr<pq_string> &in) {
    if (in->deltas.empty()) return;

    // Grab flat local references to bypass the shared_ptr wrapper entirely
    auto& deltas = in->deltas;
    auto& ints = in->ints;
    auto& amps = in->amps;

    // ====================================================================
    // ALLOCATION-FREE PRE-SCAN
    // Bypasses the entire function instantly if there are no dummy deltas
    // ====================================================================
    bool has_gobbleable_delta = false;
    for (const auto & delta : deltas) {
        const std::string& l0 = delta.labels[0];
        const std::string& l1 = delta.labels[1];
        
        if (is_dummy(l0) || is_dummy(l1)) {
            has_gobbleable_delta = true;
            break;
        }
    }
    if (!has_gobbleable_delta) return;

    // Use an unordered_map to build fully collapsed, direct single-hop lookups
    std::unordered_map<std::string, std::string> substitution_map;
    std::vector<delta_functions> remaining_deltas;
    remaining_deltas.reserve(deltas.size());

    for (const auto & delta : deltas) {
        std::string l0 = delta.labels[0];
        std::string l1 = delta.labels[1];

        // Collapse delta chains completely in memory
        while (substitution_map.find(l0) != substitution_map.end()) l0 = substitution_map[l0];
        while (substitution_map.find(l1) != substitution_map.end()) l1 = substitution_map[l1];

        if (l0 == l1) continue; 

        bool l0_is_dummy = is_dummy(l0);
        bool l1_is_dummy = is_dummy(l1);

        if (!l0_is_dummy && !l1_is_dummy) {
            remaining_deltas.push_back(delta);
            continue;
        }

        if (l0_is_dummy && l1_is_dummy) {
            if (l0 > l1) std::swap(l0, l1);
            substitution_map[l0] = l1;
        } else if (l0_is_dummy) {
            substitution_map[l0] = l1;
        } else {
            substitution_map[l1] = l0;
        }
    }

    if (substitution_map.empty()) return;

    // Fully flatten the map so EVERY key points directly to its final destination.
    // This turns our deep loops into a single O(1) hash check!
    for (auto& pair : substitution_map) {
        std::string final_dest = pair.second;
        while (substitution_map.find(final_dest) != substitution_map.end()) {
            final_dest = substitution_map[final_dest];
        }
        pair.second = final_dest;
    }

    // Helper lambda to apply substitutions with ZERO redundant string assignments
    auto apply_subs = [&](std::string &label) {
        auto it = substitution_map.find(label);
        if (it != substitution_map.end()) {
            // ONLY copy memory if the label is actually different!
            if (label != it->second) {
                label = it->second;
            }
        }
    };

    // 1. Direct iteration over integrals
    for (auto & int_pair : ints) {
        for (auto & integral : int_pair.second) {
            for (auto & label : integral.labels) apply_subs(label);
        }
    }

    // 2. Direct iteration over amplitudes
    for (auto & amp_pair : amps) {
        for (auto & amp : amp_pair.second) {
            for (auto & label : amp.labels) apply_subs(label);
        }
    }

    // 3. Update remaining deltas
    for (auto & delta : remaining_deltas) {
        for (auto & label : delta.labels) apply_subs(label);
    }

    deltas = std::move(remaining_deltas);
}

void gobble_deltas_gemini_v2(std::shared_ptr<pq_string> &in) {
    if (in->deltas.empty()) return;

    // Use a flat local vector to avoid heavy heap allocation/hashing overhead
    std::vector<std::pair<std::string, std::string>> substitution_list;
    std::vector<delta_functions> remaining_deltas;
    substitution_list.reserve(in->deltas.size());

    for (const auto & delta : in->deltas) {
        std::string l0 = delta.labels[0];
        std::string l1 = delta.labels[1];

        // Resolve existing delta chains in our flat vector
        for (const auto& sub : substitution_list) {
            if (l0 == sub.first) l0 = sub.second;
            if (l1 == sub.first) l1 = sub.second;
        }

        if (l0 == l1) continue; 

        bool l0_is_dummy = (l0.rfind("o", 0) == 0 || l0.rfind("v", 0) == 0);
        bool l1_is_dummy = (l1.rfind("o", 0) == 0 || l1.rfind("v", 0) == 0);

        if (!l0_is_dummy && !l1_is_dummy) {
            remaining_deltas.push_back(delta);
            continue;
        }

        if (l0_is_dummy && l1_is_dummy) {
            if (l0 > l1) std::swap(l0, l1);
            substitution_list.push_back({l0, l1});
        } else if (l0_is_dummy) {
            substitution_list.push_back({l0, l1});
        } else {
            substitution_list.push_back({l1, l0});
        }
    }

    if (substitution_list.empty()) return;

    // Helper lambda to apply the flat substitution sequence
    auto apply_subs = [&](std::string &label) {
        for (const auto& sub : substitution_list) {
            if (label == sub.first) {
                label = sub.second;
            }
        }
    };

    // 1. Direct iteration over integrals map without type lookup keys
    for (auto & int_pair : in->ints) {
        for (auto & integral : int_pair.second) {
            for (auto & label : integral.labels) apply_subs(label);
        }
    }

    // 2. CRITICAL FLATTENING: Direct iteration over the amps map layers 
    // loops through the underlying buckets directly without touching in->amplitude_types
    for (auto & amp_pair : in->amps) {
        for (auto & amp : amp_pair.second) {
            for (auto & label : amp.labels) apply_subs(label);
        }
    }

    // 3. Update remaining deltas
    for (auto & delta : remaining_deltas) {
        for (auto & label : delta.labels) apply_subs(label);
    }

    in->deltas = std::move(remaining_deltas);
}

void gobble_deltas_gemini_v1(std::shared_ptr<pq_string> &in) {
    if (in->deltas.empty()) return;

    std::unordered_map<std::string, std::string> substitution_map;
    std::vector<delta_functions> remaining_deltas;

    for (const auto & delta : in->deltas) {
        std::string l0 = delta.labels[0];
        std::string l1 = delta.labels[1];

        // 1. Resolve existing delta chains locally in memory
        while (substitution_map.find(l0) != substitution_map.end()) l0 = substitution_map[l0];
        while (substitution_map.find(l1) != substitution_map.end()) l1 = substitution_map[l1];

        if (l0 == l1) continue; 

        // 2. FAST CHECK: No scans! Dummies start with internal 'o' or 'v' prefixes
        bool l0_is_dummy = (l0.rfind("o", 0) == 0 || l0.rfind("v", 0) == 0);
        bool l1_is_dummy = (l1.rfind("o", 0) == 0 || l1.rfind("v", 0) == 0);

        if (!l0_is_dummy && !l1_is_dummy) {
            // Both are external fixed lines; must preserve the delta
            remaining_deltas.push_back(delta);
            continue;
        }

        // 3. Map substitutions to always clear out the dummy index
        if (l0_is_dummy && l1_is_dummy) {
            if (l0 > l1) std::swap(l0, l1);
            substitution_map[l0] = l1;
        } else if (l0_is_dummy) {
            substitution_map[l0] = l1;
        } else {
            substitution_map[l1] = l0;
        }
    }

    if (substitution_map.empty()) return;

    // 4. A single flat pass over the components to swap labels
    for (auto & int_pair : in->ints) {
        for (auto & integral : int_pair.second) {
            for (auto & label : integral.labels) {
                auto it = substitution_map.find(label);
                if (it != substitution_map.end()) label = it->second;
            }
        }
    }

    for (auto & type : in->amplitude_types) {
        if (in->amps.find(type) == in->amps.end()) continue;
        for (auto & amp : in->amps[type]) {
            for (auto & label : amp.labels) {
                auto it = substitution_map.find(label);
                if (it != substitution_map.end()) label = it->second;
            }
        }
    }

    for (auto & delta : remaining_deltas) {
        for (auto & label : delta.labels) {
            auto it = substitution_map.find(label);
            if (it != substitution_map.end()) label = it->second;
        }
    }

    in->deltas = std::move(remaining_deltas);
}

// bring a new string to normal order and add to list of normal ordered strings (fermi vacuum)
void add_new_string_true_vacuum(const std::vector<std::shared_ptr<pq_string>> &in, std::vector<std::shared_ptr<pq_string> > &ordered, int print_level, bool find_paired_permutations, bool keep_operators){


    for (auto & my_string: in ) {

        if ( print_level > 0 ) {
            printf("\n");
            printf("    ");
            printf("// starting string:\n");
            my_string->print();
        }

        // rearrange strings
        std::vector< std::shared_ptr<pq_string> > tmp;
        tmp.push_back(my_string);

        bool done_rearranging = false;
        do { 
            std::vector< std::shared_ptr<pq_string> > list;
            done_rearranging = true;
            for (const std::shared_ptr<pq_string> & pq_str : tmp) {
                bool am_i_done = swap_operators_true_vacuum(pq_str, list, keep_operators);
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
    }

    // alphabetize
    alphabetize(ordered);
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

	    // a nuclear general label expands into nuclear occupied/virtual labels,
	    // so it stays within its own species' space
	    std::string sp = is_nuclear(me_nostar) ? std::string(1, nuclear_prefix) : "";
	    std::string occ_label = sp + "o" + std::to_string(occ_label_count+1);
	    std::string vir_label = sp + "v" + std::to_string(vir_label_count+1);

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
void add_new_string_fermi_vacuum(const std::vector<std::shared_ptr<pq_string>> &in, std::vector<std::shared_ptr<pq_string> > &ordered, int print_level, bool find_paired_permutations, bool keep_operators) {
        
    std::vector< std::shared_ptr<pq_string> > new_strings[in.size()];
    #pragma omp parallel for schedule(dynamic) default(none) shared(in, new_strings, keep_operators) firstprivate(print_level)
    for (size_t k = 0; k < in.size(); k++) {
        const std::shared_ptr<pq_string>& mystring = in[k];

        // check if this string can be fully contracted ...
        int nc = 0;
        int na = 0;
        for (size_t i = 0; i < mystring->is_dagger_fermi.size(); i++) {
            if ( mystring->is_dagger_fermi[i] ) nc++;
            else na++;
        }
        if ( nc != na ) {
            continue;
        }

        // bosons, too
        nc = 0;
        na = 0;
        for (auto && bdag : mystring->is_boson_dagger) {
            if ( bdag ) nc++;
            else na++;
        }
        if ( nc != na ) {
            continue;
        }

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
                bool am_i_done = swap_operators_fermi_vacuum(pq_str, list, keep_operators);
                if ( !am_i_done ) done_rearranging = false;
            }
            tmp.clear();
            for (std::shared_ptr<pq_string> & pq_str : list) {
                if ( !pq_str->skip ) {
                    tmp.push_back(pq_str);
                }
            }
        }while(!done_rearranging);

        new_strings[k] = tmp;
        tmp.clear();
    }

    for (const auto& new_string : new_strings) {
        if ( new_string.empty() ) continue;
        for (const std::shared_ptr<pq_string> & pq_str : new_string) {
            ordered.push_back(pq_str);
        }
    }
}

} // End namespaces
