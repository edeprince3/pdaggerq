//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_helper.cc
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

#include "data.h"
#include "pq.h"
#include "pq_utils.h"

namespace pdaggerq {

/// concatinate a list of operators (a list of strings) into a single list
std::vector<std::string> concatinate_operators(std::vector<std::vector<std::string>> ops) {

    std::vector<std::string> ret;
    size_t size = 0;
    for (size_t i = 0; i < ops.size(); i++) {
        size += ops[i].size();
    }
    ret.reserve(size);
    for (size_t i = 0; i < ops.size(); i++) {
        ret.insert(ret.end(), ops[i].begin(), ops[i].end());
    }
    return ret;

}

/// remove "*" from std::string
void removeStar(std::string &x) {

  auto it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == '*');});
  x.erase(it, std::end(x));
}

/// remove "(" and ")" from std::string
void removeParentheses(std::string &x) {

  auto it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == '(');});
  x.erase(it, std::end(x));

  it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == ')');});
  x.erase(it, std::end(x));

}

// is a label classified as occupied?
bool is_occ(std::string idx) {
    if ( idx == "I" || idx == "i") {
        return true;
    }else if ( idx == "J" || idx == "j") {
        return true;
    }else if ( idx == "K" || idx == "k") {
        return true;
    }else if ( idx == "L" || idx == "l") {
        return true;
    }else if ( idx == "M" || idx == "m") {
        return true;
    }else if ( idx == "N" || idx == "n") {
        return true;
    }else if ( idx == "N" || idx == "o") {
        return true;
    }else if ( idx.at(0) == 'O' || idx.at(0) == 'o') {
        return true;
    }else if ( idx.at(0) == 'I' || idx.at(0) == 'i') {
        return true;
    }
    return false;
}

// is a label classified as virtual?
bool is_vir(std::string idx) {
    if ( idx == "A" || idx == "a") {
        return true;
    }else if ( idx == "B" || idx == "b") {
        return true;
    }else if ( idx == "C" || idx == "c") {
        return true;
    }else if ( idx == "D" || idx == "d") {
        return true;
    }else if ( idx == "E" || idx == "e") {
        return true;
    }else if ( idx == "F" || idx == "f") {
        return true;
    }else if ( idx == "F" || idx == "g") {
        return true;
    }else if ( idx.at(0) == 'V' || idx.at(0) == 'v') {
        return true;
    }else if ( idx.at(0) == 'A' || idx.at(0) == 'a') {
        return true;
    }
    return false;
}

// how many times does an index appear deltas?
int index_in_deltas(std::string idx, std::vector<delta_functions> deltas) {

    int n = 0;
    for (size_t i = 0; i < deltas.size(); i++) {
        if ( deltas[i].labels[0] == idx ) {
            n++;
        }
        if ( deltas[i].labels[1] == idx ) {
            n++;
        }
    }
    return n;
}

// how many times does an index appear integrals?
int index_in_integrals(std::string idx, std::vector<integrals> ints) {

    int n = 0;
    for (size_t i = 0; i < ints.size(); i++) {
        for (size_t j = 0; j < ints[i].labels.size(); j++) {
            if ( ints[i].labels[j] == idx ) {
                n++;
            }

        }
    }
    return n;

}

// how many times does an index appear in amplitudes?
int index_in_amplitudes(std::string idx, std::vector<amplitudes> amps) {

    int n = 0;
    for (size_t i = 0; i < amps.size(); i++) {
        for (size_t j = 0; j < amps[i].labels.size(); j++) {
            if ( amps[i].labels[j] == idx ) {
                n++;
            }

        }
    }
    return n;

}

// how many times does an index appear amplitudes, deltas, and integrals?
int index_in_anywhere(std::shared_ptr<StringData> data, std::string idx) {

    int n = 0;

    n += index_in_deltas(idx, data->deltas);
    for (size_t i = 0; i < data->integral_types.size(); i++) {
        std::string type = data->integral_types[i];
        n += index_in_integrals(idx, data->ints[type]);
    }
    for (size_t i = 0; i < data->amplitude_types.size(); i++) {
        char type = data->amplitude_types[i];
        n += index_in_amplitudes(idx, data->amps[type]);
    }

    return n;
}

/// replace one label with another (in a given set of deltas)
void replace_index_in_deltas(std::string old_idx, std::string new_idx, std::vector<delta_functions> &deltas) {

    for (size_t i = 0; i < deltas.size(); i++) {
        if ( deltas[i].labels[0] == old_idx ) {
            deltas[i].labels[0] = new_idx;
        }
    }
    for (size_t i = 0; i < deltas.size(); i++) {
        if ( deltas[i].labels[1] == old_idx ) {
            deltas[i].labels[1] = new_idx;
        }
    }
}

/// replace one label with another (in a given set of amplitudes)
void replace_index_in_amplitudes(std::string old_idx, std::string new_idx, std::vector<amplitudes> &amps) {

    for (size_t i = 0; i < amps.size(); i++) {
        for (size_t j = 0; j < amps[i].labels.size(); j++) {
            if ( amps[i].labels[j] == old_idx ) {
                amps[i].labels[j] = new_idx;
            }
        }
    }
}

/// replace one label with another (in a given set of integrals)
void replace_index_in_integrals(std::string old_idx, std::string new_idx, std::vector<integrals> &ints) {

    for (size_t i = 0; i < ints.size(); i++) {
        for (size_t j = 0; j < ints[i].labels.size(); j++) {
            if ( ints[i].labels[j] == old_idx ) {
                ints[i].labels[j] = new_idx;
            }
        }
    }
}

// swap two labels
void swap_two_labels(std::shared_ptr<StringData> data, std::string label1, std::string label2) {

    replace_index_everywhere(data, label1, "x");
    replace_index_everywhere(data, label2, label1);
    replace_index_everywhere(data, "x", label2);

}

// replace one label with another (in integrals and amplitudes)
void replace_index_everywhere(std::shared_ptr<StringData> data, std::string old_idx, std::string new_idx) {

    //replace_index_in_deltas(old_idx,new_idx);
    for (size_t i = 0; i < data->integral_types.size(); i++) {
        std::string type = data->integral_types[i];
        replace_index_in_integrals(old_idx, new_idx, data->ints[type]);
    }
    for (size_t i = 0; i < data->amplitude_types.size(); i++) {
        char type = data->amplitude_types[i];
        replace_index_in_amplitudes(old_idx, new_idx, data->amps[type]);
    }
    data->sort_labels();

}

/// compare two lists of integrals
bool compare_integrals( std::vector<integrals> ints1,
                        std::vector<integrals> ints2,
                        int & n_permute ) {

    if ( ints1.size() != ints2.size() ) return false;

    size_t nsame_ints = 0;
    for (size_t i = 0; i < ints1.size(); i++) {
        for (size_t j = 0; j < ints2.size(); j++) {

            if ( ints1[i] == ints2[j] ) {

                n_permute += ints1[i].permutations + ints2[j].permutations;

                nsame_ints++;
                break;
            }

        }
    }

    if ( nsame_ints != ints1.size() ) return false;

    return true;
}

/// compare two lists of amplitudes
bool compare_amplitudes( std::vector<amplitudes> amps1,
                         std::vector<amplitudes> amps2,
                         int & n_permute ) {

    if ( amps1.size() != amps2.size() ) return false;
   
    size_t nsame_amps = 0;
    for (size_t i = 0; i < amps1.size(); i++) {
        for (size_t j = 0; j < amps2.size(); j++) {

            if ( amps1[i] == amps2[j] ) {

                n_permute += amps1[i].permutations + amps2[j].permutations;

                nsame_amps++;
                break;
            }
        }
    }

    if ( nsame_amps != amps1.size() ) return false;

    return true;
}

// compare two strings
bool compare_strings(std::shared_ptr<pq> ordered_1, std::shared_ptr<pq> ordered_2, int & n_permute) {

    // don't forget w0
    if ( ordered_1->data->has_w0 != ordered_2->data->has_w0 ) {
        return false;
    }

    // are strings same?
    if ( ordered_1->data->symbol.size() != ordered_2->data->symbol.size() ) return false;
    int nsame_s = 0;
    for (size_t k = 0; k < ordered_1->data->symbol.size(); k++) {
        if ( ordered_1->data->symbol[k] == ordered_2->data->symbol[k] ) {
            nsame_s++;
        }
    }
    if ( nsame_s != ordered_1->data->symbol.size() ) return false;

    // same delta functions (recall these aren't sorted in any way)
    int nsame_d = 0;
    for (size_t k = 0; k < ordered_1->data->deltas.size(); k++) {
        for (size_t l = 0; l < ordered_2->data->deltas.size(); l++) {
            if ( ordered_1->data->deltas[k].labels[0] == ordered_2->data->deltas[l].labels[0]
              && ordered_1->data->deltas[k].labels[1] == ordered_2->data->deltas[l].labels[1] ) {
                nsame_d++;
                //break;
            }else if ( ordered_1->data->deltas[k].labels[0] == ordered_2->data->deltas[l].labels[1]
                    && ordered_1->data->deltas[k].labels[1] == ordered_2->data->deltas[l].labels[0] ) {
                nsame_d++;
                //break;
            }
        }
    }
    if ( nsame_d != ordered_1->data->deltas.size() ) return false;

    // amplitude comparisons, with permutations
    n_permute = 0;

    bool same_string = false;
    for (size_t i = 0; i < ordered_1->data->amplitude_types.size(); i++) {
        char type = ordered_1->data->amplitude_types[i];
        same_string = compare_amplitudes( ordered_1->data->amps[type], ordered_2->data->amps[type], n_permute);
        if ( !same_string ) return false;
    }

    // integral comparisons, with permutations
    for (size_t i = 0; i < ordered_1->data->integral_types.size(); i++) {
        std::string type = ordered_1->data->integral_types[i];
        same_string = compare_integrals( ordered_1->data->ints[type], ordered_2->data->ints[type], n_permute);
        if ( !same_string ) return false;
    }

    // permutations should be the same, too wtf
    // also need to check if the permutations are the same...
    // otherwise, we shouldn't be combining these terms
    if ( ordered_1->data->permutations.size() != ordered_2->data->permutations.size() ) {
        return false;
    }

    int nsame_permutations = 0;
    // remember, permutations come in pairs
    size_t n = ordered_1->data->permutations.size() / 2;
    int count = 0;
    for (int i = 0; i < n; i++) {

        if ( ordered_1->data->permutations[count] == ordered_2->data->permutations[count] ) {
            nsame_permutations++;
        }else if (  ordered_1->data->permutations[count]   == ordered_2->data->permutations[count+1] ) {
            nsame_permutations++;
        }else if (  ordered_1->data->permutations[count+1] == ordered_2->data->permutations[count]   ) {
            nsame_permutations++;
        }else if (  ordered_1->data->permutations[count+1] == ordered_2->data->permutations[count+1] ) {
            nsame_permutations++;
        }
        count += 2;

    }
    if ( nsame_permutations != n ) {
        return false;
    }

    return true;
}

// consolidate terms that differ by permutations
void consolidate_permutations(std::vector<std::shared_ptr<pq> > &ordered) {

    // consolidate terms that differ by permutations
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->data->skip ) continue;

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->data->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->data->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->data->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->data->skip = true;
                ordered[j]->data->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->data->sign =  1;
            }else {
                ordered[i]->data->sign = -1;
            }
            ordered[j]->data->skip = true;
        }
    }
}

// consolidate terms that differ by summed labels plus permutations
void consolidate_permutations_plus_swap(std::vector<std::shared_ptr<pq> > &ordered,
                                        std::vector<std::string> labels) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->data->skip ) continue;

        std::vector<int> find_idx;

        // ok, what labels do we have?
        for (size_t j = 0; j < labels.size(); j++) {
            int found = index_in_anywhere(ordered[i]->data, labels[j]);
            find_idx.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->data->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels
            for (size_t id1 = 0; id1 < labels.size(); id1++) {
                if ( find_idx[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels.size(); id2++) {
                    if ( find_idx[id2] != 2 ) continue;

                    std::shared_ptr<pq> newguy (new pq(ordered[i]->data->vacuum));
                    newguy->data->copy((void*)(ordered[i].get()));
                    swap_two_labels(newguy->data,labels[id1],labels[id2]);
                    newguy->data->sort_labels();

                    strings_same = compare_strings(ordered[j],newguy,n_permute);

                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->data->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->data->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->data->skip = true;
                ordered[j]->data->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->data->sign =  1;
            }else {
                ordered[i]->data->sign = -1;
            }
            ordered[j]->data->skip = true;

        }
    }
}

// consolidate terms that differ by two summed labels plus permutations
void consolidate_permutations_plus_two_swaps(
    std::vector<std::shared_ptr<pq> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->data->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = index_in_anywhere(ordered[i]->data, labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = index_in_anywhere(ordered[i]->data, labels_2[j]);
            find_2.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->data->skip ) continue;

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

                            std::shared_ptr<pq> newguy (new pq(ordered[i]->data->vacuum));
                            newguy->data->copy((void*)(ordered[i].get()));
                            swap_two_labels(newguy->data,labels_1[id1],labels_1[id2]);
                            swap_two_labels(newguy->data,labels_2[id3],labels_2[id4]);
                            newguy->data->sort_labels();

                            strings_same = compare_strings(ordered[j],newguy,n_permute);

                            if ( strings_same ) break;
                        }
                        if ( strings_same ) break;
                    }
                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

           double factor_i = ordered[i]->data->factor * ordered[i]->data->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->data->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->data->skip = true;
                ordered[j]->data->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->data->sign =  1;
            }else {
                ordered[i]->data->sign = -1;
            }
            ordered[j]->data->skip = true;

        }
    }
}

// consolidate terms that differ by permutations of non-summed labels
void consolidate_permutations_non_summed(
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

                    std::shared_ptr<pq> newguy (new pq(ordered[i]->data->vacuum));
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

/// alphabetize operators to simplify string comparisons (for true vacuum only)
void alphabetize(std::vector<std::shared_ptr<pq> > &ordered) {

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

// compare strings and remove terms that cancel

void cleanup(std::vector<std::shared_ptr<pq> > &ordered) {


    for (size_t i = 0; i < ordered.size(); i++) {

        // order amplitudes such that they're ordered t1, t2, t3, etc.
        reorder_t_amplitudes(ordered[i]);

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
        if ( ordered[i]->data->vacuum == "FERMI" ) {
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
    if ( ordered.size() == 0 ) return;
    if ( ordered[0]->data->vacuum != "FERMI" ) return;

    consolidate_permutations_non_summed(ordered,occ_labels);
    consolidate_permutations_non_summed(ordered,vir_labels);

    // re-prune
    pruned.clear();
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->data->skip ) continue;

        // for normal order relative to fermi vacuum, i doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming
        if ( ordered[i]->data->vacuum == "FERMI" ) {
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

// reorder t amplitudes as t1, t2, t3, t4
void reorder_t_amplitudes(std::shared_ptr<pq> in) {

    size_t dim = in->data->amps['t'].size();

    if ( dim == 0 ) return;

    bool* nope = (bool*)malloc(dim * sizeof(bool));
    memset((void*)nope,'\0',dim * sizeof(bool));

    std::vector<std::vector<std::string> > tmp;

    std::vector<amplitudes> tmp_new;

    for (size_t order = 1; order < 7; order++) {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if ( nope[j] ) continue;

                if ( in->data->amps['t'][j].labels.size() == 2 * order ) {
                    tmp_new.push_back(in->data->amps['t'][j]);
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
        in->data->amps['t'][i].labels.clear();
        in->data->amps['t'][i].numerical_labels.clear();
    }
    in->data->amps['t'].clear();
    for (size_t i = 0; i < tmp_new.size(); i++) {
        in->data->amps['t'].push_back(tmp_new[i]);
    }

    free(nope);

}

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



}
