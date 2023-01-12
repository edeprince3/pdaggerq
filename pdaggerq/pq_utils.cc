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
int index_in_anywhere(std::shared_ptr<pq_string> in, std::string idx) {

    int n = 0;

    n += index_in_deltas(idx, in->deltas);
    for (size_t i = 0; i < in->integral_types.size(); i++) {
        std::string type = in->integral_types[i];
        n += index_in_integrals(idx, in->ints[type]);
    }
    for (size_t i = 0; i < in->amplitude_types.size(); i++) {
        char type = in->amplitude_types[i];
        n += index_in_amplitudes(idx, in->amps[type]);
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
void swap_two_labels(std::shared_ptr<pq_string> in, std::string label1, std::string label2) {

    replace_index_everywhere(in, label1, "xyz");
    replace_index_everywhere(in, label2, label1);
    replace_index_everywhere(in, "xyz", label2);

}

// replace one label with another (in integrals and amplitudes)
void replace_index_everywhere(std::shared_ptr<pq_string> in, std::string old_idx, std::string new_idx) {

    //replace_index_in_deltas(old_idx,new_idx);
    for (size_t i = 0; i < in->integral_types.size(); i++) {
        std::string type = in->integral_types[i];
        replace_index_in_integrals(old_idx, new_idx, in->ints[type]);
    }
    for (size_t i = 0; i < in->amplitude_types.size(); i++) {
        char type = in->amplitude_types[i];
        replace_index_in_amplitudes(old_idx, new_idx, in->amps[type]);
    }
    in->sort_labels();

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
bool compare_strings(std::shared_ptr<pq_string> ordered_1, std::shared_ptr<pq_string> ordered_2, int & n_permute) {

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
    for (size_t k = 0; k < ordered_1->deltas.size(); k++) {
        for (size_t l = 0; l < ordered_2->deltas.size(); l++) {
            if ( ordered_1->deltas[k].labels[0] == ordered_2->deltas[l].labels[0]
              && ordered_1->deltas[k].labels[1] == ordered_2->deltas[l].labels[1] ) {
                nsame_d++;
                //break;
            }else if ( ordered_1->deltas[k].labels[0] == ordered_2->deltas[l].labels[1]
                    && ordered_1->deltas[k].labels[1] == ordered_2->deltas[l].labels[0] ) {
                nsame_d++;
                //break;
            }
        }
    }
    if ( nsame_d != ordered_1->deltas.size() ) return false;

    // amplitude comparisons, with permutations
    n_permute = 0;

    bool same_string = false;
    for (size_t i = 0; i < ordered_1->amplitude_types.size(); i++) {
        char type = ordered_1->amplitude_types[i];
        same_string = compare_amplitudes( ordered_1->amps[type], ordered_2->amps[type], n_permute);
        if ( !same_string ) return false;
    }

    // integral comparisons, with permutations
    for (size_t i = 0; i < ordered_1->integral_types.size(); i++) {
        std::string type = ordered_1->integral_types[i];
        same_string = compare_integrals( ordered_1->ints[type], ordered_2->ints[type], n_permute);
        if ( !same_string ) return false;
    }

    // permutations should be the same, too wtf
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

// consolidate terms that differ by permutations
void consolidate_permutations(std::vector<std::shared_ptr<pq_string> > &ordered) {

    // consolidate terms that differ by permutations
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

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

// consolidate terms that differ by summed labels plus permutations
void consolidate_permutations_plus_swap(std::vector<std::shared_ptr<pq_string> > &ordered,
                                        std::vector<std::string> labels) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_idx;

        // ok, what labels do we have?
        for (size_t j = 0; j < labels.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels[j]);
            find_idx.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping non-summed labels
            for (size_t id1 = 0; id1 < labels.size(); id1++) {
                if ( find_idx[id1] != 2 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels.size(); id2++) {
                    if ( find_idx[id2] != 2 ) continue;

                    std::shared_ptr<pq_string> newguy (new pq_string(ordered[i]->vacuum));
                    newguy->copy((void*)(ordered[i].get()));
                    swap_two_labels(newguy,labels[id1],labels[id2]);
                    newguy->sort_labels();

                    strings_same = compare_strings(ordered[j],newguy,n_permute);

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

// consolidate terms that differ by two summed labels plus permutations
void consolidate_permutations_plus_two_swaps(
    std::vector<std::shared_ptr<pq_string> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;

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

                            std::shared_ptr<pq_string> newguy (new pq_string(ordered[i]->vacuum));
                            newguy->copy((void*)(ordered[i].get()));
                            swap_two_labels(newguy,labels_1[id1],labels_1[id2]);
                            swap_two_labels(newguy,labels_2[id3],labels_2[id4]);
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

// consolidate terms that differ by permutations of non-summed labels
void consolidate_permutations_non_summed(
    std::vector<std::shared_ptr<pq_string> > &ordered,
    std::vector<std::string> labels) {
        

    for (size_t i = 0; i < ordered.size(); i++) {
        
        if ( ordered[i]->skip ) continue;
    
        std::vector<int> find_idx;
    
        // ok, what labels do we have? 
        for (size_t j = 0; j < labels.size(); j++) {
            int found = index_in_anywhere(ordered[i], labels[j]);
            // this is buggy when existing permutation labels belong to 
            // the same space as the labels we're permuting ... so skip those for now.
            bool same_space = false;
            bool is_occ1 = is_occ(labels[j]);
            for (size_t k = 0; k < ordered[i]->permutations.size(); k++) {
                bool is_occ2 = is_occ(ordered[i]->permutations[k]);
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
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            std::string permutation_1 = "";
            std::string permutation_2 = "";

            // try swapping non-summed labels
            for (size_t id1 = 0; id1 < labels.size(); id1++) {
                if ( find_idx[id1] != 1 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels.size(); id2++) {
                    if ( find_idx[id2] != 1 ) continue;

                    std::shared_ptr<pq_string> newguy (new pq_string(ordered[i]->vacuum));
                    newguy->copy((void*)(ordered[i].get()));
                    swap_two_labels(newguy,labels[id1],labels[id2]);

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

            double factor_i = ordered[i]->factor * ordered[i]->sign;
            double factor_j = ordered[j]->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

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

/// alphabetize operators to simplify string comparisons (for true vacuum only)
void alphabetize(std::vector<std::shared_ptr<pq_string> > &ordered) {

    // alphabetize string
    for (size_t i = 0; i < ordered.size(); i++) {

        // creation
        bool not_alphabetized = false;
        do {
            not_alphabetized = false;
            int ndagger = 0;
            for (size_t j = 0; j < ordered[i]->symbol.size(); j++) {
                if ( ordered[i]->is_dagger[j] ) ndagger++;
            }
            for (int j = 0; j < ndagger-1; j++) {
                int val1 = ordered[i]->symbol[j].c_str()[0];
                int val2 = ordered[i]->symbol[j+1].c_str()[0];
                if ( val2 < val1 ) {
                    std::string dum = ordered[i]->symbol[j];
                    ordered[i]->symbol[j] = ordered[i]->symbol[j+1];
                    ordered[i]->symbol[j+1] = dum;
                    ordered[i]->sign = -ordered[i]->sign;
                    not_alphabetized = true;
                    j = ordered[i]->symbol.size() + 1;
                    not_alphabetized = true;
                }
            }
        }while(not_alphabetized);
        // annihilation
        not_alphabetized = false;
        do {
            not_alphabetized = false;
            int ndagger = 0;
            for (size_t j = 0; j < ordered[i]->symbol.size(); j++) {
                if ( ordered[i]->is_dagger[j] ) ndagger++;
            }
            for (int j = ndagger; j < (int)ordered[i]->symbol.size()-1; j++) {
                int val1 = ordered[i]->symbol[j].c_str()[0];
                int val2 = ordered[i]->symbol[j+1].c_str()[0];
                if ( val2 < val1 ) {
                    std::string dum = ordered[i]->symbol[j];
                    ordered[i]->symbol[j] = ordered[i]->symbol[j+1];
                    ordered[i]->symbol[j+1] = dum;
                    ordered[i]->sign = -ordered[i]->sign;
                    not_alphabetized = true;
                    j = ordered[i]->symbol.size() + 1;
                    not_alphabetized = true;
                }
            }
        }while(not_alphabetized);
    }

    // alphabetize deltas
    for (size_t i = 0; i < ordered.size(); i++) {
        for (size_t j = 0; j < ordered[i]->deltas.size(); j++) {
            int val1 = ordered[i]->deltas[j].labels[0].c_str()[0];
            int val2 = ordered[i]->deltas[j].labels[1].c_str()[0];
            if ( val2 < val1 ) {
                std::string dum = ordered[i]->deltas[j].labels[0];
                ordered[i]->deltas[j].labels[0] = ordered[i]->deltas[j].labels[1];
                ordered[i]->deltas[j].labels[1] = dum;
            }
        }
    }
}

// compare strings and remove terms that cancel

void cleanup(std::vector<std::shared_ptr<pq_string> > &ordered) {


    for (size_t i = 0; i < ordered.size(); i++) {

        // order amplitudes such that they're ordered t1, t2, t3, etc.
        reorder_t_amplitudes(ordered[i]);

        // sort amplitude labels
        ordered[i]->sort_labels();

    }

    // prune list so it only contains non-skipped ones
    std::vector< std::shared_ptr<pq_string> > pruned;
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        // for normal order relative to fermi vacuum, i doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming
        if ( ordered[i]->vacuum == "FERMI" ) {
            if ( ordered[i]->symbol.size() != 0 ) continue;
            if ( ordered[i]->is_boson_dagger.size() != 0 ) continue;
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
    if ( ordered[0]->vacuum != "FERMI" ) return;

    consolidate_permutations_non_summed(ordered,occ_labels);
    consolidate_permutations_non_summed(ordered,vir_labels);

    // re-prune
    pruned.clear();
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        // for normal order relative to fermi vacuum, i doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming
        if ( ordered[i]->vacuum == "FERMI" ) {
            if ( ordered[i]->symbol.size() != 0 ) continue;
            if ( ordered[i]->is_boson_dagger.size() != 0 ) continue;
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
void reorder_t_amplitudes(std::shared_ptr<pq_string> in) {

    size_t dim = in->amps['t'].size();

    if ( dim == 0 ) return;

    bool* nope = (bool*)malloc(dim * sizeof(bool));
    memset((void*)nope,'\0',dim * sizeof(bool));

    std::vector<std::vector<std::string> > tmp;

    std::vector<amplitudes> tmp_new;

    for (size_t order = 1; order < 7; order++) {
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                if ( nope[j] ) continue;

                if ( in->amps['t'][j].labels.size() == 2 * order ) {
                    tmp_new.push_back(in->amps['t'][j]);
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
        in->amps['t'][i].labels.clear();
        in->amps['t'][i].numerical_labels.clear();
    }
    in->amps['t'].clear();
    for (size_t i = 0; i < tmp_new.size(); i++) {
        in->amps['t'].push_back(tmp_new[i]);
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

// re-classify fluctuation potential terms
void reclassify_integrals(std::shared_ptr<pq_string> in) {
    
    if ( in->ints["occ_repulsion"].size() > 1 ) {
       printf("\n");
       printf("only support for one integral type object per string\n");
       printf("\n");
       exit(1);
    }
    
    if ( in->ints["occ_repulsion"].size() > 0 ) {
        
        // pick summation label not included in string already
        std::vector<std::string> occ_out{"i","j","k","l","m","n","o","t","i0","i1","i2","i3","i4","i5","i6","i7","i8","i9"};
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
        
        std::string idx1 = in->ints["occ_repulsion"][0].labels[0];
        std::string idx2 = in->ints["occ_repulsion"][0].labels[1];
        
        in->ints["occ_repulsion"].clear();
        
        integrals ints;
        
        ints.labels.clear();
        ints.numerical_labels.clear();
        
        ints.labels.push_back(idx1);
        ints.labels.push_back(idx);
        ints.labels.push_back(idx2);
        ints.labels.push_back(idx);
        
        ints.sort();
        
        if ( in->ints["eri"].size() > 0 ) {
           printf("\n");
           printf("only support for one integral type object per string\n");
           printf("\n");
           exit(1);
        }
        in->ints["eri"].clear();
        in->ints["eri"].push_back(ints);
    
    }
}

// find and replace any funny labels in integrals with conventional ones. i.e., o1 -> i ,v1 -> a
void use_conventional_labels(std::shared_ptr<pq_string> in) {

    // occupied first:
    std::vector<std::string> occ_in{"o0","o1","o2","o3","o4","o5","o6","o7","o8","o9",
                                    "o10","o11","o12","o13","o14","o15","o16","o17","o18","o19",
                                    "o20","o21","o22","o23","o24","o25","o26","o27","o28","o29"};
    std::vector<std::string> occ_out{"i","j","k","l","m","n","o","t",
                                     "i0","i1","i2","i3","i4","i5","i6","i7","i8","i9",
                                     "i10","i11","i12","i13","i14","i15","i16","i17","i18","i19"};

    for (size_t i = 0; i < occ_in.size(); i++) {

        if ( index_in_anywhere(in, occ_in[i]) > 0 ) {

            for (size_t j = 0; j < occ_out.size(); j++) {

                if ( index_in_anywhere(in, occ_out[j]) == 0 ) {

                    replace_index_everywhere(in, occ_in[i], occ_out[j]);
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

        if ( index_in_anywhere(in, vir_in[i]) > 0 ) {

            for (size_t j = 0; j < vir_out.size(); j++) {

                if ( index_in_anywhere(in, vir_out[j]) == 0 ) {

                    replace_index_everywhere(in, vir_in[i], vir_out[j]);
                    break;
                }
            }
        }
    }
}

/// apply delta functions to amplitude and integral labels
void gobble_deltas(std::shared_ptr<pq_string> in) {
    
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
        if ( index_in_anywhere(in, occ_labels[i]) == 2 ) {
            sum_labels.push_back(occ_labels[i]);
        }       
    }       
    for (size_t i = 0; i < vir_labels.size(); i++) {
        if ( index_in_anywhere(in, vir_labels[i]) == 2 ) {
            sum_labels.push_back(vir_labels[i]);
        }
    }
                                    
    for (size_t i = 0; i < in->deltas.size(); i++) {
    
        // is delta label 1 in list of summation labels?
        bool have_delta1 = false;    
        for (size_t j = 0; j < sum_labels.size(); j++) {
            if ( in->deltas[i].labels[0] == sum_labels[j] ) {
                have_delta1 = true;
                break;
            }
        }   
        // is delta label 2 in list of summation labels?
        bool have_delta2 = false;
        for (size_t j = 0; j < sum_labels.size(); j++) {
            if ( in->deltas[i].labels[1] == sum_labels[j] ) {
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
        for (size_t j = 0; j < in->integral_types.size(); j++) {
            std::string type = in->integral_types[j];
            if ( have_delta1 && index_in_integrals( in->deltas[i].labels[0], in->ints[type] ) > 0 ) {
               replace_index_in_integrals( in->deltas[i].labels[0], in->deltas[i].labels[1], in->ints[type] );
               do_continue = true;
               break;
            }else if ( have_delta2 && index_in_integrals( in->deltas[i].labels[1], in->ints[type] ) > 0 ) {
               replace_index_in_integrals( in->deltas[i].labels[1], in->deltas[i].labels[0], in->ints[type] );
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
            if ( have_delta1 && index_in_amplitudes( in->deltas[i].labels[0], in->amps[type] ) > 0 ) {
               replace_index_in_amplitudes( in->deltas[i].labels[0], in->deltas[i].labels[1], in->amps[type] );
               do_continue = true;
               break;
            }else if ( have_delta2 && index_in_amplitudes( in->deltas[i].labels[1], in->amps[type] ) > 0 ) {
               replace_index_in_amplitudes( in->deltas[i].labels[1], in->deltas[i].labels[0], in->amps[type] );
               do_continue = true;
               break;
            }
        }
        if ( do_continue ) continue;

        // at this point, it is safe to assume the delta function must remain
        tmp_delta1.push_back(in->deltas[i].labels[0]);
        tmp_delta2.push_back(in->deltas[i].labels[1]);

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

// add spin labels to a string
bool add_spins(std::shared_ptr<pq_string> in, std::vector<std::shared_ptr<pq_string> > &list) {

    if ( in->skip ) return true;

    bool all_spins_added = false;

    // amplitudes
    for (size_t i = 0; i < in->amplitude_types.size(); i++) {
        char type = in->amplitude_types[i];
        for (size_t j = 0; j < in->amps[type].size(); j++) {
            for (size_t k = 0; k < in->amps[type][j].labels.size(); k++) {
                if ( in->amps[type][j].spin_labels[k] == "" ) {

                    std::shared_ptr<pq_string> sa (new pq_string(in->vacuum));
                    std::shared_ptr<pq_string> sb (new pq_string(in->vacuum));

                    sa->copy(in.get());
                    sb->copy(in.get());

                    sa->set_spin_everywhere(in->amps[type][j].labels[k], "a");
                    sb->set_spin_everywhere(in->amps[type][j].labels[k], "b");

                    //sa->amps[type][j].spin_labels[k] = "a";
                    //sb->amps[type][j].spin_labels[k] = "b";

                    list.push_back(sa);
                    list.push_back(sb);
                    return false;

                }
            }
        }
    }

    // integrals
    for (size_t i = 0; i < in->integral_types.size(); i++) {
        std::string type = in->integral_types[i];
        for (size_t j = 0; j < in->ints[type].size(); j++) {
            for (size_t k = 0; k < in->ints[type][j].labels.size(); k++) {
                if ( in->ints[type][j].spin_labels[k] == "" ) {

                    std::shared_ptr<pq_string> sa (new pq_string(in->vacuum));
                    std::shared_ptr<pq_string> sb (new pq_string(in->vacuum));

                    sa->copy(in.get());
                    sb->copy(in.get());

                    sa->set_spin_everywhere(in->ints[type][j].labels[k], "a");
                    sb->set_spin_everywhere(in->ints[type][j].labels[k], "b");

                    //sa->ints[type][j].spin_labels[k] = "a";
                    //sb->ints[type][j].spin_labels[k] = "b";

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
void spin_blocking(std::shared_ptr<pq_string> in, std::vector<std::shared_ptr<pq_string> > &spin_blocked, std::map<std::string, std::string> spin_map) {

    // check that non-summed spin labels match those specified
    std::vector<std::string> occ_labels { "i", "j", "k", "l", "m", "n", "o" };
    std::vector<std::string> vir_labels { "a", "b", "c", "d", "e", "f", "g" };

    std::map<std::string, bool> found_labels;
    
    // ok, what non-summed labels do we have in the occupied space? 
    for (size_t j = 0; j < occ_labels.size(); j++) {
        int found = index_in_anywhere(in, occ_labels[j]);
        if ( found == 1 ) {
            found_labels[occ_labels[j]] = true;
        }else{
            found_labels[occ_labels[j]] = false;
        }
    }
    
    // ok, what non-summed labels do we have in the virtual space? 
    for (size_t j = 0; j < vir_labels.size(); j++) {
        int found = index_in_anywhere(in, vir_labels[j]);
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
    in->non_summed_spin_labels = spin_map;

    // copy this term and zero spins

    std::shared_ptr<pq_string> newguy (new pq_string(in->vacuum));
    newguy->copy(in.get());

    newguy->reset_spin_labels();

    // list of expanded sums
    std::vector< std::shared_ptr<pq_string> > tmp;
    tmp.push_back(newguy);

    for (size_t i = 0; i < tmp.size(); i++) {

        // but first expand permutations where spins don't match 
        size_t n = tmp[i]->permutations.size() / 2;

        for (size_t j = 0; j < n; j++) {

            std::string idx1 = tmp[i]->permutations[2*j];
            std::string idx2 = tmp[i]->permutations[2*j+1];

            // spin 1
            std::string spin1 = "";
            spin1 = tmp[i]->non_summed_spin_labels[idx1];

            // spin 2
            std::string spin2 = "";
            spin2 = tmp[i]->non_summed_spin_labels[idx2];

            // if spins are not the same, then the permutation needs to be expanded explicitly and allowed spins redetermined
            if ( spin1 != spin2 ) {

                // first guy is just a copy
                std::shared_ptr<pq_string> newguy1 (new pq_string(in->vacuum));
                newguy1->copy((void*)tmp[i].get());

                // second guy is a copy with permuted labels and change in sign
                std::shared_ptr<pq_string> newguy2 (new pq_string(in->vacuum));
                newguy2->copy((void*)tmp[i].get());
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

    // now, expand sums 

    bool done_adding_spins = false;
    do {
        std::vector< std::shared_ptr<pq_string> > list;
        done_adding_spins = true;
        for (size_t i = 0; i < tmp.size(); i++) {
            bool am_i_done = add_spins(tmp[i], list);
            if ( !am_i_done ) done_adding_spins = false;
        }
        if ( !done_adding_spins ) {
            tmp.clear();
            for (size_t i = 0; i < list.size(); i++) {
                if ( !list[i]->skip ) {
                    tmp.push_back(list[i]);
                }
            }
        }
    }while(!done_adding_spins);


    // kill terms that have mismatched spin 
    for (size_t i = 0; i < tmp.size(); i++) {

        if ( tmp[i]->skip ) continue;

        bool killit = false;

        // amplitudes
        // TODO: this logic only works for particle-conserving amplitudes
        for (size_t j = 0; j < in->amplitude_types.size(); j++) {
            char type = in->amplitude_types[j];
            for (size_t k = 0; k < tmp[i]->amps[type].size(); k++) {

                size_t order = tmp[i]->amps[type][k].labels.size()/2;

                int left_a = 0;
                int left_b = 0;
                int right_a = 0;
                int right_b = 0;
                for (size_t l = 0; l < order; l++) {
                    if ( tmp[i]->amps[type][k].spin_labels[l] == "a" ) {
                        left_a++;
                    }else {
                        left_b++;
                    }
                    if ( tmp[i]->amps[type][k].spin_labels[l+order] == "a" ) {
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
            tmp[i]->skip = true;
            continue;
        }

        killit = false;

        // integrals
        for (size_t j = 0; j < in->integral_types.size(); j++) {
            std::string type = in->integral_types[j];
            for (size_t k = 0; k < tmp[i]->ints[type].size(); k++) {
                size_t order = tmp[i]->ints[type][k].labels.size()/2;

                int left_a = 0;
                int left_b = 0;
                int right_a = 0;
                int right_b = 0;
                for (size_t l = 0; l < order; l++) {
                    if ( tmp[i]->ints[type][k].spin_labels[l] == "a" ) {
                        left_a++;
                    }else {
                        left_b++;
                    }
                    if ( tmp[i]->ints[type][k].spin_labels[l+order] == "a" ) {
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
            tmp[i]->skip = true;
            continue;
        }

        killit = false;

        // delta functions 
        for (size_t j = 0; j < in->deltas.size(); j++) {
            if ( tmp[i]->deltas[j].spin_labels[0] != tmp[i]->deltas[j].spin_labels[1] ) {
                killit = true;
                break;
            }
        }

        if ( killit ) {
            tmp[i]->skip = true;
            continue;
        }
    }

    
    // rearrange terms so that they have standard spin order (abba -> -abab, etc.)
    for (size_t p = 0; p < tmp.size(); p++) {

        if ( tmp[p]->skip ) continue;

        // amplitudes
        for (size_t i = 0; i < in->amplitude_types.size(); i++) {
            char type = in->amplitude_types[i];
            for (size_t j = 0; j < tmp[p]->amps[type].size(); j++) {
                size_t order = tmp[p]->amps[type][j].labels.size()/2;
                if ( order > 4 ) {
                    printf("\n");
                    printf("    error: spin tracing doesn't work for higher than quadruples yet\n");
                    printf("\n");
                    exit(1);
                }
                if ( order == 2 ) {

                    // three cases that require attention: ab;ba, ba;ab, and ba;ba

                    if (       tmp[p]->amps[type][j].spin_labels[0] == "a"
                            && tmp[p]->amps[type][j].spin_labels[1] == "b"
                            && tmp[p]->amps[type][j].spin_labels[2] == "b"
                            && tmp[p]->amps[type][j].spin_labels[3] == "a" ) {

                            std::string tmp_label = tmp[p]->amps[type][j].labels[2];
                            tmp[p]->amps[type][j].labels[2] = tmp[p]->amps[type][j].labels[3];
                            tmp[p]->amps[type][j].labels[3] = tmp_label;

                            tmp[p]->amps[type][j].spin_labels[2] = "a";
                            tmp[p]->amps[type][j].spin_labels[3] = "b";

                            tmp[p]->sign *= -1;

                    }else if ( tmp[p]->amps[type][j].spin_labels[0] == "b"
                            && tmp[p]->amps[type][j].spin_labels[1] == "a"
                            && tmp[p]->amps[type][j].spin_labels[2] == "a"
                            && tmp[p]->amps[type][j].spin_labels[3] == "b" ) {

                            std::string tmp_label = tmp[p]->amps[type][j].labels[0];
                            tmp[p]->amps[type][j].labels[0] = tmp[p]->amps[type][j].labels[1];
                            tmp[p]->amps[type][j].labels[1] = tmp_label;

                            tmp[p]->amps[type][j].spin_labels[0] = "a";
                            tmp[p]->amps[type][j].spin_labels[1] = "b";

                            tmp[p]->sign *= -1;


                    }else if ( tmp[p]->amps[type][j].spin_labels[0] == "b"
                            && tmp[p]->amps[type][j].spin_labels[1] == "a"
                            && tmp[p]->amps[type][j].spin_labels[2] == "b"
                            && tmp[p]->amps[type][j].spin_labels[3] == "a" ) {

                            std::string tmp_label = tmp[p]->amps[type][j].labels[0];
                            tmp[p]->amps[type][j].labels[0] = tmp[p]->amps[type][j].labels[1];
                            tmp[p]->amps[type][j].labels[1] = tmp_label;

                            tmp[p]->amps[type][j].spin_labels[0] = "a";
                            tmp[p]->amps[type][j].spin_labels[1] = "b";

                            tmp_label = tmp[p]->amps[type][j].labels[2];
                            tmp[p]->amps[type][j].labels[2] = tmp[p]->amps[type][j].labels[3];
                            tmp[p]->amps[type][j].labels[3] = tmp_label;

                            tmp[p]->amps[type][j].spin_labels[2] = "a";
                            tmp[p]->amps[type][j].spin_labels[3] = "b";

                    }
                }else if ( order == 3 ) {

                    // target order: aaa, aab, abb, bbb
                    int sign = 1;
                    reorder_three_spins(tmp[p]->amps[type][j], 0, 1, 2, sign);
                    reorder_three_spins(tmp[p]->amps[type][j], 3, 4, 5, sign);
                    tmp[p]->sign *= sign;

                }else if ( order == 4 ) {

                    // target order: aaaa, aaab, aabb, abbb, bbbb
                    int sign = 1;
                    reorder_four_spins(tmp[p]->amps[type][j], 0, 1, 2, 3, sign);
                    reorder_four_spins(tmp[p]->amps[type][j], 4, 5, 6, 7, sign);
                    tmp[p]->sign *= sign;

                }
            }
        }

        // integrals
        for (size_t i = 0; i < in->integral_types.size(); i++) {
            std::string type = in->integral_types[i];
            for (size_t j = 0; j < tmp[p]->ints[type].size(); j++) {

                size_t order = tmp[p]->ints[type][j].labels.size()/2;

                if ( order != 2 ) continue;

                // three cases that require attention: ab;ba, ba;ab, and ba;ba

                // integrals
                if (       tmp[p]->ints[type][j].spin_labels[0] == "a"
                        && tmp[p]->ints[type][j].spin_labels[1] == "b"
                        && tmp[p]->ints[type][j].spin_labels[2] == "b"
                        && tmp[p]->ints[type][j].spin_labels[3] == "a" ) {

                        std::string tmp_label = tmp[p]->ints[type][j].labels[2];
                        tmp[p]->ints[type][j].labels[2] = tmp[p]->ints[type][j].labels[3];
                        tmp[p]->ints[type][j].labels[3] = tmp_label;

                        tmp[p]->ints[type][j].spin_labels[2] = "a";
                        tmp[p]->ints[type][j].spin_labels[3] = "b";

                        tmp[p]->sign *= -1;

                }else if ( tmp[p]->ints[type][j].spin_labels[0] == "b"
                        && tmp[p]->ints[type][j].spin_labels[1] == "a"
                        && tmp[p]->ints[type][j].spin_labels[2] == "a"
                        && tmp[p]->ints[type][j].spin_labels[3] == "b" ) {

                        std::string tmp_label = tmp[p]->ints[type][j].labels[0];
                        tmp[p]->ints[type][j].labels[0] = tmp[p]->ints[type][j].labels[1];
                        tmp[p]->ints[type][j].labels[1] = tmp_label;

                        tmp[p]->ints[type][j].spin_labels[0] = "a";
                        tmp[p]->ints[type][j].spin_labels[1] = "b";

                        tmp[p]->sign *= -1;


                }else if ( tmp[p]->ints[type][j].spin_labels[0] == "b"
                        && tmp[p]->ints[type][j].spin_labels[1] == "a"
                        && tmp[p]->ints[type][j].spin_labels[2] == "b"
                        && tmp[p]->ints[type][j].spin_labels[3] == "a" ) {

                        std::string tmp_label = tmp[p]->ints[type][j].labels[0];
                        tmp[p]->ints[type][j].labels[0] = tmp[p]->ints[type][j].labels[1];
                        tmp[p]->ints[type][j].labels[1] = tmp_label;

                        tmp[p]->ints[type][j].spin_labels[0] = "a";
                        tmp[p]->ints[type][j].spin_labels[1] = "b";

                        tmp_label = tmp[p]->ints[type][j].labels[2];
                        tmp[p]->ints[type][j].labels[2] = tmp[p]->ints[type][j].labels[3];
                        tmp[p]->ints[type][j].labels[3] = tmp_label;

                        tmp[p]->ints[type][j].spin_labels[2] = "a";
                        tmp[p]->ints[type][j].spin_labels[3] = "b";

                }
            }
        }
    }

    // 
    for (size_t i = 0; i < tmp.size(); i++) {
        if ( tmp[i]->skip ) continue;
        spin_blocked.push_back(tmp[i]);
    }

    tmp.clear();
}

// bring a new string to normal order and add to list of normal ordered strings (fermi vacuum)
void add_new_string_true_vacuum(std::shared_ptr<pq_string> in, std::vector<std::shared_ptr<pq_string> > &ordered, int print_level){

    if ( in->factor > 0.0 ) {
        in->sign = 1;
        in->factor = fabs(in->factor);
    }else {
        in->sign = -1;
        in->factor = fabs(in->factor);
    }

    for (size_t i = 0; i < in->string.size(); i++) {
        std::string me = in->string[i];
        if ( me.find("*") != std::string::npos ) {
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
        for (size_t i = 0; i < tmp.size(); i++) {
            bool am_i_done = swap_operators_true_vacuum(tmp[i], list);
            if ( !am_i_done ) done_rearranging = false;
        }
        tmp.clear();
        for (size_t i = 0; i < list.size(); i++) {
            tmp.push_back(list[i]);
        }
    }while(!done_rearranging);

    for (size_t i = 0; i < tmp.size(); i++) {
        ordered.push_back(tmp[i]);
    }
    tmp.clear();

    // alphabetize
    alphabetize(ordered);

    // try to cancel similar terms
    cleanup(ordered);
}

// bring a new string to normal order and add to list of normal ordered strings (fermi vacuum)
void add_new_string_fermi_vacuum(std::shared_ptr<pq_string> in, std::vector<std::shared_ptr<pq_string> > &ordered, int print_level){
        
    // if normal order is defined with respect to the fermi vacuum, we must
    // check here if the input string contains any general-index operators
    // (h, g). If it does, then the string must be split to account explicitly
    // for sums over 
    
    int n_gen_idx = 1;
    int n_integral_objects = 0;
    std::string integral_type = "none";
    for (size_t i = 0; i < in->integral_types.size(); i++) {
        std::string type = in->integral_types[i];
        for (size_t j = 0; j < in->ints[type].size(); j++) {
            n_integral_objects++;
            n_gen_idx = in->ints[type][j].labels.size();
            integral_type = type;
        }
    }
    if ( n_integral_objects > 1 ) {
        printf("\n");
        printf("    error: only support for a single integral object per string\n");
        printf("\n");
        exit(1);
    }   
    
    // need number of strings to be square of number of general indices  (or one)
    for (int string_num = 0; string_num < n_gen_idx * n_gen_idx; string_num++) {

        std::shared_ptr<pq_string> mystring (new pq_string("FERMI"));
            
        // factors:
        if ( in->factor > 0.0 ) {
            mystring->sign = 1;
            mystring->factor = fabs(in->factor);
        }else {
            mystring->sign = -1;
            mystring->factor = fabs(in->factor);
        }
        
        mystring->has_w0       = in->has_w0;
    
        integrals ints;

        int my_gen_idx = 0;
        for (size_t i = 0; i < in->string.size(); i++) {
            std::string me = in->string[i];
    

            std::string me_nostar = me;
            if (me_nostar.find("*") != std::string::npos ){
                removeStar(me_nostar);
            }

            // fermi vacuum 
            if ( is_vir(me_nostar) ) {
                if (me.find("*") != std::string::npos ){
                    mystring->is_dagger.push_back(true);
                    mystring->is_dagger_fermi.push_back(true);
                }else {
                    mystring->is_dagger.push_back(false);
                    mystring->is_dagger_fermi.push_back(false);
                }
                mystring->symbol.push_back(me_nostar);
            }else if ( is_occ(me_nostar) ) {
                if (me.find("*") != std::string::npos ){
                    mystring->is_dagger.push_back(true);
                    mystring->is_dagger_fermi.push_back(false);
                }else {
                    mystring->is_dagger.push_back(false);
                    mystring->is_dagger_fermi.push_back(true);
                }
                mystring->symbol.push_back(me_nostar);
            }else {

                //two-index integrals
                // 00, 01, 10, 11
                if ( n_gen_idx == 2 ) {
                    if ( my_gen_idx == 0 ) {
                        if ( string_num == 0 || string_num == 1 ) {
                            // first index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystring->is_dagger.push_back(true);
                                mystring->is_dagger_fermi.push_back(false);
                            }else {
                                mystring->is_dagger.push_back(false);
                                mystring->is_dagger_fermi.push_back(true);
                            }
                            ints.labels.push_back("o1");
                            mystring->symbol.push_back("o1");
                        }else {
                            // first index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystring->is_dagger.push_back(true);
                                mystring->is_dagger_fermi.push_back(true);
                            }else {
                                mystring->is_dagger_fermi.push_back(false);
                                mystring->is_dagger.push_back(false);
                            }
                            ints.labels.push_back("v1");
                            mystring->symbol.push_back("v1");
                        }
                    }else {
                        if ( string_num == 0 || string_num == 2 ) {
                            // second index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystring->is_dagger.push_back(true);
                                mystring->is_dagger_fermi.push_back(false);
                            }else {
                                mystring->is_dagger.push_back(false);
                                mystring->is_dagger_fermi.push_back(true);
                            }
                            ints.labels.push_back("o2");
                            mystring->symbol.push_back("o2");
                        }else {
                            // second index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystring->is_dagger.push_back(true);
                                mystring->is_dagger_fermi.push_back(true);
                            }else {
                                mystring->is_dagger.push_back(false);
                                mystring->is_dagger_fermi.push_back(false);
                            }
                            ints.labels.push_back("v2");
                            mystring->symbol.push_back("v2");
                        }
                    }
                }

                //four-index integrals

                // managing these labels is so very confusing:
                // p*q*sr (pr|qs) -> o*t*uv (ov|tu), etc.
                // p*q*sr (pr|qs) -> w*x*yz (wz|xy), etc.

                if ( n_gen_idx == 4 ) {
                    if ( my_gen_idx == 0 ) {
                        //    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
                        // 0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111
                        if ( string_num == 0 ||
                             string_num == 1 ||
                             string_num == 2 ||
                             string_num == 3 ||
                             string_num == 4 ||
                             string_num == 5 ||
                             string_num == 6 ||
                             string_num == 7 ) {

                            // first index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystring->is_dagger.push_back(true);
                                mystring->is_dagger_fermi.push_back(false);
                            }else {
                                mystring->is_dagger.push_back(false);
                                mystring->is_dagger_fermi.push_back(true);
                            }
                            ints.labels.push_back("o1");
                            mystring->symbol.push_back("o1");
                        }else {
                            // first index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystring->is_dagger.push_back(true);
                                mystring->is_dagger_fermi.push_back(true);
                            }else {
                                mystring->is_dagger.push_back(false);
                                mystring->is_dagger_fermi.push_back(false);
                            }
                            ints.labels.push_back("v1");
                            mystring->symbol.push_back("v1");
                        }
                    }else if ( my_gen_idx == 1 ) {
                        //    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
                        // 0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111
                        if ( string_num ==  0 ||
                             string_num ==  1 ||
                             string_num ==  2 ||
                             string_num ==  3 ||
                             string_num ==  8 ||
                             string_num ==  9 ||
                             string_num == 10 ||
                             string_num == 11 ) {
                            // second index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystring->is_dagger.push_back(true);
                                mystring->is_dagger_fermi.push_back(false);
                            }else {
                                mystring->is_dagger.push_back(false);
                                mystring->is_dagger_fermi.push_back(true);
                            }
                            ints.labels.push_back("o2");
                            mystring->symbol.push_back("o2");
                        }else {
                            // second index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystring->is_dagger.push_back(true);
                                mystring->is_dagger_fermi.push_back(true);
                            }else {
                                mystring->is_dagger.push_back(false);
                                mystring->is_dagger_fermi.push_back(false);
                            }
                            ints.labels.push_back("v2");
                            mystring->symbol.push_back("v2");
                        }
                    }else if ( my_gen_idx == 2 ) {
                        //    0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
                        // 0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111
                        if ( string_num ==  0 ||
                             string_num ==  1 ||
                             string_num ==  4 ||
                             string_num ==  5 ||
                             string_num ==  8 ||
                             string_num ==  9 ||
                             string_num == 12 ||
                             string_num == 13 ) {
                            // third index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystring->is_dagger.push_back(true);
                                mystring->is_dagger_fermi.push_back(false);
                            }else {
                                mystring->is_dagger.push_back(false);
                                mystring->is_dagger_fermi.push_back(true);
                            }
                            ints.labels.push_back("o3");
                            mystring->symbol.push_back("o3");
                        }else {
                            // third index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystring->is_dagger.push_back(true);
                                mystring->is_dagger_fermi.push_back(true);
                            }else {
                                mystring->is_dagger.push_back(false);
                                mystring->is_dagger_fermi.push_back(false);
                            }
                            ints.labels.push_back("v3");
                            mystring->symbol.push_back("v3");
                        }
                    }else {
                        if ( string_num ==  0 ||
                             string_num ==  2 ||
                             string_num ==  4 ||
                             string_num ==  6 ||
                             string_num ==  8 ||
                             string_num == 10 ||
                             string_num == 12 ||
                             string_num == 14 ) {
                            // fourth index occ
                            if ( me.find("*") != std::string::npos ) {
                                mystring->is_dagger.push_back(true);
                                mystring->is_dagger_fermi.push_back(false);
                            }else {
                                mystring->is_dagger.push_back(false);
                                mystring->is_dagger_fermi.push_back(true);
                            }
                            ints.labels.push_back("o4");
                            mystring->symbol.push_back("o4");
                        }else {
                            // fourth index vir
                            if ( me.find("*") != std::string::npos ) {
                                mystring->is_dagger.push_back(true);
                                mystring->is_dagger_fermi.push_back(true);
                            }else {
                                mystring->is_dagger.push_back(false);
                                mystring->is_dagger_fermi.push_back(false);
                            }
                            ints.labels.push_back("v4");
                            mystring->symbol.push_back("v4");
                        }
                    }
                }

                my_gen_idx++;

            }

        }

        for (size_t i = 0; i < in->amplitude_types.size(); i++) {
            char type = in->amplitude_types[i];
            mystring->amps[type].clear();
            for (size_t j = 0; j < in->amps[type].size(); j++) {
                mystring->amps[type].push_back( in->amps[type][j] );
            }
        }

        // now, string is complete, but ints need to be pushed onto the 
        // string, and the labels in four-index integrals need to be 
        // reordered p*q*sr(pq|sr) -> (pr|qs)
        if ( integral_type == "eri" || integral_type == "two_body" ) {

            // dirac notation: g(pqrs) p*q*sr
            std::vector<std::string> tmp;
            tmp.push_back(ints.labels[0]);
            tmp.push_back(ints.labels[1]);
            tmp.push_back(ints.labels[3]);
            tmp.push_back(ints.labels[2]);

            ints.labels.clear();
            ints.labels.push_back(tmp[0]);
            ints.labels.push_back(tmp[1]);
            ints.labels.push_back(tmp[2]);
            ints.labels.push_back(tmp[3]);

            mystring->ints[integral_type].push_back(ints);

        }else if ( integral_type != "none" ) {

            mystring->ints[integral_type].push_back(ints);

        }

        for (size_t i = 0; i < in->is_boson_dagger.size(); i++) {
            mystring->is_boson_dagger.push_back(in->is_boson_dagger[i]);
        }

        if ( print_level > 0 ) {
            printf("\n");
            printf("    ");
            printf("// starting string:\n");
            mystring->print();
        }

        // rearrange strings

        std::vector< std::shared_ptr<pq_string> > tmp;
        tmp.push_back(mystring);

        bool done_rearranging = false;
        do {
            std::vector< std::shared_ptr<pq_string> > list;
            done_rearranging = true;
            for (size_t i = 0; i < tmp.size(); i++) {
                bool am_i_done = swap_operators_fermi_vacuum(tmp[i], list);
                if ( !am_i_done ) done_rearranging = false;
            }
            tmp.clear();
            for (size_t i = 0; i < list.size(); i++) {
                if ( !list[i]->skip ) {
                    tmp.push_back(list[i]);
                }
            }
        }while(!done_rearranging);

        //ordered.clear();
        for (size_t i = 0; i < tmp.size(); i++) {
            ordered.push_back(tmp[i]);
        }
        //printf("current list size: %zu\n",ordered.size());
        tmp.clear();

    }
}

} // End namespaces

