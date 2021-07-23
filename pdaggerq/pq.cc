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
#include <math.h>
#include<sstream>

#include "pq.h"


// work-around for finite precision of std::to_string
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 14)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}

namespace pdaggerq {

pq::pq(std::string vacuum_type) {

  vacuum = vacuum_type;
  skip = false;
  data = (std::shared_ptr<StringData>)(new StringData());

}

pq::~pq() {
}

bool pq::is_occ(std::string idx) {
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

bool pq::is_vir(std::string idx) {
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

void pq::check_occ_vir() {

   // OCC: I,J,K,L,M,N
   // VIR: A,B,C,D,E,F
   // GEN: P,Q,R,S,T,U,V,W

   for (size_t i = 0; i < delta1.size(); i++ ) {
       bool first_is_occ = false;
       if ( is_occ(delta1[i]) ){
           first_is_occ = true;
       }else if ( is_vir(delta1[i]) ) {
           first_is_occ = false;
       }else {
           continue;
       }

       bool second_is_occ = false;
       if ( is_occ(delta2[i]) ){
           second_is_occ = true;
       }else if ( is_vir(delta2[i]) ) {
           second_is_occ = false;
       }else {
           continue;
       }

       if ( first_is_occ != second_is_occ ) {
           skip = true;
       }

   }

}

void pq::print_amplitudes(std::string label, std::vector<std::vector<std::string> > amplitudes) {

    if ( amplitudes.size() == 0 ) {
        return;
    }

    for (size_t i = 0; i < amplitudes.size(); i++) {

        if ( amplitudes[i].size() > 0 ) {

            size_t order = amplitudes[i].size() / 2;
            printf("%s",label.c_str());
            printf("%zu",order);
            printf("(");
            for (size_t j = 0; j < 2*order-1; j++) {
                printf("%s",amplitudes[i][j].c_str());
                printf(",");
            }
            printf("%s",amplitudes[i][2*order-1].c_str());
            printf(")");
            printf(" ");

        }
    }

}

void pq::print() {

    if ( skip ) return;

    if ( vacuum == "FERMI" && symbol.size() > 0 ) {
        // check if stings should be zero or not
        bool is_dagger_right = is_dagger_fermi[symbol.size()-1];
        bool is_dagger_left  = is_dagger_fermi[0];
        if ( !is_dagger_right || is_dagger_left ) {
            //return;
        }
    }

    //for (size_t i = 0; i < symbol.size(); i++) {
    //    printf("%5zu\n",is_dagger_fermi[i]);
    //}

    printf("    ");
    printf("//     ");
    printf("%c", sign > 0 ? '+' : '-');
    printf(" ");
    printf("%20.14lf", fabs(data->factor));
    //printf("%7.5lf", fabs(data->factor));
    printf(" ");

    if ( data->permutations.size() > 0 ) {
        // should have an even number of symbols...how many pairs?
        size_t n = data->permutations.size() / 2;
        int count = 0;
        for (int i = 0; i < n; i++) {
            printf("P(");
            printf("%s",data->permutations[count++].c_str());
            printf(",");
            printf("%s",data->permutations[count++].c_str());
            printf(")");
            printf(" ");
        }
    }

    for (size_t i = 0; i < symbol.size(); i++) {
        printf("%s",symbol[i].c_str());
        if ( is_dagger[i] ) {
            printf("%c",'*');
        }
        printf(" ");
    }
    for (size_t i = 0; i < delta1.size(); i++) {
        printf("d(%s,%s)",delta1[i].c_str(),delta2[i].c_str());
        printf(" ");
    }

    // two-electron integrals
    if ( data->tensor.size() == 4 ) {

        if ( data->tensor_type == "TWO_BODY") {
            printf("g(");
            printf("%s",data->tensor[0].c_str());
            printf(",");
            printf("%s",data->tensor[1].c_str());
            printf(",");
            printf("%s",data->tensor[2].c_str());
            printf(",");
            printf("%s",data->tensor[3].c_str());
            printf(")");
            printf(" ");
        }else {
            // dirac
            printf("<");
            printf("%s",data->tensor[0].c_str());
            printf(",");
            printf("%s",data->tensor[1].c_str());
            printf("||");
            printf("%s",data->tensor[2].c_str());
            printf(",");
            printf("%s",data->tensor[3].c_str());
            printf(">");
            printf(" ");
        }
    }

    // one-electron integrals
    if ( data->tensor.size() == 2 ) {
        if ( data->tensor_type == "CORE") {
            printf("h(");
        }else if ( data->tensor_type == "FOCK") {
            printf("f(");
        }else if ( data->tensor_type == "D+") {
            printf("d+(");
        }else if ( data->tensor_type == "D-") {
            printf("d-(");
        }
        printf("%s",data->tensor[0].c_str());
        printf(",");
        printf("%s",data->tensor[1].c_str());
        printf(")");
        printf(" ");
    }

    // left-hand amplitudes
    print_amplitudes("l",data->left_amplitudes);
    if ( data->has_l0 ) {
        printf("l0");
        printf(" ");
    }

    // right-hand amplitudes
    print_amplitudes("r",data->right_amplitudes);
    if ( data->has_r0 ) {
        printf("r0");
        printf(" ");
    }

    // t_amplitudes
    print_amplitudes("t",data->t_amplitudes);

    // u_amplitudes
    print_amplitudes("u",data->u_amplitudes);
    if ( data->has_u0 ) {
        printf("u0");
        printf(" ");
    }

    // m_amplitudes
    print_amplitudes("m",data->m_amplitudes);
    if ( data->has_m0 ) {
        printf("m0");
        printf(" ");
    }

    // s_amplitudes
    print_amplitudes("s",data->s_amplitudes);
    if ( data->has_s0 ) {
        printf("s0");
        printf(" ");
    }

    // bosons:
    for (size_t i = 0; i < data->is_boson_dagger.size(); i++) {
        if ( data->is_boson_dagger[i] ) {
            printf("B* ");
        }else {
            printf("B ");
        }
    }
    if ( data->has_w0 ) {
        printf("w0");
        printf(" ");
    }
/*
    if ( data->has_b ) {
        printf("b-");
        printf(" ");
    }
    if ( data->has_b_dagger ) {
        printf("b+");
        printf(" ");
    }
*/
    printf("\n");
}

void pq::print_amplitudes_to_string(std::string label, 
                                    std::vector<std::vector<std::string> > amplitudes, 
                                    std::vector<std::string> &my_string ) {

    if ( amplitudes.size() == 0 ) {
        return;
    }
    
    for (size_t i = 0; i < amplitudes.size(); i++) {
        
        if ( amplitudes[i].size() > 0 ) {
            
            size_t order = amplitudes[i].size() / 2;
            std::string tmp = label + std::to_string(order) + "(";
            for (int j = 0; j < 2*order-1; j++) {
                tmp += amplitudes[i][j] + ",";
            }
            tmp += amplitudes[i][2*order-1] + ")";
            my_string.push_back(tmp);
        
        }

    }
}

std::vector<std::string> pq::get_string() {

    std::vector<std::string> my_string;

    if ( skip ) return my_string;

    if ( vacuum == "FERMI" && symbol.size() > 0 ) {
        // check if stings should be zero or not
        bool is_dagger_right = is_dagger_fermi[symbol.size()-1];
        bool is_dagger_left  = is_dagger_fermi[0];
        if ( !is_dagger_right || is_dagger_left ) {
            //return;
        }
    }

    std::string tmp;
    if ( sign > 0 ) {
        tmp = "+";
    }else {
        tmp = "-";
    }
    //my_string.push_back(tmp + std::to_string(fabs(data->factor)));
    my_string.push_back(tmp + to_string_with_precision(fabs(data->factor),14));

    if ( data->permutations.size() > 0 ) {
        // should have an even number of symbols...how many pairs?
        size_t n = data->permutations.size() / 2;
        size_t count = 0;
        for (size_t i = 0; i < n; i++) {
            tmp  = "P(";
            tmp += data->permutations[count++];
            tmp += ",";
            tmp += data->permutations[count++];
            tmp += ")";
            my_string.push_back(tmp);
        }
    }

    for (size_t i = 0; i < symbol.size(); i++) {
        std::string tmp = symbol[i];
        if ( is_dagger[i] ) {
            tmp += "*";
        }
        my_string.push_back(tmp);
    }

    for (size_t i = 0; i < delta1.size(); i++) {
        std::string tmp = "d(" + delta1[i] + "," + delta2[i] + ")";
        my_string.push_back(tmp);
    }

    // two-electron integrals
    if ( data->tensor.size() == 4 ) {

        if ( data->tensor_type == "TWO_BODY") {
            std::string tmp = "g("
                            + data->tensor[0]
                            + ","
                            + data->tensor[1]
                            + ","
                            + data->tensor[2]
                            + ","
                            + data->tensor[3]
                            + ")";
            my_string.push_back(tmp);
        }else {
            // dirac
            std::string tmp = "<"
                            + data->tensor[0]
                            + ","
                            + data->tensor[1]
                            + "||"
                            + data->tensor[2]
                            + ","
                            + data->tensor[3]
                            + ">";
            my_string.push_back(tmp);
        }
    }

    // one-electron integrals
    if ( data->tensor.size() == 2 ) {
        std::string tmp;
        if ( data->tensor_type == "CORE") {
            tmp = "h(";
        }else if ( data->tensor_type == "FOCK") {
            tmp = "f(";
        }else if ( data->tensor_type == "D+") {
            tmp = "d+(";
        }else if ( data->tensor_type == "D-") {
            tmp = "d-(";
        }
        tmp += data->tensor[0]
             + ","
             + data->tensor[1]
             + ")";
        my_string.push_back(tmp);
    }

    // left-hand amplitudes
    print_amplitudes_to_string("l",data->left_amplitudes,my_string);
    if ( data->has_l0 ) {
        my_string.push_back("l0");
    }

    // right-hand amplitudes
    print_amplitudes_to_string("r",data->right_amplitudes,my_string);
    if ( data->has_r0 ) {
        my_string.push_back("r0");
    }

    // t_amplitudes
    print_amplitudes_to_string("t",data->t_amplitudes,my_string);

    // u_amplitudes
    print_amplitudes_to_string("u",data->u_amplitudes,my_string);
    if ( data->has_u0 ) {
        my_string.push_back("u0");
    }

    // m_amplitudes
    print_amplitudes_to_string("m",data->m_amplitudes,my_string);
    if ( data->has_m0 ) {
        my_string.push_back("m0");
    }

    // s_amplitudes
    print_amplitudes_to_string("s",data->s_amplitudes,my_string);
    if ( data->has_s0 ) {
        my_string.push_back("s0");
    }

    // bosons:
    for (size_t i = 0; i < data->is_boson_dagger.size(); i++) {
        if ( data->is_boson_dagger[i] ) {
            my_string.push_back("B*");
        }else {
            my_string.push_back("B");
        }
    }
    if ( data->has_w0 ) {
        my_string.push_back("w0");
    }
/*
    if ( data->has_b ) {
        my_string.push_back("b-");
    }
    if ( data->has_b_dagger ) {
        my_string.push_back("b+");
    }
*/

    return my_string;
}

bool pq::is_normal_order() {

    // don't bother bringing to normal order if we're going to skip this string
    if (skip) return true;

    // fermions
    if ( vacuum == "TRUE" ) {
        for (int i = 0; i < (int)symbol.size()-1; i++) {
            if ( !is_dagger[i] && is_dagger[i+1] ) {
                return false;
            }
        }
    }else {
        for (int i = 0; i < (int)symbol.size()-1; i++) {
            // check if stings should be zero or not
            bool is_dagger_right = is_dagger_fermi[symbol.size()-1];
            bool is_dagger_left  = is_dagger_fermi[0];
            if ( !is_dagger_right || is_dagger_left ) {
                skip = true; // added 5/28/21
                return true;
            }
            if ( !is_dagger_fermi[i] && is_dagger_fermi[i+1] ) {
                return false;
            }
        }
    }

    // bosons
    if ( !is_boson_normal_order() ) {
        return false;
    }

    return true;
}

bool pq::is_boson_normal_order() {

    if ( data->is_boson_dagger.size() == 1 ) {
        bool is_dagger_right = data->is_boson_dagger[0];
        bool is_dagger_left  = data->is_boson_dagger[0];
        if ( !is_dagger_right || is_dagger_left ) {
            skip = true; 
            return true;
        }
    }
    for (int i = 0; i < (int)data->is_boson_dagger.size()-1; i++) {

        // check if stings should be zero or not ... added 5/28/21
        bool is_dagger_right = data->is_boson_dagger[data->is_boson_dagger.size()-1];
        bool is_dagger_left  = data->is_boson_dagger[0];
        if ( !is_dagger_right || is_dagger_left ) {
            skip = true; 
            return true;
        }

        if ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] ) {
            return false;
        }
    }
    return true;

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
        for (size_t j = 0; j < ordered[i]->delta1.size(); j++) {
            int val1 = ordered[i]->delta1[j].c_str()[0];
            int val2 = ordered[i]->delta2[j].c_str()[0];
            if ( val2 < val1 ) {
                std::string dum = ordered[i]->delta1[j];
                ordered[i]->delta1[j] = ordered[i]->delta2[j];
                ordered[i]->delta2[j] = dum;
            }
        }
    }
}

void pq::swap_two_labels(std::string label1, std::string label2) {

    replace_index_everywhere(label1,"x");
    replace_index_everywhere(label2,label1);
    replace_index_everywhere("x",label2);
}

void pq::reorder_t_amplitudes() {

    size_t dim = data->t_amplitudes.size();

    if ( dim == 0 ) return;

    bool* nope = (bool*)malloc(dim * sizeof(bool));
    memset((void*)nope,'\0',dim * sizeof(bool));

    std::vector<std::vector<std::string> > tmp;

    // t1 first
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if ( nope[j] ) continue;

            if ( data->t_amplitudes[j].size() == 2 ) {
                tmp.push_back(data->t_amplitudes[j]);
                nope[j] = true;
                break;
            }
        }

    }
    // now t2
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if ( nope[j] ) continue;

            if ( data->t_amplitudes[j].size() == 4 ) {
                tmp.push_back(data->t_amplitudes[j]);
                nope[j] = true;
                break;
            }
        }

    }
    // now t3
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if ( nope[j] ) continue;

            if ( data->t_amplitudes[j].size() == 6 ) {
                tmp.push_back(data->t_amplitudes[j]);
                nope[j] = true;
                break;
            }
        }

    }
    // now t4
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if ( nope[j] ) continue;

            if ( data->t_amplitudes[j].size() == 8 ) {
                tmp.push_back(data->t_amplitudes[j]);
                nope[j] = true;
                break;
            }
        }

    }
    // now t5
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if ( nope[j] ) continue;

            if ( data->t_amplitudes[j].size() == 10 ) {
                tmp.push_back(data->t_amplitudes[j]);
                nope[j] = true;
                break;
            }
        }

    }
    // now t6
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            if ( nope[j] ) continue;

            if ( data->t_amplitudes[j].size() == 12 ) {
                tmp.push_back(data->t_amplitudes[j]);
                nope[j] = true;
                break;
            }
        }

    }
    if ( dim != tmp.size() ) { 
        printf("\n");
        printf("    something went very wrong in reorder_t_amplitudes()\n");
        printf("    this function breaks for t6 and higher. why would\n");
        printf("    you want that, anyway?\n");
        printf("\n");
        exit(1);
    }

    for (int i = 0; i < dim; i++) {
        data->t_amplitudes[i].clear();
    }
    data->t_amplitudes.clear();
    for (size_t i = 0; i < tmp.size(); i++) {
        data->t_amplitudes.push_back(tmp[i]);
    }

    free(nope);
    
}

// compare strings and remove terms that cancel

void pq::cleanup(std::vector<std::shared_ptr<pq> > &ordered) {

    // order amplitudes such that they're ordered t1, t2, t3, etc.
    for (size_t i = 0; i < ordered.size(); i++) {
        ordered[i]->reorder_t_amplitudes();
    }

    // prune list so it only contains non-skipped ones
    std::vector< std::shared_ptr<pq> > pruned;
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]-> skip ) continue;

        // for normal order relative to fermi vacuum, i doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming
        if ( vacuum == "FERMI" ) {
            if ( ordered[i]->symbol.size() != 0 ) continue;
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

/*
    // these don't seem to be necessary for any of the test cases.
    consolidate_permutations_plus_three_swaps(ordered,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_three_swaps(ordered,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_three_swaps(ordered,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_three_swaps(ordered,vir_labels,vir_labels,vir_labels);

    consolidate_permutations_plus_four_swaps(ordered,occ_labels,occ_labels,occ_labels,occ_labels);
    consolidate_permutations_plus_four_swaps(ordered,occ_labels,occ_labels,occ_labels,vir_labels);
    consolidate_permutations_plus_four_swaps(ordered,occ_labels,occ_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_four_swaps(ordered,occ_labels,vir_labels,vir_labels,vir_labels);
    consolidate_permutations_plus_four_swaps(ordered,vir_labels,vir_labels,vir_labels,vir_labels);
*/

    // probably only relevant for vacuum = fermi
    if ( vacuum != "FERMI" ) return;

    consolidate_permutations_non_summed(ordered,occ_labels);
    consolidate_permutations_non_summed(ordered,vir_labels);

}

// consolidate terms that differ by permutations of non-summed labels
void pq::consolidate_permutations_non_summed(
    std::vector<std::shared_ptr<pq> > &ordered,
    std::vector<std::string> labels) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_idx;

        // ok, what labels do we have?
        for (size_t j = 0; j < labels.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels[j]);
            find_idx.push_back(found);
        }

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            std::string permutation_1;
            std::string permutation_2;

            // try swapping non-summed labels
            for (size_t id1 = 0; id1 < labels.size(); id1++) {
                if ( find_idx[id1] != 1 ) continue;
                for (size_t id2 = id1 + 1; id2 < labels.size(); id2++) {
                    if ( find_idx[id2] != 1 ) continue;

                    std::shared_ptr<pq> newguy (new pq(vacuum));
                    newguy->copy((void*)(ordered[i].get()));
                    newguy->swap_two_labels(labels[id1],labels[id2]);
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

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, then this is a permutation
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->data->permutations.push_back(permutation_1);
                ordered[i]->data->permutations.push_back(permutation_2);
                ordered[j]->skip = true;
                break;
            }

            // otherwise, something has gone wrong in the previous consolidation step...
        }
    }
}


// consolidate terms that differ by four summed labels plus permutations
void pq::consolidate_permutations_plus_four_swaps(
    std::vector<std::shared_ptr<pq> > &ordered,
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
            int found = ordered[i]->index_in_anywhere(labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_3[j]);
            find_3.push_back(found);
        }

        // ok, what labels do we have? list 4
        for (size_t j = 0; j < labels_4.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_4[j]);
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

                                            std::shared_ptr<pq> newguy (new pq(vacuum));
                                            newguy->copy((void*)(ordered[i].get()));
                                            newguy->swap_two_labels(labels_1[id1],labels_1[id2]);
                                            newguy->swap_two_labels(labels_2[id3],labels_2[id4]);
                                            newguy->swap_two_labels(labels_3[id5],labels_3[id6]);
                                            newguy->swap_two_labels(labels_4[id7],labels_4[id8]);
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

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
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
void pq::consolidate_permutations_plus_three_swaps(
    std::vector<std::shared_ptr<pq> > &ordered,
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
            int found = ordered[i]->index_in_anywhere(labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_2[j]);
            find_2.push_back(found);
        }

        // ok, what labels do we have? list 3
        for (size_t j = 0; j < labels_3.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_3[j]);
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

                                    std::shared_ptr<pq> newguy (new pq(vacuum));
                                    newguy->copy((void*)(ordered[i].get()));
                                    newguy->swap_two_labels(labels_1[id1],labels_1[id2]);
                                    newguy->swap_two_labels(labels_2[id3],labels_2[id4]);
                                    newguy->swap_two_labels(labels_3[id5],labels_3[id6]);
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

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
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
void pq::consolidate_permutations_plus_two_swaps(
    std::vector<std::shared_ptr<pq> > &ordered,
    std::vector<std::string> labels_1,
    std::vector<std::string> labels_2) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_1;
        std::vector<int> find_2;

        // ok, what labels do we have? list 1
        for (size_t j = 0; j < labels_1.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_1[j]);
            find_1.push_back(found);
        }

        // ok, what labels do we have? list 2
        for (size_t j = 0; j < labels_2.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels_2[j]);
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

                            std::shared_ptr<pq> newguy (new pq(vacuum));
                            newguy->copy((void*)(ordered[i].get()));
                            newguy->swap_two_labels(labels_1[id1],labels_1[id2]);
                            newguy->swap_two_labels(labels_2[id3],labels_2[id4]);
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

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
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
void pq::consolidate_permutations_plus_swap(std::vector<std::shared_ptr<pq> > &ordered,
                                            std::vector<std::string> labels) {

    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        std::vector<int> find_idx;

        // ok, what labels do we have?
        for (size_t j = 0; j < labels.size(); j++) {
            int found = ordered[i]->index_in_anywhere(labels[j]);
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

                    std::shared_ptr<pq> newguy (new pq(vacuum));
                    newguy->copy((void*)(ordered[i].get()));
                    newguy->swap_two_labels(labels[id1],labels[id2]);
                    strings_same = compare_strings(ordered[j],newguy,n_permute);

                    if ( strings_same ) break;
                }
                if ( strings_same ) break;
            }

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

        }
    }
}

// consolidate terms that differ by permutations
void pq::consolidate_permutations(std::vector<std::shared_ptr<pq> > &ordered) {

    // consolidate terms that differ by permutations
    for (size_t i = 0; i < ordered.size(); i++) {

        if ( ordered[i]->skip ) continue;

        for (size_t j = i+1; j < ordered.size(); j++) {

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            if ( !strings_same ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;
        }
    }
}

bool pq::compare_strings(std::shared_ptr<pq> ordered_1, std::shared_ptr<pq> ordered_2, int & n_permute) {

    // don't forget w0, u0, r0, l0, b+, b-, m0, s0
    if ( ordered_1->data->has_u0 != ordered_2->data->has_u0 ) {
        return false;
    }
    if ( ordered_1->data->has_m0 != ordered_2->data->has_m0 ) {
        return false;
    }
    if ( ordered_1->data->has_s0 != ordered_2->data->has_s0 ) {
        return false;
    }
    if ( ordered_1->data->has_w0 != ordered_2->data->has_w0 ) {
        return false;
    }
    if ( ordered_1->data->has_r0 != ordered_2->data->has_r0 ) {
        return false;
    }
    if ( ordered_1->data->has_l0 != ordered_2->data->has_l0 ) {
        return false;
    }

    n_permute = 0;

    //printf("ok, how about these\n");
    //ordered[i]->print();
    //ordered[j]->print();

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
    for (size_t k = 0; k < ordered_1->delta1.size(); k++) {
        for (size_t l = 0; l < ordered_2->delta1.size(); l++) {
            if ( ordered_1->delta1[k] == ordered_2->delta1[l] && ordered_1->delta2[k] == ordered_2->delta2[l] ) {
                nsame_d++;
                //break;
            }else if ( ordered_1->delta2[k] == ordered_2->delta1[l] && ordered_1->delta1[k] == ordered_2->delta2[l] ) {
                nsame_d++;
                //break;
            }
        }
    }
    if ( nsame_d != ordered_1->delta1.size() ) return false;

    // t_amplitudes
    bool same_string = compare_amplitudes( ordered_1->data->t_amplitudes, ordered_2->data->t_amplitudes, n_permute);
    if ( !same_string ) return false;

    // u_amplitudes
    same_string = compare_amplitudes( ordered_1->data->u_amplitudes, ordered_2->data->u_amplitudes, n_permute);
    if ( !same_string ) return false;

    // m_amplitudes
    same_string = compare_amplitudes( ordered_1->data->m_amplitudes, ordered_2->data->m_amplitudes, n_permute);
    if ( !same_string ) return false;

    // s_amplitudes
    same_string = compare_amplitudes( ordered_1->data->s_amplitudes, ordered_2->data->s_amplitudes, n_permute);
    if ( !same_string ) return false;

    // left-hand amplitudes
    same_string = compare_amplitudes( ordered_1->data->left_amplitudes, ordered_2->data->left_amplitudes, n_permute);
    if ( !same_string ) return false;

    // right-hand amplitudes
    same_string = compare_amplitudes( ordered_1->data->right_amplitudes, ordered_2->data->right_amplitudes, n_permute);
    if ( !same_string ) return false;

    // are tensors same?
    if ( ordered_1->data->tensor_type != ordered_2->data->tensor_type ) return false;

    int nsame_t = 0;
    for (size_t k = 0; k < ordered_1->data->tensor.size(); k++) {
        if ( ordered_1->data->tensor[k] == ordered_2->data->tensor[k] ) {
            nsame_t++;
        }
    }

    // if not the same, check antisymmetry <ij||kl> = -<ji||lk> = -<ij||lk> = <ji||lk>
    if ( nsame_t != ordered_1->data->tensor.size() ) {

        if ( ordered_1->data->tensor.size() == 4 ) {

            nsame_t = 0;
            if ( ordered_1->data->tensor[0] == ordered_2->data->tensor[0] ) {
                nsame_t++;
            }
            if ( ordered_1->data->tensor[1] == ordered_2->data->tensor[1] ) {
                nsame_t++;
            }
            if ( ordered_1->data->tensor[2] == ordered_2->data->tensor[3] ) {
                nsame_t++;
            }
            if ( ordered_1->data->tensor[3] == ordered_2->data->tensor[2] ) {
                nsame_t++;
            }
            if ( nsame_t == 4 ) {
                n_permute++;
            }

        }
    }
    if ( nsame_t != ordered_1->data->tensor.size() ) {

        if ( ordered_1->data->tensor.size() == 4 ) {

            nsame_t = 0;
            if ( ordered_1->data->tensor[0] == ordered_2->data->tensor[1] ) {
                nsame_t++;
            }
            if ( ordered_1->data->tensor[1] == ordered_2->data->tensor[0] ) {
                nsame_t++;
            }
            if ( ordered_1->data->tensor[2] == ordered_2->data->tensor[2] ) {
                nsame_t++;
            }
            if ( ordered_1->data->tensor[3] == ordered_2->data->tensor[3] ) {
                nsame_t++;
            }
            if ( nsame_t == 4 ) {
                n_permute++;
            }

        }
    }
    if ( nsame_t != ordered_1->data->tensor.size() ) {

        if ( ordered_1->data->tensor.size() == 4 ) {

            nsame_t = 0;
            if ( ordered_1->data->tensor[0] == ordered_2->data->tensor[1] ) {
                nsame_t++;
            }
            if ( ordered_1->data->tensor[1] == ordered_2->data->tensor[0] ) {
                nsame_t++;
            }
            if ( ordered_1->data->tensor[2] == ordered_2->data->tensor[3] ) {
                nsame_t++;
            }
            if ( ordered_1->data->tensor[3] == ordered_2->data->tensor[2] ) {
                nsame_t++;
            }

        }
    }

    if ( nsame_t != ordered_1->data->tensor.size() ) {
        return false;
    }

    return true;
}

/// compare two lists of amplitudes
bool pq::compare_amplitudes( std::vector<std::vector<std::string> > amps1, 
                             std::vector<std::vector<std::string> > amps2, 
                             int & n_permute ) {

    // same number of amplitudes?
    if ( amps1.size() != amps2.size() ) return false;
   
    size_t nsame_amps = 0;
    for (size_t i = 0; i < amps1.size(); i++) {
        for (size_t j = 0; j < amps2.size(); j++) {

            // t1 vs t2 vs t3, etc?
            if ( amps1[i].size() != amps2[j].size() ) continue;

            // for higher than t4, just return false

            // check labels
            size_t dim = amps1[i].size();
            int nsame_idx = 0;

            // cases: 

            // dim = 2: singles, no permutations
            if ( dim == 2 ) {

                if ( amps1[i][0] == amps2[j][0] && amps1[i][1] == amps2[j][1] ) {
                    nsame_idx = 2;
                }

            // dim = 4: doubles
            }else if ( dim == 4 ) {
            
                // first part
                if ( amps1[i][0] == amps2[j][0] 
                  && amps1[i][1] == amps2[j][1] ) {

                    nsame_idx += 2;

                }else if ( amps1[i][0] == amps2[j][1] 
                        && amps1[i][1] == amps2[j][0] ) {

                    nsame_idx += 2;
                    n_permute++;

                }

                // second part
                if ( amps1[i][2] == amps2[j][2] 
                  && amps1[i][3] == amps2[j][3] ) {

                    nsame_idx += 2;

                }else if ( amps1[i][2] == amps2[j][3] 
                        && amps1[i][3] == amps2[j][2] ) {

                    nsame_idx += 2;
                    n_permute++;

                }
            
            // dim = 6: triples
            }else if ( dim == 6 ) {

                // first part
                triples_permutations(amps1[i],amps2[j],nsame_idx,n_permute,0);

                // second part
                triples_permutations(amps1[i],amps2[j],nsame_idx,n_permute,3);

            // dim = 8: quadruples
            }else if ( dim == 8 ) {

                // first part
                quadruples_permutations(amps1[i],amps2[j],nsame_idx,n_permute,0);

                // second part
                quadruples_permutations(amps1[i],amps2[j],nsame_idx,n_permute,4);

            }else {

                return false;

            }

            // if all indices are the same, the amplitudes must be the same, but we need to be careful of permutations
            if ( nsame_idx == dim ) {
                nsame_amps++;
                break;
            }
        }
    }
    if ( nsame_amps != amps1.size() ) return false;

    return true;
}

/// permutations and coincidences for triples
void pq::triples_permutations(std::vector<std::string> amps1, 
                              std::vector<std::string> amps2, 
                              int & nsame_idx, 
                              int & n_permute,
                              int off) {

   if ( amps1[0+off] == amps2[0+off] 
     && amps1[1+off] == amps2[1+off] 
     && amps1[2+off] == amps2[2+off] ) {

       nsame_idx += 3;

   }else if ( amps1[0+off] == amps2[0+off] 
           && amps1[1+off] == amps2[2+off] 
           && amps1[2+off] == amps2[1+off] ) {

       nsame_idx += 3;
       n_permute++;

   }else if ( amps1[0+off] == amps2[1+off] 
           && amps1[1+off] == amps2[0+off] 
           && amps1[2+off] == amps2[2+off] ) {

       nsame_idx += 3;
       n_permute++;

   }else if ( amps1[0+off] == amps2[1+off] 
           && amps1[1+off] == amps2[2+off] 
           && amps1[2+off] == amps2[0+off] ) {

       nsame_idx += 3;

   }else if ( amps1[0+off] == amps2[2+off] 
           && amps1[1+off] == amps2[0+off] 
           && amps1[2+off] == amps2[1+off] ) {

       nsame_idx += 3;

   }else if ( amps1[0+off] == amps2[2+off] 
           && amps1[1+off] == amps2[1+off] 
           && amps1[2+off] == amps2[0+off] ) {

       nsame_idx += 3;
       n_permute++;

   }
}

/// permutations and coincidences for quadruples
void pq::quadruples_permutations(std::vector<std::string> amps1, 
                                 std::vector<std::string> amps2, 
                                 int & nsame_idx, 
                                 int & n_permute,
                                 int off) {

    if ( amps1[0+off] == amps2[0+off] 
      && amps1[1+off] == amps2[1+off] 
      && amps1[2+off] == amps2[2+off]
      && amps1[3+off] == amps2[3+off] ) {

        nsame_idx += 4;

    }else if ( amps1[0+off] == amps2[0+off] 
            && amps1[1+off] == amps2[1+off] 
            && amps1[2+off] == amps2[3+off]
            && amps1[3+off] == amps2[2+off] ) {

        nsame_idx += 4;
        n_permute++;

    }else if ( amps1[0+off] == amps2[0+off] 
            && amps1[1+off] == amps2[2+off] 
            && amps1[2+off] == amps2[1+off]
            && amps1[3+off] == amps2[3+off] ) {

        nsame_idx += 4;
        n_permute++;

    }else if ( amps1[0+off] == amps2[0+off] 
            && amps1[1+off] == amps2[2+off] 
            && amps1[2+off] == amps2[3+off]
            && amps1[3+off] == amps2[1+off] ) {

        nsame_idx += 4;

    }else if ( amps1[0+off] == amps2[0+off] 
            && amps1[1+off] == amps2[3+off] 
            && amps1[2+off] == amps2[1+off]
            && amps1[3+off] == amps2[2+off] ) {

        nsame_idx += 4;

    }else if ( amps1[0+off] == amps2[0+off] 
            && amps1[1+off] == amps2[3+off] 
            && amps1[2+off] == amps2[2+off]
            && amps1[3+off] == amps2[1+off] ) {

        nsame_idx += 4;
        n_permute++;

    }else if ( amps1[0+off] == amps2[1+off] 
            && amps1[1+off] == amps2[0+off] 
            && amps1[2+off] == amps2[2+off]
            && amps1[3+off] == amps2[3+off] ) {

        nsame_idx += 4;
        n_permute++;

    }else if ( amps1[0+off] == amps2[1+off] 
            && amps1[1+off] == amps2[0+off] 
            && amps1[2+off] == amps2[3+off]
            && amps1[3+off] == amps2[2+off] ) {

        nsame_idx += 4;

    }else if ( amps1[0+off] == amps2[1+off] 
            && amps1[1+off] == amps2[2+off] 
            && amps1[2+off] == amps2[3+off]
            && amps1[3+off] == amps2[0+off] ) {

        nsame_idx += 4;
        n_permute++;

    }else if ( amps1[0+off] == amps2[1+off] 
            && amps1[1+off] == amps2[2+off] 
            && amps1[2+off] == amps2[0+off]
            && amps1[3+off] == amps2[3+off] ) {

        nsame_idx += 4;

    }else if ( amps1[0+off] == amps2[1+off] 
            && amps1[1+off] == amps2[3+off] 
            && amps1[2+off] == amps2[0+off]
            && amps1[3+off] == amps2[2+off] ) {

        nsame_idx += 4;
        n_permute++;

    }else if ( amps1[0+off] == amps2[1+off] 
            && amps1[1+off] == amps2[3+off] 
            && amps1[2+off] == amps2[2+off]
            && amps1[3+off] == amps2[0+off] ) {

        nsame_idx += 4;

    }else if ( amps1[0+off] == amps2[2+off] 
            && amps1[1+off] == amps2[0+off] 
            && amps1[2+off] == amps2[3+off]
            && amps1[3+off] == amps2[1+off] ) {

        nsame_idx += 4;
        n_permute++;

    }else if ( amps1[0+off] == amps2[2+off] 
            && amps1[1+off] == amps2[0+off] 
            && amps1[2+off] == amps2[1+off]
            && amps1[3+off] == amps2[3+off] ) {

        nsame_idx += 4;

    }else if ( amps1[0+off] == amps2[2+off] 
            && amps1[1+off] == amps2[1+off] 
            && amps1[2+off] == amps2[0+off]
            && amps1[3+off] == amps2[3+off] ) {

        nsame_idx += 4;
        n_permute++;

    }else if ( amps1[0+off] == amps2[2+off] 
            && amps1[1+off] == amps2[1+off] 
            && amps1[2+off] == amps2[3+off]
            && amps1[3+off] == amps2[0+off] ) {

        nsame_idx += 4;

    }else if ( amps1[0+off] == amps2[2+off] 
            && amps1[1+off] == amps2[3+off] 
            && amps1[2+off] == amps2[0+off]
            && amps1[3+off] == amps2[1+off] ) {

        nsame_idx += 4;

    }else if ( amps1[0+off] == amps2[2+off] 
            && amps1[1+off] == amps2[3+off] 
            && amps1[2+off] == amps2[1+off]
            && amps1[3+off] == amps2[0+off] ) {

        nsame_idx += 4;
        n_permute++;

    }else if ( amps1[0+off] == amps2[3+off] 
            && amps1[1+off] == amps2[0+off] 
            && amps1[2+off] == amps2[1+off]
            && amps1[3+off] == amps2[2+off] ) {

        nsame_idx += 4;
        n_permute++;

    }else if ( amps1[0+off] == amps2[3+off] 
            && amps1[1+off] == amps2[0+off] 
            && amps1[2+off] == amps2[2+off]
            && amps1[3+off] == amps2[1+off] ) {

        nsame_idx += 4;

    }else if ( amps1[0+off] == amps2[3+off] 
            && amps1[1+off] == amps2[1+off] 
            && amps1[2+off] == amps2[0+off]
            && amps1[3+off] == amps2[2+off] ) {

        nsame_idx += 4;

    }else if ( amps1[0+off] == amps2[3+off] 
            && amps1[1+off] == amps2[1+off] 
            && amps1[2+off] == amps2[2+off]
            && amps1[3+off] == amps2[0+off] ) {

        nsame_idx += 4;
        n_permute++;

    }else if ( amps1[0+off] == amps2[3+off] 
            && amps1[1+off] == amps2[2+off] 
            && amps1[2+off] == amps2[0+off]
            && amps1[3+off] == amps2[1+off] ) {

        nsame_idx += 4;
        n_permute++;

    }else if ( amps1[0+off] == amps2[3+off] 
            && amps1[1+off] == amps2[2+off] 
            && amps1[2+off] == amps2[1+off]
            && amps1[3+off] == amps2[0+off] ) {

        nsame_idx += 4;

   }
}

// copy all data, except symbols and daggers. 

void pq::shallow_copy(void * copy_me) { 

    pq * in = reinterpret_cast<pq * >(copy_me);

    // skip string?
    skip   = in->skip;
    
    // sign
    sign   = in->sign;
    
    // factor
    data->factor = in->data->factor;

    // temporary delta functions
    std::vector<std::string> tmp_delta1;
    std::vector<std::string> tmp_delta2;

    // data->tensor
    for (size_t i = 0; i < in->data->tensor.size(); i++) {
        data->tensor.push_back(in->data->tensor[i]);
    }

    // data->tensor_type
    data->tensor_type = in->data->tensor_type;

    // delta1, delta2
    for (size_t i = 0; i < in->delta1.size(); i++) {
        delta1.push_back(in->delta1[i]);
        delta2.push_back(in->delta2[i]);
    }

    // t_amplitudes
    for (size_t i = 0; i < in->data->t_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (size_t j = 0; j < in->data->t_amplitudes[i].size(); j++) {
            tmp.push_back(in->data->t_amplitudes[i][j]);
        }
        data->t_amplitudes.push_back(tmp);
    }

    // u_amplitudes
    for (size_t i = 0; i < in->data->u_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (size_t j = 0; j < in->data->u_amplitudes[i].size(); j++) {
            tmp.push_back(in->data->u_amplitudes[i][j]);
        }
        data->u_amplitudes.push_back(tmp);
    }

    // m_amplitudes
    for (size_t i = 0; i < in->data->m_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (size_t j = 0; j < in->data->m_amplitudes[i].size(); j++) {
            tmp.push_back(in->data->m_amplitudes[i][j]);
        }
        data->m_amplitudes.push_back(tmp);
    }

    // s_amplitudes
    for (size_t i = 0; i < in->data->s_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (size_t j = 0; j < in->data->s_amplitudes[i].size(); j++) {
            tmp.push_back(in->data->s_amplitudes[i][j]);
        }
        data->s_amplitudes.push_back(tmp);
    }

    // left-hand amplitudes
    for (size_t i = 0; i < in->data->left_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (size_t j = 0; j < in->data->left_amplitudes[i].size(); j++) {
            tmp.push_back(in->data->left_amplitudes[i][j]);
        }
        data->left_amplitudes.push_back(tmp);
    }

    // right-hand amplitudes
    for (size_t i = 0; i < in->data->right_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (size_t j = 0; j < in->data->right_amplitudes[i].size(); j++) {
            tmp.push_back(in->data->right_amplitudes[i][j]);
        }
        data->right_amplitudes.push_back(tmp);
    }

    // l0 
    data->has_l0 = in->data->has_l0;

    // r0 
    data->has_r0 = in->data->has_r0;

    // u0 
    data->has_u0 = in->data->has_u0;

    // m0 
    data->has_m0 = in->data->has_m0;

    // s0 
    data->has_s0 = in->data->has_s0;

    // w0 
    data->has_w0 = in->data->has_w0;

/*
    // b 
    data->has_b = in->data->has_b;

    // b_dagger 
    data->has_b_dagger = in->data->has_b_dagger;
*/

}


int pq::index_in_anywhere(std::string idx) {

    int n = 0;

    n += index_in_deltas(idx);
    n += index_in_tensor(idx);
    n += index_in_term(idx, data->t_amplitudes);
    n += index_in_term(idx, data->u_amplitudes);
    n += index_in_term(idx, data->m_amplitudes);
    n += index_in_term(idx, data->s_amplitudes);
    n += index_in_term(idx, data->left_amplitudes);
    n += index_in_term(idx, data->right_amplitudes);

    return n;

}

int pq::index_in_deltas(std::string idx) {

    int n = 0;
    for (size_t i = 0; i < delta1.size(); i++) {
        if ( delta1[i] == idx ) {
            n++;
        }
    }
    for (size_t i = 0; i < delta2.size(); i++) {
        if ( delta2[i] == idx ) {
            n++;
        }
    }
    return n;

}
int pq::index_in_tensor(std::string idx) {

    int n = 0;
    for (size_t i = 0; i < data->tensor.size(); i++) {
        if ( data->tensor[i] == idx ) {
            n++;
        }
    }
    return n;

}

int pq::index_in_term(std::string idx, std::vector<std::vector<std::string> > term) {

    int n = 0;
    for (size_t i = 0; i < term.size(); i++) {
        for (size_t j = 0; j < term[i].size(); j++) {
            if ( term[i][j] == idx ) {
                n++;
            }
           
        }
    }
    return n;

}

void pq::replace_index_everywhere(std::string old_idx, std::string new_idx) {

    //replace_index_in_deltas(old_idx,new_idx);
    replace_index_in_tensor(old_idx,new_idx);
    replace_index_in_term(old_idx,new_idx,data->t_amplitudes);
    replace_index_in_term(old_idx,new_idx,data->u_amplitudes);
    replace_index_in_term(old_idx,new_idx,data->m_amplitudes);
    replace_index_in_term(old_idx,new_idx,data->s_amplitudes);
    replace_index_in_term(old_idx,new_idx,data->left_amplitudes);
    replace_index_in_term(old_idx,new_idx,data->right_amplitudes);

}

void pq::replace_index_in_tensor(std::string old_idx, std::string new_idx) {

    for (size_t i = 0; i < data->tensor.size(); i++) {
        if ( data->tensor[i] == old_idx ) {
            data->tensor[i] = new_idx;
            // dont' return because indices may be repeated in two-electron integrals
            //return;
        }
    }

}
void pq::replace_index_in_deltas(std::string old_idx, std::string new_idx) {

    for (size_t i = 0; i < delta1.size(); i++) {
        if ( delta1[i] == old_idx ) {
            delta1[i] = new_idx;
            // dont' return because indices may be repeated in two-electron integrals
            //return;
        }
    }
    for (size_t i = 0; i < delta2.size(); i++) {
        if ( delta2[i] == old_idx ) {
            delta2[i] = new_idx;
            // dont' return because indices may be repeated in two-electron integrals
            //return;
        }
    }

}

void pq::replace_index_in_term(std::string old_idx, std::string new_idx, std::vector<std::vector<std::string> > &term) {

    for (size_t i = 0; i < term.size(); i++) {
        for (size_t j = 0; j < term[i].size(); j++) {
            if ( term[i][j] == old_idx ) {
                term[i][j] = new_idx;
            }
        }
    }

}

// find and replace any funny labels in tensors with conventional ones. i.e., o1 -> i ,v1 -> a
void pq::use_conventional_labels() {

    // occupied first:
    std::vector<std::string> occ_in{"o0","o1","o2","o3","o4","o5","o6","o7","o8","o9",
                                    "o10","o11","o12","o13","o14","o15","o16","o17","o18","o19",
                                    "o20","o21","o22","o23","o24","o25","o26","o27","o28","o29"};
    std::vector<std::string> occ_out{"i","j","k","l","m","n","o",
                                     "i0","i1","i2","i3","i4","i5","i6","i7","i8","i9",
                                     "i10","i11","i12","i13","i14","i15","i16","i17","i18","i19"};

    for (size_t i = 0; i < occ_in.size(); i++) {

        if ( index_in_anywhere(occ_in[i]) > 0 ) {

            for (size_t j = 0; j < occ_out.size(); j++) {

                //if ( !index_in_tensor(occ_out[j]) ) 
                if ( index_in_anywhere(occ_out[j]) == 0 ) {

                    //replace_index_in_tensor(occ_in[i],occ_out[j]);
                    replace_index_everywhere(occ_in[i],occ_out[j]);
                    break;
                }
            }
        }
    }

    // now virtual
    std::vector<std::string> vir_in{"v0","v1","v2","v3","v4","v5","v6","v7","v8","v9",
                                    "v10","v11","v12","v13","v14","v15","v16","v17","v18","v19",
                                    "v20","v21","v22","v23","v24","v25","v26","v27","v28","v29"};
    std::vector<std::string> vir_out{"a","b","c","d","e","f","g",
                                     "a0","a1","a2","a3","a4","a5","a6","a7","a8","a9",
                                     "a10","a11","a12","a13","a14","a15","a16","a17","a18","a19"};

    for (size_t i = 0; i < vir_in.size(); i++) {

        if ( index_in_anywhere(vir_in[i]) > 0 ) {

            for (size_t j = 0; j < vir_out.size(); j++) {

                if ( index_in_anywhere(vir_out[j]) == 0 ) {

                    //replace_index_in_tensor(vir_in[i],vir_out[j]);
                    replace_index_everywhere(vir_in[i],vir_out[j]);
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
        if ( index_in_anywhere( occ_labels[i] ) == 2 ) {
            sum_labels.push_back( occ_labels[i] );
        }
    }
    for (size_t i = 0; i < vir_labels.size(); i++) {
        if ( index_in_anywhere( vir_labels[i] ) == 2 ) {
            sum_labels.push_back( vir_labels[i] );
        }
    }

    for (size_t i = 0; i < delta1.size(); i++) {

        // is delta label 1 in list of summation labels?
        bool have_delta1 = false;
        for (size_t j = 0; j < sum_labels.size(); j++) {
            if ( delta1[i] == sum_labels[j] ) {
                have_delta1 = true;
                break;
            }
        }
        // is delta label 2 in list of summation labels?
        bool have_delta2 = false;
        for (size_t j = 0; j < sum_labels.size(); j++) {
            if ( delta2[i] == sum_labels[j] ) {
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
            replace_index_everywhere( delta1[i], delta2[i] );
            continue;
        }else if ( have_delta2 ) {
            replace_index_everywhere( delta2[i], delta1[i] );
            continue;
        }
*/

        bool delta1_in_tensor           = ( index_in_tensor( delta1[i] ) > 0 ) ? true : false;
        bool delta2_in_tensor           = ( index_in_tensor( delta2[i] ) > 0 ) ? true : false;
        bool delta1_in_t_amplitudes     = ( index_in_term( delta1[i], data->t_amplitudes ) > 0 ) ? true : false;
        bool delta2_in_t_amplitudes     = ( index_in_term( delta2[i], data->t_amplitudes ) > 0 ) ? true : false;
        bool delta1_in_left_amplitudes  = ( index_in_term( delta1[i], data->left_amplitudes ) > 0 ) ? true : false;
        bool delta2_in_left_amplitudes  = ( index_in_term( delta2[i], data->left_amplitudes ) > 0 ) ? true : false;
        bool delta1_in_right_amplitudes = ( index_in_term( delta1[i], data->right_amplitudes ) > 0 ) ? true : false;
        bool delta2_in_right_amplitudes = ( index_in_term( delta2[i], data->right_amplitudes ) > 0 ) ? true : false;
        bool delta1_in_u_amplitudes     = ( index_in_term( delta1[i], data->u_amplitudes ) > 0 ) ? true : false;
        bool delta2_in_u_amplitudes     = ( index_in_term( delta2[i], data->u_amplitudes ) > 0 ) ? true : false;
        bool delta1_in_m_amplitudes     = ( index_in_term( delta1[i], data->m_amplitudes ) > 0 ) ? true : false;
        bool delta2_in_m_amplitudes     = ( index_in_term( delta2[i], data->m_amplitudes ) > 0 ) ? true : false;
        bool delta1_in_s_amplitudes     = ( index_in_term( delta1[i], data->s_amplitudes ) > 0 ) ? true : false;
        bool delta2_in_s_amplitudes     = ( index_in_term( delta2[i], data->s_amplitudes ) > 0 ) ? true : false;

        if ( delta1_in_tensor && have_delta1 ) {
            replace_index_in_tensor( delta1[i], delta2[i] );
            continue;
        }else if ( delta2_in_tensor && have_delta2 ) {
            replace_index_in_tensor( delta2[i], delta1[i] );
            continue;
        }else if ( delta1_in_t_amplitudes && have_delta1 ) {
            replace_index_in_term( delta1[i], delta2[i], data->t_amplitudes );
            continue;
        }else if ( delta2_in_t_amplitudes && have_delta2 ) {
            replace_index_in_term( delta2[i], delta1[i], data->t_amplitudes );
            continue;
        }else if ( delta1_in_left_amplitudes && have_delta1 ) {
            replace_index_in_term( delta1[i], delta2[i], data->left_amplitudes );
            continue;
        }else if ( delta2_in_left_amplitudes && have_delta2 ) {
            replace_index_in_term( delta2[i], delta1[i], data->left_amplitudes );
            continue;
        }else if ( delta1_in_right_amplitudes && have_delta1 ) {
            replace_index_in_term( delta1[i], delta2[i], data->right_amplitudes );
            continue;
        }else if ( delta2_in_right_amplitudes && have_delta2 ) {
            replace_index_in_term( delta2[i], delta1[i], data->right_amplitudes );
            continue;
        }else if ( delta1_in_u_amplitudes && have_delta1 ) {
            replace_index_in_term( delta1[i], delta2[i], data->u_amplitudes );
            continue;
        }else if ( delta2_in_u_amplitudes && have_delta2 ) {
            replace_index_in_term( delta2[i], delta1[i], data->u_amplitudes );
            continue;
        }else if ( delta1_in_m_amplitudes && have_delta1 ) {
            replace_index_in_term( delta1[i], delta2[i], data->m_amplitudes );
            continue;
        }else if ( delta2_in_m_amplitudes && have_delta2 ) {
            replace_index_in_term( delta2[i], delta1[i], data->m_amplitudes );
            continue;
        }else if ( delta1_in_s_amplitudes && have_delta1 ) {
            replace_index_in_term( delta1[i], delta2[i], data->s_amplitudes );
            continue;
        }else if ( delta2_in_s_amplitudes && have_delta2 ) {
            replace_index_in_term( delta2[i], delta1[i], data->s_amplitudes );
            continue;
        }

        // at this point, it is safe to assume the delta function must remain
        tmp_delta1.push_back(delta1[i]);
        tmp_delta2.push_back(delta2[i]);

    }

    delta1.clear();
    delta2.clear();

    for (size_t i = 0; i < tmp_delta1.size(); i++) {
        delta1.push_back(tmp_delta1[i]);
        delta2.push_back(tmp_delta2[i]);
    }

}

// copy all data, including symbols and daggers
void pq::copy(void * copy_me) { 

    shallow_copy(copy_me);

    pq * in = reinterpret_cast<pq * >(copy_me);

    // operators
    for (size_t j = 0; j < in->symbol.size(); j++) {
        symbol.push_back(in->symbol[j]);

        // dagger?
        is_dagger.push_back(in->is_dagger[j]);

        // dagger (relative to fermi vacuum)?
        if ( vacuum == "FERMI" ) {
            is_dagger_fermi.push_back(in->is_dagger_fermi[j]);
        }
    }

    // boson daggers
    for (size_t i = 0; i < in->data->is_boson_dagger.size(); i++) {
        data->is_boson_dagger.push_back(in->data->is_boson_dagger[i]);
    }
    
}

bool pq::normal_order_true_vacuum(std::vector<std::shared_ptr<pq> > &ordered) {

    if ( skip ) return true;

    if ( is_normal_order() ) {

        // push current ordered operator onto running list
        std::shared_ptr<pq> newguy (new pq(vacuum));

        newguy->copy((void*)this);

        ordered.push_back(newguy);

        return true;
    }

    // new strings
    std::shared_ptr<pq> s1 ( new pq(vacuum) );
    std::shared_ptr<pq> s2 ( new pq(vacuum) );

    // copy data common to both new strings
    s1->shallow_copy((void*)this);
    s2->shallow_copy((void*)this);

    // rearrange operators
    for (int i = 0; i < (int)symbol.size()-1; i++) {

        bool swap = ( !is_dagger[i] && is_dagger[i+1] );

        if ( swap ) {

            s1->delta1.push_back(symbol[i]);
            s1->delta2.push_back(symbol[i+1]);

            s2->sign = -s2->sign;
            s2->symbol.push_back(symbol[i+1]);
            s2->symbol.push_back(symbol[i]);
            s2->is_dagger.push_back(is_dagger[i+1]);
            s2->is_dagger.push_back(is_dagger[i]);

            for (size_t j = i+2; j < symbol.size(); j++) {

                s1->symbol.push_back(symbol[j]);
                s2->symbol.push_back(symbol[j]);

                s1->is_dagger.push_back(is_dagger[j]);
                s2->is_dagger.push_back(is_dagger[j]);

            }
            break;

        }else {

            s1->symbol.push_back(symbol[i]);
            s2->symbol.push_back(symbol[i]);

            s1->is_dagger.push_back(is_dagger[i]);
            s2->is_dagger.push_back(is_dagger[i]);

        }
    }

    // now, s1 and s2 are closer to normal order in the fermion space
    // we should more toward normal order in the boson space, too

    if ( is_boson_normal_order() ) {

        // copy boson daggers 
        for (size_t i = 0; i < data->is_boson_dagger.size(); i++) {
            s1->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
            s2->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
        }
        //s1->normal_order_true_vacuum(ordered);
        //s2->normal_order_true_vacuum(ordered);
        ordered.push_back(s1);
        ordered.push_back(s2);
        return false;

    }else {

        // new strings
        std::shared_ptr<pq> s1a ( new pq(vacuum) );
        std::shared_ptr<pq> s1b ( new pq(vacuum) );
        std::shared_ptr<pq> s2a ( new pq(vacuum) );
        std::shared_ptr<pq> s2b ( new pq(vacuum) );

        // copy data common to new strings
        s1a->copy((void*)s1.get());
        s1b->copy((void*)s1.get());

        // ensure boson daggers are clear (they should be anyway)
        s1a->data->is_boson_dagger.clear();
        s1b->data->is_boson_dagger.clear();

        for (int i = 0; i < (int)data->is_boson_dagger.size()-1; i++) {

            // swap operators?
            bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

            if ( swap ) {

                // nothing happens to s1a. add swapped operators to s1b
                s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                // push remaining operators onto s1a and s1b
                for (size_t j = i+2; j < data->is_boson_dagger.size(); j++) {

                    s1a->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);

                }
                break;

            }else {

                s1a->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

            }
        }

        // copy data common to new strings
        s2a->copy((void*)s2.get());
        s2b->copy((void*)s2.get());

        // ensure boson daggers are clear (they should be anyway)
        s2a->data->is_boson_dagger.clear();
        s2b->data->is_boson_dagger.clear();

        for (int i = 0; i < (int)data->is_boson_dagger.size()-1; i++) {

            // swap operators?
            bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

            if ( swap ) {

                // nothing happens to s2a. add swapped operators to s2b
                s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                // push remaining operators onto s2a and s2b
                for (size_t j = i+2; j < data->is_boson_dagger.size(); j++) {

                    s2a->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);
                    s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);

                }
                break;

            }else {

                s2a->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

            }
        }

        //s1a->normal_order_true_vacuum(ordered);
        //s1b->normal_order_true_vacuum(ordered);
        //s2a->normal_order_true_vacuum(ordered);
        //s2b->normal_order_true_vacuum(ordered);
        ordered.push_back(s1a);
        ordered.push_back(s1b);
        ordered.push_back(s2a);
        ordered.push_back(s2b);
        return false;

    }

    return false;
}

bool pq::normal_order_fermi_vacuum(std::vector<std::shared_ptr<pq> > &ordered) {

    if ( skip ) return true;

    if ( is_normal_order() ) {

        // push current ordered operator onto running list
        std::shared_ptr<pq> newguy (new pq(vacuum));

        newguy->copy((void*)this);

        ordered.push_back(newguy);

        return true;
    }

    // new strings
    std::shared_ptr<pq> s1 ( new pq(vacuum) );
    std::shared_ptr<pq> s2 ( new pq(vacuum) );

    // copy data common to both new strings
    s1->shallow_copy((void*)this);
    s2->shallow_copy((void*)this);

    // rearrange operators

    int n_new_strings = 1;

    for (int i = 0; i < (int)symbol.size()-1; i++) {

        bool swap = ( !is_dagger_fermi[i] && is_dagger_fermi[i+1] );

        // four cases: **, --, *-, -*
        // **, --: change sign, swap labels
        // *-, -*: standard swap

        bool daggers_differ = ( is_dagger[i] != is_dagger[i+1] );

        if ( swap && daggers_differ ) {

            // we're going to have two new strings
            n_new_strings = 2;

            s1->delta1.push_back(symbol[i]);
            s1->delta2.push_back(symbol[i+1]);

            s2->sign = -s2->sign;
            s2->symbol.push_back(symbol[i+1]);
            s2->symbol.push_back(symbol[i]);
            s2->is_dagger.push_back(is_dagger[i+1]);
            s2->is_dagger.push_back(is_dagger[i]);
            s2->is_dagger_fermi.push_back(is_dagger_fermi[i+1]);
            s2->is_dagger_fermi.push_back(is_dagger_fermi[i]);

            for (size_t j = i+2; j < symbol.size(); j++) {

                s1->symbol.push_back(symbol[j]);
                s2->symbol.push_back(symbol[j]);

                s1->is_dagger.push_back(is_dagger[j]);
                s2->is_dagger.push_back(is_dagger[j]);

                s1->is_dagger_fermi.push_back(is_dagger_fermi[j]);
                s2->is_dagger_fermi.push_back(is_dagger_fermi[j]);

            }
            break;

        }else if ( swap && !daggers_differ )  {

            // we're only going to have one new string, with a different sign
            n_new_strings = 1;

            s1->sign = -s1->sign;
            s1->symbol.push_back(symbol[i+1]);
            s1->symbol.push_back(symbol[i]);
            s1->is_dagger.push_back(is_dagger[i+1]);
            s1->is_dagger.push_back(is_dagger[i]);
            s1->is_dagger_fermi.push_back(is_dagger_fermi[i+1]);
            s1->is_dagger_fermi.push_back(is_dagger_fermi[i]);

            for (size_t j = i+2; j < symbol.size(); j++) {

                s1->symbol.push_back(symbol[j]);

                s1->is_dagger.push_back(is_dagger[j]);

                s1->is_dagger_fermi.push_back(is_dagger_fermi[j]);

            }
            break;

        }else {

            s1->symbol.push_back(symbol[i]);
            s2->symbol.push_back(symbol[i]);

            s1->is_dagger.push_back(is_dagger[i]);
            s2->is_dagger.push_back(is_dagger[i]);

            s1->is_dagger_fermi.push_back(is_dagger_fermi[i]);
            s2->is_dagger_fermi.push_back(is_dagger_fermi[i]);

        }
    }

    // now, s1 (and s2) are closer to normal order in the fermion space
    // we should more toward normal order in the boson space, too

    if ( n_new_strings == 1 ) {

        if ( is_boson_normal_order() ) {
            if ( !skip ) {
                // copy boson daggers
                for (size_t i = 0; i < data->is_boson_dagger.size(); i++) {
                    s1->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                }
                //s1->normal_order_fermi_vacuum(ordered);
                ordered.push_back(s1);
                return false;
            }
        }else {

            // new strings
            std::shared_ptr<pq> s1a ( new pq(vacuum) );
            std::shared_ptr<pq> s1b ( new pq(vacuum) );

            // copy data common to both new strings
            s1a->copy((void*)s1.get());
            s1b->copy((void*)s1.get());

            // ensure boson daggers are clear (they should be anyway)
            s1a->data->is_boson_dagger.clear();
            s1b->data->is_boson_dagger.clear();

            for (int i = 0; i < (int)data->is_boson_dagger.size()-1; i++) {

                // swap operators?
                bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

                if ( swap ) {

                    // nothing happens to s1a. add swapped operators to s1b
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                    // push remaining operators onto s1a and s1b
                    for (size_t j = i+2; j < data->is_boson_dagger.size(); j++) {
        
                        s1a->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);
                        s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);
        
                    }
                    break;

                }else {

                    s1a->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                }
            }
            //s1a->normal_order_fermi_vacuum(ordered);
            //s1b->normal_order_fermi_vacuum(ordered);
            ordered.push_back(s1a);
            ordered.push_back(s1b);
            return false;
        }

    }else if ( n_new_strings == 2 ) {

        if ( is_boson_normal_order() ) {
            if ( !skip ) {
                // copy boson daggers
                for (size_t i = 0; i < data->is_boson_dagger.size(); i++) {
                    s1->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                    s2->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                }
                //s1->normal_order_fermi_vacuum(ordered);
                //s2->normal_order_fermi_vacuum(ordered);
                ordered.push_back(s1);
                ordered.push_back(s2);
                return false;
            }
        }else {

            // new strings
            std::shared_ptr<pq> s1a ( new pq(vacuum) );
            std::shared_ptr<pq> s1b ( new pq(vacuum) );
            std::shared_ptr<pq> s2a ( new pq(vacuum) );
            std::shared_ptr<pq> s2b ( new pq(vacuum) );

            // copy data common to new strings
            s1a->copy((void*)s1.get());
            s1b->copy((void*)s1.get());

            // ensure boson daggers are clear (they should be anyway)
            s1a->data->is_boson_dagger.clear();
            s1b->data->is_boson_dagger.clear();

            for (int i = 0; i < (int)data->is_boson_dagger.size()-1; i++) {

                // swap operators?
                bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

                if ( swap ) {

                    // nothing happens to s1a. add swapped operators to s1b
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                    // push remaining operators onto s1a and s1b
                    for (size_t j = i+2; j < data->is_boson_dagger.size(); j++) {

                        s1a->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);
                        s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);

                    }
                    break;

                }else {

                    s1a->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                }
            }

            // copy data common to new strings
            s2a->copy((void*)s2.get());
            s2b->copy((void*)s2.get());

            // ensure boson daggers are clear (they should be anyway)
            s2a->data->is_boson_dagger.clear();
            s2b->data->is_boson_dagger.clear();

            for (int i = 0; i < data->is_boson_dagger.size()-1; i++) {

                // swap operators?
                bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

                if ( swap ) {

                    // nothing happens to s2a. add swapped operators to s2b
                    s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                    s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                    // push remaining operators onto s2a and s2b
                    for (size_t j = i+2; j < data->is_boson_dagger.size(); j++) {

                        s2a->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);
                        s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[j]);

                    }
                    break;

                }else {

                    s2a->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);
                    s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                }
            }

            //s1a->normal_order_fermi_vacuum(ordered);
            //s1b->normal_order_fermi_vacuum(ordered);
            //s2a->normal_order_fermi_vacuum(ordered);
            //s2b->normal_order_fermi_vacuum(ordered);
            ordered.push_back(s1a);
            ordered.push_back(s1b);
            ordered.push_back(s2a);
            ordered.push_back(s2b);
            return false;

        }

    }
    return false;

}

bool pq::normal_order(std::vector<std::shared_ptr<pq> > &ordered) {
    if ( vacuum == "TRUE" ) {
        return normal_order_true_vacuum(ordered);
    }else {
        return normal_order_fermi_vacuum(ordered);
    }
}

// re-classify fluctuation potential terms
void pq::reclassify_tensors() {

    if ( data->tensor_type == "OCC_REPULSION") {

        // pick summation label not included in string already
        std::vector<std::string> occ_out{"i","j","k","l","m","n","o","i0","i1","i2","i3","i4","i5","i6","i7","i8","i9"};
        std::string idx;

        int skip = -999;

        for (size_t i = 0; i < occ_out.size(); i++) {
            if ( index_in_anywhere(occ_out[i]) == 0 ) {
                idx = occ_out[i];
                skip = i;
                break;
            }
        }
        if ( skip == -999 ) {
            printf("\n");
            printf("    uh oh. no suitable summation index could be found.\n");
            printf("\n");
            exit(1);
        }

        std::string idx1 = data->tensor[0];
        std::string idx2 = data->tensor[1];

        data->tensor.clear();

        data->tensor.push_back(idx1);
        data->tensor.push_back(idx);
        data->tensor.push_back(idx2);
        data->tensor.push_back(idx);

        data->tensor_type = "ERI";

    }

}

} // End namespaces

