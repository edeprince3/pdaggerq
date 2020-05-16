//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: ahat.cc
// Copyright (C) 2020 A. Eugene DePrince III
//
// Author: A. Eugene DePrince III <adeprince@fsu.edu>
// Maintainer: DePrince group
//
// This file is part of the pdaggerq package.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#include<memory>
#include<vector>
#include<iostream>
#include<string>
#include<algorithm>
#include <math.h>

#include "ahat.h"

namespace pdaggerq {

ahat::ahat(std::string vacuum_type) {

  vacuum = vacuum_type;
  skip = false;
  data = (std::shared_ptr<StringData>)(new StringData());

}

ahat::~ahat() {
}

bool ahat::is_occ(std::string idx) {
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
    }else if ( idx.at(0) == 'O' || idx.at(0) == 'o') {
        return true;
    }else if ( idx.at(0) == 'I' || idx.at(0) == 'i') {
        return true;
    }
    return false;
}

bool ahat::is_vir(std::string idx) {
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
    }else if ( idx.at(0) == 'V' || idx.at(0) == 'v') {
        return true;
    }else if ( idx.at(0) == 'A' || idx.at(0) == 'a') {
        return true;
    }
    return false;
}

void ahat::check_occ_vir() {

   // OCC: I,J,K,L,M,N
   // VIR: A,B,C,D,E,F
   // GEN: P,Q,R,S,T,U,V,W

   for (int i = 0; i < (int)delta1.size(); i++ ) {
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

void ahat::check_spin() {

    printf("\n");
    printf("    error: spin is no longer supported\n");
    printf("\n");
    exit(1);

    // check A/B in delta functions
    for (int j = 0; j < (int)delta1.size(); j++) {
        if ( delta1[j].length() == 2 ) {
            if ( delta1[j].at(1) == 'A' && delta2[j].at(1) == 'B' ) {
                skip = true;
                break;
            }else if ( delta1[j].at(1) == 'B' && delta2[j].at(1) == 'A' ) {
                skip = true;
                break;
            }
        }
    }

    // check A/B in two-index data->tensors
    if ( (int)data->tensor.size() == 2 ) {
        if ( data->tensor[0].length() == 2 ) {
            if ( data->tensor[1].length() == 2 ) {

                if ( data->tensor[0].at(1) == 'A' && data->tensor[1].at(1) == 'B' ) {
                    skip = true;
                    return;
                }else if ( data->tensor[0].at(1) == 'B' && data->tensor[1].at(1) == 'A' ) {
                    skip = true;
                    return;
                }
            
            }
        }
    }

    // check A/B in four-index data->tensors
    if ( (int)data->tensor.size() == 4 ) {
        // check bra
        if ( data->tensor[0].length() == 2 ) {
            if ( data->tensor[1].length() == 2 ) {

                if ( data->tensor[0].at(1) == 'A' && data->tensor[1].at(1) == 'B' ) {
                    skip = true;
                    return;
                }else if ( data->tensor[0].at(1) == 'B' && data->tensor[1].at(1) == 'A' ) {
                    skip = true;
                    return;
                }
            
            }
        }
        // check ket
        if ( data->tensor[2].length() == 2 ) {
            if ( data->tensor[3].length() == 2 ) {

                if ( data->tensor[2].at(1) == 'A' && data->tensor[3].at(1) == 'B' ) {
                    skip = true;
                    return;
                }else if ( data->tensor[2].at(1) == 'B' && data->tensor[3].at(1) == 'A' ) {
                    skip = true;
                    return;
                }
            
            }
        }

    }


}

void ahat::print() {
    if ( skip ) return;

    if ( vacuum == "FERMI" && (int)symbol.size() > 0 ) {
        // check if stings should be zero or not
        bool is_dagger_right = is_dagger_fermi[(int)symbol.size() - 1];
        bool is_dagger_left  = is_dagger_fermi[0];
        if ( !is_dagger_right || is_dagger_left ) {
            //return;
        }
    }

    //for (int i = 0; i < (int)symbol.size(); i++) {
    //    printf("%5i\n",(int)is_dagger_fermi[i]);
    //}

    printf("    ");
    printf("//     ");
    printf("%c", sign > 0 ? '+' : '-');
    printf(" ");
    printf("%7.5lf", fabs(data->factor));
    printf(" ");
    for (int i = 0; i < (int)symbol.size(); i++) {
        printf("%s",symbol[i].c_str());
        if ( is_dagger[i] ) {
            printf("%c",'*');
        }
        printf(" ");
    }
    for (int i = 0; i < (int)delta1.size(); i++) {
        printf("d(%s%s)",delta1[i].c_str(),delta2[i].c_str());
        printf(" ");
    }
    if ( (int)data->tensor.size() > 0 ) {
        // two-electron integrals
        if ( (int)data->tensor.size() == 4 ) {
            // mulliken
            //printf("g(");
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
            //printf(")");
        }
        // one-electron integrals
        if ( (int)data->tensor.size() == 2 ) {
            printf("h(");
            printf("%s",data->tensor[0].c_str());
            printf(",");
            printf("%s",data->tensor[1].c_str());
            printf(")");
        }
        printf(" ");
    }

    // left-hand amplitudes
    if ( (int)data->left_amplitudes.size() > 0 ) {
        for (int i = 0; i < (int)data->left_amplitudes.size(); i++) {
           
            if ( (int)data->left_amplitudes[i].size() > 0 ) {
                // l1
                if ( (int)data->left_amplitudes[i].size() == 2 ) {
                    printf("l1(");
                    printf("%s",data->left_amplitudes[i][0].c_str());
                    printf(",");
                    printf("%s",data->left_amplitudes[i][1].c_str());
                    printf(")");
                }
                // l2
                if ( (int)data->left_amplitudes[i].size() == 4 ) {
                    printf("l2(");
                    printf("%s",data->left_amplitudes[i][0].c_str());
                    printf(",");
                    printf("%s",data->left_amplitudes[i][1].c_str());
                    printf(",");
                    printf("%s",data->left_amplitudes[i][2].c_str());
                    printf(",");
                    printf("%s",data->left_amplitudes[i][3].c_str());
                    printf(")");
                }
                printf(" ");
            } 
        }
    }

    // t_amplitudes
    if ( (int)data->t_amplitudes.size() > 0 ) {
        for (int i = 0; i < (int)data->t_amplitudes.size(); i++) {
           
            if ( (int)data->t_amplitudes[i].size() > 0 ) {
                // t1
                if ( (int)data->t_amplitudes[i].size() == 2 ) {
                    printf("t1(");
                    printf("%s",data->t_amplitudes[i][0].c_str());
                    printf(",");
                    printf("%s",data->t_amplitudes[i][1].c_str());
                    printf(")");
                }
                // t2
                if ( (int)data->t_amplitudes[i].size() == 4 ) {
                    printf("t2(");
                    printf("%s",data->t_amplitudes[i][0].c_str());
                    printf(",");
                    printf("%s",data->t_amplitudes[i][1].c_str());
                    printf(",");
                    printf("%s",data->t_amplitudes[i][2].c_str());
                    printf(",");
                    printf("%s",data->t_amplitudes[i][3].c_str());
                    printf(")");
                }
                printf(" ");
            } 
        }
    }
    printf("\n");
}

bool ahat::is_normal_order() {

    // don't bother bringing to normal order if we're going to skip this string
    if (skip) return true;

    if ( vacuum == "TRUE" ) {
        for (int i = 0; i < (int)symbol.size()-1; i++) {
            if ( !is_dagger[i] && is_dagger[i+1] ) {
                return false;
            }
        }
    }else {
        for (int i = 0; i < (int)symbol.size()-1; i++) {
            // check if stings should be zero or not
            bool is_dagger_right = is_dagger_fermi[(int)symbol.size() - 1];
            bool is_dagger_left  = is_dagger_fermi[0];
            if ( !is_dagger_right || is_dagger_left ) {
                return true;
            }
            if ( !is_dagger_fermi[i] && is_dagger_fermi[i+1] ) {
                return false;
            }
        }
    }
    return true;
}

// in order to compare strings, the creation and annihilation 
// operators should be ordered in some consistent way.
// alphabetically seems reasonable enough
void ahat::alphabetize(std::vector<std::shared_ptr<ahat> > &ordered) {

    // alphabetize string
    for (int i = 0; i < (int)ordered.size(); i++) {

        // creation
        bool not_alphabetized = false;
        do {
            not_alphabetized = false;
            int ndagger = 0;
            for (int j = 0; j < (int)ordered[i]->symbol.size(); j++) {
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
                    j = (int)ordered[i]->symbol.size() + 1;
                    not_alphabetized = true;
                }
            }
        }while(not_alphabetized);
        // annihilation
        not_alphabetized = false;
        do {
            not_alphabetized = false;
            int ndagger = 0;
            for (int j = 0; j < (int)ordered[i]->symbol.size(); j++) {
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
                    j = (int)ordered[i]->symbol.size() + 1;
                    not_alphabetized = true;
                }
            }
        }while(not_alphabetized);
    }

    // alphabetize deltas
    for (int i = 0; i < (int)ordered.size(); i++) {
        for (int j = 0; j < (int)ordered[i]->delta1.size(); j++) {
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



// move all bra lables to the right-most positions in t_amplitudes and
// four-index tensors. only works for four-index amplitudes (i.e., t2)

// TODO: account for left-hand amplitudes

void ahat::update_bra_labels() {

    if ( vacuum == "FERMI" && symbol.size() != 0 ) return;

    if ( skip ) return;

    // t_amplitudes
    bool find_m = index_in_t_amplitudes("m");
    bool find_n = index_in_t_amplitudes("n");
    bool find_e = index_in_t_amplitudes("e");
    bool find_f = index_in_t_amplitudes("f");
    
    for (int i = 0; i < data->t_amplitudes.size(); i++) {

        if ( data->t_amplitudes[i].size() != 4 ) continue;

        if ( find_m && find_n ) {
            // should appear as "mn"
            if ( data->t_amplitudes[i][2] == "n" && data->t_amplitudes[i][3] == "m" ) {
                data->t_amplitudes[i][2] = "m";
                data->t_amplitudes[i][3] = "n";
                sign = -sign;
            }
        }else if ( find_m ) {
            // should appear as "-m"
            if ( data->t_amplitudes[i][2] == "m" ) {
                data->t_amplitudes[i][2] = data->t_amplitudes[i][3];
                data->t_amplitudes[i][3] = "m";
                sign = -sign;
            }
        }else if ( find_n) {
            // should appear as "-n"
            if ( data->t_amplitudes[i][2] == "n" ) {
                data->t_amplitudes[i][2] = data->t_amplitudes[i][3];
                data->t_amplitudes[i][3] = "n";
                sign = -sign;
            }
        }

        if ( find_e && find_f ) {
            // should appear as "mn"
            if ( data->t_amplitudes[i][0] == "f" && data->t_amplitudes[i][1] == "e" ) {
                data->t_amplitudes[i][0] = "e";
                data->t_amplitudes[i][1] = "f";
                sign = -sign;
            }
        }else if ( find_e ) {
            // should appear as "-e"
            if ( data->t_amplitudes[i][0] == "e" ) {
                data->t_amplitudes[i][0] = data->t_amplitudes[i][1];
                data->t_amplitudes[i][1] = "e";
                sign = -sign;
            }
        }else if ( find_f) {
            // should appear as "-f"
            if ( data->t_amplitudes[i][0] == "f" ) {
                data->t_amplitudes[i][0] = data->t_amplitudes[i][1];
                data->t_amplitudes[i][1] = "f";
                sign = -sign;
            }
        }

    }

    // tensor
    if ( data->tensor.size() != 4 ) return;

    find_m = index_in_tensor("m");
    find_n = index_in_tensor("n");
    find_e = index_in_tensor("e");
    find_f = index_in_tensor("f");
    
    if ( find_m && find_n ) {
        // should appear as "mn"
        if ( data->tensor[2] == "n" && data->tensor[3] == "m" ) {
            data->tensor[2] = "m";
            data->tensor[3] = "n";
            sign = -sign;
        }
    }else if ( find_m ) {
        // should appear as "-m"
        if ( data->tensor[2] == "m" ) {
            data->tensor[2] = data->tensor[3];
            data->tensor[3] = "m";
            sign = -sign;
        }
    }else if ( find_n) {
        // should appear as "-n"
        if ( data->tensor[2] == "n" ) {
            data->tensor[2] = data->tensor[3];
            data->tensor[3] = "n";
            sign = -sign;
        }
    }

    if ( find_e && find_f ) {
        // should appear as "mn"
        if ( data->tensor[0] == "f" && data->tensor[1] == "e" ) {
            data->tensor[0] = "e";
            data->tensor[1] = "f";
            sign = -sign;
        }
    }else if ( find_e ) {
        // should appear as "-e"
        if ( data->tensor[0] == "e" ) {
            data->tensor[0] = data->tensor[1];
            data->tensor[1] = "e";
            sign = -sign;
        }
    }else if ( find_f) {
        // should appear as "-f"
        if ( data->tensor[0] == "f" ) {
            data->tensor[0] = data->tensor[1];
            data->tensor[1] = "f";
            sign = -sign;
        }
    }

}

// prioritize summation labels as i > j > k > l and a > b > c > d.
// this means that j, k, or l should not arise in a term if i is not
// already present.
void ahat::update_summation_labels() {

    if ( vacuum == "FERMI" && symbol.size() != 0 ) return;

    if ( skip ) return;

    bool find_i = index_in_tensor("i") || index_in_t_amplitudes("i") || index_in_left_amplitudes("i");
    bool find_j = index_in_tensor("j") || index_in_t_amplitudes("j") || index_in_left_amplitudes("j");
    bool find_k = index_in_tensor("k") || index_in_t_amplitudes("k") || index_in_left_amplitudes("k");
    bool find_l = index_in_tensor("l") || index_in_t_amplitudes("l") || index_in_left_amplitudes("l");

    bool find_a = index_in_tensor("a") || index_in_t_amplitudes("a") || index_in_left_amplitudes("a");
    bool find_b = index_in_tensor("b") || index_in_t_amplitudes("b") || index_in_left_amplitudes("b");
    bool find_c = index_in_tensor("c") || index_in_t_amplitudes("c") || index_in_left_amplitudes("c");
    bool find_d = index_in_tensor("d") || index_in_t_amplitudes("d") || index_in_left_amplitudes("d");

    // i,j,k,l
    if ( !find_i && find_j && find_k && find_l ) {
        replace_index_in_tensor("l","i");
        replace_index_in_tensor("l","i");
        replace_index_in_tensor("l","i");
        replace_index_in_tensor("l","i");
        replace_index_in_t_amplitudes("l","i");
        replace_index_in_t_amplitudes("l","i");
        replace_index_in_t_amplitudes("l","i");
        replace_index_in_t_amplitudes("l","i");
        replace_index_in_left_amplitudes("l","i");
        replace_index_in_left_amplitudes("l","i");
        replace_index_in_left_amplitudes("l","i");
        replace_index_in_left_amplitudes("l","i");
    }else if ( !find_i && find_j && find_k && !find_l ) {
        replace_index_in_tensor("k","i");
        replace_index_in_tensor("k","i");
        replace_index_in_tensor("k","i");
        replace_index_in_tensor("k","i");
        replace_index_in_t_amplitudes("k","i");
        replace_index_in_t_amplitudes("k","i");
        replace_index_in_t_amplitudes("k","i");
        replace_index_in_t_amplitudes("k","i");
        replace_index_in_left_amplitudes("k","i");
        replace_index_in_left_amplitudes("k","i");
        replace_index_in_left_amplitudes("k","i");
        replace_index_in_left_amplitudes("k","i");
    }else if ( !find_i && find_j && !find_k && !find_l ) {
        replace_index_in_tensor("j","i");
        replace_index_in_tensor("j","i");
        replace_index_in_tensor("j","i");
        replace_index_in_tensor("j","i");
        replace_index_in_t_amplitudes("j","i");
        replace_index_in_t_amplitudes("j","i");
        replace_index_in_t_amplitudes("j","i");
        replace_index_in_t_amplitudes("j","i");
        replace_index_in_left_amplitudes("j","i");
        replace_index_in_left_amplitudes("j","i");
        replace_index_in_left_amplitudes("j","i");
        replace_index_in_left_amplitudes("j","i");
    }else if ( !find_i && find_j && !find_k && find_l ) {
        replace_index_in_tensor("l","i");
        replace_index_in_tensor("l","i");
        replace_index_in_tensor("l","i");
        replace_index_in_tensor("l","i");
        replace_index_in_t_amplitudes("l","i");
        replace_index_in_t_amplitudes("l","i");
        replace_index_in_t_amplitudes("l","i");
        replace_index_in_t_amplitudes("l","i");
        replace_index_in_left_amplitudes("l","i");
        replace_index_in_left_amplitudes("l","i");
        replace_index_in_left_amplitudes("l","i");
        replace_index_in_left_amplitudes("l","i");
    }else if ( !find_i && !find_j && find_k && find_l ) {
        replace_index_in_tensor("k","i");
        replace_index_in_tensor("k","i");
        replace_index_in_tensor("k","i");
        replace_index_in_tensor("k","i");
        replace_index_in_t_amplitudes("k","i");
        replace_index_in_t_amplitudes("k","i");
        replace_index_in_t_amplitudes("k","i");
        replace_index_in_t_amplitudes("k","i");
        replace_index_in_left_amplitudes("k","i");
        replace_index_in_left_amplitudes("k","i");
        replace_index_in_left_amplitudes("k","i");
        replace_index_in_left_amplitudes("k","i");
        replace_index_in_tensor("l","j");
        replace_index_in_tensor("l","j");
        replace_index_in_tensor("l","j");
        replace_index_in_tensor("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
    }else if ( !find_i && !find_j && !find_k && find_l ) {
        replace_index_in_tensor("l","i");
        replace_index_in_tensor("l","i");
        replace_index_in_tensor("l","i");
        replace_index_in_tensor("l","i");
        replace_index_in_t_amplitudes("l","i");
        replace_index_in_t_amplitudes("l","i");
        replace_index_in_t_amplitudes("l","i");
        replace_index_in_t_amplitudes("l","i");
        replace_index_in_left_amplitudes("l","i");
        replace_index_in_left_amplitudes("l","i");
        replace_index_in_left_amplitudes("l","i");
        replace_index_in_left_amplitudes("l","i");
    }else if ( !find_i && !find_j && find_k && !find_l ) {
        replace_index_in_tensor("k","i");
        replace_index_in_tensor("k","i");
        replace_index_in_tensor("k","i");
        replace_index_in_tensor("k","i");
        replace_index_in_t_amplitudes("k","i");
        replace_index_in_t_amplitudes("k","i");
        replace_index_in_t_amplitudes("k","i");
        replace_index_in_t_amplitudes("k","i");
        replace_index_in_left_amplitudes("k","i");
        replace_index_in_left_amplitudes("k","i");
        replace_index_in_left_amplitudes("k","i");
        replace_index_in_left_amplitudes("k","i");
    }else if ( find_i && !find_j && find_k && find_l ) {
        replace_index_in_tensor("l","j");
        replace_index_in_tensor("l","j");
        replace_index_in_tensor("l","j");
        replace_index_in_tensor("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
    }else if ( find_i && !find_j && !find_k && find_l ) {
        replace_index_in_tensor("l","j");
        replace_index_in_tensor("l","j");
        replace_index_in_tensor("l","j");
        replace_index_in_tensor("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
    }else if ( find_i && !find_j && find_k && !find_l ) {
        replace_index_in_tensor("k","j");
        replace_index_in_tensor("k","j");
        replace_index_in_tensor("k","j");
        replace_index_in_tensor("k","j");
        replace_index_in_t_amplitudes("k","j");
        replace_index_in_t_amplitudes("k","j");
        replace_index_in_t_amplitudes("k","j");
        replace_index_in_t_amplitudes("k","j");
        replace_index_in_left_amplitudes("k","j");
        replace_index_in_left_amplitudes("k","j");
        replace_index_in_left_amplitudes("k","j");
        replace_index_in_left_amplitudes("k","j");
    }else if ( find_i && !find_j && !find_k && find_l ) {
        replace_index_in_tensor("l","j");
        replace_index_in_tensor("l","j");
        replace_index_in_tensor("l","j");
        replace_index_in_tensor("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_t_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
        replace_index_in_left_amplitudes("l","j");
    }

    // a,b,c,d
    if ( !find_a && find_b && find_c && find_d ) {
        replace_index_in_tensor("d","a");
        replace_index_in_tensor("d","a");
        replace_index_in_tensor("d","a");
        replace_index_in_tensor("d","a");
        replace_index_in_t_amplitudes("d","a");
        replace_index_in_t_amplitudes("d","a");
        replace_index_in_t_amplitudes("d","a");
        replace_index_in_t_amplitudes("d","a");
        replace_index_in_left_amplitudes("d","a");
        replace_index_in_left_amplitudes("d","a");
        replace_index_in_left_amplitudes("d","a");
        replace_index_in_left_amplitudes("d","a");
    }else if ( !find_a && find_b && find_c && !find_d ) {
        replace_index_in_tensor("c","a");
        replace_index_in_tensor("c","a");
        replace_index_in_tensor("c","a");
        replace_index_in_tensor("c","a");
        replace_index_in_t_amplitudes("c","a");
        replace_index_in_t_amplitudes("c","a");
        replace_index_in_t_amplitudes("c","a");
        replace_index_in_t_amplitudes("c","a");
        replace_index_in_left_amplitudes("c","a");
        replace_index_in_left_amplitudes("c","a");
        replace_index_in_left_amplitudes("c","a");
        replace_index_in_left_amplitudes("c","a");
    }else if ( !find_a && find_b && !find_c && find_d ) {
        replace_index_in_tensor("d","a");
        replace_index_in_tensor("d","a");
        replace_index_in_tensor("d","a");
        replace_index_in_tensor("d","a");
        replace_index_in_t_amplitudes("d","a");
        replace_index_in_t_amplitudes("d","a");
        replace_index_in_t_amplitudes("d","a");
        replace_index_in_t_amplitudes("d","a");
        replace_index_in_left_amplitudes("d","a");
        replace_index_in_left_amplitudes("d","a");
        replace_index_in_left_amplitudes("d","a");
        replace_index_in_left_amplitudes("d","a");
    }else if ( !find_a && find_b && !find_c && !find_d ) {
        replace_index_in_tensor("b","a");
        replace_index_in_tensor("b","a");
        replace_index_in_tensor("b","a");
        replace_index_in_tensor("b","a");
        replace_index_in_t_amplitudes("b","a");
        replace_index_in_t_amplitudes("b","a");
        replace_index_in_t_amplitudes("b","a");
        replace_index_in_t_amplitudes("b","a");
        replace_index_in_left_amplitudes("b","a");
        replace_index_in_left_amplitudes("b","a");
        replace_index_in_left_amplitudes("b","a");
        replace_index_in_left_amplitudes("b","a");
    }else if ( !find_a && !find_b && find_c && find_d ) {
        replace_index_in_tensor("c","a");
        replace_index_in_tensor("c","a");
        replace_index_in_tensor("c","a");
        replace_index_in_tensor("c","a");
        replace_index_in_t_amplitudes("c","a");
        replace_index_in_t_amplitudes("c","a");
        replace_index_in_t_amplitudes("c","a");
        replace_index_in_t_amplitudes("c","a");
        replace_index_in_left_amplitudes("c","a");
        replace_index_in_left_amplitudes("c","a");
        replace_index_in_left_amplitudes("c","a");
        replace_index_in_left_amplitudes("c","a");
        replace_index_in_tensor("d","b");
        replace_index_in_tensor("d","b");
        replace_index_in_tensor("d","b");
        replace_index_in_tensor("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
    }else if ( !find_a && !find_b && !find_c && find_d ) {
        replace_index_in_tensor("d","a");
        replace_index_in_tensor("d","a");
        replace_index_in_tensor("d","a");
        replace_index_in_tensor("d","a");
        replace_index_in_t_amplitudes("d","a");
        replace_index_in_t_amplitudes("d","a");
        replace_index_in_t_amplitudes("d","a");
        replace_index_in_t_amplitudes("d","a");
        replace_index_in_left_amplitudes("d","a");
        replace_index_in_left_amplitudes("d","a");
        replace_index_in_left_amplitudes("d","a");
        replace_index_in_left_amplitudes("d","a");
    }else if ( !find_a && !find_b && find_c && !find_d ) {
        replace_index_in_tensor("c","a");
        replace_index_in_tensor("c","a");
        replace_index_in_tensor("c","a");
        replace_index_in_tensor("c","a");
        replace_index_in_t_amplitudes("c","a");
        replace_index_in_t_amplitudes("c","a");
        replace_index_in_t_amplitudes("c","a");
        replace_index_in_t_amplitudes("c","a");
        replace_index_in_left_amplitudes("c","a");
        replace_index_in_left_amplitudes("c","a");
        replace_index_in_left_amplitudes("c","a");
        replace_index_in_left_amplitudes("c","a");
    }else if ( find_a && !find_b && find_c && find_d ) {
        replace_index_in_tensor("d","b");
        replace_index_in_tensor("d","b");
        replace_index_in_tensor("d","b");
        replace_index_in_tensor("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
    }else if ( find_a && !find_b && !find_c && find_d ) {
        replace_index_in_tensor("d","b");
        replace_index_in_tensor("d","b");
        replace_index_in_tensor("d","b");
        replace_index_in_tensor("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
    }else if ( find_a && !find_b && find_c && !find_d ) {
        replace_index_in_tensor("c","b");
        replace_index_in_tensor("c","b");
        replace_index_in_tensor("c","b");
        replace_index_in_tensor("c","b");
        replace_index_in_t_amplitudes("c","b");
        replace_index_in_t_amplitudes("c","b");
        replace_index_in_t_amplitudes("c","b");
        replace_index_in_t_amplitudes("c","b");
        replace_index_in_left_amplitudes("c","b");
        replace_index_in_left_amplitudes("c","b");
        replace_index_in_left_amplitudes("c","b");
        replace_index_in_left_amplitudes("c","b");
    }else if ( find_a && !find_b && !find_c && find_d ) {
        replace_index_in_tensor("d","b");
        replace_index_in_tensor("d","b");
        replace_index_in_tensor("d","b");
        replace_index_in_tensor("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_t_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
        replace_index_in_left_amplitudes("d","b");
    }

    // now, if tensors appear as <ji||xx>, swap to -<ij|xx>, <ai||xx> = -<ia|xx>, ec.

    if ( data->tensor.size() == 4 ) {
        if ( data->tensor[0] == "j" && data->tensor[1] == "i" ) {
            data->tensor[0] = "i";
            data->tensor[1] = "j";
            sign = -sign;
        }
        if ( data->tensor[2] == "j" && data->tensor[3] == "i" ) {
            data->tensor[2] = "i";
            data->tensor[3] = "j";
            sign = -sign;
        }


        if ( data->tensor[0] == "b" && data->tensor[1] == "a" ) {
            data->tensor[0] = "a";
            data->tensor[1] = "b";
            sign = -sign;
        }
        if ( data->tensor[2] == "b" && data->tensor[3] == "a" ) {
            data->tensor[2] = "a";
            data->tensor[3] = "b";
            sign = -sign;
        }


        if ( data->tensor[0] == "a" && data->tensor[1] == "i" ) {
            data->tensor[0] = "i";
            data->tensor[1] = "a";
            sign = -sign;
        }
        if ( data->tensor[2] == "a" && data->tensor[3] == "i" ) {
            data->tensor[2] = "i";
            data->tensor[3] = "a";
            sign = -sign;
        }


        if ( data->tensor[0] == "b" && data->tensor[1] == "i" ) {
            data->tensor[0] = "i";
            data->tensor[1] = "b";
            sign = -sign;
        }
        if ( data->tensor[2] == "b" && data->tensor[3] == "i" ) {
            data->tensor[2] = "i";
            data->tensor[3] = "b";
            sign = -sign;
        }


        if ( data->tensor[0] == "a" && data->tensor[1] == "j" ) {
            data->tensor[0] = "j";
            data->tensor[1] = "a";
            sign = -sign;
        }
        if ( data->tensor[2] == "a" && data->tensor[3] == "j" ) {
            data->tensor[2] = "j";
            data->tensor[3] = "a";
            sign = -sign;
        }


        if ( data->tensor[0] == "b" && data->tensor[1] == "j" ) {
            data->tensor[0] = "j";
            data->tensor[1] = "b";
            sign = -sign;
        }
        if ( data->tensor[2] == "b" && data->tensor[3] == "j" ) {
            data->tensor[2] = "j";
            data->tensor[3] = "b";
            sign = -sign;
        }
    }

    update_bra_labels();

    // if labels are repeated in a four-index tensor, then they should be paired: <ij||jm> -> -<ij|mj>
    if ( data->tensor.size() == 4 ) {
        if ( data->tensor[0] == data->tensor[3] ) {
            std::string tmp = data->tensor[3];
            data->tensor[3] = data->tensor[2];
            data->tensor[2] = tmp;
            sign = -sign;
        }else if ( data->tensor[1] == data->tensor[2] ) {
            std::string tmp = data->tensor[2];
            data->tensor[2] = data->tensor[3];
            data->tensor[3] = tmp;
            sign = -sign;
        }

    }

}

void ahat::swap_two_labels(std::string label1, std::string label2) {

    replace_index_in_tensor(label1,"x");
    replace_index_in_tensor(label1,"x");
    replace_index_in_tensor(label1,"x");
    replace_index_in_tensor(label1,"x");
    replace_index_in_t_amplitudes(label1,"x");
    replace_index_in_t_amplitudes(label1,"x");
    replace_index_in_t_amplitudes(label1,"x");
    replace_index_in_t_amplitudes(label1,"x");
    replace_index_in_left_amplitudes(label1,"x");
    replace_index_in_left_amplitudes(label1,"x");
    replace_index_in_left_amplitudes(label1,"x");
    replace_index_in_left_amplitudes(label1,"x");
    
    replace_index_in_tensor(label2,label1);
    replace_index_in_tensor(label2,label1);
    replace_index_in_tensor(label2,label1);
    replace_index_in_tensor(label2,label1);
    replace_index_in_t_amplitudes(label2,label1);
    replace_index_in_t_amplitudes(label2,label1);
    replace_index_in_t_amplitudes(label2,label1);
    replace_index_in_t_amplitudes(label2,label1);
    replace_index_in_left_amplitudes(label2,label1);
    replace_index_in_left_amplitudes(label2,label1);
    replace_index_in_left_amplitudes(label2,label1);
    replace_index_in_left_amplitudes(label2,label1);
    
    replace_index_in_tensor("x",label2);
    replace_index_in_tensor("x",label2);
    replace_index_in_tensor("x",label2);
    replace_index_in_tensor("x",label2);
    replace_index_in_t_amplitudes("x",label2);
    replace_index_in_t_amplitudes("x",label2);
    replace_index_in_t_amplitudes("x",label2);
    replace_index_in_t_amplitudes("x",label2);
    replace_index_in_left_amplitudes("x",label2);
    replace_index_in_left_amplitudes("x",label2);
    replace_index_in_left_amplitudes("x",label2);
    replace_index_in_left_amplitudes("x",label2);

}

// once strings are alphabetized, we can compare them
// and remove terms that cancel. 

// TODO: need to consider left-hand amplitudes
void ahat::cleanup(std::vector<std::shared_ptr<ahat> > &ordered) {

    // prioritize summation labels as i > j > k > l and a > b > c > d.
    // this means that j, k, or l should not arise in a term if i is not
    // already present. only do this for vacuum_type = "FERMI"
    for (int i = 0; i < (int)ordered.size(); i++) {
        if ( vacuum != "FERMI" ) continue;
        ordered[i]->update_summation_labels();
        //ordered[i]->update_bra_labels();
    }

    // consolidate terms, including those that differ only by symmetric quantities [i.e., g(iajb) and g(jbia)]
    for (int i = 0; i < (int)ordered.size(); i++) {

        // for normal order relative to fermi vacuum, i doubt anyone will care 
        // about terms that aren't fully contracted. so, skip those because this
        // function is time consuming

        if ( vacuum == "FERMI" && ordered[i]->symbol.size() != 0 ) continue;

        if ( ordered[i]->skip ) continue;

        for (int j = i+1; j < (int)ordered.size(); j++) {

            // for normal order relative to fermi vacuum, i doubt anyone will care 
            // about terms that aren't fully contracted. so, skip those because this
            // function is time consuming

            if ( vacuum == "FERMI" && ordered[j]->symbol.size() != 0 ) continue;

            if ( ordered[j]->skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            bool find_i = ordered[j]->index_in_tensor("i") || ordered[j]->index_in_t_amplitudes("i");
            bool find_j = ordered[j]->index_in_tensor("j") || ordered[j]->index_in_t_amplitudes("j");

            bool find_a = ordered[j]->index_in_tensor("a") || ordered[j]->index_in_t_amplitudes("a");
            bool find_b = ordered[j]->index_in_tensor("b") || ordered[j]->index_in_t_amplitudes("b");

            // try swapping summation labels - only i/j, a/b swaps for now. this should be sufficient for ccsd
            if ( !strings_same && find_i && find_j ) {

                std::shared_ptr<ahat> newguy (new ahat(vacuum));
                newguy->copy((void*)(ordered[j].get()));
                newguy->swap_two_labels("i","j");
                strings_same = compare_strings(ordered[i],newguy,n_permute);

            }

            if ( !strings_same && find_a && find_b ) {

                std::shared_ptr<ahat> newguy (new ahat(vacuum));
                newguy->copy((void*)(ordered[j].get()));
                newguy->swap_two_labels("a","b");
                strings_same = compare_strings(ordered[i],newguy,n_permute);

            }

            if ( !strings_same && find_i && find_j && find_a && find_b ) {

                std::shared_ptr<ahat> newguy (new ahat(vacuum));
                newguy->copy((void*)(ordered[j].get()));
                newguy->swap_two_labels("i","j");
                newguy->swap_two_labels("a","b");
                strings_same = compare_strings(ordered[i],newguy,n_permute);
            }

            if ( !strings_same ) continue;

            //printf("same tensors\n");

            // are factors same?
            //if ( ordered[i]->data->factor != ordered[j]->data->factor ) continue;

            // are signs same?
            //if ( ordered[i]->sign != ordered[j]->sign ) continue;

            double factor_i = ordered[i]->data->factor * ordered[i]->sign;
            double factor_j = ordered[j]->data->factor * ordered[j]->sign;

            double combined_factor = factor_i + factor_j * pow(-1.0,n_permute);

            // if terms exactly cancel, do so
            if ( fabs(combined_factor) < 1e-12 ) {
                //printf("skipping\n");
                ordered[i]->skip = true;
                ordered[j]->skip = true;
                break;
            }

            // otherwise, combine terms

            //printf("combining\n");
            // well, i guess the are the same term
            ordered[i]->data->factor = fabs(combined_factor);
            if ( combined_factor > 0.0 ) {
                ordered[i]->sign =  1;
            }else {
                ordered[i]->sign = -1;
            }
            ordered[j]->skip = true;

            // break j because i'm not yet sure the best way to combine multiple terms.
            //break;
            
        }

    }

}

bool ahat::compare_strings(std::shared_ptr<ahat> ordered_1, std::shared_ptr<ahat> ordered_2, int & n_permute) {

    n_permute = 0;

    //printf("ok, how about these\n");
    //ordered[i]->print();
    //ordered[j]->print();

    // are strings same?
    if ( ordered_1->symbol.size() != ordered_2->symbol.size() ) return false;
    int nsame_s = 0;
    for (int k = 0; k < (int)ordered_1->symbol.size(); k++) {
        if ( ordered_1->symbol[k] == ordered_2->symbol[k] ) {
            nsame_s++;
        }
    }
    if ( nsame_s != ordered_1->symbol.size() ) return false;
    //printf("same strings\n");

    // same delta functions (recall these aren't sorted in any way)
    int nsame_d = 0;
    for (int k = 0; k < (int)ordered_1->delta1.size(); k++) {
        for (int l = 0; l < (int)ordered_2->delta1.size(); l++) {
            if ( ordered_1->delta1[k] == ordered_2->delta1[l] && ordered_1->delta2[k] == ordered_2->delta2[l] ) {
                nsame_d++;
                //break;
            }else if ( ordered_1->delta2[k] == ordered_2->delta1[l] && ordered_1->delta1[k] == ordered_2->delta2[l] ) {
                nsame_d++;
                //break;
            }
        }
    }
    if ( nsame_d != (int)ordered_1->delta1.size() ) return false;
    //printf("same deltas\n");

    // t_amplitudes, which can be complicated since they aren't sorted

    // same number of t_amplitudes?
    if ( ordered_1->data->t_amplitudes.size() != ordered_2->data->t_amplitudes.size() ) return false;
    
    int nsame_amps = 0;
    for (int ii = 0; ii < (int)ordered_1->data->t_amplitudes.size(); ii++) {
        for (int jj = 0; jj < (int)ordered_2->data->t_amplitudes.size(); jj++) {

            // t1 vs t2?
            if ( ordered_1->data->t_amplitudes[ii].size() != ordered_2->data->t_amplitudes[jj].size() ) continue;

            // indices?
            int nsame_idx = 0;
            for (int iii = 0; iii < (int)ordered_1->data->t_amplitudes[ii].size(); iii++) {
                for (int jjj = 0; jjj < (int)ordered_2->data->t_amplitudes[jj].size(); jjj++) {
                    if ( ordered_1->data->t_amplitudes[ii][iii] == ordered_2->data->t_amplitudes[jj][jjj] ) {
                        if ( (iii - jjj) % 2 != 0  && iii < jjj ) n_permute++;
                        nsame_idx++;
                        break;
                    }
                }
            }
            // if all indices are the same, the t_amplitudes must be the same, but we need to be careful of permutations
            if ( nsame_idx == (int)ordered_1->data->t_amplitudes[ii].size() ) {
                nsame_amps++;
                break;
            }
        }
    }
    if ( nsame_amps != (int)ordered_1->data->t_amplitudes.size() ) return false;

    // left-hand amplitudes, which can be complicated since they aren't sorted

    // same number of left-hant amplitudes?
    if ( ordered_1->data->left_amplitudes.size() != ordered_2->data->left_amplitudes.size() ) return false;
    
    int nsame_left_amps = 0;
    for (int ii = 0; ii < (int)ordered_1->data->left_amplitudes.size(); ii++) {
        for (int jj = 0; jj < (int)ordered_2->data->left_amplitudes.size(); jj++) {

            // l1 vs l2?
            if ( ordered_1->data->left_amplitudes[ii].size() != ordered_2->data->left_amplitudes[jj].size() ) continue;

            // indices?
            int nsame_idx = 0;
            for (int iii = 0; iii < (int)ordered_1->data->left_amplitudes[ii].size(); iii++) {
                for (int jjj = 0; jjj < (int)ordered_2->data->left_amplitudes[jj].size(); jjj++) {
                    if ( ordered_1->data->left_amplitudes[ii][iii] == ordered_2->data->left_amplitudes[jj][jjj] ) {
                        if ( (iii - jjj) % 2 != 0  && iii < jjj ) n_permute++;
                        nsame_idx++;
                        break;
                    }
                }
            }
            // if all indices are the same, the left-hand amplitudes must be the same, but we need to be careful of permutations
            if ( nsame_idx == (int)ordered_1->data->left_amplitudes[ii].size() ) {
                nsame_left_amps++;
                break;
            }
        }
    }
    if ( nsame_left_amps != (int)ordered_1->data->left_amplitudes.size() ) return false;

    //printf("same left-hand amps\n");
    //if ( (n_permute % 2) != 0 ) continue;

    // are tensors same?
    if ( ordered_1->data->tensor.size() != ordered_2->data->tensor.size() ) return false;

    int nsame_t = 0;
    for (int k = 0; k < (int)ordered_1->data->tensor.size(); k++) {
        if ( ordered_1->data->tensor[k] == ordered_2->data->tensor[k] ) {
            nsame_t++;
        }
    }
    // if not the same, check bras againt kets [mulliken notation: g(iajb) = (ia|jb)]
/*
    if ( nsame_t != ordered_1->data->tensor.size() ) {

        // let's just limit ourselves to four-index tensors for now
        if ( ordered_1->data->tensor.size() == 4 ) {

            int nsame_t_swap = 0;
            if ( ordered_1->data->tensor[0] != ordered_2->data->tensor[2] ||
                 ordered_1->data->tensor[1] != ordered_2->data->tensor[3] ||
                 ordered_1->data->tensor[2] != ordered_2->data->tensor[0] ||
                 ordered_1->data->tensor[3] != ordered_2->data->tensor[1]) {
                 return false;
            }

        }
    }
*/

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

// copy all data, except symbols and daggers. 

void ahat::shallow_copy(void * copy_me) { 

    ahat * in = reinterpret_cast<ahat * >(copy_me);

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
    for (int i = 0; i < (int)in->data->tensor.size(); i++) {
        data->tensor.push_back(in->data->tensor[i]);
    }

    // delta1, delta2
    for (int i = 0; i < (int)in->delta1.size(); i++) {
        delta1.push_back(in->delta1[i]);
        delta2.push_back(in->delta2[i]);
    }

    // t_amplitudes
    for (int i = 0; i < (int)in->data->t_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)in->data->t_amplitudes[i].size(); j++) {
            tmp.push_back(in->data->t_amplitudes[i][j]);
        }
        data->t_amplitudes.push_back(tmp);
    }

    // left-hand amplitudes
    for (int i = 0; i < (int)in->data->left_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)in->data->left_amplitudes[i].size(); j++) {
            tmp.push_back(in->data->left_amplitudes[i][j]);
        }
        data->left_amplitudes.push_back(tmp);
    }

}

bool ahat::index_in_tensor(std::string idx) {

    for (int i = 0; i < (int)data->tensor.size(); i++) {
        if ( data->tensor[i] == idx ) {
            return true;
        }
    }
    return false;

}

bool ahat::index_in_t_amplitudes(std::string idx) {

    for (int i = 0; i < (int)data->t_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->t_amplitudes[i].size(); j++) {
            if ( data->t_amplitudes[i][j] == idx ) {
                return true;
            }
           
        }
    }
    return false;

}

bool ahat::index_in_left_amplitudes(std::string idx) {

    for (int i = 0; i < (int)data->left_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->left_amplitudes[i].size(); j++) {
            if ( data->left_amplitudes[i][j] == idx ) {
                return true;
            }
           
        }
    }
    return false;

}

void ahat::replace_index_in_tensor(std::string old_idx, std::string new_idx) {

    for (int i = 0; i < (int)data->tensor.size(); i++) {
        if ( data->tensor[i] == old_idx ) {
            data->tensor[i] = new_idx;
            return;
        }
    }

}

void ahat::replace_index_in_t_amplitudes(std::string old_idx, std::string new_idx) {

    for (int i = 0; i < (int)data->t_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->t_amplitudes[i].size(); j++) {
            if ( data->t_amplitudes[i][j] == old_idx ) {
                data->t_amplitudes[i][j] = new_idx;
                return; 
            }
        }
    }

}

void ahat::replace_index_in_left_amplitudes(std::string old_idx, std::string new_idx) {

    for (int i = 0; i < (int)data->left_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->left_amplitudes[i].size(); j++) {
            if ( data->left_amplitudes[i][j] == old_idx ) {
                data->left_amplitudes[i][j] = new_idx;
                return; 
            }
        }
    }

}

// find and replace any funny labels in tensors with conventional ones. i.e., t -> i ,w -> a
void ahat::use_conventional_labels() {

    // occupied first:
    //std::vector<std::string> occ_in{"o","t","u","v"};
    std::vector<std::string> occ_in{"o1","o2","o3","o4"};
    std::vector<std::string> occ_out{"i","j","k","l"};

    for (int i = 0; i < (int)occ_in.size(); i++) {

        if ( index_in_tensor(occ_in[i]) ) {

            for (int j = 0; j < (int)occ_out.size(); j++) {

                if ( !index_in_tensor(occ_out[j]) ) {

                    replace_index_in_tensor(occ_in[i],occ_out[j]);
                    replace_index_in_tensor(occ_in[i],occ_out[j]);
                    replace_index_in_tensor(occ_in[i],occ_out[j]);
                    replace_index_in_tensor(occ_in[i],occ_out[j]);
                    break;
                }
            }
        }
    }

    // now virtual
    //std::vector<std::string> vir_in{"w","x","y","z"};
    std::vector<std::string> vir_in{"v1","v2","v3","v4"};
    std::vector<std::string> vir_out{"a","b","c","d"};

    for (int i = 0; i < (int)vir_in.size(); i++) {

        if ( index_in_tensor(vir_in[i]) ) {

            for (int j = 0; j < (int)vir_out.size(); j++) {

                if ( !index_in_tensor(vir_out[j]) ) {

                    replace_index_in_tensor(vir_in[i],vir_out[j]);
                    replace_index_in_tensor(vir_in[i],vir_out[j]);
                    replace_index_in_tensor(vir_in[i],vir_out[j]);
                    replace_index_in_tensor(vir_in[i],vir_out[j]);
                    break;
                }
            }
        }
    }


}

void ahat::gobble_deltas() {

    std::vector<std::string> tmp_delta1;
    std::vector<std::string> tmp_delta2;

    // possibilities:  1                2                 result
    //                 tensor           tensor            replace 1 in tensor with 2
    //                 tensor           t_amplitudes      replace 1 in tensor with 2
    //                 tensor           left amplitudes   replace 1 in tensor with 2
    //                 tensor           -                 replace 1 in tensor with 2
    //                 t_amplitudes     tensor            replace 2 in tensor with 1
    //                 left amplitudes  tensor            replace 2 in tensor with 1
    //                 -                tensor            replace 2 in tensor with 1
    //                 t_amplitudes     t_amplitudes      replace 1 in t_amplitudes with 2
    //                 t_amplitudes     left amplitudes   replace 1 in t_amplitudes with 2
    //                 t_amplitudes       -               replace 1 in t_amplitudes with 2
    //                 left amplitudes  t_amplitudes      replace 2 in t_amplitudes with 1
    //                 -                t_amplitudes      replace 2 in t_amplitudes with 1
    //                 left amplitudes  left amplitudes   replace 1 in left amplitudes with 2
    //                 left amplitudes  -                 replace 1 in left amplitudes with 2
    //                 -                left amplitudes   replace 2 in left amplitudes with 1
    for (int i = 0; i < (int)delta1.size(); i++) {

        bool delta1_in_tensor          = index_in_tensor( delta1[i] );
        bool delta2_in_tensor          = index_in_tensor( delta2[i] );
        bool delta1_in_t_amplitudes      = index_in_t_amplitudes( delta1[i] );
        bool delta2_in_t_amplitudes      = index_in_t_amplitudes( delta2[i] );
        bool delta1_in_left_amplitudes = index_in_left_amplitudes( delta1[i] );
        bool delta2_in_left_amplitudes = index_in_left_amplitudes( delta2[i] );

        if ( delta1_in_tensor ) {

            if ( delta2_in_tensor ) {

                replace_index_in_tensor( delta1[i], delta2[i] );

                continue;

            }else if ( delta2_in_t_amplitudes) {

                // replace index in tensor
                replace_index_in_tensor( delta1[i], delta2[i] );

                continue;

            }else if ( delta2_in_left_amplitudes) {

                // replace index in tensor
                replace_index_in_tensor( delta1[i], delta2[i] );

                continue;

            }else {

                // index two must come from the bra, so replace index one in tensor
                replace_index_in_tensor( delta1[i], delta2[i] );

                continue;

            }

        }else if ( delta2_in_tensor ) {

            if ( delta1_in_t_amplitudes) {

                // replace index in tensor
                replace_index_in_tensor( delta2[i], delta1[i] );

                continue;

            }else if ( delta1_in_left_amplitudes) {

                // replace index in tensor
                replace_index_in_tensor( delta2[i], delta1[i] );

                continue;

            }else {

                // index one must come from the bra, so replace index two in tensor
                replace_index_in_tensor( delta2[i], delta1[i] );

                continue;

            }

        }else if ( delta1_in_t_amplitudes ) {

            if ( delta2_in_t_amplitudes) {

                replace_index_in_t_amplitudes( delta1[i], delta2[i] );

                continue;

            }else if ( delta2_in_left_amplitudes) {

                replace_index_in_t_amplitudes( delta1[i], delta2[i] );

                continue;

            }else {

                // index two must come from the bra, so replace index one in t_amplitudes
                replace_index_in_t_amplitudes( delta1[i], delta2[i] );

                continue;

            }

        }else if ( delta2_in_t_amplitudes) {

            if ( delta1_in_left_amplitudes) {

                replace_index_in_t_amplitudes( delta2[i], delta1[i] );

                continue;

            }else {

                // index one must come from the bra, so replace index two in t_amplitudes
                replace_index_in_t_amplitudes( delta2[i], delta1[i] );

                continue;
            }

        }else if ( delta1_in_left_amplitudes ) {
            if ( delta2_in_left_amplitudes) {

                replace_index_in_left_amplitudes( delta1[i], delta2[i] );

                continue;

            }else {

                // index two must come from the bra, so replace index one in left amplitudes
                replace_index_in_left_amplitudes( delta1[i], delta2[i] );

                continue;
            }
        }else if ( delta2_in_left_amplitudes ) {

                // index one must come from the bra, so replace index two in left amplitudes
                replace_index_in_left_amplitudes( delta2[i], delta1[i] );

                continue;
        }

        // at this point, it is safe to assume the delta function must remain
        tmp_delta1.push_back(delta1[i]);
        tmp_delta2.push_back(delta2[i]);

    }


    delta1.clear();
    delta2.clear();

    for (int i = 0; i < (int)tmp_delta1.size(); i++) {
        delta1.push_back(tmp_delta1[i]);
        delta2.push_back(tmp_delta2[i]);
    }


/*
    std::vector<std::string> tmp_tensor;

    for (int j = 0; j < (int)data->tensor.size(); j++) {

        // does data->tensor index show up in a delta function? 
        bool skipme = false;
        for (int k = 0; k < (int)delta1.size(); k++) {
            if ( data->tensor[j] == delta1[k] ) {
                tmp_tensor.push_back(delta2[k]);
                skipme = true;
                break;
            }
            if ( data->tensor[j] == delta2[k] ) {
                tmp_tensor.push_back(delta1[k]);
                skipme = true;
                break;
            }
        }
        if ( skipme ) continue;

        tmp_tensor.push_back(data->tensor[j]);
    }

    std::vector<std::string> tmp_delta1;
    std::vector<std::string> tmp_delta2;
    for (int j = 0; j < (int)delta1.size(); j++) {
        bool skipme = false;
        for (int k = 0; k < (int)data->tensor.size(); k++) {
            if ( data->tensor[k] == delta1[j] ) {
                skipme = true;
                break;
            }
            if ( data->tensor[k] == delta2[j] ) {
                skipme = true;
                break;
            }
        }
        if ( skipme ) continue;
    
        tmp_delta1.push_back(delta1[j]);
        tmp_delta2.push_back(delta2[j]);
    }

    data->tensor.clear();
    for (int j = 0; j < (int)tmp_tensor.size(); j++) {
        data->tensor.push_back(tmp_tensor[j]);
    }
    delta1.clear();
    delta1.clear();
    for (int j = 0; j < (int)tmp_delta1.size(); j++) {
        delta1.push_back(tmp_delta1[j]);
        delta2.push_back(tmp_delta2[j]);
    }

    // t_amplitudes
    std::vector<std::vector<std::string>> tmp_t_amplitudes;
    for (int i = 0; i < (int)data->t_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)data->t_amplitudes[i].size(); j++) {

            // does data->amplitude index show up in a delta function?
            bool skipme = false;
            for (int k = 0; k < (int)delta1.size(); k++) {
                if ( data->t_amplitudes[i][j] == delta1[k] ) {
                    tmp.push_back(delta2[k]);
                    skipme = true;
                    break;
                }
                if ( data->t_amplitudes[i][j] == delta2[k] ) {
                    tmp.push_back(delta1[k]);
                    skipme = true;
                    break;
                }
            }
            if ( skipme ) continue;

            tmp.push_back(data->t_amplitudes[i][j]);
        }
        tmp_t_amplitudes.push_back(tmp);

        // now, remove delta functions from list that were gobbled up above
        std::vector<std::string> tmp_delta1;
        std::vector<std::string> tmp_delta2;

        for (int k = 0; k < (int)delta1.size(); k++) {
            bool skipme = false;
            for (int l = 0; l < (int)data->t_amplitudes[i].size(); l++) {
                if ( data->t_amplitudes[i][l] == delta1[k] ) {
                    skipme = true;
                    break;
                }
                if ( data->t_amplitudes[i][l] == delta2[k] ) {
                    skipme = true;
                    break;
                }
            }
            if ( skipme ) continue;
        
            tmp_delta1.push_back(delta1[k]);
            tmp_delta2.push_back(delta2[k]);
        }

        // update deltas
        delta1.clear();
        delta2.clear();
        for (int k = 0; k < (int)tmp_delta1.size(); k++) {
            delta1.push_back(tmp_delta1[k]);
            delta2.push_back(tmp_delta2[k]);
        }


    }

    // update t_amplitudes
    data->t_amplitudes.clear();
    for (int i = 0; i < (int)tmp_t_amplitudes.size(); i++) {
        data->t_amplitudes.push_back(tmp_t_amplitudes[i]);
    }
*/
    

}

// copy all data, including symbols and daggers
void ahat::copy(void * copy_me) { 

    shallow_copy(copy_me);

    ahat * in = reinterpret_cast<ahat * >(copy_me);


    // operators
    for (int j = 0; j < (int)in->symbol.size(); j++) {
        symbol.push_back(in->symbol[j]);

        // dagger?
        is_dagger.push_back(in->is_dagger[j]);

        // dagger (relative to fermi vacuum)?
        if ( vacuum == "FERMI" ) {
            is_dagger_fermi.push_back(in->is_dagger_fermi[j]);
        }
    }
    
}

void ahat::normal_order_true_vacuum(std::vector<std::shared_ptr<ahat> > &ordered) {
    if ( skip ) return;

    if ( is_normal_order() ) {

        // push current ordered operator onto running list
        std::shared_ptr<ahat> newguy (new ahat(vacuum));

        newguy->copy((void*)this);

        ordered.push_back(newguy);

        return;
    }

    // new strings
    std::shared_ptr<ahat> s1 ( new ahat(vacuum) );
    std::shared_ptr<ahat> s2 ( new ahat(vacuum) );

    // copy data common to both new strings
    s1->shallow_copy((void*)this);
    s2->shallow_copy((void*)this);

    // rearrange operators
    for (int i = 0; i < (int)symbol.size()-1; i++) {

        bool swap = ( !is_dagger[i] && is_dagger[i+1] );

        if ( swap ) {

            s1->delta1.push_back(symbol[i]);
            s1->delta2.push_back(symbol[i+1]);

            // check spin in delta functions
            //for (int j = 0; j < (int)delta1.size(); j++) {
            //    if ( s1->delta1[j].length() != 2 ) {
            //        //throw PsiException("be sure to specify spin as second character in labels",__FILE__,__LINE__);
            //        break;
            //    }
            //    //if ( s1->delta1[j].at(1) == 'A' && s1->delta2[j].at(1) == 'B' ) {
            //    //    s1->skip = true;
            //    //}else if ( s1->delta1[j].at(1) == 'B' && s1->delta2[j].at(1) == 'A' ) {
            //    //    s1->skip = true;
            //    //}
            //}

            s2->sign = -s2->sign;
            s2->symbol.push_back(symbol[i+1]);
            s2->symbol.push_back(symbol[i]);
            s2->is_dagger.push_back(is_dagger[i+1]);
            s2->is_dagger.push_back(is_dagger[i]);

            for (int j = i+2; j < (int)symbol.size(); j++) {

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

    s1->normal_order_true_vacuum(ordered);
    s2->normal_order_true_vacuum(ordered);

}

void ahat::normal_order_fermi_vacuum(std::vector<std::shared_ptr<ahat> > &ordered) {
    if ( skip ) return;

    if ( is_normal_order() ) {

        // push current ordered operator onto running list
        std::shared_ptr<ahat> newguy (new ahat(vacuum));

        newguy->copy((void*)this);

        ordered.push_back(newguy);

        return;
    }

    // new strings
    std::shared_ptr<ahat> s1 ( new ahat(vacuum) );
    std::shared_ptr<ahat> s2 ( new ahat(vacuum) );

    // copy data common to both new strings
    s1->shallow_copy((void*)this);
    s2->shallow_copy((void*)this);

    // rearrange operators

    int n_new_strings = 0;

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

            // check spin in delta functions
            //for (int j = 0; j < (int)delta1.size(); j++) {
            //    if ( s1->delta1[j].length() != 2 ) {
            //        //throw PsiException("be sure to specify spin as second character in labels",__FILE__,__LINE__);
            //        break;
            //    }
            //    //if ( s1->delta1[j].at(1) == 'A' && s1->delta2[j].at(1) == 'B' ) {
            //    //    s1->skip = true;
            //    //}else if ( s1->delta1[j].at(1) == 'B' && s1->delta2[j].at(1) == 'A' ) {
            //    //    s1->skip = true;
            //    //}
            //}

            s2->sign = -s2->sign;
            s2->symbol.push_back(symbol[i+1]);
            s2->symbol.push_back(symbol[i]);
            s2->is_dagger.push_back(is_dagger[i+1]);
            s2->is_dagger.push_back(is_dagger[i]);
            s2->is_dagger_fermi.push_back(is_dagger_fermi[i+1]);
            s2->is_dagger_fermi.push_back(is_dagger_fermi[i]);

            for (int j = i+2; j < (int)symbol.size(); j++) {

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

            //s2->sign = -s2->sign;
            //s2->symbol.push_back(symbol[i+1]);
            //s2->symbol.push_back(symbol[i]);
            //s2->is_dagger.push_back(is_dagger[i+1]);
            //s2->is_dagger.push_back(is_dagger[i]);
            //s2->is_dagger_fermi.push_back(is_dagger_fermi[i+1]);
            //s2->is_dagger_fermi.push_back(is_dagger_fermi[i]);

            for (int j = i+2; j < (int)symbol.size(); j++) {

                s1->symbol.push_back(symbol[j]);
                //s2->symbol.push_back(symbol[j]);

                s1->is_dagger.push_back(is_dagger[j]);
                //s2->is_dagger.push_back(is_dagger[j]);

                s1->is_dagger_fermi.push_back(is_dagger_fermi[j]);
                //s2->is_dagger_fermi.push_back(is_dagger_fermi[j]);

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


    if ( n_new_strings == 1 ) {
        s1->normal_order_fermi_vacuum(ordered);
    }else if ( n_new_strings == 2 ) {
        s1->normal_order_fermi_vacuum(ordered);
        s2->normal_order_fermi_vacuum(ordered);
    }

}

void ahat::normal_order(std::vector<std::shared_ptr<ahat> > &ordered) {
    if ( vacuum == "TRUE" ) {
        normal_order_true_vacuum(ordered);
    }else {
        normal_order_fermi_vacuum(ordered);
    }
}


} // End namespaces

