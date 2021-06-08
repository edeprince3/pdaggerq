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

#include "pq.h"

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

void pq::check_spin() {

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

void pq::print() {

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

    if ( (int)data->permutations.size() > 0 ) {
        // should have an even number of symbols...how many pairs?
        int n = (int)data->permutations.size() / 2;
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

    for (int i = 0; i < (int)symbol.size(); i++) {
        printf("%s",symbol[i].c_str());
        if ( is_dagger[i] ) {
            printf("%c",'*');
        }
        printf(" ");
    }
    for (int i = 0; i < (int)delta1.size(); i++) {
        printf("d(%s,%s)",delta1[i].c_str(),delta2[i].c_str());
        printf(" ");
    }

    // two-electron integrals
    if ( (int)data->tensor.size() == 4 ) {

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
    if ( (int)data->tensor.size() == 2 ) {
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
    if ( data->has_l0 ) {
        printf("l0");
        printf(" ");
    }

    // right-hand amplitudes
    if ( (int)data->right_amplitudes.size() > 0 ) {
        for (int i = 0; i < (int)data->right_amplitudes.size(); i++) {
           
            if ( (int)data->right_amplitudes[i].size() > 0 ) {
                // r1
                if ( (int)data->right_amplitudes[i].size() == 2 ) {
                    printf("r1(");
                    printf("%s",data->right_amplitudes[i][0].c_str());
                    printf(",");
                    printf("%s",data->right_amplitudes[i][1].c_str());
                    printf(")");
                }
                // r2
                if ( (int)data->right_amplitudes[i].size() == 4 ) {
                    printf("r2(");
                    printf("%s",data->right_amplitudes[i][0].c_str());
                    printf(",");
                    printf("%s",data->right_amplitudes[i][1].c_str());
                    printf(",");
                    printf("%s",data->right_amplitudes[i][2].c_str());
                    printf(",");
                    printf("%s",data->right_amplitudes[i][3].c_str());
                    printf(")");
                }
                printf(" ");
            } 
        }
    }
    if ( data->has_r0 ) {
        printf("r0");
        printf(" ");
    }

    // t_amplitudes
    if ( (int)data->t_amplitudes.size() > 0 ) {
        for (int i = 0; i < (int)data->t_amplitudes.size(); i++) {

            int order = (int)data->t_amplitudes[i].size() / 2;
            printf("t");
            printf("%i",order);
            printf("(");
            for (int j = 0; j < 2*order-1; j++) {
                printf("%s",data->t_amplitudes[i][j].c_str());
                printf(",");
            }
            printf("%s",data->t_amplitudes[i][2*order-1].c_str());
            printf(")");
            printf(" ");
          
/* 
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
                // t3
                if ( (int)data->t_amplitudes[i].size() == 6 ) {
                    printf("t3(");
                    printf("%s",data->t_amplitudes[i][0].c_str());
                    printf(",");
                    printf("%s",data->t_amplitudes[i][1].c_str());
                    printf(",");
                    printf("%s",data->t_amplitudes[i][2].c_str());
                    printf(",");
                    printf("%s",data->t_amplitudes[i][3].c_str());
                    printf(",");
                    printf("%s",data->t_amplitudes[i][4].c_str());
                    printf(",");
                    printf("%s",data->t_amplitudes[i][5].c_str());
                    printf(")");
                }
                printf(" ");
            } 
*/
        }
    }

    // u_amplitudes
    if ( (int)data->u_amplitudes.size() > 0 ) {
        for (int i = 0; i < (int)data->u_amplitudes.size(); i++) {
           
            if ( (int)data->u_amplitudes[i].size() > 0 ) {
                // u1
                if ( (int)data->u_amplitudes[i].size() == 2 ) {
                    printf("u1(");
                    printf("%s",data->u_amplitudes[i][0].c_str());
                    printf(",");
                    printf("%s",data->u_amplitudes[i][1].c_str());
                    printf(")");
                }
                // u2
                if ( (int)data->u_amplitudes[i].size() == 4 ) {
                    printf("u2(");
                    printf("%s",data->u_amplitudes[i][0].c_str());
                    printf(",");
                    printf("%s",data->u_amplitudes[i][1].c_str());
                    printf(",");
                    printf("%s",data->u_amplitudes[i][2].c_str());
                    printf(",");
                    printf("%s",data->u_amplitudes[i][3].c_str());
                    printf(")");
                }
                printf(" ");
            } 
        }
    }
    if ( data->has_u0 ) {
        printf("u0");
        printf(" ");
    }

    // m_amplitudes
    if ( (int)data->m_amplitudes.size() > 0 ) {
        for (int i = 0; i < (int)data->m_amplitudes.size(); i++) {
           
            if ( (int)data->m_amplitudes[i].size() > 0 ) {
                // m1
                if ( (int)data->m_amplitudes[i].size() == 2 ) {
                    printf("m1(");
                    printf("%s",data->m_amplitudes[i][0].c_str());
                    printf(",");
                    printf("%s",data->m_amplitudes[i][1].c_str());
                    printf(")");
                }
                // m2
                if ( (int)data->m_amplitudes[i].size() == 4 ) {
                    printf("m2(");
                    printf("%s",data->m_amplitudes[i][0].c_str());
                    printf(",");
                    printf("%s",data->m_amplitudes[i][1].c_str());
                    printf(",");
                    printf("%s",data->m_amplitudes[i][2].c_str());
                    printf(",");
                    printf("%s",data->m_amplitudes[i][3].c_str());
                    printf(")");
                }
                printf(" ");
            } 
        }
    }
    if ( data->has_m0 ) {
        printf("m0");
        printf(" ");
    }

    // s_amplitudes
    if ( (int)data->s_amplitudes.size() > 0 ) {
        for (int i = 0; i < (int)data->s_amplitudes.size(); i++) {
           
            if ( (int)data->s_amplitudes[i].size() > 0 ) {
                // s1
                if ( (int)data->s_amplitudes[i].size() == 2 ) {
                    printf("s1(");
                    printf("%s",data->s_amplitudes[i][0].c_str());
                    printf(",");
                    printf("%s",data->s_amplitudes[i][1].c_str());
                    printf(")");
                }
                // s2
                if ( (int)data->s_amplitudes[i].size() == 4 ) {
                    printf("s2(");
                    printf("%s",data->s_amplitudes[i][0].c_str());
                    printf(",");
                    printf("%s",data->s_amplitudes[i][1].c_str());
                    printf(",");
                    printf("%s",data->s_amplitudes[i][2].c_str());
                    printf(",");
                    printf("%s",data->s_amplitudes[i][3].c_str());
                    printf(")");
                }
                printf(" ");
            } 
        }
    }
    if ( data->has_s0 ) {
        printf("s0");
        printf(" ");
    }

    // bosons:
    for (int i = 0; i < (int)data->is_boson_dagger.size(); i++) {
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

std::vector<std::string> pq::get_string() {

    std::vector<std::string> my_string;

    if ( skip ) return my_string;

    if ( vacuum == "FERMI" && (int)symbol.size() > 0 ) {
        // check if stings should be zero or not
        bool is_dagger_right = is_dagger_fermi[(int)symbol.size() - 1];
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
    my_string.push_back(tmp + std::to_string(fabs(data->factor)));

    if ( (int)data->permutations.size() > 0 ) {
        // should have an even number of symbols...how many pairs?
        int n = (int)data->permutations.size() / 2;
        int count = 0;
        for (int i = 0; i < n; i++) {
            tmp  = "P(";
            tmp += data->permutations[count++];
            tmp += ",";
            tmp += data->permutations[count++];
            tmp += ")";
            my_string.push_back(tmp);
        }
    }

    for (int i = 0; i < (int)symbol.size(); i++) {
        std::string tmp = symbol[i];
        if ( is_dagger[i] ) {
            tmp += "*";
        }
        my_string.push_back(tmp);
    }

    for (int i = 0; i < (int)delta1.size(); i++) {
        std::string tmp = "d(" + delta1[i] + "," + delta2[i] + ")";
        my_string.push_back(tmp);
    }

    // two-electron integrals
    if ( (int)data->tensor.size() == 4 ) {

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
    if ( (int)data->tensor.size() == 2 ) {
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
    if ( (int)data->left_amplitudes.size() > 0 ) {

        for (int i = 0; i < (int)data->left_amplitudes.size(); i++) {
           
            std::string tmp;
            if ( (int)data->left_amplitudes[i].size() > 0 ) {
                // l1
                if ( (int)data->left_amplitudes[i].size() == 2 ) {
                    tmp = "l1("
                        + data->left_amplitudes[i][0]
                        + ","
                        + data->left_amplitudes[i][1]
                        + ")";
                }
                // l2
                if ( (int)data->left_amplitudes[i].size() == 4 ) {
                    tmp = "l2("
                        + data->left_amplitudes[i][0]
                        + ","
                        + data->left_amplitudes[i][1]
                        + ","
                        + data->left_amplitudes[i][2]
                        + ","
                        + data->left_amplitudes[i][3]
                        + ")";
                }
                my_string.push_back(tmp);
            } 
        }
    }
    if ( data->has_l0 ) {
        my_string.push_back("l0");
    }

    // right-hand amplitudes
    if ( (int)data->right_amplitudes.size() > 0 ) {
        for (int i = 0; i < (int)data->right_amplitudes.size(); i++) {
           
            if ( (int)data->right_amplitudes[i].size() > 0 ) {
                std::string tmp;
                // r1
                if ( (int)data->right_amplitudes[i].size() == 2 ) {
                    tmp = "r1("
                        + data->right_amplitudes[i][0]
                        + ","
                        + data->right_amplitudes[i][1]
                        + ")";
                }
                // r2
                if ( (int)data->right_amplitudes[i].size() == 4 ) {
                    tmp = "r2("
                        + data->right_amplitudes[i][0]
                        + ","
                        + data->right_amplitudes[i][1]
                        + ","
                        + data->right_amplitudes[i][2]
                        + ","
                        + data->right_amplitudes[i][3]
                        + ")";
                }
                my_string.push_back(tmp);
            } 
        }
    }
    if ( data->has_r0 ) {
        my_string.push_back("r0");
    }

    // t_amplitudes
    if ( (int)data->t_amplitudes.size() > 0 ) {
        for (int i = 0; i < (int)data->t_amplitudes.size(); i++) {
          
            if ( (int)data->t_amplitudes[i].size() > 0 ) {
 
                int order = (int)data->t_amplitudes[i].size() / 2;
                tmp = "t" + std::to_string(order) + "(";
                for (int j = 0; j < 2*order-1; j++) {
                    tmp += data->t_amplitudes[i][j] + ",";
                }
                tmp += data->t_amplitudes[i][2*order-1] + ")";
                my_string.push_back(tmp);

            }

/*
            if ( (int)data->t_amplitudes[i].size() > 0 ) {
                std::string tmp;
                // t1
                if ( (int)data->t_amplitudes[i].size() == 2 ) {
                    tmp = "t1("
                        + data->t_amplitudes[i][0]
                        + ","
                        + data->t_amplitudes[i][1]
                        + ")";
                }
                // t2
                if ( (int)data->t_amplitudes[i].size() == 4 ) {
                    tmp = "t2("
                        + data->t_amplitudes[i][0]
                        + ","
                        + data->t_amplitudes[i][1]
                        + ","
                        + data->t_amplitudes[i][2]
                        + ","
                        + data->t_amplitudes[i][3]
                        + ")";
                }
                // t3
                if ( (int)data->t_amplitudes[i].size() == 6 ) {
                    tmp = "t3("
                        + data->t_amplitudes[i][0]
                        + ","
                        + data->t_amplitudes[i][1]
                        + ","
                        + data->t_amplitudes[i][2]
                        + ","
                        + data->t_amplitudes[i][3]
                        + ","
                        + data->t_amplitudes[i][4]
                        + ","
                        + data->t_amplitudes[i][5]
                        + ")";
                }
                my_string.push_back(tmp);
            } 
*/
        }
    }

    // u_amplitudes
    if ( (int)data->u_amplitudes.size() > 0 ) {
        for (int i = 0; i < (int)data->u_amplitudes.size(); i++) {
           
            if ( (int)data->u_amplitudes[i].size() > 0 ) {
                std::string tmp;
                // u1
                if ( (int)data->u_amplitudes[i].size() == 2 ) {
                    tmp = "u1("
                        + data->u_amplitudes[i][0]
                        + ","
                        + data->u_amplitudes[i][1]
                        + ")";
                }
                // u2
                if ( (int)data->u_amplitudes[i].size() == 4 ) {
                    tmp = "u2("
                        + data->u_amplitudes[i][0]
                        + ","
                        + data->u_amplitudes[i][1]
                        + ","
                        + data->u_amplitudes[i][2]
                        + ","
                        + data->u_amplitudes[i][3]
                        + ")";
                }
                my_string.push_back(tmp);
            } 
        }
    }
    if ( data->has_u0 ) {
        my_string.push_back("u0");
    }

    // m_amplitudes
    if ( (int)data->m_amplitudes.size() > 0 ) {
        for (int i = 0; i < (int)data->m_amplitudes.size(); i++) {
           
            if ( (int)data->m_amplitudes[i].size() > 0 ) {
                std::string tmp;
                // m1
                if ( (int)data->m_amplitudes[i].size() == 2 ) {
                    tmp = "m1("
                        + data->m_amplitudes[i][0]
                        + ","
                        + data->m_amplitudes[i][1]
                        + ")";
                }
                // m2
                if ( (int)data->m_amplitudes[i].size() == 4 ) {
                    tmp = "m2("
                        + data->m_amplitudes[i][0]
                        + ","
                        + data->m_amplitudes[i][1]
                        + ","
                        + data->m_amplitudes[i][2]
                        + ","
                        + data->m_amplitudes[i][3]
                        + ")";
                }
                my_string.push_back(tmp);
            } 
        }
    }
    if ( data->has_m0 ) {
        my_string.push_back("m0");
    }

    // s_amplitudes
    if ( (int)data->s_amplitudes.size() > 0 ) {
        for (int i = 0; i < (int)data->s_amplitudes.size(); i++) {
           
            if ( (int)data->s_amplitudes[i].size() > 0 ) {
                std::string tmp;
                // s1
                if ( (int)data->s_amplitudes[i].size() == 2 ) {
                    tmp = "s1("
                        + data->s_amplitudes[i][0]
                        + ","
                        + data->s_amplitudes[i][1]
                        + ")";
                }
                // s2
                if ( (int)data->s_amplitudes[i].size() == 4 ) {
                    tmp = "s2("
                        + data->s_amplitudes[i][0]
                        + ","
                        + data->s_amplitudes[i][1]
                        + ","
                        + data->s_amplitudes[i][2]
                        + ","
                        + data->s_amplitudes[i][3]
                        + ")";
                }
                my_string.push_back(tmp);
            } 
        }
    }
    if ( data->has_s0 ) {
        my_string.push_back("s0");
    }

    // bosons:
    for (int i = 0; i < (int)data->is_boson_dagger.size(); i++) {
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
            bool is_dagger_right = is_dagger_fermi[(int)symbol.size() - 1];
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

    if ( (int)data->is_boson_dagger.size() == 1 ) {
        bool is_dagger_right = data->is_boson_dagger[0];
        bool is_dagger_left  = data->is_boson_dagger[0];
        if ( !is_dagger_right || is_dagger_left ) {
            skip = true; 
            return true;
        }
    }
    for (int i = 0; i < (int)data->is_boson_dagger.size() - 1; i++) {

        // check if stings should be zero or not ... added 5/28/21
        bool is_dagger_right = data->is_boson_dagger[(int)data->is_boson_dagger.size() - 1];
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

// TODO: t3, t4, etc?  
// TODO: account for left-hand amplitudes
// TODO: account for right-hand amplitudes
// TODO: need an update_ket_labes function?

// TODO: with set_left_operator replacing set_bra_state, this function might become useless

void pq::update_bra_labels() {

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

    // u_amplitudes
    find_m = index_in_u_amplitudes("m");
    find_n = index_in_u_amplitudes("n");
    find_e = index_in_u_amplitudes("e");
    find_f = index_in_u_amplitudes("f");
    
    for (int i = 0; i < data->u_amplitudes.size(); i++) {

        if ( data->u_amplitudes[i].size() != 4 ) continue;

        if ( find_m && find_n ) {
            // should appear as "mn"
            if ( data->u_amplitudes[i][2] == "n" && data->u_amplitudes[i][3] == "m" ) {
                data->u_amplitudes[i][2] = "m";
                data->u_amplitudes[i][3] = "n";
                sign = -sign;
            }
        }else if ( find_m ) {
            // should appear as "-m"
            if ( data->u_amplitudes[i][2] == "m" ) {
                data->u_amplitudes[i][2] = data->u_amplitudes[i][3];
                data->u_amplitudes[i][3] = "m";
                sign = -sign;
            }
        }else if ( find_n) {
            // should appear as "-n"
            if ( data->u_amplitudes[i][2] == "n" ) {
                data->u_amplitudes[i][2] = data->u_amplitudes[i][3];
                data->u_amplitudes[i][3] = "n";
                sign = -sign;
            }
        }

        if ( find_e && find_f ) {
            // should appear as "mn"
            if ( data->u_amplitudes[i][0] == "f" && data->u_amplitudes[i][1] == "e" ) {
                data->u_amplitudes[i][0] = "e";
                data->u_amplitudes[i][1] = "f";
                sign = -sign;
            }
        }else if ( find_e ) {
            // should appear as "-e"
            if ( data->u_amplitudes[i][0] == "e" ) {
                data->u_amplitudes[i][0] = data->u_amplitudes[i][1];
                data->u_amplitudes[i][1] = "e";
                sign = -sign;
            }
        }else if ( find_f) {
            // should appear as "-f"
            if ( data->u_amplitudes[i][0] == "f" ) {
                data->u_amplitudes[i][0] = data->u_amplitudes[i][1];
                data->u_amplitudes[i][1] = "f";
                sign = -sign;
            }
        }

    }

    // m_amplitudes
    find_m = index_in_m_amplitudes("m");
    find_n = index_in_m_amplitudes("n");
    find_e = index_in_m_amplitudes("e");
    find_f = index_in_m_amplitudes("f");
    
    for (int i = 0; i < data->m_amplitudes.size(); i++) {

        if ( data->m_amplitudes[i].size() != 4 ) continue;

        if ( find_m && find_n ) {
            // should appear as "mn"
            if ( data->m_amplitudes[i][2] == "n" && data->m_amplitudes[i][3] == "m" ) {
                data->m_amplitudes[i][2] = "m";
                data->m_amplitudes[i][3] = "n";
                sign = -sign;
            }
        }else if ( find_m ) {
            // should appear as "-m"
            if ( data->m_amplitudes[i][2] == "m" ) {
                data->m_amplitudes[i][2] = data->m_amplitudes[i][3];
                data->m_amplitudes[i][3] = "m";
                sign = -sign;
            }
        }else if ( find_n) {
            // should appear as "-n"
            if ( data->m_amplitudes[i][2] == "n" ) {
                data->m_amplitudes[i][2] = data->m_amplitudes[i][3];
                data->m_amplitudes[i][3] = "n";
                sign = -sign;
            }
        }

        if ( find_e && find_f ) {
            // should appear as "mn"
            if ( data->m_amplitudes[i][0] == "f" && data->m_amplitudes[i][1] == "e" ) {
                data->m_amplitudes[i][0] = "e";
                data->m_amplitudes[i][1] = "f";
                sign = -sign;
            }
        }else if ( find_e ) {
            // should appear as "-e"
            if ( data->m_amplitudes[i][0] == "e" ) {
                data->m_amplitudes[i][0] = data->m_amplitudes[i][1];
                data->m_amplitudes[i][1] = "e";
                sign = -sign;
            }
        }else if ( find_f) {
            // should appear as "-f"
            if ( data->m_amplitudes[i][0] == "f" ) {
                data->m_amplitudes[i][0] = data->m_amplitudes[i][1];
                data->m_amplitudes[i][1] = "f";
                sign = -sign;
            }
        }

    }

    // s_amplitudes
    find_m = index_in_s_amplitudes("m");
    find_n = index_in_s_amplitudes("n");
    find_e = index_in_s_amplitudes("e");
    find_f = index_in_s_amplitudes("f");
    
    for (int i = 0; i < data->s_amplitudes.size(); i++) {

        if ( data->s_amplitudes[i].size() != 4 ) continue;

        if ( find_m && find_n ) {
            // should appear as "mn"
            if ( data->s_amplitudes[i][2] == "n" && data->s_amplitudes[i][3] == "m" ) {
                data->s_amplitudes[i][2] = "m";
                data->s_amplitudes[i][3] = "n";
                sign = -sign;
            }
        }else if ( find_m ) {
            // should appear as "-m"
            if ( data->s_amplitudes[i][2] == "m" ) {
                data->s_amplitudes[i][2] = data->s_amplitudes[i][3];
                data->s_amplitudes[i][3] = "m";
                sign = -sign;
            }
        }else if ( find_n) {
            // should appear as "-n"
            if ( data->s_amplitudes[i][2] == "n" ) {
                data->s_amplitudes[i][2] = data->s_amplitudes[i][3];
                data->s_amplitudes[i][3] = "n";
                sign = -sign;
            }
        }

        if ( find_e && find_f ) {
            // should appear as "mn"
            if ( data->s_amplitudes[i][0] == "f" && data->s_amplitudes[i][1] == "e" ) {
                data->s_amplitudes[i][0] = "e";
                data->s_amplitudes[i][1] = "f";
                sign = -sign;
            }
        }else if ( find_e ) {
            // should appear as "-e"
            if ( data->s_amplitudes[i][0] == "e" ) {
                data->s_amplitudes[i][0] = data->s_amplitudes[i][1];
                data->s_amplitudes[i][1] = "e";
                sign = -sign;
            }
        }else if ( find_f) {
            // should appear as "-f"
            if ( data->s_amplitudes[i][0] == "f" ) {
                data->s_amplitudes[i][0] = data->s_amplitudes[i][1];
                data->s_amplitudes[i][1] = "f";
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
void pq::update_summation_labels() {

    if ( vacuum == "FERMI" && symbol.size() != 0 ) return;

    if ( skip ) return;

    int find_i = index_in_anywhere("i");
    int find_j = index_in_anywhere("j");
    int find_k = index_in_anywhere("k");
    int find_l = index_in_anywhere("l");

    int find_a = index_in_anywhere("a");
    int find_b = index_in_anywhere("b");
    int find_c = index_in_anywhere("c");
    int find_d = index_in_anywhere("d");

    // i,j,k,l
    if ( find_i == 0 && find_j == 2 && find_k == 2 && find_l == 2 ) {

        replace_index_everywhere("l","i");

    }else if ( find_i == 0 && find_j == 2 && find_k == 2 && find_l == 0 ) {

        replace_index_everywhere("k","i");

    }else if ( find_i == 0 && find_j == 2 && find_k == 0 && find_l == 0 ) {

        replace_index_everywhere("j","i");

    }else if ( find_i == 0 && find_j == 2 && find_k == 0 && find_l == 2 ) {

        replace_index_everywhere("l","i");

    }else if ( find_i == 0 && find_j == 2 && find_k == 2 && find_l == 2 ) {

        replace_index_everywhere("k","i");
        replace_index_everywhere("l","j");

    }else if ( find_i == 0 && find_j == 0 && find_k == 0 && find_l == 2 ) {

        replace_index_everywhere("l","i");

    }else if ( find_i == 0 && find_j == 0 && find_k == 2 && find_l == 0 ) {

        replace_index_everywhere("k","i");

    }else if ( find_i == 2 && find_j == 0 && find_k == 2 && find_l == 2 ) {

        replace_index_everywhere("l","j");

    }else if ( find_i == 2 && find_j == 0 && find_k == 0 && find_l == 2 ) {

        replace_index_everywhere("l","j");

    }else if ( find_i == 2 && find_j == 0 && find_k == 2 && find_l == 0 ) {

        replace_index_everywhere("k","j");

    }else if ( find_i == 2 && find_j == 0 && find_k == 0 && find_l == 2 ) {

        replace_index_everywhere("l","j");

    }

    // a,b,c,d
    if ( find_a == 0 && find_b == 2 && find_c == 2 && find_d == 2 ) {

        replace_index_everywhere("d","a");

    }else if ( find_a == 0 && find_b == 2 && find_c == 2 && find_d == 0 ) {

        replace_index_everywhere("c","a");

    }else if ( find_a == 0 && find_b == 2 && find_c == 0 && find_d == 2 ) {

        replace_index_everywhere("d","a");

    }else if ( find_a == 0 && find_b == 2 && find_c == 0 && find_d == 0 ) {

        replace_index_everywhere("b","a");

    }else if ( find_a == 0 && find_b == 0 && find_c == 2 && find_d == 2 ) {

        replace_index_everywhere("c","a");
        replace_index_everywhere("d","b");

    }else if ( find_a == 0 && find_b == 0 && find_c == 0 && find_d == 2 ) {

        replace_index_everywhere("d","a");

    }else if ( find_a == 0 && find_b == 0 && find_c == 2 && find_d == 0 ) {

        replace_index_everywhere("c","a");

    }else if ( find_a == 2 && find_b == 0 && find_c == 2 && find_d == 2 ) {

        replace_index_everywhere("d","b");

    }else if ( find_a == 2 && find_b == 0 && find_c == 0 && find_d == 2 ) {

        replace_index_everywhere("d","b");

    }else if ( find_a == 2 && find_b == 0 && find_c == 0 && find_d == 2 ) {

        replace_index_everywhere("d","b");

    }else if ( find_a == 2 && find_b == 0 && find_c == 2 && find_d == 0 ) {

        replace_index_everywhere("c","b");

    }else if ( find_a == 2 && find_b == 0 && find_c == 0 && find_d == 2 ) {

        replace_index_everywhere("d","b");

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

void pq::swap_two_labels(std::string label1, std::string label2) {

    replace_index_everywhere(label1,"x");
    replace_index_everywhere(label2,label1);
    replace_index_everywhere("x",label2);
}

void pq::reorder_t_amplitudes() {

    int dim = (int)data->t_amplitudes.size();

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
    if ( dim != (int)tmp.size() ) { 
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
    for (int i = 0; i < (int)tmp.size(); i++) {
        data->t_amplitudes.push_back(tmp[i]);
    }

    free(nope);
    
}

// once strings are alphabetized, we can compare them
// and remove terms that cancel. 

// TODO: need to consider u-amplitudes
// TODO: need to consider left-hand amplitudes
// TODO: need to consider right-hand amplitudes
void pq::cleanup(std::vector<std::shared_ptr<pq> > &ordered) {

    // order amplitudes such that they're ordered t1, t2, t3, etc.
    for (int i = 0; i < (int)ordered.size(); i++) {
        ordered[i]->reorder_t_amplitudes();
    }

    // prioritize summation labels as i > j > k > l and a > b > c > d.
    // this means that j, k, or l should not arise in a term if i is not
    // already present. only do this for vacuum_type = "FERMI"
    for (int i = 0; i < (int)ordered.size(); i++) {
        if ( vacuum != "FERMI" ) continue;
        ordered[i]->update_summation_labels();
    }

    // prune list so it only contains non-skipped ones
    std::vector< std::shared_ptr<pq> > pruned;
    for (int i = 0; i < (int)ordered.size(); i++) {

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
    for (int i = 0; i < (int)pruned.size(); i++) {
        ordered.push_back(pruned[i]);
    }
    pruned.clear();

    //printf("starting string comparisons\n");fflush(stdout);

    // consolidate terms, including those that differ only by symmetric quantities [i.e., g(iajb) and g(jbia)]
    for (int i = 0; i < (int)ordered.size(); i++) {

        if ( ordered[i]-> skip ) continue;

        // TODO: should be searching for labels in left / right / m / s amplitudes as well

        // TODO: should be searching for more labels than this ... k,l,m,n,c,d,e,f

        bool find_i = ordered[i]->index_in_tensor("i") 
                   || ordered[i]->index_in_t_amplitudes("i") 
                   || ordered[i]->index_in_u_amplitudes("i");

        bool find_j = ordered[i]->index_in_tensor("j") 
                   || ordered[i]->index_in_t_amplitudes("j") 
                   || ordered[i]->index_in_u_amplitudes("j");
                                                                                                                                             
        bool find_a = ordered[i]->index_in_tensor("a") 
                   || ordered[i]->index_in_t_amplitudes("a") 
                   || ordered[i]->index_in_u_amplitudes("a");

        bool find_b = ordered[i]->index_in_tensor("b") 
                   || ordered[i]->index_in_t_amplitudes("b") 
                   || ordered[i]->index_in_u_amplitudes("b");

        for (int j = i+1; j < (int)ordered.size(); j++) {

            if ( ordered[i]-> skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            // try swapping summation labels - only i/j, a/b swaps for now. this should be sufficient for ccsd
            if ( !strings_same && find_i && find_j ) {

                std::shared_ptr<pq> newguy (new pq(vacuum));
                newguy->copy((void*)(ordered[i].get()));
                newguy->swap_two_labels("i","j");
                strings_same = compare_strings(ordered[j],newguy,n_permute);
            }

            if ( !strings_same && find_a && find_b ) {

                std::shared_ptr<pq> newguy (new pq(vacuum));
                newguy->copy((void*)(ordered[i].get()));
                newguy->swap_two_labels("a","b");
                strings_same = compare_strings(ordered[j],newguy,n_permute);

            }

            if ( !strings_same && find_i && find_j && find_a && find_b ) {

                std::shared_ptr<pq> newguy (new pq(vacuum));
                newguy->copy((void*)(ordered[i].get()));
                newguy->swap_two_labels("i","j");
                newguy->swap_two_labels("a","b");
                strings_same = compare_strings(ordered[j],newguy,n_permute);
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

    if ( vacuum != "FERMI" ) return;

    // consolidate terms that differ by permutations (occupied)

    // TODO: account for r,l,m,s amplitudes

    for (int i = 0; i < (int)ordered.size(); i++) {

        if ( ordered[i]-> skip ) continue;

        std::vector<bool> find_idx;
        std::vector<std::string> labels { "i", "j", "k", "l", "m", "n" };

        // ok, what labels do we have?
        for (int j = 0; j < (int)labels.size(); j++) {
            bool found = ordered[i]->index_in_tensor(labels[j]) 
                      || ordered[i]->index_in_t_amplitudes(labels[j]) 
                      || ordered[i]->index_in_u_amplitudes(labels[j]);
            find_idx.push_back(found);
        }

        for (int j = i+1; j < (int)ordered.size(); j++) {

            if ( ordered[i]-> skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            std::string permutation_1;
            std::string permutation_2;

            // try swapping summation labels
            for (int id1 = 0; id1 < (int)labels.size(); id1++) {
                if ( !find_idx[id1] ) continue;
                for (int id2 = id1 + 1; id2 < (int)labels.size(); id2++) {
                    if ( !find_idx[id2] ) continue;

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

    // TODO: consolidate terms that differ by permutations (virtual)
    for (int i = 0; i < (int)ordered.size(); i++) {

        if ( ordered[i]-> skip ) continue;

        std::vector<bool> find_idx;
        std::vector<std::string> labels { "a", "b", "c", "d", "e", "f" };

        // ok, what labels do we have?
        for (int j = 0; j < (int)labels.size(); j++) {
            bool found = ordered[i]->index_in_tensor(labels[j]) 
                      || ordered[i]->index_in_t_amplitudes(labels[j]) 
                      || ordered[i]->index_in_u_amplitudes(labels[j]);
            find_idx.push_back(found);
        }

        for (int j = i+1; j < (int)ordered.size(); j++) {

            if ( ordered[i]-> skip ) continue;

            int n_permute;
            bool strings_same = compare_strings(ordered[i],ordered[j],n_permute);

            std::string permutation_1;
            std::string permutation_2;

            // try swapping summation labels
            for (int id1 = 0; id1 < (int)labels.size(); id1++) {
                if ( !find_idx[id1] ) continue;
                for (int id2 = id1 + 1; id2 < (int)labels.size(); id2++) {
                    if ( !find_idx[id2] ) continue;

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
/*
    if ( ordered_1->data->has_b != ordered_2->data->has_b ) {
        return false;
    }
    if ( ordered_1->data->has_b_dagger != ordered_2->data->has_b_dagger ) {
        return false;
    }
*/

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

            // t1 vs t2 vs t3, etc?
            if ( ordered_1->data->t_amplitudes[ii].size() != ordered_2->data->t_amplitudes[jj].size() ) continue;

            // need to carefully consider if this works for t3 or higher (i doubt it does so just return false...)
            if ( ordered_1->data->t_amplitudes[ii].size() >= 6 ) return false;

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

    // u_amplitudes, which can be complicated since they aren't sorted

    // same number of u_amplitudes?
    if ( ordered_1->data->u_amplitudes.size() != ordered_2->data->u_amplitudes.size() ) return false;
    
    int nsame_u_amps = 0;
    for (int ii = 0; ii < (int)ordered_1->data->u_amplitudes.size(); ii++) {
        for (int jj = 0; jj < (int)ordered_2->data->u_amplitudes.size(); jj++) {

            // u1 vs u2?
            if ( ordered_1->data->u_amplitudes[ii].size() != ordered_2->data->u_amplitudes[jj].size() ) continue;

            // indices?
            int nsame_idx = 0;
            for (int iii = 0; iii < (int)ordered_1->data->u_amplitudes[ii].size(); iii++) {
                for (int jjj = 0; jjj < (int)ordered_2->data->u_amplitudes[jj].size(); jjj++) {
                    if ( ordered_1->data->u_amplitudes[ii][iii] == ordered_2->data->u_amplitudes[jj][jjj] ) {
                        if ( (iii - jjj) % 2 != 0  && iii < jjj ) n_permute++;
                        nsame_idx++;
                        break;
                    }
                }
            }
            // if all indices are the same, the u_amplitudes must be the same, but we need to be careful of permutations
            if ( nsame_idx == (int)ordered_1->data->u_amplitudes[ii].size() ) {
                nsame_u_amps++;
                break;
            }
        }
    }
    if ( nsame_u_amps != (int)ordered_1->data->u_amplitudes.size() ) return false;

    // m_amplitudes, which can be complicated since they aren't sorted

    // same number of m_amplitudes?
    if ( ordered_1->data->m_amplitudes.size() != ordered_2->data->m_amplitudes.size() ) return false;
    
    int nsame_m_amps = 0;
    for (int ii = 0; ii < (int)ordered_1->data->m_amplitudes.size(); ii++) {
        for (int jj = 0; jj < (int)ordered_2->data->m_amplitudes.size(); jj++) {

            // m1 vs m2?
            if ( ordered_1->data->m_amplitudes[ii].size() != ordered_2->data->m_amplitudes[jj].size() ) continue;

            // indices?
            int nsame_idx = 0;
            for (int iii = 0; iii < (int)ordered_1->data->m_amplitudes[ii].size(); iii++) {
                for (int jjj = 0; jjj < (int)ordered_2->data->m_amplitudes[jj].size(); jjj++) {
                    if ( ordered_1->data->m_amplitudes[ii][iii] == ordered_2->data->m_amplitudes[jj][jjj] ) {
                        if ( (iii - jjj) % 2 != 0  && iii < jjj ) n_permute++;
                        nsame_idx++;
                        break;
                    }
                }
            }
            // if all indices are the same, the u_amplitudes must be the same, but we need to be careful of permutations
            if ( nsame_idx == (int)ordered_1->data->m_amplitudes[ii].size() ) {
                nsame_m_amps++;
                break;
            }
        }
    }
    if ( nsame_m_amps != (int)ordered_1->data->m_amplitudes.size() ) return false;

    // s_amplitudes, which can be complicated since they aren't sorted

    // same number of s_amplitudes?
    if ( ordered_1->data->s_amplitudes.size() != ordered_2->data->s_amplitudes.size() ) return false;
    
    int nsame_s_amps = 0;
    for (int ii = 0; ii < (int)ordered_1->data->s_amplitudes.size(); ii++) {
        for (int jj = 0; jj < (int)ordered_2->data->s_amplitudes.size(); jj++) {

            // s1 vs s2?
            if ( ordered_1->data->s_amplitudes[ii].size() != ordered_2->data->s_amplitudes[jj].size() ) continue;

            // indices?
            int nsame_idx = 0;
            for (int iii = 0; iii < (int)ordered_1->data->s_amplitudes[ii].size(); iii++) {
                for (int jjj = 0; jjj < (int)ordered_2->data->s_amplitudes[jj].size(); jjj++) {
                    if ( ordered_1->data->s_amplitudes[ii][iii] == ordered_2->data->s_amplitudes[jj][jjj] ) {
                        if ( (iii - jjj) % 2 != 0  && iii < jjj ) n_permute++;
                        nsame_idx++;
                        break;
                    }
                }
            }
            // if all indices are the same, the s_amplitudes must be the same, but we need to be careful of permutations
            if ( nsame_idx == (int)ordered_1->data->s_amplitudes[ii].size() ) {
                nsame_s_amps++;
                break;
            }
        }
    }
    if ( nsame_s_amps != (int)ordered_1->data->s_amplitudes.size() ) return false;

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

    // right-hand amplitudes, which can be complicated since they aren't sorted

    // same number of right-hant amplitudes?
    if ( ordered_1->data->right_amplitudes.size() != ordered_2->data->right_amplitudes.size() ) return false;
    
    int nsame_right_amps = 0;
    for (int ii = 0; ii < (int)ordered_1->data->right_amplitudes.size(); ii++) {
        for (int jj = 0; jj < (int)ordered_2->data->right_amplitudes.size(); jj++) {

            // r1 vs r2?
            if ( ordered_1->data->right_amplitudes[ii].size() != ordered_2->data->right_amplitudes[jj].size() ) continue;

            // indices?
            int nsame_idx = 0;
            for (int iii = 0; iii < (int)ordered_1->data->right_amplitudes[ii].size(); iii++) {
                for (int jjj = 0; jjj < (int)ordered_2->data->right_amplitudes[jj].size(); jjj++) {
                    if ( ordered_1->data->right_amplitudes[ii][iii] == ordered_2->data->right_amplitudes[jj][jjj] ) {
                        if ( (iii - jjj) % 2 != 0  && iii < jjj ) n_permute++;
                        nsame_idx++;
                        break;
                    }
                }
            }
            // if all indices are the same, the right-hand amplitudes must be the same, but we need to be careful of permutations
            if ( nsame_idx == (int)ordered_1->data->right_amplitudes[ii].size() ) {
                nsame_right_amps++;
                break;
            }
        }
    }
    if ( nsame_right_amps != (int)ordered_1->data->right_amplitudes.size() ) return false;


    //printf("same right-hand amps\n");
    //if ( (n_permute % 2) != 0 ) continue;

    // are tensors same?
    //if ( ordered_1->data->tensor.size() != ordered_2->data->tensor.size() ) return false;
    if ( ordered_1->data->tensor_type != ordered_2->data->tensor_type ) return false;

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
    for (int i = 0; i < (int)in->data->tensor.size(); i++) {
        data->tensor.push_back(in->data->tensor[i]);
    }

    // data->tensor_type
    data->tensor_type = in->data->tensor_type;

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

    // u_amplitudes
    for (int i = 0; i < (int)in->data->u_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)in->data->u_amplitudes[i].size(); j++) {
            tmp.push_back(in->data->u_amplitudes[i][j]);
        }
        data->u_amplitudes.push_back(tmp);
    }

    // m_amplitudes
    for (int i = 0; i < (int)in->data->m_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)in->data->m_amplitudes[i].size(); j++) {
            tmp.push_back(in->data->m_amplitudes[i][j]);
        }
        data->m_amplitudes.push_back(tmp);
    }

    // s_amplitudes
    for (int i = 0; i < (int)in->data->s_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)in->data->s_amplitudes[i].size(); j++) {
            tmp.push_back(in->data->s_amplitudes[i][j]);
        }
        data->s_amplitudes.push_back(tmp);
    }

    // left-hand amplitudes
    for (int i = 0; i < (int)in->data->left_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)in->data->left_amplitudes[i].size(); j++) {
            tmp.push_back(in->data->left_amplitudes[i][j]);
        }
        data->left_amplitudes.push_back(tmp);
    }

    // right-hand amplitudes
    for (int i = 0; i < (int)in->data->right_amplitudes.size(); i++) {
        std::vector<std::string> tmp;
        for (int j = 0; j < (int)in->data->right_amplitudes[i].size(); j++) {
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

    if ( index_in_tensor(idx) ) {
        n++;
    }else if ( index_in_t_amplitudes(idx) ) {
        n++;
    }else if ( index_in_u_amplitudes(idx) ) {
        n++;
    }else if ( index_in_m_amplitudes(idx) ) {
        n++;
    }else if ( index_in_s_amplitudes(idx) ) {
        n++;
    }else if ( index_in_left_amplitudes(idx) ) {
        n++;
    }else if ( index_in_right_amplitudes(idx) ) {
        n++;
    }

    return n;

}

bool pq::index_in_tensor(std::string idx) {

    for (int i = 0; i < (int)data->tensor.size(); i++) {
        if ( data->tensor[i] == idx ) {
            return true;
        }
    }
    return false;

}

bool pq::index_in_t_amplitudes(std::string idx) {

    for (int i = 0; i < (int)data->t_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->t_amplitudes[i].size(); j++) {
            if ( data->t_amplitudes[i][j] == idx ) {
                return true;
            }
           
        }
    }
    return false;

}

bool pq::index_in_u_amplitudes(std::string idx) {

    for (int i = 0; i < (int)data->u_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->u_amplitudes[i].size(); j++) {
            if ( data->u_amplitudes[i][j] == idx ) {
                return true;
            }
           
        }
    }
    return false;

}

bool pq::index_in_m_amplitudes(std::string idx) {

    for (int i = 0; i < (int)data->m_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->m_amplitudes[i].size(); j++) {
            if ( data->m_amplitudes[i][j] == idx ) {
                return true;
            }
           
        }
    }
    return false;

}

bool pq::index_in_s_amplitudes(std::string idx) {

    for (int i = 0; i < (int)data->s_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->s_amplitudes[i].size(); j++) {
            if ( data->s_amplitudes[i][j] == idx ) {
                return true;
            }
           
        }
    }
    return false;

}

bool pq::index_in_left_amplitudes(std::string idx) {

    for (int i = 0; i < (int)data->left_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->left_amplitudes[i].size(); j++) {
            if ( data->left_amplitudes[i][j] == idx ) {
                return true;
            }
           
        }
    }
    return false;

}

bool pq::index_in_right_amplitudes(std::string idx) {

    for (int i = 0; i < (int)data->right_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->right_amplitudes[i].size(); j++) {
            if ( data->right_amplitudes[i][j] == idx ) {
                return true;
            }
           
        }
    }
    return false;

}

void pq::replace_index_everywhere(std::string old_idx, std::string new_idx) {

    replace_index_in_tensor(old_idx,new_idx);
    replace_index_in_t_amplitudes(old_idx,new_idx);
    replace_index_in_u_amplitudes(old_idx,new_idx);
    replace_index_in_m_amplitudes(old_idx,new_idx);
    replace_index_in_s_amplitudes(old_idx,new_idx);
    replace_index_in_left_amplitudes(old_idx,new_idx);
    replace_index_in_right_amplitudes(old_idx,new_idx);

}

void pq::replace_index_in_tensor(std::string old_idx, std::string new_idx) {

    for (int i = 0; i < (int)data->tensor.size(); i++) {
        if ( data->tensor[i] == old_idx ) {
            data->tensor[i] = new_idx;
            // dont' return because indices may be repeated in two-electron integrals
            //return;
        }
    }

}

void pq::replace_index_in_t_amplitudes(std::string old_idx, std::string new_idx) {

    for (int i = 0; i < (int)data->t_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->t_amplitudes[i].size(); j++) {
            if ( data->t_amplitudes[i][j] == old_idx ) {
                data->t_amplitudes[i][j] = new_idx;
                return; 
            }
        }
    }

}

void pq::replace_index_in_u_amplitudes(std::string old_idx, std::string new_idx) {

    for (int i = 0; i < (int)data->u_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->u_amplitudes[i].size(); j++) {
            if ( data->u_amplitudes[i][j] == old_idx ) {
                data->u_amplitudes[i][j] = new_idx;
                return; 
            }
        }
    }

}

void pq::replace_index_in_m_amplitudes(std::string old_idx, std::string new_idx) {

    for (int i = 0; i < (int)data->m_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->m_amplitudes[i].size(); j++) {
            if ( data->m_amplitudes[i][j] == old_idx ) {
                data->m_amplitudes[i][j] = new_idx;
                return; 
            }
        }
    }

}

void pq::replace_index_in_s_amplitudes(std::string old_idx, std::string new_idx) {

    for (int i = 0; i < (int)data->s_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->s_amplitudes[i].size(); j++) {
            if ( data->s_amplitudes[i][j] == old_idx ) {
                data->s_amplitudes[i][j] = new_idx;
                return; 
            }
        }
    }

}

void pq::replace_index_in_left_amplitudes(std::string old_idx, std::string new_idx) {

    for (int i = 0; i < (int)data->left_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->left_amplitudes[i].size(); j++) {
            if ( data->left_amplitudes[i][j] == old_idx ) {
                data->left_amplitudes[i][j] = new_idx;
                return; 
            }
        }
    }

}

void pq::replace_index_in_right_amplitudes(std::string old_idx, std::string new_idx) {

    for (int i = 0; i < (int)data->right_amplitudes.size(); i++) {
        for (int j = 0; j < (int)data->right_amplitudes[i].size(); j++) {
            if ( data->right_amplitudes[i][j] == old_idx ) {
                data->right_amplitudes[i][j] = new_idx;
                return; 
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

    for (int i = 0; i < (int)occ_in.size(); i++) {

        if ( index_in_anywhere(occ_in[i]) > 0 ) {

            for (int j = 0; j < (int)occ_out.size(); j++) {

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

    for (int i = 0; i < (int)vir_in.size(); i++) {

        if ( index_in_anywhere(vir_in[i]) > 0 ) {

            for (int j = 0; j < (int)vir_out.size(); j++) {

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

    for (int i = 0; i < (int)delta1.size(); i++) {

        bool delta1_in_tensor           = index_in_tensor( delta1[i] );
        bool delta2_in_tensor           = index_in_tensor( delta2[i] );
        bool delta1_in_t_amplitudes     = index_in_t_amplitudes( delta1[i] );
        bool delta2_in_t_amplitudes     = index_in_t_amplitudes( delta2[i] );
        bool delta1_in_left_amplitudes  = index_in_left_amplitudes( delta1[i] );
        bool delta2_in_left_amplitudes  = index_in_left_amplitudes( delta2[i] );
        bool delta1_in_right_amplitudes = index_in_right_amplitudes( delta1[i] );
        bool delta2_in_right_amplitudes = index_in_right_amplitudes( delta2[i] );
        bool delta1_in_u_amplitudes     = index_in_u_amplitudes( delta1[i] );
        bool delta2_in_u_amplitudes     = index_in_u_amplitudes( delta2[i] );
        bool delta1_in_m_amplitudes     = index_in_m_amplitudes( delta1[i] );
        bool delta2_in_m_amplitudes     = index_in_m_amplitudes( delta2[i] );
        bool delta1_in_s_amplitudes     = index_in_s_amplitudes( delta1[i] );
        bool delta2_in_s_amplitudes     = index_in_s_amplitudes( delta2[i] );

        if ( delta1_in_tensor ) {
            replace_index_in_tensor( delta1[i], delta2[i] );
            continue;
        }else if ( delta2_in_tensor ) {
                replace_index_in_tensor( delta2[i], delta1[i] );
            continue;
        }else if ( delta1_in_t_amplitudes ) {
            replace_index_in_t_amplitudes( delta1[i], delta2[i] );
            continue;
        }else if ( delta2_in_t_amplitudes ) {
            replace_index_in_t_amplitudes( delta2[i], delta1[i] );
            continue;
        }else if ( delta1_in_left_amplitudes ) {
            replace_index_in_left_amplitudes( delta1[i], delta2[i] );
            continue;
        }else if ( delta2_in_left_amplitudes ) {
            replace_index_in_left_amplitudes( delta2[i], delta1[i] );
            continue;
        }else if ( delta1_in_right_amplitudes ) {
            replace_index_in_right_amplitudes( delta1[i], delta2[i] );
            continue;
        }else if ( delta2_in_right_amplitudes ) {
            replace_index_in_right_amplitudes( delta2[i], delta1[i] );
            continue;
        }else if ( delta1_in_u_amplitudes ) {
            replace_index_in_u_amplitudes( delta1[i], delta2[i] );
            continue;
        }else if ( delta2_in_u_amplitudes ) {
            replace_index_in_u_amplitudes( delta2[i], delta1[i] );
            continue;
        }else if ( delta1_in_m_amplitudes ) {
            replace_index_in_m_amplitudes( delta1[i], delta2[i] );
            continue;
        }else if ( delta2_in_m_amplitudes ) {
            replace_index_in_m_amplitudes( delta2[i], delta1[i] );
            continue;
        }else if ( delta1_in_s_amplitudes ) {
            replace_index_in_s_amplitudes( delta1[i], delta2[i] );
            continue;
        }else if ( delta2_in_s_amplitudes ) {
            replace_index_in_s_amplitudes( delta2[i], delta1[i] );
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

}

// copy all data, including symbols and daggers
void pq::copy(void * copy_me) { 

    shallow_copy(copy_me);

    pq * in = reinterpret_cast<pq * >(copy_me);

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

    // boson daggers
    for (int i = 0; i < (int)in->data->is_boson_dagger.size(); i++) {
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

    // now, s1 and s2 are closer to normal order in the fermion space
    // we should more toward normal order in the boson space, too

    if ( is_boson_normal_order() ) {

        // copy boson daggers 
        for (int i = 0; i < (int)data->is_boson_dagger.size(); i++) {
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

        for (int i = 0; i < (int)data->is_boson_dagger.size() - 1; i++) {

            // swap operators?
            bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

            if ( swap ) {

                // nothing happens to s1a. add swapped operators to s1b
                s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                // push remaining operators onto s1a and s1b
                for (int j = i+2; j < (int)data->is_boson_dagger.size(); j++) {

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

        for (int i = 0; i < (int)data->is_boson_dagger.size() - 1; i++) {

            // swap operators?
            bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

            if ( swap ) {

                // nothing happens to s2a. add swapped operators to s2b
                s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                // push remaining operators onto s2a and s2b
                for (int j = i+2; j < (int)data->is_boson_dagger.size(); j++) {

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

            for (int j = i+2; j < (int)symbol.size(); j++) {

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
                for (int i = 0; i < (int)data->is_boson_dagger.size(); i++) {
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

            for (int i = 0; i < (int)data->is_boson_dagger.size() - 1; i++) {

                // swap operators?
                bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

                if ( swap ) {

                    // nothing happens to s1a. add swapped operators to s1b
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                    // push remaining operators onto s1a and s1b
                    for (int j = i+2; j < (int)data->is_boson_dagger.size(); j++) {
        
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
                for (int i = 0; i < (int)data->is_boson_dagger.size(); i++) {
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

            for (int i = 0; i < (int)data->is_boson_dagger.size() - 1; i++) {

                // swap operators?
                bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

                if ( swap ) {

                    // nothing happens to s1a. add swapped operators to s1b
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                    s1b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                    // push remaining operators onto s1a and s1b
                    for (int j = i+2; j < (int)data->is_boson_dagger.size(); j++) {

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

            for (int i = 0; i < (int)data->is_boson_dagger.size() - 1; i++) {

                // swap operators?
                bool swap = ( !data->is_boson_dagger[i] && data->is_boson_dagger[i+1] );

                if ( swap ) {

                    // nothing happens to s2a. add swapped operators to s2b
                    s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i+1]);
                    s2b->data->is_boson_dagger.push_back(data->is_boson_dagger[i]);

                    // push remaining operators onto s2a and s2b
                    for (int j = i+2; j < (int)data->is_boson_dagger.size(); j++) {

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
        std::vector<std::string> occ_out{"i","j","k","l","i0","i1","i2","i3","i4","i5","i6","i7","i8","i9"};
        std::string idx;

        int skip = -999;

        for (int i = 0; i < (int)occ_out.size(); i++) {
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

