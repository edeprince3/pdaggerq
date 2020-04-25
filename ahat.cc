/*
 * @BEGIN LICENSE
 *
 * pdaggerq by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2017 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "psi4/psi4-dec.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"

#include<iostream>
#include<string>
#include<algorithm>

#include "ahat.h"

#include <math.h>

namespace psi{ namespace pdaggerq {

ahat::ahat() {
}
ahat::~ahat() {
}

void ahat::check_occ_vir() {

   // OCC: I,J,K,L,M,N
   // VIR: A,B,C,D,E,F
   // GEN: P,Q,R,S,T,U,V,W

   for (int i = 0; i < (int)delta1.size(); i++ ) {
       bool first_is_occ = false;
       if ( delta1[i].at(0) == 'I') {
           first_is_occ = true;
       }else if ( delta1[i].at(0) == 'J') {
           first_is_occ = true;
       }else if ( delta1[i].at(0) == 'K') {
           first_is_occ = true;
       }else if ( delta1[i].at(0) == 'L') {
           first_is_occ = true;
       }else if ( delta1[i].at(0) == 'M') {
           first_is_occ = true;
       }else if ( delta1[i].at(0) == 'N') {
           first_is_occ = true;
       }else if ( delta1[i].at(0) == 'A') {
           first_is_occ = false;
       }else if ( delta1[i].at(0) == 'B') {
           first_is_occ = false;
       }else if ( delta1[i].at(0) == 'C') {
           first_is_occ = false;
       }else if ( delta1[i].at(0) == 'D') {
           first_is_occ = false;
       }else if ( delta1[i].at(0) == 'E') {
           first_is_occ = false;
       }else if ( delta1[i].at(0) == 'F') {
           first_is_occ = false;
       }else {
           continue;
       }

       bool second_is_occ = false;
       if ( delta2[i].at(0) == 'I') {
           second_is_occ = true;
       }else if ( delta2[i].at(0) == 'J') {
           second_is_occ = true;
       }else if ( delta2[i].at(0) == 'K') {
           second_is_occ = true;
       }else if ( delta2[i].at(0) == 'L') {
           second_is_occ = true;
       }else if ( delta2[i].at(0) == 'M') {
           second_is_occ = true;
       }else if ( delta2[i].at(0) == 'N') {
           second_is_occ = true;
       }else if ( delta2[i].at(0) == 'A') {
           second_is_occ = false;
       }else if ( delta2[i].at(0) == 'B') {
           second_is_occ = false;
       }else if ( delta2[i].at(0) == 'C') {
           second_is_occ = false;
       }else if ( delta2[i].at(0) == 'D') {
           second_is_occ = false;
       }else if ( delta2[i].at(0) == 'E') {
           second_is_occ = false;
       }else if ( delta2[i].at(0) == 'F') {
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

    // check A/B in two-index tensors
    if ( (int)tensor.size() == 2 ) {
        if ( tensor[0].length() == 2 ) {
            if ( tensor[1].length() == 2 ) {

                if ( tensor[0].at(1) == 'A' && tensor[1].at(1) == 'B' ) {
                    skip = true;
                    return;
                }else if ( tensor[0].at(1) == 'B' && tensor[1].at(1) == 'A' ) {
                    skip = true;
                    return;
                }
            
            }
        }
    }

    // check A/B in four-index tensors
    if ( (int)tensor.size() == 4 ) {
        // check bra
        if ( tensor[0].length() == 2 ) {
            if ( tensor[1].length() == 2 ) {

                if ( tensor[0].at(1) == 'A' && tensor[1].at(1) == 'B' ) {
                    skip = true;
                    return;
                }else if ( tensor[0].at(1) == 'B' && tensor[1].at(1) == 'A' ) {
                    skip = true;
                    return;
                }
            
            }
        }
        // check ket
        if ( tensor[2].length() == 2 ) {
            if ( tensor[3].length() == 2 ) {

                if ( tensor[2].at(1) == 'A' && tensor[3].at(1) == 'B' ) {
                    skip = true;
                    return;
                }else if ( tensor[2].at(1) == 'B' && tensor[3].at(1) == 'A' ) {
                    skip = true;
                    return;
                }
            
            }
        }

    }


}

void ahat::print() {
    if ( skip ) return;
    printf("    ");
    printf("//     ");
    printf("%c", sign > 0 ? '+' : '-');
    printf(" ");
    printf("%7.5lf", fabs(factor));
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
    if ( (int)tensor.size() > 0 ) {
        // two-electron integrals
        if ( (int)tensor.size() == 4 ) {
            printf("(");
            for (int i = 0; i < 2; i++) {
                printf("%s",tensor[i].c_str());
            }
            printf("|");
            for (int i = 2; i < 4; i++) {
                printf("%s",tensor[i].c_str());
            }
            printf(")");
        }
        // one-electron integrals
        if ( (int)tensor.size() == 2 ) {
            printf("h(");
            for (int i = 0; i < 2; i++) {
                printf("%s",tensor[i].c_str());
            }
            printf(")");
        }
        printf(" ");
    }
    // amplitudes(1)
    if ( (int)amplitudes1.size() > 0 ) {
        // t1
        if ( (int)amplitudes1.size() == 2 ) {
            printf("t1(");
            for (int i = 0; i < 2; i++) {
                printf("%s",amplitudes1[i].c_str());
            }
            printf(")");
        }
        // t2
        if ( (int)amplitudes1.size() == 4 ) {
            printf("t2(");
            for (int i = 0; i < 4; i++) {
                printf("%s",amplitudes1[i].c_str());
            }
            printf(")");
        }
        printf(" ");
    }
    // amplitudes(2)
    if ( (int)amplitudes2.size() > 0 ) {
        // t1
        if ( (int)amplitudes2.size() == 2 ) {
            printf("t1(");
            for (int i = 0; i < 2; i++) {
                printf("%s",amplitudes2[i].c_str());
            }
            printf(")");
        }
        // t2
        if ( (int)amplitudes2.size() == 4 ) {
            printf("t2(");
            for (int i = 0; i < 4; i++) {
                printf("%s",amplitudes2[i].c_str());
            }
            printf(")");
        }
    }
    printf("\n");
}

bool ahat::is_normal_order() {

    // don't bother bringing to normal order if we're going to skip this string
    if (skip) return true;

    for (int i = 0; i < (int)symbol.size()-1; i++) {
        if ( !is_dagger[i] && is_dagger[i+1] ) {
            return false;
        }
    }
    return true;
}

// in order to compare strings, the creation and annihilation 
// operators should be ordered in some consistent way.
// alphabetically seems reasonable enough
void ahat::alphabetize(std::vector<ahat *> &ordered) {

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

// once strings are alphabetized, we can compare them
// and remove terms that cancel
void ahat::cleanup(std::vector<ahat *> &ordered) {

    for (int i = 0; i < (int)ordered.size(); i++) {

        for (int j = i+1; j < (int)ordered.size(); j++) {
            
            // same factor
            if ( ordered[i]->factor == ordered[j]->factor ) {

                // opposite sign
                if ( ordered[i]->sign == -ordered[j]->sign ) {

                    // same normal-ordered operator
                    if ( ordered[i]->symbol.size() == ordered[j]->symbol.size() ) {
                        int nsame_s = 0;
                        for (int k = 0; k < (int)ordered[i]->symbol.size(); k++) {
                            if ( ordered[i]->symbol[k] == ordered[j]->symbol[k] ) {
                                nsame_s++;
                            }
                        }
                        if ( nsame_s == ordered[i]->symbol.size() ) {
                            // same tensor
                            if ( ordered[i]->tensor.size() == ordered[j]->tensor.size() ) {
                                int nsame_t = 0;
                                for (int k = 0; k < (int)tensor.size(); k++) {
                                    if ( ordered[i]->tensor[k] == ordered[j]->tensor[k] ) {
                                        nsame_t++;
                                    }
                                }
                                if ( nsame_t == ordered[i]->tensor.size() ) {
                                    // same delta functions (recall these aren't sorted in any way)
                                    int nsame_d = 0;
                                    for (int k = 0; k < (int)ordered[i]->delta1.size(); k++) {
                                        for (int l = 0; l < (int)ordered[j]->delta1.size(); l++) {
                                            if ( ordered[i]->delta1[k] == ordered[j]->delta1[l] && ordered[i]->delta2[k] == ordered[j]->delta2[l] ) {
                                                nsame_d++;
                                            }else if ( ordered[i]->delta2[k] == ordered[j]->delta1[l] && ordered[i]->delta1[k] == ordered[j]->delta2[l] ) {
                                                nsame_d++;
                                            }
                                        }
                                    }
                                    if ( nsame_d == (int)ordered[i]->delta1.size() ) {
                                        ordered[i]->skip = true;
                                        ordered[j]->skip = true;
                                    }
                                }
                            }
                        }
                    }

                }

            }
            
        }

    }

}

void ahat::normal_order(std::vector<ahat *> &ordered) {
    if ( skip ) return;

    if ( is_normal_order() ) {

        // push current ordered operator onto running list
        ahat * newguy (new ahat());

        // skip string?
        newguy->skip   = skip;

        // sign
        newguy->sign   = sign;

        // factor
        newguy->factor = factor;

        // dagger?
        for (int j = 0; j < (int)is_dagger.size(); j++) {
            newguy->is_dagger.push_back(is_dagger[j]);
        }

        // operators
        for (int j = 0; j < (int)symbol.size(); j++) {
            newguy->symbol.push_back(symbol[j]);
        }

        // tensor
        for (int j = 0; j < (int)tensor.size(); j++) {
            // does tensor index show up in a delta function?
            bool skipme = false;
            for (int k = 0; k < (int)delta1.size(); k++) {
                if ( tensor[j] == delta1[k] ) {
                    newguy->tensor.push_back(delta2[k]);
                    skipme = true;
                    break;
                }
                if ( tensor[j] == delta2[k] ) {
                    newguy->tensor.push_back(delta1[k]);
                    skipme = true;
                    break;
                }
            }
            if ( skipme ) continue;
            newguy->tensor.push_back(tensor[j]);
        }
        for (int j = 0; j < (int)delta1.size(); j++) {
            bool skipme = false;
            for (int k = 0; k < (int)tensor.size(); k++) {
                if ( tensor[k] == delta1[j] ) {
                    skipme = true;
                    break;
                }
                if ( tensor[k] == delta2[j] ) {
                    skipme = true;
                    break;
                }
            }
            if ( skipme ) continue;

            newguy->delta1.push_back(delta1[j]);
            newguy->delta2.push_back(delta2[j]);
        }

        // amplitudes
        for (int j = 0; j < (int)amplitudes1.size(); j++) {
            newguy->amplitudes1.push_back(amplitudes1[j]);
        }
        // amplitudes
        for (int j = 0; j < (int)amplitudes2.size(); j++) {
            newguy->amplitudes2.push_back(amplitudes2[j]);
        }

        ordered.push_back(newguy);

        return;
    }

    // new strings
    std::shared_ptr<ahat> s1 ( new ahat() );
    std::shared_ptr<ahat> s2 ( new ahat() );

    for (int i = 0; i < (int)tensor.size(); i++) {
        s1->tensor.push_back(tensor[i]);
        s2->tensor.push_back(tensor[i]);
    }
    // amplitudes
    for (int j = 0; j < (int)amplitudes1.size(); j++) {
        s1->amplitudes1.push_back(amplitudes1[j]);
        s2->amplitudes1.push_back(amplitudes1[j]);
    }
    // amplitudes
    for (int j = 0; j < (int)amplitudes2.size(); j++) {
        s1->amplitudes2.push_back(amplitudes2[j]);
        s2->amplitudes2.push_back(amplitudes2[j]);
    }

    s1->skip = skip;
    s2->skip = skip;

    s1->sign = sign;
    s2->sign = sign;

    s1->factor = factor;
    s2->factor = factor;

    for (int i = 0; i < (int)delta1.size(); i++) {
        s1->delta1.push_back(delta1[i]);
        s2->delta1.push_back(delta1[i]);

        s1->delta2.push_back(delta2[i]);
        s2->delta2.push_back(delta2[i]);
    }

    for (int i = 0; i < (int)symbol.size()-1; i++) {

        if ( !is_dagger[i] && is_dagger[i+1] ) {

            s1->delta1.push_back(symbol[i]);
            s1->delta2.push_back(symbol[i+1]);

            // check spin in delta functions
            for (int j = 0; j < (int)delta1.size(); j++) {
                if ( s1->delta1[j].length() != 2 ) {
                    //throw PsiException("be sure to specify spin as second character in labels",__FILE__,__LINE__);
                    break;
                }
                if ( s1->delta1[j].at(1) == 'A' && s1->delta2[j].at(1) == 'B' ) {
                    s1->skip = true;
                }else if ( s1->delta1[j].at(1) == 'B' && s1->delta2[j].at(1) == 'A' ) {
                    s1->skip = true;
                }
            }

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

    s1->normal_order(ordered);
    s2->normal_order(ordered);

}

}} // End namespaces

