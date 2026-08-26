//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: pq_swap_operators.cc
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

#include "pq_tensor.h"
#include "pq_string.h"
#include "pq_utils.h"

namespace pdaggerq {

bool swap_operators_fermi_vacuum(const std::shared_ptr<pq_string> &in, std::vector<std::shared_ptr<pq_string> > &ordered, bool keep_operators) {

    if ( in->skip ) return true;

    if ( in->is_normal_order(keep_operators) ) {

        // push current ordered operator onto running list
        ordered.push_back(in);
        return true;
    }

    // new strings
    std::shared_ptr<pq_string> s1 = std::make_shared<pq_string>(in.get(), false);
    std::shared_ptr<pq_string> s2 = std::make_shared<pq_string>(in.get(), false);

    // rearrange operators

    int n_new_strings = 1;
    for (int i = 0; i < (int)in->symbol.size()-1; i++) {

        bool swap = ( !in->is_dagger_fermi[i] && in->is_dagger_fermi[i+1] );

        // four cases: **, --, *-, -*
        // **, --: change sign, swap labels
        // *-, -*: standard swap

        bool daggers_differ = ( in->is_dagger[i] != in->is_dagger[i+1] );

        // operators of different species (e.g. electron vs nuclear) live in
        // disjoint orbital spaces: they never contract, and -- in the commuting
        // convention for distinct particle types -- they swap without a sign
        // change. only same-species creator/annihilator pairs contract.
        bool same_species = ( is_nuclear(in->symbol[i]) == is_nuclear(in->symbol[i+1]) );
        bool can_contract = daggers_differ && same_species;

        if ( swap && can_contract ) {

            // we're going to have two new strings
            n_new_strings = 2;

            // delta function
            delta_functions deltas;
            deltas.labels.push_back(in->symbol[i]);
            deltas.labels.push_back(in->symbol[i+1]);
            deltas.sort();
            s1->deltas.push_back(deltas);

            s2->sign = -s2->sign;
            s2->symbol.push_back(in->symbol[i+1]);
            s2->symbol.push_back(in->symbol[i]);
            s2->is_dagger.push_back(in->is_dagger[i+1]);
            s2->is_dagger.push_back(in->is_dagger[i]);
            s2->is_dagger_fermi.push_back(in->is_dagger_fermi[i+1]);
            s2->is_dagger_fermi.push_back(in->is_dagger_fermi[i]);

            for (size_t j = i+2; j < in->symbol.size(); j++) {

                s1->symbol.push_back(in->symbol[j]);
                s2->symbol.push_back(in->symbol[j]);

                s1->is_dagger.push_back(in->is_dagger[j]);
                s2->is_dagger.push_back(in->is_dagger[j]);

                s1->is_dagger_fermi.push_back(in->is_dagger_fermi[j]);
                s2->is_dagger_fermi.push_back(in->is_dagger_fermi[j]);

            }
            break;

        }else if ( swap )  {

            // a swap with no contraction: same-species same-kind pairs (** or --)
            // anticommute (sign change); distinct-species pairs commute (no change)
            n_new_strings = 1;

            if ( same_species ) s1->sign = -s1->sign;
            s1->symbol.push_back(in->symbol[i+1]);
            s1->symbol.push_back(in->symbol[i]);
            s1->is_dagger.push_back(in->is_dagger[i+1]);
            s1->is_dagger.push_back(in->is_dagger[i]);
            s1->is_dagger_fermi.push_back(in->is_dagger_fermi[i+1]);
            s1->is_dagger_fermi.push_back(in->is_dagger_fermi[i]);

            for (size_t j = i+2; j < in->symbol.size(); j++) {

                s1->symbol.push_back(in->symbol[j]);
                s1->is_dagger.push_back(in->is_dagger[j]);
                s1->is_dagger_fermi.push_back(in->is_dagger_fermi[j]);

            }
            break;

        }else {

            s1->symbol.push_back(in->symbol[i]);
            s2->symbol.push_back(in->symbol[i]);

            s1->is_dagger.push_back(in->is_dagger[i]);
            s2->is_dagger.push_back(in->is_dagger[i]);

            s1->is_dagger_fermi.push_back(in->is_dagger_fermi[i]);
            s2->is_dagger_fermi.push_back(in->is_dagger_fermi[i]);

        }
    }

    // now, s1 (and s2) are closer to normal order in the fermion space
    // we should more toward normal order in the boson space, too

    if ( n_new_strings == 1 ) {

        if ( in->is_boson_normal_order(keep_operators) ) {
            if ( !in->skip ) {
                // copy boson daggers
                for (size_t i = 0; i < in->is_boson_dagger.size(); i++) {
                    s1->is_boson_dagger.push_back(in->is_boson_dagger[i]);
                }
                ordered.push_back(s1);
                return false;
            }
        }else {

            // new strings
            std::shared_ptr<pq_string> s1a = std::make_shared<pq_string>(*s1);
            std::shared_ptr<pq_string> s1b = std::make_shared<pq_string>(*s1);

            // ensure boson daggers are clear (they should be anyway)
            s1a->is_boson_dagger.clear();
            s1b->is_boson_dagger.clear();

            for (int i = 0; i < (int)in->is_boson_dagger.size()-1; i++) {

                // swap operators?
                bool swap = ( !in->is_boson_dagger[i] && in->is_boson_dagger[i+1] );

                if ( swap ) {

                    // nothing happens to s1a. add swapped operators to s1b
                    s1b->is_boson_dagger.push_back(in->is_boson_dagger[i+1]);
                    s1b->is_boson_dagger.push_back(in->is_boson_dagger[i]);

                    // push remaining operators onto s1a and s1b
                    for (size_t j = i+2; j < in->is_boson_dagger.size(); j++) {

                        s1a->is_boson_dagger.push_back(in->is_boson_dagger[j]);
                        s1b->is_boson_dagger.push_back(in->is_boson_dagger[j]);
                    }
                    break;

                }else {

                    s1a->is_boson_dagger.push_back(in->is_boson_dagger[i]);
                    s1b->is_boson_dagger.push_back(in->is_boson_dagger[i]);
                }
            }
            ordered.push_back(s1a);
            ordered.push_back(s1b);
            return false;
        }

    }else if ( n_new_strings == 2 ) {

        if ( in->is_boson_normal_order(keep_operators) ) {
            if ( !in->skip ) {
                // copy boson daggers
                for (size_t i = 0; i < in->is_boson_dagger.size(); i++) {
                    s1->is_boson_dagger.push_back(in->is_boson_dagger[i]);
                    s2->is_boson_dagger.push_back(in->is_boson_dagger[i]);
                }
                ordered.push_back(s1);
                ordered.push_back(s2);
                return false;
            }
        }else {

            // new strings
            std::shared_ptr<pq_string> s1a = std::make_shared<pq_string>(*s1);
            std::shared_ptr<pq_string> s1b = std::make_shared<pq_string>(*s1);
            std::shared_ptr<pq_string> s2a = std::make_shared<pq_string>(*s2);
            std::shared_ptr<pq_string> s2b = std::make_shared<pq_string>(*s2);

            // ensure boson daggers are clear (they should be anyway)
            s1a->is_boson_dagger.clear();
            s1b->is_boson_dagger.clear();

            for (int i = 0; i < (int)in->is_boson_dagger.size()-1; i++) {

                // swap operators?
                bool swap = ( !in->is_boson_dagger[i] && in->is_boson_dagger[i+1] );

                if ( swap ) {

                    // nothing happens to s1a. add swapped operators to s1b
                    s1b->is_boson_dagger.push_back(in->is_boson_dagger[i+1]);
                    s1b->is_boson_dagger.push_back(in->is_boson_dagger[i]);

                    // push remaining operators onto s1a and s1b
                    for (size_t j = i+2; j < in->is_boson_dagger.size(); j++) {

                        s1a->is_boson_dagger.push_back(in->is_boson_dagger[j]);
                        s1b->is_boson_dagger.push_back(in->is_boson_dagger[j]);
                    }
                    break;

                }else {

                    s1a->is_boson_dagger.push_back(in->is_boson_dagger[i]);
                    s1b->is_boson_dagger.push_back(in->is_boson_dagger[i]);
                }
            }

            // ensure boson daggers are clear (they should be anyway)
            s2a->is_boson_dagger.clear();
            s2b->is_boson_dagger.clear();

            for (int i = 0; i < in->is_boson_dagger.size()-1; i++) {

                // swap operators?
                bool swap = ( !in->is_boson_dagger[i] && in->is_boson_dagger[i+1] );

                if ( swap ) {

                    // nothing happens to s2a. add swapped operators to s2b
                    s2b->is_boson_dagger.push_back(in->is_boson_dagger[i+1]);
                    s2b->is_boson_dagger.push_back(in->is_boson_dagger[i]);

                    // push remaining operators onto s2a and s2b
                    for (size_t j = i+2; j < in->is_boson_dagger.size(); j++) {

                        s2a->is_boson_dagger.push_back(in->is_boson_dagger[j]);
                        s2b->is_boson_dagger.push_back(in->is_boson_dagger[j]);
                    }
                    break;

                }else {

                    s2a->is_boson_dagger.push_back(in->is_boson_dagger[i]);
                    s2b->is_boson_dagger.push_back(in->is_boson_dagger[i]);
                }
            }

            ordered.push_back(s1a);
            ordered.push_back(s1b);
            ordered.push_back(s2a);
            ordered.push_back(s2b);
            return false;
        }
    }
    return false;
}

bool swap_operators_fermi_vacuum_no_deltas(std::shared_ptr<pq_string> &in) {

    if ( in->skip ) return true;

    // don't call this function here because it will set the "skip" flag
    //if ( in->is_normal_order() ) {
    //    return true;
    //}

    // rearrange operators

    bool am_i_done = true;

    for (int i = 0; i < (int)in->symbol.size()-1; i++) {

        bool swap = ( !in->is_dagger_fermi[i] && in->is_dagger_fermi[i+1] );

        if ( swap ) {

            in->sign = -in->sign;

            std::string tmp_symbol = in->symbol[i];
            in->symbol[i] = in->symbol[i+1];
            in->symbol[i+1] = tmp_symbol;

            bool tmp_is_dagger = in->is_dagger[i];
            in->is_dagger[i] = in->is_dagger[i+1];
            in->is_dagger[i+1] = tmp_is_dagger;

            bool tmp_is_dagger_fermi = in->is_dagger_fermi[i];
            in->is_dagger_fermi[i] = in->is_dagger_fermi[i+1];
            in->is_dagger_fermi[i+1] = tmp_is_dagger_fermi;

            am_i_done = false;
        }
    }

    // now, the operator is closer to normal order in the fermion space
    // we should more toward normal order in the boson space, too

    // don't call this function here because it will set the "skip" flag
    //if ( in->is_boson_normal_order() ) {
    //    return false;
    //}

    for (int i = 0; i < (int)in->is_boson_dagger.size()-1; i++) {

        // swap operators?
        bool swap = ( !in->is_boson_dagger[i] && in->is_boson_dagger[i+1] );

        if ( swap ) {
            bool tmp = in->is_boson_dagger[i];
            in->is_boson_dagger[i] = in->is_boson_dagger[i+1];
            in->is_boson_dagger[i+1] = tmp;
            
            am_i_done = false;
        }
    }

    return am_i_done;
}

bool swap_operators_true_vacuum(const std::shared_ptr<pq_string> &in, std::vector<std::shared_ptr<pq_string> > &ordered, bool keep_operators) {

    if ( in->skip ) return true;

    if ( in->is_normal_order(keep_operators) ) {

        // push current ordered operator onto running list
        ordered.push_back(in);

        return true;
    }

    // new strings
    std::shared_ptr<pq_string> s1 = std::make_shared<pq_string>(in.get(), false);
    std::shared_ptr<pq_string> s2 = std::make_shared<pq_string>(in.get(), false);

    // rearrange operators
    for (int i = 0; i < (int)in->symbol.size()-1; i++) {

        bool swap = ( !in->is_dagger[i] && in->is_dagger[i+1] );

        if ( swap ) {

            delta_functions deltas;
            deltas.labels.push_back(in->symbol[i]);
            deltas.labels.push_back(in->symbol[i+1]);
            deltas.sort();
            s1->deltas.push_back(deltas);

            s2->sign = -s2->sign;
            s2->symbol.push_back(in->symbol[i+1]);
            s2->symbol.push_back(in->symbol[i]);
            s2->is_dagger.push_back(in->is_dagger[i+1]);
            s2->is_dagger.push_back(in->is_dagger[i]);

            for (size_t j = i+2; j < in->symbol.size(); j++) {

                s1->symbol.push_back(in->symbol[j]);
                s2->symbol.push_back(in->symbol[j]);

                s1->is_dagger.push_back(in->is_dagger[j]);
                s2->is_dagger.push_back(in->is_dagger[j]);
            }
            break;

        }else {

            s1->symbol.push_back(in->symbol[i]);
            s2->symbol.push_back(in->symbol[i]);

            s1->is_dagger.push_back(in->is_dagger[i]);
            s2->is_dagger.push_back(in->is_dagger[i]);
        }
    }

    // now, s1 and s2 are closer to normal order in the fermion space
    // we should more toward normal order in the boson space, too

    if ( in->is_boson_normal_order(keep_operators) ) {

        // copy boson daggers 
        for (size_t i = 0; i < in->is_boson_dagger.size(); i++) {
            s1->is_boson_dagger.push_back(in->is_boson_dagger[i]);
            s2->is_boson_dagger.push_back(in->is_boson_dagger[i]);
        }
        ordered.push_back(s1);
        ordered.push_back(s2);
        return false;

    }else {

        // new strings
        std::shared_ptr<pq_string> s1a = std::make_shared<pq_string>(*s1);
        std::shared_ptr<pq_string> s1b = std::make_shared<pq_string>(*s1);
        std::shared_ptr<pq_string> s2a = std::make_shared<pq_string>(*s2);
        std::shared_ptr<pq_string> s2b = std::make_shared<pq_string>(*s2);

        // ensure boson daggers are clear (they should be anyway)
        s1a->is_boson_dagger.clear();
        s1b->is_boson_dagger.clear();

        for (int i = 0; i < (int)in->is_boson_dagger.size()-1; i++) {

            // swap operators?
            bool swap = ( !in->is_boson_dagger[i] && in->is_boson_dagger[i+1] );

            if ( swap ) {

                // nothing happens to s1a. add swapped operators to s1b
                s1b->is_boson_dagger.push_back(in->is_boson_dagger[i+1]);
                s1b->is_boson_dagger.push_back(in->is_boson_dagger[i]);

                // push remaining operators onto s1a and s1b
                for (size_t j = i+2; j < in->is_boson_dagger.size(); j++) {

                    s1a->is_boson_dagger.push_back(in->is_boson_dagger[j]);
                    s1b->is_boson_dagger.push_back(in->is_boson_dagger[j]);
                }
                break;

            }else {

                s1a->is_boson_dagger.push_back(in->is_boson_dagger[i]);
                s1b->is_boson_dagger.push_back(in->is_boson_dagger[i]);
            }
        }

        // ensure boson daggers are clear (they should be anyway)
        s2a->is_boson_dagger.clear();
        s2b->is_boson_dagger.clear();

        for (int i = 0; i < (int)in->is_boson_dagger.size()-1; i++) {

            // swap operators?
            bool swap = ( !in->is_boson_dagger[i] && in->is_boson_dagger[i+1] );

            if ( swap ) {

                // nothing happens to s2a. add swapped operators to s2b
                s2b->is_boson_dagger.push_back(in->is_boson_dagger[i+1]);
                s2b->is_boson_dagger.push_back(in->is_boson_dagger[i]);

                // push remaining operators onto s2a and s2b
                for (size_t j = i+2; j < in->is_boson_dagger.size(); j++) {

                    s2a->is_boson_dagger.push_back(in->is_boson_dagger[j]);
                    s2b->is_boson_dagger.push_back(in->is_boson_dagger[j]);

                }
                break;

            }else {

                s2a->is_boson_dagger.push_back(in->is_boson_dagger[i]);
                s2b->is_boson_dagger.push_back(in->is_boson_dagger[i]);
            }
        }

        ordered.push_back(s1a);
        ordered.push_back(s1b);
        ordered.push_back(s2a);
        ordered.push_back(s2b);
        return false;
    }

    return false;
}

}
