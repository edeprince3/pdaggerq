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

#include "pq_cumulant_expansion.h"
#include "pq_string.h"
#include "pq_tensor.h"

namespace pdaggerq {

/// replace rdms with cumulant expansion, ignoring the n-body cumulant
void cumulant_expansion(std::vector<std::shared_ptr<pq_string> > &ordered, std::vector<int> ignore_cumulant_rdms) {

    if ( ignore_cumulant_rdms.size() == 0 ) {
        return;
    }

    // TODO: D4
    for (size_t n = 3; n > 1; n--) {

        if(std::find(ignore_cumulant_rdms.begin(), ignore_cumulant_rdms.end(), n) == ignore_cumulant_rdms.end()) {
            continue;
        }

        bool done_expanding_rdms = false;
        do {

            std::vector< std::shared_ptr<pq_string> > list;

            // add strings that don't contain target rdm to list before starting. otherwise, expand_rdms() will miss them

            for (std::shared_ptr<pq_string> & pq_str : ordered) {

                auto rdm_pos = pq_str->amps.find('D');
                if ( rdm_pos == pq_str->amps.end() ) {

                    // no rdms
                    list.push_back(pq_str);

                }else {

                    std::vector<amplitudes> & rdms = rdm_pos->second;


                    bool found_rdm = false;
                    for (size_t i = 0; i < rdms.size(); i++) {
                        if ( rdms[i].labels.size() == 2 * n ) {
                            found_rdm = true;
                        }
                    }
                    if ( !found_rdm ) {
                        // no target rdm
                        list.push_back(pq_str);
                    }
                }
            }

            done_expanding_rdms = true;
            for (std::shared_ptr<pq_string> & pq_str : ordered) {
                bool am_i_done = expand_rdms(pq_str, list, n);
                if ( !am_i_done ) done_expanding_rdms = false;
            }
            if ( !done_expanding_rdms ) {
                ordered.clear();
                for (std::shared_ptr<pq_string> & pq_str : list) {
                    if ( !pq_str->skip ) {
                        ordered.push_back(pq_str);
                    }
                }
            }
        }while(!done_expanding_rdms);
    }
}

/// expand rdms in an input string using cumulant expansion, ignoring the n-body cumulant
bool expand_rdms(const std::shared_ptr<pq_string>& in, std::vector<std::shared_ptr<pq_string> > &list, int order) {

    if ( in->skip ) return true;
            
    // get rdms
    auto rdm_pos = in->amps.find('D');
    if ( rdm_pos == in->amps.end() ) return true; // no rdms
                
    std::vector<amplitudes> & rdms = rdm_pos->second;
                
    bool found_rdm = false;
    for (size_t i = 0; i < rdms.size(); i++) {
        if ( rdms[i].labels.size() == 2 * order ) {
            found_rdm = true;
        }               
    }                   
    if ( !found_rdm ) {
        return true; // no rdms of correct order
    }
                        
    std::vector<std::shared_ptr<pq_string> > newguys;
    int n_new_terms = 2;
    if ( order == 3 ) { 
        n_new_terms = 15;
    }                       
    for (int i = 0; i < n_new_terms; i++) { 
        std::shared_ptr<pq_string> tmp (new pq_string(in->vacuum));
        tmp->copy(in.get());
        newguys.push_back(tmp);
    } 

    std::vector<amplitudes> new_rdms;
    for (size_t i = 0; i < rdms.size(); i++) {
        if ( rdms[i].labels.size() != 2 * order ) {
            continue;
        }

        std::vector<std::string> upper;
        for (size_t j = 0; j < order; j++) {
            upper.push_back(rdms[i].labels[j]);
        }
        std::vector<std::string> lower;
        for (size_t j = 0; j < order; j++) {
            lower.push_back(rdms[i].labels[2 * order - j - 1]);
        }

        if ( order == 2 ) {

            std::vector<std::string> lab = rdms[i].labels;

            // d2("0123")

            // term 0: + d1("02") * d2("13");

            // overwrite rdm
            newguys[0]->amps['D'][i].labels = {lab[0], lab[2]};
            newguys[0]->amps['D'][i].n_create = 1;
            newguys[0]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[0]->set_amplitudes('D', 1, 1, {lab[1], lab[3]});

            // term 1: - d1("03") * d2("12");

            // overwrite rdm
            newguys[1]->amps['D'][i].labels = {lab[0], lab[3]};
            newguys[1]->amps['D'][i].n_create = 1;
            newguys[1]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[1]->set_amplitudes('D', 1, 1, {lab[1], lab[2]});

            // adjust sign
            newguys[1]->sign *= -1;

        }else if ( order == 3 ) {

            std::vector<std::string> lab = rdms[i].labels;

            // d3("012345")

            // term 0: + d1("03") * d2("1245");

            // overwrite rdm
            newguys[0]->amps['D'][i].labels = {lab[0], lab[3]};
            newguys[0]->amps['D'][i].n_create = 1;
            newguys[0]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[0]->set_amplitudes('D', 2, 2, {lab[1], lab[2], lab[4], lab[5]});

            // term 1: - d1("04") * d2("1235");

            // overwrite rdm
            newguys[1]->amps['D'][i].labels = {lab[0], lab[4]};
            newguys[1]->amps['D'][i].n_create = 1;
            newguys[1]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[1]->set_amplitudes('D', 2, 2, {lab[1], lab[2], lab[3], lab[5]});

            // adjust sign
            newguys[1]->sign *= -1;

            // term 2: - d1("05") * d2("1243");

            // overwrite rdm
            newguys[2]->amps['D'][i].labels = {lab[0], lab[5]};
            newguys[2]->amps['D'][i].n_create = 1;
            newguys[2]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[2]->set_amplitudes('D', 2, 2, {lab[1], lab[2], lab[3], lab[4]});

            // adjust sign
            newguys[2]->sign *= -1;

            // term 3: + d1("14") * d2("0235");

            // overwrite rdm
            newguys[3]->amps['D'][i].labels = {lab[1], lab[4]};
            newguys[3]->amps['D'][i].n_create = 1;
            newguys[3]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[3]->set_amplitudes('D', 2, 2, {lab[0], lab[2], lab[3], lab[5]});

            // term 4: - d1("13") * d2("0245");

            // overwrite rdm
            newguys[4]->amps['D'][i].labels = {lab[1], lab[3]};
            newguys[4]->amps['D'][i].n_create = 1;
            newguys[4]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[4]->set_amplitudes('D', 2, 2, {lab[0], lab[2], lab[4], lab[5]});

            // adjust sign
            newguys[4]->sign *= -1;

            // term 5: - d1("15") * d2("0234");

            // overwrite rdm
            newguys[5]->amps['D'][i].labels = {lab[1], lab[5]};
            newguys[5]->amps['D'][i].n_create = 1;
            newguys[5]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[5]->set_amplitudes('D', 2, 2, {lab[0], lab[2], lab[3], lab[4]});

            // adjust sign
            newguys[5]->sign *= -1;

            // term 6: + d1("25") * d2("0134");

            // overwrite rdm
            newguys[6]->amps['D'][i].labels = {lab[2], lab[5]};
            newguys[6]->amps['D'][i].n_create = 1;
            newguys[6]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[6]->set_amplitudes('D', 2, 2, {lab[0], lab[1], lab[3], lab[4]});

            // term 7: - d1("23") * d2("0154");

            // overwrite rdm
            newguys[7]->amps['D'][i].labels = {lab[2], lab[3]};
            newguys[7]->amps['D'][i].n_create = 1;
            newguys[7]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[7]->set_amplitudes('D', 2, 2, {lab[0], lab[1], lab[5], lab[4]});

            // adjust sign
            newguys[7]->sign *= -1;

            // term 8: - d1("24") * d2("0135");

            // overwrite rdm
            newguys[8]->amps['D'][i].labels = {lab[2], lab[4]};
            newguys[8]->amps['D'][i].n_create = 1;
            newguys[8]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[8]->set_amplitudes('D', 2, 2, {lab[0], lab[1], lab[3], lab[5]});

            // adjust sign
            newguys[8]->sign *= -1;

            // term 9: - 2.0 * d1("03") * d1("14") * d1("25");

            // overwrite rdm
            newguys[9]->amps['D'][i].labels = {lab[0], lab[3]};
            newguys[9]->amps['D'][i].n_create = 1;
            newguys[9]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[9]->set_amplitudes('D', 1, 1, {lab[1], lab[4]});

            // and another
            newguys[9]->set_amplitudes('D', 1, 1, {lab[2], lab[5]});

            // adjust sign
            newguys[9]->sign *= -1;

            // adjust factor
            newguys[9]->factor *= 2.0;

            // term 10: - 2.0 * d1("04") * d1("15") * d1("23");

            // overwrite rdm
            newguys[10]->amps['D'][i].labels = {lab[0], lab[4]};
            newguys[10]->amps['D'][i].n_create = 1;
            newguys[10]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[10]->set_amplitudes('D', 1, 1, {lab[1], lab[5]});

            // and another
            newguys[10]->set_amplitudes('D', 1, 1, {lab[2], lab[3]});

            // adjust sign
            newguys[10]->sign *= -1;

            // adjust factor
            newguys[10]->factor *= 2.0;

            // term 11: - 2.0 * d1("05") * d1("13") * d1("24");

            // overwrite rdm
            newguys[11]->amps['D'][i].labels = {lab[0], lab[5]};
            newguys[11]->amps['D'][i].n_create = 1;
            newguys[11]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[11]->set_amplitudes('D', 1, 1, {lab[1], lab[3]});

            // and another
            newguys[11]->set_amplitudes('D', 1, 1, {lab[2], lab[4]});

            // adjust sign
            newguys[11]->sign *= -1;

            // adjust factor
            newguys[11]->factor *= 2.0;

            // term 12: + 2.0 * d1("03") * d1("15") * d1("24");

            // overwrite rdm
            newguys[12]->amps['D'][i].labels = {lab[0], lab[3]};
            newguys[12]->amps['D'][i].n_create = 1;
            newguys[12]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[12]->set_amplitudes('D', 1, 1, {lab[1], lab[5]});

            // and another
            newguys[12]->set_amplitudes('D', 1, 1, {lab[2], lab[4]});

            // adjust factor
            newguys[12]->factor *= 2.0;

            // term 13: + 2.0 * d1("05") * d1("14") * d1("23");

            // overwrite rdm
            newguys[13]->amps['D'][i].labels = {lab[0], lab[5]};
            newguys[13]->amps['D'][i].n_create = 1;
            newguys[13]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[13]->set_amplitudes('D', 1, 1, {lab[1], lab[4]});

            // and another
            newguys[13]->set_amplitudes('D', 1, 1, {lab[2], lab[3]});

            // adjust factor
            newguys[13]->factor *= 2.0;

            // term 14: + 2.0 * d1("04") * d1("13") * d1("25");

            // overwrite rdm
            newguys[14]->amps['D'][i].labels = {lab[0], lab[4]};
            newguys[14]->amps['D'][i].n_create = 1;
            newguys[14]->amps['D'][i].n_annihilate = 1;

            // and add another
            newguys[14]->set_amplitudes('D', 1, 1, {lab[1], lab[3]});

            // and another
            newguys[14]->set_amplitudes('D', 1, 1, {lab[2], lab[5]});

            // adjust factor
            newguys[14]->factor *= 2.0;
        }
    }

    // new expanded list
    for (int i = 0; i < n_new_terms; i++) {
        list.push_back(newguys[i]);
    }
    return false;
}


} // End namespaces
