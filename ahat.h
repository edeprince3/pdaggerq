//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: ahat.h
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

#ifndef SQE_H
#define SQE_H

#include "data.h"

namespace pdaggerq {

class ahat {

  private:

    bool is_normal_order();

    bool index_in_tensor(std::string idx);
    bool index_in_t_amplitudes(std::string idx);
    bool index_in_left_amplitudes(std::string idx);
    bool index_in_right_amplitudes(std::string idx);

    void replace_index_everywhere(std::string old_idx, std::string new_idx);
    void replace_index_in_tensor(std::string old_idx, std::string new_idx);
    void replace_index_in_t_amplitudes(std::string old_idx, std::string new_idx);
    void replace_index_in_left_amplitudes(std::string old_idx, std::string new_idx);
    void replace_index_in_right_amplitudes(std::string old_idx, std::string new_idx);

    bool compare_strings(std::shared_ptr<ahat> ordered_1, std::shared_ptr<ahat> ordered_2, int & n_permute);
    void update_summation_labels();
    void update_bra_labels();

    void swap_two_labels(std::string label1, std::string label2);


  public:

    ahat(std::string vacuum_type);
    ~ahat();

    std::string vacuum;

    bool skip     = false;
    int sign      = 1;

    void shallow_copy(void * copy_me);
    void copy(void * copy_me);

    std::vector<std::string> symbol;
    std::vector<bool> is_dagger;
    std::vector<bool> is_dagger_fermi;
    std::vector<std::string> delta1;
    std::vector<std::string> delta2;

    std::shared_ptr<StringData> data;

    void print();
    void check_spin();
    void check_occ_vir();
    void gobble_deltas();
    void use_conventional_labels();

    void normal_order(std::vector<std::shared_ptr<ahat> > &ordered);
    void normal_order_fermi_vacuum(std::vector<std::shared_ptr<ahat> > &ordered);
    void normal_order_true_vacuum(std::vector<std::shared_ptr<ahat> > &ordered);

    void alphabetize(std::vector<std::shared_ptr<ahat> > &ordered);
    void cleanup(std::vector<std::shared_ptr<ahat> > &ordered);

    bool is_occ(std::string idx);
    bool is_vir(std::string idx);
};

}

#endif
