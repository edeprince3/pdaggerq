//
// pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
// Filename: data.h
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

#ifndef DATA_H
#define DATA_H

namespace pdaggerq {

class StringData {

  private:


  public:

    /// constructor
    StringData(){};

    /// descructor
    ~StringData(){};

    /// factor
    double factor = 1.0;

    /// list: labels for fermionic creation / annihilation operators 
    std::vector<std::string> string;

    /// list: labels for 1- or 2-index tensor
    std::vector<std::string> tensor;

    /// tensor type (CORE, ERI, D+, D-)
    std::string tensor_type;

    /// list: labels u amplitudes
    std::vector<std::vector<std::string> > u_amplitudes;

    /// list: labels t amplitudes
    std::vector<std::vector<std::string> > t_amplitudes;

    /// list: labels left-hand eom-cc amplitudes
    std::vector<std::vector<std::string> > left_amplitudes;

    /// list: labels right-hand amplitudes
    std::vector<std::vector<std::string> > right_amplitudes;

    /// list: labels left-hand amplitudes plus boson
    std::vector<std::vector<std::string> > m_amplitudes;

    /// list: labels right-hand amplitudes plus boson
    std::vector<std::vector<std::string> > s_amplitudes;

    /// should we account for l0 in EOM-CC?
    bool has_l0 = false;

    /// should we account for r0 in EOM-CC?
    bool has_r0 = false;

    /// should we account for u0?
    bool has_u0 = false;

    /// should we account for m0?
    bool has_m0 = false;

    /// should we account for s0?
    bool has_s0 = false;

    /// should we account for w0?
    bool has_w0 = false;

    /// should we account for terms with just b?
    bool has_b = false;

    /// should we account for terms with just b+?
    bool has_b_dagger = false;

    /// list: is bosonic operator creator or annihilator?
    std::vector<bool> is_boson_dagger;

};

}

#endif
