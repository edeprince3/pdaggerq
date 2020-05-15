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

    StringData(){};
    ~StringData(){};

    double factor = 1.0;
    std::vector<std::string> string;
    std::vector<std::string> tensor;
    std::vector<std::vector<std::string> > amplitudes;
    std::vector<std::vector<std::string> > left_amplitudes;

};

}

#endif
