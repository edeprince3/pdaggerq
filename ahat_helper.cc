/*
 *@BEGIN LICENSE
 *
 * pdaggerq, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Copyright (c) 2014, The Florida State University. All rights reserved.
 * 
 *@END LICENSE
 *
 */

#ifndef _python_api2_h_
#define _python_api2_h_

#include<iostream>
#include<string>
#include<algorithm>

#include "psi4/psi4-dec.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"
#include <psi4/libpsi4util/process.h>



#include "ahat.h"
#include "ahat_helper.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "psi4/libpsi4util/process.h"
#include "psi4/libmints/wavefunction.h"

using namespace psi;

namespace py = pybind11;
using namespace pybind11::literals;

namespace psi { namespace pdaggerq {

void export_ahat_helper(py::module& m) {
    py::class_<pdaggerq::ahat_helper, std::shared_ptr<pdaggerq::ahat_helper> >(m, "ahat_helper")
        .def(py::init< >())
        .def("add_new_string", &ahat_helper::add_new_string)
        .def("finalize", &ahat_helper::finalize);
}

PYBIND11_MODULE(pdaggerq, m) {
    m.doc() = "Python API of pdaggerq: A code for bringing strings of creation / annihilation operators to normal order.";
    export_ahat_helper(m);
}

void removeStar(std::string &x)
{ 
  auto it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == '*');});
  x.erase(it, std::end(x));
}

ahat_helper::ahat_helper()
{
}

ahat_helper::~ahat_helper()
{
}

void ahat_helper::add_new_string(Options& options,std::string stringnum){

    std::shared_ptr<ahat> mystring (new ahat());

    if ( options["SQFACTOR"+stringnum].has_changed() ) {
        if ( options.get_double("SQFACTOR"+stringnum) > 0.0 ) {
            mystring->sign = 1;
            mystring->factor = fabs(options.get_double("SQFACTOR"+stringnum));
        }else {
            mystring->sign = -1;
            mystring->factor = fabs(options.get_double("SQFACTOR"+stringnum));
        }
    }

    if ( options["SQSTRING"+stringnum].has_changed() ) {
        for (int i = 0; i < (int)options["SQSTRING"+stringnum].size(); i++) {
            std::string me = options["SQSTRING"+stringnum][i].to_string();
            if ( me.find("*") != std::string::npos ) {
                removeStar(me);
                mystring->is_dagger.push_back(true);
            }else {
                mystring->is_dagger.push_back(false);
            }
            mystring->symbol.push_back(me);
        }
    }

    if ( options["SQTENSOR"+stringnum].has_changed() ) {
        for (int i = 0; i < (int)options["SQTENSOR"+stringnum].size(); i++) {
            std::string me = options["SQTENSOR"+stringnum][i].to_string();
            mystring->tensor.push_back(me);
        }
    }

    if ( options["SQAMPS"+stringnum+"_A"].has_changed() ) {
        for (int i = 0; i < (int)options["SQAMPS"+stringnum+"_A"].size(); i++) {
            std::string me = options["SQAMPS"+stringnum+"_A"][i].to_string();
            mystring->amplitudes1.push_back(me);
        }
    }
    if ( options["SQAMPS"+stringnum+"_B"].has_changed() ) {
        for (int i = 0; i < (int)options["SQAMPS"+stringnum+"_B"].size(); i++) {
            std::string me = options["SQAMPS"+stringnum+"_B"][i].to_string();
            mystring->amplitudes2.push_back(me);
        }
    }

    printf("\n");
    printf("    ");
    printf("// starting string:\n");
    mystring->print();

    // rearrange strings
    mystring->normal_order(ordered);

    // alphabetize
    mystring->alphabetize(ordered);

    // cancel terms
    mystring->cleanup(ordered);

}

void ahat_helper::finalize() { 
//,std::vector<ahat *> &out)
    
    std::vector< ahat* > out;
            
    bool *vanish = (bool*)malloc(ordered.size()*sizeof(bool));
    memset((void*)vanish,'\0',ordered.size()*sizeof(bool));
    for (int i = 0; i < (int)ordered.size(); i++) {
        for (int j = i+1; j < (int)ordered.size(); j++) {
        
    
            bool strings_differ = false;
    
            // check strings
            if ( ordered[i]->symbol.size() == ordered[j]->symbol.size() ) {
                for (int k = 0; k < (int)ordered[i]->symbol.size(); k++) {
                
                    // strings differ?
                    if ( ordered[i]->symbol[k] != ordered[j]->symbol[k] ) {
                        strings_differ = true;
                    }
            
                }
            }else {
                strings_differ = true;
            }
            if ( strings_differ ) continue;
            
            // check deltas
            if ( ordered[i]->delta1.size() == ordered[j]->delta1.size() ) {
                for (int k = 0; k < (int)ordered[i]->delta1.size(); k++) {

                    // strings differ?
                    if ( ordered[i]->delta1[k] != ordered[j]->delta1[k] || ordered[i]->delta2[k] != ordered[j]->delta2[k] ) {
                        strings_differ = true;
                    }
        
                }
            }else {
                strings_differ = true;
            }
            if ( strings_differ ) continue;
        
            // check tensors
            if ( ordered[i]->tensor.size() == ordered[j]->tensor.size() ) {
                for (int k = 0; k < (int)ordered[i]->tensor.size(); k++) {
    
                    // strings differ?
                    if ( ordered[i]->tensor[k] != ordered[j]->tensor[k] ) {

                        strings_differ = true;
                    }

                }
            }else {
                strings_differ = true;
            }
            if ( strings_differ ) continue;

            // at this point, we know the strings are the same.  what about the factor?
            int fac1 = ordered[i]->factor;
            int fac2 = ordered[j]->factor;
            if ( fabs(fac1 + fac2) < 1e-8 ) {
                vanish[i] = true;
                vanish[j] = true;
            }


        }
    }
    for (int i = 0; i < (int)ordered.size(); i++) {
        if ( !vanish[i] ) {
            out.push_back(ordered[i]);
        }

    }

    ordered.clear();
    for (int i = 0; i < (int)out.size(); i++) {
        ordered.push_back(out[i]);

        // check spin
        ordered[i]->check_spin();

        // check for occ/vir pairs in delta functions
        ordered[i]->check_occ_vir();

    }

    printf("\n");
    printf("    ");
    printf("// normal-ordered strings:\n");
    for (int i = 0; i < (int)ordered.size(); i++) {
        ordered[i]->print();
    }
    printf("\n");

}


}} // End namespaces

#endif
