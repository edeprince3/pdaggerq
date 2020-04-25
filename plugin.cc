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
#include <psi4/libpsi4util/process.h>

#include<iostream>
#include<string>
#include<algorithm>

#include "ahat.h"

#include <math.h>

namespace psi{ namespace pdaggerq {

extern "C" PSI_API
int read_options(std::string name, Options& options)
{
    if (name == "AHAT"|| options.read_globals()) {
        /*- print code? -*/
        options.add_bool("PRINT_CODE", false);

        /*- a string of creation/annihilation operators to rearrange -*/
        options.add("SQSTRING",new ArrayType());
        /*- a string of indices representing a tensor -*/
        options.add("SQTENSOR",new ArrayType());
        /*- the multiplicative factor for the given string -*/
        options.add_double("SQFACTOR",1.0);
        /*- a string of indices representing an amplitude -*/
        options.add("SQAMPS_A",new ArrayType());
        /*- a string of indices representing an amplitude -*/
        options.add("SQAMPS_B",new ArrayType());

        /*- a string of creation/annihilation operators to rearrange -*/
        options.add("SQSTRING2",new ArrayType());
        /*- a string of indices representing a tensor -*/
        options.add("SQTENSOR2",new ArrayType());
        /*- the multiplicative factor for the given string -*/
        options.add_double("SQFACTOR2",1.0);
        /*- a string of indices representing an amplitude -*/
        options.add("SQAMPS2_A",new ArrayType());
        /*- a string of indices representing an amplitude -*/
        options.add("SQAMPS2_B",new ArrayType());

        /*- a string of creation/annihilation operators to rearrange -*/
        options.add("SQSTRING3",new ArrayType());
        /*- a string of indices representing a tensor -*/
        options.add("SQTENSOR3",new ArrayType());
        /*- the multiplicative factor for the given string -*/
        options.add_double("SQFACTOR3",1.0);
        /*- a string of indices representing an amplitude -*/
        options.add("SQAMPS3_A",new ArrayType());
        /*- a string of indices representing an amplitude -*/
        options.add("SQAMPS3_B",new ArrayType());

        /*- a string of creation/annihilation operators to rearrange -*/
        options.add("SQSTRING4",new ArrayType());
        /*- a string of indices representing a tensor -*/
        options.add("SQTENSOR4",new ArrayType());
        /*- the multiplicative factor for the given string -*/
        options.add_double("SQFACTOR4",1.0);
        /*- a string of indices representing an amplitude -*/
        options.add("SQAMPS4_A",new ArrayType());
        /*- a string of indices representing an amplitude -*/
        options.add("SQAMPS4_B",new ArrayType());
    }

    return true;
}

void removeStar(std::string &x)
{
  auto it = std::remove_if(std::begin(x),std::end(x),[](char c){return (c == '*');});
  x.erase(it, std::end(x));
}

void AddNewString(Options& options,std::vector< ahat* > &ordered,std::string stringnum){

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

void consolidate(std::vector<ahat *> &in,std::vector<ahat *> &out) {

    bool *vanish = (bool*)malloc(in.size()*sizeof(bool));
    memset((void*)vanish,'\0',in.size()*sizeof(bool));
    for (int i = 0; i < (int)in.size(); i++) {
        for (int j = i+1; j < (int)in.size(); j++) {


            bool strings_differ = false;

            // check strings
            if ( in[i]->symbol.size() == in[j]->symbol.size() ) {
                for (int k = 0; k < (int)in[i]->symbol.size(); k++) {

                    // strings differ?
                    if ( in[i]->symbol[k] != in[j]->symbol[k] ) {
                        strings_differ = true;
                    }

                }
            }else {
                strings_differ = true;
            }
            if ( strings_differ ) continue;

            // check deltas
            if ( in[i]->delta1.size() == in[j]->delta1.size() ) {
                for (int k = 0; k < (int)in[i]->delta1.size(); k++) {

                    // strings differ?
                    if ( in[i]->delta1[k] != in[j]->delta1[k] || in[i]->delta2[k] != in[j]->delta2[k] ) {
                        strings_differ = true;
                    }

                }
            }else {
                strings_differ = true;
            }
            if ( strings_differ ) continue;

            // check tensors
            if ( in[i]->tensor.size() == in[j]->tensor.size() ) {
                for (int k = 0; k < (int)in[i]->tensor.size(); k++) {

                    // strings differ?
                    if ( in[i]->tensor[k] != in[j]->tensor[k] ) {

                        strings_differ = true;
                    }

                }
            }else {
                strings_differ = true;
            }
            if ( strings_differ ) continue;
            
            // at this point, we know the strings are the same.  what about the factor?
            int fac1 = in[i]->factor;
            int fac2 = in[j]->factor;
            if ( fabs(fac1 + fac2) < 1e-8 ) {
                vanish[i] = true;
                vanish[j] = true;
                //printf("these terms will cancel\n");
                //in[i]->print();
                //in[j]->print();
            }


        }
    }
    for (int i = 0; i < (int)in.size(); i++) {
        if ( !vanish[i] ) {
            out.push_back(in[i]);
        }
        
    }


}

extern "C" PSI_API
std::shared_ptr<Wavefunction> pdaggerq(std::shared_ptr<Wavefunction> ref_wfn, Options& options)
{
    std::vector< ahat* > ordered;

    if ( options["SQSTRING"].has_changed() ) {
        AddNewString(options,ordered,"");
    }
    if ( options["SQSTRING2"].has_changed() ) {
        AddNewString(options,ordered,"2");
    }
    if ( options["SQSTRING3"].has_changed() ) {
        AddNewString(options,ordered,"3");
    }
    if ( options["SQSTRING4"].has_changed() ) {
        AddNewString(options,ordered,"4");
    }

    printf("\n");
    printf("    ");
    printf("// normal-ordered strings:\n");

    std::vector< ahat* > pruned;
    consolidate(ordered,pruned);

    for (int i = 0; i < (int)pruned.size(); i++) {
        //pruned[i]->check_occ_vir();
        pruned[i]->check_spin();
        pruned[i]->print();
    }
    printf("\n");

    Process::environment.globals["CURRENT ENERGY"] = 0.0;

    return ref_wfn;
}

}} // End namespaces

