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
#include "ahat_helper.h"

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

extern "C" PSI_API
std::shared_ptr<Wavefunction> pdaggerq(std::shared_ptr<Wavefunction> ref_wfn, Options& options)
{
    //std::vector< ahat* > ordered;

    std::shared_ptr<ahat_helper> helper (new ahat_helper());

    if ( options["SQSTRING"].has_changed() ) {
        helper->add_new_string(options,"");
    }
    if ( options["SQSTRING2"].has_changed() ) {
        helper->add_new_string(options,"2");
    }
    if ( options["SQSTRING3"].has_changed() ) {
        helper->add_new_string(options,"3");
    }
    if ( options["SQSTRING4"].has_changed() ) {
        helper->add_new_string(options,"4");
    }

    helper->finalize();

    Process::environment.globals["CURRENT ENERGY"] = 0.0;

    return ref_wfn;
}

}} // End namespaces

