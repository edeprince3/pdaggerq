# pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
# Copyright (C) 2020 A. Eugene DePrince III
#
# This file is part of the pdaggerq package.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""

example script for running ccsd, ccsdt, ccsdqt, and eom-ccsdt using 
pdaggerq-generated equations. Integrals come from psi4.

"""

import numpy as np
from numpy import einsum
import psi4

from cc_tools import ccsd
from cc_tools import ccsd_t
from cc_tools import ccsdt
from cc_tools import ccsdtq

def main():

    # CCSD / H2O / STO-3G

    # set molecule
    mol = psi4.geometry("""
    0 1
         O            0.000000000000     0.000000000000    -0.068516219320    
         H            0.000000000000    -0.790689573744     0.543701060715    
         H            0.000000000000     0.790689573744     0.543701060715    
    no_reorient
    nocom
    symmetry c1
    """)  
    
    # set options
    psi4_options = {
        'basis': 'sto-3g',
        'scf_type': 'pk',
        'e_convergence': 1e-10,
        'd_convergence': 1e-10
    }
    psi4.set_options(psi4_options)

    # run ccsd
    import time
    s1 = time.time()
    en = ccsd(mol, do_eom_ccsd = False, use_spin_orbital_basis = True)
    e1 = time.time()

    s2 = time.time()
    en = ccsd(mol, do_eom_ccsd = False, use_spin_orbital_basis = False)
    e2 = time.time()

    print(e1-s1, e2-s2)

    # check ccsd energy against psi4
    assert np.isclose(en,-75.019715133639338, rtol = 1e-8, atol = 1e-8)

    print('    CCSD Total Energy..........................................................PASSED')
    print('')

    # run ccsd(t)
    en = ccsd_t(mol)

    # check ccsd energy against psi4
    assert np.isclose(en,-75.019790965805612, rtol = 1e-8, atol = 1e-8)

    print('    CCSD(T) Total Energy..........................................................PASSED')
    print('')

    # CCSDT / HF / 6-31G

    # set molecule
    mol = psi4.geometry("""
    0 1
         H 0.0 0.0 0.0
         F 0.0 0.0 3.023561812617029
    units bohr
    no_reorient
    nocom
    symmetry c1
    """)  
    
    # set options
    psi4_options = {
        'basis': '6-31g',
        'scf_type': 'pk',
        'e_convergence': 1e-10,
        'd_convergence': 1e-10
    }
    psi4.set_options(psi4_options)

    # run ccsdt spin-blocked
    s1 = time.time()
    en = ccsdt(mol, use_spin_orbital_basis = False)
    e1 = time.time()

    # check ccsdt energy against nwchem
    assert np.isclose(en,-100.008956600850908,rtol = 1e-8, atol = 1e-8)

    print('    CCSDT Total Energy..........................................................PASSED')
    print('')

    ## run spin-orbital ccsdt
    #s2 = time.time()
    #en = ccsdt(mol)
    #e2 = time.time()

    ## check ccsdt energy against nwchem
    #assert np.isclose(en,-100.008956600850908,rtol = 1e-8, atol = 1e-8)

    #print('    CCSDT Total Energy..........................................................PASSED')
    #print('')

    #print(e1-s1, e2-s2)

    # run spin-blocked ccsdtq
    en = ccsdtq(mol)

    # check ccsdtq energy against nwchem
    assert np.isclose(en,-100.009723511692869,rtol = 1e-8, atol = 1e-8)

    print('    CCSDTQ Total Energy..........................................................PASSED')
    print('')

if __name__ == "__main__":
    main()

