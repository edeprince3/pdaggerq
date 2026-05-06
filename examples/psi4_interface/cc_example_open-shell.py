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
import time

from cc_tools import ccsd
from cc_tools import ccsd_t
from cc_tools import ccsdt
from cc_tools import ccsdtq

def main():

    # CCSD / OH radical UHF / STO-3G

    # set molecule
    mol = psi4.geometry("""
    0 2
        O  0.0  0.0  0.0
        H  0.0  0.0  0.96966
    no_reorient
    nocom
    symmetry c1
    """)

    # set options
    psi4_options = {
        'basis': '6-31g',
        'reference': 'uhf',
        'scf_type': 'pk',
        'e_convergence': 1e-10,
        'd_convergence': 1e-10
    }
    psi4.set_options(psi4_options)

    # run ccsd
    s1 = time.time()
    en1 = ccsd(mol, do_eom_ccsd = False, use_spin_orbital_basis = True)
    e1 = time.time()
    time_1 = e1-s1

    # check ccsd energy against psi4
    assert np.isclose(en1, -75.461980701869820, rtol=1e-8, atol=1e-8)

    print('    Spin-Orbital CCSD Total Energy.............................................PASSED')
    print('')

    s2 = time.time()
    en2 = ccsd(mol, do_eom_ccsd = False, use_spin_orbital_basis = False)
    e2 = time.time()
    time_2 = e2-s2

    # check ccsd energy against psi4
    assert np.isclose(en2, -75.461980701869820, rtol=1e-8, atol=1e-8)

    print('    Spin-Traced CCSD Total Energy..............................................PASSED')
    print('')

    print(f"CCSD energy in spin-orbital basis: {en1: 30.20f}")
    print(f"CCSD energy in spin-traced basis:  {en2: 30.20f}")
    print(f"Difference in energy:              {en2-en1: 30.20f}")
    print(f"CCSD time in spin-orbital basis:   {time_1: 10.3f}")
    print(f"CCSD time in spin-traced basis:    {time_2: 10.3f}")
    print(f"Difference in time:                {time_2-time_1: 10.3f}")
    print("")

    # run ccsd(t)
    en = ccsd_t(mol)

    # check ccsd(t) energy against psi4
    assert np.isclose(en,-75.462537797835523, rtol = 1e-8, atol = 1e-8)

    print('    CCSD(T) Total Energy..........................................................PASSED')
    print('')

    # CCSDT / OH radical ROHF/6-31G

    # set options
    psi4_options = {
        'basis': '6-31g',
        'reference': 'rohf',
        'scf_type': 'pk',
        'e_convergence': 1e-10,
        'd_convergence': 1e-10
    }

    psi4.set_options(psi4_options)

    # run spin-orbital ccsdt
    s1 = time.time()
    en1 = ccsdt(mol, use_spin_orbital_basis = True)
    e1 = time.time()
    time_1 = e1-s1

    # check ccsdt energy against gamess/ccpy
    assert np.isclose(en1,-75.462630429073740856,rtol = 1e-8, atol = 1e-8)

    print('    Spin-Orbital CCSDT Total Energy.............................................PASSED')
    print('')

    # run ccsdt spin-blocked
    s2 = time.time()
    en2 = ccsdt(mol, use_spin_orbital_basis = False)
    e2 = time.time()
    time_2 = e2-s2

    # check ccsdt energy against gamess/ccpy
    assert np.isclose(en2, -75.462630429073740856, rtol=1e-8, atol=1e-8)

    print('    Spin-Traced CCSDT Total Energy..............................................PASSED')
    print('')

    print(f"CCSDT energy in spin-orbital basis: {en1: 30.20f}")
    print(f"CCSDT energy in spin-traced basis:  {en2: 30.20f}")
    print(f"Difference in energy:               {en2 - en1: 30.20f}")
    print(f"CCSDT time in spin-orbital basis:   {time_1: 10.3f}")
    print(f"CCSDT time in spin-traced basis:    {time_2: 10.3f}")
    print(f"Difference in time:                 {time_2 - time_1: 10.3f}")
    print("")

    # CCSDTQ / Be triplet ROHF/6-31G

    mol = psi4.geometry("""
        0 3
             Be  0.0  0.0  0.0    
        no_reorient
        nocom
        symmetry c1
        """)

    # set options
    psi4_options = {
        'basis': '6-31g',
        'reference': 'rohf',
        'scf_type': 'pk',
        'e_convergence': 1e-10,
        'd_convergence': 1e-10
    }

    psi4.set_options(psi4_options)

    # run spin-blocked ccsdtq
    s1 = time.time()
    en1 = ccsdtq(mol)
    e1 = time.time()
    time_1 = e1 - s1

    # check ccsdtq energy against ccpy
    assert np.isclose(en1, -14.508386418276780105, rtol=1e-8, atol=1e-8)

    print('    Spin-Traced CCSDTQ Total Energy..............................................PASSED')
    print('')

    print(f"CCSDTQ energy in spin-traced basis: {en1: 30.20f}")
    print(f"CCSDTQ time in spin-traced basis:   {time_1: 10.3f}")
    print("")

if __name__ == "__main__":
    main()

