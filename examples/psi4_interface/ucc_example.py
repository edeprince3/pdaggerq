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

example script for running uccsd(3) and uccsd(4) using
pdaggerq-generated equations. Integrals come from psi4.

"""

import numpy as np
from numpy import einsum
import psi4

from cc_tools import ucc3
from cc_tools import ucc4

def main():

    # UCC3, UCC4 / H2O / STO-3G

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

    ucc3_en = ucc3(mol)

    # check ucc3 energy
    assert np.isclose(ucc3_en, -75.020316846984, rtol = 1e-8, atol = 1e-8)

    print('    UCCSD(3) Total Energy........................................................PASSED')
    print('')
    #exit()

    ucc4_en = ucc4(mol)

    # check ucc4 energy
    assert np.isclose(ucc4_en, -75.019768257645, rtol = 1e-8, atol = 1e-8)

    print('    UCCSD(4) Total Energy........................................................PASSED')
    print('')
    #exit()

if __name__ == "__main__":
    main()

