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
Driver for spin-orbital CCSD and EOM-CCSD. (EOM-)CCSD code  generated with pdaggerq. Integrals come from psi4.
"""
import numpy as np
from numpy import einsum

def main():

    import sys
    import numpy as np
    
    import psi4

    from spatial_to_spin_orbital import spatial_to_spin_orbital_oei
    from spatial_to_spin_orbital import spatial_to_spin_orbital_tei

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
    psi4_options_dict = {
        'basis': 'sto-3g',
        'scf_type': 'pk',
        'e_convergence': 1e-10,
        'd_convergence': 1e-10
    }
    psi4.set_options(psi4_options_dict)
    
    # compute the Hartree-Fock energy and wave function
    scf_e, wfn = psi4.energy('SCF', return_wfn=True)

    # number of doubly occupied orbitals
    no   = wfn.nalpha()
    
    # total number of orbitals
    nmo     = wfn.nmo()
    
    # number of virtual orbitals
    nv   = nmo - no
    
    # orbital energies
    epsilon     = np.asarray(wfn.epsilon_a())
    
    # molecular orbitals (spatial):
    C = wfn.Ca()

    # use Psi4's MintsHelper to generate integrals
    mints = psi4.core.MintsHelper(wfn.basisset())

    # build the one-electron integrals
    H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    H = np.einsum('uj,vi,uv', C, C, H)

    # unpack one-electron integrals in spin-orbital basis
    sH   = spatial_to_spin_orbital_oei(H,nmo,no)
    
    # build the two-electron integrals:
    tei = np.asarray(mints.mo_eri(C, C, C, C))

    # unpack two-electron integrals in spin-orbital basis
    stei = spatial_to_spin_orbital_tei(tei,nmo,no)

    # antisymmetrize g(ijkl) = <ij|kl> - <ij|lk> = (ik|jl) - (il|jk)
    gtei = np.einsum('ikjl->ijkl', stei) - np.einsum('iljk->ijkl', stei)

    # occupied, virtual slices
    n = np.newaxis
    o = slice(None, 2 * no)
    v = slice(2 * no, None)

    # build spin-orbital fock matrix
    fock = sH + np.einsum('piqi->pq', gtei[:, o, :, o])

    # orbital energies
    eps = np.zeros(2*nmo)

    for i in range(0,2*nmo):
        eps[i] = fock[i,i]

    # energy denominators
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    nsvirt = 2 * nv
    nsocc = 2 * no

    # hartree-fock energy
    hf_energy = 1.0 * einsum('ii', fock[o, o]) -0.5 * einsum('ijij', gtei[o, o, o, o])

    # ccsd iterations
    from ccsd import ccsd_iterations
    from ccsd import ccsd_energy

    t1, t2 = ccsd_iterations(np.zeros((nsvirt, nsocc)), np.zeros((nsvirt, nsvirt, nsocc, nsocc)), fock, gtei, o, v, e_ai, e_abij,
                      hf_energy, e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)

    cc_energy = ccsd_energy(t1, t2, fock, gtei, o, v)

    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    print("")
    print("    CCSD Correlation Energy: {: 20.12f}".format(cc_energy - hf_energy))
    print("    CCSD Total Energy:       {: 20.12f}".format(cc_energy + nuclear_repulsion_energy))
    print("")

    # now eom-ccsd?
    print("    ==> EOM-CCSD <==")
    print("")
    from eom_ccsd import build_eom_ccsd_H

    H = build_eom_ccsd_H(fock, gtei, o, v, t1, t2, nsocc, nsvirt)

    print('    eigenvalues of e(-T) H e(T):')
    print('')

    print('    %5s %20s %20s' % ('state', 'total energy','excitation energy'))
    en, vec = np.linalg.eig(H)
    en.sort()
    for i in range (1,min(21,len(en))):
        print('    %5i %20.12f %20.12f' % ( i, en[i] + nuclear_repulsion_energy,en[i]-cc_energy ))

    print('')

if __name__ == "__main__":
    main()

