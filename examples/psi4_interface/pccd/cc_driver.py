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
Driver for spin-orbital pCCD
"""
import numpy as np
from numpy import einsum

def main():

    import sys
    import numpy as np
    
    import psi4

    from spatial_to_spin_orbital import spatial_to_spin_orbital_oei
    from spatial_to_spin_orbital import spatial_to_spin_orbital_tei

    do_eom_ccsd = True

    # set molecule
    mol = psi4.geometry("""
    0 1
         #O            0.000000000000     0.000000000000    -0.068516219320    
         #H            0.000000000000    -0.790689573744     0.543701060715    
         #H            0.000000000000     0.790689573744     0.543701060715    
         N
         N 1 2.1
    no_reorient
    nocom
    symmetry c1
    """)  
    
    # set options
    psi4_options_dict = {
        'basis': 'sto-3g',
        'scf_type': 'pk',
        'e_convergence': 1e-14,
        'd_convergence': 1e-14
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
    tmp = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    tmp = np.einsum('uj,vi,uv', C, C, tmp)

    H = np.zeros((nmo,nmo))
    for p in range (nmo):
        H[p,p] = tmp[p,p]

    # unpack one-electron integrals in spin-orbital basis
    sH   = spatial_to_spin_orbital_oei(H,nmo,no)
    
    # build the two-electron integrals:
    tmp = np.asarray(mints.mo_eri(C, C, C, C))

    tei = np.zeros((nmo,nmo,nmo,nmo))
    for p in range (nmo):
        for q in range (nmo):
            tei[p,q,p,q] = tmp[p,q,p,q]
            tei[p,p,q,q] = tmp[p,p,q,q]
            tei[p,q,q,p] = tmp[p,q,q,p]

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

# enforce seniority zero structure on integrals

    # build the one-electron integrals
    #H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    #H = np.einsum('uj,vi,uv', C, C, H)
    #tmp = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    #tmp = np.einsum('uj,vi,uv', C, C, tmp)
    #for p in range (nmo):
    #    for q in range (nmo):
    #        H[p,q] = 0.0
    #for p in range (nmo):
    #    H[p,p] = tmp[p,p]

    ## unpack one-electron integrals in spin-orbital basis
    #sH   = spatial_to_spin_orbital_oei(H,nmo,no)
   
    ## build the two-electron integrals:
    #tei = np.asarray(mints.mo_eri(C, C, C, C))
    #tmp = np.asarray(mints.mo_eri(C, C, C, C))

    #for p in range (nmo):
    #    for q in range (nmo):
    #        for r in range (nmo):
    #            for s in range (nmo):
    #                tei[p,q,r,s] = 0.0
    #for p in range (nmo):
    #    for q in range (nmo):
    #        tei[p,q,p,q] = tmp[p,q,p,q]
    #        tei[p,p,q,q] = tmp[p,p,q,q]
    #        tei[p,q,q,p] = tmp[p,q,q,p]

    ## unpack two-electron integrals in spin-orbital basis
    #stei = spatial_to_spin_orbital_tei(tei,nmo,no)

    ## antisymmetrize g(ijkl) = <ij|kl> - <ij|lk> = (ik|jl) - (il|jk)
    #gtei = np.einsum('ikjl->ijkl', stei) - np.einsum('iljk->ijkl', stei)

    ## occupied, virtual slices
    #n = np.newaxis
    #o = slice(None, 2 * no)
    #v = slice(2 * no, None)

    ## build spin-orbital fock matrix
    #fock = sH + np.einsum('piqi->pq', gtei[:, o, :, o])
    #for i in range (0,2*nmo):
    #    for j in range (0,2*nmo):
    #        if i == j:
    #            continue
    #        fock[i,j] = 0.0

# end of enforcing seniority zero structure on integrals

    # ccsd iterations
    from pccd import ccsd_iterations
    from pccd import ccsd_energy

    # solve t-amplitude equations
    t1 = np.zeros((nsvirt, nsocc))
    t2 = np.zeros((nsvirt, nsvirt, nsocc, nsocc))
    t1, t2 = ccsd_iterations(t1, t2, fock, gtei, o, v, e_ai, e_abij,
                      hf_energy, nsocc, nsvirt, e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)

    # solve lambda-amplitude equations
    from pccd import lambda_iterations
    t1, t2, l1, l2 = lambda_iterations(t1, t2, fock, gtei, o, v, e_ai, e_abij,
                      hf_energy, nsocc, nsvirt, e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)

    cc_energy = ccsd_energy(t1, t2, fock, gtei, o, v)

    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    print("")
    print("    pCCD Correlation Energy: {: 20.12f}".format(cc_energy - hf_energy))
    print("    pCCD Total Energy:       {: 20.12f}".format(cc_energy + nuclear_repulsion_energy))
    print("")

    # build 1rdm
    from pccd import ccsd_d1
    d1hf = np.eye(2 * nmo)
    opdm = ccsd_d1(t1, t2, l1, l2, d1hf, o, v)

    opdm_a = opdm[::2, ::2]
    opdm_b = opdm[1::2, 1::2]
    opdm_s = (opdm_a + opdm_b + opdm_a.T + opdm_b.T) / 2
    print("")
    print(" # Natural Orbital Occupation Numbers (spin free) #")
    print("")
    for i in range (nmo):
        print("%5i: %20.12lf" % (i, opdm_s[i,i]) )
    print("")

if __name__ == "__main__":
    main()

