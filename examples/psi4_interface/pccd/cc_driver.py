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

run_hilbert = False
if run_hilbert == True:
    import sys
    sys.path.insert(0, '/Users/deprince/edeprince3/hilbert/public')
    import hilbert

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

    basis = 'sto-3g'
    
    # set options
    psi4_options_dict = {
        'basis': basis,
        'scf_type': 'cd',
        'cholesky_tolerance': 1e-12,
        'df_basis_scf': 'aug-cc-pv5z-jkfit',
        'df_ints_io': 'save',
        'e_convergence': 1e-6,
        'r_convergence': 1e-6,
        'd_convergence': 1e-6,
        'maxiter': 100,
    }
    psi4.set_options(psi4_options_dict)
    
    # compute the Hartree-Fock energy and wave function
    scf_e, wfn = psi4.energy('SCF', return_wfn=True)

    # run hilbert's pccd and doci?
    if run_hilbert == True:
        psi4_options_dict = {
            'basis': basis,
            'scf_type': 'cd',
            'cholesky_tolerance': 1e-6,
            'scf_type': 'cd',
            'df_basis_scf': 'aug-cc-pv5z-jkfit',
            'df_ints_io': 'save',
            'e_convergence': 1e-6,
            'r_convergence': 1e-6,
            'd_convergence': 1e-6,
            'optimize_orbitals': False,
            'orbopt_gradient_convergence': 1e-4,
            'orbopt_energy_convergence': 1e-6,
            'orbopt_maxiter': 10,
            'maxiter': 100,
            'p2rdm_type': 'ccd',
        }
        psi4.set_options(psi4_options_dict)

        # grab options object
        options = psi4.core.get_options()
        options.set_current_module('HILBERT')

        doci_psi4 = hilbert.DOCIHelper(wfn,options)
        doci_psi4.compute_energy()

        pccd_psi4 = hilbert.pp2RDMHelper(doci_psi4,options)
        pccd_psi4.compute_energy()

    # end of hilbert's pccd and doci

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
    #C = doci_psi4.Ca()

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

    #print("")
    #print("    ==> t amplitudes <== ")
    #print("")
    #for a in range (nv):
    #    for i in range (no):
    #        print("%5i %5i %20.12lf" % (a, i, t2[a,a+nv,i,i+no]))
    #print("")
    #print("    ==> lambda amplitudes <== ")
    #print("")
    #for i in range (no):
    #    for a in range (nv):
    #        print("%5i %5i %20.12lf" % (i, a, l2[i,i+no,a,a+nv]))

    cc_energy = ccsd_energy(t1, t2, fock, gtei, o, v)

    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    print("")
    print("    pCCD correlation energy: {: 20.12f}".format(cc_energy - hf_energy))
    print("    * pCCD total energy:       {: 20.12f}".format(cc_energy + nuclear_repulsion_energy))
    print("")

    # build 1rdm
    from pccd import ccsd_d1
    d1hf = np.eye(2 * nmo)
    opdm = ccsd_d1(t1, t2, l1, l2, d1hf, o, v)

    print("")
    print(" # Natural Orbital Occupation Numbers (spin free) #")
    print("")
    for i in range (no):
        print("%5i: %20.12lf" % (i, 2*opdm[i,i]) )
    for i in range (2*no,2*no+nv):
        print("%5i: %20.12lf" % (i, 2*opdm[i,i]) )
    print("")

if __name__ == "__main__":
    main()


