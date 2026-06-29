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
Driver for spin-orbital CCSD and EOM-CCSD. (EOM-)CCSD code generated with pdaggerq. Integrals come from psi4.
"""

import scipy
from scipy.sparse.linalg import LinearOperator

import numpy as np
from numpy import einsum

# psi4
import psi4

# ucc4 iterations
from ucc4 import ucc4_iterations
from ucc4 import ucc4_energy

# quccsd iterations
from quccsd import quccsd_iterations
from quccsd import quccsd_energy

# ucc3 iterations
from ucc3 import ucc3_iterations
from ucc3 import ucc3_energy

# ccsd iterations
from ccsd import ccsd_iterations
from ccsd import coupled_cluster_energy
from ccsd import ccsd_iterations_with_spin
from ccsd import ccsd_energy_with_spin

# ccsdt iterations
from ccsdt import ccsdt_iterations
from ccsdt import ccsdt_iterations_with_spin

# (t)
from ccsdt import perturbative_triples_correction

# ccsdyq iterations
from ccsdtq import ccsdtq_iterations_with_spin

def spatial_to_spin_orbital_oei(ha, hb, n, noa, nob):
    """

    :param ha: one-electron orbitals, alpha part
    :param hb: one-electron orbitals, beta part
    :param n: number of spatial orbitals
    :param noa: number of alpha occupied orbitals
    :param nob: number of beta occupied orbitals
    :return:  spin-orbital one-electron integrals, sh
    """

    # build spin-orbital oeis
    sh = np.zeros((2*n,2*n))

    # shape of each axis in spin-orbital basis: |oa|ob|va|vb|
    soa = slice(None, noa)
    sob = slice(noa, noa+nob)
    sva = slice(noa+nob, n+nob)
    svb = slice(n+nob, None)
    # shape of spatial orbital axis: |oa|va| and |ob|vb|
    oa = slice(None, noa)
    va = slice(noa, None)
    ob = slice(None, nob)
    vb = slice(nob, None)

    # alpha blocks
    sh[soa, soa] = ha[oa, oa]
    sh[sva, sva] = ha[va, va]
    sh[sva, soa] = ha[va, oa]
    sh[soa, sva] = ha[oa, va]

    # beta blocks
    sh[sob, sob] = hb[ob, ob]
    sh[svb, svb] = hb[vb, vb]
    sh[svb, sob] = hb[vb, ob]
    sh[sob, svb] = hb[ob, vb]

    return sh

def spatial_to_spin_orbital_tei(gaa, gab, gbb, n, noa, nob):
    """

    :param gaa: antisymmetrized two-electron integrals in physicist' notation, alpha-alpha portion
    :param gab: two-electron integrals in physicist' notation, alpha-beta portion
    :param gbb: antisymmetrized two-electron integrals in physicist' notation, beta-beta portion
    :param n: number of spatial orbitals
    :param noa: number of alpha occupied orbitals
    :param nob: number of beta occupied orbitals
    :return:  spin-orbital two-electron integrals, sg
    """

    # build spin-orbital teis
    sg = np.zeros((2*n,2*n,2*n,2*n))

    # shape of each axis in spin-orbital basis: |oa|ob|va|vb|
    soa = slice(None, noa)
    sob = slice(noa, noa+nob)
    sva = slice(noa+nob, n+nob)
    svb = slice(n+nob, None)
    # shape of spatial orbital axis: |oa|va| and |ob|vb|
    oa  = slice(None,noa)
    va  = slice(noa,None)
    ob  = slice(None,nob)
    vb  = slice(nob,None)

    # populate TEI
    def to_bin(x):
        return '{:04b}'.format(x)

    soa = (soa, sva)
    moa = (oa, va)
    sob = (sob, svb)
    mob = (ob, vb)

    # go from 0000 to 1111, 0 = occ slice, 1 = vir slice
    # equivalent to looping from oooo to vvvv slice
    for i in range(16):
        p,q,r,s = [int(x) for x in to_bin(i)]
        # <p,q||r,s> <- aaaa block
        sg[soa[p], soa[q], soa[r], soa[s]] = gaa[moa[p], moa[q], moa[r], moa[s]]
        # <p,q||r,s> <- abab block, antisymmetrize on the fly (needed for spin-orbital code)
        sg[soa[p], sob[q], soa[r], sob[s]] =  gab[moa[p], mob[q], moa[r], mob[s]]
        sg[soa[p], sob[q], sob[r], soa[s]] = -gab[moa[p], mob[q], moa[s], mob[r]].transpose(0,1,3,2)
        sg[sob[p], soa[q], soa[r], sob[s]] = -gab[moa[q], mob[p], moa[r], mob[s]].transpose(1,0,2,3)
        sg[sob[p], soa[q], sob[r], soa[s]] =  gab[moa[q], mob[p], moa[s], mob[r]].transpose(1,0,3,2)
        # <p,q||r,s> <- bbbb block
        sg[sob[p], sob[q], sob[r], sob[s]] = gbb[mob[p], mob[q], mob[r], mob[s]]

    return sg

def get_integrals_with_spin():
    """

    get one- and two-electron integrals from psi4, with spin

    :return nsocc: number of occupied orbitals
    :return nsvirt: number of virtual orbitals
    :return fock: the fock matrix (spin-orbital basis)
    :return gtei: antisymmetrized two-electron integrals (spin-orbital basis)

    """

    # compute the Hartree-Fock energy and wave function
    scf_e, wfn = psi4.energy('SCF', return_wfn=True)

    # number of doubly occupied orbitals
    noa = wfn.nalpha()
    nob = wfn.nbeta()
    
    # total number of orbitals
    nmo = wfn.nmo()
    
    # number of virtual orbitals
    nva = nmo - noa
    nvb = nmo - nob

    # molecular orbitals (spatial):
    Ca = wfn.Ca()
    Cb = wfn.Cb()

    # use Psi4's MintsHelper to generate integrals
    mints = psi4.core.MintsHelper(wfn.basisset())

    # build the one-electron integrals
    H = np.asarray(mints.ao_kinetic()) + np.asarray(mints.ao_potential())
    Ha = np.einsum('uj,vi,uv', Ca, Ca, H)
    Hb = np.einsum('uj,vi,uv', Cb, Cb, H)

    # build the two-electron integrals:
    g_aaaa = np.asarray(mints.mo_eri(Ca, Ca, Ca, Ca))
    g_bbbb = np.asarray(mints.mo_eri(Cb, Cb, Cb, Cb))
    g_abab = np.asarray(mints.mo_eri(Ca, Ca, Cb, Cb))

    # antisymmetrize g(ijkl) = <ij|kl> - <ij|lk> = (ik|jl) - (il|jk)
    g_aaaa = np.einsum('ikjl->ijkl', g_aaaa) - np.einsum('iljk->ijkl', g_aaaa)
    g_bbbb = np.einsum('ikjl->ijkl', g_bbbb) - np.einsum('iljk->ijkl', g_bbbb)
    g_abab = np.einsum('ikjl->ijkl', g_abab)

    # occupied slices
    oa = slice(None, noa)
    ob = slice(None, nob)

    # build spin-orbital fock matrix
    fa = Ha + np.einsum('piqi->pq', g_aaaa[:, oa, :, oa]) + np.einsum('piqi->pq', g_abab[:, ob, :, ob])
    fb = Hb + np.einsum('piqi->pq', g_bbbb[:, ob, :, ob]) + np.einsum('ipiq->pq', g_abab[oa, : , oa, :])

    return noa, nob, nva, nvb, fa, fb, g_aaaa, g_bbbb, g_abab

def get_integrals():
    """

    get one- and two-electron integrals from psi4

    :return nsocc: number of occupied spin-orbitals
    :return nsvirt: number of virtual spin-orbitals
    :return fock: the fock matrix (spin-orbital basis)
    :return gtei: antisymmetrized two-electron integrals (spin-orbital basis)

    """

    noa, nob, nva, nvb, fa, fb, g_aaaa, g_bbbb, g_abab = get_integrals_with_spin()

    nsocc  = noa + nob
    nsvirt = nva + nvb

    fock = spatial_to_spin_orbital_oei(fa, fb, (nsocc+nsvirt)//2, noa, nob)
    gtei = spatial_to_spin_orbital_tei(g_aaaa, g_abab, g_bbbb, (nsocc+nsvirt)//2, noa, nob)

    return nsocc, nsvirt, fock, gtei

def ccsd_with_spin(mol, do_eom_ccsd = False):
    """

    run ccsd, with spin

    :param mol: a psi4 molecule
    :param do_eom_ccsd: do run eom-ccsd? default false
    :return cc_energy: the total ccsd energy

    """

    nocc_a, nocc_b, nvirt_a, nvirt_b, fa, fb, g_aaaa, g_bbbb, g_abab  = get_integrals_with_spin()
    
    # occupied, virtual slices
    n = np.newaxis
    oa = slice(None, nocc_a)
    ob = slice(None, nocc_b)
    va = slice(nocc_a, None)
    vb = slice(nocc_b, None)

    # orbital energies
    row, col = fa.shape
    eps_a = np.zeros(row)
    for i in range(0,row):
        eps_a[i] = fa[i,i]

    row, col = fb.shape
    eps_b = np.zeros(row)
    for i in range(0,row):
        eps_b[i] = fb[i,i]

    # energy denominators
    e_aaaa_abij = 1 / ( - eps_a[va, n, n, n] 
                        - eps_a[n, va, n, n] 
                        + eps_a[n, n, oa, n] 
                        + eps_a[n, n, n, oa] ) 
    e_bbbb_abij = 1 / ( - eps_b[vb, n, n, n] 
                        - eps_b[n, vb, n, n] 
                        + eps_b[n, n, ob, n] 
                        + eps_b[n, n, n, ob] ) 
    e_abab_abij = 1 / ( - eps_a[va, n, n, n] 
                        - eps_b[n, vb, n, n] 
                        + eps_a[n, n, oa, n] 
                        + eps_b[n, n, n, ob] ) 

    e_aa_ai = 1 / (-eps_a[va, n] + eps_a[n, oa])
    e_bb_ai = 1 / (-eps_b[vb, n] + eps_b[n, ob])

    # hartree-fock energy
    hf_energy = ( einsum('ii', fa[oa, oa]) + einsum('ii', fb[ob, ob])
              - 0.5 * einsum('ijij', g_aaaa[oa, oa, oa, oa])
              - 0.5 * einsum('ijij', g_bbbb[ob, ob, ob, ob])
              - 1.0 * einsum('ijij', g_abab[oa, ob, oa, ob]) )

    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    #print('hartree-fock energy: {: 20.12f}'.format(hf_energy + nuclear_repulsion_energy))

    t1_aa = np.zeros((nvirt_a, nocc_a))
    t1_bb = np.zeros((nvirt_b, nocc_b))

    t2_aaaa = np.zeros((nvirt_a, nvirt_a, nocc_a, nocc_a))
    t2_bbbb = np.zeros((nvirt_b, nvirt_b, nocc_b, nocc_b))
    t2_abab = np.zeros((nvirt_a, nvirt_b, nocc_a, nocc_b))

    t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab = ccsd_iterations_with_spin(t1_aa, t1_bb,
            t2_aaaa, t2_bbbb, t2_abab, fa, fb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb, 
            e_aa_ai, e_bb_ai, e_aaaa_abij, e_bbbb_abij, e_abab_abij,
            hf_energy, e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)

    cc_energy = ccsd_energy_with_spin(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, fa, fb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)

    print("")
    print("    CCSD Correlation Energy: {: 20.12f}".format(cc_energy - hf_energy))
    print("    CCSD Total Energy:       {: 20.12f}".format(cc_energy + nuclear_repulsion_energy))
    print("")

    if not do_eom_ccsd: 
        return cc_energy + nuclear_repulsion_energy

    # now eom-ccsd?
    nstates = 11

    print("    ==> EOM-CCSD <==")
    print("")

    from eom_ccsd import HbarOperatorWithSpin

    # unique oo/vv pairs
    i_idx_a, j_idx_a = np.triu_indices(nocc_a, k=1)
    i_idx_b, j_idx_b = np.triu_indices(nocc_b, k=1)
    a_idx_a, b_idx_a = np.triu_indices(nvirt_a, k=1)
    a_idx_b, b_idx_b = np.triu_indices(nvirt_b, k=1)

    dim = 1 
    dim += nocc_a * nvirt_a
    dim += nocc_b * nvirt_b
    dim += len(i_idx_a) * len(a_idx_a)
    dim += len(i_idx_b) * len(a_idx_b)
    dim += nocc_a * nvirt_a * nocc_b * nvirt_b

    Hbar = HbarOperatorWithSpin(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, fa, fb, g_aaaa, g_bbbb, g_abab, nocc_a, nocc_b, nvirt_a, nvirt_b)
    HbarR = LinearOperator((dim, dim), matvec=Hbar.matvec, dtype=np.float64)

    ex, rvec = scipy.sparse.linalg.eigs(HbarR, k=nstates)
    idx = np.argsort(ex)
    ex = ex[idx]
    rvec = rvec[:, idx]

    print('    eigenvalues of e(-T) H e(T):')
    print('')

    print('    %5s %20s %20s' % ('state', 'total energy','excitation energy'))
    for i in range (1, nstates):
        print('    %5i %20.12f %20.12f' % ( i, ex[i].real + nuclear_repulsion_energy, ex[i].real - cc_energy ))

    print('')

    return cc_energy + nuclear_repulsion_energy

def ccsd(mol, do_eom_ccsd = False, use_spin_orbital_basis = True):
    """

    run ccsd

    :param mol: a psi4 molecule
    :param do_eom_ccsd: do run eom-ccsd? default false
    :param use_spin_orbital_basis: do use spin-obital basis? default false
    :return cc_energy: the total ccsd energy

    """

    if not use_spin_orbital_basis : 
        return ccsd_with_spin(mol, do_eom_ccsd = do_eom_ccsd)

    nsocc, nsvirt, fock, tei = get_integrals()
    
    # occupied, virtual slices
    n = np.newaxis
    o = slice(None, nsocc)
    v = slice(nsocc, None)

    # orbital energies
    row, col = fock.shape
    eps = np.zeros(row)
    for i in range(0,row):
        eps[i] = fock[i,i]

    # energy denominators
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    # hartree-fock energy
    hf_energy = 1.0 * einsum('ii', fock[o, o]) -0.5 * einsum('ijij', tei[o, o, o, o])

    t1 = np.zeros((nsvirt, nsocc))
    t2 = np.zeros((nsvirt, nsvirt, nsocc, nsocc))
    t1, t2 = ccsd_iterations(t1, t2, fock, tei, o, v, e_ai, e_abij,
                      hf_energy, e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)

    cc_energy = coupled_cluster_energy(t1, t2, fock, tei, o, v)

    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    print("")
    print("    CCSD Correlation Energy: {: 20.12f}".format(cc_energy - hf_energy))
    print("    CCSD Total Energy:       {: 20.12f}".format(cc_energy + nuclear_repulsion_energy))
    print("")

    if not do_eom_ccsd: 
        return cc_energy + nuclear_repulsion_energy

    # now eom-ccsd?
    nstates = 11

    print("    ==> EOM-CCSD <==")
    print("")

    # full diagonalization (for testing)
    full_diagonalization = False

    if full_diagonalization:

        from eom_ccsd import build_eom_ccsd_H

        # populate core list for super inefficicent implementation of CVS approximation
        core_list = []
        for i in range (0, nsocc):
            core_list.append(i)
        H = build_eom_ccsd_H(fock, tei, o, v, t1, t2, nsocc, nsvirt, core_list)

        print('    eigenvalues of e(-T) H e(T):')
        print('')

        print('    %5s %20s %20s' % ('state', 'total energy','excitation energy'))
        en, vec = np.linalg.eig(H)
        en.sort()
        for i in range (1,min(nstates,len(en))):
            print('    %5i %20.12f %20.12f' % ( i, en[i] + nuclear_repulsion_energy,en[i] - cc_energy ))

        print('')

    # sparse diagonalization

    from eom_ccsd import HbarOperator

    # unique oo/vv pairs
    i_idx, j_idx = np.triu_indices(nsocc, k=1)
    a_idx, b_idx = np.triu_indices(nsvirt, k=1)

    dim = 1 + nsocc*nsvirt + len(i_idx) * len(a_idx)

    Hbar = HbarOperator(t1, t2, fock, tei, nsocc, nsvirt)
    HbarR = LinearOperator((dim, dim), matvec=Hbar.matvec, dtype=np.float64)

    ex, rvec = scipy.sparse.linalg.eigs(HbarR, k=nstates)
    idx = np.argsort(ex)
    ex = ex[idx]
    rvec = rvec[:, idx]

    print('    eigenvalues of e(-T) H e(T):')
    print('')

    print('    %5s %20s %20s' % ('state', 'total energy','excitation energy'))
    for i in range (1, nstates):
        print('    %5i %20.12f %20.12f' % ( i, ex[i].real + nuclear_repulsion_energy, ex[i].real - cc_energy ))

    print('')

    return cc_energy + nuclear_repulsion_energy

def quccsd(mol, do_eom_ccsd = False):
    """

    run quccsd

    :param mol: a psi4 molecule
    :param do_eom_ccsd: do run eom-ccsd? default false
    :return cc_energy: the total quccsd energy

    """

    nsocc, nsvirt, fock, tei = get_integrals()
    
    # occupied, virtual slices
    n = np.newaxis
    o = slice(None, nsocc)
    v = slice(nsocc, None)

    # orbital energies
    row, col = fock.shape
    eps = np.zeros(row)
    for i in range(0,row):
        eps[i] = fock[i,i]

    # energy denominators
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    # hartree-fock energy
    hf_energy = 1.0 * einsum('ii', fock[o, o]) -0.5 * einsum('ijij', tei[o, o, o, o])

    t1 = np.zeros((nsvirt, nsocc))
    t2 = np.zeros((nsvirt, nsvirt, nsocc, nsocc))
    t1, t2 = quccsd_iterations(t1, t2, fock, tei, o, v, e_ai, e_abij,
                      hf_energy, e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)

    cc_energy = quccsd_energy(t1, t2, fock, tei, o, v)

    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    print("")
    print("    QUCCSD Correlation Energy: {: 20.12f}".format(cc_energy - hf_energy))
    print("    QUCCSD Total Energy:       {: 20.12f}".format(cc_energy + nuclear_repulsion_energy))
    print("")

    if not do_eom_ccsd: 
        return cc_energy + nuclear_repulsion_energy

    print("    error: EOM-QUCCSD is not implemented.")
    print("")

    ## now eom-ccsd?
    #print("    ==> EOM-CCSD <==")
    #print("")
    #from eom_ccsd import build_eom_quccsd_H

    ## populate core list for super inefficicent implementation of CVS approximation
    #core_list = []
    #for i in range (0, nsocc):
    #    core_list.append(i)
    #H = build_eom_quccsd_H(fock, tei, o, v, t1, t2, nsocc, nsvirt, core_list)

    #print('    eigenvalues of e(-T) H e(T):')
    #print('')

    #print('    %5s %20s %20s' % ('state', 'total energy','excitation energy'))
    #en, vec = np.linalg.eig(H)
    #en.sort()
    #for i in range (1,min(21,len(en))):
    #    print('    %5i %20.12f %20.12f' % ( i, en[i] + nuclear_repulsion_energy,en[i]-cc_energy ))

    #print('')

    #return cc_energy + nuclear_repulsion_energy

def ucc3(mol):
    """

    run ucc3

    :param mol: a psi4 molecule
    :return cc_energy: the total ucc3 energy

    """

    nsocc, nsvirt, fock, tei = get_integrals()
    
    # occupied, virtual slices
    n = np.newaxis
    o = slice(None, nsocc)
    v = slice(nsocc, None)

    # orbital energies
    row, col = fock.shape
    eps = np.zeros(row)
    for i in range(0,row):
        eps[i] = fock[i,i]

    # energy denominators
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    # hartree-fock energy
    hf_energy = 1.0 * einsum('ii', fock[o, o]) -0.5 * einsum('ijij', tei[o, o, o, o])

    t1 = np.zeros((nsvirt, nsocc))
    t2 = np.zeros((nsvirt, nsvirt, nsocc, nsocc))
    t1, t2 = ucc3_iterations(t1, t2, fock, tei, o, v, e_ai, e_abij,
                      hf_energy, e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)

    cc_energy = ucc3_energy(t1, t2, fock, tei, o, v)

    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    print("")
    print("    UCC(3) Correlation Energy: {: 20.12f}".format(cc_energy - hf_energy))
    print("    UCC(3) Total Energy:       {: 20.12f}".format(cc_energy + nuclear_repulsion_energy))
    print("")

    return cc_energy + nuclear_repulsion_energy

def ucc4(mol):
    """

    run ucc4

    :param mol: a psi4 molecule
    :return cc_energy: the total ucc4 energy

    """

    nsocc, nsvirt, fock, tei = get_integrals()
    
    # occupied, virtual slices
    n = np.newaxis
    o = slice(None, nsocc)
    v = slice(nsocc, None)

    # orbital energies
    row, col = fock.shape
    eps = np.zeros(row)
    for i in range(0,row):
        eps[i] = fock[i,i]

    # energy denominators
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    # hartree-fock energy
    hf_energy = 1.0 * einsum('ii', fock[o, o]) -0.5 * einsum('ijij', tei[o, o, o, o])

    t1 = np.zeros((nsvirt, nsocc))
    t2 = np.zeros((nsvirt, nsvirt, nsocc, nsocc))
    t1, t2 = ucc4_iterations(t1, t2, fock, tei, o, v, e_ai, e_abij,
                      hf_energy, e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)

    cc_energy = ucc4_energy(t1, t2, fock, tei, o, v)

    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    print("")
    print("    UCC(4) Correlation Energy: {: 20.12f}".format(cc_energy - hf_energy))
    print("    UCC(4) Total Energy:       {: 20.12f}".format(cc_energy + nuclear_repulsion_energy))
    print("")

    return cc_energy + nuclear_repulsion_energy

def ccsd_t(mol):
    """

    run ccsd(t)

    :param mol: a psi4 molecule
    :return cc_energy: the total ccsdt energy

    """

    nsocc, nsvirt, fock, tei = get_integrals()
    
    # occupied, virtual slices
    n = np.newaxis
    o = slice(None, nsocc)
    v = slice(nsocc, None)

    # orbital energies
    row, col = fock.shape
    eps = np.zeros(row)
    for i in range(0,row):
        eps[i] = fock[i,i]

    # energy denominators
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    # hartree-fock energy
    hf_energy = 1.0 * einsum('ii', fock[o, o]) -0.5 * einsum('ijij', tei[o, o, o, o])

    # ccsd
    t1 = np.zeros((nsvirt, nsocc))
    t2 = np.zeros((nsvirt, nsvirt, nsocc, nsocc))
    t1, t2 = ccsd_iterations(t1, t2, fock, tei, o, v, e_ai, e_abij, 
                      hf_energy, e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)

    cc_energy = coupled_cluster_energy(t1, t2, fock, tei, o, v)

    # triples 
    t3 = np.zeros((nsvirt, nsvirt, nsvirt, nsocc, nsocc, nsocc))
    e_abcijk = 1 / (-eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n] + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o])
    et = perturbative_triples_correction(t1, t2, t3, fock, tei, o, v, e_abcijk)

    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    print("")
    print("    CCSD Correlation Energy:    {: 20.12f}".format(cc_energy - hf_energy))
    print("    (T) Correction:             {: 20.12f}".format(et))
    print("    CCSD(T) Total Energy:       {: 20.12f}".format(cc_energy + et + nuclear_repulsion_energy))
    print("")

    return cc_energy + et + nuclear_repulsion_energy

def ccsdt_with_spin(mol):
    """

    run ccsdt, with spin

    :param mol: a psi4 molecule
    :return cc_energy: the total ccsdt energy

    """

    nocc_a, nocc_b, nvirt_a, nvirt_b, fa, fb, g_aaaa, g_bbbb, g_abab  = get_integrals_with_spin()
    
    # occupied, virtual slices
    n = np.newaxis
    oa = slice(None, nocc_a)
    ob = slice(None, nocc_b)
    va = slice(nocc_a, None)
    vb = slice(nocc_b, None)

    # orbital energies
    row, col = fa.shape
    eps_a = np.zeros(row)
    for i in range(0,row):
        eps_a[i] = fa[i,i]

    row, col = fb.shape
    eps_b = np.zeros(row)
    for i in range(0,row):
        eps_b[i] = fb[i,i]

    # energy denominators
    e_aaaaaa_abcijk = 1 / ( - eps_a[va, n,  n,  n,  n,  n] 
                            - eps_a[n, va,  n,  n,  n,  n] 
                            - eps_a[n,  n, va,  n,  n,  n] 
                            + eps_a[n,  n,  n, oa,  n,  n] 
                            + eps_a[n,  n,  n,  n, oa,  n] 
                            + eps_a[n,  n,  n,  n,  n, oa]  )
    e_aabaab_abcijk = 1 / ( - eps_a[va, n,  n,  n,  n,  n] 
                            - eps_a[n, va,  n,  n,  n,  n] 
                            - eps_b[n,  n, vb,  n,  n,  n] 
                            + eps_a[n,  n,  n, oa,  n,  n] 
                            + eps_a[n,  n,  n,  n, oa,  n] 
                            + eps_b[n,  n,  n,  n,  n, ob]  )
    e_abbabb_abcijk = 1 / ( - eps_a[va, n,  n,  n,  n,  n] 
                            - eps_b[n, vb,  n,  n,  n,  n] 
                            - eps_b[n,  n, vb,  n,  n,  n] 
                            + eps_a[n,  n,  n, oa,  n,  n] 
                            + eps_b[n,  n,  n,  n, ob,  n] 
                            + eps_b[n,  n,  n,  n,  n, ob]  )
    e_bbbbbb_abcijk = 1 / ( - eps_b[vb, n,  n,  n,  n,  n] 
                            - eps_b[n, vb,  n,  n,  n,  n] 
                            - eps_b[n,  n, vb,  n,  n,  n] 
                            + eps_b[n,  n,  n, ob,  n,  n] 
                            + eps_b[n,  n,  n,  n, ob,  n] 
                            + eps_b[n,  n,  n,  n,  n, ob]  )

    e_aaaa_abij = 1 / ( - eps_a[va, n, n, n] 
                        - eps_a[n, va, n, n] 
                        + eps_a[n, n, oa, n] 
                        + eps_a[n, n, n, oa] ) 
    e_bbbb_abij = 1 / ( - eps_b[vb, n, n, n] 
                        - eps_b[n, vb, n, n] 
                        + eps_b[n, n, ob, n] 
                        + eps_b[n, n, n, ob] ) 
    e_abab_abij = 1 / ( - eps_a[va, n, n, n] 
                        - eps_b[n, vb, n, n] 
                        + eps_a[n, n, oa, n] 
                        + eps_b[n, n, n, ob] ) 

    e_aa_ai = 1 / (-eps_a[va, n] + eps_a[n, oa])
    e_bb_ai = 1 / (-eps_b[vb, n] + eps_b[n, ob])

    # hartree-fock energy
    hf_energy = ( einsum('ii', fa[oa, oa]) + einsum('ii', fb[ob, ob])
              - 0.5 * einsum('ijij', g_aaaa[oa, oa, oa, oa])
              - 0.5 * einsum('ijij', g_bbbb[ob, ob, ob, ob])
              - 1.0 * einsum('ijij', g_abab[oa, ob, oa, ob]) )

    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    #print('hartree-fock energy: {: 20.12f}'.format(hf_energy + nuclear_repulsion_energy))

    t1_aa = np.zeros((nvirt_a, nocc_a))
    t1_bb = np.zeros((nvirt_b, nocc_b))

    t2_aaaa = np.zeros((nvirt_a, nvirt_a, nocc_a, nocc_a))
    t2_bbbb = np.zeros((nvirt_b, nvirt_b, nocc_b, nocc_b))
    t2_abab = np.zeros((nvirt_a, nvirt_b, nocc_a, nocc_b))

    t3_aaaaaa = np.zeros((nvirt_a, nvirt_a, nvirt_a, nocc_a, nocc_a, nocc_a))
    t3_aabaab = np.zeros((nvirt_a, nvirt_a, nvirt_b, nocc_a, nocc_a, nocc_b))
    t3_abbabb = np.zeros((nvirt_a, nvirt_b, nvirt_b, nocc_a, nocc_b, nocc_b))
    t3_bbbbbb = np.zeros((nvirt_b, nvirt_b, nvirt_b, nocc_b, nocc_b, nocc_b))

    t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb = ccsdt_iterations_with_spin(t1_aa, t1_bb,
            t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, fa, fb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb, 
            e_aa_ai, e_bb_ai, e_aaaa_abij, e_bbbb_abij, e_abab_abij, e_aaaaaa_abcijk, e_aabaab_abcijk, e_abbabb_abcijk, e_bbbbbb_abcijk,
            hf_energy, e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)

    cc_energy = ccsd_energy_with_spin(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, fa, fb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)

    print("")
    print("    CCSDT Correlation Energy: {: 20.12f}".format(cc_energy - hf_energy))
    print("    CCSDT Total Energy:       {: 20.12f}".format(cc_energy + nuclear_repulsion_energy))
    print("")

    return cc_energy + nuclear_repulsion_energy

def ccsdt(mol, use_spin_orbital_basis = True):
    """

    run ccsdt

    :param mol: a psi4 molecule
    :return cc_energy: the total ccsdt energy

    """

    if not use_spin_orbital_basis : 
        return ccsdt_with_spin(mol)

    nsocc, nsvirt, fock, tei = get_integrals()
    
    # occupied, virtual slices
    n = np.newaxis
    o = slice(None, nsocc)
    v = slice(nsocc, None)

    # orbital energies
    row, col = fock.shape
    eps = np.zeros(row)
    for i in range(0,row):
        eps[i] = fock[i,i]

    # energy denominators
    e_abcijk = 1 / (-eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n] + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o])
    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    # hartree-fock energy
    hf_energy = 1.0 * einsum('ii', fock[o, o]) -0.5 * einsum('ijij', tei[o, o, o, o])

    t1 = np.zeros((nsvirt, nsocc))
    t2 = np.zeros((nsvirt, nsvirt, nsocc, nsocc))
    t3 = np.zeros((nsvirt, nsvirt, nsvirt, nsocc, nsocc, nsocc))
    t1, t2, t3 = ccsdt_iterations(t1, t2, t3, fock, tei, o, v, e_ai, e_abij, e_abcijk,
                      hf_energy, e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)

    #t1, t2 = ccsd_iterations(t1, t2, fock, tei, o, v, e_ai, e_abij,
    #                  hf_energy, e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)

    cc_energy = coupled_cluster_energy(t1, t2, fock, tei, o, v)

    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    print("")
    print("    CCSDT Correlation Energy: {: 20.12f}".format(cc_energy - hf_energy))
    print("    CCSDT Total Energy:       {: 20.12f}".format(cc_energy + nuclear_repulsion_energy))
    print("")

    return cc_energy + nuclear_repulsion_energy

def ccsdtq(mol):
    """

    run ccsdtq, with spin

    :param mol: a psi4 molecule
    :return cc_energy: the total ccsdt energy

    """

    nocc_a, nocc_b, nvirt_a, nvirt_b, fa, fb, g_aaaa, g_bbbb, g_abab  = get_integrals_with_spin()
    
    # occupied, virtual slices
    n = np.newaxis
    oa = slice(None, nocc_a)
    ob = slice(None, nocc_b)
    va = slice(nocc_a, None)
    vb = slice(nocc_b, None)

    # orbital energies
    row, col = fa.shape
    eps_a = np.zeros(row)
    for i in range(0,row):
        eps_a[i] = fa[i,i]

    row, col = fb.shape
    eps_b = np.zeros(row)
    for i in range(0,row):
        eps_b[i] = fb[i,i]

    # energy denominators
    e_aaaaaaaa_abcdijkl = 1 / ( - eps_a[va, n,  n,  n,  n,  n,  n,  n] 
                                - eps_a[n, va,  n,  n,  n,  n,  n,  n] 
                                - eps_a[n,  n, va,  n,  n,  n,  n,  n] 
                                + eps_a[n,  n,  n, va,  n,  n,  n,  n] 
                                + eps_a[n,  n,  n,  n, oa,  n,  n,  n] 
                                + eps_a[n,  n,  n,  n,  n, oa,  n,  n] 
                                + eps_a[n,  n,  n,  n,  n,  n, oa,  n] 
                                + eps_a[n,  n,  n,  n,  n,  n,  n, oa] )
    e_aaabaaab_abcdijkl = 1 / ( - eps_a[va, n,  n,  n,  n,  n,  n,  n] 
                                - eps_a[n, va,  n,  n,  n,  n,  n,  n] 
                                - eps_a[n,  n, va,  n,  n,  n,  n,  n] 
                                + eps_b[n,  n,  n, vb,  n,  n,  n,  n] 
                                + eps_a[n,  n,  n,  n, oa,  n,  n,  n] 
                                + eps_a[n,  n,  n,  n,  n, oa,  n,  n] 
                                + eps_a[n,  n,  n,  n,  n,  n, oa,  n] 
                                + eps_b[n,  n,  n,  n,  n,  n,  n, ob] )
    e_aabbaabb_abcdijkl = 1 / ( - eps_a[va, n,  n,  n,  n,  n,  n,  n] 
                                - eps_a[n, va,  n,  n,  n,  n,  n,  n] 
                                - eps_b[n,  n, vb,  n,  n,  n,  n,  n] 
                                + eps_b[n,  n,  n, vb,  n,  n,  n,  n] 
                                + eps_a[n,  n,  n,  n, oa,  n,  n,  n] 
                                + eps_a[n,  n,  n,  n,  n, oa,  n,  n] 
                                + eps_b[n,  n,  n,  n,  n,  n, ob,  n] 
                                + eps_b[n,  n,  n,  n,  n,  n,  n, ob] )
    e_abbbabbb_abcdijkl = 1 / ( - eps_a[va, n,  n,  n,  n,  n,  n,  n] 
                                - eps_b[n, vb,  n,  n,  n,  n,  n,  n] 
                                - eps_b[n,  n, vb,  n,  n,  n,  n,  n] 
                                + eps_b[n,  n,  n, vb,  n,  n,  n,  n] 
                                + eps_a[n,  n,  n,  n, oa,  n,  n,  n] 
                                + eps_b[n,  n,  n,  n,  n, ob,  n,  n] 
                                + eps_b[n,  n,  n,  n,  n,  n, ob,  n] 
                                + eps_b[n,  n,  n,  n,  n,  n,  n, ob] )
    e_bbbbbbbb_abcdijkl = 1 / ( - eps_b[vb, n,  n,  n,  n,  n,  n,  n] 
                                - eps_b[n, vb,  n,  n,  n,  n,  n,  n] 
                                - eps_b[n,  n, vb,  n,  n,  n,  n,  n] 
                                + eps_b[n,  n,  n, vb,  n,  n,  n,  n] 
                                + eps_b[n,  n,  n,  n, ob,  n,  n,  n] 
                                + eps_b[n,  n,  n,  n,  n, ob,  n,  n] 
                                + eps_b[n,  n,  n,  n,  n,  n, ob,  n] 
                                + eps_b[n,  n,  n,  n,  n,  n,  n, ob] )

    e_aaaaaa_abcijk = 1 / ( - eps_a[va, n,  n,  n,  n,  n] 
                            - eps_a[n, va,  n,  n,  n,  n] 
                            - eps_a[n,  n, va,  n,  n,  n] 
                            + eps_a[n,  n,  n, oa,  n,  n] 
                            + eps_a[n,  n,  n,  n, oa,  n] 
                            + eps_a[n,  n,  n,  n,  n, oa]  )
    e_aabaab_abcijk = 1 / ( - eps_a[va, n,  n,  n,  n,  n] 
                            - eps_a[n, va,  n,  n,  n,  n] 
                            - eps_b[n,  n, vb,  n,  n,  n] 
                            + eps_a[n,  n,  n, oa,  n,  n] 
                            + eps_a[n,  n,  n,  n, oa,  n] 
                            + eps_b[n,  n,  n,  n,  n, ob]  )
    e_abbabb_abcijk = 1 / ( - eps_a[va, n,  n,  n,  n,  n] 
                            - eps_b[n, vb,  n,  n,  n,  n] 
                            - eps_b[n,  n, vb,  n,  n,  n] 
                            + eps_a[n,  n,  n, oa,  n,  n] 
                            + eps_b[n,  n,  n,  n, ob,  n] 
                            + eps_b[n,  n,  n,  n,  n, ob]  )
    e_bbbbbb_abcijk = 1 / ( - eps_b[vb, n,  n,  n,  n,  n] 
                            - eps_b[n, vb,  n,  n,  n,  n] 
                            - eps_b[n,  n, vb,  n,  n,  n] 
                            + eps_b[n,  n,  n, ob,  n,  n] 
                            + eps_b[n,  n,  n,  n, ob,  n] 
                            + eps_b[n,  n,  n,  n,  n, ob]  )

    e_aaaa_abij = 1 / ( - eps_a[va, n, n, n] 
                        - eps_a[n, va, n, n] 
                        + eps_a[n, n, oa, n] 
                        + eps_a[n, n, n, oa] ) 
    e_bbbb_abij = 1 / ( - eps_b[vb, n, n, n] 
                        - eps_b[n, vb, n, n] 
                        + eps_b[n, n, ob, n] 
                        + eps_b[n, n, n, ob] ) 
    e_abab_abij = 1 / ( - eps_a[va, n, n, n] 
                        - eps_b[n, vb, n, n] 
                        + eps_a[n, n, oa, n] 
                        + eps_b[n, n, n, ob] ) 

    e_aa_ai = 1 / (-eps_a[va, n] + eps_a[n, oa])
    e_bb_ai = 1 / (-eps_b[vb, n] + eps_b[n, ob])

    # hartree-fock energy
    hf_energy = ( einsum('ii', fa[oa, oa]) + einsum('ii', fb[ob, ob])
              - 0.5 * einsum('ijij', g_aaaa[oa, oa, oa, oa])
              - 0.5 * einsum('ijij', g_bbbb[ob, ob, ob, ob])
              - 1.0 * einsum('ijij', g_abab[oa, ob, oa, ob]) )

    nuclear_repulsion_energy = mol.nuclear_repulsion_energy()

    #print('hartree-fock energy: {: 20.12f}'.format(hf_energy + nuclear_repulsion_energy))

    t1_aa = np.zeros((nvirt_a, nocc_a))
    t1_bb = np.zeros((nvirt_b, nocc_b))

    t2_aaaa = np.zeros((nvirt_a, nvirt_a, nocc_a, nocc_a))
    t2_bbbb = np.zeros((nvirt_b, nvirt_b, nocc_b, nocc_b))
    t2_abab = np.zeros((nvirt_a, nvirt_b, nocc_a, nocc_b))

    t3_aaaaaa = np.zeros((nvirt_a, nvirt_a, nvirt_a, nocc_a, nocc_a, nocc_a))
    t3_aabaab = np.zeros((nvirt_a, nvirt_a, nvirt_b, nocc_a, nocc_a, nocc_b))
    t3_abbabb = np.zeros((nvirt_a, nvirt_b, nvirt_b, nocc_a, nocc_b, nocc_b))
    t3_bbbbbb = np.zeros((nvirt_b, nvirt_b, nvirt_b, nocc_b, nocc_b, nocc_b))

    t4_aaaaaaaa = np.zeros((nvirt_a, nvirt_a, nvirt_a, nvirt_a, nocc_a, nocc_a, nocc_a, nocc_a))
    t4_aaabaaab = np.zeros((nvirt_a, nvirt_a, nvirt_a, nvirt_b, nocc_a, nocc_a, nocc_a, nocc_b))
    t4_aabbaabb = np.zeros((nvirt_a, nvirt_a, nvirt_b, nvirt_b, nocc_a, nocc_a, nocc_b, nocc_b))
    t4_abbbabbb = np.zeros((nvirt_a, nvirt_b, nvirt_b, nvirt_b, nocc_a, nocc_b, nocc_b, nocc_b))
    t4_bbbbbbbb = np.zeros((nvirt_b, nvirt_b, nvirt_b, nvirt_b, nocc_b, nocc_b, nocc_b, nocc_b))

    t1_aa, t1_bb, \
    t2_aaaa, t2_bbbb, t2_abab, \
    t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, \
    t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb = \
        ccsdtq_iterations_with_spin(t1_aa, t1_bb,
                                    t2_aaaa, t2_bbbb, t2_abab, 
                                    t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, 
                                    t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb,
                                    fa, fb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb, 
                                    e_aa_ai, e_bb_ai, 
                                    e_aaaa_abij, e_bbbb_abij, e_abab_abij, 
                                    e_aaaaaa_abcijk, e_aabaab_abcijk, e_abbabb_abcijk, e_bbbbbb_abcijk,
                                    e_aaaaaaaa_abcdijkl, e_aaabaaab_abcdijkl, e_aabbaabb_abcdijkl, e_abbbabbb_abcdijkl, e_bbbbbbbb_abcdijkl,
                                    hf_energy, e_convergence=1e-8, r_convergence=1e-4, diis_size=8, diis_start_cycle=4)

    cc_energy = ccsd_energy_with_spin(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, fa, fb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)

    print("")
    print("    CCSDTQ Correlation Energy: {: 20.12f}".format(cc_energy - hf_energy))
    print("    CCSDTQ Total Energy:       {: 20.12f}".format(cc_energy + nuclear_repulsion_energy))
    print("")

    return cc_energy + nuclear_repulsion_energy

if __name__ == "__main__":
    main()
