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

import numpy as np
from numpy import einsum

# psi4
import psi4

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

def spatial_to_spin_orbital_oei(h, n, no):
    """
    get spin-orbital-basis one-electron integrals

    :param h: one-electron orbitals
    :param n: number of spatial orbitals
    :param no: number of (doubly) occupied orbitals
    :return:  spin-orbital one-electron integrals, sh
    """

    # build spin-orbital oeis
    sh = np.zeros((2*n,2*n))

    # index 1
    for i in range(0,n):
        ia = i
        ib = i
    
        # alpha occ do nothing
        if ( ia < no ):
            ia = i
        # alpha vir shift up by no
        else :
            ia += no
        # beta occ
        if ( ib < no ):
            ib += no
        else :
            ib += n
    
        # index 2
        for j in range(0,n):
            ja = j
            jb = j
    
            # alpha occ
            if ( ja < no ):
                ja = j
            # alpha vir
            else :
                ja += no
            # beta occ
            if ( jb < no ):
                jb += no
            # beta vir
            else :
                jb += n

            # Haa
            sh[ia,ja] = h[i,j]
            # Hbb
            sh[ib,jb] = h[i,j]
    
    return sh

def spatial_to_spin_orbital_tei(g, n, no):
    """
    get spin-orbital-basis two-electron integrals

    :param g: two-electron integrals in chemists' notation
    :param n: number of spatial orbitals
    :param no: number of (doubly) occupied orbitals
    :return:  spin-orbital two-electron integrals, sg
    """

    # build spin-orbital teis
    sg = np.zeros((2*n,2*n,2*n,2*n))

    # index 1
    for i in range(0,n):
        ia = i
        ib = i
    
        # alpha occ do nothing
        if ( ia < no ):
            ia = i
        # alpha vir shift up by no
        else :
            ia += no
        # beta occ
        if ( ib < no ):
            ib += no
        else :
            ib += n
    
        # index 2
        for j in range(0,n):
            ja = j
            jb = j
    
            # alpha occ
            if ( ja < no ):
                ja = j
            # alpha vir
            else :
                ja += no
            # beta occ
            if ( jb < no ):
                jb += no
            # beta vir
            else :
                jb += n

            # index 3
            for k in range(0,n):
                ka = k
                kb = k
    
                # alpha occ
                if ( ka < no ):
                    ka = k
                # alpha vir
                else :
                    ka += no
                # beta occ
                if ( kb < no ):
                    kb += no
                # beta vir
                else :
                    kb += n
    
                # index 4
                for l in range(0,n):
                    la = l
                    lb = l
    
                    # alpha occ
                    if ( la < no ):
                        la = l
                    # alpha vir
                    else :
                        la += no
                    # beta occ
                    if ( lb < no ):
                        lb += no
                    # beta vir
                    else :
                        lb += n
                     
                    # (aa|aa)
                    sg[ia,ja,ka,la] = g[i,j,k,l]
                    # (aa|bb)
                    sg[ia,ja,kb,lb] = g[i,j,k,l]
                    # (bb|aa)
                    sg[ib,jb,ka,la] = g[i,j,k,l]
                    # (bb|bb)
                    sg[ib,jb,kb,lb] = g[i,j,k,l]
    
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

    # orbital energies
    epsilon_a     = np.asarray(wfn.epsilon_a())
    epsilon_b     = np.asarray(wfn.epsilon_b())
    
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

    # occupied, virtual slices
    n = np.newaxis
    oa = slice(None, noa)
    ob = slice(None, nob)
    va = slice(noa, None)
    vb = slice(nob, None)

    # build spin-orbital fock matrix
    fa = Ha + np.einsum('piqi->pq', g_aaaa[:, oa, :, oa]) + np.einsum('piqi->pq', g_abab[:, ob, :, ob])
    fb = Hb + np.einsum('piqi->pq', g_bbbb[:, ob, :, ob]) + np.einsum('ipiq->pq', g_abab[oa, : , oa, :])

    return noa, nob, nva, nvb, fa, fb, g_aaaa, g_bbbb, g_abab 

def get_integrals():
    """

    get one- and two-electron integrals from psi4

    :return nsocc: number of occupied orbitals
    :return nsvirt: number of virtual orbitals
    :return fock: the fock matrix (spin-orbital basis)
    :return gtei: antisymmetrized two-electron integrals (spin-orbital basis)

    """

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

    nsvirt = 2 * nv
    nsocc = 2 * no

    return nsocc, nsvirt, fock, gtei

def ccsd_with_spin(mol):
    """

    run ccsd, with spin

    :param mol: a psi4 molecule
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
        return ccsd_with_spin(mol)

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
    print("    ==> EOM-CCSD <==")
    print("")
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
    for i in range (1,min(21,len(en))):
        print('    %5i %20.12f %20.12f' % ( i, en[i] + nuclear_repulsion_energy,en[i]-cc_energy ))

    print('')

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
