#
# pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
# Copyright (C) 2026 A. Eugene DePrince III
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
generate and return integrals from psi4
"""

import numpy as np
from numpy import einsum

# psi4
import psi4

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

def spatial_to_spin_orbital_tei(gaa, gab, gbb, n, noa, nob, antisymmetrize_eris = True):
    """

    :param gaa: antisymmetrized two-electron integrals in physicist' notation, alpha-alpha portion
    :param gab: two-electron integrals in physicist' notation, alpha-beta portion
    :param gbb: antisymmetrized two-electron integrals in physicist' notation, beta-beta portion
    :param n: number of spatial orbitals
    :param noa: number of alpha occupied orbitals
    :param nob: number of beta occupied orbitals
    :param antisymmetrize_eris: do antisymmetrize the eris?
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
        sg[sob[p], soa[q], sob[r], soa[s]] =  gab[moa[q], mob[p], moa[s], mob[r]].transpose(1,0,3,2)
        if antisymmetrize_eris:
            sg[soa[p], sob[q], sob[r], soa[s]] = -gab[moa[p], mob[q], moa[s], mob[r]].transpose(0,1,3,2)
            sg[sob[p], soa[q], soa[r], sob[s]] = -gab[moa[q], mob[p], moa[r], mob[s]].transpose(1,0,2,3)
        # <p,q||r,s> <- bbbb block
        sg[sob[p], sob[q], sob[r], sob[s]] = gbb[mob[p], mob[q], mob[r], mob[s]]

    return sg

def get_quadrupole_integrals_with_spin(wfn):
    """

    get quadrupole integrals from psi4, with spin

    :param wfn: psi4 wave function object

    :return q_aa: alpha-spin quadrupole integrals
    :return q_bb: beta-spin quadrupole integrals

    """

    # molecular orbitals (spatial):
    Ca = wfn.Ca()
    Cb = wfn.Cb()

    # use Psi4's MintsHelper to generate integrals
    mints = psi4.core.MintsHelper(wfn.basisset())

    q = np.asarray(mints.ao_quadrupole())
    q_aa = []
    q_bb = []
    for i in range (len(q)):
        q_aa.append(np.einsum('uj,vi,uv', Ca, Ca, q[i]))
        q_bb.append(np.einsum('uj,vi,uv', Cb, Cb, q[i]))

    return q_aa, q_bb

def get_quadrupole_integrals(wfn):
    """
    get quadrupole integrals from psi4, in spin-orbital basis

    :param wfn: psi4 wave function object

    :return q: spin-orbital basis quadrupole integrals

    """

    # number of occupied orbitals
    noa = wfn.nalpha()
    nob = wfn.nbeta()

    # total number of orbitals
    nmo = wfn.nmo()

    q_aa, q_bb = get_quadrupole_integrals_with_spin(wfn)

    q = []
    for i in range (len(q_aa)):
        q.append(spatial_to_spin_orbital_oei(q_aa[i], q_bb[i], nmo, noa, nob))

    return q

def get_dipole_integrals_with_spin(wfn):
    """

    get dipole integrals from psi4, with spin

    :param wfn: psi4 wave function object

    :return mu_aa: alpha-spin dipole integrals
    :return mu_bb: beta-spin dipole integrals

    """

    # molecular orbitals (spatial):
    Ca = wfn.Ca()
    Cb = wfn.Cb()

    # use Psi4's MintsHelper to generate integrals
    mints = psi4.core.MintsHelper(wfn.basisset())

    mu = np.asarray(mints.ao_dipole())
    mu_aa = []
    mu_bb = []
    for i in range (len(mu)):
        mu_aa.append(np.einsum('uj,vi,uv', Ca, Ca, mu[i]))
        mu_bb.append(np.einsum('uj,vi,uv', Cb, Cb, mu[i]))

    return mu_aa, mu_bb

def get_dipole_integrals(wfn):
    """
    get dipole integrals from psi4, in spin-orbital basis

    :param wfn: psi4 wave function object

    :return mu: spin-orbital basis dipole integrals

    """

    # number of occupied orbitals
    noa = wfn.nalpha()
    nob = wfn.nbeta()

    # total number of orbitals
    nmo = wfn.nmo()

    mu_aa, mu_bb = get_dipole_integrals_with_spin(wfn)

    mu = []
    for i in range (len(mu_aa)):
        mu.append(spatial_to_spin_orbital_oei(mu_aa[i], mu_bb[i], nmo, noa, nob))

    return mu

def get_df_integrals_with_spin(wfn):
    """

    get one-electron and three-center integrals from psi4, with spin

    :param wfn: psi4 wave function object

    :return noa: number of occupied orbitals (alpha)
    :return nob: number of occupied orbitals (beta)
    :return nva: number of virtual orbitals (alpha)
    :return nvb: number of virtual orbitals (beta)
    :return fa: the fock matrix (alpha)
    :return fb: the fock matrix (beta)
    :return B_aa: three-center integrals (aa)
    :return B_bb: three-center integrals (bb)

    """

    # number of doubly occupied orbitals
    noa = wfn.nalpha()
    nob = wfn.nbeta()
    
    # total number of orbitals
    nmo = wfn.nmo()
    
    # number of virtual orbitals
    nva = nmo - noa
    nvb = nmo - nob

    # use Psi4's MintsHelper to generate integrals
    mints = psi4.core.MintsHelper(wfn.basisset())

    aux_basisset = wfn.get_basisset("DF_BASIS_SCF")

    df = psi4.core.DFHelper(wfn.basisset(), aux_basisset)
    df.set_method("DIRECT")
    df.initialize()

    df.add_space("va", wfn.Ca_subset("AO", "VIR"))
    df.add_space("oa", wfn.Ca_subset("AO", "OCC"))

    df.add_space("vb", wfn.Cb_subset("AO", "VIR"))
    df.add_space("ob", wfn.Cb_subset("AO", "OCC"))

    df.add_transformation("Bov_aa", "oa", "va")
    df.add_transformation("Bov_bb", "ob", "vb")

    df.transform()

    Bov_aa = df.get_tensor("Bov_aa").np
    Bov_bb = df.get_tensor("Bov_bb").np

    eps_a = wfn.epsilon_a().np
    eps_b = wfn.epsilon_b().np

    return noa, nob, nva, nvb, eps_a, eps_b, Bov_aa, Bov_bb

def get_integrals_with_spin(wfn, antisymmetrize_eris = True, nfzc = 0):
    """

    get one- and two-electron integrals from psi4, with spin

    :param wfn: psi4 wave function object
    :param antisymmetrize_eris: do antisymmetrize the eris
    :param nfzc: number of frozen core orbitals

    :return noa: number of occupied orbitals (alpha)
    :return nob: number of occupied orbitals (beta)
    :return nva: number of virtual orbitals (alpha)
    :return nvb: number of virtual orbitals (beta)
    :return fa: the fock matrix (alpha)
    :return fb: the fock matrix (beta)
    :return g_aaaa: antisymmetrized two-electron integrals (aaaa)
    :return g_bbbb: antisymmetrized two-electron integrals (bbbb)
    :return g_abab: antisymmetrized two-electron integrals (abab)
    :return efzc: frozen core energy

    """

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
    g_aaaa = np.einsum('ikjl->ijkl', g_aaaa)
    g_bbbb = np.einsum('ikjl->ijkl', g_bbbb)
    g_abab = np.einsum('ikjl->ijkl', g_abab)
    if antisymmetrize_eris:
        g_aaaa -= np.einsum('ijlk->ijkl', g_aaaa)
        g_bbbb -= np.einsum('ijlk->ijkl', g_bbbb)

    # occupied slices
    oa = slice(None, noa)
    ob = slice(None, nob)

    # build spin-orbital fock matrix
    fa = Ha + np.einsum('piqi->pq', g_aaaa[:, oa, :, oa]) + np.einsum('piqi->pq', g_abab[:, ob, :, ob])
    fb = Hb + np.einsum('piqi->pq', g_bbbb[:, ob, :, ob]) + np.einsum('ipiq->pq', g_abab[oa, : , oa, :])
    if not antisymmetrize_eris:
        fa -= np.einsum('piiq->pq', g_aaaa[:, oa, oa, :])
        fb -= np.einsum('piiq->pq', g_bbbb[:, ob, ob, :])

    # frozen-core energy
    efzc = ( einsum('ii', Ha[:nfzc, :nfzc]) + einsum('ii', Hb[:nfzc, :nfzc])
           + 0.5 * einsum('ijij', g_aaaa[:nfzc, :nfzc, :nfzc, :nfzc])
           + 0.5 * einsum('ijij', g_bbbb[:nfzc, :nfzc, :nfzc, :nfzc])
           + 1.0 * einsum('ijij', g_abab[:nfzc, :nfzc, :nfzc, :nfzc]) )

    # adjust for frozen core
    noa -= nfzc
    nob -= nfzc
    fa = fa[nfzc:, nfzc:]
    fb = fb[nfzc:, nfzc:]
    g_aaaa = g_aaaa[nfzc:, nfzc:, nfzc:, nfzc:]
    g_bbbb = g_bbbb[nfzc:, nfzc:, nfzc:, nfzc:]
    g_abab = g_abab[nfzc:, nfzc:, nfzc:, nfzc:]

    return noa, nob, nva, nvb, fa, fb, g_aaaa, g_bbbb, g_abab, efzc

def get_integrals(wfn, antisymmetrize_eris = True, nfzc = 0):
    """

    get one- and two-electron integrals from psi4

    :param wfn: psi4 wave function object
    :param antisymmetrize_eris: do antisymmetrize the eris?
    :param nfzc: number of frozen core orbitals

    :return nsocc: number of occupied spin-orbitals
    :return nsvirt: number of virtual spin-orbitals
    :return fock: the fock matrix (spin-orbital basis)
    :return gtei: (antisymmetrized) two-electron integrals (spin-orbital basis)
    :return efzc: frozen core energy

    """

    noa, nob, nva, nvb, fa, fb, g_aaaa, g_bbbb, g_abab, efzc = get_integrals_with_spin(wfn, antisymmetrize_eris = antisymmetrize_eris, nfzc = nfzc)

    nsocc  = noa + nob
    nsvirt = nva + nvb

    fock = spatial_to_spin_orbital_oei(fa, fb, (nsocc+nsvirt)//2, noa, nob)
    gtei = spatial_to_spin_orbital_tei(g_aaaa, g_abab, g_bbbb, (nsocc+nsvirt)//2, noa, nob, antisymmetrize_eris = antisymmetrize_eris)

    return nsocc, nsvirt, fock, gtei, efzc

