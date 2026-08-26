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
generate and return integrals from pyscf

This is the pyscf-backed counterpart of pdaggerq.numerical.utils.integrals 
"""

import numpy as np
from numpy import einsum

# pyscf
from pyscf import ao2mo


def _spin_mo_data(mf):
    """
    pull (Ca, Cb, noa, nob) out of a pyscf mean-field object, regardless of whether it is
    restricted (RHF/ROHF -- one shared orbital set, mo_occ in {0, 1, 2}) or unrestricted
    (UHF -- independent alpha/beta orbital sets and occupation vectors).

    :param mf: a converged pyscf mean-field object (RHF, ROHF, or UHF)

    :return Ca: alpha-spin MO coefficients
    :return Cb: beta-spin MO coefficients
    :return noa: number of occupied alpha orbitals
    :return nob: number of occupied beta orbitals
    """

    # pyscf stores UHF's mo_coeff/mo_occ as either a 2-tuple of (nao,nmo)/(nmo,) arrays or a
    # single stacked (2,nao,nmo)/(2,nmo) array, depending on version -- np.asarray(...) turns
    # either representation into an actual leading dimension we can check and index the same way.
    mo_coeff = np.asarray(mf.mo_coeff)
    if mo_coeff.ndim == 3:
        # UHF: independent alpha/beta orbitals and occupations
        Ca, Cb = mo_coeff[0], mo_coeff[1]
        occ = np.asarray(mf.mo_occ)
        occ_a, occ_b = occ[0], occ[1]
        noa = int(np.sum(occ_a > 0.5))
        nob = int(np.sum(occ_b > 0.5))
    else:
        # RHF/ROHF: one shared orbital set; occupation numbers are 0/2 (RHF) or 0/1/2 (ROHF)
        Ca = Cb = mo_coeff
        occ = np.asarray(mf.mo_occ)
        noa = int(np.sum(occ > 0.5))
        nob = int(np.sum(occ > 1.5))

    return Ca, Cb, noa, nob


def spatial_to_spin_orbital_oei(ha, hb, n, noa, nob):
    """
    identical to pdaggerq.numerical.utils.integrals.spatial_to_spin_orbital_oei -- duplicated
    here so this module has no dependency on the psi4-backed one.

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
    identical to pdaggerq.numerical.utils.integrals.spatial_to_spin_orbital_tei -- duplicated
    here so this module has no dependency on the psi4-backed one.

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


def get_quadrupole_integrals_with_spin(mf, nfzc = 0):
    """

    get quadrupole integrals from pyscf, with spin

    Raw second-moment integrals <mu|r_i r_j|nu> (no trace removal, no electrostatic
    prefactor) in the six-component order XX, XY, XZ, YY, YZ, ZZ -- matching the ordering
    of psi4's mints.ao_quadrupole().

    :param mf: a converged pyscf mean-field object (RHF, ROHF, or UHF)
    :param nfzc: number of frozen core orbitals

    :return q_aa: alpha-spin quadrupole integrals
    :return q_bb: beta-spin quadrupole integrals

    """

    # molecular orbitals (spatial):
    Ca, Cb, _, _ = _spin_mo_data(mf)
    mol = mf.mol

    # AO quadrupole integrals: <mu| r_i r_j |nu>, shape (3, 3, nao, nao)
    nao = mol.nao_nr()
    rr = mol.intor('int1e_rr').reshape(3, 3, nao, nao)
    comps = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]  # XX, XY, XZ, YY, YZ, ZZ

    q_aa = []
    q_bb = []
    for (i, j) in comps:
        q_aa.append(-np.einsum('up,vq,uv->pq', Ca, Ca, rr[i, j])[nfzc:, nfzc:])
        q_bb.append(-np.einsum('up,vq,uv->pq', Cb, Cb, rr[i, j])[nfzc:, nfzc:])

    return q_aa, q_bb


def get_quadrupole_integrals(mf):
    """
    get quadrupole integrals from pyscf, in spin-orbital basis

    :param mf: a converged pyscf mean-field object (RHF, ROHF, or UHF)

    :return q: spin-orbital basis quadrupole integrals

    """

    # number of occupied orbitals
    Ca, Cb, noa, nob = _spin_mo_data(mf)

    # total number of orbitals
    nmo = Ca.shape[1]

    q_aa, q_bb = get_quadrupole_integrals_with_spin(mf)

    q = []
    for i in range (len(q_aa)):
        q.append(spatial_to_spin_orbital_oei(q_aa[i], q_bb[i], nmo, noa, nob))

    return q


def get_dipole_integrals_with_spin(mf, nfzc = 0):
    """

    get dipole integrals from pyscf, with spin

    :param mf: a converged pyscf mean-field object (RHF, ROHF, or UHF)
    :param nfzc: number of frozen core orbitals

    :return mu_aa: alpha-spin dipole integrals
    :return mu_bb: beta-spin dipole integrals

    """

    # molecular orbitals (spatial):
    Ca, Cb, _, _ = _spin_mo_data(mf)
    mol = mf.mol

    # AO dipole integrals: <mu| r_i |nu>, shape (3, nao, nao)
    r = mol.intor('int1e_r')

    mu_aa = []
    mu_bb = []
    for i in range (3):
        mu_aa.append(-np.einsum('up,vq,uv->pq', Ca, Ca, r[i])[nfzc:, nfzc:])
        mu_bb.append(-np.einsum('up,vq,uv->pq', Cb, Cb, r[i])[nfzc:, nfzc:])

    return mu_aa, mu_bb


def get_dipole_integrals(mf):
    """
    get dipole integrals from pyscf, in spin-orbital basis

    :param mf: a converged pyscf mean-field object (RHF, ROHF, or UHF)

    :return mu: spin-orbital basis dipole integrals

    """

    # number of occupied orbitals
    Ca, Cb, noa, nob = _spin_mo_data(mf)

    # total number of orbitals
    nmo = Ca.shape[1]

    mu_aa, mu_bb = get_dipole_integrals_with_spin(mf)

    mu = []
    for i in range (len(mu_aa)):
        mu.append(spatial_to_spin_orbital_oei(mu_aa[i], mu_bb[i], nmo, noa, nob))

    return mu


def get_core_hamiltonian_with_spin(mf, nfzc = 0):
    """

    get core Hamiltonian integrals from pyscf, with spin

    :param mf: a converged pyscf mean-field object (RHF, ROHF, or UHF)
    :param nfzc: number of frozen core orbitals

    :return Ha: the core Hamiltonian matrix (alpha)
    :return Hb: the core Hamiltonian matrix (beta)

    """

    # molecular orbitals (spatial):
    Ca, Cb, _, _ = _spin_mo_data(mf)
    mol = mf.mol

    # build the one-electron integrals (kinetic + nuclear attraction)
    H = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    Ha = np.einsum('up,vq,uv->pq', Ca, Ca, H)
    Hb = np.einsum('up,vq,uv->pq', Cb, Cb, H)

    return Ha[nfzc:, nfzc:], Hb[nfzc:, nfzc:]


def get_integrals_with_spin(mf, antisymmetrize_eris = True, nfzc = 0):
    """

    get one- and two-electron integrals from pyscf, with spin

    :param mf: a converged pyscf mean-field object (RHF, ROHF, or UHF)
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

    # number of occupied/virtual orbitals
    Ca, Cb, noa, nob = _spin_mo_data(mf)
    mol = mf.mol

    # total number of orbitals
    nmo = Ca.shape[1]

    # number of virtual orbitals
    nva = nmo - noa
    nvb = nmo - nob

    # build the one-electron integrals
    H = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
    Ha = np.einsum('up,vq,uv->pq', Ca, Ca, H)
    Hb = np.einsum('up,vq,uv->pq', Cb, Cb, H)

    # build the two-electron integrals, chemist's notation (pq|rs), one full ao2mo
    # transform per spin block -- mirrors the psi4-backed module's mints.mo_eri(...) calls
    g_aaaa = ao2mo.general(mol, (Ca, Ca, Ca, Ca), compact=False).reshape(nmo, nmo, nmo, nmo)
    g_bbbb = ao2mo.general(mol, (Cb, Cb, Cb, Cb), compact=False).reshape(nmo, nmo, nmo, nmo)
    g_abab = ao2mo.general(mol, (Ca, Ca, Cb, Cb), compact=False).reshape(nmo, nmo, nmo, nmo)

    # antisymmetrize g(ijkl) = <ij|kl> - <ij|lk> = (ik|jl) - (il|jk)
    g_aaaa = np.einsum('ikjl->ijkl', g_aaaa)
    g_bbbb = np.einsum('ikjl->ijkl', g_bbbb)
    g_abab = np.einsum('ikjl->ijkl', g_abab)
    if antisymmetrize_eris:
        g_aaaa = g_aaaa - np.einsum('ijlk->ijkl', g_aaaa)
        g_bbbb = g_bbbb - np.einsum('ijlk->ijkl', g_bbbb)

    # occupied slices
    oa = slice(None, noa)
    ob = slice(None, nob)

    # build spin-orbital fock matrix
    fa = Ha + np.einsum('piqi->pq', g_aaaa[:, oa, :, oa]) + np.einsum('piqi->pq', g_abab[:, ob, :, ob])
    fb = Hb + np.einsum('piqi->pq', g_bbbb[:, ob, :, ob]) + np.einsum('ipiq->pq', g_abab[oa, : , oa, :])
    if not antisymmetrize_eris:
        fa = fa - np.einsum('piiq->pq', g_aaaa[:, oa, oa, :])
        fb = fb - np.einsum('piiq->pq', g_bbbb[:, ob, ob, :])

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


def get_integrals(mf, antisymmetrize_eris = True, nfzc = 0):
    """

    get one- and two-electron integrals from pyscf

    :param mf: a converged pyscf mean-field object (RHF, ROHF, or UHF)
    :param antisymmetrize_eris: do antisymmetrize the eris?
    :param nfzc: number of frozen core orbitals

    :return nsocc: number of occupied spin-orbitals
    :return nsvirt: number of virtual spin-orbitals
    :return fock: the fock matrix (spin-orbital basis)
    :return gtei: (antisymmetrized) two-electron integrals (spin-orbital basis)
    :return efzc: frozen core energy

    """

    noa, nob, nva, nvb, fa, fb, g_aaaa, g_bbbb, g_abab, efzc = get_integrals_with_spin(mf, antisymmetrize_eris = antisymmetrize_eris, nfzc = nfzc)

    nsocc  = noa + nob
    nsvirt = nva + nvb

    fock = spatial_to_spin_orbital_oei(fa, fb, (nsocc+nsvirt)//2, noa, nob)
    gtei = spatial_to_spin_orbital_tei(g_aaaa, g_abab, g_bbbb, (nsocc+nsvirt)//2, noa, nob, antisymmetrize_eris = antisymmetrize_eris)

    return nsocc, nsvirt, fock, gtei, efzc

def electric_dipole(mf, opdm_a, opdm_b, nfzc = 0, print_level = 1):
    """
    :param mf: a converged pyscf mean-field object (RHF, ROHF, or UHF)
    :param opdm_a: one-particle density matrix (alpha)
    :param opdm_b: one-particle density matrix (beta)
    :param nfzc: the number of frozen core
    :param print_level: 0 for no printing, anything greater than 0 for printing
    """

    from pdaggerq.numerical.utils.pyscf_integrals import get_dipole_integrals_with_spin
    mu_aa, mu_bb = get_dipole_integrals_with_spin(mf)

    # Expand alpha-spin OPDM to include frozen core orbitals
    full_opdm_a = np.block([
        [np.eye(nfzc), np.zeros((nfzc, opdm_a.shape[0]))],
        [np.zeros((opdm_a.shape[0], nfzc)), opdm_a]
    ])

    # Expand beta-spin OPDM to include frozen core orbitals
    full_opdm_b = np.block([
        [np.eye(nfzc), np.zeros((nfzc, opdm_b.shape[0]))],
        [np.zeros((opdm_b.shape[0], nfzc)), opdm_b]
    ])

    # Evaluate electric dipole
    dipole_x = np.einsum('pq,qp->', mu_aa[0], full_opdm_a)
    dipole_x += np.einsum('pq,qp->', mu_bb[0], full_opdm_b)

    dipole_y = np.einsum('pq,qp->', mu_aa[1], full_opdm_a)
    dipole_y += np.einsum('pq,qp->', mu_bb[1], full_opdm_b)

    dipole_z = np.einsum('pq,qp->', mu_aa[2], full_opdm_a)
    dipole_z += np.einsum('pq,qp->', mu_bb[2], full_opdm_b)

    if print_level > 0:
        print('')
        print('    Dipole x: %20.12f' % (dipole_x))
        print('    Dipole y: %20.12f' % (dipole_y))
        print('    Dipole z: %20.12f' % (dipole_z))
        print('')

    return [dipole_x, dipole_y, dipole_z]

def electric_quadrupole(mf, opdm_a, opdm_b, nfzc = 0, print_level = 1):
    """
    :param mf: a converged pyscf mean-field object (RHF, ROHF, or UHF)
    :param opdm_a: one-particle density matrix (alpha)
    :param opdm_b: one-particle density matrix (beta)
    :param nfzc: the number of frozen core
    :param print_level: 0 for no printing, anything greater than 0 for printing
    """

    from pdaggerq.numerical.utils.pyscf_integrals import get_quadrupole_integrals_with_spin
    q_aa, q_bb = get_quadrupole_integrals_with_spin(mf)

    # Expand alpha-spin OPDM to include frozen core orbitals
    full_opdm_a = np.block([
        [np.eye(nfzc), np.zeros((nfzc, opdm_a.shape[0]))],
        [np.zeros((opdm_a.shape[0], nfzc)), opdm_a]
    ])

    # Expand beta-spin OPDM to include frozen core orbitals
    full_opdm_b = np.block([
        [np.eye(nfzc), np.zeros((nfzc, opdm_b.shape[0]))],
        [np.zeros((opdm_b.shape[0], nfzc)), opdm_b]
    ])

    # Evaluate electric quadrupole
    q_xx = np.einsum('pq,qp->', q_aa[0], full_opdm_a)
    q_xx += np.einsum('pq,qp->', q_bb[0], full_opdm_b)

    q_xy = np.einsum('pq,qp->', q_aa[1], full_opdm_a)
    q_xy += np.einsum('pq,qp->', q_bb[1], full_opdm_b)

    q_xz = np.einsum('pq,qp->', q_aa[2], full_opdm_a)
    q_xz += np.einsum('pq,qp->', q_bb[2], full_opdm_b)

    q_yy = np.einsum('pq,qp->', q_aa[3], full_opdm_a)
    q_yy += np.einsum('pq,qp->', q_bb[3], full_opdm_b)

    q_yz = np.einsum('pq,qp->', q_aa[4], full_opdm_a)
    q_yz += np.einsum('pq,qp->', q_bb[4], full_opdm_b)

    q_zz = np.einsum('pq,qp->', q_aa[5], full_opdm_a)
    q_zz += np.einsum('pq,qp->', q_bb[5], full_opdm_b)

    if print_level > 0:
        print('')
        print('    Quadrupole xx: %20.12f' % (q_xx))
        print('    Quadrupole xy: %20.12f' % (q_xy))
        print('    Quadrupole xz: %20.12f' % (q_xz))
        print('    Quadrupole yy: %20.12f' % (q_yy))
        print('    Quadrupole yz: %20.12f' % (q_yz))
        print('    Quadrupole zz: %20.12f' % (q_zz))
        print('')

    return [q_xx, q_xy, q_xz, q_yy, q_yz, q_zz]

def one_electron_energy(mf, opdm_a, opdm_b, nfzc = 0, print_level = 1):
    """
    calculate one-electron part of the energy from the opdm

    :param mf: a converged pyscf mean-field object (RHF, ROHF, or UHF)
    :param opdm_a: one-particle density matrix (alpha)
    :param opdm_b: one-particle density matrix (beta)
    :param nfzc: the number of frozen core
    :param print_level: 0 for no printing, anything greater than 0 for printing
    """

    from pdaggerq.numerical.utils.pyscf_integrals import get_core_hamiltonian_with_spin
    Ha, Hb = get_core_hamiltonian_with_spin(mf, nfzc = nfzc)

    # Evaluate one-electron energy
    one_electron_energy = np.einsum('pq,pq->', Ha, opdm_a)
    one_electron_energy += np.einsum('pq,pq->', Hb, opdm_b)

    if print_level > 0:
        print('')
        print('    One-electron energy: %20.12f' % (one_electron_energy))
        print('')

    return one_electron_energy

def two_electron_energy(mf, tpdm_aaaa, tpdm_abab, tpdm_bbbb, opdm_a, opdm_b, nfzc = 0, print_level = 1):
    """
    calculate two-electron part of the energy from the tpdm (plus frozen core part from opdm)

    :param mf: a converged pyscf mean-field object (RHF, ROHF, or UHF)
    :param tpdm_aaaa: two-particle density matrix (aaaa)
    :param tpdm_abab: two-particle density matrix (abab)
    :param tpdm_bbbb: two-particle density matrix (bbbb)
    :param opdm_a: alpha-spin one-particle density matrix
    :param opdm_b: beta-spin one-particle density matrix
    :param nfzc: the number of frozen core
    :param print_level: 0 for no printing, anything greater than 0 for printing
    """

    from pdaggerq.numerical.utils.pyscf_integrals import get_integrals_with_spin
    noa, nob, nva, nvb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, efzc  = get_integrals_with_spin(mf)

    # Evaluate two-electron energy
    two_electron_energy = 0.25 * np.einsum('pqrs,pqrs->', g_aaaa[nfzc:, nfzc:, nfzc:, nfzc:], tpdm_aaaa)
    two_electron_energy += np.einsum('pqrs,pqrs->', g_abab[nfzc:, nfzc:, nfzc:, nfzc:], tpdm_abab)
    two_electron_energy += 0.25 * np.einsum('pqrs,pqrs->', g_bbbb[nfzc:, nfzc:, nfzc:, nfzc:], tpdm_bbbb)

    if nfzc > 0:
        two_electron_energy += np.einsum('piqi,pq->', g_aaaa[nfzc:, :nfzc, nfzc:, :nfzc], opdm_a)
        two_electron_energy += np.einsum('ipiq,pq->', g_abab[:nfzc, nfzc:, :nfzc, nfzc:], opdm_b)
        two_electron_energy += np.einsum('piqi,pq->', g_bbbb[nfzc:, :nfzc, nfzc:, :nfzc], opdm_b)
        two_electron_energy += np.einsum('piqi,pq->', g_abab[nfzc:, :nfzc, nfzc:, :nfzc], opdm_a)

    if print_level > 0:
        print('')
        print('    Two-electron energy: %20.12f' % (two_electron_energy))
        print('')

    return two_electron_energy
