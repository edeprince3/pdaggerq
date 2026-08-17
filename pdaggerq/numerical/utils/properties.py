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
evaluate one- and two-electron properties using input opdm / tpdm and integrals from psi4
"""

import numpy as np
from numpy import einsum

def electric_dipole(wfn, opdm_a, opdm_b, nfzc = 0, print_level = 1):
    """
    :param wfn: psi4 wave function object
    :param opdm_a: one-particle density matrix (alpha)
    :param opdm_b: one-particle density matrix (beta)
    :param nfzc: the number of frozen core
    :param print_level: 0 for no printing, anything greater than 0 for printing
    """

    from pdaggerq.numerical.utils.integrals import get_dipole_integrals_with_spin
    mu_aa, mu_bb = get_dipole_integrals_with_spin(wfn)

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

def electric_quadrupole(wfn, opdm_a, opdm_b, nfzc = 0, print_level = 1):
    """
    :param wfn: psi4 wave function object
    :param opdm_a: one-particle density matrix (alpha)
    :param opdm_b: one-particle density matrix (beta)
    :param nfzc: the number of frozen core
    :param print_level: 0 for no printing, anything greater than 0 for printing
    """

    from pdaggerq.numerical.utils.integrals import get_quadrupole_integrals_with_spin
    q_aa, q_bb = get_quadrupole_integrals_with_spin(wfn)

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

def one_electron_energy(wfn, opdm_a, opdm_b, nfzc = 0, print_level = 1):
    """
    calculate one-electron part of the energy from the opdm

    :param wfn: psi4 wave function object 
    :param opdm_a: one-particle density matrix (alpha)
    :param opdm_b: one-particle density matrix (beta)
    :param nfzc: the number of frozen core
    :param print_level: 0 for no printing, anything greater than 0 for printing
    """

    from pdaggerq.numerical.utils.integrals import get_core_hamiltonian_with_spin
    Ha, Hb = get_core_hamiltonian_with_spin(wfn, nfzc = nfzc)

    # Evaluate one-electron energy
    one_electron_energy = np.einsum('pq,pq->', Ha, opdm_a)
    one_electron_energy += np.einsum('pq,pq->', Hb, opdm_b)

    if print_level > 0:
        print('')
        print('    One-electron energy: %20.12f' % (one_electron_energy))
        print('')

    return one_electron_energy

def two_electron_energy(wfn, tpdm_aaaa, tpdm_abab, tpdm_bbbb, opdm_a, opdm_b, nfzc = 0, print_level = 1):
    """
    calculate two-electron part of the energy from the tpdm (plus frozen core part from opdm)

    :param wfn: psi4 wave function object 
    :param tpdm_aaaa: two-particle density matrix (aaaa)
    :param tpdm_abab: two-particle density matrix (abab)
    :param tpdm_bbbb: two-particle density matrix (bbbb)
    :param opdm_a: alpha-spin one-particle density matrix
    :param opdm_b: beta-spin one-particle density matrix
    :param nfzc: the number of frozen core
    :param print_level: 0 for no printing, anything greater than 0 for printing
    """

    from pdaggerq.numerical.utils.integrals import get_integrals_with_spin
    noa, nob, nva, nvb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, efzc  = get_integrals_with_spin(wfn)

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
