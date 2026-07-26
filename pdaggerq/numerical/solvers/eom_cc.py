# hilbert2 - python-based quantum chemistry
# Copyright (C) 2025 A. Eugene DePrince III
#
# This file is part of the hilbert2 package.
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
eom-ccsd sigma vectors and the full eom-ccsd hamiltonian
"""

import numpy as np
from numpy import einsum

import scipy
from scipy.sparse.linalg import LinearOperator

from pdaggerq.numerical.solvers.cc_hbar import HbarOperator

import types

import copy

class eom_ccsd:

    def __init__(self, 
                 ccsd,
                 nstates = 5,
                 R_list=[],
                 L_list=[],
                 density_matrix_func=None):
        """
        initialize EOMCC class

        :params ccsd: the ccsd class
        :params R_list: list of R operator dictionaries
        :params L_list: list of L operator dictionaries
        """

        self.ccsd = ccsd
        self.nstates = nstates
        self.R_list = R_list
        self.L_list = L_list

        if density_matrix_func is not None:
            self.density_matrix = types.MethodType(density_matrix_func, self)

    def right_solver(self):

        print('    ==> right-hand EOMCC <==')
        print('')

        # build Hbar operator object
        Hbar = HbarOperator(self.ccsd, R_list = self.R_list)

        # diagonalize Hbar
        dim = Hbar.right_amplitude_size
        HbarR = LinearOperator((dim, dim), matvec=Hbar.matvec_right, dtype=np.float64)

        ex, rvec = scipy.sparse.linalg.eigs(HbarR, k=self.nstates, which='SR')
        idx = np.argsort(ex)
        ex = ex[idx]
        rvec = rvec[:, idx]

        print('    %5s %20s %20s' % ('state', 'total energy','excitation energy'))
        for i in range (self.nstates):
            print('    %5i %20.12f %20.12f' % ( i, ex[i].real + self.ccsd.energy + self.ccsd.efzc + self.ccsd.nuclear_repulsion_energy, ex[i].real))

        print('')
       
        # save energies and right-hand amplitudes
        self.eom_cc_energy = []
        for i in range (self.nstates):
            self.eom_cc_energy.append(ex[i])

        self.R = []
        for i in range (self.nstates):
            Hbar.unpack_eom_vectors(rvec[:, i], Hbar.R, Hbar.R_meta)
            self.R.append(copy.deepcopy(Hbar.R))

    def left_solver(self):

        print('    ==> left-hand EOMCC <==')
        print('')

        # build Hbar operator object
        Hbar = HbarOperator(self.ccsd, L_list = self.L_list)

        # diagonalize Hbar
        dim = Hbar.left_amplitude_size
        LHbar = LinearOperator((dim, dim), matvec=Hbar.matvec_left, dtype=np.float64)

        ex, lvec = scipy.sparse.linalg.eigs(LHbar, k=self.nstates, which='SR')
        idx = np.argsort(ex)
        ex = ex[idx]
        lvec = lvec[:, idx]

        print('    %5s %20s %20s' % ('state', 'total energy','excitation energy'))
        for i in range (self.nstates):
            print('    %5i %20.12f %20.12f' % ( i, ex[i].real + self.ccsd.energy + self.ccsd.efzc + self.ccsd.nuclear_repulsion_energy, ex[i].real))

        print('')
       
        # save energies and left-hand amplitudes
        self.eom_cc_energy = []
        for i in range (self.nstates):
            self.eom_cc_energy.append(ex[i])

        self.L = []
        for i in range (self.nstates):
            Hbar.unpack_eom_vectors(lvec[:, i], Hbar.L, Hbar.L_meta)
            self.L.append(copy.deepcopy(Hbar.L))

    def oscillator_strengths(self):

        # Pack eigenvectors for biorthogonalization
        Hbar = HbarOperator(self.ccsd, L_list = self.L_list, R_list = self.R_list)

        dim = Hbar.left_amplitude_size
        R_mat = np.zeros((dim, self.nstates), dtype = np.complex128)
        L_mat = np.zeros((dim, self.nstates), dtype = np.complex128)
        M = np.zeros((self.nstates, self.nstates))

        for i in range (self.nstates):
            L_mat[:, i] = Hbar.pack_eom_vectors(self.L[i], Hbar.L_meta)
            R_mat[:, i] = Hbar.pack_eom_vectors(self.R[i], Hbar.R_meta)

        # Biorthogonalize
        L_mat, R_mat = self.LU_biorthonormalization(L_mat, R_mat)

        # Unpack biorthogonalized L and R
        for i in range (self.nstates):
            Hbar.unpack_eom_vectors(R_mat[:, i], self.R[i], Hbar.R_meta)
            Hbar.unpack_eom_vectors(L_mat[:, i], self.L[i], Hbar.L_meta)

        # Compute oscillator strengths

        print('    ==> EOMCC oscillator strengths <==')
        print('')
        print('    %7s %7s %10s %10s %10s %10s %10s %10s %10s' % ('', '', '<L1|mu|R2>', '', '', '<L2|mu|R1>', '', '', ''))
        print('    %7s %7s %10s %10s %10s %10s %10s %10s %10s' % ('state 1', 'state 2', 'x', 'y', 'z', 'x', 'y', 'z', 'osc'))

        from pdaggerq.numerical.utils.integrals import get_dipole_integrals_with_spin
        dipole_aa, dipole_bb = get_dipole_integrals_with_spin(self.ccsd.wfn, nfzc = self.ccsd.nfzc)

        f = np.zeros((self.nstates, self.nstates), dtype=np.complex128)

        for i in range (self.nstates):
            for j in range (i+1, self.nstates):

                tdm = self.density_matrix(i, j)

                tdp_ij = np.zeros((3), dtype=np.complex128)

                for xyz in range (3):
                    tdp_ij[xyz] += np.einsum('ij,ij->', tdm['aa_oo'], dipole_aa[xyz][self.ccsd.oa, self.ccsd.oa])
                    tdp_ij[xyz] += np.einsum('ia,ia->', tdm['aa_ov'], dipole_aa[xyz][self.ccsd.oa, self.ccsd.va])
                    tdp_ij[xyz] += np.einsum('ai,ai->', tdm['aa_vo'], dipole_aa[xyz][self.ccsd.va, self.ccsd.oa])
                    tdp_ij[xyz] += np.einsum('ab,ab->', tdm['aa_vv'], dipole_aa[xyz][self.ccsd.va, self.ccsd.va])
                    tdp_ij[xyz] += np.einsum('ij,ij->', tdm['bb_oo'], dipole_bb[xyz][self.ccsd.ob, self.ccsd.ob])
                    tdp_ij[xyz] += np.einsum('ia,ia->', tdm['bb_ov'], dipole_bb[xyz][self.ccsd.ob, self.ccsd.vb])
                    tdp_ij[xyz] += np.einsum('ai,ai->', tdm['bb_vo'], dipole_bb[xyz][self.ccsd.vb, self.ccsd.ob])
                    tdp_ij[xyz] += np.einsum('ab,ab->', tdm['bb_vv'], dipole_bb[xyz][self.ccsd.vb, self.ccsd.vb])

                tdm = self.density_matrix(j, i)
                    
                tdp_ji = np.zeros((3), dtype=np.complex128)

                for xyz in range (3):
                    tdp_ji[xyz] += np.einsum('ij,ij->', tdm['aa_oo'], dipole_aa[xyz][self.ccsd.oa, self.ccsd.oa])
                    tdp_ji[xyz] += np.einsum('ia,ia->', tdm['aa_ov'], dipole_aa[xyz][self.ccsd.oa, self.ccsd.va])
                    tdp_ji[xyz] += np.einsum('ai,ai->', tdm['aa_vo'], dipole_aa[xyz][self.ccsd.va, self.ccsd.oa])
                    tdp_ji[xyz] += np.einsum('ab,ab->', tdm['aa_vv'], dipole_aa[xyz][self.ccsd.va, self.ccsd.va])
                    tdp_ji[xyz] += np.einsum('ij,ij->', tdm['bb_oo'], dipole_bb[xyz][self.ccsd.ob, self.ccsd.ob])
                    tdp_ji[xyz] += np.einsum('ia,ia->', tdm['bb_ov'], dipole_bb[xyz][self.ccsd.ob, self.ccsd.vb])
                    tdp_ji[xyz] += np.einsum('ai,ai->', tdm['bb_vo'], dipole_bb[xyz][self.ccsd.vb, self.ccsd.ob])
                    tdp_ji[xyz] += np.einsum('ab,ab->', tdm['bb_vv'], dipole_bb[xyz][self.ccsd.vb, self.ccsd.vb])

                en_i = self.eom_cc_energy[i]
                en_j = self.eom_cc_energy[j]
                w = en_j - en_i

                f[i, j] = 2./3. * w * np.einsum('i,i->', tdp_ij, tdp_ji)
                print('    %7i %7i %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f' 
                    % (i, j, tdp_ij[0].real, tdp_ij[1].real, tdp_ij[2].real,
                    tdp_ji[0].real, tdp_ji[1].real, tdp_ji[2].real, f[i, j].real))
        print('')

        return f

    def LU_biorthonormalization(self, L, R):
    
        for i in range(len(L[0])):
            L[:,i] /= np.dot(L[:,i],R[:,i])
    
        M = np.matmul(L.T, R)
        ML, MU = scipy.linalg.lu(M, permute_l=True)
    
        L = np.matmul(np.linalg.inv(ML),L.T).T
        R = np.matmul(R,np.linalg.inv(MU))
    
        # normalize L and R vectors, <R|R> = <L|R> = 1
        for i in range(len(R[0])):
            R[:,i] /= np.linalg.norm(R[:,i])
            L[:,i] /= np.dot(L[:,i],R[:,i])
    
        return L, R

