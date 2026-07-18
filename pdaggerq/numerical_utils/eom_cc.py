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

from pdaggerq.numerical_utils.cc_hbar import HbarOperator

import types

class eom_ccsd:

    def __init__(self, 
                 ccsd,
                 nstates = 5,
                 right_sigma0_func=None,
                 right_sigma1_func=None,
                 right_sigma2_func=None,
                 left_sigma0_func=None,
                 left_sigma1_func=None,
                 left_sigma2_func=None,
                 density_matrix_func=None):
        """
        initialize EOM-CCSD class

        :params ccsd: the ccsd class
        :params right_sigma0_func: python function for the 0 part of the right-hand sigma vector
        :params right_sigma1_func: python function for the 1 part of the right-hand sigma vector
        :params right_sigma2_func: python function for the 2 part of the right-hand sigma vector
        :params left_sigma0_func: python function for the 0 part of the left-hand sigma vector
        :params left_sigma1_func: python function for the 1 part of the left-hand sigma vector
        :params left_sigma2_func: python function for the 2 part of the left-hand sigma vector
        """

        self.ccsd = ccsd
        self.nstates = nstates
        self.right_sigma0_func = right_sigma0_func
        self.right_sigma1_func = right_sigma1_func
        self.right_sigma2_func = right_sigma2_func
        self.left_sigma0_func = left_sigma0_func
        self.left_sigma1_func = left_sigma1_func
        self.left_sigma2_func = left_sigma2_func

        if density_matrix_func is not None:
            self.density_matrix = types.MethodType(density_matrix_func, self)

        if ccsd.use_spin_orbital_basis:
            raise Exception("spin-orbital eomcc is not implemented")
        else:
            self.r0 = None
            self.r1_aa = None
            self.r1_bb = None
            self.r2_aaaa = None
            self.r2_abab = None
            self.r2_bbbb = None
            self.l0 = None
            self.l1_aa = None
            self.l1_bb = None
            self.l2_aaaa = None
            self.l2_abab = None
            self.l2_bbbb = None

    def right_solver(self):

        print('    ==> right-hand EOM-CCSD <==')
        print('')

        # build Hbar operator object

        # unique oo/vv pairs
        i_idx_a, j_idx_a = np.triu_indices(self.ccsd.noa, k=1)
        i_idx_b, j_idx_b = np.triu_indices(self.ccsd.nob, k=1)
        a_idx_a, b_idx_a = np.triu_indices(self.ccsd.nva, k=1)
        a_idx_b, b_idx_b = np.triu_indices(self.ccsd.nvb, k=1)

        dim = 1
        dim += self.ccsd.noa * self.ccsd.nva
        dim += self.ccsd.nob * self.ccsd.nvb
        dim += len(i_idx_a) * len(a_idx_a)
        dim += len(i_idx_b) * len(a_idx_b)
        dim += self.ccsd.noa * self.ccsd.nva * self.ccsd.nob * self.ccsd.nvb

        Hbar = HbarOperator(self.ccsd, 
                            right_sigma0_func = self.right_sigma0_func,
                            right_sigma1_func = self.right_sigma1_func,
                            right_sigma2_func = self.right_sigma2_func)

        # diagonalize Hbar
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

        self.r0 = []
        self.r1_aa = []
        self.r1_bb = []
        self.r2_aaaa = []
        self.r2_abab = []
        self.r2_bbbb = []
        for i in range (self.nstates):
            r0, r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb = Hbar.unpack_right_amplitudes(rvec[:, i])
            self.r0.append(r0)
            self.r1_aa.append(r1_aa)
            self.r1_bb.append(r1_bb)
            self.r2_aaaa.append(r2_aaaa)
            self.r2_abab.append(r2_abab)
            self.r2_bbbb.append(r2_bbbb)

    def left_solver(self):

        print('    ==> left-hand EOM-CCSD <==')
        print('')

        # build Hbar operator object

        # unique oo/vv pairs
        i_idx_a, j_idx_a = np.triu_indices(self.ccsd.noa, k=1)
        i_idx_b, j_idx_b = np.triu_indices(self.ccsd.nob, k=1)
        a_idx_a, b_idx_a = np.triu_indices(self.ccsd.nva, k=1)
        a_idx_b, b_idx_b = np.triu_indices(self.ccsd.nvb, k=1)

        dim = 1
        dim += self.ccsd.noa * self.ccsd.nva
        dim += self.ccsd.nob * self.ccsd.nvb
        dim += len(i_idx_a) * len(a_idx_a)
        dim += len(i_idx_b) * len(a_idx_b)
        dim += self.ccsd.noa * self.ccsd.nva * self.ccsd.nob * self.ccsd.nvb

        Hbar = HbarOperator(self.ccsd,
                            left_sigma0_func = self.left_sigma0_func,
                            left_sigma1_func = self.left_sigma1_func,
                            left_sigma2_func = self.left_sigma2_func)

        # diagonalize Hbar
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

        self.l0 = []
        self.l1_aa = []
        self.l1_bb = []
        self.l2_aaaa = []
        self.l2_abab = []
        self.l2_bbbb = []
        for i in range (self.nstates):
            l0, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb = Hbar.unpack_left_amplitudes(lvec[:, i])
            self.l0.append(l0)
            self.l1_aa.append(l1_aa)
            self.l1_bb.append(l1_bb)
            self.l2_aaaa.append(l2_aaaa)
            self.l2_abab.append(l2_abab)
            self.l2_bbbb.append(l2_bbbb)

    def oscillator_strengths(self):

        # 
        # Biorthogonalize L and R
        # 

        Hbar = HbarOperator(self.ccsd)

        L = Hbar.pack_left_amplitudes(self.l0[0],
            self.l1_aa[0],
            self.l1_bb[0],
            self.l2_aaaa[0],
            self.l2_abab[0],
            self.l2_bbbb[0]
        )
        dim = len(L)

        R_mat = np.zeros((dim, self.nstates), dtype = np.complex128)
        L_mat = np.zeros((dim, self.nstates), dtype = np.complex128)
        M = np.zeros((self.nstates, self.nstates))

        for i in range (self.nstates):
            L_mat[:, i] = Hbar.pack_right_amplitudes(self.l0[i],
                self.l1_aa[i].transpose(1,0),
                self.l1_bb[i].transpose(1,0),
                self.l2_aaaa[i].transpose(2,3,0,1),
                self.l2_abab[i].transpose(2,3,0,1),
                self.l2_bbbb[i].transpose(2,3,0,1)
            )
            R_mat[:, i] = Hbar.pack_right_amplitudes(self.r0[i],
                self.r1_aa[i],
                self.r1_bb[i],
                self.r2_aaaa[i],
                self.r2_abab[i],
                self.r2_bbbb[i]
            )

        L_mat, R_mat = self.LU_biorthonormalization(L_mat, R_mat)

        # Unpack biorthogonalized L and R
        for i in range (self.nstates):
            self.r0[i], self.r1_aa[i], self.r1_bb[i], self.r2_aaaa[i], self.r2_abab[i], self.r2_bbbb[i] = Hbar.unpack_right_amplitudes(R_mat[:, i])
            self.l0[i], self.l1_aa[i], self.l1_bb[i], self.l2_aaaa[i], self.l2_abab[i], self.l2_bbbb[i] = Hbar.unpack_right_amplitudes(L_mat[:, i])
            self.l1_aa[i] = self.l1_aa[i].transpose(1,0)
            self.l1_bb[i] = self.l1_bb[i].transpose(1,0)
            self.l2_aaaa[i] = self.l2_aaaa[i].transpose(2,3,0,1)
            self.l2_abab[i] = self.l2_abab[i].transpose(2,3,0,1)
            self.l2_bbbb[i] = self.l2_bbbb[i].transpose(2,3,0,1)
            

        #
        # Compute oscillator strengths
        #

        print('    ==> EOM-CCSD oscillator strengths <==')
        print('')
        print('    %7s %7s %10s %10s %10s %10s %10s %10s %10s %10s' % ('', '', '<L1|mu|R2>', '', '', '<L2|mu|R1>', '', '', '', ''))
        print('    %7s %7s %10s %10s %10s %10s %10s %10s %10s %10s' % ('state 1', 'state 2', 'x', 'y', 'z', 'x', 'y', 'z', 'osc', 'w'))

        from pdaggerq.numerical_utils.integrals import get_dipole_integrals_with_spin
        dipole_aa, dipole_bb = get_dipole_integrals_with_spin(self.ccsd.wfn, nfzc = self.ccsd.nfzc)

        f = np.zeros((self.nstates, self.nstates))

        for i in range (self.nstates):
            for j in range (i, self.nstates):

                tdm = self.density_matrix(i, j)

                tdp_ij = np.zeros((3))

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
                    
                tdp_ji = np.zeros((3))

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
                print('    %7i %7i %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f %10.6f' 
                    % (i, j, tdp_ij[0], tdp_ij[1], tdp_ij[2],
                    tdp_ji[0], tdp_ji[1], tdp_ji[2], f[i, j], w))
        print('')

        return f

    def LU_biorthonormalization(self, L, R):
    
        for i in range(len(L[0])):
            L[:,i] /= np.dot(L[:,i],R[:,i])
    
        M = np.matmul(L.T, R)
    
        # x, ML, MU = scipy.linalg.lu(M, permute_l=False)
        ML, MU = scipy.linalg.lu(M, permute_l=True)
    
        L = np.matmul(np.linalg.inv(ML),L.T).T
        R = np.matmul(R,np.linalg.inv(MU))
    
        # for numerical reason, renormalize L and R vectors
        for i in range(len(R[0])):
            R[:,i] /= np.linalg.norm(R[:,i]) # make sure <R|R>=1 for numerical stability
            L[:,i] /= np.dot(L[:,i],R[:,i]) # re-binormalize <L|R>=1 for numerical stability
    
        chkmat = np.matmul(L.T,R)
    
        #print('',flush=True)
        #print('    Checking <L_mu|R_nu> ...',flush=True)
        ## print(np.abs(chkmat),flush=True)
    
        #if np.any(np.abs(chkmat - np.eye(len(L[0])))>1.0e-12):
        #   adx=np.argwhere(np.abs(chkmat - np.eye(len(L[0])))>1.0e-12)
        #   print('    Looks like L and R could be messed up. Check the following elements:',flush=True)
        #   for a in adx:
        #       print('    < {: 3d} | {: 3d} > = {: .6e}'.format(a[0],a[1],chkmat[a[0],a[1]]),flush=True)
        #else:
        #   print('    Looks like L and R are properly biorthonormalized',flush=True)
        #print('',flush=True)
    
        return L, R

