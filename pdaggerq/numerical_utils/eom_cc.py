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

class eom_ccsd:

    def __init__(self, 
                 ccsd,
                 right_sigma0_func=None,
                 right_sigma1_func=None,
                 right_sigma2_func=None,
                 left_sigma0_func=None,
                 left_sigma1_func=None,
                 left_sigma2_func=None):
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
        self.right_sigma0_func = right_sigma0_func
        self.right_sigma1_func = right_sigma1_func
        self.right_sigma2_func = right_sigma2_func
        self.left_sigma0_func = left_sigma0_func
        self.left_sigma1_func = left_sigma1_func
        self.left_sigma2_func = left_sigma2_func

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

    def right_solver(self, nstates = 5):

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

        ex, rvec = scipy.sparse.linalg.eigs(HbarR, k=nstates, which='SR')
        idx = np.argsort(ex)
        ex = ex[idx]
        rvec = rvec[:, idx]

        print('    %5s %20s %20s' % ('state', 'total energy','excitation energy'))
        for i in range (0, nstates):
            print('    %5i %20.12f %20.12f' % ( i, ex[i].real + self.ccsd.energy + self.ccsd.efzc + self.ccsd.nuclear_repulsion_energy, ex[i].real))

        print('')
       
        # save energies and right-hand amplitudes
        self.eom_cc_energy = []
        for i in range (0, nstates):
            self.eom_cc_energy.append(ex[i])

        self.r0 = []
        self.r1_aa = []
        self.r1_bb = []
        self.r2_aaaa = []
        self.r2_abab = []
        self.r2_bbbb = []
        for i in range (0, nstates):
            r0, r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb = Hbar.unpack_right_amplitudes(rvec[:, i])
            self.r0.append(r0)
            self.r1_aa.append(r1_aa)
            self.r1_bb.append(r1_bb)
            self.r2_aaaa.append(r2_aaaa)
            self.r2_abab.append(r2_abab)
            self.r2_bbbb.append(r2_bbbb)

    def left_solver(self, nstates = 5):

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

        ex, lvec = scipy.sparse.linalg.eigs(LHbar, k=nstates, which='SR')
        idx = np.argsort(ex)
        ex = ex[idx]
        lvec = lvec[:, idx]

        print('    %5s %20s %20s' % ('state', 'total energy','excitation energy'))
        for i in range (0, nstates):
            print('    %5i %20.12f %20.12f' % ( i, ex[i].real + self.ccsd.energy + self.ccsd.efzc + self.ccsd.nuclear_repulsion_energy, ex[i].real))

        print('')
       
        # save energies and left-hand amplitudes
        self.eom_cc_energy = []
        for i in range (0, nstates):
            self.eom_cc_energy.append(ex[i])

        self.l0 = []
        self.l1_aa = []
        self.l1_bb = []
        self.l2_aaaa = []
        self.l2_abab = []
        self.l2_bbbb = []
        for i in range (0, nstates):
            l0, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb = Hbar.unpack_left_amplitudes(lvec[:, i])
            self.l0.append(l0)
            self.l1_aa.append(l1_aa)
            self.l1_bb.append(l1_bb)
            self.l2_aaaa.append(l2_aaaa)
            self.l2_abab.append(l2_abab)
            self.l2_bbbb.append(l2_bbbb)

