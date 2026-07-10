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

import types

class HbarOperator:
    """
    Hbar as a LinearOperator, with spin-traced expressions
    """

    def __init__(self, 
                 ccsd,
                 include_reference = True,
                 right_sigma0_func=None, 
                 right_sigma1_func=None,
                 right_sigma2_func=None,
                 left_sigma0_func=None, 
                 left_sigma1_func=None,
                 left_sigma2_func=None):
        """
        initialize HBarOperator

        :param ccsd: ccsd object
        :param include_reference: do include reference in Hbar basis?
        :params right_sigma0_func: python function for the 0 part of the right-hand sigma vector
        :params right_sigma1_func: python function for the 1 part of the right-hand sigma vector
        :params right_sigma2_func: python function for the 2 part of the right-hand sigma vector
        :params left_sigma0_func: python function for the 0 part of the left-hand sigma vector
        :params left_sigma1_func: python function for the 1 part of the left-hand sigma vector
        :params left_sigma2_func: python function for the 2 part of the left-hand sigma vector
        """

        self.ccsd = ccsd
        self.include_reference = include_reference
        if right_sigma0_func is not None:
            self.right_sigma0 = types.MethodType(right_sigma0_func, self)
        if right_sigma1_func is not None:
            self.right_sigma1 = types.MethodType(right_sigma1_func, self)
        if right_sigma2_func is not None:
            self.right_sigma2 = types.MethodType(right_sigma2_func, self)
        if left_sigma0_func is not None:
            self.left_sigma0 = types.MethodType(left_sigma0_func, self)
        if left_sigma1_func is not None:
            self.left_sigma1 = types.MethodType(left_sigma1_func, self)
        if left_sigma2_func is not None:
            self.left_sigma2 = types.MethodType(left_sigma2_func, self)

        # unique oo/vv pairs
        self.i_idx_a, self.j_idx_a = np.triu_indices(ccsd.noa, k=1)
        self.i_idx_b, self.j_idx_b = np.triu_indices(ccsd.nob, k=1)
        self.a_idx_a, self.b_idx_a = np.triu_indices(ccsd.nva, k=1)
        self.a_idx_b, self.b_idx_b = np.triu_indices(ccsd.nvb, k=1)

    def unpack_right_amplitudes(self, R):

        start = 0

        r0 = R[0]
        start += 1

        r1_aa = R[start:start + self.ccsd.noa*self.ccsd.nva].reshape(self.ccsd.t1_aa.shape)
        start += self.ccsd.noa*self.ccsd.nva

        r1_bb = R[start:start + self.ccsd.nob*self.ccsd.nvb].reshape(self.ccsd.t1_bb.shape)
        start += self.ccsd.nob*self.ccsd.nvb

        # antisymmetrized eom-cc alpha-alpha doubles amplitudes
        r2_aaaa = self.unpack_antisym(R[start:start+len(self.i_idx_a)*len(self.a_idx_a)], self.a_idx_a, self.b_idx_a, self.i_idx_a, self.j_idx_a, self.ccsd.nva, self.ccsd.noa)
        start += len(self.i_idx_a)*len(self.a_idx_a)

        r2_abab = R[start:start+self.ccsd.noa*self.ccsd.nva*self.ccsd.nob*self.ccsd.nvb].reshape(self.ccsd.t2_abab.shape)
        start += self.ccsd.noa*self.ccsd.nva*self.ccsd.nob*self.ccsd.nvb

        # antisymmetrized eom-cc beta-beta doubles amplitudes
        r2_bbbb = self.unpack_antisym(R[start:], self.a_idx_b, self.b_idx_b, self.i_idx_b, self.j_idx_b, self.ccsd.nvb, self.ccsd.nob)


        return r0, r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb

    def unpack_left_amplitudes(self, L):

        start = 0

        l0 = L[0]
        start += 1

        l1_aa = L[start:start + self.ccsd.noa*self.ccsd.nva].reshape(self.ccsd.noa, self.ccsd.nva)
        start += self.ccsd.noa*self.ccsd.nva

        l1_bb = L[start:start + self.ccsd.nob*self.ccsd.nvb].reshape(self.ccsd.nob, self.ccsd.nvb)
        start += self.ccsd.nob*self.ccsd.nvb

        # antisymmetrized eom-cc alpha-alpha doubles amplitudes
        l2_aaaa = self.unpack_antisym(L[start:start+len(self.i_idx_a)*len(self.a_idx_a)], self.i_idx_a, self.j_idx_a, self.a_idx_a, self.b_idx_a, self.ccsd.noa, self.ccsd.nva)
        start += len(self.i_idx_a)*len(self.a_idx_a)

        l2_abab = L[start:start+self.ccsd.noa*self.ccsd.nva*self.ccsd.nob*self.ccsd.nvb].reshape(self.ccsd.noa, self.ccsd.nob, self.ccsd.nva, self.ccsd.nvb)
        start += self.ccsd.noa*self.ccsd.nva*self.ccsd.nob*self.ccsd.nvb

        # antisymmetrized eom-cc beta-beta doubles amplitudes
        l2_bbbb = self.unpack_antisym(L[start:], self.i_idx_b, self.j_idx_b, self.a_idx_b, self.b_idx_b, self.ccsd.nob, self.ccsd.nvb)

        return l0, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb

    def matvec_right(self, R):
        """
        evaluate the action of Hbar on a vector, sigma = H.R

        :param R: the vector (flat, unique elements only)

        :return sigma: the sigma vector (flat, unique elements only)
        """

        bigR = R
        if not self.include_reference:
            bigR = np.hstack((0.0, R))
        r0, r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb = self.unpack_right_amplitudes(bigR)

        sigma0 = self.right_sigma0(r0, r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb)
        sigma1_aa, sigma1_bb = self.right_sigma1(r0, r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb)
        sigma2_aaaa, sigma2_abab, sigma2_bbbb = self.right_sigma2(r0, r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb)

        # packed same-spin doubles parts of the eom-cc sigma vector
        sigma2_aaaa = self.pack_antisym(sigma2_aaaa, self.a_idx_a, self.b_idx_a, self.i_idx_a, self.j_idx_a)
        sigma2_bbbb = self.pack_antisym(sigma2_bbbb, self.a_idx_b, self.b_idx_b, self.i_idx_b, self.j_idx_b)

        if self.include_reference:
            sigma = np.hstack((sigma0, sigma1_aa.flatten(), sigma1_bb.flatten(), sigma2_aaaa.flatten(), sigma2_abab.flatten(), sigma2_bbbb.flatten()))
        else:
            sigma = np.hstack((sigma1_aa.flatten(), sigma1_bb.flatten(), sigma2_aaaa.flatten(), sigma2_abab.flatten(), sigma2_bbbb.flatten()))
        return sigma - self.ccsd.energy * R

    def matvec_left(self, L):
        """
        evaluate the action of Hbar on a vector, sigma = L.H

        :param L: the vector (flat, unique elements only)

        :return sigma: the sigma vector (flat, unique elements only)
        """

        bigL = L
        if not self.include_reference:
            bigL = np.hstack((0.0, L))
        l0, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb = self.unpack_left_amplitudes(bigL)

        sigma0 = self.left_sigma0(l0, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb)
        sigma1_aa, sigma1_bb = self.left_sigma1(l0, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb)
        sigma2_aaaa, sigma2_abab, sigma2_bbbb = self.left_sigma2(l0, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb)

        # packed same-spin doubles parts of the eom-cc sigma vector
        sigma2_aaaa = self.pack_antisym(sigma2_aaaa, self.i_idx_a, self.j_idx_a, self.a_idx_a, self.b_idx_a)
        sigma2_bbbb = self.pack_antisym(sigma2_bbbb, self.i_idx_b, self.j_idx_b, self.a_idx_b, self.b_idx_b)

        if self.include_reference:
            sigma = np.hstack((sigma0, sigma1_aa.flatten(), sigma1_bb.flatten(), sigma2_aaaa.flatten(), sigma2_abab.flatten(), sigma2_bbbb.flatten()))
        else:
            sigma = np.hstack((sigma1_aa.flatten(), sigma1_bb.flatten(), sigma2_aaaa.flatten(), sigma2_abab.flatten(), sigma2_bbbb.flatten()))
        return sigma - self.ccsd.energy * L

    def unpack_antisym(self, rvec, a_idx, b_idx, i_idx, j_idx, nv, no):
        """
        unpack a flat array (a<b, i<j) into full antisymmetrized r2[a,b,i,j]
    
        :param rvec: the flat array
        :param a_idx: first of a<b pair (list)
        :param b_idx: second of a<b pair (list)
        :param i_idx: first of i<j pair (list)
        :param j_idx: second of i<j pair (list)
        :param nv: number of virtual orbitals
        :param no: number of occupied orbitals
    
        :return r2: the full antisymmetrized array
        """
    
        r2 = np.zeros((nv, nv, no, no), dtype = rvec.dtype)
        tmp = rvec.reshape(len(a_idx), len(i_idx))
        for sign, (aa, bb, ii, jj) in [
            (+1, (a_idx, b_idx, i_idx, j_idx)),
            (-1, (b_idx, a_idx, i_idx, j_idx)),
            (-1, (a_idx, b_idx, j_idx, i_idx)),
            (+1, (b_idx, a_idx, j_idx, i_idx)),
        ]:
            r2[aa[:, None], bb[:, None], ii[None, :], jj[None, :]] += sign * tmp
        return r2
    
    def pack_antisym(self, t2, a_idx, b_idx, i_idx, j_idx):
        """
        pack antisymmetrized tensor t2[a,b,i,j] into a flat array with a<b, i<j.
    
        :param t2: the antisymmetrized array
        :param a_idx: first of a<b pair (list)
        :param b_idx: second of a<b pair (list)
        :param i_idx: first of i<j pair (list)
        :param j_idx: second of i<j pair (list)
    
        :return flat: the flat array
        """
    
        flat = t2[a_idx[:, None], b_idx[:, None], i_idx[None, :], j_idx[None, :]]
        return flat.reshape(-1)

