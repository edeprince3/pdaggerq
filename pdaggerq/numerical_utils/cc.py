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
CCSD and CCSDT
"""

import numpy as np
from numpy import einsum
import types

from pdaggerq.numerical_utils.integrals import get_integrals
from pdaggerq.numerical_utils.integrals import get_integrals_with_spin

from pdaggerq.numerical_utils.diis import DIIS

class cc:

    def __init__(self, wfn, 
        mol, 
        use_spin_orbital_basis = False, 
        nfzc = 0, 
        cc_energy_func=None, 
        t1_residual_func=None, 
        t2_residual_func=None,
        t3_residual_func=None,
        cc_pseudoenergy_func=None, 
        l1_residual_func=None, 
        l2_residual_func=None):

        """
        initialize CC class

        :params wfn: a psi4 wave function
        :params mol: a psi4 molecule object
        :params use_spin_orbital_basis: do use a spin-orbital basis?
        :params nfzc: number of frozen core
        :params cc_energy_func: python function for evaluating the cc energy
        :params t1_residual_func: python function for the t1 residual equations
        :params t2_residual_func: python function for the t2 residual equations
        :params t3_residual_func: python function for the t3 residual equations
        :params cc_pseudoenergy_func: python function for evaluating the lambda cc pseudoenergy
        :params l1_residual_func: python function for the l1 residual equations
        :params l2_residual_func: python function for the l2 residual equations
        """

        self.wfn = wfn
        self.mol = mol
        self.use_spin_orbital_basis = use_spin_orbital_basis

        if self.use_spin_orbital_basis:

            self.nsocc, self.nsvirt, self.f, self.g, self.efzc = get_integrals(self.wfn, nfzc = nfzc)

            # occupied, virtual slices
            n = np.newaxis
            self.o = slice(None, self.nsocc)
            self.v = slice(self.nsocc, None)

            # orbital energies
            row, col = self.f.shape
            eps = np.zeros(row)
            for i in range(0,row):
                eps[i] = self.f[i,i]

            # energy denominators
            self.e_abij = 1 / (-eps[self.v, n, n, n] - eps[n, self.v, n, n] + eps[n, n, self.o, n] + eps[
                n, n, n, self.o])
            self.e_ai = 1 / (-eps[self.v, n] + eps[n, self.o])

            # hartree-fock energy
            self.hf_energy = 1.0 * einsum('ii', self.f[self.o, self.o]) -0.5 * einsum('ijij', self.g[self.o, self.o, self.o, self.o])

            self.t1 = np.zeros((self.nsvirt, self.nsocc))
            self.t2 = np.zeros((self.nsvirt, self.nsvirt, self.nsocc, self.nsocc))

            if t3_residual_func is not None:
                raise Exception("spin-orbital CCSDT is not implemented")

            self.l1 = None
            self.l2 = None

        else:

            noa, nob, nva, nvb, self.f_aa, self.f_bb, self.g_aaaa, self.g_bbbb, self.g_abab, self.efzc  = get_integrals_with_spin(self.wfn, nfzc = nfzc)

            # occupied, virtual slices
            n = np.newaxis
            oa = slice(None, noa)
            ob = slice(None, nob)
            va = slice(noa, None)
            vb = slice(nob, None)

            # orbital energies
            row, col = self.f_aa.shape
            eps_a = np.zeros(row)
            for i in range(0,row):
                eps_a[i] = self.f_aa[i,i]

            row, col = self.f_bb.shape
            eps_b = np.zeros(row)
            for i in range(0,row):
                eps_b[i] = self.f_bb[i,i]

            # energy denominators
            self.e_aaaa_abij = 1 / ( - eps_a[va, n, n, n]
                                     - eps_a[n, va, n, n]
                                     + eps_a[n, n, oa, n]
                                     + eps_a[n, n, n, oa] )
            self.e_bbbb_abij = 1 / ( - eps_b[vb, n, n, n]
                                     - eps_b[n, vb, n, n]
                                     + eps_b[n, n, ob, n]
                                     + eps_b[n, n, n, ob] )
            self.e_abab_abij = 1 / ( - eps_a[va, n, n, n]
                                     - eps_b[n, vb, n, n]
                                     + eps_a[n, n, oa, n]
                                     + eps_b[n, n, n, ob] )

            if t3_residual_func is not None:
                self.e_aaaaaa_abcijk = 1 / ( - eps_a[va, n,  n,  n,  n,  n]
                                             - eps_a[n, va,  n,  n,  n,  n]
                                             - eps_a[n,  n, va,  n,  n,  n]
                                             + eps_a[n,  n,  n, oa,  n,  n]
                                             + eps_a[n,  n,  n,  n, oa,  n]
                                             + eps_a[n,  n,  n,  n,  n, oa] )
                self.e_aabaab_abcijk = 1 / ( - eps_a[va, n,  n,  n,  n,  n]
                                             - eps_a[n, va,  n,  n,  n,  n]
                                             - eps_b[n,  n, vb,  n,  n,  n]
                                             + eps_a[n,  n,  n, oa,  n,  n]
                                             + eps_a[n,  n,  n,  n, oa,  n]
                                             + eps_b[n,  n,  n,  n,  n, ob] )
                self.e_abbabb_abcijk = 1 / ( - eps_a[va, n,  n,  n,  n,  n]
                                             - eps_b[n, vb,  n,  n,  n,  n]
                                             - eps_b[n,  n, vb,  n,  n,  n]
                                             + eps_a[n,  n,  n, oa,  n,  n]
                                             + eps_b[n,  n,  n,  n, ob,  n]
                                             + eps_b[n,  n,  n,  n,  n, ob] )
                self.e_bbbbbb_abcijk = 1 / ( - eps_b[vb, n,  n,  n,  n,  n]
                                             - eps_b[n, vb,  n,  n,  n,  n]
                                             - eps_b[n,  n, vb,  n,  n,  n]
                                             + eps_b[n,  n,  n, ob,  n,  n]
                                             + eps_b[n,  n,  n,  n, ob,  n]
                                             + eps_b[n,  n,  n,  n,  n, ob] )
            else:
                self.e_aaaaaa_abcijk = np.zeros((1))
                self.e_aabaab_abcijk = np.zeros((1))
                self.e_abbabb_abcijk = np.zeros((1))
                self.e_bbbbbb_abcijk = np.zeros((1))

            self.e_aa_ai = 1 / (-eps_a[va, n] + eps_a[n, oa])
            self.e_bb_ai = 1 / (-eps_b[vb, n] + eps_b[n, ob])

            # hartree-fock energy
            self.hf_energy = ( einsum('ii', self.f_aa[oa, oa]) + einsum('ii', self.f_bb[ob, ob])
                           - 0.5 * einsum('ijij', self.g_aaaa[oa, oa, oa, oa])
                           - 0.5 * einsum('ijij', self.g_bbbb[ob, ob, ob, ob])
                           - 1.0 * einsum('ijij', self.g_abab[oa, ob, oa, ob]) )

            self.t1_aa = np.zeros((nva, noa))
            self.t1_bb = np.zeros((nvb, nob))

            self.t2_aaaa = np.zeros((nva, nva, noa, noa))
            self.t2_bbbb = np.zeros((nvb, nvb, nob, nob))
            self.t2_abab = np.zeros((nva, nvb, noa, nob))

            if t3_residual_func is not None:
                self.t3_aaaaaa = np.zeros((nva, nva, nva, noa, noa, noa))
                self.t3_aabaab = np.zeros((nva, nva, nvb, noa, noa, nob))
                self.t3_abbabb = np.zeros((nva, nvb, nvb, noa, nob, nob))
                self.t3_bbbbbb = np.zeros((nvb, nvb, nvb, nob, nob, nob))
            else:
                self.t3_aaaaaa = np.zeros((1))
                self.t3_aabaab = np.zeros((1))
                self.t3_abbabb = np.zeros((1))
                self.t3_bbbbbb = np.zeros((1))

            self.l1_aa = None
            self.l1_bb = None
            self.l2_aaaa = None
            self.l2_bbbb = None
            self.l2_abab = None

            self.noa = noa
            self.nob = nob
            self.nva = nva
            self.nvb = nvb
            self.oa = oa
            self.ob = ob
            self.va = va
            self.vb = vb
   
        self.nuclear_repulsion_energy = self.mol.nuclear_repulsion_energy()

        self.cc_energy = types.MethodType(cc_energy_func, self)
        self.t1_residual = types.MethodType(t1_residual_func, self)
        self.t2_residual = types.MethodType(t2_residual_func, self)
        if t3_residual_func is not None:
            self.t3_residual = types.MethodType(t3_residual_func, self)
        else:
            self.t3_residual = None

        if cc_pseudoenergy_func is not None:
            self.cc_pseudoenergy = types.MethodType(cc_pseudoenergy_func, self)
        if l1_residual_func is not None:
            self.l1_residual = types.MethodType(l1_residual_func, self)
        else:
            self.l1_residual = None

        if l2_residual_func is not None:
            self.l2_residual = types.MethodType(l2_residual_func, self)
        else:
            self.l2_residual = None

    def t_solver(self):
        """
    
        run ccsd amplitude equations
    
        :return energy: the total ccsd energy
    
        """

        if self.use_spin_orbital_basis:
            self.cc_iterations(e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)
            energy = self.cc_energy()
        else:
            self.cc_iterations_with_spin(e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)
            energy = self.cc_energy()
                        
        print("")
        print("    CCSD Correlation Energy: {: 20.12f}".format(energy - self.hf_energy))
        print("    CCSD Total Energy:       {: 20.12f}".format(energy + self.nuclear_repulsion_energy + self.efzc))
        print("")

        self.energy = energy

        return energy + self.nuclear_repulsion_energy + self.efzc

    def lambda_solver(self):
        """
    
        run ccsd lambda iterations
    
        """

        if self.use_spin_orbital_basis:
            raise Exception("spin-orbital CCSD lambda is not implemented")
        self.ccsd_lambda_iterations_with_spin(e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)

        return self.cc_pseudoenergy() #+ self.nuclear_repulsion_energy + self.efzc
                        
    def cc_iterations_with_spin(self, max_iter=500,e_convergence=1e-8,r_convergence=1e-8,diis_size=None, diis_start_cycle=4):

        # initialize diis if diis_size is not None
        # else normal scf iterate
    
        if diis_size is not None:
            diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
            t1_aa_end = self.t1_aa.size
            t1_bb_end = t1_aa_end + self.t1_bb.size
            t2_aaaa_end = t1_bb_end + self.t2_aaaa.size
            t2_abab_end = t2_aaaa_end + self.t2_abab.size
            t2_bbbb_end = t2_abab_end + self.t2_bbbb.size
            t3_aaaaaa_end = t2_bbbb_end + self.t3_aaaaaa.size
            t3_aabaab_end = t3_aaaaaa_end + self.t3_aabaab.size
            t3_abbabb_end = t3_aabaab_end + self.t3_abbabb.size
            t3_bbbbbb_end = t3_abbabb_end + self.t3_bbbbbb.size
            old_vec = np.hstack((self.t1_aa.flatten(), 
                                 self.t1_bb.flatten(), 
                                 self.t2_aaaa.flatten(), 
                                 self.t2_abab.flatten(), 
                                 self.t2_bbbb.flatten(), 
                                 self.t3_aaaaaa.flatten(),
                                 self.t3_aabaab.flatten(),
                                 self.t3_abbabb.flatten(),
                                 self.t3_bbbbbb.flatten(),
                               ))
    
        fock_e_aa_ai = np.reciprocal(self.e_aa_ai)
        fock_e_bb_ai = np.reciprocal(self.e_bb_ai)
    
        fock_e_aaaa_abij = np.reciprocal(self.e_aaaa_abij)
        fock_e_bbbb_abij = np.reciprocal(self.e_bbbb_abij)
        fock_e_abab_abij = np.reciprocal(self.e_abab_abij)

        if self.t3_residual is not None:
            fock_e_aaaaaa_abcijk = np.reciprocal(self.e_aaaaaa_abcijk)
            fock_e_aabaab_abcijk = np.reciprocal(self.e_aabaab_abcijk)
            fock_e_abbabb_abcijk = np.reciprocal(self.e_abbabb_abcijk)
            fock_e_bbbbbb_abcijk = np.reciprocal(self.e_bbbbbb_abcijk)
        else:
            fock_e_aaaaaa_abcijk = np.zeros((1))
            fock_e_aabaab_abcijk = np.zeros((1))
            fock_e_abbabb_abcijk = np.zeros((1))
            fock_e_bbbbbb_abcijk = np.zeros((1))

        old_energy = self.cc_energy()
    
        print("")
        print("    ==> CCSD amplitude equations <==")
        print("")
        print("     Iter               Energy                 |dE|                 |dT|")
        for idx in range(max_iter):
    
            residual_t1_aa, residual_t1_bb = self.t1_residual()
    
            residual_t2_aaaa, residual_t2_abab, residual_t2_bbbb = self.t2_residual()
    
            if self.t3_residual is not None:
                residual_t3_aaaaaa, residual_t3_aabaab, residual_t3_abbabb, residual_t3_bbbbbb = self.t3_residual()
            else:
                residual_t3_aaaaaa = np.zeros((1))
                residual_t3_aabaab = np.zeros((1))
                residual_t3_abbabb = np.zeros((1))
                residual_t3_bbbbbb = np.zeros((1))

            res_norm = ( np.linalg.norm(residual_t1_aa)
                       + np.linalg.norm(residual_t1_bb)
                       + np.linalg.norm(residual_t2_aaaa)
                       + np.linalg.norm(residual_t2_bbbb)
                       + np.linalg.norm(residual_t2_abab) 
                       + np.linalg.norm(residual_t3_aaaaaa) 
                       + np.linalg.norm(residual_t3_aabaab) 
                       + np.linalg.norm(residual_t3_abbabb) 
                       + np.linalg.norm(residual_t3_bbbbbb) 
                       )

    
            t1_aa_res = residual_t1_aa + fock_e_aa_ai * self.t1_aa
            t1_bb_res = residual_t1_bb + fock_e_bb_ai * self.t1_bb
    
            t2_aaaa_res = residual_t2_aaaa + fock_e_aaaa_abij * self.t2_aaaa
            t2_bbbb_res = residual_t2_bbbb + fock_e_bbbb_abij * self.t2_bbbb
            t2_abab_res = residual_t2_abab + fock_e_abab_abij * self.t2_abab

            t3_aaaaaa_res = residual_t3_aaaaaa + fock_e_aaaaaa_abcijk * self.t3_aaaaaa
            t3_aabaab_res = residual_t3_aabaab + fock_e_aabaab_abcijk * self.t3_aabaab
            t3_abbabb_res = residual_t3_abbabb + fock_e_abbabb_abcijk * self.t3_abbabb
            t3_bbbbbb_res = residual_t3_bbbbbb + fock_e_bbbbbb_abcijk * self.t3_bbbbbb
    
            new_t1_aa = t1_aa_res * self.e_aa_ai
            new_t1_bb = t1_bb_res * self.e_bb_ai
    
            new_t2_aaaa = t2_aaaa_res * self.e_aaaa_abij
            new_t2_bbbb = t2_bbbb_res * self.e_bbbb_abij
            new_t2_abab = t2_abab_res * self.e_abab_abij

            new_t3_aaaaaa = t3_aaaaaa_res * self.e_aaaaaa_abcijk
            new_t3_aabaab = t3_aabaab_res * self.e_aabaab_abcijk
            new_t3_abbabb = t3_abbabb_res * self.e_abbabb_abcijk
            new_t3_bbbbbb = t3_bbbbbb_res * self.e_bbbbbb_abcijk
    
            # diis update
            if diis_size is not None:
                vectorized_iterate = np.hstack((new_t1_aa.flatten(), 
                                                new_t1_bb.flatten(), 
                                                new_t2_aaaa.flatten(), 
                                                new_t2_abab.flatten(), 
                                                new_t2_bbbb.flatten(), 
                                                new_t3_aaaaaa.flatten(),
                                                new_t3_aabaab.flatten(),
                                                new_t3_abbabb.flatten(),
                                                new_t3_bbbbbb.flatten(),
                                              ))


                error_vec = old_vec - vectorized_iterate
                new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                     error_vec)

                new_t1_aa = new_vectorized_iterate[:t1_aa_end].reshape(self.t1_aa.shape)
                new_t1_bb = new_vectorized_iterate[t1_aa_end:t1_bb_end].reshape(self.t1_bb.shape)
    
                new_t2_aaaa = new_vectorized_iterate[t1_bb_end:t2_aaaa_end].reshape(self.t2_aaaa.shape)
                new_t2_abab = new_vectorized_iterate[t2_aaaa_end:t2_abab_end].reshape(self.t2_abab.shape)
                new_t2_bbbb = new_vectorized_iterate[t2_abab_end:t2_bbbb_end].reshape(self.t2_bbbb.shape)
    
                new_t3_aaaaaa = new_vectorized_iterate[t2_bbbb_end:t3_aaaaaa_end].reshape(self.t3_aaaaaa.shape)
                new_t3_aabaab = new_vectorized_iterate[t3_aaaaaa_end:t3_aabaab_end].reshape(self.t3_aabaab.shape)
                new_t3_abbabb = new_vectorized_iterate[t3_aabaab_end:t3_abbabb_end].reshape(self.t3_abbabb.shape)
                new_t3_bbbbbb = new_vectorized_iterate[t3_abbabb_end:t3_bbbbbb_end].reshape(self.t3_bbbbbb.shape)

                old_vec = new_vectorized_iterate
    
            self.t1_aa = new_t1_aa
            self.t1_bb = new_t1_bb
    
            self.t2_aaaa = new_t2_aaaa
            self.t2_bbbb = new_t2_bbbb
            self.t2_abab = new_t2_abab

            self.t3_aaaaaa = new_t3_aaaaaa
            self.t3_aabaab = new_t3_aabaab
            self.t3_abbabb = new_t3_abbabb
            self.t3_bbbbbb = new_t3_bbbbbb

            current_energy = self.cc_energy()
    
            delta_e = np.abs(old_energy - current_energy)
    
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(idx, current_energy - self.hf_energy, delta_e, res_norm))
            if delta_e < e_convergence and res_norm < r_convergence:
                break
            else:
                old_energy = current_energy
    
        else:
            raise ValueError("CCSD iterations did not converge")
    
    def cc_iterations(self, max_iter=500, e_convergence=1e-8,r_convergence=1e-8,diis_size=None, diis_start_cycle=4):
               
        # initialize diis if diis_size is not None
        # else normal scf iterate
    
        if diis_size is not None:
            diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
            t1_dim = self.t1.size
            old_vec = np.hstack((self.t1.flatten(), self.t2.flatten()))
    
        fock_e_ai = np.reciprocal(self.e_ai)
        fock_e_abij = np.reciprocal(self.e_abij)
        old_energy = self.cc_energy()
    
        print("")
        print("    ==> CCSD amplitude equations <==")
        print("")
        print("     Iter               Energy                 |dE|                 |dT|")
        for idx in range(max_iter):
    
            residual_singles = self.t1_residual()
            residual_doubles = self.t2_residual()
    
            res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles)
    
            singles_res = residual_singles + fock_e_ai * self.t1
            doubles_res = residual_doubles + fock_e_abij * self.t2
    
            new_singles = singles_res * self.e_ai
            new_doubles = doubles_res * self.e_abij
    
            # diis update
            if diis_size is not None:
                vectorized_iterate = np.hstack(
                    (new_singles.flatten(), new_doubles.flatten()))
                error_vec = old_vec - vectorized_iterate
                new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                     error_vec)
                new_singles = new_vectorized_iterate[:t1_dim].reshape(self.t1.shape)
                new_doubles = new_vectorized_iterate[t1_dim:].reshape(self.t2.shape)
                old_vec = new_vectorized_iterate
    
            self.t1 = new_singles
            self.t2 = new_doubles
            current_energy = self.cc_energy()

            delta_e = np.abs(old_energy - current_energy)
    
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(idx, current_energy - self.hf_energy, delta_e, res_norm))
            if delta_e < e_convergence and res_norm < r_convergence:
                break
            else:
                old_energy = current_energy
    
        else:
            raise ValueError("CCSD iterations did not converge")
    
    def ccsd_lambda_iterations_with_spin(self, max_iter=500,e_convergence=1e-8,r_convergence=1e-8,diis_size=None, diis_start_cycle=4):

        self.l1_aa = self.t1_aa.transpose(1, 0)
        self.l1_bb = self.t1_bb.transpose(1, 0)
        self.l2_aaaa = self.t2_aaaa.transpose(2, 3, 0, 1)
        self.l2_bbbb = self.t2_bbbb.transpose(2, 3, 0, 1)
        self.l2_abab = self.t2_abab.transpose(2, 3, 0, 1)

        # initialize diis if diis_size is not None
        # else normal scf iterate
    
        if diis_size is not None:
            diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
            l1_aa_end = self.l1_aa.size
            l1_bb_end = l1_aa_end + self.l1_bb.size
            l2_aaaa_end = l1_bb_end + self.l2_aaaa.size
            l2_bbbb_end = l2_aaaa_end + self.l2_bbbb.size
            l2_abab_end = l2_bbbb_end + self.l2_abab.size
            old_vec = np.hstack((self.l1_aa.flatten(), self.l1_bb.flatten(), self.l2_aaaa.flatten(), self.l2_bbbb.flatten(), self.l2_abab.flatten()))
    
        # inverse diagonal fock should be rearranged for lambdas
        fock_e_aa_ia = np.reciprocal(self.e_aa_ai).transpose(1, 0)
        fock_e_bb_ia = np.reciprocal(self.e_bb_ai).transpose(1, 0)
    
        fock_e_aaaa_ijab = np.reciprocal(self.e_aaaa_abij).transpose(2, 3, 0, 1)
        fock_e_bbbb_ijab = np.reciprocal(self.e_bbbb_abij).transpose(2, 3, 0, 1)
        fock_e_abab_ijab = np.reciprocal(self.e_abab_abij).transpose(2, 3, 0, 1)
    
        old_energy = self.cc_pseudoenergy()
    
        print("")
        print("    ==> CCSD lambda amplitude equations <==")
        print("")
        print("     Iter         Pseudoenergy                 |dE|            |dlambda|")
        for idx in range(max_iter):
    
            residual_l1_aa, residual_l1_bb = self.l1_residual()
    
            residual_l2_aaaa, residual_l2_abab, residual_l2_bbbb = self.l2_residual()
    
            res_norm = ( np.linalg.norm(residual_l1_aa)
                       + np.linalg.norm(residual_l1_bb)
                       + np.linalg.norm(residual_l2_aaaa)
                       + np.linalg.norm(residual_l2_bbbb)
                       + np.linalg.norm(residual_l2_abab) )
    
            l1_aa_res = residual_l1_aa + fock_e_aa_ia * self.l1_aa
            l1_bb_res = residual_l1_bb + fock_e_bb_ia * self.l1_bb
    
            l2_aaaa_res = residual_l2_aaaa + fock_e_aaaa_ijab * self.l2_aaaa
            l2_bbbb_res = residual_l2_bbbb + fock_e_bbbb_ijab * self.l2_bbbb
            l2_abab_res = residual_l2_abab + fock_e_abab_ijab * self.l2_abab
    
            new_l1_aa = l1_aa_res * self.e_aa_ai.transpose(1, 0)
            new_l1_bb = l1_bb_res * self.e_bb_ai.transpose(1, 0)
    
            new_l2_aaaa = l2_aaaa_res * self.e_aaaa_abij.transpose(2, 3, 0, 1)
            new_l2_bbbb = l2_bbbb_res * self.e_bbbb_abij.transpose(2, 3, 0, 1)
            new_l2_abab = l2_abab_res * self.e_abab_abij.transpose(2, 3, 0, 1)
    
            # diis update
            if diis_size is not None:
                vectorized_iterate = np.hstack(
                    (new_l1_aa.flatten(), new_l1_bb.flatten(), new_l2_aaaa.flatten(), new_l2_bbbb.flatten(), new_l2_abab.flatten()))
                error_vec = old_vec - vectorized_iterate
                new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                     error_vec)
                new_l1_aa = new_vectorized_iterate[:l1_aa_end].reshape(self.l1_aa.shape)
                new_l1_bb = new_vectorized_iterate[l1_aa_end:l1_bb_end].reshape(self.l1_bb.shape)
    
                new_l2_aaaa = new_vectorized_iterate[l1_bb_end:l2_aaaa_end].reshape(self.l2_aaaa.shape)
                new_l2_bbbb = new_vectorized_iterate[l2_aaaa_end:l2_bbbb_end].reshape(self.l2_bbbb.shape)
                new_l2_abab = new_vectorized_iterate[l2_bbbb_end:l2_abab_end].reshape(self.l2_abab.shape)
    
                old_vec = new_vectorized_iterate
    
            self.l1_aa = new_l1_aa
            self.l1_bb = new_l1_bb
    
            self.l2_aaaa = new_l2_aaaa
            self.l2_bbbb = new_l2_bbbb
            self.l2_abab = new_l2_abab

            current_energy = self.cc_pseudoenergy()
    
            delta_e = np.abs(old_energy - current_energy)
    
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(idx, current_energy, delta_e, res_norm))
            if delta_e < e_convergence and res_norm < r_convergence:
                break
            else:
                old_energy = current_energy
    
        else:
            raise ValueError("CCSD lambda iterations did not converge")

        print("")
        print("    CCSD lambda pseudoenergy:       {: 20.12f}".format(current_energy))
        print("")

