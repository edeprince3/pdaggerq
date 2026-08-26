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
Solvers for CC and lambda CC. Should work for up to quadruple excitations, plus up to 4 photons in QED-CC
"""
 
import numpy as np
from numpy import einsum
import types
import math
import itertools
 
import copy
 
from pdaggerq.numerical.utils.backend import detect_backend, get_integrals_module, \
    nuclear_repulsion_energy, nuclear_dipole_components
from pdaggerq.numerical.utils.diis import DIIS
 
class cc:
 
    def __init__(self, wfn,
        mol,
        nfzc = 0,
        e_convergence = 1e-8,
        r_convergence = 1e-6,
        diis_size = 8,
        diis_start_cycle = 4,
        is_qed = False,
        cc_energy_func = None,
        T_list = [],
        L_list = [],
        cc_pseudoenergy_func=None,
        cavity_frequency=0.07349864501573,
        cavity_lambda=[0.0, 0.0, 0.0]):
 
        """
        initialize CC class
 
        :params wfn: a psi4 wave function, or a converged pyscf mean-field object (RHF, ROHF,
            or UHF). The backend is detected automatically (see pdaggerq.numerical.utils.backend)
            and used to pick the matching integrals module -- no other argument changes needed.
        :params mol: a psi4 molecule object, or a pyscf Mole object (matching whichever package
            wfn came from)
        :params e_convergence: energy convergence thershold
        :params r_convergence: residual equation convergence thershold
        :params nfzc: number of frozen core
        :params T_list: list of dictionaries representing cluster amplitudes
        :params L_list: list of dictionaries representing lambda amplitudes
        :params cc_energy_func: python function for evaluating the cc energy
        :params cc_pseudoenergy_func: python function for evaluating the lambda cc pseudoenergy
        :params is_qed: is this qed-cc?
        :params cavity_frequency: cavity frequency for qed-cc
        :params cavity_lambda: cavity coupling strength for qed-cc
        """
 
        self.wfn = wfn
        self.nfzc = nfzc
        self.mol = mol
        self.e_convergence = e_convergence
        self.r_convergence = r_convergence
        self.diis_size = diis_size
        self.diis_start_cycle = diis_start_cycle
 
        # figure out once whether wfn/mol are psi4 or pyscf objects, and which integrals
        # module matches -- everything below that used to hardcode psi4 goes through these
        self.backend = detect_backend(wfn)
        self._integrals = get_integrals_module(self.backend)
 
        # qed-cc
        self.cavity_frequency = cavity_frequency
        self.cavity_lambda = np.array(cavity_lambda)
        self.is_qed = is_qed
 
        if self.is_qed and self.nfzc > 0:
            raise Exception("QED-CC does not work with frozen core")
 
        noa, nob, nva, nvb, self.f_aa, self.f_bb, self.g_aaaa, self.g_bbbb, self.g_abab, self.efzc  = self._integrals.get_integrals_with_spin(self.wfn, nfzc = self.nfzc)
 
        # occupied, virtual slices
        n = np.newaxis
        oa = slice(None, noa)
        ob = slice(None, nob)
        va = slice(noa, None)
        vb = slice(nob, None)
        self.oa = oa
        self.ob = ob
        self.va = va
        self.vb = vb
        self.slices = {
            'oa': oa,
            'va': va,
            'ob': ob,
            'vb': vb
        }
 
        self.noa = noa
        self.nob = nob
        self.nva = nva
        self.nvb = nvb
        self.dims = {
            'va': self.nva,
            'oa': self.noa,
            'vb': self.nvb,
            'ob': self.nob
        }
 
 
        # DSE and bilinear coupling contributions for QED
        if not self.is_qed:
            self.enuc_dse = 0.0
        else:
            tmp_dipole_aa, tmp_dipole_bb = self._integrals.get_dipole_integrals_with_spin(self.wfn, nfzc = self.nfzc)
 
            # lambda-weighted dipole integrals
            dipole_aa = np.zeros_like(tmp_dipole_aa[0])
            dipole_bb = np.zeros_like(tmp_dipole_bb[0])
            for i in range (0, 3):
                dipole_aa += tmp_dipole_aa[i] * self.cavity_lambda[i]
                dipole_bb += tmp_dipole_bb[i] * self.cavity_lambda[i]
 
            # update eris with dse term
            self.g_aaaa += einsum('ik,jl->ijkl', dipole_aa, dipole_aa)
            self.g_abab += einsum('ik,jl->ijkl', dipole_aa, dipole_bb)
            self.g_bbbb += einsum('ik,jl->ijkl', dipole_bb, dipole_bb)
 
            self.g_aaaa -= einsum('il,jk->ijkl', dipole_aa, dipole_aa)
            self.g_bbbb -= einsum('il,jk->ijkl', dipole_bb, dipole_bb)
 
            # update fock matrix with dse term from dipole integrals
            dip_a = np.einsum('ii->', dipole_aa[oa, oa])
            dip_b = np.einsum('ii->', dipole_bb[ob, ob])
 
            self.f_aa += (dip_a + dip_b) * dipole_aa
            self.f_aa -= np.einsum('pi,iq->pq', dipole_aa[:, oa], dipole_aa[oa, :])
 
            self.f_bb += (dip_a + dip_b) * dipole_bb
            self.f_bb -= np.einsum('pi,iq->pq', dipole_bb[:, ob], dipole_bb[ob, :])
 
            # update fock matrix with dse term from quadrupole integrals
            tmp_quadrupole_aa, tmp_quadrupole_bb = self._integrals.get_quadrupole_integrals_with_spin(self.wfn, nfzc = self.nfzc)
 
            quadrupole_aa = np.zeros_like(tmp_quadrupole_aa[0])
            quadrupole_bb = np.zeros_like(tmp_quadrupole_bb[0])
            idx = 0
            for i in range(3):
                for j in range(i, 3):
                    factor = 1.0 if i == j else 2.0
                    quadrupole_aa += factor * self.cavity_lambda[i] * self.cavity_lambda[j] * tmp_quadrupole_aa[idx]
                    quadrupole_bb += factor * self.cavity_lambda[i] * self.cavity_lambda[j] * tmp_quadrupole_bb[idx]
                    idx += 1
 
            self.f_aa -= 0.5 * quadrupole_aa
            self.f_bb -= 0.5 * quadrupole_bb
 
            # nuclear dipole contribution to the Fock matrix
            nuc_dip_x, nuc_dip_y, nuc_dip_z = nuclear_dipole_components(mol, self.backend)
            self.nuc_dip = self.cavity_lambda[0] * nuc_dip_x \
                +  self.cavity_lambda[1] * nuc_dip_y \
                +  self.cavity_lambda[2] * nuc_dip_z
            self.f_aa += self.nuc_dip * dipole_aa
            self.f_bb += self.nuc_dip * dipole_bb
 
            self.enuc_dse = 0.5 * self.nuc_dip**2
 
            # scale dipole integrals for the bilinear coupling term
            self.dipole_aa = np.sqrt(0.5 * self.cavity_frequency) * dipole_aa
            self.dipole_bb = np.sqrt(0.5 * self.cavity_frequency) * dipole_bb
 
        # orbital energies
        self.eps = {}
        row, col = self.f_aa.shape
        self.eps['a'] = np.zeros(row)
        for i in range(0,row):
            self.eps['a'][i] = self.f_aa[i,i]
 
        row, col = self.f_bb.shape
        self.eps['b'] = np.zeros(row)
        for i in range(0,row):
            self.eps['b'][i] = self.f_bb[i,i]
 
        self.T = {}
        self.T_residual = {}
        self.D = {}

        # initialize T amplitude dictionaries
        self.T = {}
        self.T_residual = {}
        self.T_meta = {}
        self.initialize_amplitudes(T_list, self.T, self.T_residual, self.T_meta, function = 'residual')

        # initialize denominator dictionaries
        self.D = {}
        self.initialize_denominators(T_list, self.D)
 
        # initialize lambda amplitude dictionaries
        self.L = {}
        self.L_residual = {}
        self.L_meta = {}
        self.initialize_amplitudes(L_list, self.L, self.L_residual, self.L_meta, function = 'residual')
 
        # hartree-fock energy
        self.hf_energy = ( einsum('ii', self.f_aa[oa, oa]) + einsum('ii', self.f_bb[ob, ob])
                       - 0.5 * einsum('ijij', self.g_aaaa[oa, oa, oa, oa])
                       - 0.5 * einsum('ijij', self.g_bbbb[ob, ob, ob, ob])
                       - 1.0 * einsum('ijij', self.g_abab[oa, ob, oa, ob]) )
 
        self.nuclear_repulsion_energy = nuclear_repulsion_energy(self.mol, self.backend)
 
        # cc energy function
        self.cc_energy = types.MethodType(cc_energy_func, self)

    def initialize_denominators(self, amp_list, denom):
        """
        Initialize amplitude denominator dictionaries, mirroring initialize_amplitudes'
        base_name/spaces/spins structure so the two stay in lockstep.
        :param amp_list: list of amplitude dictionaries containing spaces / spins / nph
        :param denom: denominator dictionary to populate
        """
    
        dims = {
            'va': self.nva,
            'oa': self.noa,
            'vb': self.nvb,
            'ob': self.nob
        }
    
        for my_amp in amp_list:
    
            raw_spaces = my_amp.get('spaces', [])
            if raw_spaces and len(raw_spaces) != 2:
                raise Exception("amp_list spaces should have exactly two elements (left/right)")
            full_spaces = "".join(raw_spaces)
    
            order = max(len(raw_spaces[0]), len(raw_spaces[1])) if raw_spaces else 0
    
            nph = my_amp.get('nph', 0)
    
            base_name = str(order)
            if nph > 0:
                base_name += '_' + str(nph) + 'p'
    
            denom[base_name] = {}
    
            rank = len(full_spaces)
    
            if 'spins' in my_amp and rank > 0:
                for raw_spins in my_amp['spins']:
    
                    if isinstance(raw_spins, (list, tuple)):
                        full_spins = "".join(raw_spins)
                    else:
                        full_spins = raw_spins
    
                    shape = tuple(dims[space + spin] for space, spin in zip(full_spaces, full_spins))
    
                    d = np.zeros(shape)
                    for i, (space, spin) in enumerate(zip(full_spaces, full_spins)):
                        sign = 1.0 if space == 'o' else -1.0
                        eps_1d = self.eps[spin][self.slices[space + spin]]
    
                        b_idx = [None] * rank
                        b_idx[i] = slice(None)
    
                        d += sign * eps_1d[tuple(b_idx)]
    
                    if nph > 0:
                        d -= nph * self.cavity_frequency
    
                    denom[base_name][full_spins] = 1.0 / d
            else:
                # scalar / pure-photon amplitude: denominator is purely photonic (or, if
                # nph == 0 too, this amp is a bare scalar with no well-defined denominator --
                # matches how initialize_amplitudes stores a dummy zeros((1,)) in that case)
                d = -nph * self.cavity_frequency if nph > 0 else 0.0
                denom[base_name][''] = 1.0 / d

    def initialize_amplitudes(self, amp_list, amp, amp_residual, amp_meta, function = 'residual'):
        """
        Initialize T or lambda amplitude dictionaries
        :param amp_list: list of amplitude dictionaries containing spaces / spins / residual function
        :param amp: amplitude dictionary
        :param amp_residual: residual function dictionary
        :param amp_meta: meta-data dictionary for left/right space/spin information
        :param function: the function in the amp_list element that we wish to initialize
        """

        dims = {
            'va': self.nva,
            'oa': self.noa,
            'vb': self.nvb,
            'ob': self.nob
        } 
        
        for my_amp in amp_list:
        
            # [v,o], [vv,oo], etc.
            raw_spaces = my_amp.get('spaces', [])
        
            if raw_spaces and len(raw_spaces) != 2:
                raise Exception("amp_list spaces should have exactly two elements (left/right)")
            full_spaces = "".join(raw_spaces)
        
            # Amplitude order (e.g., T1 -> 1, T2 -> 2)
            if raw_spaces:
                order = max(len(raw_spaces[0]), len(raw_spaces[1]))
            else:
                order = 0

            # Number of photons
            nph = my_amp.get('nph', 0)

            # Base key for this rank (e.g., '1', '2', '0_1p' for 1 photon, '1_1p', etc.)
            base_name = str(order)
            if nph > 0:
                base_name += '_' + str(nph) + 'p'

            # Initialize nested dictionaries for this rank
            amp[base_name] = {}

            # Bind residual function to instance
            amp_residual[base_name] = types.MethodType(my_amp[function], self)

            # If no spins are provided (like for r0 or pure photons), 
            # we provide a dummy list [[]] so the packer loops exactly once.
            raw_spins = my_amp.get('spins', [])
            if not raw_spins:
                raw_spins = [[]]

            # Store the exact structural boundaries for the solver
            amp_meta[base_name] = {
                'raw_spaces': raw_spaces,
                'raw_spins': raw_spins
            }

            if 'spins' in my_amp and len(full_spaces) > 0:
                for raw_spins in my_amp['spins']:

                    if isinstance(raw_spins, (list, tuple)):
                        full_spins = "".join(raw_spins)
                    else:
                        full_spins = raw_spins

                    shape = tuple(dims[space + spin] for space, spin in zip(full_spaces, full_spins))
                    amp[base_name][full_spins] = np.zeros(shape, dtype=np.float64)
            else:
                amp[base_name][''] = np.zeros((1,), dtype=np.float64)
 
    def initialize_lambda(self, L_list = [], cc_pseudoenergy_func = None):
        """
        wrapper for initialize_amplitudes for initializing lambda externally
        """

        self.L = {}
        self.L_residual = {}
        self.L_meta = {}
        self.initialize_amplitudes(L_list, self.L, self.L_residual, self.L_meta, function = 'residual')

        # lambda CC pseudoenergy function
        if cc_pseudoenergy_func is not None:
            self.cc_pseudoenergy = types.MethodType(cc_pseudoenergy_func, self)
        else:
            self.cc_pseudoenergy = lambda: 0.0

    #def initialize_lambda(self, L_list = [], cc_pseudoenergy_func = None):
 
    #    # lambda CC pseudoenergy function
    #    if cc_pseudoenergy_func is not None:
    #        self.cc_pseudoenergy = types.MethodType(cc_pseudoenergy_func, self)
    #    else:
    #        self.cc_pseudoenergy = lambda: 0.0
 
    #    self.L = {}
    #    self.L_residual = {}
    #    for myL in L_list:
 
    #        # vo, vvoo, etc.
    #        spaces = myL.get('spaces', '')
 
    #        # Amplitude order (e.g., L1 -> 1, L2 -> 2)
    #        order = len(spaces) // 2
 
    #        # Number of photons
    #        nph = myL.get('nph', 0)
 
    #        # Base key for this rank (e.g., '1', '2', '0_1p' for 1 photon, '1_1p', etc.)
    #        base_name = str(order)
    #        if nph > 0:
    #            base_name += '_' + str(nph) + 'p'
 
    #        # Initialize nested dictionaries for this rank
    #        self.L[base_name] = {}
 
    #        # Bind residual method to instance
    #        self.L_residual[base_name] = types.MethodType(myL['residual'], self)
 
    #        # Fermionic amplitudes (has spins e.g. 'aa', 'abab')
    #        if 'spins' in myL and len(spaces) > 0:
    #            rank = len(spaces)
 
    #            for spins in myL['spins']:
    #                shape = tuple(self.dims[space + spin] for space, spin in zip(spaces, spins))
 
    #                # Store tensors in nested dicts
    #                self.L[base_name][spins] = np.zeros(shape)
 
    #        # Pure photon / zero-fermion amplitudes
    #        else:
    #            if nph > 0:
    #                # 1-element 1D array for scalar photon amplitudes
    #                self.L[base_name][''] = np.zeros((1,))
 
    def t_solver(self):
        """
 
        run ccsd amplitude equations
 
        :return energy: the total ccsd energy
 
        """
 
        self.cc_iterations_with_spin(e_convergence=self.e_convergence, r_convergence=self.r_convergence, diis_size=self.diis_size, diis_start_cycle=self.diis_start_cycle)
        energy = self.cc_energy()
 
        print("")
        print("    CC Correlation Energy: {: 20.12f}".format(energy - self.hf_energy))
 
        print("    CC Total Energy:       {: 20.12f}".format(energy + self.enuc_dse + self.nuclear_repulsion_energy + self.efzc))
        print("")
 
        self.energy = energy
 
        return energy + self.enuc_dse + self.nuclear_repulsion_energy + self.efzc
 
    def lambda_solver(self):
        """
 
        run ccsd lambda iterations
 
        """
 
        self.cc_iterations_with_spin(e_convergence=self.e_convergence, r_convergence=self.r_convergence, diis_size=self.diis_size, diis_start_cycle=self.diis_start_cycle, is_lambda = True)
        energy = self.cc_pseudoenergy()
 
        print("")
        print("    CC lambda pseudoenergy:       {: 20.12f}".format(energy))
        print("")
 
        return self.cc_pseudoenergy() #+ self.nuclear_repulsion_energy + self.efzc
 
    def pack_diis_vectors(self, amps, residuals):
        """
        Flattens all amplitudes and corresponding residuals into matching 1D vectors.
 
        :param amps: dict mapping base_name -> {spins: amp_tensor}
        :param residuals: dict mapping base_name -> {spins: res_tensor}
        :return: (amp_vec, err_vec) as 1D NumPy arrays
        """
        amp_list = []
        err_list = []
 
        for base_name, spin_dict in amps.items():
            res_dict = residuals[base_name]
            for spins, amp_tensor in spin_dict.items():
                res_tensor = res_dict[spins]
 
                # flatten/ravel both arrays into 1D slices
                amp_list.append(amp_tensor.ravel())
                err_list.append(res_tensor.ravel())
 
        amp_vec = np.concatenate(amp_list)
        err_vec = np.concatenate(err_list)
 
        return amp_vec, err_vec
 
    def unpack_amp_vector(self, flat_amp_vec, target_amps):
        """
        Unpacks a 1D vector (e.g. from DIIS) back into the target_amps dictionary in-place.
        """
        idx = 0
        # Iterate over the target dictionary directly
        for base_name, spin_dict in target_amps.items():
            for spins, current_tensor in spin_dict.items():
                size = current_tensor.size
                shape = current_tensor.shape
 
                # Extract slice, reshape, and overwrite the tensor in-place
                target_amps[base_name][spins] = flat_amp_vec[idx:idx + size].reshape(shape)
                idx += size
 
    @property
    def amplitude_size(self):
        """
        Calculates the total number of scalar elements across all T amplitude tensors.
        """
        return sum(
            tensor.size
            for spin_dict in self.T.values()
            for tensor in spin_dict.values()
        )
 
    def cc_iterations_with_spin(self, max_iter=500,e_convergence=1e-8,r_convergence=1e-8, diis_size=8, diis_start_cycle=4, is_lambda=False):
        """
        solve cc / lambda cc equations, with spin traced quantities
 
        :param max_iter: maximum number of iterations
        :param e_convergence: energy convergence between iterations
        :param r_convergence: convergence of the 2-norm of the residuals
        :param diis_size: dimension of the DIIS extrapolation
        :param diis_start_cycle: when to start the DIIS extrapolation
        :param is_lambda: are we solving CC or lambda-CC?
        """
 
        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        old_vec = np.zeros((self.amplitude_size))
 
        old_energy = self.cc_energy()
 
        print("")
        if is_lambda:
            print("    ==> CC lambda amplitude equations <==")
            print("")
            print("     Iter         Pseudoenergy                 |dE|            |dlambda|")
 
        else:
            print("    ==> CC amplitude equations <==")
            print("")
            print("     Iter               Energy                 |dE|                 |dT|")
 
        residuals = {}
 
        # Alias the dictionaries based on the target
        amps = self.L if is_lambda else self.T
        amps_meta = self.L_meta if is_lambda else self.T_meta
        res_funcs = self.L_residual if is_lambda else self.T_residual
 
        for idx in range(max_iter):
 
            res_norm = 0.0
 
            # Loop through each excitation rank
            for base_name, res_func in res_funcs.items():
 
                # Evaluate residual tensors
                res_dict = res_func()[base_name]
 
                # Store spin-channel dict in residuals
                residuals[base_name] = res_dict
 
                # Loop through spin channels
                for spins, res_tensor in res_dict.items():
 
                    # Accumulate squared residual norm across all ranks/spins
                    res_norm += float(np.linalg.norm(res_tensor))**2
 
                    # Update T or lambda
                    amps[base_name][spins] += res_tensor * self.D[base_name][spins]
 
            # Calculate norm from squared norm
            res_norm = np.sqrt(res_norm)
 
            # diis update
            amp_vec, err_vec = self.pack_diis_vectors(amps, residuals)
            new_amp_vec = diis_update.compute_new_vec(amp_vec, err_vec)
            self.unpack_amp_vector(new_amp_vec, amps)

            # explicitly antisymmetrize amplitudes
            self.antisymmetrize_amplitudes(amps, amps_meta)
 
            # Calculate new energy
            current_energy = self.cc_energy() - self.hf_energy if not is_lambda else self.cc_pseudoenergy()
 
            delta_e = np.abs(old_energy - current_energy)
 
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(idx, current_energy, delta_e, res_norm))
            if delta_e < e_convergence and res_norm < r_convergence:
                break
            else:
                old_energy = current_energy
        else:
            if is_lambda:
                raise ValueError("lambda-CC iterations did not converge")
            else:
                raise ValueError("CC iterations did not converge")
 
    def opdm(self, opdm_func = None):
 
        if opdm_func is None:
            raise Exception("provide an opdm_func")
 
        # Evaluate OPDM
        self.density_matrix = types.MethodType(opdm_func, self)
        opdm = self.density_matrix()
 
        # Alpha spin OPDM
        opdm_a = np.block([
            [opdm['aa_oo'], opdm['aa_ov']],
            [opdm['aa_vo'], opdm['aa_vv']]
        ])
 
        # Beta spin OPDM
        opdm_b = np.block([
            [opdm['bb_oo'], opdm['bb_ov']],
            [opdm['bb_vo'], opdm['bb_vv']]
        ])
 
        return opdm_a, opdm_b
 
    def tpdm(self, tpdm_func = None):
 
        if tpdm_func is None:
            raise Exception("provide an tpdm_func")
 
        # Evaluate TPDM
        self.density_matrix = types.MethodType(tpdm_func, self)
        tpdm = self.density_matrix()
 
        # aaaa spin TPDM
        tpdm_aaaa = np.block([
            [  # Axis 0 = o
                [  # Axis 1 = o
                    [tpdm['aaaa_oooo'], tpdm['aaaa_ooov']],  # Axis 2 = o (Axis 3 = o, v)
                    [tpdm['aaaa_oovo'], tpdm['aaaa_oovv']]   # Axis 2 = v (Axis 3 = o, v)
                ],
                [  # Axis 1 = v
                    [tpdm['aaaa_ovoo'], tpdm['aaaa_ovov']],  # Axis 2 = o (Axis 3 = o, v)
                    [tpdm['aaaa_ovvo'], tpdm['aaaa_ovvv']]   # Axis 2 = v (Axis 3 = o, v)
                ]
            ],
            [  # Axis 0 = v
                [  # Axis 1 = o
                    [tpdm['aaaa_vooo'], tpdm['aaaa_voov']],  # Axis 2 = o (Axis 3 = o, v)
                    [tpdm['aaaa_vovo'], tpdm['aaaa_vovv']]   # Axis 2 = v (Axis 3 = o, v)
                ],
                [  # Axis 1 = v
                    [tpdm['aaaa_vvoo'], tpdm['aaaa_vvov']],  # Axis 2 = o (Axis 3 = o, v)
                    [tpdm['aaaa_vvvo'], tpdm['aaaa_vvvv']]   # Axis 2 = v (Axis 3 = o, v)
                ]
            ]
        ])
 
        # bbbb spin TPDM
        tpdm_bbbb = np.block([
            [  # Axis 0 = o
                [  # Axis 1 = o
                    [tpdm['bbbb_oooo'], tpdm['bbbb_ooov']],  # Axis 2 = o (Axis 3 = o, v)
                    [tpdm['bbbb_oovo'], tpdm['bbbb_oovv']]   # Axis 2 = v (Axis 3 = o, v)
                ],
                [  # Axis 1 = v
                    [tpdm['bbbb_ovoo'], tpdm['bbbb_ovov']],  # Axis 2 = o (Axis 3 = o, v)
                    [tpdm['bbbb_ovvo'], tpdm['bbbb_ovvv']]   # Axis 2 = v (Axis 3 = o, v)
                ]
            ],
            [  # Axis 0 = v
                [  # Axis 1 = o
                    [tpdm['bbbb_vooo'], tpdm['bbbb_voov']],  # Axis 2 = o (Axis 3 = o, v)
                    [tpdm['bbbb_vovo'], tpdm['bbbb_vovv']]   # Axis 2 = v (Axis 3 = o, v)
                ],
                [  # Axis 1 = v
                    [tpdm['bbbb_vvoo'], tpdm['bbbb_vvov']],  # Axis 2 = o (Axis 3 = o, v)
                    [tpdm['bbbb_vvvo'], tpdm['bbbb_vvvv']]   # Axis 2 = v (Axis 3 = o, v)
                ]
            ]
        ])
 
        # abab spin TPDM
        tpdm_abab = np.block([
            [  # Axis 0 = o
                [  # Axis 1 = o
                    [tpdm['abab_oooo'], tpdm['abab_ooov']],  # Axis 2 = o (Axis 3 = o, v)
                    [tpdm['abab_oovo'], tpdm['abab_oovv']]   # Axis 2 = v (Axis 3 = o, v)
                ],
                [  # Axis 1 = v
                    [tpdm['abab_ovoo'], tpdm['abab_ovov']],  # Axis 2 = o (Axis 3 = o, v)
                    [tpdm['abab_ovvo'], tpdm['abab_ovvv']]   # Axis 2 = v (Axis 3 = o, v)
                ]
            ],
            [  # Axis 0 = v
                [  # Axis 1 = o
                    [tpdm['abab_vooo'], tpdm['abab_voov']],  # Axis 2 = o (Axis 3 = o, v)
                    [tpdm['abab_vovo'], tpdm['abab_vovv']]   # Axis 2 = v (Axis 3 = o, v)
                ],
                [  # Axis 1 = v
                    [tpdm['abab_vvoo'], tpdm['abab_vvov']],  # Axis 2 = o (Axis 3 = o, v)
                    [tpdm['abab_vvvo'], tpdm['abab_vvvv']]   # Axis 2 = v (Axis 3 = o, v)
                ]
            ]
        ])
 
        return tpdm_aaaa, tpdm_abab, tpdm_bbbb

    def get_parity(self, p):
        """
        Computes the parity of a permutation using inversion counting.
        Returns 1 for even permutations, -1 for odd permutations.
        """
        swaps = 0
        for i in range(len(p)):
            for j in range(i + 1, len(p)):
                if p[i] > p[j]:
                    swaps += 1
        return 1 if swaps % 2 == 0 else -1

    def antisymmetrize_tensor(self, tensor, raw_spaces, raw_spins):
        """
        Antisymmetrizes a dense tensor in-place, projecting it onto the exactly
        antisymmetric subspace (respecting left/right operator boundaries).
    
        Unlike the EOMCC version this is adapted from -- which unpacks a genuinely
        sparse (redundant-zero) representation, so summing permutations with sign
        reconstructs the full tensor with no double counting -- every element of
        `tensor` here is already populated (and only approximately antisymmetric,
        e.g. from residual/DIIS floating-point noise). Summing all permutations
        with sign therefore overcounts by prod(k!) over each permutable group of
        size k, so the result must be normalized by that factor to be a proper
        projector (idempotent, and a no-op on an already-exact input).
        """
        # 1. Identify groups of identical, permutable indices (unchanged from EOMCC)
        offset = 0
        groups = []
    
        for space_group, spin_group in zip(raw_spaces, raw_spins):
    
            partition_groups = {}
            for i, (sp, spn) in enumerate(zip(space_group, spin_group)):
                key = (sp, spn)
                if key not in partition_groups:
                    partition_groups[key] = []
                partition_groups[key].append(offset + i)
    
            for key, indices in partition_groups.items():
                if len(indices) > 1:
                    groups.append(indices)
    
            offset += len(space_group)
    
        # If there are no groups to permute (e.g. a rank-1 amplitude), we are done
        if not groups:
            return
    
        # 2. Generate all allowed permutations for each group, and the
        # normalization (product of each group's size factorial)
        group_perms = []
        norm = 1
        for g in groups:
            perms = [(p, self.get_parity(p)) for p in itertools.permutations(g)]
            group_perms.append(perms)
            norm *= math.factorial(len(g))
    
        # 3. Copy the (approximately antisymmetric) input, then accumulate the
        # properly antisymmetrized result back into the original array
        dense_tensor = tensor.copy()
        tensor.fill(0.0)
    
        for combined in itertools.product(*group_perms):
    
            axes = list(range(tensor.ndim))
            total_sign = 1
    
            for original_group, (permuted_indices, sign) in zip(groups, combined):
                for orig_idx, perm_idx in zip(original_group, permuted_indices):
                    axes[orig_idx] = perm_idx
                total_sign *= sign
    
            tensor += total_sign * dense_tensor.transpose(axes)
    
        tensor /= norm
    
    
    def antisymmetrize_amplitudes(self, amps, meta_dict):
        """
        Antisymmetrize every tensor in an amplitude dictionary (e.g. self.T or
        self.L) in place.
        :param amps: amplitude dict, keyed by base_name -> spin_key -> tensor,
                     as built by initialize_amplitudes
        :param meta_dict: meta-data dict (e.g. self.T_meta / self.L_meta) giving
                           the raw_spaces/raw_spins structure for each base_name,
                           as built by initialize_amplitudes
        """
        for base_name, meta in meta_dict.items():
            raw_spaces = meta['raw_spaces']
            if not raw_spaces:
                continue  # scalar / pure-photon amplitude -- nothing to antisymmetrize
    
            for raw_spins in meta['raw_spins']:
                spin_key = "".join(raw_spins) if isinstance(raw_spins, (list, tuple)) else raw_spins
                self.antisymmetrize_tensor(amps[base_name][spin_key], raw_spaces, raw_spins)
