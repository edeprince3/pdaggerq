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

from pdaggerq.numerical_utils.integrals import get_integrals
from pdaggerq.numerical_utils.integrals import get_integrals_with_spin

from pdaggerq.numerical_utils.diis import DIIS

class cc:

    def __init__(self, wfn, 
        mol, 
        nfzc = 0, 
        is_qed = False, 
        cc_energy_func = None, 
        T_list = [],
        L_list = [],
        cc_pseudoenergy_func=None, 
        cavity_frequency=1.0,
        cavity_lambda=0.0):

        """
        initialize CC class

        :params wfn: a psi4 wave function
        :params mol: a psi4 molecule object
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

        # qed-cc
        self.cavity_frequency = cavity_frequency
        self.cavity_lambda = np.array(cavity_lambda)
        self.is_qed = is_qed

        if self.is_qed and self.nfzc > 0:
            raise Exception("QED-CC does not work with frozen core")

        noa, nob, nva, nvb, self.f_aa, self.f_bb, self.g_aaaa, self.g_bbbb, self.g_abab, self.efzc  = get_integrals_with_spin(self.wfn, nfzc = self.nfzc)

        # occupied, virtual slices
        n = np.newaxis
        oa = slice(None, noa)
        ob = slice(None, nob)
        va = slice(noa, None)
        vb = slice(nob, None)
        self.slices = {
            'oa': oa,
            'va': va,
            'ob': ob,
            'vb': vb
        }

        # DSE and bilinear coupling contributions for QED
        if not self.is_qed:
            self.enuc_dse = 0.0
        else:
            from pdaggerq.numerical_utils.integrals import get_dipole_integrals_with_spin
            tmp_dipole_aa, tmp_dipole_bb = get_dipole_integrals_with_spin(self.wfn)
                             
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
            from pdaggerq.numerical_utils.integrals import get_quadrupole_integrals_with_spin
            tmp_quadrupole_aa, tmp_quadrupole_bb = get_quadrupole_integrals_with_spin(self.wfn)

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
            nuc_dip_x = 0.0;
            nuc_dip_y = 0.0;
            nuc_dip_z = 0.0;
            for i in range (mol.natom()):
                nuc_dip_x += mol.Z(i) * mol.x(i);
                nuc_dip_y += mol.Z(i) * mol.y(i);
                nuc_dip_z += mol.Z(i) * mol.z(i);
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

        # cluster amplitudes
        dims = {
            'va': nva,
            'oa': noa,
            'vb': nvb,
            'ob': nob
        }

        self.T = {}
        self.T_residual = {}
        self.D = {}
        
        for myT in T_list:
        
            # vo, vvoo, etc.
            spaces = myT.get('spaces', '')
        
            # Amplitude order (e.g., T1 -> 1, T2 -> 2)
            order = len(spaces) // 2
        
            # Number of photons
            nph = myT.get('nph', 0)
        
            # Base key for this rank (e.g., '1', '2', '0_1p' for 1 photon, '1_1p', etc.)
            base_name = str(order) 
            if nph > 0:
                base_name += '_' + str(nph) + 'p'
        
            # Initialize nested dictionaries for this rank
            self.T[base_name] = {}
            self.D[base_name] = {}
        
            # Bind residual method to instance
            self.T_residual[base_name] = types.MethodType(myT['residual'], self)
        
            # Fermionic amplitudes (has spins e.g. 'aa', 'abab')
            if 'spins' in myT and len(spaces) > 0:
                rank = len(spaces)
        
                for spins in myT['spins']:
                    shape = tuple(dims[space + spin] for space, spin in zip(spaces, spins))
                    
                    # Initialize denominator accumulator for this spin block
                    denom = np.zeros(shape)
        
                    # Accumulate orbital energy differences
                    for i, (space, spin) in enumerate(zip(spaces, spins)):
                        sign = 1.0 if space == 'o' else -1.0
                        eps_1d = self.eps[spin][self.slices[space + spin]]
        
                        b_idx = [n] * rank
                        b_idx[i] = slice(None)
        
                        denom += sign * eps_1d[tuple(b_idx)]
        
                    # Subtract photon energy if coupled to photons
                    if nph > 0:
                        denom -= nph * self.cavity_frequency
        
                    # Store tensors in nested dicts
                    self.T[base_name][spins] = np.zeros(shape)
                    self.D[base_name][spins] = 1.0 / denom
        
            # Pure photon / zero-fermion amplitudes
            else:
                if nph > 0:
                    # 1-element 1D array for scalar photon amplitudes
                    self.T[base_name][''] = np.zeros((1,))
                    # Energy shift is strictly -nph * omega
                    self.D[base_name][''] = np.array([-1.0 / (nph * self.cavity_frequency)])

        # lambda amplitudes
        self.L = {}
        self.L_residual = {}
        for myL in L_list:
        
            # vo, vvoo, etc.
            spaces = myL.get('spaces', '')
        
            # Amplitude order (e.g., L1 -> 1, L2 -> 2)
            order = len(spaces) // 2
        
            # Number of photons
            nph = myL.get('nph', 0)
        
            # Base key for this rank (e.g., '1', '2', '0_1p' for 1 photon, '1_1p', etc.)
            base_name = str(order) 
            if nph > 0:
                base_name += '_' + str(nph) + 'p'
        
            # Initialize nested dictionaries for this rank
            self.L[base_name] = {}
        
            # Bind residual method to instance
            self.L_residual[base_name] = types.MethodType(myL['residual'], self)
        
            # Fermionic amplitudes (has spins e.g. 'aa', 'abab')
            if 'spins' in myL and len(spaces) > 0:
                rank = len(spaces)
        
                for spins in myL['spins']:
                    shape = tuple(dims[space + spin] for space, spin in zip(spaces, spins))
                    
                    # Store tensors in nested dicts
                    self.L[base_name][spins] = np.zeros(shape)
        
            # Pure photon / zero-fermion amplitudes
            else:
                if nph > 0:
                    # 1-element 1D array for scalar photon amplitudes
                    self.L[base_name][''] = np.zeros((1,))

        # hartree-fock energy
        self.hf_energy = ( einsum('ii', self.f_aa[oa, oa]) + einsum('ii', self.f_bb[ob, ob])
                       - 0.5 * einsum('ijij', self.g_aaaa[oa, oa, oa, oa])
                       - 0.5 * einsum('ijij', self.g_bbbb[ob, ob, ob, ob])
                       - 1.0 * einsum('ijij', self.g_abab[oa, ob, oa, ob]) )

        self.noa = noa
        self.nob = nob
        self.nva = nva
        self.nvb = nvb
        self.oa = oa
        self.ob = ob
        self.va = va
        self.vb = vb
   
        self.nuclear_repulsion_energy = self.mol.nuclear_repulsion_energy()

        # cc energy function
        self.cc_energy = types.MethodType(cc_energy_func, self)

        # lambda CC pseudoenergy function
        if cc_pseudoenergy_func is not None:
            self.cc_pseudoenergy = types.MethodType(cc_pseudoenergy_func, self)

    def t_solver(self):
        """
    
        run ccsd amplitude equations
    
        :return energy: the total ccsd energy
    
        """

        self.cc_iterations_with_spin(e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4)
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

        self.cc_iterations_with_spin(e_convergence=1e-10, r_convergence=1e-10, diis_size=8, diis_start_cycle=4, is_lambda = True)
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
    
