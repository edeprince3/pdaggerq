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
eom-cc sigma vectors Hbar.R and L.Hbar
"""

import numpy as np
from numpy import einsum

import scipy
from scipy.sparse.linalg import LinearOperator

import types

import itertools

class HbarOperator:
    """
    Hbar as a LinearOperator, with spin-traced expressions
    """

    def __init__(self, 
                 cc,
                 R_list = [], 
                 L_list = []):
        """
        initialize HBarOperator

        :param cc: cc object
        :params R_list: list of R operator dictionaries
        :params L_list: list of L operator dictionaries
        """

        self.cc = cc

        # right-hand amplitudes
        self.R = {}
        self.R_sigma = {}
        self.R_meta = {}
        self.initialize_amplitudes(R_list, self.R, self.R_sigma, self.R_meta)

        # left-hand amplitudes
        self.L = {}
        self.L_sigma = {}
        self.L_meta = {}
        self.initialize_amplitudes(L_list, self.L, self.L_sigma, self.L_meta)

    @property
    def right_amplitude_size(self):
        """ 
        Calculates the total number of unique, symmetry-allowed scalar elements 
        across all R amplitude tensors.
        """ 
        total_size = 0
        
        for base_name, meta in self.R_meta.items():
            raw_spaces = meta['raw_spaces']
            
            for raw_spins in meta['raw_spins']:
                spin_key = "".join(raw_spins)
                tensor = self.R[base_name][spin_key]
                
                # Generate the boolean mask for this tensor block
                mask = self.get_unique_mask(raw_spaces, raw_spins, tensor.shape)
                
                # Count the number of True values (unique elements)
                total_size += np.count_nonzero(mask)
                
        return total_size

    @property
    def left_amplitude_size(self):
        """ 
        Calculates the total number of unique, symmetry-allowed scalar elements 
        across all L amplitude tensors.
        """ 
        total_size = 0
        
        for base_name, meta in self.L_meta.items():
            raw_spaces = meta['raw_spaces']
            
            for raw_spins in meta['raw_spins']:
                spin_key = "".join(raw_spins)
                tensor = self.L[base_name][spin_key]
                
                # Generate the boolean mask for this tensor block
                mask = self.get_unique_mask(raw_spaces, raw_spins, tensor.shape)
                
                # Count the number of True values (unique elements)
                total_size += np.count_nonzero(mask)
                
        return total_size

    def initialize_amplitudes(self, R_list, R, R_sigma, R_meta, function = 'sigma'):
        """
        Initialize right- or left-hand EOMCC amplitude dictionaries
        :param R_list: list of amplitude dictionaries containing spaces / spins / sigma function
        :param R: amplitude dictionary
        :param R_sigma: sigma-vector function dictionary
        :param R_meta: meta-data dictionary for left/right space/spin information
        :param function: the function in the R_list element that we wish to initialize
        """

        dims = {
            'va': self.cc.nva,
            'oa': self.cc.noa,
            'vb': self.cc.nvb,
            'ob': self.cc.nob
        } 

        for myR in R_list:

            # [v,o], [vv,oo], etc.
            raw_spaces = myR.get('spaces', [])

            if raw_spaces and len(raw_spaces) != 2:
                raise Exception("R_list spaces should have exactly two elements (left/right)")
            full_spaces = "".join(raw_spaces)

            # Amplitude order (e.g., R1 -> 1, R2 -> 2)
            if raw_spaces:
                order = max(len(raw_spaces[0]), len(raw_spaces[1]))
            else:
                order = 0

            # Number of photons
            nph = myR.get('nph', 0)

            # Base key for this rank (e.g., '1', '2', '0_1p' for 1 photon, '1_1p', etc.)
            base_name = str(order)
            if nph > 0:
                base_name += '_' + str(nph) + 'p'

            # Initialize nested dictionaries for this rank
            R[base_name] = {}

            # Bind sigma-build function to instance (or xi or eta functions for cc response)
            R_sigma[base_name] = types.MethodType(myR[function], self)

            # If no spins are provided (like for r0 or pure photons), 
            # we provide a dummy list [[]] so the packer loops exactly once.
            raw_spins = myR.get('spins', [])
            if not raw_spins:
                raw_spins = [[]]

            # Store the exact structural boundaries for the solver
            R_meta[base_name] = {
                'raw_spaces': raw_spaces,
                'raw_spins': raw_spins
            }

            if 'spins' in myR and len(full_spaces) > 0:
                for raw_spins in myR['spins']:

                    if isinstance(raw_spins, (list, tuple)):
                        full_spins = "".join(raw_spins)
                    else:
                        full_spins = raw_spins

                    shape = tuple(dims[space + spin] for space, spin in zip(full_spaces, full_spins))
                    R[base_name][full_spins] = np.zeros(shape, dtype=np.float64)
            else:
                R[base_name][''] = np.zeros((1,), dtype=np.float64)

    def get_unique_mask(self, raw_spaces, raw_spins, shape):
        """
        Generates a boolean mask for the unique elements of a tensor, 
        isolating the left and right operator spaces.
        
        Example: 
            raw_spaces = ['vv', 'oo'], raw_spins = ['aa', 'aa']
            Enforces: idx_0 < idx_1 (left) AND idx_2 < idx_3 (right)
        """
        mask = np.ones(shape, dtype=bool)
        grid = np.indices(shape)
        
        offset = 0
        # Loop over the left (creation) and right (annihilation) partitions
        for space_group, spin_group in zip(raw_spaces, raw_spins):
            
            # Enforce idx_n < idx_{n+1} only for identical fermions within this partition
            for i in range(len(space_group) - 1):
                if space_group[i] == space_group[i+1] and spin_group[i] == spin_group[i+1]:
                    mask &= (grid[offset + i] < grid[offset + i + 1])
                    
            # Advance the offset to the next partition (e.g., move past the left indices)
            offset += len(space_group)
            
        return mask

    def pack_eom_vectors(self, amps, meta_dict):
        """
        Flattens the unique, symmetry-allowed elements of the EOM amplitudes.
        """
        vec_list = []
        
        for base_name, meta in meta_dict.items():
            raw_spaces = meta['raw_spaces']
            
            for raw_spins in meta['raw_spins']:
                # Convert [['aa', 'aa']] -> 'aaaa' to access the tensor
                spin_key = "".join(raw_spins)
                tensor = amps[base_name][spin_key]
                
                mask = self.get_unique_mask(raw_spaces, raw_spins, tensor.shape)
                vec_list.append(tensor[mask])
                
        return np.concatenate(vec_list)

    def unpack_eom_vectors(self, flat_vec, amps, meta_dict):
        """
        Unpacks a 1D Davidson vector back into the target amplitude tensors.
        """
        idx = 0
        
        for base_name, meta in meta_dict.items():
            raw_spaces = meta['raw_spaces']
            
            for raw_spins in meta['raw_spins']:
                spin_key = "".join(raw_spins)
                current_tensor = amps[base_name][spin_key]
                
                mask = self.get_unique_mask(raw_spaces, raw_spins, current_tensor.shape)
                n_elements = np.count_nonzero(mask)
                
                # Zero the tensor, then fill the unique elements
                current_tensor.fill(0.0)
                current_tensor[mask] = flat_vec[idx:idx + n_elements]
                
                idx += n_elements
                
                # Antisymmetrize
                self.antisymmetrize_tensor(current_tensor, raw_spaces, raw_spins)

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
        Antisymmetrizes a sparse tensor in-place, strictly respecting the 
        boundaries between left/right operators.
        """
        # 1. Identify groups of identical, permutable indices
        offset = 0
        groups = []
        
        for space_group, spin_group in zip(raw_spaces, raw_spins):
            
            # Group indices by (space, spin) within this partition
            partition_groups = {}
            for i, (sp, spn) in enumerate(zip(space_group, spin_group)):
                key = (sp, spn)
                if key not in partition_groups:
                    partition_groups[key] = []
                partition_groups[key].append(offset + i)
            
            # We only care about groups with 2 or more identical indices
            for key, indices in partition_groups.items():
                if len(indices) > 1:
                    groups.append(indices)
                    
            # Advance the offset past this partition
            offset += len(space_group)
            
        # If there are no groups to permute (e.g., R1 'v', 'o'), we are done
        if not groups:
            return
    
        # 2. Generate all allowed permutations for each group
        group_perms = []
        for g in groups:
            # Store tuples of (permuted_indices, parity_sign)
            perms = [(p, self.get_parity(p)) for p in itertools.permutations(g)]
            group_perms.append(perms)
    
        # 3. Create a sparse copy of the tensor, then zero out the original
        sparse_tensor = tensor.copy()
        tensor.fill(0.0)
    
        # 4. Cartesian product of all permutations across all groups
        for combined in itertools.product(*group_perms):
            # 'combined' is one specific permutation state for the whole tensor
            
            axes = list(range(tensor.ndim))
            total_sign = 1
            
            # Apply the permutation mappings and multiply the signs
            for original_group, (permuted_indices, sign) in zip(groups, combined):
                for orig_idx, perm_idx in zip(original_group, permuted_indices):
                    axes[orig_idx] = perm_idx
                total_sign *= sign
                
            # Transpose the sparse tensor and accumulate it directly into the result!
            tensor += total_sign * sparse_tensor.transpose(axes)

    def matvec_right(self, R):
        """
        evaluate the action of Hbar on a vector, sigma = H.R

        :param R: the vector (flat, unique elements only)

        :return sigma: the sigma vector (flat, unique elements only)
        """

        # Unpack amplitudes
        self.unpack_eom_vectors(R, self.R, self.R_meta)

        # Loop through each excitation rank and evaluate sigma vectors
        sigmas = {}
        for base_name, sigma_func in self.R_sigma.items():
        
            # Evaluate sigma vector tensors
            sigma_dict = sigma_func()[base_name]
            
            # Store spin-channel dict in sigma
            sigmas[base_name] = sigma_dict

        # Repack sigma vector
        sigma = self.pack_eom_vectors(sigmas, self.R_meta)

        return sigma - self.cc.energy * R

    def matvec_left(self, L):
        """
        evaluate the action of Hbar on a vector, sigma = L.H

        :param L: the vector (flat, unique elements only)

        :return sigma: the sigma vector (flat, unique elements only)
        """

        # Unpack amplitudes
        self.unpack_eom_vectors(L, self.L, self.L_meta)

        # Loop through each excitation rank and evaluate sigma vectors
        sigmas = {}
        for base_name, sigma_func in self.L_sigma.items():
        
            # Evaluate sigma vector tensors
            sigma_dict = sigma_func()[base_name]
            
            # Store spin-channel dict in sigma
            sigmas[base_name] = sigma_dict

        # Repack sigma vector
        sigma = self.pack_eom_vectors(sigmas, self.L_meta)

        return sigma - self.cc.energy * L
