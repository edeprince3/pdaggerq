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
eom-cc response theory
"""

import types

import numpy as np
from numpy import einsum

import scipy
from scipy.sparse.linalg import LinearOperator

import copy

from pdaggerq.numerical.solvers.cc_hbar import HbarOperator
from pdaggerq.numerical.utils.integrals import get_dipole_integrals_with_spin
from pdaggerq.numerical.utils.diis import DIIS

class cc_response:

    def __init__(self, cc, R_list = [], perturb = 'dipole'):
        """
        Initialize CC response class

        :params cc: the cc class
        :params R_list: list of R operator dictionaries
        :params perturb: the perturbation type

        """

        self.cc = cc
        self.perturb = perturb

        if R_list is not None:
            self.R_list = R_list

        if self.perturb == 'dipole':
            self.V_aa, self.V_bb = get_dipole_integrals_with_spin(self.cc.wfn, nfzc = self.cc.nfzc)
        else:
            raise Exception("invalid perturbing operator")

        # build Hbar operator object
        self.Hbar = HbarOperator(self.cc, R_list = self.R_list)

        # initialize positive-frequency right-hand response vectors
        self.Hbar.R = {}
        self.Hbar.R_sigma = {}
        self.Hbar.R_meta = {}
        self.Hbar.initialize_amplitudes(self.R_list, self.Hbar.R, self.Hbar.R_sigma, self.Hbar.R_meta)

        # initialize xi_mu = <mu| Vbar |0> 
        self.xi = []
        for i in range (3):

            # dipole integrals in i-direction
            self.Hbar.h_aa = self.V_aa[i]
            self.Hbar.h_bb = self.V_bb[i]

            # initialize xi_mu = <mu| Vbar |0>
            self.Hbar.xi = {}
            self.Hbar.xi_func = {}
            self.Hbar.xi_meta = {}
            self.Hbar.initialize_amplitudes(self.R_list, self.Hbar.xi, self.Hbar.xi_func, self.Hbar.xi_meta, function = 'xi')

            # Evaluate xi tensors
            xi = {}
            for base_name, xi_func in self.Hbar.xi_func.items():
                xi_dict = xi_func()[base_name]
                xi[base_name] = xi_dict
                    
            # Flatten xi and add to the CC response class
            self.xi.append(self.Hbar.pack_eom_vectors(xi, self.Hbar.xi_meta))

    def polarizability(self, hessian_func = None):
        """
        Evaluate cc polarizality

        :params hessian_func: the function to evaluate the hessian contribution to the polarizability
        """

        # hessian contributionto the polarizability
        self.Hbar.hessian = types.MethodType(hessian_func, self.Hbar)

        # initialize eta_nu = <0|(1+Lambda)[Vbar,tau_nu]|0> 
        self.eta = []
        for i in range (3):

            # dipole integrals in i-direction
            self.Hbar.h_aa = self.V_aa[i]
            self.Hbar.h_bb = self.V_bb[i]

            # eta_nu = <0|(1+Lambda)[Vbar,tau_nu]|0> 
            self.Hbar.eta = {}
            self.Hbar.eta_func = {}
            self.Hbar.eta_meta = {}
            self.Hbar.initialize_amplitudes(self.R_list, self.Hbar.eta, self.Hbar.eta_func, self.Hbar.eta_meta, function = 'eta')

            # Evaluate eta tensors
            eta = {}
            for base_name, eta_func in self.Hbar.eta_func.items():
                eta_dict = eta_func()[base_name]
                eta[base_name] = eta_dict
                    
            # Flatten eta and add to the CC response class
            self.eta.append(self.Hbar.pack_eom_vectors(eta, self.Hbar.eta_meta))

        dirs = ['x', 'y', 'z']
        alpha = np.zeros((3,3))
        for i in range (3):
            flat_right_i = self.Hbar.pack_eom_vectors(self.right_response[i], self.Hbar.R_meta)
            flat_right_neg_i = self.Hbar.pack_eom_vectors(self.right_neg_response[i], self.Hbar.R_meta)

            for j in range (3):
                flat_right_j = self.Hbar.pack_eom_vectors(self.right_response[j], self.Hbar.R_meta)
                flat_right_neg_j = self.Hbar.pack_eom_vectors(self.right_neg_response[j], self.Hbar.R_meta)

                # eta.t(w)
                val = np.dot(flat_right_i, self.eta[j])
                val += np.dot(flat_right_j, self.eta[i])

                # eta.t(-w)
                val += np.dot(flat_right_neg_i, self.eta[j])
                val += np.dot(flat_right_neg_j, self.eta[i])

                self.Hbar.R = self.right_response[i]
                self.Hbar.R_neg = self.right_neg_response[j]
                hessian = 0.5 * self.Hbar.hessian()

                alpha[i, j] = -0.5*val -0.5*hessian
                print('    alpha(%s,%s) = %20.12f' % (dirs[i], dirs[j], alpha[i, j]))

        print('')

        return alpha

    def first_order_response_solver(self, omega=0.0, r_convergence = 1e-8):
        """
        Solve CC first-order response equations
        
        :param omega: perturbing frequency
        """
        print('')
        print('    ==> CC response properties <==')
        print('')
        print('    omega        = %20.12f' % (omega))
        print('    perturbation = %20s' % (self.perturb))
        print('')
    
        self.right_response = []
        self.right_neg_response = []
        
        for i in range(3):
    
            # Solve right-hand response equations at +omega
            R = self.right_response_iterations(
                self.xi[i], f"{self.perturb}[{i}]", omega=omega, r_convergence = r_convergence
            )
    
            Rneg = R
            if np.abs(omega) > 1e-12:
                # Solve right-hand response equations at -omega
                Rneg = self.right_response_iterations(
                    self.xi[i], f"{self.perturb}[{i}]", omega=-omega, r_convergence = r_convergence
                )
    
            # Unpack and store +omega response
            self.Hbar.unpack_eom_vectors(R, self.Hbar.R, self.Hbar.R_meta)
            self.right_response.append(copy.deepcopy(self.Hbar.R))
    
            # Unpack and store -omega response
            self.Hbar.unpack_eom_vectors(Rneg, self.Hbar.R, self.Hbar.R_meta)
            self.right_neg_response.append(copy.deepcopy(self.Hbar.R))
    
        print('')

    def right_response_iterations(self, xi, perturb, omega=0.0, max_iter=500, r_convergence=1e-8, diis_size=8, diis_start_cycle=2):
    
        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
    
        # Preconditioner 1 / (diag(H) - omega)

        # Check if response operator structure matches ground-state denominator structure
        d_keys = set(self.cc.D.keys())
        r_keys = set(self.Hbar.R_meta.keys())
        
        if not r_keys.issubset(d_keys):
            raise NotImplementedError(
                f"CC response theory is currently only supported for particle-conserving (EE) methods. "
                f"Response operator sectors {list(r_keys)} do not match ground-state denominator sectors {list(d_keys)}."
            )

        Hdiag = - 1.0 / self.Hbar.pack_eom_vectors(self.cc.D, self.Hbar.R_meta)
        precon = 1.0 / ( Hdiag - omega)
    
        print('')
        print('    ==> CC right-hand first-order response equations <==')
        print('')
        print('    omega        = %20.12f' % (omega))
        print('    perturbation = %20s' % (perturb))
        print('')
        print('     Iter          Pseudoalpha              |dT(1)|')
    
        oldR = np.zeros_like(xi)
    
        for idx in range(max_iter):
    
            damp = 0.0
            if idx > 10:
                damp = 0.0
    
            # Matrix-vector product \bar{H} * R_old
            HbarR = self.Hbar.matvec_right(oldR)
    
            # Explicit residual: r = -\xi - \bar{H} R_old + \omega R_old
            residual = -xi - HbarR + omega * oldR
    
            # Jacobi update step: R_new = R_old + precon * residual
            R_step = oldR + precon * residual
            R = damp * oldR + (1.0 - damp) * R_step
    
            # Error vector for DIIS and convergence
            dR = R - oldR
    
            # DIIS extrapolation
            oldR = diis_update.compute_new_vec(R, dR)
            R = oldR
    
            nrm = np.linalg.norm(dR)
            print("     {: 5d} {: 20.12f} {: 20.12f}".format(idx, -2 * np.dot(R, xi), nrm))
            
            if nrm < r_convergence:
                break
    
        else:
            raise ValueError("CC right-hand first-order response iterations did not converge")

        return R
