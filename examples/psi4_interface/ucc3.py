# pdaggerq - A code for bringing strings of creation / annihilation operators to normal order.
# Copyright (C) 2020 A. Eugene DePrince III
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
spin-orbital UCC3 amplitude equations
"""

import numpy as np
from numpy import einsum

def ucc3_iterations(t1, t2, fock, g, o, v, e_ai, e_abij, hf_energy, max_iter=100, 
        e_convergence=1e-8,r_convergence=1e-8,diis_size=None, diis_start_cycle=4):
           
    # initialize diis if diis_size is not None
    # else normal scf iterate

    if diis_size is not None:
        from diis import DIIS
        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        t1_dim = t1.size
        old_vec = np.hstack((t1.flatten(), t2.flatten()))

    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)
    old_energy = ucc3_energy(t1, t2, fock, g, o, v)

    print("")
    print("    ==> UCC3 amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(max_iter):

        residual_singles = 1.0 * ucc3_singles_residual(t1, t2, fock, g, o, v)
        residual_doubles = ucc3_doubles_residual(t1, t2, fock, g, o, v)

        res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles)

        singles_res = 1.0 * (residual_singles + fock_e_ai * t1)
        doubles_res = residual_doubles + fock_e_abij * t2

        new_singles = singles_res * e_ai
        new_doubles = doubles_res * e_abij

        # diis update
        if diis_size is not None:
            vectorized_iterate = np.hstack(
                (new_singles.flatten(), new_doubles.flatten()))
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                 error_vec)
            new_singles = new_vectorized_iterate[:t1_dim].reshape(t1.shape)
            new_doubles = new_vectorized_iterate[t1_dim:].reshape(t2.shape)
            old_vec = new_vectorized_iterate

        current_energy = ucc3_energy(new_singles, new_doubles, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(idx, current_energy - hf_energy, delta_e, res_norm))
        if delta_e < e_convergence and res_norm < r_convergence:
            # assign t1 and t2 variables for future use before breaking
            t1 = new_singles
            t2 = new_doubles
            break
        else:
            # assign t1 and t2 and old_energy for next iteration
            t1 = new_singles
            t2 = new_doubles
            old_energy = current_energy

    else:
        raise ValueError("UCC3 iterations did not converge")


    return t1, t2



def ucc3_energy(t1, t2, f, g, o, v):

    #    < 0 | e(-T) H e(T) | 0> :
    
    #	  1.00 f(i,i)
    energy =  1.00 * einsum('ii', f[o, o])
    
    #	 -0.50 <j,i||j,i>
    energy += -0.50 * einsum('jiji', g[o, o, o, o])
    
    #	  1.00 f(i,a)*t1(a,i)
    energy +=  1.00 * einsum('ia,ai', f[o, v], t1)
    
    #	  1.00 f(a,i)*t1(a,i)
    energy +=  1.00 * einsum('ai,ai', f[v, o], t1)
    
    #	  0.250 <j,i||a,b>*t2(a,b,j,i)
    energy +=  0.250 * einsum('jiab,abji', g[o, o, v, v], t2)
    
    #	  0.250 <b,a||i,j>*t2(a,b,j,i)
    energy +=  0.250 * einsum('baij,abji', g[v, v, o, o], t2)
    
    #	 -0.50 f(a,i)*t2(b,a,i,j)*t1(b,j)
    energy += -0.50 * einsum('ai,baij,bj', f[v, o], t2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.50 f(i,a)*t2(a,b,j,i)*t1(b,j)
    energy += -0.50 * einsum('ia,abji,bj', f[o, v], t2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.50 f(j,i)*t2(b,a,i,k)*t2(a,b,k,j)
    energy += -0.50 * einsum('ji,baik,abkj', f[o, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.50 f(a,b)*t2(c,a,i,j)*t2(b,c,j,i)
    energy +=  0.50 * einsum('ab,caij,bcji', f[v, v], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.50 <k,i||j,i>*t2(b,a,j,l)*t2(a,b,l,k)
    energy +=  0.50 * einsum('kiji,bajl,ablk', g[o, o, o, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.50 <j,l||i,l>*t2(b,a,i,k)*t2(a,b,k,j)
    energy += -0.50 * einsum('jlil,baik,abkj', g[o, o, o, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.1250 <l,k||i,j>*t2(b,a,j,i)*t2(a,b,l,k)
    energy +=  0.1250 * einsum('lkij,baji,ablk', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.00 <j,a||b,i>*t2(c,a,i,k)*t2(b,c,k,j)
    energy +=  1.00 * einsum('jabi,caik,bckj', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||c,d>*t2(a,b,i,j)*t2(c,d,j,i)
    energy +=  0.1250 * einsum('bacd,abij,cdji', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    return energy

def ucc3_singles_residual(t1, t2, f, g, o, v):

    #    < 0 | m* e e(-T) H e(T) | 0> :
    
    #	  1.00 f(e,m)
    singles_res =  1.00 * einsum('em->em', f[v, o])
    
    #	 -1.00 f(i,a)*t2(a,e,m,i)
    singles_res += -1.00 * einsum('ia,aemi->em', f[o, v], t2)
    
    return singles_res


def ucc3_doubles_residual(t1, t2, f, g, o, v):

    #    < 0 | m* n* f e e(-T) H e(T) | 0> :
    
    #	  1.00 <e,f||m,n>
    doubles_res =  1.00 * einsum('efmn->efmn', g[v, v, o, o])
    
    #	 -1.00 P(m,n)f(i,n)*t2(e,f,m,i)
    contracted_intermediate = -1.00 * einsum('in,efmi->efmn', f[o, o], t2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.00 P(e,f)f(e,a)*t2(a,f,m,n)
    contracted_intermediate =  1.00 * einsum('ea,afmn->efmn', f[v, v], t2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  0.50 <j,i||m,n>*t2(e,f,j,i)
    doubles_res +=  0.50 * einsum('jimn,efji->efmn', g[o, o, o, o], t2)
    
    #	  1.00 P(m,n)*P(e,f)<i,e||a,n>*t2(a,f,m,i)
    contracted_intermediate =  1.00 * einsum('iean,afmi->efmn', g[o, v, v, o], t2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  0.50 <e,f||a,b>*t2(a,b,m,n)
    doubles_res +=  0.50 * einsum('efab,abmn->efmn', g[v, v, v, v], t2)
    
    return doubles_res
    
