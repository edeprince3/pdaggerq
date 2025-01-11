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
spin-orbital QUCCSD amplitude equations
"""

import numpy as np
from numpy import einsum

def quccsd_iterations(t1, t2, fock, g, o, v, e_ai, e_abij, hf_energy, max_iter=100, 
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
    old_energy = quccsd_energy(t1, t2, fock, g, o, v)

    print("")
    print("    ==> QUCCSD amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(max_iter):

        residual_singles = 1.0 * quccsd_singles_residual(t1, t2, fock, g, o, v)
        residual_doubles = quccsd_doubles_residual(t1, t2, fock, g, o, v)

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

        current_energy = quccsd_energy(new_singles, new_doubles, fock, g, o, v)
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
        raise ValueError("QUCCSD iterations did not converge")


    return t1, t2



    
def quccsd_energy(t1, t2, f, g, o, v):
    
    #    < 0 | e(-T) H e(T) | 0> :
    
    #	  1.00 f(i,i)
    energy =  1.00 * einsum('ii', f[o, o])
    
    #	  1.00 f(i,a)*t1(a,i)
    energy +=  1.00 * einsum('ia,ai', f[o, v], t1)
    
    #	  1.00 f(a,i)*t1(a,i)
    energy +=  1.00 * einsum('ai,ai', f[v, o], t1)
    
    #	 -0.50 <j,i||j,i>
    energy += -0.50 * einsum('jiji', g[o, o, o, o])
    
    #	  0.1250 <j,i||a,b>*t2(a,b,j,i)
    energy +=  0.1250 * einsum('jiab,abji', g[o, o, v, v], t2)
    
    #	  0.1250 <b,a||i,j>*t2(a,b,j,i)
    energy +=  0.1250 * einsum('baij,abji', g[v, v, o, o], t2)
    
    #	 -0.083333333333333 <j,i||a,b>*t1(a,i)*t1(b,j)
    energy += -0.083333333333333 * einsum('jiab,ai,bj', g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.083333333333333 <b,a||i,j>*t1(a,i)*t1(b,j)
    energy += -0.083333333333333 * einsum('baij,ai,bj', g[v, v, o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.3333333333333 <k,j||a,i>*t1(a,j)*t1(b,i)*t1(b,k)
    energy += -0.3333333333333 * einsum('kjai,aj,bi,bk', g[o, o, v, o], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.3333333333333 <i,a||b,c>*t1(a,j)*t1(b,i)*t1(c,j)
    energy += -0.3333333333333 * einsum('iabc,aj,bi,cj', g[o, v, v, v], t1, t1, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.3333333333333 <k,a||i,j>*t1(a,j)*t1(b,i)*t1(b,k)
    energy +=  0.3333333333333 * einsum('kaij,aj,bi,bk', g[o, v, o, o], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.3333333333333 <b,a||c,i>*t1(a,j)*t1(b,i)*t1(c,j)
    energy +=  0.3333333333333 * einsum('baci,aj,bi,cj', g[v, v, v, o], t1, t1, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.0416666666666667 <k,j||a,b>*t1(a,i)*t2(b,c,k,j)*t1(c,i)
    energy += -0.0416666666666667 * einsum('kjab,ai,bckj,ci', g[o, o, v, v], t1, t2, t1, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.0416666666666667 <j,i||b,c>*t1(a,i)*t1(a,k)*t2(b,c,k,j)
    energy += -0.0416666666666667 * einsum('jibc,ai,ak,bckj', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	  0.083333333333333 <l,k||i,j>*t2(a,b,l,k)*t1(a,j)*t1(b,i)
    energy +=  0.083333333333333 * einsum('lkij,ablk,aj,bi', g[o, o, o, o], t2, t1, t1, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3333333333333 <j,a||b,i>*t1(a,k)*t2(b,c,k,j)*t1(c,i)
    energy += -0.3333333333333 * einsum('jabi,ak,bckj,ci', g[o, v, v, o], t1, t2, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.083333333333333 <b,a||c,d>*t1(a,j)*t1(b,i)*t2(c,d,j,i)
    energy +=  0.083333333333333 * einsum('bacd,aj,bi,cdji', g[v, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.083333333333333 <l,k||i,j>*t2(b,a,j,i)*t1(a,k)*t1(b,l)
    energy +=  0.083333333333333 * einsum('lkij,baji,ak,bl', g[o, o, o, o], t2, t1, t1, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.3333333333333 <k,a||b,i>*t2(c,a,i,j)*t1(b,j)*t1(c,k)
    energy += -0.3333333333333 * einsum('kabi,caij,bj,ck', g[o, v, v, o], t2, t1, t1, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  0.083333333333333 <b,a||c,d>*t2(a,b,i,j)*t1(c,i)*t1(d,j)
    energy +=  0.083333333333333 * einsum('bacd,abij,ci,dj', g[v, v, v, v], t2, t1, t1, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.083333333333333 <b,a||i,j>*t2(c,a,j,i)*t1(b,k)*t1(c,k)
    energy += -0.083333333333333 * einsum('baij,caji,bk,ck', g[v, v, o, o], t2, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.083333333333333 <b,a||i,j>*t2(a,b,i,k)*t1(c,j)*t1(c,k)
    energy += -0.083333333333333 * einsum('baij,abik,cj,ck', g[v, v, o, o], t2, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.0416666666666667 <k,i||a,i>*t1(a,j)*t2(c,b,j,l)*t2(b,c,l,k)
    energy += -0.0416666666666667 * einsum('kiai,aj,cbjl,bclk', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    
    #	 -0.083333333333333 <k,j||a,i>*t1(a,j)*t2(c,b,i,l)*t2(b,c,l,k)
    energy += -0.083333333333333 * einsum('kjai,aj,cbil,bclk', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.083333333333333 <l,k||a,i>*t1(a,j)*t2(c,b,i,j)*t2(b,c,l,k)
    energy += -0.083333333333333 * einsum('lkai,aj,cbij,bclk', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    
    #	  0.16666666666667 <k,j||b,i>*t2(c,a,i,l)*t1(a,j)*t2(b,c,l,k)
    energy +=  0.16666666666667 * einsum('kjbi,cail,aj,bclk', g[o, o, v, o], t2, t1, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.083333333333333 <i,a||b,c>*t2(d,a,j,k)*t1(b,i)*t2(c,d,k,j)
    energy += -0.083333333333333 * einsum('iabc,dajk,bi,cdkj', g[o, v, v, v], t2, t1, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.16666666666667 <j,a||b,c>*t2(d,a,i,k)*t1(b,i)*t2(c,d,k,j)
    energy +=  0.16666666666667 * einsum('jabc,daik,bi,cdkj', g[o, v, v, v], t2, t1, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.083333333333333 <i,a||c,d>*t2(b,a,j,k)*t1(b,i)*t2(c,d,k,j)
    energy += -0.083333333333333 * einsum('iacd,bajk,bi,cdkj', g[o, v, v, v], t2, t1, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    #	 -0.083333333333333 <a,i||j,i>*t1(a,k)*t2(c,b,j,l)*t2(b,c,l,k)
    energy += -0.083333333333333 * einsum('aiji,ak,cbjl,bclk', g[v, o, o, o], t1, t2, t2, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    
    #	 -0.083333333333333 <l,a||i,l>*t1(a,j)*t2(c,b,i,k)*t2(b,c,k,j)
    energy += -0.083333333333333 * einsum('lail,aj,cbik,bckj', g[o, v, o, o], t1, t2, t2, optimize=['einsum_path', (2, 3), (1, 2), (0, 1)])
    
    #	  0.16666666666667 <k,a||i,j>*t1(a,j)*t2(c,b,i,l)*t2(b,c,l,k)
    energy +=  0.16666666666667 * einsum('kaij,aj,cbil,bclk', g[o, v, o, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.083333333333333 <k,a||i,j>*t1(a,l)*t2(c,b,j,i)*t2(b,c,l,k)
    energy +=  0.083333333333333 * einsum('kaij,al,cbji,bclk', g[o, v, o, o], t1, t2, t2, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    
    #	 -0.250 <k,a||i,j>*t2(c,a,i,l)*t2(b,c,l,k)*t1(b,j)
    energy += -0.250 * einsum('kaij,cail,bclk,bj', g[o, v, o, o], t2, t2, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.16666666666667 <b,a||c,i>*t2(d,a,j,k)*t1(b,i)*t2(c,d,k,j)
    energy +=  0.16666666666667 * einsum('baci,dajk,bi,cdkj', g[v, v, v, o], t2, t1, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.3333333333333 <b,a||c,i>*t2(d,a,i,k)*t1(b,j)*t2(c,d,k,j)
    energy += -0.3333333333333 * einsum('baci,daik,bj,cdkj', g[v, v, v, o], t2, t1, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.083333333333333 <b,a||c,i>*t2(a,b,j,k)*t2(c,d,k,j)*t1(d,i)
    energy +=  0.083333333333333 * einsum('baci,abjk,cdkj,di', g[v, v, v, o], t2, t2, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.0416666666666667 <k,i||a,b>*t2(a,b,j,i)*t1(c,j)*t1(c,k)
    energy +=  0.0416666666666667 * einsum('kiab,abji,cj,ck', g[o, o, v, v], t2, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.0416666666666667 <j,i||a,c>*t2(a,b,j,i)*t1(b,k)*t1(c,k)
    energy +=  0.0416666666666667 * einsum('jiac,abji,bk,ck', g[o, o, v, v], t2, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.083333333333333 <k,i||a,c>*t2(a,b,j,i)*t1(b,j)*t1(c,k)
    energy += -0.083333333333333 * einsum('kiac,abji,bj,ck', g[o, o, v, v], t2, t1, t1, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	 -0.083333333333333 <b,a||i,j>*t1(a,i)*t2(c,b,j,k)*t1(c,k)
    energy += -0.083333333333333 * einsum('baij,ai,cbjk,ck', g[v, v, o, o], t1, t2, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.083333333333333 <l,a||i,j>*t2(b,a,j,k)*t2(b,c,l,k)*t1(c,i)
    energy +=  0.083333333333333 * einsum('laij,bajk,bclk,ci', g[o, v, o, o], t2, t2, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.0416666666666667 <i,l||c,l>*t2(a,b,j,i)*t2(b,a,j,k)*t1(c,k)
    energy += -0.0416666666666667 * einsum('ilcl,abji,bajk,ck', g[o, o, v, o], t2, t2, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.16666666666667 <l,j||a,i>*t2(a,b,k,j)*t2(c,b,i,k)*t1(c,l)
    energy +=  0.16666666666667 * einsum('ljai,abkj,cbik,cl', g[o, o, v, o], t2, t2, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.083333333333333 <l,j||c,i>*t2(b,a,i,k)*t2(a,b,k,j)*t1(c,l)
    energy +=  0.083333333333333 * einsum('ljci,baik,abkj,cl', g[o, o, v, o], t2, t2, t1, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	  0.16666666666667 <i,a||b,d>*t2(c,a,j,k)*t2(b,c,j,i)*t1(d,k)
    energy +=  0.16666666666667 * einsum('iabd,cajk,bcji,dk', g[o, v, v, v], t2, t2, t1, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	  0.083333333333333 <k,a||b,d>*t2(c,a,i,j)*t2(b,c,j,i)*t1(d,k)
    energy +=  0.083333333333333 * einsum('kabd,caij,bcji,dk', g[o, v, v, v], t2, t2, t1, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	  0.02083333333333333 <k,i||a,b>*t2(a,b,j,i)*t2(d,c,j,l)*t2(c,d,l,k)
    energy +=  0.02083333333333333 * einsum('kiab,abji,dcjl,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.010 <l,k||a,b>*t2(a,b,j,i)*t2(d,c,i,j)*t2(c,d,l,k)
    energy += -0.010 * einsum('lkab,abji,dcij,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.02083333333333333 <j,i||a,c>*t2(a,b,j,i)*t2(d,b,k,l)*t2(c,d,l,k)
    energy +=  0.02083333333333333 * einsum('jiac,abji,dbkl,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.083333333333333 <k,i||a,c>*t2(a,b,j,i)*t2(d,b,j,l)*t2(c,d,l,k)
    energy += -0.083333333333333 * einsum('kiac,abji,dbjl,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.02083333333333333 <l,k||a,c>*t2(a,b,j,i)*t2(d,b,i,j)*t2(c,d,l,k)
    energy +=  0.02083333333333333 * einsum('lkac,abji,dbij,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	 -0.010 <j,i||c,d>*t2(a,b,j,i)*t2(b,a,k,l)*t2(c,d,l,k)
    energy += -0.010 * einsum('jicd,abji,bakl,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.02083333333333333 <k,i||c,d>*t2(a,b,j,i)*t2(b,a,j,l)*t2(c,d,l,k)
    energy +=  0.02083333333333333 * einsum('kicd,abji,bajl,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	  0.0416666666666667 <b,a||i,j>*t2(a,b,j,k)*t2(d,c,i,l)*t2(c,d,l,k)
    energy +=  0.0416666666666667 * einsum('baij,abjk,dcil,cdlk', g[v, v, o, o], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.010416666666666666 <b,a||i,j>*t2(a,b,k,l)*t2(d,c,j,i)*t2(c,d,l,k)
    energy += -0.010416666666666666 * einsum('baij,abkl,dcji,cdlk', g[v, v, o, o], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.0416666666666667 <b,a||i,j>*t2(d,a,k,l)*t2(c,b,j,i)*t2(c,d,l,k)
    energy +=  0.0416666666666667 * einsum('baij,dakl,cbji,cdlk', g[v, v, o, o], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.083333333333333 <b,a||i,j>*t2(d,a,i,l)*t2(c,b,j,k)*t2(c,d,l,k)
    energy += -0.083333333333333 * einsum('baij,dail,cbjk,cdlk', g[v, v, o, o], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    return energy
    
    
def quccsd_singles_residual(t1, t2, f, g, o, v):
    
    #    < 0 | i* a e(-T) H e(T) | 0> :
    
    #	  1.00 f(a,i)
    singles_res =  1.00 * einsum('ai->ai', f[v, o])
    
    #	 -1.00 f(j,i)*t1(a,j)
    singles_res += -1.00 * einsum('ji,aj->ai', f[o, o], t1)
    
    #	  1.00 f(a,b)*t1(b,i)
    singles_res +=  1.00 * einsum('ab,bi->ai', f[v, v], t1)
    
    #	 -1.00 f(j,b)*t2(b,a,i,j)
    singles_res += -1.00 * einsum('jb,baij->ai', f[o, v], t2)
    
    #	  1.00 <j,a||b,i>*t1(b,j)
    singles_res +=  1.00 * einsum('jabi,bj->ai', g[o, v, v, o], t1)
    
    #	 -0.50 <b,a||i,j>*t1(b,j)
    singles_res += -0.50 * einsum('baij,bj->ai', g[v, v, o, o], t1)
    
    #	 -0.50 <k,j||b,i>*t2(b,a,k,j)
    singles_res += -0.50 * einsum('kjbi,bakj->ai', g[o, o, v, o], t2)
    
    #	 -0.50 <j,a||b,c>*t2(b,c,i,j)
    singles_res += -0.50 * einsum('jabc,bcij->ai', g[o, v, v, v], t2)
    
    #	  0.416666666666667 <k,j||b,c>*t2(c,a,i,k)*t1(b,j)
    singles_res +=  0.416666666666667 * einsum('kjbc,caik,bj->ai', g[o, o, v, v], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.3333333333333 <k,j||b,c>*t2(c,a,k,j)*t1(b,i)
    singles_res +=  0.3333333333333 * einsum('kjbc,cakj,bi->ai', g[o, o, v, v], t2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.3333333333333 <k,j||b,c>*t1(a,j)*t2(b,c,i,k)
    singles_res +=  0.3333333333333 * einsum('kjbc,aj,bcik->ai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.16666666666667 <c,b||i,j>*t1(a,k)*t2(b,c,j,k)
    singles_res += -0.16666666666667 * einsum('cbij,ak,bcjk->ai', g[v, v, o, o], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.16666666666667 <b,a||j,k>*t2(c,b,k,j)*t1(c,i)
    singles_res += -0.16666666666667 * einsum('bajk,cbkj,ci->ai', g[v, v, o, o], t2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.3333333333333 <b,a||i,j>*t2(c,b,j,k)*t1(c,k)
    singles_res +=  0.3333333333333 * einsum('baij,cbjk,ck->ai', g[v, v, o, o], t2, t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.00 <k,j||b,i>*t1(a,k)*t1(b,j)
    singles_res +=  1.00 * einsum('kjbi,ak,bj->ai', g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.00 <j,a||b,c>*t1(b,j)*t1(c,i)
    singles_res +=  1.00 * einsum('jabc,bj,ci->ai', g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.50 <k,b||i,j>*t1(a,k)*t1(b,j)
    singles_res += -0.50 * einsum('kbij,ak,bj->ai', g[o, v, o, o], t1, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.50 <k,a||i,j>*t1(b,j)*t1(b,k)
    singles_res +=  0.50 * einsum('kaij,bj,bk->ai', g[o, v, o, o], t1, t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.50 <b,a||c,j>*t1(b,j)*t1(c,i)
    singles_res += -0.50 * einsum('bacj,bj,ci->ai', g[v, v, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.50 <b,a||c,i>*t1(b,j)*t1(c,j)
    singles_res +=  0.50 * einsum('baci,bj,cj->ai', g[v, v, v, o], t1, t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.250 <l,k||i,j>*t2(b,a,l,k)*t1(b,j)
    singles_res += -0.250 * einsum('lkij,balk,bj->ai', g[o, o, o, o], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.50 <j,b||c,i>*t2(c,a,k,j)*t1(b,k)
    singles_res +=  0.50 * einsum('jbci,cakj,bk->ai', g[o, v, v, o], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.50 <k,a||b,j>*t2(b,c,i,k)*t1(c,j)
    singles_res +=  0.50 * einsum('kabj,bcik,cj->ai', g[o, v, v, o], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.250 <b,a||c,d>*t1(b,j)*t2(c,d,i,j)
    singles_res += -0.250 * einsum('bacd,bj,cdij->ai', g[v, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.50 <k,b||i,j>*t2(c,a,l,k)*t2(c,b,j,l)
    singles_res +=  0.50 * einsum('kbij,calk,cbjl->ai', g[o, v, o, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.1250 <l,a||j,k>*t2(b,c,i,l)*t2(c,b,k,j)
    singles_res += -0.1250 * einsum('lajk,bcil,cbkj->ai', g[o, v, o, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.250 <k,a||i,j>*t2(c,b,j,l)*t2(b,c,l,k)
    singles_res +=  0.250 * einsum('kaij,cbjl,bclk->ai', g[o, v, o, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.1250 <c,b||d,i>*t2(d,a,k,j)*t2(b,c,j,k)
    singles_res += -0.1250 * einsum('cbdi,dakj,bcjk->ai', g[v, v, v, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.50 <b,a||c,j>*t2(d,b,j,k)*t2(c,d,i,k)
    singles_res +=  0.50 * einsum('bacj,dbjk,cdik->ai', g[v, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.250 <b,a||c,i>*t2(d,b,j,k)*t2(c,d,k,j)
    singles_res +=  0.250 * einsum('baci,dbjk,cdkj->ai', g[v, v, v, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    return singles_res
    
    
def quccsd_doubles_residual(t1, t2, f, g, o, v):
    
    #    < 0 | i* j* b a e(-T) H e(T) | 0> :
    
    #	 -1.00 P(i,j)f(k,j)*t2(a,b,i,k)
    contracted_intermediate = -1.00 * einsum('kj,abik->abij', f[o, o], t2)
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.00 P(a,b)f(a,c)*t2(c,b,i,j)
    contracted_intermediate =  1.00 * einsum('ac,cbij->abij', f[v, v], t2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.00 <a,b||i,j>
    doubles_res +=  1.00 * einsum('abij->abij', g[v, v, o, o])
    
    #	  1.00 P(a,b)<k,a||i,j>*t1(b,k)
    contracted_intermediate =  1.00 * einsum('kaij,bk->abij', g[o, v, o, o], t1)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.00 P(i,j)<a,b||c,j>*t1(c,i)
    contracted_intermediate =  1.00 * einsum('abcj,ci->abij', g[v, v, v, o], t1)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.50 <l,k||i,j>*t2(a,b,l,k)
    doubles_res +=  0.50 * einsum('lkij,ablk->abij', g[o, o, o, o], t2)
    
    #	  1.00 P(i,j)*P(a,b)<k,a||c,j>*t2(c,b,i,k)
    contracted_intermediate =  1.00 * einsum('kacj,cbik->abij', g[o, v, v, o], t2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.50 <a,b||c,d>*t2(c,d,i,j)
    doubles_res +=  0.50 * einsum('abcd,cdij->abij', g[v, v, v, v], t2)
    
    #	  0.3333333333333 P(a,b)<c,a||i,j>*t1(b,k)*t1(c,k)
    contracted_intermediate =  0.3333333333333 * einsum('caij,bk,ck->abij', g[v, v, o, o], t1, t1, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.3333333333333 P(i,j)<a,b||j,k>*t1(c,i)*t1(c,k)
    contracted_intermediate =  0.3333333333333 * einsum('abjk,ci,ck->abij', g[v, v, o, o], t1, t1, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3333333333333 P(i,j)<l,k||c,d>*t2(a,b,i,l)*t2(c,d,j,k)
    contracted_intermediate = -0.3333333333333 * einsum('lkcd,abil,cdjk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.16666666666667 <l,k||c,d>*t2(a,b,l,k)*t2(c,d,i,j)
    doubles_res +=  0.16666666666667 * einsum('lkcd,ablk,cdij->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.3333333333333 <l,k||c,d>*t2(c,a,l,k)*t2(d,b,i,j)
    doubles_res += -0.3333333333333 * einsum('lkcd,calk,dbij->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.6666666666667 P(i,j)<l,k||c,d>*t2(c,a,j,k)*t2(d,b,i,l)
    contracted_intermediate =  0.6666666666667 * einsum('lkcd,cajk,dbil->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.3333333333333 <l,k||c,d>*t2(c,a,i,j)*t2(d,b,l,k)
    doubles_res += -0.3333333333333 * einsum('lkcd,caij,dblk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.16666666666667 P(i,j)<d,c||j,k>*t2(a,b,i,l)*t2(c,d,k,l)
    contracted_intermediate = -0.16666666666667 * einsum('dcjk,abil,cdkl->abij', g[v, v, o, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.083333333333333 <d,c||i,j>*t2(a,b,l,k)*t2(c,d,k,l)
    doubles_res +=  0.083333333333333 * einsum('dcij,ablk,cdkl->abij', g[v, v, o, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.16666666666667 P(a,b)<c,a||k,l>*t2(d,b,i,j)*t2(d,c,l,k)
    contracted_intermediate = -0.16666666666667 * einsum('cakl,dbij,dclk->abij', g[v, v, o, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.3333333333333 P(i,j)*P(a,b)<c,a||j,k>*t2(d,b,i,l)*t2(d,c,k,l)
    contracted_intermediate =  0.3333333333333 * einsum('cajk,dbil,dckl->abij', g[v, v, o, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.16666666666667 P(a,b)<c,a||i,j>*t2(d,b,l,k)*t2(d,c,k,l)
    contracted_intermediate = -0.16666666666667 * einsum('caij,dblk,dckl->abij', g[v, v, o, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.083333333333333 <a,b||k,l>*t2(c,d,i,j)*t2(d,c,l,k)
    doubles_res +=  0.083333333333333 * einsum('abkl,cdij,dclk->abij', g[v, v, o, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.16666666666667 P(i,j)<a,b||j,k>*t2(c,d,i,l)*t2(d,c,k,l)
    contracted_intermediate = -0.16666666666667 * einsum('abjk,cdil,dckl->abij', g[v, v, o, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.00 <l,k||i,j>*t1(a,k)*t1(b,l)
    doubles_res += -1.00 * einsum('lkij,ak,bl->abij', g[o, o, o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.00 P(i,j)*P(a,b)<k,a||c,j>*t1(b,k)*t1(c,i)
    contracted_intermediate =  1.00 * einsum('kacj,bk,ci->abij', g[o, v, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.00 <a,b||c,d>*t1(c,j)*t1(d,i)
    doubles_res += -1.00 * einsum('abcd,cj,di->abij', g[v, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.00 P(i,j)<l,k||c,j>*t2(a,b,i,l)*t1(c,k)
    contracted_intermediate =  1.00 * einsum('lkcj,abil,ck->abij', g[o, o, v, o], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.50 P(i,j)<l,k||c,j>*t2(a,b,l,k)*t1(c,i)
    contracted_intermediate =  0.50 * einsum('lkcj,ablk,ci->abij', g[o, o, v, o], t2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<l,k||c,j>*t1(a,k)*t2(c,b,i,l)
    contracted_intermediate = -1.00 * einsum('lkcj,ak,cbil->abij', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.00 P(a,b)<k,a||c,d>*t2(d,b,i,j)*t1(c,k)
    contracted_intermediate =  1.00 * einsum('kacd,dbij,ck->abij', g[o, v, v, v], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.00 P(i,j)*P(a,b)<k,a||c,d>*t2(d,b,i,k)*t1(c,j)
    contracted_intermediate = -1.00 * einsum('kacd,dbik,cj->abij', g[o, v, v, v], t2, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.50 P(a,b)<k,a||c,d>*t1(b,k)*t2(c,d,i,j)
    contracted_intermediate =  0.50 * einsum('kacd,bk,cdij->abij', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.50 P(i,j)<l,c||j,k>*t2(a,b,i,l)*t1(c,k)
    contracted_intermediate = -0.50 * einsum('lcjk,abil,ck->abij', g[o, v, o, o], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.50 <k,c||i,j>*t2(a,b,l,k)*t1(c,l)
    doubles_res += -0.50 * einsum('kcij,ablk,cl->abij', g[o, v, o, o], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.50 P(i,j)*P(a,b)<l,a||j,k>*t2(c,b,i,l)*t1(c,k)
    contracted_intermediate =  0.50 * einsum('lajk,cbil,ck->abij', g[o, v, o, o], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.50 P(a,b)<c,a||d,k>*t2(d,b,i,j)*t1(c,k)
    contracted_intermediate = -0.50 * einsum('cadk,dbij,ck->abij', g[v, v, v, o], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  0.50 P(i,j)*P(a,b)<c,a||d,j>*t2(d,b,i,k)*t1(c,k)
    contracted_intermediate =  0.50 * einsum('cadj,dbik,ck->abij', g[v, v, v, o], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.50 <a,b||c,k>*t2(c,d,i,j)*t1(d,k)
    doubles_res += -0.50 * einsum('abck,cdij,dk->abij', g[v, v, v, o], t2, t1, optimize=['einsum_path', (0, 2), (0, 1)])
    
    return doubles_res
    
