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
spin-orbital UCC4 amplitude equations
"""

import numpy as np
from numpy import einsum

def ucc4_iterations(t1, t2, fock, g, o, v, e_ai, e_abij, hf_energy, max_iter=100, 
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
    old_energy = ucc4_energy(t1, t2, fock, g, o, v)

    print("")
    print("    ==> UCC4 amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(max_iter):

        residual_singles = 1.0 * ucc4_singles_residual(t1, t2, fock, g, o, v)
        residual_doubles = ucc4_doubles_residual(t1, t2, fock, g, o, v)

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

        current_energy = ucc4_energy(new_singles, new_doubles, fock, g, o, v)
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
        raise ValueError("UCC4 iterations did not converge")


    return t1, t2

def ucc4_energy(t1, t2, f, g, o, v):

    #    < 0 | e(-T) H e(T) | 0> :
    
    #	  1.0000 f(i,i)
    energy =  1.000000000000000 * einsum('ii', f[o, o])
    
    #	 -0.5000 <j,i||j,i>
    energy += -0.500000000000000 * einsum('jiji', g[o, o, o, o])
    
    #	  1.0000 f(i,a)*t1(a,i)
    energy +=  1.000000000000000 * einsum('ia,ai', f[o, v], t1)
    
    #	  1.0000 f(a,i)*t1(a,i)
    energy +=  1.000000000000000 * einsum('ai,ai', f[v, o], t1)
    
    #	  0.2500 <j,i||a,b>*t2(a,b,j,i)
    energy +=  0.250000000000000 * einsum('jiab,abji', g[o, o, v, v], t2)
    
    #	  0.2500 <b,a||i,j>*t2(a,b,j,i)
    energy +=  0.250000000000000 * einsum('baij,abji', g[v, v, o, o], t2)
    
    #	 -1.0000 f(j,i)*t1(a,i)*t1(a,j)
    energy += -1.000000000000000 * einsum('ji,ai,aj', f[o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f(a,b)*t1(a,i)*t1(b,i)
    energy +=  1.000000000000000 * einsum('ab,ai,bi', f[v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 f(a,i)*t1(b,j)*t2(b,a,i,j)
    energy += -0.500000000000000 * einsum('ai,bj,baij', f[v, o], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 f(i,a)*t1(b,j)*t2(a,b,j,i)
    energy += -0.500000000000000 * einsum('ia,bj,abji', f[o, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 f(j,i)*t2(b,a,i,k)*t2(a,b,k,j)
    energy += -0.500000000000000 * einsum('ji,baik,abkj', f[o, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 f(a,b)*t2(c,a,i,j)*t2(b,c,j,i)
    energy +=  0.500000000000000 * einsum('ab,caij,bcji', f[v, v], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <k,j||a,i>*t1(b,i)*t2(a,b,k,j)
    energy += -0.500000000000000 * einsum('kjai,bi,abkj', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <i,a||b,c>*t1(a,j)*t2(b,c,j,i)
    energy += -0.500000000000000 * einsum('iabc,aj,bcji', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <k,a||i,j>*t1(b,k)*t2(b,a,j,i)
    energy +=  0.500000000000000 * einsum('kaij,bk,baji', g[o, v, o, o], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <b,a||c,i>*t1(c,j)*t2(a,b,i,j)
    energy +=  0.500000000000000 * einsum('baci,cj,abij', g[v, v, v, o], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.1250 <l,k||i,j>*t2(b,a,j,i)*t2(a,b,l,k)
    energy +=  0.125000000000000 * einsum('lkij,baji,ablk', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,a||b,i>*t2(c,a,i,k)*t2(b,c,k,j)
    energy +=  1.000000000000000 * einsum('jabi,caik,bckj', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.1250 <b,a||c,d>*t2(a,b,i,j)*t2(c,d,j,i)
    energy +=  0.125000000000000 * einsum('bacd,abij,cdji', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.1667 f(j,a)*t1(a,i)*t2(c,b,i,k)*t2(b,c,k,j)
    energy += -0.166666666666670 * einsum('ja,ai,cbik,bckj', f[o, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1667 f(i,b)*t1(a,i)*t2(c,a,j,k)*t2(b,c,k,j)
    energy += -0.166666666666670 * einsum('ib,ai,cajk,bckj', f[o, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.3333 f(a,i)*t1(a,j)*t2(c,b,i,k)*t2(b,c,k,j)
    energy += -0.333333333333330 * einsum('ai,aj,cbik,bckj', f[v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.3333 f(a,i)*t1(b,i)*t2(c,a,j,k)*t2(b,c,k,j)
    energy += -0.333333333333330 * einsum('ai,bi,cajk,bckj', f[v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1667 f(j,b)*t1(a,i)*t2(c,a,i,k)*t2(b,c,k,j)
    energy += -0.166666666666670 * einsum('jb,ai,caik,bckj', f[o, v], t1, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	  0.3333 f(i,a)*t1(c,k)*t2(c,b,j,k)*t2(a,b,j,i)
    energy +=  0.333333333333330 * einsum('ia,ck,cbjk,abji', f[o, v], t1, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	  0.1667 f(k,a)*t1(c,k)*t2(c,b,i,j)*t2(a,b,j,i)
    energy +=  0.166666666666670 * einsum('ka,ck,cbij,abji', f[o, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1667 f(i,c)*t1(c,k)*t2(b,a,j,k)*t2(a,b,j,i)
    energy +=  0.166666666666670 * einsum('ic,ck,bajk,abji', f[o, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.1667 f(a,i)*t1(c,k)*t2(b,a,i,j)*t2(b,c,k,j)
    energy +=  0.166666666666670 * einsum('ai,ck,baij,bckj', f[v, o], t1, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.0833 <k,i||a,b>*t2(d,c,j,l)*t2(a,b,j,i)*t2(c,d,l,k)
    energy += -0.083333333333330 * einsum('kiab,dcjl,abji,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.0208 <l,k||a,b>*t2(d,c,i,j)*t2(a,b,j,i)*t2(c,d,l,k)
    energy +=  0.020833333333330 * einsum('lkab,dcij,abji,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.0833 <j,i||a,c>*t2(d,b,k,l)*t2(a,b,j,i)*t2(c,d,l,k)
    energy += -0.083333333333330 * einsum('jiac,dbkl,abji,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.3333 <k,i||a,c>*t2(d,b,j,l)*t2(a,b,j,i)*t2(c,d,l,k)
    energy +=  0.333333333333330 * einsum('kiac,dbjl,abji,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.0833 <l,k||a,c>*t2(d,b,i,j)*t2(a,b,j,i)*t2(c,d,l,k)
    energy += -0.083333333333330 * einsum('lkac,dbij,abji,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	  0.0208 <j,i||c,d>*t2(b,a,k,l)*t2(a,b,j,i)*t2(c,d,l,k)
    energy +=  0.020833333333330 * einsum('jicd,bakl,abji,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.0833 <k,i||c,d>*t2(b,a,j,l)*t2(a,b,j,i)*t2(c,d,l,k)
    energy += -0.083333333333330 * einsum('kicd,bajl,abji,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	 -0.1667 <b,a||i,j>*t2(d,c,i,l)*t2(a,b,j,k)*t2(c,d,l,k)
    energy += -0.166666666666670 * einsum('baij,dcil,abjk,cdlk', g[v, v, o, o], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.0417 <b,a||i,j>*t2(d,c,j,i)*t2(a,b,k,l)*t2(c,d,l,k)
    energy +=  0.041666666666670 * einsum('baij,dcji,abkl,cdlk', g[v, v, o, o], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.1667 <b,a||i,j>*t2(d,a,k,l)*t2(c,b,j,i)*t2(c,d,l,k)
    energy += -0.166666666666670 * einsum('baij,dakl,cbji,cdlk', g[v, v, o, o], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.3333 <b,a||i,j>*t2(d,a,i,l)*t2(c,b,j,k)*t2(c,d,l,k)
    energy +=  0.333333333333330 * einsum('baij,dail,cbjk,cdlk', g[v, v, o, o], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.0104 <j,i||a,b>*t2(d,c,k,l)*t2(a,b,j,i)*t2(c,d,l,k)
    energy +=  0.010416666666670 * einsum('jiab,dckl,abji,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.0104 <b,a||i,j>*t2(a,b,j,i)*t2(d,c,k,l)*t2(c,d,l,k)
    energy +=  0.010416666666670 * einsum('baij,abji,dckl,cdlk', g[v, v, o, o], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.0104 <l,k||c,d>*t2(b,a,i,j)*t2(a,b,j,i)*t2(c,d,l,k)
    energy += -0.010416666666670 * einsum('lkcd,baij,abji,cdlk', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	 -0.0104 <d,c||k,l>*t2(b,a,i,j)*t2(a,b,j,i)*t2(c,d,l,k)
    energy += -0.010416666666670 * einsum('dckl,baij,abji,cdlk', g[v, v, o, o], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	 -0.0833 f(j,i)*t2(d,c,i,m)*t2(b,a,k,l)*t2(a,b,k,j)*t2(c,d,m,l)
    energy += -0.083333333333330 * einsum('ji,dcim,bakl,abkj,cdml', f[o, o], t2, t2, t2, t2, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.0417 f(j,i)*t2(d,c,i,k)*t2(b,a,l,m)*t2(a,b,k,j)*t2(c,d,m,l)
    energy += -0.041666666666670 * einsum('ji,dcik,balm,abkj,cdml', f[o, o], t2, t2, t2, t2, optimize=['einsum_path', (1, 4), (1, 2), (1, 2), (0, 1)])
    
    #	  0.3333 f(j,i)*t2(d,b,i,m)*t2(c,a,k,l)*t2(a,b,k,j)*t2(c,d,m,l)
    energy +=  0.333333333333330 * einsum('ji,dbim,cakl,abkj,cdml', f[o, o], t2, t2, t2, t2, optimize=['einsum_path', (1, 4), (1, 2), (1, 2), (0, 1)])
    
    #	  0.1667 f(j,i)*t2(d,b,i,k)*t2(c,a,l,m)*t2(a,b,k,j)*t2(c,d,m,l)
    energy +=  0.166666666666670 * einsum('ji,dbik,calm,abkj,cdml', f[o, o], t2, t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	 -0.0833 f(j,i)*t2(b,a,i,m)*t2(d,c,k,l)*t2(a,b,k,j)*t2(c,d,m,l)
    energy += -0.083333333333330 * einsum('ji,baim,dckl,abkj,cdml', f[o, o], t2, t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.0833 f(a,b)*t2(e,a,k,l)*t2(d,c,i,j)*t2(b,c,j,i)*t2(d,e,l,k)
    energy +=  0.083333333333330 * einsum('ab,eakl,dcij,bcji,delk', f[v, v], t2, t2, t2, t2, optimize=['einsum_path', (1, 4), (0, 3), (0, 1), (0, 1)])
    
    #	 -0.3333 f(a,b)*t2(e,a,j,l)*t2(d,c,i,k)*t2(b,c,j,i)*t2(d,e,l,k)
    energy += -0.333333333333330 * einsum('ab,eajl,dcik,bcji,delk', f[v, v], t2, t2, t2, t2, optimize=['einsum_path', (1, 4), (1, 2), (1, 2), (0, 1)])
    
    #	  0.0833 f(a,b)*t2(e,a,i,j)*t2(d,c,k,l)*t2(b,c,j,i)*t2(d,e,l,k)
    energy +=  0.083333333333330 * einsum('ab,eaij,dckl,bcji,delk', f[v, v], t2, t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.0417 f(a,b)*t2(c,a,k,l)*t2(e,d,i,j)*t2(b,c,j,i)*t2(d,e,l,k)
    energy +=  0.041666666666670 * einsum('ab,cakl,edij,bcji,delk', f[v, v], t2, t2, t2, t2, optimize=['einsum_path', (1, 4), (1, 2), (1, 2), (0, 1)])
    
    #	 -0.1667 f(a,b)*t2(c,a,j,l)*t2(e,d,i,k)*t2(b,c,j,i)*t2(d,e,l,k)
    energy += -0.166666666666670 * einsum('ab,cajl,edik,bcji,delk', f[v, v], t2, t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1), (0, 1)])
    
    #	  0.0208 f(l,k)*t2(b,a,i,j)*t2(a,b,j,i)*t2(d,c,k,m)*t2(c,d,m,l)
    energy +=  0.020833333333330 * einsum('lk,baij,abji,dckm,cdml', f[o, o], t2, t2, t2, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.0208 f(c,d)*t2(b,a,i,j)*t2(a,b,j,i)*t2(e,c,k,l)*t2(d,e,l,k)
    energy += -0.020833333333330 * einsum('cd,baij,abji,eckl,delk', f[v, v], t2, t2, t2, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.0208 f(j,i)*t2(d,c,l,m)*t2(b,a,i,k)*t2(a,b,k,j)*t2(c,d,m,l)
    energy += -0.020833333333330 * einsum('ji,dclm,baik,abkj,cdml', f[o, o], t2, t2, t2, t2, optimize=['einsum_path', (1, 4), (1, 2), (0, 2), (0, 1)])
    
    #	  0.0208 f(a,b)*t2(e,d,k,l)*t2(c,a,i,j)*t2(b,c,j,i)*t2(d,e,l,k)
    energy +=  0.020833333333330 * einsum('ab,edkl,caij,bcji,delk', f[v, v], t2, t2, t2, t2, optimize=['einsum_path', (1, 4), (1, 2), (0, 2), (0, 1)])
    
    return energy
    
    
def ucc4_singles_residual(t1, t2, f, g, o, v):
    
    #    < 0 | m* e e(-T) H e(T) | 0> :
    
    #	  1.0000 f(e,m)
    singles_res =  1.000000000000000 * einsum('em->em', f[v, o])
    
    #	 -1.0000 f(i,m)*t1(e,i)
    singles_res += -1.000000000000000 * einsum('im,ei->em', f[o, o], t1)
    
    #	  1.0000 f(e,a)*t1(a,m)
    singles_res +=  1.000000000000000 * einsum('ea,am->em', f[v, v], t1)
    
    #	 -1.0000 f(i,a)*t2(a,e,m,i)
    singles_res += -1.000000000000000 * einsum('ia,aemi->em', f[o, v], t2)
    
    #	  1.0000 <i,e||a,m>*t1(a,i)
    singles_res +=  1.000000000000000 * einsum('ieam,ai->em', g[o, v, v, o], t1)
    
    #	 -1.0000 <a,e||m,i>*t1(a,i)
    singles_res += -1.000000000000000 * einsum('aemi,ai->em', g[v, v, o, o], t1)
    
    #	 -0.5000 <j,i||a,m>*t2(a,e,j,i)
    singles_res += -0.500000000000000 * einsum('jiam,aeji->em', g[o, o, v, o], t2)
    
    #	 -0.5000 <i,e||a,b>*t2(a,b,m,i)
    singles_res += -0.500000000000000 * einsum('ieab,abmi->em', g[o, v, v, v], t2)
    
    #	  1.0000 f(j,i)*t1(a,i)*t2(a,e,m,j)
    singles_res +=  1.000000000000000 * einsum('ji,ai,aemj->em', f[o, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f(a,b)*t1(a,i)*t2(b,e,m,i)
    singles_res += -1.000000000000000 * einsum('ab,ai,bemi->em', f[v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 f(i,m)*t1(a,j)*t2(a,e,j,i)
    singles_res += -0.500000000000000 * einsum('im,aj,aeji->em', f[o, o], t1, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 f(e,a)*t1(b,i)*t2(a,b,m,i)
    singles_res +=  0.500000000000000 * einsum('ea,bi,abmi->em', f[v, v], t1, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 f(a,i)*t2(b,a,i,j)*t2(b,e,m,j)
    singles_res +=  0.500000000000000 * einsum('ai,baij,bemj->em', f[v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 f(a,m)*t2(b,a,i,j)*t2(b,e,j,i)
    singles_res +=  0.250000000000000 * einsum('am,baij,beji->em', f[v, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 f(e,i)*t2(b,a,i,j)*t2(a,b,m,j)
    singles_res +=  0.250000000000000 * einsum('ei,baij,abmj->em', f[v, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <k,a||i,j>*t2(b,a,j,i)*t2(b,e,m,k)
    singles_res += -0.500000000000000 * einsum('kaij,baji,bemk->em', g[o, v, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,a||m,i>*t2(b,a,i,k)*t2(b,e,k,j)
    singles_res +=  1.000000000000000 * einsum('jami,baik,bekj->em', g[o, v, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <k,e||i,j>*t2(b,a,j,i)*t2(a,b,m,k)
    singles_res += -0.250000000000000 * einsum('keij,baji,abmk->em', g[o, v, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <j,e||m,i>*t2(b,a,i,k)*t2(a,b,k,j)
    singles_res +=  0.500000000000000 * einsum('jemi,baik,abkj->em', g[o, v, o, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 <b,a||c,i>*t2(a,b,i,j)*t2(c,e,m,j)
    singles_res += -0.500000000000000 * einsum('baci,abij,cemj->em', g[v, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <b,a||c,m>*t2(a,b,i,j)*t2(c,e,j,i)
    singles_res += -0.250000000000000 * einsum('bacm,abij,ceji->em', g[v, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,e||b,i>*t2(c,a,i,j)*t2(b,c,m,j)
    singles_res +=  1.000000000000000 * einsum('aebi,caij,bcmj->em', g[v, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <a,e||b,m>*t2(c,a,i,j)*t2(b,c,j,i)
    singles_res +=  0.500000000000000 * einsum('aebm,caij,bcji->em', g[v, v, v, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.3333 f(i,a)*t2(c,b,j,k)*t2(a,b,j,i)*t2(c,e,m,k)
    singles_res += -0.333333333333330 * einsum('ia,cbjk,abji,cemk->em', f[o, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    
    #	 -0.1667 f(i,a)*t2(c,b,j,k)*t2(a,b,m,i)*t2(c,e,k,j)
    singles_res += -0.166666666666670 * einsum('ia,cbjk,abmi,cekj->em', f[o, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1667 f(k,a)*t2(c,b,i,j)*t2(a,b,j,i)*t2(c,e,m,k)
    singles_res += -0.166666666666670 * einsum('ka,cbij,abji,cemk->em', f[o, v], t2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.3333 f(j,a)*t2(c,b,i,k)*t2(a,b,m,i)*t2(c,e,k,j)
    singles_res +=  0.333333333333330 * einsum('ja,cbik,abmi,cekj->em', f[o, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1667 f(i,a)*t2(c,b,j,k)*t2(a,e,j,i)*t2(b,c,m,k)
    singles_res += -0.166666666666670 * einsum('ia,cbjk,aeji,bcmk->em', f[o, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	 -0.1667 f(k,a)*t2(c,b,i,j)*t2(a,e,j,i)*t2(b,c,m,k)
    singles_res += -0.166666666666670 * einsum('ka,cbij,aeji,bcmk->em', f[o, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1667 f(j,a)*t2(c,b,i,k)*t2(a,e,m,i)*t2(b,c,k,j)
    singles_res +=  0.166666666666670 * einsum('ja,cbik,aemi,bckj->em', f[o, v], t2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.1667 f(i,c)*t2(b,a,j,k)*t2(a,b,j,i)*t2(c,e,m,k)
    singles_res += -0.166666666666670 * einsum('ic,bajk,abji,cemk->em', f[o, v], t2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    
    #	  0.3333 f(i,b)*t2(c,a,j,k)*t2(a,e,j,i)*t2(b,c,m,k)
    singles_res +=  0.333333333333330 * einsum('ib,cajk,aeji,bcmk->em', f[o, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    
    #	  0.1667 f(i,b)*t2(c,a,j,k)*t2(a,e,m,i)*t2(b,c,k,j)
    singles_res +=  0.166666666666670 * einsum('ib,cajk,aemi,bckj->em', f[o, v], t2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    
    #	 -0.0833 f(j,c)*t2(b,a,i,k)*t2(a,b,m,i)*t2(c,e,k,j)
    singles_res += -0.083333333333330 * einsum('jc,baik,abmi,cekj->em', f[o, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	 -0.0833 f(k,b)*t2(c,a,i,j)*t2(a,e,j,i)*t2(b,c,m,k)
    singles_res += -0.083333333333330 * einsum('kb,caij,aeji,bcmk->em', f[o, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    
    #	  0.1667 f(j,b)*t2(c,a,i,k)*t2(a,e,m,i)*t2(b,c,k,j)
    singles_res +=  0.166666666666670 * einsum('jb,caik,aemi,bckj->em', f[o, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    
    return singles_res
    
    
def ucc4_doubles_residual(t1, t2, f, g, o, v):
    
    #    < 0 | m* n* f e e(-T) H e(T) | 0> :
    
    #	  1.0000 <e,f||m,n>
    doubles_res =  1.000000000000000 * einsum('efmn->efmn', g[v, v, o, o])
    
    #	 -1.0000 P(m,n)f(i,n)*t2(e,f,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('in,efmi->efmn', f[o, o], t2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(e,f)f(e,a)*t2(a,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('ea,afmn->efmn', f[v, v], t2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  1.0000 P(e,f)<i,e||m,n>*t1(f,i)
    contracted_intermediate =  1.000000000000000 * einsum('iemn,fi->efmn', g[o, v, o, o], t1)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  1.0000 P(m,n)<e,f||a,n>*t1(a,m)
    contracted_intermediate =  1.000000000000000 * einsum('efan,am->efmn', g[v, v, v, o], t1)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.5000 <j,i||m,n>*t2(e,f,j,i)
    doubles_res +=  0.500000000000000 * einsum('jimn,efji->efmn', g[o, o, o, o], t2)
    
    #	  1.0000 P(m,n)*P(e,f)<i,e||a,n>*t2(a,f,m,i)
    contracted_intermediate =  1.000000000000000 * einsum('iean,afmi->efmn', g[o, v, v, o], t2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  0.5000 <e,f||a,b>*t2(a,b,m,n)
    doubles_res +=  0.500000000000000 * einsum('efab,abmn->efmn', g[v, v, v, v], t2)
    
    #	 -1.0000 P(m,n)f(i,a)*t1(a,n)*t2(e,f,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('ia,an,efmi->efmn', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -1.0000 P(e,f)f(i,a)*t1(e,i)*t2(a,f,m,n)
    contracted_intermediate = -1.000000000000000 * einsum('ia,ei,afmn->efmn', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -0.5000 P(m,n)f(a,n)*t1(a,i)*t2(e,f,m,i)
    contracted_intermediate = -0.500000000000000 * einsum('an,ai,efmi->efmn', f[v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -0.5000 P(e,f)f(e,i)*t1(a,i)*t2(a,f,m,n)
    contracted_intermediate = -0.500000000000000 * einsum('ei,ai,afmn->efmn', f[v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -0.5000 P(m,n)<j,i||a,b>*t2(a,b,n,i)*t2(e,f,m,j)
    contracted_intermediate = -0.500000000000000 * einsum('jiab,abni,efmj->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.2500 <j,i||a,b>*t2(a,b,m,n)*t2(e,f,j,i)
    doubles_res +=  0.250000000000000 * einsum('jiab,abmn,efji->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||a,b>*t2(a,e,j,i)*t2(b,f,m,n)
    doubles_res += -0.500000000000000 * einsum('jiab,aeji,bfmn->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(m,n)<j,i||a,b>*t2(a,e,n,i)*t2(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,aeni,bfmj->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -0.5000 <j,i||a,b>*t2(a,e,m,n)*t2(b,f,j,i)
    doubles_res += -0.500000000000000 * einsum('jiab,aemn,bfji->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 P(m,n)<b,a||n,i>*t2(a,b,i,j)*t2(e,f,m,j)
    contracted_intermediate = -0.250000000000000 * einsum('bani,abij,efmj->efmn', g[v, v, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.1250 <b,a||m,n>*t2(a,b,i,j)*t2(e,f,j,i)
    doubles_res +=  0.125000000000000 * einsum('bamn,abij,efji->efmn', g[v, v, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 P(e,f)<a,e||i,j>*t2(b,a,j,i)*t2(b,f,m,n)
    contracted_intermediate = -0.250000000000000 * einsum('aeij,baji,bfmn->efmn', g[v, v, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  0.5000 P(m,n)*P(e,f)<a,e||n,i>*t2(b,a,i,j)*t2(b,f,m,j)
    contracted_intermediate =  0.500000000000000 * einsum('aeni,baij,bfmj->efmn', g[v, v, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	 -0.2500 P(e,f)<a,e||m,n>*t2(b,a,i,j)*t2(b,f,j,i)
    contracted_intermediate = -0.250000000000000 * einsum('aemn,baij,bfji->efmn', g[v, v, o, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  0.1250 <e,f||i,j>*t2(b,a,j,i)*t2(a,b,m,n)
    doubles_res +=  0.125000000000000 * einsum('efij,baji,abmn->efmn', g[v, v, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 P(m,n)<e,f||n,i>*t2(b,a,i,j)*t2(a,b,m,j)
    contracted_intermediate = -0.250000000000000 * einsum('efni,baij,abmj->efmn', g[v, v, o, o], t2, t2, optimize=['einsum_path', (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.2500 P(m,n)f(j,i)*t2(b,a,i,k)*t2(a,b,n,j)*t2(e,f,m,k)
    contracted_intermediate =  0.250000000000000 * einsum('ji,baik,abnj,efmk->efmn', f[o, o], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -0.2500 f(j,i)*t2(b,a,i,k)*t2(a,b,m,n)*t2(e,f,k,j)
    doubles_res += -0.250000000000000 * einsum('ji,baik,abmn,efkj->efmn', f[o, o], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 P(e,f)f(j,i)*t2(b,a,i,k)*t2(a,e,k,j)*t2(b,f,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('ji,baik,aekj,bfmn->efmn', f[o, o], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -0.5000 P(m,n)*P(e,f)f(j,i)*t2(b,a,i,k)*t2(a,e,n,j)*t2(b,f,m,k)
    contracted_intermediate = -0.500000000000000 * einsum('ji,baik,aenj,bfmk->efmn', f[o, o], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  0.2500 P(m,n)f(j,i)*t2(b,a,i,k)*t2(e,f,n,j)*t2(a,b,m,k)
    contracted_intermediate =  0.250000000000000 * einsum('ji,baik,efnj,abmk->efmn', f[o, o], t2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -0.5000 P(m,n)f(a,b)*t2(c,a,i,j)*t2(b,c,n,i)*t2(e,f,m,j)
    contracted_intermediate = -0.500000000000000 * einsum('ab,caij,bcni,efmj->efmn', f[v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.2500 f(a,b)*t2(c,a,i,j)*t2(b,c,m,n)*t2(e,f,j,i)
    doubles_res +=  0.250000000000000 * einsum('ab,caij,bcmn,efji->efmn', f[v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.2500 P(e,f)f(a,b)*t2(c,a,i,j)*t2(b,e,j,i)*t2(c,f,m,n)
    contracted_intermediate = -0.250000000000000 * einsum('ab,caij,beji,cfmn->efmn', f[v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  0.5000 P(m,n)*P(e,f)f(a,b)*t2(c,a,i,j)*t2(b,e,n,i)*t2(c,f,m,j)
    contracted_intermediate =  0.500000000000000 * einsum('ab,caij,beni,cfmj->efmn', f[v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	 -0.2500 P(e,f)f(a,b)*t2(c,a,i,j)*t2(b,e,m,n)*t2(c,f,j,i)
    contracted_intermediate = -0.250000000000000 * einsum('ab,caij,bemn,cfji->efmn', f[v, v], t2, t2, t2, optimize=['einsum_path', (1, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -0.0833 P(m,n)f(i,n)*t2(b,a,j,k)*t2(a,b,j,i)*t2(e,f,m,k)
    contracted_intermediate = -0.083333333333330 * einsum('in,bajk,abji,efmk->efmn', f[o, o], t2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -0.0417 P(m,n)f(i,n)*t2(b,a,j,k)*t2(a,b,m,i)*t2(e,f,k,j)
    contracted_intermediate = -0.041666666666670 * einsum('in,bajk,abmi,efkj->efmn', f[o, o], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.3333 P(m,n)*P(e,f)f(i,n)*t2(b,a,j,k)*t2(a,e,j,i)*t2(b,f,m,k)
    contracted_intermediate =  0.333333333333330 * einsum('in,bajk,aeji,bfmk->efmn', f[o, o], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  0.0833 P(m,n)*P(e,f)f(i,n)*t2(b,a,j,k)*t2(a,e,m,i)*t2(b,f,k,j)
    contracted_intermediate =  0.083333333333330 * einsum('in,bajk,aemi,bfkj->efmn', f[o, o], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	 -0.0833 P(m,n)f(i,n)*t2(b,a,j,k)*t2(e,f,j,i)*t2(a,b,m,k)
    contracted_intermediate = -0.083333333333330 * einsum('in,bajk,efji,abmk->efmn', f[o, o], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.0833 P(e,f)f(e,a)*t2(c,b,i,j)*t2(a,b,j,i)*t2(c,f,m,n)
    contracted_intermediate =  0.083333333333330 * einsum('ea,cbij,abji,cfmn->efmn', f[v, v], t2, t2, t2, optimize=['einsum_path', (1, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -0.3333 P(m,n)*P(e,f)f(e,a)*t2(c,b,i,j)*t2(a,b,n,i)*t2(c,f,m,j)
    contracted_intermediate = -0.333333333333330 * einsum('ea,cbij,abni,cfmj->efmn', f[v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  0.0833 P(e,f)f(e,a)*t2(c,b,i,j)*t2(a,b,m,n)*t2(c,f,j,i)
    contracted_intermediate =  0.083333333333330 * einsum('ea,cbij,abmn,cfji->efmn', f[v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  0.0417 P(e,f)f(e,a)*t2(c,b,i,j)*t2(a,f,j,i)*t2(b,c,m,n)
    contracted_intermediate =  0.041666666666670 * einsum('ea,cbij,afji,bcmn->efmn', f[v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -0.0833 P(m,n)*P(e,f)f(e,a)*t2(c,b,i,j)*t2(a,f,n,i)*t2(b,c,m,j)
    contracted_intermediate = -0.083333333333330 * einsum('ea,cbij,afni,bcmj->efmn', f[v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	 -0.1667 P(m,n)*P(e,f)f(j,m)*t2(b,a,i,k)*t2(a,e,n,i)*t2(b,f,k,j)
    contracted_intermediate = -0.166666666666670 * einsum('jm,baik,aeni,bfkj->efmn', f[o, o], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  0.1667 P(m,n)*P(e,f)f(f,b)*t2(c,a,i,j)*t2(a,e,n,i)*t2(b,c,m,j)
    contracted_intermediate =  0.166666666666670 * einsum('fb,caij,aeni,bcmj->efmn', f[v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    return doubles_res
    
