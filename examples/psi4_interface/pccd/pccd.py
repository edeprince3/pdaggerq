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
spin-orbital pccd amplitude equations
"""
import numpy as np
from numpy import einsum

def ccsd_energy(t1, t2, f, g, o, v):

    #    < 0 | e(-T) H e(T) | 0> :
    
    #	  1.0000 f(i,i)
    energy =  1.000000000000000 * einsum('ii', f[o, o])
    
    #	  1.0000 f(i,a)*t1(a,i)
    energy +=  1.000000000000000 * einsum('ia,ai', f[o, v], t1)
    
    #	 -0.5000 <j,i||j,i>
    energy += -0.500000000000000 * einsum('jiji', g[o, o, o, o])
    
    #	  0.2500 <j,i||a,b>*t2(a,b,j,i)
    energy +=  0.250000000000000 * einsum('jiab,abji', g[o, o, v, v], t2)
    
    #	 -0.5000 <j,i||a,b>*t1(a,i)*t1(b,j)
    energy += -0.500000000000000 * einsum('jiab,ai,bj', g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])

    return energy
    
def singles_residual(t1, t2, f, g, o, v):
    
    #    < 0 | m* e e(-T) H e(T) | 0> :
    
    #	  1.0000 f(e,m)
    singles_res =  1.000000000000000 * einsum('em->em', f[v, o])
    
    #	 -1.0000 f(i,m)*t1(e,i)
    singles_res += -1.000000000000000 * einsum('im,ei->em', f[o, o], t1)
    
    #	  1.0000 f(e,a)*t1(a,m)
    singles_res +=  1.000000000000000 * einsum('ea,am->em', f[v, v], t1)
    
    #	 -1.0000 f(i,a)*t2(a,e,m,i)
    singles_res += -1.000000000000000 * einsum('ia,aemi->em', f[o, v], t2)
    
    #	 -1.0000 f(i,a)*t1(a,m)*t1(e,i)
    singles_res += -1.000000000000000 * einsum('ia,am,ei->em', f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <i,e||a,m>*t1(a,i)
    singles_res +=  1.000000000000000 * einsum('ieam,ai->em', g[o, v, v, o], t1)
    
    #	 -0.5000 <j,i||a,m>*t2(a,e,j,i)
    singles_res += -0.500000000000000 * einsum('jiam,aeji->em', g[o, o, v, o], t2)
    
    #	 -0.5000 <i,e||a,b>*t2(a,b,m,i)
    singles_res += -0.500000000000000 * einsum('ieab,abmi->em', g[o, v, v, v], t2)
    
    #	  1.0000 <j,i||a,b>*t1(a,i)*t2(b,e,m,j)
    singles_res +=  1.000000000000000 * einsum('jiab,ai,bemj->em', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <j,i||a,b>*t1(a,m)*t2(b,e,j,i)
    singles_res +=  0.500000000000000 * einsum('jiab,am,beji->em', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <j,i||a,b>*t1(e,i)*t2(a,b,m,j)
    singles_res +=  0.500000000000000 * einsum('jiab,ei,abmj->em', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <j,i||a,m>*t1(a,i)*t1(e,j)
    singles_res +=  1.000000000000000 * einsum('jiam,ai,ej->em', g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <i,e||a,b>*t1(a,i)*t1(b,m)
    singles_res +=  1.000000000000000 * einsum('ieab,ai,bm->em', g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,i||a,b>*t1(a,i)*t1(b,m)*t1(e,j)
    singles_res +=  1.000000000000000 * einsum('jiab,ai,bm,ej->em', g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    return singles_res    

def doubles_residual(t1, t2, f, g, o, v):
    
    #    < 0 | m* n* f e e(-T) H e(T) | 0> :
    
    #	 -1.0000 P(m,n)f(i,n)*t2(e,f,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('in,efmi->efmn', f[o, o], t2)
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(e,f)f(e,a)*t2(a,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('ea,afmn->efmn', f[v, v], t2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -1.0000 P(m,n)f(i,a)*t1(a,n)*t2(e,f,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('ia,an,efmi->efmn', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -1.0000 P(e,f)f(i,a)*t1(e,i)*t2(a,f,m,n)
    contracted_intermediate = -1.000000000000000 * einsum('ia,ei,afmn->efmn', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  1.0000 <e,f||m,n>
    doubles_res +=  1.000000000000000 * einsum('efmn->efmn', g[v, v, o, o])
    
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
    
    #	  1.0000 P(m,n)<j,i||a,n>*t1(a,i)*t2(e,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jian,ai,efmj->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.5000 P(m,n)<j,i||a,n>*t1(a,m)*t2(e,f,j,i)
    contracted_intermediate =  0.500000000000000 * einsum('jian,am,efji->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -1.0000 P(m,n)*P(e,f)<j,i||a,n>*t1(e,i)*t2(a,f,m,j)
    contracted_intermediate = -1.000000000000000 * einsum('jian,ei,afmj->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  1.0000 P(e,f)<i,e||a,b>*t1(a,i)*t2(b,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('ieab,ai,bfmn->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -1.0000 P(m,n)*P(e,f)<i,e||a,b>*t1(a,n)*t2(b,f,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('ieab,an,bfmi->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  0.5000 P(e,f)<i,e||a,b>*t1(f,i)*t2(a,b,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('ieab,fi,abmn->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -1.0000 <j,i||m,n>*t1(e,i)*t1(f,j)
    doubles_res += -1.000000000000000 * einsum('jimn,ei,fj->efmn', g[o, o, o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(m,n)*P(e,f)<i,e||a,n>*t1(a,m)*t1(f,i)
    contracted_intermediate =  1.000000000000000 * einsum('iean,am,fi->efmn', g[o, v, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	 -1.0000 <e,f||a,b>*t1(a,n)*t1(b,m)
    doubles_res += -1.000000000000000 * einsum('efab,an,bm->efmn', g[v, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
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
    
    #	  1.0000 P(m,n)<j,i||a,b>*t1(a,i)*t1(b,n)*t2(e,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,ai,bn,efmj->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(e,f)<j,i||a,b>*t1(a,i)*t1(e,j)*t2(b,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,ai,ej,bfmn->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -0.5000 <j,i||a,b>*t1(a,n)*t1(b,m)*t2(e,f,j,i)
    doubles_res += -0.500000000000000 * einsum('jiab,an,bm,efji->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(m,n)*P(e,f)<j,i||a,b>*t1(a,n)*t1(e,i)*t2(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,an,ei,bfmj->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	 -0.5000 <j,i||a,b>*t1(e,i)*t1(f,j)*t2(a,b,m,n)
    doubles_res += -0.500000000000000 * einsum('jiab,ei,fj,abmn->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(m,n)<j,i||a,n>*t1(a,m)*t1(e,i)*t1(f,j)
    contracted_intermediate = -1.000000000000000 * einsum('jian,am,ei,fj->efmn', g[o, o, v, o], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -1.0000 P(e,f)<i,e||a,b>*t1(a,n)*t1(b,m)*t1(f,i)
    contracted_intermediate = -1.000000000000000 * einsum('ieab,an,bm,fi->efmn', g[o, v, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  1.0000 <j,i||a,b>*t1(a,n)*t1(b,m)*t1(e,i)*t1(f,j)
    doubles_res +=  1.000000000000000 * einsum('jiab,an,bm,ei,fj->efmn', g[o, o, v, v], t1, t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    return doubles_res

def ccsd_iterations(t1, t2, fock, g, o, v, e_ai, e_abij, hf_energy, nsocc, nsvirt, max_iter=100, 
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
    old_energy = ccsd_energy(t1, t2, fock, g, o, v)

    print("")
    print("    ==> pCCD amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(max_iter):

        residual_singles = singles_residual(t1, t2, fock, g, o, v)
        residual_doubles = doubles_residual(t1, t2, fock, g, o, v)

        residual_singles = np.zeros((nsvirt, nsocc))

        res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles)

        singles_res = residual_singles + fock_e_ai * t1
        doubles_res = residual_doubles + fock_e_abij * t2

        singles_res = np.zeros((nsvirt, nsocc))

        new_singles = singles_res * e_ai
        new_doubles = doubles_res * e_abij

        new_singles = np.zeros((nsvirt, nsocc))

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

        current_energy = ccsd_energy(new_singles, new_doubles, fock, g, o, v)
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
        raise ValueError("pCCD iterations did not converge")

    return t1, t2



def lambda_iterations(t1, t2, fock, g, o, v, e_ai, e_abij, hf_energy, nsocc, nsvirt, max_iter=100, 
        e_convergence=1e-8,r_convergence=1e-8,diis_size=None, diis_start_cycle=4):
          
    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)

    lfock_e_ai = fock_e_ai.transpose(1, 0)
    lfock_e_abij = fock_e_abij.transpose(2, 3, 0, 1)

    # diagonal fock should be rearranged for lambda
    le_ai = e_ai.transpose(1, 0)
    le_abij = e_abij.transpose(2, 3, 0, 1)

    l1 = t1.transpose(1, 0)
    l2 = t2.transpose(2, 3, 0, 1)

    # initialize diis if diis_size is not None
    # else normal scf iterate

    if diis_size is not None:
        from diis import DIIS
        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        t1_dim = t1.size
        old_vec = np.hstack((t1.flatten(), t2.flatten()))

    old_energy = 0.0 #lagrangian_energy(t1, t2, l1, l2, fock, g, o, v)

    print("")
    print("    ==> pCCD lambda amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(max_iter):

        lsingles_res = lambda_singles(t1, t2, l1, l2,  fock, g, o, v)
        ldoubles_res = lambda_doubles(t1, t2, l1, l2, fock, g, o, v)

        lsingles_res = np.zeros((nsocc, nsvirt))

        total_lambda_res = np.linalg.norm(lsingles_res) + np.linalg.norm(ldoubles_res)
        lsingles_res += lfock_e_ai * l1
        ldoubles_res += lfock_e_abij * l2

        for i in range (nsocc):
            for a in range (nsvirt):
                lsingles_res[i,a] = 0.0

        lnew_singles = lsingles_res * le_ai
        lnew_doubles = ldoubles_res * le_abij

        # diis update
        if diis_size is not None:
            vectorized_iterate = np.hstack(
                (lnew_singles.flatten(), lnew_doubles.flatten()))
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                 error_vec)
            lnew_singles = new_vectorized_iterate[:l1.size].reshape(l1.shape)
            lnew_doubles = new_vectorized_iterate[l1.size:].reshape(l2.shape)
            old_vec = new_vectorized_iterate

        current_energy = lagrangian_energy(t1, t2, lnew_singles, lnew_doubles, fock, g, o, v)
        pseudo_energy = 0.25 * einsum('jiab,jiab', g[o, o, v, v], l2)

        delta_e = np.abs(old_energy - pseudo_energy)

        if delta_e < e_convergence and total_lambda_res < r_convergence:
            l1 = lnew_singles
            l2 = lnew_doubles
            break
        else:
            l1 = lnew_singles
            l2 = lnew_doubles
            old_energy = pseudo_energy
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(
                idx, old_energy, delta_e,
                np.linalg.norm(lambda_singles(t1, t2, l1, l2, fock, g, o, v)) +
                np.linalg.norm(lambda_doubles(t1, t2, l1, l2, fock, g, o, v)),
                pseudo_energy
            ))
    else:
        raise ValueError("Did not converge")

    return t1, t2, l1, l2


def lagrangian_energy(t1, t2, l1, l2, f, g, o, v):
    """
    L(t1, t2, l1, l2) = <0|e(-T)H e(T)|0> + \sum_{i}l_{i}<i|e(-T)H e(T)|0>

    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param l1: lagrange multiplier for singles (nocc x nvirt)
    :param l2: lagrange multiplier for doubles (nocc x nocc x nvirt x nvirt)
    :param f: fock operator defined as soei + np.einsum('piiq->pq', astei[:, o, o, :])
              where soei is 1 electron integrals (spinorb) and astei is
              antisymmetric 2 electron integrals in openfermion format
              <12|21>.  <ij|kl> - <ij|lk>
    :param g: antisymmetric 2 electron integrals. See fock input.
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    """
    l_energy = ccsd_energy(t1, t2, f, g, o, v)
    l_energy += np.einsum('me,em', l1, singles_residual(t1, t2, f, g, o, v))
    l_energy += np.einsum('mnef,efmn', l2, doubles_residual(t1, t2, f, g, o, v))
    return l_energy


def lambda_singles(t1, t2, l1, l2, f, g, o, v):
    """
    Derivative of Lagrangian w.r.t t1

    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param l1: lagrange multiplier for singles (nocc x nvirt)
    :param l2: lagrange multiplier for doubles (nocc x nocc x nvirt x nvirt)
    :param f: fock operator defined as soei + np.einsum('piiq->pq', astei[:, o, o, :])
              where soei is 1 electron integrals (spinorb) and astei is
              antisymmetric 2 electron integrals in openfermion format
              <12|21>.  <ij|kl> - <ij|lk>
    :param g: antisymmetric 2 electron integrals. See fock input.
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    """
    #	  1.0000 f(m,e)
    lambda_one = 1.0 * einsum('me->me', f[o, v])

    #	 -1.0000 <i,m||e,a>*t1(a,i)
    lambda_one += -1.0 * einsum('imea,ai->me', g[o, o, v, v], t1)

    #	 -1.0000 f(m,i)*l1(i,e)
    lambda_one += -1.0 * einsum('mi,ie->me', f[o, o], l1)

    #	  1.0000 f(a,e)*l1(m,a)
    lambda_one += 1.0 * einsum('ae,ma->me', f[v, v], l1)

    #	 -1.0000 f(i,e)*l1(m,a)*t1(a,i)
    lambda_one += -1.0 * einsum('ie,ma,ai->me', f[o, v], l1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 f(m,a)*l1(i,e)*t1(a,i)
    lambda_one += -1.0 * einsum('ma,ie,ai->me', f[o, v], l1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 f(j,e)*l2(i,m,b,a)*t2(b,a,i,j)
    lambda_one += -0.5 * einsum('je,imba,baij->me', f[o, v], l2, t2,
                                optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -0.5000 f(m,b)*l2(i,j,e,a)*t2(b,a,i,j)
    lambda_one += -0.5 * einsum('mb,ijea,baij->me', f[o, v], l2, t2,
                                optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 <m,a||e,i>*l1(i,a)
    lambda_one += 1.0 * einsum('maei,ia->me', g[o, v, v, o], l1)

    #	  0.5000 <m,a||i,j>*l2(i,j,a,e)
    lambda_one += 0.5 * einsum('maij,ijae->me', g[o, v, o, o], l2)

    #	  0.5000 <b,a||e,i>*l2(m,i,b,a)
    lambda_one += 0.5 * einsum('baei,miba->me', g[v, v, v, o], l2)

    #	  1.0000 <j,m||e,i>*l1(i,a)*t1(a,j)
    lambda_one += 1.0 * einsum('jmei,ia,aj->me', g[o, o, v, o], l1, t1,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 <j,m||a,i>*l1(i,e)*t1(a,j)
    lambda_one += -1.0 * einsum('jmai,ie,aj->me', g[o, o, v, o], l1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,a||e,b>*l1(i,a)*t1(b,i)
    lambda_one += 1.0 * einsum('maeb,ia,bi->me', g[o, v, v, v], l1, t1,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 <i,a||e,b>*l1(m,a)*t1(b,i)
    lambda_one += -1.0 * einsum('iaeb,ma,bi->me', g[o, v, v, v], l1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <k,m||i,j>*l2(i,j,e,a)*t1(a,k)
    lambda_one += -0.5 * einsum('kmij,ijea,ak->me', g[o, o, o, o], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,b||e,i>*l2(m,i,b,a)*t1(a,j)
    lambda_one += 1.0 * einsum('jbei,miba,aj->me', g[o, v, v, o], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,a||b,j>*l2(i,j,a,e)*t1(b,i)
    lambda_one += 1.0 * einsum('mabj,ijae,bi->me', g[o, v, v, o], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <b,a||e,c>*l2(i,m,b,a)*t1(c,i)
    lambda_one += -0.5 * einsum('baec,imba,ci->me', g[v, v, v, v], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,m||e,b>*l1(i,a)*t2(b,a,i,j)
    lambda_one += 1.0 * einsum('jmeb,ia,baij->me', g[o, o, v, v], l1, t2,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	  0.5000 <j,i||e,b>*l1(m,a)*t2(b,a,j,i)
    lambda_one += 0.5 * einsum('jieb,ma,baji->me', g[o, o, v, v], l1, t2,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <j,m||a,b>*l1(i,e)*t2(a,b,i,j)
    lambda_one += 0.5 * einsum('jmab,ie,abij->me', g[o, o, v, v], l1, t2,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <k,m||e,j>*l2(i,j,b,a)*t2(b,a,i,k)
    lambda_one += 0.5 * einsum('kmej,ijba,baik->me', g[o, o, v, o], l2, t2,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	  0.2500 <k,j||e,i>*l2(m,i,b,a)*t2(b,a,k,j)
    lambda_one += 0.25 * einsum('kjei,miba,bakj->me', g[o, o, v, o], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <k,m||b,j>*l2(i,j,e,a)*t2(b,a,i,k)
    lambda_one += -1.0 * einsum('kmbj,ijea,baik->me', g[o, o, v, o], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,b||e,c>*l2(i,j,b,a)*t2(c,a,i,j)
    lambda_one += 0.5 * einsum('mbec,ijba,caij->me', g[o, v, v, v], l2, t2,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 <j,b||e,c>*l2(i,m,b,a)*t2(c,a,i,j)
    lambda_one += -1.0 * einsum('jbec,imba,caij->me', g[o, v, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 <m,a||b,c>*l2(i,j,a,e)*t2(b,c,i,j)
    lambda_one += 0.25 * einsum('mabc,ijae,bcij->me', g[o, v, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,m||e,b>*l1(i,a)*t1(b,i)*t1(a,j)
    lambda_one += 1.0 * einsum('jmeb,ia,bi,aj->me', g[o, o, v, v], l1, t1, t1,
                               optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])

    #	 -1.0000 <j,i||e,b>*l1(m,a)*t1(b,i)*t1(a,j)
    lambda_one += -1.0 * einsum('jieb,ma,bi,aj->me', g[o, o, v, v], l1, t1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1),
                                          (0, 1)])

    #	 -1.0000 <j,m||a,b>*l1(i,e)*t1(a,j)*t1(b,i)
    lambda_one += -1.0 * einsum('jmab,ie,aj,bi->me', g[o, o, v, v], l1, t1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1),
                                          (0, 1)])

    #	 -0.5000 <k,j||e,i>*l2(m,i,b,a)*t1(b,j)*t1(a,k)
    lambda_one += -0.5 * einsum('kjei,miba,bj,ak->me', g[o, o, v, o], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	 -1.0000 <k,m||b,j>*l2(i,j,e,a)*t1(b,i)*t1(a,k)
    lambda_one += -1.0 * einsum('kmbj,ijea,bi,ak->me', g[o, o, v, o], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	 -1.0000 <j,b||e,c>*l2(i,m,b,a)*t1(c,i)*t1(a,j)
    lambda_one += -1.0 * einsum('jbec,imba,ci,aj->me', g[o, v, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	 -0.5000 <m,a||b,c>*l2(i,j,a,e)*t1(b,j)*t1(c,i)
    lambda_one += -0.5 * einsum('mabc,ijae,bj,ci->me', g[o, v, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	  0.5000 <k,m||e,c>*l2(i,j,b,a)*t1(c,j)*t2(b,a,i,k)
    lambda_one += 0.5 * einsum('kmec,ijba,cj,baik->me', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])

    #	  0.5000 <k,m||e,c>*l2(i,j,b,a)*t1(b,k)*t2(c,a,i,j)
    lambda_one += 0.5 * einsum('kmec,ijba,bk,caij->me', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])

    #	 -0.5000 <k,j||e,c>*l2(i,m,b,a)*t1(c,j)*t2(b,a,i,k)
    lambda_one += -0.5 * einsum('kjec,imba,cj,baik->me', g[o, o, v, v], l2, t1,
                                t2, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	 -0.2500 <k,j||e,c>*l2(i,m,b,a)*t1(c,i)*t2(b,a,k,j)
    lambda_one += -0.25 * einsum('kjec,imba,ci,bakj->me', g[o, o, v, v], l2, t1,
                                 t2, optimize=['einsum_path', (0, 2), (0, 1),
                                               (0, 1)])

    #	  1.0000 <k,j||e,c>*l2(i,m,b,a)*t1(b,j)*t2(c,a,i,k)
    lambda_one += 1.0 * einsum('kjec,imba,bj,caik->me', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])

    #	 -0.5000 <k,m||b,c>*l2(i,j,e,a)*t1(b,k)*t2(c,a,i,j)
    lambda_one += -0.5 * einsum('kmbc,ijea,bk,caij->me', g[o, o, v, v], l2, t1,
                                t2, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	  1.0000 <k,m||b,c>*l2(i,j,e,a)*t1(b,j)*t2(c,a,i,k)
    lambda_one += 1.0 * einsum('kmbc,ijea,bj,caik->me', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])

    #	 -0.2500 <k,m||b,c>*l2(i,j,e,a)*t1(a,k)*t2(b,c,i,j)
    lambda_one += -0.25 * einsum('kmbc,ijea,ak,bcij->me', g[o, o, v, v], l2, t1,
                                 t2, optimize=['einsum_path', (0, 2), (0, 1),
                                               (0, 1)])

    #	  0.5000 <k,j||e,c>*l2(i,m,b,a)*t1(c,i)*t1(b,j)*t1(a,k)
    lambda_one += 0.5 * einsum('kjec,imba,ci,bj,ak->me', g[o, o, v, v], l2, t1,
                               t1, t1,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1),
                                         (0, 1)])

    #	  0.5000 <k,m||b,c>*l2(i,j,e,a)*t1(b,j)*t1(c,i)*t1(a,k)
    lambda_one += 0.5 * einsum('kmbc,ijea,bj,ci,ak->me', g[o, o, v, v], l2, t1,
                               t1, t1,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1),
                                         (0, 1)])
    return lambda_one


def lambda_doubles(t1, t2, l1, l2, f, g, o, v):
    """
    Lagrangian derivative with respect to t2

    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param l1: lagrange multiplier for singles (nocc x nvirt)
    :param l2: lagrange multiplier for doubles (nocc x nocc x nvirt x nvirt)
    :param f: fock operator defined as soei + np.einsum('piiq->pq', astei[:, o, o, :])
              where soei is 1 electron integrals (spinorb) and astei is
              antisymmetric 2 electron integrals in openfermion format
              <12|21>.  <ij|kl> - <ij|lk>
    :param g: antisymmetric 2 electron integrals. See fock input.
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    """
    #	  1.0000 <m,n||e,f>
    lambda_two = 1.0 * einsum('mnef->mnef', g[o, o, v, v])

    #	 -1.0000 P(m,n)*P(e,f)f(n,e)*l1(m,f)
    contracted_intermediate = -1.0 * einsum('ne,mf->mnef', f[o, v], l1)
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	 -1.0000 P(m,n)f(n,i)*l2(m,i,e,f)
    contracted_intermediate = -1.0 * einsum('ni,mief->mnef', f[o, o], l2)
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	  1.0000 P(e,f)f(a,e)*l2(m,n,a,f)
    contracted_intermediate = 1.0 * einsum('ae,mnaf->mnef', f[v, v], l2)
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	  1.0000 P(e,f)f(i,e)*l2(m,n,f,a)*t1(a,i)
    contracted_intermediate = 1.0 * einsum('ie,mnfa,ai->mnef', f[o, v], l2, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	  1.0000 P(m,n)f(n,a)*l2(i,m,e,f)*t1(a,i)
    contracted_intermediate = 1.0 * einsum('na,imef,ai->mnef', f[o, v], l2, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	 -1.0000 P(e,f)<m,n||e,i>*l1(i,f)
    contracted_intermediate = -1.0 * einsum('mnei,if->mnef', g[o, o, v, o], l1)
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	 -1.0000 P(m,n)<n,a||e,f>*l1(m,a)
    contracted_intermediate = -1.0 * einsum('naef,ma->mnef', g[o, v, v, v], l1)
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	  0.5000 <m,n||i,j>*l2(i,j,e,f)
    lambda_two += 0.5 * einsum('mnij,ijef->mnef', g[o, o, o, o], l2)

    #	  1.0000 P(m,n)*P(e,f)<n,a||e,i>*l2(m,i,a,f)
    contracted_intermediate = 1.0 * einsum('naei,miaf->mnef', g[o, v, v, o], l2)
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	  0.5000 <b,a||e,f>*l2(m,n,b,a)
    lambda_two += 0.5 * einsum('baef,mnba->mnef', g[v, v, v, v], l2)

    #	 -1.0000 P(m,n)<i,n||e,f>*l1(m,a)*t1(a,i)
    contracted_intermediate = -1.0 * einsum('inef,ma,ai->mnef', g[o, o, v, v],
                                            l1, t1,
                                            optimize=['einsum_path', (1, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	 -1.0000 P(e,f)<m,n||e,a>*l1(i,f)*t1(a,i)
    contracted_intermediate = -1.0 * einsum('mnea,if,ai->mnef', g[o, o, v, v],
                                            l1, t1,
                                            optimize=['einsum_path', (1, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	  1.0000 P(m,n)*P(e,f)<i,n||e,a>*l1(m,f)*t1(a,i)
    contracted_intermediate = 1.0 * einsum('inea,mf,ai->mnef', g[o, o, v, v],
                                           l1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	 -1.0000 P(m,n)*P(e,f)<j,n||e,i>*l2(m,i,f,a)*t1(a,j)
    contracted_intermediate = -1.0 * einsum('jnei,mifa,aj->mnef', g[o, o, v, o],
                                            l2, t1,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	  1.0000 <m,n||a,j>*l2(i,j,e,f)*t1(a,i)
    lambda_two += 1.0 * einsum('mnaj,ijef,ai->mnef', g[o, o, v, o], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(m,n)<j,n||a,i>*l2(m,i,e,f)*t1(a,j)
    contracted_intermediate = -1.0 * einsum('jnai,mief,aj->mnef', g[o, o, v, o],
                                            l2, t1,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	  1.0000 <i,b||e,f>*l2(m,n,b,a)*t1(a,i)
    lambda_two += 1.0 * einsum('ibef,mnba,ai->mnef', g[o, v, v, v], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(m,n)*P(e,f)<n,a||e,b>*l2(i,m,a,f)*t1(b,i)
    contracted_intermediate = -1.0 * einsum('naeb,imaf,bi->mnef', g[o, v, v, v],
                                            l2, t1,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	 -1.0000 P(e,f)<i,a||e,b>*l2(m,n,a,f)*t1(b,i)
    contracted_intermediate = -1.0 * einsum('iaeb,mnaf,bi->mnef', g[o, v, v, v],
                                            l2, t1,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	 -0.5000 P(m,n)<j,n||e,f>*l2(i,m,b,a)*t2(b,a,i,j)
    contracted_intermediate = -0.5 * einsum('jnef,imba,baij->mnef',
                                            g[o, o, v, v], l2, t2,
                                            optimize=['einsum_path', (1, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	  0.2500 <j,i||e,f>*l2(m,n,b,a)*t2(b,a,j,i)
    lambda_two += 0.25 * einsum('jief,mnba,baji->mnef', g[o, o, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 P(e,f)<m,n||e,b>*l2(i,j,f,a)*t2(b,a,i,j)
    contracted_intermediate = -0.5 * einsum('mneb,ijfa,baij->mnef',
                                            g[o, o, v, v], l2, t2,
                                            optimize=['einsum_path', (1, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	  1.0000 P(m,n)*P(e,f)<j,n||e,b>*l2(i,m,f,a)*t2(b,a,i,j)
    contracted_intermediate = 1.0 * einsum('jneb,imfa,baij->mnef',
                                           g[o, o, v, v], l2, t2,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	 -0.5000 P(e,f)<j,i||e,b>*l2(m,n,f,a)*t2(b,a,j,i)
    contracted_intermediate = -0.5 * einsum('jieb,mnfa,baji->mnef',
                                            g[o, o, v, v], l2, t2,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	  0.2500 <m,n||a,b>*l2(i,j,e,f)*t2(a,b,i,j)
    lambda_two += 0.25 * einsum('mnab,ijef,abij->mnef', g[o, o, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 P(m,n)<j,n||a,b>*l2(i,m,e,f)*t2(a,b,i,j)
    contracted_intermediate = -0.5 * einsum('jnab,imef,abij->mnef',
                                            g[o, o, v, v], l2, t2,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	 -0.5000 <j,i||e,f>*l2(m,n,b,a)*t1(b,i)*t1(a,j)
    lambda_two += -0.5 * einsum('jief,mnba,bi,aj->mnef', g[o, o, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	  1.0000 P(m,n)*P(e,f)<j,n||e,b>*l2(i,m,f,a)*t1(b,i)*t1(a,j)
    contracted_intermediate = 1.0 * einsum('jneb,imfa,bi,aj->mnef',
                                           g[o, o, v, v], l2, t1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1), (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	  1.0000 P(e,f)<j,i||e,b>*l2(m,n,f,a)*t1(b,i)*t1(a,j)
    contracted_intermediate = 1.0 * einsum('jieb,mnfa,bi,aj->mnef',
                                           g[o, o, v, v], l2, t1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (1, 2), (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	 -0.5000 <m,n||a,b>*l2(i,j,e,f)*t1(a,j)*t1(b,i)
    lambda_two += -0.5 * einsum('mnab,ijef,aj,bi->mnef', g[o, o, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	  1.0000 P(m,n)<j,n||a,b>*l2(i,m,e,f)*t1(a,j)*t1(b,i)
    contracted_intermediate = 1.0 * einsum('jnab,imef,aj,bi->mnef',
                                           g[o, o, v, v], l2, t1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (1, 2), (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)
    return lambda_two



def ccsd_d1(t1, t2, l1, l2, kd, o, v):
    """
    Compute CCSD 1-RDM

    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param l1: lagrange multiplier for singles (nocc x nvirt)
    :param l2: lagrange multiplier for doubles (nocc x nocc x nvirt x nvirt)
    :param kd: identity matrix (|spin-orb| x |spin-orb|)
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    """
    opdm = np.zeros_like(kd)

    #    D1(m,n):
    # 	  1.0000 d(m,n)
    # 	 ['+1.000000', 'd(m,n)']
    opdm[o, o] += 1.0 * einsum('mn->mn', kd[o, o])

    # 	 -1.0000 l1(n,a)*t1(a,m)
    # 	 ['-1.000000', 'l1(n,a)', 't1(a,m)']
    opdm[o, o] += -1.0 * einsum('na,am->mn', l1, t1)

    # 	 -0.5000 l2(i,n,b,a)*t2(b,a,i,m)
    # 	 ['-0.500000', 'l2(i,n,b,a)', 't2(b,a,i,m)']
    opdm[o, o] += -0.5 * einsum('inba,baim->mn', l2, t2)

    #    D1(e,f):

    #	  1.0000 l1(i,e)*t1(f,i)
    #	 ['+1.000000', 'l1(i,e)', 't1(f,i)']
    opdm[v, v] += 1.0 * einsum('ie,fi->ef', l1, t1)

    #	  0.5000 l2(i,j,e,a)*t2(f,a,i,j)
    #	 ['+0.500000', 'l2(i,j,e,a)', 't2(f,a,i,j)']
    opdm[v, v] += 0.5 * einsum('ijea,faij->ef', l2, t2)

    #    D1(e,m):

    #	  1.0000 l1(m,e)
    #	 ['+1.000000', 'l1(m,e)']
    opdm[v, o] += 1.0 * einsum('me->em', l1)

    #    D1(m,e):

    #	  1.0000 t1(e,m)
    #	 ['+1.000000', 't1(e,m)']
    opdm[o, v] += 1.0 * einsum('em->me', t1)

    #	 -1.0000 l1(i,a)*t2(e,a,i,m)
    #	 ['-1.000000', 'l1(i,a)', 't2(e,a,i,m)']
    opdm[o, v] += -1.0 * einsum('ia,eaim->me', l1, t2)

    #	 -1.0000 l1(i,a)*t1(e,i)*t1(a,m)
    #	 ['-1.000000', 'l1(i,a)', 't1(e,i)', 't1(a,m)']
    opdm[o, v] += -1.0 * einsum('ia,ei,am->me', l1, t1, t1,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 l2(i,j,b,a)*t1(e,j)*t2(b,a,i,m)
    #	 ['-0.500000', 'l2(i,j,b,a)', 't1(e,j)', 't2(b,a,i,m)']
    opdm[o, v] += -0.5 * einsum('ijba,ej,baim->me', l2, t1, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 l2(i,j,b,a)*t1(b,m)*t2(e,a,i,j)
    #	 ['-0.500000', 'l2(i,j,b,a)', 't1(b,m)', 't2(e,a,i,j)']
    opdm[o, v] += -0.5 * einsum('ijba,bm,eaij->me', l2, t1, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    return opdm


def ccsd_d2(t1, t2, l1, l2, kd, o, v):
    """
    Compute CCSD 2-RDM

    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param l1: lagrange multiplier for singles (nocc x nvirt)
    :param l2: lagrange multiplier for doubles (nocc x nocc x nvirt x nvirt)
    :param kd: identity matrix (|spin-orb| x |spin-orb|)
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    """
    nso = kd.shape[0]
    tpdm = np.zeros((nso, nso, nso, nso))
    #    D2(i,j,k,l):

    #	 ['+1.000000', 'd(j,l)', 'd(i,k)']
    #	  1.0000 d(j,l)*d(i,k)
    tpdm[o, o, o, o] += 1.0 * einsum('jl,ik->ijkl', kd[o, o], kd[o, o])

    #	 ['-1.000000', 'd(i,l)', 'd(j,k)']
    #	 -1.0000 d(i,l)*d(j,k)
    tpdm[o, o, o, o] += -1.0 * einsum('il,jk->ijkl', kd[o, o], kd[o, o])

    #	 ['-1.000000', 'd(j,l)', 'l1(k,a)', 't1(a,i)']
    #	 -1.0000 d(j,l)*l1(k,a)*t1(a,i)
    tpdm[o, o, o, o] += -1.0 * einsum('jl,ka,ai->ijkl', kd[o, o], l1, t1,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['+1.000000', 'd(i,l)', 'l1(k,a)', 't1(a,j)']
    #	  1.0000 d(i,l)*l1(k,a)*t1(a,j)
    tpdm[o, o, o, o] += 1.0 * einsum('il,ka,aj->ijkl', kd[o, o], l1, t1,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['+1.000000', 'd(j,k)', 'l1(l,a)', 't1(a,i)']
    #	  1.0000 d(j,k)*l1(l,a)*t1(a,i)
    tpdm[o, o, o, o] += 1.0 * einsum('jk,la,ai->ijkl', kd[o, o], l1, t1,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['-1.000000', 'd(i,k)', 'l1(l,a)', 't1(a,j)']
    #	 -1.0000 d(i,k)*l1(l,a)*t1(a,j)
    tpdm[o, o, o, o] += -1.0 * einsum('ik,la,aj->ijkl', kd[o, o], l1, t1,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['-0.500000', 'd(j,l)', 'l2(m,k,b,a)', 't2(b,a,m,i)']
    #	 -0.5000 d(j,l)*l2(m,k,b,a)*t2(b,a,m,i)
    tpdm[o, o, o, o] += -0.5 * einsum('jl,mkba,bami->ijkl', kd[o, o], l2, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['+0.500000', 'd(i,l)', 'l2(m,k,b,a)', 't2(b,a,m,j)']
    #	  0.5000 d(i,l)*l2(m,k,b,a)*t2(b,a,m,j)
    tpdm[o, o, o, o] += 0.5 * einsum('il,mkba,bamj->ijkl', kd[o, o], l2, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['+0.500000', 'd(j,k)', 'l2(m,l,b,a)', 't2(b,a,m,i)']
    #	  0.5000 d(j,k)*l2(m,l,b,a)*t2(b,a,m,i)
    tpdm[o, o, o, o] += 0.5 * einsum('jk,mlba,bami->ijkl', kd[o, o], l2, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['-0.500000', 'd(i,k)', 'l2(m,l,b,a)', 't2(b,a,m,j)']
    #	 -0.5000 d(i,k)*l2(m,l,b,a)*t2(b,a,m,j)
    tpdm[o, o, o, o] += -0.5 * einsum('ik,mlba,bamj->ijkl', kd[o, o], l2, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	 ['+0.500000', 'l2(k,l,b,a)', 't2(b,a,i,j)']
    #	  0.5000 l2(k,l,b,a)*t2(b,a,i,j)
    tpdm[o, o, o, o] += 0.5 * einsum('klba,baij->ijkl', l2, t2)

    #	 ['-1.000000', 'l2(k,l,b,a)', 't1(b,j)', 't1(a,i)']
    #	 -1.0000 l2(k,l,b,a)*t1(b,j)*t1(a,i)
    tpdm[o, o, o, o] += -1.0 * einsum('klba,bj,ai->ijkl', l2, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #    D2(i,j,k,a):

    #	 -1.0000 d(j,k)*t1(a,i)
    tpdm[o, o, o, v] += -1.0 * einsum('jk,ai->ijka', kd[o, o], t1)

    #	  1.0000 d(i,k)*t1(a,j)
    tpdm[o, o, o, v] += 1.0 * einsum('ik,aj->ijka', kd[o, o], t1)

    #	  1.0000 d(j,k)*l1(l,b)*t2(a,b,l,i)
    tpdm[o, o, o, v] += 1.0 * einsum('jk,lb,abli->ijka', kd[o, o], l1, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 d(i,k)*l1(l,b)*t2(a,b,l,j)
    tpdm[o, o, o, v] += -1.0 * einsum('ik,lb,ablj->ijka', kd[o, o], l1, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 l1(k,b)*t2(a,b,i,j)
    tpdm[o, o, o, v] += 1.0 * einsum('kb,abij->ijka', l1, t2)

    #	  1.0000 d(j,k)*l1(l,b)*t1(a,l)*t1(b,i)
    tpdm[o, o, o, v] += 1.0 * einsum('jk,lb,al,bi->ijka', kd[o, o], l1, t1, t1,
                                     optimize=['einsum_path', (1, 2), (1, 2),
                                               (0, 1)])

    #	 -1.0000 d(i,k)*l1(l,b)*t1(a,l)*t1(b,j)
    tpdm[o, o, o, v] += -1.0 * einsum('ik,lb,al,bj->ijka', kd[o, o], l1, t1, t1,
                                      optimize=['einsum_path', (1, 2), (1, 2),
                                                (0, 1)])

    #	 -1.0000 P(i,j)l1(k,b)*t1(a,j)*t1(b,i)
    contracted_intermediate = -1.0 * einsum('kb,aj,bi->ijka', l1, t1, t1,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    tpdm[o, o, o, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijka->jika', contracted_intermediate)

    #	  0.5000 d(j,k)*l2(l,m,c,b)*t1(a,m)*t2(c,b,l,i)
    tpdm[o, o, o, v] += 0.5 * einsum('jk,lmcb,am,cbli->ijka', kd[o, o], l2, t1,
                                     t2,
                                     optimize=['einsum_path', (1, 3), (1, 2),
                                               (0, 1)])

    #	  0.5000 d(j,k)*l2(l,m,c,b)*t1(c,i)*t2(a,b,l,m)
    tpdm[o, o, o, v] += 0.5 * einsum('jk,lmcb,ci,ablm->ijka', kd[o, o], l2, t1,
                                     t2,
                                     optimize=['einsum_path', (1, 3), (1, 2),
                                               (0, 1)])

    #	 -0.5000 d(i,k)*l2(l,m,c,b)*t1(a,m)*t2(c,b,l,j)
    tpdm[o, o, o, v] += -0.5 * einsum('ik,lmcb,am,cblj->ijka', kd[o, o], l2, t1,
                                      t2,
                                      optimize=['einsum_path', (1, 3), (1, 2),
                                                (0, 1)])

    #	 -0.5000 d(i,k)*l2(l,m,c,b)*t1(c,j)*t2(a,b,l,m)
    tpdm[o, o, o, v] += -0.5 * einsum('ik,lmcb,cj,ablm->ijka', kd[o, o], l2, t1,
                                      t2,
                                      optimize=['einsum_path', (1, 3), (1, 2),
                                                (0, 1)])

    #	 -0.5000 P(i,j)l2(l,k,c,b)*t1(a,j)*t2(c,b,l,i)
    contracted_intermediate = -0.5 * einsum('lkcb,aj,cbli->ijka', l2, t1, t2,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    tpdm[o, o, o, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijka->jika', contracted_intermediate)

    #	 -0.5000 l2(l,k,c,b)*t1(a,l)*t2(c,b,i,j)
    tpdm[o, o, o, v] += -0.5 * einsum('lkcb,al,cbij->ijka', l2, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)l2(l,k,c,b)*t1(c,j)*t2(a,b,l,i)
    contracted_intermediate = 1.0 * einsum('lkcb,cj,abli->ijka', l2, t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    tpdm[o, o, o, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijka->jika', contracted_intermediate)

    #	  1.0000 l2(l,k,c,b)*t1(a,l)*t1(c,j)*t1(b,i)
    tpdm[o, o, o, v] += 1.0 * einsum('lkcb,al,cj,bi->ijka', l2, t1, t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #    D2(i,j,a,l):

    #	  1.0000 d(j,l)*t1(a,i)
    tpdm[o, o, v, o] += 1.0 * einsum('jl,ai->ijal', kd[o, o], t1)

    #	 -1.0000 d(i,l)*t1(a,j)
    tpdm[o, o, v, o] += -1.0 * einsum('il,aj->ijal', kd[o, o], t1)

    #	 -1.0000 d(j,l)*l1(k,b)*t2(a,b,k,i)
    tpdm[o, o, v, o] += -1.0 * einsum('jl,kb,abki->ijal', kd[o, o], l1, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 d(i,l)*l1(k,b)*t2(a,b,k,j)
    tpdm[o, o, v, o] += 1.0 * einsum('il,kb,abkj->ijal', kd[o, o], l1, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 l1(l,b)*t2(a,b,i,j)
    tpdm[o, o, v, o] += -1.0 * einsum('lb,abij->ijal', l1, t2)

    #	 -1.0000 d(j,l)*l1(k,b)*t1(a,k)*t1(b,i)
    tpdm[o, o, v, o] += -1.0 * einsum('jl,kb,ak,bi->ijal', kd[o, o], l1, t1, t1,
                                      optimize=['einsum_path', (1, 2), (1, 2),
                                                (0, 1)])

    #	  1.0000 d(i,l)*l1(k,b)*t1(a,k)*t1(b,j)
    tpdm[o, o, v, o] += 1.0 * einsum('il,kb,ak,bj->ijal', kd[o, o], l1, t1, t1,
                                     optimize=['einsum_path', (1, 2), (1, 2),
                                               (0, 1)])

    #	  1.0000 P(i,j)l1(l,b)*t1(a,j)*t1(b,i)
    contracted_intermediate = 1.0 * einsum('lb,aj,bi->ijal', l1, t1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    tpdm[o, o, v, o] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijal->jial', contracted_intermediate)

    #	 -0.5000 d(j,l)*l2(k,m,c,b)*t1(a,m)*t2(c,b,k,i)
    tpdm[o, o, v, o] += -0.5 * einsum('jl,kmcb,am,cbki->ijal', kd[o, o], l2, t1,
                                      t2,
                                      optimize=['einsum_path', (1, 3), (1, 2),
                                                (0, 1)])

    #	 -0.5000 d(j,l)*l2(k,m,c,b)*t1(c,i)*t2(a,b,k,m)
    tpdm[o, o, v, o] += -0.5 * einsum('jl,kmcb,ci,abkm->ijal', kd[o, o], l2, t1,
                                      t2,
                                      optimize=['einsum_path', (1, 3), (1, 2),
                                                (0, 1)])

    #	  0.5000 d(i,l)*l2(k,m,c,b)*t1(a,m)*t2(c,b,k,j)
    tpdm[o, o, v, o] += 0.5 * einsum('il,kmcb,am,cbkj->ijal', kd[o, o], l2, t1,
                                     t2,
                                     optimize=['einsum_path', (1, 3), (1, 2),
                                               (0, 1)])

    #	  0.5000 d(i,l)*l2(k,m,c,b)*t1(c,j)*t2(a,b,k,m)
    tpdm[o, o, v, o] += 0.5 * einsum('il,kmcb,cj,abkm->ijal', kd[o, o], l2, t1,
                                     t2,
                                     optimize=['einsum_path', (1, 3), (1, 2),
                                               (0, 1)])

    #	  0.5000 P(i,j)l2(k,l,c,b)*t1(a,j)*t2(c,b,k,i)
    contracted_intermediate = 0.5 * einsum('klcb,aj,cbki->ijal', l2, t1, t2,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    tpdm[o, o, v, o] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijal->jial', contracted_intermediate)

    #	  0.5000 l2(k,l,c,b)*t1(a,k)*t2(c,b,i,j)
    tpdm[o, o, v, o] += 0.5 * einsum('klcb,ak,cbij->ijal', l2, t1, t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 P(i,j)l2(k,l,c,b)*t1(c,j)*t2(a,b,k,i)
    contracted_intermediate = -1.0 * einsum('klcb,cj,abki->ijal', l2, t1, t2,
                                            optimize=['einsum_path', (0, 1),
                                                      (0, 1)])
    tpdm[o, o, v, o] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijal->jial', contracted_intermediate)

    #	 -1.0000 l2(k,l,c,b)*t1(a,k)*t1(c,j)*t1(b,i)
    tpdm[o, o, v, o] += -1.0 * einsum('klcb,ak,cj,bi->ijal', l2, t1, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #    D2(i,a,k,l):

    #	 -1.0000 d(i,l)*l1(k,a)
    tpdm[o, v, o, o] += -1.0 * einsum('il,ka->iakl', kd[o, o], l1)

    #	  1.0000 d(i,k)*l1(l,a)
    tpdm[o, v, o, o] += 1.0 * einsum('ik,la->iakl', kd[o, o], l1)

    #	  1.0000 l2(k,l,a,b)*t1(b,i)
    tpdm[o, v, o, o] += 1.0 * einsum('klab,bi->iakl', l2, t1)

    #    D2(a,j,k,l):

    #	  1.0000 d(j,l)*l1(k,a)
    tpdm[v, o, o, o] += 1.0 * einsum('jl,ka->ajkl', kd[o, o], l1)

    #	 -1.0000 d(j,k)*l1(l,a)
    tpdm[v, o, o, o] += -1.0 * einsum('jk,la->ajkl', kd[o, o], l1)

    #	 -1.0000 l2(k,l,a,b)*t1(b,j)
    tpdm[v, o, o, o] += -1.0 * einsum('klab,bj->ajkl', l2, t1)

    #    D2(a,b,c,d):

    #	  0.5000 l2(i,j,a,b)*t2(c,d,i,j)
    tpdm[v, v, v, v] += 0.5 * einsum('ijab,cdij->abcd', l2, t2)

    #	 -1.0000 l2(i,j,a,b)*t1(c,j)*t1(d,i)
    tpdm[v, v, v, v] += -1.0 * einsum('ijab,cj,di->abcd', l2, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #    D2(a,b,c,i):

    #	  1.0000 l2(j,i,a,b)*t1(c,j)
    tpdm[v, v, v, o] += 1.0 * einsum('jiab,cj->abci', l2, t1)

    #    D2(a,b,i,d):

    #	 -1.0000 l2(j,i,a,b)*t1(d,j)
    tpdm[v, v, o, v] += -1.0 * einsum('jiab,dj->abid', l2, t1)

    #    D2(i,b,c,d):

    #	 -1.0000 l1(j,b)*t2(c,d,j,i)
    tpdm[o, v, v, v] += -1.0 * einsum('jb,cdji->ibcd', l1, t2)

    #	  1.0000 P(c,d)l1(j,b)*t1(c,i)*t1(d,j)
    contracted_intermediate = 1.0 * einsum('jb,ci,dj->ibcd', l1, t1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    tpdm[o, v, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ibcd->ibdc', contracted_intermediate)

    #	  0.5000 P(c,d)l2(j,k,b,a)*t1(c,i)*t2(d,a,j,k)
    contracted_intermediate = 0.5 * einsum('jkba,ci,dajk->ibcd', l2, t1, t2,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    tpdm[o, v, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ibcd->ibdc', contracted_intermediate)

    #	 -1.0000 P(c,d)l2(j,k,b,a)*t1(c,k)*t2(d,a,j,i)
    contracted_intermediate = -1.0 * einsum('jkba,ck,daji->ibcd', l2, t1, t2,
                                            optimize=['einsum_path', (0, 1),
                                                      (0, 1)])
    tpdm[o, v, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ibcd->ibdc', contracted_intermediate)

    #	  0.5000 l2(j,k,b,a)*t1(a,i)*t2(c,d,j,k)
    tpdm[o, v, v, v] += 0.5 * einsum('jkba,ai,cdjk->ibcd', l2, t1, t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 l2(j,k,b,a)*t1(c,k)*t1(d,j)*t1(a,i)
    tpdm[o, v, v, v] += -1.0 * einsum('jkba,ck,dj,ai->ibcd', l2, t1, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #    D2(a,i,c,d):

    #	  1.0000 l1(j,a)*t2(c,d,j,i)
    tpdm[v, o, v, v] += 1.0 * einsum('ja,cdji->aicd', l1, t2)

    #	 -1.0000 P(c,d)l1(j,a)*t1(c,i)*t1(d,j)
    contracted_intermediate = -1.0 * einsum('ja,ci,dj->aicd', l1, t1, t1,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    tpdm[v, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'aicd->aidc', contracted_intermediate)

    #	 -0.5000 P(c,d)l2(j,k,a,b)*t1(c,i)*t2(d,b,j,k)
    contracted_intermediate = -0.5 * einsum('jkab,ci,dbjk->aicd', l2, t1, t2,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    tpdm[v, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'aicd->aidc', contracted_intermediate)

    #	  1.0000 P(c,d)l2(j,k,a,b)*t1(c,k)*t2(d,b,j,i)
    contracted_intermediate = 1.0 * einsum('jkab,ck,dbji->aicd', l2, t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    tpdm[v, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'aicd->aidc', contracted_intermediate)

    #	 -0.5000 l2(j,k,a,b)*t1(b,i)*t2(c,d,j,k)
    tpdm[v, o, v, v] += -0.5 * einsum('jkab,bi,cdjk->aicd', l2, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 l2(j,k,a,b)*t1(c,k)*t1(d,j)*t1(b,i)
    tpdm[v, o, v, v] += 1.0 * einsum('jkab,ck,dj,bi->aicd', l2, t1, t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #    D2(i,j,a,b):

    #	  1.0000 t2(a,b,i,j)
    tpdm[o, o, v, v] += 1.0 * einsum('abij->ijab', t2)

    #	 -1.0000 P(i,j)t1(a,j)*t1(b,i)
    contracted_intermediate = -1.0 * einsum('aj,bi->ijab', t1, t1)
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)l1(k,c)*t1(a,j)*t2(b,c,k,i)
    contracted_intermediate = 1.0 * einsum('kc,aj,bcki->ijab', l1, t1, t2,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	  1.0000 P(a,b)l1(k,c)*t1(a,k)*t2(b,c,i,j)
    contracted_intermediate = 1.0 * einsum('kc,ak,bcij->ijab', l1, t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->ijba', contracted_intermediate)

    #	  1.0000 P(i,j)l1(k,c)*t1(c,j)*t2(a,b,k,i)
    contracted_intermediate = 1.0 * einsum('kc,cj,abki->ijab', l1, t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate)

    #	 -0.5000 P(i,j)l2(k,l,d,c)*t2(a,b,l,j)*t2(d,c,k,i)
    contracted_intermediate = -0.5 * einsum('kldc,ablj,dcki->ijab', l2, t2, t2,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate)

    #	  0.2500 l2(k,l,d,c)*t2(a,b,k,l)*t2(d,c,i,j)
    tpdm[o, o, v, v] += 0.25 * einsum('kldc,abkl,dcij->ijab', l2, t2, t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 l2(k,l,d,c)*t2(a,d,i,j)*t2(b,c,k,l)
    tpdm[o, o, v, v] += -0.5 * einsum('kldc,adij,bckl->ijab', l2, t2, t2,
                                      optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 P(i,j)l2(k,l,d,c)*t2(a,d,l,j)*t2(b,c,k,i)
    contracted_intermediate = 1.0 * einsum('kldc,adlj,bcki->ijab', l2, t2, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate)

    #	 -0.5000 l2(k,l,d,c)*t2(a,d,k,l)*t2(b,c,i,j)
    tpdm[o, o, v, v] += -0.5 * einsum('kldc,adkl,bcij->ijab', l2, t2, t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)l1(k,c)*t1(a,j)*t1(b,k)*t1(c,i)
    contracted_intermediate = 1.0 * einsum('kc,aj,bk,ci->ijab', l1, t1, t1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (1, 2), (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	  0.5000 P(i,j)*P(a,b)l2(k,l,d,c)*t1(a,j)*t1(b,l)*t2(d,c,k,i)
    contracted_intermediate = 0.5 * einsum('kldc,aj,bl,dcki->ijab', l2, t1, t1,
                                           t2, optimize=['einsum_path', (0, 3),
                                                         (1, 2), (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	  0.5000 P(i,j)*P(a,b)l2(k,l,d,c)*t1(a,j)*t1(d,i)*t2(b,c,k,l)
    contracted_intermediate = 0.5 * einsum('kldc,aj,di,bckl->ijab', l2, t1, t1,
                                           t2, optimize=['einsum_path', (0, 3),
                                                         (1, 2), (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	 -0.5000 l2(k,l,d,c)*t1(a,l)*t1(b,k)*t2(d,c,i,j)
    tpdm[o, o, v, v] += -0.5 * einsum('kldc,al,bk,dcij->ijab', l2, t1, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)l2(k,l,d,c)*t1(a,l)*t1(d,j)*t2(b,c,k,i)
    contracted_intermediate = 1.0 * einsum('kldc,al,dj,bcki->ijab', l2, t1, t1,
                                           t2, optimize=['einsum_path', (0, 1),
                                                         (0, 2), (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	 -0.5000 l2(k,l,d,c)*t1(d,j)*t1(c,i)*t2(a,b,k,l)
    tpdm[o, o, v, v] += -0.5 * einsum('kldc,dj,ci,abkl->ijab', l2, t1, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #	  1.0000 l2(k,l,d,c)*t1(a,l)*t1(b,k)*t1(d,j)*t1(c,i)
    tpdm[o, o, v, v] += 1.0 * einsum('kldc,al,bk,dj,ci->ijab', l2, t1, t1, t1,
                                     t1,
                                     optimize=['einsum_path', (0, 1), (0, 3),
                                               (0, 2), (0, 1)])

    #    D2(a,b,i,j):

    #	  1.0000 l2(i,j,a,b)
    tpdm[v, v, o, o] += 1.0 * einsum('ijab->abij', l2)

    #    D2(i,a,j,b):

    #	  1.0000 d(i,j)*l1(k,a)*t1(b,k)
    tpdm[o, v, o, v] += 1.0 * einsum('ij,ka,bk->iajb', kd[o, o], l1, t1,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 l1(j,a)*t1(b,i)
    tpdm[o, v, o, v] += -1.0 * einsum('ja,bi->iajb', l1, t1)

    #	  0.5000 d(i,j)*l2(k,l,a,c)*t2(b,c,k,l)
    tpdm[o, v, o, v] += 0.5 * einsum('ij,klac,bckl->iajb', kd[o, o], l2, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 l2(k,j,a,c)*t2(b,c,k,i)
    tpdm[o, v, o, v] += -1.0 * einsum('kjac,bcki->iajb', l2, t2)

    #	 -1.0000 l2(k,j,a,c)*t1(b,k)*t1(c,i)
    tpdm[o, v, o, v] += -1.0 * einsum('kjac,bk,ci->iajb', l2, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #    D2(a,i,j,b):

    #	 -1.0000 d(i,j)*l1(k,a)*t1(b,k)
    tpdm[v, o, o, v] += -1.0 * einsum('ij,ka,bk->aijb', kd[o, o], l1, t1,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 l1(j,a)*t1(b,i)
    tpdm[v, o, o, v] += 1.0 * einsum('ja,bi->aijb', l1, t1)

    #	 -0.5000 d(i,j)*l2(k,l,a,c)*t2(b,c,k,l)
    tpdm[v, o, o, v] += -0.5 * einsum('ij,klac,bckl->aijb', kd[o, o], l2, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 l2(k,j,a,c)*t2(b,c,k,i)
    tpdm[v, o, o, v] += 1.0 * einsum('kjac,bcki->aijb', l2, t2)

    #	  1.0000 l2(k,j,a,c)*t1(b,k)*t1(c,i)
    tpdm[v, o, o, v] += 1.0 * einsum('kjac,bk,ci->aijb', l2, t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #    D2(i,a,b,j):

    #	 -1.0000 d(i,j)*l1(k,a)*t1(b,k)
    tpdm[o, v, v, o] += -1.0 * einsum('ij,ka,bk->iabj', kd[o, o], l1, t1,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 l1(j,a)*t1(b,i)
    tpdm[o, v, v, o] += 1.0 * einsum('ja,bi->iabj', l1, t1)

    #	 -0.5000 d(i,j)*l2(k,l,a,c)*t2(b,c,k,l)
    tpdm[o, v, v, o] += -0.5 * einsum('ij,klac,bckl->iabj', kd[o, o], l2, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 l2(k,j,a,c)*t2(b,c,k,i)
    tpdm[o, v, v, o] += 1.0 * einsum('kjac,bcki->iabj', l2, t2)

    #	  1.0000 l2(k,j,a,c)*t1(b,k)*t1(c,i)
    tpdm[o, v, v, o] += 1.0 * einsum('kjac,bk,ci->iabj', l2, t1, t1,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #    D2(a,i,b,j):

    #	  1.0000 d(i,j)*l1(k,a)*t1(b,k)
    tpdm[v, o, v, o] += 1.0 * einsum('ij,ka,bk->aibj', kd[o, o], l1, t1,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 l1(j,a)*t1(b,i)
    tpdm[v, o, v, o] += -1.0 * einsum('ja,bi->aibj', l1, t1)

    #	  0.5000 d(i,j)*l2(k,l,a,c)*t2(b,c,k,l)
    tpdm[v, o, v, o] += 0.5 * einsum('ij,klac,bckl->aibj', kd[o, o], l2, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 l2(k,j,a,c)*t2(b,c,k,i)
    tpdm[v, o, v, o] += -1.0 * einsum('kjac,bcki->aibj', l2, t2)

    #	 -1.0000 l2(k,j,a,c)*t1(b,k)*t1(c,i)
    tpdm[v, o, v, o] += -1.0 * einsum('kjac,bk,ci->aibj', l2, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    return tpdm


