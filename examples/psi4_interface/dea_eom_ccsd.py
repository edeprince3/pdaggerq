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
Functions to build DEA-EOM-CCSD Hamiltonian
"""

import numpy as np
from numpy import einsum

def dea_eom_ccsd_hamiltonian_22(kd, f, g, o, v, t1, t2):

    #    H(a,b;d,e) = <0|b a e(-T) H e(T) d* e*|0>
    
    #	  1.0000 d(a,d)*d(b,e)*f(i,i)
    H =  1.000000000000000 * einsum('ad,be,ii->abde', kd[v, v], kd[v, v], f[o, o], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*f(i,i)
    H += -1.000000000000000 * einsum('bd,ae,ii->abde', kd[v, v], kd[v, v], f[o, o], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 d(b,e)*f(a,d)
    H +=  1.000000000000000 * einsum('be,ad->abde', kd[v, v], f[v, v])
    
    #	 -1.0000 d(b,d)*f(a,e)
    H += -1.000000000000000 * einsum('bd,ae->abde', kd[v, v], f[v, v])
    
    #	 -1.0000 d(a,e)*f(b,d)
    H += -1.000000000000000 * einsum('ae,bd->abde', kd[v, v], f[v, v])
    
    #	  1.0000 d(a,d)*f(b,e)
    H +=  1.000000000000000 * einsum('ad,be->abde', kd[v, v], f[v, v])
    
    #	  1.0000 d(a,d)*d(b,e)*f(i,c)*t1(c,i)
    H +=  1.000000000000000 * einsum('ad,be,ic,ci->abde', kd[v, v], kd[v, v], f[o, v], t1, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*f(i,c)*t1(c,i)
    H += -1.000000000000000 * einsum('bd,ae,ic,ci->abde', kd[v, v], kd[v, v], f[o, v], t1, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,e)*f(i,d)*t1(a,i)
    H += -1.000000000000000 * einsum('be,id,ai->abde', kd[v, v], f[o, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*f(i,e)*t1(a,i)
    H +=  1.000000000000000 * einsum('bd,ie,ai->abde', kd[v, v], f[o, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(a,e)*f(i,d)*t1(b,i)
    H +=  1.000000000000000 * einsum('ae,id,bi->abde', kd[v, v], f[o, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*f(i,e)*t1(b,i)
    H += -1.000000000000000 * einsum('ad,ie,bi->abde', kd[v, v], f[o, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(a,d)*d(b,e)*<j,i||j,i>
    H += -0.500000000000000 * einsum('ad,be,jiji->abde', kd[v, v], kd[v, v], g[o, o, o, o], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 d(b,d)*d(a,e)*<j,i||j,i>
    H +=  0.500000000000000 * einsum('bd,ae,jiji->abde', kd[v, v], kd[v, v], g[o, o, o, o], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <a,b||d,e>
    H +=  1.000000000000000 * einsum('abde->abde', g[v, v, v, v])
    
    #	  1.0000 d(b,e)*<i,a||c,d>*t1(c,i)
    H +=  1.000000000000000 * einsum('be,iacd,ci->abde', kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*<i,a||c,e>*t1(c,i)
    H += -1.000000000000000 * einsum('bd,iace,ci->abde', kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 P(a,b)<i,a||d,e>*t1(b,i)
    contracted_intermediate =  1.000000000000000 * einsum('iade,bi->abde', g[o, v, v, v], t1)
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abde->bade', contracted_intermediate) 
    
    #	 -1.0000 d(a,e)*<i,b||c,d>*t1(c,i)
    H += -1.000000000000000 * einsum('ae,ibcd,ci->abde', kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*<i,b||c,e>*t1(c,i)
    H +=  1.000000000000000 * einsum('ad,ibce,ci->abde', kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.2500 d(a,d)*d(b,e)*<j,i||c,f>*t2(c,f,j,i)
    H +=  0.250000000000000 * einsum('ad,be,jicf,cfji->abde', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 d(b,d)*d(a,e)*<j,i||c,f>*t2(c,f,j,i)
    H += -0.250000000000000 * einsum('bd,ae,jicf,cfji->abde', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(b,e)*<j,i||c,d>*t2(c,a,j,i)
    H += -0.500000000000000 * einsum('be,jicd,caji->abde', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(b,d)*<j,i||c,e>*t2(c,a,j,i)
    H +=  0.500000000000000 * einsum('bd,jice,caji->abde', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(a,e)*<j,i||c,d>*t2(c,b,j,i)
    H +=  0.500000000000000 * einsum('ae,jicd,cbji->abde', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(a,d)*<j,i||c,e>*t2(c,b,j,i)
    H += -0.500000000000000 * einsum('ad,jice,cbji->abde', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 <j,i||d,e>*t2(a,b,j,i)
    H +=  0.500000000000000 * einsum('jide,abji->abde', g[o, o, v, v], t2)
    
    #	 -0.5000 d(a,d)*d(b,e)*<j,i||c,f>*t1(c,i)*t1(f,j)
    H += -0.500000000000000 * einsum('ad,be,jicf,ci,fj->abde', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (2, 3), (2, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(b,d)*d(a,e)*<j,i||c,f>*t1(c,i)*t1(f,j)
    H +=  0.500000000000000 * einsum('bd,ae,jicf,ci,fj->abde', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (2, 3), (2, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,e)*<j,i||c,d>*t1(c,i)*t1(a,j)
    H +=  1.000000000000000 * einsum('be,jicd,ci,aj->abde', kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*<j,i||c,e>*t1(c,i)*t1(a,j)
    H += -1.000000000000000 * einsum('bd,jice,ci,aj->abde', kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(a,e)*<j,i||c,d>*t1(c,i)*t1(b,j)
    H += -1.000000000000000 * einsum('ae,jicd,ci,bj->abde', kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*<j,i||c,e>*t1(c,i)*t1(b,j)
    H +=  1.000000000000000 * einsum('ad,jice,ci,bj->abde', kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 <j,i||d,e>*t1(a,i)*t1(b,j)
    H += -1.000000000000000 * einsum('jide,ai,bj->abde', g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    
    return H


def dea_eom_ccsd_hamiltonian_32(kd, f, g, o, v, t1, t2):

    #    H(a,b,c,i;d,e) = <0|i* c b a e(-T) H e(T) d* e*|0>
    
    #	  1.0000 d(b,d)*d(c,e)*f(a,i)
    H =  1.000000000000000 * einsum('bd,ce,ai->abcide', kd[v, v], kd[v, v], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(c,d)*d(b,e)*f(a,i)
    H += -1.000000000000000 * einsum('cd,be,ai->abcide', kd[v, v], kd[v, v], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(a,d)*d(c,e)*f(b,i)
    H += -1.000000000000000 * einsum('ad,ce,bi->abcide', kd[v, v], kd[v, v], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(c,d)*d(a,e)*f(b,i)
    H +=  1.000000000000000 * einsum('cd,ae,bi->abcide', kd[v, v], kd[v, v], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(a,d)*d(b,e)*f(c,i)
    H +=  1.000000000000000 * einsum('ad,be,ci->abcide', kd[v, v], kd[v, v], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(b,d)*d(a,e)*f(c,i)
    H += -1.000000000000000 * einsum('bd,ae,ci->abcide', kd[v, v], kd[v, v], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(b,d)*d(c,e)*f(j,i)*t1(a,j)
    H += -1.000000000000000 * einsum('bd,ce,ji,aj->abcide', kd[v, v], kd[v, v], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,d)*d(b,e)*f(j,i)*t1(a,j)
    H +=  1.000000000000000 * einsum('cd,be,ji,aj->abcide', kd[v, v], kd[v, v], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(a,d)*d(c,e)*f(j,i)*t1(b,j)
    H +=  1.000000000000000 * einsum('ad,ce,ji,bj->abcide', kd[v, v], kd[v, v], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(a,e)*f(j,i)*t1(b,j)
    H += -1.000000000000000 * einsum('cd,ae,ji,bj->abcide', kd[v, v], kd[v, v], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(b,e)*f(j,i)*t1(c,j)
    H += -1.000000000000000 * einsum('ad,be,ji,cj->abcide', kd[v, v], kd[v, v], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,d)*d(a,e)*f(j,i)*t1(c,j)
    H +=  1.000000000000000 * einsum('bd,ae,ji,cj->abcide', kd[v, v], kd[v, v], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,e)*f(a,f)*t1(f,i)
    H +=  1.000000000000000 * einsum('bd,ce,af,fi->abcide', kd[v, v], kd[v, v], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,e)*f(a,f)*t1(f,i)
    H += -1.000000000000000 * einsum('cd,be,af,fi->abcide', kd[v, v], kd[v, v], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,e)*f(b,f)*t1(f,i)
    H += -1.000000000000000 * einsum('ad,ce,bf,fi->abcide', kd[v, v], kd[v, v], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,e)*f(b,f)*t1(f,i)
    H +=  1.000000000000000 * einsum('cd,ae,bf,fi->abcide', kd[v, v], kd[v, v], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(a,d)*d(b,e)*f(c,f)*t1(f,i)
    H +=  1.000000000000000 * einsum('ad,be,cf,fi->abcide', kd[v, v], kd[v, v], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*f(c,f)*t1(f,i)
    H += -1.000000000000000 * einsum('bd,ae,cf,fi->abcide', kd[v, v], kd[v, v], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(c,e)*f(j,f)*t2(f,a,i,j)
    H += -1.000000000000000 * einsum('bd,ce,jf,faij->abcide', kd[v, v], kd[v, v], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,d)*d(b,e)*f(j,f)*t2(f,a,i,j)
    H +=  1.000000000000000 * einsum('cd,be,jf,faij->abcide', kd[v, v], kd[v, v], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(a,d)*d(c,e)*f(j,f)*t2(f,b,i,j)
    H +=  1.000000000000000 * einsum('ad,ce,jf,fbij->abcide', kd[v, v], kd[v, v], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(a,e)*f(j,f)*t2(f,b,i,j)
    H += -1.000000000000000 * einsum('cd,ae,jf,fbij->abcide', kd[v, v], kd[v, v], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(b,e)*f(j,f)*t2(f,c,i,j)
    H += -1.000000000000000 * einsum('ad,be,jf,fcij->abcide', kd[v, v], kd[v, v], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,d)*d(a,e)*f(j,f)*t2(f,c,i,j)
    H +=  1.000000000000000 * einsum('bd,ae,jf,fcij->abcide', kd[v, v], kd[v, v], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,e)*f(j,d)*t2(a,b,i,j)
    H += -1.000000000000000 * einsum('ce,jd,abij->abcide', kd[v, v], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*f(j,e)*t2(a,b,i,j)
    H +=  1.000000000000000 * einsum('cd,je,abij->abcide', kd[v, v], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(b,e)*f(j,d)*t2(a,c,i,j)
    H +=  1.000000000000000 * einsum('be,jd,acij->abcide', kd[v, v], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*f(j,e)*t2(a,c,i,j)
    H += -1.000000000000000 * einsum('bd,je,acij->abcide', kd[v, v], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(a,e)*f(j,d)*t2(b,c,i,j)
    H += -1.000000000000000 * einsum('ae,jd,bcij->abcide', kd[v, v], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*f(j,e)*t2(b,c,i,j)
    H +=  1.000000000000000 * einsum('ad,je,bcij->abcide', kd[v, v], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(c,e)*f(j,f)*t1(f,i)*t1(a,j)
    H += -1.000000000000000 * einsum('bd,ce,jf,fi,aj->abcide', kd[v, v], kd[v, v], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(b,e)*f(j,f)*t1(f,i)*t1(a,j)
    H +=  1.000000000000000 * einsum('cd,be,jf,fi,aj->abcide', kd[v, v], kd[v, v], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*d(c,e)*f(j,f)*t1(f,i)*t1(b,j)
    H +=  1.000000000000000 * einsum('ad,ce,jf,fi,bj->abcide', kd[v, v], kd[v, v], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(a,e)*f(j,f)*t1(f,i)*t1(b,j)
    H += -1.000000000000000 * einsum('cd,ae,jf,fi,bj->abcide', kd[v, v], kd[v, v], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(b,e)*f(j,f)*t1(f,i)*t1(c,j)
    H += -1.000000000000000 * einsum('ad,be,jf,fi,cj->abcide', kd[v, v], kd[v, v], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(a,e)*f(j,f)*t1(f,i)*t1(c,j)
    H +=  1.000000000000000 * einsum('bd,ae,jf,fi,cj->abcide', kd[v, v], kd[v, v], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,e)*<a,b||d,i>
    H += -1.000000000000000 * einsum('ce,abdi->abcide', kd[v, v], g[v, v, v, o])
    
    #	  1.0000 d(c,d)*<a,b||e,i>
    H +=  1.000000000000000 * einsum('cd,abei->abcide', kd[v, v], g[v, v, v, o])
    
    #	  1.0000 d(b,e)*<a,c||d,i>
    H +=  1.000000000000000 * einsum('be,acdi->abcide', kd[v, v], g[v, v, v, o])
    
    #	 -1.0000 d(b,d)*<a,c||e,i>
    H += -1.000000000000000 * einsum('bd,acei->abcide', kd[v, v], g[v, v, v, o])
    
    #	 -1.0000 d(a,e)*<b,c||d,i>
    H += -1.000000000000000 * einsum('ae,bcdi->abcide', kd[v, v], g[v, v, v, o])
    
    #	  1.0000 d(a,d)*<b,c||e,i>
    H +=  1.000000000000000 * einsum('ad,bcei->abcide', kd[v, v], g[v, v, v, o])
    
    #	  1.0000 d(b,d)*d(c,e)*<j,a||f,i>*t1(f,j)
    H +=  1.000000000000000 * einsum('bd,ce,jafi,fj->abcide', kd[v, v], kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,e)*<j,a||f,i>*t1(f,j)
    H += -1.000000000000000 * einsum('cd,be,jafi,fj->abcide', kd[v, v], kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 P(a,b)d(c,e)*<j,a||d,i>*t1(b,j)
    contracted_intermediate = -1.000000000000000 * einsum('ce,jadi,bj->abcide', kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->bacide', contracted_intermediate) 
    
    #	  1.0000 P(a,b)d(c,d)*<j,a||e,i>*t1(b,j)
    contracted_intermediate =  1.000000000000000 * einsum('cd,jaei,bj->abcide', kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->bacide', contracted_intermediate) 
    
    #	  1.0000 P(a,c)d(b,e)*<j,a||d,i>*t1(c,j)
    contracted_intermediate =  1.000000000000000 * einsum('be,jadi,cj->abcide', kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->cbaide', contracted_intermediate) 
    
    #	 -1.0000 P(a,c)d(b,d)*<j,a||e,i>*t1(c,j)
    contracted_intermediate = -1.000000000000000 * einsum('bd,jaei,cj->abcide', kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->cbaide', contracted_intermediate) 
    
    #	 -1.0000 d(a,d)*d(c,e)*<j,b||f,i>*t1(f,j)
    H += -1.000000000000000 * einsum('ad,ce,jbfi,fj->abcide', kd[v, v], kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,e)*<j,b||f,i>*t1(f,j)
    H +=  1.000000000000000 * einsum('cd,ae,jbfi,fj->abcide', kd[v, v], kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 P(b,c)d(a,e)*<j,b||d,i>*t1(c,j)
    contracted_intermediate = -1.000000000000000 * einsum('ae,jbdi,cj->abcide', kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->acbide', contracted_intermediate) 
    
    #	  1.0000 P(b,c)d(a,d)*<j,b||e,i>*t1(c,j)
    contracted_intermediate =  1.000000000000000 * einsum('ad,jbei,cj->abcide', kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->acbide', contracted_intermediate) 
    
    #	  1.0000 d(a,d)*d(b,e)*<j,c||f,i>*t1(f,j)
    H +=  1.000000000000000 * einsum('ad,be,jcfi,fj->abcide', kd[v, v], kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*<j,c||f,i>*t1(f,j)
    H += -1.000000000000000 * einsum('bd,ae,jcfi,fj->abcide', kd[v, v], kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,e)*<a,b||f,d>*t1(f,i)
    H +=  1.000000000000000 * einsum('ce,abfd,fi->abcide', kd[v, v], g[v, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*<a,b||f,e>*t1(f,i)
    H += -1.000000000000000 * einsum('cd,abfe,fi->abcide', kd[v, v], g[v, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(b,e)*<a,c||f,d>*t1(f,i)
    H += -1.000000000000000 * einsum('be,acfd,fi->abcide', kd[v, v], g[v, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*<a,c||f,e>*t1(f,i)
    H +=  1.000000000000000 * einsum('bd,acfe,fi->abcide', kd[v, v], g[v, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(a,e)*<b,c||f,d>*t1(f,i)
    H +=  1.000000000000000 * einsum('ae,bcfd,fi->abcide', kd[v, v], g[v, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*<b,c||f,e>*t1(f,i)
    H += -1.000000000000000 * einsum('ad,bcfe,fi->abcide', kd[v, v], g[v, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(b,d)*d(c,e)*<k,j||f,i>*t2(f,a,k,j)
    H += -0.500000000000000 * einsum('bd,ce,kjfi,fakj->abcide', kd[v, v], kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(c,d)*d(b,e)*<k,j||f,i>*t2(f,a,k,j)
    H +=  0.500000000000000 * einsum('cd,be,kjfi,fakj->abcide', kd[v, v], kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(a,d)*d(c,e)*<k,j||f,i>*t2(f,b,k,j)
    H +=  0.500000000000000 * einsum('ad,ce,kjfi,fbkj->abcide', kd[v, v], kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(c,d)*d(a,e)*<k,j||f,i>*t2(f,b,k,j)
    H += -0.500000000000000 * einsum('cd,ae,kjfi,fbkj->abcide', kd[v, v], kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(a,d)*d(b,e)*<k,j||f,i>*t2(f,c,k,j)
    H += -0.500000000000000 * einsum('ad,be,kjfi,fckj->abcide', kd[v, v], kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(b,d)*d(a,e)*<k,j||f,i>*t2(f,c,k,j)
    H +=  0.500000000000000 * einsum('bd,ae,kjfi,fckj->abcide', kd[v, v], kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(c,e)*<k,j||d,i>*t2(a,b,k,j)
    H += -0.500000000000000 * einsum('ce,kjdi,abkj->abcide', kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(c,d)*<k,j||e,i>*t2(a,b,k,j)
    H +=  0.500000000000000 * einsum('cd,kjei,abkj->abcide', kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(b,e)*<k,j||d,i>*t2(a,c,k,j)
    H +=  0.500000000000000 * einsum('be,kjdi,ackj->abcide', kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(b,d)*<k,j||e,i>*t2(a,c,k,j)
    H += -0.500000000000000 * einsum('bd,kjei,ackj->abcide', kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(a,e)*<k,j||d,i>*t2(b,c,k,j)
    H += -0.500000000000000 * einsum('ae,kjdi,bckj->abcide', kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(a,d)*<k,j||e,i>*t2(b,c,k,j)
    H +=  0.500000000000000 * einsum('ad,kjei,bckj->abcide', kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(b,d)*d(c,e)*<j,a||f,g>*t2(f,g,i,j)
    H += -0.500000000000000 * einsum('bd,ce,jafg,fgij->abcide', kd[v, v], kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(c,d)*d(b,e)*<j,a||f,g>*t2(f,g,i,j)
    H +=  0.500000000000000 * einsum('cd,be,jafg,fgij->abcide', kd[v, v], kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 P(a,b)d(c,e)*<j,a||f,d>*t2(f,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('ce,jafd,fbij->abcide', kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->bacide', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)d(c,d)*<j,a||f,e>*t2(f,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('cd,jafe,fbij->abcide', kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->bacide', contracted_intermediate) 
    
    #	 -1.0000 P(a,c)d(b,e)*<j,a||f,d>*t2(f,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('be,jafd,fcij->abcide', kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->cbaide', contracted_intermediate) 
    
    #	  1.0000 P(a,c)d(b,d)*<j,a||f,e>*t2(f,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('bd,jafe,fcij->abcide', kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->cbaide', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<j,a||d,e>*t2(b,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('jade,bcij->abcide', g[o, v, v, v], t2)
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->bacide', contracted_intermediate) 
    
    #	  0.5000 d(a,d)*d(c,e)*<j,b||f,g>*t2(f,g,i,j)
    H +=  0.500000000000000 * einsum('ad,ce,jbfg,fgij->abcide', kd[v, v], kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(c,d)*d(a,e)*<j,b||f,g>*t2(f,g,i,j)
    H += -0.500000000000000 * einsum('cd,ae,jbfg,fgij->abcide', kd[v, v], kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 P(b,c)d(a,e)*<j,b||f,d>*t2(f,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('ae,jbfd,fcij->abcide', kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->acbide', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)d(a,d)*<j,b||f,e>*t2(f,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('ad,jbfe,fcij->abcide', kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->acbide', contracted_intermediate) 
    
    #	 -0.5000 d(a,d)*d(b,e)*<j,c||f,g>*t2(f,g,i,j)
    H += -0.500000000000000 * einsum('ad,be,jcfg,fgij->abcide', kd[v, v], kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(b,d)*d(a,e)*<j,c||f,g>*t2(f,g,i,j)
    H +=  0.500000000000000 * einsum('bd,ae,jcfg,fgij->abcide', kd[v, v], kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 <j,c||d,e>*t2(a,b,i,j)
    H += -1.000000000000000 * einsum('jcde,abij->abcide', g[o, v, v, v], t2)
    
    #	  1.0000 d(b,d)*d(c,e)*<k,j||f,g>*t1(f,j)*t2(g,a,i,k)
    H +=  1.000000000000000 * einsum('bd,ce,kjfg,fj,gaik->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,e)*<k,j||f,g>*t1(f,j)*t2(g,a,i,k)
    H += -1.000000000000000 * einsum('cd,be,kjfg,fj,gaik->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,e)*<k,j||f,g>*t1(f,j)*t2(g,b,i,k)
    H += -1.000000000000000 * einsum('ad,ce,kjfg,fj,gbik->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,e)*<k,j||f,g>*t1(f,j)*t2(g,b,i,k)
    H +=  1.000000000000000 * einsum('cd,ae,kjfg,fj,gbik->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*d(b,e)*<k,j||f,g>*t1(f,j)*t2(g,c,i,k)
    H +=  1.000000000000000 * einsum('ad,be,kjfg,fj,gcik->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*<k,j||f,g>*t1(f,j)*t2(g,c,i,k)
    H += -1.000000000000000 * einsum('bd,ae,kjfg,fj,gcik->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,e)*<k,j||f,d>*t1(f,j)*t2(a,b,i,k)
    H +=  1.000000000000000 * einsum('ce,kjfd,fj,abik->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*<k,j||f,e>*t1(f,j)*t2(a,b,i,k)
    H += -1.000000000000000 * einsum('cd,kjfe,fj,abik->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(b,e)*<k,j||f,d>*t1(f,j)*t2(a,c,i,k)
    H += -1.000000000000000 * einsum('be,kjfd,fj,acik->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*<k,j||f,e>*t1(f,j)*t2(a,c,i,k)
    H +=  1.000000000000000 * einsum('bd,kjfe,fj,acik->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(a,e)*<k,j||f,d>*t1(f,j)*t2(b,c,i,k)
    H +=  1.000000000000000 * einsum('ae,kjfd,fj,bcik->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*<k,j||f,e>*t1(f,j)*t2(b,c,i,k)
    H += -1.000000000000000 * einsum('ad,kjfe,fj,bcik->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 d(b,d)*d(c,e)*<k,j||f,g>*t1(f,i)*t2(g,a,k,j)
    H +=  0.500000000000000 * einsum('bd,ce,kjfg,fi,gakj->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(c,d)*d(b,e)*<k,j||f,g>*t1(f,i)*t2(g,a,k,j)
    H += -0.500000000000000 * einsum('cd,be,kjfg,fi,gakj->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(a,d)*d(c,e)*<k,j||f,g>*t1(f,i)*t2(g,b,k,j)
    H += -0.500000000000000 * einsum('ad,ce,kjfg,fi,gbkj->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(c,d)*d(a,e)*<k,j||f,g>*t1(f,i)*t2(g,b,k,j)
    H +=  0.500000000000000 * einsum('cd,ae,kjfg,fi,gbkj->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(a,d)*d(b,e)*<k,j||f,g>*t1(f,i)*t2(g,c,k,j)
    H +=  0.500000000000000 * einsum('ad,be,kjfg,fi,gckj->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(b,d)*d(a,e)*<k,j||f,g>*t1(f,i)*t2(g,c,k,j)
    H += -0.500000000000000 * einsum('bd,ae,kjfg,fi,gckj->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(c,e)*<k,j||f,d>*t1(f,i)*t2(a,b,k,j)
    H +=  0.500000000000000 * einsum('ce,kjfd,fi,abkj->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 d(c,d)*<k,j||f,e>*t1(f,i)*t2(a,b,k,j)
    H += -0.500000000000000 * einsum('cd,kjfe,fi,abkj->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 d(b,e)*<k,j||f,d>*t1(f,i)*t2(a,c,k,j)
    H += -0.500000000000000 * einsum('be,kjfd,fi,ackj->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 d(b,d)*<k,j||f,e>*t1(f,i)*t2(a,c,k,j)
    H +=  0.500000000000000 * einsum('bd,kjfe,fi,ackj->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 d(a,e)*<k,j||f,d>*t1(f,i)*t2(b,c,k,j)
    H +=  0.500000000000000 * einsum('ae,kjfd,fi,bckj->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 d(a,d)*<k,j||f,e>*t1(f,i)*t2(b,c,k,j)
    H += -0.500000000000000 * einsum('ad,kjfe,fi,bckj->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  0.5000 d(b,d)*d(c,e)*<k,j||f,g>*t1(a,j)*t2(f,g,i,k)
    H +=  0.500000000000000 * einsum('bd,ce,kjfg,aj,fgik->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(c,d)*d(b,e)*<k,j||f,g>*t1(a,j)*t2(f,g,i,k)
    H += -0.500000000000000 * einsum('cd,be,kjfg,aj,fgik->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 P(a,b)d(c,e)*<k,j||f,d>*t1(a,j)*t2(f,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('ce,kjfd,aj,fbik->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->bacide', contracted_intermediate) 
    
    #	  1.0000 P(a,b)d(c,d)*<k,j||f,e>*t1(a,j)*t2(f,b,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('cd,kjfe,aj,fbik->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->bacide', contracted_intermediate) 
    
    #	  1.0000 P(a,c)d(b,e)*<k,j||f,d>*t1(a,j)*t2(f,c,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('be,kjfd,aj,fcik->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->cbaide', contracted_intermediate) 
    
    #	 -1.0000 P(a,c)d(b,d)*<k,j||f,e>*t1(a,j)*t2(f,c,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('bd,kjfe,aj,fcik->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->cbaide', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<k,j||d,e>*t1(a,j)*t2(b,c,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('kjde,aj,bcik->abcide', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->bacide', contracted_intermediate) 
    
    #	 -0.5000 d(a,d)*d(c,e)*<k,j||f,g>*t1(b,j)*t2(f,g,i,k)
    H += -0.500000000000000 * einsum('ad,ce,kjfg,bj,fgik->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(c,d)*d(a,e)*<k,j||f,g>*t1(b,j)*t2(f,g,i,k)
    H +=  0.500000000000000 * einsum('cd,ae,kjfg,bj,fgik->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 P(b,c)d(a,e)*<k,j||f,d>*t1(b,j)*t2(f,c,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('ae,kjfd,bj,fcik->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->acbide', contracted_intermediate) 
    
    #	  1.0000 P(b,c)d(a,d)*<k,j||f,e>*t1(b,j)*t2(f,c,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('ad,kjfe,bj,fcik->abcide', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->acbide', contracted_intermediate) 
    
    #	  0.5000 d(a,d)*d(b,e)*<k,j||f,g>*t1(c,j)*t2(f,g,i,k)
    H +=  0.500000000000000 * einsum('ad,be,kjfg,cj,fgik->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(b,d)*d(a,e)*<k,j||f,g>*t1(c,j)*t2(f,g,i,k)
    H += -0.500000000000000 * einsum('bd,ae,kjfg,cj,fgik->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 <k,j||d,e>*t1(c,j)*t2(a,b,i,k)
    H +=  1.000000000000000 * einsum('kjde,cj,abik->abcide', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,e)*<k,j||f,i>*t1(f,j)*t1(a,k)
    H +=  1.000000000000000 * einsum('bd,ce,kjfi,fj,ak->abcide', kd[v, v], kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,e)*<k,j||f,i>*t1(f,j)*t1(a,k)
    H += -1.000000000000000 * einsum('cd,be,kjfi,fj,ak->abcide', kd[v, v], kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,e)*<k,j||f,i>*t1(f,j)*t1(b,k)
    H += -1.000000000000000 * einsum('ad,ce,kjfi,fj,bk->abcide', kd[v, v], kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,e)*<k,j||f,i>*t1(f,j)*t1(b,k)
    H +=  1.000000000000000 * einsum('cd,ae,kjfi,fj,bk->abcide', kd[v, v], kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*d(b,e)*<k,j||f,i>*t1(f,j)*t1(c,k)
    H +=  1.000000000000000 * einsum('ad,be,kjfi,fj,ck->abcide', kd[v, v], kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*<k,j||f,i>*t1(f,j)*t1(c,k)
    H += -1.000000000000000 * einsum('bd,ae,kjfi,fj,ck->abcide', kd[v, v], kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,e)*<k,j||d,i>*t1(a,j)*t1(b,k)
    H +=  1.000000000000000 * einsum('ce,kjdi,aj,bk->abcide', kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*<k,j||e,i>*t1(a,j)*t1(b,k)
    H += -1.000000000000000 * einsum('cd,kjei,aj,bk->abcide', kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(b,e)*<k,j||d,i>*t1(a,j)*t1(c,k)
    H += -1.000000000000000 * einsum('be,kjdi,aj,ck->abcide', kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*<k,j||e,i>*t1(a,j)*t1(c,k)
    H +=  1.000000000000000 * einsum('bd,kjei,aj,ck->abcide', kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(a,e)*<k,j||d,i>*t1(b,j)*t1(c,k)
    H +=  1.000000000000000 * einsum('ae,kjdi,bj,ck->abcide', kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*<k,j||e,i>*t1(b,j)*t1(c,k)
    H += -1.000000000000000 * einsum('ad,kjei,bj,ck->abcide', kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,e)*<j,a||f,g>*t1(f,j)*t1(g,i)
    H +=  1.000000000000000 * einsum('bd,ce,jafg,fj,gi->abcide', kd[v, v], kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,e)*<j,a||f,g>*t1(f,j)*t1(g,i)
    H += -1.000000000000000 * einsum('cd,be,jafg,fj,gi->abcide', kd[v, v], kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(a,b)d(c,e)*<j,a||f,d>*t1(f,i)*t1(b,j)
    contracted_intermediate =  1.000000000000000 * einsum('ce,jafd,fi,bj->abcide', kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->bacide', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)d(c,d)*<j,a||f,e>*t1(f,i)*t1(b,j)
    contracted_intermediate = -1.000000000000000 * einsum('cd,jafe,fi,bj->abcide', kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->bacide', contracted_intermediate) 
    
    #	 -1.0000 P(a,c)d(b,e)*<j,a||f,d>*t1(f,i)*t1(c,j)
    contracted_intermediate = -1.000000000000000 * einsum('be,jafd,fi,cj->abcide', kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->cbaide', contracted_intermediate) 
    
    #	  1.0000 P(a,c)d(b,d)*<j,a||f,e>*t1(f,i)*t1(c,j)
    contracted_intermediate =  1.000000000000000 * einsum('bd,jafe,fi,cj->abcide', kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->cbaide', contracted_intermediate) 
    
    #	 -1.0000 d(a,d)*d(c,e)*<j,b||f,g>*t1(f,j)*t1(g,i)
    H += -1.000000000000000 * einsum('ad,ce,jbfg,fj,gi->abcide', kd[v, v], kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,e)*<j,b||f,g>*t1(f,j)*t1(g,i)
    H +=  1.000000000000000 * einsum('cd,ae,jbfg,fj,gi->abcide', kd[v, v], kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(b,c)d(a,e)*<j,b||f,d>*t1(f,i)*t1(c,j)
    contracted_intermediate =  1.000000000000000 * einsum('ae,jbfd,fi,cj->abcide', kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->acbide', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)d(a,d)*<j,b||f,e>*t1(f,i)*t1(c,j)
    contracted_intermediate = -1.000000000000000 * einsum('ad,jbfe,fi,cj->abcide', kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcide->acbide', contracted_intermediate) 
    
    #	  1.0000 d(a,d)*d(b,e)*<j,c||f,g>*t1(f,j)*t1(g,i)
    H +=  1.000000000000000 * einsum('ad,be,jcfg,fj,gi->abcide', kd[v, v], kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*<j,c||f,g>*t1(f,j)*t1(g,i)
    H += -1.000000000000000 * einsum('bd,ae,jcfg,fj,gi->abcide', kd[v, v], kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,e)*<k,j||f,g>*t1(f,j)*t1(g,i)*t1(a,k)
    H +=  1.000000000000000 * einsum('bd,ce,kjfg,fj,gi,ak->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,e)*<k,j||f,g>*t1(f,j)*t1(g,i)*t1(a,k)
    H += -1.000000000000000 * einsum('cd,be,kjfg,fj,gi,ak->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,e)*<k,j||f,g>*t1(f,j)*t1(g,i)*t1(b,k)
    H += -1.000000000000000 * einsum('ad,ce,kjfg,fj,gi,bk->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,e)*<k,j||f,g>*t1(f,j)*t1(g,i)*t1(b,k)
    H +=  1.000000000000000 * einsum('cd,ae,kjfg,fj,gi,bk->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*d(b,e)*<k,j||f,g>*t1(f,j)*t1(g,i)*t1(c,k)
    H +=  1.000000000000000 * einsum('ad,be,kjfg,fj,gi,ck->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*<k,j||f,g>*t1(f,j)*t1(g,i)*t1(c,k)
    H += -1.000000000000000 * einsum('bd,ae,kjfg,fj,gi,ck->abcide', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,e)*<k,j||f,d>*t1(f,i)*t1(a,j)*t1(b,k)
    H += -1.000000000000000 * einsum('ce,kjfd,fi,aj,bk->abcide', kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*<k,j||f,e>*t1(f,i)*t1(a,j)*t1(b,k)
    H +=  1.000000000000000 * einsum('cd,kjfe,fi,aj,bk->abcide', kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    #	  1.0000 d(b,e)*<k,j||f,d>*t1(f,i)*t1(a,j)*t1(c,k)
    H +=  1.000000000000000 * einsum('be,kjfd,fi,aj,ck->abcide', kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*<k,j||f,e>*t1(f,i)*t1(a,j)*t1(c,k)
    H += -1.000000000000000 * einsum('bd,kjfe,fi,aj,ck->abcide', kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    #	 -1.0000 d(a,e)*<k,j||f,d>*t1(f,i)*t1(b,j)*t1(c,k)
    H += -1.000000000000000 * einsum('ae,kjfd,fi,bj,ck->abcide', kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*<k,j||f,e>*t1(f,i)*t1(b,j)*t1(c,k)
    H +=  1.000000000000000 * einsum('ad,kjfe,fi,bj,ck->abcide', kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    
    return H


def dea_eom_ccsd_hamiltonian_23(kd, f, g, o, v, t1, t2):

    #    H(a,b;d,e,f,j) = <0|b a e(-T) H e(T) d* e* f* j|0>
    
    #	  1.0000 d(a,e)*d(b,f)*f(j,d)
    H =  1.000000000000000 * einsum('ae,bf,jd->abdefj', kd[v, v], kd[v, v], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(b,e)*d(a,f)*f(j,d)
    H += -1.000000000000000 * einsum('be,af,jd->abdefj', kd[v, v], kd[v, v], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(a,d)*d(b,f)*f(j,e)
    H += -1.000000000000000 * einsum('ad,bf,je->abdefj', kd[v, v], kd[v, v], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(a,d)*d(b,e)*f(j,f)
    H +=  1.000000000000000 * einsum('ad,be,jf->abdefj', kd[v, v], kd[v, v], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(b,d)*d(a,f)*f(j,e)
    H +=  1.000000000000000 * einsum('bd,af,je->abdefj', kd[v, v], kd[v, v], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(b,d)*d(a,e)*f(j,f)
    H += -1.000000000000000 * einsum('bd,ae,jf->abdefj', kd[v, v], kd[v, v], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(b,f)*<j,a||d,e>
    H +=  1.000000000000000 * einsum('bf,jade->abdefj', kd[v, v], g[o, v, v, v])
    
    #	 -1.0000 d(b,e)*<j,a||d,f>
    H += -1.000000000000000 * einsum('be,jadf->abdefj', kd[v, v], g[o, v, v, v])
    
    #	  1.0000 d(b,d)*<j,a||e,f>
    H +=  1.000000000000000 * einsum('bd,jaef->abdefj', kd[v, v], g[o, v, v, v])
    
    #	 -1.0000 d(a,f)*<j,b||d,e>
    H += -1.000000000000000 * einsum('af,jbde->abdefj', kd[v, v], g[o, v, v, v])
    
    #	  1.0000 d(a,e)*<j,b||d,f>
    H +=  1.000000000000000 * einsum('ae,jbdf->abdefj', kd[v, v], g[o, v, v, v])
    
    #	 -1.0000 d(a,d)*<j,b||e,f>
    H += -1.000000000000000 * einsum('ad,jbef->abdefj', kd[v, v], g[o, v, v, v])
    
    #	 -1.0000 d(a,e)*d(b,f)*<j,i||c,d>*t1(c,i)
    H += -1.000000000000000 * einsum('ae,bf,jicd,ci->abdefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,e)*d(a,f)*<j,i||c,d>*t1(c,i)
    H +=  1.000000000000000 * einsum('be,af,jicd,ci->abdefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(a,d)*d(b,f)*<j,i||c,e>*t1(c,i)
    H +=  1.000000000000000 * einsum('ad,bf,jice,ci->abdefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(b,e)*<j,i||c,f>*t1(c,i)
    H += -1.000000000000000 * einsum('ad,be,jicf,ci->abdefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,f)*<j,i||c,e>*t1(c,i)
    H += -1.000000000000000 * einsum('bd,af,jice,ci->abdefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,d)*d(a,e)*<j,i||c,f>*t1(c,i)
    H +=  1.000000000000000 * einsum('bd,ae,jicf,ci->abdefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(b,f)*<j,i||d,e>*t1(a,i)
    H += -1.000000000000000 * einsum('bf,jide,ai->abdefj', kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(b,e)*<j,i||d,f>*t1(a,i)
    H +=  1.000000000000000 * einsum('be,jidf,ai->abdefj', kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*<j,i||e,f>*t1(a,i)
    H += -1.000000000000000 * einsum('bd,jief,ai->abdefj', kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(a,f)*<j,i||d,e>*t1(b,i)
    H +=  1.000000000000000 * einsum('af,jide,bi->abdefj', kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(a,e)*<j,i||d,f>*t1(b,i)
    H += -1.000000000000000 * einsum('ae,jidf,bi->abdefj', kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*<j,i||e,f>*t1(b,i)
    H +=  1.000000000000000 * einsum('ad,jief,bi->abdefj', kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    
    return H


def dea_eom_ccsd_hamiltonian_33(kd, f, g, o, v, t1, t2):

    #    H(a,b,c,i;d,e,f,j) = <0|i* c b a e(-T) H e(T) d* e* f* j|0>
    
    #	  1.0000 d(a,d)*d(b,e)*d(c,f)*d(i,j)*f(k,k)
    H =  1.000000000000000 * einsum('ad,be,cf,ij,kk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,e)*d(b,f)*d(i,j)*f(k,k)
    H += -1.000000000000000 * einsum('ad,ce,bf,ij,kk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*d(c,f)*d(i,j)*f(k,k)
    H += -1.000000000000000 * einsum('bd,ae,cf,ij,kk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,e)*d(a,f)*d(i,j)*f(k,k)
    H +=  1.000000000000000 * einsum('bd,ce,af,ij,kk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,e)*d(b,f)*d(i,j)*f(k,k)
    H +=  1.000000000000000 * einsum('cd,ae,bf,ij,kk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,e)*d(a,f)*d(i,j)*f(k,k)
    H += -1.000000000000000 * einsum('cd,be,af,ij,kk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(b,e)*d(c,f)*f(j,i)
    H += -1.000000000000000 * einsum('ad,be,cf,ji->abcidefj', kd[v, v], kd[v, v], kd[v, v], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(a,d)*d(c,e)*d(b,f)*f(j,i)
    H +=  1.000000000000000 * einsum('ad,ce,bf,ji->abcidefj', kd[v, v], kd[v, v], kd[v, v], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(b,d)*d(a,e)*d(c,f)*f(j,i)
    H +=  1.000000000000000 * einsum('bd,ae,cf,ji->abcidefj', kd[v, v], kd[v, v], kd[v, v], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(b,d)*d(c,e)*d(a,f)*f(j,i)
    H += -1.000000000000000 * einsum('bd,ce,af,ji->abcidefj', kd[v, v], kd[v, v], kd[v, v], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(c,d)*d(a,e)*d(b,f)*f(j,i)
    H += -1.000000000000000 * einsum('cd,ae,bf,ji->abcidefj', kd[v, v], kd[v, v], kd[v, v], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(c,d)*d(b,e)*d(a,f)*f(j,i)
    H +=  1.000000000000000 * einsum('cd,be,af,ji->abcidefj', kd[v, v], kd[v, v], kd[v, v], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(b,e)*d(c,f)*d(i,j)*f(a,d)
    H +=  1.000000000000000 * einsum('be,cf,ij,ad->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(c,e)*d(b,f)*d(i,j)*f(a,d)
    H += -1.000000000000000 * einsum('ce,bf,ij,ad->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(b,d)*d(c,f)*d(i,j)*f(a,e)
    H += -1.000000000000000 * einsum('bd,cf,ij,ae->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(b,d)*d(c,e)*d(i,j)*f(a,f)
    H +=  1.000000000000000 * einsum('bd,ce,ij,af->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(c,d)*d(b,f)*d(i,j)*f(a,e)
    H +=  1.000000000000000 * einsum('cd,bf,ij,ae->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(c,d)*d(b,e)*d(i,j)*f(a,f)
    H += -1.000000000000000 * einsum('cd,be,ij,af->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(a,e)*d(c,f)*d(i,j)*f(b,d)
    H += -1.000000000000000 * einsum('ae,cf,ij,bd->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(c,e)*d(a,f)*d(i,j)*f(b,d)
    H +=  1.000000000000000 * einsum('ce,af,ij,bd->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(a,d)*d(c,f)*d(i,j)*f(b,e)
    H +=  1.000000000000000 * einsum('ad,cf,ij,be->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(a,d)*d(c,e)*d(i,j)*f(b,f)
    H += -1.000000000000000 * einsum('ad,ce,ij,bf->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(c,d)*d(a,f)*d(i,j)*f(b,e)
    H += -1.000000000000000 * einsum('cd,af,ij,be->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(c,d)*d(a,e)*d(i,j)*f(b,f)
    H +=  1.000000000000000 * einsum('cd,ae,ij,bf->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(a,e)*d(b,f)*d(i,j)*f(c,d)
    H +=  1.000000000000000 * einsum('ae,bf,ij,cd->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(b,e)*d(a,f)*d(i,j)*f(c,d)
    H += -1.000000000000000 * einsum('be,af,ij,cd->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(a,d)*d(b,f)*d(i,j)*f(c,e)
    H += -1.000000000000000 * einsum('ad,bf,ij,ce->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(a,d)*d(b,e)*d(i,j)*f(c,f)
    H +=  1.000000000000000 * einsum('ad,be,ij,cf->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(b,d)*d(a,f)*d(i,j)*f(c,e)
    H +=  1.000000000000000 * einsum('bd,af,ij,ce->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(b,d)*d(a,e)*d(i,j)*f(c,f)
    H += -1.000000000000000 * einsum('bd,ae,ij,cf->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(a,d)*d(b,e)*d(c,f)*d(i,j)*f(k,g)*t1(g,k)
    H +=  1.000000000000000 * einsum('ad,be,cf,ij,kg,gk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,e)*d(b,f)*d(i,j)*f(k,g)*t1(g,k)
    H += -1.000000000000000 * einsum('ad,ce,bf,ij,kg,gk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*d(c,f)*d(i,j)*f(k,g)*t1(g,k)
    H += -1.000000000000000 * einsum('bd,ae,cf,ij,kg,gk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,e)*d(a,f)*d(i,j)*f(k,g)*t1(g,k)
    H +=  1.000000000000000 * einsum('bd,ce,af,ij,kg,gk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,e)*d(b,f)*d(i,j)*f(k,g)*t1(g,k)
    H +=  1.000000000000000 * einsum('cd,ae,bf,ij,kg,gk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,e)*d(a,f)*d(i,j)*f(k,g)*t1(g,k)
    H += -1.000000000000000 * einsum('cd,be,af,ij,kg,gk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(b,e)*d(c,f)*f(j,g)*t1(g,i)
    H += -1.000000000000000 * einsum('ad,be,cf,jg,gi->abcidefj', kd[v, v], kd[v, v], kd[v, v], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*d(c,e)*d(b,f)*f(j,g)*t1(g,i)
    H +=  1.000000000000000 * einsum('ad,ce,bf,jg,gi->abcidefj', kd[v, v], kd[v, v], kd[v, v], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(a,e)*d(c,f)*f(j,g)*t1(g,i)
    H +=  1.000000000000000 * einsum('bd,ae,cf,jg,gi->abcidefj', kd[v, v], kd[v, v], kd[v, v], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(c,e)*d(a,f)*f(j,g)*t1(g,i)
    H += -1.000000000000000 * einsum('bd,ce,af,jg,gi->abcidefj', kd[v, v], kd[v, v], kd[v, v], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(a,e)*d(b,f)*f(j,g)*t1(g,i)
    H += -1.000000000000000 * einsum('cd,ae,bf,jg,gi->abcidefj', kd[v, v], kd[v, v], kd[v, v], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(b,e)*d(a,f)*f(j,g)*t1(g,i)
    H +=  1.000000000000000 * einsum('cd,be,af,jg,gi->abcidefj', kd[v, v], kd[v, v], kd[v, v], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,e)*d(c,f)*d(i,j)*f(k,d)*t1(a,k)
    H += -1.000000000000000 * einsum('be,cf,ij,kd,ak->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,e)*d(b,f)*d(i,j)*f(k,d)*t1(a,k)
    H +=  1.000000000000000 * einsum('ce,bf,ij,kd,ak->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,f)*d(i,j)*f(k,e)*t1(a,k)
    H +=  1.000000000000000 * einsum('bd,cf,ij,ke,ak->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(c,e)*d(i,j)*f(k,f)*t1(a,k)
    H += -1.000000000000000 * einsum('bd,ce,ij,kf,ak->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,f)*d(i,j)*f(k,e)*t1(a,k)
    H += -1.000000000000000 * einsum('cd,bf,ij,ke,ak->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(b,e)*d(i,j)*f(k,f)*t1(a,k)
    H +=  1.000000000000000 * einsum('cd,be,ij,kf,ak->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,e)*d(c,f)*d(i,j)*f(k,d)*t1(b,k)
    H +=  1.000000000000000 * einsum('ae,cf,ij,kd,bk->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,e)*d(a,f)*d(i,j)*f(k,d)*t1(b,k)
    H += -1.000000000000000 * einsum('ce,af,ij,kd,bk->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,f)*d(i,j)*f(k,e)*t1(b,k)
    H += -1.000000000000000 * einsum('ad,cf,ij,ke,bk->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*d(c,e)*d(i,j)*f(k,f)*t1(b,k)
    H +=  1.000000000000000 * einsum('ad,ce,ij,kf,bk->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,f)*d(i,j)*f(k,e)*t1(b,k)
    H +=  1.000000000000000 * einsum('cd,af,ij,ke,bk->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(a,e)*d(i,j)*f(k,f)*t1(b,k)
    H += -1.000000000000000 * einsum('cd,ae,ij,kf,bk->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,e)*d(b,f)*d(i,j)*f(k,d)*t1(c,k)
    H += -1.000000000000000 * einsum('ae,bf,ij,kd,ck->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,e)*d(a,f)*d(i,j)*f(k,d)*t1(c,k)
    H +=  1.000000000000000 * einsum('be,af,ij,kd,ck->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*d(b,f)*d(i,j)*f(k,e)*t1(c,k)
    H +=  1.000000000000000 * einsum('ad,bf,ij,ke,ck->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(b,e)*d(i,j)*f(k,f)*t1(c,k)
    H += -1.000000000000000 * einsum('ad,be,ij,kf,ck->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,f)*d(i,j)*f(k,e)*t1(c,k)
    H += -1.000000000000000 * einsum('bd,af,ij,ke,ck->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(a,e)*d(i,j)*f(k,f)*t1(c,k)
    H +=  1.000000000000000 * einsum('bd,ae,ij,kf,ck->abcidefj', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(a,d)*d(b,e)*d(c,f)*d(i,j)*<l,k||l,k>
    H += -0.500000000000000 * einsum('ad,be,cf,ij,lklk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(a,d)*d(c,e)*d(b,f)*d(i,j)*<l,k||l,k>
    H +=  0.500000000000000 * einsum('ad,ce,bf,ij,lklk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(b,d)*d(a,e)*d(c,f)*d(i,j)*<l,k||l,k>
    H +=  0.500000000000000 * einsum('bd,ae,cf,ij,lklk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(b,d)*d(c,e)*d(a,f)*d(i,j)*<l,k||l,k>
    H += -0.500000000000000 * einsum('bd,ce,af,ij,lklk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(c,d)*d(a,e)*d(b,f)*d(i,j)*<l,k||l,k>
    H += -0.500000000000000 * einsum('cd,ae,bf,ij,lklk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(c,d)*d(b,e)*d(a,f)*d(i,j)*<l,k||l,k>
    H +=  0.500000000000000 * einsum('cd,be,af,ij,lklk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,e)*d(c,f)*<j,a||d,i>
    H +=  1.000000000000000 * einsum('be,cf,jadi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(c,e)*d(b,f)*<j,a||d,i>
    H += -1.000000000000000 * einsum('ce,bf,jadi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(b,d)*d(c,f)*<j,a||e,i>
    H += -1.000000000000000 * einsum('bd,cf,jaei->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(b,d)*d(c,e)*<j,a||f,i>
    H +=  1.000000000000000 * einsum('bd,ce,jafi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(c,d)*d(b,f)*<j,a||e,i>
    H +=  1.000000000000000 * einsum('cd,bf,jaei->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(c,d)*d(b,e)*<j,a||f,i>
    H += -1.000000000000000 * einsum('cd,be,jafi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(a,e)*d(c,f)*<j,b||d,i>
    H += -1.000000000000000 * einsum('ae,cf,jbdi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(c,e)*d(a,f)*<j,b||d,i>
    H +=  1.000000000000000 * einsum('ce,af,jbdi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(a,d)*d(c,f)*<j,b||e,i>
    H +=  1.000000000000000 * einsum('ad,cf,jbei->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(a,d)*d(c,e)*<j,b||f,i>
    H += -1.000000000000000 * einsum('ad,ce,jbfi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(c,d)*d(a,f)*<j,b||e,i>
    H += -1.000000000000000 * einsum('cd,af,jbei->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(c,d)*d(a,e)*<j,b||f,i>
    H +=  1.000000000000000 * einsum('cd,ae,jbfi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(a,e)*d(b,f)*<j,c||d,i>
    H +=  1.000000000000000 * einsum('ae,bf,jcdi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(b,e)*d(a,f)*<j,c||d,i>
    H += -1.000000000000000 * einsum('be,af,jcdi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(a,d)*d(b,f)*<j,c||e,i>
    H += -1.000000000000000 * einsum('ad,bf,jcei->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(a,d)*d(b,e)*<j,c||f,i>
    H +=  1.000000000000000 * einsum('ad,be,jcfi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(b,d)*d(a,f)*<j,c||e,i>
    H +=  1.000000000000000 * einsum('bd,af,jcei->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(b,d)*d(a,e)*<j,c||f,i>
    H += -1.000000000000000 * einsum('bd,ae,jcfi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(c,f)*d(i,j)*<a,b||d,e>
    H +=  1.000000000000000 * einsum('cf,ij,abde->abcidefj', kd[v, v], kd[o, o], g[v, v, v, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(c,e)*d(i,j)*<a,b||d,f>
    H += -1.000000000000000 * einsum('ce,ij,abdf->abcidefj', kd[v, v], kd[o, o], g[v, v, v, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(c,d)*d(i,j)*<a,b||e,f>
    H +=  1.000000000000000 * einsum('cd,ij,abef->abcidefj', kd[v, v], kd[o, o], g[v, v, v, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(b,f)*d(i,j)*<a,c||d,e>
    H += -1.000000000000000 * einsum('bf,ij,acde->abcidefj', kd[v, v], kd[o, o], g[v, v, v, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(b,e)*d(i,j)*<a,c||d,f>
    H +=  1.000000000000000 * einsum('be,ij,acdf->abcidefj', kd[v, v], kd[o, o], g[v, v, v, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(b,d)*d(i,j)*<a,c||e,f>
    H += -1.000000000000000 * einsum('bd,ij,acef->abcidefj', kd[v, v], kd[o, o], g[v, v, v, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(a,f)*d(i,j)*<b,c||d,e>
    H +=  1.000000000000000 * einsum('af,ij,bcde->abcidefj', kd[v, v], kd[o, o], g[v, v, v, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(a,e)*d(i,j)*<b,c||d,f>
    H += -1.000000000000000 * einsum('ae,ij,bcdf->abcidefj', kd[v, v], kd[o, o], g[v, v, v, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(a,d)*d(i,j)*<b,c||e,f>
    H +=  1.000000000000000 * einsum('ad,ij,bcef->abcidefj', kd[v, v], kd[o, o], g[v, v, v, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(a,d)*d(b,e)*d(c,f)*<j,k||g,i>*t1(g,k)
    H +=  1.000000000000000 * einsum('ad,be,cf,jkgi,gk->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,e)*d(b,f)*<j,k||g,i>*t1(g,k)
    H += -1.000000000000000 * einsum('ad,ce,bf,jkgi,gk->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*d(c,f)*<j,k||g,i>*t1(g,k)
    H += -1.000000000000000 * einsum('bd,ae,cf,jkgi,gk->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,e)*d(a,f)*<j,k||g,i>*t1(g,k)
    H +=  1.000000000000000 * einsum('bd,ce,af,jkgi,gk->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,e)*d(b,f)*<j,k||g,i>*t1(g,k)
    H +=  1.000000000000000 * einsum('cd,ae,bf,jkgi,gk->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,e)*d(a,f)*<j,k||g,i>*t1(g,k)
    H += -1.000000000000000 * einsum('cd,be,af,jkgi,gk->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,e)*d(c,f)*<j,k||d,i>*t1(a,k)
    H += -1.000000000000000 * einsum('be,cf,jkdi,ak->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,e)*d(b,f)*<j,k||d,i>*t1(a,k)
    H +=  1.000000000000000 * einsum('ce,bf,jkdi,ak->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,f)*<j,k||e,i>*t1(a,k)
    H +=  1.000000000000000 * einsum('bd,cf,jkei,ak->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(c,e)*<j,k||f,i>*t1(a,k)
    H += -1.000000000000000 * einsum('bd,ce,jkfi,ak->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,f)*<j,k||e,i>*t1(a,k)
    H += -1.000000000000000 * einsum('cd,bf,jkei,ak->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,d)*d(b,e)*<j,k||f,i>*t1(a,k)
    H +=  1.000000000000000 * einsum('cd,be,jkfi,ak->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(a,e)*d(c,f)*<j,k||d,i>*t1(b,k)
    H +=  1.000000000000000 * einsum('ae,cf,jkdi,bk->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,e)*d(a,f)*<j,k||d,i>*t1(b,k)
    H += -1.000000000000000 * einsum('ce,af,jkdi,bk->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,f)*<j,k||e,i>*t1(b,k)
    H += -1.000000000000000 * einsum('ad,cf,jkei,bk->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(a,d)*d(c,e)*<j,k||f,i>*t1(b,k)
    H +=  1.000000000000000 * einsum('ad,ce,jkfi,bk->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,f)*<j,k||e,i>*t1(b,k)
    H +=  1.000000000000000 * einsum('cd,af,jkei,bk->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(a,e)*<j,k||f,i>*t1(b,k)
    H += -1.000000000000000 * einsum('cd,ae,jkfi,bk->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(a,e)*d(b,f)*<j,k||d,i>*t1(c,k)
    H += -1.000000000000000 * einsum('ae,bf,jkdi,ck->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,e)*d(a,f)*<j,k||d,i>*t1(c,k)
    H +=  1.000000000000000 * einsum('be,af,jkdi,ck->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(a,d)*d(b,f)*<j,k||e,i>*t1(c,k)
    H +=  1.000000000000000 * einsum('ad,bf,jkei,ck->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(b,e)*<j,k||f,i>*t1(c,k)
    H += -1.000000000000000 * einsum('ad,be,jkfi,ck->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,f)*<j,k||e,i>*t1(c,k)
    H += -1.000000000000000 * einsum('bd,af,jkei,ck->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,d)*d(a,e)*<j,k||f,i>*t1(c,k)
    H +=  1.000000000000000 * einsum('bd,ae,jkfi,ck->abcidefj', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,e)*d(c,f)*d(i,j)*<k,a||g,d>*t1(g,k)
    H +=  1.000000000000000 * einsum('be,cf,ij,kagd,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,e)*d(b,f)*d(i,j)*<k,a||g,d>*t1(g,k)
    H += -1.000000000000000 * einsum('ce,bf,ij,kagd,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(c,f)*d(i,j)*<k,a||g,e>*t1(g,k)
    H += -1.000000000000000 * einsum('bd,cf,ij,kage,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,e)*d(i,j)*<k,a||g,f>*t1(g,k)
    H +=  1.000000000000000 * einsum('bd,ce,ij,kagf,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(b,f)*d(i,j)*<k,a||g,e>*t1(g,k)
    H +=  1.000000000000000 * einsum('cd,bf,ij,kage,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,e)*d(i,j)*<k,a||g,f>*t1(g,k)
    H += -1.000000000000000 * einsum('cd,be,ij,kagf,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,e)*d(c,f)*<j,a||g,d>*t1(g,i)
    H += -1.000000000000000 * einsum('be,cf,jagd,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,e)*d(b,f)*<j,a||g,d>*t1(g,i)
    H +=  1.000000000000000 * einsum('ce,bf,jagd,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,f)*<j,a||g,e>*t1(g,i)
    H +=  1.000000000000000 * einsum('bd,cf,jage,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(c,e)*<j,a||g,f>*t1(g,i)
    H += -1.000000000000000 * einsum('bd,ce,jagf,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,f)*<j,a||g,e>*t1(g,i)
    H += -1.000000000000000 * einsum('cd,bf,jage,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,d)*d(b,e)*<j,a||g,f>*t1(g,i)
    H +=  1.000000000000000 * einsum('cd,be,jagf,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 P(a,b)d(c,f)*d(i,j)*<k,a||d,e>*t1(b,k)
    contracted_intermediate =  1.000000000000000 * einsum('cf,ij,kade,bk->abcidefj', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcidefj->bacidefj', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)d(c,e)*d(i,j)*<k,a||d,f>*t1(b,k)
    contracted_intermediate = -1.000000000000000 * einsum('ce,ij,kadf,bk->abcidefj', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcidefj->bacidefj', contracted_intermediate) 
    
    #	  1.0000 P(a,b)d(c,d)*d(i,j)*<k,a||e,f>*t1(b,k)
    contracted_intermediate =  1.000000000000000 * einsum('cd,ij,kaef,bk->abcidefj', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcidefj->bacidefj', contracted_intermediate) 
    
    #	 -1.0000 P(a,c)d(b,f)*d(i,j)*<k,a||d,e>*t1(c,k)
    contracted_intermediate = -1.000000000000000 * einsum('bf,ij,kade,ck->abcidefj', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcidefj->cbaidefj', contracted_intermediate) 
    
    #	  1.0000 P(a,c)d(b,e)*d(i,j)*<k,a||d,f>*t1(c,k)
    contracted_intermediate =  1.000000000000000 * einsum('be,ij,kadf,ck->abcidefj', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcidefj->cbaidefj', contracted_intermediate) 
    
    #	 -1.0000 P(a,c)d(b,d)*d(i,j)*<k,a||e,f>*t1(c,k)
    contracted_intermediate = -1.000000000000000 * einsum('bd,ij,kaef,ck->abcidefj', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcidefj->cbaidefj', contracted_intermediate) 
    
    #	 -1.0000 d(a,e)*d(c,f)*d(i,j)*<k,b||g,d>*t1(g,k)
    H += -1.000000000000000 * einsum('ae,cf,ij,kbgd,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,e)*d(a,f)*d(i,j)*<k,b||g,d>*t1(g,k)
    H +=  1.000000000000000 * einsum('ce,af,ij,kbgd,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*d(c,f)*d(i,j)*<k,b||g,e>*t1(g,k)
    H +=  1.000000000000000 * einsum('ad,cf,ij,kbge,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,e)*d(i,j)*<k,b||g,f>*t1(g,k)
    H += -1.000000000000000 * einsum('ad,ce,ij,kbgf,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(a,f)*d(i,j)*<k,b||g,e>*t1(g,k)
    H += -1.000000000000000 * einsum('cd,af,ij,kbge,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,e)*d(i,j)*<k,b||g,f>*t1(g,k)
    H +=  1.000000000000000 * einsum('cd,ae,ij,kbgf,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,e)*d(c,f)*<j,b||g,d>*t1(g,i)
    H +=  1.000000000000000 * einsum('ae,cf,jbgd,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,e)*d(a,f)*<j,b||g,d>*t1(g,i)
    H += -1.000000000000000 * einsum('ce,af,jbgd,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,f)*<j,b||g,e>*t1(g,i)
    H += -1.000000000000000 * einsum('ad,cf,jbge,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(a,d)*d(c,e)*<j,b||g,f>*t1(g,i)
    H +=  1.000000000000000 * einsum('ad,ce,jbgf,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,f)*<j,b||g,e>*t1(g,i)
    H +=  1.000000000000000 * einsum('cd,af,jbge,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(a,e)*<j,b||g,f>*t1(g,i)
    H += -1.000000000000000 * einsum('cd,ae,jbgf,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 P(b,c)d(a,f)*d(i,j)*<k,b||d,e>*t1(c,k)
    contracted_intermediate =  1.000000000000000 * einsum('af,ij,kbde,ck->abcidefj', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcidefj->acbidefj', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)d(a,e)*d(i,j)*<k,b||d,f>*t1(c,k)
    contracted_intermediate = -1.000000000000000 * einsum('ae,ij,kbdf,ck->abcidefj', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcidefj->acbidefj', contracted_intermediate) 
    
    #	  1.0000 P(b,c)d(a,d)*d(i,j)*<k,b||e,f>*t1(c,k)
    contracted_intermediate =  1.000000000000000 * einsum('ad,ij,kbef,ck->abcidefj', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    H +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcidefj->acbidefj', contracted_intermediate) 
    
    #	  1.0000 d(a,e)*d(b,f)*d(i,j)*<k,c||g,d>*t1(g,k)
    H +=  1.000000000000000 * einsum('ae,bf,ij,kcgd,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,e)*d(a,f)*d(i,j)*<k,c||g,d>*t1(g,k)
    H += -1.000000000000000 * einsum('be,af,ij,kcgd,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(b,f)*d(i,j)*<k,c||g,e>*t1(g,k)
    H += -1.000000000000000 * einsum('ad,bf,ij,kcge,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*d(b,e)*d(i,j)*<k,c||g,f>*t1(g,k)
    H +=  1.000000000000000 * einsum('ad,be,ij,kcgf,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(a,f)*d(i,j)*<k,c||g,e>*t1(g,k)
    H +=  1.000000000000000 * einsum('bd,af,ij,kcge,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*d(i,j)*<k,c||g,f>*t1(g,k)
    H += -1.000000000000000 * einsum('bd,ae,ij,kcgf,gk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,e)*d(b,f)*<j,c||g,d>*t1(g,i)
    H += -1.000000000000000 * einsum('ae,bf,jcgd,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,e)*d(a,f)*<j,c||g,d>*t1(g,i)
    H +=  1.000000000000000 * einsum('be,af,jcgd,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(a,d)*d(b,f)*<j,c||g,e>*t1(g,i)
    H +=  1.000000000000000 * einsum('ad,bf,jcge,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(b,e)*<j,c||g,f>*t1(g,i)
    H += -1.000000000000000 * einsum('ad,be,jcgf,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,f)*<j,c||g,e>*t1(g,i)
    H += -1.000000000000000 * einsum('bd,af,jcge,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,d)*d(a,e)*<j,c||g,f>*t1(g,i)
    H +=  1.000000000000000 * einsum('bd,ae,jcgf,gi->abcidefj', kd[v, v], kd[v, v], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 d(a,d)*d(b,e)*d(c,f)*d(i,j)*<l,k||g,h>*t2(g,h,l,k)
    H +=  0.250000000000000 * einsum('ad,be,cf,ij,lkgh,ghlk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 d(a,d)*d(c,e)*d(b,f)*d(i,j)*<l,k||g,h>*t2(g,h,l,k)
    H += -0.250000000000000 * einsum('ad,ce,bf,ij,lkgh,ghlk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 d(b,d)*d(a,e)*d(c,f)*d(i,j)*<l,k||g,h>*t2(g,h,l,k)
    H += -0.250000000000000 * einsum('bd,ae,cf,ij,lkgh,ghlk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 d(b,d)*d(c,e)*d(a,f)*d(i,j)*<l,k||g,h>*t2(g,h,l,k)
    H +=  0.250000000000000 * einsum('bd,ce,af,ij,lkgh,ghlk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 d(c,d)*d(a,e)*d(b,f)*d(i,j)*<l,k||g,h>*t2(g,h,l,k)
    H +=  0.250000000000000 * einsum('cd,ae,bf,ij,lkgh,ghlk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 d(c,d)*d(b,e)*d(a,f)*d(i,j)*<l,k||g,h>*t2(g,h,l,k)
    H += -0.250000000000000 * einsum('cd,be,af,ij,lkgh,ghlk->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(a,d)*d(b,e)*d(c,f)*<j,k||g,h>*t2(g,h,i,k)
    H += -0.500000000000000 * einsum('ad,be,cf,jkgh,ghik->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(a,d)*d(c,e)*d(b,f)*<j,k||g,h>*t2(g,h,i,k)
    H +=  0.500000000000000 * einsum('ad,ce,bf,jkgh,ghik->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(b,d)*d(a,e)*d(c,f)*<j,k||g,h>*t2(g,h,i,k)
    H +=  0.500000000000000 * einsum('bd,ae,cf,jkgh,ghik->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(b,d)*d(c,e)*d(a,f)*<j,k||g,h>*t2(g,h,i,k)
    H += -0.500000000000000 * einsum('bd,ce,af,jkgh,ghik->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(c,d)*d(a,e)*d(b,f)*<j,k||g,h>*t2(g,h,i,k)
    H += -0.500000000000000 * einsum('cd,ae,bf,jkgh,ghik->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(c,d)*d(b,e)*d(a,f)*<j,k||g,h>*t2(g,h,i,k)
    H +=  0.500000000000000 * einsum('cd,be,af,jkgh,ghik->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(b,e)*d(c,f)*d(i,j)*<l,k||g,d>*t2(g,a,l,k)
    H += -0.500000000000000 * einsum('be,cf,ij,lkgd,galk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(c,e)*d(b,f)*d(i,j)*<l,k||g,d>*t2(g,a,l,k)
    H +=  0.500000000000000 * einsum('ce,bf,ij,lkgd,galk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(b,d)*d(c,f)*d(i,j)*<l,k||g,e>*t2(g,a,l,k)
    H +=  0.500000000000000 * einsum('bd,cf,ij,lkge,galk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(b,d)*d(c,e)*d(i,j)*<l,k||g,f>*t2(g,a,l,k)
    H += -0.500000000000000 * einsum('bd,ce,ij,lkgf,galk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(c,d)*d(b,f)*d(i,j)*<l,k||g,e>*t2(g,a,l,k)
    H += -0.500000000000000 * einsum('cd,bf,ij,lkge,galk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(c,d)*d(b,e)*d(i,j)*<l,k||g,f>*t2(g,a,l,k)
    H +=  0.500000000000000 * einsum('cd,be,ij,lkgf,galk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,e)*d(c,f)*<j,k||g,d>*t2(g,a,i,k)
    H +=  1.000000000000000 * einsum('be,cf,jkgd,gaik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,e)*d(b,f)*<j,k||g,d>*t2(g,a,i,k)
    H += -1.000000000000000 * einsum('ce,bf,jkgd,gaik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(c,f)*<j,k||g,e>*t2(g,a,i,k)
    H += -1.000000000000000 * einsum('bd,cf,jkge,gaik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,e)*<j,k||g,f>*t2(g,a,i,k)
    H +=  1.000000000000000 * einsum('bd,ce,jkgf,gaik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,d)*d(b,f)*<j,k||g,e>*t2(g,a,i,k)
    H +=  1.000000000000000 * einsum('cd,bf,jkge,gaik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,e)*<j,k||g,f>*t2(g,a,i,k)
    H += -1.000000000000000 * einsum('cd,be,jkgf,gaik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(a,e)*d(c,f)*d(i,j)*<l,k||g,d>*t2(g,b,l,k)
    H +=  0.500000000000000 * einsum('ae,cf,ij,lkgd,gblk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(c,e)*d(a,f)*d(i,j)*<l,k||g,d>*t2(g,b,l,k)
    H += -0.500000000000000 * einsum('ce,af,ij,lkgd,gblk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(a,d)*d(c,f)*d(i,j)*<l,k||g,e>*t2(g,b,l,k)
    H += -0.500000000000000 * einsum('ad,cf,ij,lkge,gblk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(a,d)*d(c,e)*d(i,j)*<l,k||g,f>*t2(g,b,l,k)
    H +=  0.500000000000000 * einsum('ad,ce,ij,lkgf,gblk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(c,d)*d(a,f)*d(i,j)*<l,k||g,e>*t2(g,b,l,k)
    H +=  0.500000000000000 * einsum('cd,af,ij,lkge,gblk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(c,d)*d(a,e)*d(i,j)*<l,k||g,f>*t2(g,b,l,k)
    H += -0.500000000000000 * einsum('cd,ae,ij,lkgf,gblk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,e)*d(c,f)*<j,k||g,d>*t2(g,b,i,k)
    H += -1.000000000000000 * einsum('ae,cf,jkgd,gbik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,e)*d(a,f)*<j,k||g,d>*t2(g,b,i,k)
    H +=  1.000000000000000 * einsum('ce,af,jkgd,gbik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(a,d)*d(c,f)*<j,k||g,e>*t2(g,b,i,k)
    H +=  1.000000000000000 * einsum('ad,cf,jkge,gbik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,e)*<j,k||g,f>*t2(g,b,i,k)
    H += -1.000000000000000 * einsum('ad,ce,jkgf,gbik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(a,f)*<j,k||g,e>*t2(g,b,i,k)
    H += -1.000000000000000 * einsum('cd,af,jkge,gbik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,e)*<j,k||g,f>*t2(g,b,i,k)
    H +=  1.000000000000000 * einsum('cd,ae,jkgf,gbik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(a,e)*d(b,f)*d(i,j)*<l,k||g,d>*t2(g,c,l,k)
    H += -0.500000000000000 * einsum('ae,bf,ij,lkgd,gclk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(b,e)*d(a,f)*d(i,j)*<l,k||g,d>*t2(g,c,l,k)
    H +=  0.500000000000000 * einsum('be,af,ij,lkgd,gclk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(a,d)*d(b,f)*d(i,j)*<l,k||g,e>*t2(g,c,l,k)
    H +=  0.500000000000000 * einsum('ad,bf,ij,lkge,gclk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(a,d)*d(b,e)*d(i,j)*<l,k||g,f>*t2(g,c,l,k)
    H += -0.500000000000000 * einsum('ad,be,ij,lkgf,gclk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(b,d)*d(a,f)*d(i,j)*<l,k||g,e>*t2(g,c,l,k)
    H += -0.500000000000000 * einsum('bd,af,ij,lkge,gclk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(b,d)*d(a,e)*d(i,j)*<l,k||g,f>*t2(g,c,l,k)
    H +=  0.500000000000000 * einsum('bd,ae,ij,lkgf,gclk->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,e)*d(b,f)*<j,k||g,d>*t2(g,c,i,k)
    H +=  1.000000000000000 * einsum('ae,bf,jkgd,gcik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(b,e)*d(a,f)*<j,k||g,d>*t2(g,c,i,k)
    H += -1.000000000000000 * einsum('be,af,jkgd,gcik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(b,f)*<j,k||g,e>*t2(g,c,i,k)
    H += -1.000000000000000 * einsum('ad,bf,jkge,gcik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(a,d)*d(b,e)*<j,k||g,f>*t2(g,c,i,k)
    H +=  1.000000000000000 * einsum('ad,be,jkgf,gcik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,d)*d(a,f)*<j,k||g,e>*t2(g,c,i,k)
    H +=  1.000000000000000 * einsum('bd,af,jkge,gcik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*<j,k||g,f>*t2(g,c,i,k)
    H += -1.000000000000000 * einsum('bd,ae,jkgf,gcik->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(c,f)*d(i,j)*<l,k||d,e>*t2(a,b,l,k)
    H +=  0.500000000000000 * einsum('cf,ij,lkde,ablk->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(c,e)*d(i,j)*<l,k||d,f>*t2(a,b,l,k)
    H += -0.500000000000000 * einsum('ce,ij,lkdf,ablk->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(c,d)*d(i,j)*<l,k||e,f>*t2(a,b,l,k)
    H +=  0.500000000000000 * einsum('cd,ij,lkef,ablk->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(c,f)*<j,k||d,e>*t2(a,b,i,k)
    H += -1.000000000000000 * einsum('cf,jkde,abik->abcidefj', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(c,e)*<j,k||d,f>*t2(a,b,i,k)
    H +=  1.000000000000000 * einsum('ce,jkdf,abik->abcidefj', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*<j,k||e,f>*t2(a,b,i,k)
    H += -1.000000000000000 * einsum('cd,jkef,abik->abcidefj', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(b,f)*d(i,j)*<l,k||d,e>*t2(a,c,l,k)
    H += -0.500000000000000 * einsum('bf,ij,lkde,aclk->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(b,e)*d(i,j)*<l,k||d,f>*t2(a,c,l,k)
    H +=  0.500000000000000 * einsum('be,ij,lkdf,aclk->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(b,d)*d(i,j)*<l,k||e,f>*t2(a,c,l,k)
    H += -0.500000000000000 * einsum('bd,ij,lkef,aclk->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(b,f)*<j,k||d,e>*t2(a,c,i,k)
    H +=  1.000000000000000 * einsum('bf,jkde,acik->abcidefj', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(b,e)*<j,k||d,f>*t2(a,c,i,k)
    H += -1.000000000000000 * einsum('be,jkdf,acik->abcidefj', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*<j,k||e,f>*t2(a,c,i,k)
    H +=  1.000000000000000 * einsum('bd,jkef,acik->abcidefj', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(a,f)*d(i,j)*<l,k||d,e>*t2(b,c,l,k)
    H +=  0.500000000000000 * einsum('af,ij,lkde,bclk->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(a,e)*d(i,j)*<l,k||d,f>*t2(b,c,l,k)
    H += -0.500000000000000 * einsum('ae,ij,lkdf,bclk->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(a,d)*d(i,j)*<l,k||e,f>*t2(b,c,l,k)
    H +=  0.500000000000000 * einsum('ad,ij,lkef,bclk->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(a,f)*<j,k||d,e>*t2(b,c,i,k)
    H += -1.000000000000000 * einsum('af,jkde,bcik->abcidefj', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(a,e)*<j,k||d,f>*t2(b,c,i,k)
    H +=  1.000000000000000 * einsum('ae,jkdf,bcik->abcidefj', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*<j,k||e,f>*t2(b,c,i,k)
    H += -1.000000000000000 * einsum('ad,jkef,bcik->abcidefj', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(a,d)*d(b,e)*d(c,f)*d(i,j)*<l,k||g,h>*t1(g,k)*t1(h,l)
    H += -0.500000000000000 * einsum('ad,be,cf,ij,lkgh,gk,hl->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(a,d)*d(c,e)*d(b,f)*d(i,j)*<l,k||g,h>*t1(g,k)*t1(h,l)
    H +=  0.500000000000000 * einsum('ad,ce,bf,ij,lkgh,gk,hl->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(b,d)*d(a,e)*d(c,f)*d(i,j)*<l,k||g,h>*t1(g,k)*t1(h,l)
    H +=  0.500000000000000 * einsum('bd,ae,cf,ij,lkgh,gk,hl->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(b,d)*d(c,e)*d(a,f)*d(i,j)*<l,k||g,h>*t1(g,k)*t1(h,l)
    H += -0.500000000000000 * einsum('bd,ce,af,ij,lkgh,gk,hl->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(c,d)*d(a,e)*d(b,f)*d(i,j)*<l,k||g,h>*t1(g,k)*t1(h,l)
    H += -0.500000000000000 * einsum('cd,ae,bf,ij,lkgh,gk,hl->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(c,d)*d(b,e)*d(a,f)*d(i,j)*<l,k||g,h>*t1(g,k)*t1(h,l)
    H +=  0.500000000000000 * einsum('cd,be,af,ij,lkgh,gk,hl->abcidefj', kd[v, v], kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(a,d)*d(b,e)*d(c,f)*<j,k||g,h>*t1(g,k)*t1(h,i)
    H +=  0.500000000000000 * einsum('ad,be,cf,jkgh,gk,hi->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(a,d)*d(c,e)*d(b,f)*<j,k||g,h>*t1(g,k)*t1(h,i)
    H += -0.500000000000000 * einsum('ad,ce,bf,jkgh,gk,hi->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(b,d)*d(a,e)*d(c,f)*<j,k||g,h>*t1(g,k)*t1(h,i)
    H += -0.500000000000000 * einsum('bd,ae,cf,jkgh,gk,hi->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(b,d)*d(c,e)*d(a,f)*<j,k||g,h>*t1(g,k)*t1(h,i)
    H +=  0.500000000000000 * einsum('bd,ce,af,jkgh,gk,hi->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(c,d)*d(a,e)*d(b,f)*<j,k||g,h>*t1(g,k)*t1(h,i)
    H +=  0.500000000000000 * einsum('cd,ae,bf,jkgh,gk,hi->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(c,d)*d(b,e)*d(a,f)*<j,k||g,h>*t1(g,k)*t1(h,i)
    H += -0.500000000000000 * einsum('cd,be,af,jkgh,gk,hi->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,e)*d(c,f)*d(i,j)*<l,k||g,d>*t1(g,k)*t1(a,l)
    H +=  1.000000000000000 * einsum('be,cf,ij,lkgd,gk,al->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,e)*d(b,f)*d(i,j)*<l,k||g,d>*t1(g,k)*t1(a,l)
    H += -1.000000000000000 * einsum('ce,bf,ij,lkgd,gk,al->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(c,f)*d(i,j)*<l,k||g,e>*t1(g,k)*t1(a,l)
    H += -1.000000000000000 * einsum('bd,cf,ij,lkge,gk,al->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,e)*d(i,j)*<l,k||g,f>*t1(g,k)*t1(a,l)
    H +=  1.000000000000000 * einsum('bd,ce,ij,lkgf,gk,al->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(b,f)*d(i,j)*<l,k||g,e>*t1(g,k)*t1(a,l)
    H +=  1.000000000000000 * einsum('cd,bf,ij,lkge,gk,al->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,e)*d(i,j)*<l,k||g,f>*t1(g,k)*t1(a,l)
    H += -1.000000000000000 * einsum('cd,be,ij,lkgf,gk,al->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,e)*d(c,f)*d(i,j)*<l,k||g,d>*t1(g,k)*t1(b,l)
    H += -1.000000000000000 * einsum('ae,cf,ij,lkgd,gk,bl->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,e)*d(a,f)*d(i,j)*<l,k||g,d>*t1(g,k)*t1(b,l)
    H +=  1.000000000000000 * einsum('ce,af,ij,lkgd,gk,bl->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*d(c,f)*d(i,j)*<l,k||g,e>*t1(g,k)*t1(b,l)
    H +=  1.000000000000000 * einsum('ad,cf,ij,lkge,gk,bl->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,e)*d(i,j)*<l,k||g,f>*t1(g,k)*t1(b,l)
    H += -1.000000000000000 * einsum('ad,ce,ij,lkgf,gk,bl->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(a,f)*d(i,j)*<l,k||g,e>*t1(g,k)*t1(b,l)
    H += -1.000000000000000 * einsum('cd,af,ij,lkge,gk,bl->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,e)*d(i,j)*<l,k||g,f>*t1(g,k)*t1(b,l)
    H +=  1.000000000000000 * einsum('cd,ae,ij,lkgf,gk,bl->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,e)*d(b,f)*d(i,j)*<l,k||g,d>*t1(g,k)*t1(c,l)
    H +=  1.000000000000000 * einsum('ae,bf,ij,lkgd,gk,cl->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,e)*d(a,f)*d(i,j)*<l,k||g,d>*t1(g,k)*t1(c,l)
    H += -1.000000000000000 * einsum('be,af,ij,lkgd,gk,cl->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(b,f)*d(i,j)*<l,k||g,e>*t1(g,k)*t1(c,l)
    H += -1.000000000000000 * einsum('ad,bf,ij,lkge,gk,cl->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*d(b,e)*d(i,j)*<l,k||g,f>*t1(g,k)*t1(c,l)
    H +=  1.000000000000000 * einsum('ad,be,ij,lkgf,gk,cl->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(a,f)*d(i,j)*<l,k||g,e>*t1(g,k)*t1(c,l)
    H +=  1.000000000000000 * einsum('bd,af,ij,lkge,gk,cl->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*d(i,j)*<l,k||g,f>*t1(g,k)*t1(c,l)
    H += -1.000000000000000 * einsum('bd,ae,ij,lkgf,gk,cl->abcidefj', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(a,d)*d(b,e)*d(c,f)*<j,k||g,h>*t1(g,i)*t1(h,k)
    H += -0.500000000000000 * einsum('ad,be,cf,jkgh,gi,hk->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(a,d)*d(c,e)*d(b,f)*<j,k||g,h>*t1(g,i)*t1(h,k)
    H +=  0.500000000000000 * einsum('ad,ce,bf,jkgh,gi,hk->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(b,d)*d(a,e)*d(c,f)*<j,k||g,h>*t1(g,i)*t1(h,k)
    H +=  0.500000000000000 * einsum('bd,ae,cf,jkgh,gi,hk->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(b,d)*d(c,e)*d(a,f)*<j,k||g,h>*t1(g,i)*t1(h,k)
    H += -0.500000000000000 * einsum('bd,ce,af,jkgh,gi,hk->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(c,d)*d(a,e)*d(b,f)*<j,k||g,h>*t1(g,i)*t1(h,k)
    H += -0.500000000000000 * einsum('cd,ae,bf,jkgh,gi,hk->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(c,d)*d(b,e)*d(a,f)*<j,k||g,h>*t1(g,i)*t1(h,k)
    H +=  0.500000000000000 * einsum('cd,be,af,jkgh,gi,hk->abcidefj', kd[v, v], kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 3), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,e)*d(c,f)*<j,k||g,d>*t1(g,i)*t1(a,k)
    H +=  1.000000000000000 * einsum('be,cf,jkgd,gi,ak->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,e)*d(b,f)*<j,k||g,d>*t1(g,i)*t1(a,k)
    H += -1.000000000000000 * einsum('ce,bf,jkgd,gi,ak->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(c,f)*<j,k||g,e>*t1(g,i)*t1(a,k)
    H += -1.000000000000000 * einsum('bd,cf,jkge,gi,ak->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(c,e)*<j,k||g,f>*t1(g,i)*t1(a,k)
    H +=  1.000000000000000 * einsum('bd,ce,jkgf,gi,ak->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(b,f)*<j,k||g,e>*t1(g,i)*t1(a,k)
    H +=  1.000000000000000 * einsum('cd,bf,jkge,gi,ak->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(b,e)*<j,k||g,f>*t1(g,i)*t1(a,k)
    H += -1.000000000000000 * einsum('cd,be,jkgf,gi,ak->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,e)*d(c,f)*<j,k||g,d>*t1(g,i)*t1(b,k)
    H += -1.000000000000000 * einsum('ae,cf,jkgd,gi,bk->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,e)*d(a,f)*<j,k||g,d>*t1(g,i)*t1(b,k)
    H +=  1.000000000000000 * einsum('ce,af,jkgd,gi,bk->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*d(c,f)*<j,k||g,e>*t1(g,i)*t1(b,k)
    H +=  1.000000000000000 * einsum('ad,cf,jkge,gi,bk->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(c,e)*<j,k||g,f>*t1(g,i)*t1(b,k)
    H += -1.000000000000000 * einsum('ad,ce,jkgf,gi,bk->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(a,f)*<j,k||g,e>*t1(g,i)*t1(b,k)
    H += -1.000000000000000 * einsum('cd,af,jkge,gi,bk->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,d)*d(a,e)*<j,k||g,f>*t1(g,i)*t1(b,k)
    H +=  1.000000000000000 * einsum('cd,ae,jkgf,gi,bk->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,e)*d(b,f)*<j,k||g,d>*t1(g,i)*t1(c,k)
    H +=  1.000000000000000 * einsum('ae,bf,jkgd,gi,ck->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,e)*d(a,f)*<j,k||g,d>*t1(g,i)*t1(c,k)
    H += -1.000000000000000 * einsum('be,af,jkgd,gi,ck->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(b,f)*<j,k||g,e>*t1(g,i)*t1(c,k)
    H += -1.000000000000000 * einsum('ad,bf,jkge,gi,ck->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,d)*d(b,e)*<j,k||g,f>*t1(g,i)*t1(c,k)
    H +=  1.000000000000000 * einsum('ad,be,jkgf,gi,ck->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(a,f)*<j,k||g,e>*t1(g,i)*t1(c,k)
    H +=  1.000000000000000 * einsum('bd,af,jkge,gi,ck->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,d)*d(a,e)*<j,k||g,f>*t1(g,i)*t1(c,k)
    H += -1.000000000000000 * einsum('bd,ae,jkgf,gi,ck->abcidefj', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,f)*d(i,j)*<l,k||d,e>*t1(a,k)*t1(b,l)
    H += -1.000000000000000 * einsum('cf,ij,lkde,ak,bl->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(c,e)*d(i,j)*<l,k||d,f>*t1(a,k)*t1(b,l)
    H +=  1.000000000000000 * einsum('ce,ij,lkdf,ak,bl->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(c,d)*d(i,j)*<l,k||e,f>*t1(a,k)*t1(b,l)
    H += -1.000000000000000 * einsum('cd,ij,lkef,ak,bl->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,f)*d(i,j)*<l,k||d,e>*t1(a,k)*t1(c,l)
    H +=  1.000000000000000 * einsum('bf,ij,lkde,ak,cl->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(b,e)*d(i,j)*<l,k||d,f>*t1(a,k)*t1(c,l)
    H += -1.000000000000000 * einsum('be,ij,lkdf,ak,cl->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(b,d)*d(i,j)*<l,k||e,f>*t1(a,k)*t1(c,l)
    H +=  1.000000000000000 * einsum('bd,ij,lkef,ak,cl->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,f)*d(i,j)*<l,k||d,e>*t1(b,k)*t1(c,l)
    H += -1.000000000000000 * einsum('af,ij,lkde,bk,cl->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(a,e)*d(i,j)*<l,k||d,f>*t1(b,k)*t1(c,l)
    H +=  1.000000000000000 * einsum('ae,ij,lkdf,bk,cl->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(a,d)*d(i,j)*<l,k||e,f>*t1(b,k)*t1(c,l)
    H += -1.000000000000000 * einsum('ad,ij,lkef,bk,cl->abcidefj', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    
    return H

def pack_dea_eom_ccsd_H(H22, H23, H32, H33, nsocc, nsvirt):

    n2 = 0
    for a in range (0, nsvirt):
        for b in range (a+1, nsvirt):
            n2 += 1

    n3 = 0
    for a in range (0, nsvirt):
        for b in range (a+1, nsvirt):
            for c in range (b+1, nsvirt):
                for i in range (0, nsocc):
                    n3 += 1

    dim = n2 + n3
    H = np.zeros((dim,dim))

    # 22 block
    ab = 0
    for a in range (0, nsvirt):
        for b in range (a+1, nsvirt):
            de = 0
            for d in range (0, nsvirt):
                for e in range (d+1, nsvirt):
                    H[ab,de] = H22[a,b,d,e]
                    de += 1
            ab += 1

    # 23, 32 blocks
    ab = 0
    for a in range (0, nsvirt):
        for b in range (a+1, nsvirt):
            defj = n2
            for d in range (0, nsvirt):
                for e in range (d+1, nsvirt):
                     for f in range (e+1, nsvirt):
                         for j in range (0, nsocc):
                             H[ab,defj] = H23[a,b,d,e,f,j]
                             H[defj,ab] = H32[d,e,f,j,a,b]
                             defj += 1
            ab += 1

    # 33 block
    abci = n2
    for a in range (0, nsvirt):
        for b in range (a+1, nsvirt):
            for c in range (b+1, nsvirt):
                for i in range (0, nsocc):
                    defj = n2
                    for d in range (0, nsvirt):
                        for e in range (d+1, nsvirt):
                             for f in range (e+1, nsvirt):
                                 for j in range (0, nsocc):
                                     H[abci,defj] = H33[a,b,c,i,d,e,f,j]
                                     defj += 1
                    abci += 1

    return H

def build_dea_eom_ccsd_H(f, g, o, v, t1, t2, nsocc, nsvirt):

    kd = np.zeros((nsocc+nsvirt,nsocc+nsvirt))
    for i in range (0,nsocc+nsvirt):
        kd[i,i] = 1.0

    H22 = dea_eom_ccsd_hamiltonian_22(kd, f, g, o, v, t1, t2)
    H23 = dea_eom_ccsd_hamiltonian_23(kd, f, g, o, v, t1, t2)
    H32 = dea_eom_ccsd_hamiltonian_32(kd, f, g, o, v, t1, t2)
    H33 = dea_eom_ccsd_hamiltonian_33(kd, f, g, o, v, t1, t2)

    H = pack_dea_eom_ccsd_H(H22, H23, H32, H33, nsocc, nsvirt)

    return H
