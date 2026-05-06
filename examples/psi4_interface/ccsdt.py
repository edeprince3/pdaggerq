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
spin-orbital CCSDT amplitude equations
"""
import numpy as np
from numpy import einsum

from ccsd import coupled_cluster_energy

def ccsdt_singles_residual(t1, t2, t3, f, g, o, v):

    #    < 0 | i* a e(-T) H e(T) | 0> :
    
    #	  1.0000 f(a,i)
    singles_res =  1.000000000000000 * einsum('ai->ai', f[v, o])
    
    #	 -1.0000 f(j,i)*t1(a,j)
    singles_res += -1.000000000000000 * einsum('ji,aj->ai', f[o, o], t1)
    
    #	  1.0000 f(a,b)*t1(b,i)
    singles_res +=  1.000000000000000 * einsum('ab,bi->ai', f[v, v], t1)
    
    #	 -1.0000 f(j,b)*t2(b,a,i,j)
    singles_res += -1.000000000000000 * einsum('jb,baij->ai', f[o, v], t2)
    
    #	 -1.0000 f(j,b)*t1(b,i)*t1(a,j)
    singles_res += -1.000000000000000 * einsum('jb,bi,aj->ai', f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,a||b,i>*t1(b,j)
    singles_res +=  1.000000000000000 * einsum('jabi,bj->ai', g[o, v, v, o], t1)
    
    #	 -0.5000 <k,j||b,i>*t2(b,a,k,j)
    singles_res += -0.500000000000000 * einsum('kjbi,bakj->ai', g[o, o, v, o], t2)
    
    #	 -0.5000 <j,a||b,c>*t2(b,c,i,j)
    singles_res += -0.500000000000000 * einsum('jabc,bcij->ai', g[o, v, v, v], t2)
    
    #	  0.2500 <k,j||b,c>*t3(b,c,a,i,k,j)
    singles_res +=  0.250000000000000 * einsum('kjbc,bcaikj->ai', g[o, o, v, v], t3)
    
    #	  1.0000 <k,j||b,c>*t1(b,j)*t2(c,a,i,k)
    singles_res +=  1.000000000000000 * einsum('kjbc,bj,caik->ai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,j||b,c>*t1(b,i)*t2(c,a,k,j)
    singles_res +=  0.500000000000000 * einsum('kjbc,bi,cakj->ai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <k,j||b,c>*t1(a,j)*t2(b,c,i,k)
    singles_res +=  0.500000000000000 * einsum('kjbc,aj,bcik->ai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <k,j||b,i>*t1(b,j)*t1(a,k)
    singles_res +=  1.000000000000000 * einsum('kjbi,bj,ak->ai', g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,a||b,c>*t1(b,j)*t1(c,i)
    singles_res +=  1.000000000000000 * einsum('jabc,bj,ci->ai', g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,j||b,c>*t1(b,j)*t1(c,i)*t1(a,k)
    singles_res +=  1.000000000000000 * einsum('kjbc,bj,ci,ak->ai', g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    return singles_res


def ccsdt_doubles_residual(t1, t2, t3, f, g, o, v):

    #    < 0 | i* j* b a e(-T) H e(T) | 0> :
    
    #	 -1.0000 P(i,j)f(k,j)*t2(a,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('kj,abik->abij', f[o, o], t2)
    doubles_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(a,b)f(a,c)*t2(c,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('ac,cbij->abij', f[v, v], t2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 f(k,c)*t3(c,a,b,i,j,k)
    doubles_res +=  1.000000000000000 * einsum('kc,cabijk->abij', f[o, v], t3)
    
    #	 -1.0000 P(i,j)f(k,c)*t1(c,j)*t2(a,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('kc,cj,abik->abij', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)f(k,c)*t1(a,k)*t2(c,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('kc,ak,cbij->abij', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 <a,b||i,j>
    doubles_res +=  1.000000000000000 * einsum('abij->abij', g[v, v, o, o])
    
    #	  1.0000 P(a,b)<k,a||i,j>*t1(b,k)
    contracted_intermediate =  1.000000000000000 * einsum('kaij,bk->abij', g[o, v, o, o], t1)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<a,b||c,j>*t1(c,i)
    contracted_intermediate =  1.000000000000000 * einsum('abcj,ci->abij', g[v, v, v, o], t1)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 <l,k||i,j>*t2(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkij,ablk->abij', g[o, o, o, o], t2)
    
    #	  1.0000 P(i,j)*P(a,b)<k,a||c,j>*t2(c,b,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('kacj,cbik->abij', g[o, v, v, o], t2)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 <a,b||c,d>*t2(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('abcd,cdij->abij', g[v, v, v, v], t2)
    
    #	  0.5000 P(i,j)<l,k||c,j>*t3(c,a,b,i,l,k)
    contracted_intermediate =  0.500000000000000 * einsum('lkcj,cabilk->abij', g[o, o, v, o], t3)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<k,a||c,d>*t3(c,d,b,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('kacd,cdbijk->abij', g[o, v, v, v], t3)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<l,k||c,j>*t1(c,k)*t2(a,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcj,ck,abil->abij', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<l,k||c,j>*t1(c,i)*t2(a,b,l,k)
    contracted_intermediate =  0.500000000000000 * einsum('lkcj,ci,ablk->abij', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<l,k||c,j>*t1(a,k)*t2(c,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lkcj,ak,cbil->abij', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<k,a||c,d>*t1(c,k)*t2(d,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('kacd,ck,dbij->abij', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<k,a||c,d>*t1(c,j)*t2(d,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('kacd,cj,dbik->abij', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<k,a||c,d>*t1(b,k)*t2(c,d,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('kacd,bk,cdij->abij', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 <l,k||c,d>*t1(c,k)*t3(d,a,b,i,j,l)
    doubles_res += -1.000000000000000 * einsum('lkcd,ck,dabijl->abij', g[o, o, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 P(i,j)<l,k||c,d>*t1(c,j)*t3(d,a,b,i,l,k)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,cj,dabilk->abij', g[o, o, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<l,k||c,d>*t1(a,k)*t3(c,d,b,i,j,l)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,ak,cdbijl->abij', g[o, o, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -1.0000 <l,k||i,j>*t1(a,k)*t1(b,l)
    doubles_res += -1.000000000000000 * einsum('lkij,ak,bl->abij', g[o, o, o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(i,j)*P(a,b)<k,a||c,j>*t1(c,i)*t1(b,k)
    contracted_intermediate =  1.000000000000000 * einsum('kacj,ci,bk->abij', g[o, v, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 <a,b||c,d>*t1(c,j)*t1(d,i)
    doubles_res += -1.000000000000000 * einsum('abcd,cj,di->abij', g[v, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 P(i,j)<l,k||c,d>*t2(c,d,j,k)*t2(a,b,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,cdjk,abil->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  0.2500 <l,k||c,d>*t2(c,d,i,j)*t2(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdij,ablk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>*t2(c,a,l,k)*t2(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(i,j)<l,k||c,d>*t2(c,a,j,k)*t2(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,cajk,dbil->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -0.5000 <l,k||c,d>*t2(c,a,i,j)*t2(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,caij,dblk->abij', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 P(i,j)<l,k||c,d>*t1(c,k)*t1(d,j)*t2(a,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,ck,dj,abil->abij', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<l,k||c,d>*t1(c,k)*t1(a,l)*t2(d,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,ck,al,dbij->abij', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	 -0.5000 <l,k||c,d>*t1(c,j)*t1(d,i)*t2(a,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,cj,di,ablk->abij', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(i,j)*P(a,b)<l,k||c,d>*t1(c,j)*t1(a,k)*t2(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,cj,ak,dbil->abij', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -0.5000 <l,k||c,d>*t1(a,k)*t1(b,l)*t2(c,d,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,ak,bl,cdij->abij', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(i,j)<l,k||c,j>*t1(c,i)*t1(a,k)*t1(b,l)
    contracted_intermediate = -1.000000000000000 * einsum('lkcj,ci,ak,bl->abij', g[o, o, v, o], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<k,a||c,d>*t1(c,j)*t1(d,i)*t1(b,k)
    contracted_intermediate = -1.000000000000000 * einsum('kacd,cj,di,bk->abij', g[o, v, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
    #	  1.0000 <l,k||c,d>*t1(c,j)*t1(d,i)*t1(a,k)*t1(b,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,cj,di,ak,bl->abij', g[o, o, v, v], t1, t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    return doubles_res


def ccsdt_triples_residual(t1, t2, t3, f, g, o, v):

    #    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :
    
    #	 -1.0000 P(j,k)f(l,k)*t3(a,b,c,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('lk,abcijl->abcijk', f[o, o], t3)
    triples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 f(l,i)*t3(a,b,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('li,abcjkl->abcijk', f[o, o], t3)
    
    #	  1.0000 P(a,b)f(a,d)*t3(d,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('ad,dbcijk->abcijk', f[v, v], t3)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  1.0000 f(c,d)*t3(d,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('cd,dabijk->abcijk', f[v, v], t3)
    
    #	 -1.0000 P(j,k)f(l,d)*t1(d,k)*t3(a,b,c,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('ld,dk,abcijl->abcijk', f[o, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 f(l,d)*t1(d,i)*t3(a,b,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('ld,di,abcjkl->abcijk', f[o, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(a,b)f(l,d)*t1(a,l)*t3(d,b,c,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('ld,al,dbcijk->abcijk', f[o, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -1.0000 f(l,d)*t1(c,l)*t3(d,a,b,i,j,k)
    triples_res += -1.000000000000000 * einsum('ld,cl,dabijk->abcijk', f[o, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(i,j)*P(a,b)f(l,d)*t2(d,a,j,k)*t2(b,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('ld,dajk,bcil->abcijk', f[o, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)f(l,d)*t2(d,a,i,j)*t2(b,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('ld,daij,bckl->abcijk', f[o, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)f(l,d)*t2(d,c,j,k)*t2(a,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('ld,dcjk,abil->abcijk', f[o, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	 -1.0000 f(l,d)*t2(d,c,i,j)*t2(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('ld,dcij,abkl->abcijk', f[o, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(i,j)*P(a,b)<l,a||j,k>*t2(b,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lajk,bcil->abcijk', g[o, v, o, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<l,a||i,j>*t2(b,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('laij,bckl->abcijk', g[o, v, o, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<l,c||j,k>*t2(a,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcjk,abil->abcijk', g[o, v, o, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	 -1.0000 <l,c||i,j>*t2(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcij,abkl->abcijk', g[o, v, o, o], t2)
    
    #	 -1.0000 P(j,k)*P(b,c)<a,b||d,k>*t2(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abdk,dcij->abcijk', g[v, v, v, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<a,b||d,i>*t2(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abdi,dcjk->abcijk', g[v, v, v, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<b,c||d,k>*t2(d,a,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('bcdk,daij->abcijk', g[v, v, v, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 <b,c||d,i>*t2(d,a,j,k)
    triples_res += -1.000000000000000 * einsum('bcdi,dajk->abcijk', g[v, v, v, o], t2)
    
    #	  0.5000 P(i,j)<m,l||j,k>*t3(a,b,c,i,m,l)
    contracted_intermediate =  0.500000000000000 * einsum('mljk,abciml->abcijk', g[o, o, o, o], t3)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	  0.5000 <m,l||i,j>*t3(a,b,c,k,m,l)
    triples_res +=  0.500000000000000 * einsum('mlij,abckml->abcijk', g[o, o, o, o], t3)
    
    #	  1.0000 P(j,k)*P(a,b)<l,a||d,k>*t3(d,b,c,i,j,l)
    contracted_intermediate =  1.000000000000000 * einsum('ladk,dbcijl->abcijk', g[o, v, v, o], t3)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<l,a||d,i>*t3(d,b,c,j,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('ladi,dbcjkl->abcijk', g[o, v, v, o], t3)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<l,c||d,k>*t3(d,a,b,i,j,l)
    contracted_intermediate =  1.000000000000000 * einsum('lcdk,dabijl->abcijk', g[o, v, v, o], t3)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	  1.0000 <l,c||d,i>*t3(d,a,b,j,k,l)
    triples_res +=  1.000000000000000 * einsum('lcdi,dabjkl->abcijk', g[o, v, v, o], t3)
    
    #	  0.5000 P(b,c)<a,b||d,e>*t3(d,e,c,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('abde,decijk->abcijk', g[v, v, v, v], t3)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	  0.5000 <b,c||d,e>*t3(d,e,a,i,j,k)
    triples_res +=  0.500000000000000 * einsum('bcde,deaijk->abcijk', g[v, v, v, v], t3)
    
    #	  1.0000 P(i,j)*P(a,b)<m,l||j,k>*t1(a,l)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mljk,al,bcim->abcijk', g[o, o, o, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,l||j,k>*t1(c,l)*t2(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mljk,cl,abim->abcijk', g[o, o, o, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,l||i,j>*t1(a,l)*t2(b,c,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlij,al,bckm->abcijk', g[o, o, o, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  1.0000 <m,l||i,j>*t1(c,l)*t2(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mlij,cl,abkm->abcijk', g[o, o, o, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(i,j)*P(a,b)<l,a||d,k>*t1(d,j)*t2(b,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('ladk,dj,bcil->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<l,a||d,k>*t1(b,l)*t2(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('ladk,bl,dcij->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<l,a||d,j>*t1(d,k)*t2(b,c,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('ladj,dk,bcil->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<l,a||d,i>*t1(d,k)*t2(b,c,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('ladi,dk,bcjl->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	 -1.0000 P(b,c)<l,a||d,i>*t1(b,l)*t2(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('ladi,bl,dcjk->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,c)<l,b||d,k>*t1(a,l)*t2(d,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lbdk,al,dcij->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->cbaikj', contracted_intermediate) 
    
    #	  1.0000 P(a,c)<l,b||d,i>*t1(a,l)*t2(d,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('lbdi,al,dcjk->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<l,c||d,k>*t1(d,j)*t2(a,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcdk,dj,abil->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<l,c||d,k>*t1(a,l)*t2(d,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('lcdk,al,dbij->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<l,c||d,j>*t1(d,k)*t2(a,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lcdj,dk,abil->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<l,c||d,i>*t1(d,k)*t2(a,b,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcdi,dk,abjl->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<l,c||d,i>*t1(a,l)*t2(d,b,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('lcdi,al,dbjk->abcijk', g[o, v, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<a,b||d,e>*t1(d,k)*t2(e,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('abde,dk,ecij->abcijk', g[v, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<a,b||d,e>*t1(d,i)*t2(e,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('abde,di,ecjk->abcijk', g[v, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<b,c||d,e>*t1(d,k)*t2(e,a,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('bcde,dk,eaij->abcijk', g[v, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	  1.0000 <b,c||d,e>*t1(d,i)*t2(e,a,j,k)
    triples_res +=  1.000000000000000 * einsum('bcde,di,eajk->abcijk', g[v, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(j,k)<m,l||d,k>*t1(d,l)*t3(a,b,c,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,dl,abcijm->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,l||d,k>*t1(d,j)*t3(a,b,c,i,m,l)
    contracted_intermediate =  0.500000000000000 * einsum('mldk,dj,abciml->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,l||d,k>*t1(a,l)*t3(d,b,c,i,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldk,al,dbcijm->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<m,l||d,k>*t1(c,l)*t3(d,a,b,i,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldk,cl,dabijm->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)<m,l||d,j>*t1(d,k)*t3(a,b,c,i,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mldj,dk,abciml->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate) 
    
    #	  1.0000 <m,l||d,i>*t1(d,l)*t3(a,b,c,j,k,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dl,abcjkm->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 P(j,k)<m,l||d,i>*t1(d,k)*t3(a,b,c,j,m,l)
    contracted_intermediate =  0.500000000000000 * einsum('mldi,dk,abcjml->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,l||d,i>*t1(a,l)*t3(d,b,c,j,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldi,al,dbcjkm->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -1.0000 <m,l||d,i>*t1(c,l)*t3(d,a,b,j,k,m)
    triples_res += -1.000000000000000 * einsum('mldi,cl,dabjkm->abcijk', g[o, o, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(a,b)<l,a||d,e>*t1(d,l)*t3(e,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dl,ebcijk->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<l,a||d,e>*t1(d,k)*t3(e,b,c,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('lade,dk,ebcijl->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<l,a||d,e>*t1(d,i)*t3(e,b,c,j,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('lade,di,ebcjkl->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  0.5000 P(b,c)<l,a||d,e>*t1(b,l)*t3(d,e,c,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('lade,bl,decijk->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	 -0.5000 P(a,c)<l,b||d,e>*t1(a,l)*t3(d,e,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lbde,al,decijk->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate) 
    
    #	  1.0000 <l,c||d,e>*t1(d,l)*t3(e,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dl,eabijk->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 P(j,k)<l,c||d,e>*t1(d,k)*t3(e,a,b,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcde,dk,eabijl->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 <l,c||d,e>*t1(d,i)*t3(e,a,b,j,k,l)
    triples_res += -1.000000000000000 * einsum('lcde,di,eabjkl->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 P(a,b)<l,c||d,e>*t1(a,l)*t3(d,e,b,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('lcde,al,debijk->abcijk', g[o, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<m,l||d,e>*t2(d,e,k,l)*t3(a,b,c,i,j,m)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,dekl,abcijm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -0.5000 <m,l||d,e>*t2(d,e,i,l)*t3(a,b,c,j,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,deil,abcjkm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 P(i,j)<m,l||d,e>*t2(d,e,j,k)*t3(a,b,c,i,m,l)
    contracted_intermediate =  0.250000000000000 * einsum('mlde,dejk,abciml->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	  0.2500 <m,l||d,e>*t2(d,e,i,j)*t3(a,b,c,k,m,l)
    triples_res +=  0.250000000000000 * einsum('mlde,deij,abckml->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 P(a,b)<m,l||d,e>*t2(d,a,m,l)*t3(e,b,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,daml,ebcijk->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,l||d,e>*t2(d,a,k,l)*t3(e,b,c,i,j,m)
    contracted_intermediate =  1.000000000000010 * einsum('mlde,dakl,ebcijm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,l||d,e>*t2(d,a,i,l)*t3(e,b,c,j,k,m)
    contracted_intermediate =  1.000000000000010 * einsum('mlde,dail,ebcjkm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<m,l||d,e>*t2(d,a,j,k)*t3(e,b,c,i,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,dajk,ebciml->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,l||d,e>*t2(d,a,i,j)*t3(e,b,c,k,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,daij,ebckml->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -0.5000 <m,l||d,e>*t2(d,c,m,l)*t3(e,a,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,dcml,eabijk->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(j,k)<m,l||d,e>*t2(d,c,k,l)*t3(e,a,b,i,j,m)
    contracted_intermediate =  1.000000000000010 * einsum('mlde,dckl,eabijm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	  1.0000 <m,l||d,e>*t2(d,c,i,l)*t3(e,a,b,j,k,m)
    triples_res +=  1.000000000000010 * einsum('mlde,dcil,eabjkm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 P(i,j)<m,l||d,e>*t2(d,c,j,k)*t3(e,a,b,i,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,dcjk,eabiml->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	 -0.5000 <m,l||d,e>*t2(d,c,i,j)*t3(e,a,b,k,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dcij,eabkml->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 P(b,c)<m,l||d,e>*t2(a,b,m,l)*t3(d,e,c,i,j,k)
    contracted_intermediate =  0.250000000000000 * einsum('mlde,abml,decijk->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(b,c)<m,l||d,e>*t2(a,b,k,l)*t3(d,e,c,i,j,m)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,abkl,decijm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<m,l||d,e>*t2(a,b,i,l)*t3(d,e,c,j,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,abil,decjkm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	  0.2500 <m,l||d,e>*t2(b,c,m,l)*t3(d,e,a,i,j,k)
    triples_res +=  0.250000000000000 * einsum('mlde,bcml,deaijk->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 P(j,k)<m,l||d,e>*t2(b,c,k,l)*t3(d,e,a,i,j,m)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,bckl,deaijm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -0.5000 <m,l||d,e>*t2(b,c,i,l)*t3(d,e,a,j,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,bcil,deajkm->abcijk', g[o, o, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 P(i,j)*P(a,b)<m,l||d,k>*t2(d,a,j,l)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,dajl,bcim->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<m,l||d,k>*t2(d,a,i,j)*t2(b,c,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mldk,daij,bcml->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,l||d,k>*t2(d,c,j,l)*t2(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,dcjl,abim->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<m,l||d,k>*t2(d,c,i,j)*t2(a,b,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mldk,dcij,abml->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,l||d,j>*t2(d,a,k,l)*t2(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,dakl,bcim->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<m,l||d,j>*t2(d,c,k,l)*t2(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,dckl,abim->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,l||d,i>*t2(d,a,k,l)*t2(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,dakl,bcjm->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,l||d,i>*t2(d,a,j,k)*t2(b,c,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mldi,dajk,bcml->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<m,l||d,i>*t2(d,c,k,l)*t2(a,b,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,dckl,abjm->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -0.5000 <m,l||d,i>*t2(d,c,j,k)*t2(a,b,m,l)
    triples_res += -0.500000000000000 * einsum('mldi,dcjk,abml->abcijk', g[o, o, v, o], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 P(i,j)*P(a,b)<l,a||d,e>*t2(d,e,j,k)*t2(b,c,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('lade,dejk,bcil->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<l,a||d,e>*t2(d,e,i,j)*t2(b,c,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('lade,deij,bckl->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<l,a||d,e>*t2(d,b,k,l)*t2(e,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dbkl,ecij->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<l,a||d,e>*t2(d,b,i,l)*t2(e,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dbil,ecjk->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<l,a||d,e>*t2(d,b,j,k)*t2(e,c,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dbjk,ecil->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<l,a||d,e>*t2(d,b,i,j)*t2(e,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dbij,eckl->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<l,c||d,e>*t2(d,e,j,k)*t2(a,b,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('lcde,dejk,abil->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	 -0.5000 <l,c||d,e>*t2(d,e,i,j)*t2(a,b,k,l)
    triples_res += -0.500000000000000 * einsum('lcde,deij,abkl->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(j,k)<l,c||d,e>*t2(d,a,k,l)*t2(e,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,dakl,ebij->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	  1.0000 <l,c||d,e>*t2(d,a,i,l)*t2(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dail,ebjk->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(i,j)<l,c||d,e>*t2(d,a,j,k)*t2(e,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,dajk,ebil->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	  1.0000 <l,c||d,e>*t2(d,a,i,j)*t2(e,b,k,l)
    triples_res +=  1.000000000000000 * einsum('lcde,daij,ebkl->abcijk', g[o, v, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 P(i,j)*P(a,b)<m,l||d,e>*t1(d,l)*t2(e,a,j,k)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dl,eajk,bcim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,l||d,e>*t1(d,l)*t2(e,a,i,j)*t2(b,c,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dl,eaij,bckm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,l||d,e>*t1(d,l)*t2(e,c,j,k)*t2(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dl,ecjk,abim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	  1.0000 <m,l||d,e>*t1(d,l)*t2(e,c,i,j)*t2(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ecij,abkm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,e>*t1(d,k)*t2(e,a,j,l)*t2(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,dk,eajl,bcim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(a,b)<m,l||d,e>*t1(d,k)*t2(e,a,i,j)*t2(b,c,m,l)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,dk,eaij,bcml->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,l||d,e>*t1(d,k)*t2(e,c,j,l)*t2(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,dk,ecjl,abim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	  0.5000 P(j,k)<m,l||d,e>*t1(d,k)*t2(e,c,i,j)*t2(a,b,m,l)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,dk,ecij,abml->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,l||d,e>*t1(d,j)*t2(e,a,k,l)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dj,eakl,bcim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<m,l||d,e>*t1(d,j)*t2(e,c,k,l)*t2(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dj,eckl,abim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,l||d,e>*t1(d,i)*t2(e,a,k,l)*t2(b,c,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,di,eakl,bcjm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,l||d,e>*t1(d,i)*t2(e,a,j,k)*t2(b,c,m,l)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,di,eajk,bcml->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<m,l||d,e>*t1(d,i)*t2(e,c,k,l)*t2(a,b,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,di,eckl,abjm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	  0.5000 <m,l||d,e>*t1(d,i)*t2(e,c,j,k)*t2(a,b,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,di,ecjk,abml->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 P(i,j)*P(a,b)<m,l||d,e>*t1(a,l)*t2(d,e,j,k)*t2(b,c,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,al,dejk,bcim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,l||d,e>*t1(a,l)*t2(d,e,i,j)*t2(b,c,k,m)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,al,deij,bckm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,l||d,e>*t1(a,l)*t2(d,b,k,m)*t2(e,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,dbkm,ecij->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,l||d,e>*t1(a,l)*t2(d,b,i,m)*t2(e,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,dbim,ecjk->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,e>*t1(a,l)*t2(d,b,j,k)*t2(e,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,dbjk,ecim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,l||d,e>*t1(a,l)*t2(d,b,i,j)*t2(e,c,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,dbij,eckm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<m,l||d,e>*t1(c,l)*t2(d,e,j,k)*t2(a,b,i,m)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,cl,dejk,abim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	  0.5000 <m,l||d,e>*t1(c,l)*t2(d,e,i,j)*t2(a,b,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,cl,deij,abkm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(j,k)<m,l||d,e>*t1(c,l)*t2(d,a,k,m)*t2(e,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,cl,dakm,ebij->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 <m,l||d,e>*t1(c,l)*t2(d,a,i,m)*t2(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,cl,daim,ebjk->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(i,j)<m,l||d,e>*t1(c,l)*t2(d,a,j,k)*t2(e,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,cl,dajk,ebim->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	 -1.0000 <m,l||d,e>*t1(c,l)*t2(d,a,i,j)*t2(e,b,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,cl,daij,ebkm->abcijk', g[o, o, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 P(i,j)*P(a,b)<m,l||d,k>*t1(d,j)*t1(a,l)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,dj,al,bcim->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<m,l||d,k>*t1(d,j)*t1(c,l)*t2(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,dj,cl,abim->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<m,l||d,k>*t1(a,l)*t1(b,m)*t2(d,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,al,bm,dcij->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<m,l||d,k>*t1(b,l)*t1(c,m)*t2(d,a,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,bl,cm,daij->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,l||d,j>*t1(d,k)*t1(a,l)*t2(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,dk,al,bcim->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<m,l||d,j>*t1(d,k)*t1(c,l)*t2(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,dk,cl,abim->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,l||d,i>*t1(d,k)*t1(a,l)*t2(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,dk,al,bcjm->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<m,l||d,i>*t1(d,k)*t1(c,l)*t2(a,b,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,dk,cl,abjm->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<m,l||d,i>*t1(a,l)*t1(b,m)*t2(d,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,al,bm,dcjk->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	  1.0000 <m,l||d,i>*t1(b,l)*t1(c,m)*t2(d,a,j,k)
    triples_res +=  1.000000000000000 * einsum('mldi,bl,cm,dajk->abcijk', g[o, o, v, o], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(i,j)*P(a,b)<l,a||d,e>*t1(d,k)*t1(e,j)*t2(b,c,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dk,ej,bcil->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<l,a||d,e>*t1(d,k)*t1(b,l)*t2(e,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dk,bl,ecij->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<l,a||d,e>*t1(d,j)*t1(e,i)*t2(b,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dj,ei,bckl->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  1.0000 P(b,c)<l,a||d,e>*t1(d,i)*t1(b,l)*t2(e,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('lade,di,bl,ecjk->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,c)<l,b||d,e>*t1(d,k)*t1(a,l)*t2(e,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('lbde,dk,al,ecij->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->cbaikj', contracted_intermediate) 
    
    #	 -1.0000 P(a,c)<l,b||d,e>*t1(d,i)*t1(a,l)*t2(e,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('lbde,di,al,ecjk->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<l,c||d,e>*t1(d,k)*t1(e,j)*t2(a,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,dk,ej,abil->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<l,c||d,e>*t1(d,k)*t1(a,l)*t2(e,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,dk,al,ebij->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	  1.0000 <l,c||d,e>*t1(d,j)*t1(e,i)*t2(a,b,k,l)
    triples_res +=  1.000000000000000 * einsum('lcde,dj,ei,abkl->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(a,b)<l,c||d,e>*t1(d,i)*t1(a,l)*t2(e,b,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,di,al,ebjk->abcijk', g[o, v, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<m,l||d,e>*t1(d,l)*t1(e,k)*t3(a,b,c,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dl,ek,abcijm->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	  1.0000 <m,l||d,e>*t1(d,l)*t1(e,i)*t3(a,b,c,j,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ei,abcjkm->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(a,b)<m,l||d,e>*t1(d,l)*t1(a,m)*t3(e,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dl,am,ebcijk->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  1.0000 <m,l||d,e>*t1(d,l)*t1(c,m)*t3(e,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,cm,eabijk->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 P(i,j)<m,l||d,e>*t1(d,k)*t1(e,j)*t3(a,b,c,i,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,dk,ej,abciml->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,l||d,e>*t1(d,k)*t1(a,l)*t3(e,b,c,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dk,al,ebcijm->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<m,l||d,e>*t1(d,k)*t1(c,l)*t3(e,a,b,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dk,cl,eabijm->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -0.5000 <m,l||d,e>*t1(d,j)*t1(e,i)*t3(a,b,c,k,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dj,ei,abckml->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(a,b)<m,l||d,e>*t1(d,i)*t1(a,l)*t3(e,b,c,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,di,al,ebcjkm->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  1.0000 <m,l||d,e>*t1(d,i)*t1(c,l)*t3(e,a,b,j,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,cl,eabjkm->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 P(b,c)<m,l||d,e>*t1(a,l)*t1(b,m)*t3(d,e,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,al,bm,decijk->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	 -0.5000 <m,l||d,e>*t1(b,l)*t1(c,m)*t3(d,e,a,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,bl,cm,deaijk->abcijk', g[o, o, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,e>*t1(d,k)*t1(e,j)*t1(a,l)*t2(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,dk,ej,al,bcim->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<m,l||d,e>*t1(d,k)*t1(e,j)*t1(c,l)*t2(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,dk,ej,cl,abim->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<m,l||d,e>*t1(d,k)*t1(a,l)*t1(b,m)*t2(e,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,dk,al,bm,ecij->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<m,l||d,e>*t1(d,k)*t1(b,l)*t1(c,m)*t2(e,a,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,dk,bl,cm,eaij->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,l||d,e>*t1(d,j)*t1(e,i)*t1(a,l)*t2(b,c,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,dj,ei,al,bckm->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -1.0000 <m,l||d,e>*t1(d,j)*t1(e,i)*t1(c,l)*t2(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,ei,cl,abkm->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 P(b,c)<m,l||d,e>*t1(d,i)*t1(a,l)*t1(b,m)*t2(e,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,di,al,bm,ecjk->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate) 
    
    #	 -1.0000 <m,l||d,e>*t1(d,i)*t1(b,l)*t1(c,m)*t2(e,a,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,di,bl,cm,eajk->abcijk', g[o, o, v, v], t1, t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    return triples_res

def ccsdt_iterations_with_spin(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, 
        t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb,
        f_aa, f_bb, g_aaaa, g_bbbb, g_abab,
        oa, ob, va, vb, e_aa_ai, e_bb_ai, e_aaaa_abij, e_bbbb_abij, e_abab_abij, 
        e_aaaaaa_abcijk, e_aabaab_abcijk, e_abbabb_abcijk, e_bbbbbb_abcijk,
        hf_energy, max_iter=100,
        e_convergence=1e-8,r_convergence=1e-8,diis_size=None, diis_start_cycle=4):

    # initialize diis if diis_size is not None
    # else normal scf iterate

    if diis_size is not None:
        from diis import DIIS
        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        t1_aa_end = t1_aa.size
        t1_bb_end = t1_aa_end + t1_bb.size
        t2_aaaa_end = t1_bb_end + t2_aaaa.size
        t2_bbbb_end = t2_aaaa_end + t2_bbbb.size
        t2_abab_end = t2_bbbb_end + t2_abab.size
        t3_aaaaaa_end = t2_abab_end + t3_aaaaaa.size
        t3_aabaab_end = t3_aaaaaa_end + t3_aabaab.size
        t3_abbabb_end = t3_aabaab_end + t3_abbabb.size
        t3_bbbbbb_end = t3_abbabb_end + t3_bbbbbb.size
        old_vec = np.hstack((t1_aa.flatten(), t1_bb.flatten(), 
                             t2_aaaa.flatten(), t2_bbbb.flatten(), t2_abab.flatten(), 
                             t3_aaaaaa.flatten(), t3_aabaab.flatten(), t3_abbabb.flatten(), t3_bbbbbb.flatten()))

    fock_e_aa_ai = np.reciprocal(e_aa_ai)
    fock_e_bb_ai = np.reciprocal(e_bb_ai)

    fock_e_aaaa_abij = np.reciprocal(e_aaaa_abij)
    fock_e_bbbb_abij = np.reciprocal(e_bbbb_abij)
    fock_e_abab_abij = np.reciprocal(e_abab_abij)

    fock_e_aaaaaa_abcijk = np.reciprocal(e_aaaaaa_abcijk)
    fock_e_aabaab_abcijk = np.reciprocal(e_aabaab_abcijk)
    fock_e_abbabb_abcijk = np.reciprocal(e_abbabb_abcijk)
    fock_e_bbbbbb_abcijk = np.reciprocal(e_bbbbbb_abcijk)

    from ccsd import ccsd_energy_with_spin
    old_energy = ccsd_energy_with_spin(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)

    print("")
    print("    ==> CCSDT amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(max_iter):

        residual_t1_aa = ccsdt_t1_aa_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)
        residual_t1_bb = ccsdt_t1_bb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)

        residual_t2_aaaa = ccsdt_t2_aaaa_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)
        residual_t2_bbbb = ccsdt_t2_bbbb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)
        residual_t2_abab = ccsdt_t2_abab_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)

        residual_t3_aaaaaa = ccsdt_t3_aaaaaa_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)
        residual_t3_aabaab = ccsdt_t3_aabaab_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)
        residual_t3_abbabb = ccsdt_t3_abbabb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)
        residual_t3_bbbbbb = ccsdt_t3_bbbbbb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)

        res_norm = ( np.linalg.norm(residual_t1_aa)
                   + np.linalg.norm(residual_t1_bb)
                   + np.linalg.norm(residual_t2_aaaa)
                   + np.linalg.norm(residual_t2_bbbb)
                   + np.linalg.norm(residual_t2_abab) 
                   + np.linalg.norm(residual_t3_aaaaaa) 
                   + np.linalg.norm(residual_t3_aabaab) 
                   + np.linalg.norm(residual_t3_abbabb) 
                   + np.linalg.norm(residual_t3_bbbbbb) )

        t1_aa_res = residual_t1_aa + fock_e_aa_ai * t1_aa
        t1_bb_res = residual_t1_bb + fock_e_bb_ai * t1_bb

        t2_aaaa_res = residual_t2_aaaa + fock_e_aaaa_abij * t2_aaaa
        t2_bbbb_res = residual_t2_bbbb + fock_e_bbbb_abij * t2_bbbb
        t2_abab_res = residual_t2_abab + fock_e_abab_abij * t2_abab

        t3_aaaaaa_res = residual_t3_aaaaaa + fock_e_aaaaaa_abcijk * t3_aaaaaa
        t3_aabaab_res = residual_t3_aabaab + fock_e_aabaab_abcijk * t3_aabaab
        t3_abbabb_res = residual_t3_abbabb + fock_e_abbabb_abcijk * t3_abbabb
        t3_bbbbbb_res = residual_t3_bbbbbb + fock_e_bbbbbb_abcijk * t3_bbbbbb

        new_t1_aa = t1_aa_res * e_aa_ai
        new_t1_bb = t1_bb_res * e_bb_ai

        new_t2_aaaa = t2_aaaa_res * e_aaaa_abij
        new_t2_bbbb = t2_bbbb_res * e_bbbb_abij
        new_t2_abab = t2_abab_res * e_abab_abij

        new_t3_aaaaaa = t3_aaaaaa_res * e_aaaaaa_abcijk
        new_t3_aabaab = t3_aabaab_res * e_aabaab_abcijk
        new_t3_abbabb = t3_abbabb_res * e_abbabb_abcijk
        new_t3_bbbbbb = t3_bbbbbb_res * e_bbbbbb_abcijk

        # diis update
        if diis_size is not None:
            vectorized_iterate = np.hstack(
                (new_t1_aa.flatten(), new_t1_bb.flatten(), 
                 new_t2_aaaa.flatten(), new_t2_bbbb.flatten(), new_t2_abab.flatten(), 
                 new_t3_aaaaaa.flatten(), new_t3_aabaab.flatten(), new_t3_abbabb.flatten(), new_t3_bbbbbb.flatten()))
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                 error_vec)
            new_t1_aa = new_vectorized_iterate[:t1_aa_end].reshape(t1_aa.shape)
            new_t1_bb = new_vectorized_iterate[t1_aa_end:t1_bb_end].reshape(t1_bb.shape)

            new_t2_aaaa = new_vectorized_iterate[t1_bb_end:t2_aaaa_end].reshape(t2_aaaa.shape)
            new_t2_bbbb = new_vectorized_iterate[t2_aaaa_end:t2_bbbb_end].reshape(t2_bbbb.shape)
            new_t2_abab = new_vectorized_iterate[t2_bbbb_end:t2_abab_end].reshape(t2_abab.shape)

            new_t3_aaaaaa = new_vectorized_iterate[t2_abab_end:t3_aaaaaa_end].reshape(t3_aaaaaa.shape)
            new_t3_aabaab = new_vectorized_iterate[t3_aaaaaa_end:t3_aabaab_end].reshape(t3_aabaab.shape)
            new_t3_abbabb = new_vectorized_iterate[t3_aabaab_end:t3_abbabb_end].reshape(t3_abbabb.shape)
            new_t3_bbbbbb = new_vectorized_iterate[t3_abbabb_end:t3_bbbbbb_end].reshape(t3_bbbbbb.shape)

            old_vec = new_vectorized_iterate

        current_energy = ccsd_energy_with_spin(new_t1_aa, new_t1_bb, new_t2_aaaa, new_t2_bbbb, new_t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb)

        delta_e = np.abs(old_energy - current_energy)

        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(idx, current_energy - hf_energy, delta_e, res_norm))
        if delta_e < e_convergence and res_norm < r_convergence:
            # assign t1 and t2 variables for future use before breaking
            t1_aa = new_t1_aa
            t1_bb = new_t1_bb

            t2_aaaa = new_t2_aaaa
            t2_bbbb = new_t2_bbbb
            t2_abab = new_t2_abab

            t3_aaaaaa = new_t3_aaaaaa
            t3_aabaab = new_t3_aabaab
            t3_abbabb = new_t3_abbabb
            t3_bbbbbb = new_t3_bbbbbb

            break
        else:
            # assign t1 and t2 and old_energy for next iteration
            t1_aa = new_t1_aa
            t1_bb = new_t1_bb

            t2_aaaa = new_t2_aaaa
            t2_bbbb = new_t2_bbbb
            t2_abab = new_t2_abab

            t3_aaaaaa = new_t3_aaaaaa
            t3_aabaab = new_t3_aabaab
            t3_abbabb = new_t3_abbabb
            t3_bbbbbb = new_t3_bbbbbb

            old_energy = current_energy

    else:
        raise ValueError("CCSDT iterations did not converge")


    return t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb

def ccsdt_iterations(t1, t2, t3, fock, g, o, v, e_ai, e_abij, e_abcijk, hf_energy, max_iter=100, 
        e_convergence=1e-8,r_convergence=1e-8,diis_size=None, diis_start_cycle=4):
           

    # initialize diis if diis_size is not None
    # else normal scf iterate

    if diis_size is not None:
        from diis import DIIS
        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        t1_dim = t1.size
        t2_dim = t2.size 
        old_vec = np.hstack((t1.flatten(), t2.flatten(), t3.flatten()))

    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)
    fock_e_abcijk = np.reciprocal(e_abcijk)
    old_energy = coupled_cluster_energy(t1, t2, fock, g, o, v)

    print("")
    print("    ==> CCSDT amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(max_iter):

        residual_singles = ccsdt_singles_residual(t1, t2, t3, fock, g, o, v)
        residual_doubles = ccsdt_doubles_residual(t1, t2, t3, fock, g, o, v)
        residual_triples = ccsdt_triples_residual(t1, t2, t3, fock, g, o, v)

        res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles) + np.linalg.norm(residual_triples)

        singles_res = residual_singles + fock_e_ai * t1
        doubles_res = residual_doubles + fock_e_abij * t2
        triples_res = residual_triples + fock_e_abcijk * t3

        new_singles = singles_res * e_ai
        new_doubles = doubles_res * e_abij
        new_triples = triples_res * e_abcijk

        # diis update
        if diis_size is not None:
            vectorized_iterate = np.hstack(
                (new_singles.flatten(), new_doubles.flatten(), new_triples.flatten()))
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                 error_vec)
            new_singles = new_vectorized_iterate[:t1_dim].reshape(t1.shape)
            new_doubles = new_vectorized_iterate[t1_dim:t1_dim+t2_dim].reshape(t2.shape)
            new_triples = new_vectorized_iterate[t1_dim+t2_dim:].reshape(t3.shape)
            old_vec = new_vectorized_iterate

        current_energy = coupled_cluster_energy(new_singles, new_doubles, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(idx, current_energy - hf_energy, delta_e, res_norm))

        # assign t1, t2, t3 
        t1 = new_singles
        t2 = new_doubles
        t3 = new_triples
        old_energy = current_energy

        if delta_e < e_convergence and res_norm < r_convergence:
            break

    else:
        raise ValueError("CCSD iterations did not converge")

    return t1, t2, t3

#    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :
#
#['-1.00000000000000', 'P(j,k)', 'f(l,k)', 't3(a,b,c,i,j,l)']
#['-1.00000000000000', 'f(l,i)', 't3(a,b,c,j,k,l)']
#['+1.00000000000000', 'P(a,b)', 'f(a,d)', 't3(d,b,c,i,j,k)']
#['+1.00000000000000', 'f(c,d)', 't3(d,a,b,i,j,k)']
#['-1.00000000000000', 'P(i,j)', 'P(a,b)', '<l,a||j,k>', 't2(b,c,i,l)']
#['-1.00000000000000', 'P(a,b)', '<l,a||i,j>', 't2(b,c,k,l)']
#['-1.00000000000000', 'P(i,j)', '<l,c||j,k>', 't2(a,b,i,l)']
#['-1.00000000000000', '<l,c||i,j>', 't2(a,b,k,l)']
#['-1.00000000000000', 'P(j,k)', 'P(b,c)', '<a,b||d,k>', 't2(d,c,i,j)']
#['-1.00000000000000', 'P(b,c)', '<a,b||d,i>', 't2(d,c,j,k)']
#['-1.00000000000000', 'P(j,k)', '<b,c||d,k>', 't2(d,a,i,j)']
#['-1.00000000000000', '<b,c||d,i>', 't2(d,a,j,k)']

def perturbative_triples_residual(t1, t2, t3, f, g, o, v):
    """

    evaluate perturbative triples residual 

    triples_res = < abc;ijk | e(-t3) F e(t3) + e(-t2) V e(t2) | 0 >

    :param t2: CCSD doubles amplitudes
    :param t3: approximate triples amplitudes (should be zero)
    :param fock: fock matrix
    :param g: two-electron integrals
    :param o: occupied orbitals slice
    :param v: virtual orbitals slice

    :return triples_res: defined above

    """

    #        -1.0000 P(j,k)f(l,k)*t3(a,b,c,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('lk,abcijl->abcijk', f[o, o], t3)
    triples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)
    
    #        -1.0000 f(l,i)*t3(a,b,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('li,abcjkl->abcijk', f[o, o], t3)
    
    #         1.0000 P(a,b)f(a,d)*t3(d,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('ad,dbcijk->abcijk', f[v, v], t3)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)
    
    #         1.0000 f(c,d)*t3(d,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('cd,dabijk->abcijk', f[v, v], t3)
    
    #        -1.0000 P(i,j)*P(a,b)<l,a||j,k>*t2(b,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lajk,bcil->abcijk', g[o, v, o, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)
    
    #        -1.0000 P(a,b)<l,a||i,j>*t2(b,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('laij,bckl->abcijk', g[o, v, o, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)
    
    #        -1.0000 P(i,j)<l,c||j,k>*t2(a,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcjk,abil->abcijk', g[o, v, o, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)
    
    #        -1.0000 <l,c||i,j>*t2(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcij,abkl->abcijk', g[o, v, o, o], t2)
    
    #        -1.0000 P(j,k)*P(b,c)<a,b||d,k>*t2(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abdk,dcij->abcijk', g[v, v, v, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)
    
    #        -1.0000 P(b,c)<a,b||d,i>*t2(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abdi,dcjk->abcijk', g[v, v, v, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)
    
    #        -1.0000 P(j,k)<b,c||d,k>*t2(d,a,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('bcdk,daij->abcijk', g[v, v, v, o], t2)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)
    
    #        -1.0000 <b,c||d,i>*t2(d,a,j,k)
    triples_res += -1.000000000000000 * einsum('bcdi,dajk->abcijk', g[v, v, v, o], t2)
    
    return triples_res

#    E(t)
#
#['+0.25000000000000', '<k,j||b,c>', 'l1(i,a)', 't3(b,c,a,i,k,j)']
#['+0.25000000000000', '<l,k||c,j>', 'l2(i,j,b,a)', 't3(c,b,a,i,l,k)']
#['+0.25000000000000', '<k,b||c,d>', 'l2(i,j,b,a)', 't3(c,d,a,i,j,k)']

def perturbative_triples_energy(t1, t2, t3, f, g, o, v):
    """

    evaluate the (T) correction to CCSD energy

    :param t1: CCSD singles amplitudes
    :param t2: CCSD doubles amplitudes
    :param t3: approximate triples amplitudes 
    :param fock: fock matrix
    :param g: two-electron integrals
    :param o: occupied orbitals slice
    :param v: virtual orbitals slice

    :return energy: the perurbative triples correction to the ccsd energy

    """

    l1 = t1.transpose(1, 0)
    l2 = t2.transpose(2, 3, 0, 1)

    #         0.2500 <k,j||b,c>*l1(i,a)*t3(b,c,a,i,k,j)
    energy =  0.250000000000000 * einsum('kjbc,ia,bcaikj', g[o, o, v, v], l1, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #         0.2500 <l,k||c,j>*l2(i,j,b,a)*t3(c,b,a,i,l,k)
    energy +=  0.250000000000000 * einsum('lkcj,ijba,cbailk', g[o, o, v, o], l2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #         0.2500 <k,b||c,d>*l2(i,j,b,a)*t3(c,d,a,i,j,k)
    energy +=  0.250000000000000 * einsum('kbcd,ijba,cdaijk', g[o, v, v, v], l2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    
    return energy

def perturbative_triples_correction(t1, t2, t3, fock, g, o, v, e_abcijk):
    """

    evaluate the (T) correction to CCSD energy

    :param t1: CCSD singles amplitudes
    :param t2: CCSD doubles amplitudes
    :param t3: approximate triples amplitudes (to be determined)
    :param fock: fock matrix
    :param g: two-electron integrals
    :param o: occupied orbitals slice
    :param v: virtual orbitals slice
    :param e_abcijk: triples energy denominator 

    :return et: the perurbative triples correction to the ccsd energy, (t)

    """

    fock_e_abcijk = np.reciprocal(e_abcijk)

    residual_triples = perturbative_triples_residual(t1, t2, t3, fock, g, o, v)
    triples_res = residual_triples + fock_e_abcijk * t3
    new_triples = triples_res * e_abcijk
    t3 = new_triples

    et = perturbative_triples_energy(t1, t2, t3, fock, g, o, v)

    return et


def ccsdt_t1_aa_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* a e(-T) H e(T) | 0> :

    #	  1.0000 f_aa(a,i)
    singles_res  =  1.000000000000000 * einsum('ai->ai', f_aa[va, oa])

    #	 -1.0000 f_aa(j,i)*t1_aa(a,j)
    singles_res += -1.000000000000000 * einsum('ji,aj->ai', f_aa[oa, oa], t1_aa)

    #	  1.0000 f_aa(a,b)*t1_aa(b,i)
    singles_res +=  1.000000000000000 * einsum('ab,bi->ai', f_aa[va, va], t1_aa)

    #	 -1.0000 f_aa(j,b)*t2_aaaa(b,a,i,j)
    singles_res += -1.000000000000000 * einsum('jb,baij->ai', f_aa[oa, va], t2_aaaa)

    #	  1.0000 f_bb(j,b)*t2_abab(a,b,i,j)
    singles_res +=  1.000000000000000 * einsum('jb,abij->ai', f_bb[ob, vb], t2_abab)

    #	 -1.0000 f_aa(j,b)*t1_aa(a,j)*t1_aa(b,i)
    singles_res += -1.000000000000000 * einsum('jb,aj,bi->ai', f_aa[oa, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <j,a||b,i>_aaaa*t1_aa(b,j)
    singles_res +=  1.000000000000000 * einsum('jabi,bj->ai', g_aaaa[oa, va, va, oa], t1_aa)

    #	  1.0000 <a,j||i,b>_abab*t1_bb(b,j)
    singles_res +=  1.000000000000000 * einsum('ajib,bj->ai', g_abab[va, ob, oa, vb], t1_bb)

    #	 -0.5000 <k,j||b,i>_aaaa*t2_aaaa(b,a,k,j)
    singles_res += -0.500000000000000 * einsum('kjbi,bakj->ai', g_aaaa[oa, oa, va, oa], t2_aaaa)

    #	 -0.5000 <k,j||i,b>_abab*t2_abab(a,b,k,j)
    singles_res += -0.500000000000000 * einsum('kjib,abkj->ai', g_abab[oa, ob, oa, vb], t2_abab)

    #	 -0.5000 <j,k||i,b>_abab*t2_abab(a,b,j,k)
    singles_res += -0.500000000000000 * einsum('jkib,abjk->ai', g_abab[oa, ob, oa, vb], t2_abab)

    #	 -0.5000 <j,a||b,c>_aaaa*t2_aaaa(b,c,i,j)
    singles_res += -0.500000000000000 * einsum('jabc,bcij->ai', g_aaaa[oa, va, va, va], t2_aaaa)

    #	  0.5000 <a,j||b,c>_abab*t2_abab(b,c,i,j)
    singles_res +=  0.500000000000000 * einsum('ajbc,bcij->ai', g_abab[va, ob, va, vb], t2_abab)

    #	  0.5000 <a,j||c,b>_abab*t2_abab(c,b,i,j)
    singles_res +=  0.500000000000000 * einsum('ajcb,cbij->ai', g_abab[va, ob, va, vb], t2_abab)

    #	  0.2500 <k,j||b,c>_aaaa*t3_aaaaaa(b,c,a,i,k,j)
    singles_res +=  0.250000000000000 * einsum('kjbc,bcaikj->ai', g_aaaa[oa, oa, va, va], t3_aaaaaa)

    #	 -0.2500 <k,j||b,c>_abab*t3_aabaab(b,a,c,i,k,j)
    singles_res += -0.250000000000000 * einsum('kjbc,bacikj->ai', g_abab[oa, ob, va, vb], t3_aabaab)

    #	 -0.2500 <j,k||b,c>_abab*t3_aabaab(b,a,c,i,j,k)
    singles_res += -0.250000000000000 * einsum('jkbc,bacijk->ai', g_abab[oa, ob, va, vb], t3_aabaab)

    #	  0.2500 <k,j||c,b>_abab*t3_aabaab(a,c,b,i,k,j)
    singles_res +=  0.250000000000000 * einsum('kjcb,acbikj->ai', g_abab[oa, ob, va, vb], t3_aabaab)

    #	  0.2500 <j,k||c,b>_abab*t3_aabaab(a,c,b,i,j,k)
    singles_res +=  0.250000000000000 * einsum('jkcb,acbijk->ai', g_abab[oa, ob, va, vb], t3_aabaab)

    #	 -0.2500 <k,j||b,c>_bbbb*t3_abbabb(a,c,b,i,k,j)
    singles_res += -0.250000000000000 * einsum('kjbc,acbikj->ai', g_bbbb[ob, ob, vb, vb], t3_abbabb)

    #	  1.0000 <k,j||b,c>_aaaa*t2_aaaa(c,a,i,k)*t1_aa(b,j)
    singles_res +=  1.000000000000000 * einsum('kjbc,caik,bj->ai', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <k,j||c,b>_abab*t2_aaaa(c,a,i,k)*t1_bb(b,j)
    singles_res += -1.000000000000000 * einsum('kjcb,caik,bj->ai', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,k||b,c>_abab*t2_abab(a,c,i,k)*t1_aa(b,j)
    singles_res +=  1.000000000000000 * einsum('jkbc,acik,bj->ai', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <k,j||b,c>_bbbb*t2_abab(a,c,i,k)*t1_bb(b,j)
    singles_res += -1.000000000000000 * einsum('kjbc,acik,bj->ai', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <k,j||b,c>_aaaa*t2_aaaa(c,a,k,j)*t1_aa(b,i)
    singles_res +=  0.500000000000000 * einsum('kjbc,cakj,bi->ai', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <k,j||b,c>_abab*t2_abab(a,c,k,j)*t1_aa(b,i)
    singles_res += -0.500000000000000 * einsum('kjbc,ackj,bi->ai', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <j,k||b,c>_abab*t2_abab(a,c,j,k)*t1_aa(b,i)
    singles_res += -0.500000000000000 * einsum('jkbc,acjk,bi->ai', g_abab[oa, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <k,j||b,c>_aaaa*t1_aa(a,j)*t2_aaaa(b,c,i,k)
    singles_res +=  0.500000000000000 * einsum('kjbc,aj,bcik->ai', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <j,k||b,c>_abab*t1_aa(a,j)*t2_abab(b,c,i,k)
    singles_res += -0.500000000000000 * einsum('jkbc,aj,bcik->ai', g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <j,k||c,b>_abab*t1_aa(a,j)*t2_abab(c,b,i,k)
    singles_res += -0.500000000000000 * einsum('jkcb,aj,cbik->ai', g_abab[oa, ob, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <k,j||b,i>_aaaa*t1_aa(a,k)*t1_aa(b,j)
    singles_res +=  1.000000000000000 * einsum('kjbi,ak,bj->ai', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <k,j||i,b>_abab*t1_aa(a,k)*t1_bb(b,j)
    singles_res += -1.000000000000000 * einsum('kjib,ak,bj->ai', g_abab[oa, ob, oa, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,a||b,c>_aaaa*t1_aa(b,j)*t1_aa(c,i)
    singles_res +=  1.000000000000000 * einsum('jabc,bj,ci->ai', g_aaaa[oa, va, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <a,j||c,b>_abab*t1_bb(b,j)*t1_aa(c,i)
    singles_res +=  1.000000000000000 * einsum('ajcb,bj,ci->ai', g_abab[va, ob, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <k,j||b,c>_aaaa*t1_aa(a,k)*t1_aa(b,j)*t1_aa(c,i)
    singles_res +=  1.000000000000000 * einsum('kjbc,ak,bj,ci->ai', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 <k,j||c,b>_abab*t1_aa(a,k)*t1_bb(b,j)*t1_aa(c,i)
    singles_res += -1.000000000000000 * einsum('kjcb,ak,bj,ci->ai', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    return singles_res


def ccsdt_t1_bb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* a e(-T) H e(T) | 0> :

    #	  1.0000 f_bb(a,i)
    singles_res  =  1.000000000000000 * einsum('ai->ai', f_bb[vb, ob])

    #	 -1.0000 f_bb(j,i)*t1_bb(a,j)
    singles_res += -1.000000000000000 * einsum('ji,aj->ai', f_bb[ob, ob], t1_bb)

    #	  1.0000 f_bb(a,b)*t1_bb(b,i)
    singles_res +=  1.000000000000000 * einsum('ab,bi->ai', f_bb[vb, vb], t1_bb)

    #	  1.0000 f_aa(j,b)*t2_abab(b,a,j,i)
    singles_res +=  1.000000000000000 * einsum('jb,baji->ai', f_aa[oa, va], t2_abab)

    #	 -1.0000 f_bb(j,b)*t2_bbbb(b,a,i,j)
    singles_res += -1.000000000000000 * einsum('jb,baij->ai', f_bb[ob, vb], t2_bbbb)

    #	 -1.0000 f_bb(j,b)*t1_bb(a,j)*t1_bb(b,i)
    singles_res += -1.000000000000000 * einsum('jb,aj,bi->ai', f_bb[ob, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,a||b,i>_abab*t1_aa(b,j)
    singles_res +=  1.000000000000000 * einsum('jabi,bj->ai', g_abab[oa, vb, va, ob], t1_aa)

    #	  1.0000 <j,a||b,i>_bbbb*t1_bb(b,j)
    singles_res +=  1.000000000000000 * einsum('jabi,bj->ai', g_bbbb[ob, vb, vb, ob], t1_bb)

    #	 -0.5000 <k,j||b,i>_abab*t2_abab(b,a,k,j)
    singles_res += -0.500000000000000 * einsum('kjbi,bakj->ai', g_abab[oa, ob, va, ob], t2_abab)

    #	 -0.5000 <j,k||b,i>_abab*t2_abab(b,a,j,k)
    singles_res += -0.500000000000000 * einsum('jkbi,bajk->ai', g_abab[oa, ob, va, ob], t2_abab)

    #	 -0.5000 <k,j||b,i>_bbbb*t2_bbbb(b,a,k,j)
    singles_res += -0.500000000000000 * einsum('kjbi,bakj->ai', g_bbbb[ob, ob, vb, ob], t2_bbbb)

    #	  0.5000 <j,a||b,c>_abab*t2_abab(b,c,j,i)
    singles_res +=  0.500000000000000 * einsum('jabc,bcji->ai', g_abab[oa, vb, va, vb], t2_abab)

    #	  0.5000 <j,a||c,b>_abab*t2_abab(c,b,j,i)
    singles_res +=  0.500000000000000 * einsum('jacb,cbji->ai', g_abab[oa, vb, va, vb], t2_abab)

    #	 -0.5000 <j,a||b,c>_bbbb*t2_bbbb(b,c,i,j)
    singles_res += -0.500000000000000 * einsum('jabc,bcij->ai', g_bbbb[ob, vb, vb, vb], t2_bbbb)

    #	 -0.2500 <k,j||b,c>_aaaa*t3_aabaab(b,c,a,j,k,i)
    singles_res += -0.250000000000000 * einsum('kjbc,bcajki->ai', g_aaaa[oa, oa, va, va], t3_aabaab)

    #	 -0.2500 <k,j||b,c>_abab*t3_abbabb(b,c,a,k,i,j)
    singles_res += -0.250000000000000 * einsum('kjbc,bcakij->ai', g_abab[oa, ob, va, vb], t3_abbabb)

    #	  0.2500 <j,k||b,c>_abab*t3_abbabb(b,c,a,j,k,i)
    singles_res +=  0.250000000000000 * einsum('jkbc,bcajki->ai', g_abab[oa, ob, va, vb], t3_abbabb)

    #	 -0.2500 <k,j||c,b>_abab*t3_abbabb(c,b,a,k,i,j)
    singles_res += -0.250000000000000 * einsum('kjcb,cbakij->ai', g_abab[oa, ob, va, vb], t3_abbabb)

    #	  0.2500 <j,k||c,b>_abab*t3_abbabb(c,b,a,j,k,i)
    singles_res +=  0.250000000000000 * einsum('jkcb,cbajki->ai', g_abab[oa, ob, va, vb], t3_abbabb)

    #	  0.2500 <k,j||b,c>_bbbb*t3_bbbbbb(b,c,a,i,k,j)
    singles_res +=  0.250000000000000 * einsum('kjbc,bcaikj->ai', g_bbbb[ob, ob, vb, vb], t3_bbbbbb)

    #	 -1.0000 <k,j||b,c>_aaaa*t2_abab(c,a,k,i)*t1_aa(b,j)
    singles_res += -1.000000000000000 * einsum('kjbc,caki,bj->ai', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <k,j||c,b>_abab*t2_abab(c,a,k,i)*t1_bb(b,j)
    singles_res +=  1.000000000000000 * einsum('kjcb,caki,bj->ai', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <j,k||b,c>_abab*t2_bbbb(c,a,i,k)*t1_aa(b,j)
    singles_res += -1.000000000000000 * einsum('jkbc,caik,bj->ai', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <k,j||b,c>_bbbb*t2_bbbb(c,a,i,k)*t1_bb(b,j)
    singles_res +=  1.000000000000000 * einsum('kjbc,caik,bj->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <k,j||c,b>_abab*t2_abab(c,a,k,j)*t1_bb(b,i)
    singles_res += -0.500000000000000 * einsum('kjcb,cakj,bi->ai', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <j,k||c,b>_abab*t2_abab(c,a,j,k)*t1_bb(b,i)
    singles_res += -0.500000000000000 * einsum('jkcb,cajk,bi->ai', g_abab[oa, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <k,j||b,c>_bbbb*t2_bbbb(c,a,k,j)*t1_bb(b,i)
    singles_res +=  0.500000000000000 * einsum('kjbc,cakj,bi->ai', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <k,j||b,c>_abab*t1_bb(a,j)*t2_abab(b,c,k,i)
    singles_res += -0.500000000000000 * einsum('kjbc,aj,bcki->ai', g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <k,j||c,b>_abab*t1_bb(a,j)*t2_abab(c,b,k,i)
    singles_res += -0.500000000000000 * einsum('kjcb,aj,cbki->ai', g_abab[oa, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <k,j||b,c>_bbbb*t1_bb(a,j)*t2_bbbb(b,c,i,k)
    singles_res +=  0.500000000000000 * einsum('kjbc,aj,bcik->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <j,k||b,i>_abab*t1_bb(a,k)*t1_aa(b,j)
    singles_res += -1.000000000000000 * einsum('jkbi,ak,bj->ai', g_abab[oa, ob, va, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <k,j||b,i>_bbbb*t1_bb(a,k)*t1_bb(b,j)
    singles_res +=  1.000000000000000 * einsum('kjbi,ak,bj->ai', g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,a||b,c>_abab*t1_aa(b,j)*t1_bb(c,i)
    singles_res +=  1.000000000000000 * einsum('jabc,bj,ci->ai', g_abab[oa, vb, va, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <j,a||b,c>_bbbb*t1_bb(b,j)*t1_bb(c,i)
    singles_res +=  1.000000000000000 * einsum('jabc,bj,ci->ai', g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <j,k||b,c>_abab*t1_bb(a,k)*t1_aa(b,j)*t1_bb(c,i)
    singles_res += -1.000000000000000 * einsum('jkbc,ak,bj,ci->ai', g_abab[oa, ob, va, vb], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	  1.0000 <k,j||b,c>_bbbb*t1_bb(a,k)*t1_bb(b,j)*t1_bb(c,i)
    singles_res +=  1.000000000000000 * einsum('kjbc,ak,bj,ci->ai', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    return singles_res


def ccsdt_t2_aaaa_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* b a e(-T) H e(T) | 0> :

    #	 -1.0000 P(i,j)f_aa(k,j)*t2_aaaa(a,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('kj,abik->abij', f_aa[oa, oa], t2_aaaa)
    doubles_res  =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  1.0000 P(a,b)f_aa(a,c)*t2_aaaa(c,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('ac,cbij->abij', f_aa[va, va], t2_aaaa)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  1.0000 f_aa(k,c)*t3_aaaaaa(c,a,b,i,j,k)
    doubles_res +=  1.000000000000000 * einsum('kc,cabijk->abij', f_aa[oa, va], t3_aaaaaa)

    #	 -1.0000 f_bb(k,c)*t3_aabaab(b,a,c,i,j,k)
    doubles_res += -1.000000000000000 * einsum('kc,bacijk->abij', f_bb[ob, vb], t3_aabaab)

    #	 -1.0000 P(i,j)f_aa(k,c)*t2_aaaa(a,b,i,k)*t1_aa(c,j)
    contracted_intermediate = -1.000000000000000 * einsum('kc,abik,cj->abij', f_aa[oa, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -1.0000 P(a,b)f_aa(k,c)*t1_aa(a,k)*t2_aaaa(c,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('kc,ak,cbij->abij', f_aa[oa, va], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  1.0000 <a,b||i,j>_aaaa
    doubles_res +=  1.000000000000000 * einsum('abij->abij', g_aaaa[va, va, oa, oa])

    #	  1.0000 P(a,b)<k,a||i,j>_aaaa*t1_aa(b,k)
    contracted_intermediate =  1.000000000000000 * einsum('kaij,bk->abij', g_aaaa[oa, va, oa, oa], t1_aa)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  1.0000 P(i,j)<a,b||c,j>_aaaa*t1_aa(c,i)
    contracted_intermediate =  1.000000000000000 * einsum('abcj,ci->abij', g_aaaa[va, va, va, oa], t1_aa)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 <l,k||i,j>_aaaa*t2_aaaa(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkij,ablk->abij', g_aaaa[oa, oa, oa, oa], t2_aaaa)

    #	  1.0000 P(i,j)*P(a,b)<k,a||c,j>_aaaa*t2_aaaa(c,b,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('kacj,cbik->abij', g_aaaa[oa, va, va, oa], t2_aaaa)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<a,k||j,c>_abab*t2_abab(b,c,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('akjc,bcik->abij', g_abab[va, ob, oa, vb], t2_abab)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	  0.5000 <a,b||c,d>_aaaa*t2_aaaa(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('abcd,cdij->abij', g_aaaa[va, va, va, va], t2_aaaa)

    #	  0.5000 P(i,j)<l,k||c,j>_aaaa*t3_aaaaaa(c,a,b,i,l,k)
    contracted_intermediate =  0.500000000000000 * einsum('lkcj,cabilk->abij', g_aaaa[oa, oa, va, oa], t3_aaaaaa)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 P(i,j)<l,k||j,c>_abab*t3_aabaab(b,a,c,i,l,k)
    contracted_intermediate =  0.500000000000000 * einsum('lkjc,bacilk->abij', g_abab[oa, ob, oa, vb], t3_aabaab)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 P(i,j)<k,l||j,c>_abab*t3_aabaab(b,a,c,i,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('kljc,bacikl->abij', g_abab[oa, ob, oa, vb], t3_aabaab)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 P(a,b)<k,a||c,d>_aaaa*t3_aaaaaa(c,d,b,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('kacd,cdbijk->abij', g_aaaa[oa, va, va, va], t3_aaaaaa)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  0.5000 P(a,b)<a,k||c,d>_abab*t3_aabaab(c,b,d,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('akcd,cbdijk->abij', g_abab[va, ob, va, vb], t3_aabaab)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -0.5000 P(a,b)<a,k||d,c>_abab*t3_aabaab(b,d,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('akdc,bdcijk->abij', g_abab[va, ob, va, vb], t3_aabaab)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  1.0000 P(i,j)<l,k||c,j>_aaaa*t2_aaaa(a,b,i,l)*t1_aa(c,k)
    contracted_intermediate =  1.000000000000000 * einsum('lkcj,abil,ck->abij', g_aaaa[oa, oa, va, oa], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -1.0000 P(i,j)<l,k||j,c>_abab*t2_aaaa(a,b,i,l)*t1_bb(c,k)
    contracted_intermediate = -1.000000000000000 * einsum('lkjc,abil,ck->abij', g_abab[oa, ob, oa, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 P(i,j)<l,k||c,j>_aaaa*t2_aaaa(a,b,l,k)*t1_aa(c,i)
    contracted_intermediate =  0.500000000000000 * einsum('lkcj,ablk,ci->abij', g_aaaa[oa, oa, va, oa], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<l,k||c,j>_aaaa*t1_aa(a,k)*t2_aaaa(c,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lkcj,ak,cbil->abij', g_aaaa[oa, oa, va, oa], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<k,l||j,c>_abab*t1_aa(a,k)*t2_abab(b,c,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('kljc,ak,bcil->abij', g_abab[oa, ob, oa, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	  1.0000 P(a,b)<k,a||c,d>_aaaa*t2_aaaa(d,b,i,j)*t1_aa(c,k)
    contracted_intermediate =  1.000000000000000 * einsum('kacd,dbij,ck->abij', g_aaaa[oa, va, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  1.0000 P(a,b)<a,k||d,c>_abab*t2_aaaa(d,b,i,j)*t1_bb(c,k)
    contracted_intermediate =  1.000000000000000 * einsum('akdc,dbij,ck->abij', g_abab[va, ob, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<k,a||c,d>_aaaa*t2_aaaa(d,b,i,k)*t1_aa(c,j)
    contracted_intermediate = -1.000000000000000 * einsum('kacd,dbik,cj->abij', g_aaaa[oa, va, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<a,k||c,d>_abab*t2_abab(b,d,i,k)*t1_aa(c,j)
    contracted_intermediate = -1.000000000000000 * einsum('akcd,bdik,cj->abij', g_abab[va, ob, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	  0.5000 P(a,b)<k,a||c,d>_aaaa*t1_aa(b,k)*t2_aaaa(c,d,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('kacd,bk,cdij->abij', g_aaaa[oa, va, va, va], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -1.0000 <l,k||c,d>_aaaa*t3_aaaaaa(d,a,b,i,j,l)*t1_aa(c,k)
    doubles_res += -1.000000000000000 * einsum('lkcd,dabijl,ck->abij', g_aaaa[oa, oa, va, va], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,k||d,c>_abab*t3_aaaaaa(d,a,b,i,j,l)*t1_bb(c,k)
    doubles_res +=  1.000000000000000 * einsum('lkdc,dabijl,ck->abij', g_abab[oa, ob, va, vb], t3_aaaaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <k,l||c,d>_abab*t3_aabaab(b,a,d,i,j,l)*t1_aa(c,k)
    doubles_res += -1.000000000000000 * einsum('klcd,badijl,ck->abij', g_abab[oa, ob, va, vb], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,k||c,d>_bbbb*t3_aabaab(b,a,d,i,j,l)*t1_bb(c,k)
    doubles_res +=  1.000000000000000 * einsum('lkcd,badijl,ck->abij', g_bbbb[ob, ob, vb, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 P(i,j)<l,k||c,d>_aaaa*t3_aaaaaa(d,a,b,i,l,k)*t1_aa(c,j)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,dabilk,cj->abij', g_aaaa[oa, oa, va, va], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 P(i,j)<l,k||c,d>_abab*t3_aabaab(b,a,d,i,l,k)*t1_aa(c,j)
    contracted_intermediate =  0.500000000000000 * einsum('lkcd,badilk,cj->abij', g_abab[oa, ob, va, vb], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 P(i,j)<k,l||c,d>_abab*t3_aabaab(b,a,d,i,k,l)*t1_aa(c,j)
    contracted_intermediate =  0.500000000000000 * einsum('klcd,badikl,cj->abij', g_abab[oa, ob, va, vb], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -0.5000 P(a,b)<l,k||c,d>_aaaa*t1_aa(a,k)*t3_aaaaaa(c,d,b,i,j,l)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,ak,cdbijl->abij', g_aaaa[oa, oa, va, va], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -0.5000 P(a,b)<k,l||c,d>_abab*t1_aa(a,k)*t3_aabaab(c,b,d,i,j,l)
    contracted_intermediate = -0.500000000000000 * einsum('klcd,ak,cbdijl->abij', g_abab[oa, ob, va, vb], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  0.5000 P(a,b)<k,l||d,c>_abab*t1_aa(a,k)*t3_aabaab(b,d,c,i,j,l)
    contracted_intermediate =  0.500000000000000 * einsum('kldc,ak,bdcijl->abij', g_abab[oa, ob, va, vb], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -1.0000 <l,k||i,j>_aaaa*t1_aa(a,k)*t1_aa(b,l)
    doubles_res += -1.000000000000000 * einsum('lkij,ak,bl->abij', g_aaaa[oa, oa, oa, oa], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<k,a||c,j>_aaaa*t1_aa(b,k)*t1_aa(c,i)
    contracted_intermediate =  1.000000000000000 * einsum('kacj,bk,ci->abij', g_aaaa[oa, va, va, oa], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	 -1.0000 <a,b||c,d>_aaaa*t1_aa(c,j)*t1_aa(d,i)
    doubles_res += -1.000000000000000 * einsum('abcd,cj,di->abij', g_aaaa[va, va, va, va], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 P(i,j)<l,k||c,d>_aaaa*t2_aaaa(a,b,i,l)*t2_aaaa(c,d,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,abil,cdjk->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -0.5000 P(i,j)<l,k||c,d>_abab*t2_aaaa(a,b,i,l)*t2_abab(c,d,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,abil,cdjk->abij', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -0.5000 P(i,j)<l,k||d,c>_abab*t2_aaaa(a,b,i,l)*t2_abab(d,c,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lkdc,abil,dcjk->abij', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.2500 <l,k||c,d>_aaaa*t2_aaaa(a,b,l,k)*t2_aaaa(c,d,i,j)
    doubles_res +=  0.250000000000000 * einsum('lkcd,ablk,cdij->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,k||c,d>_aaaa*t2_aaaa(c,a,l,k)*t2_aaaa(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,k||d,c>_abab*t2_abab(a,c,l,k)*t2_aaaa(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkdc,aclk,dbij->abij', g_abab[oa, ob, va, vb], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <k,l||d,c>_abab*t2_abab(a,c,k,l)*t2_aaaa(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('kldc,ackl,dbij->abij', g_abab[oa, ob, va, vb], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)<l,k||c,d>_aaaa*t2_aaaa(c,a,j,k)*t2_aaaa(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,cajk,dbil->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  1.0000 P(i,j)<k,l||c,d>_abab*t2_aaaa(c,a,j,k)*t2_abab(b,d,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('klcd,cajk,bdil->abij', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  1.0000 P(i,j)<l,k||d,c>_abab*t2_abab(a,c,j,k)*t2_aaaa(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkdc,acjk,dbil->abij', g_abab[oa, ob, va, vb], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  1.0000 P(i,j)<l,k||c,d>_bbbb*t2_abab(a,c,j,k)*t2_abab(b,d,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,acjk,bdil->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -0.5000 <l,k||c,d>_aaaa*t2_aaaa(c,a,i,j)*t2_aaaa(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,caij,dblk->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,k||c,d>_abab*t2_aaaa(c,a,i,j)*t2_abab(b,d,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcd,caij,bdlk->abij', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <k,l||c,d>_abab*t2_aaaa(c,a,i,j)*t2_abab(b,d,k,l)
    doubles_res +=  0.500000000000000 * einsum('klcd,caij,bdkl->abij', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 P(i,j)<l,k||c,d>_aaaa*t2_aaaa(a,b,i,l)*t1_aa(c,k)*t1_aa(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,abil,ck,dj->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -1.0000 P(i,j)<l,k||d,c>_abab*t2_aaaa(a,b,i,l)*t1_bb(c,k)*t1_aa(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('lkdc,abil,ck,dj->abij', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  1.0000 P(a,b)<l,k||c,d>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,j)*t1_aa(c,k)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,al,dbij,ck->abij', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,k||d,c>_abab*t1_aa(a,l)*t2_aaaa(d,b,i,j)*t1_bb(c,k)
    contracted_intermediate = -1.000000000000000 * einsum('lkdc,al,dbij,ck->abij', g_abab[oa, ob, va, vb], t1_aa, t2_aaaa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -0.5000 <l,k||c,d>_aaaa*t2_aaaa(a,b,l,k)*t1_aa(c,j)*t1_aa(d,i)
    doubles_res += -0.500000000000000 * einsum('lkcd,ablk,cj,di->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<l,k||c,d>_aaaa*t1_aa(a,k)*t2_aaaa(d,b,i,l)*t1_aa(c,j)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,ak,dbil,cj->abij', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<k,l||c,d>_abab*t1_aa(a,k)*t2_abab(b,d,i,l)*t1_aa(c,j)
    contracted_intermediate =  1.000000000000000 * einsum('klcd,ak,bdil,cj->abij', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	 -0.5000 <l,k||c,d>_aaaa*t1_aa(a,k)*t1_aa(b,l)*t2_aaaa(c,d,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,ak,bl,cdij->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)<l,k||c,j>_aaaa*t1_aa(a,k)*t1_aa(b,l)*t1_aa(c,i)
    contracted_intermediate = -1.000000000000000 * einsum('lkcj,ak,bl,ci->abij', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -1.0000 P(a,b)<k,a||c,d>_aaaa*t1_aa(b,k)*t1_aa(c,j)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('kacd,bk,cj,di->abij', g_aaaa[oa, va, va, va], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  1.0000 <l,k||c,d>_aaaa*t1_aa(a,k)*t1_aa(b,l)*t1_aa(c,j)*t1_aa(d,i)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ak,bl,cj,di->abij', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])

    return doubles_res


def ccsdt_t2_bbbb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* b a e(-T) H e(T) | 0> :

    #	 -1.0000 P(i,j)f_bb(k,j)*t2_bbbb(a,b,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('kj,abik->abij', f_bb[ob, ob], t2_bbbb)
    doubles_res  =  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  1.0000 P(a,b)f_bb(a,c)*t2_bbbb(c,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('ac,cbij->abij', f_bb[vb, vb], t2_bbbb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -1.0000 f_aa(k,c)*t3_abbabb(c,a,b,k,j,i)
    doubles_res += -1.000000000000000 * einsum('kc,cabkji->abij', f_aa[oa, va], t3_abbabb)

    #	  1.0000 f_bb(k,c)*t3_bbbbbb(c,a,b,i,j,k)
    doubles_res +=  1.000000000000000 * einsum('kc,cabijk->abij', f_bb[ob, vb], t3_bbbbbb)

    #	 -1.0000 P(i,j)f_bb(k,c)*t2_bbbb(a,b,i,k)*t1_bb(c,j)
    contracted_intermediate = -1.000000000000000 * einsum('kc,abik,cj->abij', f_bb[ob, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -1.0000 P(a,b)f_bb(k,c)*t1_bb(a,k)*t2_bbbb(c,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('kc,ak,cbij->abij', f_bb[ob, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  1.0000 <a,b||i,j>_bbbb
    doubles_res +=  1.000000000000000 * einsum('abij->abij', g_bbbb[vb, vb, ob, ob])

    #	  1.0000 P(a,b)<k,a||i,j>_bbbb*t1_bb(b,k)
    contracted_intermediate =  1.000000000000000 * einsum('kaij,bk->abij', g_bbbb[ob, vb, ob, ob], t1_bb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  1.0000 P(i,j)<a,b||c,j>_bbbb*t1_bb(c,i)
    contracted_intermediate =  1.000000000000000 * einsum('abcj,ci->abij', g_bbbb[vb, vb, vb, ob], t1_bb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 <l,k||i,j>_bbbb*t2_bbbb(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkij,ablk->abij', g_bbbb[ob, ob, ob, ob], t2_bbbb)

    #	 -1.0000 P(i,j)*P(a,b)<k,a||c,j>_abab*t2_abab(c,b,k,i)
    contracted_intermediate = -1.000000000000000 * einsum('kacj,cbki->abij', g_abab[oa, vb, va, ob], t2_abab)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<k,a||c,j>_bbbb*t2_bbbb(c,b,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('kacj,cbik->abij', g_bbbb[ob, vb, vb, ob], t2_bbbb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	  0.5000 <a,b||c,d>_bbbb*t2_bbbb(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('abcd,cdij->abij', g_bbbb[vb, vb, vb, vb], t2_bbbb)

    #	 -0.5000 P(i,j)<l,k||c,j>_abab*t3_abbabb(c,a,b,l,i,k)
    contracted_intermediate = -0.500000000000000 * einsum('lkcj,cablik->abij', g_abab[oa, ob, va, ob], t3_abbabb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 P(i,j)<k,l||c,j>_abab*t3_abbabb(c,a,b,k,l,i)
    contracted_intermediate =  0.500000000000000 * einsum('klcj,cabkli->abij', g_abab[oa, ob, va, ob], t3_abbabb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 P(i,j)<l,k||c,j>_bbbb*t3_bbbbbb(c,a,b,i,l,k)
    contracted_intermediate =  0.500000000000000 * einsum('lkcj,cabilk->abij', g_bbbb[ob, ob, vb, ob], t3_bbbbbb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -0.5000 P(a,b)<k,a||c,d>_abab*t3_abbabb(c,d,b,k,j,i)
    contracted_intermediate = -0.500000000000000 * einsum('kacd,cdbkji->abij', g_abab[oa, vb, va, vb], t3_abbabb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -0.5000 P(a,b)<k,a||d,c>_abab*t3_abbabb(d,c,b,k,j,i)
    contracted_intermediate = -0.500000000000000 * einsum('kadc,dcbkji->abij', g_abab[oa, vb, va, vb], t3_abbabb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  0.5000 P(a,b)<k,a||c,d>_bbbb*t3_bbbbbb(c,d,b,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('kacd,cdbijk->abij', g_bbbb[ob, vb, vb, vb], t3_bbbbbb)
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -1.0000 P(i,j)<k,l||c,j>_abab*t2_bbbb(a,b,i,l)*t1_aa(c,k)
    contracted_intermediate = -1.000000000000000 * einsum('klcj,abil,ck->abij', g_abab[oa, ob, va, ob], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  1.0000 P(i,j)<l,k||c,j>_bbbb*t2_bbbb(a,b,i,l)*t1_bb(c,k)
    contracted_intermediate =  1.000000000000000 * einsum('lkcj,abil,ck->abij', g_bbbb[ob, ob, vb, ob], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 P(i,j)<l,k||c,j>_bbbb*t2_bbbb(a,b,l,k)*t1_bb(c,i)
    contracted_intermediate =  0.500000000000000 * einsum('lkcj,ablk,ci->abij', g_bbbb[ob, ob, vb, ob], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<l,k||c,j>_abab*t1_bb(a,k)*t2_abab(c,b,l,i)
    contracted_intermediate =  1.000000000000000 * einsum('lkcj,ak,cbli->abij', g_abab[oa, ob, va, ob], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<l,k||c,j>_bbbb*t1_bb(a,k)*t2_bbbb(c,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lkcj,ak,cbil->abij', g_bbbb[ob, ob, vb, ob], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	  1.0000 P(a,b)<k,a||c,d>_abab*t2_bbbb(d,b,i,j)*t1_aa(c,k)
    contracted_intermediate =  1.000000000000000 * einsum('kacd,dbij,ck->abij', g_abab[oa, vb, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  1.0000 P(a,b)<k,a||c,d>_bbbb*t2_bbbb(d,b,i,j)*t1_bb(c,k)
    contracted_intermediate =  1.000000000000000 * einsum('kacd,dbij,ck->abij', g_bbbb[ob, vb, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<k,a||d,c>_abab*t2_abab(d,b,k,i)*t1_bb(c,j)
    contracted_intermediate = -1.000000000000000 * einsum('kadc,dbki,cj->abij', g_abab[oa, vb, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<k,a||c,d>_bbbb*t2_bbbb(d,b,i,k)*t1_bb(c,j)
    contracted_intermediate = -1.000000000000000 * einsum('kacd,dbik,cj->abij', g_bbbb[ob, vb, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	  0.5000 P(a,b)<k,a||c,d>_bbbb*t1_bb(b,k)*t2_bbbb(c,d,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('kacd,bk,cdij->abij', g_bbbb[ob, vb, vb, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  1.0000 <l,k||c,d>_aaaa*t3_abbabb(d,a,b,l,j,i)*t1_aa(c,k)
    doubles_res +=  1.000000000000000 * einsum('lkcd,dablji,ck->abij', g_aaaa[oa, oa, va, va], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,k||d,c>_abab*t3_abbabb(d,a,b,l,j,i)*t1_bb(c,k)
    doubles_res += -1.000000000000000 * einsum('lkdc,dablji,ck->abij', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <k,l||c,d>_abab*t3_bbbbbb(d,a,b,i,j,l)*t1_aa(c,k)
    doubles_res +=  1.000000000000000 * einsum('klcd,dabijl,ck->abij', g_abab[oa, ob, va, vb], t3_bbbbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,k||c,d>_bbbb*t3_bbbbbb(d,a,b,i,j,l)*t1_bb(c,k)
    doubles_res += -1.000000000000000 * einsum('lkcd,dabijl,ck->abij', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 P(i,j)<l,k||d,c>_abab*t3_abbabb(d,a,b,l,i,k)*t1_bb(c,j)
    contracted_intermediate = -0.500000000000000 * einsum('lkdc,dablik,cj->abij', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 P(i,j)<k,l||d,c>_abab*t3_abbabb(d,a,b,k,l,i)*t1_bb(c,j)
    contracted_intermediate =  0.500000000000000 * einsum('kldc,dabkli,cj->abij', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -0.5000 P(i,j)<l,k||c,d>_bbbb*t3_bbbbbb(d,a,b,i,l,k)*t1_bb(c,j)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,dabilk,cj->abij', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 P(a,b)<l,k||c,d>_abab*t1_bb(a,k)*t3_abbabb(c,d,b,l,j,i)
    contracted_intermediate =  0.500000000000000 * einsum('lkcd,ak,cdblji->abij', g_abab[oa, ob, va, vb], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  0.5000 P(a,b)<l,k||d,c>_abab*t1_bb(a,k)*t3_abbabb(d,c,b,l,j,i)
    contracted_intermediate =  0.500000000000000 * einsum('lkdc,ak,dcblji->abij', g_abab[oa, ob, va, vb], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -0.5000 P(a,b)<l,k||c,d>_bbbb*t1_bb(a,k)*t3_bbbbbb(c,d,b,i,j,l)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,ak,cdbijl->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -1.0000 <l,k||i,j>_bbbb*t1_bb(a,k)*t1_bb(b,l)
    doubles_res += -1.000000000000000 * einsum('lkij,ak,bl->abij', g_bbbb[ob, ob, ob, ob], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<k,a||c,j>_bbbb*t1_bb(b,k)*t1_bb(c,i)
    contracted_intermediate =  1.000000000000000 * einsum('kacj,bk,ci->abij', g_bbbb[ob, vb, vb, ob], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	 -1.0000 <a,b||c,d>_bbbb*t1_bb(c,j)*t1_bb(d,i)
    doubles_res += -1.000000000000000 * einsum('abcd,cj,di->abij', g_bbbb[vb, vb, vb, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 P(i,j)<k,l||c,d>_abab*t2_bbbb(a,b,i,l)*t2_abab(c,d,k,j)
    contracted_intermediate = -0.500000000000000 * einsum('klcd,abil,cdkj->abij', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -0.5000 P(i,j)<k,l||d,c>_abab*t2_bbbb(a,b,i,l)*t2_abab(d,c,k,j)
    contracted_intermediate = -0.500000000000000 * einsum('kldc,abil,dckj->abij', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -0.5000 P(i,j)<l,k||c,d>_bbbb*t2_bbbb(a,b,i,l)*t2_bbbb(c,d,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lkcd,abil,cdjk->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.2500 <l,k||c,d>_bbbb*t2_bbbb(a,b,l,k)*t2_bbbb(c,d,i,j)
    doubles_res +=  0.250000000000000 * einsum('lkcd,ablk,cdij->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,k||c,d>_abab*t2_abab(c,a,l,k)*t2_bbbb(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <k,l||c,d>_abab*t2_abab(c,a,k,l)*t2_bbbb(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('klcd,cakl,dbij->abij', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,k||c,d>_bbbb*t2_bbbb(c,a,l,k)*t2_bbbb(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)<l,k||c,d>_aaaa*t2_abab(c,a,k,j)*t2_abab(d,b,l,i)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,cakj,dbli->abij', g_aaaa[oa, oa, va, va], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  1.0000 P(i,j)<k,l||c,d>_abab*t2_abab(c,a,k,j)*t2_bbbb(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('klcd,cakj,dbil->abij', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  1.0000 P(i,j)<l,k||d,c>_abab*t2_bbbb(c,a,j,k)*t2_abab(d,b,l,i)
    contracted_intermediate =  1.000000000000000 * einsum('lkdc,cajk,dbli->abij', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  1.0000 P(i,j)<l,k||c,d>_bbbb*t2_bbbb(c,a,j,k)*t2_bbbb(d,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,cajk,dbil->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  0.5000 <l,k||d,c>_abab*t2_bbbb(c,a,i,j)*t2_abab(d,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkdc,caij,dblk->abij', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <k,l||d,c>_abab*t2_bbbb(c,a,i,j)*t2_abab(d,b,k,l)
    doubles_res +=  0.500000000000000 * einsum('kldc,caij,dbkl->abij', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,k||c,d>_bbbb*t2_bbbb(c,a,i,j)*t2_bbbb(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,caij,dblk->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)<k,l||c,d>_abab*t2_bbbb(a,b,i,l)*t1_aa(c,k)*t1_bb(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('klcd,abil,ck,dj->abij', g_abab[oa, ob, va, vb], t2_bbbb, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	  1.0000 P(i,j)<l,k||c,d>_bbbb*t2_bbbb(a,b,i,l)*t1_bb(c,k)*t1_bb(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,abil,ck,dj->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -1.0000 P(a,b)<k,l||c,d>_abab*t1_bb(a,l)*t2_bbbb(d,b,i,j)*t1_aa(c,k)
    contracted_intermediate = -1.000000000000000 * einsum('klcd,al,dbij,ck->abij', g_abab[oa, ob, va, vb], t1_bb, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  1.0000 P(a,b)<l,k||c,d>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,i,j)*t1_bb(c,k)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,al,dbij,ck->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	 -0.5000 <l,k||c,d>_bbbb*t2_bbbb(a,b,l,k)*t1_bb(c,j)*t1_bb(d,i)
    doubles_res += -0.500000000000000 * einsum('lkcd,ablk,cj,di->abij', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<l,k||d,c>_abab*t1_bb(a,k)*t2_abab(d,b,l,i)*t1_bb(c,j)
    contracted_intermediate =  1.000000000000000 * einsum('lkdc,ak,dbli,cj->abij', g_abab[oa, ob, va, vb], t1_bb, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<l,k||c,d>_bbbb*t1_bb(a,k)*t2_bbbb(d,b,i,l)*t1_bb(c,j)
    contracted_intermediate =  1.000000000000000 * einsum('lkcd,ak,dbil,cj->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate)

    #	 -0.5000 <l,k||c,d>_bbbb*t1_bb(a,k)*t1_bb(b,l)*t2_bbbb(c,d,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,ak,bl,cdij->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)<l,k||c,j>_bbbb*t1_bb(a,k)*t1_bb(b,l)*t1_bb(c,i)
    contracted_intermediate = -1.000000000000000 * einsum('lkcj,ak,bl,ci->abij', g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)

    #	 -1.0000 P(a,b)<k,a||c,d>_bbbb*t1_bb(b,k)*t1_bb(c,j)*t1_bb(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('kacd,bk,cj,di->abij', g_bbbb[ob, vb, vb, vb], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate)

    #	  1.0000 <l,k||c,d>_bbbb*t1_bb(a,k)*t1_bb(b,l)*t1_bb(c,j)*t1_bb(d,i)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ak,bl,cj,di->abij', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (2, 3), (0, 2), (0, 1)])

    return doubles_res


def ccsdt_t2_abab_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* b a e(-T) H e(T) | 0> :

    #	 -1.0000 f_bb(k,j)*t2_abab(a,b,i,k)
    doubles_res  = -1.000000000000000 * einsum('kj,abik->abij', f_bb[ob, ob], t2_abab)

    #	 -1.0000 f_aa(k,i)*t2_abab(a,b,k,j)
    doubles_res += -1.000000000000000 * einsum('ki,abkj->abij', f_aa[oa, oa], t2_abab)

    #	  1.0000 f_aa(a,c)*t2_abab(c,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('ac,cbij->abij', f_aa[va, va], t2_abab)

    #	  1.0000 f_bb(b,c)*t2_abab(a,c,i,j)
    doubles_res +=  1.000000000000000 * einsum('bc,acij->abij', f_bb[vb, vb], t2_abab)

    #	 -1.0000 f_aa(k,c)*t3_aabaab(c,a,b,i,k,j)
    doubles_res += -1.000000000000000 * einsum('kc,cabikj->abij', f_aa[oa, va], t3_aabaab)

    #	 -1.0000 f_bb(k,c)*t3_abbabb(a,c,b,i,j,k)
    doubles_res += -1.000000000000000 * einsum('kc,acbijk->abij', f_bb[ob, vb], t3_abbabb)

    #	 -1.0000 f_bb(k,c)*t2_abab(a,b,i,k)*t1_bb(c,j)
    doubles_res += -1.000000000000000 * einsum('kc,abik,cj->abij', f_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 f_aa(k,c)*t2_abab(a,b,k,j)*t1_aa(c,i)
    doubles_res += -1.000000000000000 * einsum('kc,abkj,ci->abij', f_aa[oa, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 f_aa(k,c)*t1_aa(a,k)*t2_abab(c,b,i,j)
    doubles_res += -1.000000000000000 * einsum('kc,ak,cbij->abij', f_aa[oa, va], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 f_bb(k,c)*t2_abab(a,c,i,j)*t1_bb(b,k)
    doubles_res += -1.000000000000000 * einsum('kc,acij,bk->abij', f_bb[ob, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <a,b||i,j>_abab
    doubles_res +=  1.000000000000000 * einsum('abij->abij', g_abab[va, vb, oa, ob])

    #	 -1.0000 <a,k||i,j>_abab*t1_bb(b,k)
    doubles_res += -1.000000000000000 * einsum('akij,bk->abij', g_abab[va, ob, oa, ob], t1_bb)

    #	 -1.0000 <k,b||i,j>_abab*t1_aa(a,k)
    doubles_res += -1.000000000000000 * einsum('kbij,ak->abij', g_abab[oa, vb, oa, ob], t1_aa)

    #	  1.0000 <a,b||c,j>_abab*t1_aa(c,i)
    doubles_res +=  1.000000000000000 * einsum('abcj,ci->abij', g_abab[va, vb, va, ob], t1_aa)

    #	  1.0000 <a,b||i,c>_abab*t1_bb(c,j)
    doubles_res +=  1.000000000000000 * einsum('abic,cj->abij', g_abab[va, vb, oa, vb], t1_bb)

    #	  0.5000 <l,k||i,j>_abab*t2_abab(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkij,ablk->abij', g_abab[oa, ob, oa, ob], t2_abab)

    #	  0.5000 <k,l||i,j>_abab*t2_abab(a,b,k,l)
    doubles_res +=  0.500000000000000 * einsum('klij,abkl->abij', g_abab[oa, ob, oa, ob], t2_abab)

    #	 -1.0000 <a,k||c,j>_abab*t2_abab(c,b,i,k)
    doubles_res += -1.000000000000000 * einsum('akcj,cbik->abij', g_abab[va, ob, va, ob], t2_abab)

    #	 -1.0000 <k,b||c,j>_abab*t2_aaaa(c,a,i,k)
    doubles_res += -1.000000000000000 * einsum('kbcj,caik->abij', g_abab[oa, vb, va, ob], t2_aaaa)

    #	  1.0000 <k,b||c,j>_bbbb*t2_abab(a,c,i,k)
    doubles_res +=  1.000000000000000 * einsum('kbcj,acik->abij', g_bbbb[ob, vb, vb, ob], t2_abab)

    #	  1.0000 <k,a||c,i>_aaaa*t2_abab(c,b,k,j)
    doubles_res +=  1.000000000000000 * einsum('kaci,cbkj->abij', g_aaaa[oa, va, va, oa], t2_abab)

    #	 -1.0000 <a,k||i,c>_abab*t2_bbbb(c,b,j,k)
    doubles_res += -1.000000000000000 * einsum('akic,cbjk->abij', g_abab[va, ob, oa, vb], t2_bbbb)

    #	 -1.0000 <k,b||i,c>_abab*t2_abab(a,c,k,j)
    doubles_res += -1.000000000000000 * einsum('kbic,ackj->abij', g_abab[oa, vb, oa, vb], t2_abab)

    #	  0.5000 <a,b||c,d>_abab*t2_abab(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('abcd,cdij->abij', g_abab[va, vb, va, vb], t2_abab)

    #	  0.5000 <a,b||d,c>_abab*t2_abab(d,c,i,j)
    doubles_res +=  0.500000000000000 * einsum('abdc,dcij->abij', g_abab[va, vb, va, vb], t2_abab)

    #	  0.5000 <l,k||c,j>_abab*t3_aabaab(c,a,b,i,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcj,cabilk->abij', g_abab[oa, ob, va, ob], t3_aabaab)

    #	  0.5000 <k,l||c,j>_abab*t3_aabaab(c,a,b,i,k,l)
    doubles_res +=  0.500000000000000 * einsum('klcj,cabikl->abij', g_abab[oa, ob, va, ob], t3_aabaab)

    #	 -0.5000 <l,k||c,j>_bbbb*t3_abbabb(a,c,b,i,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcj,acbilk->abij', g_bbbb[ob, ob, vb, ob], t3_abbabb)

    #	  0.5000 <l,k||c,i>_aaaa*t3_aabaab(c,a,b,k,l,j)
    doubles_res +=  0.500000000000000 * einsum('lkci,cabklj->abij', g_aaaa[oa, oa, va, oa], t3_aabaab)

    #	  0.5000 <l,k||i,c>_abab*t3_abbabb(a,c,b,l,j,k)
    doubles_res +=  0.500000000000000 * einsum('lkic,acbljk->abij', g_abab[oa, ob, oa, vb], t3_abbabb)

    #	 -0.5000 <k,l||i,c>_abab*t3_abbabb(a,c,b,k,l,j)
    doubles_res += -0.500000000000000 * einsum('klic,acbklj->abij', g_abab[oa, ob, oa, vb], t3_abbabb)

    #	 -0.5000 <k,a||c,d>_aaaa*t3_aabaab(c,d,b,i,k,j)
    doubles_res += -0.500000000000000 * einsum('kacd,cdbikj->abij', g_aaaa[oa, va, va, va], t3_aabaab)

    #	 -0.5000 <a,k||c,d>_abab*t3_abbabb(c,d,b,i,j,k)
    doubles_res += -0.500000000000000 * einsum('akcd,cdbijk->abij', g_abab[va, ob, va, vb], t3_abbabb)

    #	 -0.5000 <a,k||d,c>_abab*t3_abbabb(d,c,b,i,j,k)
    doubles_res += -0.500000000000000 * einsum('akdc,dcbijk->abij', g_abab[va, ob, va, vb], t3_abbabb)

    #	 -0.5000 <k,b||c,d>_abab*t3_aabaab(c,a,d,i,k,j)
    doubles_res += -0.500000000000000 * einsum('kbcd,cadikj->abij', g_abab[oa, vb, va, vb], t3_aabaab)

    #	  0.5000 <k,b||d,c>_abab*t3_aabaab(a,d,c,i,k,j)
    doubles_res +=  0.500000000000000 * einsum('kbdc,adcikj->abij', g_abab[oa, vb, va, vb], t3_aabaab)

    #	  0.5000 <k,b||c,d>_bbbb*t3_abbabb(a,d,c,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('kbcd,adcijk->abij', g_bbbb[ob, vb, vb, vb], t3_abbabb)

    #	 -1.0000 <k,l||c,j>_abab*t2_abab(a,b,i,l)*t1_aa(c,k)
    doubles_res += -1.000000000000000 * einsum('klcj,abil,ck->abij', g_abab[oa, ob, va, ob], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,k||c,j>_bbbb*t2_abab(a,b,i,l)*t1_bb(c,k)
    doubles_res +=  1.000000000000000 * einsum('lkcj,abil,ck->abij', g_bbbb[ob, ob, vb, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,k||c,i>_aaaa*t2_abab(a,b,l,j)*t1_aa(c,k)
    doubles_res +=  1.000000000000000 * einsum('lkci,ablj,ck->abij', g_aaaa[oa, oa, va, oa], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,k||i,c>_abab*t2_abab(a,b,l,j)*t1_bb(c,k)
    doubles_res += -1.000000000000000 * einsum('lkic,ablj,ck->abij', g_abab[oa, ob, oa, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,k||c,j>_abab*t2_abab(a,b,l,k)*t1_aa(c,i)
    doubles_res +=  0.500000000000000 * einsum('lkcj,ablk,ci->abij', g_abab[oa, ob, va, ob], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <k,l||c,j>_abab*t2_abab(a,b,k,l)*t1_aa(c,i)
    doubles_res +=  0.500000000000000 * einsum('klcj,abkl,ci->abij', g_abab[oa, ob, va, ob], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,k||i,c>_abab*t2_abab(a,b,l,k)*t1_bb(c,j)
    doubles_res +=  0.500000000000000 * einsum('lkic,ablk,cj->abij', g_abab[oa, ob, oa, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <k,l||i,c>_abab*t2_abab(a,b,k,l)*t1_bb(c,j)
    doubles_res +=  0.500000000000000 * einsum('klic,abkl,cj->abij', g_abab[oa, ob, oa, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <k,l||c,j>_abab*t1_aa(a,k)*t2_abab(c,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('klcj,ak,cbil->abij', g_abab[oa, ob, va, ob], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,k||c,j>_abab*t2_aaaa(c,a,i,l)*t1_bb(b,k)
    doubles_res +=  1.000000000000000 * einsum('lkcj,cail,bk->abij', g_abab[oa, ob, va, ob], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,k||c,j>_bbbb*t2_abab(a,c,i,l)*t1_bb(b,k)
    doubles_res += -1.000000000000000 * einsum('lkcj,acil,bk->abij', g_bbbb[ob, ob, vb, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,k||c,i>_aaaa*t1_aa(a,k)*t2_abab(c,b,l,j)
    doubles_res += -1.000000000000000 * einsum('lkci,ak,cblj->abij', g_aaaa[oa, oa, va, oa], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <k,l||i,c>_abab*t1_aa(a,k)*t2_bbbb(c,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('klic,ak,cbjl->abij', g_abab[oa, ob, oa, vb], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,k||i,c>_abab*t2_abab(a,c,l,j)*t1_bb(b,k)
    doubles_res +=  1.000000000000000 * einsum('lkic,aclj,bk->abij', g_abab[oa, ob, oa, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <k,a||c,d>_aaaa*t2_abab(d,b,i,j)*t1_aa(c,k)
    doubles_res +=  1.000000000000000 * einsum('kacd,dbij,ck->abij', g_aaaa[oa, va, va, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <a,k||d,c>_abab*t2_abab(d,b,i,j)*t1_bb(c,k)
    doubles_res +=  1.000000000000000 * einsum('akdc,dbij,ck->abij', g_abab[va, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <k,b||c,d>_abab*t2_abab(a,d,i,j)*t1_aa(c,k)
    doubles_res +=  1.000000000000000 * einsum('kbcd,adij,ck->abij', g_abab[oa, vb, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <k,b||c,d>_bbbb*t2_abab(a,d,i,j)*t1_bb(c,k)
    doubles_res +=  1.000000000000000 * einsum('kbcd,adij,ck->abij', g_bbbb[ob, vb, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <a,k||d,c>_abab*t2_abab(d,b,i,k)*t1_bb(c,j)
    doubles_res += -1.000000000000000 * einsum('akdc,dbik,cj->abij', g_abab[va, ob, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <k,b||d,c>_abab*t2_aaaa(d,a,i,k)*t1_bb(c,j)
    doubles_res += -1.000000000000000 * einsum('kbdc,daik,cj->abij', g_abab[oa, vb, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <k,b||c,d>_bbbb*t2_abab(a,d,i,k)*t1_bb(c,j)
    doubles_res += -1.000000000000000 * einsum('kbcd,adik,cj->abij', g_bbbb[ob, vb, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <k,a||c,d>_aaaa*t2_abab(d,b,k,j)*t1_aa(c,i)
    doubles_res += -1.000000000000000 * einsum('kacd,dbkj,ci->abij', g_aaaa[oa, va, va, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <a,k||c,d>_abab*t2_bbbb(d,b,j,k)*t1_aa(c,i)
    doubles_res += -1.000000000000000 * einsum('akcd,dbjk,ci->abij', g_abab[va, ob, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <k,b||c,d>_abab*t2_abab(a,d,k,j)*t1_aa(c,i)
    doubles_res += -1.000000000000000 * einsum('kbcd,adkj,ci->abij', g_abab[oa, vb, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <a,k||c,d>_abab*t1_bb(b,k)*t2_abab(c,d,i,j)
    doubles_res += -0.500000000000000 * einsum('akcd,bk,cdij->abij', g_abab[va, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <a,k||d,c>_abab*t1_bb(b,k)*t2_abab(d,c,i,j)
    doubles_res += -0.500000000000000 * einsum('akdc,bk,dcij->abij', g_abab[va, ob, va, vb], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <k,b||c,d>_abab*t1_aa(a,k)*t2_abab(c,d,i,j)
    doubles_res += -0.500000000000000 * einsum('kbcd,ak,cdij->abij', g_abab[oa, vb, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <k,b||d,c>_abab*t1_aa(a,k)*t2_abab(d,c,i,j)
    doubles_res += -0.500000000000000 * einsum('kbdc,ak,dcij->abij', g_abab[oa, vb, va, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,k||c,d>_aaaa*t3_aabaab(d,a,b,i,l,j)*t1_aa(c,k)
    doubles_res +=  1.000000000000000 * einsum('lkcd,dabilj,ck->abij', g_aaaa[oa, oa, va, va], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,k||d,c>_abab*t3_aabaab(d,a,b,i,l,j)*t1_bb(c,k)
    doubles_res += -1.000000000000000 * einsum('lkdc,dabilj,ck->abij', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <k,l||c,d>_abab*t3_abbabb(a,d,b,i,j,l)*t1_aa(c,k)
    doubles_res += -1.000000000000000 * einsum('klcd,adbijl,ck->abij', g_abab[oa, ob, va, vb], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,k||c,d>_bbbb*t3_abbabb(a,d,b,i,j,l)*t1_bb(c,k)
    doubles_res +=  1.000000000000000 * einsum('lkcd,adbijl,ck->abij', g_bbbb[ob, ob, vb, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,k||d,c>_abab*t3_aabaab(d,a,b,i,l,k)*t1_bb(c,j)
    doubles_res +=  0.500000000000000 * einsum('lkdc,dabilk,cj->abij', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <k,l||d,c>_abab*t3_aabaab(d,a,b,i,k,l)*t1_bb(c,j)
    doubles_res +=  0.500000000000000 * einsum('kldc,dabikl,cj->abij', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,k||c,d>_bbbb*t3_abbabb(a,d,b,i,l,k)*t1_bb(c,j)
    doubles_res +=  0.500000000000000 * einsum('lkcd,adbilk,cj->abij', g_bbbb[ob, ob, vb, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,k||c,d>_aaaa*t3_aabaab(d,a,b,k,l,j)*t1_aa(c,i)
    doubles_res += -0.500000000000000 * einsum('lkcd,dabklj,ci->abij', g_aaaa[oa, oa, va, va], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,k||c,d>_abab*t3_abbabb(a,d,b,l,j,k)*t1_aa(c,i)
    doubles_res +=  0.500000000000000 * einsum('lkcd,adbljk,ci->abij', g_abab[oa, ob, va, vb], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <k,l||c,d>_abab*t3_abbabb(a,d,b,k,l,j)*t1_aa(c,i)
    doubles_res += -0.500000000000000 * einsum('klcd,adbklj,ci->abij', g_abab[oa, ob, va, vb], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,k||c,d>_aaaa*t1_aa(a,k)*t3_aabaab(c,d,b,i,l,j)
    doubles_res +=  0.500000000000000 * einsum('lkcd,ak,cdbilj->abij', g_aaaa[oa, oa, va, va], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <k,l||c,d>_abab*t1_aa(a,k)*t3_abbabb(c,d,b,i,j,l)
    doubles_res +=  0.500000000000000 * einsum('klcd,ak,cdbijl->abij', g_abab[oa, ob, va, vb], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <k,l||d,c>_abab*t1_aa(a,k)*t3_abbabb(d,c,b,i,j,l)
    doubles_res +=  0.500000000000000 * einsum('kldc,ak,dcbijl->abij', g_abab[oa, ob, va, vb], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,k||c,d>_abab*t3_aabaab(c,a,d,i,l,j)*t1_bb(b,k)
    doubles_res +=  0.500000000000000 * einsum('lkcd,cadilj,bk->abij', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,k||d,c>_abab*t3_aabaab(a,d,c,i,l,j)*t1_bb(b,k)
    doubles_res += -0.500000000000000 * einsum('lkdc,adcilj,bk->abij', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,k||c,d>_bbbb*t3_abbabb(a,d,c,i,j,l)*t1_bb(b,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,adcijl,bk->abij', g_bbbb[ob, ob, vb, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <k,l||i,j>_abab*t1_aa(a,k)*t1_bb(b,l)
    doubles_res +=  1.000000000000000 * einsum('klij,ak,bl->abij', g_abab[oa, ob, oa, ob], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <a,k||c,j>_abab*t1_bb(b,k)*t1_aa(c,i)
    doubles_res += -1.000000000000000 * einsum('akcj,bk,ci->abij', g_abab[va, ob, va, ob], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <k,b||c,j>_abab*t1_aa(a,k)*t1_aa(c,i)
    doubles_res += -1.000000000000000 * einsum('kbcj,ak,ci->abij', g_abab[oa, vb, va, ob], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <a,k||i,c>_abab*t1_bb(b,k)*t1_bb(c,j)
    doubles_res += -1.000000000000000 * einsum('akic,bk,cj->abij', g_abab[va, ob, oa, vb], t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <k,b||i,c>_abab*t1_aa(a,k)*t1_bb(c,j)
    doubles_res += -1.000000000000000 * einsum('kbic,ak,cj->abij', g_abab[oa, vb, oa, vb], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <a,b||d,c>_abab*t1_bb(c,j)*t1_aa(d,i)
    doubles_res +=  1.000000000000000 * einsum('abdc,cj,di->abij', g_abab[va, vb, va, vb], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <k,l||c,d>_abab*t2_abab(a,b,i,l)*t2_abab(c,d,k,j)
    doubles_res += -0.500000000000000 * einsum('klcd,abil,cdkj->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <k,l||d,c>_abab*t2_abab(a,b,i,l)*t2_abab(d,c,k,j)
    doubles_res += -0.500000000000000 * einsum('kldc,abil,dckj->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,k||c,d>_bbbb*t2_abab(a,b,i,l)*t2_bbbb(c,d,j,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,abil,cdjk->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,k||c,d>_aaaa*t2_abab(a,b,l,j)*t2_aaaa(c,d,i,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,ablj,cdik->abij', g_aaaa[oa, oa, va, va], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,k||c,d>_abab*t2_abab(a,b,l,j)*t2_abab(c,d,i,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,ablj,cdik->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,k||d,c>_abab*t2_abab(a,b,l,j)*t2_abab(d,c,i,k)
    doubles_res += -0.500000000000000 * einsum('lkdc,ablj,dcik->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 <l,k||c,d>_abab*t2_abab(a,b,l,k)*t2_abab(c,d,i,j)
    doubles_res +=  0.250000000000000 * einsum('lkcd,ablk,cdij->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.2500 <l,k||d,c>_abab*t2_abab(a,b,l,k)*t2_abab(d,c,i,j)
    doubles_res +=  0.250000000000000 * einsum('lkdc,ablk,dcij->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.2500 <k,l||c,d>_abab*t2_abab(a,b,k,l)*t2_abab(c,d,i,j)
    doubles_res +=  0.250000000000000 * einsum('klcd,abkl,cdij->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.2500 <k,l||d,c>_abab*t2_abab(a,b,k,l)*t2_abab(d,c,i,j)
    doubles_res +=  0.250000000000000 * einsum('kldc,abkl,dcij->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,k||c,d>_aaaa*t2_aaaa(c,a,l,k)*t2_abab(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,k||d,c>_abab*t2_abab(a,c,l,k)*t2_abab(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkdc,aclk,dbij->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <k,l||d,c>_abab*t2_abab(a,c,k,l)*t2_abab(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('kldc,ackl,dbij->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <k,l||d,c>_abab*t2_abab(a,c,k,j)*t2_abab(d,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('kldc,ackj,dbil->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,k||c,d>_aaaa*t2_aaaa(c,a,i,k)*t2_abab(d,b,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkcd,caik,dblj->abij', g_aaaa[oa, oa, va, va], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <k,l||c,d>_abab*t2_aaaa(c,a,i,k)*t2_bbbb(d,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('klcd,caik,dbjl->abij', g_abab[oa, ob, va, vb], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,k||d,c>_abab*t2_abab(a,c,i,k)*t2_abab(d,b,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkdc,acik,dblj->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,k||c,d>_bbbb*t2_abab(a,c,i,k)*t2_bbbb(d,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,acik,dbjl->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,k||d,c>_abab*t2_abab(a,c,i,j)*t2_abab(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkdc,acij,dblk->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <k,l||d,c>_abab*t2_abab(a,c,i,j)*t2_abab(d,b,k,l)
    doubles_res += -0.500000000000000 * einsum('kldc,acij,dbkl->abij', g_abab[oa, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,k||c,d>_bbbb*t2_abab(a,c,i,j)*t2_bbbb(d,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcd,acij,dblk->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <k,l||c,d>_abab*t2_abab(a,b,i,l)*t1_aa(c,k)*t1_bb(d,j)
    doubles_res += -1.000000000000000 * einsum('klcd,abil,ck,dj->abij', g_abab[oa, ob, va, vb], t2_abab, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	  1.0000 <l,k||c,d>_bbbb*t2_abab(a,b,i,l)*t1_bb(c,k)*t1_bb(d,j)
    doubles_res +=  1.000000000000000 * einsum('lkcd,abil,ck,dj->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	  1.0000 <l,k||c,d>_aaaa*t2_abab(a,b,l,j)*t1_aa(c,k)*t1_aa(d,i)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ablj,ck,di->abij', g_aaaa[oa, oa, va, va], t2_abab, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 <l,k||d,c>_abab*t2_abab(a,b,l,j)*t1_bb(c,k)*t1_aa(d,i)
    doubles_res += -1.000000000000000 * einsum('lkdc,ablj,ck,di->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 <l,k||c,d>_aaaa*t1_aa(a,l)*t2_abab(d,b,i,j)*t1_aa(c,k)
    doubles_res +=  1.000000000000000 * einsum('lkcd,al,dbij,ck->abij', g_aaaa[oa, oa, va, va], t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <l,k||d,c>_abab*t1_aa(a,l)*t2_abab(d,b,i,j)*t1_bb(c,k)
    doubles_res += -1.000000000000000 * einsum('lkdc,al,dbij,ck->abij', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <k,l||c,d>_abab*t2_abab(a,d,i,j)*t1_bb(b,l)*t1_aa(c,k)
    doubles_res += -1.000000000000000 * einsum('klcd,adij,bl,ck->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <l,k||c,d>_bbbb*t2_abab(a,d,i,j)*t1_bb(b,l)*t1_bb(c,k)
    doubles_res +=  1.000000000000000 * einsum('lkcd,adij,bl,ck->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  0.5000 <l,k||d,c>_abab*t2_abab(a,b,l,k)*t1_bb(c,j)*t1_aa(d,i)
    doubles_res +=  0.500000000000000 * einsum('lkdc,ablk,cj,di->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  0.5000 <k,l||d,c>_abab*t2_abab(a,b,k,l)*t1_bb(c,j)*t1_aa(d,i)
    doubles_res +=  0.500000000000000 * einsum('kldc,abkl,cj,di->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 <k,l||d,c>_abab*t1_aa(a,k)*t2_abab(d,b,i,l)*t1_bb(c,j)
    doubles_res +=  1.000000000000000 * einsum('kldc,ak,dbil,cj->abij', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	  1.0000 <l,k||d,c>_abab*t2_aaaa(d,a,i,l)*t1_bb(b,k)*t1_bb(c,j)
    doubles_res +=  1.000000000000000 * einsum('lkdc,dail,bk,cj->abij', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <l,k||c,d>_bbbb*t2_abab(a,d,i,l)*t1_bb(b,k)*t1_bb(c,j)
    doubles_res +=  1.000000000000000 * einsum('lkcd,adil,bk,cj->abij', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <l,k||c,d>_aaaa*t1_aa(a,k)*t2_abab(d,b,l,j)*t1_aa(c,i)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ak,dblj,ci->abij', g_aaaa[oa, oa, va, va], t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <k,l||c,d>_abab*t1_aa(a,k)*t2_bbbb(d,b,j,l)*t1_aa(c,i)
    doubles_res +=  1.000000000000000 * einsum('klcd,ak,dbjl,ci->abij', g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <l,k||c,d>_abab*t2_abab(a,d,l,j)*t1_bb(b,k)*t1_aa(c,i)
    doubles_res +=  1.000000000000000 * einsum('lkcd,adlj,bk,ci->abij', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  0.5000 <k,l||c,d>_abab*t1_aa(a,k)*t1_bb(b,l)*t2_abab(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('klcd,ak,bl,cdij->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	  0.5000 <k,l||d,c>_abab*t1_aa(a,k)*t1_bb(b,l)*t2_abab(d,c,i,j)
    doubles_res +=  0.500000000000000 * einsum('kldc,ak,bl,dcij->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	  1.0000 <k,l||c,j>_abab*t1_aa(a,k)*t1_bb(b,l)*t1_aa(c,i)
    doubles_res +=  1.000000000000000 * einsum('klcj,ak,bl,ci->abij', g_abab[oa, ob, va, ob], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <k,l||i,c>_abab*t1_aa(a,k)*t1_bb(b,l)*t1_bb(c,j)
    doubles_res +=  1.000000000000000 * einsum('klic,ak,bl,cj->abij', g_abab[oa, ob, oa, vb], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -1.0000 <a,k||d,c>_abab*t1_bb(b,k)*t1_bb(c,j)*t1_aa(d,i)
    doubles_res += -1.000000000000000 * einsum('akdc,bk,cj,di->abij', g_abab[va, ob, va, vb], t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 <k,b||d,c>_abab*t1_aa(a,k)*t1_bb(c,j)*t1_aa(d,i)
    doubles_res += -1.000000000000000 * einsum('kbdc,ak,cj,di->abij', g_abab[oa, vb, va, vb], t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <k,l||d,c>_abab*t1_aa(a,k)*t1_bb(b,l)*t1_bb(c,j)*t1_aa(d,i)
    doubles_res +=  1.000000000000000 * einsum('kldc,ak,bl,cj,di->abij', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (0, 2), (0, 1)])

    return doubles_res


def ccsdt_t3_aaaaaa_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :

    #	 -1.0000 P(j,k)f_aa(l,k)*t3_aaaaaa(a,b,c,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('lk,abcijl->abcijk', f_aa[oa, oa], t3_aaaaaa)
    triples_res  =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 f_aa(l,i)*t3_aaaaaa(a,b,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('li,abcjkl->abcijk', f_aa[oa, oa], t3_aaaaaa)

    #	  1.0000 P(a,b)f_aa(a,d)*t3_aaaaaa(d,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('ad,dbcijk->abcijk', f_aa[va, va], t3_aaaaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 f_aa(c,d)*t3_aaaaaa(d,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('cd,dabijk->abcijk', f_aa[va, va], t3_aaaaaa)

    #	 -1.0000 P(j,k)f_aa(l,d)*t3_aaaaaa(a,b,c,i,j,l)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('ld,abcijl,dk->abcijk', f_aa[oa, va], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 f_aa(l,d)*t3_aaaaaa(a,b,c,j,k,l)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('ld,abcjkl,di->abcijk', f_aa[oa, va], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 P(a,b)f_aa(l,d)*t1_aa(a,l)*t3_aaaaaa(d,b,c,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('ld,al,dbcijk->abcijk', f_aa[oa, va], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 f_aa(l,d)*t3_aaaaaa(d,a,b,i,j,k)*t1_aa(c,l)
    triples_res += -1.000000000000000 * einsum('ld,dabijk,cl->abcijk', f_aa[oa, va], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)*P(a,b)f_aa(l,d)*t2_aaaa(d,a,j,k)*t2_aaaa(b,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('ld,dajk,bcil->abcijk', f_aa[oa, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(a,b)f_aa(l,d)*t2_aaaa(d,a,i,j)*t2_aaaa(b,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('ld,daij,bckl->abcijk', f_aa[oa, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(i,j)f_aa(l,d)*t2_aaaa(a,b,i,l)*t2_aaaa(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('ld,abil,dcjk->abcijk', f_aa[oa, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 f_aa(l,d)*t2_aaaa(a,b,k,l)*t2_aaaa(d,c,i,j)
    triples_res += -1.000000000000000 * einsum('ld,abkl,dcij->abcijk', f_aa[oa, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 P(i,j)*P(a,b)<l,a||j,k>_aaaa*t2_aaaa(b,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lajk,bcil->abcijk', g_aaaa[oa, va, oa, oa], t2_aaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,a||i,j>_aaaa*t2_aaaa(b,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('laij,bckl->abcijk', g_aaaa[oa, va, oa, oa], t2_aaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(i,j)<l,c||j,k>_aaaa*t2_aaaa(a,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcjk,abil->abcijk', g_aaaa[oa, va, oa, oa], t2_aaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 <l,c||i,j>_aaaa*t2_aaaa(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcij,abkl->abcijk', g_aaaa[oa, va, oa, oa], t2_aaaa)

    #	 -1.0000 P(j,k)*P(b,c)<a,b||d,k>_aaaa*t2_aaaa(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abdk,dcij->abcijk', g_aaaa[va, va, va, oa], t2_aaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	 -1.0000 P(b,c)<a,b||d,i>_aaaa*t2_aaaa(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abdi,dcjk->abcijk', g_aaaa[va, va, va, oa], t2_aaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -1.0000 P(j,k)<b,c||d,k>_aaaa*t2_aaaa(d,a,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('bcdk,daij->abcijk', g_aaaa[va, va, va, oa], t2_aaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <b,c||d,i>_aaaa*t2_aaaa(d,a,j,k)
    triples_res += -1.000000000000000 * einsum('bcdi,dajk->abcijk', g_aaaa[va, va, va, oa], t2_aaaa)

    #	  0.5000 P(i,j)<m,l||j,k>_aaaa*t3_aaaaaa(a,b,c,i,m,l)
    contracted_intermediate =  0.500000000000000 * einsum('mljk,abciml->abcijk', g_aaaa[oa, oa, oa, oa], t3_aaaaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 <m,l||i,j>_aaaa*t3_aaaaaa(a,b,c,k,m,l)
    triples_res +=  0.500000000000000 * einsum('mlij,abckml->abcijk', g_aaaa[oa, oa, oa, oa], t3_aaaaaa)

    #	  1.0000 P(j,k)*P(a,b)<l,a||d,k>_aaaa*t3_aaaaaa(d,b,c,i,j,l)
    contracted_intermediate =  1.000000000000000 * einsum('ladk,dbcijl->abcijk', g_aaaa[oa, va, va, oa], t3_aaaaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<a,l||k,d>_abab*t3_aabaab(c,b,d,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('alkd,cbdijl->abcijk', g_abab[va, ob, oa, vb], t3_aabaab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||d,i>_aaaa*t3_aaaaaa(d,b,c,j,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('ladi,dbcjkl->abcijk', g_aaaa[oa, va, va, oa], t3_aaaaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<a,l||i,d>_abab*t3_aabaab(c,b,d,j,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('alid,cbdjkl->abcijk', g_abab[va, ob, oa, vb], t3_aabaab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(j,k)<l,c||d,k>_aaaa*t3_aaaaaa(d,a,b,i,j,l)
    contracted_intermediate =  1.000000000000000 * einsum('lcdk,dabijl->abcijk', g_aaaa[oa, va, va, oa], t3_aaaaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<c,l||k,d>_abab*t3_aabaab(b,a,d,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('clkd,badijl->abcijk', g_abab[va, ob, oa, vb], t3_aabaab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <l,c||d,i>_aaaa*t3_aaaaaa(d,a,b,j,k,l)
    triples_res +=  1.000000000000000 * einsum('lcdi,dabjkl->abcijk', g_aaaa[oa, va, va, oa], t3_aaaaaa)

    #	 -1.0000 <c,l||i,d>_abab*t3_aabaab(b,a,d,j,k,l)
    triples_res += -1.000000000000000 * einsum('clid,badjkl->abcijk', g_abab[va, ob, oa, vb], t3_aabaab)

    #	  0.5000 P(b,c)<a,b||d,e>_aaaa*t3_aaaaaa(d,e,c,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('abde,decijk->abcijk', g_aaaa[va, va, va, va], t3_aaaaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.5000 <b,c||d,e>_aaaa*t3_aaaaaa(d,e,a,i,j,k)
    triples_res +=  0.500000000000000 * einsum('bcde,deaijk->abcijk', g_aaaa[va, va, va, va], t3_aaaaaa)

    #	  1.0000 P(i,j)*P(a,b)<m,l||j,k>_aaaa*t1_aa(a,l)*t2_aaaa(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mljk,al,bcim->abcijk', g_aaaa[oa, oa, oa, oa], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||j,k>_aaaa*t2_aaaa(a,b,i,m)*t1_aa(c,l)
    contracted_intermediate =  1.000000000000000 * einsum('mljk,abim,cl->abcijk', g_aaaa[oa, oa, oa, oa], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||i,j>_aaaa*t1_aa(a,l)*t2_aaaa(b,c,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlij,al,bckm->abcijk', g_aaaa[oa, oa, oa, oa], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <m,l||i,j>_aaaa*t2_aaaa(a,b,k,m)*t1_aa(c,l)
    triples_res +=  1.000000000000000 * einsum('mlij,abkm,cl->abcijk', g_aaaa[oa, oa, oa, oa], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)*P(a,b)<l,a||d,k>_aaaa*t2_aaaa(b,c,i,l)*t1_aa(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('ladk,bcil,dj->abcijk', g_aaaa[oa, va, va, oa], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(b,c)<l,a||d,k>_aaaa*t1_aa(b,l)*t2_aaaa(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('ladk,bl,dcij->abcijk', g_aaaa[oa, va, va, oa], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  1.0000 P(i,k)*P(a,b)<l,a||d,j>_aaaa*t2_aaaa(b,c,i,l)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('ladj,bcil,dk->abcijk', g_aaaa[oa, va, va, oa], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<l,a||d,i>_aaaa*t2_aaaa(b,c,j,l)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('ladi,bcjl,dk->abcijk', g_aaaa[oa, va, va, oa], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(b,c)<l,a||d,i>_aaaa*t1_aa(b,l)*t2_aaaa(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('ladi,bl,dcjk->abcijk', g_aaaa[oa, va, va, oa], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,c)<l,b||d,k>_aaaa*t1_aa(a,l)*t2_aaaa(d,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lbdk,al,dcij->abcijk', g_aaaa[oa, va, va, oa], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->cbaikj', contracted_intermediate)

    #	  1.0000 P(a,c)<l,b||d,i>_aaaa*t1_aa(a,l)*t2_aaaa(d,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('lbdi,al,dcjk->abcijk', g_aaaa[oa, va, va, oa], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate)

    #	 -1.0000 P(i,j)<l,c||d,k>_aaaa*t2_aaaa(a,b,i,l)*t1_aa(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('lcdk,abil,dj->abcijk', g_aaaa[oa, va, va, oa], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<l,c||d,k>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('lcdk,al,dbij->abcijk', g_aaaa[oa, va, va, oa], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(i,k)<l,c||d,j>_aaaa*t2_aaaa(a,b,i,l)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lcdj,abil,dk->abcijk', g_aaaa[oa, va, va, oa], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	 -1.0000 P(j,k)<l,c||d,i>_aaaa*t2_aaaa(a,b,j,l)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lcdi,abjl,dk->abcijk', g_aaaa[oa, va, va, oa], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,c||d,i>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('lcdi,al,dbjk->abcijk', g_aaaa[oa, va, va, oa], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(j,k)*P(b,c)<a,b||d,e>_aaaa*t2_aaaa(e,c,i,j)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('abde,ecij,dk->abcijk', g_aaaa[va, va, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  1.0000 P(b,c)<a,b||d,e>_aaaa*t2_aaaa(e,c,j,k)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('abde,ecjk,di->abcijk', g_aaaa[va, va, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  1.0000 P(j,k)<b,c||d,e>_aaaa*t2_aaaa(e,a,i,j)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('bcde,eaij,dk->abcijk', g_aaaa[va, va, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <b,c||d,e>_aaaa*t2_aaaa(e,a,j,k)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('bcde,eajk,di->abcijk', g_aaaa[va, va, va, va], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 P(j,k)<m,l||d,k>_aaaa*t3_aaaaaa(a,b,c,i,j,m)*t1_aa(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,abcijm,dl->abcijk', g_aaaa[oa, oa, va, oa], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||k,d>_abab*t3_aaaaaa(a,b,c,i,j,m)*t1_bb(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('mlkd,abcijm,dl->abcijk', g_abab[oa, ob, oa, vb], t3_aaaaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 P(i,j)<m,l||d,k>_aaaa*t3_aaaaaa(a,b,c,i,m,l)*t1_aa(d,j)
    contracted_intermediate =  0.500000000000000 * einsum('mldk,abciml,dj->abcijk', g_aaaa[oa, oa, va, oa], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<m,l||d,k>_aaaa*t1_aa(a,l)*t3_aaaaaa(d,b,c,i,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldk,al,dbcijm->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<l,m||k,d>_abab*t1_aa(a,l)*t3_aabaab(c,b,d,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmkd,al,cbdijm->abcijk', g_abab[oa, ob, oa, vb], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,k>_aaaa*t3_aaaaaa(d,a,b,i,j,m)*t1_aa(c,l)
    contracted_intermediate = -1.000000000000000 * einsum('mldk,dabijm,cl->abcijk', g_aaaa[oa, oa, va, oa], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,m||k,d>_abab*t3_aabaab(b,a,d,i,j,m)*t1_aa(c,l)
    contracted_intermediate =  1.000000000000000 * einsum('lmkd,badijm,cl->abcijk', g_abab[oa, ob, oa, vb], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 P(i,k)<m,l||d,j>_aaaa*t3_aaaaaa(a,b,c,i,m,l)*t1_aa(d,k)
    contracted_intermediate = -0.500000000000000 * einsum('mldj,abciml,dk->abcijk', g_aaaa[oa, oa, va, oa], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	  1.0000 <m,l||d,i>_aaaa*t3_aaaaaa(a,b,c,j,k,m)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('mldi,abcjkm,dl->abcijk', g_aaaa[oa, oa, va, oa], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <m,l||i,d>_abab*t3_aaaaaa(a,b,c,j,k,m)*t1_bb(d,l)
    triples_res += -1.000000000000000 * einsum('mlid,abcjkm,dl->abcijk', g_abab[oa, ob, oa, vb], t3_aaaaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 P(j,k)<m,l||d,i>_aaaa*t3_aaaaaa(a,b,c,j,m,l)*t1_aa(d,k)
    contracted_intermediate =  0.500000000000000 * einsum('mldi,abcjml,dk->abcijk', g_aaaa[oa, oa, va, oa], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,i>_aaaa*t1_aa(a,l)*t3_aaaaaa(d,b,c,j,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldi,al,dbcjkm->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,m||i,d>_abab*t1_aa(a,l)*t3_aabaab(c,b,d,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmid,al,cbdjkm->abcijk', g_abab[oa, ob, oa, vb], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 <m,l||d,i>_aaaa*t3_aaaaaa(d,a,b,j,k,m)*t1_aa(c,l)
    triples_res += -1.000000000000000 * einsum('mldi,dabjkm,cl->abcijk', g_aaaa[oa, oa, va, oa], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,m||i,d>_abab*t3_aabaab(b,a,d,j,k,m)*t1_aa(c,l)
    triples_res +=  1.000000000000000 * einsum('lmid,badjkm,cl->abcijk', g_abab[oa, ob, oa, vb], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 P(a,b)<l,a||d,e>_aaaa*t3_aaaaaa(e,b,c,i,j,k)*t1_aa(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('lade,ebcijk,dl->abcijk', g_aaaa[oa, va, va, va], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<a,l||e,d>_abab*t3_aaaaaa(e,b,c,i,j,k)*t1_bb(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('aled,ebcijk,dl->abcijk', g_abab[va, ob, va, vb], t3_aaaaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<l,a||d,e>_aaaa*t3_aaaaaa(e,b,c,i,j,l)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lade,ebcijl,dk->abcijk', g_aaaa[oa, va, va, va], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<a,l||d,e>_abab*t3_aabaab(c,b,e,i,j,l)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('alde,cbeijl,dk->abcijk', g_abab[va, ob, va, vb], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,a||d,e>_aaaa*t3_aaaaaa(e,b,c,j,k,l)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('lade,ebcjkl,di->abcijk', g_aaaa[oa, va, va, va], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<a,l||d,e>_abab*t3_aabaab(c,b,e,j,k,l)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('alde,cbejkl,di->abcijk', g_abab[va, ob, va, vb], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(b,c)<l,a||d,e>_aaaa*t1_aa(b,l)*t3_aaaaaa(d,e,c,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('lade,bl,decijk->abcijk', g_aaaa[oa, va, va, va], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -0.5000 P(a,c)<l,b||d,e>_aaaa*t1_aa(a,l)*t3_aaaaaa(d,e,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lbde,al,decijk->abcijk', g_aaaa[oa, va, va, va], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate)

    #	  1.0000 <l,c||d,e>_aaaa*t3_aaaaaa(e,a,b,i,j,k)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('lcde,eabijk,dl->abcijk', g_aaaa[oa, va, va, va], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <c,l||e,d>_abab*t3_aaaaaa(e,a,b,i,j,k)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('cled,eabijk,dl->abcijk', g_abab[va, ob, va, vb], t3_aaaaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)<l,c||d,e>_aaaa*t3_aaaaaa(e,a,b,i,j,l)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lcde,eabijl,dk->abcijk', g_aaaa[oa, va, va, va], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<c,l||d,e>_abab*t3_aabaab(b,a,e,i,j,l)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('clde,baeijl,dk->abcijk', g_abab[va, ob, va, vb], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <l,c||d,e>_aaaa*t3_aaaaaa(e,a,b,j,k,l)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('lcde,eabjkl,di->abcijk', g_aaaa[oa, va, va, va], t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <c,l||d,e>_abab*t3_aabaab(b,a,e,j,k,l)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('clde,baejkl,di->abcijk', g_abab[va, ob, va, vb], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 P(a,b)<l,c||d,e>_aaaa*t1_aa(a,l)*t3_aaaaaa(d,e,b,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('lcde,al,debijk->abcijk', g_aaaa[oa, va, va, va], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(j,k)<m,l||d,e>_aaaa*t3_aaaaaa(a,b,c,i,j,m)*t2_aaaa(d,e,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,abcijm,dekl->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 P(j,k)<m,l||d,e>_abab*t3_aaaaaa(a,b,c,i,j,m)*t2_abab(d,e,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,abcijm,dekl->abcijk', g_abab[oa, ob, va, vb], t3_aaaaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 P(j,k)<m,l||e,d>_abab*t3_aaaaaa(a,b,c,i,j,m)*t2_abab(e,d,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('mled,abcijm,edkl->abcijk', g_abab[oa, ob, va, vb], t3_aaaaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 <m,l||d,e>_aaaa*t3_aaaaaa(a,b,c,j,k,m)*t2_aaaa(d,e,i,l)
    triples_res += -0.500000000000000 * einsum('mlde,abcjkm,deil->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_abab*t3_aaaaaa(a,b,c,j,k,m)*t2_abab(d,e,i,l)
    triples_res += -0.500000000000000 * einsum('mlde,abcjkm,deil->abcijk', g_abab[oa, ob, va, vb], t3_aaaaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||e,d>_abab*t3_aaaaaa(a,b,c,j,k,m)*t2_abab(e,d,i,l)
    triples_res += -0.500000000000000 * einsum('mled,abcjkm,edil->abcijk', g_abab[oa, ob, va, vb], t3_aaaaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 P(i,j)<m,l||d,e>_aaaa*t3_aaaaaa(a,b,c,i,m,l)*t2_aaaa(d,e,j,k)
    contracted_intermediate =  0.250000000000000 * einsum('mlde,abciml,dejk->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.2500 <m,l||d,e>_aaaa*t3_aaaaaa(a,b,c,k,m,l)*t2_aaaa(d,e,i,j)
    triples_res +=  0.250000000000000 * einsum('mlde,abckml,deij->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 P(a,b)<m,l||d,e>_aaaa*t2_aaaa(d,a,m,l)*t3_aaaaaa(e,b,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,daml,ebcijk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||e,d>_abab*t2_abab(a,d,m,l)*t3_aaaaaa(e,b,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mled,adml,ebcijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<l,m||e,d>_abab*t2_abab(a,d,l,m)*t3_aaaaaa(e,b,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lmed,adlm,ebcijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||d,e>_aaaa*t2_aaaa(d,a,k,l)*t3_aaaaaa(e,b,c,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dakl,ebcijm->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<l,m||d,e>_abab*t2_aaaa(d,a,k,l)*t3_aabaab(c,b,e,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,dakl,cbeijm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||e,d>_abab*t2_abab(a,d,k,l)*t3_aaaaaa(e,b,c,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mled,adkl,ebcijm->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||d,e>_bbbb*t2_abab(a,d,k,l)*t3_aabaab(c,b,e,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,adkl,cbeijm->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_aaaa*t2_aaaa(d,a,i,l)*t3_aaaaaa(e,b,c,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dail,ebcjkm->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,m||d,e>_abab*t2_aaaa(d,a,i,l)*t3_aabaab(c,b,e,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,dail,cbejkm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||e,d>_abab*t2_abab(a,d,i,l)*t3_aaaaaa(e,b,c,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mled,adil,ebcjkm->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_bbbb*t2_abab(a,d,i,l)*t3_aabaab(c,b,e,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,adil,cbejkm->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(i,j)*P(a,b)<m,l||d,e>_aaaa*t2_aaaa(d,a,j,k)*t3_aaaaaa(e,b,c,i,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,dajk,ebciml->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  0.5000 P(i,j)*P(a,b)<m,l||d,e>_abab*t2_aaaa(d,a,j,k)*t3_aabaab(c,b,e,i,m,l)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,dajk,cbeiml->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  0.5000 P(i,j)*P(a,b)<l,m||d,e>_abab*t2_aaaa(d,a,j,k)*t3_aabaab(c,b,e,i,l,m)
    contracted_intermediate =  0.500000000000000 * einsum('lmde,dajk,cbeilm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||d,e>_aaaa*t2_aaaa(d,a,i,j)*t3_aaaaaa(e,b,c,k,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,daij,ebckml->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(a,b)<m,l||d,e>_abab*t2_aaaa(d,a,i,j)*t3_aabaab(c,b,e,k,m,l)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,daij,cbekml->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(a,b)<l,m||d,e>_abab*t2_aaaa(d,a,i,j)*t3_aabaab(c,b,e,k,l,m)
    contracted_intermediate =  0.500000000000000 * einsum('lmde,daij,cbeklm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 <m,l||d,e>_aaaa*t3_aaaaaa(e,a,b,i,j,k)*t2_aaaa(d,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,eabijk,dcml->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||e,d>_abab*t3_aaaaaa(e,a,b,i,j,k)*t2_abab(c,d,m,l)
    triples_res += -0.500000000000000 * einsum('mled,eabijk,cdml->abcijk', g_abab[oa, ob, va, vb], t3_aaaaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,m||e,d>_abab*t3_aaaaaa(e,a,b,i,j,k)*t2_abab(c,d,l,m)
    triples_res += -0.500000000000000 * einsum('lmed,eabijk,cdlm->abcijk', g_abab[oa, ob, va, vb], t3_aaaaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 P(j,k)<m,l||d,e>_aaaa*t3_aaaaaa(e,a,b,i,j,m)*t2_aaaa(d,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,eabijm,dckl->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||e,d>_abab*t3_aaaaaa(e,a,b,i,j,m)*t2_abab(c,d,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('mled,eabijm,cdkl->abcijk', g_abab[oa, ob, va, vb], t3_aaaaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,m||d,e>_abab*t3_aabaab(b,a,e,i,j,m)*t2_aaaa(d,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,baeijm,dckl->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,e>_bbbb*t3_aabaab(b,a,e,i,j,m)*t2_abab(c,d,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,baeijm,cdkl->abcijk', g_bbbb[ob, ob, vb, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <m,l||d,e>_aaaa*t3_aaaaaa(e,a,b,j,k,m)*t2_aaaa(d,c,i,l)
    triples_res +=  1.000000000000000 * einsum('mlde,eabjkm,dcil->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||e,d>_abab*t3_aaaaaa(e,a,b,j,k,m)*t2_abab(c,d,i,l)
    triples_res +=  1.000000000000000 * einsum('mled,eabjkm,cdil->abcijk', g_abab[oa, ob, va, vb], t3_aaaaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,m||d,e>_abab*t3_aabaab(b,a,e,j,k,m)*t2_aaaa(d,c,i,l)
    triples_res +=  1.000000000000000 * einsum('lmde,baejkm,dcil->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t3_aabaab(b,a,e,j,k,m)*t2_abab(c,d,i,l)
    triples_res +=  1.000000000000000 * einsum('mlde,baejkm,cdil->abcijk', g_bbbb[ob, ob, vb, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 P(i,j)<m,l||d,e>_aaaa*t3_aaaaaa(e,a,b,i,m,l)*t2_aaaa(d,c,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,eabiml,dcjk->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 P(i,j)<m,l||d,e>_abab*t3_aabaab(b,a,e,i,m,l)*t2_aaaa(d,c,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,baeiml,dcjk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 P(i,j)<l,m||d,e>_abab*t3_aabaab(b,a,e,i,l,m)*t2_aaaa(d,c,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('lmde,baeilm,dcjk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -0.5000 <m,l||d,e>_aaaa*t3_aaaaaa(e,a,b,k,m,l)*t2_aaaa(d,c,i,j)
    triples_res += -0.500000000000000 * einsum('mlde,eabkml,dcij->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <m,l||d,e>_abab*t3_aabaab(b,a,e,k,m,l)*t2_aaaa(d,c,i,j)
    triples_res +=  0.500000000000000 * einsum('mlde,baekml,dcij->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,m||d,e>_abab*t3_aabaab(b,a,e,k,l,m)*t2_aaaa(d,c,i,j)
    triples_res +=  0.500000000000000 * einsum('lmde,baeklm,dcij->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.2500 P(b,c)<m,l||d,e>_aaaa*t2_aaaa(a,b,m,l)*t3_aaaaaa(d,e,c,i,j,k)
    contracted_intermediate =  0.250000000000000 * einsum('mlde,abml,decijk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -0.5000 P(j,k)*P(b,c)<m,l||d,e>_aaaa*t2_aaaa(a,b,k,l)*t3_aaaaaa(d,e,c,i,j,m)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,abkl,decijm->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	 -0.5000 P(j,k)*P(b,c)<l,m||d,e>_abab*t2_aaaa(a,b,k,l)*t3_aabaab(d,c,e,i,j,m)
    contracted_intermediate = -0.500000000000000 * einsum('lmde,abkl,dceijm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  0.5000 P(j,k)*P(b,c)<l,m||e,d>_abab*t2_aaaa(a,b,k,l)*t3_aabaab(c,e,d,i,j,m)
    contracted_intermediate =  0.500000000000000 * einsum('lmed,abkl,cedijm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	 -0.5000 P(b,c)<m,l||d,e>_aaaa*t2_aaaa(a,b,i,l)*t3_aaaaaa(d,e,c,j,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,abil,decjkm->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -0.5000 P(b,c)<l,m||d,e>_abab*t2_aaaa(a,b,i,l)*t3_aabaab(d,c,e,j,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('lmde,abil,dcejkm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.5000 P(b,c)<l,m||e,d>_abab*t2_aaaa(a,b,i,l)*t3_aabaab(c,e,d,j,k,m)
    contracted_intermediate =  0.500000000000000 * einsum('lmed,abil,cedjkm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.2500 <m,l||d,e>_aaaa*t3_aaaaaa(d,e,a,i,j,k)*t2_aaaa(b,c,m,l)
    triples_res +=  0.250000000000000 * einsum('mlde,deaijk,bcml->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 P(j,k)<m,l||d,e>_aaaa*t3_aaaaaa(d,e,a,i,j,m)*t2_aaaa(b,c,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,deaijm,bckl->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 P(j,k)<l,m||d,e>_abab*t3_aabaab(d,a,e,i,j,m)*t2_aaaa(b,c,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('lmde,daeijm,bckl->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 P(j,k)<l,m||e,d>_abab*t3_aabaab(a,e,d,i,j,m)*t2_aaaa(b,c,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('lmed,aedijm,bckl->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 <m,l||d,e>_aaaa*t3_aaaaaa(d,e,a,j,k,m)*t2_aaaa(b,c,i,l)
    triples_res += -0.500000000000000 * einsum('mlde,deajkm,bcil->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,m||d,e>_abab*t3_aabaab(d,a,e,j,k,m)*t2_aaaa(b,c,i,l)
    triples_res += -0.500000000000000 * einsum('lmde,daejkm,bcil->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,m||e,d>_abab*t3_aabaab(a,e,d,j,k,m)*t2_aaaa(b,c,i,l)
    triples_res +=  0.500000000000000 * einsum('lmed,aedjkm,bcil->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<m,l||d,k>_aaaa*t2_aaaa(d,a,j,l)*t2_aaaa(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,dajl,bcim->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<m,l||k,d>_abab*t2_abab(a,d,j,l)*t2_aaaa(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlkd,adjl,bcim->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -0.5000 P(j,k)*P(a,b)<m,l||d,k>_aaaa*t2_aaaa(d,a,i,j)*t2_aaaa(b,c,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mldk,daij,bcml->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||d,k>_aaaa*t2_aaaa(a,b,i,m)*t2_aaaa(d,c,j,l)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,abim,dcjl->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||k,d>_abab*t2_aaaa(a,b,i,m)*t2_abab(c,d,j,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlkd,abim,cdjl->abcijk', g_abab[oa, ob, oa, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -0.5000 P(j,k)<m,l||d,k>_aaaa*t2_aaaa(a,b,m,l)*t2_aaaa(d,c,i,j)
    contracted_intermediate = -0.500000000000000 * einsum('mldk,abml,dcij->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(i,k)*P(a,b)<m,l||d,j>_aaaa*t2_aaaa(d,a,k,l)*t2_aaaa(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,dakl,bcim->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate)

    #	 -1.0000 P(i,k)*P(a,b)<m,l||j,d>_abab*t2_abab(a,d,k,l)*t2_aaaa(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mljd,adkl,bcim->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate)

    #	 -1.0000 P(i,k)<m,l||d,j>_aaaa*t2_aaaa(a,b,i,m)*t2_aaaa(d,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,abim,dckl->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	 -1.0000 P(i,k)<m,l||j,d>_abab*t2_aaaa(a,b,i,m)*t2_abab(c,d,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('mljd,abim,cdkl->abcijk', g_abab[oa, ob, oa, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||d,i>_aaaa*t2_aaaa(d,a,k,l)*t2_aaaa(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,dakl,bcjm->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||i,d>_abab*t2_abab(a,d,k,l)*t2_aaaa(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlid,adkl,bcjm->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||d,i>_aaaa*t2_aaaa(d,a,j,k)*t2_aaaa(b,c,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mldi,dajk,bcml->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,i>_aaaa*t2_aaaa(a,b,j,m)*t2_aaaa(d,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,abjm,dckl->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||i,d>_abab*t2_aaaa(a,b,j,m)*t2_abab(c,d,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlid,abjm,cdkl->abcijk', g_abab[oa, ob, oa, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 <m,l||d,i>_aaaa*t2_aaaa(a,b,m,l)*t2_aaaa(d,c,j,k)
    triples_res += -0.500000000000000 * einsum('mldi,abml,dcjk->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 P(i,j)*P(a,b)<l,a||d,e>_aaaa*t2_aaaa(b,c,i,l)*t2_aaaa(d,e,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lade,bcil,dejk->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -0.5000 P(a,b)<l,a||d,e>_aaaa*t2_aaaa(b,c,k,l)*t2_aaaa(d,e,i,j)
    contracted_intermediate = -0.500000000000000 * einsum('lade,bckl,deij->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<l,a||d,e>_aaaa*t2_aaaa(d,b,k,l)*t2_aaaa(e,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dbkl,ecij->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<a,l||e,d>_abab*t2_abab(b,d,k,l)*t2_aaaa(e,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('aled,bdkl,ecij->abcijk', g_abab[va, ob, va, vb], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||d,e>_aaaa*t2_aaaa(d,b,i,l)*t2_aaaa(e,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dbil,ecjk->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<a,l||e,d>_abab*t2_abab(b,d,i,l)*t2_aaaa(e,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('aled,bdil,ecjk->abcijk', g_abab[va, ob, va, vb], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<l,a||d,e>_aaaa*t2_aaaa(d,b,j,k)*t2_aaaa(e,c,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dbjk,ecil->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<a,l||d,e>_abab*t2_aaaa(d,b,j,k)*t2_abab(c,e,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('alde,dbjk,ceil->abcijk', g_abab[va, ob, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||d,e>_aaaa*t2_aaaa(d,b,i,j)*t2_aaaa(e,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dbij,eckl->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<a,l||d,e>_abab*t2_aaaa(d,b,i,j)*t2_abab(c,e,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('alde,dbij,cekl->abcijk', g_abab[va, ob, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(i,j)<l,c||d,e>_aaaa*t2_aaaa(a,b,i,l)*t2_aaaa(d,e,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lcde,abil,dejk->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -0.5000 <l,c||d,e>_aaaa*t2_aaaa(a,b,k,l)*t2_aaaa(d,e,i,j)
    triples_res += -0.500000000000000 * einsum('lcde,abkl,deij->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(j,k)<l,c||d,e>_aaaa*t2_aaaa(d,a,k,l)*t2_aaaa(e,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,dakl,ebij->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<c,l||e,d>_abab*t2_abab(a,d,k,l)*t2_aaaa(e,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('cled,adkl,ebij->abcijk', g_abab[va, ob, va, vb], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <l,c||d,e>_aaaa*t2_aaaa(d,a,i,l)*t2_aaaa(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dail,ebjk->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <c,l||e,d>_abab*t2_abab(a,d,i,l)*t2_aaaa(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('cled,adil,ebjk->abcijk', g_abab[va, ob, va, vb], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)<l,c||d,e>_aaaa*t2_aaaa(d,a,j,k)*t2_aaaa(e,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,dajk,ebil->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(i,j)<c,l||d,e>_abab*t2_aaaa(d,a,j,k)*t2_abab(b,e,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('clde,dajk,beil->abcijk', g_abab[va, ob, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 <l,c||d,e>_aaaa*t2_aaaa(d,a,i,j)*t2_aaaa(e,b,k,l)
    triples_res +=  1.000000000000000 * einsum('lcde,daij,ebkl->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <c,l||d,e>_abab*t2_aaaa(d,a,i,j)*t2_abab(b,e,k,l)
    triples_res +=  1.000000000000000 * einsum('clde,daij,bekl->abcijk', g_abab[va, ob, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<m,l||d,e>_aaaa*t2_aaaa(e,a,j,k)*t2_aaaa(b,c,i,m)*t1_aa(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,eajk,bcim,dl->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<m,l||e,d>_abab*t2_aaaa(e,a,j,k)*t2_aaaa(b,c,i,m)*t1_bb(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('mled,eajk,bcim,dl->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_aaaa, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_aaaa*t2_aaaa(e,a,i,j)*t2_aaaa(b,c,k,m)*t1_aa(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,eaij,bckm,dl->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||e,d>_abab*t2_aaaa(e,a,i,j)*t2_aaaa(b,c,k,m)*t1_bb(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('mled,eaij,bckm,dl->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_aaaa, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||d,e>_aaaa*t2_aaaa(a,b,i,m)*t2_aaaa(e,c,j,k)*t1_aa(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,abim,ecjk,dl->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(i,j)<m,l||e,d>_abab*t2_aaaa(a,b,i,m)*t2_aaaa(e,c,j,k)*t1_bb(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('mled,abim,ecjk,dl->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_aaaa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 <m,l||d,e>_aaaa*t2_aaaa(a,b,k,m)*t2_aaaa(e,c,i,j)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('mlde,abkm,ecij,dl->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t2_aaaa(a,b,k,m)*t2_aaaa(e,c,i,j)*t1_bb(d,l)
    triples_res += -1.000000000000000 * einsum('mled,abkm,ecij,dl->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_aaaa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,e>_aaaa*t2_aaaa(e,a,j,l)*t2_aaaa(b,c,i,m)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,eajl,bcim,dk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<m,l||d,e>_abab*t2_abab(a,e,j,l)*t2_aaaa(b,c,i,m)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,aejl,bcim,dk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  0.5000 P(j,k)*P(a,b)<m,l||d,e>_aaaa*t2_aaaa(e,a,i,j)*t2_aaaa(b,c,m,l)*t1_aa(d,k)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,eaij,bcml,dk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(i,j)<m,l||d,e>_aaaa*t2_aaaa(a,b,i,m)*t2_aaaa(e,c,j,l)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,abim,ecjl,dk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||d,e>_abab*t2_aaaa(a,b,i,m)*t2_abab(c,e,j,l)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,abim,cejl,dk->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 P(j,k)<m,l||d,e>_aaaa*t2_aaaa(a,b,m,l)*t2_aaaa(e,c,i,j)*t1_aa(d,k)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,abml,ecij,dk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(i,k)*P(a,b)<m,l||d,e>_aaaa*t2_aaaa(e,a,k,l)*t2_aaaa(b,c,i,m)*t1_aa(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,eakl,bcim,dj->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate)

    #	 -1.0000 P(i,k)*P(a,b)<m,l||d,e>_abab*t2_abab(a,e,k,l)*t2_aaaa(b,c,i,m)*t1_aa(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,aekl,bcim,dj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate)

    #	  1.0000 P(i,k)<m,l||d,e>_aaaa*t2_aaaa(a,b,i,m)*t2_aaaa(e,c,k,l)*t1_aa(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,abim,eckl,dj->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	 -1.0000 P(i,k)<m,l||d,e>_abab*t2_aaaa(a,b,i,m)*t2_abab(c,e,k,l)*t1_aa(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,abim,cekl,dj->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<m,l||d,e>_aaaa*t2_aaaa(e,a,k,l)*t2_aaaa(b,c,j,m)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,eakl,bcjm,di->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||d,e>_abab*t2_abab(a,e,k,l)*t2_aaaa(b,c,j,m)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,aekl,bcjm,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  0.5000 P(a,b)<m,l||d,e>_aaaa*t2_aaaa(e,a,j,k)*t2_aaaa(b,c,m,l)*t1_aa(d,i)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,eajk,bcml,di->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,e>_aaaa*t2_aaaa(a,b,j,m)*t2_aaaa(e,c,k,l)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,abjm,eckl,di->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,e>_abab*t2_aaaa(a,b,j,m)*t2_abab(c,e,k,l)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,abjm,cekl,di->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,b,m,l)*t2_aaaa(e,c,j,k)*t1_aa(d,i)
    triples_res +=  0.500000000000000 * einsum('mlde,abml,ecjk,di->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	  0.5000 P(i,j)*P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(b,c,i,m)*t2_aaaa(d,e,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,al,bcim,dejk->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  0.5000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(b,c,k,m)*t2_aaaa(d,e,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,al,bckm,deij->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,k,m)*t2_aaaa(e,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,dbkm,ecij->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,d,k,m)*t2_aaaa(e,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lmed,al,bdkm,ecij->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,m)*t2_aaaa(e,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,dbim,ecjk->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,d,i,m)*t2_aaaa(e,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmed,al,bdim,ecjk->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,j,k)*t2_aaaa(e,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,dbjk,ecim->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<l,m||d,e>_abab*t1_aa(a,l)*t2_aaaa(d,b,j,k)*t2_abab(c,e,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('lmde,al,dbjk,ceim->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,j)*t2_aaaa(e,c,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,dbij,eckm->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,m||d,e>_abab*t1_aa(a,l)*t2_aaaa(d,b,i,j)*t2_abab(c,e,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('lmde,al,dbij,cekm->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(i,j)<m,l||d,e>_aaaa*t2_aaaa(a,b,i,m)*t1_aa(c,l)*t2_aaaa(d,e,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,abim,cl,dejk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,b,k,m)*t1_aa(c,l)*t2_aaaa(d,e,i,j)
    triples_res +=  0.500000000000000 * einsum('mlde,abkm,cl,deij->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)<m,l||d,e>_aaaa*t2_aaaa(d,a,k,m)*t2_aaaa(e,b,i,j)*t1_aa(c,l)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,dakm,ebij,cl->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,m||e,d>_abab*t2_abab(a,d,k,m)*t2_aaaa(e,b,i,j)*t1_aa(c,l)
    contracted_intermediate =  1.000000000000000 * einsum('lmed,adkm,ebij,cl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,a,i,m)*t2_aaaa(e,b,j,k)*t1_aa(c,l)
    triples_res += -1.000000000000000 * einsum('mlde,daim,ebjk,cl->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <l,m||e,d>_abab*t2_abab(a,d,i,m)*t2_aaaa(e,b,j,k)*t1_aa(c,l)
    triples_res +=  1.000000000000000 * einsum('lmed,adim,ebjk,cl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)<m,l||d,e>_aaaa*t2_aaaa(d,a,j,k)*t2_aaaa(e,b,i,m)*t1_aa(c,l)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,dajk,ebim,cl->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(i,j)<l,m||d,e>_abab*t2_aaaa(d,a,j,k)*t2_abab(b,e,i,m)*t1_aa(c,l)
    contracted_intermediate = -1.000000000000000 * einsum('lmde,dajk,beim,cl->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,a,i,j)*t2_aaaa(e,b,k,m)*t1_aa(c,l)
    triples_res += -1.000000000000000 * einsum('mlde,daij,ebkm,cl->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	 -1.0000 <l,m||d,e>_abab*t2_aaaa(d,a,i,j)*t2_abab(b,e,k,m)*t1_aa(c,l)
    triples_res += -1.000000000000000 * einsum('lmde,daij,bekm,cl->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<m,l||d,k>_aaaa*t1_aa(a,l)*t2_aaaa(b,c,i,m)*t1_aa(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,al,bcim,dj->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||d,k>_aaaa*t2_aaaa(a,b,i,m)*t1_aa(c,l)*t1_aa(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,abim,cl,dj->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(j,k)*P(b,c)<m,l||d,k>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t2_aaaa(d,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,al,bm,dcij->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,k>_aaaa*t2_aaaa(d,a,i,j)*t1_aa(b,l)*t1_aa(c,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,daij,bl,cm->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(i,k)*P(a,b)<m,l||d,j>_aaaa*t1_aa(a,l)*t2_aaaa(b,c,i,m)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,al,bcim,dk->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate)

    #	 -1.0000 P(i,k)<m,l||d,j>_aaaa*t2_aaaa(a,b,i,m)*t1_aa(c,l)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,abim,cl,dk->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||d,i>_aaaa*t1_aa(a,l)*t2_aaaa(b,c,j,m)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,al,bcjm,dk->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,i>_aaaa*t2_aaaa(a,b,j,m)*t1_aa(c,l)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,abjm,cl,dk->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(b,c)<m,l||d,i>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t2_aaaa(d,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,al,bm,dcjk->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  1.0000 <m,l||d,i>_aaaa*t2_aaaa(d,a,j,k)*t1_aa(b,l)*t1_aa(c,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dajk,bl,cm->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<l,a||d,e>_aaaa*t2_aaaa(b,c,i,l)*t1_aa(d,k)*t1_aa(e,j)
    contracted_intermediate =  1.000000000000000 * einsum('lade,bcil,dk,ej->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(j,k)*P(b,c)<l,a||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(e,c,i,j)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lade,bl,ecij,dk->abcijk', g_aaaa[oa, va, va, va], t1_aa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||d,e>_aaaa*t2_aaaa(b,c,k,l)*t1_aa(d,j)*t1_aa(e,i)
    contracted_intermediate =  1.000000000000000 * einsum('lade,bckl,dj,ei->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(b,c)<l,a||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(e,c,j,k)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('lade,bl,ecjk,di->abcijk', g_aaaa[oa, va, va, va], t1_aa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,c)<l,b||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(e,c,i,j)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lbde,al,ecij,dk->abcijk', g_aaaa[oa, va, va, va], t1_aa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->cbaikj', contracted_intermediate)

    #	 -1.0000 P(a,c)<l,b||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(e,c,j,k)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('lbde,al,ecjk,di->abcijk', g_aaaa[oa, va, va, va], t1_aa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate)

    #	  1.0000 P(i,j)<l,c||d,e>_aaaa*t2_aaaa(a,b,i,l)*t1_aa(d,k)*t1_aa(e,j)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,abil,dk,ej->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<l,c||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(e,b,i,j)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,al,ebij,dk->abcijk', g_aaaa[oa, va, va, va], t1_aa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 <l,c||d,e>_aaaa*t2_aaaa(a,b,k,l)*t1_aa(d,j)*t1_aa(e,i)
    triples_res +=  1.000000000000000 * einsum('lcde,abkl,dj,ei->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 P(a,b)<l,c||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(e,b,j,k)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,al,ebjk,di->abcijk', g_aaaa[oa, va, va, va], t1_aa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,e>_aaaa*t3_aaaaaa(a,b,c,i,j,m)*t1_aa(d,l)*t1_aa(e,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,abcijm,dl,ek->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||e,d>_abab*t3_aaaaaa(a,b,c,i,j,m)*t1_bb(d,l)*t1_aa(e,k)
    contracted_intermediate = -1.000000000000000 * einsum('mled,abcijm,dl,ek->abcijk', g_abab[oa, ob, va, vb], t3_aaaaaa, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <m,l||d,e>_aaaa*t3_aaaaaa(a,b,c,j,k,m)*t1_aa(d,l)*t1_aa(e,i)
    triples_res +=  1.000000000000000 * einsum('mlde,abcjkm,dl,ei->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t3_aaaaaa(a,b,c,j,k,m)*t1_bb(d,l)*t1_aa(e,i)
    triples_res += -1.000000000000000 * einsum('mled,abcjkm,dl,ei->abcijk', g_abab[oa, ob, va, vb], t3_aaaaaa, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,m)*t3_aaaaaa(e,b,c,i,j,k)*t1_aa(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,am,ebcijk,dl->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||e,d>_abab*t1_aa(a,m)*t3_aaaaaa(e,b,c,i,j,k)*t1_bb(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('mled,am,ebcijk,dl->abcijk', g_abab[oa, ob, va, vb], t1_aa, t3_aaaaaa, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <m,l||d,e>_aaaa*t3_aaaaaa(e,a,b,i,j,k)*t1_aa(c,m)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('mlde,eabijk,cm,dl->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t3_aaaaaa(e,a,b,i,j,k)*t1_aa(c,m)*t1_bb(d,l)
    triples_res += -1.000000000000000 * einsum('mled,eabijk,cm,dl->abcijk', g_abab[oa, ob, va, vb], t3_aaaaaa, t1_aa, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	 -0.5000 P(i,j)<m,l||d,e>_aaaa*t3_aaaaaa(a,b,c,i,m,l)*t1_aa(d,k)*t1_aa(e,j)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,abciml,dk,ej->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t3_aaaaaa(e,b,c,i,j,m)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,al,ebcijm,dk->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<l,m||d,e>_abab*t1_aa(a,l)*t3_aabaab(c,b,e,i,j,m)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,al,cbeijm,dk->abcijk', g_abab[oa, ob, va, vb], t1_aa, t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,e>_aaaa*t3_aaaaaa(e,a,b,i,j,m)*t1_aa(c,l)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,eabijm,cl,dk->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,m||d,e>_abab*t3_aabaab(b,a,e,i,j,m)*t1_aa(c,l)*t1_aa(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,baeijm,cl,dk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 <m,l||d,e>_aaaa*t3_aaaaaa(a,b,c,k,m,l)*t1_aa(d,j)*t1_aa(e,i)
    triples_res += -0.500000000000000 * einsum('mlde,abckml,dj,ei->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t3_aaaaaa(e,b,c,j,k,m)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,al,ebcjkm,di->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t3_aaaaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,m||d,e>_abab*t1_aa(a,l)*t3_aabaab(c,b,e,j,k,m)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,al,cbejkm,di->abcijk', g_abab[oa, ob, va, vb], t1_aa, t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <m,l||d,e>_aaaa*t3_aaaaaa(e,a,b,j,k,m)*t1_aa(c,l)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('mlde,eabjkm,cl,di->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 <l,m||d,e>_abab*t3_aabaab(b,a,e,j,k,m)*t1_aa(c,l)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('lmde,baejkm,cl,di->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -0.5000 P(b,c)<m,l||d,e>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t3_aaaaaa(d,e,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,al,bm,decijk->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -0.5000 <m,l||d,e>_aaaa*t3_aaaaaa(d,e,a,i,j,k)*t1_aa(b,l)*t1_aa(c,m)
    triples_res += -0.500000000000000 * einsum('mlde,deaijk,bl,cm->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(b,c,i,m)*t1_aa(d,k)*t1_aa(e,j)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,bcim,dk,ej->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(i,j)<m,l||d,e>_aaaa*t2_aaaa(a,b,i,m)*t1_aa(c,l)*t1_aa(d,k)*t1_aa(e,j)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,abim,cl,dk,ej->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(b,c)<m,l||d,e>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t2_aaaa(e,c,i,j)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,bm,ecij,dk->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,e>_aaaa*t2_aaaa(e,a,i,j)*t1_aa(b,l)*t1_aa(c,m)*t1_aa(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,eaij,bl,cm,dk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(b,c,k,m)*t1_aa(d,j)*t1_aa(e,i)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,bckm,dj,ei->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 <m,l||d,e>_aaaa*t2_aaaa(a,b,k,m)*t1_aa(c,l)*t1_aa(d,j)*t1_aa(e,i)
    triples_res += -1.000000000000000 * einsum('mlde,abkm,cl,dj,ei->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 P(b,c)<m,l||d,e>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t2_aaaa(e,c,j,k)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,bm,ecjk,di->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -1.0000 <m,l||d,e>_aaaa*t2_aaaa(e,a,j,k)*t1_aa(b,l)*t1_aa(c,m)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('mlde,eajk,bl,cm,di->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (1, 3), (1, 2), (0, 1)])

    return triples_res


def ccsdt_t3_aabaab_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :

    #	 -1.0000 f_bb(l,k)*t3_aabaab(a,b,c,i,j,l)
    triples_res  = -1.000000000000000 * einsum('lk,abcijl->abcijk', f_bb[ob, ob], t3_aabaab)

    #	 -1.0000 f_aa(l,j)*t3_aabaab(a,b,c,i,l,k)
    triples_res += -1.000000000000000 * einsum('lj,abcilk->abcijk', f_aa[oa, oa], t3_aabaab)

    #	  1.0000 f_aa(l,i)*t3_aabaab(a,b,c,j,l,k)
    triples_res +=  1.000000000000000 * einsum('li,abcjlk->abcijk', f_aa[oa, oa], t3_aabaab)

    #	  1.0000 P(a,b)f_aa(a,d)*t3_aabaab(d,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('ad,dbcijk->abcijk', f_aa[va, va], t3_aabaab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 f_bb(c,d)*t3_aabaab(b,a,d,i,j,k)
    triples_res += -1.000000000000000 * einsum('cd,badijk->abcijk', f_bb[vb, vb], t3_aabaab)

    #	 -1.0000 f_bb(l,d)*t3_aabaab(a,b,c,i,j,l)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('ld,abcijl,dk->abcijk', f_bb[ob, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 f_aa(l,d)*t3_aabaab(a,b,c,i,l,k)*t1_aa(d,j)
    triples_res += -1.000000000000000 * einsum('ld,abcilk,dj->abcijk', f_aa[oa, va], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 f_aa(l,d)*t3_aabaab(a,b,c,j,l,k)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('ld,abcjlk,di->abcijk', f_aa[oa, va], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 P(a,b)f_aa(l,d)*t1_aa(a,l)*t3_aabaab(d,b,c,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('ld,al,dbcijk->abcijk', f_aa[oa, va], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 f_bb(l,d)*t3_aabaab(b,a,d,i,j,k)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('ld,badijk,cl->abcijk', f_bb[ob, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)f_bb(l,d)*t2_abab(a,d,j,k)*t2_abab(b,c,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('ld,adjk,bcil->abcijk', f_bb[ob, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(a,b)f_aa(l,d)*t2_aaaa(d,a,i,j)*t2_abab(b,c,l,k)
    contracted_intermediate =  1.000000000000000 * einsum('ld,daij,bclk->abcijk', f_aa[oa, va], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(i,j)f_aa(l,d)*t2_aaaa(a,b,i,l)*t2_abab(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('ld,abil,dcjk->abcijk', f_aa[oa, va], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<a,l||j,k>_abab*t2_abab(b,c,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('aljk,bcil->abcijk', g_abab[va, ob, oa, ob], t2_abab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||i,j>_aaaa*t2_abab(b,c,l,k)
    contracted_intermediate =  1.000000000000000 * einsum('laij,bclk->abcijk', g_aaaa[oa, va, oa, oa], t2_abab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(i,j)<l,c||j,k>_abab*t2_aaaa(a,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcjk,abil->abcijk', g_abab[oa, vb, oa, ob], t2_aaaa)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 <a,c||d,k>_abab*t2_aaaa(d,b,i,j)
    triples_res +=  1.000000000000000 * einsum('acdk,dbij->abcijk', g_abab[va, vb, va, ob], t2_aaaa)

    #	  1.0000 <a,b||d,j>_aaaa*t2_abab(d,c,i,k)
    triples_res +=  1.000000000000000 * einsum('abdj,dcik->abcijk', g_aaaa[va, va, va, oa], t2_abab)

    #	 -1.0000 <a,c||j,d>_abab*t2_abab(b,d,i,k)
    triples_res += -1.000000000000000 * einsum('acjd,bdik->abcijk', g_abab[va, vb, oa, vb], t2_abab)

    #	 -1.0000 <a,b||d,i>_aaaa*t2_abab(d,c,j,k)
    triples_res += -1.000000000000000 * einsum('abdi,dcjk->abcijk', g_aaaa[va, va, va, oa], t2_abab)

    #	  1.0000 <a,c||i,d>_abab*t2_abab(b,d,j,k)
    triples_res +=  1.000000000000000 * einsum('acid,bdjk->abcijk', g_abab[va, vb, oa, vb], t2_abab)

    #	 -1.0000 <b,c||d,k>_abab*t2_aaaa(d,a,i,j)
    triples_res += -1.000000000000000 * einsum('bcdk,daij->abcijk', g_abab[va, vb, va, ob], t2_aaaa)

    #	  1.0000 <b,c||j,d>_abab*t2_abab(a,d,i,k)
    triples_res +=  1.000000000000000 * einsum('bcjd,adik->abcijk', g_abab[va, vb, oa, vb], t2_abab)

    #	 -1.0000 <b,c||i,d>_abab*t2_abab(a,d,j,k)
    triples_res += -1.000000000000000 * einsum('bcid,adjk->abcijk', g_abab[va, vb, oa, vb], t2_abab)

    #	  0.5000 P(i,j)<m,l||j,k>_abab*t3_aabaab(a,b,c,i,m,l)
    contracted_intermediate =  0.500000000000000 * einsum('mljk,abciml->abcijk', g_abab[oa, ob, oa, ob], t3_aabaab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 P(i,j)<l,m||j,k>_abab*t3_aabaab(a,b,c,i,l,m)
    contracted_intermediate =  0.500000000000000 * einsum('lmjk,abcilm->abcijk', g_abab[oa, ob, oa, ob], t3_aabaab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -0.5000 <m,l||i,j>_aaaa*t3_aabaab(a,b,c,l,m,k)
    triples_res += -0.500000000000000 * einsum('mlij,abclmk->abcijk', g_aaaa[oa, oa, oa, oa], t3_aabaab)

    #	 -1.0000 P(a,b)<a,l||d,k>_abab*t3_aabaab(d,b,c,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('aldk,dbcijl->abcijk', g_abab[va, ob, va, ob], t3_aabaab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||d,j>_aaaa*t3_aabaab(d,b,c,i,l,k)
    contracted_intermediate =  1.000000000000000 * einsum('ladj,dbcilk->abcijk', g_aaaa[oa, va, va, oa], t3_aabaab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<a,l||j,d>_abab*t3_abbabb(b,d,c,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('aljd,bdcikl->abcijk', g_abab[va, ob, oa, vb], t3_abbabb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,a||d,i>_aaaa*t3_aabaab(d,b,c,j,l,k)
    contracted_intermediate = -1.000000000000000 * einsum('ladi,dbcjlk->abcijk', g_aaaa[oa, va, va, oa], t3_aabaab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<a,l||i,d>_abab*t3_abbabb(b,d,c,j,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('alid,bdcjkl->abcijk', g_abab[va, ob, oa, vb], t3_abbabb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <l,c||d,k>_abab*t3_aaaaaa(d,a,b,i,j,l)
    triples_res +=  1.000000000000000 * einsum('lcdk,dabijl->abcijk', g_abab[oa, vb, va, ob], t3_aaaaaa)

    #	 -1.0000 <l,c||d,k>_bbbb*t3_aabaab(b,a,d,i,j,l)
    triples_res += -1.000000000000000 * einsum('lcdk,badijl->abcijk', g_bbbb[ob, vb, vb, ob], t3_aabaab)

    #	  1.0000 <l,c||j,d>_abab*t3_aabaab(b,a,d,i,l,k)
    triples_res +=  1.000000000000000 * einsum('lcjd,badilk->abcijk', g_abab[oa, vb, oa, vb], t3_aabaab)

    #	 -1.0000 <l,c||i,d>_abab*t3_aabaab(b,a,d,j,l,k)
    triples_res += -1.000000000000000 * einsum('lcid,badjlk->abcijk', g_abab[oa, vb, oa, vb], t3_aabaab)

    #	  0.5000 <a,b||d,e>_aaaa*t3_aabaab(d,e,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('abde,decijk->abcijk', g_aaaa[va, va, va, va], t3_aabaab)

    #	  0.5000 <a,c||d,e>_abab*t3_aabaab(d,b,e,i,j,k)
    triples_res +=  0.500000000000000 * einsum('acde,dbeijk->abcijk', g_abab[va, vb, va, vb], t3_aabaab)

    #	 -0.5000 <a,c||e,d>_abab*t3_aabaab(b,e,d,i,j,k)
    triples_res += -0.500000000000000 * einsum('aced,bedijk->abcijk', g_abab[va, vb, va, vb], t3_aabaab)

    #	 -0.5000 <b,c||d,e>_abab*t3_aabaab(d,a,e,i,j,k)
    triples_res += -0.500000000000000 * einsum('bcde,daeijk->abcijk', g_abab[va, vb, va, vb], t3_aabaab)

    #	  0.5000 <b,c||e,d>_abab*t3_aabaab(a,e,d,i,j,k)
    triples_res +=  0.500000000000000 * einsum('bced,aedijk->abcijk', g_abab[va, vb, va, vb], t3_aabaab)

    #	 -1.0000 P(i,j)*P(a,b)<l,m||j,k>_abab*t1_aa(a,l)*t2_abab(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('lmjk,al,bcim->abcijk', g_abab[oa, ob, oa, ob], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||j,k>_abab*t2_aaaa(a,b,i,m)*t1_bb(c,l)
    contracted_intermediate =  1.000000000000000 * einsum('mljk,abim,cl->abcijk', g_abab[oa, ob, oa, ob], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||i,j>_aaaa*t1_aa(a,l)*t2_abab(b,c,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlij,al,bcmk->abcijk', g_aaaa[oa, oa, oa, oa], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<a,l||d,k>_abab*t2_abab(b,c,i,l)*t1_aa(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('aldk,bcil,dj->abcijk', g_abab[va, ob, va, ob], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 <a,l||d,k>_abab*t2_aaaa(d,b,i,j)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('aldk,dbij,cl->abcijk', g_abab[va, ob, va, ob], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,a||d,j>_aaaa*t1_aa(b,l)*t2_abab(d,c,i,k)
    triples_res +=  1.000000000000000 * einsum('ladj,bl,dcik->abcijk', g_aaaa[oa, va, va, oa], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <a,l||j,d>_abab*t2_abab(b,d,i,k)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('aljd,bdik,cl->abcijk', g_abab[va, ob, oa, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(a,b)<a,l||j,d>_abab*t2_abab(b,c,i,l)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('aljd,bcil,dk->abcijk', g_abab[va, ob, oa, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||d,j>_aaaa*t2_abab(b,c,l,k)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('ladj,bclk,di->abcijk', g_aaaa[oa, va, va, oa], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<a,l||i,d>_abab*t2_abab(b,c,j,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('alid,bcjl,dk->abcijk', g_abab[va, ob, oa, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,a||d,i>_aaaa*t2_abab(b,c,l,k)*t1_aa(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('ladi,bclk,dj->abcijk', g_aaaa[oa, va, va, oa], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 <l,a||d,i>_aaaa*t1_aa(b,l)*t2_abab(d,c,j,k)
    triples_res += -1.000000000000000 * einsum('ladi,bl,dcjk->abcijk', g_aaaa[oa, va, va, oa], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <a,l||i,d>_abab*t2_abab(b,d,j,k)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('alid,bdjk,cl->abcijk', g_abab[va, ob, oa, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <b,l||d,k>_abab*t2_aaaa(d,a,i,j)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('bldk,daij,cl->abcijk', g_abab[va, ob, va, ob], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,b||d,j>_aaaa*t1_aa(a,l)*t2_abab(d,c,i,k)
    triples_res += -1.000000000000000 * einsum('lbdj,al,dcik->abcijk', g_aaaa[oa, va, va, oa], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <b,l||j,d>_abab*t2_abab(a,d,i,k)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('bljd,adik,cl->abcijk', g_abab[va, ob, oa, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,b||d,i>_aaaa*t1_aa(a,l)*t2_abab(d,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lbdi,al,dcjk->abcijk', g_aaaa[oa, va, va, oa], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <b,l||i,d>_abab*t2_abab(a,d,j,k)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('blid,adjk,cl->abcijk', g_abab[va, ob, oa, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 P(i,j)<l,c||d,k>_abab*t2_aaaa(a,b,i,l)*t1_aa(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('lcdk,abil,dj->abcijk', g_abab[oa, vb, va, ob], t2_aaaa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,c||d,k>_abab*t1_aa(a,l)*t2_aaaa(d,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('lcdk,al,dbij->abcijk', g_abab[oa, vb, va, ob], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,c||j,d>_abab*t1_aa(a,l)*t2_abab(b,d,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('lcjd,al,bdik->abcijk', g_abab[oa, vb, oa, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 <l,c||j,d>_abab*t2_aaaa(a,b,i,l)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('lcjd,abil,dk->abcijk', g_abab[oa, vb, oa, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,c||i,d>_abab*t2_aaaa(a,b,j,l)*t1_bb(d,k)
    triples_res +=  1.000000000000000 * einsum('lcid,abjl,dk->abcijk', g_abab[oa, vb, oa, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(a,b)<l,c||i,d>_abab*t1_aa(a,l)*t2_abab(b,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('lcid,al,bdjk->abcijk', g_abab[oa, vb, oa, vb], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <a,c||e,d>_abab*t2_aaaa(e,b,i,j)*t1_bb(d,k)
    triples_res +=  1.000000000000000 * einsum('aced,ebij,dk->abcijk', g_abab[va, vb, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <a,b||d,e>_aaaa*t2_abab(e,c,i,k)*t1_aa(d,j)
    triples_res += -1.000000000000000 * einsum('abde,ecik,dj->abcijk', g_aaaa[va, va, va, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <a,c||d,e>_abab*t2_abab(b,e,i,k)*t1_aa(d,j)
    triples_res += -1.000000000000000 * einsum('acde,beik,dj->abcijk', g_abab[va, vb, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <a,b||d,e>_aaaa*t2_abab(e,c,j,k)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('abde,ecjk,di->abcijk', g_aaaa[va, va, va, va], t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <a,c||d,e>_abab*t2_abab(b,e,j,k)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('acde,bejk,di->abcijk', g_abab[va, vb, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <b,c||e,d>_abab*t2_aaaa(e,a,i,j)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('bced,eaij,dk->abcijk', g_abab[va, vb, va, vb], t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <b,c||d,e>_abab*t2_abab(a,e,i,k)*t1_aa(d,j)
    triples_res +=  1.000000000000000 * einsum('bcde,aeik,dj->abcijk', g_abab[va, vb, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <b,c||d,e>_abab*t2_abab(a,e,j,k)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('bcde,aejk,di->abcijk', g_abab[va, vb, va, vb], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,m||d,k>_abab*t3_aabaab(a,b,c,i,j,m)*t1_aa(d,l)
    triples_res += -1.000000000000000 * einsum('lmdk,abcijm,dl->abcijk', g_abab[oa, ob, va, ob], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,k>_bbbb*t3_aabaab(a,b,c,i,j,m)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('mldk,abcijm,dl->abcijk', g_bbbb[ob, ob, vb, ob], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,j>_aaaa*t3_aabaab(a,b,c,i,m,k)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('mldj,abcimk,dl->abcijk', g_aaaa[oa, oa, va, oa], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <m,l||j,d>_abab*t3_aabaab(a,b,c,i,m,k)*t1_bb(d,l)
    triples_res += -1.000000000000000 * einsum('mljd,abcimk,dl->abcijk', g_abab[oa, ob, oa, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 P(i,j)<m,l||d,k>_abab*t3_aabaab(a,b,c,i,m,l)*t1_aa(d,j)
    contracted_intermediate =  0.500000000000000 * einsum('mldk,abciml,dj->abcijk', g_abab[oa, ob, va, ob], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 P(i,j)<l,m||d,k>_abab*t3_aabaab(a,b,c,i,l,m)*t1_aa(d,j)
    contracted_intermediate =  0.500000000000000 * einsum('lmdk,abcilm,dj->abcijk', g_abab[oa, ob, va, ob], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(a,b)<l,m||d,k>_abab*t1_aa(a,l)*t3_aabaab(d,b,c,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmdk,al,dbcijm->abcijk', g_abab[oa, ob, va, ob], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,j>_aaaa*t1_aa(a,l)*t3_aabaab(d,b,c,i,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,al,dbcimk->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,m||j,d>_abab*t1_aa(a,l)*t3_abbabb(b,d,c,i,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('lmjd,al,bdcikm->abcijk', g_abab[oa, ob, oa, vb], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 <m,l||d,k>_abab*t3_aaaaaa(d,a,b,i,j,m)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('mldk,dabijm,cl->abcijk', g_abab[oa, ob, va, ob], t3_aaaaaa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <m,l||d,k>_bbbb*t3_aabaab(b,a,d,i,j,m)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mldk,badijm,cl->abcijk', g_bbbb[ob, ob, vb, ob], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <m,l||j,d>_abab*t3_aabaab(b,a,d,i,m,k)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('mljd,badimk,cl->abcijk', g_abab[oa, ob, oa, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <m,l||j,d>_abab*t3_aabaab(a,b,c,i,m,l)*t1_bb(d,k)
    triples_res +=  0.500000000000000 * einsum('mljd,abciml,dk->abcijk', g_abab[oa, ob, oa, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,m||j,d>_abab*t3_aabaab(a,b,c,i,l,m)*t1_bb(d,k)
    triples_res +=  0.500000000000000 * einsum('lmjd,abcilm,dk->abcijk', g_abab[oa, ob, oa, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,j>_aaaa*t3_aabaab(a,b,c,l,m,k)*t1_aa(d,i)
    triples_res += -0.500000000000000 * einsum('mldj,abclmk,di->abcijk', g_aaaa[oa, oa, va, oa], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <m,l||d,i>_aaaa*t3_aabaab(a,b,c,j,m,k)*t1_aa(d,l)
    triples_res += -1.000000000000000 * einsum('mldi,abcjmk,dl->abcijk', g_aaaa[oa, oa, va, oa], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||i,d>_abab*t3_aabaab(a,b,c,j,m,k)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('mlid,abcjmk,dl->abcijk', g_abab[oa, ob, oa, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||i,d>_abab*t3_aabaab(a,b,c,j,m,l)*t1_bb(d,k)
    triples_res += -0.500000000000000 * einsum('mlid,abcjml,dk->abcijk', g_abab[oa, ob, oa, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,m||i,d>_abab*t3_aabaab(a,b,c,j,l,m)*t1_bb(d,k)
    triples_res += -0.500000000000000 * einsum('lmid,abcjlm,dk->abcijk', g_abab[oa, ob, oa, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,i>_aaaa*t3_aabaab(a,b,c,l,m,k)*t1_aa(d,j)
    triples_res +=  0.500000000000000 * einsum('mldi,abclmk,dj->abcijk', g_aaaa[oa, oa, va, oa], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(a,b)<m,l||d,i>_aaaa*t1_aa(a,l)*t3_aabaab(d,b,c,j,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,al,dbcjmk->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,m||i,d>_abab*t1_aa(a,l)*t3_abbabb(b,d,c,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmid,al,bdcjkm->abcijk', g_abab[oa, ob, oa, vb], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <m,l||i,d>_abab*t3_aabaab(b,a,d,j,m,k)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mlid,badjmk,cl->abcijk', g_abab[oa, ob, oa, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(a,b)<l,a||d,e>_aaaa*t3_aabaab(e,b,c,i,j,k)*t1_aa(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('lade,ebcijk,dl->abcijk', g_aaaa[oa, va, va, va], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<a,l||e,d>_abab*t3_aabaab(e,b,c,i,j,k)*t1_bb(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('aled,ebcijk,dl->abcijk', g_abab[va, ob, va, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<a,l||e,d>_abab*t3_aabaab(e,b,c,i,j,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('aled,ebcijl,dk->abcijk', g_abab[va, ob, va, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,a||d,e>_aaaa*t3_aabaab(e,b,c,i,l,k)*t1_aa(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('lade,ebcilk,dj->abcijk', g_aaaa[oa, va, va, va], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<a,l||d,e>_abab*t3_abbabb(b,e,c,i,k,l)*t1_aa(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('alde,becikl,dj->abcijk', g_abab[va, ob, va, vb], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||d,e>_aaaa*t3_aabaab(e,b,c,j,l,k)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('lade,ebcjlk,di->abcijk', g_aaaa[oa, va, va, va], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<a,l||d,e>_abab*t3_abbabb(b,e,c,j,k,l)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('alde,becjkl,di->abcijk', g_abab[va, ob, va, vb], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 <l,a||d,e>_aaaa*t1_aa(b,l)*t3_aabaab(d,e,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lade,bl,decijk->abcijk', g_aaaa[oa, va, va, va], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <a,l||d,e>_abab*t3_aabaab(d,b,e,i,j,k)*t1_bb(c,l)
    triples_res += -0.500000000000000 * einsum('alde,dbeijk,cl->abcijk', g_abab[va, ob, va, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <a,l||e,d>_abab*t3_aabaab(b,e,d,i,j,k)*t1_bb(c,l)
    triples_res +=  0.500000000000000 * einsum('aled,bedijk,cl->abcijk', g_abab[va, ob, va, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,b||d,e>_aaaa*t1_aa(a,l)*t3_aabaab(d,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lbde,al,decijk->abcijk', g_aaaa[oa, va, va, va], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <b,l||d,e>_abab*t3_aabaab(d,a,e,i,j,k)*t1_bb(c,l)
    triples_res +=  0.500000000000000 * einsum('blde,daeijk,cl->abcijk', g_abab[va, ob, va, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <b,l||e,d>_abab*t3_aabaab(a,e,d,i,j,k)*t1_bb(c,l)
    triples_res += -0.500000000000000 * einsum('bled,aedijk,cl->abcijk', g_abab[va, ob, va, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,c||d,e>_abab*t3_aabaab(b,a,e,i,j,k)*t1_aa(d,l)
    triples_res += -1.000000000000000 * einsum('lcde,baeijk,dl->abcijk', g_abab[oa, vb, va, vb], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,c||d,e>_bbbb*t3_aabaab(b,a,e,i,j,k)*t1_bb(d,l)
    triples_res += -1.000000000000000 * einsum('lcde,baeijk,dl->abcijk', g_bbbb[ob, vb, vb, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,c||e,d>_abab*t3_aaaaaa(e,a,b,i,j,l)*t1_bb(d,k)
    triples_res +=  1.000000000000000 * einsum('lced,eabijl,dk->abcijk', g_abab[oa, vb, va, vb], t3_aaaaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,c||d,e>_bbbb*t3_aabaab(b,a,e,i,j,l)*t1_bb(d,k)
    triples_res +=  1.000000000000000 * einsum('lcde,baeijl,dk->abcijk', g_bbbb[ob, vb, vb, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,c||d,e>_abab*t3_aabaab(b,a,e,i,l,k)*t1_aa(d,j)
    triples_res +=  1.000000000000000 * einsum('lcde,baeilk,dj->abcijk', g_abab[oa, vb, va, vb], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,c||d,e>_abab*t3_aabaab(b,a,e,j,l,k)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('lcde,baejlk,di->abcijk', g_abab[oa, vb, va, vb], t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 P(a,b)<l,c||d,e>_abab*t1_aa(a,l)*t3_aabaab(d,b,e,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lcde,al,dbeijk->abcijk', g_abab[oa, vb, va, vb], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(a,b)<l,c||e,d>_abab*t1_aa(a,l)*t3_aabaab(b,e,d,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('lced,al,bedijk->abcijk', g_abab[oa, vb, va, vb], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 <l,m||d,e>_abab*t3_aabaab(a,b,c,i,j,m)*t2_abab(d,e,l,k)
    triples_res += -0.500000000000000 * einsum('lmde,abcijm,delk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,m||e,d>_abab*t3_aabaab(a,b,c,i,j,m)*t2_abab(e,d,l,k)
    triples_res += -0.500000000000000 * einsum('lmed,abcijm,edlk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_bbbb*t3_aabaab(a,b,c,i,j,m)*t2_bbbb(d,e,k,l)
    triples_res += -0.500000000000000 * einsum('mlde,abcijm,dekl->abcijk', g_bbbb[ob, ob, vb, vb], t3_aabaab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_aaaa*t3_aabaab(a,b,c,i,m,k)*t2_aaaa(d,e,j,l)
    triples_res += -0.500000000000000 * einsum('mlde,abcimk,dejl->abcijk', g_aaaa[oa, oa, va, va], t3_aabaab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_abab*t3_aabaab(a,b,c,i,m,k)*t2_abab(d,e,j,l)
    triples_res += -0.500000000000000 * einsum('mlde,abcimk,dejl->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||e,d>_abab*t3_aabaab(a,b,c,i,m,k)*t2_abab(e,d,j,l)
    triples_res += -0.500000000000000 * einsum('mled,abcimk,edjl->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_aaaa*t3_aabaab(a,b,c,j,m,k)*t2_aaaa(d,e,i,l)
    triples_res +=  0.500000000000000 * einsum('mlde,abcjmk,deil->abcijk', g_aaaa[oa, oa, va, va], t3_aabaab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_abab*t3_aabaab(a,b,c,j,m,k)*t2_abab(d,e,i,l)
    triples_res +=  0.500000000000000 * einsum('mlde,abcjmk,deil->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||e,d>_abab*t3_aabaab(a,b,c,j,m,k)*t2_abab(e,d,i,l)
    triples_res +=  0.500000000000000 * einsum('mled,abcjmk,edil->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 P(i,j)<m,l||d,e>_abab*t3_aabaab(a,b,c,i,m,l)*t2_abab(d,e,j,k)
    contracted_intermediate =  0.250000000000000 * einsum('mlde,abciml,dejk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.2500 P(i,j)<m,l||e,d>_abab*t3_aabaab(a,b,c,i,m,l)*t2_abab(e,d,j,k)
    contracted_intermediate =  0.250000000000000 * einsum('mled,abciml,edjk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.2500 P(i,j)<l,m||d,e>_abab*t3_aabaab(a,b,c,i,l,m)*t2_abab(d,e,j,k)
    contracted_intermediate =  0.250000000000000 * einsum('lmde,abcilm,dejk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.2500 P(i,j)<l,m||e,d>_abab*t3_aabaab(a,b,c,i,l,m)*t2_abab(e,d,j,k)
    contracted_intermediate =  0.250000000000000 * einsum('lmed,abcilm,edjk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -0.2500 <m,l||d,e>_aaaa*t3_aabaab(a,b,c,l,m,k)*t2_aaaa(d,e,i,j)
    triples_res += -0.250000000000000 * einsum('mlde,abclmk,deij->abcijk', g_aaaa[oa, oa, va, va], t3_aabaab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 P(a,b)<m,l||d,e>_aaaa*t2_aaaa(d,a,m,l)*t3_aabaab(e,b,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,daml,ebcijk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||e,d>_abab*t2_abab(a,d,m,l)*t3_aabaab(e,b,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mled,adml,ebcijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<l,m||e,d>_abab*t2_abab(a,d,l,m)*t3_aabaab(e,b,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lmed,adlm,ebcijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,m||e,d>_abab*t2_abab(a,d,l,k)*t3_aabaab(e,b,c,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmed,adlk,ebcijm->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_aaaa*t2_aaaa(d,a,j,l)*t3_aabaab(e,b,c,i,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dajl,ebcimk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,m||d,e>_abab*t2_aaaa(d,a,j,l)*t3_abbabb(b,e,c,i,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('lmde,dajl,becikm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||e,d>_abab*t2_abab(a,d,j,l)*t3_aabaab(e,b,c,i,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mled,adjl,ebcimk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,e>_bbbb*t2_abab(a,d,j,l)*t3_abbabb(b,e,c,i,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,adjl,becikm->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,e>_aaaa*t2_aaaa(d,a,i,l)*t3_aabaab(e,b,c,j,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,dail,ebcjmk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,m||d,e>_abab*t2_aaaa(d,a,i,l)*t3_abbabb(b,e,c,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,dail,becjkm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||e,d>_abab*t2_abab(a,d,i,l)*t3_aabaab(e,b,c,j,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mled,adil,ebcjmk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_bbbb*t2_abab(a,d,i,l)*t3_abbabb(b,e,c,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,adil,becjkm->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(i,j)*P(a,b)<m,l||e,d>_abab*t2_abab(a,d,j,k)*t3_aabaab(e,b,c,i,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mled,adjk,ebciml->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -0.5000 P(i,j)*P(a,b)<l,m||e,d>_abab*t2_abab(a,d,j,k)*t3_aabaab(e,b,c,i,l,m)
    contracted_intermediate = -0.500000000000000 * einsum('lmed,adjk,ebcilm->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -0.5000 P(i,j)*P(a,b)<m,l||d,e>_bbbb*t2_abab(a,d,j,k)*t3_abbabb(b,e,c,i,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,adjk,beciml->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  0.5000 P(a,b)<m,l||d,e>_aaaa*t2_aaaa(d,a,i,j)*t3_aabaab(e,b,c,l,m,k)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,daij,ebclmk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||d,e>_abab*t2_aaaa(d,a,i,j)*t3_abbabb(b,e,c,m,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,daij,becmkl->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(a,b)<l,m||d,e>_abab*t2_aaaa(d,a,i,j)*t3_abbabb(b,e,c,l,m,k)
    contracted_intermediate =  0.500000000000000 * einsum('lmde,daij,beclmk->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 <m,l||d,e>_abab*t3_aabaab(b,a,e,i,j,k)*t2_abab(d,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,baeijk,dcml->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,m||d,e>_abab*t3_aabaab(b,a,e,i,j,k)*t2_abab(d,c,l,m)
    triples_res +=  0.500000000000000 * einsum('lmde,baeijk,dclm->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_bbbb*t3_aabaab(b,a,e,i,j,k)*t2_bbbb(d,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,baeijk,dcml->abcijk', g_bbbb[ob, ob, vb, vb], t3_aabaab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_aaaa*t3_aaaaaa(e,a,b,i,j,m)*t2_abab(d,c,l,k)
    triples_res += -1.000000000000000 * einsum('mlde,eabijm,dclk->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t3_aaaaaa(e,a,b,i,j,m)*t2_bbbb(d,c,k,l)
    triples_res += -1.000000000000000 * einsum('mled,eabijm,dckl->abcijk', g_abab[oa, ob, va, vb], t3_aaaaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,m||d,e>_abab*t3_aabaab(b,a,e,i,j,m)*t2_abab(d,c,l,k)
    triples_res += -1.000000000000000 * einsum('lmde,baeijm,dclk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t3_aabaab(b,a,e,i,j,m)*t2_bbbb(d,c,k,l)
    triples_res += -1.000000000000000 * einsum('mlde,baeijm,dckl->abcijk', g_bbbb[ob, ob, vb, vb], t3_aabaab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_abab*t3_aabaab(b,a,e,i,m,k)*t2_abab(d,c,j,l)
    triples_res += -1.000000000000000 * einsum('mlde,baeimk,dcjl->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <m,l||d,e>_abab*t3_aabaab(b,a,e,j,m,k)*t2_abab(d,c,i,l)
    triples_res +=  1.000000000000000 * einsum('mlde,baejmk,dcil->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 P(i,j)<m,l||d,e>_aaaa*t3_aaaaaa(e,a,b,i,m,l)*t2_abab(d,c,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,eabiml,dcjk->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 P(i,j)<m,l||d,e>_abab*t3_aabaab(b,a,e,i,m,l)*t2_abab(d,c,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,baeiml,dcjk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 P(i,j)<l,m||d,e>_abab*t3_aabaab(b,a,e,i,l,m)*t2_abab(d,c,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('lmde,baeilm,dcjk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.2500 <m,l||d,e>_aaaa*t2_aaaa(a,b,m,l)*t3_aabaab(d,e,c,i,j,k)
    triples_res +=  0.250000000000000 * einsum('mlde,abml,decijk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.2500 <m,l||d,e>_abab*t2_abab(a,c,m,l)*t3_aabaab(d,b,e,i,j,k)
    triples_res +=  0.250000000000000 * einsum('mlde,acml,dbeijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.2500 <m,l||e,d>_abab*t2_abab(a,c,m,l)*t3_aabaab(b,e,d,i,j,k)
    triples_res += -0.250000000000000 * einsum('mled,acml,bedijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.2500 <l,m||d,e>_abab*t2_abab(a,c,l,m)*t3_aabaab(d,b,e,i,j,k)
    triples_res +=  0.250000000000000 * einsum('lmde,aclm,dbeijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.2500 <l,m||e,d>_abab*t2_abab(a,c,l,m)*t3_aabaab(b,e,d,i,j,k)
    triples_res += -0.250000000000000 * einsum('lmed,aclm,bedijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||d,e>_aaaa*t2_abab(a,c,l,k)*t3_aaaaaa(d,e,b,i,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,aclk,debijm->abcijk', g_aaaa[oa, oa, va, va], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,m||d,e>_abab*t2_abab(a,c,l,k)*t3_aabaab(d,b,e,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmde,aclk,dbeijm->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,m||e,d>_abab*t2_abab(a,c,l,k)*t3_aabaab(b,e,d,i,j,m)
    triples_res +=  0.500000000000000 * einsum('lmed,aclk,bedijm->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,b,j,l)*t3_aabaab(d,e,c,i,m,k)
    triples_res += -0.500000000000000 * einsum('mlde,abjl,decimk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,m||d,e>_abab*t2_aaaa(a,b,j,l)*t3_abbabb(d,e,c,i,k,m)
    triples_res += -0.500000000000000 * einsum('lmde,abjl,decikm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,m||e,d>_abab*t2_aaaa(a,b,j,l)*t3_abbabb(e,d,c,i,k,m)
    triples_res += -0.500000000000000 * einsum('lmed,abjl,edcikm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_abab*t2_abab(a,c,j,l)*t3_aabaab(d,b,e,i,m,k)
    triples_res += -0.500000000000000 * einsum('mlde,acjl,dbeimk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||e,d>_abab*t2_abab(a,c,j,l)*t3_aabaab(b,e,d,i,m,k)
    triples_res +=  0.500000000000000 * einsum('mled,acjl,bedimk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_bbbb*t2_abab(a,c,j,l)*t3_abbabb(b,e,d,i,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,acjl,bedikm->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,b,i,l)*t3_aabaab(d,e,c,j,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,abil,decjmk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,m||d,e>_abab*t2_aaaa(a,b,i,l)*t3_abbabb(d,e,c,j,k,m)
    triples_res +=  0.500000000000000 * einsum('lmde,abil,decjkm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,m||e,d>_abab*t2_aaaa(a,b,i,l)*t3_abbabb(e,d,c,j,k,m)
    triples_res +=  0.500000000000000 * einsum('lmed,abil,edcjkm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_abab*t2_abab(a,c,i,l)*t3_aabaab(d,b,e,j,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,acil,dbejmk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||e,d>_abab*t2_abab(a,c,i,l)*t3_aabaab(b,e,d,j,m,k)
    triples_res += -0.500000000000000 * einsum('mled,acil,bedjmk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_bbbb*t2_abab(a,c,i,l)*t3_abbabb(b,e,d,j,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,acil,bedjkm->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.2500 <m,l||d,e>_abab*t3_aabaab(d,a,e,i,j,k)*t2_abab(b,c,m,l)
    triples_res += -0.250000000000000 * einsum('mlde,daeijk,bcml->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.2500 <l,m||d,e>_abab*t3_aabaab(d,a,e,i,j,k)*t2_abab(b,c,l,m)
    triples_res += -0.250000000000000 * einsum('lmde,daeijk,bclm->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 <m,l||e,d>_abab*t3_aabaab(a,e,d,i,j,k)*t2_abab(b,c,m,l)
    triples_res +=  0.250000000000000 * einsum('mled,aedijk,bcml->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 <l,m||e,d>_abab*t3_aabaab(a,e,d,i,j,k)*t2_abab(b,c,l,m)
    triples_res +=  0.250000000000000 * einsum('lmed,aedijk,bclm->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_aaaa*t3_aaaaaa(d,e,a,i,j,m)*t2_abab(b,c,l,k)
    triples_res +=  0.500000000000000 * einsum('mlde,deaijm,bclk->abcijk', g_aaaa[oa, oa, va, va], t3_aaaaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,m||d,e>_abab*t3_aabaab(d,a,e,i,j,m)*t2_abab(b,c,l,k)
    triples_res +=  0.500000000000000 * einsum('lmde,daeijm,bclk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,m||e,d>_abab*t3_aabaab(a,e,d,i,j,m)*t2_abab(b,c,l,k)
    triples_res += -0.500000000000000 * einsum('lmed,aedijm,bclk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <m,l||d,e>_abab*t3_aabaab(d,a,e,i,m,k)*t2_abab(b,c,j,l)
    triples_res +=  0.500000000000000 * einsum('mlde,daeimk,bcjl->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||e,d>_abab*t3_aabaab(a,e,d,i,m,k)*t2_abab(b,c,j,l)
    triples_res += -0.500000000000000 * einsum('mled,aedimk,bcjl->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||d,e>_bbbb*t3_abbabb(a,e,d,i,k,m)*t2_abab(b,c,j,l)
    triples_res += -0.500000000000000 * einsum('mlde,aedikm,bcjl->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||d,e>_abab*t3_aabaab(d,a,e,j,m,k)*t2_abab(b,c,i,l)
    triples_res += -0.500000000000000 * einsum('mlde,daejmk,bcil->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <m,l||e,d>_abab*t3_aabaab(a,e,d,j,m,k)*t2_abab(b,c,i,l)
    triples_res +=  0.500000000000000 * einsum('mled,aedjmk,bcil->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <m,l||d,e>_bbbb*t3_abbabb(a,e,d,j,k,m)*t2_abab(b,c,i,l)
    triples_res +=  0.500000000000000 * einsum('mlde,aedjkm,bcil->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 P(i,j)*P(a,b)<l,m||d,k>_abab*t2_aaaa(d,a,j,l)*t2_abab(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('lmdk,dajl,bcim->abcijk', g_abab[oa, ob, va, ob], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,k>_bbbb*t2_abab(a,d,j,l)*t2_abab(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldk,adjl,bcim->abcijk', g_bbbb[ob, ob, vb, ob], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||d,k>_abab*t2_aaaa(d,a,i,j)*t2_abab(b,c,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mldk,daij,bcml->abcijk', g_abab[oa, ob, va, ob], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<l,m||d,k>_abab*t2_aaaa(d,a,i,j)*t2_abab(b,c,l,m)
    contracted_intermediate = -0.500000000000000 * einsum('lmdk,daij,bclm->abcijk', g_abab[oa, ob, va, ob], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(a,b)<m,l||j,d>_abab*t2_abab(a,d,i,k)*t2_abab(b,c,m,l)
    contracted_intermediate =  0.500000000000000 * einsum('mljd,adik,bcml->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(a,b)<l,m||j,d>_abab*t2_abab(a,d,i,k)*t2_abab(b,c,l,m)
    contracted_intermediate =  0.500000000000000 * einsum('lmjd,adik,bclm->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||d,k>_abab*t2_aaaa(a,b,i,m)*t2_abab(d,c,j,l)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,abim,dcjl->abcijk', g_abab[oa, ob, va, ob], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 <m,l||d,j>_aaaa*t2_aaaa(a,b,m,l)*t2_abab(d,c,i,k)
    triples_res +=  0.500000000000000 * einsum('mldj,abml,dcik->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 P(a,b)<l,m||j,d>_abab*t2_abab(a,d,l,k)*t2_abab(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('lmjd,adlk,bcim->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,j>_aaaa*t2_aaaa(d,a,i,l)*t2_abab(b,c,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,dail,bcmk->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||j,d>_abab*t2_abab(a,d,i,l)*t2_abab(b,c,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mljd,adil,bcmk->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <m,l||d,j>_aaaa*t2_aaaa(a,b,i,m)*t2_abab(d,c,l,k)
    triples_res +=  1.000000000000000 * einsum('mldj,abim,dclk->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||j,d>_abab*t2_aaaa(a,b,i,m)*t2_bbbb(d,c,k,l)
    triples_res +=  1.000000000000000 * einsum('mljd,abim,dckl->abcijk', g_abab[oa, ob, oa, vb], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 P(a,b)<l,m||i,d>_abab*t2_abab(a,d,l,k)*t2_abab(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmid,adlk,bcjm->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,i>_aaaa*t2_aaaa(d,a,j,l)*t2_abab(b,c,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,dajl,bcmk->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||i,d>_abab*t2_abab(a,d,j,l)*t2_abab(b,c,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlid,adjl,bcmk->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||i,d>_abab*t2_abab(a,d,j,k)*t2_abab(b,c,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlid,adjk,bcml->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<l,m||i,d>_abab*t2_abab(a,d,j,k)*t2_abab(b,c,l,m)
    contracted_intermediate = -0.500000000000000 * einsum('lmid,adjk,bclm->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 <m,l||d,i>_aaaa*t2_aaaa(a,b,j,m)*t2_abab(d,c,l,k)
    triples_res += -1.000000000000000 * einsum('mldi,abjm,dclk->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <m,l||i,d>_abab*t2_aaaa(a,b,j,m)*t2_bbbb(d,c,k,l)
    triples_res += -1.000000000000000 * einsum('mlid,abjm,dckl->abcijk', g_abab[oa, ob, oa, vb], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,i>_aaaa*t2_aaaa(a,b,m,l)*t2_abab(d,c,j,k)
    triples_res += -0.500000000000000 * einsum('mldi,abml,dcjk->abcijk', g_aaaa[oa, oa, va, oa], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 P(i,j)*P(a,b)<a,l||d,e>_abab*t2_abab(b,c,i,l)*t2_abab(d,e,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('alde,bcil,dejk->abcijk', g_abab[va, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  0.5000 P(i,j)*P(a,b)<a,l||e,d>_abab*t2_abab(b,c,i,l)*t2_abab(e,d,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('aled,bcil,edjk->abcijk', g_abab[va, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  0.5000 P(a,b)<l,a||d,e>_aaaa*t2_abab(b,c,l,k)*t2_aaaa(d,e,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('lade,bclk,deij->abcijk', g_aaaa[oa, va, va, va], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,a||d,e>_aaaa*t2_aaaa(d,b,j,l)*t2_abab(e,c,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('lade,dbjl,ecik->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<a,l||e,d>_abab*t2_abab(b,d,j,l)*t2_abab(e,c,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('aled,bdjl,ecik->abcijk', g_abab[va, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||d,e>_aaaa*t2_aaaa(d,b,i,l)*t2_abab(e,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dbil,ecjk->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<a,l||e,d>_abab*t2_abab(b,d,i,l)*t2_abab(e,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('aled,bdil,ecjk->abcijk', g_abab[va, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<a,l||e,d>_abab*t2_abab(b,d,j,k)*t2_abab(e,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('aled,bdjk,ecil->abcijk', g_abab[va, ob, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,a||d,e>_aaaa*t2_aaaa(d,b,i,j)*t2_abab(e,c,l,k)
    contracted_intermediate = -1.000000000000000 * einsum('lade,dbij,eclk->abcijk', g_aaaa[oa, va, va, va], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<a,l||d,e>_abab*t2_aaaa(d,b,i,j)*t2_bbbb(e,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('alde,dbij,eckl->abcijk', g_abab[va, ob, va, vb], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(i,j)<l,c||d,e>_abab*t2_aaaa(a,b,i,l)*t2_abab(d,e,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lcde,abil,dejk->abcijk', g_abab[oa, vb, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -0.5000 P(i,j)<l,c||e,d>_abab*t2_aaaa(a,b,i,l)*t2_abab(e,d,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lced,abil,edjk->abcijk', g_abab[oa, vb, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 <l,c||e,d>_abab*t2_abab(a,d,l,k)*t2_aaaa(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('lced,adlk,ebij->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,c||d,e>_abab*t2_aaaa(d,a,j,l)*t2_abab(b,e,i,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dajl,beik->abcijk', g_abab[oa, vb, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,c||d,e>_bbbb*t2_abab(a,d,j,l)*t2_abab(b,e,i,k)
    triples_res += -1.000000000000000 * einsum('lcde,adjl,beik->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,c||d,e>_abab*t2_aaaa(d,a,i,l)*t2_abab(b,e,j,k)
    triples_res += -1.000000000000000 * einsum('lcde,dail,bejk->abcijk', g_abab[oa, vb, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,c||d,e>_bbbb*t2_abab(a,d,i,l)*t2_abab(b,e,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,adil,bejk->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)<l,c||e,d>_abab*t2_abab(a,d,j,k)*t2_aaaa(e,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lced,adjk,ebil->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(i,j)<l,c||d,e>_bbbb*t2_abab(a,d,j,k)*t2_abab(b,e,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,adjk,beil->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 <l,c||d,e>_abab*t2_aaaa(d,a,i,j)*t2_abab(b,e,l,k)
    triples_res +=  1.000000000000000 * einsum('lcde,daij,belk->abcijk', g_abab[oa, vb, va, vb], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<l,m||d,e>_abab*t2_abab(a,e,j,k)*t2_abab(b,c,i,m)*t1_aa(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,aejk,bcim,dl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,e>_bbbb*t2_abab(a,e,j,k)*t2_abab(b,c,i,m)*t1_bb(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,aejk,bcim,dl->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,e>_aaaa*t2_aaaa(e,a,i,j)*t2_abab(b,c,m,k)*t1_aa(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,eaij,bcmk,dl->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_abab, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||e,d>_abab*t2_aaaa(e,a,i,j)*t2_abab(b,c,m,k)*t1_bb(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mled,eaij,bcmk,dl->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||d,e>_aaaa*t2_aaaa(a,b,i,m)*t2_abab(e,c,j,k)*t1_aa(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,abim,ecjk,dl->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_abab, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(i,j)<m,l||e,d>_abab*t2_aaaa(a,b,i,m)*t2_abab(e,c,j,k)*t1_bb(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('mled,abim,ecjk,dl->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<l,m||e,d>_abab*t2_aaaa(e,a,j,l)*t2_abab(b,c,i,m)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lmed,eajl,bcim,dk->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<m,l||d,e>_bbbb*t2_abab(a,e,j,l)*t2_abab(b,c,i,m)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,aejl,bcim,dk->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||e,d>_abab*t2_aaaa(e,a,i,j)*t2_abab(b,c,m,l)*t1_bb(d,k)
    contracted_intermediate = -0.500000000000000 * einsum('mled,eaij,bcml,dk->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<l,m||e,d>_abab*t2_aaaa(e,a,i,j)*t2_abab(b,c,l,m)*t1_bb(d,k)
    contracted_intermediate = -0.500000000000000 * einsum('lmed,eaij,bclm,dk->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(a,b)<m,l||d,e>_abab*t2_abab(a,e,i,k)*t2_abab(b,c,m,l)*t1_aa(d,j)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,aeik,bcml,dj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(a,b)<l,m||d,e>_abab*t2_abab(a,e,i,k)*t2_abab(b,c,l,m)*t1_aa(d,j)
    contracted_intermediate =  0.500000000000000 * einsum('lmde,aeik,bclm,dj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||e,d>_abab*t2_aaaa(a,b,i,m)*t2_abab(e,c,j,l)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mled,abim,ecjl,dk->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,b,m,l)*t2_abab(e,c,i,k)*t1_aa(d,j)
    triples_res += -0.500000000000000 * einsum('mlde,abml,ecik,dj->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -1.0000 P(a,b)<l,m||d,e>_abab*t2_abab(a,e,l,k)*t2_abab(b,c,i,m)*t1_aa(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('lmde,aelk,bcim,dj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_aaaa*t2_aaaa(e,a,i,l)*t2_abab(b,c,m,k)*t1_aa(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,eail,bcmk,dj->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,e>_abab*t2_abab(a,e,i,l)*t2_abab(b,c,m,k)*t1_aa(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,aeil,bcmk,dj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 <m,l||d,e>_aaaa*t2_aaaa(a,b,i,m)*t2_abab(e,c,l,k)*t1_aa(d,j)
    triples_res += -1.000000000000000 * einsum('mlde,abim,eclk,dj->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_abab*t2_aaaa(a,b,i,m)*t2_bbbb(e,c,k,l)*t1_aa(d,j)
    triples_res +=  1.000000000000000 * einsum('mlde,abim,eckl,dj->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 P(a,b)<l,m||d,e>_abab*t2_abab(a,e,l,k)*t2_abab(b,c,j,m)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,aelk,bcjm,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,e>_aaaa*t2_aaaa(e,a,j,l)*t2_abab(b,c,m,k)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,eajl,bcmk,di->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_abab*t2_abab(a,e,j,l)*t2_abab(b,c,m,k)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,aejl,bcmk,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||d,e>_abab*t2_abab(a,e,j,k)*t2_abab(b,c,m,l)*t1_aa(d,i)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,aejk,bcml,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<l,m||d,e>_abab*t2_abab(a,e,j,k)*t2_abab(b,c,l,m)*t1_aa(d,i)
    contracted_intermediate = -0.500000000000000 * einsum('lmde,aejk,bclm,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <m,l||d,e>_aaaa*t2_aaaa(a,b,j,m)*t2_abab(e,c,l,k)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('mlde,abjm,eclk,di->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_abab*t2_aaaa(a,b,j,m)*t2_bbbb(e,c,k,l)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('mlde,abjm,eckl,di->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,b,m,l)*t2_abab(e,c,j,k)*t1_aa(d,i)
    triples_res +=  0.500000000000000 * einsum('mlde,abml,ecjk,di->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -0.5000 P(i,j)*P(a,b)<l,m||d,e>_abab*t1_aa(a,l)*t2_abab(b,c,i,m)*t2_abab(d,e,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lmde,al,bcim,dejk->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -0.5000 P(i,j)*P(a,b)<l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,c,i,m)*t2_abab(e,d,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lmed,al,bcim,edjk->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_abab(b,c,m,k)*t2_aaaa(d,e,i,j)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,al,bcmk,deij->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,j,m)*t2_abab(e,c,i,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,al,dbjm,ecik->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,d,j,m)*t2_abab(e,c,i,k)
    contracted_intermediate = -1.000000000000000 * einsum('lmed,al,bdjm,ecik->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,m)*t2_abab(e,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,dbim,ecjk->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,d,i,m)*t2_abab(e,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmed,al,bdim,ecjk->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,d,j,k)*t2_abab(e,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmed,al,bdjk,ecim->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,j)*t2_abab(e,c,m,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,al,dbij,ecmk->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,m||d,e>_abab*t1_aa(a,l)*t2_aaaa(d,b,i,j)*t2_bbbb(e,c,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,al,dbij,eckm->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(i,j)<m,l||d,e>_abab*t2_aaaa(a,b,i,m)*t1_bb(c,l)*t2_abab(d,e,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,abim,cl,dejk->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 P(i,j)<m,l||e,d>_abab*t2_aaaa(a,b,i,m)*t1_bb(c,l)*t2_abab(e,d,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('mled,abim,cl,edjk->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 <m,l||e,d>_abab*t2_abab(a,d,m,k)*t2_aaaa(e,b,i,j)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mled,admk,ebij,cl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_aaaa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_abab*t2_aaaa(d,a,j,m)*t2_abab(b,e,i,k)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('mlde,dajm,beik,cl->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,j,m)*t2_abab(b,e,i,k)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mlde,adjm,beik,cl->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_abab*t2_aaaa(d,a,i,m)*t2_abab(b,e,j,k)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mlde,daim,bejk,cl->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,i,m)*t2_abab(b,e,j,k)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('mlde,adim,bejk,cl->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)<m,l||e,d>_abab*t2_abab(a,d,j,k)*t2_aaaa(e,b,i,m)*t1_bb(c,l)
    contracted_intermediate = -1.000000000000000 * einsum('mled,adjk,ebim,cl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_aaaa, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(i,j)<m,l||d,e>_bbbb*t2_abab(a,d,j,k)*t2_abab(b,e,i,m)*t1_bb(c,l)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,adjk,beim,cl->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 <m,l||d,e>_abab*t2_aaaa(d,a,i,j)*t2_abab(b,e,m,k)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('mlde,daij,bemk,cl->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	 -1.0000 P(i,j)*P(a,b)<l,m||d,k>_abab*t1_aa(a,l)*t2_abab(b,c,i,m)*t1_aa(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('lmdk,al,bcim,dj->abcijk', g_abab[oa, ob, va, ob], t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||d,k>_abab*t2_aaaa(a,b,i,m)*t1_bb(c,l)*t1_aa(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,abim,cl,dj->abcijk', g_abab[oa, ob, va, ob], t2_aaaa, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 <l,m||d,k>_abab*t1_aa(a,l)*t2_aaaa(d,b,i,j)*t1_bb(c,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,al,dbij,cm->abcijk', g_abab[oa, ob, va, ob], t1_aa, t2_aaaa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -1.0000 <m,l||d,j>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t2_abab(d,c,i,k)
    triples_res += -1.000000000000000 * einsum('mldj,al,bm,dcik->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <l,m||j,d>_abab*t1_aa(a,l)*t2_abab(b,d,i,k)*t1_bb(c,m)
    triples_res += -1.000000000000000 * einsum('lmjd,al,bdik,cm->abcijk', g_abab[oa, ob, oa, vb], t1_aa, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <l,m||d,k>_abab*t2_aaaa(d,a,i,j)*t1_aa(b,l)*t1_bb(c,m)
    triples_res += -1.000000000000000 * einsum('lmdk,daij,bl,cm->abcijk', g_abab[oa, ob, va, ob], t2_aaaa, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	  1.0000 <l,m||j,d>_abab*t2_abab(a,d,i,k)*t1_aa(b,l)*t1_bb(c,m)
    triples_res +=  1.000000000000000 * einsum('lmjd,adik,bl,cm->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 P(a,b)<l,m||j,d>_abab*t1_aa(a,l)*t2_abab(b,c,i,m)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lmjd,al,bcim,dk->abcijk', g_abab[oa, ob, oa, vb], t1_aa, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,j>_aaaa*t1_aa(a,l)*t2_abab(b,c,m,k)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,al,bcmk,di->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <m,l||j,d>_abab*t2_aaaa(a,b,i,m)*t1_bb(c,l)*t1_bb(d,k)
    triples_res +=  1.000000000000000 * einsum('mljd,abim,cl,dk->abcijk', g_abab[oa, ob, oa, vb], t2_aaaa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 P(a,b)<l,m||i,d>_abab*t1_aa(a,l)*t2_abab(b,c,j,m)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmid,al,bcjm,dk->abcijk', g_abab[oa, ob, oa, vb], t1_aa, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,i>_aaaa*t1_aa(a,l)*t2_abab(b,c,m,k)*t1_aa(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,al,bcmk,dj->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 <m,l||i,d>_abab*t2_aaaa(a,b,j,m)*t1_bb(c,l)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('mlid,abjm,cl,dk->abcijk', g_abab[oa, ob, oa, vb], t2_aaaa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,i>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t2_abab(d,c,j,k)
    triples_res +=  1.000000000000000 * einsum('mldi,al,bm,dcjk->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <l,m||i,d>_abab*t1_aa(a,l)*t2_abab(b,d,j,k)*t1_bb(c,m)
    triples_res +=  1.000000000000000 * einsum('lmid,al,bdjk,cm->abcijk', g_abab[oa, ob, oa, vb], t1_aa, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <l,m||i,d>_abab*t2_abab(a,d,j,k)*t1_aa(b,l)*t1_bb(c,m)
    triples_res += -1.000000000000000 * einsum('lmid,adjk,bl,cm->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<a,l||e,d>_abab*t2_abab(b,c,i,l)*t1_bb(d,k)*t1_aa(e,j)
    contracted_intermediate =  1.000000000000000 * einsum('aled,bcil,dk,ej->abcijk', g_abab[va, ob, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 <a,l||e,d>_abab*t2_aaaa(e,b,i,j)*t1_bb(c,l)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('aled,ebij,cl,dk->abcijk', g_abab[va, ob, va, vb], t2_aaaa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	 -1.0000 <l,a||d,e>_aaaa*t1_aa(b,l)*t2_abab(e,c,i,k)*t1_aa(d,j)
    triples_res += -1.000000000000000 * einsum('lade,bl,ecik,dj->abcijk', g_aaaa[oa, va, va, va], t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	  1.0000 <a,l||d,e>_abab*t2_abab(b,e,i,k)*t1_bb(c,l)*t1_aa(d,j)
    triples_res +=  1.000000000000000 * einsum('alde,beik,cl,dj->abcijk', g_abab[va, ob, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 P(a,b)<l,a||d,e>_aaaa*t2_abab(b,c,l,k)*t1_aa(d,j)*t1_aa(e,i)
    contracted_intermediate = -1.000000000000000 * einsum('lade,bclk,dj,ei->abcijk', g_aaaa[oa, va, va, va], t2_abab, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <l,a||d,e>_aaaa*t1_aa(b,l)*t2_abab(e,c,j,k)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('lade,bl,ecjk,di->abcijk', g_aaaa[oa, va, va, va], t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -1.0000 <a,l||d,e>_abab*t2_abab(b,e,j,k)*t1_bb(c,l)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('alde,bejk,cl,di->abcijk', g_abab[va, ob, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <b,l||e,d>_abab*t2_aaaa(e,a,i,j)*t1_bb(c,l)*t1_bb(d,k)
    triples_res +=  1.000000000000000 * einsum('bled,eaij,cl,dk->abcijk', g_abab[va, ob, va, vb], t2_aaaa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  1.0000 <l,b||d,e>_aaaa*t1_aa(a,l)*t2_abab(e,c,i,k)*t1_aa(d,j)
    triples_res +=  1.000000000000000 * einsum('lbde,al,ecik,dj->abcijk', g_aaaa[oa, va, va, va], t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -1.0000 <b,l||d,e>_abab*t2_abab(a,e,i,k)*t1_bb(c,l)*t1_aa(d,j)
    triples_res += -1.000000000000000 * einsum('blde,aeik,cl,dj->abcijk', g_abab[va, ob, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <l,b||d,e>_aaaa*t1_aa(a,l)*t2_abab(e,c,j,k)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('lbde,al,ecjk,di->abcijk', g_aaaa[oa, va, va, va], t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	  1.0000 <b,l||d,e>_abab*t2_abab(a,e,j,k)*t1_bb(c,l)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('blde,aejk,cl,di->abcijk', g_abab[va, ob, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)<l,c||e,d>_abab*t2_aaaa(a,b,i,l)*t1_bb(d,k)*t1_aa(e,j)
    contracted_intermediate = -1.000000000000000 * einsum('lced,abil,dk,ej->abcijk', g_abab[oa, vb, va, vb], t2_aaaa, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,c||e,d>_abab*t1_aa(a,l)*t2_aaaa(e,b,i,j)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lced,al,ebij,dk->abcijk', g_abab[oa, vb, va, vb], t1_aa, t2_aaaa, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,c||d,e>_abab*t1_aa(a,l)*t2_abab(b,e,i,k)*t1_aa(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,al,beik,dj->abcijk', g_abab[oa, vb, va, vb], t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,c||d,e>_abab*t1_aa(a,l)*t2_abab(b,e,j,k)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('lcde,al,bejk,di->abcijk', g_abab[oa, vb, va, vb], t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 <l,m||d,e>_abab*t3_aabaab(a,b,c,i,j,m)*t1_aa(d,l)*t1_bb(e,k)
    triples_res += -1.000000000000000 * einsum('lmde,abcijm,dl,ek->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t3_aabaab(a,b,c,i,j,m)*t1_bb(d,l)*t1_bb(e,k)
    triples_res +=  1.000000000000000 * einsum('mlde,abcijm,dl,ek->abcijk', g_bbbb[ob, ob, vb, vb], t3_aabaab, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_aaaa*t3_aabaab(a,b,c,i,m,k)*t1_aa(d,l)*t1_aa(e,j)
    triples_res +=  1.000000000000000 * einsum('mlde,abcimk,dl,ej->abcijk', g_aaaa[oa, oa, va, va], t3_aabaab, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t3_aabaab(a,b,c,i,m,k)*t1_bb(d,l)*t1_aa(e,j)
    triples_res += -1.000000000000000 * einsum('mled,abcimk,dl,ej->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_aaaa*t3_aabaab(a,b,c,j,m,k)*t1_aa(d,l)*t1_aa(e,i)
    triples_res += -1.000000000000000 * einsum('mlde,abcjmk,dl,ei->abcijk', g_aaaa[oa, oa, va, va], t3_aabaab, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 <m,l||e,d>_abab*t3_aabaab(a,b,c,j,m,k)*t1_bb(d,l)*t1_aa(e,i)
    triples_res +=  1.000000000000000 * einsum('mled,abcjmk,dl,ei->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,m)*t3_aabaab(e,b,c,i,j,k)*t1_aa(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,am,ebcijk,dl->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t3_aabaab, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||e,d>_abab*t1_aa(a,m)*t3_aabaab(e,b,c,i,j,k)*t1_bb(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('mled,am,ebcijk,dl->abcijk', g_abab[oa, ob, va, vb], t1_aa, t3_aabaab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <l,m||d,e>_abab*t3_aabaab(b,a,e,i,j,k)*t1_bb(c,m)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('lmde,baeijk,cm,dl->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t3_aabaab(b,a,e,i,j,k)*t1_bb(c,m)*t1_bb(d,l)
    triples_res += -1.000000000000000 * einsum('mlde,baeijk,cm,dl->abcijk', g_bbbb[ob, ob, vb, vb], t3_aabaab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  0.5000 P(i,j)<m,l||e,d>_abab*t3_aabaab(a,b,c,i,m,l)*t1_bb(d,k)*t1_aa(e,j)
    contracted_intermediate =  0.500000000000000 * einsum('mled,abciml,dk,ej->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 P(i,j)<l,m||e,d>_abab*t3_aabaab(a,b,c,i,l,m)*t1_bb(d,k)*t1_aa(e,j)
    contracted_intermediate =  0.500000000000000 * einsum('lmed,abcilm,dk,ej->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(a,b)<l,m||e,d>_abab*t1_aa(a,l)*t3_aabaab(e,b,c,i,j,m)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmed,al,ebcijm,dk->abcijk', g_abab[oa, ob, va, vb], t1_aa, t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t3_aabaab(e,b,c,i,m,k)*t1_aa(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,al,ebcimk,dj->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,m||d,e>_abab*t1_aa(a,l)*t3_abbabb(b,e,c,i,k,m)*t1_aa(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('lmde,al,becikm,dj->abcijk', g_abab[oa, ob, va, vb], t1_aa, t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 <m,l||e,d>_abab*t3_aaaaaa(e,a,b,i,j,m)*t1_bb(c,l)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('mled,eabijm,cl,dk->abcijk', g_abab[oa, ob, va, vb], t3_aaaaaa, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t3_aabaab(b,a,e,i,j,m)*t1_bb(c,l)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('mlde,baeijm,cl,dk->abcijk', g_bbbb[ob, ob, vb, vb], t3_aabaab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_abab*t3_aabaab(b,a,e,i,m,k)*t1_bb(c,l)*t1_aa(d,j)
    triples_res += -1.000000000000000 * einsum('mlde,baeimk,cl,dj->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_aaaa*t3_aabaab(a,b,c,l,m,k)*t1_aa(d,j)*t1_aa(e,i)
    triples_res +=  0.500000000000000 * einsum('mlde,abclmk,dj,ei->abcijk', g_aaaa[oa, oa, va, va], t3_aabaab, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t3_aabaab(e,b,c,j,m,k)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,ebcjmk,di->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t3_aabaab, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,m||d,e>_abab*t1_aa(a,l)*t3_abbabb(b,e,c,j,k,m)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,al,becjkm,di->abcijk', g_abab[oa, ob, va, vb], t1_aa, t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <m,l||d,e>_abab*t3_aabaab(b,a,e,j,m,k)*t1_bb(c,l)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('mlde,baejmk,cl,di->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t3_aabaab(d,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,al,bm,decijk->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  0.5000 <l,m||d,e>_abab*t1_aa(a,l)*t3_aabaab(d,b,e,i,j,k)*t1_bb(c,m)
    triples_res +=  0.500000000000000 * einsum('lmde,al,dbeijk,cm->abcijk', g_abab[oa, ob, va, vb], t1_aa, t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -0.5000 <l,m||e,d>_abab*t1_aa(a,l)*t3_aabaab(b,e,d,i,j,k)*t1_bb(c,m)
    triples_res += -0.500000000000000 * einsum('lmed,al,bedijk,cm->abcijk', g_abab[oa, ob, va, vb], t1_aa, t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -0.5000 <l,m||d,e>_abab*t3_aabaab(d,a,e,i,j,k)*t1_aa(b,l)*t1_bb(c,m)
    triples_res += -0.500000000000000 * einsum('lmde,daeijk,bl,cm->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  0.5000 <l,m||e,d>_abab*t3_aabaab(a,e,d,i,j,k)*t1_aa(b,l)*t1_bb(c,m)
    triples_res +=  0.500000000000000 * einsum('lmed,aedijk,bl,cm->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)*P(a,b)<l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,c,i,m)*t1_bb(d,k)*t1_aa(e,j)
    contracted_intermediate = -1.000000000000000 * einsum('lmed,al,bcim,dk,ej->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||e,d>_abab*t2_aaaa(a,b,i,m)*t1_bb(c,l)*t1_bb(d,k)*t1_aa(e,j)
    contracted_intermediate =  1.000000000000000 * einsum('mled,abim,cl,dk,ej->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_aaaa(e,b,i,j)*t1_bb(c,m)*t1_bb(d,k)
    triples_res +=  1.000000000000000 * einsum('lmed,al,ebij,cm,dk->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_aaaa, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (2, 3), (1, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t2_abab(e,c,i,k)*t1_aa(d,j)
    triples_res +=  1.000000000000000 * einsum('mlde,al,bm,ecik,dj->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (1, 2), (0, 1)])

    #	 -1.0000 <l,m||d,e>_abab*t1_aa(a,l)*t2_abab(b,e,i,k)*t1_bb(c,m)*t1_aa(d,j)
    triples_res += -1.000000000000000 * einsum('lmde,al,beik,cm,dj->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <l,m||e,d>_abab*t2_aaaa(e,a,i,j)*t1_aa(b,l)*t1_bb(c,m)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('lmed,eaij,bl,cm,dk->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (2, 3), (1, 2), (0, 1)])

    #	  1.0000 <l,m||d,e>_abab*t2_abab(a,e,i,k)*t1_aa(b,l)*t1_bb(c,m)*t1_aa(d,j)
    triples_res +=  1.000000000000000 * einsum('lmde,aeik,bl,cm,dj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])

    #	  1.0000 P(a,b)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_abab(b,c,m,k)*t1_aa(d,j)*t1_aa(e,i)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,al,bcmk,dj,ei->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_abab, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t2_abab(e,c,j,k)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('mlde,al,bm,ecjk,di->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t1_aa, t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (1, 2), (0, 1)])

    #	  1.0000 <l,m||d,e>_abab*t1_aa(a,l)*t2_abab(b,e,j,k)*t1_bb(c,m)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('lmde,al,bejk,cm,di->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <l,m||d,e>_abab*t2_abab(a,e,j,k)*t1_aa(b,l)*t1_bb(c,m)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('lmde,aejk,bl,cm,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_aa, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 3), (0, 2), (0, 1)])

    return triples_res


def ccsdt_t3_abbabb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :

    #	 -1.0000 P(j,k)f_bb(l,k)*t3_abbabb(a,b,c,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('lk,abcijl->abcijk', f_bb[ob, ob], t3_abbabb)
    triples_res  =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 f_aa(l,i)*t3_abbabb(a,b,c,l,k,j)
    triples_res +=  1.000000000000000 * einsum('li,abclkj->abcijk', f_aa[oa, oa], t3_abbabb)

    #	  1.0000 f_aa(a,d)*t3_abbabb(d,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('ad,dbcijk->abcijk', f_aa[va, va], t3_abbabb)

    #	  1.0000 f_bb(b,d)*t3_abbabb(a,d,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('bd,adcijk->abcijk', f_bb[vb, vb], t3_abbabb)

    #	 -1.0000 f_bb(c,d)*t3_abbabb(a,d,b,i,j,k)
    triples_res += -1.000000000000000 * einsum('cd,adbijk->abcijk', f_bb[vb, vb], t3_abbabb)

    #	 -1.0000 P(j,k)f_bb(l,d)*t3_abbabb(a,b,c,i,j,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('ld,abcijl,dk->abcijk', f_bb[ob, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 f_aa(l,d)*t3_abbabb(a,b,c,l,k,j)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('ld,abclkj,di->abcijk', f_aa[oa, va], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 f_aa(l,d)*t1_aa(a,l)*t3_abbabb(d,b,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('ld,al,dbcijk->abcijk', f_aa[oa, va], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 f_bb(l,d)*t3_abbabb(a,d,c,i,j,k)*t1_bb(b,l)
    triples_res += -1.000000000000000 * einsum('ld,adcijk,bl->abcijk', f_bb[ob, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 f_bb(l,d)*t3_abbabb(a,d,b,i,j,k)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('ld,adbijk,cl->abcijk', f_bb[ob, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 f_bb(l,d)*t2_abab(a,c,i,l)*t2_bbbb(d,b,j,k)
    triples_res +=  1.000000000000000 * einsum('ld,acil,dbjk->abcijk', f_bb[ob, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 f_bb(l,d)*t2_abab(a,d,i,k)*t2_bbbb(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('ld,adik,bcjl->abcijk', f_bb[ob, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 f_aa(l,d)*t2_abab(a,c,l,j)*t2_abab(d,b,i,k)
    triples_res +=  1.000000000000000 * einsum('ld,aclj,dbik->abcijk', f_aa[oa, va], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 f_bb(l,d)*t2_abab(a,d,i,j)*t2_bbbb(b,c,k,l)
    triples_res +=  1.000000000000000 * einsum('ld,adij,bckl->abcijk', f_bb[ob, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 f_aa(l,d)*t2_abab(a,c,l,k)*t2_abab(d,b,i,j)
    triples_res += -1.000000000000000 * einsum('ld,aclk,dbij->abcijk', f_aa[oa, va], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 f_bb(l,d)*t2_abab(a,b,i,l)*t2_bbbb(d,c,j,k)
    triples_res += -1.000000000000000 * einsum('ld,abil,dcjk->abcijk', f_bb[ob, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 f_aa(l,d)*t2_abab(a,b,l,j)*t2_abab(d,c,i,k)
    triples_res += -1.000000000000000 * einsum('ld,ablj,dcik->abcijk', f_aa[oa, va], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 f_aa(l,d)*t2_abab(a,b,l,k)*t2_abab(d,c,i,j)
    triples_res +=  1.000000000000000 * einsum('ld,ablk,dcij->abcijk', f_aa[oa, va], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,b||j,k>_bbbb*t2_abab(a,c,i,l)
    triples_res +=  1.000000000000000 * einsum('lbjk,acil->abcijk', g_bbbb[ob, vb, ob, ob], t2_abab)

    #	 -1.0000 <a,l||i,k>_abab*t2_bbbb(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('alik,bcjl->abcijk', g_abab[va, ob, oa, ob], t2_bbbb)

    #	  1.0000 <l,b||i,k>_abab*t2_abab(a,c,l,j)
    triples_res +=  1.000000000000000 * einsum('lbik,aclj->abcijk', g_abab[oa, vb, oa, ob], t2_abab)

    #	  1.0000 <a,l||i,j>_abab*t2_bbbb(b,c,k,l)
    triples_res +=  1.000000000000000 * einsum('alij,bckl->abcijk', g_abab[va, ob, oa, ob], t2_bbbb)

    #	 -1.0000 <l,b||i,j>_abab*t2_abab(a,c,l,k)
    triples_res += -1.000000000000000 * einsum('lbij,aclk->abcijk', g_abab[oa, vb, oa, ob], t2_abab)

    #	 -1.0000 <l,c||j,k>_bbbb*t2_abab(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('lcjk,abil->abcijk', g_bbbb[ob, vb, ob, ob], t2_abab)

    #	 -1.0000 <l,c||i,k>_abab*t2_abab(a,b,l,j)
    triples_res += -1.000000000000000 * einsum('lcik,ablj->abcijk', g_abab[oa, vb, oa, ob], t2_abab)

    #	  1.0000 <l,c||i,j>_abab*t2_abab(a,b,l,k)
    triples_res +=  1.000000000000000 * einsum('lcij,ablk->abcijk', g_abab[oa, vb, oa, ob], t2_abab)

    #	 -1.0000 P(j,k)*P(b,c)<a,b||d,k>_abab*t2_abab(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abdk,dcij->abcijk', g_abab[va, vb, va, ob], t2_abab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  1.0000 P(b,c)<a,b||i,d>_abab*t2_bbbb(d,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('abid,dcjk->abcijk', g_abab[va, vb, oa, vb], t2_bbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  1.0000 P(j,k)<b,c||d,k>_bbbb*t2_abab(a,d,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('bcdk,adij->abcijk', g_bbbb[vb, vb, vb, ob], t2_abab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 <m,l||j,k>_bbbb*t3_abbabb(a,b,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mljk,abciml->abcijk', g_bbbb[ob, ob, ob, ob], t3_abbabb)

    #	  0.5000 <m,l||i,k>_abab*t3_abbabb(a,b,c,m,j,l)
    triples_res +=  0.500000000000000 * einsum('mlik,abcmjl->abcijk', g_abab[oa, ob, oa, ob], t3_abbabb)

    #	 -0.5000 <l,m||i,k>_abab*t3_abbabb(a,b,c,l,m,j)
    triples_res += -0.500000000000000 * einsum('lmik,abclmj->abcijk', g_abab[oa, ob, oa, ob], t3_abbabb)

    #	 -0.5000 <m,l||i,j>_abab*t3_abbabb(a,b,c,m,k,l)
    triples_res += -0.500000000000000 * einsum('mlij,abcmkl->abcijk', g_abab[oa, ob, oa, ob], t3_abbabb)

    #	  0.5000 <l,m||i,j>_abab*t3_abbabb(a,b,c,l,m,k)
    triples_res +=  0.500000000000000 * einsum('lmij,abclmk->abcijk', g_abab[oa, ob, oa, ob], t3_abbabb)

    #	 -1.0000 P(j,k)<a,l||d,k>_abab*t3_abbabb(d,b,c,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('aldk,dbcijl->abcijk', g_abab[va, ob, va, ob], t3_abbabb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,b||d,k>_abab*t3_aabaab(d,a,c,i,l,j)
    contracted_intermediate =  1.000000000000000 * einsum('lbdk,dacilj->abcijk', g_abab[oa, vb, va, ob], t3_aabaab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,b||d,k>_bbbb*t3_abbabb(a,d,c,i,j,l)
    contracted_intermediate =  1.000000000000000 * einsum('lbdk,adcijl->abcijk', g_bbbb[ob, vb, vb, ob], t3_abbabb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <l,a||d,i>_aaaa*t3_abbabb(d,b,c,l,k,j)
    triples_res += -1.000000000000000 * einsum('ladi,dbclkj->abcijk', g_aaaa[oa, va, va, oa], t3_abbabb)

    #	  1.0000 <a,l||i,d>_abab*t3_bbbbbb(d,b,c,j,k,l)
    triples_res +=  1.000000000000000 * einsum('alid,dbcjkl->abcijk', g_abab[va, ob, oa, vb], t3_bbbbbb)

    #	  1.0000 <l,b||i,d>_abab*t3_abbabb(a,d,c,l,k,j)
    triples_res +=  1.000000000000000 * einsum('lbid,adclkj->abcijk', g_abab[oa, vb, oa, vb], t3_abbabb)

    #	 -1.0000 P(j,k)<l,c||d,k>_abab*t3_aabaab(d,a,b,i,l,j)
    contracted_intermediate = -1.000000000000000 * einsum('lcdk,dabilj->abcijk', g_abab[oa, vb, va, ob], t3_aabaab)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<l,c||d,k>_bbbb*t3_abbabb(a,d,b,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcdk,adbijl->abcijk', g_bbbb[ob, vb, vb, ob], t3_abbabb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <l,c||i,d>_abab*t3_abbabb(a,d,b,l,k,j)
    triples_res += -1.000000000000000 * einsum('lcid,adblkj->abcijk', g_abab[oa, vb, oa, vb], t3_abbabb)

    #	  0.5000 P(b,c)<a,b||d,e>_abab*t3_abbabb(d,e,c,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('abde,decijk->abcijk', g_abab[va, vb, va, vb], t3_abbabb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.5000 P(b,c)<a,b||e,d>_abab*t3_abbabb(e,d,c,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('abed,edcijk->abcijk', g_abab[va, vb, va, vb], t3_abbabb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -0.5000 <b,c||d,e>_bbbb*t3_abbabb(a,e,d,i,j,k)
    triples_res += -0.500000000000000 * einsum('bcde,aedijk->abcijk', g_bbbb[vb, vb, vb, vb], t3_abbabb)

    #	 -1.0000 <m,l||j,k>_bbbb*t2_abab(a,c,i,m)*t1_bb(b,l)
    triples_res += -1.000000000000000 * einsum('mljk,acim,bl->abcijk', g_bbbb[ob, ob, ob, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,m||i,k>_abab*t1_aa(a,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmik,al,bcjm->abcijk', g_abab[oa, ob, oa, ob], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <m,l||i,k>_abab*t2_abab(a,c,m,j)*t1_bb(b,l)
    triples_res += -1.000000000000000 * einsum('mlik,acmj,bl->abcijk', g_abab[oa, ob, oa, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <m,l||j,k>_bbbb*t2_abab(a,b,i,m)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mljk,abim,cl->abcijk', g_bbbb[ob, ob, ob, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||i,k>_abab*t2_abab(a,b,m,j)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mlik,abmj,cl->abcijk', g_abab[oa, ob, oa, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,m||i,j>_abab*t1_aa(a,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmij,al,bckm->abcijk', g_abab[oa, ob, oa, ob], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <m,l||i,j>_abab*t2_abab(a,c,m,k)*t1_bb(b,l)
    triples_res +=  1.000000000000000 * einsum('mlij,acmk,bl->abcijk', g_abab[oa, ob, oa, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <m,l||i,j>_abab*t2_abab(a,b,m,k)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('mlij,abmk,cl->abcijk', g_abab[oa, ob, oa, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,b||d,k>_bbbb*t2_abab(a,c,i,l)*t1_bb(d,j)
    triples_res +=  1.000000000000000 * einsum('lbdk,acil,dj->abcijk', g_bbbb[ob, vb, vb, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <a,l||d,k>_abab*t2_bbbb(b,c,j,l)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('aldk,bcjl,di->abcijk', g_abab[va, ob, va, ob], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,b||d,k>_abab*t2_abab(a,c,l,j)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('lbdk,aclj,di->abcijk', g_abab[oa, vb, va, ob], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(j,k)*P(b,c)<a,l||d,k>_abab*t1_bb(b,l)*t2_abab(d,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('aldk,bl,dcij->abcijk', g_abab[va, ob, va, ob], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	 -1.0000 <l,b||d,j>_bbbb*t2_abab(a,c,i,l)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('lbdj,acil,dk->abcijk', g_bbbb[ob, vb, vb, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <a,l||d,j>_abab*t2_bbbb(b,c,k,l)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('aldj,bckl,di->abcijk', g_abab[va, ob, va, ob], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,b||d,j>_abab*t2_abab(a,c,l,k)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('lbdj,aclk,di->abcijk', g_abab[oa, vb, va, ob], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 P(j,k)<a,l||i,d>_abab*t2_bbbb(b,c,j,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('alid,bcjl,dk->abcijk', g_abab[va, ob, oa, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,b||i,d>_abab*t2_abab(a,c,l,j)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lbid,aclj,dk->abcijk', g_abab[oa, vb, oa, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(b,c)<a,l||i,d>_abab*t1_bb(b,l)*t2_bbbb(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('alid,bl,dcjk->abcijk', g_abab[va, ob, oa, vb], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  1.0000 P(j,k)<l,b||d,k>_abab*t1_aa(a,l)*t2_abab(d,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lbdk,al,dcij->abcijk', g_abab[oa, vb, va, ob], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,b||d,k>_bbbb*t2_abab(a,d,i,j)*t1_bb(c,l)
    contracted_intermediate =  1.000000000000000 * einsum('lbdk,adij,cl->abcijk', g_bbbb[ob, vb, vb, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <l,b||i,d>_abab*t1_aa(a,l)*t2_bbbb(d,c,j,k)
    triples_res += -1.000000000000000 * einsum('lbid,al,dcjk->abcijk', g_abab[oa, vb, oa, vb], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,c||d,k>_bbbb*t2_abab(a,b,i,l)*t1_bb(d,j)
    triples_res += -1.000000000000000 * einsum('lcdk,abil,dj->abcijk', g_bbbb[ob, vb, vb, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,c||d,k>_abab*t2_abab(a,b,l,j)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('lcdk,ablj,di->abcijk', g_abab[oa, vb, va, ob], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 P(j,k)<l,c||d,k>_abab*t1_aa(a,l)*t2_abab(d,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('lcdk,al,dbij->abcijk', g_abab[oa, vb, va, ob], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<l,c||d,k>_bbbb*t2_abab(a,d,i,j)*t1_bb(b,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcdk,adij,bl->abcijk', g_bbbb[ob, vb, vb, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <l,c||d,j>_bbbb*t2_abab(a,b,i,l)*t1_bb(d,k)
    triples_res +=  1.000000000000000 * einsum('lcdj,abil,dk->abcijk', g_bbbb[ob, vb, vb, ob], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,c||d,j>_abab*t2_abab(a,b,l,k)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('lcdj,ablk,di->abcijk', g_abab[oa, vb, va, ob], t2_abab, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 P(j,k)<l,c||i,d>_abab*t2_abab(a,b,l,j)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lcid,ablj,dk->abcijk', g_abab[oa, vb, oa, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <l,c||i,d>_abab*t1_aa(a,l)*t2_bbbb(d,b,j,k)
    triples_res +=  1.000000000000000 * einsum('lcid,al,dbjk->abcijk', g_abab[oa, vb, oa, vb], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 P(j,k)*P(b,c)<a,b||e,d>_abab*t2_abab(e,c,i,j)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('abed,ecij,dk->abcijk', g_abab[va, vb, va, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  1.0000 P(b,c)<a,b||d,e>_abab*t2_bbbb(e,c,j,k)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('abde,ecjk,di->abcijk', g_abab[va, vb, va, vb], t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -1.0000 P(j,k)<b,c||d,e>_bbbb*t2_abab(a,e,i,j)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('bcde,aeij,dk->abcijk', g_bbbb[vb, vb, vb, vb], t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<l,m||d,k>_abab*t3_abbabb(a,b,c,i,j,m)*t1_aa(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('lmdk,abcijm,dl->abcijk', g_abab[oa, ob, va, ob], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,k>_bbbb*t3_abbabb(a,b,c,i,j,m)*t1_bb(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,abcijm,dl->abcijk', g_bbbb[ob, ob, vb, ob], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 <m,l||d,k>_bbbb*t3_abbabb(a,b,c,i,m,l)*t1_bb(d,j)
    triples_res +=  0.500000000000000 * einsum('mldk,abciml,dj->abcijk', g_bbbb[ob, ob, vb, ob], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,k>_abab*t3_abbabb(a,b,c,m,j,l)*t1_aa(d,i)
    triples_res +=  0.500000000000000 * einsum('mldk,abcmjl,di->abcijk', g_abab[oa, ob, va, ob], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,m||d,k>_abab*t3_abbabb(a,b,c,l,m,j)*t1_aa(d,i)
    triples_res += -0.500000000000000 * einsum('lmdk,abclmj,di->abcijk', g_abab[oa, ob, va, ob], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(j,k)<l,m||d,k>_abab*t1_aa(a,l)*t3_abbabb(d,b,c,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmdk,al,dbcijm->abcijk', g_abab[oa, ob, va, ob], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,k>_abab*t3_aabaab(d,a,c,i,m,j)*t1_bb(b,l)
    contracted_intermediate = -1.000000000000000 * einsum('mldk,dacimj,bl->abcijk', g_abab[oa, ob, va, ob], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,k>_bbbb*t3_abbabb(a,d,c,i,j,m)*t1_bb(b,l)
    contracted_intermediate = -1.000000000000000 * einsum('mldk,adcijm,bl->abcijk', g_bbbb[ob, ob, vb, ob], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,k>_abab*t3_aabaab(d,a,b,i,m,j)*t1_bb(c,l)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,dabimj,cl->abcijk', g_abab[oa, ob, va, ob], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,k>_bbbb*t3_abbabb(a,d,b,i,j,m)*t1_bb(c,l)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,adbijm,cl->abcijk', g_bbbb[ob, ob, vb, ob], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 <m,l||d,j>_bbbb*t3_abbabb(a,b,c,i,m,l)*t1_bb(d,k)
    triples_res += -0.500000000000000 * einsum('mldj,abciml,dk->abcijk', g_bbbb[ob, ob, vb, ob], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,j>_abab*t3_abbabb(a,b,c,m,k,l)*t1_aa(d,i)
    triples_res += -0.500000000000000 * einsum('mldj,abcmkl,di->abcijk', g_abab[oa, ob, va, ob], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,m||d,j>_abab*t3_abbabb(a,b,c,l,m,k)*t1_aa(d,i)
    triples_res +=  0.500000000000000 * einsum('lmdj,abclmk,di->abcijk', g_abab[oa, ob, va, ob], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <m,l||d,i>_aaaa*t3_abbabb(a,b,c,m,k,j)*t1_aa(d,l)
    triples_res += -1.000000000000000 * einsum('mldi,abcmkj,dl->abcijk', g_aaaa[oa, oa, va, oa], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||i,d>_abab*t3_abbabb(a,b,c,m,k,j)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('mlid,abcmkj,dl->abcijk', g_abab[oa, ob, oa, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 P(j,k)<m,l||i,d>_abab*t3_abbabb(a,b,c,m,j,l)*t1_bb(d,k)
    contracted_intermediate =  0.500000000000000 * einsum('mlid,abcmjl,dk->abcijk', g_abab[oa, ob, oa, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 P(j,k)<l,m||i,d>_abab*t3_abbabb(a,b,c,l,m,j)*t1_bb(d,k)
    contracted_intermediate = -0.500000000000000 * einsum('lmid,abclmj,dk->abcijk', g_abab[oa, ob, oa, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <m,l||d,i>_aaaa*t1_aa(a,l)*t3_abbabb(d,b,c,m,k,j)
    triples_res +=  1.000000000000000 * einsum('mldi,al,dbcmkj->abcijk', g_aaaa[oa, oa, va, oa], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,m||i,d>_abab*t1_aa(a,l)*t3_bbbbbb(d,b,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('lmid,al,dbcjkm->abcijk', g_abab[oa, ob, oa, vb], t1_aa, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <m,l||i,d>_abab*t3_abbabb(a,d,c,m,k,j)*t1_bb(b,l)
    triples_res += -1.000000000000000 * einsum('mlid,adcmkj,bl->abcijk', g_abab[oa, ob, oa, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <m,l||i,d>_abab*t3_abbabb(a,d,b,m,k,j)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mlid,adbmkj,cl->abcijk', g_abab[oa, ob, oa, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,a||d,e>_aaaa*t3_abbabb(e,b,c,i,j,k)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('lade,ebcijk,dl->abcijk', g_aaaa[oa, va, va, va], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <a,l||e,d>_abab*t3_abbabb(e,b,c,i,j,k)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('aled,ebcijk,dl->abcijk', g_abab[va, ob, va, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,b||d,e>_abab*t3_abbabb(a,e,c,i,j,k)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('lbde,aecijk,dl->abcijk', g_abab[oa, vb, va, vb], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,b||d,e>_bbbb*t3_abbabb(a,e,c,i,j,k)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('lbde,aecijk,dl->abcijk', g_bbbb[ob, vb, vb, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)<a,l||e,d>_abab*t3_abbabb(e,b,c,i,j,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('aled,ebcijl,dk->abcijk', g_abab[va, ob, va, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,b||e,d>_abab*t3_aabaab(e,a,c,i,l,j)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lbed,eacilj,dk->abcijk', g_abab[oa, vb, va, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<l,b||d,e>_bbbb*t3_abbabb(a,e,c,i,j,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lbde,aecijl,dk->abcijk', g_bbbb[ob, vb, vb, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <l,a||d,e>_aaaa*t3_abbabb(e,b,c,l,k,j)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('lade,ebclkj,di->abcijk', g_aaaa[oa, va, va, va], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <a,l||d,e>_abab*t3_bbbbbb(e,b,c,j,k,l)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('alde,ebcjkl,di->abcijk', g_abab[va, ob, va, vb], t3_bbbbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,b||d,e>_abab*t3_abbabb(a,e,c,l,k,j)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('lbde,aeclkj,di->abcijk', g_abab[oa, vb, va, vb], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 P(b,c)<a,l||d,e>_abab*t1_bb(b,l)*t3_abbabb(d,e,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('alde,bl,decijk->abcijk', g_abab[va, ob, va, vb], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -0.5000 P(b,c)<a,l||e,d>_abab*t1_bb(b,l)*t3_abbabb(e,d,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('aled,bl,edcijk->abcijk', g_abab[va, ob, va, vb], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -0.5000 <l,b||d,e>_abab*t1_aa(a,l)*t3_abbabb(d,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lbde,al,decijk->abcijk', g_abab[oa, vb, va, vb], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,b||e,d>_abab*t1_aa(a,l)*t3_abbabb(e,d,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lbed,al,edcijk->abcijk', g_abab[oa, vb, va, vb], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,b||d,e>_bbbb*t3_abbabb(a,e,d,i,j,k)*t1_bb(c,l)
    triples_res += -0.500000000000000 * einsum('lbde,aedijk,cl->abcijk', g_bbbb[ob, vb, vb, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,c||d,e>_abab*t3_abbabb(a,e,b,i,j,k)*t1_aa(d,l)
    triples_res += -1.000000000000000 * einsum('lcde,aebijk,dl->abcijk', g_abab[oa, vb, va, vb], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,c||d,e>_bbbb*t3_abbabb(a,e,b,i,j,k)*t1_bb(d,l)
    triples_res += -1.000000000000000 * einsum('lcde,aebijk,dl->abcijk', g_bbbb[ob, vb, vb, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)<l,c||e,d>_abab*t3_aabaab(e,a,b,i,l,j)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lced,eabilj,dk->abcijk', g_abab[oa, vb, va, vb], t3_aabaab, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,c||d,e>_bbbb*t3_abbabb(a,e,b,i,j,l)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,aebijl,dk->abcijk', g_bbbb[ob, vb, vb, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <l,c||d,e>_abab*t3_abbabb(a,e,b,l,k,j)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('lcde,aeblkj,di->abcijk', g_abab[oa, vb, va, vb], t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,c||d,e>_abab*t1_aa(a,l)*t3_abbabb(d,e,b,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lcde,al,debijk->abcijk', g_abab[oa, vb, va, vb], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,c||e,d>_abab*t1_aa(a,l)*t3_abbabb(e,d,b,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lced,al,edbijk->abcijk', g_abab[oa, vb, va, vb], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,c||d,e>_bbbb*t3_abbabb(a,e,d,i,j,k)*t1_bb(b,l)
    triples_res +=  0.500000000000000 * einsum('lcde,aedijk,bl->abcijk', g_bbbb[ob, vb, vb, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 P(j,k)<l,m||d,e>_abab*t3_abbabb(a,b,c,i,j,m)*t2_abab(d,e,l,k)
    contracted_intermediate = -0.500000000000000 * einsum('lmde,abcijm,delk->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 P(j,k)<l,m||e,d>_abab*t3_abbabb(a,b,c,i,j,m)*t2_abab(e,d,l,k)
    contracted_intermediate = -0.500000000000000 * einsum('lmed,abcijm,edlk->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 P(j,k)<m,l||d,e>_bbbb*t3_abbabb(a,b,c,i,j,m)*t2_bbbb(d,e,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,abcijm,dekl->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 <m,l||d,e>_aaaa*t3_abbabb(a,b,c,m,k,j)*t2_aaaa(d,e,i,l)
    triples_res +=  0.500000000000000 * einsum('mlde,abcmkj,deil->abcijk', g_aaaa[oa, oa, va, va], t3_abbabb, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_abab*t3_abbabb(a,b,c,m,k,j)*t2_abab(d,e,i,l)
    triples_res +=  0.500000000000000 * einsum('mlde,abcmkj,deil->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||e,d>_abab*t3_abbabb(a,b,c,m,k,j)*t2_abab(e,d,i,l)
    triples_res +=  0.500000000000000 * einsum('mled,abcmkj,edil->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 <m,l||d,e>_bbbb*t3_abbabb(a,b,c,i,m,l)*t2_bbbb(d,e,j,k)
    triples_res +=  0.250000000000000 * einsum('mlde,abciml,dejk->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 <m,l||d,e>_abab*t3_abbabb(a,b,c,m,j,l)*t2_abab(d,e,i,k)
    triples_res +=  0.250000000000000 * einsum('mlde,abcmjl,deik->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 <m,l||e,d>_abab*t3_abbabb(a,b,c,m,j,l)*t2_abab(e,d,i,k)
    triples_res +=  0.250000000000000 * einsum('mled,abcmjl,edik->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.2500 <l,m||d,e>_abab*t3_abbabb(a,b,c,l,m,j)*t2_abab(d,e,i,k)
    triples_res += -0.250000000000000 * einsum('lmde,abclmj,deik->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.2500 <l,m||e,d>_abab*t3_abbabb(a,b,c,l,m,j)*t2_abab(e,d,i,k)
    triples_res += -0.250000000000000 * einsum('lmed,abclmj,edik->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.2500 <m,l||d,e>_abab*t3_abbabb(a,b,c,m,k,l)*t2_abab(d,e,i,j)
    triples_res += -0.250000000000000 * einsum('mlde,abcmkl,deij->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.2500 <m,l||e,d>_abab*t3_abbabb(a,b,c,m,k,l)*t2_abab(e,d,i,j)
    triples_res += -0.250000000000000 * einsum('mled,abcmkl,edij->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 <l,m||d,e>_abab*t3_abbabb(a,b,c,l,m,k)*t2_abab(d,e,i,j)
    triples_res +=  0.250000000000000 * einsum('lmde,abclmk,deij->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 <l,m||e,d>_abab*t3_abbabb(a,b,c,l,m,k)*t2_abab(e,d,i,j)
    triples_res +=  0.250000000000000 * einsum('lmed,abclmk,edij->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,a,m,l)*t3_abbabb(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,daml,ebcijk->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||e,d>_abab*t2_abab(a,d,m,l)*t3_abbabb(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mled,adml,ebcijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,m||e,d>_abab*t2_abab(a,d,l,m)*t3_abbabb(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lmed,adlm,ebcijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||d,e>_abab*t3_abbabb(a,e,c,i,j,k)*t2_abab(d,b,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,aecijk,dbml->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,m||d,e>_abab*t3_abbabb(a,e,c,i,j,k)*t2_abab(d,b,l,m)
    triples_res += -0.500000000000000 * einsum('lmde,aecijk,dblm->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_bbbb*t3_abbabb(a,e,c,i,j,k)*t2_bbbb(d,b,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,aecijk,dbml->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 P(j,k)<l,m||e,d>_abab*t2_abab(a,d,l,k)*t3_abbabb(e,b,c,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmed,adlk,ebcijm->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,e>_aaaa*t3_aabaab(e,a,c,i,m,j)*t2_abab(d,b,l,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,eacimj,dblk->abcijk', g_aaaa[oa, oa, va, va], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||e,d>_abab*t3_aabaab(e,a,c,i,m,j)*t2_bbbb(d,b,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('mled,eacimj,dbkl->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,m||d,e>_abab*t3_abbabb(a,e,c,i,j,m)*t2_abab(d,b,l,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,aecijm,dblk->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,e>_bbbb*t3_abbabb(a,e,c,i,j,m)*t2_bbbb(d,b,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,aecijm,dbkl->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,a,i,l)*t3_abbabb(e,b,c,m,k,j)
    triples_res += -1.000000000000000 * einsum('mlde,dail,ebcmkj->abcijk', g_aaaa[oa, oa, va, va], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,m||d,e>_abab*t2_aaaa(d,a,i,l)*t3_bbbbbb(e,b,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,dail,ebcjkm->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t2_abab(a,d,i,l)*t3_abbabb(e,b,c,m,k,j)
    triples_res += -1.000000000000000 * einsum('mled,adil,ebcmkj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,i,l)*t3_bbbbbb(e,b,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,adil,ebcjkm->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <m,l||d,e>_abab*t3_abbabb(a,e,c,m,k,j)*t2_abab(d,b,i,l)
    triples_res += -1.000000000000000 * einsum('mlde,aecmkj,dbil->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||e,d>_abab*t3_aabaab(e,a,c,i,m,l)*t2_bbbb(d,b,j,k)
    triples_res += -0.500000000000000 * einsum('mled,eaciml,dbjk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,m||e,d>_abab*t3_aabaab(e,a,c,i,l,m)*t2_bbbb(d,b,j,k)
    triples_res += -0.500000000000000 * einsum('lmed,eacilm,dbjk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||d,e>_bbbb*t3_abbabb(a,e,c,i,m,l)*t2_bbbb(d,b,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,aeciml,dbjk->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||e,d>_abab*t2_abab(a,d,i,k)*t3_abbabb(e,b,c,m,j,l)
    triples_res += -0.500000000000000 * einsum('mled,adik,ebcmjl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,m||e,d>_abab*t2_abab(a,d,i,k)*t3_abbabb(e,b,c,l,m,j)
    triples_res +=  0.500000000000000 * einsum('lmed,adik,ebclmj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_bbbb*t2_abab(a,d,i,k)*t3_bbbbbb(e,b,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,adik,ebcjml->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_aaaa*t3_aabaab(e,a,c,l,m,j)*t2_abab(d,b,i,k)
    triples_res +=  0.500000000000000 * einsum('mlde,eaclmj,dbik->abcijk', g_aaaa[oa, oa, va, va], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||d,e>_abab*t3_abbabb(a,e,c,m,j,l)*t2_abab(d,b,i,k)
    triples_res += -0.500000000000000 * einsum('mlde,aecmjl,dbik->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,m||d,e>_abab*t3_abbabb(a,e,c,l,m,j)*t2_abab(d,b,i,k)
    triples_res +=  0.500000000000000 * einsum('lmde,aeclmj,dbik->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <m,l||e,d>_abab*t2_abab(a,d,i,j)*t3_abbabb(e,b,c,m,k,l)
    triples_res +=  0.500000000000000 * einsum('mled,adij,ebcmkl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,m||e,d>_abab*t2_abab(a,d,i,j)*t3_abbabb(e,b,c,l,m,k)
    triples_res += -0.500000000000000 * einsum('lmed,adij,ebclmk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_bbbb*t2_abab(a,d,i,j)*t3_bbbbbb(e,b,c,k,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,adij,ebckml->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_aaaa*t3_aabaab(e,a,c,l,m,k)*t2_abab(d,b,i,j)
    triples_res += -0.500000000000000 * einsum('mlde,eaclmk,dbij->abcijk', g_aaaa[oa, oa, va, va], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <m,l||d,e>_abab*t3_abbabb(a,e,c,m,k,l)*t2_abab(d,b,i,j)
    triples_res +=  0.500000000000000 * einsum('mlde,aecmkl,dbij->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,m||d,e>_abab*t3_abbabb(a,e,c,l,m,k)*t2_abab(d,b,i,j)
    triples_res += -0.500000000000000 * einsum('lmde,aeclmk,dbij->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <m,l||d,e>_abab*t3_abbabb(a,e,b,i,j,k)*t2_abab(d,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,aebijk,dcml->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,m||d,e>_abab*t3_abbabb(a,e,b,i,j,k)*t2_abab(d,c,l,m)
    triples_res +=  0.500000000000000 * einsum('lmde,aebijk,dclm->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_bbbb*t3_abbabb(a,e,b,i,j,k)*t2_bbbb(d,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,aebijk,dcml->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 P(j,k)<m,l||d,e>_aaaa*t3_aabaab(e,a,b,i,m,j)*t2_abab(d,c,l,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,eabimj,dclk->abcijk', g_aaaa[oa, oa, va, va], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||e,d>_abab*t3_aabaab(e,a,b,i,m,j)*t2_bbbb(d,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('mled,eabimj,dckl->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<l,m||d,e>_abab*t3_abbabb(a,e,b,i,j,m)*t2_abab(d,c,l,k)
    contracted_intermediate = -1.000000000000000 * einsum('lmde,aebijm,dclk->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,e>_bbbb*t3_abbabb(a,e,b,i,j,m)*t2_bbbb(d,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,aebijm,dckl->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <m,l||d,e>_abab*t3_abbabb(a,e,b,m,k,j)*t2_abab(d,c,i,l)
    triples_res +=  1.000000000000000 * einsum('mlde,aebmkj,dcil->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <m,l||e,d>_abab*t3_aabaab(e,a,b,i,m,l)*t2_bbbb(d,c,j,k)
    triples_res +=  0.500000000000000 * einsum('mled,eabiml,dcjk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,m||e,d>_abab*t3_aabaab(e,a,b,i,l,m)*t2_bbbb(d,c,j,k)
    triples_res +=  0.500000000000000 * einsum('lmed,eabilm,dcjk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <m,l||d,e>_bbbb*t3_abbabb(a,e,b,i,m,l)*t2_bbbb(d,c,j,k)
    triples_res +=  0.500000000000000 * einsum('mlde,aebiml,dcjk->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||d,e>_aaaa*t3_aabaab(e,a,b,l,m,j)*t2_abab(d,c,i,k)
    triples_res += -0.500000000000000 * einsum('mlde,eablmj,dcik->abcijk', g_aaaa[oa, oa, va, va], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <m,l||d,e>_abab*t3_abbabb(a,e,b,m,j,l)*t2_abab(d,c,i,k)
    triples_res +=  0.500000000000000 * einsum('mlde,aebmjl,dcik->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,m||d,e>_abab*t3_abbabb(a,e,b,l,m,j)*t2_abab(d,c,i,k)
    triples_res += -0.500000000000000 * einsum('lmde,aeblmj,dcik->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <m,l||d,e>_aaaa*t3_aabaab(e,a,b,l,m,k)*t2_abab(d,c,i,j)
    triples_res +=  0.500000000000000 * einsum('mlde,eablmk,dcij->abcijk', g_aaaa[oa, oa, va, va], t3_aabaab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||d,e>_abab*t3_abbabb(a,e,b,m,k,l)*t2_abab(d,c,i,j)
    triples_res += -0.500000000000000 * einsum('mlde,aebmkl,dcij->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,m||d,e>_abab*t3_abbabb(a,e,b,l,m,k)*t2_abab(d,c,i,j)
    triples_res +=  0.500000000000000 * einsum('lmde,aeblmk,dcij->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.2500 P(b,c)<m,l||d,e>_abab*t2_abab(a,b,m,l)*t3_abbabb(d,e,c,i,j,k)
    contracted_intermediate =  0.250000000000000 * einsum('mlde,abml,decijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.2500 P(b,c)<m,l||e,d>_abab*t2_abab(a,b,m,l)*t3_abbabb(e,d,c,i,j,k)
    contracted_intermediate =  0.250000000000000 * einsum('mled,abml,edcijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.2500 P(b,c)<l,m||d,e>_abab*t2_abab(a,b,l,m)*t3_abbabb(d,e,c,i,j,k)
    contracted_intermediate =  0.250000000000000 * einsum('lmde,ablm,decijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.2500 P(b,c)<l,m||e,d>_abab*t2_abab(a,b,l,m)*t3_abbabb(e,d,c,i,j,k)
    contracted_intermediate =  0.250000000000000 * einsum('lmed,ablm,edcijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -0.5000 P(j,k)*P(b,c)<m,l||d,e>_aaaa*t2_abab(a,b,l,k)*t3_aabaab(d,e,c,i,m,j)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,ablk,decimj->abcijk', g_aaaa[oa, oa, va, va], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	 -0.5000 P(j,k)*P(b,c)<l,m||d,e>_abab*t2_abab(a,b,l,k)*t3_abbabb(d,e,c,i,j,m)
    contracted_intermediate = -0.500000000000000 * einsum('lmde,ablk,decijm->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	 -0.5000 P(j,k)*P(b,c)<l,m||e,d>_abab*t2_abab(a,b,l,k)*t3_abbabb(e,d,c,i,j,m)
    contracted_intermediate = -0.500000000000000 * einsum('lmed,ablk,edcijm->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  0.5000 P(b,c)<m,l||d,e>_abab*t2_abab(a,b,i,l)*t3_abbabb(d,e,c,m,k,j)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,abil,decmkj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.5000 P(b,c)<m,l||e,d>_abab*t2_abab(a,b,i,l)*t3_abbabb(e,d,c,m,k,j)
    contracted_intermediate =  0.500000000000000 * einsum('mled,abil,edcmkj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -0.5000 P(b,c)<m,l||d,e>_bbbb*t2_abab(a,b,i,l)*t3_bbbbbb(d,e,c,j,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,abil,decjkm->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -0.2500 <m,l||d,e>_bbbb*t3_abbabb(a,e,d,i,j,k)*t2_bbbb(b,c,m,l)
    triples_res += -0.250000000000000 * einsum('mlde,aedijk,bcml->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 P(j,k)<m,l||d,e>_abab*t3_aabaab(d,a,e,i,m,j)*t2_bbbb(b,c,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,daeimj,bckl->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 P(j,k)<m,l||e,d>_abab*t3_aabaab(a,e,d,i,m,j)*t2_bbbb(b,c,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('mled,aedimj,bckl->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 P(j,k)<m,l||d,e>_bbbb*t3_abbabb(a,e,d,i,j,m)*t2_bbbb(b,c,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,aedijm,bckl->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <l,m||d,k>_abab*t2_abab(a,c,i,m)*t2_abab(d,b,l,j)
    triples_res += -1.000000000000000 * einsum('lmdk,acim,dblj->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,k>_bbbb*t2_abab(a,c,i,m)*t2_bbbb(d,b,j,l)
    triples_res += -1.000000000000000 * einsum('mldk,acim,dbjl->abcijk', g_bbbb[ob, ob, vb, ob], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,m||d,k>_abab*t2_aaaa(d,a,i,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,dail,bcjm->abcijk', g_abab[oa, ob, va, ob], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <m,l||d,k>_bbbb*t2_abab(a,d,i,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,adil,bcjm->abcijk', g_bbbb[ob, ob, vb, ob], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <m,l||d,k>_abab*t2_abab(a,c,m,j)*t2_abab(d,b,i,l)
    triples_res += -1.000000000000000 * einsum('mldk,acmj,dbil->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 P(j,k)<m,l||d,k>_bbbb*t2_abab(a,d,i,j)*t2_bbbb(b,c,m,l)
    contracted_intermediate =  0.500000000000000 * einsum('mldk,adij,bcml->abcijk', g_bbbb[ob, ob, vb, ob], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 P(j,k)<m,l||d,k>_abab*t2_abab(a,c,m,l)*t2_abab(d,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('mldk,acml,dbij->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 P(j,k)<l,m||d,k>_abab*t2_abab(a,c,l,m)*t2_abab(d,b,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('lmdk,aclm,dbij->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <l,m||d,k>_abab*t2_abab(a,b,i,m)*t2_abab(d,c,l,j)
    triples_res +=  1.000000000000000 * einsum('lmdk,abim,dclj->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,k>_bbbb*t2_abab(a,b,i,m)*t2_bbbb(d,c,j,l)
    triples_res +=  1.000000000000000 * einsum('mldk,abim,dcjl->abcijk', g_bbbb[ob, ob, vb, ob], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,k>_abab*t2_abab(a,b,m,j)*t2_abab(d,c,i,l)
    triples_res +=  1.000000000000000 * einsum('mldk,abmj,dcil->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 P(j,k)<m,l||d,k>_abab*t2_abab(a,b,m,l)*t2_abab(d,c,i,j)
    contracted_intermediate = -0.500000000000000 * einsum('mldk,abml,dcij->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 P(j,k)<l,m||d,k>_abab*t2_abab(a,b,l,m)*t2_abab(d,c,i,j)
    contracted_intermediate = -0.500000000000000 * einsum('lmdk,ablm,dcij->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <l,m||d,j>_abab*t2_abab(a,c,i,m)*t2_abab(d,b,l,k)
    triples_res +=  1.000000000000000 * einsum('lmdj,acim,dblk->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,j>_bbbb*t2_abab(a,c,i,m)*t2_bbbb(d,b,k,l)
    triples_res +=  1.000000000000000 * einsum('mldj,acim,dbkl->abcijk', g_bbbb[ob, ob, vb, ob], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,m||d,j>_abab*t2_aaaa(d,a,i,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmdj,dail,bckm->abcijk', g_abab[oa, ob, va, ob], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <m,l||d,j>_bbbb*t2_abab(a,d,i,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mldj,adil,bckm->abcijk', g_bbbb[ob, ob, vb, ob], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <m,l||d,j>_abab*t2_abab(a,c,m,k)*t2_abab(d,b,i,l)
    triples_res +=  1.000000000000000 * einsum('mldj,acmk,dbil->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,m||d,j>_abab*t2_abab(a,b,i,m)*t2_abab(d,c,l,k)
    triples_res += -1.000000000000000 * einsum('lmdj,abim,dclk->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,j>_bbbb*t2_abab(a,b,i,m)*t2_bbbb(d,c,k,l)
    triples_res += -1.000000000000000 * einsum('mldj,abim,dckl->abcijk', g_bbbb[ob, ob, vb, ob], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,j>_abab*t2_abab(a,b,m,k)*t2_abab(d,c,i,l)
    triples_res += -1.000000000000000 * einsum('mldj,abmk,dcil->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(j,k)<l,m||i,d>_abab*t2_abab(a,d,l,k)*t2_bbbb(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmid,adlk,bcjm->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,i>_aaaa*t2_abab(a,c,m,j)*t2_abab(d,b,l,k)
    contracted_intermediate = -1.000000000000000 * einsum('mldi,acmj,dblk->abcijk', g_aaaa[oa, oa, va, oa], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||i,d>_abab*t2_abab(a,c,m,j)*t2_bbbb(d,b,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('mlid,acmj,dbkl->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 <m,l||i,d>_abab*t2_abab(a,c,m,l)*t2_bbbb(d,b,j,k)
    triples_res += -0.500000000000000 * einsum('mlid,acml,dbjk->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <l,m||i,d>_abab*t2_abab(a,c,l,m)*t2_bbbb(d,b,j,k)
    triples_res += -0.500000000000000 * einsum('lmid,aclm,dbjk->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(j,k)<m,l||d,i>_aaaa*t2_abab(a,b,m,j)*t2_abab(d,c,l,k)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,abmj,dclk->abcijk', g_aaaa[oa, oa, va, oa], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||i,d>_abab*t2_abab(a,b,m,j)*t2_bbbb(d,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlid,abmj,dckl->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 <m,l||i,d>_abab*t2_abab(a,b,m,l)*t2_bbbb(d,c,j,k)
    triples_res +=  0.500000000000000 * einsum('mlid,abml,dcjk->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,m||i,d>_abab*t2_abab(a,b,l,m)*t2_bbbb(d,c,j,k)
    triples_res +=  0.500000000000000 * einsum('lmid,ablm,dcjk->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,b||d,e>_bbbb*t2_abab(a,c,i,l)*t2_bbbb(d,e,j,k)
    triples_res +=  0.500000000000000 * einsum('lbde,acil,dejk->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <a,l||d,e>_abab*t2_bbbb(b,c,j,l)*t2_abab(d,e,i,k)
    triples_res += -0.500000000000000 * einsum('alde,bcjl,deik->abcijk', g_abab[va, ob, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <a,l||e,d>_abab*t2_bbbb(b,c,j,l)*t2_abab(e,d,i,k)
    triples_res += -0.500000000000000 * einsum('aled,bcjl,edik->abcijk', g_abab[va, ob, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,b||d,e>_abab*t2_abab(a,c,l,j)*t2_abab(d,e,i,k)
    triples_res +=  0.500000000000000 * einsum('lbde,aclj,deik->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,b||e,d>_abab*t2_abab(a,c,l,j)*t2_abab(e,d,i,k)
    triples_res +=  0.500000000000000 * einsum('lbed,aclj,edik->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <a,l||d,e>_abab*t2_bbbb(b,c,k,l)*t2_abab(d,e,i,j)
    triples_res +=  0.500000000000000 * einsum('alde,bckl,deij->abcijk', g_abab[va, ob, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <a,l||e,d>_abab*t2_bbbb(b,c,k,l)*t2_abab(e,d,i,j)
    triples_res +=  0.500000000000000 * einsum('aled,bckl,edij->abcijk', g_abab[va, ob, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,b||d,e>_abab*t2_abab(a,c,l,k)*t2_abab(d,e,i,j)
    triples_res += -0.500000000000000 * einsum('lbde,aclk,deij->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,b||e,d>_abab*t2_abab(a,c,l,k)*t2_abab(e,d,i,j)
    triples_res += -0.500000000000000 * einsum('lbed,aclk,edij->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)<l,a||d,e>_aaaa*t2_abab(d,b,l,k)*t2_abab(e,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('lade,dblk,ecij->abcijk', g_aaaa[oa, va, va, va], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<a,l||e,d>_abab*t2_bbbb(d,b,k,l)*t2_abab(e,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('aled,dbkl,ecij->abcijk', g_abab[va, ob, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,b||e,d>_abab*t2_abab(a,d,l,k)*t2_abab(e,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lbed,adlk,ecij->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <a,l||d,e>_abab*t2_abab(d,b,i,l)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('alde,dbil,ecjk->abcijk', g_abab[va, ob, va, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,b||d,e>_abab*t2_aaaa(d,a,i,l)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('lbde,dail,ecjk->abcijk', g_abab[oa, vb, va, vb], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,b||d,e>_bbbb*t2_abab(a,d,i,l)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lbde,adil,ecjk->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <a,l||e,d>_abab*t2_bbbb(d,b,j,k)*t2_abab(e,c,i,l)
    triples_res +=  1.000000000000000 * einsum('aled,dbjk,ecil->abcijk', g_abab[va, ob, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,a||d,e>_aaaa*t2_abab(d,b,i,k)*t2_abab(e,c,l,j)
    triples_res +=  1.000000000000000 * einsum('lade,dbik,eclj->abcijk', g_aaaa[oa, va, va, va], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <a,l||d,e>_abab*t2_abab(d,b,i,k)*t2_bbbb(e,c,j,l)
    triples_res +=  1.000000000000000 * einsum('alde,dbik,ecjl->abcijk', g_abab[va, ob, va, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,b||e,d>_abab*t2_abab(a,d,i,k)*t2_abab(e,c,l,j)
    triples_res += -1.000000000000000 * einsum('lbed,adik,eclj->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,b||d,e>_bbbb*t2_abab(a,d,i,k)*t2_bbbb(e,c,j,l)
    triples_res += -1.000000000000000 * einsum('lbde,adik,ecjl->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,a||d,e>_aaaa*t2_abab(d,b,i,j)*t2_abab(e,c,l,k)
    triples_res += -1.000000000000000 * einsum('lade,dbij,eclk->abcijk', g_aaaa[oa, va, va, va], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <a,l||d,e>_abab*t2_abab(d,b,i,j)*t2_bbbb(e,c,k,l)
    triples_res += -1.000000000000000 * einsum('alde,dbij,eckl->abcijk', g_abab[va, ob, va, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,b||e,d>_abab*t2_abab(a,d,i,j)*t2_abab(e,c,l,k)
    triples_res +=  1.000000000000000 * einsum('lbed,adij,eclk->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,b||d,e>_bbbb*t2_abab(a,d,i,j)*t2_bbbb(e,c,k,l)
    triples_res +=  1.000000000000000 * einsum('lbde,adij,eckl->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,c||d,e>_bbbb*t2_abab(a,b,i,l)*t2_bbbb(d,e,j,k)
    triples_res += -0.500000000000000 * einsum('lcde,abil,dejk->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,c||d,e>_abab*t2_abab(a,b,l,j)*t2_abab(d,e,i,k)
    triples_res += -0.500000000000000 * einsum('lcde,ablj,deik->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,c||e,d>_abab*t2_abab(a,b,l,j)*t2_abab(e,d,i,k)
    triples_res += -0.500000000000000 * einsum('lced,ablj,edik->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,c||d,e>_abab*t2_abab(a,b,l,k)*t2_abab(d,e,i,j)
    triples_res +=  0.500000000000000 * einsum('lcde,ablk,deij->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <l,c||e,d>_abab*t2_abab(a,b,l,k)*t2_abab(e,d,i,j)
    triples_res +=  0.500000000000000 * einsum('lced,ablk,edij->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)<l,c||e,d>_abab*t2_abab(a,d,l,k)*t2_abab(e,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('lced,adlk,ebij->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <l,c||d,e>_abab*t2_aaaa(d,a,i,l)*t2_bbbb(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dail,ebjk->abcijk', g_abab[oa, vb, va, vb], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <l,c||d,e>_bbbb*t2_abab(a,d,i,l)*t2_bbbb(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('lcde,adil,ebjk->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,c||e,d>_abab*t2_abab(a,d,i,k)*t2_abab(e,b,l,j)
    triples_res +=  1.000000000000000 * einsum('lced,adik,eblj->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,c||d,e>_bbbb*t2_abab(a,d,i,k)*t2_bbbb(e,b,j,l)
    triples_res +=  1.000000000000000 * einsum('lcde,adik,ebjl->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,c||e,d>_abab*t2_abab(a,d,i,j)*t2_abab(e,b,l,k)
    triples_res += -1.000000000000000 * einsum('lced,adij,eblk->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,c||d,e>_bbbb*t2_abab(a,d,i,j)*t2_bbbb(e,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcde,adij,ebkl->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,m||d,e>_abab*t2_abab(a,c,i,m)*t2_bbbb(e,b,j,k)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('lmde,acim,ebjk,dl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,c,i,m)*t2_bbbb(e,b,j,k)*t1_bb(d,l)
    triples_res += -1.000000000000000 * einsum('mlde,acim,ebjk,dl->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	 -1.0000 <l,m||d,e>_abab*t2_abab(a,e,i,k)*t2_bbbb(b,c,j,m)*t1_aa(d,l)
    triples_res += -1.000000000000000 * einsum('lmde,aeik,bcjm,dl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(a,e,i,k)*t2_bbbb(b,c,j,m)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('mlde,aeik,bcjm,dl->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_aaaa*t2_abab(a,c,m,j)*t2_abab(e,b,i,k)*t1_aa(d,l)
    triples_res += -1.000000000000000 * einsum('mlde,acmj,ebik,dl->abcijk', g_aaaa[oa, oa, va, va], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <m,l||e,d>_abab*t2_abab(a,c,m,j)*t2_abab(e,b,i,k)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('mled,acmj,ebik,dl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <l,m||d,e>_abab*t2_abab(a,e,i,j)*t2_bbbb(b,c,k,m)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('lmde,aeij,bckm,dl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,e,i,j)*t2_bbbb(b,c,k,m)*t1_bb(d,l)
    triples_res += -1.000000000000000 * einsum('mlde,aeij,bckm,dl->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_aaaa*t2_abab(a,c,m,k)*t2_abab(e,b,i,j)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('mlde,acmk,ebij,dl->abcijk', g_aaaa[oa, oa, va, va], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t2_abab(a,c,m,k)*t2_abab(e,b,i,j)*t1_bb(d,l)
    triples_res += -1.000000000000000 * einsum('mled,acmk,ebij,dl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <l,m||d,e>_abab*t2_abab(a,b,i,m)*t2_bbbb(e,c,j,k)*t1_aa(d,l)
    triples_res += -1.000000000000000 * einsum('lmde,abim,ecjk,dl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(a,b,i,m)*t2_bbbb(e,c,j,k)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('mlde,abim,ecjk,dl->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_aaaa*t2_abab(a,b,m,j)*t2_abab(e,c,i,k)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('mlde,abmj,ecik,dl->abcijk', g_aaaa[oa, oa, va, va], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t2_abab(a,b,m,j)*t2_abab(e,c,i,k)*t1_bb(d,l)
    triples_res += -1.000000000000000 * einsum('mled,abmj,ecik,dl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_aaaa*t2_abab(a,b,m,k)*t2_abab(e,c,i,j)*t1_aa(d,l)
    triples_res += -1.000000000000000 * einsum('mlde,abmk,ecij,dl->abcijk', g_aaaa[oa, oa, va, va], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <m,l||e,d>_abab*t2_abab(a,b,m,k)*t2_abab(e,c,i,j)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('mled,abmk,ecij,dl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <l,m||e,d>_abab*t2_abab(a,c,i,m)*t2_abab(e,b,l,j)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('lmed,acim,eblj,dk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(a,c,i,m)*t2_bbbb(e,b,j,l)*t1_bb(d,k)
    triples_res +=  1.000000000000000 * einsum('mlde,acim,ebjl,dk->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  1.0000 <l,m||e,d>_abab*t2_aaaa(e,a,i,l)*t2_bbbb(b,c,j,m)*t1_bb(d,k)
    triples_res +=  1.000000000000000 * einsum('lmed,eail,bcjm,dk->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(b,c,j,m)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('mlde,aeil,bcjm,dk->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t2_abab(a,c,m,j)*t2_abab(e,b,i,l)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('mled,acmj,ebil,dk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -0.5000 P(j,k)<m,l||d,e>_bbbb*t2_abab(a,e,i,j)*t2_bbbb(b,c,m,l)*t1_bb(d,k)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,aeij,bcml,dk->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 P(j,k)<m,l||e,d>_abab*t2_abab(a,c,m,l)*t2_abab(e,b,i,j)*t1_bb(d,k)
    contracted_intermediate =  0.500000000000000 * einsum('mled,acml,ebij,dk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 P(j,k)<l,m||e,d>_abab*t2_abab(a,c,l,m)*t2_abab(e,b,i,j)*t1_bb(d,k)
    contracted_intermediate =  0.500000000000000 * einsum('lmed,aclm,ebij,dk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <l,m||e,d>_abab*t2_abab(a,b,i,m)*t2_abab(e,c,l,j)*t1_bb(d,k)
    triples_res +=  1.000000000000000 * einsum('lmed,abim,eclj,dk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,b,i,m)*t2_bbbb(e,c,j,l)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('mlde,abim,ecjl,dk->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  1.0000 <m,l||e,d>_abab*t2_abab(a,b,m,j)*t2_abab(e,c,i,l)*t1_bb(d,k)
    triples_res +=  1.000000000000000 * einsum('mled,abmj,ecil,dk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -0.5000 P(j,k)<m,l||e,d>_abab*t2_abab(a,b,m,l)*t2_abab(e,c,i,j)*t1_bb(d,k)
    contracted_intermediate = -0.500000000000000 * einsum('mled,abml,ecij,dk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 P(j,k)<l,m||e,d>_abab*t2_abab(a,b,l,m)*t2_abab(e,c,i,j)*t1_bb(d,k)
    contracted_intermediate = -0.500000000000000 * einsum('lmed,ablm,ecij,dk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <l,m||e,d>_abab*t2_abab(a,c,i,m)*t2_abab(e,b,l,k)*t1_bb(d,j)
    triples_res +=  1.000000000000000 * einsum('lmed,acim,eblk,dj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,c,i,m)*t2_bbbb(e,b,k,l)*t1_bb(d,j)
    triples_res += -1.000000000000000 * einsum('mlde,acim,ebkl,dj->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	 -1.0000 <l,m||e,d>_abab*t2_aaaa(e,a,i,l)*t2_bbbb(b,c,k,m)*t1_bb(d,j)
    triples_res += -1.000000000000000 * einsum('lmed,eail,bckm,dj->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(a,e,i,l)*t2_bbbb(b,c,k,m)*t1_bb(d,j)
    triples_res +=  1.000000000000000 * einsum('mlde,aeil,bckm,dj->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <m,l||e,d>_abab*t2_abab(a,c,m,k)*t2_abab(e,b,i,l)*t1_bb(d,j)
    triples_res +=  1.000000000000000 * einsum('mled,acmk,ebil,dj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <l,m||e,d>_abab*t2_abab(a,b,i,m)*t2_abab(e,c,l,k)*t1_bb(d,j)
    triples_res += -1.000000000000000 * einsum('lmed,abim,eclk,dj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(a,b,i,m)*t2_bbbb(e,c,k,l)*t1_bb(d,j)
    triples_res +=  1.000000000000000 * einsum('mlde,abim,eckl,dj->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t2_abab(a,b,m,k)*t2_abab(e,c,i,l)*t1_bb(d,j)
    triples_res += -1.000000000000000 * einsum('mled,abmk,ecil,dj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 P(j,k)<l,m||d,e>_abab*t2_abab(a,e,l,k)*t2_bbbb(b,c,j,m)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,aelk,bcjm,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,e>_aaaa*t2_abab(a,c,m,j)*t2_abab(e,b,l,k)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,acmj,eblk,di->abcijk', g_aaaa[oa, oa, va, va], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,e>_abab*t2_abab(a,c,m,j)*t2_bbbb(e,b,k,l)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,acmj,ebkl,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 <m,l||d,e>_abab*t2_abab(a,c,m,l)*t2_bbbb(e,b,j,k)*t1_aa(d,i)
    triples_res += -0.500000000000000 * einsum('mlde,acml,ebjk,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -0.5000 <l,m||d,e>_abab*t2_abab(a,c,l,m)*t2_bbbb(e,b,j,k)*t1_aa(d,i)
    triples_res += -0.500000000000000 * einsum('lmde,aclm,ebjk,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)<m,l||d,e>_aaaa*t2_abab(a,b,m,j)*t2_abab(e,c,l,k)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,abmj,eclk,di->abcijk', g_aaaa[oa, oa, va, va], t2_abab, t2_abab, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,e>_abab*t2_abab(a,b,m,j)*t2_bbbb(e,c,k,l)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,abmj,eckl,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 <m,l||d,e>_abab*t2_abab(a,b,m,l)*t2_bbbb(e,c,j,k)*t1_aa(d,i)
    triples_res +=  0.500000000000000 * einsum('mlde,abml,ecjk,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  0.5000 <l,m||d,e>_abab*t2_abab(a,b,l,m)*t2_bbbb(e,c,j,k)*t1_aa(d,i)
    triples_res +=  0.500000000000000 * einsum('lmde,ablm,ecjk,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_bbbb*t2_abab(a,c,i,m)*t1_bb(b,l)*t2_bbbb(d,e,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,acim,bl,dejk->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  0.5000 <l,m||d,e>_abab*t1_aa(a,l)*t2_bbbb(b,c,j,m)*t2_abab(d,e,i,k)
    triples_res +=  0.500000000000000 * einsum('lmde,al,bcjm,deik->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	  0.5000 <l,m||e,d>_abab*t1_aa(a,l)*t2_bbbb(b,c,j,m)*t2_abab(e,d,i,k)
    triples_res +=  0.500000000000000 * einsum('lmed,al,bcjm,edik->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_abab*t2_abab(a,c,m,j)*t1_bb(b,l)*t2_abab(d,e,i,k)
    triples_res += -0.500000000000000 * einsum('mlde,acmj,bl,deik->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -0.5000 <m,l||e,d>_abab*t2_abab(a,c,m,j)*t1_bb(b,l)*t2_abab(e,d,i,k)
    triples_res += -0.500000000000000 * einsum('mled,acmj,bl,edik->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -0.5000 <l,m||d,e>_abab*t1_aa(a,l)*t2_bbbb(b,c,k,m)*t2_abab(d,e,i,j)
    triples_res += -0.500000000000000 * einsum('lmde,al,bckm,deij->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -0.5000 <l,m||e,d>_abab*t1_aa(a,l)*t2_bbbb(b,c,k,m)*t2_abab(e,d,i,j)
    triples_res += -0.500000000000000 * einsum('lmed,al,bckm,edij->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_abab*t2_abab(a,c,m,k)*t1_bb(b,l)*t2_abab(d,e,i,j)
    triples_res +=  0.500000000000000 * einsum('mlde,acmk,bl,deij->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  0.5000 <m,l||e,d>_abab*t2_abab(a,c,m,k)*t1_bb(b,l)*t2_abab(e,d,i,j)
    triples_res +=  0.500000000000000 * einsum('mled,acmk,bl,edij->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 P(j,k)<m,l||d,e>_aaaa*t1_aa(a,l)*t2_abab(d,b,m,k)*t2_abab(e,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,al,dbmk,ecij->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<l,m||e,d>_abab*t1_aa(a,l)*t2_bbbb(d,b,k,m)*t2_abab(e,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('lmed,al,dbkm,ecij->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||e,d>_abab*t2_abab(a,d,m,k)*t1_bb(b,l)*t2_abab(e,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('mled,admk,bl,ecij->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <l,m||d,e>_abab*t1_aa(a,l)*t2_abab(d,b,i,m)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lmde,al,dbim,ecjk->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_abab*t2_aaaa(d,a,i,m)*t1_bb(b,l)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,daim,bl,ecjk->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,i,m)*t1_bb(b,l)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,adim,bl,ecjk->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_bbbb(d,b,j,k)*t2_abab(e,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmed,al,dbjk,ecim->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_abab(d,b,i,k)*t2_abab(e,c,m,j)
    triples_res += -1.000000000000000 * einsum('mlde,al,dbik,ecmj->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -1.0000 <l,m||d,e>_abab*t1_aa(a,l)*t2_abab(d,b,i,k)*t2_bbbb(e,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmde,al,dbik,ecjm->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	  1.0000 <m,l||e,d>_abab*t2_abab(a,d,i,k)*t1_bb(b,l)*t2_abab(e,c,m,j)
    triples_res +=  1.000000000000000 * einsum('mled,adik,bl,ecmj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,i,k)*t1_bb(b,l)*t2_bbbb(e,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,adik,bl,ecjm->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_abab(d,b,i,j)*t2_abab(e,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,al,dbij,ecmk->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	  1.0000 <l,m||d,e>_abab*t1_aa(a,l)*t2_abab(d,b,i,j)*t2_bbbb(e,c,k,m)
    triples_res +=  1.000000000000000 * einsum('lmde,al,dbij,eckm->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t2_abab(a,d,i,j)*t1_bb(b,l)*t2_abab(e,c,m,k)
    triples_res += -1.000000000000000 * einsum('mled,adij,bl,ecmk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,i,j)*t1_bb(b,l)*t2_bbbb(e,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,adij,bl,eckm->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_bbbb*t2_abab(a,b,i,m)*t1_bb(c,l)*t2_bbbb(d,e,j,k)
    triples_res +=  0.500000000000000 * einsum('mlde,abim,cl,dejk->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  0.5000 <m,l||d,e>_abab*t2_abab(a,b,m,j)*t1_bb(c,l)*t2_abab(d,e,i,k)
    triples_res +=  0.500000000000000 * einsum('mlde,abmj,cl,deik->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  0.5000 <m,l||e,d>_abab*t2_abab(a,b,m,j)*t1_bb(c,l)*t2_abab(e,d,i,k)
    triples_res +=  0.500000000000000 * einsum('mled,abmj,cl,edik->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_abab*t2_abab(a,b,m,k)*t1_bb(c,l)*t2_abab(d,e,i,j)
    triples_res += -0.500000000000000 * einsum('mlde,abmk,cl,deij->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -0.5000 <m,l||e,d>_abab*t2_abab(a,b,m,k)*t1_bb(c,l)*t2_abab(e,d,i,j)
    triples_res += -0.500000000000000 * einsum('mled,abmk,cl,edij->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t2_abab, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 P(j,k)<m,l||e,d>_abab*t2_abab(a,d,m,k)*t2_abab(e,b,i,j)*t1_bb(c,l)
    contracted_intermediate =  1.000000000000000 * einsum('mled,admk,ebij,cl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <m,l||d,e>_abab*t2_aaaa(d,a,i,m)*t2_bbbb(e,b,j,k)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('mlde,daim,ebjk,cl->abcijk', g_abab[oa, ob, va, vb], t2_aaaa, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,i,m)*t2_bbbb(e,b,j,k)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mlde,adim,ebjk,cl->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t2_abab(a,d,i,k)*t2_abab(e,b,m,j)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('mled,adik,ebmj,cl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,i,k)*t2_bbbb(e,b,j,m)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('mlde,adik,ebjm,cl->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 <m,l||e,d>_abab*t2_abab(a,d,i,j)*t2_abab(e,b,m,k)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mled,adij,ebmk,cl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,i,j)*t2_bbbb(e,b,k,m)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mlde,adij,ebkm,cl->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,k>_bbbb*t2_abab(a,c,i,m)*t1_bb(b,l)*t1_bb(d,j)
    triples_res += -1.000000000000000 * einsum('mldk,acim,bl,dj->abcijk', g_bbbb[ob, ob, vb, ob], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  1.0000 <l,m||d,k>_abab*t1_aa(a,l)*t2_bbbb(b,c,j,m)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('lmdk,al,bcjm,di->abcijk', g_abab[oa, ob, va, ob], t1_aa, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	 -1.0000 <m,l||d,k>_abab*t2_abab(a,c,m,j)*t1_bb(b,l)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('mldk,acmj,bl,di->abcijk', g_abab[oa, ob, va, ob], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,k>_bbbb*t2_abab(a,b,i,m)*t1_bb(c,l)*t1_bb(d,j)
    triples_res +=  1.000000000000000 * einsum('mldk,abim,cl,dj->abcijk', g_bbbb[ob, ob, vb, ob], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  1.0000 <m,l||d,k>_abab*t2_abab(a,b,m,j)*t1_bb(c,l)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('mldk,abmj,cl,di->abcijk', g_abab[oa, ob, va, ob], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)*P(b,c)<l,m||d,k>_abab*t1_aa(a,l)*t1_bb(b,m)*t2_abab(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('lmdk,al,bm,dcij->abcijk', g_abab[oa, ob, va, ob], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,k>_bbbb*t2_abab(a,d,i,j)*t1_bb(b,l)*t1_bb(c,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldk,adij,bl,cm->abcijk', g_bbbb[ob, ob, vb, ob], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <m,l||d,j>_bbbb*t2_abab(a,c,i,m)*t1_bb(b,l)*t1_bb(d,k)
    triples_res +=  1.000000000000000 * einsum('mldj,acim,bl,dk->abcijk', g_bbbb[ob, ob, vb, ob], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	 -1.0000 <l,m||d,j>_abab*t1_aa(a,l)*t2_bbbb(b,c,k,m)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('lmdj,al,bckm,di->abcijk', g_abab[oa, ob, va, ob], t1_aa, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])

    #	  1.0000 <m,l||d,j>_abab*t2_abab(a,c,m,k)*t1_bb(b,l)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('mldj,acmk,bl,di->abcijk', g_abab[oa, ob, va, ob], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,j>_bbbb*t2_abab(a,b,i,m)*t1_bb(c,l)*t1_bb(d,k)
    triples_res += -1.000000000000000 * einsum('mldj,abim,cl,dk->abcijk', g_bbbb[ob, ob, vb, ob], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	 -1.0000 <m,l||d,j>_abab*t2_abab(a,b,m,k)*t1_bb(c,l)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('mldj,abmk,cl,di->abcijk', g_abab[oa, ob, va, ob], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 P(j,k)<l,m||i,d>_abab*t1_aa(a,l)*t2_bbbb(b,c,j,m)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmid,al,bcjm,dk->abcijk', g_abab[oa, ob, oa, vb], t1_aa, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||i,d>_abab*t2_abab(a,c,m,j)*t1_bb(b,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlid,acmj,bl,dk->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||i,d>_abab*t2_abab(a,b,m,j)*t1_bb(c,l)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlid,abmj,cl,dk->abcijk', g_abab[oa, ob, oa, vb], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(b,c)<l,m||i,d>_abab*t1_aa(a,l)*t1_bb(b,m)*t2_bbbb(d,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmid,al,bm,dcjk->abcijk', g_abab[oa, ob, oa, vb], t1_aa, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -1.0000 <l,b||d,e>_bbbb*t2_abab(a,c,i,l)*t1_bb(d,k)*t1_bb(e,j)
    triples_res += -1.000000000000000 * einsum('lbde,acil,dk,ej->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	 -1.0000 <a,l||e,d>_abab*t2_bbbb(b,c,j,l)*t1_bb(d,k)*t1_aa(e,i)
    triples_res += -1.000000000000000 * einsum('aled,bcjl,dk,ei->abcijk', g_abab[va, ob, va, vb], t2_bbbb, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	  1.0000 <l,b||e,d>_abab*t2_abab(a,c,l,j)*t1_bb(d,k)*t1_aa(e,i)
    triples_res +=  1.000000000000000 * einsum('lbed,aclj,dk,ei->abcijk', g_abab[oa, vb, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 P(j,k)*P(b,c)<a,l||e,d>_abab*t1_bb(b,l)*t2_abab(e,c,i,j)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('aled,bl,ecij,dk->abcijk', g_abab[va, ob, va, vb], t1_bb, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  1.0000 <a,l||e,d>_abab*t2_bbbb(b,c,k,l)*t1_bb(d,j)*t1_aa(e,i)
    triples_res +=  1.000000000000000 * einsum('aled,bckl,dj,ei->abcijk', g_abab[va, ob, va, vb], t2_bbbb, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	 -1.0000 <l,b||e,d>_abab*t2_abab(a,c,l,k)*t1_bb(d,j)*t1_aa(e,i)
    triples_res += -1.000000000000000 * einsum('lbed,aclk,dj,ei->abcijk', g_abab[oa, vb, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 P(b,c)<a,l||d,e>_abab*t1_bb(b,l)*t2_bbbb(e,c,j,k)*t1_aa(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('alde,bl,ecjk,di->abcijk', g_abab[va, ob, va, vb], t1_bb, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  1.0000 P(j,k)<l,b||e,d>_abab*t1_aa(a,l)*t2_abab(e,c,i,j)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lbed,al,ecij,dk->abcijk', g_abab[oa, vb, va, vb], t1_aa, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<l,b||d,e>_bbbb*t2_abab(a,e,i,j)*t1_bb(c,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lbde,aeij,cl,dk->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <l,b||d,e>_abab*t1_aa(a,l)*t2_bbbb(e,c,j,k)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('lbde,al,ecjk,di->abcijk', g_abab[oa, vb, va, vb], t1_aa, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <l,c||d,e>_bbbb*t2_abab(a,b,i,l)*t1_bb(d,k)*t1_bb(e,j)
    triples_res +=  1.000000000000000 * einsum('lcde,abil,dk,ej->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	 -1.0000 <l,c||e,d>_abab*t2_abab(a,b,l,j)*t1_bb(d,k)*t1_aa(e,i)
    triples_res += -1.000000000000000 * einsum('lced,ablj,dk,ei->abcijk', g_abab[oa, vb, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)<l,c||e,d>_abab*t1_aa(a,l)*t2_abab(e,b,i,j)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lced,al,ebij,dk->abcijk', g_abab[oa, vb, va, vb], t1_aa, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,c||d,e>_bbbb*t2_abab(a,e,i,j)*t1_bb(b,l)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,aeij,bl,dk->abcijk', g_bbbb[ob, vb, vb, vb], t2_abab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <l,c||e,d>_abab*t2_abab(a,b,l,k)*t1_bb(d,j)*t1_aa(e,i)
    triples_res +=  1.000000000000000 * einsum('lced,ablk,dj,ei->abcijk', g_abab[oa, vb, va, vb], t2_abab, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 <l,c||d,e>_abab*t1_aa(a,l)*t2_bbbb(e,b,j,k)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('lcde,al,ebjk,di->abcijk', g_abab[oa, vb, va, vb], t1_aa, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)<l,m||d,e>_abab*t3_abbabb(a,b,c,i,j,m)*t1_aa(d,l)*t1_bb(e,k)
    contracted_intermediate = -1.000000000000000 * einsum('lmde,abcijm,dl,ek->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,e>_bbbb*t3_abbabb(a,b,c,i,j,m)*t1_bb(d,l)*t1_bb(e,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,abcijm,dl,ek->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <m,l||d,e>_aaaa*t3_abbabb(a,b,c,m,k,j)*t1_aa(d,l)*t1_aa(e,i)
    triples_res += -1.000000000000000 * einsum('mlde,abcmkj,dl,ei->abcijk', g_aaaa[oa, oa, va, va], t3_abbabb, t1_aa, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 <m,l||e,d>_abab*t3_abbabb(a,b,c,m,k,j)*t1_bb(d,l)*t1_aa(e,i)
    triples_res +=  1.000000000000000 * einsum('mled,abcmkj,dl,ei->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(a,m)*t3_abbabb(e,b,c,i,j,k)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('mlde,am,ebcijk,dl->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t3_abbabb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t1_aa(a,m)*t3_abbabb(e,b,c,i,j,k)*t1_bb(d,l)
    triples_res += -1.000000000000000 * einsum('mled,am,ebcijk,dl->abcijk', g_abab[oa, ob, va, vb], t1_aa, t3_abbabb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <l,m||d,e>_abab*t3_abbabb(a,e,c,i,j,k)*t1_bb(b,m)*t1_aa(d,l)
    triples_res += -1.000000000000000 * einsum('lmde,aecijk,bm,dl->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t3_abbabb(a,e,c,i,j,k)*t1_bb(b,m)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('mlde,aecijk,bm,dl->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <l,m||d,e>_abab*t3_abbabb(a,e,b,i,j,k)*t1_bb(c,m)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('lmde,aebijk,cm,dl->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t3_abbabb(a,e,b,i,j,k)*t1_bb(c,m)*t1_bb(d,l)
    triples_res += -1.000000000000000 * einsum('mlde,aebijk,cm,dl->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_bbbb*t3_abbabb(a,b,c,i,m,l)*t1_bb(d,k)*t1_bb(e,j)
    triples_res += -0.500000000000000 * einsum('mlde,abciml,dk,ej->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	  0.5000 <m,l||e,d>_abab*t3_abbabb(a,b,c,m,j,l)*t1_bb(d,k)*t1_aa(e,i)
    triples_res +=  0.500000000000000 * einsum('mled,abcmjl,dk,ei->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -0.5000 <l,m||e,d>_abab*t3_abbabb(a,b,c,l,m,j)*t1_bb(d,k)*t1_aa(e,i)
    triples_res += -0.500000000000000 * einsum('lmed,abclmj,dk,ei->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 P(j,k)<l,m||e,d>_abab*t1_aa(a,l)*t3_abbabb(e,b,c,i,j,m)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmed,al,ebcijm,dk->abcijk', g_abab[oa, ob, va, vb], t1_aa, t3_abbabb, t1_bb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||e,d>_abab*t3_aabaab(e,a,c,i,m,j)*t1_bb(b,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mled,eacimj,bl,dk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,e>_bbbb*t3_abbabb(a,e,c,i,j,m)*t1_bb(b,l)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,aecijm,bl,dk->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||e,d>_abab*t3_aabaab(e,a,b,i,m,j)*t1_bb(c,l)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mled,eabimj,cl,dk->abcijk', g_abab[oa, ob, va, vb], t3_aabaab, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,e>_bbbb*t3_abbabb(a,e,b,i,j,m)*t1_bb(c,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,aebijm,cl,dk->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 <m,l||e,d>_abab*t3_abbabb(a,b,c,m,k,l)*t1_bb(d,j)*t1_aa(e,i)
    triples_res += -0.500000000000000 * einsum('mled,abcmkl,dj,ei->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  0.5000 <l,m||e,d>_abab*t3_abbabb(a,b,c,l,m,k)*t1_bb(d,j)*t1_aa(e,i)
    triples_res +=  0.500000000000000 * einsum('lmed,abclmk,dj,ei->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, t1_aa, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t3_abbabb(e,b,c,m,k,j)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('mlde,al,ebcmkj,di->abcijk', g_aaaa[oa, oa, va, va], t1_aa, t3_abbabb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <l,m||d,e>_abab*t1_aa(a,l)*t3_bbbbbb(e,b,c,j,k,m)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('lmde,al,ebcjkm,di->abcijk', g_abab[oa, ob, va, vb], t1_aa, t3_bbbbbb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_abab*t3_abbabb(a,e,c,m,k,j)*t1_bb(b,l)*t1_aa(d,i)
    triples_res += -1.000000000000000 * einsum('mlde,aecmkj,bl,di->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_abab*t3_abbabb(a,e,b,m,k,j)*t1_bb(c,l)*t1_aa(d,i)
    triples_res +=  1.000000000000000 * einsum('mlde,aebmkj,cl,di->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  0.5000 P(b,c)<l,m||d,e>_abab*t1_aa(a,l)*t1_bb(b,m)*t3_abbabb(d,e,c,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('lmde,al,bm,decijk->abcijk', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.5000 P(b,c)<l,m||e,d>_abab*t1_aa(a,l)*t1_bb(b,m)*t3_abbabb(e,d,c,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('lmed,al,bm,edcijk->abcijk', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.5000 <m,l||d,e>_bbbb*t3_abbabb(a,e,d,i,j,k)*t1_bb(b,l)*t1_bb(c,m)
    triples_res +=  0.500000000000000 * einsum('mlde,aedijk,bl,cm->abcijk', g_bbbb[ob, ob, vb, vb], t3_abbabb, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(a,c,i,m)*t1_bb(b,l)*t1_bb(d,k)*t1_bb(e,j)
    triples_res +=  1.000000000000000 * einsum('mlde,acim,bl,dk,ej->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (2, 3), (1, 2), (0, 1)])

    #	  1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_bbbb(b,c,j,m)*t1_bb(d,k)*t1_aa(e,i)
    triples_res +=  1.000000000000000 * einsum('lmed,al,bcjm,dk,ei->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t2_abab(a,c,m,j)*t1_bb(b,l)*t1_bb(d,k)*t1_aa(e,i)
    triples_res += -1.000000000000000 * einsum('mled,acmj,bl,dk,ei->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,b,i,m)*t1_bb(c,l)*t1_bb(d,k)*t1_bb(e,j)
    triples_res += -1.000000000000000 * einsum('mlde,abim,cl,dk,ej->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (2, 3), (1, 2), (0, 1)])

    #	  1.0000 <m,l||e,d>_abab*t2_abab(a,b,m,j)*t1_bb(c,l)*t1_bb(d,k)*t1_aa(e,i)
    triples_res +=  1.000000000000000 * einsum('mled,abmj,cl,dk,ei->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)*P(b,c)<l,m||e,d>_abab*t1_aa(a,l)*t1_bb(b,m)*t2_abab(e,c,i,j)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lmed,al,bm,ecij,dk->abcijk', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t2_abab, t1_bb, optimize=['einsum_path', (0, 1), (2, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,e>_bbbb*t2_abab(a,e,i,j)*t1_bb(b,l)*t1_bb(c,m)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,aeij,bl,cm,dk->abcijk', g_bbbb[ob, ob, vb, vb], t2_abab, t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_bbbb(b,c,k,m)*t1_bb(d,j)*t1_aa(e,i)
    triples_res += -1.000000000000000 * einsum('lmed,al,bckm,dj,ei->abcijk', g_abab[oa, ob, va, vb], t1_aa, t2_bbbb, t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (1, 2), (0, 1)])

    #	  1.0000 <m,l||e,d>_abab*t2_abab(a,c,m,k)*t1_bb(b,l)*t1_bb(d,j)*t1_aa(e,i)
    triples_res +=  1.000000000000000 * einsum('mled,acmk,bl,dj,ei->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 3), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||e,d>_abab*t2_abab(a,b,m,k)*t1_bb(c,l)*t1_bb(d,j)*t1_aa(e,i)
    triples_res += -1.000000000000000 * einsum('mled,abmk,cl,dj,ei->abcijk', g_abab[oa, ob, va, vb], t2_abab, t1_bb, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 3), (0, 2), (0, 1)])

    #	  1.0000 P(b,c)<l,m||d,e>_abab*t1_aa(a,l)*t1_bb(b,m)*t2_bbbb(e,c,j,k)*t1_aa(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,al,bm,ecjk,di->abcijk', g_abab[oa, ob, va, vb], t1_aa, t1_bb, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 1), (1, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    return triples_res


def ccsdt_t3_bbbbbb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):

    #    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :

    #	 -1.0000 P(j,k)f_bb(l,k)*t3_bbbbbb(a,b,c,i,j,l)
    contracted_intermediate = -1.000000000000000 * einsum('lk,abcijl->abcijk', f_bb[ob, ob], t3_bbbbbb)
    triples_res  =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 f_bb(l,i)*t3_bbbbbb(a,b,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('li,abcjkl->abcijk', f_bb[ob, ob], t3_bbbbbb)

    #	  1.0000 P(a,b)f_bb(a,d)*t3_bbbbbb(d,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('ad,dbcijk->abcijk', f_bb[vb, vb], t3_bbbbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 f_bb(c,d)*t3_bbbbbb(d,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('cd,dabijk->abcijk', f_bb[vb, vb], t3_bbbbbb)

    #	 -1.0000 P(j,k)f_bb(l,d)*t3_bbbbbb(a,b,c,i,j,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('ld,abcijl,dk->abcijk', f_bb[ob, vb], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 f_bb(l,d)*t3_bbbbbb(a,b,c,j,k,l)*t1_bb(d,i)
    triples_res += -1.000000000000000 * einsum('ld,abcjkl,di->abcijk', f_bb[ob, vb], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(a,b)f_bb(l,d)*t1_bb(a,l)*t3_bbbbbb(d,b,c,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('ld,al,dbcijk->abcijk', f_bb[ob, vb], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 f_bb(l,d)*t3_bbbbbb(d,a,b,i,j,k)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('ld,dabijk,cl->abcijk', f_bb[ob, vb], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 P(i,j)*P(a,b)f_bb(l,d)*t2_bbbb(d,a,j,k)*t2_bbbb(b,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('ld,dajk,bcil->abcijk', f_bb[ob, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(a,b)f_bb(l,d)*t2_bbbb(d,a,i,j)*t2_bbbb(b,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('ld,daij,bckl->abcijk', f_bb[ob, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(i,j)f_bb(l,d)*t2_bbbb(a,b,i,l)*t2_bbbb(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('ld,abil,dcjk->abcijk', f_bb[ob, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 f_bb(l,d)*t2_bbbb(a,b,k,l)*t2_bbbb(d,c,i,j)
    triples_res += -1.000000000000000 * einsum('ld,abkl,dcij->abcijk', f_bb[ob, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)*P(a,b)<l,a||j,k>_bbbb*t2_bbbb(b,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lajk,bcil->abcijk', g_bbbb[ob, vb, ob, ob], t2_bbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,a||i,j>_bbbb*t2_bbbb(b,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('laij,bckl->abcijk', g_bbbb[ob, vb, ob, ob], t2_bbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(i,j)<l,c||j,k>_bbbb*t2_bbbb(a,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('lcjk,abil->abcijk', g_bbbb[ob, vb, ob, ob], t2_bbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 <l,c||i,j>_bbbb*t2_bbbb(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcij,abkl->abcijk', g_bbbb[ob, vb, ob, ob], t2_bbbb)

    #	 -1.0000 P(j,k)*P(b,c)<a,b||d,k>_bbbb*t2_bbbb(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abdk,dcij->abcijk', g_bbbb[vb, vb, vb, ob], t2_bbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	 -1.0000 P(b,c)<a,b||d,i>_bbbb*t2_bbbb(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abdi,dcjk->abcijk', g_bbbb[vb, vb, vb, ob], t2_bbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -1.0000 P(j,k)<b,c||d,k>_bbbb*t2_bbbb(d,a,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('bcdk,daij->abcijk', g_bbbb[vb, vb, vb, ob], t2_bbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <b,c||d,i>_bbbb*t2_bbbb(d,a,j,k)
    triples_res += -1.000000000000000 * einsum('bcdi,dajk->abcijk', g_bbbb[vb, vb, vb, ob], t2_bbbb)

    #	  0.5000 P(i,j)<m,l||j,k>_bbbb*t3_bbbbbb(a,b,c,i,m,l)
    contracted_intermediate =  0.500000000000000 * einsum('mljk,abciml->abcijk', g_bbbb[ob, ob, ob, ob], t3_bbbbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 <m,l||i,j>_bbbb*t3_bbbbbb(a,b,c,k,m,l)
    triples_res +=  0.500000000000000 * einsum('mlij,abckml->abcijk', g_bbbb[ob, ob, ob, ob], t3_bbbbbb)

    #	 -1.0000 P(j,k)*P(a,b)<l,a||d,k>_abab*t3_abbabb(d,b,c,l,j,i)
    contracted_intermediate = -1.000000000000000 * einsum('ladk,dbclji->abcijk', g_abab[oa, vb, va, ob], t3_abbabb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<l,a||d,k>_bbbb*t3_bbbbbb(d,b,c,i,j,l)
    contracted_intermediate =  1.000000000000000 * einsum('ladk,dbcijl->abcijk', g_bbbb[ob, vb, vb, ob], t3_bbbbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,a||d,i>_abab*t3_abbabb(d,b,c,l,k,j)
    contracted_intermediate = -1.000000000000000 * einsum('ladi,dbclkj->abcijk', g_abab[oa, vb, va, ob], t3_abbabb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||d,i>_bbbb*t3_bbbbbb(d,b,c,j,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('ladi,dbcjkl->abcijk', g_bbbb[ob, vb, vb, ob], t3_bbbbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(j,k)<l,c||d,k>_abab*t3_abbabb(d,a,b,l,j,i)
    contracted_intermediate = -1.000000000000000 * einsum('lcdk,dablji->abcijk', g_abab[oa, vb, va, ob], t3_abbabb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,c||d,k>_bbbb*t3_bbbbbb(d,a,b,i,j,l)
    contracted_intermediate =  1.000000000000000 * einsum('lcdk,dabijl->abcijk', g_bbbb[ob, vb, vb, ob], t3_bbbbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <l,c||d,i>_abab*t3_abbabb(d,a,b,l,k,j)
    triples_res += -1.000000000000000 * einsum('lcdi,dablkj->abcijk', g_abab[oa, vb, va, ob], t3_abbabb)

    #	  1.0000 <l,c||d,i>_bbbb*t3_bbbbbb(d,a,b,j,k,l)
    triples_res +=  1.000000000000000 * einsum('lcdi,dabjkl->abcijk', g_bbbb[ob, vb, vb, ob], t3_bbbbbb)

    #	  0.5000 P(b,c)<a,b||d,e>_bbbb*t3_bbbbbb(d,e,c,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('abde,decijk->abcijk', g_bbbb[vb, vb, vb, vb], t3_bbbbbb)
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.5000 <b,c||d,e>_bbbb*t3_bbbbbb(d,e,a,i,j,k)
    triples_res +=  0.500000000000000 * einsum('bcde,deaijk->abcijk', g_bbbb[vb, vb, vb, vb], t3_bbbbbb)

    #	  1.0000 P(i,j)*P(a,b)<m,l||j,k>_bbbb*t1_bb(a,l)*t2_bbbb(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mljk,al,bcim->abcijk', g_bbbb[ob, ob, ob, ob], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||j,k>_bbbb*t2_bbbb(a,b,i,m)*t1_bb(c,l)
    contracted_intermediate =  1.000000000000000 * einsum('mljk,abim,cl->abcijk', g_bbbb[ob, ob, ob, ob], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||i,j>_bbbb*t1_bb(a,l)*t2_bbbb(b,c,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlij,al,bckm->abcijk', g_bbbb[ob, ob, ob, ob], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <m,l||i,j>_bbbb*t2_bbbb(a,b,k,m)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mlij,abkm,cl->abcijk', g_bbbb[ob, ob, ob, ob], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)*P(a,b)<l,a||d,k>_bbbb*t2_bbbb(b,c,i,l)*t1_bb(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('ladk,bcil,dj->abcijk', g_bbbb[ob, vb, vb, ob], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(b,c)<l,a||d,k>_bbbb*t1_bb(b,l)*t2_bbbb(d,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('ladk,bl,dcij->abcijk', g_bbbb[ob, vb, vb, ob], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  1.0000 P(i,k)*P(a,b)<l,a||d,j>_bbbb*t2_bbbb(b,c,i,l)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('ladj,bcil,dk->abcijk', g_bbbb[ob, vb, vb, ob], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<l,a||d,i>_bbbb*t2_bbbb(b,c,j,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('ladi,bcjl,dk->abcijk', g_bbbb[ob, vb, vb, ob], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(b,c)<l,a||d,i>_bbbb*t1_bb(b,l)*t2_bbbb(d,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('ladi,bl,dcjk->abcijk', g_bbbb[ob, vb, vb, ob], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,c)<l,b||d,k>_bbbb*t1_bb(a,l)*t2_bbbb(d,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lbdk,al,dcij->abcijk', g_bbbb[ob, vb, vb, ob], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->cbaikj', contracted_intermediate)

    #	  1.0000 P(a,c)<l,b||d,i>_bbbb*t1_bb(a,l)*t2_bbbb(d,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('lbdi,al,dcjk->abcijk', g_bbbb[ob, vb, vb, ob], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate)

    #	 -1.0000 P(i,j)<l,c||d,k>_bbbb*t2_bbbb(a,b,i,l)*t1_bb(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('lcdk,abil,dj->abcijk', g_bbbb[ob, vb, vb, ob], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<l,c||d,k>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('lcdk,al,dbij->abcijk', g_bbbb[ob, vb, vb, ob], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(i,k)<l,c||d,j>_bbbb*t2_bbbb(a,b,i,l)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lcdj,abil,dk->abcijk', g_bbbb[ob, vb, vb, ob], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	 -1.0000 P(j,k)<l,c||d,i>_bbbb*t2_bbbb(a,b,j,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lcdi,abjl,dk->abcijk', g_bbbb[ob, vb, vb, ob], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,c||d,i>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('lcdi,al,dbjk->abcijk', g_bbbb[ob, vb, vb, ob], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(j,k)*P(b,c)<a,b||d,e>_bbbb*t2_bbbb(e,c,i,j)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('abde,ecij,dk->abcijk', g_bbbb[vb, vb, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  1.0000 P(b,c)<a,b||d,e>_bbbb*t2_bbbb(e,c,j,k)*t1_bb(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('abde,ecjk,di->abcijk', g_bbbb[vb, vb, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  1.0000 P(j,k)<b,c||d,e>_bbbb*t2_bbbb(e,a,i,j)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('bcde,eaij,dk->abcijk', g_bbbb[vb, vb, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <b,c||d,e>_bbbb*t2_bbbb(e,a,j,k)*t1_bb(d,i)
    triples_res +=  1.000000000000000 * einsum('bcde,eajk,di->abcijk', g_bbbb[vb, vb, vb, vb], t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)<l,m||d,k>_abab*t3_bbbbbb(a,b,c,i,j,m)*t1_aa(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('lmdk,abcijm,dl->abcijk', g_abab[oa, ob, va, ob], t3_bbbbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,k>_bbbb*t3_bbbbbb(a,b,c,i,j,m)*t1_bb(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,abcijm,dl->abcijk', g_bbbb[ob, ob, vb, ob], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 P(i,j)<m,l||d,k>_bbbb*t3_bbbbbb(a,b,c,i,m,l)*t1_bb(d,j)
    contracted_intermediate =  0.500000000000000 * einsum('mldk,abciml,dj->abcijk', g_bbbb[ob, ob, vb, ob], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||d,k>_abab*t1_bb(a,l)*t3_abbabb(d,b,c,m,j,i)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,al,dbcmji->abcijk', g_abab[oa, ob, va, ob], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<m,l||d,k>_bbbb*t1_bb(a,l)*t3_bbbbbb(d,b,c,i,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldk,al,dbcijm->abcijk', g_bbbb[ob, ob, vb, ob], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,k>_abab*t3_abbabb(d,a,b,m,j,i)*t1_bb(c,l)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,dabmji,cl->abcijk', g_abab[oa, ob, va, ob], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,k>_bbbb*t3_bbbbbb(d,a,b,i,j,m)*t1_bb(c,l)
    contracted_intermediate = -1.000000000000000 * einsum('mldk,dabijm,cl->abcijk', g_bbbb[ob, ob, vb, ob], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 P(i,k)<m,l||d,j>_bbbb*t3_bbbbbb(a,b,c,i,m,l)*t1_bb(d,k)
    contracted_intermediate = -0.500000000000000 * einsum('mldj,abciml,dk->abcijk', g_bbbb[ob, ob, vb, ob], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	 -1.0000 <l,m||d,i>_abab*t3_bbbbbb(a,b,c,j,k,m)*t1_aa(d,l)
    triples_res += -1.000000000000000 * einsum('lmdi,abcjkm,dl->abcijk', g_abab[oa, ob, va, ob], t3_bbbbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,i>_bbbb*t3_bbbbbb(a,b,c,j,k,m)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('mldi,abcjkm,dl->abcijk', g_bbbb[ob, ob, vb, ob], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 P(j,k)<m,l||d,i>_bbbb*t3_bbbbbb(a,b,c,j,m,l)*t1_bb(d,k)
    contracted_intermediate =  0.500000000000000 * einsum('mldi,abcjml,dk->abcijk', g_bbbb[ob, ob, vb, ob], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,i>_abab*t1_bb(a,l)*t3_abbabb(d,b,c,m,k,j)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,al,dbcmkj->abcijk', g_abab[oa, ob, va, ob], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,i>_bbbb*t1_bb(a,l)*t3_bbbbbb(d,b,c,j,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldi,al,dbcjkm->abcijk', g_bbbb[ob, ob, vb, ob], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <m,l||d,i>_abab*t3_abbabb(d,a,b,m,k,j)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mldi,dabmkj,cl->abcijk', g_abab[oa, ob, va, ob], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <m,l||d,i>_bbbb*t3_bbbbbb(d,a,b,j,k,m)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('mldi,dabjkm,cl->abcijk', g_bbbb[ob, ob, vb, ob], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(a,b)<l,a||d,e>_abab*t3_bbbbbb(e,b,c,i,j,k)*t1_aa(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('lade,ebcijk,dl->abcijk', g_abab[oa, vb, va, vb], t3_bbbbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||d,e>_bbbb*t3_bbbbbb(e,b,c,i,j,k)*t1_bb(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('lade,ebcijk,dl->abcijk', g_bbbb[ob, vb, vb, vb], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<l,a||e,d>_abab*t3_abbabb(e,b,c,l,j,i)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('laed,ebclji,dk->abcijk', g_abab[oa, vb, va, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<l,a||d,e>_bbbb*t3_bbbbbb(e,b,c,i,j,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lade,ebcijl,dk->abcijk', g_bbbb[ob, vb, vb, vb], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,a||e,d>_abab*t3_abbabb(e,b,c,l,k,j)*t1_bb(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('laed,ebclkj,di->abcijk', g_abab[oa, vb, va, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,a||d,e>_bbbb*t3_bbbbbb(e,b,c,j,k,l)*t1_bb(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('lade,ebcjkl,di->abcijk', g_bbbb[ob, vb, vb, vb], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(b,c)<l,a||d,e>_bbbb*t1_bb(b,l)*t3_bbbbbb(d,e,c,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('lade,bl,decijk->abcijk', g_bbbb[ob, vb, vb, vb], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -0.5000 P(a,c)<l,b||d,e>_bbbb*t1_bb(a,l)*t3_bbbbbb(d,e,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lbde,al,decijk->abcijk', g_bbbb[ob, vb, vb, vb], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate)

    #	  1.0000 <l,c||d,e>_abab*t3_bbbbbb(e,a,b,i,j,k)*t1_aa(d,l)
    triples_res +=  1.000000000000000 * einsum('lcde,eabijk,dl->abcijk', g_abab[oa, vb, va, vb], t3_bbbbbb, t1_aa, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,c||d,e>_bbbb*t3_bbbbbb(e,a,b,i,j,k)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('lcde,eabijk,dl->abcijk', g_bbbb[ob, vb, vb, vb], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)<l,c||e,d>_abab*t3_abbabb(e,a,b,l,j,i)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lced,eablji,dk->abcijk', g_abab[oa, vb, va, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<l,c||d,e>_bbbb*t3_bbbbbb(e,a,b,i,j,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lcde,eabijl,dk->abcijk', g_bbbb[ob, vb, vb, vb], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <l,c||e,d>_abab*t3_abbabb(e,a,b,l,k,j)*t1_bb(d,i)
    triples_res += -1.000000000000000 * einsum('lced,eablkj,di->abcijk', g_abab[oa, vb, va, vb], t3_abbabb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <l,c||d,e>_bbbb*t3_bbbbbb(e,a,b,j,k,l)*t1_bb(d,i)
    triples_res += -1.000000000000000 * einsum('lcde,eabjkl,di->abcijk', g_bbbb[ob, vb, vb, vb], t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 P(a,b)<l,c||d,e>_bbbb*t1_bb(a,l)*t3_bbbbbb(d,e,b,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('lcde,al,debijk->abcijk', g_bbbb[ob, vb, vb, vb], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(j,k)<l,m||d,e>_abab*t3_bbbbbb(a,b,c,i,j,m)*t2_abab(d,e,l,k)
    contracted_intermediate = -0.500000000000000 * einsum('lmde,abcijm,delk->abcijk', g_abab[oa, ob, va, vb], t3_bbbbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 P(j,k)<l,m||e,d>_abab*t3_bbbbbb(a,b,c,i,j,m)*t2_abab(e,d,l,k)
    contracted_intermediate = -0.500000000000000 * einsum('lmed,abcijm,edlk->abcijk', g_abab[oa, ob, va, vb], t3_bbbbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 P(j,k)<m,l||d,e>_bbbb*t3_bbbbbb(a,b,c,i,j,m)*t2_bbbb(d,e,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,abcijm,dekl->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 <l,m||d,e>_abab*t3_bbbbbb(a,b,c,j,k,m)*t2_abab(d,e,l,i)
    triples_res += -0.500000000000000 * einsum('lmde,abcjkm,deli->abcijk', g_abab[oa, ob, va, vb], t3_bbbbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,m||e,d>_abab*t3_bbbbbb(a,b,c,j,k,m)*t2_abab(e,d,l,i)
    triples_res += -0.500000000000000 * einsum('lmed,abcjkm,edli->abcijk', g_abab[oa, ob, va, vb], t3_bbbbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_bbbb*t3_bbbbbb(a,b,c,j,k,m)*t2_bbbb(d,e,i,l)
    triples_res += -0.500000000000000 * einsum('mlde,abcjkm,deil->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 P(i,j)<m,l||d,e>_bbbb*t3_bbbbbb(a,b,c,i,m,l)*t2_bbbb(d,e,j,k)
    contracted_intermediate =  0.250000000000000 * einsum('mlde,abciml,dejk->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.2500 <m,l||d,e>_bbbb*t3_bbbbbb(a,b,c,k,m,l)*t2_bbbb(d,e,i,j)
    triples_res +=  0.250000000000000 * einsum('mlde,abckml,deij->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 P(a,b)<m,l||d,e>_abab*t2_abab(d,a,m,l)*t3_bbbbbb(e,b,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,daml,ebcijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<l,m||d,e>_abab*t2_abab(d,a,l,m)*t3_bbbbbb(e,b,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lmde,dalm,ebcijk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||d,e>_bbbb*t2_bbbb(d,a,m,l)*t3_bbbbbb(e,b,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,daml,ebcijk->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||d,e>_aaaa*t2_abab(d,a,l,k)*t3_abbabb(e,b,c,m,j,i)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dalk,ebcmji->abcijk', g_aaaa[oa, oa, va, va], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<l,m||d,e>_abab*t2_abab(d,a,l,k)*t3_bbbbbb(e,b,c,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,dalk,ebcijm->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||e,d>_abab*t2_bbbb(d,a,k,l)*t3_abbabb(e,b,c,m,j,i)
    contracted_intermediate =  1.000000000000000 * einsum('mled,dakl,ebcmji->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||d,e>_bbbb*t2_bbbb(d,a,k,l)*t3_bbbbbb(e,b,c,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dakl,ebcijm->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_aaaa*t2_abab(d,a,l,i)*t3_abbabb(e,b,c,m,k,j)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dali,ebcmkj->abcijk', g_aaaa[oa, oa, va, va], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,m||d,e>_abab*t2_abab(d,a,l,i)*t3_bbbbbb(e,b,c,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,dali,ebcjkm->abcijk', g_abab[oa, ob, va, vb], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||e,d>_abab*t2_bbbb(d,a,i,l)*t3_abbabb(e,b,c,m,k,j)
    contracted_intermediate =  1.000000000000000 * einsum('mled,dail,ebcmkj->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_bbbb*t2_bbbb(d,a,i,l)*t3_bbbbbb(e,b,c,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,dail,ebcjkm->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(i,j)*P(a,b)<m,l||e,d>_abab*t2_bbbb(d,a,j,k)*t3_abbabb(e,b,c,m,i,l)
    contracted_intermediate = -0.500000000000000 * einsum('mled,dajk,ebcmil->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  0.5000 P(i,j)*P(a,b)<l,m||e,d>_abab*t2_bbbb(d,a,j,k)*t3_abbabb(e,b,c,l,m,i)
    contracted_intermediate =  0.500000000000000 * einsum('lmed,dajk,ebclmi->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -0.5000 P(i,j)*P(a,b)<m,l||d,e>_bbbb*t2_bbbb(d,a,j,k)*t3_bbbbbb(e,b,c,i,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,dajk,ebciml->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||e,d>_abab*t2_bbbb(d,a,i,j)*t3_abbabb(e,b,c,m,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('mled,daij,ebcmkl->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(a,b)<l,m||e,d>_abab*t2_bbbb(d,a,i,j)*t3_abbabb(e,b,c,l,m,k)
    contracted_intermediate =  0.500000000000000 * einsum('lmed,daij,ebclmk->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||d,e>_bbbb*t2_bbbb(d,a,i,j)*t3_bbbbbb(e,b,c,k,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,daij,ebckml->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 <m,l||d,e>_abab*t3_bbbbbb(e,a,b,i,j,k)*t2_abab(d,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,eabijk,dcml->abcijk', g_abab[oa, ob, va, vb], t3_bbbbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <l,m||d,e>_abab*t3_bbbbbb(e,a,b,i,j,k)*t2_abab(d,c,l,m)
    triples_res += -0.500000000000000 * einsum('lmde,eabijk,dclm->abcijk', g_abab[oa, ob, va, vb], t3_bbbbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,l||d,e>_bbbb*t3_bbbbbb(e,a,b,i,j,k)*t2_bbbb(d,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,eabijk,dcml->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 P(j,k)<m,l||d,e>_aaaa*t3_abbabb(e,a,b,m,j,i)*t2_abab(d,c,l,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,eabmji,dclk->abcijk', g_aaaa[oa, oa, va, va], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||e,d>_abab*t3_abbabb(e,a,b,m,j,i)*t2_bbbb(d,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('mled,eabmji,dckl->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,m||d,e>_abab*t3_bbbbbb(e,a,b,i,j,m)*t2_abab(d,c,l,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmde,eabijm,dclk->abcijk', g_abab[oa, ob, va, vb], t3_bbbbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,e>_bbbb*t3_bbbbbb(e,a,b,i,j,m)*t2_bbbb(d,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,eabijm,dckl->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <m,l||d,e>_aaaa*t3_abbabb(e,a,b,m,k,j)*t2_abab(d,c,l,i)
    triples_res +=  1.000000000000000 * einsum('mlde,eabmkj,dcli->abcijk', g_aaaa[oa, oa, va, va], t3_abbabb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||e,d>_abab*t3_abbabb(e,a,b,m,k,j)*t2_bbbb(d,c,i,l)
    triples_res +=  1.000000000000000 * einsum('mled,eabmkj,dcil->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,m||d,e>_abab*t3_bbbbbb(e,a,b,j,k,m)*t2_abab(d,c,l,i)
    triples_res +=  1.000000000000000 * einsum('lmde,eabjkm,dcli->abcijk', g_abab[oa, ob, va, vb], t3_bbbbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t3_bbbbbb(e,a,b,j,k,m)*t2_bbbb(d,c,i,l)
    triples_res +=  1.000000000000000 * einsum('mlde,eabjkm,dcil->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 P(i,j)<m,l||e,d>_abab*t3_abbabb(e,a,b,m,i,l)*t2_bbbb(d,c,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mled,eabmil,dcjk->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 P(i,j)<l,m||e,d>_abab*t3_abbabb(e,a,b,l,m,i)*t2_bbbb(d,c,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('lmed,eablmi,dcjk->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -0.5000 P(i,j)<m,l||d,e>_bbbb*t3_bbbbbb(e,a,b,i,m,l)*t2_bbbb(d,c,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,eabiml,dcjk->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -0.5000 <m,l||e,d>_abab*t3_abbabb(e,a,b,m,k,l)*t2_bbbb(d,c,i,j)
    triples_res += -0.500000000000000 * einsum('mled,eabmkl,dcij->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <l,m||e,d>_abab*t3_abbabb(e,a,b,l,m,k)*t2_bbbb(d,c,i,j)
    triples_res +=  0.500000000000000 * einsum('lmed,eablmk,dcij->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||d,e>_bbbb*t3_bbbbbb(e,a,b,k,m,l)*t2_bbbb(d,c,i,j)
    triples_res += -0.500000000000000 * einsum('mlde,eabkml,dcij->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.2500 P(b,c)<m,l||d,e>_bbbb*t2_bbbb(a,b,m,l)*t3_bbbbbb(d,e,c,i,j,k)
    contracted_intermediate =  0.250000000000000 * einsum('mlde,abml,decijk->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.5000 P(j,k)*P(b,c)<m,l||d,e>_abab*t2_bbbb(a,b,k,l)*t3_abbabb(d,e,c,m,j,i)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,abkl,decmji->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  0.5000 P(j,k)*P(b,c)<m,l||e,d>_abab*t2_bbbb(a,b,k,l)*t3_abbabb(e,d,c,m,j,i)
    contracted_intermediate =  0.500000000000000 * einsum('mled,abkl,edcmji->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	 -0.5000 P(j,k)*P(b,c)<m,l||d,e>_bbbb*t2_bbbb(a,b,k,l)*t3_bbbbbb(d,e,c,i,j,m)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,abkl,decijm->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  0.5000 P(b,c)<m,l||d,e>_abab*t2_bbbb(a,b,i,l)*t3_abbabb(d,e,c,m,k,j)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,abil,decmkj->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.5000 P(b,c)<m,l||e,d>_abab*t2_bbbb(a,b,i,l)*t3_abbabb(e,d,c,m,k,j)
    contracted_intermediate =  0.500000000000000 * einsum('mled,abil,edcmkj->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -0.5000 P(b,c)<m,l||d,e>_bbbb*t2_bbbb(a,b,i,l)*t3_bbbbbb(d,e,c,j,k,m)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,abil,decjkm->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  0.2500 <m,l||d,e>_bbbb*t3_bbbbbb(d,e,a,i,j,k)*t2_bbbb(b,c,m,l)
    triples_res +=  0.250000000000000 * einsum('mlde,deaijk,bcml->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 P(j,k)<m,l||d,e>_abab*t3_abbabb(d,e,a,m,j,i)*t2_bbbb(b,c,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,deamji,bckl->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 P(j,k)<m,l||e,d>_abab*t3_abbabb(e,d,a,m,j,i)*t2_bbbb(b,c,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('mled,edamji,bckl->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 P(j,k)<m,l||d,e>_bbbb*t3_bbbbbb(d,e,a,i,j,m)*t2_bbbb(b,c,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,deaijm,bckl->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 <m,l||d,e>_abab*t3_abbabb(d,e,a,m,k,j)*t2_bbbb(b,c,i,l)
    triples_res +=  0.500000000000000 * einsum('mlde,deamkj,bcil->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <m,l||e,d>_abab*t3_abbabb(e,d,a,m,k,j)*t2_bbbb(b,c,i,l)
    triples_res +=  0.500000000000000 * einsum('mled,edamkj,bcil->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <m,l||d,e>_bbbb*t3_bbbbbb(d,e,a,j,k,m)*t2_bbbb(b,c,i,l)
    triples_res += -0.500000000000000 * einsum('mlde,deajkm,bcil->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<l,m||d,k>_abab*t2_abab(d,a,l,j)*t2_bbbb(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmdk,dalj,bcim->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<m,l||d,k>_bbbb*t2_bbbb(d,a,j,l)*t2_bbbb(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,dajl,bcim->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -0.5000 P(j,k)*P(a,b)<m,l||d,k>_bbbb*t2_bbbb(d,a,i,j)*t2_bbbb(b,c,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mldk,daij,bcml->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(i,j)<l,m||d,k>_abab*t2_bbbb(a,b,i,m)*t2_abab(d,c,l,j)
    contracted_intermediate =  1.000000000000000 * einsum('lmdk,abim,dclj->abcijk', g_abab[oa, ob, va, ob], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||d,k>_bbbb*t2_bbbb(a,b,i,m)*t2_bbbb(d,c,j,l)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,abim,dcjl->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -0.5000 P(j,k)<m,l||d,k>_bbbb*t2_bbbb(a,b,m,l)*t2_bbbb(d,c,i,j)
    contracted_intermediate = -0.500000000000000 * einsum('mldk,abml,dcij->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(i,k)*P(a,b)<l,m||d,j>_abab*t2_abab(d,a,l,k)*t2_bbbb(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('lmdj,dalk,bcim->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate)

    #	 -1.0000 P(i,k)*P(a,b)<m,l||d,j>_bbbb*t2_bbbb(d,a,k,l)*t2_bbbb(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,dakl,bcim->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate)

    #	 -1.0000 P(i,k)<l,m||d,j>_abab*t2_bbbb(a,b,i,m)*t2_abab(d,c,l,k)
    contracted_intermediate = -1.000000000000000 * einsum('lmdj,abim,dclk->abcijk', g_abab[oa, ob, va, ob], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	 -1.0000 P(i,k)<m,l||d,j>_bbbb*t2_bbbb(a,b,i,m)*t2_bbbb(d,c,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,abim,dckl->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<l,m||d,i>_abab*t2_abab(d,a,l,k)*t2_bbbb(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('lmdi,dalk,bcjm->abcijk', g_abab[oa, ob, va, ob], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||d,i>_bbbb*t2_bbbb(d,a,k,l)*t2_bbbb(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,dakl,bcjm->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -0.5000 P(a,b)<m,l||d,i>_bbbb*t2_bbbb(d,a,j,k)*t2_bbbb(b,c,m,l)
    contracted_intermediate = -0.500000000000000 * einsum('mldi,dajk,bcml->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(j,k)<l,m||d,i>_abab*t2_bbbb(a,b,j,m)*t2_abab(d,c,l,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmdi,abjm,dclk->abcijk', g_abab[oa, ob, va, ob], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,i>_bbbb*t2_bbbb(a,b,j,m)*t2_bbbb(d,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,abjm,dckl->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 <m,l||d,i>_bbbb*t2_bbbb(a,b,m,l)*t2_bbbb(d,c,j,k)
    triples_res += -0.500000000000000 * einsum('mldi,abml,dcjk->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 P(i,j)*P(a,b)<l,a||d,e>_bbbb*t2_bbbb(b,c,i,l)*t2_bbbb(d,e,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lade,bcil,dejk->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -0.5000 P(a,b)<l,a||d,e>_bbbb*t2_bbbb(b,c,k,l)*t2_bbbb(d,e,i,j)
    contracted_intermediate = -0.500000000000000 * einsum('lade,bckl,deij->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<l,a||d,e>_abab*t2_abab(d,b,l,k)*t2_bbbb(e,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('lade,dblk,ecij->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<l,a||d,e>_bbbb*t2_bbbb(d,b,k,l)*t2_bbbb(e,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dbkl,ecij->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,a||d,e>_abab*t2_abab(d,b,l,i)*t2_bbbb(e,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('lade,dbli,ecjk->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||d,e>_bbbb*t2_bbbb(d,b,i,l)*t2_bbbb(e,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dbil,ecjk->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<l,a||e,d>_abab*t2_bbbb(d,b,j,k)*t2_abab(e,c,l,i)
    contracted_intermediate =  1.000000000000000 * einsum('laed,dbjk,ecli->abcijk', g_abab[oa, vb, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<l,a||d,e>_bbbb*t2_bbbb(d,b,j,k)*t2_bbbb(e,c,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dbjk,ecil->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||e,d>_abab*t2_bbbb(d,b,i,j)*t2_abab(e,c,l,k)
    contracted_intermediate =  1.000000000000000 * einsum('laed,dbij,eclk->abcijk', g_abab[oa, vb, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||d,e>_bbbb*t2_bbbb(d,b,i,j)*t2_bbbb(e,c,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('lade,dbij,eckl->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -0.5000 P(i,j)<l,c||d,e>_bbbb*t2_bbbb(a,b,i,l)*t2_bbbb(d,e,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('lcde,abil,dejk->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -0.5000 <l,c||d,e>_bbbb*t2_bbbb(a,b,k,l)*t2_bbbb(d,e,i,j)
    triples_res += -0.500000000000000 * einsum('lcde,abkl,deij->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(j,k)<l,c||d,e>_abab*t2_abab(d,a,l,k)*t2_bbbb(e,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('lcde,dalk,ebij->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<l,c||d,e>_bbbb*t2_bbbb(d,a,k,l)*t2_bbbb(e,b,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,dakl,ebij->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <l,c||d,e>_abab*t2_abab(d,a,l,i)*t2_bbbb(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('lcde,dali,ebjk->abcijk', g_abab[oa, vb, va, vb], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <l,c||d,e>_bbbb*t2_bbbb(d,a,i,l)*t2_bbbb(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dail,ebjk->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)<l,c||e,d>_abab*t2_bbbb(d,a,j,k)*t2_abab(e,b,l,i)
    contracted_intermediate =  1.000000000000000 * einsum('lced,dajk,ebli->abcijk', g_abab[oa, vb, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(i,j)<l,c||d,e>_bbbb*t2_bbbb(d,a,j,k)*t2_bbbb(e,b,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,dajk,ebil->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 <l,c||e,d>_abab*t2_bbbb(d,a,i,j)*t2_abab(e,b,l,k)
    triples_res +=  1.000000000000000 * einsum('lced,daij,eblk->abcijk', g_abab[oa, vb, va, vb], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <l,c||d,e>_bbbb*t2_bbbb(d,a,i,j)*t2_bbbb(e,b,k,l)
    triples_res +=  1.000000000000000 * einsum('lcde,daij,ebkl->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)*P(a,b)<l,m||d,e>_abab*t2_bbbb(e,a,j,k)*t2_bbbb(b,c,i,m)*t1_aa(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('lmde,eajk,bcim,dl->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)<m,l||d,e>_bbbb*t2_bbbb(e,a,j,k)*t2_bbbb(b,c,i,m)*t1_bb(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,eajk,bcim,dl->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(a,b)<l,m||d,e>_abab*t2_bbbb(e,a,i,j)*t2_bbbb(b,c,k,m)*t1_aa(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('lmde,eaij,bckm,dl->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_bbbb*t2_bbbb(e,a,i,j)*t2_bbbb(b,c,k,m)*t1_bb(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,eaij,bckm,dl->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(i,j)<l,m||d,e>_abab*t2_bbbb(a,b,i,m)*t2_bbbb(e,c,j,k)*t1_aa(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('lmde,abim,ecjk,dl->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||d,e>_bbbb*t2_bbbb(a,b,i,m)*t2_bbbb(e,c,j,k)*t1_bb(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,abim,ecjk,dl->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 <l,m||d,e>_abab*t2_bbbb(a,b,k,m)*t2_bbbb(e,c,i,j)*t1_aa(d,l)
    triples_res += -1.000000000000000 * einsum('lmde,abkm,ecij,dl->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t2_bbbb, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t2_bbbb(a,b,k,m)*t2_bbbb(e,c,i,j)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('mlde,abkm,ecij,dl->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<l,m||e,d>_abab*t2_abab(e,a,l,j)*t2_bbbb(b,c,i,m)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmed,ealj,bcim,dk->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,e>_bbbb*t2_bbbb(e,a,j,l)*t2_bbbb(b,c,i,m)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,eajl,bcim,dk->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  0.5000 P(j,k)*P(a,b)<m,l||d,e>_bbbb*t2_bbbb(e,a,i,j)*t2_bbbb(b,c,m,l)*t1_bb(d,k)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,eaij,bcml,dk->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(i,j)<l,m||e,d>_abab*t2_bbbb(a,b,i,m)*t2_abab(e,c,l,j)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lmed,abim,eclj,dk->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(i,j)<m,l||d,e>_bbbb*t2_bbbb(a,b,i,m)*t2_bbbb(e,c,j,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,abim,ecjl,dk->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 P(j,k)<m,l||d,e>_bbbb*t2_bbbb(a,b,m,l)*t2_bbbb(e,c,i,j)*t1_bb(d,k)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,abml,ecij,dk->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(i,k)*P(a,b)<l,m||e,d>_abab*t2_abab(e,a,l,k)*t2_bbbb(b,c,i,m)*t1_bb(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('lmed,ealk,bcim,dj->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate)

    #	  1.0000 P(i,k)*P(a,b)<m,l||d,e>_bbbb*t2_bbbb(e,a,k,l)*t2_bbbb(b,c,i,m)*t1_bb(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,eakl,bcim,dj->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate)

    #	 -1.0000 P(i,k)<l,m||e,d>_abab*t2_bbbb(a,b,i,m)*t2_abab(e,c,l,k)*t1_bb(d,j)
    contracted_intermediate = -1.000000000000000 * einsum('lmed,abim,eclk,dj->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	  1.0000 P(i,k)<m,l||d,e>_bbbb*t2_bbbb(a,b,i,m)*t2_bbbb(e,c,k,l)*t1_bb(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,abim,eckl,dj->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<l,m||e,d>_abab*t2_abab(e,a,l,k)*t2_bbbb(b,c,j,m)*t1_bb(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('lmed,ealk,bcjm,di->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<m,l||d,e>_bbbb*t2_bbbb(e,a,k,l)*t2_bbbb(b,c,j,m)*t1_bb(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,eakl,bcjm,di->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  0.5000 P(a,b)<m,l||d,e>_bbbb*t2_bbbb(e,a,j,k)*t2_bbbb(b,c,m,l)*t1_bb(d,i)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,eajk,bcml,di->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(j,k)<l,m||e,d>_abab*t2_bbbb(a,b,j,m)*t2_abab(e,c,l,k)*t1_bb(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('lmed,abjm,eclk,di->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,e>_bbbb*t2_bbbb(a,b,j,m)*t2_bbbb(e,c,k,l)*t1_bb(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,abjm,eckl,di->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(a,b,m,l)*t2_bbbb(e,c,j,k)*t1_bb(d,i)
    triples_res +=  0.500000000000000 * einsum('mlde,abml,ecjk,di->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  0.5000 P(i,j)*P(a,b)<m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(b,c,i,m)*t2_bbbb(d,e,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,al,bcim,dejk->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  0.5000 P(a,b)<m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(b,c,k,m)*t2_bbbb(d,e,i,j)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,al,bckm,deij->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||d,e>_abab*t1_bb(a,l)*t2_abab(d,b,m,k)*t2_bbbb(e,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,al,dbmk,ecij->abcijk', g_abab[oa, ob, va, vb], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,b)<m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,k,m)*t2_bbbb(e,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,dbkm,ecij->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_abab*t1_bb(a,l)*t2_abab(d,b,m,i)*t2_bbbb(e,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,al,dbmi,ecjk->abcijk', g_abab[oa, ob, va, vb], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,i,m)*t2_bbbb(e,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,dbim,ecjk->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<m,l||e,d>_abab*t1_bb(a,l)*t2_bbbb(d,b,j,k)*t2_abab(e,c,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('mled,al,dbjk,ecmi->abcijk', g_abab[oa, ob, va, vb], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,j,k)*t2_bbbb(e,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,dbjk,ecim->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||e,d>_abab*t1_bb(a,l)*t2_bbbb(d,b,i,j)*t2_abab(e,c,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('mled,al,dbij,ecmk->abcijk', g_abab[oa, ob, va, vb], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,i,j)*t2_bbbb(e,c,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,dbij,eckm->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  0.5000 P(i,j)<m,l||d,e>_bbbb*t2_bbbb(a,b,i,m)*t1_bb(c,l)*t2_bbbb(d,e,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('mlde,abim,cl,dejk->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(a,b,k,m)*t1_bb(c,l)*t2_bbbb(d,e,i,j)
    triples_res +=  0.500000000000000 * einsum('mlde,abkm,cl,deij->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])

    #	  1.0000 P(j,k)<m,l||d,e>_abab*t2_abab(d,a,m,k)*t2_bbbb(e,b,i,j)*t1_bb(c,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,damk,ebij,cl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,e>_bbbb*t2_bbbb(d,a,k,m)*t2_bbbb(e,b,i,j)*t1_bb(c,l)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,dakm,ebij,cl->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 <m,l||d,e>_abab*t2_abab(d,a,m,i)*t2_bbbb(e,b,j,k)*t1_bb(c,l)
    triples_res +=  1.000000000000000 * einsum('mlde,dami,ebjk,cl->abcijk', g_abab[oa, ob, va, vb], t2_abab, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,a,i,m)*t2_bbbb(e,b,j,k)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('mlde,daim,ebjk,cl->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)<m,l||e,d>_abab*t2_bbbb(d,a,j,k)*t2_abab(e,b,m,i)*t1_bb(c,l)
    contracted_intermediate = -1.000000000000000 * einsum('mled,dajk,ebmi,cl->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(i,j)<m,l||d,e>_bbbb*t2_bbbb(d,a,j,k)*t2_bbbb(e,b,i,m)*t1_bb(c,l)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,dajk,ebim,cl->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 <m,l||e,d>_abab*t2_bbbb(d,a,i,j)*t2_abab(e,b,m,k)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('mled,daij,ebmk,cl->abcijk', g_abab[oa, ob, va, vb], t2_bbbb, t2_abab, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	 -1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,a,i,j)*t2_bbbb(e,b,k,m)*t1_bb(c,l)
    triples_res += -1.000000000000000 * einsum('mlde,daij,ebkm,cl->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<m,l||d,k>_bbbb*t1_bb(a,l)*t2_bbbb(b,c,i,m)*t1_bb(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,al,bcim,dj->abcijk', g_bbbb[ob, ob, vb, ob], t1_bb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(i,j)<m,l||d,k>_bbbb*t2_bbbb(a,b,i,m)*t1_bb(c,l)*t1_bb(d,j)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,abim,cl,dj->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(j,k)*P(b,c)<m,l||d,k>_bbbb*t1_bb(a,l)*t1_bb(b,m)*t2_bbbb(d,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,al,bm,dcij->abcijk', g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,k>_bbbb*t2_bbbb(d,a,i,j)*t1_bb(b,l)*t1_bb(c,m)
    contracted_intermediate =  1.000000000000000 * einsum('mldk,daij,bl,cm->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(i,k)*P(a,b)<m,l||d,j>_bbbb*t1_bb(a,l)*t2_bbbb(b,c,i,m)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,al,bcim,dk->abcijk', g_bbbb[ob, ob, vb, ob], t1_bb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->backji', contracted_intermediate)

    #	 -1.0000 P(i,k)<m,l||d,j>_bbbb*t2_bbbb(a,b,i,m)*t1_bb(c,l)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mldj,abim,cl,dk->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abckji', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||d,i>_bbbb*t1_bb(a,l)*t2_bbbb(b,c,j,m)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,al,bcjm,dk->abcijk', g_bbbb[ob, ob, vb, ob], t1_bb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,i>_bbbb*t2_bbbb(a,b,j,m)*t1_bb(c,l)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,abjm,cl,dk->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(b,c)<m,l||d,i>_bbbb*t1_bb(a,l)*t1_bb(b,m)*t2_bbbb(d,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('mldi,al,bm,dcjk->abcijk', g_bbbb[ob, ob, vb, ob], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	  1.0000 <m,l||d,i>_bbbb*t2_bbbb(d,a,j,k)*t1_bb(b,l)*t1_bb(c,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dajk,bl,cm->abcijk', g_bbbb[ob, ob, vb, ob], t2_bbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)<l,a||d,e>_bbbb*t2_bbbb(b,c,i,l)*t1_bb(d,k)*t1_bb(e,j)
    contracted_intermediate =  1.000000000000000 * einsum('lade,bcil,dk,ej->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	  1.0000 P(j,k)*P(b,c)<l,a||d,e>_bbbb*t1_bb(b,l)*t2_bbbb(e,c,i,j)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lade,bl,ecij,dk->abcijk', g_bbbb[ob, vb, vb, vb], t1_bb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	  1.0000 P(a,b)<l,a||d,e>_bbbb*t2_bbbb(b,c,k,l)*t1_bb(d,j)*t1_bb(e,i)
    contracted_intermediate =  1.000000000000000 * einsum('lade,bckl,dj,ei->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(b,c)<l,a||d,e>_bbbb*t1_bb(b,l)*t2_bbbb(e,c,j,k)*t1_bb(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('lade,bl,ecjk,di->abcijk', g_bbbb[ob, vb, vb, vb], t1_bb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(a,c)<l,b||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(e,c,i,j)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('lbde,al,ecij,dk->abcijk', g_bbbb[ob, vb, vb, vb], t1_bb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->cbaikj', contracted_intermediate)

    #	 -1.0000 P(a,c)<l,b||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(e,c,j,k)*t1_bb(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('lbde,al,ecjk,di->abcijk', g_bbbb[ob, vb, vb, vb], t1_bb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->cbaijk', contracted_intermediate)

    #	  1.0000 P(i,j)<l,c||d,e>_bbbb*t2_bbbb(a,b,i,l)*t1_bb(d,k)*t1_bb(e,j)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,abil,dk,ej->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<l,c||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(e,b,i,j)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,al,ebij,dk->abcijk', g_bbbb[ob, vb, vb, vb], t1_bb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 <l,c||d,e>_bbbb*t2_bbbb(a,b,k,l)*t1_bb(d,j)*t1_bb(e,i)
    triples_res +=  1.000000000000000 * einsum('lcde,abkl,dj,ei->abcijk', g_bbbb[ob, vb, vb, vb], t2_bbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	  1.0000 P(a,b)<l,c||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(e,b,j,k)*t1_bb(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('lcde,al,ebjk,di->abcijk', g_bbbb[ob, vb, vb, vb], t1_bb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 P(j,k)<l,m||d,e>_abab*t3_bbbbbb(a,b,c,i,j,m)*t1_aa(d,l)*t1_bb(e,k)
    contracted_intermediate = -1.000000000000000 * einsum('lmde,abcijm,dl,ek->abcijk', g_abab[oa, ob, va, vb], t3_bbbbbb, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,e>_bbbb*t3_bbbbbb(a,b,c,i,j,m)*t1_bb(d,l)*t1_bb(e,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,abcijm,dl,ek->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 <l,m||d,e>_abab*t3_bbbbbb(a,b,c,j,k,m)*t1_aa(d,l)*t1_bb(e,i)
    triples_res += -1.000000000000000 * einsum('lmde,abcjkm,dl,ei->abcijk', g_abab[oa, ob, va, vb], t3_bbbbbb, t1_aa, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t3_bbbbbb(a,b,c,j,k,m)*t1_bb(d,l)*t1_bb(e,i)
    triples_res +=  1.000000000000000 * einsum('mlde,abcjkm,dl,ei->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	 -1.0000 P(a,b)<l,m||d,e>_abab*t1_bb(a,m)*t3_bbbbbb(e,b,c,i,j,k)*t1_aa(d,l)
    contracted_intermediate = -1.000000000000000 * einsum('lmde,am,ebcijk,dl->abcijk', g_abab[oa, ob, va, vb], t1_bb, t3_bbbbbb, t1_aa, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_bbbb*t1_bb(a,m)*t3_bbbbbb(e,b,c,i,j,k)*t1_bb(d,l)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,am,ebcijk,dl->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 <l,m||d,e>_abab*t3_bbbbbb(e,a,b,i,j,k)*t1_bb(c,m)*t1_aa(d,l)
    triples_res += -1.000000000000000 * einsum('lmde,eabijk,cm,dl->abcijk', g_abab[oa, ob, va, vb], t3_bbbbbb, t1_bb, t1_aa, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t3_bbbbbb(e,a,b,i,j,k)*t1_bb(c,m)*t1_bb(d,l)
    triples_res +=  1.000000000000000 * einsum('mlde,eabijk,cm,dl->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -0.5000 P(i,j)<m,l||d,e>_bbbb*t3_bbbbbb(a,b,c,i,m,l)*t1_bb(d,k)*t1_bb(e,j)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,abciml,dk,ej->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||e,d>_abab*t1_bb(a,l)*t3_abbabb(e,b,c,m,j,i)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mled,al,ebcmji,dk->abcijk', g_abab[oa, ob, va, vb], t1_bb, t3_abbabb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)*P(a,b)<m,l||d,e>_bbbb*t1_bb(a,l)*t3_bbbbbb(e,b,c,i,j,m)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,al,ebcijm,dk->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||e,d>_abab*t3_abbabb(e,a,b,m,j,i)*t1_bb(c,l)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mled,eabmji,cl,dk->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	  1.0000 P(j,k)<m,l||d,e>_bbbb*t3_bbbbbb(e,a,b,i,j,m)*t1_bb(c,l)*t1_bb(d,k)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,eabijm,cl,dk->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -0.5000 <m,l||d,e>_bbbb*t3_bbbbbb(a,b,c,k,m,l)*t1_bb(d,j)*t1_bb(e,i)
    triples_res += -0.500000000000000 * einsum('mlde,abckml,dj,ei->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	  1.0000 P(a,b)<m,l||e,d>_abab*t1_bb(a,l)*t3_abbabb(e,b,c,m,k,j)*t1_bb(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('mled,al,ebcmkj,di->abcijk', g_abab[oa, ob, va, vb], t1_bb, t3_abbabb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 P(a,b)<m,l||d,e>_bbbb*t1_bb(a,l)*t3_bbbbbb(e,b,c,j,k,m)*t1_bb(d,i)
    contracted_intermediate =  1.000000000000000 * einsum('mlde,al,ebcjkm,di->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t3_bbbbbb, t1_bb, optimize=['einsum_path', (0, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	  1.0000 <m,l||e,d>_abab*t3_abbabb(e,a,b,m,k,j)*t1_bb(c,l)*t1_bb(d,i)
    triples_res +=  1.000000000000000 * einsum('mled,eabmkj,cl,di->abcijk', g_abab[oa, ob, va, vb], t3_abbabb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	  1.0000 <m,l||d,e>_bbbb*t3_bbbbbb(e,a,b,j,k,m)*t1_bb(c,l)*t1_bb(d,i)
    triples_res +=  1.000000000000000 * einsum('mlde,eabjkm,cl,di->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])

    #	 -0.5000 P(b,c)<m,l||d,e>_bbbb*t1_bb(a,l)*t1_bb(b,m)*t3_bbbbbb(d,e,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('mlde,al,bm,decijk->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -0.5000 <m,l||d,e>_bbbb*t3_bbbbbb(d,e,a,i,j,k)*t1_bb(b,l)*t1_bb(c,m)
    triples_res += -0.500000000000000 * einsum('mlde,deaijk,bl,cm->abcijk', g_bbbb[ob, ob, vb, vb], t3_bbbbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 P(i,j)*P(a,b)<m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(b,c,i,m)*t1_bb(d,k)*t1_bb(e,j)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,bcim,dk,ej->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (2, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)  + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->bacjik', contracted_intermediate)

    #	 -1.0000 P(i,j)<m,l||d,e>_bbbb*t2_bbbb(a,b,i,m)*t1_bb(c,l)*t1_bb(d,k)*t1_bb(e,j)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,abim,cl,dk,ej->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (2, 3), (1, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcjik', contracted_intermediate)

    #	 -1.0000 P(j,k)*P(b,c)<m,l||d,e>_bbbb*t1_bb(a,l)*t1_bb(b,m)*t2_bbbb(e,c,i,j)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,bm,ecij,dk->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 4), (2, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)  + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)  +  1.00000 * einsum('abcijk->acbikj', contracted_intermediate)

    #	 -1.0000 P(j,k)<m,l||d,e>_bbbb*t2_bbbb(e,a,i,j)*t1_bb(b,l)*t1_bb(c,m)*t1_bb(d,k)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,eaij,bl,cm,dk->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate)

    #	 -1.0000 P(a,b)<m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(b,c,k,m)*t1_bb(d,j)*t1_bb(e,i)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,bckm,dj,ei->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t2_bbbb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (2, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate)

    #	 -1.0000 <m,l||d,e>_bbbb*t2_bbbb(a,b,k,m)*t1_bb(c,l)*t1_bb(d,j)*t1_bb(e,i)
    triples_res += -1.000000000000000 * einsum('mlde,abkm,cl,dj,ei->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 3), (2, 3), (1, 2), (0, 1)])

    #	 -1.0000 P(b,c)<m,l||d,e>_bbbb*t1_bb(a,l)*t1_bb(b,m)*t2_bbbb(e,c,j,k)*t1_bb(d,i)
    contracted_intermediate = -1.000000000000000 * einsum('mlde,al,bm,ecjk,di->abcijk', g_bbbb[ob, ob, vb, vb], t1_bb, t1_bb, t2_bbbb, t1_bb, optimize=['einsum_path', (0, 4), (2, 3), (0, 2), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->acbijk', contracted_intermediate)

    #	 -1.0000 <m,l||d,e>_bbbb*t2_bbbb(e,a,j,k)*t1_bb(b,l)*t1_bb(c,m)*t1_bb(d,i)
    triples_res += -1.000000000000000 * einsum('mlde,eajk,bl,cm,di->abcijk', g_bbbb[ob, ob, vb, vb], t2_bbbb, t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 4), (0, 3), (0, 2), (0, 1)])

    return triples_res