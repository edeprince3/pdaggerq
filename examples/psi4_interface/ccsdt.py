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
    
    o = oa
    v = va
    
    #	  1.0000 f_aa(a,i)
    singles_res =  1.000000000000000 * einsum('ai', f_aa[v, o])
    
    #	 -1.0000 f_aa(j,i)*t1_aa(a,j)
    singles_res += -1.000000000000000 * einsum('ji,aj', f_aa[o, o], t1_aa)
    
    #	  1.0000 f_aa(a,b)*t1_aa(b,i)
    singles_res +=  1.000000000000000 * einsum('ab,bi', f_aa[v, v], t1_aa)
    
    #	 -1.0000 f_aa(j,b)*t2_aaaa(b,a,i,j)
    singles_res += -1.000000000000000 * einsum('jb,baij', f_aa[o, v], t2_aaaa)
    
    #	  1.0000 f_bb(j,b)*t2_abab(a,b,i,j)
    singles_res +=  1.000000000000000 * einsum('jb,abij', f_bb[o, v], t2_abab)
    
    #	 -1.0000 f_aa(j,b)*t1_aa(b,i)*t1_aa(a,j)
    singles_res += -1.000000000000000 * einsum('jb,bi,aj', f_aa[o, v], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,a||b,i>_aaaa*t1_aa(b,j)
    singles_res +=  1.000000000000000 * einsum('jabi,bj', g_aaaa[o, v, v, o], t1_aa)
    
    #	  1.0000 <a,j||i,b>_abab*t1_bb(b,j)
    singles_res +=  1.000000000000000 * einsum('ajib,bj', g_abab[v, o, o, v], t1_bb)
    
    #	 -0.5000 <k,j||b,i>_aaaa*t2_aaaa(b,a,k,j)
    singles_res += -0.500000000000000 * einsum('kjbi,bakj', g_aaaa[o, o, v, o], t2_aaaa)
    
    #	 -0.5000 <k,j||i,b>_abab*t2_abab(a,b,k,j)
    singles_res += -0.500000000000000 * einsum('kjib,abkj', g_abab[o, o, o, v], t2_abab)
    
    #	 -0.5000 <j,k||i,b>_abab*t2_abab(a,b,j,k)
    singles_res += -0.500000000000000 * einsum('jkib,abjk', g_abab[o, o, o, v], t2_abab)
    
    #	 -0.5000 <j,a||b,c>_aaaa*t2_aaaa(b,c,i,j)
    singles_res += -0.500000000000000 * einsum('jabc,bcij', g_aaaa[o, v, v, v], t2_aaaa)
    
    #	  0.5000 <a,j||b,c>_abab*t2_abab(b,c,i,j)
    singles_res +=  0.500000000000000 * einsum('ajbc,bcij', g_abab[v, o, v, v], t2_abab)
    
    #	  0.5000 <a,j||c,b>_abab*t2_abab(c,b,i,j)
    singles_res +=  0.500000000000000 * einsum('ajcb,cbij', g_abab[v, o, v, v], t2_abab)
    
    #	  0.2500 <k,j||b,c>_aaaa*t3_aaaaaa(b,c,a,i,k,j)
    singles_res +=  0.250000000000000 * einsum('kjbc,bcaikj', g_aaaa[o, o, v, v], t3_aaaaaa)
    
    #	 -0.2500 <k,j||b,c>_abab*t3_aabaab(b,a,c,i,k,j)
    singles_res += -0.250000000000000 * einsum('kjbc,bacikj', g_abab[o, o, v, v], t3_aabaab)
    
    #	 -0.2500 <j,k||b,c>_abab*t3_aabaab(b,a,c,i,j,k)
    singles_res += -0.250000000000000 * einsum('jkbc,bacijk', g_abab[o, o, v, v], t3_aabaab)
    
    #	  0.2500 <k,j||c,b>_abab*t3_aabaab(a,c,b,i,k,j)
    singles_res +=  0.250000000000000 * einsum('kjcb,acbikj', g_abab[o, o, v, v], t3_aabaab)
    
    #	  0.2500 <j,k||c,b>_abab*t3_aabaab(a,c,b,i,j,k)
    singles_res +=  0.250000000000000 * einsum('jkcb,acbijk', g_abab[o, o, v, v], t3_aabaab)
    
    #	 -0.2500 <k,j||b,c>_bbbb*t3_abbabb(a,c,b,i,k,j)
    singles_res += -0.250000000000000 * einsum('kjbc,acbikj', g_bbbb[o, o, v, v], t3_abbabb)
    
    #	  1.0000 <k,j||b,c>_aaaa*t1_aa(b,j)*t2_aaaa(c,a,i,k)
    singles_res +=  1.000000000000000 * einsum('kjbc,bj,caik', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,k||b,c>_abab*t1_aa(b,j)*t2_abab(a,c,i,k)
    singles_res +=  1.000000000000000 * einsum('jkbc,bj,acik', g_abab[o, o, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,j||c,b>_abab*t1_bb(b,j)*t2_aaaa(c,a,i,k)
    singles_res += -1.000000000000000 * einsum('kjcb,bj,caik', g_abab[o, o, v, v], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,j||b,c>_bbbb*t1_bb(b,j)*t2_abab(a,c,i,k)
    singles_res += -1.000000000000000 * einsum('kjbc,bj,acik', g_bbbb[o, o, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,j||b,c>_aaaa*t1_aa(b,i)*t2_aaaa(c,a,k,j)
    singles_res +=  0.500000000000000 * einsum('kjbc,bi,cakj', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <k,j||b,c>_abab*t1_aa(b,i)*t2_abab(a,c,k,j)
    singles_res += -0.500000000000000 * einsum('kjbc,bi,ackj', g_abab[o, o, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <j,k||b,c>_abab*t1_aa(b,i)*t2_abab(a,c,j,k)
    singles_res += -0.500000000000000 * einsum('jkbc,bi,acjk', g_abab[o, o, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <k,j||b,c>_aaaa*t1_aa(a,j)*t2_aaaa(b,c,i,k)
    singles_res +=  0.500000000000000 * einsum('kjbc,aj,bcik', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <j,k||b,c>_abab*t1_aa(a,j)*t2_abab(b,c,i,k)
    singles_res += -0.500000000000000 * einsum('jkbc,aj,bcik', g_abab[o, o, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <j,k||c,b>_abab*t1_aa(a,j)*t2_abab(c,b,i,k)
    singles_res += -0.500000000000000 * einsum('jkcb,aj,cbik', g_abab[o, o, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <k,j||b,i>_aaaa*t1_aa(b,j)*t1_aa(a,k)
    singles_res +=  1.000000000000000 * einsum('kjbi,bj,ak', g_aaaa[o, o, v, o], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,j||i,b>_abab*t1_bb(b,j)*t1_aa(a,k)
    singles_res += -1.000000000000000 * einsum('kjib,bj,ak', g_abab[o, o, o, v], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,a||b,c>_aaaa*t1_aa(b,j)*t1_aa(c,i)
    singles_res +=  1.000000000000000 * einsum('jabc,bj,ci', g_aaaa[o, v, v, v], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,j||c,b>_abab*t1_bb(b,j)*t1_aa(c,i)
    singles_res +=  1.000000000000000 * einsum('ajcb,bj,ci', g_abab[v, o, v, v], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,j||b,c>_aaaa*t1_aa(b,j)*t1_aa(c,i)*t1_aa(a,k)
    singles_res +=  1.000000000000000 * einsum('kjbc,bj,ci,ak', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <k,j||c,b>_abab*t1_bb(b,j)*t1_aa(c,i)*t1_aa(a,k)
    singles_res += -1.000000000000000 * einsum('kjcb,bj,ci,ak', g_abab[o, o, v, v], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    return singles_res
    
    
def ccsdt_t1_bb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):
    
    #    < 0 | i* a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	  1.0000 f_bb(a,i)
    singles_res =  1.000000000000000 * einsum('ai', f_bb[v, o])
    
    #	 -1.0000 f_bb(j,i)*t1_bb(a,j)
    singles_res += -1.000000000000000 * einsum('ji,aj', f_bb[o, o], t1_bb)
    
    #	  1.0000 f_bb(a,b)*t1_bb(b,i)
    singles_res +=  1.000000000000000 * einsum('ab,bi', f_bb[v, v], t1_bb)
    
    #	  1.0000 f_aa(j,b)*t2_abab(b,a,j,i)
    singles_res +=  1.000000000000000 * einsum('jb,baji', f_aa[o, v], t2_abab)
    
    #	 -1.0000 f_bb(j,b)*t2_bbbb(b,a,i,j)
    singles_res += -1.000000000000000 * einsum('jb,baij', f_bb[o, v], t2_bbbb)
    
    #	 -1.0000 f_bb(j,b)*t1_bb(b,i)*t1_bb(a,j)
    singles_res += -1.000000000000000 * einsum('jb,bi,aj', f_bb[o, v], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,a||b,i>_abab*t1_aa(b,j)
    singles_res +=  1.000000000000000 * einsum('jabi,bj', g_abab[o, v, v, o], t1_aa)
    
    #	  1.0000 <j,a||b,i>_bbbb*t1_bb(b,j)
    singles_res +=  1.000000000000000 * einsum('jabi,bj', g_bbbb[o, v, v, o], t1_bb)
    
    #	 -0.5000 <k,j||b,i>_abab*t2_abab(b,a,k,j)
    singles_res += -0.500000000000000 * einsum('kjbi,bakj', g_abab[o, o, v, o], t2_abab)
    
    #	 -0.5000 <j,k||b,i>_abab*t2_abab(b,a,j,k)
    singles_res += -0.500000000000000 * einsum('jkbi,bajk', g_abab[o, o, v, o], t2_abab)
    
    #	 -0.5000 <k,j||b,i>_bbbb*t2_bbbb(b,a,k,j)
    singles_res += -0.500000000000000 * einsum('kjbi,bakj', g_bbbb[o, o, v, o], t2_bbbb)
    
    #	  0.5000 <j,a||b,c>_abab*t2_abab(b,c,j,i)
    singles_res +=  0.500000000000000 * einsum('jabc,bcji', g_abab[o, v, v, v], t2_abab)
    
    #	  0.5000 <j,a||c,b>_abab*t2_abab(c,b,j,i)
    singles_res +=  0.500000000000000 * einsum('jacb,cbji', g_abab[o, v, v, v], t2_abab)
    
    #	 -0.5000 <j,a||b,c>_bbbb*t2_bbbb(b,c,i,j)
    singles_res += -0.500000000000000 * einsum('jabc,bcij', g_bbbb[o, v, v, v], t2_bbbb)
    
    #	 -0.2500 <k,j||b,c>_aaaa*t3_aabaab(b,c,a,j,k,i)
    singles_res += -0.250000000000000 * einsum('kjbc,bcajki', g_aaaa[o, o, v, v], t3_aabaab)
    
    #	 -0.2500 <k,j||b,c>_abab*t3_abbabb(b,c,a,k,i,j)
    singles_res += -0.250000000000000 * einsum('kjbc,bcakij', g_abab[o, o, v, v], t3_abbabb)
    
    #	  0.2500 <j,k||b,c>_abab*t3_abbabb(b,c,a,j,k,i)
    singles_res +=  0.250000000000000 * einsum('jkbc,bcajki', g_abab[o, o, v, v], t3_abbabb)
    
    #	 -0.2500 <k,j||c,b>_abab*t3_abbabb(c,b,a,k,i,j)
    singles_res += -0.250000000000000 * einsum('kjcb,cbakij', g_abab[o, o, v, v], t3_abbabb)
    
    #	  0.2500 <j,k||c,b>_abab*t3_abbabb(c,b,a,j,k,i)
    singles_res +=  0.250000000000000 * einsum('jkcb,cbajki', g_abab[o, o, v, v], t3_abbabb)
    
    #	  0.2500 <k,j||b,c>_bbbb*t3_bbbbbb(b,c,a,i,k,j)
    singles_res +=  0.250000000000000 * einsum('kjbc,bcaikj', g_bbbb[o, o, v, v], t3_bbbbbb)
    
    #	 -1.0000 <k,j||b,c>_aaaa*t1_aa(b,j)*t2_abab(c,a,k,i)
    singles_res += -1.000000000000000 * einsum('kjbc,bj,caki', g_aaaa[o, o, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <j,k||b,c>_abab*t1_aa(b,j)*t2_bbbb(c,a,i,k)
    singles_res += -1.000000000000000 * einsum('jkbc,bj,caik', g_abab[o, o, v, v], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,j||c,b>_abab*t1_bb(b,j)*t2_abab(c,a,k,i)
    singles_res +=  1.000000000000000 * einsum('kjcb,bj,caki', g_abab[o, o, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,j||b,c>_bbbb*t1_bb(b,j)*t2_bbbb(c,a,i,k)
    singles_res +=  1.000000000000000 * einsum('kjbc,bj,caik', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,j||c,b>_abab*t1_bb(b,i)*t2_abab(c,a,k,j)
    singles_res += -0.500000000000000 * einsum('kjcb,bi,cakj', g_abab[o, o, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <j,k||c,b>_abab*t1_bb(b,i)*t2_abab(c,a,j,k)
    singles_res += -0.500000000000000 * einsum('jkcb,bi,cajk', g_abab[o, o, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <k,j||b,c>_bbbb*t1_bb(b,i)*t2_bbbb(c,a,k,j)
    singles_res +=  0.500000000000000 * einsum('kjbc,bi,cakj', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <k,j||b,c>_abab*t1_bb(a,j)*t2_abab(b,c,k,i)
    singles_res += -0.500000000000000 * einsum('kjbc,aj,bcki', g_abab[o, o, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <k,j||c,b>_abab*t1_bb(a,j)*t2_abab(c,b,k,i)
    singles_res += -0.500000000000000 * einsum('kjcb,aj,cbki', g_abab[o, o, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <k,j||b,c>_bbbb*t1_bb(a,j)*t2_bbbb(b,c,i,k)
    singles_res +=  0.500000000000000 * einsum('kjbc,aj,bcik', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <j,k||b,i>_abab*t1_aa(b,j)*t1_bb(a,k)
    singles_res += -1.000000000000000 * einsum('jkbi,bj,ak', g_abab[o, o, v, o], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,j||b,i>_bbbb*t1_bb(b,j)*t1_bb(a,k)
    singles_res +=  1.000000000000000 * einsum('kjbi,bj,ak', g_bbbb[o, o, v, o], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,a||b,c>_abab*t1_aa(b,j)*t1_bb(c,i)
    singles_res +=  1.000000000000000 * einsum('jabc,bj,ci', g_abab[o, v, v, v], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,a||b,c>_bbbb*t1_bb(b,j)*t1_bb(c,i)
    singles_res +=  1.000000000000000 * einsum('jabc,bj,ci', g_bbbb[o, v, v, v], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <j,k||b,c>_abab*t1_aa(b,j)*t1_bb(c,i)*t1_bb(a,k)
    singles_res += -1.000000000000000 * einsum('jkbc,bj,ci,ak', g_abab[o, o, v, v], t1_aa, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <k,j||b,c>_bbbb*t1_bb(b,j)*t1_bb(c,i)*t1_bb(a,k)
    singles_res +=  1.000000000000000 * einsum('kjbc,bj,ci,ak', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    return singles_res
    
    
def ccsdt_t2_aaaa_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):
    
    #    < 0 | i* j* b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 f_aa(k,j)*t2_aaaa(a,b,i,k)
    doubles_res = -1.000000000000000 * einsum('kj,abik', f_aa[o, o], t2_aaaa)
    
    #	  1.0000 f_aa(k,i)*t2_aaaa(a,b,j,k)
    doubles_res +=  1.000000000000000 * einsum('ki,abjk', f_aa[o, o], t2_aaaa)
    
    #	  1.0000 f_aa(a,c)*t2_aaaa(c,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('ac,cbij', f_aa[v, v], t2_aaaa)
    
    #	 -1.0000 f_aa(b,c)*t2_aaaa(c,a,i,j)
    doubles_res += -1.000000000000000 * einsum('bc,caij', f_aa[v, v], t2_aaaa)
    
    #	  1.0000 f_aa(k,c)*t3_aaaaaa(c,a,b,i,j,k)
    doubles_res +=  1.000000000000000 * einsum('kc,cabijk', f_aa[o, v], t3_aaaaaa)
    
    #	 -1.0000 f_bb(k,c)*t3_aabaab(b,a,c,i,j,k)
    doubles_res += -1.000000000000000 * einsum('kc,bacijk', f_bb[o, v], t3_aabaab)
    
    #	 -1.0000 f_aa(k,c)*t1_aa(c,j)*t2_aaaa(a,b,i,k)
    doubles_res += -1.000000000000000 * einsum('kc,cj,abik', f_aa[o, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(k,c)*t1_aa(c,i)*t2_aaaa(a,b,j,k)
    doubles_res +=  1.000000000000000 * einsum('kc,ci,abjk', f_aa[o, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(k,c)*t1_aa(a,k)*t2_aaaa(c,b,i,j)
    doubles_res += -1.000000000000000 * einsum('kc,ak,cbij', f_aa[o, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(k,c)*t1_aa(b,k)*t2_aaaa(c,a,i,j)
    doubles_res +=  1.000000000000000 * einsum('kc,bk,caij', f_aa[o, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,b||i,j>_aaaa
    doubles_res +=  1.000000000000000 * einsum('abij', g_aaaa[v, v, o, o])
    
    #	  1.0000 <k,a||i,j>_aaaa*t1_aa(b,k)
    doubles_res +=  1.000000000000000 * einsum('kaij,bk', g_aaaa[o, v, o, o], t1_aa)
    
    #	 -1.0000 <k,b||i,j>_aaaa*t1_aa(a,k)
    doubles_res += -1.000000000000000 * einsum('kbij,ak', g_aaaa[o, v, o, o], t1_aa)
    
    #	  1.0000 <a,b||c,j>_aaaa*t1_aa(c,i)
    doubles_res +=  1.000000000000000 * einsum('abcj,ci', g_aaaa[v, v, v, o], t1_aa)
    
    #	 -1.0000 <a,b||c,i>_aaaa*t1_aa(c,j)
    doubles_res += -1.000000000000000 * einsum('abci,cj', g_aaaa[v, v, v, o], t1_aa)
    
    #	  0.5000 <l,k||i,j>_aaaa*t2_aaaa(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkij,ablk', g_aaaa[o, o, o, o], t2_aaaa)
    
    #	  1.0000 <k,a||c,j>_aaaa*t2_aaaa(c,b,i,k)
    doubles_res +=  1.000000000000000 * einsum('kacj,cbik', g_aaaa[o, v, v, o], t2_aaaa)
    
    #	 -1.0000 <a,k||j,c>_abab*t2_abab(b,c,i,k)
    doubles_res += -1.000000000000000 * einsum('akjc,bcik', g_abab[v, o, o, v], t2_abab)
    
    #	 -1.0000 <k,a||c,i>_aaaa*t2_aaaa(c,b,j,k)
    doubles_res += -1.000000000000000 * einsum('kaci,cbjk', g_aaaa[o, v, v, o], t2_aaaa)
    
    #	  1.0000 <a,k||i,c>_abab*t2_abab(b,c,j,k)
    doubles_res +=  1.000000000000000 * einsum('akic,bcjk', g_abab[v, o, o, v], t2_abab)
    
    #	 -1.0000 <k,b||c,j>_aaaa*t2_aaaa(c,a,i,k)
    doubles_res += -1.000000000000000 * einsum('kbcj,caik', g_aaaa[o, v, v, o], t2_aaaa)
    
    #	  1.0000 <b,k||j,c>_abab*t2_abab(a,c,i,k)
    doubles_res +=  1.000000000000000 * einsum('bkjc,acik', g_abab[v, o, o, v], t2_abab)
    
    #	  1.0000 <k,b||c,i>_aaaa*t2_aaaa(c,a,j,k)
    doubles_res +=  1.000000000000000 * einsum('kbci,cajk', g_aaaa[o, v, v, o], t2_aaaa)
    
    #	 -1.0000 <b,k||i,c>_abab*t2_abab(a,c,j,k)
    doubles_res += -1.000000000000000 * einsum('bkic,acjk', g_abab[v, o, o, v], t2_abab)
    
    #	  0.5000 <a,b||c,d>_aaaa*t2_aaaa(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('abcd,cdij', g_aaaa[v, v, v, v], t2_aaaa)
    
    #	  0.5000 <l,k||c,j>_aaaa*t3_aaaaaa(c,a,b,i,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcj,cabilk', g_aaaa[o, o, v, o], t3_aaaaaa)
    
    #	  0.5000 <l,k||j,c>_abab*t3_aabaab(b,a,c,i,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkjc,bacilk', g_abab[o, o, o, v], t3_aabaab)
    
    #	  0.5000 <k,l||j,c>_abab*t3_aabaab(b,a,c,i,k,l)
    doubles_res +=  0.500000000000000 * einsum('kljc,bacikl', g_abab[o, o, o, v], t3_aabaab)
    
    #	 -0.5000 <l,k||c,i>_aaaa*t3_aaaaaa(c,a,b,j,l,k)
    doubles_res += -0.500000000000000 * einsum('lkci,cabjlk', g_aaaa[o, o, v, o], t3_aaaaaa)
    
    #	 -0.5000 <l,k||i,c>_abab*t3_aabaab(b,a,c,j,l,k)
    doubles_res += -0.500000000000000 * einsum('lkic,bacjlk', g_abab[o, o, o, v], t3_aabaab)
    
    #	 -0.5000 <k,l||i,c>_abab*t3_aabaab(b,a,c,j,k,l)
    doubles_res += -0.500000000000000 * einsum('klic,bacjkl', g_abab[o, o, o, v], t3_aabaab)
    
    #	  0.5000 <k,a||c,d>_aaaa*t3_aaaaaa(c,d,b,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('kacd,cdbijk', g_aaaa[o, v, v, v], t3_aaaaaa)
    
    #	  0.5000 <a,k||c,d>_abab*t3_aabaab(c,b,d,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('akcd,cbdijk', g_abab[v, o, v, v], t3_aabaab)
    
    #	 -0.5000 <a,k||d,c>_abab*t3_aabaab(b,d,c,i,j,k)
    doubles_res += -0.500000000000000 * einsum('akdc,bdcijk', g_abab[v, o, v, v], t3_aabaab)
    
    #	 -0.5000 <k,b||c,d>_aaaa*t3_aaaaaa(c,d,a,i,j,k)
    doubles_res += -0.500000000000000 * einsum('kbcd,cdaijk', g_aaaa[o, v, v, v], t3_aaaaaa)
    
    #	 -0.5000 <b,k||c,d>_abab*t3_aabaab(c,a,d,i,j,k)
    doubles_res += -0.500000000000000 * einsum('bkcd,cadijk', g_abab[v, o, v, v], t3_aabaab)
    
    #	  0.5000 <b,k||d,c>_abab*t3_aabaab(a,d,c,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('bkdc,adcijk', g_abab[v, o, v, v], t3_aabaab)
    
    #	  1.0000 <l,k||c,j>_aaaa*t1_aa(c,k)*t2_aaaa(a,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcj,ck,abil', g_aaaa[o, o, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||j,c>_abab*t1_bb(c,k)*t2_aaaa(a,b,i,l)
    doubles_res += -1.000000000000000 * einsum('lkjc,ck,abil', g_abab[o, o, o, v], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,j>_aaaa*t1_aa(c,i)*t2_aaaa(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcj,ci,ablk', g_aaaa[o, o, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,j>_aaaa*t1_aa(a,k)*t2_aaaa(c,b,i,l)
    doubles_res += -1.000000000000000 * einsum('lkcj,ak,cbil', g_aaaa[o, o, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,l||j,c>_abab*t1_aa(a,k)*t2_abab(b,c,i,l)
    doubles_res +=  1.000000000000000 * einsum('kljc,ak,bcil', g_abab[o, o, o, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,j>_aaaa*t1_aa(b,k)*t2_aaaa(c,a,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcj,bk,cail', g_aaaa[o, o, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,l||j,c>_abab*t1_aa(b,k)*t2_abab(a,c,i,l)
    doubles_res += -1.000000000000000 * einsum('kljc,bk,acil', g_abab[o, o, o, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,i>_aaaa*t1_aa(c,k)*t2_aaaa(a,b,j,l)
    doubles_res += -1.000000000000000 * einsum('lkci,ck,abjl', g_aaaa[o, o, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||i,c>_abab*t1_bb(c,k)*t2_aaaa(a,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('lkic,ck,abjl', g_abab[o, o, o, v], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,i>_aaaa*t1_aa(c,j)*t2_aaaa(a,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkci,cj,ablk', g_aaaa[o, o, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,i>_aaaa*t1_aa(a,k)*t2_aaaa(c,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('lkci,ak,cbjl', g_aaaa[o, o, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,l||i,c>_abab*t1_aa(a,k)*t2_abab(b,c,j,l)
    doubles_res += -1.000000000000000 * einsum('klic,ak,bcjl', g_abab[o, o, o, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,i>_aaaa*t1_aa(b,k)*t2_aaaa(c,a,j,l)
    doubles_res += -1.000000000000000 * einsum('lkci,bk,cajl', g_aaaa[o, o, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,l||i,c>_abab*t1_aa(b,k)*t2_abab(a,c,j,l)
    doubles_res +=  1.000000000000000 * einsum('klic,bk,acjl', g_abab[o, o, o, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,a||c,d>_aaaa*t1_aa(c,k)*t2_aaaa(d,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('kacd,ck,dbij', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,k||d,c>_abab*t1_bb(c,k)*t2_aaaa(d,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('akdc,ck,dbij', g_abab[v, o, v, v], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,a||c,d>_aaaa*t1_aa(c,j)*t2_aaaa(d,b,i,k)
    doubles_res += -1.000000000000000 * einsum('kacd,cj,dbik', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,k||c,d>_abab*t1_aa(c,j)*t2_abab(b,d,i,k)
    doubles_res += -1.000000000000000 * einsum('akcd,cj,bdik', g_abab[v, o, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,a||c,d>_aaaa*t1_aa(c,i)*t2_aaaa(d,b,j,k)
    doubles_res +=  1.000000000000000 * einsum('kacd,ci,dbjk', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,k||c,d>_abab*t1_aa(c,i)*t2_abab(b,d,j,k)
    doubles_res +=  1.000000000000000 * einsum('akcd,ci,bdjk', g_abab[v, o, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,a||c,d>_aaaa*t1_aa(b,k)*t2_aaaa(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('kacd,bk,cdij', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,b||c,d>_aaaa*t1_aa(c,k)*t2_aaaa(d,a,i,j)
    doubles_res += -1.000000000000000 * einsum('kbcd,ck,daij', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,k||d,c>_abab*t1_bb(c,k)*t2_aaaa(d,a,i,j)
    doubles_res += -1.000000000000000 * einsum('bkdc,ck,daij', g_abab[v, o, v, v], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,b||c,d>_aaaa*t1_aa(c,j)*t2_aaaa(d,a,i,k)
    doubles_res +=  1.000000000000000 * einsum('kbcd,cj,daik', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,k||c,d>_abab*t1_aa(c,j)*t2_abab(a,d,i,k)
    doubles_res +=  1.000000000000000 * einsum('bkcd,cj,adik', g_abab[v, o, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,b||c,d>_aaaa*t1_aa(c,i)*t2_aaaa(d,a,j,k)
    doubles_res += -1.000000000000000 * einsum('kbcd,ci,dajk', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,k||c,d>_abab*t1_aa(c,i)*t2_abab(a,d,j,k)
    doubles_res += -1.000000000000000 * einsum('bkcd,ci,adjk', g_abab[v, o, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,b||c,d>_aaaa*t1_aa(a,k)*t2_aaaa(c,d,i,j)
    doubles_res += -0.500000000000000 * einsum('kbcd,ak,cdij', g_aaaa[o, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_aaaa*t1_aa(c,k)*t3_aaaaaa(d,a,b,i,j,l)
    doubles_res += -1.000000000000000 * einsum('lkcd,ck,dabijl', g_aaaa[o, o, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,l||c,d>_abab*t1_aa(c,k)*t3_aabaab(b,a,d,i,j,l)
    doubles_res += -1.000000000000000 * einsum('klcd,ck,badijl', g_abab[o, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||d,c>_abab*t1_bb(c,k)*t3_aaaaaa(d,a,b,i,j,l)
    doubles_res +=  1.000000000000000 * einsum('lkdc,ck,dabijl', g_abab[o, o, v, v], t1_bb, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t1_bb(c,k)*t3_aabaab(b,a,d,i,j,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ck,badijl', g_bbbb[o, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_aaaa*t1_aa(c,j)*t3_aaaaaa(d,a,b,i,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,cj,dabilk', g_aaaa[o, o, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_abab*t1_aa(c,j)*t3_aabaab(b,a,d,i,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcd,cj,badilk', g_abab[o, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,l||c,d>_abab*t1_aa(c,j)*t3_aabaab(b,a,d,i,k,l)
    doubles_res +=  0.500000000000000 * einsum('klcd,cj,badikl', g_abab[o, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_aaaa*t1_aa(c,i)*t3_aaaaaa(d,a,b,j,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcd,ci,dabjlk', g_aaaa[o, o, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_abab*t1_aa(c,i)*t3_aabaab(b,a,d,j,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,ci,badjlk', g_abab[o, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||c,d>_abab*t1_aa(c,i)*t3_aabaab(b,a,d,j,k,l)
    doubles_res += -0.500000000000000 * einsum('klcd,ci,badjkl', g_abab[o, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_aaaa*t1_aa(a,k)*t3_aaaaaa(c,d,b,i,j,l)
    doubles_res += -0.500000000000000 * einsum('lkcd,ak,cdbijl', g_aaaa[o, o, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||c,d>_abab*t1_aa(a,k)*t3_aabaab(c,b,d,i,j,l)
    doubles_res += -0.500000000000000 * einsum('klcd,ak,cbdijl', g_abab[o, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,l||d,c>_abab*t1_aa(a,k)*t3_aabaab(b,d,c,i,j,l)
    doubles_res +=  0.500000000000000 * einsum('kldc,ak,bdcijl', g_abab[o, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_aaaa*t1_aa(b,k)*t3_aaaaaa(c,d,a,i,j,l)
    doubles_res +=  0.500000000000000 * einsum('lkcd,bk,cdaijl', g_aaaa[o, o, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,l||c,d>_abab*t1_aa(b,k)*t3_aabaab(c,a,d,i,j,l)
    doubles_res +=  0.500000000000000 * einsum('klcd,bk,cadijl', g_abab[o, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||d,c>_abab*t1_aa(b,k)*t3_aabaab(a,d,c,i,j,l)
    doubles_res += -0.500000000000000 * einsum('kldc,bk,adcijl', g_abab[o, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||i,j>_aaaa*t1_aa(a,k)*t1_aa(b,l)
    doubles_res += -1.000000000000000 * einsum('lkij,ak,bl', g_aaaa[o, o, o, o], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,a||c,j>_aaaa*t1_aa(c,i)*t1_aa(b,k)
    doubles_res +=  1.000000000000000 * einsum('kacj,ci,bk', g_aaaa[o, v, v, o], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,a||c,i>_aaaa*t1_aa(c,j)*t1_aa(b,k)
    doubles_res += -1.000000000000000 * einsum('kaci,cj,bk', g_aaaa[o, v, v, o], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,b||c,j>_aaaa*t1_aa(c,i)*t1_aa(a,k)
    doubles_res += -1.000000000000000 * einsum('kbcj,ci,ak', g_aaaa[o, v, v, o], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,b||c,i>_aaaa*t1_aa(c,j)*t1_aa(a,k)
    doubles_res +=  1.000000000000000 * einsum('kbci,cj,ak', g_aaaa[o, v, v, o], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,b||c,d>_aaaa*t1_aa(c,j)*t1_aa(d,i)
    doubles_res += -1.000000000000000 * einsum('abcd,cj,di', g_aaaa[v, v, v, v], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_aaaa*t2_aaaa(c,d,j,k)*t2_aaaa(a,b,i,l)
    doubles_res += -0.500000000000000 * einsum('lkcd,cdjk,abil', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_abab*t2_abab(c,d,j,k)*t2_aaaa(a,b,i,l)
    doubles_res += -0.500000000000000 * einsum('lkcd,cdjk,abil', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||d,c>_abab*t2_abab(d,c,j,k)*t2_aaaa(a,b,i,l)
    doubles_res += -0.500000000000000 * einsum('lkdc,dcjk,abil', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_aaaa*t2_aaaa(c,d,i,k)*t2_aaaa(a,b,j,l)
    doubles_res +=  0.500000000000000 * einsum('lkcd,cdik,abjl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_abab*t2_abab(c,d,i,k)*t2_aaaa(a,b,j,l)
    doubles_res +=  0.500000000000000 * einsum('lkcd,cdik,abjl', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||d,c>_abab*t2_abab(d,c,i,k)*t2_aaaa(a,b,j,l)
    doubles_res +=  0.500000000000000 * einsum('lkdc,dcik,abjl', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,k||c,d>_aaaa*t2_aaaa(c,d,i,j)*t2_aaaa(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdij,ablk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_aaaa*t2_aaaa(c,a,l,k)*t2_aaaa(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||d,c>_abab*t2_abab(a,c,l,k)*t2_aaaa(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkdc,aclk,dbij', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||d,c>_abab*t2_abab(a,c,k,l)*t2_aaaa(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('kldc,ackl,dbij', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t2_aaaa(c,a,j,k)*t2_aaaa(d,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,cajk,dbil', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,l||c,d>_abab*t2_aaaa(c,a,j,k)*t2_abab(b,d,i,l)
    doubles_res +=  1.000000000000000 * einsum('klcd,cajk,bdil', g_abab[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||d,c>_abab*t2_abab(a,c,j,k)*t2_aaaa(d,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkdc,acjk,dbil', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t2_abab(a,c,j,k)*t2_abab(b,d,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,acjk,bdil', g_bbbb[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_aaaa*t2_aaaa(c,a,i,k)*t2_aaaa(d,b,j,l)
    doubles_res += -1.000000000000000 * einsum('lkcd,caik,dbjl', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,l||c,d>_abab*t2_aaaa(c,a,i,k)*t2_abab(b,d,j,l)
    doubles_res += -1.000000000000000 * einsum('klcd,caik,bdjl', g_abab[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||d,c>_abab*t2_abab(a,c,i,k)*t2_aaaa(d,b,j,l)
    doubles_res += -1.000000000000000 * einsum('lkdc,acik,dbjl', g_abab[o, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_bbbb*t2_abab(a,c,i,k)*t2_abab(b,d,j,l)
    doubles_res += -1.000000000000000 * einsum('lkcd,acik,bdjl', g_bbbb[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_aaaa*t2_aaaa(c,a,i,j)*t2_aaaa(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,caij,dblk', g_aaaa[o, o, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_abab*t2_aaaa(c,a,i,j)*t2_abab(b,d,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcd,caij,bdlk', g_abab[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <k,l||c,d>_abab*t2_aaaa(c,a,i,j)*t2_abab(b,d,k,l)
    doubles_res +=  0.500000000000000 * einsum('klcd,caij,bdkl', g_abab[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t1_aa(c,k)*t1_aa(d,j)*t2_aaaa(a,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ck,dj,abil', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||d,c>_abab*t1_bb(c,k)*t1_aa(d,j)*t2_aaaa(a,b,i,l)
    doubles_res += -1.000000000000000 * einsum('lkdc,ck,dj,abil', g_abab[o, o, v, v], t1_bb, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_aaaa*t1_aa(c,k)*t1_aa(d,i)*t2_aaaa(a,b,j,l)
    doubles_res += -1.000000000000000 * einsum('lkcd,ck,di,abjl', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||d,c>_abab*t1_bb(c,k)*t1_aa(d,i)*t2_aaaa(a,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('lkdc,ck,di,abjl', g_abab[o, o, v, v], t1_bb, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t1_aa(c,k)*t1_aa(a,l)*t2_aaaa(d,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ck,al,dbij', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||d,c>_abab*t1_bb(c,k)*t1_aa(a,l)*t2_aaaa(d,b,i,j)
    doubles_res += -1.000000000000000 * einsum('lkdc,ck,al,dbij', g_abab[o, o, v, v], t1_bb, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_aaaa*t1_aa(c,k)*t1_aa(b,l)*t2_aaaa(d,a,i,j)
    doubles_res += -1.000000000000000 * einsum('lkcd,ck,bl,daij', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||d,c>_abab*t1_bb(c,k)*t1_aa(b,l)*t2_aaaa(d,a,i,j)
    doubles_res +=  1.000000000000000 * einsum('lkdc,ck,bl,daij', g_abab[o, o, v, v], t1_bb, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_aaaa*t1_aa(c,j)*t1_aa(d,i)*t2_aaaa(a,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,cj,di,ablk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t1_aa(c,j)*t1_aa(a,k)*t2_aaaa(d,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,cj,ak,dbil', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <k,l||c,d>_abab*t1_aa(c,j)*t1_aa(a,k)*t2_abab(b,d,i,l)
    doubles_res +=  1.000000000000000 * einsum('klcd,cj,ak,bdil', g_abab[o, o, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_aaaa*t1_aa(c,j)*t1_aa(b,k)*t2_aaaa(d,a,i,l)
    doubles_res += -1.000000000000000 * einsum('lkcd,cj,bk,dail', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <k,l||c,d>_abab*t1_aa(c,j)*t1_aa(b,k)*t2_abab(a,d,i,l)
    doubles_res += -1.000000000000000 * einsum('klcd,cj,bk,adil', g_abab[o, o, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_aaaa*t1_aa(c,i)*t1_aa(a,k)*t2_aaaa(d,b,j,l)
    doubles_res += -1.000000000000000 * einsum('lkcd,ci,ak,dbjl', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <k,l||c,d>_abab*t1_aa(c,i)*t1_aa(a,k)*t2_abab(b,d,j,l)
    doubles_res += -1.000000000000000 * einsum('klcd,ci,ak,bdjl', g_abab[o, o, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t1_aa(c,i)*t1_aa(b,k)*t2_aaaa(d,a,j,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ci,bk,dajl', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <k,l||c,d>_abab*t1_aa(c,i)*t1_aa(b,k)*t2_abab(a,d,j,l)
    doubles_res +=  1.000000000000000 * einsum('klcd,ci,bk,adjl', g_abab[o, o, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_aaaa*t1_aa(a,k)*t1_aa(b,l)*t2_aaaa(c,d,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,ak,bl,cdij', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||c,j>_aaaa*t1_aa(c,i)*t1_aa(a,k)*t1_aa(b,l)
    doubles_res += -1.000000000000000 * einsum('lkcj,ci,ak,bl', g_aaaa[o, o, v, o], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,i>_aaaa*t1_aa(c,j)*t1_aa(a,k)*t1_aa(b,l)
    doubles_res +=  1.000000000000000 * einsum('lkci,cj,ak,bl', g_aaaa[o, o, v, o], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <k,a||c,d>_aaaa*t1_aa(c,j)*t1_aa(d,i)*t1_aa(b,k)
    doubles_res += -1.000000000000000 * einsum('kacd,cj,di,bk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <k,b||c,d>_aaaa*t1_aa(c,j)*t1_aa(d,i)*t1_aa(a,k)
    doubles_res +=  1.000000000000000 * einsum('kbcd,cj,di,ak', g_aaaa[o, v, v, v], t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t1_aa(c,j)*t1_aa(d,i)*t1_aa(a,k)*t1_aa(b,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,cj,di,ak,bl', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    return doubles_res
    
    
def ccsdt_t2_bbbb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):
    
    #    < 0 | i* j* b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 f_bb(k,j)*t2_bbbb(a,b,i,k)
    doubles_res = -1.000000000000000 * einsum('kj,abik', f_bb[o, o], t2_bbbb)
    
    #	  1.0000 f_bb(k,i)*t2_bbbb(a,b,j,k)
    doubles_res +=  1.000000000000000 * einsum('ki,abjk', f_bb[o, o], t2_bbbb)
    
    #	  1.0000 f_bb(a,c)*t2_bbbb(c,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('ac,cbij', f_bb[v, v], t2_bbbb)
    
    #	 -1.0000 f_bb(b,c)*t2_bbbb(c,a,i,j)
    doubles_res += -1.000000000000000 * einsum('bc,caij', f_bb[v, v], t2_bbbb)
    
    #	 -1.0000 f_aa(k,c)*t3_abbabb(c,a,b,k,j,i)
    doubles_res += -1.000000000000000 * einsum('kc,cabkji', f_aa[o, v], t3_abbabb)
    
    #	  1.0000 f_bb(k,c)*t3_bbbbbb(c,a,b,i,j,k)
    doubles_res +=  1.000000000000000 * einsum('kc,cabijk', f_bb[o, v], t3_bbbbbb)
    
    #	 -1.0000 f_bb(k,c)*t1_bb(c,j)*t2_bbbb(a,b,i,k)
    doubles_res += -1.000000000000000 * einsum('kc,cj,abik', f_bb[o, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(k,c)*t1_bb(c,i)*t2_bbbb(a,b,j,k)
    doubles_res +=  1.000000000000000 * einsum('kc,ci,abjk', f_bb[o, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(k,c)*t1_bb(a,k)*t2_bbbb(c,b,i,j)
    doubles_res += -1.000000000000000 * einsum('kc,ak,cbij', f_bb[o, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(k,c)*t1_bb(b,k)*t2_bbbb(c,a,i,j)
    doubles_res +=  1.000000000000000 * einsum('kc,bk,caij', f_bb[o, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,b||i,j>_bbbb
    doubles_res +=  1.000000000000000 * einsum('abij', g_bbbb[v, v, o, o])
    
    #	  1.0000 <k,a||i,j>_bbbb*t1_bb(b,k)
    doubles_res +=  1.000000000000000 * einsum('kaij,bk', g_bbbb[o, v, o, o], t1_bb)
    
    #	 -1.0000 <k,b||i,j>_bbbb*t1_bb(a,k)
    doubles_res += -1.000000000000000 * einsum('kbij,ak', g_bbbb[o, v, o, o], t1_bb)
    
    #	  1.0000 <a,b||c,j>_bbbb*t1_bb(c,i)
    doubles_res +=  1.000000000000000 * einsum('abcj,ci', g_bbbb[v, v, v, o], t1_bb)
    
    #	 -1.0000 <a,b||c,i>_bbbb*t1_bb(c,j)
    doubles_res += -1.000000000000000 * einsum('abci,cj', g_bbbb[v, v, v, o], t1_bb)
    
    #	  0.5000 <l,k||i,j>_bbbb*t2_bbbb(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkij,ablk', g_bbbb[o, o, o, o], t2_bbbb)
    
    #	 -1.0000 <k,a||c,j>_abab*t2_abab(c,b,k,i)
    doubles_res += -1.000000000000000 * einsum('kacj,cbki', g_abab[o, v, v, o], t2_abab)
    
    #	  1.0000 <k,a||c,j>_bbbb*t2_bbbb(c,b,i,k)
    doubles_res +=  1.000000000000000 * einsum('kacj,cbik', g_bbbb[o, v, v, o], t2_bbbb)
    
    #	  1.0000 <k,a||c,i>_abab*t2_abab(c,b,k,j)
    doubles_res +=  1.000000000000000 * einsum('kaci,cbkj', g_abab[o, v, v, o], t2_abab)
    
    #	 -1.0000 <k,a||c,i>_bbbb*t2_bbbb(c,b,j,k)
    doubles_res += -1.000000000000000 * einsum('kaci,cbjk', g_bbbb[o, v, v, o], t2_bbbb)
    
    #	  1.0000 <k,b||c,j>_abab*t2_abab(c,a,k,i)
    doubles_res +=  1.000000000000000 * einsum('kbcj,caki', g_abab[o, v, v, o], t2_abab)
    
    #	 -1.0000 <k,b||c,j>_bbbb*t2_bbbb(c,a,i,k)
    doubles_res += -1.000000000000000 * einsum('kbcj,caik', g_bbbb[o, v, v, o], t2_bbbb)
    
    #	 -1.0000 <k,b||c,i>_abab*t2_abab(c,a,k,j)
    doubles_res += -1.000000000000000 * einsum('kbci,cakj', g_abab[o, v, v, o], t2_abab)
    
    #	  1.0000 <k,b||c,i>_bbbb*t2_bbbb(c,a,j,k)
    doubles_res +=  1.000000000000000 * einsum('kbci,cajk', g_bbbb[o, v, v, o], t2_bbbb)
    
    #	  0.5000 <a,b||c,d>_bbbb*t2_bbbb(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('abcd,cdij', g_bbbb[v, v, v, v], t2_bbbb)
    
    #	 -0.5000 <l,k||c,j>_abab*t3_abbabb(c,a,b,l,i,k)
    doubles_res += -0.500000000000000 * einsum('lkcj,cablik', g_abab[o, o, v, o], t3_abbabb)
    
    #	  0.5000 <k,l||c,j>_abab*t3_abbabb(c,a,b,k,l,i)
    doubles_res +=  0.500000000000000 * einsum('klcj,cabkli', g_abab[o, o, v, o], t3_abbabb)
    
    #	  0.5000 <l,k||c,j>_bbbb*t3_bbbbbb(c,a,b,i,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcj,cabilk', g_bbbb[o, o, v, o], t3_bbbbbb)
    
    #	  0.5000 <l,k||c,i>_abab*t3_abbabb(c,a,b,l,j,k)
    doubles_res +=  0.500000000000000 * einsum('lkci,cabljk', g_abab[o, o, v, o], t3_abbabb)
    
    #	 -0.5000 <k,l||c,i>_abab*t3_abbabb(c,a,b,k,l,j)
    doubles_res += -0.500000000000000 * einsum('klci,cabklj', g_abab[o, o, v, o], t3_abbabb)
    
    #	 -0.5000 <l,k||c,i>_bbbb*t3_bbbbbb(c,a,b,j,l,k)
    doubles_res += -0.500000000000000 * einsum('lkci,cabjlk', g_bbbb[o, o, v, o], t3_bbbbbb)
    
    #	 -0.5000 <k,a||c,d>_abab*t3_abbabb(c,d,b,k,j,i)
    doubles_res += -0.500000000000000 * einsum('kacd,cdbkji', g_abab[o, v, v, v], t3_abbabb)
    
    #	 -0.5000 <k,a||d,c>_abab*t3_abbabb(d,c,b,k,j,i)
    doubles_res += -0.500000000000000 * einsum('kadc,dcbkji', g_abab[o, v, v, v], t3_abbabb)
    
    #	  0.5000 <k,a||c,d>_bbbb*t3_bbbbbb(c,d,b,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('kacd,cdbijk', g_bbbb[o, v, v, v], t3_bbbbbb)
    
    #	  0.5000 <k,b||c,d>_abab*t3_abbabb(c,d,a,k,j,i)
    doubles_res +=  0.500000000000000 * einsum('kbcd,cdakji', g_abab[o, v, v, v], t3_abbabb)
    
    #	  0.5000 <k,b||d,c>_abab*t3_abbabb(d,c,a,k,j,i)
    doubles_res +=  0.500000000000000 * einsum('kbdc,dcakji', g_abab[o, v, v, v], t3_abbabb)
    
    #	 -0.5000 <k,b||c,d>_bbbb*t3_bbbbbb(c,d,a,i,j,k)
    doubles_res += -0.500000000000000 * einsum('kbcd,cdaijk', g_bbbb[o, v, v, v], t3_bbbbbb)
    
    #	 -1.0000 <k,l||c,j>_abab*t1_aa(c,k)*t2_bbbb(a,b,i,l)
    doubles_res += -1.000000000000000 * einsum('klcj,ck,abil', g_abab[o, o, v, o], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,j>_bbbb*t1_bb(c,k)*t2_bbbb(a,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcj,ck,abil', g_bbbb[o, o, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,j>_bbbb*t1_bb(c,i)*t2_bbbb(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcj,ci,ablk', g_bbbb[o, o, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,j>_abab*t1_bb(a,k)*t2_abab(c,b,l,i)
    doubles_res +=  1.000000000000000 * einsum('lkcj,ak,cbli', g_abab[o, o, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,j>_bbbb*t1_bb(a,k)*t2_bbbb(c,b,i,l)
    doubles_res += -1.000000000000000 * einsum('lkcj,ak,cbil', g_bbbb[o, o, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,j>_abab*t1_bb(b,k)*t2_abab(c,a,l,i)
    doubles_res += -1.000000000000000 * einsum('lkcj,bk,cali', g_abab[o, o, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,j>_bbbb*t1_bb(b,k)*t2_bbbb(c,a,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcj,bk,cail', g_bbbb[o, o, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,l||c,i>_abab*t1_aa(c,k)*t2_bbbb(a,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('klci,ck,abjl', g_abab[o, o, v, o], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,i>_bbbb*t1_bb(c,k)*t2_bbbb(a,b,j,l)
    doubles_res += -1.000000000000000 * einsum('lkci,ck,abjl', g_bbbb[o, o, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,i>_bbbb*t1_bb(c,j)*t2_bbbb(a,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkci,cj,ablk', g_bbbb[o, o, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,i>_abab*t1_bb(a,k)*t2_abab(c,b,l,j)
    doubles_res += -1.000000000000000 * einsum('lkci,ak,cblj', g_abab[o, o, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,i>_bbbb*t1_bb(a,k)*t2_bbbb(c,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('lkci,ak,cbjl', g_bbbb[o, o, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,i>_abab*t1_bb(b,k)*t2_abab(c,a,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkci,bk,calj', g_abab[o, o, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,i>_bbbb*t1_bb(b,k)*t2_bbbb(c,a,j,l)
    doubles_res += -1.000000000000000 * einsum('lkci,bk,cajl', g_bbbb[o, o, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,a||c,d>_abab*t1_aa(c,k)*t2_bbbb(d,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('kacd,ck,dbij', g_abab[o, v, v, v], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,a||c,d>_bbbb*t1_bb(c,k)*t2_bbbb(d,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('kacd,ck,dbij', g_bbbb[o, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,a||d,c>_abab*t1_bb(c,j)*t2_abab(d,b,k,i)
    doubles_res += -1.000000000000000 * einsum('kadc,cj,dbki', g_abab[o, v, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,a||c,d>_bbbb*t1_bb(c,j)*t2_bbbb(d,b,i,k)
    doubles_res += -1.000000000000000 * einsum('kacd,cj,dbik', g_bbbb[o, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,a||d,c>_abab*t1_bb(c,i)*t2_abab(d,b,k,j)
    doubles_res +=  1.000000000000000 * einsum('kadc,ci,dbkj', g_abab[o, v, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,a||c,d>_bbbb*t1_bb(c,i)*t2_bbbb(d,b,j,k)
    doubles_res +=  1.000000000000000 * einsum('kacd,ci,dbjk', g_bbbb[o, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,a||c,d>_bbbb*t1_bb(b,k)*t2_bbbb(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('kacd,bk,cdij', g_bbbb[o, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,b||c,d>_abab*t1_aa(c,k)*t2_bbbb(d,a,i,j)
    doubles_res += -1.000000000000000 * einsum('kbcd,ck,daij', g_abab[o, v, v, v], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,b||c,d>_bbbb*t1_bb(c,k)*t2_bbbb(d,a,i,j)
    doubles_res += -1.000000000000000 * einsum('kbcd,ck,daij', g_bbbb[o, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,b||d,c>_abab*t1_bb(c,j)*t2_abab(d,a,k,i)
    doubles_res +=  1.000000000000000 * einsum('kbdc,cj,daki', g_abab[o, v, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,b||c,d>_bbbb*t1_bb(c,j)*t2_bbbb(d,a,i,k)
    doubles_res +=  1.000000000000000 * einsum('kbcd,cj,daik', g_bbbb[o, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,b||d,c>_abab*t1_bb(c,i)*t2_abab(d,a,k,j)
    doubles_res += -1.000000000000000 * einsum('kbdc,ci,dakj', g_abab[o, v, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,b||c,d>_bbbb*t1_bb(c,i)*t2_bbbb(d,a,j,k)
    doubles_res += -1.000000000000000 * einsum('kbcd,ci,dajk', g_bbbb[o, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,b||c,d>_bbbb*t1_bb(a,k)*t2_bbbb(c,d,i,j)
    doubles_res += -0.500000000000000 * einsum('kbcd,ak,cdij', g_bbbb[o, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t1_aa(c,k)*t3_abbabb(d,a,b,l,j,i)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ck,dablji', g_aaaa[o, o, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,l||c,d>_abab*t1_aa(c,k)*t3_bbbbbb(d,a,b,i,j,l)
    doubles_res +=  1.000000000000000 * einsum('klcd,ck,dabijl', g_abab[o, o, v, v], t1_aa, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||d,c>_abab*t1_bb(c,k)*t3_abbabb(d,a,b,l,j,i)
    doubles_res += -1.000000000000000 * einsum('lkdc,ck,dablji', g_abab[o, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_bbbb*t1_bb(c,k)*t3_bbbbbb(d,a,b,i,j,l)
    doubles_res += -1.000000000000000 * einsum('lkcd,ck,dabijl', g_bbbb[o, o, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||d,c>_abab*t1_bb(c,j)*t3_abbabb(d,a,b,l,i,k)
    doubles_res += -0.500000000000000 * einsum('lkdc,cj,dablik', g_abab[o, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,l||d,c>_abab*t1_bb(c,j)*t3_abbabb(d,a,b,k,l,i)
    doubles_res +=  0.500000000000000 * einsum('kldc,cj,dabkli', g_abab[o, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_bbbb*t1_bb(c,j)*t3_bbbbbb(d,a,b,i,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,cj,dabilk', g_bbbb[o, o, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||d,c>_abab*t1_bb(c,i)*t3_abbabb(d,a,b,l,j,k)
    doubles_res +=  0.500000000000000 * einsum('lkdc,ci,dabljk', g_abab[o, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||d,c>_abab*t1_bb(c,i)*t3_abbabb(d,a,b,k,l,j)
    doubles_res += -0.500000000000000 * einsum('kldc,ci,dabklj', g_abab[o, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_bbbb*t1_bb(c,i)*t3_bbbbbb(d,a,b,j,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcd,ci,dabjlk', g_bbbb[o, o, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_abab*t1_bb(a,k)*t3_abbabb(c,d,b,l,j,i)
    doubles_res +=  0.500000000000000 * einsum('lkcd,ak,cdblji', g_abab[o, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||d,c>_abab*t1_bb(a,k)*t3_abbabb(d,c,b,l,j,i)
    doubles_res +=  0.500000000000000 * einsum('lkdc,ak,dcblji', g_abab[o, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_bbbb*t1_bb(a,k)*t3_bbbbbb(c,d,b,i,j,l)
    doubles_res += -0.500000000000000 * einsum('lkcd,ak,cdbijl', g_bbbb[o, o, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_abab*t1_bb(b,k)*t3_abbabb(c,d,a,l,j,i)
    doubles_res += -0.500000000000000 * einsum('lkcd,bk,cdalji', g_abab[o, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||d,c>_abab*t1_bb(b,k)*t3_abbabb(d,c,a,l,j,i)
    doubles_res += -0.500000000000000 * einsum('lkdc,bk,dcalji', g_abab[o, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_bbbb*t1_bb(b,k)*t3_bbbbbb(c,d,a,i,j,l)
    doubles_res +=  0.500000000000000 * einsum('lkcd,bk,cdaijl', g_bbbb[o, o, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||i,j>_bbbb*t1_bb(a,k)*t1_bb(b,l)
    doubles_res += -1.000000000000000 * einsum('lkij,ak,bl', g_bbbb[o, o, o, o], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,a||c,j>_bbbb*t1_bb(c,i)*t1_bb(b,k)
    doubles_res +=  1.000000000000000 * einsum('kacj,ci,bk', g_bbbb[o, v, v, o], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,a||c,i>_bbbb*t1_bb(c,j)*t1_bb(b,k)
    doubles_res += -1.000000000000000 * einsum('kaci,cj,bk', g_bbbb[o, v, v, o], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,b||c,j>_bbbb*t1_bb(c,i)*t1_bb(a,k)
    doubles_res += -1.000000000000000 * einsum('kbcj,ci,ak', g_bbbb[o, v, v, o], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,b||c,i>_bbbb*t1_bb(c,j)*t1_bb(a,k)
    doubles_res +=  1.000000000000000 * einsum('kbci,cj,ak', g_bbbb[o, v, v, o], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,b||c,d>_bbbb*t1_bb(c,j)*t1_bb(d,i)
    doubles_res += -1.000000000000000 * einsum('abcd,cj,di', g_bbbb[v, v, v, v], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||c,d>_abab*t2_abab(c,d,k,j)*t2_bbbb(a,b,i,l)
    doubles_res += -0.500000000000000 * einsum('klcd,cdkj,abil', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||d,c>_abab*t2_abab(d,c,k,j)*t2_bbbb(a,b,i,l)
    doubles_res += -0.500000000000000 * einsum('kldc,dckj,abil', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_bbbb*t2_bbbb(c,d,j,k)*t2_bbbb(a,b,i,l)
    doubles_res += -0.500000000000000 * einsum('lkcd,cdjk,abil', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,l||c,d>_abab*t2_abab(c,d,k,i)*t2_bbbb(a,b,j,l)
    doubles_res +=  0.500000000000000 * einsum('klcd,cdki,abjl', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,l||d,c>_abab*t2_abab(d,c,k,i)*t2_bbbb(a,b,j,l)
    doubles_res +=  0.500000000000000 * einsum('kldc,dcki,abjl', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_bbbb*t2_bbbb(c,d,i,k)*t2_bbbb(a,b,j,l)
    doubles_res +=  0.500000000000000 * einsum('lkcd,cdik,abjl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,k||c,d>_bbbb*t2_bbbb(c,d,i,j)*t2_bbbb(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdij,ablk', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_abab*t2_abab(c,a,l,k)*t2_bbbb(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||c,d>_abab*t2_abab(c,a,k,l)*t2_bbbb(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('klcd,cakl,dbij', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_bbbb*t2_bbbb(c,a,l,k)*t2_bbbb(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t2_abab(c,a,k,j)*t2_abab(d,b,l,i)
    doubles_res +=  1.000000000000000 * einsum('lkcd,cakj,dbli', g_aaaa[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,l||c,d>_abab*t2_abab(c,a,k,j)*t2_bbbb(d,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('klcd,cakj,dbil', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||d,c>_abab*t2_bbbb(c,a,j,k)*t2_abab(d,b,l,i)
    doubles_res +=  1.000000000000000 * einsum('lkdc,cajk,dbli', g_abab[o, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t2_bbbb(c,a,j,k)*t2_bbbb(d,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,cajk,dbil', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_aaaa*t2_abab(c,a,k,i)*t2_abab(d,b,l,j)
    doubles_res += -1.000000000000000 * einsum('lkcd,caki,dblj', g_aaaa[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,l||c,d>_abab*t2_abab(c,a,k,i)*t2_bbbb(d,b,j,l)
    doubles_res += -1.000000000000000 * einsum('klcd,caki,dbjl', g_abab[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||d,c>_abab*t2_bbbb(c,a,i,k)*t2_abab(d,b,l,j)
    doubles_res += -1.000000000000000 * einsum('lkdc,caik,dblj', g_abab[o, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_bbbb*t2_bbbb(c,a,i,k)*t2_bbbb(d,b,j,l)
    doubles_res += -1.000000000000000 * einsum('lkcd,caik,dbjl', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||d,c>_abab*t2_bbbb(c,a,i,j)*t2_abab(d,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkdc,caij,dblk', g_abab[o, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <k,l||d,c>_abab*t2_bbbb(c,a,i,j)*t2_abab(d,b,k,l)
    doubles_res +=  0.500000000000000 * einsum('kldc,caij,dbkl', g_abab[o, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_bbbb*t2_bbbb(c,a,i,j)*t2_bbbb(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,caij,dblk', g_bbbb[o, o, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <k,l||c,d>_abab*t1_aa(c,k)*t1_bb(d,j)*t2_bbbb(a,b,i,l)
    doubles_res += -1.000000000000000 * einsum('klcd,ck,dj,abil', g_abab[o, o, v, v], t1_aa, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t1_bb(c,k)*t1_bb(d,j)*t2_bbbb(a,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ck,dj,abil', g_bbbb[o, o, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <k,l||c,d>_abab*t1_aa(c,k)*t1_bb(d,i)*t2_bbbb(a,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('klcd,ck,di,abjl', g_abab[o, o, v, v], t1_aa, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_bbbb*t1_bb(c,k)*t1_bb(d,i)*t2_bbbb(a,b,j,l)
    doubles_res += -1.000000000000000 * einsum('lkcd,ck,di,abjl', g_bbbb[o, o, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <k,l||c,d>_abab*t1_aa(c,k)*t1_bb(a,l)*t2_bbbb(d,b,i,j)
    doubles_res += -1.000000000000000 * einsum('klcd,ck,al,dbij', g_abab[o, o, v, v], t1_aa, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t1_bb(c,k)*t1_bb(a,l)*t2_bbbb(d,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ck,al,dbij', g_bbbb[o, o, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <k,l||c,d>_abab*t1_aa(c,k)*t1_bb(b,l)*t2_bbbb(d,a,i,j)
    doubles_res +=  1.000000000000000 * einsum('klcd,ck,bl,daij', g_abab[o, o, v, v], t1_aa, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_bbbb*t1_bb(c,k)*t1_bb(b,l)*t2_bbbb(d,a,i,j)
    doubles_res += -1.000000000000000 * einsum('lkcd,ck,bl,daij', g_bbbb[o, o, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_bbbb*t1_bb(c,j)*t1_bb(d,i)*t2_bbbb(a,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcd,cj,di,ablk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||d,c>_abab*t1_bb(c,j)*t1_bb(a,k)*t2_abab(d,b,l,i)
    doubles_res +=  1.000000000000000 * einsum('lkdc,cj,ak,dbli', g_abab[o, o, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t1_bb(c,j)*t1_bb(a,k)*t2_bbbb(d,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,cj,ak,dbil', g_bbbb[o, o, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||d,c>_abab*t1_bb(c,j)*t1_bb(b,k)*t2_abab(d,a,l,i)
    doubles_res += -1.000000000000000 * einsum('lkdc,cj,bk,dali', g_abab[o, o, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_bbbb*t1_bb(c,j)*t1_bb(b,k)*t2_bbbb(d,a,i,l)
    doubles_res += -1.000000000000000 * einsum('lkcd,cj,bk,dail', g_bbbb[o, o, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||d,c>_abab*t1_bb(c,i)*t1_bb(a,k)*t2_abab(d,b,l,j)
    doubles_res += -1.000000000000000 * einsum('lkdc,ci,ak,dblj', g_abab[o, o, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||c,d>_bbbb*t1_bb(c,i)*t1_bb(a,k)*t2_bbbb(d,b,j,l)
    doubles_res += -1.000000000000000 * einsum('lkcd,ci,ak,dbjl', g_bbbb[o, o, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||d,c>_abab*t1_bb(c,i)*t1_bb(b,k)*t2_abab(d,a,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkdc,ci,bk,dalj', g_abab[o, o, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t1_bb(c,i)*t1_bb(b,k)*t2_bbbb(d,a,j,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ci,bk,dajl', g_bbbb[o, o, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_bbbb*t1_bb(a,k)*t1_bb(b,l)*t2_bbbb(c,d,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,ak,bl,cdij', g_bbbb[o, o, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||c,j>_bbbb*t1_bb(c,i)*t1_bb(a,k)*t1_bb(b,l)
    doubles_res += -1.000000000000000 * einsum('lkcj,ci,ak,bl', g_bbbb[o, o, v, o], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,i>_bbbb*t1_bb(c,j)*t1_bb(a,k)*t1_bb(b,l)
    doubles_res +=  1.000000000000000 * einsum('lkci,cj,ak,bl', g_bbbb[o, o, v, o], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <k,a||c,d>_bbbb*t1_bb(c,j)*t1_bb(d,i)*t1_bb(b,k)
    doubles_res += -1.000000000000000 * einsum('kacd,cj,di,bk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <k,b||c,d>_bbbb*t1_bb(c,j)*t1_bb(d,i)*t1_bb(a,k)
    doubles_res +=  1.000000000000000 * einsum('kbcd,cj,di,ak', g_bbbb[o, v, v, v], t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t1_bb(c,j)*t1_bb(d,i)*t1_bb(a,k)*t1_bb(b,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,cj,di,ak,bl', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    return doubles_res
    
    
def ccsdt_t2_abab_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):
    
    #    < 0 | i* j* b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 f_bb(k,j)*t2_abab(a,b,i,k)
    doubles_res = -1.000000000000000 * einsum('kj,abik', f_bb[o, o], t2_abab)
    
    #	 -1.0000 f_aa(k,i)*t2_abab(a,b,k,j)
    doubles_res += -1.000000000000000 * einsum('ki,abkj', f_aa[o, o], t2_abab)
    
    #	  1.0000 f_aa(a,c)*t2_abab(c,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('ac,cbij', f_aa[v, v], t2_abab)
    
    #	  1.0000 f_bb(b,c)*t2_abab(a,c,i,j)
    doubles_res +=  1.000000000000000 * einsum('bc,acij', f_bb[v, v], t2_abab)
    
    #	 -1.0000 f_aa(k,c)*t3_aabaab(c,a,b,i,k,j)
    doubles_res += -1.000000000000000 * einsum('kc,cabikj', f_aa[o, v], t3_aabaab)
    
    #	 -1.0000 f_bb(k,c)*t3_abbabb(a,c,b,i,j,k)
    doubles_res += -1.000000000000000 * einsum('kc,acbijk', f_bb[o, v], t3_abbabb)
    
    #	 -1.0000 f_bb(k,c)*t1_bb(c,j)*t2_abab(a,b,i,k)
    doubles_res += -1.000000000000000 * einsum('kc,cj,abik', f_bb[o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(k,c)*t1_aa(c,i)*t2_abab(a,b,k,j)
    doubles_res += -1.000000000000000 * einsum('kc,ci,abkj', f_aa[o, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(k,c)*t1_aa(a,k)*t2_abab(c,b,i,j)
    doubles_res += -1.000000000000000 * einsum('kc,ak,cbij', f_aa[o, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(k,c)*t1_bb(b,k)*t2_abab(a,c,i,j)
    doubles_res += -1.000000000000000 * einsum('kc,bk,acij', f_bb[o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,b||i,j>_abab
    doubles_res +=  1.000000000000000 * einsum('abij', g_abab[v, v, o, o])
    
    #	 -1.0000 <a,k||i,j>_abab*t1_bb(b,k)
    doubles_res += -1.000000000000000 * einsum('akij,bk', g_abab[v, o, o, o], t1_bb)
    
    #	 -1.0000 <k,b||i,j>_abab*t1_aa(a,k)
    doubles_res += -1.000000000000000 * einsum('kbij,ak', g_abab[o, v, o, o], t1_aa)
    
    #	  1.0000 <a,b||c,j>_abab*t1_aa(c,i)
    doubles_res +=  1.000000000000000 * einsum('abcj,ci', g_abab[v, v, v, o], t1_aa)
    
    #	  1.0000 <a,b||i,c>_abab*t1_bb(c,j)
    doubles_res +=  1.000000000000000 * einsum('abic,cj', g_abab[v, v, o, v], t1_bb)
    
    #	  0.5000 <l,k||i,j>_abab*t2_abab(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkij,ablk', g_abab[o, o, o, o], t2_abab)
    
    #	  0.5000 <k,l||i,j>_abab*t2_abab(a,b,k,l)
    doubles_res +=  0.500000000000000 * einsum('klij,abkl', g_abab[o, o, o, o], t2_abab)
    
    #	 -1.0000 <a,k||c,j>_abab*t2_abab(c,b,i,k)
    doubles_res += -1.000000000000000 * einsum('akcj,cbik', g_abab[v, o, v, o], t2_abab)
    
    #	  1.0000 <k,a||c,i>_aaaa*t2_abab(c,b,k,j)
    doubles_res +=  1.000000000000000 * einsum('kaci,cbkj', g_aaaa[o, v, v, o], t2_abab)
    
    #	 -1.0000 <a,k||i,c>_abab*t2_bbbb(c,b,j,k)
    doubles_res += -1.000000000000000 * einsum('akic,cbjk', g_abab[v, o, o, v], t2_bbbb)
    
    #	 -1.0000 <k,b||c,j>_abab*t2_aaaa(c,a,i,k)
    doubles_res += -1.000000000000000 * einsum('kbcj,caik', g_abab[o, v, v, o], t2_aaaa)
    
    #	  1.0000 <k,b||c,j>_bbbb*t2_abab(a,c,i,k)
    doubles_res +=  1.000000000000000 * einsum('kbcj,acik', g_bbbb[o, v, v, o], t2_abab)
    
    #	 -1.0000 <k,b||i,c>_abab*t2_abab(a,c,k,j)
    doubles_res += -1.000000000000000 * einsum('kbic,ackj', g_abab[o, v, o, v], t2_abab)
    
    #	  0.5000 <a,b||c,d>_abab*t2_abab(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('abcd,cdij', g_abab[v, v, v, v], t2_abab)
    
    #	  0.5000 <a,b||d,c>_abab*t2_abab(d,c,i,j)
    doubles_res +=  0.500000000000000 * einsum('abdc,dcij', g_abab[v, v, v, v], t2_abab)
    
    #	  0.5000 <l,k||c,j>_abab*t3_aabaab(c,a,b,i,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcj,cabilk', g_abab[o, o, v, o], t3_aabaab)
    
    #	  0.5000 <k,l||c,j>_abab*t3_aabaab(c,a,b,i,k,l)
    doubles_res +=  0.500000000000000 * einsum('klcj,cabikl', g_abab[o, o, v, o], t3_aabaab)
    
    #	 -0.5000 <l,k||c,j>_bbbb*t3_abbabb(a,c,b,i,l,k)
    doubles_res += -0.500000000000000 * einsum('lkcj,acbilk', g_bbbb[o, o, v, o], t3_abbabb)
    
    #	  0.5000 <l,k||c,i>_aaaa*t3_aabaab(c,a,b,k,l,j)
    doubles_res +=  0.500000000000000 * einsum('lkci,cabklj', g_aaaa[o, o, v, o], t3_aabaab)
    
    #	  0.5000 <l,k||i,c>_abab*t3_abbabb(a,c,b,l,j,k)
    doubles_res +=  0.500000000000000 * einsum('lkic,acbljk', g_abab[o, o, o, v], t3_abbabb)
    
    #	 -0.5000 <k,l||i,c>_abab*t3_abbabb(a,c,b,k,l,j)
    doubles_res += -0.500000000000000 * einsum('klic,acbklj', g_abab[o, o, o, v], t3_abbabb)
    
    #	 -0.5000 <k,a||c,d>_aaaa*t3_aabaab(c,d,b,i,k,j)
    doubles_res += -0.500000000000000 * einsum('kacd,cdbikj', g_aaaa[o, v, v, v], t3_aabaab)
    
    #	 -0.5000 <a,k||c,d>_abab*t3_abbabb(c,d,b,i,j,k)
    doubles_res += -0.500000000000000 * einsum('akcd,cdbijk', g_abab[v, o, v, v], t3_abbabb)
    
    #	 -0.5000 <a,k||d,c>_abab*t3_abbabb(d,c,b,i,j,k)
    doubles_res += -0.500000000000000 * einsum('akdc,dcbijk', g_abab[v, o, v, v], t3_abbabb)
    
    #	 -0.5000 <k,b||c,d>_abab*t3_aabaab(c,a,d,i,k,j)
    doubles_res += -0.500000000000000 * einsum('kbcd,cadikj', g_abab[o, v, v, v], t3_aabaab)
    
    #	  0.5000 <k,b||d,c>_abab*t3_aabaab(a,d,c,i,k,j)
    doubles_res +=  0.500000000000000 * einsum('kbdc,adcikj', g_abab[o, v, v, v], t3_aabaab)
    
    #	  0.5000 <k,b||c,d>_bbbb*t3_abbabb(a,d,c,i,j,k)
    doubles_res +=  0.500000000000000 * einsum('kbcd,adcijk', g_bbbb[o, v, v, v], t3_abbabb)
    
    #	 -1.0000 <k,l||c,j>_abab*t1_aa(c,k)*t2_abab(a,b,i,l)
    doubles_res += -1.000000000000000 * einsum('klcj,ck,abil', g_abab[o, o, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,j>_bbbb*t1_bb(c,k)*t2_abab(a,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcj,ck,abil', g_bbbb[o, o, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,j>_abab*t1_aa(c,i)*t2_abab(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcj,ci,ablk', g_abab[o, o, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,l||c,j>_abab*t1_aa(c,i)*t2_abab(a,b,k,l)
    doubles_res +=  0.500000000000000 * einsum('klcj,ci,abkl', g_abab[o, o, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,l||c,j>_abab*t1_aa(a,k)*t2_abab(c,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('klcj,ak,cbil', g_abab[o, o, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,j>_abab*t1_bb(b,k)*t2_aaaa(c,a,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcj,bk,cail', g_abab[o, o, v, o], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,j>_bbbb*t1_bb(b,k)*t2_abab(a,c,i,l)
    doubles_res += -1.000000000000000 * einsum('lkcj,bk,acil', g_bbbb[o, o, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,i>_aaaa*t1_aa(c,k)*t2_abab(a,b,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkci,ck,ablj', g_aaaa[o, o, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||i,c>_abab*t1_bb(c,k)*t2_abab(a,b,l,j)
    doubles_res += -1.000000000000000 * einsum('lkic,ck,ablj', g_abab[o, o, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||i,c>_abab*t1_bb(c,j)*t2_abab(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkic,cj,ablk', g_abab[o, o, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,l||i,c>_abab*t1_bb(c,j)*t2_abab(a,b,k,l)
    doubles_res +=  0.500000000000000 * einsum('klic,cj,abkl', g_abab[o, o, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||c,i>_aaaa*t1_aa(a,k)*t2_abab(c,b,l,j)
    doubles_res += -1.000000000000000 * einsum('lkci,ak,cblj', g_aaaa[o, o, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,l||i,c>_abab*t1_aa(a,k)*t2_bbbb(c,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('klic,ak,cbjl', g_abab[o, o, o, v], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||i,c>_abab*t1_bb(b,k)*t2_abab(a,c,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkic,bk,aclj', g_abab[o, o, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,a||c,d>_aaaa*t1_aa(c,k)*t2_abab(d,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('kacd,ck,dbij', g_aaaa[o, v, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,k||d,c>_abab*t1_bb(c,k)*t2_abab(d,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('akdc,ck,dbij', g_abab[v, o, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,k||d,c>_abab*t1_bb(c,j)*t2_abab(d,b,i,k)
    doubles_res += -1.000000000000000 * einsum('akdc,cj,dbik', g_abab[v, o, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,a||c,d>_aaaa*t1_aa(c,i)*t2_abab(d,b,k,j)
    doubles_res += -1.000000000000000 * einsum('kacd,ci,dbkj', g_aaaa[o, v, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,k||c,d>_abab*t1_aa(c,i)*t2_bbbb(d,b,j,k)
    doubles_res += -1.000000000000000 * einsum('akcd,ci,dbjk', g_abab[v, o, v, v], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <a,k||c,d>_abab*t1_bb(b,k)*t2_abab(c,d,i,j)
    doubles_res += -0.500000000000000 * einsum('akcd,bk,cdij', g_abab[v, o, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <a,k||d,c>_abab*t1_bb(b,k)*t2_abab(d,c,i,j)
    doubles_res += -0.500000000000000 * einsum('akdc,bk,dcij', g_abab[v, o, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,b||c,d>_abab*t1_aa(c,k)*t2_abab(a,d,i,j)
    doubles_res +=  1.000000000000000 * einsum('kbcd,ck,adij', g_abab[o, v, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,b||c,d>_bbbb*t1_bb(c,k)*t2_abab(a,d,i,j)
    doubles_res +=  1.000000000000000 * einsum('kbcd,ck,adij', g_bbbb[o, v, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,b||d,c>_abab*t1_bb(c,j)*t2_aaaa(d,a,i,k)
    doubles_res += -1.000000000000000 * einsum('kbdc,cj,daik', g_abab[o, v, v, v], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,b||c,d>_bbbb*t1_bb(c,j)*t2_abab(a,d,i,k)
    doubles_res += -1.000000000000000 * einsum('kbcd,cj,adik', g_bbbb[o, v, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,b||c,d>_abab*t1_aa(c,i)*t2_abab(a,d,k,j)
    doubles_res += -1.000000000000000 * einsum('kbcd,ci,adkj', g_abab[o, v, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,b||c,d>_abab*t1_aa(a,k)*t2_abab(c,d,i,j)
    doubles_res += -0.500000000000000 * einsum('kbcd,ak,cdij', g_abab[o, v, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,b||d,c>_abab*t1_aa(a,k)*t2_abab(d,c,i,j)
    doubles_res += -0.500000000000000 * einsum('kbdc,ak,dcij', g_abab[o, v, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t1_aa(c,k)*t3_aabaab(d,a,b,i,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ck,dabilj', g_aaaa[o, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,l||c,d>_abab*t1_aa(c,k)*t3_abbabb(a,d,b,i,j,l)
    doubles_res += -1.000000000000000 * einsum('klcd,ck,adbijl', g_abab[o, o, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,k||d,c>_abab*t1_bb(c,k)*t3_aabaab(d,a,b,i,l,j)
    doubles_res += -1.000000000000000 * einsum('lkdc,ck,dabilj', g_abab[o, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t1_bb(c,k)*t3_abbabb(a,d,b,i,j,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ck,adbijl', g_bbbb[o, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||d,c>_abab*t1_bb(c,j)*t3_aabaab(d,a,b,i,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkdc,cj,dabilk', g_abab[o, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,l||d,c>_abab*t1_bb(c,j)*t3_aabaab(d,a,b,i,k,l)
    doubles_res +=  0.500000000000000 * einsum('kldc,cj,dabikl', g_abab[o, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_bbbb*t1_bb(c,j)*t3_abbabb(a,d,b,i,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcd,cj,adbilk', g_bbbb[o, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_aaaa*t1_aa(c,i)*t3_aabaab(d,a,b,k,l,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,ci,dabklj', g_aaaa[o, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_abab*t1_aa(c,i)*t3_abbabb(a,d,b,l,j,k)
    doubles_res +=  0.500000000000000 * einsum('lkcd,ci,adbljk', g_abab[o, o, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||c,d>_abab*t1_aa(c,i)*t3_abbabb(a,d,b,k,l,j)
    doubles_res += -0.500000000000000 * einsum('klcd,ci,adbklj', g_abab[o, o, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_aaaa*t1_aa(a,k)*t3_aabaab(c,d,b,i,l,j)
    doubles_res +=  0.500000000000000 * einsum('lkcd,ak,cdbilj', g_aaaa[o, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,l||c,d>_abab*t1_aa(a,k)*t3_abbabb(c,d,b,i,j,l)
    doubles_res +=  0.500000000000000 * einsum('klcd,ak,cdbijl', g_abab[o, o, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,l||d,c>_abab*t1_aa(a,k)*t3_abbabb(d,c,b,i,j,l)
    doubles_res +=  0.500000000000000 * einsum('kldc,ak,dcbijl', g_abab[o, o, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_abab*t1_bb(b,k)*t3_aabaab(c,a,d,i,l,j)
    doubles_res +=  0.500000000000000 * einsum('lkcd,bk,cadilj', g_abab[o, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||d,c>_abab*t1_bb(b,k)*t3_aabaab(a,d,c,i,l,j)
    doubles_res += -0.500000000000000 * einsum('lkdc,bk,adcilj', g_abab[o, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_bbbb*t1_bb(b,k)*t3_abbabb(a,d,c,i,j,l)
    doubles_res += -0.500000000000000 * einsum('lkcd,bk,adcijl', g_bbbb[o, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,l||i,j>_abab*t1_aa(a,k)*t1_bb(b,l)
    doubles_res +=  1.000000000000000 * einsum('klij,ak,bl', g_abab[o, o, o, o], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,k||c,j>_abab*t1_aa(c,i)*t1_bb(b,k)
    doubles_res += -1.000000000000000 * einsum('akcj,ci,bk', g_abab[v, o, v, o], t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,k||i,c>_abab*t1_bb(c,j)*t1_bb(b,k)
    doubles_res += -1.000000000000000 * einsum('akic,cj,bk', g_abab[v, o, o, v], t1_bb, t1_bb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,b||c,j>_abab*t1_aa(c,i)*t1_aa(a,k)
    doubles_res += -1.000000000000000 * einsum('kbcj,ci,ak', g_abab[o, v, v, o], t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <k,b||i,c>_abab*t1_bb(c,j)*t1_aa(a,k)
    doubles_res += -1.000000000000000 * einsum('kbic,cj,ak', g_abab[o, v, o, v], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,b||d,c>_abab*t1_bb(c,j)*t1_aa(d,i)
    doubles_res +=  1.000000000000000 * einsum('abdc,cj,di', g_abab[v, v, v, v], t1_bb, t1_aa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||c,d>_abab*t2_abab(c,d,k,j)*t2_abab(a,b,i,l)
    doubles_res += -0.500000000000000 * einsum('klcd,cdkj,abil', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||d,c>_abab*t2_abab(d,c,k,j)*t2_abab(a,b,i,l)
    doubles_res += -0.500000000000000 * einsum('kldc,dckj,abil', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_bbbb*t2_bbbb(c,d,j,k)*t2_abab(a,b,i,l)
    doubles_res += -0.500000000000000 * einsum('lkcd,cdjk,abil', g_bbbb[o, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_aaaa*t2_aaaa(c,d,i,k)*t2_abab(a,b,l,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,cdik,ablj', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_abab*t2_abab(c,d,i,k)*t2_abab(a,b,l,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,cdik,ablj', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||d,c>_abab*t2_abab(d,c,i,k)*t2_abab(a,b,l,j)
    doubles_res += -0.500000000000000 * einsum('lkdc,dcik,ablj', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,k||c,d>_abab*t2_abab(c,d,i,j)*t2_abab(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdij,ablk', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <k,l||c,d>_abab*t2_abab(c,d,i,j)*t2_abab(a,b,k,l)
    doubles_res +=  0.250000000000000 * einsum('klcd,cdij,abkl', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,k||d,c>_abab*t2_abab(d,c,i,j)*t2_abab(a,b,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkdc,dcij,ablk', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <k,l||d,c>_abab*t2_abab(d,c,i,j)*t2_abab(a,b,k,l)
    doubles_res +=  0.250000000000000 * einsum('kldc,dcij,abkl', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||c,d>_aaaa*t2_aaaa(c,a,l,k)*t2_abab(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkcd,calk,dbij', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||d,c>_abab*t2_abab(a,c,l,k)*t2_abab(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('lkdc,aclk,dbij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <k,l||d,c>_abab*t2_abab(a,c,k,l)*t2_abab(d,b,i,j)
    doubles_res += -0.500000000000000 * einsum('kldc,ackl,dbij', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,l||d,c>_abab*t2_abab(a,c,k,j)*t2_abab(d,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('kldc,ackj,dbil', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t2_aaaa(c,a,i,k)*t2_abab(d,b,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkcd,caik,dblj', g_aaaa[o, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,l||c,d>_abab*t2_aaaa(c,a,i,k)*t2_bbbb(d,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('klcd,caik,dbjl', g_abab[o, o, v, v], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||d,c>_abab*t2_abab(a,c,i,k)*t2_abab(d,b,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkdc,acik,dblj', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t2_abab(a,c,i,k)*t2_bbbb(d,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,acik,dbjl', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,k||d,c>_abab*t2_abab(a,c,i,j)*t2_abab(d,b,l,k)
    doubles_res += -0.500000000000000 * einsum('lkdc,acij,dblk', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <k,l||d,c>_abab*t2_abab(a,c,i,j)*t2_abab(d,b,k,l)
    doubles_res += -0.500000000000000 * einsum('kldc,acij,dbkl', g_abab[o, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,k||c,d>_bbbb*t2_abab(a,c,i,j)*t2_bbbb(d,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkcd,acij,dblk', g_bbbb[o, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <k,l||c,d>_abab*t1_aa(c,k)*t1_bb(d,j)*t2_abab(a,b,i,l)
    doubles_res += -1.000000000000000 * einsum('klcd,ck,dj,abil', g_abab[o, o, v, v], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t1_bb(c,k)*t1_bb(d,j)*t2_abab(a,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ck,dj,abil', g_bbbb[o, o, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t1_aa(c,k)*t1_aa(d,i)*t2_abab(a,b,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ck,di,ablj', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||d,c>_abab*t1_bb(c,k)*t1_aa(d,i)*t2_abab(a,b,l,j)
    doubles_res += -1.000000000000000 * einsum('lkdc,ck,di,ablj', g_abab[o, o, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t1_aa(c,k)*t1_aa(a,l)*t2_abab(d,b,i,j)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ck,al,dbij', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,k||d,c>_abab*t1_bb(c,k)*t1_aa(a,l)*t2_abab(d,b,i,j)
    doubles_res += -1.000000000000000 * einsum('lkdc,ck,al,dbij', g_abab[o, o, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <k,l||c,d>_abab*t1_aa(c,k)*t1_bb(b,l)*t2_abab(a,d,i,j)
    doubles_res += -1.000000000000000 * einsum('klcd,ck,bl,adij', g_abab[o, o, v, v], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t1_bb(c,k)*t1_bb(b,l)*t2_abab(a,d,i,j)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ck,bl,adij', g_bbbb[o, o, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <l,k||d,c>_abab*t1_bb(c,j)*t1_aa(d,i)*t2_abab(a,b,l,k)
    doubles_res +=  0.500000000000000 * einsum('lkdc,cj,di,ablk', g_abab[o, o, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <k,l||d,c>_abab*t1_bb(c,j)*t1_aa(d,i)*t2_abab(a,b,k,l)
    doubles_res +=  0.500000000000000 * einsum('kldc,cj,di,abkl', g_abab[o, o, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <k,l||d,c>_abab*t1_bb(c,j)*t1_aa(a,k)*t2_abab(d,b,i,l)
    doubles_res +=  1.000000000000000 * einsum('kldc,cj,ak,dbil', g_abab[o, o, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||d,c>_abab*t1_bb(c,j)*t1_bb(b,k)*t2_aaaa(d,a,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkdc,cj,bk,dail', g_abab[o, o, v, v], t1_bb, t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_bbbb*t1_bb(c,j)*t1_bb(b,k)*t2_abab(a,d,i,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,cj,bk,adil', g_bbbb[o, o, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_aaaa*t1_aa(c,i)*t1_aa(a,k)*t2_abab(d,b,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ci,ak,dblj', g_aaaa[o, o, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <k,l||c,d>_abab*t1_aa(c,i)*t1_aa(a,k)*t2_bbbb(d,b,j,l)
    doubles_res +=  1.000000000000000 * einsum('klcd,ci,ak,dbjl', g_abab[o, o, v, v], t1_aa, t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,k||c,d>_abab*t1_aa(c,i)*t1_bb(b,k)*t2_abab(a,d,l,j)
    doubles_res +=  1.000000000000000 * einsum('lkcd,ci,bk,adlj', g_abab[o, o, v, v], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <k,l||c,d>_abab*t1_aa(a,k)*t1_bb(b,l)*t2_abab(c,d,i,j)
    doubles_res +=  0.500000000000000 * einsum('klcd,ak,bl,cdij', g_abab[o, o, v, v], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <k,l||d,c>_abab*t1_aa(a,k)*t1_bb(b,l)*t2_abab(d,c,i,j)
    doubles_res +=  0.500000000000000 * einsum('kldc,ak,bl,dcij', g_abab[o, o, v, v], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <k,l||c,j>_abab*t1_aa(c,i)*t1_aa(a,k)*t1_bb(b,l)
    doubles_res +=  1.000000000000000 * einsum('klcj,ci,ak,bl', g_abab[o, o, v, o], t1_aa, t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <k,l||i,c>_abab*t1_bb(c,j)*t1_aa(a,k)*t1_bb(b,l)
    doubles_res +=  1.000000000000000 * einsum('klic,cj,ak,bl', g_abab[o, o, o, v], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,k||d,c>_abab*t1_bb(c,j)*t1_aa(d,i)*t1_bb(b,k)
    doubles_res += -1.000000000000000 * einsum('akdc,cj,di,bk', g_abab[v, o, v, v], t1_bb, t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <k,b||d,c>_abab*t1_bb(c,j)*t1_aa(d,i)*t1_aa(a,k)
    doubles_res += -1.000000000000000 * einsum('kbdc,cj,di,ak', g_abab[o, v, v, v], t1_bb, t1_aa, t1_aa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <k,l||d,c>_abab*t1_bb(c,j)*t1_aa(d,i)*t1_aa(a,k)*t1_bb(b,l)
    doubles_res +=  1.000000000000000 * einsum('kldc,cj,di,ak,bl', g_abab[o, o, v, v], t1_bb, t1_aa, t1_aa, t1_bb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    return doubles_res
    
def ccsdt_t3_aaaaaa_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):
    
    #    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 f_aa(l,k)*t3_aaaaaa(a,b,c,i,j,l)
    triples_res = -1.000000000000000 * einsum('lk,abcijl->abcijk', f_aa[o, o], t3_aaaaaa)
    
    #	  1.0000 f_aa(l,j)*t3_aaaaaa(a,b,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('lj,abcikl->abcijk', f_aa[o, o], t3_aaaaaa)
    
    #	 -1.0000 f_aa(l,i)*t3_aaaaaa(a,b,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('li,abcjkl->abcijk', f_aa[o, o], t3_aaaaaa)
    
    #	  1.0000 f_aa(a,d)*t3_aaaaaa(d,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('ad,dbcijk->abcijk', f_aa[v, v], t3_aaaaaa)
    
    #	 -1.0000 f_aa(b,d)*t3_aaaaaa(d,a,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('bd,dacijk->abcijk', f_aa[v, v], t3_aaaaaa)
    
    #	  1.0000 f_aa(c,d)*t3_aaaaaa(d,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('cd,dabijk->abcijk', f_aa[v, v], t3_aaaaaa)
    
    #	 -1.0000 f_aa(l,d)*t1_aa(d,k)*t3_aaaaaa(a,b,c,i,j,l)
    triples_res += -1.000000000000000 * einsum('ld,dk,abcijl->abcijk', f_aa[o, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(l,d)*t1_aa(d,j)*t3_aaaaaa(a,b,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('ld,dj,abcikl->abcijk', f_aa[o, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t1_aa(d,i)*t3_aaaaaa(a,b,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('ld,di,abcjkl->abcijk', f_aa[o, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t1_aa(a,l)*t3_aaaaaa(d,b,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('ld,al,dbcijk->abcijk', f_aa[o, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(l,d)*t1_aa(b,l)*t3_aaaaaa(d,a,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('ld,bl,dacijk->abcijk', f_aa[o, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t1_aa(c,l)*t3_aaaaaa(d,a,b,i,j,k)
    triples_res += -1.000000000000000 * einsum('ld,cl,dabijk->abcijk', f_aa[o, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t2_aaaa(d,a,j,k)*t2_aaaa(b,c,i,l)
    triples_res += -1.000000000000000 * einsum('ld,dajk,bcil->abcijk', f_aa[o, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(l,d)*t2_aaaa(d,a,i,k)*t2_aaaa(b,c,j,l)
    triples_res +=  1.000000000000000 * einsum('ld,daik,bcjl->abcijk', f_aa[o, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t2_aaaa(d,a,i,j)*t2_aaaa(b,c,k,l)
    triples_res += -1.000000000000000 * einsum('ld,daij,bckl->abcijk', f_aa[o, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(l,d)*t2_aaaa(d,b,j,k)*t2_aaaa(a,c,i,l)
    triples_res +=  1.000000000000000 * einsum('ld,dbjk,acil->abcijk', f_aa[o, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t2_aaaa(d,b,i,k)*t2_aaaa(a,c,j,l)
    triples_res += -1.000000000000000 * einsum('ld,dbik,acjl->abcijk', f_aa[o, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(l,d)*t2_aaaa(d,b,i,j)*t2_aaaa(a,c,k,l)
    triples_res +=  1.000000000000000 * einsum('ld,dbij,ackl->abcijk', f_aa[o, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t2_aaaa(d,c,j,k)*t2_aaaa(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('ld,dcjk,abil->abcijk', f_aa[o, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(l,d)*t2_aaaa(d,c,i,k)*t2_aaaa(a,b,j,l)
    triples_res +=  1.000000000000000 * einsum('ld,dcik,abjl->abcijk', f_aa[o, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t2_aaaa(d,c,i,j)*t2_aaaa(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('ld,dcij,abkl->abcijk', f_aa[o, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||j,k>_aaaa*t2_aaaa(b,c,i,l)
    triples_res += -1.000000000000000 * einsum('lajk,bcil->abcijk', g_aaaa[o, v, o, o], t2_aaaa)
    
    #	  1.0000 <l,a||i,k>_aaaa*t2_aaaa(b,c,j,l)
    triples_res +=  1.000000000000000 * einsum('laik,bcjl->abcijk', g_aaaa[o, v, o, o], t2_aaaa)
    
    #	 -1.0000 <l,a||i,j>_aaaa*t2_aaaa(b,c,k,l)
    triples_res += -1.000000000000000 * einsum('laij,bckl->abcijk', g_aaaa[o, v, o, o], t2_aaaa)
    
    #	  1.0000 <l,b||j,k>_aaaa*t2_aaaa(a,c,i,l)
    triples_res +=  1.000000000000000 * einsum('lbjk,acil->abcijk', g_aaaa[o, v, o, o], t2_aaaa)
    
    #	 -1.0000 <l,b||i,k>_aaaa*t2_aaaa(a,c,j,l)
    triples_res += -1.000000000000000 * einsum('lbik,acjl->abcijk', g_aaaa[o, v, o, o], t2_aaaa)
    
    #	  1.0000 <l,b||i,j>_aaaa*t2_aaaa(a,c,k,l)
    triples_res +=  1.000000000000000 * einsum('lbij,ackl->abcijk', g_aaaa[o, v, o, o], t2_aaaa)
    
    #	 -1.0000 <l,c||j,k>_aaaa*t2_aaaa(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('lcjk,abil->abcijk', g_aaaa[o, v, o, o], t2_aaaa)
    
    #	  1.0000 <l,c||i,k>_aaaa*t2_aaaa(a,b,j,l)
    triples_res +=  1.000000000000000 * einsum('lcik,abjl->abcijk', g_aaaa[o, v, o, o], t2_aaaa)
    
    #	 -1.0000 <l,c||i,j>_aaaa*t2_aaaa(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcij,abkl->abcijk', g_aaaa[o, v, o, o], t2_aaaa)
    
    #	 -1.0000 <a,b||d,k>_aaaa*t2_aaaa(d,c,i,j)
    triples_res += -1.000000000000000 * einsum('abdk,dcij->abcijk', g_aaaa[v, v, v, o], t2_aaaa)
    
    #	  1.0000 <a,b||d,j>_aaaa*t2_aaaa(d,c,i,k)
    triples_res +=  1.000000000000000 * einsum('abdj,dcik->abcijk', g_aaaa[v, v, v, o], t2_aaaa)
    
    #	 -1.0000 <a,b||d,i>_aaaa*t2_aaaa(d,c,j,k)
    triples_res += -1.000000000000000 * einsum('abdi,dcjk->abcijk', g_aaaa[v, v, v, o], t2_aaaa)
    
    #	  1.0000 <a,c||d,k>_aaaa*t2_aaaa(d,b,i,j)
    triples_res +=  1.000000000000000 * einsum('acdk,dbij->abcijk', g_aaaa[v, v, v, o], t2_aaaa)
    
    #	 -1.0000 <a,c||d,j>_aaaa*t2_aaaa(d,b,i,k)
    triples_res += -1.000000000000000 * einsum('acdj,dbik->abcijk', g_aaaa[v, v, v, o], t2_aaaa)
    
    #	  1.0000 <a,c||d,i>_aaaa*t2_aaaa(d,b,j,k)
    triples_res +=  1.000000000000000 * einsum('acdi,dbjk->abcijk', g_aaaa[v, v, v, o], t2_aaaa)
    
    #	 -1.0000 <b,c||d,k>_aaaa*t2_aaaa(d,a,i,j)
    triples_res += -1.000000000000000 * einsum('bcdk,daij->abcijk', g_aaaa[v, v, v, o], t2_aaaa)
    
    #	  1.0000 <b,c||d,j>_aaaa*t2_aaaa(d,a,i,k)
    triples_res +=  1.000000000000000 * einsum('bcdj,daik->abcijk', g_aaaa[v, v, v, o], t2_aaaa)
    
    #	 -1.0000 <b,c||d,i>_aaaa*t2_aaaa(d,a,j,k)
    triples_res += -1.000000000000000 * einsum('bcdi,dajk->abcijk', g_aaaa[v, v, v, o], t2_aaaa)
    
    #	  0.5000 <m,l||j,k>_aaaa*t3_aaaaaa(a,b,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mljk,abciml->abcijk', g_aaaa[o, o, o, o], t3_aaaaaa)
    
    #	 -0.5000 <m,l||i,k>_aaaa*t3_aaaaaa(a,b,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mlik,abcjml->abcijk', g_aaaa[o, o, o, o], t3_aaaaaa)
    
    #	  0.5000 <m,l||i,j>_aaaa*t3_aaaaaa(a,b,c,k,m,l)
    triples_res +=  0.500000000000000 * einsum('mlij,abckml->abcijk', g_aaaa[o, o, o, o], t3_aaaaaa)
    
    #	  1.0000 <l,a||d,k>_aaaa*t3_aaaaaa(d,b,c,i,j,l)
    triples_res +=  1.000000000000000 * einsum('ladk,dbcijl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa)
    
    #	 -1.0000 <a,l||k,d>_abab*t3_aabaab(c,b,d,i,j,l)
    triples_res += -1.000000000000000 * einsum('alkd,cbdijl->abcijk', g_abab[v, o, o, v], t3_aabaab)
    
    #	 -1.0000 <l,a||d,j>_aaaa*t3_aaaaaa(d,b,c,i,k,l)
    triples_res += -1.000000000000000 * einsum('ladj,dbcikl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa)
    
    #	  1.0000 <a,l||j,d>_abab*t3_aabaab(c,b,d,i,k,l)
    triples_res +=  1.000000000000000 * einsum('aljd,cbdikl->abcijk', g_abab[v, o, o, v], t3_aabaab)
    
    #	  1.0000 <l,a||d,i>_aaaa*t3_aaaaaa(d,b,c,j,k,l)
    triples_res +=  1.000000000000000 * einsum('ladi,dbcjkl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa)
    
    #	 -1.0000 <a,l||i,d>_abab*t3_aabaab(c,b,d,j,k,l)
    triples_res += -1.000000000000000 * einsum('alid,cbdjkl->abcijk', g_abab[v, o, o, v], t3_aabaab)
    
    #	 -1.0000 <l,b||d,k>_aaaa*t3_aaaaaa(d,a,c,i,j,l)
    triples_res += -1.000000000000000 * einsum('lbdk,dacijl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa)
    
    #	  1.0000 <b,l||k,d>_abab*t3_aabaab(c,a,d,i,j,l)
    triples_res +=  1.000000000000000 * einsum('blkd,cadijl->abcijk', g_abab[v, o, o, v], t3_aabaab)
    
    #	  1.0000 <l,b||d,j>_aaaa*t3_aaaaaa(d,a,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('lbdj,dacikl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa)
    
    #	 -1.0000 <b,l||j,d>_abab*t3_aabaab(c,a,d,i,k,l)
    triples_res += -1.000000000000000 * einsum('bljd,cadikl->abcijk', g_abab[v, o, o, v], t3_aabaab)
    
    #	 -1.0000 <l,b||d,i>_aaaa*t3_aaaaaa(d,a,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('lbdi,dacjkl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa)
    
    #	  1.0000 <b,l||i,d>_abab*t3_aabaab(c,a,d,j,k,l)
    triples_res +=  1.000000000000000 * einsum('blid,cadjkl->abcijk', g_abab[v, o, o, v], t3_aabaab)
    
    #	  1.0000 <l,c||d,k>_aaaa*t3_aaaaaa(d,a,b,i,j,l)
    triples_res +=  1.000000000000000 * einsum('lcdk,dabijl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa)
    
    #	 -1.0000 <c,l||k,d>_abab*t3_aabaab(b,a,d,i,j,l)
    triples_res += -1.000000000000000 * einsum('clkd,badijl->abcijk', g_abab[v, o, o, v], t3_aabaab)
    
    #	 -1.0000 <l,c||d,j>_aaaa*t3_aaaaaa(d,a,b,i,k,l)
    triples_res += -1.000000000000000 * einsum('lcdj,dabikl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa)
    
    #	  1.0000 <c,l||j,d>_abab*t3_aabaab(b,a,d,i,k,l)
    triples_res +=  1.000000000000000 * einsum('cljd,badikl->abcijk', g_abab[v, o, o, v], t3_aabaab)
    
    #	  1.0000 <l,c||d,i>_aaaa*t3_aaaaaa(d,a,b,j,k,l)
    triples_res +=  1.000000000000000 * einsum('lcdi,dabjkl->abcijk', g_aaaa[o, v, v, o], t3_aaaaaa)
    
    #	 -1.0000 <c,l||i,d>_abab*t3_aabaab(b,a,d,j,k,l)
    triples_res += -1.000000000000000 * einsum('clid,badjkl->abcijk', g_abab[v, o, o, v], t3_aabaab)
    
    #	  0.5000 <a,b||d,e>_aaaa*t3_aaaaaa(d,e,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('abde,decijk->abcijk', g_aaaa[v, v, v, v], t3_aaaaaa)
    
    #	 -0.5000 <a,c||d,e>_aaaa*t3_aaaaaa(d,e,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('acde,debijk->abcijk', g_aaaa[v, v, v, v], t3_aaaaaa)
    
    #	  0.5000 <b,c||d,e>_aaaa*t3_aaaaaa(d,e,a,i,j,k)
    triples_res +=  0.500000000000000 * einsum('bcde,deaijk->abcijk', g_aaaa[v, v, v, v], t3_aaaaaa)
    
    #	  1.0000 <m,l||j,k>_aaaa*t1_aa(a,l)*t2_aaaa(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mljk,al,bcim->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||j,k>_aaaa*t1_aa(b,l)*t2_aaaa(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mljk,bl,acim->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||j,k>_aaaa*t1_aa(c,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mljk,cl,abim->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,k>_aaaa*t1_aa(a,l)*t2_aaaa(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlik,al,bcjm->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,k>_aaaa*t1_aa(b,l)*t2_aaaa(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlik,bl,acjm->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,k>_aaaa*t1_aa(c,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlik,cl,abjm->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,j>_aaaa*t1_aa(a,l)*t2_aaaa(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlij,al,bckm->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,j>_aaaa*t1_aa(b,l)*t2_aaaa(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlij,bl,ackm->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,j>_aaaa*t1_aa(c,l)*t2_aaaa(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mlij,cl,abkm->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,k>_aaaa*t1_aa(d,j)*t2_aaaa(b,c,i,l)
    triples_res += -1.000000000000000 * einsum('ladk,dj,bcil->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,k>_aaaa*t1_aa(d,i)*t2_aaaa(b,c,j,l)
    triples_res +=  1.000000000000000 * einsum('ladk,di,bcjl->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,k>_aaaa*t1_aa(b,l)*t2_aaaa(d,c,i,j)
    triples_res += -1.000000000000000 * einsum('ladk,bl,dcij->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,k>_aaaa*t1_aa(c,l)*t2_aaaa(d,b,i,j)
    triples_res +=  1.000000000000000 * einsum('ladk,cl,dbij->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,j>_aaaa*t1_aa(d,k)*t2_aaaa(b,c,i,l)
    triples_res +=  1.000000000000000 * einsum('ladj,dk,bcil->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,j>_aaaa*t1_aa(d,i)*t2_aaaa(b,c,k,l)
    triples_res += -1.000000000000000 * einsum('ladj,di,bckl->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,j>_aaaa*t1_aa(b,l)*t2_aaaa(d,c,i,k)
    triples_res +=  1.000000000000000 * einsum('ladj,bl,dcik->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,j>_aaaa*t1_aa(c,l)*t2_aaaa(d,b,i,k)
    triples_res += -1.000000000000000 * einsum('ladj,cl,dbik->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,i>_aaaa*t1_aa(d,k)*t2_aaaa(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('ladi,dk,bcjl->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,i>_aaaa*t1_aa(d,j)*t2_aaaa(b,c,k,l)
    triples_res +=  1.000000000000000 * einsum('ladi,dj,bckl->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,i>_aaaa*t1_aa(b,l)*t2_aaaa(d,c,j,k)
    triples_res += -1.000000000000000 * einsum('ladi,bl,dcjk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,i>_aaaa*t1_aa(c,l)*t2_aaaa(d,b,j,k)
    triples_res +=  1.000000000000000 * einsum('ladi,cl,dbjk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,k>_aaaa*t1_aa(d,j)*t2_aaaa(a,c,i,l)
    triples_res +=  1.000000000000000 * einsum('lbdk,dj,acil->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,k>_aaaa*t1_aa(d,i)*t2_aaaa(a,c,j,l)
    triples_res += -1.000000000000000 * einsum('lbdk,di,acjl->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,k>_aaaa*t1_aa(a,l)*t2_aaaa(d,c,i,j)
    triples_res +=  1.000000000000000 * einsum('lbdk,al,dcij->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,k>_aaaa*t1_aa(c,l)*t2_aaaa(d,a,i,j)
    triples_res += -1.000000000000000 * einsum('lbdk,cl,daij->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,j>_aaaa*t1_aa(d,k)*t2_aaaa(a,c,i,l)
    triples_res += -1.000000000000000 * einsum('lbdj,dk,acil->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,j>_aaaa*t1_aa(d,i)*t2_aaaa(a,c,k,l)
    triples_res +=  1.000000000000000 * einsum('lbdj,di,ackl->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,j>_aaaa*t1_aa(a,l)*t2_aaaa(d,c,i,k)
    triples_res += -1.000000000000000 * einsum('lbdj,al,dcik->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,j>_aaaa*t1_aa(c,l)*t2_aaaa(d,a,i,k)
    triples_res +=  1.000000000000000 * einsum('lbdj,cl,daik->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,i>_aaaa*t1_aa(d,k)*t2_aaaa(a,c,j,l)
    triples_res +=  1.000000000000000 * einsum('lbdi,dk,acjl->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,i>_aaaa*t1_aa(d,j)*t2_aaaa(a,c,k,l)
    triples_res += -1.000000000000000 * einsum('lbdi,dj,ackl->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,i>_aaaa*t1_aa(a,l)*t2_aaaa(d,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lbdi,al,dcjk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,i>_aaaa*t1_aa(c,l)*t2_aaaa(d,a,j,k)
    triples_res += -1.000000000000000 * einsum('lbdi,cl,dajk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,k>_aaaa*t1_aa(d,j)*t2_aaaa(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('lcdk,dj,abil->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,k>_aaaa*t1_aa(d,i)*t2_aaaa(a,b,j,l)
    triples_res +=  1.000000000000000 * einsum('lcdk,di,abjl->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,k>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,j)
    triples_res += -1.000000000000000 * einsum('lcdk,al,dbij->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,k>_aaaa*t1_aa(b,l)*t2_aaaa(d,a,i,j)
    triples_res +=  1.000000000000000 * einsum('lcdk,bl,daij->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,j>_aaaa*t1_aa(d,k)*t2_aaaa(a,b,i,l)
    triples_res +=  1.000000000000000 * einsum('lcdj,dk,abil->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,j>_aaaa*t1_aa(d,i)*t2_aaaa(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcdj,di,abkl->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,j>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,k)
    triples_res +=  1.000000000000000 * einsum('lcdj,al,dbik->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,j>_aaaa*t1_aa(b,l)*t2_aaaa(d,a,i,k)
    triples_res += -1.000000000000000 * einsum('lcdj,bl,daik->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,i>_aaaa*t1_aa(d,k)*t2_aaaa(a,b,j,l)
    triples_res += -1.000000000000000 * einsum('lcdi,dk,abjl->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,i>_aaaa*t1_aa(d,j)*t2_aaaa(a,b,k,l)
    triples_res +=  1.000000000000000 * einsum('lcdi,dj,abkl->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,i>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,j,k)
    triples_res += -1.000000000000000 * einsum('lcdi,al,dbjk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,i>_aaaa*t1_aa(b,l)*t2_aaaa(d,a,j,k)
    triples_res +=  1.000000000000000 * einsum('lcdi,bl,dajk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,b||d,e>_aaaa*t1_aa(d,k)*t2_aaaa(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('abde,dk,ecij->abcijk', g_aaaa[v, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,b||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('abde,dj,ecik->abcijk', g_aaaa[v, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,b||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('abde,di,ecjk->abcijk', g_aaaa[v, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,c||d,e>_aaaa*t1_aa(d,k)*t2_aaaa(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('acde,dk,ebij->abcijk', g_aaaa[v, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,c||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,b,i,k)
    triples_res +=  1.000000000000000 * einsum('acde,dj,ebik->abcijk', g_aaaa[v, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,c||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('acde,di,ebjk->abcijk', g_aaaa[v, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,c||d,e>_aaaa*t1_aa(d,k)*t2_aaaa(e,a,i,j)
    triples_res +=  1.000000000000000 * einsum('bcde,dk,eaij->abcijk', g_aaaa[v, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,c||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,a,i,k)
    triples_res += -1.000000000000000 * einsum('bcde,dj,eaik->abcijk', g_aaaa[v, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,c||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,a,j,k)
    triples_res +=  1.000000000000000 * einsum('bcde,di,eajk->abcijk', g_aaaa[v, v, v, v], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_aaaa*t1_aa(d,l)*t3_aaaaaa(a,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dl,abcijm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||k,d>_abab*t1_bb(d,l)*t3_aaaaaa(a,b,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('mlkd,dl,abcijm->abcijk', g_abab[o, o, o, v], t1_bb, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,k>_aaaa*t1_aa(d,j)*t3_aaaaaa(a,b,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mldk,dj,abciml->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,k>_aaaa*t1_aa(d,i)*t3_aaaaaa(a,b,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mldk,di,abcjml->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_aaaa*t1_aa(a,l)*t3_aaaaaa(d,b,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,al,dbcijm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||k,d>_abab*t1_aa(a,l)*t3_aabaab(c,b,d,i,j,m)
    triples_res +=  1.000000000000000 * einsum('lmkd,al,cbdijm->abcijk', g_abab[o, o, o, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_aaaa*t1_aa(b,l)*t3_aaaaaa(d,a,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,bl,dacijm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||k,d>_abab*t1_aa(b,l)*t3_aabaab(c,a,d,i,j,m)
    triples_res += -1.000000000000000 * einsum('lmkd,bl,cadijm->abcijk', g_abab[o, o, o, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_aaaa*t1_aa(c,l)*t3_aaaaaa(d,a,b,i,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,cl,dabijm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||k,d>_abab*t1_aa(c,l)*t3_aabaab(b,a,d,i,j,m)
    triples_res +=  1.000000000000000 * einsum('lmkd,cl,badijm->abcijk', g_abab[o, o, o, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t1_aa(d,l)*t3_aaaaaa(a,b,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('mldj,dl,abcikm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||j,d>_abab*t1_bb(d,l)*t3_aaaaaa(a,b,c,i,k,m)
    triples_res +=  1.000000000000000 * einsum('mljd,dl,abcikm->abcijk', g_abab[o, o, o, v], t1_bb, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,j>_aaaa*t1_aa(d,k)*t3_aaaaaa(a,b,c,i,m,l)
    triples_res += -0.500000000000000 * einsum('mldj,dk,abciml->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,j>_aaaa*t1_aa(d,i)*t3_aaaaaa(a,b,c,k,m,l)
    triples_res +=  0.500000000000000 * einsum('mldj,di,abckml->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t1_aa(a,l)*t3_aaaaaa(d,b,c,i,k,m)
    triples_res +=  1.000000000000000 * einsum('mldj,al,dbcikm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||j,d>_abab*t1_aa(a,l)*t3_aabaab(c,b,d,i,k,m)
    triples_res += -1.000000000000000 * einsum('lmjd,al,cbdikm->abcijk', g_abab[o, o, o, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t1_aa(b,l)*t3_aaaaaa(d,a,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('mldj,bl,dacikm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||j,d>_abab*t1_aa(b,l)*t3_aabaab(c,a,d,i,k,m)
    triples_res +=  1.000000000000000 * einsum('lmjd,bl,cadikm->abcijk', g_abab[o, o, o, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t1_aa(c,l)*t3_aaaaaa(d,a,b,i,k,m)
    triples_res +=  1.000000000000000 * einsum('mldj,cl,dabikm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||j,d>_abab*t1_aa(c,l)*t3_aabaab(b,a,d,i,k,m)
    triples_res += -1.000000000000000 * einsum('lmjd,cl,badikm->abcijk', g_abab[o, o, o, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t1_aa(d,l)*t3_aaaaaa(a,b,c,j,k,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dl,abcjkm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,d>_abab*t1_bb(d,l)*t3_aaaaaa(a,b,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('mlid,dl,abcjkm->abcijk', g_abab[o, o, o, v], t1_bb, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,i>_aaaa*t1_aa(d,k)*t3_aaaaaa(a,b,c,j,m,l)
    triples_res +=  0.500000000000000 * einsum('mldi,dk,abcjml->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,i>_aaaa*t1_aa(d,j)*t3_aaaaaa(a,b,c,k,m,l)
    triples_res += -0.500000000000000 * einsum('mldi,dj,abckml->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t1_aa(a,l)*t3_aaaaaa(d,b,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('mldi,al,dbcjkm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||i,d>_abab*t1_aa(a,l)*t3_aabaab(c,b,d,j,k,m)
    triples_res +=  1.000000000000000 * einsum('lmid,al,cbdjkm->abcijk', g_abab[o, o, o, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t1_aa(b,l)*t3_aaaaaa(d,a,c,j,k,m)
    triples_res +=  1.000000000000000 * einsum('mldi,bl,dacjkm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||i,d>_abab*t1_aa(b,l)*t3_aabaab(c,a,d,j,k,m)
    triples_res += -1.000000000000000 * einsum('lmid,bl,cadjkm->abcijk', g_abab[o, o, o, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t1_aa(c,l)*t3_aaaaaa(d,a,b,j,k,m)
    triples_res += -1.000000000000000 * einsum('mldi,cl,dabjkm->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||i,d>_abab*t1_aa(c,l)*t3_aabaab(b,a,d,j,k,m)
    triples_res +=  1.000000000000000 * einsum('lmid,cl,badjkm->abcijk', g_abab[o, o, o, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t1_aa(d,l)*t3_aaaaaa(e,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lade,dl,ebcijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||e,d>_abab*t1_bb(d,l)*t3_aaaaaa(e,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('aled,dl,ebcijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t1_aa(d,k)*t3_aaaaaa(e,b,c,i,j,l)
    triples_res += -1.000000000000000 * einsum('lade,dk,ebcijl->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||d,e>_abab*t1_aa(d,k)*t3_aabaab(c,b,e,i,j,l)
    triples_res += -1.000000000000000 * einsum('alde,dk,cbeijl->abcijk', g_abab[v, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t1_aa(d,j)*t3_aaaaaa(e,b,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('lade,dj,ebcikl->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||d,e>_abab*t1_aa(d,j)*t3_aabaab(c,b,e,i,k,l)
    triples_res +=  1.000000000000000 * einsum('alde,dj,cbeikl->abcijk', g_abab[v, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t1_aa(d,i)*t3_aaaaaa(e,b,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('lade,di,ebcjkl->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||d,e>_abab*t1_aa(d,i)*t3_aabaab(c,b,e,j,k,l)
    triples_res += -1.000000000000000 * einsum('alde,di,cbejkl->abcijk', g_abab[v, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,a||d,e>_aaaa*t1_aa(b,l)*t3_aaaaaa(d,e,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lade,bl,decijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,a||d,e>_aaaa*t1_aa(c,l)*t3_aaaaaa(d,e,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('lade,cl,debijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t1_aa(d,l)*t3_aaaaaa(e,a,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('lbde,dl,eacijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,l||e,d>_abab*t1_bb(d,l)*t3_aaaaaa(e,a,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('bled,dl,eacijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_aaaa*t1_aa(d,k)*t3_aaaaaa(e,a,c,i,j,l)
    triples_res +=  1.000000000000000 * einsum('lbde,dk,eacijl->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,l||d,e>_abab*t1_aa(d,k)*t3_aabaab(c,a,e,i,j,l)
    triples_res +=  1.000000000000000 * einsum('blde,dk,caeijl->abcijk', g_abab[v, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t1_aa(d,j)*t3_aaaaaa(e,a,c,i,k,l)
    triples_res += -1.000000000000000 * einsum('lbde,dj,eacikl->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,l||d,e>_abab*t1_aa(d,j)*t3_aabaab(c,a,e,i,k,l)
    triples_res += -1.000000000000000 * einsum('blde,dj,caeikl->abcijk', g_abab[v, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_aaaa*t1_aa(d,i)*t3_aaaaaa(e,a,c,j,k,l)
    triples_res +=  1.000000000000000 * einsum('lbde,di,eacjkl->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,l||d,e>_abab*t1_aa(d,i)*t3_aabaab(c,a,e,j,k,l)
    triples_res +=  1.000000000000000 * einsum('blde,di,caejkl->abcijk', g_abab[v, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,b||d,e>_aaaa*t1_aa(a,l)*t3_aaaaaa(d,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lbde,al,decijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,b||d,e>_aaaa*t1_aa(c,l)*t3_aaaaaa(d,e,a,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lbde,cl,deaijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_aaaa*t1_aa(d,l)*t3_aaaaaa(e,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dl,eabijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <c,l||e,d>_abab*t1_bb(d,l)*t3_aaaaaa(e,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('cled,dl,eabijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_aaaa*t1_aa(d,k)*t3_aaaaaa(e,a,b,i,j,l)
    triples_res += -1.000000000000000 * einsum('lcde,dk,eabijl->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <c,l||d,e>_abab*t1_aa(d,k)*t3_aabaab(b,a,e,i,j,l)
    triples_res += -1.000000000000000 * einsum('clde,dk,baeijl->abcijk', g_abab[v, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_aaaa*t1_aa(d,j)*t3_aaaaaa(e,a,b,i,k,l)
    triples_res +=  1.000000000000000 * einsum('lcde,dj,eabikl->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <c,l||d,e>_abab*t1_aa(d,j)*t3_aabaab(b,a,e,i,k,l)
    triples_res +=  1.000000000000000 * einsum('clde,dj,baeikl->abcijk', g_abab[v, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_aaaa*t1_aa(d,i)*t3_aaaaaa(e,a,b,j,k,l)
    triples_res += -1.000000000000000 * einsum('lcde,di,eabjkl->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <c,l||d,e>_abab*t1_aa(d,i)*t3_aabaab(b,a,e,j,k,l)
    triples_res += -1.000000000000000 * einsum('clde,di,baejkl->abcijk', g_abab[v, o, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,c||d,e>_aaaa*t1_aa(a,l)*t3_aaaaaa(d,e,b,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lcde,al,debijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,c||d,e>_aaaa*t1_aa(b,l)*t3_aaaaaa(d,e,a,i,j,k)
    triples_res += -0.500000000000000 * einsum('lcde,bl,deaijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,e,k,l)*t3_aaaaaa(a,b,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,dekl,abcijm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_abab(d,e,k,l)*t3_aaaaaa(a,b,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,dekl,abcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_abab(e,d,k,l)*t3_aaaaaa(a,b,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('mled,edkl,abcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,e,j,l)*t3_aaaaaa(a,b,c,i,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,dejl,abcikm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_abab(d,e,j,l)*t3_aaaaaa(a,b,c,i,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,dejl,abcikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_abab(e,d,j,l)*t3_aaaaaa(a,b,c,i,k,m)
    triples_res +=  0.500000000000000 * einsum('mled,edjl,abcikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,e,i,l)*t3_aaaaaa(a,b,c,j,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,deil,abcjkm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_abab(d,e,i,l)*t3_aaaaaa(a,b,c,j,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,deil,abcjkm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_abab(e,d,i,l)*t3_aaaaaa(a,b,c,j,k,m)
    triples_res += -0.500000000000000 * einsum('mled,edil,abcjkm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_aaaa*t2_aaaa(d,e,j,k)*t3_aaaaaa(a,b,c,i,m,l)
    triples_res +=  0.250000000000000 * einsum('mlde,dejk,abciml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,l||d,e>_aaaa*t2_aaaa(d,e,i,k)*t3_aaaaaa(a,b,c,j,m,l)
    triples_res += -0.250000000000000 * einsum('mlde,deik,abcjml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_aaaa*t2_aaaa(d,e,i,j)*t3_aaaaaa(a,b,c,k,m,l)
    triples_res +=  0.250000000000000 * einsum('mlde,deij,abckml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,a,m,l)*t3_aaaaaa(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,daml,ebcijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_abab(a,d,m,l)*t3_aaaaaa(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mled,adml,ebcijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(a,d,l,m)*t3_aaaaaa(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lmed,adlm,ebcijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,a,k,l)*t3_aaaaaa(e,b,c,i,j,m)
    triples_res +=  1.000000000000010 * einsum('mlde,dakl,ebcijm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_aaaa(d,a,k,l)*t3_aabaab(c,b,e,i,j,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dakl,cbeijm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_abab(a,d,k,l)*t3_aaaaaa(e,b,c,i,j,m)
    triples_res +=  1.000000000000010 * einsum('mled,adkl,ebcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,k,l)*t3_aabaab(c,b,e,i,j,m)
    triples_res +=  1.000000000000010 * einsum('mlde,adkl,cbeijm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,a,j,l)*t3_aaaaaa(e,b,c,i,k,m)
    triples_res += -1.000000000000010 * einsum('mlde,dajl,ebcikm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_aaaa(d,a,j,l)*t3_aabaab(c,b,e,i,k,m)
    triples_res += -1.000000000000010 * einsum('lmde,dajl,cbeikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_abab(a,d,j,l)*t3_aaaaaa(e,b,c,i,k,m)
    triples_res += -1.000000000000010 * einsum('mled,adjl,ebcikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,j,l)*t3_aabaab(c,b,e,i,k,m)
    triples_res += -1.000000000000010 * einsum('mlde,adjl,cbeikm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,a,i,l)*t3_aaaaaa(e,b,c,j,k,m)
    triples_res +=  1.000000000000010 * einsum('mlde,dail,ebcjkm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_aaaa(d,a,i,l)*t3_aabaab(c,b,e,j,k,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dail,cbejkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_abab(a,d,i,l)*t3_aaaaaa(e,b,c,j,k,m)
    triples_res +=  1.000000000000010 * einsum('mled,adil,ebcjkm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,i,l)*t3_aabaab(c,b,e,j,k,m)
    triples_res +=  1.000000000000010 * einsum('mlde,adil,cbejkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,a,j,k)*t3_aaaaaa(e,b,c,i,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dajk,ebciml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_aaaa(d,a,j,k)*t3_aabaab(c,b,e,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dajk,cbeiml->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_aaaa(d,a,j,k)*t3_aabaab(c,b,e,i,l,m)
    triples_res +=  0.500000000000000 * einsum('lmde,dajk,cbeilm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,a,i,k)*t3_aaaaaa(e,b,c,j,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,daik,ebcjml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_aaaa(d,a,i,k)*t3_aabaab(c,b,e,j,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,daik,cbejml->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_aaaa(d,a,i,k)*t3_aabaab(c,b,e,j,l,m)
    triples_res += -0.500000000000000 * einsum('lmde,daik,cbejlm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,a,i,j)*t3_aaaaaa(e,b,c,k,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,daij,ebckml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_aaaa(d,a,i,j)*t3_aabaab(c,b,e,k,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,daij,cbekml->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_aaaa(d,a,i,j)*t3_aabaab(c,b,e,k,l,m)
    triples_res +=  0.500000000000000 * einsum('lmde,daij,cbeklm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,b,m,l)*t3_aaaaaa(e,a,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('mlde,dbml,eacijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_abab(b,d,m,l)*t3_aaaaaa(e,a,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('mled,bdml,eacijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_abab(b,d,l,m)*t3_aaaaaa(e,a,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lmed,bdlm,eacijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,b,k,l)*t3_aaaaaa(e,a,c,i,j,m)
    triples_res += -1.000000000000010 * einsum('mlde,dbkl,eacijm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_aaaa(d,b,k,l)*t3_aabaab(c,a,e,i,j,m)
    triples_res += -1.000000000000010 * einsum('lmde,dbkl,caeijm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_abab(b,d,k,l)*t3_aaaaaa(e,a,c,i,j,m)
    triples_res += -1.000000000000010 * einsum('mled,bdkl,eacijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(b,d,k,l)*t3_aabaab(c,a,e,i,j,m)
    triples_res += -1.000000000000010 * einsum('mlde,bdkl,caeijm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,b,j,l)*t3_aaaaaa(e,a,c,i,k,m)
    triples_res +=  1.000000000000010 * einsum('mlde,dbjl,eacikm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_aaaa(d,b,j,l)*t3_aabaab(c,a,e,i,k,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dbjl,caeikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_abab(b,d,j,l)*t3_aaaaaa(e,a,c,i,k,m)
    triples_res +=  1.000000000000010 * einsum('mled,bdjl,eacikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(b,d,j,l)*t3_aabaab(c,a,e,i,k,m)
    triples_res +=  1.000000000000010 * einsum('mlde,bdjl,caeikm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,b,i,l)*t3_aaaaaa(e,a,c,j,k,m)
    triples_res += -1.000000000000010 * einsum('mlde,dbil,eacjkm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_aaaa(d,b,i,l)*t3_aabaab(c,a,e,j,k,m)
    triples_res += -1.000000000000010 * einsum('lmde,dbil,caejkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_abab(b,d,i,l)*t3_aaaaaa(e,a,c,j,k,m)
    triples_res += -1.000000000000010 * einsum('mled,bdil,eacjkm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(b,d,i,l)*t3_aabaab(c,a,e,j,k,m)
    triples_res += -1.000000000000010 * einsum('mlde,bdil,caejkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,b,j,k)*t3_aaaaaa(e,a,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dbjk,eaciml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_aaaa(d,b,j,k)*t3_aabaab(c,a,e,i,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dbjk,caeiml->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_aaaa(d,b,j,k)*t3_aabaab(c,a,e,i,l,m)
    triples_res += -0.500000000000000 * einsum('lmde,dbjk,caeilm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,b,i,k)*t3_aaaaaa(e,a,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dbik,eacjml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_aaaa(d,b,i,k)*t3_aabaab(c,a,e,j,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dbik,caejml->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_aaaa(d,b,i,k)*t3_aabaab(c,a,e,j,l,m)
    triples_res +=  0.500000000000000 * einsum('lmde,dbik,caejlm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,b,i,j)*t3_aaaaaa(e,a,c,k,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dbij,eackml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_aaaa(d,b,i,j)*t3_aabaab(c,a,e,k,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dbij,caekml->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_aaaa(d,b,i,j)*t3_aabaab(c,a,e,k,l,m)
    triples_res += -0.500000000000000 * einsum('lmde,dbij,caeklm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,c,m,l)*t3_aaaaaa(e,a,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,dcml,eabijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_abab(c,d,m,l)*t3_aaaaaa(e,a,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('mled,cdml,eabijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(c,d,l,m)*t3_aaaaaa(e,a,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('lmed,cdlm,eabijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,c,k,l)*t3_aaaaaa(e,a,b,i,j,m)
    triples_res +=  1.000000000000010 * einsum('mlde,dckl,eabijm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_aaaa(d,c,k,l)*t3_aabaab(b,a,e,i,j,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dckl,baeijm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_abab(c,d,k,l)*t3_aaaaaa(e,a,b,i,j,m)
    triples_res +=  1.000000000000010 * einsum('mled,cdkl,eabijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(c,d,k,l)*t3_aabaab(b,a,e,i,j,m)
    triples_res +=  1.000000000000010 * einsum('mlde,cdkl,baeijm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,c,j,l)*t3_aaaaaa(e,a,b,i,k,m)
    triples_res += -1.000000000000010 * einsum('mlde,dcjl,eabikm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_aaaa(d,c,j,l)*t3_aabaab(b,a,e,i,k,m)
    triples_res += -1.000000000000010 * einsum('lmde,dcjl,baeikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_abab(c,d,j,l)*t3_aaaaaa(e,a,b,i,k,m)
    triples_res += -1.000000000000010 * einsum('mled,cdjl,eabikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(c,d,j,l)*t3_aabaab(b,a,e,i,k,m)
    triples_res += -1.000000000000010 * einsum('mlde,cdjl,baeikm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,c,i,l)*t3_aaaaaa(e,a,b,j,k,m)
    triples_res +=  1.000000000000010 * einsum('mlde,dcil,eabjkm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_aaaa(d,c,i,l)*t3_aabaab(b,a,e,j,k,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dcil,baejkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_abab(c,d,i,l)*t3_aaaaaa(e,a,b,j,k,m)
    triples_res +=  1.000000000000010 * einsum('mled,cdil,eabjkm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(c,d,i,l)*t3_aabaab(b,a,e,j,k,m)
    triples_res +=  1.000000000000010 * einsum('mlde,cdil,baejkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,c,j,k)*t3_aaaaaa(e,a,b,i,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dcjk,eabiml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_aaaa(d,c,j,k)*t3_aabaab(b,a,e,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dcjk,baeiml->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_aaaa(d,c,j,k)*t3_aabaab(b,a,e,i,l,m)
    triples_res +=  0.500000000000000 * einsum('lmde,dcjk,baeilm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,c,i,k)*t3_aaaaaa(e,a,b,j,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dcik,eabjml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_aaaa(d,c,i,k)*t3_aabaab(b,a,e,j,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dcik,baejml->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_aaaa(d,c,i,k)*t3_aabaab(b,a,e,j,l,m)
    triples_res += -0.500000000000000 * einsum('lmde,dcik,baejlm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,c,i,j)*t3_aaaaaa(e,a,b,k,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dcij,eabkml->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_aaaa(d,c,i,j)*t3_aabaab(b,a,e,k,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dcij,baekml->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_aaaa(d,c,i,j)*t3_aabaab(b,a,e,k,l,m)
    triples_res +=  0.500000000000000 * einsum('lmde,dcij,baeklm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_aaaa*t2_aaaa(a,b,m,l)*t3_aaaaaa(d,e,c,i,j,k)
    triples_res +=  0.250000000000000 * einsum('mlde,abml,decijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,b,k,l)*t3_aaaaaa(d,e,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,abkl,decijm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_aaaa(a,b,k,l)*t3_aabaab(d,c,e,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmde,abkl,dceijm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_aaaa(a,b,k,l)*t3_aabaab(c,e,d,i,j,m)
    triples_res +=  0.500000000000000 * einsum('lmed,abkl,cedijm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,b,j,l)*t3_aaaaaa(d,e,c,i,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,abjl,decikm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_aaaa(a,b,j,l)*t3_aabaab(d,c,e,i,k,m)
    triples_res +=  0.500000000000000 * einsum('lmde,abjl,dceikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_aaaa(a,b,j,l)*t3_aabaab(c,e,d,i,k,m)
    triples_res += -0.500000000000000 * einsum('lmed,abjl,cedikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,b,i,l)*t3_aaaaaa(d,e,c,j,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,abil,decjkm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_aaaa(a,b,i,l)*t3_aabaab(d,c,e,j,k,m)
    triples_res += -0.500000000000000 * einsum('lmde,abil,dcejkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_aaaa(a,b,i,l)*t3_aabaab(c,e,d,j,k,m)
    triples_res +=  0.500000000000000 * einsum('lmed,abil,cedjkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <m,l||d,e>_aaaa*t2_aaaa(a,c,m,l)*t3_aaaaaa(d,e,b,i,j,k)
    triples_res += -0.250000000000000 * einsum('mlde,acml,debijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,c,k,l)*t3_aaaaaa(d,e,b,i,j,m)
    triples_res +=  0.500000000000000 * einsum('mlde,ackl,debijm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_aaaa(a,c,k,l)*t3_aabaab(d,b,e,i,j,m)
    triples_res +=  0.500000000000000 * einsum('lmde,ackl,dbeijm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_aaaa(a,c,k,l)*t3_aabaab(b,e,d,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmed,ackl,bedijm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,c,j,l)*t3_aaaaaa(d,e,b,i,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,acjl,debikm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_aaaa(a,c,j,l)*t3_aabaab(d,b,e,i,k,m)
    triples_res += -0.500000000000000 * einsum('lmde,acjl,dbeikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_aaaa(a,c,j,l)*t3_aabaab(b,e,d,i,k,m)
    triples_res +=  0.500000000000000 * einsum('lmed,acjl,bedikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,c,i,l)*t3_aaaaaa(d,e,b,j,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,acil,debjkm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_aaaa(a,c,i,l)*t3_aabaab(d,b,e,j,k,m)
    triples_res +=  0.500000000000000 * einsum('lmde,acil,dbejkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_aaaa(a,c,i,l)*t3_aabaab(b,e,d,j,k,m)
    triples_res += -0.500000000000000 * einsum('lmed,acil,bedjkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_aaaa*t2_aaaa(b,c,m,l)*t3_aaaaaa(d,e,a,i,j,k)
    triples_res +=  0.250000000000000 * einsum('mlde,bcml,deaijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(b,c,k,l)*t3_aaaaaa(d,e,a,i,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,bckl,deaijm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_aaaa(b,c,k,l)*t3_aabaab(d,a,e,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmde,bckl,daeijm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_aaaa(b,c,k,l)*t3_aabaab(a,e,d,i,j,m)
    triples_res +=  0.500000000000000 * einsum('lmed,bckl,aedijm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(b,c,j,l)*t3_aaaaaa(d,e,a,i,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,bcjl,deaikm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_aaaa(b,c,j,l)*t3_aabaab(d,a,e,i,k,m)
    triples_res +=  0.500000000000000 * einsum('lmde,bcjl,daeikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_aaaa(b,c,j,l)*t3_aabaab(a,e,d,i,k,m)
    triples_res += -0.500000000000000 * einsum('lmed,bcjl,aedikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(b,c,i,l)*t3_aaaaaa(d,e,a,j,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,bcil,deajkm->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_aaaa(b,c,i,l)*t3_aabaab(d,a,e,j,k,m)
    triples_res += -0.500000000000000 * einsum('lmde,bcil,daejkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_aaaa(b,c,i,l)*t3_aabaab(a,e,d,j,k,m)
    triples_res +=  0.500000000000000 * einsum('lmed,bcil,aedjkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_aaaa*t2_aaaa(d,a,j,l)*t2_aaaa(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dajl,bcim->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||k,d>_abab*t2_abab(a,d,j,l)*t2_aaaa(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlkd,adjl,bcim->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_aaaa*t2_aaaa(d,a,i,l)*t2_aaaa(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,dail,bcjm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||k,d>_abab*t2_abab(a,d,i,l)*t2_aaaa(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlkd,adil,bcjm->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,k>_aaaa*t2_aaaa(d,a,i,j)*t2_aaaa(b,c,m,l)
    triples_res += -0.500000000000000 * einsum('mldk,daij,bcml->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_aaaa*t2_aaaa(d,b,j,l)*t2_aaaa(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mldk,dbjl,acim->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||k,d>_abab*t2_abab(b,d,j,l)*t2_aaaa(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlkd,bdjl,acim->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_aaaa*t2_aaaa(d,b,i,l)*t2_aaaa(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dbil,acjm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||k,d>_abab*t2_abab(b,d,i,l)*t2_aaaa(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlkd,bdil,acjm->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,k>_aaaa*t2_aaaa(d,b,i,j)*t2_aaaa(a,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mldk,dbij,acml->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_aaaa*t2_aaaa(d,c,j,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dcjl,abim->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||k,d>_abab*t2_abab(c,d,j,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mlkd,cdjl,abim->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_aaaa*t2_aaaa(d,c,i,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,dcil,abjm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||k,d>_abab*t2_abab(c,d,i,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlkd,cdil,abjm->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,k>_aaaa*t2_aaaa(d,c,i,j)*t2_aaaa(a,b,m,l)
    triples_res += -0.500000000000000 * einsum('mldk,dcij,abml->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t2_aaaa(d,a,k,l)*t2_aaaa(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('mldj,dakl,bcim->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||j,d>_abab*t2_abab(a,d,k,l)*t2_aaaa(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('mljd,adkl,bcim->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t2_aaaa(d,a,i,l)*t2_aaaa(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mldj,dail,bckm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||j,d>_abab*t2_abab(a,d,i,l)*t2_aaaa(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mljd,adil,bckm->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,j>_aaaa*t2_aaaa(d,a,i,k)*t2_aaaa(b,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mldj,daik,bcml->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t2_aaaa(d,b,k,l)*t2_aaaa(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mldj,dbkl,acim->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||j,d>_abab*t2_abab(b,d,k,l)*t2_aaaa(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mljd,bdkl,acim->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t2_aaaa(d,b,i,l)*t2_aaaa(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('mldj,dbil,ackm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||j,d>_abab*t2_abab(b,d,i,l)*t2_aaaa(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('mljd,bdil,ackm->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,j>_aaaa*t2_aaaa(d,b,i,k)*t2_aaaa(a,c,m,l)
    triples_res += -0.500000000000000 * einsum('mldj,dbik,acml->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t2_aaaa(d,c,k,l)*t2_aaaa(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mldj,dckl,abim->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||j,d>_abab*t2_abab(c,d,k,l)*t2_aaaa(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mljd,cdkl,abim->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t2_aaaa(d,c,i,l)*t2_aaaa(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mldj,dcil,abkm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||j,d>_abab*t2_abab(c,d,i,l)*t2_aaaa(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mljd,cdil,abkm->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,j>_aaaa*t2_aaaa(d,c,i,k)*t2_aaaa(a,b,m,l)
    triples_res +=  0.500000000000000 * einsum('mldj,dcik,abml->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t2_aaaa(d,a,k,l)*t2_aaaa(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dakl,bcjm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,d>_abab*t2_abab(a,d,k,l)*t2_aaaa(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlid,adkl,bcjm->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t2_aaaa(d,a,j,l)*t2_aaaa(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mldi,dajl,bckm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,d>_abab*t2_abab(a,d,j,l)*t2_aaaa(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlid,adjl,bckm->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,i>_aaaa*t2_aaaa(d,a,j,k)*t2_aaaa(b,c,m,l)
    triples_res += -0.500000000000000 * einsum('mldi,dajk,bcml->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t2_aaaa(d,b,k,l)*t2_aaaa(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('mldi,dbkl,acjm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,d>_abab*t2_abab(b,d,k,l)*t2_aaaa(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlid,bdkl,acjm->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t2_aaaa(d,b,j,l)*t2_aaaa(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dbjl,ackm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,d>_abab*t2_abab(b,d,j,l)*t2_aaaa(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlid,bdjl,ackm->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,i>_aaaa*t2_aaaa(d,b,j,k)*t2_aaaa(a,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mldi,dbjk,acml->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t2_aaaa(d,c,k,l)*t2_aaaa(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dckl,abjm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,d>_abab*t2_abab(c,d,k,l)*t2_aaaa(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mlid,cdkl,abjm->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t2_aaaa(d,c,j,l)*t2_aaaa(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('mldi,dcjl,abkm->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,d>_abab*t2_abab(c,d,j,l)*t2_aaaa(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('mlid,cdjl,abkm->abcijk', g_abab[o, o, o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,i>_aaaa*t2_aaaa(d,c,j,k)*t2_aaaa(a,b,m,l)
    triples_res += -0.500000000000000 * einsum('mldi,dcjk,abml->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,a||d,e>_aaaa*t2_aaaa(d,e,j,k)*t2_aaaa(b,c,i,l)
    triples_res += -0.500000000000000 * einsum('lade,dejk,bcil->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,a||d,e>_aaaa*t2_aaaa(d,e,i,k)*t2_aaaa(b,c,j,l)
    triples_res +=  0.500000000000000 * einsum('lade,deik,bcjl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,a||d,e>_aaaa*t2_aaaa(d,e,i,j)*t2_aaaa(b,c,k,l)
    triples_res += -0.500000000000000 * einsum('lade,deij,bckl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t2_aaaa(d,b,k,l)*t2_aaaa(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('lade,dbkl,ecij->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||e,d>_abab*t2_abab(b,d,k,l)*t2_aaaa(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('aled,bdkl,ecij->abcijk', g_abab[v, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t2_aaaa(d,b,j,l)*t2_aaaa(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('lade,dbjl,ecik->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||e,d>_abab*t2_abab(b,d,j,l)*t2_aaaa(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('aled,bdjl,ecik->abcijk', g_abab[v, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t2_aaaa(d,b,i,l)*t2_aaaa(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lade,dbil,ecjk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||e,d>_abab*t2_abab(b,d,i,l)*t2_aaaa(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('aled,bdil,ecjk->abcijk', g_abab[v, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t2_aaaa(d,b,j,k)*t2_aaaa(e,c,i,l)
    triples_res +=  1.000000000000000 * einsum('lade,dbjk,ecil->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <a,l||d,e>_abab*t2_aaaa(d,b,j,k)*t2_abab(c,e,i,l)
    triples_res +=  1.000000000000000 * einsum('alde,dbjk,ceil->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t2_aaaa(d,b,i,k)*t2_aaaa(e,c,j,l)
    triples_res += -1.000000000000000 * einsum('lade,dbik,ecjl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <a,l||d,e>_abab*t2_aaaa(d,b,i,k)*t2_abab(c,e,j,l)
    triples_res += -1.000000000000000 * einsum('alde,dbik,cejl->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t2_aaaa(d,b,i,j)*t2_aaaa(e,c,k,l)
    triples_res +=  1.000000000000000 * einsum('lade,dbij,eckl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <a,l||d,e>_abab*t2_aaaa(d,b,i,j)*t2_abab(c,e,k,l)
    triples_res +=  1.000000000000000 * einsum('alde,dbij,cekl->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,b||d,e>_aaaa*t2_aaaa(d,e,j,k)*t2_aaaa(a,c,i,l)
    triples_res +=  0.500000000000000 * einsum('lbde,dejk,acil->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,b||d,e>_aaaa*t2_aaaa(d,e,i,k)*t2_aaaa(a,c,j,l)
    triples_res += -0.500000000000000 * einsum('lbde,deik,acjl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,b||d,e>_aaaa*t2_aaaa(d,e,i,j)*t2_aaaa(a,c,k,l)
    triples_res +=  0.500000000000000 * einsum('lbde,deij,ackl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t2_aaaa(d,a,k,l)*t2_aaaa(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('lbde,dakl,ecij->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,l||e,d>_abab*t2_abab(a,d,k,l)*t2_aaaa(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('bled,adkl,ecij->abcijk', g_abab[v, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_aaaa*t2_aaaa(d,a,j,l)*t2_aaaa(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('lbde,dajl,ecik->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,l||e,d>_abab*t2_abab(a,d,j,l)*t2_aaaa(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('bled,adjl,ecik->abcijk', g_abab[v, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t2_aaaa(d,a,i,l)*t2_aaaa(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('lbde,dail,ecjk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,l||e,d>_abab*t2_abab(a,d,i,l)*t2_aaaa(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('bled,adil,ecjk->abcijk', g_abab[v, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t2_aaaa(d,a,j,k)*t2_aaaa(e,c,i,l)
    triples_res += -1.000000000000000 * einsum('lbde,dajk,ecil->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <b,l||d,e>_abab*t2_aaaa(d,a,j,k)*t2_abab(c,e,i,l)
    triples_res += -1.000000000000000 * einsum('blde,dajk,ceil->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_aaaa*t2_aaaa(d,a,i,k)*t2_aaaa(e,c,j,l)
    triples_res +=  1.000000000000000 * einsum('lbde,daik,ecjl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <b,l||d,e>_abab*t2_aaaa(d,a,i,k)*t2_abab(c,e,j,l)
    triples_res +=  1.000000000000000 * einsum('blde,daik,cejl->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t2_aaaa(d,a,i,j)*t2_aaaa(e,c,k,l)
    triples_res += -1.000000000000000 * einsum('lbde,daij,eckl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <b,l||d,e>_abab*t2_aaaa(d,a,i,j)*t2_abab(c,e,k,l)
    triples_res += -1.000000000000000 * einsum('blde,daij,cekl->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,c||d,e>_aaaa*t2_aaaa(d,e,j,k)*t2_aaaa(a,b,i,l)
    triples_res += -0.500000000000000 * einsum('lcde,dejk,abil->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,c||d,e>_aaaa*t2_aaaa(d,e,i,k)*t2_aaaa(a,b,j,l)
    triples_res +=  0.500000000000000 * einsum('lcde,deik,abjl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,c||d,e>_aaaa*t2_aaaa(d,e,i,j)*t2_aaaa(a,b,k,l)
    triples_res += -0.500000000000000 * einsum('lcde,deij,abkl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_aaaa*t2_aaaa(d,a,k,l)*t2_aaaa(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('lcde,dakl,ebij->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <c,l||e,d>_abab*t2_abab(a,d,k,l)*t2_aaaa(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('cled,adkl,ebij->abcijk', g_abab[v, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_aaaa*t2_aaaa(d,a,j,l)*t2_aaaa(e,b,i,k)
    triples_res += -1.000000000000000 * einsum('lcde,dajl,ebik->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <c,l||e,d>_abab*t2_abab(a,d,j,l)*t2_aaaa(e,b,i,k)
    triples_res +=  1.000000000000000 * einsum('cled,adjl,ebik->abcijk', g_abab[v, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_aaaa*t2_aaaa(d,a,i,l)*t2_aaaa(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dail,ebjk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <c,l||e,d>_abab*t2_abab(a,d,i,l)*t2_aaaa(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('cled,adil,ebjk->abcijk', g_abab[v, o, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_aaaa*t2_aaaa(d,a,j,k)*t2_aaaa(e,b,i,l)
    triples_res +=  1.000000000000000 * einsum('lcde,dajk,ebil->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <c,l||d,e>_abab*t2_aaaa(d,a,j,k)*t2_abab(b,e,i,l)
    triples_res +=  1.000000000000000 * einsum('clde,dajk,beil->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_aaaa*t2_aaaa(d,a,i,k)*t2_aaaa(e,b,j,l)
    triples_res += -1.000000000000000 * einsum('lcde,daik,ebjl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <c,l||d,e>_abab*t2_aaaa(d,a,i,k)*t2_abab(b,e,j,l)
    triples_res += -1.000000000000000 * einsum('clde,daik,bejl->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_aaaa*t2_aaaa(d,a,i,j)*t2_aaaa(e,b,k,l)
    triples_res +=  1.000000000000000 * einsum('lcde,daij,ebkl->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <c,l||d,e>_abab*t2_aaaa(d,a,i,j)*t2_abab(b,e,k,l)
    triples_res +=  1.000000000000000 * einsum('clde,daij,bekl->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_aaaa(e,a,j,k)*t2_aaaa(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,eajk,bcim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_aaaa(e,a,j,k)*t2_aaaa(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('mled,dl,eajk,bcim->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_aaaa(e,a,i,k)*t2_aaaa(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,eaik,bcjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_aaaa(e,a,i,k)*t2_aaaa(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mled,dl,eaik,bcjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_aaaa(e,a,i,j)*t2_aaaa(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,eaij,bckm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_aaaa(e,a,i,j)*t2_aaaa(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mled,dl,eaij,bckm->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_aaaa(e,b,j,k)*t2_aaaa(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ebjk,acim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_aaaa(e,b,j,k)*t2_aaaa(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mled,dl,ebjk,acim->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_aaaa(e,b,i,k)*t2_aaaa(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ebik,acjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_aaaa(e,b,i,k)*t2_aaaa(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('mled,dl,ebik,acjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_aaaa(e,b,i,j)*t2_aaaa(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ebij,ackm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_aaaa(e,b,i,j)*t2_aaaa(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mled,dl,ebij,ackm->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_aaaa(e,c,j,k)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ecjk,abim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_aaaa(e,c,j,k)*t2_aaaa(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mled,dl,ecjk,abim->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_aaaa(e,c,i,k)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ecik,abjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_aaaa(e,c,i,k)*t2_aaaa(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mled,dl,ecik,abjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_aaaa(e,c,i,j)*t2_aaaa(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ecij,abkm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_aaaa(e,c,i,j)*t2_aaaa(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('mled,dl,ecij,abkm->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t2_aaaa(e,a,j,l)*t2_aaaa(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,eajl,bcim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,k)*t2_abab(a,e,j,l)*t2_aaaa(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,aejl,bcim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t2_aaaa(e,a,i,l)*t2_aaaa(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,eail,bcjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,k)*t2_abab(a,e,i,l)*t2_aaaa(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,aeil,bcjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(d,k)*t2_aaaa(e,a,i,j)*t2_aaaa(b,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dk,eaij,bcml->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t2_aaaa(e,b,j,l)*t2_aaaa(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,ebjl,acim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,k)*t2_abab(b,e,j,l)*t2_aaaa(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,bejl,acim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t2_aaaa(e,b,i,l)*t2_aaaa(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,ebil,acjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,k)*t2_abab(b,e,i,l)*t2_aaaa(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,beil,acjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(d,k)*t2_aaaa(e,b,i,j)*t2_aaaa(a,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dk,ebij,acml->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t2_aaaa(e,c,j,l)*t2_aaaa(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,ecjl,abim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,k)*t2_abab(c,e,j,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,cejl,abim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t2_aaaa(e,c,i,l)*t2_aaaa(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,ecil,abjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,k)*t2_abab(c,e,i,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,ceil,abjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(d,k)*t2_aaaa(e,c,i,j)*t2_aaaa(a,b,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dk,ecij,abml->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,a,k,l)*t2_aaaa(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,eakl,bcim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,j)*t2_abab(a,e,k,l)*t2_aaaa(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,aekl,bcim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,a,i,l)*t2_aaaa(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,eail,bckm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,j)*t2_abab(a,e,i,l)*t2_aaaa(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,aeil,bckm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,a,i,k)*t2_aaaa(b,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dj,eaik,bcml->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,b,k,l)*t2_aaaa(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,ebkl,acim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,j)*t2_abab(b,e,k,l)*t2_aaaa(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,bekl,acim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,b,i,l)*t2_aaaa(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,ebil,ackm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,j)*t2_abab(b,e,i,l)*t2_aaaa(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,beil,ackm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,b,i,k)*t2_aaaa(a,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dj,ebik,acml->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,c,k,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,eckl,abim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,j)*t2_abab(c,e,k,l)*t2_aaaa(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,cekl,abim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,c,i,l)*t2_aaaa(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,ecil,abkm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,j)*t2_abab(c,e,i,l)*t2_aaaa(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,ceil,abkm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,c,i,k)*t2_aaaa(a,b,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dj,ecik,abml->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,a,k,l)*t2_aaaa(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,di,eakl,bcjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t2_abab(a,e,k,l)*t2_aaaa(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,aekl,bcjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,a,j,l)*t2_aaaa(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,eajl,bckm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t2_abab(a,e,j,l)*t2_aaaa(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,di,aejl,bckm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,a,j,k)*t2_aaaa(b,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,di,eajk,bcml->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,b,k,l)*t2_aaaa(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,ebkl,acjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t2_abab(b,e,k,l)*t2_aaaa(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,di,bekl,acjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,b,j,l)*t2_aaaa(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,di,ebjl,ackm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t2_abab(b,e,j,l)*t2_aaaa(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,bejl,ackm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,b,j,k)*t2_aaaa(a,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,di,ebjk,acml->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,c,k,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,di,eckl,abjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t2_abab(c,e,k,l)*t2_aaaa(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,cekl,abjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,c,j,l)*t2_aaaa(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,ecjl,abkm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t2_abab(c,e,j,l)*t2_aaaa(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,di,cejl,abkm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,c,j,k)*t2_aaaa(a,b,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,di,ecjk,abml->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,e,j,k)*t2_aaaa(b,c,i,m)
    triples_res +=  0.500000000000000 * einsum('mlde,al,dejk,bcim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,e,i,k)*t2_aaaa(b,c,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,al,deik,bcjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,e,i,j)*t2_aaaa(b,c,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,al,deij,bckm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,k,m)*t2_aaaa(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('mlde,al,dbkm,ecij->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,d,k,m)*t2_aaaa(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('lmed,al,bdkm,ecij->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,j,m)*t2_aaaa(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('mlde,al,dbjm,ecik->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,d,j,m)*t2_aaaa(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('lmed,al,bdjm,ecik->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,m)*t2_aaaa(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,al,dbim,ecjk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,d,i,m)*t2_aaaa(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lmed,al,bdim,ecjk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,j,k)*t2_aaaa(e,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,al,dbjk,ecim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(a,l)*t2_aaaa(d,b,j,k)*t2_abab(c,e,i,m)
    triples_res += -1.000000000000000 * einsum('lmde,al,dbjk,ceim->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,k)*t2_aaaa(e,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,al,dbik,ecjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(a,l)*t2_aaaa(d,b,i,k)*t2_abab(c,e,j,m)
    triples_res +=  1.000000000000000 * einsum('lmde,al,dbik,cejm->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,j)*t2_aaaa(e,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,al,dbij,eckm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(a,l)*t2_aaaa(d,b,i,j)*t2_abab(c,e,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,al,dbij,cekm->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(d,e,j,k)*t2_aaaa(a,c,i,m)
    triples_res += -0.500000000000000 * einsum('mlde,bl,dejk,acim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(d,e,i,k)*t2_aaaa(a,c,j,m)
    triples_res +=  0.500000000000000 * einsum('mlde,bl,deik,acjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(d,e,i,j)*t2_aaaa(a,c,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,bl,deij,ackm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(d,a,k,m)*t2_aaaa(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('mlde,bl,dakm,ecij->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_aa(b,l)*t2_abab(a,d,k,m)*t2_aaaa(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('lmed,bl,adkm,ecij->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(d,a,j,m)*t2_aaaa(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('mlde,bl,dajm,ecik->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_aa(b,l)*t2_abab(a,d,j,m)*t2_aaaa(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('lmed,bl,adjm,ecik->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(d,a,i,m)*t2_aaaa(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,bl,daim,ecjk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_aa(b,l)*t2_abab(a,d,i,m)*t2_aaaa(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('lmed,bl,adim,ecjk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(d,a,j,k)*t2_aaaa(e,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,bl,dajk,ecim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(b,l)*t2_aaaa(d,a,j,k)*t2_abab(c,e,i,m)
    triples_res +=  1.000000000000000 * einsum('lmde,bl,dajk,ceim->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(d,a,i,k)*t2_aaaa(e,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,bl,daik,ecjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(b,l)*t2_aaaa(d,a,i,k)*t2_abab(c,e,j,m)
    triples_res += -1.000000000000000 * einsum('lmde,bl,daik,cejm->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(d,a,i,j)*t2_aaaa(e,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,bl,daij,eckm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(b,l)*t2_aaaa(d,a,i,j)*t2_abab(c,e,k,m)
    triples_res +=  1.000000000000000 * einsum('lmde,bl,daij,cekm->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(c,l)*t2_aaaa(d,e,j,k)*t2_aaaa(a,b,i,m)
    triples_res +=  0.500000000000000 * einsum('mlde,cl,dejk,abim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(c,l)*t2_aaaa(d,e,i,k)*t2_aaaa(a,b,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,cl,deik,abjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(c,l)*t2_aaaa(d,e,i,j)*t2_aaaa(a,b,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,cl,deij,abkm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(c,l)*t2_aaaa(d,a,k,m)*t2_aaaa(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('mlde,cl,dakm,ebij->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_aa(c,l)*t2_abab(a,d,k,m)*t2_aaaa(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('lmed,cl,adkm,ebij->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(c,l)*t2_aaaa(d,a,j,m)*t2_aaaa(e,b,i,k)
    triples_res +=  1.000000000000000 * einsum('mlde,cl,dajm,ebik->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_aa(c,l)*t2_abab(a,d,j,m)*t2_aaaa(e,b,i,k)
    triples_res += -1.000000000000000 * einsum('lmed,cl,adjm,ebik->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(c,l)*t2_aaaa(d,a,i,m)*t2_aaaa(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,cl,daim,ebjk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_aa(c,l)*t2_abab(a,d,i,m)*t2_aaaa(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('lmed,cl,adim,ebjk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(c,l)*t2_aaaa(d,a,j,k)*t2_aaaa(e,b,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,cl,dajk,ebim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(c,l)*t2_aaaa(d,a,j,k)*t2_abab(b,e,i,m)
    triples_res += -1.000000000000000 * einsum('lmde,cl,dajk,beim->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(c,l)*t2_aaaa(d,a,i,k)*t2_aaaa(e,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,cl,daik,ebjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(c,l)*t2_aaaa(d,a,i,k)*t2_abab(b,e,j,m)
    triples_res +=  1.000000000000000 * einsum('lmde,cl,daik,bejm->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(c,l)*t2_aaaa(d,a,i,j)*t2_aaaa(e,b,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,cl,daij,ebkm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(c,l)*t2_aaaa(d,a,i,j)*t2_abab(b,e,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,cl,daij,bekm->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_aaaa*t1_aa(d,j)*t1_aa(a,l)*t2_aaaa(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dj,al,bcim->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_aaaa*t1_aa(d,j)*t1_aa(b,l)*t2_aaaa(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mldk,dj,bl,acim->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_aaaa*t1_aa(d,j)*t1_aa(c,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dj,cl,abim->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_aaaa*t1_aa(d,i)*t1_aa(a,l)*t2_aaaa(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,di,al,bcjm->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_aaaa*t1_aa(d,i)*t1_aa(b,l)*t2_aaaa(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,di,bl,acjm->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_aaaa*t1_aa(d,i)*t1_aa(c,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,di,cl,abjm->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t2_aaaa(d,c,i,j)
    triples_res +=  1.000000000000000 * einsum('mldk,al,bm,dcij->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_aaaa*t1_aa(a,l)*t1_aa(c,m)*t2_aaaa(d,b,i,j)
    triples_res += -1.000000000000000 * einsum('mldk,al,cm,dbij->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_aaaa*t1_aa(b,l)*t1_aa(c,m)*t2_aaaa(d,a,i,j)
    triples_res +=  1.000000000000000 * einsum('mldk,bl,cm,daij->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t1_aa(d,k)*t1_aa(a,l)*t2_aaaa(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('mldj,dk,al,bcim->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t1_aa(d,k)*t1_aa(b,l)*t2_aaaa(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mldj,dk,bl,acim->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t1_aa(d,k)*t1_aa(c,l)*t2_aaaa(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mldj,dk,cl,abim->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t1_aa(d,i)*t1_aa(a,l)*t2_aaaa(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mldj,di,al,bckm->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t1_aa(d,i)*t1_aa(b,l)*t2_aaaa(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('mldj,di,bl,ackm->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t1_aa(d,i)*t1_aa(c,l)*t2_aaaa(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mldj,di,cl,abkm->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t2_aaaa(d,c,i,k)
    triples_res += -1.000000000000000 * einsum('mldj,al,bm,dcik->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t1_aa(a,l)*t1_aa(c,m)*t2_aaaa(d,b,i,k)
    triples_res +=  1.000000000000000 * einsum('mldj,al,cm,dbik->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t1_aa(b,l)*t1_aa(c,m)*t2_aaaa(d,a,i,k)
    triples_res += -1.000000000000000 * einsum('mldj,bl,cm,daik->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t1_aa(d,k)*t1_aa(a,l)*t2_aaaa(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dk,al,bcjm->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t1_aa(d,k)*t1_aa(b,l)*t2_aaaa(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('mldi,dk,bl,acjm->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t1_aa(d,k)*t1_aa(c,l)*t2_aaaa(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dk,cl,abjm->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t1_aa(d,j)*t1_aa(a,l)*t2_aaaa(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mldi,dj,al,bckm->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t1_aa(d,j)*t1_aa(b,l)*t2_aaaa(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dj,bl,ackm->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t1_aa(d,j)*t1_aa(c,l)*t2_aaaa(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('mldi,dj,cl,abkm->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t2_aaaa(d,c,j,k)
    triples_res +=  1.000000000000000 * einsum('mldi,al,bm,dcjk->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t1_aa(a,l)*t1_aa(c,m)*t2_aaaa(d,b,j,k)
    triples_res += -1.000000000000000 * einsum('mldi,al,cm,dbjk->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t1_aa(b,l)*t1_aa(c,m)*t2_aaaa(d,a,j,k)
    triples_res +=  1.000000000000000 * einsum('mldi,bl,cm,dajk->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,j)*t2_aaaa(b,c,i,l)
    triples_res +=  1.000000000000000 * einsum('lade,dk,ej,bcil->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,i)*t2_aaaa(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('lade,dk,ei,bcjl->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t1_aa(d,k)*t1_aa(b,l)*t2_aaaa(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('lade,dk,bl,ecij->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t1_aa(d,k)*t1_aa(c,l)*t2_aaaa(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('lade,dk,cl,ebij->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t1_aa(d,j)*t1_aa(e,i)*t2_aaaa(b,c,k,l)
    triples_res +=  1.000000000000000 * einsum('lade,dj,ei,bckl->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t1_aa(d,j)*t1_aa(b,l)*t2_aaaa(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('lade,dj,bl,ecik->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t1_aa(d,j)*t1_aa(c,l)*t2_aaaa(e,b,i,k)
    triples_res +=  1.000000000000000 * einsum('lade,dj,cl,ebik->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t1_aa(d,i)*t1_aa(b,l)*t2_aaaa(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lade,di,bl,ecjk->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t1_aa(d,i)*t1_aa(c,l)*t2_aaaa(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('lade,di,cl,ebjk->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,j)*t2_aaaa(a,c,i,l)
    triples_res += -1.000000000000000 * einsum('lbde,dk,ej,acil->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,i)*t2_aaaa(a,c,j,l)
    triples_res +=  1.000000000000000 * einsum('lbde,dk,ei,acjl->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t1_aa(d,k)*t1_aa(a,l)*t2_aaaa(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('lbde,dk,al,ecij->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_aaaa*t1_aa(d,k)*t1_aa(c,l)*t2_aaaa(e,a,i,j)
    triples_res +=  1.000000000000000 * einsum('lbde,dk,cl,eaij->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t1_aa(d,j)*t1_aa(e,i)*t2_aaaa(a,c,k,l)
    triples_res += -1.000000000000000 * einsum('lbde,dj,ei,ackl->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_aaaa*t1_aa(d,j)*t1_aa(a,l)*t2_aaaa(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('lbde,dj,al,ecik->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t1_aa(d,j)*t1_aa(c,l)*t2_aaaa(e,a,i,k)
    triples_res += -1.000000000000000 * einsum('lbde,dj,cl,eaik->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t1_aa(d,i)*t1_aa(a,l)*t2_aaaa(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('lbde,di,al,ecjk->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_aaaa*t1_aa(d,i)*t1_aa(c,l)*t2_aaaa(e,a,j,k)
    triples_res +=  1.000000000000000 * einsum('lbde,di,cl,eajk->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,j)*t2_aaaa(a,b,i,l)
    triples_res +=  1.000000000000000 * einsum('lcde,dk,ej,abil->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,i)*t2_aaaa(a,b,j,l)
    triples_res += -1.000000000000000 * einsum('lcde,dk,ei,abjl->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_aaaa*t1_aa(d,k)*t1_aa(a,l)*t2_aaaa(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('lcde,dk,al,ebij->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_aaaa*t1_aa(d,k)*t1_aa(b,l)*t2_aaaa(e,a,i,j)
    triples_res += -1.000000000000000 * einsum('lcde,dk,bl,eaij->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_aaaa*t1_aa(d,j)*t1_aa(e,i)*t2_aaaa(a,b,k,l)
    triples_res +=  1.000000000000000 * einsum('lcde,dj,ei,abkl->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_aaaa*t1_aa(d,j)*t1_aa(a,l)*t2_aaaa(e,b,i,k)
    triples_res += -1.000000000000000 * einsum('lcde,dj,al,ebik->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_aaaa*t1_aa(d,j)*t1_aa(b,l)*t2_aaaa(e,a,i,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dj,bl,eaik->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_aaaa*t1_aa(d,i)*t1_aa(a,l)*t2_aaaa(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,di,al,ebjk->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_aaaa*t1_aa(d,i)*t1_aa(b,l)*t2_aaaa(e,a,j,k)
    triples_res += -1.000000000000000 * einsum('lcde,di,bl,eajk->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t1_aa(e,k)*t3_aaaaaa(a,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ek,abcijm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t1_aa(e,k)*t3_aaaaaa(a,b,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('mled,dl,ek,abcijm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t1_aa(e,j)*t3_aaaaaa(a,b,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ej,abcikm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t1_aa(e,j)*t3_aaaaaa(a,b,c,i,k,m)
    triples_res +=  1.000000000000000 * einsum('mled,dl,ej,abcikm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t1_aa(e,i)*t3_aaaaaa(a,b,c,j,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ei,abcjkm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t1_aa(e,i)*t3_aaaaaa(a,b,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('mled,dl,ei,abcjkm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t1_aa(a,m)*t3_aaaaaa(e,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,am,ebcijk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t1_aa(a,m)*t3_aaaaaa(e,b,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('mled,dl,am,ebcijk->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t1_aa(b,m)*t3_aaaaaa(e,a,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,dl,bm,eacijk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t1_aa(b,m)*t3_aaaaaa(e,a,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('mled,dl,bm,eacijk->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t1_aa(c,m)*t3_aaaaaa(e,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,cm,eabijk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t1_aa(c,m)*t3_aaaaaa(e,a,b,i,j,k)
    triples_res += -1.000000000000000 * einsum('mled,dl,cm,eabijk->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,j)*t3_aaaaaa(a,b,c,i,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dk,ej,abciml->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,i)*t3_aaaaaa(a,b,c,j,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dk,ei,abcjml->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(a,l)*t3_aaaaaa(e,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,al,ebcijm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,k)*t1_aa(a,l)*t3_aabaab(c,b,e,i,j,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dk,al,cbeijm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(b,l)*t3_aaaaaa(e,a,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,bl,eacijm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,k)*t1_aa(b,l)*t3_aabaab(c,a,e,i,j,m)
    triples_res += -1.000000000000000 * einsum('lmde,dk,bl,caeijm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(c,l)*t3_aaaaaa(e,a,b,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,cl,eabijm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,k)*t1_aa(c,l)*t3_aabaab(b,a,e,i,j,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dk,cl,baeijm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(e,i)*t3_aaaaaa(a,b,c,k,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dj,ei,abckml->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(a,l)*t3_aaaaaa(e,b,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,al,ebcikm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,j)*t1_aa(a,l)*t3_aabaab(c,b,e,i,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,dj,al,cbeikm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(b,l)*t3_aaaaaa(e,a,c,i,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,bl,eacikm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,j)*t1_aa(b,l)*t3_aabaab(c,a,e,i,k,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dj,bl,caeikm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(c,l)*t3_aaaaaa(e,a,b,i,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,cl,eabikm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,j)*t1_aa(c,l)*t3_aabaab(b,a,e,i,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,dj,cl,baeikm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t1_aa(a,l)*t3_aaaaaa(e,b,c,j,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,al,ebcjkm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t1_aa(a,l)*t3_aabaab(c,b,e,j,k,m)
    triples_res +=  1.000000000000000 * einsum('lmde,di,al,cbejkm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t1_aa(b,l)*t3_aaaaaa(e,a,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,di,bl,eacjkm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t1_aa(b,l)*t3_aabaab(c,a,e,j,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,di,bl,caejkm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t1_aa(c,l)*t3_aaaaaa(e,a,b,j,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,cl,eabjkm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t1_aa(c,l)*t3_aabaab(b,a,e,j,k,m)
    triples_res +=  1.000000000000000 * einsum('lmde,di,cl,baejkm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t3_aaaaaa(d,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,al,bm,decijk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(a,l)*t1_aa(c,m)*t3_aaaaaa(d,e,b,i,j,k)
    triples_res +=  0.500000000000000 * einsum('mlde,al,cm,debijk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(b,l)*t1_aa(c,m)*t3_aaaaaa(d,e,a,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,bl,cm,deaijk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,j)*t1_aa(a,l)*t2_aaaa(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,ej,al,bcim->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,j)*t1_aa(b,l)*t2_aaaa(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,ej,bl,acim->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,j)*t1_aa(c,l)*t2_aaaa(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,ej,cl,abim->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,i)*t1_aa(a,l)*t2_aaaa(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,ei,al,bcjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,i)*t1_aa(b,l)*t2_aaaa(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,ei,bl,acjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(e,i)*t1_aa(c,l)*t2_aaaa(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,ei,cl,abjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(a,l)*t1_aa(b,m)*t2_aaaa(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('mlde,dk,al,bm,ecij->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(a,l)*t1_aa(c,m)*t2_aaaa(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,al,cm,ebij->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,k)*t1_aa(b,l)*t1_aa(c,m)*t2_aaaa(e,a,i,j)
    triples_res += -1.000000000000000 * einsum('mlde,dk,bl,cm,eaij->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(e,i)*t1_aa(a,l)*t2_aaaa(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,ei,al,bckm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(e,i)*t1_aa(b,l)*t2_aaaa(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,ei,bl,ackm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(e,i)*t1_aa(c,l)*t2_aaaa(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,ei,cl,abkm->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(a,l)*t1_aa(b,m)*t2_aaaa(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,al,bm,ecik->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(a,l)*t1_aa(c,m)*t2_aaaa(e,b,i,k)
    triples_res += -1.000000000000000 * einsum('mlde,dj,al,cm,ebik->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(b,l)*t1_aa(c,m)*t2_aaaa(e,a,i,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,bl,cm,eaik->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t1_aa(a,l)*t1_aa(b,m)*t2_aaaa(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,di,al,bm,ecjk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t1_aa(a,l)*t1_aa(c,m)*t2_aaaa(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,di,al,cm,ebjk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t1_aa(b,l)*t1_aa(c,m)*t2_aaaa(e,a,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,di,bl,cm,eajk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    return triples_res
    
    
def ccsdt_t3_aabaab_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):
    
    #    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 f_bb(l,k)*t3_aabaab(a,b,c,i,j,l)
    triples_res = -1.000000000000000 * einsum('lk,abcijl->abcijk', f_bb[o, o], t3_aabaab)
    
    #	 -1.0000 f_aa(l,j)*t3_aabaab(a,b,c,i,l,k)
    triples_res += -1.000000000000000 * einsum('lj,abcilk->abcijk', f_aa[o, o], t3_aabaab)
    
    #	  1.0000 f_aa(l,i)*t3_aabaab(a,b,c,j,l,k)
    triples_res +=  1.000000000000000 * einsum('li,abcjlk->abcijk', f_aa[o, o], t3_aabaab)
    
    #	  1.0000 f_aa(a,d)*t3_aabaab(d,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('ad,dbcijk->abcijk', f_aa[v, v], t3_aabaab)
    
    #	 -1.0000 f_aa(b,d)*t3_aabaab(d,a,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('bd,dacijk->abcijk', f_aa[v, v], t3_aabaab)
    
    #	 -1.0000 f_bb(c,d)*t3_aabaab(b,a,d,i,j,k)
    triples_res += -1.000000000000000 * einsum('cd,badijk->abcijk', f_bb[v, v], t3_aabaab)
    
    #	 -1.0000 f_bb(l,d)*t1_bb(d,k)*t3_aabaab(a,b,c,i,j,l)
    triples_res += -1.000000000000000 * einsum('ld,dk,abcijl->abcijk', f_bb[o, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t1_aa(d,j)*t3_aabaab(a,b,c,i,l,k)
    triples_res += -1.000000000000000 * einsum('ld,dj,abcilk->abcijk', f_aa[o, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(l,d)*t1_aa(d,i)*t3_aabaab(a,b,c,j,l,k)
    triples_res +=  1.000000000000000 * einsum('ld,di,abcjlk->abcijk', f_aa[o, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t1_aa(a,l)*t3_aabaab(d,b,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('ld,al,dbcijk->abcijk', f_aa[o, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(l,d)*t1_aa(b,l)*t3_aabaab(d,a,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('ld,bl,dacijk->abcijk', f_aa[o, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(l,d)*t1_bb(c,l)*t3_aabaab(b,a,d,i,j,k)
    triples_res +=  1.000000000000000 * einsum('ld,cl,badijk->abcijk', f_bb[o, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(l,d)*t2_abab(a,d,j,k)*t2_abab(b,c,i,l)
    triples_res +=  1.000000000000000 * einsum('ld,adjk,bcil->abcijk', f_bb[o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(l,d)*t2_abab(a,d,i,k)*t2_abab(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('ld,adik,bcjl->abcijk', f_bb[o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(l,d)*t2_aaaa(d,a,i,j)*t2_abab(b,c,l,k)
    triples_res +=  1.000000000000000 * einsum('ld,daij,bclk->abcijk', f_aa[o, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(l,d)*t2_abab(b,d,j,k)*t2_abab(a,c,i,l)
    triples_res += -1.000000000000000 * einsum('ld,bdjk,acil->abcijk', f_bb[o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(l,d)*t2_abab(b,d,i,k)*t2_abab(a,c,j,l)
    triples_res +=  1.000000000000000 * einsum('ld,bdik,acjl->abcijk', f_bb[o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t2_aaaa(d,b,i,j)*t2_abab(a,c,l,k)
    triples_res += -1.000000000000000 * einsum('ld,dbij,aclk->abcijk', f_aa[o, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t2_abab(d,c,j,k)*t2_aaaa(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('ld,dcjk,abil->abcijk', f_aa[o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(l,d)*t2_abab(d,c,i,k)*t2_aaaa(a,b,j,l)
    triples_res +=  1.000000000000000 * einsum('ld,dcik,abjl->abcijk', f_aa[o, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||j,k>_abab*t2_abab(b,c,i,l)
    triples_res +=  1.000000000000000 * einsum('aljk,bcil->abcijk', g_abab[v, o, o, o], t2_abab)
    
    #	 -1.0000 <a,l||i,k>_abab*t2_abab(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('alik,bcjl->abcijk', g_abab[v, o, o, o], t2_abab)
    
    #	  1.0000 <l,a||i,j>_aaaa*t2_abab(b,c,l,k)
    triples_res +=  1.000000000000000 * einsum('laij,bclk->abcijk', g_aaaa[o, v, o, o], t2_abab)
    
    #	 -1.0000 <b,l||j,k>_abab*t2_abab(a,c,i,l)
    triples_res += -1.000000000000000 * einsum('bljk,acil->abcijk', g_abab[v, o, o, o], t2_abab)
    
    #	  1.0000 <b,l||i,k>_abab*t2_abab(a,c,j,l)
    triples_res +=  1.000000000000000 * einsum('blik,acjl->abcijk', g_abab[v, o, o, o], t2_abab)
    
    #	 -1.0000 <l,b||i,j>_aaaa*t2_abab(a,c,l,k)
    triples_res += -1.000000000000000 * einsum('lbij,aclk->abcijk', g_aaaa[o, v, o, o], t2_abab)
    
    #	 -1.0000 <l,c||j,k>_abab*t2_aaaa(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('lcjk,abil->abcijk', g_abab[o, v, o, o], t2_aaaa)
    
    #	  1.0000 <l,c||i,k>_abab*t2_aaaa(a,b,j,l)
    triples_res +=  1.000000000000000 * einsum('lcik,abjl->abcijk', g_abab[o, v, o, o], t2_aaaa)
    
    #	  1.0000 <a,b||d,j>_aaaa*t2_abab(d,c,i,k)
    triples_res +=  1.000000000000000 * einsum('abdj,dcik->abcijk', g_aaaa[v, v, v, o], t2_abab)
    
    #	 -1.0000 <a,b||d,i>_aaaa*t2_abab(d,c,j,k)
    triples_res += -1.000000000000000 * einsum('abdi,dcjk->abcijk', g_aaaa[v, v, v, o], t2_abab)
    
    #	  1.0000 <a,c||d,k>_abab*t2_aaaa(d,b,i,j)
    triples_res +=  1.000000000000000 * einsum('acdk,dbij->abcijk', g_abab[v, v, v, o], t2_aaaa)
    
    #	 -1.0000 <a,c||j,d>_abab*t2_abab(b,d,i,k)
    triples_res += -1.000000000000000 * einsum('acjd,bdik->abcijk', g_abab[v, v, o, v], t2_abab)
    
    #	  1.0000 <a,c||i,d>_abab*t2_abab(b,d,j,k)
    triples_res +=  1.000000000000000 * einsum('acid,bdjk->abcijk', g_abab[v, v, o, v], t2_abab)
    
    #	 -1.0000 <b,c||d,k>_abab*t2_aaaa(d,a,i,j)
    triples_res += -1.000000000000000 * einsum('bcdk,daij->abcijk', g_abab[v, v, v, o], t2_aaaa)
    
    #	  1.0000 <b,c||j,d>_abab*t2_abab(a,d,i,k)
    triples_res +=  1.000000000000000 * einsum('bcjd,adik->abcijk', g_abab[v, v, o, v], t2_abab)
    
    #	 -1.0000 <b,c||i,d>_abab*t2_abab(a,d,j,k)
    triples_res += -1.000000000000000 * einsum('bcid,adjk->abcijk', g_abab[v, v, o, v], t2_abab)
    
    #	  0.5000 <m,l||j,k>_abab*t3_aabaab(a,b,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mljk,abciml->abcijk', g_abab[o, o, o, o], t3_aabaab)
    
    #	  0.5000 <l,m||j,k>_abab*t3_aabaab(a,b,c,i,l,m)
    triples_res +=  0.500000000000000 * einsum('lmjk,abcilm->abcijk', g_abab[o, o, o, o], t3_aabaab)
    
    #	 -0.5000 <m,l||i,k>_abab*t3_aabaab(a,b,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mlik,abcjml->abcijk', g_abab[o, o, o, o], t3_aabaab)
    
    #	 -0.5000 <l,m||i,k>_abab*t3_aabaab(a,b,c,j,l,m)
    triples_res += -0.500000000000000 * einsum('lmik,abcjlm->abcijk', g_abab[o, o, o, o], t3_aabaab)
    
    #	 -0.5000 <m,l||i,j>_aaaa*t3_aabaab(a,b,c,l,m,k)
    triples_res += -0.500000000000000 * einsum('mlij,abclmk->abcijk', g_aaaa[o, o, o, o], t3_aabaab)
    
    #	 -1.0000 <a,l||d,k>_abab*t3_aabaab(d,b,c,i,j,l)
    triples_res += -1.000000000000000 * einsum('aldk,dbcijl->abcijk', g_abab[v, o, v, o], t3_aabaab)
    
    #	  1.0000 <l,a||d,j>_aaaa*t3_aabaab(d,b,c,i,l,k)
    triples_res +=  1.000000000000000 * einsum('ladj,dbcilk->abcijk', g_aaaa[o, v, v, o], t3_aabaab)
    
    #	  1.0000 <a,l||j,d>_abab*t3_abbabb(b,d,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('aljd,bdcikl->abcijk', g_abab[v, o, o, v], t3_abbabb)
    
    #	 -1.0000 <l,a||d,i>_aaaa*t3_aabaab(d,b,c,j,l,k)
    triples_res += -1.000000000000000 * einsum('ladi,dbcjlk->abcijk', g_aaaa[o, v, v, o], t3_aabaab)
    
    #	 -1.0000 <a,l||i,d>_abab*t3_abbabb(b,d,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('alid,bdcjkl->abcijk', g_abab[v, o, o, v], t3_abbabb)
    
    #	  1.0000 <b,l||d,k>_abab*t3_aabaab(d,a,c,i,j,l)
    triples_res +=  1.000000000000000 * einsum('bldk,dacijl->abcijk', g_abab[v, o, v, o], t3_aabaab)
    
    #	 -1.0000 <l,b||d,j>_aaaa*t3_aabaab(d,a,c,i,l,k)
    triples_res += -1.000000000000000 * einsum('lbdj,dacilk->abcijk', g_aaaa[o, v, v, o], t3_aabaab)
    
    #	 -1.0000 <b,l||j,d>_abab*t3_abbabb(a,d,c,i,k,l)
    triples_res += -1.000000000000000 * einsum('bljd,adcikl->abcijk', g_abab[v, o, o, v], t3_abbabb)
    
    #	  1.0000 <l,b||d,i>_aaaa*t3_aabaab(d,a,c,j,l,k)
    triples_res +=  1.000000000000000 * einsum('lbdi,dacjlk->abcijk', g_aaaa[o, v, v, o], t3_aabaab)
    
    #	  1.0000 <b,l||i,d>_abab*t3_abbabb(a,d,c,j,k,l)
    triples_res +=  1.000000000000000 * einsum('blid,adcjkl->abcijk', g_abab[v, o, o, v], t3_abbabb)
    
    #	  1.0000 <l,c||d,k>_abab*t3_aaaaaa(d,a,b,i,j,l)
    triples_res +=  1.000000000000000 * einsum('lcdk,dabijl->abcijk', g_abab[o, v, v, o], t3_aaaaaa)
    
    #	 -1.0000 <l,c||d,k>_bbbb*t3_aabaab(b,a,d,i,j,l)
    triples_res += -1.000000000000000 * einsum('lcdk,badijl->abcijk', g_bbbb[o, v, v, o], t3_aabaab)
    
    #	  1.0000 <l,c||j,d>_abab*t3_aabaab(b,a,d,i,l,k)
    triples_res +=  1.000000000000000 * einsum('lcjd,badilk->abcijk', g_abab[o, v, o, v], t3_aabaab)
    
    #	 -1.0000 <l,c||i,d>_abab*t3_aabaab(b,a,d,j,l,k)
    triples_res += -1.000000000000000 * einsum('lcid,badjlk->abcijk', g_abab[o, v, o, v], t3_aabaab)
    
    #	  0.5000 <a,b||d,e>_aaaa*t3_aabaab(d,e,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('abde,decijk->abcijk', g_aaaa[v, v, v, v], t3_aabaab)
    
    #	  0.5000 <a,c||d,e>_abab*t3_aabaab(d,b,e,i,j,k)
    triples_res +=  0.500000000000000 * einsum('acde,dbeijk->abcijk', g_abab[v, v, v, v], t3_aabaab)
    
    #	 -0.5000 <a,c||e,d>_abab*t3_aabaab(b,e,d,i,j,k)
    triples_res += -0.500000000000000 * einsum('aced,bedijk->abcijk', g_abab[v, v, v, v], t3_aabaab)
    
    #	 -0.5000 <b,c||d,e>_abab*t3_aabaab(d,a,e,i,j,k)
    triples_res += -0.500000000000000 * einsum('bcde,daeijk->abcijk', g_abab[v, v, v, v], t3_aabaab)
    
    #	  0.5000 <b,c||e,d>_abab*t3_aabaab(a,e,d,i,j,k)
    triples_res +=  0.500000000000000 * einsum('bced,aedijk->abcijk', g_abab[v, v, v, v], t3_aabaab)
    
    #	 -1.0000 <l,m||j,k>_abab*t1_aa(a,l)*t2_abab(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmjk,al,bcim->abcijk', g_abab[o, o, o, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||j,k>_abab*t1_aa(b,l)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmjk,bl,acim->abcijk', g_abab[o, o, o, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||j,k>_abab*t1_bb(c,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mljk,cl,abim->abcijk', g_abab[o, o, o, o], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||i,k>_abab*t1_aa(a,l)*t2_abab(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmik,al,bcjm->abcijk', g_abab[o, o, o, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||i,k>_abab*t1_aa(b,l)*t2_abab(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmik,bl,acjm->abcijk', g_abab[o, o, o, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,k>_abab*t1_bb(c,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlik,cl,abjm->abcijk', g_abab[o, o, o, o], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,j>_aaaa*t1_aa(a,l)*t2_abab(b,c,m,k)
    triples_res += -1.000000000000000 * einsum('mlij,al,bcmk->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,j>_aaaa*t1_aa(b,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlij,bl,acmk->abcijk', g_aaaa[o, o, o, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||d,k>_abab*t1_aa(d,j)*t2_abab(b,c,i,l)
    triples_res +=  1.000000000000000 * einsum('aldk,dj,bcil->abcijk', g_abab[v, o, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||d,k>_abab*t1_aa(d,i)*t2_abab(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('aldk,di,bcjl->abcijk', g_abab[v, o, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||d,k>_abab*t1_bb(c,l)*t2_aaaa(d,b,i,j)
    triples_res += -1.000000000000000 * einsum('aldk,cl,dbij->abcijk', g_abab[v, o, v, o], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||j,d>_abab*t1_bb(d,k)*t2_abab(b,c,i,l)
    triples_res +=  1.000000000000000 * einsum('aljd,dk,bcil->abcijk', g_abab[v, o, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,j>_aaaa*t1_aa(d,i)*t2_abab(b,c,l,k)
    triples_res +=  1.000000000000000 * einsum('ladj,di,bclk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,j>_aaaa*t1_aa(b,l)*t2_abab(d,c,i,k)
    triples_res +=  1.000000000000000 * einsum('ladj,bl,dcik->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||j,d>_abab*t1_bb(c,l)*t2_abab(b,d,i,k)
    triples_res +=  1.000000000000000 * einsum('aljd,cl,bdik->abcijk', g_abab[v, o, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||i,d>_abab*t1_bb(d,k)*t2_abab(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('alid,dk,bcjl->abcijk', g_abab[v, o, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,i>_aaaa*t1_aa(d,j)*t2_abab(b,c,l,k)
    triples_res += -1.000000000000000 * einsum('ladi,dj,bclk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,i>_aaaa*t1_aa(b,l)*t2_abab(d,c,j,k)
    triples_res += -1.000000000000000 * einsum('ladi,bl,dcjk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||i,d>_abab*t1_bb(c,l)*t2_abab(b,d,j,k)
    triples_res += -1.000000000000000 * einsum('alid,cl,bdjk->abcijk', g_abab[v, o, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,l||d,k>_abab*t1_aa(d,j)*t2_abab(a,c,i,l)
    triples_res += -1.000000000000000 * einsum('bldk,dj,acil->abcijk', g_abab[v, o, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,l||d,k>_abab*t1_aa(d,i)*t2_abab(a,c,j,l)
    triples_res +=  1.000000000000000 * einsum('bldk,di,acjl->abcijk', g_abab[v, o, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,l||d,k>_abab*t1_bb(c,l)*t2_aaaa(d,a,i,j)
    triples_res +=  1.000000000000000 * einsum('bldk,cl,daij->abcijk', g_abab[v, o, v, o], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,l||j,d>_abab*t1_bb(d,k)*t2_abab(a,c,i,l)
    triples_res += -1.000000000000000 * einsum('bljd,dk,acil->abcijk', g_abab[v, o, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,j>_aaaa*t1_aa(d,i)*t2_abab(a,c,l,k)
    triples_res += -1.000000000000000 * einsum('lbdj,di,aclk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,j>_aaaa*t1_aa(a,l)*t2_abab(d,c,i,k)
    triples_res += -1.000000000000000 * einsum('lbdj,al,dcik->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,l||j,d>_abab*t1_bb(c,l)*t2_abab(a,d,i,k)
    triples_res += -1.000000000000000 * einsum('bljd,cl,adik->abcijk', g_abab[v, o, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,l||i,d>_abab*t1_bb(d,k)*t2_abab(a,c,j,l)
    triples_res +=  1.000000000000000 * einsum('blid,dk,acjl->abcijk', g_abab[v, o, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,i>_aaaa*t1_aa(d,j)*t2_abab(a,c,l,k)
    triples_res +=  1.000000000000000 * einsum('lbdi,dj,aclk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,i>_aaaa*t1_aa(a,l)*t2_abab(d,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lbdi,al,dcjk->abcijk', g_aaaa[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,l||i,d>_abab*t1_bb(c,l)*t2_abab(a,d,j,k)
    triples_res +=  1.000000000000000 * einsum('blid,cl,adjk->abcijk', g_abab[v, o, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,k>_abab*t1_aa(d,j)*t2_aaaa(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('lcdk,dj,abil->abcijk', g_abab[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,k>_abab*t1_aa(d,i)*t2_aaaa(a,b,j,l)
    triples_res +=  1.000000000000000 * einsum('lcdk,di,abjl->abcijk', g_abab[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,k>_abab*t1_aa(a,l)*t2_aaaa(d,b,i,j)
    triples_res += -1.000000000000000 * einsum('lcdk,al,dbij->abcijk', g_abab[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,k>_abab*t1_aa(b,l)*t2_aaaa(d,a,i,j)
    triples_res +=  1.000000000000000 * einsum('lcdk,bl,daij->abcijk', g_abab[o, v, v, o], t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||j,d>_abab*t1_bb(d,k)*t2_aaaa(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('lcjd,dk,abil->abcijk', g_abab[o, v, o, v], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||j,d>_abab*t1_aa(a,l)*t2_abab(b,d,i,k)
    triples_res +=  1.000000000000000 * einsum('lcjd,al,bdik->abcijk', g_abab[o, v, o, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||j,d>_abab*t1_aa(b,l)*t2_abab(a,d,i,k)
    triples_res += -1.000000000000000 * einsum('lcjd,bl,adik->abcijk', g_abab[o, v, o, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||i,d>_abab*t1_bb(d,k)*t2_aaaa(a,b,j,l)
    triples_res +=  1.000000000000000 * einsum('lcid,dk,abjl->abcijk', g_abab[o, v, o, v], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||i,d>_abab*t1_aa(a,l)*t2_abab(b,d,j,k)
    triples_res += -1.000000000000000 * einsum('lcid,al,bdjk->abcijk', g_abab[o, v, o, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||i,d>_abab*t1_aa(b,l)*t2_abab(a,d,j,k)
    triples_res +=  1.000000000000000 * einsum('lcid,bl,adjk->abcijk', g_abab[o, v, o, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,b||d,e>_aaaa*t1_aa(d,j)*t2_abab(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('abde,dj,ecik->abcijk', g_aaaa[v, v, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,b||d,e>_aaaa*t1_aa(d,i)*t2_abab(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('abde,di,ecjk->abcijk', g_aaaa[v, v, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,c||e,d>_abab*t1_bb(d,k)*t2_aaaa(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('aced,dk,ebij->abcijk', g_abab[v, v, v, v], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,c||d,e>_abab*t1_aa(d,j)*t2_abab(b,e,i,k)
    triples_res += -1.000000000000000 * einsum('acde,dj,beik->abcijk', g_abab[v, v, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,c||d,e>_abab*t1_aa(d,i)*t2_abab(b,e,j,k)
    triples_res +=  1.000000000000000 * einsum('acde,di,bejk->abcijk', g_abab[v, v, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,c||e,d>_abab*t1_bb(d,k)*t2_aaaa(e,a,i,j)
    triples_res += -1.000000000000000 * einsum('bced,dk,eaij->abcijk', g_abab[v, v, v, v], t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,c||d,e>_abab*t1_aa(d,j)*t2_abab(a,e,i,k)
    triples_res +=  1.000000000000000 * einsum('bcde,dj,aeik->abcijk', g_abab[v, v, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,c||d,e>_abab*t1_aa(d,i)*t2_abab(a,e,j,k)
    triples_res += -1.000000000000000 * einsum('bcde,di,aejk->abcijk', g_abab[v, v, v, v], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t1_aa(d,l)*t3_aabaab(a,b,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('lmdk,dl,abcijm->abcijk', g_abab[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t1_bb(d,l)*t3_aabaab(a,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dl,abcijm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,k>_abab*t1_aa(d,j)*t3_aabaab(a,b,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mldk,dj,abciml->abcijk', g_abab[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,m||d,k>_abab*t1_aa(d,j)*t3_aabaab(a,b,c,i,l,m)
    triples_res +=  0.500000000000000 * einsum('lmdk,dj,abcilm->abcijk', g_abab[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,k>_abab*t1_aa(d,i)*t3_aabaab(a,b,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mldk,di,abcjml->abcijk', g_abab[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||d,k>_abab*t1_aa(d,i)*t3_aabaab(a,b,c,j,l,m)
    triples_res += -0.500000000000000 * einsum('lmdk,di,abcjlm->abcijk', g_abab[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t1_aa(a,l)*t3_aabaab(d,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,al,dbcijm->abcijk', g_abab[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t1_aa(b,l)*t3_aabaab(d,a,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('lmdk,bl,dacijm->abcijk', g_abab[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_abab*t1_bb(c,l)*t3_aaaaaa(d,a,b,i,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,cl,dabijm->abcijk', g_abab[o, o, v, o], t1_bb, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t1_bb(c,l)*t3_aabaab(b,a,d,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,cl,badijm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t1_aa(d,l)*t3_aabaab(a,b,c,i,m,k)
    triples_res +=  1.000000000000000 * einsum('mldj,dl,abcimk->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||j,d>_abab*t1_bb(d,l)*t3_aabaab(a,b,c,i,m,k)
    triples_res += -1.000000000000000 * einsum('mljd,dl,abcimk->abcijk', g_abab[o, o, o, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||j,d>_abab*t1_bb(d,k)*t3_aabaab(a,b,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mljd,dk,abciml->abcijk', g_abab[o, o, o, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,m||j,d>_abab*t1_bb(d,k)*t3_aabaab(a,b,c,i,l,m)
    triples_res +=  0.500000000000000 * einsum('lmjd,dk,abcilm->abcijk', g_abab[o, o, o, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,j>_aaaa*t1_aa(d,i)*t3_aabaab(a,b,c,l,m,k)
    triples_res += -0.500000000000000 * einsum('mldj,di,abclmk->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t1_aa(a,l)*t3_aabaab(d,b,c,i,m,k)
    triples_res += -1.000000000000000 * einsum('mldj,al,dbcimk->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||j,d>_abab*t1_aa(a,l)*t3_abbabb(b,d,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('lmjd,al,bdcikm->abcijk', g_abab[o, o, o, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t1_aa(b,l)*t3_aabaab(d,a,c,i,m,k)
    triples_res +=  1.000000000000000 * einsum('mldj,bl,dacimk->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||j,d>_abab*t1_aa(b,l)*t3_abbabb(a,d,c,i,k,m)
    triples_res +=  1.000000000000000 * einsum('lmjd,bl,adcikm->abcijk', g_abab[o, o, o, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||j,d>_abab*t1_bb(c,l)*t3_aabaab(b,a,d,i,m,k)
    triples_res += -1.000000000000000 * einsum('mljd,cl,badimk->abcijk', g_abab[o, o, o, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t1_aa(d,l)*t3_aabaab(a,b,c,j,m,k)
    triples_res += -1.000000000000000 * einsum('mldi,dl,abcjmk->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,d>_abab*t1_bb(d,l)*t3_aabaab(a,b,c,j,m,k)
    triples_res +=  1.000000000000000 * einsum('mlid,dl,abcjmk->abcijk', g_abab[o, o, o, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||i,d>_abab*t1_bb(d,k)*t3_aabaab(a,b,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mlid,dk,abcjml->abcijk', g_abab[o, o, o, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||i,d>_abab*t1_bb(d,k)*t3_aabaab(a,b,c,j,l,m)
    triples_res += -0.500000000000000 * einsum('lmid,dk,abcjlm->abcijk', g_abab[o, o, o, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,i>_aaaa*t1_aa(d,j)*t3_aabaab(a,b,c,l,m,k)
    triples_res +=  0.500000000000000 * einsum('mldi,dj,abclmk->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t1_aa(a,l)*t3_aabaab(d,b,c,j,m,k)
    triples_res +=  1.000000000000000 * einsum('mldi,al,dbcjmk->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||i,d>_abab*t1_aa(a,l)*t3_abbabb(b,d,c,j,k,m)
    triples_res +=  1.000000000000000 * einsum('lmid,al,bdcjkm->abcijk', g_abab[o, o, o, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t1_aa(b,l)*t3_aabaab(d,a,c,j,m,k)
    triples_res += -1.000000000000000 * einsum('mldi,bl,dacjmk->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||i,d>_abab*t1_aa(b,l)*t3_abbabb(a,d,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('lmid,bl,adcjkm->abcijk', g_abab[o, o, o, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,d>_abab*t1_bb(c,l)*t3_aabaab(b,a,d,j,m,k)
    triples_res +=  1.000000000000000 * einsum('mlid,cl,badjmk->abcijk', g_abab[o, o, o, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t1_aa(d,l)*t3_aabaab(e,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lade,dl,ebcijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||e,d>_abab*t1_bb(d,l)*t3_aabaab(e,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('aled,dl,ebcijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||e,d>_abab*t1_bb(d,k)*t3_aabaab(e,b,c,i,j,l)
    triples_res += -1.000000000000000 * einsum('aled,dk,ebcijl->abcijk', g_abab[v, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t1_aa(d,j)*t3_aabaab(e,b,c,i,l,k)
    triples_res += -1.000000000000000 * einsum('lade,dj,ebcilk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||d,e>_abab*t1_aa(d,j)*t3_abbabb(b,e,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('alde,dj,becikl->abcijk', g_abab[v, o, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t1_aa(d,i)*t3_aabaab(e,b,c,j,l,k)
    triples_res +=  1.000000000000000 * einsum('lade,di,ebcjlk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||d,e>_abab*t1_aa(d,i)*t3_abbabb(b,e,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('alde,di,becjkl->abcijk', g_abab[v, o, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,a||d,e>_aaaa*t1_aa(b,l)*t3_aabaab(d,e,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lade,bl,decijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <a,l||d,e>_abab*t1_bb(c,l)*t3_aabaab(d,b,e,i,j,k)
    triples_res += -0.500000000000000 * einsum('alde,cl,dbeijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <a,l||e,d>_abab*t1_bb(c,l)*t3_aabaab(b,e,d,i,j,k)
    triples_res +=  0.500000000000000 * einsum('aled,cl,bedijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t1_aa(d,l)*t3_aabaab(e,a,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('lbde,dl,eacijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,l||e,d>_abab*t1_bb(d,l)*t3_aabaab(e,a,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('bled,dl,eacijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,l||e,d>_abab*t1_bb(d,k)*t3_aabaab(e,a,c,i,j,l)
    triples_res +=  1.000000000000000 * einsum('bled,dk,eacijl->abcijk', g_abab[v, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_aaaa*t1_aa(d,j)*t3_aabaab(e,a,c,i,l,k)
    triples_res +=  1.000000000000000 * einsum('lbde,dj,eacilk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,l||d,e>_abab*t1_aa(d,j)*t3_abbabb(a,e,c,i,k,l)
    triples_res += -1.000000000000000 * einsum('blde,dj,aecikl->abcijk', g_abab[v, o, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t1_aa(d,i)*t3_aabaab(e,a,c,j,l,k)
    triples_res += -1.000000000000000 * einsum('lbde,di,eacjlk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,l||d,e>_abab*t1_aa(d,i)*t3_abbabb(a,e,c,j,k,l)
    triples_res +=  1.000000000000000 * einsum('blde,di,aecjkl->abcijk', g_abab[v, o, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,b||d,e>_aaaa*t1_aa(a,l)*t3_aabaab(d,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lbde,al,decijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <b,l||d,e>_abab*t1_bb(c,l)*t3_aabaab(d,a,e,i,j,k)
    triples_res +=  0.500000000000000 * einsum('blde,cl,daeijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <b,l||e,d>_abab*t1_bb(c,l)*t3_aabaab(a,e,d,i,j,k)
    triples_res += -0.500000000000000 * einsum('bled,cl,aedijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_abab*t1_aa(d,l)*t3_aabaab(b,a,e,i,j,k)
    triples_res += -1.000000000000000 * einsum('lcde,dl,baeijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t1_bb(d,l)*t3_aabaab(b,a,e,i,j,k)
    triples_res += -1.000000000000000 * einsum('lcde,dl,baeijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||e,d>_abab*t1_bb(d,k)*t3_aaaaaa(e,a,b,i,j,l)
    triples_res +=  1.000000000000000 * einsum('lced,dk,eabijl->abcijk', g_abab[o, v, v, v], t1_bb, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t1_bb(d,k)*t3_aabaab(b,a,e,i,j,l)
    triples_res +=  1.000000000000000 * einsum('lcde,dk,baeijl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_abab*t1_aa(d,j)*t3_aabaab(b,a,e,i,l,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dj,baeilk->abcijk', g_abab[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_abab*t1_aa(d,i)*t3_aabaab(b,a,e,j,l,k)
    triples_res += -1.000000000000000 * einsum('lcde,di,baejlk->abcijk', g_abab[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,c||d,e>_abab*t1_aa(a,l)*t3_aabaab(d,b,e,i,j,k)
    triples_res += -0.500000000000000 * einsum('lcde,al,dbeijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,c||e,d>_abab*t1_aa(a,l)*t3_aabaab(b,e,d,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lced,al,bedijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,c||d,e>_abab*t1_aa(b,l)*t3_aabaab(d,a,e,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lcde,bl,daeijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,c||e,d>_abab*t1_aa(b,l)*t3_aabaab(a,e,d,i,j,k)
    triples_res += -0.500000000000000 * einsum('lced,bl,aedijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_abab(d,e,l,k)*t3_aabaab(a,b,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmde,delk,abcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(e,d,l,k)*t3_aabaab(a,b,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmed,edlk,abcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,e,k,l)*t3_aabaab(a,b,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,dekl,abcijm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,e,j,l)*t3_aabaab(a,b,c,i,m,k)
    triples_res += -0.500000000000000 * einsum('mlde,dejl,abcimk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_abab(d,e,j,l)*t3_aabaab(a,b,c,i,m,k)
    triples_res += -0.500000000000000 * einsum('mlde,dejl,abcimk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_abab(e,d,j,l)*t3_aabaab(a,b,c,i,m,k)
    triples_res += -0.500000000000000 * einsum('mled,edjl,abcimk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,e,i,l)*t3_aabaab(a,b,c,j,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,deil,abcjmk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_abab(d,e,i,l)*t3_aabaab(a,b,c,j,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,deil,abcjmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_abab(e,d,i,l)*t3_aabaab(a,b,c,j,m,k)
    triples_res +=  0.500000000000000 * einsum('mled,edil,abcjmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_abab*t2_abab(d,e,j,k)*t3_aabaab(a,b,c,i,m,l)
    triples_res +=  0.250000000000000 * einsum('mlde,dejk,abciml->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,m||d,e>_abab*t2_abab(d,e,j,k)*t3_aabaab(a,b,c,i,l,m)
    triples_res +=  0.250000000000000 * einsum('lmde,dejk,abcilm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,l||e,d>_abab*t2_abab(e,d,j,k)*t3_aabaab(a,b,c,i,m,l)
    triples_res +=  0.250000000000000 * einsum('mled,edjk,abciml->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,m||e,d>_abab*t2_abab(e,d,j,k)*t3_aabaab(a,b,c,i,l,m)
    triples_res +=  0.250000000000000 * einsum('lmed,edjk,abcilm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,l||d,e>_abab*t2_abab(d,e,i,k)*t3_aabaab(a,b,c,j,m,l)
    triples_res += -0.250000000000000 * einsum('mlde,deik,abcjml->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <l,m||d,e>_abab*t2_abab(d,e,i,k)*t3_aabaab(a,b,c,j,l,m)
    triples_res += -0.250000000000000 * einsum('lmde,deik,abcjlm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,l||e,d>_abab*t2_abab(e,d,i,k)*t3_aabaab(a,b,c,j,m,l)
    triples_res += -0.250000000000000 * einsum('mled,edik,abcjml->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <l,m||e,d>_abab*t2_abab(e,d,i,k)*t3_aabaab(a,b,c,j,l,m)
    triples_res += -0.250000000000000 * einsum('lmed,edik,abcjlm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,l||d,e>_aaaa*t2_aaaa(d,e,i,j)*t3_aabaab(a,b,c,l,m,k)
    triples_res += -0.250000000000000 * einsum('mlde,deij,abclmk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,a,m,l)*t3_aabaab(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,daml,ebcijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_abab(a,d,m,l)*t3_aabaab(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mled,adml,ebcijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(a,d,l,m)*t3_aabaab(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lmed,adlm,ebcijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t2_abab(a,d,l,k)*t3_aabaab(e,b,c,i,j,m)
    triples_res +=  1.000000000000010 * einsum('lmed,adlk,ebcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,a,j,l)*t3_aabaab(e,b,c,i,m,k)
    triples_res +=  1.000000000000010 * einsum('mlde,dajl,ebcimk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_aaaa(d,a,j,l)*t3_abbabb(b,e,c,i,k,m)
    triples_res += -1.000000000000010 * einsum('lmde,dajl,becikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_abab(a,d,j,l)*t3_aabaab(e,b,c,i,m,k)
    triples_res +=  1.000000000000010 * einsum('mled,adjl,ebcimk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,j,l)*t3_abbabb(b,e,c,i,k,m)
    triples_res += -1.000000000000010 * einsum('mlde,adjl,becikm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,a,i,l)*t3_aabaab(e,b,c,j,m,k)
    triples_res += -1.000000000000010 * einsum('mlde,dail,ebcjmk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_aaaa(d,a,i,l)*t3_abbabb(b,e,c,j,k,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dail,becjkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_abab(a,d,i,l)*t3_aabaab(e,b,c,j,m,k)
    triples_res += -1.000000000000010 * einsum('mled,adil,ebcjmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,i,l)*t3_abbabb(b,e,c,j,k,m)
    triples_res +=  1.000000000000010 * einsum('mlde,adil,becjkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_abab(a,d,j,k)*t3_aabaab(e,b,c,i,m,l)
    triples_res += -0.500000000000000 * einsum('mled,adjk,ebciml->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(a,d,j,k)*t3_aabaab(e,b,c,i,l,m)
    triples_res += -0.500000000000000 * einsum('lmed,adjk,ebcilm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_abab(a,d,j,k)*t3_abbabb(b,e,c,i,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,adjk,beciml->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_abab(a,d,i,k)*t3_aabaab(e,b,c,j,m,l)
    triples_res +=  0.500000000000000 * einsum('mled,adik,ebcjml->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_abab(a,d,i,k)*t3_aabaab(e,b,c,j,l,m)
    triples_res +=  0.500000000000000 * einsum('lmed,adik,ebcjlm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_abab(a,d,i,k)*t3_abbabb(b,e,c,j,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,adik,becjml->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,a,i,j)*t3_aabaab(e,b,c,l,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,daij,ebclmk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_aaaa(d,a,i,j)*t3_abbabb(b,e,c,m,k,l)
    triples_res += -0.500000000000000 * einsum('mlde,daij,becmkl->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_aaaa(d,a,i,j)*t3_abbabb(b,e,c,l,m,k)
    triples_res +=  0.500000000000000 * einsum('lmde,daij,beclmk->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,b,m,l)*t3_aabaab(e,a,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('mlde,dbml,eacijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_abab(b,d,m,l)*t3_aabaab(e,a,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('mled,bdml,eacijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_abab(b,d,l,m)*t3_aabaab(e,a,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lmed,bdlm,eacijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t2_abab(b,d,l,k)*t3_aabaab(e,a,c,i,j,m)
    triples_res += -1.000000000000010 * einsum('lmed,bdlk,eacijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,b,j,l)*t3_aabaab(e,a,c,i,m,k)
    triples_res += -1.000000000000010 * einsum('mlde,dbjl,eacimk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_aaaa(d,b,j,l)*t3_abbabb(a,e,c,i,k,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dbjl,aecikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_abab(b,d,j,l)*t3_aabaab(e,a,c,i,m,k)
    triples_res += -1.000000000000010 * einsum('mled,bdjl,eacimk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_abab(b,d,j,l)*t3_abbabb(a,e,c,i,k,m)
    triples_res +=  1.000000000000010 * einsum('mlde,bdjl,aecikm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,b,i,l)*t3_aabaab(e,a,c,j,m,k)
    triples_res +=  1.000000000000010 * einsum('mlde,dbil,eacjmk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_aaaa(d,b,i,l)*t3_abbabb(a,e,c,j,k,m)
    triples_res += -1.000000000000010 * einsum('lmde,dbil,aecjkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_abab(b,d,i,l)*t3_aabaab(e,a,c,j,m,k)
    triples_res +=  1.000000000000010 * einsum('mled,bdil,eacjmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(b,d,i,l)*t3_abbabb(a,e,c,j,k,m)
    triples_res += -1.000000000000010 * einsum('mlde,bdil,aecjkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_abab(b,d,j,k)*t3_aabaab(e,a,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mled,bdjk,eaciml->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_abab(b,d,j,k)*t3_aabaab(e,a,c,i,l,m)
    triples_res +=  0.500000000000000 * einsum('lmed,bdjk,eacilm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_abab(b,d,j,k)*t3_abbabb(a,e,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,bdjk,aeciml->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_abab(b,d,i,k)*t3_aabaab(e,a,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mled,bdik,eacjml->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(b,d,i,k)*t3_aabaab(e,a,c,j,l,m)
    triples_res += -0.500000000000000 * einsum('lmed,bdik,eacjlm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_abab(b,d,i,k)*t3_abbabb(a,e,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,bdik,aecjml->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,b,i,j)*t3_aabaab(e,a,c,l,m,k)
    triples_res += -0.500000000000000 * einsum('mlde,dbij,eaclmk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_aaaa(d,b,i,j)*t3_abbabb(a,e,c,m,k,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dbij,aecmkl->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_aaaa(d,b,i,j)*t3_abbabb(a,e,c,l,m,k)
    triples_res += -0.500000000000000 * einsum('lmde,dbij,aeclmk->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_abab(d,c,m,l)*t3_aabaab(b,a,e,i,j,k)
    triples_res +=  0.500000000000000 * einsum('mlde,dcml,baeijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_abab(d,c,l,m)*t3_aabaab(b,a,e,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lmde,dclm,baeijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,c,m,l)*t3_aabaab(b,a,e,i,j,k)
    triples_res +=  0.500000000000000 * einsum('mlde,dcml,baeijk->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_abab(d,c,l,k)*t3_aaaaaa(e,a,b,i,j,m)
    triples_res += -1.000000000000010 * einsum('mlde,dclk,eabijm->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_abab(d,c,l,k)*t3_aabaab(b,a,e,i,j,m)
    triples_res += -1.000000000000010 * einsum('lmde,dclk,baeijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_bbbb(d,c,k,l)*t3_aaaaaa(e,a,b,i,j,m)
    triples_res += -1.000000000000010 * einsum('mled,dckl,eabijm->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,c,k,l)*t3_aabaab(b,a,e,i,j,m)
    triples_res += -1.000000000000010 * einsum('mlde,dckl,baeijm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t2_abab(d,c,j,l)*t3_aabaab(b,a,e,i,m,k)
    triples_res += -1.000000000000010 * einsum('mlde,dcjl,baeimk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t2_abab(d,c,i,l)*t3_aabaab(b,a,e,j,m,k)
    triples_res +=  1.000000000000010 * einsum('mlde,dcil,baejmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_abab(d,c,j,k)*t3_aaaaaa(e,a,b,i,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dcjk,eabiml->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_abab(d,c,j,k)*t3_aabaab(b,a,e,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dcjk,baeiml->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_abab(d,c,j,k)*t3_aabaab(b,a,e,i,l,m)
    triples_res +=  0.500000000000000 * einsum('lmde,dcjk,baeilm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_abab(d,c,i,k)*t3_aaaaaa(e,a,b,j,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dcik,eabjml->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_abab(d,c,i,k)*t3_aabaab(b,a,e,j,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dcik,baejml->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_abab(d,c,i,k)*t3_aabaab(b,a,e,j,l,m)
    triples_res += -0.500000000000000 * einsum('lmde,dcik,baejlm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_aaaa*t2_aaaa(a,b,m,l)*t3_aabaab(d,e,c,i,j,k)
    triples_res +=  0.250000000000000 * einsum('mlde,abml,decijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,b,j,l)*t3_aabaab(d,e,c,i,m,k)
    triples_res += -0.500000000000000 * einsum('mlde,abjl,decimk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_aaaa(a,b,j,l)*t3_abbabb(d,e,c,i,k,m)
    triples_res += -0.500000000000000 * einsum('lmde,abjl,decikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_aaaa(a,b,j,l)*t3_abbabb(e,d,c,i,k,m)
    triples_res += -0.500000000000000 * einsum('lmed,abjl,edcikm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(a,b,i,l)*t3_aabaab(d,e,c,j,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,abil,decjmk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_aaaa(a,b,i,l)*t3_abbabb(d,e,c,j,k,m)
    triples_res +=  0.500000000000000 * einsum('lmde,abil,decjkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_aaaa(a,b,i,l)*t3_abbabb(e,d,c,j,k,m)
    triples_res +=  0.500000000000000 * einsum('lmed,abil,edcjkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_abab*t2_abab(a,c,m,l)*t3_aabaab(d,b,e,i,j,k)
    triples_res +=  0.250000000000000 * einsum('mlde,acml,dbeijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,l||e,d>_abab*t2_abab(a,c,m,l)*t3_aabaab(b,e,d,i,j,k)
    triples_res += -0.250000000000000 * einsum('mled,acml,bedijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,m||d,e>_abab*t2_abab(a,c,l,m)*t3_aabaab(d,b,e,i,j,k)
    triples_res +=  0.250000000000000 * einsum('lmde,aclm,dbeijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <l,m||e,d>_abab*t2_abab(a,c,l,m)*t3_aabaab(b,e,d,i,j,k)
    triples_res += -0.250000000000000 * einsum('lmed,aclm,bedijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_abab(a,c,l,k)*t3_aaaaaa(d,e,b,i,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,aclk,debijm->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_abab(a,c,l,k)*t3_aabaab(d,b,e,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmde,aclk,dbeijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_abab(a,c,l,k)*t3_aabaab(b,e,d,i,j,m)
    triples_res +=  0.500000000000000 * einsum('lmed,aclk,bedijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_abab(a,c,j,l)*t3_aabaab(d,b,e,i,m,k)
    triples_res += -0.500000000000000 * einsum('mlde,acjl,dbeimk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_abab(a,c,j,l)*t3_aabaab(b,e,d,i,m,k)
    triples_res +=  0.500000000000000 * einsum('mled,acjl,bedimk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_abab(a,c,j,l)*t3_abbabb(b,e,d,i,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,acjl,bedikm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_abab(a,c,i,l)*t3_aabaab(d,b,e,j,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,acil,dbejmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_abab(a,c,i,l)*t3_aabaab(b,e,d,j,m,k)
    triples_res += -0.500000000000000 * einsum('mled,acil,bedjmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_abab(a,c,i,l)*t3_abbabb(b,e,d,j,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,acil,bedjkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <m,l||d,e>_abab*t2_abab(b,c,m,l)*t3_aabaab(d,a,e,i,j,k)
    triples_res += -0.250000000000000 * einsum('mlde,bcml,daeijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,l||e,d>_abab*t2_abab(b,c,m,l)*t3_aabaab(a,e,d,i,j,k)
    triples_res +=  0.250000000000000 * einsum('mled,bcml,aedijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <l,m||d,e>_abab*t2_abab(b,c,l,m)*t3_aabaab(d,a,e,i,j,k)
    triples_res += -0.250000000000000 * einsum('lmde,bclm,daeijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,m||e,d>_abab*t2_abab(b,c,l,m)*t3_aabaab(a,e,d,i,j,k)
    triples_res +=  0.250000000000000 * einsum('lmed,bclm,aedijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_abab(b,c,l,k)*t3_aaaaaa(d,e,a,i,j,m)
    triples_res +=  0.500000000000000 * einsum('mlde,bclk,deaijm->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aaaaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_abab(b,c,l,k)*t3_aabaab(d,a,e,i,j,m)
    triples_res +=  0.500000000000000 * einsum('lmde,bclk,daeijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(b,c,l,k)*t3_aabaab(a,e,d,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmed,bclk,aedijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_abab(b,c,j,l)*t3_aabaab(d,a,e,i,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,bcjl,daeimk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_abab(b,c,j,l)*t3_aabaab(a,e,d,i,m,k)
    triples_res += -0.500000000000000 * einsum('mled,bcjl,aedimk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_abab(b,c,j,l)*t3_abbabb(a,e,d,i,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,bcjl,aedikm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_abab(b,c,i,l)*t3_aabaab(d,a,e,j,m,k)
    triples_res += -0.500000000000000 * einsum('mlde,bcil,daejmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_abab(b,c,i,l)*t3_aabaab(a,e,d,j,m,k)
    triples_res +=  0.500000000000000 * einsum('mled,bcil,aedjmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_abab(b,c,i,l)*t3_abbabb(a,e,d,j,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,bcil,aedjkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t2_aaaa(d,a,j,l)*t2_abab(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmdk,dajl,bcim->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t2_abab(a,d,j,l)*t2_abab(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('mldk,adjl,bcim->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t2_aaaa(d,a,i,l)*t2_abab(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,dail,bcjm->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t2_abab(a,d,i,l)*t2_abab(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,adil,bcjm->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,k>_abab*t2_aaaa(d,a,i,j)*t2_abab(b,c,m,l)
    triples_res += -0.500000000000000 * einsum('mldk,daij,bcml->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,k>_abab*t2_aaaa(d,a,i,j)*t2_abab(b,c,l,m)
    triples_res += -0.500000000000000 * einsum('lmdk,daij,bclm->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t2_aaaa(d,b,j,l)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,dbjl,acim->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t2_abab(b,d,j,l)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mldk,bdjl,acim->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t2_aaaa(d,b,i,l)*t2_abab(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmdk,dbil,acjm->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t2_abab(b,d,i,l)*t2_abab(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,bdil,acjm->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,k>_abab*t2_aaaa(d,b,i,j)*t2_abab(a,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mldk,dbij,acml->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,k>_abab*t2_aaaa(d,b,i,j)*t2_abab(a,c,l,m)
    triples_res +=  0.500000000000000 * einsum('lmdk,dbij,aclm->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_abab*t2_abab(d,c,j,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dcjl,abim->abcijk', g_abab[o, o, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_abab*t2_abab(d,c,i,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,dcil,abjm->abcijk', g_abab[o, o, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||j,d>_abab*t2_abab(a,d,l,k)*t2_abab(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmjd,adlk,bcim->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t2_aaaa(d,a,i,l)*t2_abab(b,c,m,k)
    triples_res += -1.000000000000000 * einsum('mldj,dail,bcmk->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||j,d>_abab*t2_abab(a,d,i,l)*t2_abab(b,c,m,k)
    triples_res += -1.000000000000000 * einsum('mljd,adil,bcmk->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||j,d>_abab*t2_abab(a,d,i,k)*t2_abab(b,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mljd,adik,bcml->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||j,d>_abab*t2_abab(a,d,i,k)*t2_abab(b,c,l,m)
    triples_res +=  0.500000000000000 * einsum('lmjd,adik,bclm->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||j,d>_abab*t2_abab(b,d,l,k)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmjd,bdlk,acim->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t2_aaaa(d,b,i,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mldj,dbil,acmk->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||j,d>_abab*t2_abab(b,d,i,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mljd,bdil,acmk->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||j,d>_abab*t2_abab(b,d,i,k)*t2_abab(a,c,m,l)
    triples_res += -0.500000000000000 * einsum('mljd,bdik,acml->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||j,d>_abab*t2_abab(b,d,i,k)*t2_abab(a,c,l,m)
    triples_res += -0.500000000000000 * einsum('lmjd,bdik,aclm->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t2_abab(d,c,l,k)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mldj,dclk,abim->abcijk', g_aaaa[o, o, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||j,d>_abab*t2_bbbb(d,c,k,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mljd,dckl,abim->abcijk', g_abab[o, o, o, v], t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,j>_aaaa*t2_abab(d,c,i,k)*t2_aaaa(a,b,m,l)
    triples_res +=  0.500000000000000 * einsum('mldj,dcik,abml->abcijk', g_aaaa[o, o, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||i,d>_abab*t2_abab(a,d,l,k)*t2_abab(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmid,adlk,bcjm->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t2_aaaa(d,a,j,l)*t2_abab(b,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mldi,dajl,bcmk->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,d>_abab*t2_abab(a,d,j,l)*t2_abab(b,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlid,adjl,bcmk->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||i,d>_abab*t2_abab(a,d,j,k)*t2_abab(b,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlid,adjk,bcml->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||i,d>_abab*t2_abab(a,d,j,k)*t2_abab(b,c,l,m)
    triples_res += -0.500000000000000 * einsum('lmid,adjk,bclm->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||i,d>_abab*t2_abab(b,d,l,k)*t2_abab(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmid,bdlk,acjm->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t2_aaaa(d,b,j,l)*t2_abab(a,c,m,k)
    triples_res += -1.000000000000000 * einsum('mldi,dbjl,acmk->abcijk', g_aaaa[o, o, v, o], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,d>_abab*t2_abab(b,d,j,l)*t2_abab(a,c,m,k)
    triples_res += -1.000000000000000 * einsum('mlid,bdjl,acmk->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||i,d>_abab*t2_abab(b,d,j,k)*t2_abab(a,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlid,bdjk,acml->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||i,d>_abab*t2_abab(b,d,j,k)*t2_abab(a,c,l,m)
    triples_res +=  0.500000000000000 * einsum('lmid,bdjk,aclm->abcijk', g_abab[o, o, o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t2_abab(d,c,l,k)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mldi,dclk,abjm->abcijk', g_aaaa[o, o, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,d>_abab*t2_bbbb(d,c,k,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlid,dckl,abjm->abcijk', g_abab[o, o, o, v], t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,i>_aaaa*t2_abab(d,c,j,k)*t2_aaaa(a,b,m,l)
    triples_res += -0.500000000000000 * einsum('mldi,dcjk,abml->abcijk', g_aaaa[o, o, v, o], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <a,l||d,e>_abab*t2_abab(d,e,j,k)*t2_abab(b,c,i,l)
    triples_res +=  0.500000000000000 * einsum('alde,dejk,bcil->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <a,l||e,d>_abab*t2_abab(e,d,j,k)*t2_abab(b,c,i,l)
    triples_res +=  0.500000000000000 * einsum('aled,edjk,bcil->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <a,l||d,e>_abab*t2_abab(d,e,i,k)*t2_abab(b,c,j,l)
    triples_res += -0.500000000000000 * einsum('alde,deik,bcjl->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <a,l||e,d>_abab*t2_abab(e,d,i,k)*t2_abab(b,c,j,l)
    triples_res += -0.500000000000000 * einsum('aled,edik,bcjl->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,a||d,e>_aaaa*t2_aaaa(d,e,i,j)*t2_abab(b,c,l,k)
    triples_res +=  0.500000000000000 * einsum('lade,deij,bclk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t2_aaaa(d,b,j,l)*t2_abab(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('lade,dbjl,ecik->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||e,d>_abab*t2_abab(b,d,j,l)*t2_abab(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('aled,bdjl,ecik->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t2_aaaa(d,b,i,l)*t2_abab(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lade,dbil,ecjk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||e,d>_abab*t2_abab(b,d,i,l)*t2_abab(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('aled,bdil,ecjk->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||e,d>_abab*t2_abab(b,d,j,k)*t2_abab(e,c,i,l)
    triples_res += -1.000000000000000 * einsum('aled,bdjk,ecil->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <a,l||e,d>_abab*t2_abab(b,d,i,k)*t2_abab(e,c,j,l)
    triples_res +=  1.000000000000000 * einsum('aled,bdik,ecjl->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t2_aaaa(d,b,i,j)*t2_abab(e,c,l,k)
    triples_res += -1.000000000000000 * einsum('lade,dbij,eclk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <a,l||d,e>_abab*t2_aaaa(d,b,i,j)*t2_bbbb(e,c,k,l)
    triples_res += -1.000000000000000 * einsum('alde,dbij,eckl->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <b,l||d,e>_abab*t2_abab(d,e,j,k)*t2_abab(a,c,i,l)
    triples_res += -0.500000000000000 * einsum('blde,dejk,acil->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <b,l||e,d>_abab*t2_abab(e,d,j,k)*t2_abab(a,c,i,l)
    triples_res += -0.500000000000000 * einsum('bled,edjk,acil->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <b,l||d,e>_abab*t2_abab(d,e,i,k)*t2_abab(a,c,j,l)
    triples_res +=  0.500000000000000 * einsum('blde,deik,acjl->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <b,l||e,d>_abab*t2_abab(e,d,i,k)*t2_abab(a,c,j,l)
    triples_res +=  0.500000000000000 * einsum('bled,edik,acjl->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,b||d,e>_aaaa*t2_aaaa(d,e,i,j)*t2_abab(a,c,l,k)
    triples_res += -0.500000000000000 * einsum('lbde,deij,aclk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_aaaa*t2_aaaa(d,a,j,l)*t2_abab(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('lbde,dajl,ecik->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,l||e,d>_abab*t2_abab(a,d,j,l)*t2_abab(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('bled,adjl,ecik->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t2_aaaa(d,a,i,l)*t2_abab(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('lbde,dail,ecjk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,l||e,d>_abab*t2_abab(a,d,i,l)*t2_abab(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('bled,adil,ecjk->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,l||e,d>_abab*t2_abab(a,d,j,k)*t2_abab(e,c,i,l)
    triples_res +=  1.000000000000000 * einsum('bled,adjk,ecil->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <b,l||e,d>_abab*t2_abab(a,d,i,k)*t2_abab(e,c,j,l)
    triples_res += -1.000000000000000 * einsum('bled,adik,ecjl->abcijk', g_abab[v, o, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_aaaa*t2_aaaa(d,a,i,j)*t2_abab(e,c,l,k)
    triples_res +=  1.000000000000000 * einsum('lbde,daij,eclk->abcijk', g_aaaa[o, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <b,l||d,e>_abab*t2_aaaa(d,a,i,j)*t2_bbbb(e,c,k,l)
    triples_res +=  1.000000000000000 * einsum('blde,daij,eckl->abcijk', g_abab[v, o, v, v], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,c||d,e>_abab*t2_abab(d,e,j,k)*t2_aaaa(a,b,i,l)
    triples_res += -0.500000000000000 * einsum('lcde,dejk,abil->abcijk', g_abab[o, v, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,c||e,d>_abab*t2_abab(e,d,j,k)*t2_aaaa(a,b,i,l)
    triples_res += -0.500000000000000 * einsum('lced,edjk,abil->abcijk', g_abab[o, v, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,c||d,e>_abab*t2_abab(d,e,i,k)*t2_aaaa(a,b,j,l)
    triples_res +=  0.500000000000000 * einsum('lcde,deik,abjl->abcijk', g_abab[o, v, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,c||e,d>_abab*t2_abab(e,d,i,k)*t2_aaaa(a,b,j,l)
    triples_res +=  0.500000000000000 * einsum('lced,edik,abjl->abcijk', g_abab[o, v, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||e,d>_abab*t2_abab(a,d,l,k)*t2_aaaa(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('lced,adlk,ebij->abcijk', g_abab[o, v, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_abab*t2_aaaa(d,a,j,l)*t2_abab(b,e,i,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dajl,beik->abcijk', g_abab[o, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t2_abab(a,d,j,l)*t2_abab(b,e,i,k)
    triples_res += -1.000000000000000 * einsum('lcde,adjl,beik->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_abab*t2_aaaa(d,a,i,l)*t2_abab(b,e,j,k)
    triples_res += -1.000000000000000 * einsum('lcde,dail,bejk->abcijk', g_abab[o, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t2_abab(a,d,i,l)*t2_abab(b,e,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,adil,bejk->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||e,d>_abab*t2_abab(a,d,j,k)*t2_aaaa(e,b,i,l)
    triples_res +=  1.000000000000000 * einsum('lced,adjk,ebil->abcijk', g_abab[o, v, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t2_abab(a,d,j,k)*t2_abab(b,e,i,l)
    triples_res +=  1.000000000000000 * einsum('lcde,adjk,beil->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||e,d>_abab*t2_abab(a,d,i,k)*t2_aaaa(e,b,j,l)
    triples_res += -1.000000000000000 * einsum('lced,adik,ebjl->abcijk', g_abab[o, v, v, v], t2_abab, t2_aaaa, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t2_abab(a,d,i,k)*t2_abab(b,e,j,l)
    triples_res += -1.000000000000000 * einsum('lcde,adik,bejl->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_abab*t2_aaaa(d,a,i,j)*t2_abab(b,e,l,k)
    triples_res +=  1.000000000000000 * einsum('lcde,daij,belk->abcijk', g_abab[o, v, v, v], t2_aaaa, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_abab(a,e,j,k)*t2_abab(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dl,aejk,bcim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_abab(a,e,j,k)*t2_abab(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,aejk,bcim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_abab(a,e,i,k)*t2_abab(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmde,dl,aeik,bcjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_abab(a,e,i,k)*t2_abab(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,aeik,bcjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_aaaa(e,a,i,j)*t2_abab(b,c,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,dl,eaij,bcmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_aaaa(e,a,i,j)*t2_abab(b,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mled,dl,eaij,bcmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_abab(b,e,j,k)*t2_abab(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmde,dl,bejk,acim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_abab(b,e,j,k)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,bejk,acim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_abab(b,e,i,k)*t2_abab(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dl,beik,acjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_abab(b,e,i,k)*t2_abab(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,beik,acjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_aaaa(e,b,i,j)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ebij,acmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_aaaa(e,b,i,j)*t2_abab(a,c,m,k)
    triples_res += -1.000000000000000 * einsum('mled,dl,ebij,acmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_abab(e,c,j,k)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ecjk,abim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_abab(e,c,j,k)*t2_aaaa(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mled,dl,ecjk,abim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_abab(e,c,i,k)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ecik,abjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_abab(e,c,i,k)*t2_aaaa(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mled,dl,ecik,abjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t2_aaaa(e,a,j,l)*t2_abab(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmed,dk,eajl,bcim->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_abab(a,e,j,l)*t2_abab(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,aejl,bcim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t2_aaaa(e,a,i,l)*t2_abab(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,eail,bcjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_abab(a,e,i,l)*t2_abab(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,aeil,bcjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t1_bb(d,k)*t2_aaaa(e,a,i,j)*t2_abab(b,c,m,l)
    triples_res += -0.500000000000000 * einsum('mled,dk,eaij,bcml->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t1_bb(d,k)*t2_aaaa(e,a,i,j)*t2_abab(b,c,l,m)
    triples_res += -0.500000000000000 * einsum('lmed,dk,eaij,bclm->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t2_aaaa(e,b,j,l)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,ebjl,acim->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_abab(b,e,j,l)*t2_abab(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,bejl,acim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t2_aaaa(e,b,i,l)*t2_abab(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmed,dk,ebil,acjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_abab(b,e,i,l)*t2_abab(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,beil,acjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t1_bb(d,k)*t2_aaaa(e,b,i,j)*t2_abab(a,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mled,dk,ebij,acml->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t1_bb(d,k)*t2_aaaa(e,b,i,j)*t2_abab(a,c,l,m)
    triples_res +=  0.500000000000000 * einsum('lmed,dk,ebij,aclm->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t2_abab(e,c,j,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mled,dk,ecjl,abim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t2_abab(e,c,i,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mled,dk,ecil,abjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,j)*t2_abab(a,e,l,k)*t2_abab(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmde,dj,aelk,bcim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,a,i,l)*t2_abab(b,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,eail,bcmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,j)*t2_abab(a,e,i,l)*t2_abab(b,c,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,dj,aeil,bcmk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t1_aa(d,j)*t2_abab(a,e,i,k)*t2_abab(b,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dj,aeik,bcml->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t1_aa(d,j)*t2_abab(a,e,i,k)*t2_abab(b,c,l,m)
    triples_res +=  0.500000000000000 * einsum('lmde,dj,aeik,bclm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,j)*t2_abab(b,e,l,k)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dj,belk,acim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t2_aaaa(e,b,i,l)*t2_abab(a,c,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,dj,ebil,acmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,j)*t2_abab(b,e,i,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,beil,acmk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t1_aa(d,j)*t2_abab(b,e,i,k)*t2_abab(a,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dj,beik,acml->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t1_aa(d,j)*t2_abab(b,e,i,k)*t2_abab(a,c,l,m)
    triples_res += -0.500000000000000 * einsum('lmde,dj,beik,aclm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t2_abab(e,c,l,k)*t2_aaaa(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,eclk,abim->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,j)*t2_bbbb(e,c,k,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,eckl,abim->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(d,j)*t2_abab(e,c,i,k)*t2_aaaa(a,b,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dj,ecik,abml->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t2_abab(a,e,l,k)*t2_abab(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmde,di,aelk,bcjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,a,j,l)*t2_abab(b,c,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,di,eajl,bcmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t2_abab(a,e,j,l)*t2_abab(b,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,di,aejl,bcmk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t1_aa(d,i)*t2_abab(a,e,j,k)*t2_abab(b,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,di,aejk,bcml->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t1_aa(d,i)*t2_abab(a,e,j,k)*t2_abab(b,c,l,m)
    triples_res += -0.500000000000000 * einsum('lmde,di,aejk,bclm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t2_abab(b,e,l,k)*t2_abab(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmde,di,belk,acjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_aaaa(e,b,j,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,di,ebjl,acmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t2_abab(b,e,j,l)*t2_abab(a,c,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,di,bejl,acmk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t1_aa(d,i)*t2_abab(b,e,j,k)*t2_abab(a,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,di,bejk,acml->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t1_aa(d,i)*t2_abab(b,e,j,k)*t2_abab(a,c,l,m)
    triples_res +=  0.500000000000000 * einsum('lmde,di,bejk,aclm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_abab(e,c,l,k)*t2_aaaa(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,eclk,abjm->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t2_bbbb(e,c,k,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,di,eckl,abjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_abab(e,c,j,k)*t2_aaaa(a,b,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,di,ecjk,abml->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t1_aa(a,l)*t2_abab(d,e,j,k)*t2_abab(b,c,i,m)
    triples_res += -0.500000000000000 * einsum('lmde,al,dejk,bcim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t1_aa(a,l)*t2_abab(e,d,j,k)*t2_abab(b,c,i,m)
    triples_res += -0.500000000000000 * einsum('lmed,al,edjk,bcim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t1_aa(a,l)*t2_abab(d,e,i,k)*t2_abab(b,c,j,m)
    triples_res +=  0.500000000000000 * einsum('lmde,al,deik,bcjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t1_aa(a,l)*t2_abab(e,d,i,k)*t2_abab(b,c,j,m)
    triples_res +=  0.500000000000000 * einsum('lmed,al,edik,bcjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,e,i,j)*t2_abab(b,c,m,k)
    triples_res += -0.500000000000000 * einsum('mlde,al,deij,bcmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,j,m)*t2_abab(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('mlde,al,dbjm,ecik->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,d,j,m)*t2_abab(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('lmed,al,bdjm,ecik->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,m)*t2_abab(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,al,dbim,ecjk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,d,i,m)*t2_abab(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lmed,al,bdim,ecjk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,d,j,k)*t2_abab(e,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmed,al,bdjk,ecim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_abab(b,d,i,k)*t2_abab(e,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmed,al,bdik,ecjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_aaaa(d,b,i,j)*t2_abab(e,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,al,dbij,ecmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(a,l)*t2_aaaa(d,b,i,j)*t2_bbbb(e,c,k,m)
    triples_res +=  1.000000000000000 * einsum('lmde,al,dbij,eckm->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t1_aa(b,l)*t2_abab(d,e,j,k)*t2_abab(a,c,i,m)
    triples_res +=  0.500000000000000 * einsum('lmde,bl,dejk,acim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t1_aa(b,l)*t2_abab(e,d,j,k)*t2_abab(a,c,i,m)
    triples_res +=  0.500000000000000 * einsum('lmed,bl,edjk,acim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t1_aa(b,l)*t2_abab(d,e,i,k)*t2_abab(a,c,j,m)
    triples_res += -0.500000000000000 * einsum('lmde,bl,deik,acjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t1_aa(b,l)*t2_abab(e,d,i,k)*t2_abab(a,c,j,m)
    triples_res += -0.500000000000000 * einsum('lmed,bl,edik,acjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(d,e,i,j)*t2_abab(a,c,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,bl,deij,acmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(d,a,j,m)*t2_abab(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('mlde,bl,dajm,ecik->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_aa(b,l)*t2_abab(a,d,j,m)*t2_abab(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('lmed,bl,adjm,ecik->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(d,a,i,m)*t2_abab(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,bl,daim,ecjk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_aa(b,l)*t2_abab(a,d,i,m)*t2_abab(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('lmed,bl,adim,ecjk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_aa(b,l)*t2_abab(a,d,j,k)*t2_abab(e,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmed,bl,adjk,ecim->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_aa(b,l)*t2_abab(a,d,i,k)*t2_abab(e,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmed,bl,adik,ecjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(b,l)*t2_aaaa(d,a,i,j)*t2_abab(e,c,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,bl,daij,ecmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(b,l)*t2_aaaa(d,a,i,j)*t2_bbbb(e,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,bl,daij,eckm->abcijk', g_abab[o, o, v, v], t1_aa, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t1_bb(c,l)*t2_abab(d,e,j,k)*t2_aaaa(a,b,i,m)
    triples_res +=  0.500000000000000 * einsum('mlde,cl,dejk,abim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t1_bb(c,l)*t2_abab(e,d,j,k)*t2_aaaa(a,b,i,m)
    triples_res +=  0.500000000000000 * einsum('mled,cl,edjk,abim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t1_bb(c,l)*t2_abab(d,e,i,k)*t2_aaaa(a,b,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,cl,deik,abjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t1_bb(c,l)*t2_abab(e,d,i,k)*t2_aaaa(a,b,j,m)
    triples_res += -0.500000000000000 * einsum('mled,cl,edik,abjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(c,l)*t2_abab(a,d,m,k)*t2_aaaa(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('mled,cl,admk,ebij->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_bb(c,l)*t2_aaaa(d,a,j,m)*t2_abab(b,e,i,k)
    triples_res += -1.000000000000000 * einsum('mlde,cl,dajm,beik->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_abab(a,d,j,m)*t2_abab(b,e,i,k)
    triples_res +=  1.000000000000000 * einsum('mlde,cl,adjm,beik->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_bb(c,l)*t2_aaaa(d,a,i,m)*t2_abab(b,e,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,cl,daim,bejk->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_abab(a,d,i,m)*t2_abab(b,e,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,cl,adim,bejk->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(c,l)*t2_abab(a,d,j,k)*t2_aaaa(e,b,i,m)
    triples_res += -1.000000000000000 * einsum('mled,cl,adjk,ebim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_abab(a,d,j,k)*t2_abab(b,e,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,cl,adjk,beim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(c,l)*t2_abab(a,d,i,k)*t2_aaaa(e,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mled,cl,adik,ebjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_aaaa, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_abab(a,d,i,k)*t2_abab(b,e,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,cl,adik,bejm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_bb(c,l)*t2_aaaa(d,a,i,j)*t2_abab(b,e,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,cl,daij,bemk->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t1_aa(d,j)*t1_aa(a,l)*t2_abab(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmdk,dj,al,bcim->abcijk', g_abab[o, o, v, o], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t1_aa(d,j)*t1_aa(b,l)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,dj,bl,acim->abcijk', g_abab[o, o, v, o], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_abab*t1_aa(d,j)*t1_bb(c,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dj,cl,abim->abcijk', g_abab[o, o, v, o], t1_aa, t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t1_aa(d,i)*t1_aa(a,l)*t2_abab(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,di,al,bcjm->abcijk', g_abab[o, o, v, o], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t1_aa(d,i)*t1_aa(b,l)*t2_abab(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmdk,di,bl,acjm->abcijk', g_abab[o, o, v, o], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_abab*t1_aa(d,i)*t1_bb(c,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,di,cl,abjm->abcijk', g_abab[o, o, v, o], t1_aa, t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t1_aa(a,l)*t1_bb(c,m)*t2_aaaa(d,b,i,j)
    triples_res +=  1.000000000000000 * einsum('lmdk,al,cm,dbij->abcijk', g_abab[o, o, v, o], t1_aa, t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t1_aa(b,l)*t1_bb(c,m)*t2_aaaa(d,a,i,j)
    triples_res += -1.000000000000000 * einsum('lmdk,bl,cm,daij->abcijk', g_abab[o, o, v, o], t1_aa, t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||j,d>_abab*t1_bb(d,k)*t1_aa(a,l)*t2_abab(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmjd,dk,al,bcim->abcijk', g_abab[o, o, o, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||j,d>_abab*t1_bb(d,k)*t1_aa(b,l)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmjd,dk,bl,acim->abcijk', g_abab[o, o, o, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||j,d>_abab*t1_bb(d,k)*t1_bb(c,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mljd,dk,cl,abim->abcijk', g_abab[o, o, o, v], t1_bb, t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t1_aa(d,i)*t1_aa(a,l)*t2_abab(b,c,m,k)
    triples_res += -1.000000000000000 * einsum('mldj,di,al,bcmk->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_aaaa*t1_aa(d,i)*t1_aa(b,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mldj,di,bl,acmk->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t2_abab(d,c,i,k)
    triples_res += -1.000000000000000 * einsum('mldj,al,bm,dcik->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||j,d>_abab*t1_aa(a,l)*t1_bb(c,m)*t2_abab(b,d,i,k)
    triples_res += -1.000000000000000 * einsum('lmjd,al,cm,bdik->abcijk', g_abab[o, o, o, v], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||j,d>_abab*t1_aa(b,l)*t1_bb(c,m)*t2_abab(a,d,i,k)
    triples_res +=  1.000000000000000 * einsum('lmjd,bl,cm,adik->abcijk', g_abab[o, o, o, v], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||i,d>_abab*t1_bb(d,k)*t1_aa(a,l)*t2_abab(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmid,dk,al,bcjm->abcijk', g_abab[o, o, o, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||i,d>_abab*t1_bb(d,k)*t1_aa(b,l)*t2_abab(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmid,dk,bl,acjm->abcijk', g_abab[o, o, o, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||i,d>_abab*t1_bb(d,k)*t1_bb(c,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlid,dk,cl,abjm->abcijk', g_abab[o, o, o, v], t1_bb, t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t1_aa(d,j)*t1_aa(a,l)*t2_abab(b,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mldi,dj,al,bcmk->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t1_aa(d,j)*t1_aa(b,l)*t2_abab(a,c,m,k)
    triples_res += -1.000000000000000 * einsum('mldi,dj,bl,acmk->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t2_abab(d,c,j,k)
    triples_res +=  1.000000000000000 * einsum('mldi,al,bm,dcjk->abcijk', g_aaaa[o, o, v, o], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||i,d>_abab*t1_aa(a,l)*t1_bb(c,m)*t2_abab(b,d,j,k)
    triples_res +=  1.000000000000000 * einsum('lmid,al,cm,bdjk->abcijk', g_abab[o, o, o, v], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||i,d>_abab*t1_aa(b,l)*t1_bb(c,m)*t2_abab(a,d,j,k)
    triples_res += -1.000000000000000 * einsum('lmid,bl,cm,adjk->abcijk', g_abab[o, o, o, v], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <a,l||e,d>_abab*t1_bb(d,k)*t1_aa(e,j)*t2_abab(b,c,i,l)
    triples_res +=  1.000000000000000 * einsum('aled,dk,ej,bcil->abcijk', g_abab[v, o, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,l||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t2_abab(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('aled,dk,ei,bcjl->abcijk', g_abab[v, o, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,l||e,d>_abab*t1_bb(d,k)*t1_bb(c,l)*t2_aaaa(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('aled,dk,cl,ebij->abcijk', g_abab[v, o, v, v], t1_bb, t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t1_aa(d,j)*t1_aa(e,i)*t2_abab(b,c,l,k)
    triples_res += -1.000000000000000 * einsum('lade,dj,ei,bclk->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t1_aa(d,j)*t1_aa(b,l)*t2_abab(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('lade,dj,bl,ecik->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <a,l||d,e>_abab*t1_aa(d,j)*t1_bb(c,l)*t2_abab(b,e,i,k)
    triples_res +=  1.000000000000000 * einsum('alde,dj,cl,beik->abcijk', g_abab[v, o, v, v], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t1_aa(d,i)*t1_aa(b,l)*t2_abab(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lade,di,bl,ecjk->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,l||d,e>_abab*t1_aa(d,i)*t1_bb(c,l)*t2_abab(b,e,j,k)
    triples_res += -1.000000000000000 * einsum('alde,di,cl,bejk->abcijk', g_abab[v, o, v, v], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <b,l||e,d>_abab*t1_bb(d,k)*t1_aa(e,j)*t2_abab(a,c,i,l)
    triples_res += -1.000000000000000 * einsum('bled,dk,ej,acil->abcijk', g_abab[v, o, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <b,l||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t2_abab(a,c,j,l)
    triples_res +=  1.000000000000000 * einsum('bled,dk,ei,acjl->abcijk', g_abab[v, o, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <b,l||e,d>_abab*t1_bb(d,k)*t1_bb(c,l)*t2_aaaa(e,a,i,j)
    triples_res +=  1.000000000000000 * einsum('bled,dk,cl,eaij->abcijk', g_abab[v, o, v, v], t1_bb, t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_aaaa*t1_aa(d,j)*t1_aa(e,i)*t2_abab(a,c,l,k)
    triples_res +=  1.000000000000000 * einsum('lbde,dj,ei,aclk->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_aaaa*t1_aa(d,j)*t1_aa(a,l)*t2_abab(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('lbde,dj,al,ecik->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <b,l||d,e>_abab*t1_aa(d,j)*t1_bb(c,l)*t2_abab(a,e,i,k)
    triples_res += -1.000000000000000 * einsum('blde,dj,cl,aeik->abcijk', g_abab[v, o, v, v], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_aaaa*t1_aa(d,i)*t1_aa(a,l)*t2_abab(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('lbde,di,al,ecjk->abcijk', g_aaaa[o, v, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <b,l||d,e>_abab*t1_aa(d,i)*t1_bb(c,l)*t2_abab(a,e,j,k)
    triples_res +=  1.000000000000000 * einsum('blde,di,cl,aejk->abcijk', g_abab[v, o, v, v], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||e,d>_abab*t1_bb(d,k)*t1_aa(e,j)*t2_aaaa(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('lced,dk,ej,abil->abcijk', g_abab[o, v, v, v], t1_bb, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t2_aaaa(a,b,j,l)
    triples_res +=  1.000000000000000 * einsum('lced,dk,ei,abjl->abcijk', g_abab[o, v, v, v], t1_bb, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||e,d>_abab*t1_bb(d,k)*t1_aa(a,l)*t2_aaaa(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('lced,dk,al,ebij->abcijk', g_abab[o, v, v, v], t1_bb, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||e,d>_abab*t1_bb(d,k)*t1_aa(b,l)*t2_aaaa(e,a,i,j)
    triples_res +=  1.000000000000000 * einsum('lced,dk,bl,eaij->abcijk', g_abab[o, v, v, v], t1_bb, t1_aa, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_abab*t1_aa(d,j)*t1_aa(a,l)*t2_abab(b,e,i,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dj,al,beik->abcijk', g_abab[o, v, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_abab*t1_aa(d,j)*t1_aa(b,l)*t2_abab(a,e,i,k)
    triples_res += -1.000000000000000 * einsum('lcde,dj,bl,aeik->abcijk', g_abab[o, v, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_abab*t1_aa(d,i)*t1_aa(a,l)*t2_abab(b,e,j,k)
    triples_res += -1.000000000000000 * einsum('lcde,di,al,bejk->abcijk', g_abab[o, v, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_abab*t1_aa(d,i)*t1_aa(b,l)*t2_abab(a,e,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,di,bl,aejk->abcijk', g_abab[o, v, v, v], t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t1_bb(e,k)*t3_aabaab(a,b,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('lmde,dl,ek,abcijm->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t1_bb(e,k)*t3_aabaab(a,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ek,abcijm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t1_aa(e,j)*t3_aabaab(a,b,c,i,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ej,abcimk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t1_aa(e,j)*t3_aabaab(a,b,c,i,m,k)
    triples_res += -1.000000000000000 * einsum('mled,dl,ej,abcimk->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t1_aa(e,i)*t3_aabaab(a,b,c,j,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ei,abcjmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t1_aa(e,i)*t3_aabaab(a,b,c,j,m,k)
    triples_res +=  1.000000000000000 * einsum('mled,dl,ei,abcjmk->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t1_aa(a,m)*t3_aabaab(e,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,am,ebcijk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t1_aa(a,m)*t3_aabaab(e,b,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('mled,dl,am,ebcijk->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t1_aa(b,m)*t3_aabaab(e,a,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,dl,bm,eacijk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t1_aa(b,m)*t3_aabaab(e,a,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('mled,dl,bm,eacijk->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t1_bb(c,m)*t3_aabaab(b,a,e,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lmde,dl,cm,baeijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t1_bb(c,m)*t3_aabaab(b,a,e,i,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,dl,cm,baeijk->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t1_bb(d,k)*t1_aa(e,j)*t3_aabaab(a,b,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mled,dk,ej,abciml->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(e,j)*t3_aabaab(a,b,c,i,l,m)
    triples_res +=  0.500000000000000 * einsum('lmed,dk,ej,abcilm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t3_aabaab(a,b,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mled,dk,ei,abcjml->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t3_aabaab(a,b,c,j,l,m)
    triples_res += -0.500000000000000 * einsum('lmed,dk,ei,abcjlm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(a,l)*t3_aabaab(e,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,al,ebcijm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(b,l)*t3_aabaab(e,a,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('lmed,dk,bl,eacijm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t1_bb(c,l)*t3_aaaaaa(e,a,b,i,j,m)
    triples_res += -1.000000000000000 * einsum('mled,dk,cl,eabijm->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_aaaaaa, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(c,l)*t3_aabaab(b,a,e,i,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,cl,baeijm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(e,i)*t3_aabaab(a,b,c,l,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,dj,ei,abclmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(a,l)*t3_aabaab(e,b,c,i,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,al,ebcimk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,j)*t1_aa(a,l)*t3_abbabb(b,e,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,dj,al,becikm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(b,l)*t3_aabaab(e,a,c,i,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,dj,bl,eacimk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,j)*t1_aa(b,l)*t3_abbabb(a,e,c,i,k,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dj,bl,aecikm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,j)*t1_bb(c,l)*t3_aabaab(b,a,e,i,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,dj,cl,baeimk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t1_aa(a,l)*t3_aabaab(e,b,c,j,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,di,al,ebcjmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t1_aa(a,l)*t3_abbabb(b,e,c,j,k,m)
    triples_res +=  1.000000000000000 * einsum('lmde,di,al,becjkm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t1_aa(b,l)*t3_aabaab(e,a,c,j,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,di,bl,eacjmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t1_aa(b,l)*t3_abbabb(a,e,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,di,bl,aecjkm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t1_bb(c,l)*t3_aabaab(b,a,e,j,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,di,cl,baejmk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t1_aa(a,l)*t1_aa(b,m)*t3_aabaab(d,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,al,bm,decijk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t1_aa(a,l)*t1_bb(c,m)*t3_aabaab(d,b,e,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lmde,al,cm,dbeijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t1_aa(a,l)*t1_bb(c,m)*t3_aabaab(b,e,d,i,j,k)
    triples_res += -0.500000000000000 * einsum('lmed,al,cm,bedijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t1_aa(b,l)*t1_bb(c,m)*t3_aabaab(d,a,e,i,j,k)
    triples_res += -0.500000000000000 * einsum('lmde,bl,cm,daeijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t1_aa(b,l)*t1_bb(c,m)*t3_aabaab(a,e,d,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lmed,bl,cm,aedijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(e,j)*t1_aa(a,l)*t2_abab(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmed,dk,ej,al,bcim->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(e,j)*t1_aa(b,l)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,ej,bl,acim->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t1_aa(e,j)*t1_bb(c,l)*t2_aaaa(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mled,dk,ej,cl,abim->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t1_aa(a,l)*t2_abab(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,ei,al,bcjm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t1_aa(b,l)*t2_abab(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmed,dk,ei,bl,acjm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t1_bb(c,l)*t2_aaaa(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mled,dk,ei,cl,abjm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(a,l)*t1_bb(c,m)*t2_aaaa(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,al,cm,ebij->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(b,l)*t1_bb(c,m)*t2_aaaa(e,a,i,j)
    triples_res += -1.000000000000000 * einsum('lmed,dk,bl,cm,eaij->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, t2_aaaa, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(e,i)*t1_aa(a,l)*t2_abab(b,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,ei,al,bcmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(e,i)*t1_aa(b,l)*t2_abab(a,c,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,dj,ei,bl,acmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,j)*t1_aa(a,l)*t1_aa(b,m)*t2_abab(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,al,bm,ecik->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,j)*t1_aa(a,l)*t1_bb(c,m)*t2_abab(b,e,i,k)
    triples_res += -1.000000000000000 * einsum('lmde,dj,al,cm,beik->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,j)*t1_aa(b,l)*t1_bb(c,m)*t2_abab(a,e,i,k)
    triples_res +=  1.000000000000000 * einsum('lmde,dj,bl,cm,aeik->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t1_aa(a,l)*t1_aa(b,m)*t2_abab(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,di,al,bm,ecjk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t1_aa(a,l)*t1_bb(c,m)*t2_abab(b,e,j,k)
    triples_res +=  1.000000000000000 * einsum('lmde,di,al,cm,bejk->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t1_aa(b,l)*t1_bb(c,m)*t2_abab(a,e,j,k)
    triples_res += -1.000000000000000 * einsum('lmde,di,bl,cm,aejk->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    return triples_res
    
    
def ccsdt_t3_abbabb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):
    
    #    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 f_bb(l,k)*t3_abbabb(a,b,c,i,j,l)
    triples_res = -1.000000000000000 * einsum('lk,abcijl->abcijk', f_bb[o, o], t3_abbabb)
    
    #	  1.0000 f_bb(l,j)*t3_abbabb(a,b,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('lj,abcikl->abcijk', f_bb[o, o], t3_abbabb)
    
    #	  1.0000 f_aa(l,i)*t3_abbabb(a,b,c,l,k,j)
    triples_res +=  1.000000000000000 * einsum('li,abclkj->abcijk', f_aa[o, o], t3_abbabb)
    
    #	  1.0000 f_aa(a,d)*t3_abbabb(d,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('ad,dbcijk->abcijk', f_aa[v, v], t3_abbabb)
    
    #	  1.0000 f_bb(b,d)*t3_abbabb(a,d,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('bd,adcijk->abcijk', f_bb[v, v], t3_abbabb)
    
    #	 -1.0000 f_bb(c,d)*t3_abbabb(a,d,b,i,j,k)
    triples_res += -1.000000000000000 * einsum('cd,adbijk->abcijk', f_bb[v, v], t3_abbabb)
    
    #	 -1.0000 f_bb(l,d)*t1_bb(d,k)*t3_abbabb(a,b,c,i,j,l)
    triples_res += -1.000000000000000 * einsum('ld,dk,abcijl->abcijk', f_bb[o, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(l,d)*t1_bb(d,j)*t3_abbabb(a,b,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('ld,dj,abcikl->abcijk', f_bb[o, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(l,d)*t1_aa(d,i)*t3_abbabb(a,b,c,l,k,j)
    triples_res +=  1.000000000000000 * einsum('ld,di,abclkj->abcijk', f_aa[o, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t1_aa(a,l)*t3_abbabb(d,b,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('ld,al,dbcijk->abcijk', f_aa[o, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(l,d)*t1_bb(b,l)*t3_abbabb(a,d,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('ld,bl,adcijk->abcijk', f_bb[o, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(l,d)*t1_bb(c,l)*t3_abbabb(a,d,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('ld,cl,adbijk->abcijk', f_bb[o, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(l,d)*t2_abab(a,d,i,k)*t2_bbbb(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('ld,adik,bcjl->abcijk', f_bb[o, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(l,d)*t2_abab(a,d,i,j)*t2_bbbb(b,c,k,l)
    triples_res +=  1.000000000000000 * einsum('ld,adij,bckl->abcijk', f_bb[o, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(l,d)*t2_bbbb(d,b,j,k)*t2_abab(a,c,i,l)
    triples_res +=  1.000000000000000 * einsum('ld,dbjk,acil->abcijk', f_bb[o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(l,d)*t2_abab(d,b,i,k)*t2_abab(a,c,l,j)
    triples_res +=  1.000000000000000 * einsum('ld,dbik,aclj->abcijk', f_aa[o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t2_abab(d,b,i,j)*t2_abab(a,c,l,k)
    triples_res += -1.000000000000000 * einsum('ld,dbij,aclk->abcijk', f_aa[o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(l,d)*t2_bbbb(d,c,j,k)*t2_abab(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('ld,dcjk,abil->abcijk', f_bb[o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_aa(l,d)*t2_abab(d,c,i,k)*t2_abab(a,b,l,j)
    triples_res += -1.000000000000000 * einsum('ld,dcik,ablj->abcijk', f_aa[o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_aa(l,d)*t2_abab(d,c,i,j)*t2_abab(a,b,l,k)
    triples_res +=  1.000000000000000 * einsum('ld,dcij,ablk->abcijk', f_aa[o, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||i,k>_abab*t2_bbbb(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('alik,bcjl->abcijk', g_abab[v, o, o, o], t2_bbbb)
    
    #	  1.0000 <a,l||i,j>_abab*t2_bbbb(b,c,k,l)
    triples_res +=  1.000000000000000 * einsum('alij,bckl->abcijk', g_abab[v, o, o, o], t2_bbbb)
    
    #	  1.0000 <l,b||j,k>_bbbb*t2_abab(a,c,i,l)
    triples_res +=  1.000000000000000 * einsum('lbjk,acil->abcijk', g_bbbb[o, v, o, o], t2_abab)
    
    #	  1.0000 <l,b||i,k>_abab*t2_abab(a,c,l,j)
    triples_res +=  1.000000000000000 * einsum('lbik,aclj->abcijk', g_abab[o, v, o, o], t2_abab)
    
    #	 -1.0000 <l,b||i,j>_abab*t2_abab(a,c,l,k)
    triples_res += -1.000000000000000 * einsum('lbij,aclk->abcijk', g_abab[o, v, o, o], t2_abab)
    
    #	 -1.0000 <l,c||j,k>_bbbb*t2_abab(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('lcjk,abil->abcijk', g_bbbb[o, v, o, o], t2_abab)
    
    #	 -1.0000 <l,c||i,k>_abab*t2_abab(a,b,l,j)
    triples_res += -1.000000000000000 * einsum('lcik,ablj->abcijk', g_abab[o, v, o, o], t2_abab)
    
    #	  1.0000 <l,c||i,j>_abab*t2_abab(a,b,l,k)
    triples_res +=  1.000000000000000 * einsum('lcij,ablk->abcijk', g_abab[o, v, o, o], t2_abab)
    
    #	 -1.0000 <a,b||d,k>_abab*t2_abab(d,c,i,j)
    triples_res += -1.000000000000000 * einsum('abdk,dcij->abcijk', g_abab[v, v, v, o], t2_abab)
    
    #	  1.0000 <a,b||d,j>_abab*t2_abab(d,c,i,k)
    triples_res +=  1.000000000000000 * einsum('abdj,dcik->abcijk', g_abab[v, v, v, o], t2_abab)
    
    #	  1.0000 <a,b||i,d>_abab*t2_bbbb(d,c,j,k)
    triples_res +=  1.000000000000000 * einsum('abid,dcjk->abcijk', g_abab[v, v, o, v], t2_bbbb)
    
    #	  1.0000 <a,c||d,k>_abab*t2_abab(d,b,i,j)
    triples_res +=  1.000000000000000 * einsum('acdk,dbij->abcijk', g_abab[v, v, v, o], t2_abab)
    
    #	 -1.0000 <a,c||d,j>_abab*t2_abab(d,b,i,k)
    triples_res += -1.000000000000000 * einsum('acdj,dbik->abcijk', g_abab[v, v, v, o], t2_abab)
    
    #	 -1.0000 <a,c||i,d>_abab*t2_bbbb(d,b,j,k)
    triples_res += -1.000000000000000 * einsum('acid,dbjk->abcijk', g_abab[v, v, o, v], t2_bbbb)
    
    #	  1.0000 <b,c||d,k>_bbbb*t2_abab(a,d,i,j)
    triples_res +=  1.000000000000000 * einsum('bcdk,adij->abcijk', g_bbbb[v, v, v, o], t2_abab)
    
    #	 -1.0000 <b,c||d,j>_bbbb*t2_abab(a,d,i,k)
    triples_res += -1.000000000000000 * einsum('bcdj,adik->abcijk', g_bbbb[v, v, v, o], t2_abab)
    
    #	  0.5000 <m,l||j,k>_bbbb*t3_abbabb(a,b,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mljk,abciml->abcijk', g_bbbb[o, o, o, o], t3_abbabb)
    
    #	  0.5000 <m,l||i,k>_abab*t3_abbabb(a,b,c,m,j,l)
    triples_res +=  0.500000000000000 * einsum('mlik,abcmjl->abcijk', g_abab[o, o, o, o], t3_abbabb)
    
    #	 -0.5000 <l,m||i,k>_abab*t3_abbabb(a,b,c,l,m,j)
    triples_res += -0.500000000000000 * einsum('lmik,abclmj->abcijk', g_abab[o, o, o, o], t3_abbabb)
    
    #	 -0.5000 <m,l||i,j>_abab*t3_abbabb(a,b,c,m,k,l)
    triples_res += -0.500000000000000 * einsum('mlij,abcmkl->abcijk', g_abab[o, o, o, o], t3_abbabb)
    
    #	  0.5000 <l,m||i,j>_abab*t3_abbabb(a,b,c,l,m,k)
    triples_res +=  0.500000000000000 * einsum('lmij,abclmk->abcijk', g_abab[o, o, o, o], t3_abbabb)
    
    #	 -1.0000 <a,l||d,k>_abab*t3_abbabb(d,b,c,i,j,l)
    triples_res += -1.000000000000000 * einsum('aldk,dbcijl->abcijk', g_abab[v, o, v, o], t3_abbabb)
    
    #	  1.0000 <a,l||d,j>_abab*t3_abbabb(d,b,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('aldj,dbcikl->abcijk', g_abab[v, o, v, o], t3_abbabb)
    
    #	 -1.0000 <l,a||d,i>_aaaa*t3_abbabb(d,b,c,l,k,j)
    triples_res += -1.000000000000000 * einsum('ladi,dbclkj->abcijk', g_aaaa[o, v, v, o], t3_abbabb)
    
    #	  1.0000 <a,l||i,d>_abab*t3_bbbbbb(d,b,c,j,k,l)
    triples_res +=  1.000000000000000 * einsum('alid,dbcjkl->abcijk', g_abab[v, o, o, v], t3_bbbbbb)
    
    #	  1.0000 <l,b||d,k>_abab*t3_aabaab(d,a,c,i,l,j)
    triples_res +=  1.000000000000000 * einsum('lbdk,dacilj->abcijk', g_abab[o, v, v, o], t3_aabaab)
    
    #	  1.0000 <l,b||d,k>_bbbb*t3_abbabb(a,d,c,i,j,l)
    triples_res +=  1.000000000000000 * einsum('lbdk,adcijl->abcijk', g_bbbb[o, v, v, o], t3_abbabb)
    
    #	 -1.0000 <l,b||d,j>_abab*t3_aabaab(d,a,c,i,l,k)
    triples_res += -1.000000000000000 * einsum('lbdj,dacilk->abcijk', g_abab[o, v, v, o], t3_aabaab)
    
    #	 -1.0000 <l,b||d,j>_bbbb*t3_abbabb(a,d,c,i,k,l)
    triples_res += -1.000000000000000 * einsum('lbdj,adcikl->abcijk', g_bbbb[o, v, v, o], t3_abbabb)
    
    #	  1.0000 <l,b||i,d>_abab*t3_abbabb(a,d,c,l,k,j)
    triples_res +=  1.000000000000000 * einsum('lbid,adclkj->abcijk', g_abab[o, v, o, v], t3_abbabb)
    
    #	 -1.0000 <l,c||d,k>_abab*t3_aabaab(d,a,b,i,l,j)
    triples_res += -1.000000000000000 * einsum('lcdk,dabilj->abcijk', g_abab[o, v, v, o], t3_aabaab)
    
    #	 -1.0000 <l,c||d,k>_bbbb*t3_abbabb(a,d,b,i,j,l)
    triples_res += -1.000000000000000 * einsum('lcdk,adbijl->abcijk', g_bbbb[o, v, v, o], t3_abbabb)
    
    #	  1.0000 <l,c||d,j>_abab*t3_aabaab(d,a,b,i,l,k)
    triples_res +=  1.000000000000000 * einsum('lcdj,dabilk->abcijk', g_abab[o, v, v, o], t3_aabaab)
    
    #	  1.0000 <l,c||d,j>_bbbb*t3_abbabb(a,d,b,i,k,l)
    triples_res +=  1.000000000000000 * einsum('lcdj,adbikl->abcijk', g_bbbb[o, v, v, o], t3_abbabb)
    
    #	 -1.0000 <l,c||i,d>_abab*t3_abbabb(a,d,b,l,k,j)
    triples_res += -1.000000000000000 * einsum('lcid,adblkj->abcijk', g_abab[o, v, o, v], t3_abbabb)
    
    #	  0.5000 <a,b||d,e>_abab*t3_abbabb(d,e,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('abde,decijk->abcijk', g_abab[v, v, v, v], t3_abbabb)
    
    #	  0.5000 <a,b||e,d>_abab*t3_abbabb(e,d,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('abed,edcijk->abcijk', g_abab[v, v, v, v], t3_abbabb)
    
    #	 -0.5000 <a,c||d,e>_abab*t3_abbabb(d,e,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('acde,debijk->abcijk', g_abab[v, v, v, v], t3_abbabb)
    
    #	 -0.5000 <a,c||e,d>_abab*t3_abbabb(e,d,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('aced,edbijk->abcijk', g_abab[v, v, v, v], t3_abbabb)
    
    #	 -0.5000 <b,c||d,e>_bbbb*t3_abbabb(a,e,d,i,j,k)
    triples_res += -0.500000000000000 * einsum('bcde,aedijk->abcijk', g_bbbb[v, v, v, v], t3_abbabb)
    
    #	 -1.0000 <m,l||j,k>_bbbb*t1_bb(b,l)*t2_abab(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mljk,bl,acim->abcijk', g_bbbb[o, o, o, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||j,k>_bbbb*t1_bb(c,l)*t2_abab(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mljk,cl,abim->abcijk', g_bbbb[o, o, o, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||i,k>_abab*t1_aa(a,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmik,al,bcjm->abcijk', g_abab[o, o, o, o], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,k>_abab*t1_bb(b,l)*t2_abab(a,c,m,j)
    triples_res += -1.000000000000000 * einsum('mlik,bl,acmj->abcijk', g_abab[o, o, o, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,k>_abab*t1_bb(c,l)*t2_abab(a,b,m,j)
    triples_res +=  1.000000000000000 * einsum('mlik,cl,abmj->abcijk', g_abab[o, o, o, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||i,j>_abab*t1_aa(a,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmij,al,bckm->abcijk', g_abab[o, o, o, o], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,j>_abab*t1_bb(b,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlij,bl,acmk->abcijk', g_abab[o, o, o, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,j>_abab*t1_bb(c,l)*t2_abab(a,b,m,k)
    triples_res += -1.000000000000000 * einsum('mlij,cl,abmk->abcijk', g_abab[o, o, o, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||d,k>_abab*t1_aa(d,i)*t2_bbbb(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('aldk,di,bcjl->abcijk', g_abab[v, o, v, o], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||d,k>_abab*t1_bb(b,l)*t2_abab(d,c,i,j)
    triples_res +=  1.000000000000000 * einsum('aldk,bl,dcij->abcijk', g_abab[v, o, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||d,k>_abab*t1_bb(c,l)*t2_abab(d,b,i,j)
    triples_res += -1.000000000000000 * einsum('aldk,cl,dbij->abcijk', g_abab[v, o, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||d,j>_abab*t1_aa(d,i)*t2_bbbb(b,c,k,l)
    triples_res +=  1.000000000000000 * einsum('aldj,di,bckl->abcijk', g_abab[v, o, v, o], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||d,j>_abab*t1_bb(b,l)*t2_abab(d,c,i,k)
    triples_res += -1.000000000000000 * einsum('aldj,bl,dcik->abcijk', g_abab[v, o, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||d,j>_abab*t1_bb(c,l)*t2_abab(d,b,i,k)
    triples_res +=  1.000000000000000 * einsum('aldj,cl,dbik->abcijk', g_abab[v, o, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||i,d>_abab*t1_bb(d,k)*t2_bbbb(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('alid,dk,bcjl->abcijk', g_abab[v, o, o, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||i,d>_abab*t1_bb(d,j)*t2_bbbb(b,c,k,l)
    triples_res +=  1.000000000000000 * einsum('alid,dj,bckl->abcijk', g_abab[v, o, o, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||i,d>_abab*t1_bb(b,l)*t2_bbbb(d,c,j,k)
    triples_res += -1.000000000000000 * einsum('alid,bl,dcjk->abcijk', g_abab[v, o, o, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||i,d>_abab*t1_bb(c,l)*t2_bbbb(d,b,j,k)
    triples_res +=  1.000000000000000 * einsum('alid,cl,dbjk->abcijk', g_abab[v, o, o, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,k>_bbbb*t1_bb(d,j)*t2_abab(a,c,i,l)
    triples_res +=  1.000000000000000 * einsum('lbdk,dj,acil->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,k>_abab*t1_aa(d,i)*t2_abab(a,c,l,j)
    triples_res +=  1.000000000000000 * einsum('lbdk,di,aclj->abcijk', g_abab[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,k>_abab*t1_aa(a,l)*t2_abab(d,c,i,j)
    triples_res +=  1.000000000000000 * einsum('lbdk,al,dcij->abcijk', g_abab[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,k>_bbbb*t1_bb(c,l)*t2_abab(a,d,i,j)
    triples_res +=  1.000000000000000 * einsum('lbdk,cl,adij->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,j>_bbbb*t1_bb(d,k)*t2_abab(a,c,i,l)
    triples_res += -1.000000000000000 * einsum('lbdj,dk,acil->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,j>_abab*t1_aa(d,i)*t2_abab(a,c,l,k)
    triples_res += -1.000000000000000 * einsum('lbdj,di,aclk->abcijk', g_abab[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,j>_abab*t1_aa(a,l)*t2_abab(d,c,i,k)
    triples_res += -1.000000000000000 * einsum('lbdj,al,dcik->abcijk', g_abab[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,j>_bbbb*t1_bb(c,l)*t2_abab(a,d,i,k)
    triples_res += -1.000000000000000 * einsum('lbdj,cl,adik->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||i,d>_abab*t1_bb(d,k)*t2_abab(a,c,l,j)
    triples_res +=  1.000000000000000 * einsum('lbid,dk,aclj->abcijk', g_abab[o, v, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||i,d>_abab*t1_bb(d,j)*t2_abab(a,c,l,k)
    triples_res += -1.000000000000000 * einsum('lbid,dj,aclk->abcijk', g_abab[o, v, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||i,d>_abab*t1_aa(a,l)*t2_bbbb(d,c,j,k)
    triples_res += -1.000000000000000 * einsum('lbid,al,dcjk->abcijk', g_abab[o, v, o, v], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,k>_bbbb*t1_bb(d,j)*t2_abab(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('lcdk,dj,abil->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,k>_abab*t1_aa(d,i)*t2_abab(a,b,l,j)
    triples_res += -1.000000000000000 * einsum('lcdk,di,ablj->abcijk', g_abab[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,k>_abab*t1_aa(a,l)*t2_abab(d,b,i,j)
    triples_res += -1.000000000000000 * einsum('lcdk,al,dbij->abcijk', g_abab[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,k>_bbbb*t1_bb(b,l)*t2_abab(a,d,i,j)
    triples_res += -1.000000000000000 * einsum('lcdk,bl,adij->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,j>_bbbb*t1_bb(d,k)*t2_abab(a,b,i,l)
    triples_res +=  1.000000000000000 * einsum('lcdj,dk,abil->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,j>_abab*t1_aa(d,i)*t2_abab(a,b,l,k)
    triples_res +=  1.000000000000000 * einsum('lcdj,di,ablk->abcijk', g_abab[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,j>_abab*t1_aa(a,l)*t2_abab(d,b,i,k)
    triples_res +=  1.000000000000000 * einsum('lcdj,al,dbik->abcijk', g_abab[o, v, v, o], t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,j>_bbbb*t1_bb(b,l)*t2_abab(a,d,i,k)
    triples_res +=  1.000000000000000 * einsum('lcdj,bl,adik->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||i,d>_abab*t1_bb(d,k)*t2_abab(a,b,l,j)
    triples_res += -1.000000000000000 * einsum('lcid,dk,ablj->abcijk', g_abab[o, v, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||i,d>_abab*t1_bb(d,j)*t2_abab(a,b,l,k)
    triples_res +=  1.000000000000000 * einsum('lcid,dj,ablk->abcijk', g_abab[o, v, o, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||i,d>_abab*t1_aa(a,l)*t2_bbbb(d,b,j,k)
    triples_res +=  1.000000000000000 * einsum('lcid,al,dbjk->abcijk', g_abab[o, v, o, v], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,b||e,d>_abab*t1_bb(d,k)*t2_abab(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('abed,dk,ecij->abcijk', g_abab[v, v, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,b||e,d>_abab*t1_bb(d,j)*t2_abab(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('abed,dj,ecik->abcijk', g_abab[v, v, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,b||d,e>_abab*t1_aa(d,i)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('abde,di,ecjk->abcijk', g_abab[v, v, v, v], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,c||e,d>_abab*t1_bb(d,k)*t2_abab(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('aced,dk,ebij->abcijk', g_abab[v, v, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,c||e,d>_abab*t1_bb(d,j)*t2_abab(e,b,i,k)
    triples_res += -1.000000000000000 * einsum('aced,dj,ebik->abcijk', g_abab[v, v, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,c||d,e>_abab*t1_aa(d,i)*t2_bbbb(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('acde,di,ebjk->abcijk', g_abab[v, v, v, v], t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,c||d,e>_bbbb*t1_bb(d,k)*t2_abab(a,e,i,j)
    triples_res += -1.000000000000000 * einsum('bcde,dk,aeij->abcijk', g_bbbb[v, v, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,c||d,e>_bbbb*t1_bb(d,j)*t2_abab(a,e,i,k)
    triples_res +=  1.000000000000000 * einsum('bcde,dj,aeik->abcijk', g_bbbb[v, v, v, v], t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t1_aa(d,l)*t3_abbabb(a,b,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('lmdk,dl,abcijm->abcijk', g_abab[o, o, v, o], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t1_bb(d,l)*t3_abbabb(a,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dl,abcijm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,k>_bbbb*t1_bb(d,j)*t3_abbabb(a,b,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mldk,dj,abciml->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,k>_abab*t1_aa(d,i)*t3_abbabb(a,b,c,m,j,l)
    triples_res +=  0.500000000000000 * einsum('mldk,di,abcmjl->abcijk', g_abab[o, o, v, o], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||d,k>_abab*t1_aa(d,i)*t3_abbabb(a,b,c,l,m,j)
    triples_res += -0.500000000000000 * einsum('lmdk,di,abclmj->abcijk', g_abab[o, o, v, o], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t1_aa(a,l)*t3_abbabb(d,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,al,dbcijm->abcijk', g_abab[o, o, v, o], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_abab*t1_bb(b,l)*t3_aabaab(d,a,c,i,m,j)
    triples_res += -1.000000000000000 * einsum('mldk,bl,dacimj->abcijk', g_abab[o, o, v, o], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t1_bb(b,l)*t3_abbabb(a,d,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,bl,adcijm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_abab*t1_bb(c,l)*t3_aabaab(d,a,b,i,m,j)
    triples_res +=  1.000000000000000 * einsum('mldk,cl,dabimj->abcijk', g_abab[o, o, v, o], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t1_bb(c,l)*t3_abbabb(a,d,b,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,cl,adbijm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,j>_abab*t1_aa(d,l)*t3_abbabb(a,b,c,i,k,m)
    triples_res +=  1.000000000000000 * einsum('lmdj,dl,abcikm->abcijk', g_abab[o, o, v, o], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t1_bb(d,l)*t3_abbabb(a,b,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('mldj,dl,abcikm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,j>_bbbb*t1_bb(d,k)*t3_abbabb(a,b,c,i,m,l)
    triples_res += -0.500000000000000 * einsum('mldj,dk,abciml->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,j>_abab*t1_aa(d,i)*t3_abbabb(a,b,c,m,k,l)
    triples_res += -0.500000000000000 * einsum('mldj,di,abcmkl->abcijk', g_abab[o, o, v, o], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,m||d,j>_abab*t1_aa(d,i)*t3_abbabb(a,b,c,l,m,k)
    triples_res +=  0.500000000000000 * einsum('lmdj,di,abclmk->abcijk', g_abab[o, o, v, o], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,j>_abab*t1_aa(a,l)*t3_abbabb(d,b,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('lmdj,al,dbcikm->abcijk', g_abab[o, o, v, o], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_abab*t1_bb(b,l)*t3_aabaab(d,a,c,i,m,k)
    triples_res +=  1.000000000000000 * einsum('mldj,bl,dacimk->abcijk', g_abab[o, o, v, o], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_bbbb*t1_bb(b,l)*t3_abbabb(a,d,c,i,k,m)
    triples_res +=  1.000000000000000 * einsum('mldj,bl,adcikm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_abab*t1_bb(c,l)*t3_aabaab(d,a,b,i,m,k)
    triples_res += -1.000000000000000 * einsum('mldj,cl,dabimk->abcijk', g_abab[o, o, v, o], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t1_bb(c,l)*t3_abbabb(a,d,b,i,k,m)
    triples_res += -1.000000000000000 * einsum('mldj,cl,adbikm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t1_aa(d,l)*t3_abbabb(a,b,c,m,k,j)
    triples_res += -1.000000000000000 * einsum('mldi,dl,abcmkj->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,d>_abab*t1_bb(d,l)*t3_abbabb(a,b,c,m,k,j)
    triples_res +=  1.000000000000000 * einsum('mlid,dl,abcmkj->abcijk', g_abab[o, o, o, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||i,d>_abab*t1_bb(d,k)*t3_abbabb(a,b,c,m,j,l)
    triples_res +=  0.500000000000000 * einsum('mlid,dk,abcmjl->abcijk', g_abab[o, o, o, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||i,d>_abab*t1_bb(d,k)*t3_abbabb(a,b,c,l,m,j)
    triples_res += -0.500000000000000 * einsum('lmid,dk,abclmj->abcijk', g_abab[o, o, o, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||i,d>_abab*t1_bb(d,j)*t3_abbabb(a,b,c,m,k,l)
    triples_res += -0.500000000000000 * einsum('mlid,dj,abcmkl->abcijk', g_abab[o, o, o, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,m||i,d>_abab*t1_bb(d,j)*t3_abbabb(a,b,c,l,m,k)
    triples_res +=  0.500000000000000 * einsum('lmid,dj,abclmk->abcijk', g_abab[o, o, o, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t1_aa(a,l)*t3_abbabb(d,b,c,m,k,j)
    triples_res +=  1.000000000000000 * einsum('mldi,al,dbcmkj->abcijk', g_aaaa[o, o, v, o], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||i,d>_abab*t1_aa(a,l)*t3_bbbbbb(d,b,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('lmid,al,dbcjkm->abcijk', g_abab[o, o, o, v], t1_aa, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,d>_abab*t1_bb(b,l)*t3_abbabb(a,d,c,m,k,j)
    triples_res += -1.000000000000000 * einsum('mlid,bl,adcmkj->abcijk', g_abab[o, o, o, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,d>_abab*t1_bb(c,l)*t3_abbabb(a,d,b,m,k,j)
    triples_res +=  1.000000000000000 * einsum('mlid,cl,adbmkj->abcijk', g_abab[o, o, o, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t1_aa(d,l)*t3_abbabb(e,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lade,dl,ebcijk->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||e,d>_abab*t1_bb(d,l)*t3_abbabb(e,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('aled,dl,ebcijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||e,d>_abab*t1_bb(d,k)*t3_abbabb(e,b,c,i,j,l)
    triples_res += -1.000000000000000 * einsum('aled,dk,ebcijl->abcijk', g_abab[v, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||e,d>_abab*t1_bb(d,j)*t3_abbabb(e,b,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('aled,dj,ebcikl->abcijk', g_abab[v, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t1_aa(d,i)*t3_abbabb(e,b,c,l,k,j)
    triples_res +=  1.000000000000000 * einsum('lade,di,ebclkj->abcijk', g_aaaa[o, v, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||d,e>_abab*t1_aa(d,i)*t3_bbbbbb(e,b,c,j,k,l)
    triples_res +=  1.000000000000000 * einsum('alde,di,ebcjkl->abcijk', g_abab[v, o, v, v], t1_aa, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <a,l||d,e>_abab*t1_bb(b,l)*t3_abbabb(d,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('alde,bl,decijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <a,l||e,d>_abab*t1_bb(b,l)*t3_abbabb(e,d,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('aled,bl,edcijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <a,l||d,e>_abab*t1_bb(c,l)*t3_abbabb(d,e,b,i,j,k)
    triples_res +=  0.500000000000000 * einsum('alde,cl,debijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <a,l||e,d>_abab*t1_bb(c,l)*t3_abbabb(e,d,b,i,j,k)
    triples_res +=  0.500000000000000 * einsum('aled,cl,edbijk->abcijk', g_abab[v, o, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_abab*t1_aa(d,l)*t3_abbabb(a,e,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lbde,dl,aecijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_bbbb*t1_bb(d,l)*t3_abbabb(a,e,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lbde,dl,aecijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||e,d>_abab*t1_bb(d,k)*t3_aabaab(e,a,c,i,l,j)
    triples_res +=  1.000000000000000 * einsum('lbed,dk,eacilj->abcijk', g_abab[o, v, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t1_bb(d,k)*t3_abbabb(a,e,c,i,j,l)
    triples_res += -1.000000000000000 * einsum('lbde,dk,aecijl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||e,d>_abab*t1_bb(d,j)*t3_aabaab(e,a,c,i,l,k)
    triples_res += -1.000000000000000 * einsum('lbed,dj,eacilk->abcijk', g_abab[o, v, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_bbbb*t1_bb(d,j)*t3_abbabb(a,e,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('lbde,dj,aecikl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_abab*t1_aa(d,i)*t3_abbabb(a,e,c,l,k,j)
    triples_res +=  1.000000000000000 * einsum('lbde,di,aeclkj->abcijk', g_abab[o, v, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,b||d,e>_abab*t1_aa(a,l)*t3_abbabb(d,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lbde,al,decijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,b||e,d>_abab*t1_aa(a,l)*t3_abbabb(e,d,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lbed,al,edcijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,b||d,e>_bbbb*t1_bb(c,l)*t3_abbabb(a,e,d,i,j,k)
    triples_res += -0.500000000000000 * einsum('lbde,cl,aedijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_abab*t1_aa(d,l)*t3_abbabb(a,e,b,i,j,k)
    triples_res += -1.000000000000000 * einsum('lcde,dl,aebijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t1_bb(d,l)*t3_abbabb(a,e,b,i,j,k)
    triples_res += -1.000000000000000 * einsum('lcde,dl,aebijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||e,d>_abab*t1_bb(d,k)*t3_aabaab(e,a,b,i,l,j)
    triples_res += -1.000000000000000 * einsum('lced,dk,eabilj->abcijk', g_abab[o, v, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t1_bb(d,k)*t3_abbabb(a,e,b,i,j,l)
    triples_res +=  1.000000000000000 * einsum('lcde,dk,aebijl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||e,d>_abab*t1_bb(d,j)*t3_aabaab(e,a,b,i,l,k)
    triples_res +=  1.000000000000000 * einsum('lced,dj,eabilk->abcijk', g_abab[o, v, v, v], t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t1_bb(d,j)*t3_abbabb(a,e,b,i,k,l)
    triples_res += -1.000000000000000 * einsum('lcde,dj,aebikl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_abab*t1_aa(d,i)*t3_abbabb(a,e,b,l,k,j)
    triples_res += -1.000000000000000 * einsum('lcde,di,aeblkj->abcijk', g_abab[o, v, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,c||d,e>_abab*t1_aa(a,l)*t3_abbabb(d,e,b,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lcde,al,debijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,c||e,d>_abab*t1_aa(a,l)*t3_abbabb(e,d,b,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lced,al,edbijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,c||d,e>_bbbb*t1_bb(b,l)*t3_abbabb(a,e,d,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lcde,bl,aedijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_abab(d,e,l,k)*t3_abbabb(a,b,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmde,delk,abcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(e,d,l,k)*t3_abbabb(a,b,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmed,edlk,abcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,e,k,l)*t3_abbabb(a,b,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,dekl,abcijm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_abab(d,e,l,j)*t3_abbabb(a,b,c,i,k,m)
    triples_res +=  0.500000000000000 * einsum('lmde,delj,abcikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_abab(e,d,l,j)*t3_abbabb(a,b,c,i,k,m)
    triples_res +=  0.500000000000000 * einsum('lmed,edlj,abcikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,e,j,l)*t3_abbabb(a,b,c,i,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,dejl,abcikm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,e,i,l)*t3_abbabb(a,b,c,m,k,j)
    triples_res +=  0.500000000000000 * einsum('mlde,deil,abcmkj->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_abab(d,e,i,l)*t3_abbabb(a,b,c,m,k,j)
    triples_res +=  0.500000000000000 * einsum('mlde,deil,abcmkj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_abab(e,d,i,l)*t3_abbabb(a,b,c,m,k,j)
    triples_res +=  0.500000000000000 * einsum('mled,edil,abcmkj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_bbbb*t2_bbbb(d,e,j,k)*t3_abbabb(a,b,c,i,m,l)
    triples_res +=  0.250000000000000 * einsum('mlde,dejk,abciml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_abab*t2_abab(d,e,i,k)*t3_abbabb(a,b,c,m,j,l)
    triples_res +=  0.250000000000000 * einsum('mlde,deik,abcmjl->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <l,m||d,e>_abab*t2_abab(d,e,i,k)*t3_abbabb(a,b,c,l,m,j)
    triples_res += -0.250000000000000 * einsum('lmde,deik,abclmj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,l||e,d>_abab*t2_abab(e,d,i,k)*t3_abbabb(a,b,c,m,j,l)
    triples_res +=  0.250000000000000 * einsum('mled,edik,abcmjl->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <l,m||e,d>_abab*t2_abab(e,d,i,k)*t3_abbabb(a,b,c,l,m,j)
    triples_res += -0.250000000000000 * einsum('lmed,edik,abclmj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,l||d,e>_abab*t2_abab(d,e,i,j)*t3_abbabb(a,b,c,m,k,l)
    triples_res += -0.250000000000000 * einsum('mlde,deij,abcmkl->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,m||d,e>_abab*t2_abab(d,e,i,j)*t3_abbabb(a,b,c,l,m,k)
    triples_res +=  0.250000000000000 * einsum('lmde,deij,abclmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,l||e,d>_abab*t2_abab(e,d,i,j)*t3_abbabb(a,b,c,m,k,l)
    triples_res += -0.250000000000000 * einsum('mled,edij,abcmkl->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,m||e,d>_abab*t2_abab(e,d,i,j)*t3_abbabb(a,b,c,l,m,k)
    triples_res +=  0.250000000000000 * einsum('lmed,edij,abclmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_aaaa(d,a,m,l)*t3_abbabb(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,daml,ebcijk->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_abab(a,d,m,l)*t3_abbabb(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mled,adml,ebcijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(a,d,l,m)*t3_abbabb(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lmed,adlm,ebcijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t2_abab(a,d,l,k)*t3_abbabb(e,b,c,i,j,m)
    triples_res +=  1.000000000000010 * einsum('lmed,adlk,ebcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t2_abab(a,d,l,j)*t3_abbabb(e,b,c,i,k,m)
    triples_res += -1.000000000000010 * einsum('lmed,adlj,ebcikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_aaaa(d,a,i,l)*t3_abbabb(e,b,c,m,k,j)
    triples_res += -1.000000000000010 * einsum('mlde,dail,ebcmkj->abcijk', g_aaaa[o, o, v, v], t2_aaaa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_aaaa(d,a,i,l)*t3_bbbbbb(e,b,c,j,k,m)
    triples_res += -1.000000000000010 * einsum('lmde,dail,ebcjkm->abcijk', g_abab[o, o, v, v], t2_aaaa, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_abab(a,d,i,l)*t3_abbabb(e,b,c,m,k,j)
    triples_res += -1.000000000000010 * einsum('mled,adil,ebcmkj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_abab(a,d,i,l)*t3_bbbbbb(e,b,c,j,k,m)
    triples_res += -1.000000000000010 * einsum('mlde,adil,ebcjkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_abab(a,d,i,k)*t3_abbabb(e,b,c,m,j,l)
    triples_res += -0.500000000000000 * einsum('mled,adik,ebcmjl->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_abab(a,d,i,k)*t3_abbabb(e,b,c,l,m,j)
    triples_res +=  0.500000000000000 * einsum('lmed,adik,ebclmj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_abab(a,d,i,k)*t3_bbbbbb(e,b,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,adik,ebcjml->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_abab(a,d,i,j)*t3_abbabb(e,b,c,m,k,l)
    triples_res +=  0.500000000000000 * einsum('mled,adij,ebcmkl->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(a,d,i,j)*t3_abbabb(e,b,c,l,m,k)
    triples_res += -0.500000000000000 * einsum('lmed,adij,ebclmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_abab(a,d,i,j)*t3_bbbbbb(e,b,c,k,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,adij,ebckml->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_abab(d,b,m,l)*t3_abbabb(a,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,dbml,aecijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_abab(d,b,l,m)*t3_abbabb(a,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lmde,dblm,aecijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,b,m,l)*t3_abbabb(a,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,dbml,aecijk->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_abab(d,b,l,k)*t3_aabaab(e,a,c,i,m,j)
    triples_res += -1.000000000000010 * einsum('mlde,dblk,eacimj->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_abab(d,b,l,k)*t3_abbabb(a,e,c,i,j,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dblk,aecijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_bbbb(d,b,k,l)*t3_aabaab(e,a,c,i,m,j)
    triples_res += -1.000000000000010 * einsum('mled,dbkl,eacimj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,b,k,l)*t3_abbabb(a,e,c,i,j,m)
    triples_res +=  1.000000000000010 * einsum('mlde,dbkl,aecijm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_abab(d,b,l,j)*t3_aabaab(e,a,c,i,m,k)
    triples_res +=  1.000000000000010 * einsum('mlde,dblj,eacimk->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_abab(d,b,l,j)*t3_abbabb(a,e,c,i,k,m)
    triples_res += -1.000000000000010 * einsum('lmde,dblj,aecikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_bbbb(d,b,j,l)*t3_aabaab(e,a,c,i,m,k)
    triples_res +=  1.000000000000010 * einsum('mled,dbjl,eacimk->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,b,j,l)*t3_abbabb(a,e,c,i,k,m)
    triples_res += -1.000000000000010 * einsum('mlde,dbjl,aecikm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t2_abab(d,b,i,l)*t3_abbabb(a,e,c,m,k,j)
    triples_res += -1.000000000000010 * einsum('mlde,dbil,aecmkj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_bbbb(d,b,j,k)*t3_aabaab(e,a,c,i,m,l)
    triples_res += -0.500000000000000 * einsum('mled,dbjk,eaciml->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_bbbb(d,b,j,k)*t3_aabaab(e,a,c,i,l,m)
    triples_res += -0.500000000000000 * einsum('lmed,dbjk,eacilm->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,b,j,k)*t3_abbabb(a,e,c,i,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dbjk,aeciml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_abab(d,b,i,k)*t3_aabaab(e,a,c,l,m,j)
    triples_res +=  0.500000000000000 * einsum('mlde,dbik,eaclmj->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_abab(d,b,i,k)*t3_abbabb(a,e,c,m,j,l)
    triples_res += -0.500000000000000 * einsum('mlde,dbik,aecmjl->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_abab(d,b,i,k)*t3_abbabb(a,e,c,l,m,j)
    triples_res +=  0.500000000000000 * einsum('lmde,dbik,aeclmj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_abab(d,b,i,j)*t3_aabaab(e,a,c,l,m,k)
    triples_res += -0.500000000000000 * einsum('mlde,dbij,eaclmk->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_abab(d,b,i,j)*t3_abbabb(a,e,c,m,k,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dbij,aecmkl->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_abab(d,b,i,j)*t3_abbabb(a,e,c,l,m,k)
    triples_res += -0.500000000000000 * einsum('lmde,dbij,aeclmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_abab(d,c,m,l)*t3_abbabb(a,e,b,i,j,k)
    triples_res +=  0.500000000000000 * einsum('mlde,dcml,aebijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_abab(d,c,l,m)*t3_abbabb(a,e,b,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lmde,dclm,aebijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,c,m,l)*t3_abbabb(a,e,b,i,j,k)
    triples_res +=  0.500000000000000 * einsum('mlde,dcml,aebijk->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_abab(d,c,l,k)*t3_aabaab(e,a,b,i,m,j)
    triples_res +=  1.000000000000010 * einsum('mlde,dclk,eabimj->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_abab(d,c,l,k)*t3_abbabb(a,e,b,i,j,m)
    triples_res += -1.000000000000010 * einsum('lmde,dclk,aebijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_bbbb(d,c,k,l)*t3_aabaab(e,a,b,i,m,j)
    triples_res +=  1.000000000000010 * einsum('mled,dckl,eabimj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,c,k,l)*t3_abbabb(a,e,b,i,j,m)
    triples_res += -1.000000000000010 * einsum('mlde,dckl,aebijm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_abab(d,c,l,j)*t3_aabaab(e,a,b,i,m,k)
    triples_res += -1.000000000000010 * einsum('mlde,dclj,eabimk->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_abab(d,c,l,j)*t3_abbabb(a,e,b,i,k,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dclj,aebikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_bbbb(d,c,j,l)*t3_aabaab(e,a,b,i,m,k)
    triples_res += -1.000000000000010 * einsum('mled,dcjl,eabimk->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,c,j,l)*t3_abbabb(a,e,b,i,k,m)
    triples_res +=  1.000000000000010 * einsum('mlde,dcjl,aebikm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t2_abab(d,c,i,l)*t3_abbabb(a,e,b,m,k,j)
    triples_res +=  1.000000000000010 * einsum('mlde,dcil,aebmkj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_bbbb(d,c,j,k)*t3_aabaab(e,a,b,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mled,dcjk,eabiml->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_bbbb(d,c,j,k)*t3_aabaab(e,a,b,i,l,m)
    triples_res +=  0.500000000000000 * einsum('lmed,dcjk,eabilm->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,c,j,k)*t3_abbabb(a,e,b,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dcjk,aebiml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_abab(d,c,i,k)*t3_aabaab(e,a,b,l,m,j)
    triples_res += -0.500000000000000 * einsum('mlde,dcik,eablmj->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_abab(d,c,i,k)*t3_abbabb(a,e,b,m,j,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dcik,aebmjl->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_abab(d,c,i,k)*t3_abbabb(a,e,b,l,m,j)
    triples_res += -0.500000000000000 * einsum('lmde,dcik,aeblmj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_abab(d,c,i,j)*t3_aabaab(e,a,b,l,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,dcij,eablmk->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_abab(d,c,i,j)*t3_abbabb(a,e,b,m,k,l)
    triples_res += -0.500000000000000 * einsum('mlde,dcij,aebmkl->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_abab(d,c,i,j)*t3_abbabb(a,e,b,l,m,k)
    triples_res +=  0.500000000000000 * einsum('lmde,dcij,aeblmk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_abab*t2_abab(a,b,m,l)*t3_abbabb(d,e,c,i,j,k)
    triples_res +=  0.250000000000000 * einsum('mlde,abml,decijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,l||e,d>_abab*t2_abab(a,b,m,l)*t3_abbabb(e,d,c,i,j,k)
    triples_res +=  0.250000000000000 * einsum('mled,abml,edcijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,m||d,e>_abab*t2_abab(a,b,l,m)*t3_abbabb(d,e,c,i,j,k)
    triples_res +=  0.250000000000000 * einsum('lmde,ablm,decijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <l,m||e,d>_abab*t2_abab(a,b,l,m)*t3_abbabb(e,d,c,i,j,k)
    triples_res +=  0.250000000000000 * einsum('lmed,ablm,edcijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_abab(a,b,l,k)*t3_aabaab(d,e,c,i,m,j)
    triples_res += -0.500000000000000 * einsum('mlde,ablk,decimj->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_abab(a,b,l,k)*t3_abbabb(d,e,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmde,ablk,decijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(a,b,l,k)*t3_abbabb(e,d,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmed,ablk,edcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_abab(a,b,l,j)*t3_aabaab(d,e,c,i,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,ablj,decimk->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_abab(a,b,l,j)*t3_abbabb(d,e,c,i,k,m)
    triples_res +=  0.500000000000000 * einsum('lmde,ablj,decikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_abab(a,b,l,j)*t3_abbabb(e,d,c,i,k,m)
    triples_res +=  0.500000000000000 * einsum('lmed,ablj,edcikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_abab(a,b,i,l)*t3_abbabb(d,e,c,m,k,j)
    triples_res +=  0.500000000000000 * einsum('mlde,abil,decmkj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_abab(a,b,i,l)*t3_abbabb(e,d,c,m,k,j)
    triples_res +=  0.500000000000000 * einsum('mled,abil,edcmkj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_abab(a,b,i,l)*t3_bbbbbb(d,e,c,j,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,abil,decjkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <m,l||d,e>_abab*t2_abab(a,c,m,l)*t3_abbabb(d,e,b,i,j,k)
    triples_res += -0.250000000000000 * einsum('mlde,acml,debijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,l||e,d>_abab*t2_abab(a,c,m,l)*t3_abbabb(e,d,b,i,j,k)
    triples_res += -0.250000000000000 * einsum('mled,acml,edbijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <l,m||d,e>_abab*t2_abab(a,c,l,m)*t3_abbabb(d,e,b,i,j,k)
    triples_res += -0.250000000000000 * einsum('lmde,aclm,debijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <l,m||e,d>_abab*t2_abab(a,c,l,m)*t3_abbabb(e,d,b,i,j,k)
    triples_res += -0.250000000000000 * einsum('lmed,aclm,edbijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_aaaa*t2_abab(a,c,l,k)*t3_aabaab(d,e,b,i,m,j)
    triples_res +=  0.500000000000000 * einsum('mlde,aclk,debimj->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_abab(a,c,l,k)*t3_abbabb(d,e,b,i,j,m)
    triples_res +=  0.500000000000000 * einsum('lmde,aclk,debijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_abab(a,c,l,k)*t3_abbabb(e,d,b,i,j,m)
    triples_res +=  0.500000000000000 * einsum('lmed,aclk,edbijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_aaaa*t2_abab(a,c,l,j)*t3_aabaab(d,e,b,i,m,k)
    triples_res += -0.500000000000000 * einsum('mlde,aclj,debimk->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_abab(a,c,l,j)*t3_abbabb(d,e,b,i,k,m)
    triples_res += -0.500000000000000 * einsum('lmde,aclj,debikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(a,c,l,j)*t3_abbabb(e,d,b,i,k,m)
    triples_res += -0.500000000000000 * einsum('lmed,aclj,edbikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_abab(a,c,i,l)*t3_abbabb(d,e,b,m,k,j)
    triples_res += -0.500000000000000 * einsum('mlde,acil,debmkj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_abab(a,c,i,l)*t3_abbabb(e,d,b,m,k,j)
    triples_res += -0.500000000000000 * einsum('mled,acil,edbmkj->abcijk', g_abab[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_abab(a,c,i,l)*t3_bbbbbb(d,e,b,j,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,acil,debjkm->abcijk', g_bbbb[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <m,l||d,e>_bbbb*t2_bbbb(b,c,m,l)*t3_abbabb(a,e,d,i,j,k)
    triples_res += -0.250000000000000 * einsum('mlde,bcml,aedijk->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_bbbb(b,c,k,l)*t3_aabaab(d,a,e,i,m,j)
    triples_res += -0.500000000000000 * einsum('mlde,bckl,daeimj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_bbbb(b,c,k,l)*t3_aabaab(a,e,d,i,m,j)
    triples_res +=  0.500000000000000 * einsum('mled,bckl,aedimj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(b,c,k,l)*t3_abbabb(a,e,d,i,j,m)
    triples_res +=  0.500000000000000 * einsum('mlde,bckl,aedijm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_bbbb(b,c,j,l)*t3_aabaab(d,a,e,i,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,bcjl,daeimk->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_bbbb(b,c,j,l)*t3_aabaab(a,e,d,i,m,k)
    triples_res += -0.500000000000000 * einsum('mled,bcjl,aedimk->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_aabaab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(b,c,j,l)*t3_abbabb(a,e,d,i,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,bcjl,aedikm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t2_aaaa(d,a,i,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,dail,bcjm->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t2_abab(a,d,i,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,adil,bcjm->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,k>_bbbb*t2_abab(a,d,i,j)*t2_bbbb(b,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mldk,adij,bcml->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t2_abab(d,b,l,j)*t2_abab(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmdk,dblj,acim->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t2_bbbb(d,b,j,l)*t2_abab(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mldk,dbjl,acim->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_abab*t2_abab(d,b,i,l)*t2_abab(a,c,m,j)
    triples_res += -1.000000000000000 * einsum('mldk,dbil,acmj->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,k>_abab*t2_abab(d,b,i,j)*t2_abab(a,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mldk,dbij,acml->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,k>_abab*t2_abab(d,b,i,j)*t2_abab(a,c,l,m)
    triples_res +=  0.500000000000000 * einsum('lmdk,dbij,aclm->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t2_abab(d,c,l,j)*t2_abab(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,dclj,abim->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t2_bbbb(d,c,j,l)*t2_abab(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dcjl,abim->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_abab*t2_abab(d,c,i,l)*t2_abab(a,b,m,j)
    triples_res +=  1.000000000000000 * einsum('mldk,dcil,abmj->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,k>_abab*t2_abab(d,c,i,j)*t2_abab(a,b,m,l)
    triples_res += -0.500000000000000 * einsum('mldk,dcij,abml->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,k>_abab*t2_abab(d,c,i,j)*t2_abab(a,b,l,m)
    triples_res += -0.500000000000000 * einsum('lmdk,dcij,ablm->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,j>_abab*t2_aaaa(d,a,i,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmdj,dail,bckm->abcijk', g_abab[o, o, v, o], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t2_abab(a,d,i,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mldj,adil,bckm->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,j>_bbbb*t2_abab(a,d,i,k)*t2_bbbb(b,c,m,l)
    triples_res += -0.500000000000000 * einsum('mldj,adik,bcml->abcijk', g_bbbb[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,j>_abab*t2_abab(d,b,l,k)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmdj,dblk,acim->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_bbbb*t2_bbbb(d,b,k,l)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mldj,dbkl,acim->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_abab*t2_abab(d,b,i,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mldj,dbil,acmk->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,j>_abab*t2_abab(d,b,i,k)*t2_abab(a,c,m,l)
    triples_res += -0.500000000000000 * einsum('mldj,dbik,acml->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,j>_abab*t2_abab(d,b,i,k)*t2_abab(a,c,l,m)
    triples_res += -0.500000000000000 * einsum('lmdj,dbik,aclm->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,j>_abab*t2_abab(d,c,l,k)*t2_abab(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('lmdj,dclk,abim->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t2_bbbb(d,c,k,l)*t2_abab(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mldj,dckl,abim->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_abab*t2_abab(d,c,i,l)*t2_abab(a,b,m,k)
    triples_res += -1.000000000000000 * einsum('mldj,dcil,abmk->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,j>_abab*t2_abab(d,c,i,k)*t2_abab(a,b,m,l)
    triples_res +=  0.500000000000000 * einsum('mldj,dcik,abml->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,j>_abab*t2_abab(d,c,i,k)*t2_abab(a,b,l,m)
    triples_res +=  0.500000000000000 * einsum('lmdj,dcik,ablm->abcijk', g_abab[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||i,d>_abab*t2_abab(a,d,l,k)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmid,adlk,bcjm->abcijk', g_abab[o, o, o, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||i,d>_abab*t2_abab(a,d,l,j)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmid,adlj,bckm->abcijk', g_abab[o, o, o, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t2_abab(d,b,l,k)*t2_abab(a,c,m,j)
    triples_res += -1.000000000000000 * einsum('mldi,dblk,acmj->abcijk', g_aaaa[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,d>_abab*t2_bbbb(d,b,k,l)*t2_abab(a,c,m,j)
    triples_res += -1.000000000000000 * einsum('mlid,dbkl,acmj->abcijk', g_abab[o, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t2_abab(d,b,l,j)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mldi,dblj,acmk->abcijk', g_aaaa[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,d>_abab*t2_bbbb(d,b,j,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlid,dbjl,acmk->abcijk', g_abab[o, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||i,d>_abab*t2_bbbb(d,b,j,k)*t2_abab(a,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlid,dbjk,acml->abcijk', g_abab[o, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||i,d>_abab*t2_bbbb(d,b,j,k)*t2_abab(a,c,l,m)
    triples_res += -0.500000000000000 * einsum('lmid,dbjk,aclm->abcijk', g_abab[o, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_aaaa*t2_abab(d,c,l,k)*t2_abab(a,b,m,j)
    triples_res +=  1.000000000000000 * einsum('mldi,dclk,abmj->abcijk', g_aaaa[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,d>_abab*t2_bbbb(d,c,k,l)*t2_abab(a,b,m,j)
    triples_res +=  1.000000000000000 * einsum('mlid,dckl,abmj->abcijk', g_abab[o, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_aaaa*t2_abab(d,c,l,j)*t2_abab(a,b,m,k)
    triples_res += -1.000000000000000 * einsum('mldi,dclj,abmk->abcijk', g_aaaa[o, o, v, o], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,d>_abab*t2_bbbb(d,c,j,l)*t2_abab(a,b,m,k)
    triples_res += -1.000000000000000 * einsum('mlid,dcjl,abmk->abcijk', g_abab[o, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||i,d>_abab*t2_bbbb(d,c,j,k)*t2_abab(a,b,m,l)
    triples_res +=  0.500000000000000 * einsum('mlid,dcjk,abml->abcijk', g_abab[o, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||i,d>_abab*t2_bbbb(d,c,j,k)*t2_abab(a,b,l,m)
    triples_res +=  0.500000000000000 * einsum('lmid,dcjk,ablm->abcijk', g_abab[o, o, o, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <a,l||d,e>_abab*t2_abab(d,e,i,k)*t2_bbbb(b,c,j,l)
    triples_res += -0.500000000000000 * einsum('alde,deik,bcjl->abcijk', g_abab[v, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <a,l||e,d>_abab*t2_abab(e,d,i,k)*t2_bbbb(b,c,j,l)
    triples_res += -0.500000000000000 * einsum('aled,edik,bcjl->abcijk', g_abab[v, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <a,l||d,e>_abab*t2_abab(d,e,i,j)*t2_bbbb(b,c,k,l)
    triples_res +=  0.500000000000000 * einsum('alde,deij,bckl->abcijk', g_abab[v, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <a,l||e,d>_abab*t2_abab(e,d,i,j)*t2_bbbb(b,c,k,l)
    triples_res +=  0.500000000000000 * einsum('aled,edij,bckl->abcijk', g_abab[v, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t2_abab(d,b,l,k)*t2_abab(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('lade,dblk,ecij->abcijk', g_aaaa[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||e,d>_abab*t2_bbbb(d,b,k,l)*t2_abab(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('aled,dbkl,ecij->abcijk', g_abab[v, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t2_abab(d,b,l,j)*t2_abab(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('lade,dblj,ecik->abcijk', g_aaaa[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||e,d>_abab*t2_bbbb(d,b,j,l)*t2_abab(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('aled,dbjl,ecik->abcijk', g_abab[v, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,l||d,e>_abab*t2_abab(d,b,i,l)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('alde,dbil,ecjk->abcijk', g_abab[v, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,l||e,d>_abab*t2_bbbb(d,b,j,k)*t2_abab(e,c,i,l)
    triples_res +=  1.000000000000000 * einsum('aled,dbjk,ecil->abcijk', g_abab[v, o, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_aaaa*t2_abab(d,b,i,k)*t2_abab(e,c,l,j)
    triples_res +=  1.000000000000000 * einsum('lade,dbik,eclj->abcijk', g_aaaa[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <a,l||d,e>_abab*t2_abab(d,b,i,k)*t2_bbbb(e,c,j,l)
    triples_res +=  1.000000000000000 * einsum('alde,dbik,ecjl->abcijk', g_abab[v, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_aaaa*t2_abab(d,b,i,j)*t2_abab(e,c,l,k)
    triples_res += -1.000000000000000 * einsum('lade,dbij,eclk->abcijk', g_aaaa[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <a,l||d,e>_abab*t2_abab(d,b,i,j)*t2_bbbb(e,c,k,l)
    triples_res += -1.000000000000000 * einsum('alde,dbij,eckl->abcijk', g_abab[v, o, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,b||d,e>_bbbb*t2_bbbb(d,e,j,k)*t2_abab(a,c,i,l)
    triples_res +=  0.500000000000000 * einsum('lbde,dejk,acil->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,b||d,e>_abab*t2_abab(d,e,i,k)*t2_abab(a,c,l,j)
    triples_res +=  0.500000000000000 * einsum('lbde,deik,aclj->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,b||e,d>_abab*t2_abab(e,d,i,k)*t2_abab(a,c,l,j)
    triples_res +=  0.500000000000000 * einsum('lbed,edik,aclj->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,b||d,e>_abab*t2_abab(d,e,i,j)*t2_abab(a,c,l,k)
    triples_res += -0.500000000000000 * einsum('lbde,deij,aclk->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,b||e,d>_abab*t2_abab(e,d,i,j)*t2_abab(a,c,l,k)
    triples_res += -0.500000000000000 * einsum('lbed,edij,aclk->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||e,d>_abab*t2_abab(a,d,l,k)*t2_abab(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('lbed,adlk,ecij->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||e,d>_abab*t2_abab(a,d,l,j)*t2_abab(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('lbed,adlj,ecik->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_abab*t2_aaaa(d,a,i,l)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('lbde,dail,ecjk->abcijk', g_abab[o, v, v, v], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_bbbb*t2_abab(a,d,i,l)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lbde,adil,ecjk->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||e,d>_abab*t2_abab(a,d,i,k)*t2_abab(e,c,l,j)
    triples_res += -1.000000000000000 * einsum('lbed,adik,eclj->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t2_abab(a,d,i,k)*t2_bbbb(e,c,j,l)
    triples_res += -1.000000000000000 * einsum('lbde,adik,ecjl->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||e,d>_abab*t2_abab(a,d,i,j)*t2_abab(e,c,l,k)
    triples_res +=  1.000000000000000 * einsum('lbed,adij,eclk->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_bbbb*t2_abab(a,d,i,j)*t2_bbbb(e,c,k,l)
    triples_res +=  1.000000000000000 * einsum('lbde,adij,eckl->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,c||d,e>_bbbb*t2_bbbb(d,e,j,k)*t2_abab(a,b,i,l)
    triples_res += -0.500000000000000 * einsum('lcde,dejk,abil->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,c||d,e>_abab*t2_abab(d,e,i,k)*t2_abab(a,b,l,j)
    triples_res += -0.500000000000000 * einsum('lcde,deik,ablj->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,c||e,d>_abab*t2_abab(e,d,i,k)*t2_abab(a,b,l,j)
    triples_res += -0.500000000000000 * einsum('lced,edik,ablj->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,c||d,e>_abab*t2_abab(d,e,i,j)*t2_abab(a,b,l,k)
    triples_res +=  0.500000000000000 * einsum('lcde,deij,ablk->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,c||e,d>_abab*t2_abab(e,d,i,j)*t2_abab(a,b,l,k)
    triples_res +=  0.500000000000000 * einsum('lced,edij,ablk->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||e,d>_abab*t2_abab(a,d,l,k)*t2_abab(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('lced,adlk,ebij->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||e,d>_abab*t2_abab(a,d,l,j)*t2_abab(e,b,i,k)
    triples_res +=  1.000000000000000 * einsum('lced,adlj,ebik->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_abab*t2_aaaa(d,a,i,l)*t2_bbbb(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dail,ebjk->abcijk', g_abab[o, v, v, v], t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t2_abab(a,d,i,l)*t2_bbbb(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('lcde,adil,ebjk->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||e,d>_abab*t2_abab(a,d,i,k)*t2_abab(e,b,l,j)
    triples_res +=  1.000000000000000 * einsum('lced,adik,eblj->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t2_abab(a,d,i,k)*t2_bbbb(e,b,j,l)
    triples_res +=  1.000000000000000 * einsum('lcde,adik,ebjl->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||e,d>_abab*t2_abab(a,d,i,j)*t2_abab(e,b,l,k)
    triples_res += -1.000000000000000 * einsum('lced,adij,eblk->abcijk', g_abab[o, v, v, v], t2_abab, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t2_abab(a,d,i,j)*t2_bbbb(e,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcde,adij,ebkl->abcijk', g_bbbb[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_abab(a,e,i,k)*t2_bbbb(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmde,dl,aeik,bcjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_abab(a,e,i,k)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,aeik,bcjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_abab(a,e,i,j)*t2_bbbb(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dl,aeij,bckm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_abab(a,e,i,j)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,aeij,bckm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_bbbb(e,b,j,k)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dl,ebjk,acim->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_bbbb(e,b,j,k)*t2_abab(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ebjk,acim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_abab(e,b,i,k)*t2_abab(a,c,m,j)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ebik,acmj->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_abab(e,b,i,k)*t2_abab(a,c,m,j)
    triples_res +=  1.000000000000000 * einsum('mled,dl,ebik,acmj->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_abab(e,b,i,j)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ebij,acmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_abab(e,b,i,j)*t2_abab(a,c,m,k)
    triples_res += -1.000000000000000 * einsum('mled,dl,ebij,acmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_bbbb(e,c,j,k)*t2_abab(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('lmde,dl,ecjk,abim->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_bbbb(e,c,j,k)*t2_abab(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ecjk,abim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_abab(e,c,i,k)*t2_abab(a,b,m,j)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ecik,abmj->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_abab(e,c,i,k)*t2_abab(a,b,m,j)
    triples_res += -1.000000000000000 * einsum('mled,dl,ecik,abmj->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t2_abab(e,c,i,j)*t2_abab(a,b,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ecij,abmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t2_abab(e,c,i,j)*t2_abab(a,b,m,k)
    triples_res +=  1.000000000000000 * einsum('mled,dl,ecij,abmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t2_aaaa(e,a,i,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,eail,bcjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_abab(a,e,i,l)*t2_bbbb(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,aeil,bcjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_abab(a,e,i,j)*t2_bbbb(b,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dk,aeij,bcml->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t2_abab(e,b,l,j)*t2_abab(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmed,dk,eblj,acim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,b,j,l)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,ebjl,acim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t2_abab(e,b,i,l)*t2_abab(a,c,m,j)
    triples_res += -1.000000000000000 * einsum('mled,dk,ebil,acmj->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t1_bb(d,k)*t2_abab(e,b,i,j)*t2_abab(a,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mled,dk,ebij,acml->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t1_bb(d,k)*t2_abab(e,b,i,j)*t2_abab(a,c,l,m)
    triples_res +=  0.500000000000000 * einsum('lmed,dk,ebij,aclm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t2_abab(e,c,l,j)*t2_abab(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,eclj,abim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,c,j,l)*t2_abab(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,ecjl,abim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t2_abab(e,c,i,l)*t2_abab(a,b,m,j)
    triples_res +=  1.000000000000000 * einsum('mled,dk,ecil,abmj->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t1_bb(d,k)*t2_abab(e,c,i,j)*t2_abab(a,b,m,l)
    triples_res += -0.500000000000000 * einsum('mled,dk,ecij,abml->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t1_bb(d,k)*t2_abab(e,c,i,j)*t2_abab(a,b,l,m)
    triples_res += -0.500000000000000 * einsum('lmed,dk,ecij,ablm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,j)*t2_aaaa(e,a,i,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmed,dj,eail,bckm->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t2_abab(a,e,i,l)*t2_bbbb(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,aeil,bckm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(d,j)*t2_abab(a,e,i,k)*t2_bbbb(b,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dj,aeik,bcml->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,j)*t2_abab(e,b,l,k)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dj,eblk,acim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,b,k,l)*t2_abab(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,ebkl,acim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,j)*t2_abab(e,b,i,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mled,dj,ebil,acmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t1_bb(d,j)*t2_abab(e,b,i,k)*t2_abab(a,c,m,l)
    triples_res += -0.500000000000000 * einsum('mled,dj,ebik,acml->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t1_bb(d,j)*t2_abab(e,b,i,k)*t2_abab(a,c,l,m)
    triples_res += -0.500000000000000 * einsum('lmed,dj,ebik,aclm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,j)*t2_abab(e,c,l,k)*t2_abab(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('lmed,dj,eclk,abim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,c,k,l)*t2_abab(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,eckl,abim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,j)*t2_abab(e,c,i,l)*t2_abab(a,b,m,k)
    triples_res += -1.000000000000000 * einsum('mled,dj,ecil,abmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t1_bb(d,j)*t2_abab(e,c,i,k)*t2_abab(a,b,m,l)
    triples_res +=  0.500000000000000 * einsum('mled,dj,ecik,abml->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t1_bb(d,j)*t2_abab(e,c,i,k)*t2_abab(a,b,l,m)
    triples_res +=  0.500000000000000 * einsum('lmed,dj,ecik,ablm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t2_abab(a,e,l,k)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmde,di,aelk,bcjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t2_abab(a,e,l,j)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,di,aelj,bckm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_abab(e,b,l,k)*t2_abab(a,c,m,j)
    triples_res +=  1.000000000000000 * einsum('mlde,di,eblk,acmj->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t2_bbbb(e,b,k,l)*t2_abab(a,c,m,j)
    triples_res += -1.000000000000000 * einsum('mlde,di,ebkl,acmj->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_abab(e,b,l,j)*t2_abab(a,c,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,di,eblj,acmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t2_bbbb(e,b,j,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,di,ebjl,acmk->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t1_aa(d,i)*t2_bbbb(e,b,j,k)*t2_abab(a,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,di,ebjk,acml->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t1_aa(d,i)*t2_bbbb(e,b,j,k)*t2_abab(a,c,l,m)
    triples_res += -0.500000000000000 * einsum('lmde,di,ebjk,aclm->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_abab(e,c,l,k)*t2_abab(a,b,m,j)
    triples_res += -1.000000000000000 * einsum('mlde,di,eclk,abmj->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t2_bbbb(e,c,k,l)*t2_abab(a,b,m,j)
    triples_res +=  1.000000000000000 * einsum('mlde,di,eckl,abmj->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t2_abab(e,c,l,j)*t2_abab(a,b,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,di,eclj,abmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t2_bbbb(e,c,j,l)*t2_abab(a,b,m,k)
    triples_res += -1.000000000000000 * einsum('mlde,di,ecjl,abmk->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t1_aa(d,i)*t2_bbbb(e,c,j,k)*t2_abab(a,b,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,di,ecjk,abml->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t1_aa(d,i)*t2_bbbb(e,c,j,k)*t2_abab(a,b,l,m)
    triples_res +=  0.500000000000000 * einsum('lmde,di,ecjk,ablm->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t1_aa(a,l)*t2_abab(d,e,i,k)*t2_bbbb(b,c,j,m)
    triples_res +=  0.500000000000000 * einsum('lmde,al,deik,bcjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t1_aa(a,l)*t2_abab(e,d,i,k)*t2_bbbb(b,c,j,m)
    triples_res +=  0.500000000000000 * einsum('lmed,al,edik,bcjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t1_aa(a,l)*t2_abab(d,e,i,j)*t2_bbbb(b,c,k,m)
    triples_res += -0.500000000000000 * einsum('lmde,al,deij,bckm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t1_aa(a,l)*t2_abab(e,d,i,j)*t2_bbbb(b,c,k,m)
    triples_res += -0.500000000000000 * einsum('lmed,al,edij,bckm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_abab(d,b,m,k)*t2_abab(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('mlde,al,dbmk,ecij->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_bbbb(d,b,k,m)*t2_abab(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('lmed,al,dbkm,ecij->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_abab(d,b,m,j)*t2_abab(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('mlde,al,dbmj,ecik->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_bbbb(d,b,j,m)*t2_abab(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('lmed,al,dbjm,ecik->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(a,l)*t2_abab(d,b,i,m)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lmde,al,dbim,ecjk->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_aa(a,l)*t2_bbbb(d,b,j,k)*t2_abab(e,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmed,al,dbjk,ecim->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_abab(d,b,i,k)*t2_abab(e,c,m,j)
    triples_res += -1.000000000000000 * einsum('mlde,al,dbik,ecmj->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(a,l)*t2_abab(d,b,i,k)*t2_bbbb(e,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmde,al,dbik,ecjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(a,l)*t2_abab(d,b,i,j)*t2_abab(e,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlde,al,dbij,ecmk->abcijk', g_aaaa[o, o, v, v], t1_aa, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(a,l)*t2_abab(d,b,i,j)*t2_bbbb(e,c,k,m)
    triples_res +=  1.000000000000000 * einsum('lmde,al,dbij,eckm->abcijk', g_abab[o, o, v, v], t1_aa, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(b,l)*t2_bbbb(d,e,j,k)*t2_abab(a,c,i,m)
    triples_res += -0.500000000000000 * einsum('mlde,bl,dejk,acim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t1_bb(b,l)*t2_abab(d,e,i,k)*t2_abab(a,c,m,j)
    triples_res += -0.500000000000000 * einsum('mlde,bl,deik,acmj->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t1_bb(b,l)*t2_abab(e,d,i,k)*t2_abab(a,c,m,j)
    triples_res += -0.500000000000000 * einsum('mled,bl,edik,acmj->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t1_bb(b,l)*t2_abab(d,e,i,j)*t2_abab(a,c,m,k)
    triples_res +=  0.500000000000000 * einsum('mlde,bl,deij,acmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t1_bb(b,l)*t2_abab(e,d,i,j)*t2_abab(a,c,m,k)
    triples_res +=  0.500000000000000 * einsum('mled,bl,edij,acmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(b,l)*t2_abab(a,d,m,k)*t2_abab(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('mled,bl,admk,ecij->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(b,l)*t2_abab(a,d,m,j)*t2_abab(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('mled,bl,admj,ecik->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_bb(b,l)*t2_aaaa(d,a,i,m)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,bl,daim,ecjk->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(b,l)*t2_abab(a,d,i,m)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,bl,adim,ecjk->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(b,l)*t2_abab(a,d,i,k)*t2_abab(e,c,m,j)
    triples_res +=  1.000000000000000 * einsum('mled,bl,adik,ecmj->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(b,l)*t2_abab(a,d,i,k)*t2_bbbb(e,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,bl,adik,ecjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(b,l)*t2_abab(a,d,i,j)*t2_abab(e,c,m,k)
    triples_res += -1.000000000000000 * einsum('mled,bl,adij,ecmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(b,l)*t2_abab(a,d,i,j)*t2_bbbb(e,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,bl,adij,eckm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_bbbb(d,e,j,k)*t2_abab(a,b,i,m)
    triples_res +=  0.500000000000000 * einsum('mlde,cl,dejk,abim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t1_bb(c,l)*t2_abab(d,e,i,k)*t2_abab(a,b,m,j)
    triples_res +=  0.500000000000000 * einsum('mlde,cl,deik,abmj->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t1_bb(c,l)*t2_abab(e,d,i,k)*t2_abab(a,b,m,j)
    triples_res +=  0.500000000000000 * einsum('mled,cl,edik,abmj->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t1_bb(c,l)*t2_abab(d,e,i,j)*t2_abab(a,b,m,k)
    triples_res += -0.500000000000000 * einsum('mlde,cl,deij,abmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t1_bb(c,l)*t2_abab(e,d,i,j)*t2_abab(a,b,m,k)
    triples_res += -0.500000000000000 * einsum('mled,cl,edij,abmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(c,l)*t2_abab(a,d,m,k)*t2_abab(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('mled,cl,admk,ebij->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(c,l)*t2_abab(a,d,m,j)*t2_abab(e,b,i,k)
    triples_res += -1.000000000000000 * einsum('mled,cl,admj,ebik->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_bb(c,l)*t2_aaaa(d,a,i,m)*t2_bbbb(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,cl,daim,ebjk->abcijk', g_abab[o, o, v, v], t1_bb, t2_aaaa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_abab(a,d,i,m)*t2_bbbb(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,cl,adim,ebjk->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(c,l)*t2_abab(a,d,i,k)*t2_abab(e,b,m,j)
    triples_res += -1.000000000000000 * einsum('mled,cl,adik,ebmj->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_abab(a,d,i,k)*t2_bbbb(e,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,cl,adik,ebjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(c,l)*t2_abab(a,d,i,j)*t2_abab(e,b,m,k)
    triples_res +=  1.000000000000000 * einsum('mled,cl,adij,ebmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_abab(a,d,i,j)*t2_bbbb(e,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,cl,adij,ebkm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t1_bb(d,j)*t1_bb(b,l)*t2_abab(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mldk,dj,bl,acim->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t1_bb(d,j)*t1_bb(c,l)*t2_abab(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dj,cl,abim->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t1_aa(d,i)*t1_aa(a,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,di,al,bcjm->abcijk', g_abab[o, o, v, o], t1_aa, t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_abab*t1_aa(d,i)*t1_bb(b,l)*t2_abab(a,c,m,j)
    triples_res += -1.000000000000000 * einsum('mldk,di,bl,acmj->abcijk', g_abab[o, o, v, o], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_abab*t1_aa(d,i)*t1_bb(c,l)*t2_abab(a,b,m,j)
    triples_res +=  1.000000000000000 * einsum('mldk,di,cl,abmj->abcijk', g_abab[o, o, v, o], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t1_aa(a,l)*t1_bb(b,m)*t2_abab(d,c,i,j)
    triples_res += -1.000000000000000 * einsum('lmdk,al,bm,dcij->abcijk', g_abab[o, o, v, o], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t1_aa(a,l)*t1_bb(c,m)*t2_abab(d,b,i,j)
    triples_res +=  1.000000000000000 * einsum('lmdk,al,cm,dbij->abcijk', g_abab[o, o, v, o], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t1_bb(b,l)*t1_bb(c,m)*t2_abab(a,d,i,j)
    triples_res += -1.000000000000000 * einsum('mldk,bl,cm,adij->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_bbbb*t1_bb(d,k)*t1_bb(b,l)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mldj,dk,bl,acim->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t1_bb(d,k)*t1_bb(c,l)*t2_abab(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mldj,dk,cl,abim->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,j>_abab*t1_aa(d,i)*t1_aa(a,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmdj,di,al,bckm->abcijk', g_abab[o, o, v, o], t1_aa, t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_abab*t1_aa(d,i)*t1_bb(b,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mldj,di,bl,acmk->abcijk', g_abab[o, o, v, o], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_abab*t1_aa(d,i)*t1_bb(c,l)*t2_abab(a,b,m,k)
    triples_res += -1.000000000000000 * einsum('mldj,di,cl,abmk->abcijk', g_abab[o, o, v, o], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,j>_abab*t1_aa(a,l)*t1_bb(b,m)*t2_abab(d,c,i,k)
    triples_res +=  1.000000000000000 * einsum('lmdj,al,bm,dcik->abcijk', g_abab[o, o, v, o], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,j>_abab*t1_aa(a,l)*t1_bb(c,m)*t2_abab(d,b,i,k)
    triples_res += -1.000000000000000 * einsum('lmdj,al,cm,dbik->abcijk', g_abab[o, o, v, o], t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_bbbb*t1_bb(b,l)*t1_bb(c,m)*t2_abab(a,d,i,k)
    triples_res +=  1.000000000000000 * einsum('mldj,bl,cm,adik->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||i,d>_abab*t1_bb(d,k)*t1_aa(a,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmid,dk,al,bcjm->abcijk', g_abab[o, o, o, v], t1_bb, t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||i,d>_abab*t1_bb(d,k)*t1_bb(b,l)*t2_abab(a,c,m,j)
    triples_res += -1.000000000000000 * einsum('mlid,dk,bl,acmj->abcijk', g_abab[o, o, o, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||i,d>_abab*t1_bb(d,k)*t1_bb(c,l)*t2_abab(a,b,m,j)
    triples_res +=  1.000000000000000 * einsum('mlid,dk,cl,abmj->abcijk', g_abab[o, o, o, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||i,d>_abab*t1_bb(d,j)*t1_aa(a,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmid,dj,al,bckm->abcijk', g_abab[o, o, o, v], t1_bb, t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||i,d>_abab*t1_bb(d,j)*t1_bb(b,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mlid,dj,bl,acmk->abcijk', g_abab[o, o, o, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||i,d>_abab*t1_bb(d,j)*t1_bb(c,l)*t2_abab(a,b,m,k)
    triples_res += -1.000000000000000 * einsum('mlid,dj,cl,abmk->abcijk', g_abab[o, o, o, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||i,d>_abab*t1_aa(a,l)*t1_bb(b,m)*t2_bbbb(d,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lmid,al,bm,dcjk->abcijk', g_abab[o, o, o, v], t1_aa, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||i,d>_abab*t1_aa(a,l)*t1_bb(c,m)*t2_bbbb(d,b,j,k)
    triples_res += -1.000000000000000 * einsum('lmid,al,cm,dbjk->abcijk', g_abab[o, o, o, v], t1_aa, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,l||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t2_bbbb(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('aled,dk,ei,bcjl->abcijk', g_abab[v, o, v, v], t1_bb, t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <a,l||e,d>_abab*t1_bb(d,k)*t1_bb(b,l)*t2_abab(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('aled,dk,bl,ecij->abcijk', g_abab[v, o, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,l||e,d>_abab*t1_bb(d,k)*t1_bb(c,l)*t2_abab(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('aled,dk,cl,ebij->abcijk', g_abab[v, o, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <a,l||e,d>_abab*t1_bb(d,j)*t1_aa(e,i)*t2_bbbb(b,c,k,l)
    triples_res +=  1.000000000000000 * einsum('aled,dj,ei,bckl->abcijk', g_abab[v, o, v, v], t1_bb, t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,l||e,d>_abab*t1_bb(d,j)*t1_bb(b,l)*t2_abab(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('aled,dj,bl,ecik->abcijk', g_abab[v, o, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <a,l||e,d>_abab*t1_bb(d,j)*t1_bb(c,l)*t2_abab(e,b,i,k)
    triples_res +=  1.000000000000000 * einsum('aled,dj,cl,ebik->abcijk', g_abab[v, o, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <a,l||d,e>_abab*t1_aa(d,i)*t1_bb(b,l)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('alde,di,bl,ecjk->abcijk', g_abab[v, o, v, v], t1_aa, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <a,l||d,e>_abab*t1_aa(d,i)*t1_bb(c,l)*t2_bbbb(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('alde,di,cl,ebjk->abcijk', g_abab[v, o, v, v], t1_aa, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,j)*t2_abab(a,c,i,l)
    triples_res += -1.000000000000000 * einsum('lbde,dk,ej,acil->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t2_abab(a,c,l,j)
    triples_res +=  1.000000000000000 * einsum('lbed,dk,ei,aclj->abcijk', g_abab[o, v, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||e,d>_abab*t1_bb(d,k)*t1_aa(a,l)*t2_abab(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('lbed,dk,al,ecij->abcijk', g_abab[o, v, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t1_bb(d,k)*t1_bb(c,l)*t2_abab(a,e,i,j)
    triples_res += -1.000000000000000 * einsum('lbde,dk,cl,aeij->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||e,d>_abab*t1_bb(d,j)*t1_aa(e,i)*t2_abab(a,c,l,k)
    triples_res += -1.000000000000000 * einsum('lbed,dj,ei,aclk->abcijk', g_abab[o, v, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||e,d>_abab*t1_bb(d,j)*t1_aa(a,l)*t2_abab(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('lbed,dj,al,ecik->abcijk', g_abab[o, v, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_bbbb*t1_bb(d,j)*t1_bb(c,l)*t2_abab(a,e,i,k)
    triples_res +=  1.000000000000000 * einsum('lbde,dj,cl,aeik->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_abab*t1_aa(d,i)*t1_aa(a,l)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('lbde,di,al,ecjk->abcijk', g_abab[o, v, v, v], t1_aa, t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,j)*t2_abab(a,b,i,l)
    triples_res +=  1.000000000000000 * einsum('lcde,dk,ej,abil->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t2_abab(a,b,l,j)
    triples_res += -1.000000000000000 * einsum('lced,dk,ei,ablj->abcijk', g_abab[o, v, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||e,d>_abab*t1_bb(d,k)*t1_aa(a,l)*t2_abab(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('lced,dk,al,ebij->abcijk', g_abab[o, v, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t1_bb(d,k)*t1_bb(b,l)*t2_abab(a,e,i,j)
    triples_res +=  1.000000000000000 * einsum('lcde,dk,bl,aeij->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||e,d>_abab*t1_bb(d,j)*t1_aa(e,i)*t2_abab(a,b,l,k)
    triples_res +=  1.000000000000000 * einsum('lced,dj,ei,ablk->abcijk', g_abab[o, v, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||e,d>_abab*t1_bb(d,j)*t1_aa(a,l)*t2_abab(e,b,i,k)
    triples_res +=  1.000000000000000 * einsum('lced,dj,al,ebik->abcijk', g_abab[o, v, v, v], t1_bb, t1_aa, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t1_bb(d,j)*t1_bb(b,l)*t2_abab(a,e,i,k)
    triples_res += -1.000000000000000 * einsum('lcde,dj,bl,aeik->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_abab*t1_aa(d,i)*t1_aa(a,l)*t2_bbbb(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,di,al,ebjk->abcijk', g_abab[o, v, v, v], t1_aa, t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t1_bb(e,k)*t3_abbabb(a,b,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('lmde,dl,ek,abcijm->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t1_bb(e,k)*t3_abbabb(a,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ek,abcijm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t1_bb(e,j)*t3_abbabb(a,b,c,i,k,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dl,ej,abcikm->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t1_bb(e,j)*t3_abbabb(a,b,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ej,abcikm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t1_aa(e,i)*t3_abbabb(a,b,c,m,k,j)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ei,abcmkj->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t1_aa(e,i)*t3_abbabb(a,b,c,m,k,j)
    triples_res +=  1.000000000000000 * einsum('mled,dl,ei,abcmkj->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t1_aa(d,l)*t1_aa(a,m)*t3_abbabb(e,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,am,ebcijk->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,l)*t1_aa(a,m)*t3_abbabb(e,b,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('mled,dl,am,ebcijk->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t1_bb(b,m)*t3_abbabb(a,e,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('lmde,dl,bm,aecijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t1_bb(b,m)*t3_abbabb(a,e,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,bm,aecijk->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t1_bb(c,m)*t3_abbabb(a,e,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lmde,dl,cm,aebijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t1_bb(c,m)*t3_abbabb(a,e,b,i,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,dl,cm,aebijk->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,j)*t3_abbabb(a,b,c,i,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dk,ej,abciml->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t3_abbabb(a,b,c,m,j,l)
    triples_res +=  0.500000000000000 * einsum('mled,dk,ei,abcmjl->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t3_abbabb(a,b,c,l,m,j)
    triples_res += -0.500000000000000 * einsum('lmed,dk,ei,abclmj->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(a,l)*t3_abbabb(e,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,al,ebcijm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t1_bb(b,l)*t3_aabaab(e,a,c,i,m,j)
    triples_res += -1.000000000000000 * einsum('mled,dk,bl,eacimj->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(b,l)*t3_abbabb(a,e,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,bl,aecijm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t1_bb(c,l)*t3_aabaab(e,a,b,i,m,j)
    triples_res +=  1.000000000000000 * einsum('mled,dk,cl,eabimj->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(c,l)*t3_abbabb(a,e,b,i,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,cl,aebijm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t1_bb(d,j)*t1_aa(e,i)*t3_abbabb(a,b,c,m,k,l)
    triples_res += -0.500000000000000 * einsum('mled,dj,ei,abcmkl->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t1_bb(d,j)*t1_aa(e,i)*t3_abbabb(a,b,c,l,m,k)
    triples_res +=  0.500000000000000 * einsum('lmed,dj,ei,abclmk->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,j)*t1_aa(a,l)*t3_abbabb(e,b,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('lmed,dj,al,ebcikm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,j)*t1_bb(b,l)*t3_aabaab(e,a,c,i,m,k)
    triples_res +=  1.000000000000000 * einsum('mled,dj,bl,eacimk->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t1_bb(b,l)*t3_abbabb(a,e,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,bl,aecikm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,j)*t1_bb(c,l)*t3_aabaab(e,a,b,i,m,k)
    triples_res += -1.000000000000000 * einsum('mled,dj,cl,eabimk->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_aabaab, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t1_bb(c,l)*t3_abbabb(a,e,b,i,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,cl,aebikm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t1_aa(d,i)*t1_aa(a,l)*t3_abbabb(e,b,c,m,k,j)
    triples_res += -1.000000000000000 * einsum('mlde,di,al,ebcmkj->abcijk', g_aaaa[o, o, v, v], t1_aa, t1_aa, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t1_aa(a,l)*t3_bbbbbb(e,b,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,di,al,ebcjkm->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t1_bb(b,l)*t3_abbabb(a,e,c,m,k,j)
    triples_res += -1.000000000000000 * einsum('mlde,di,bl,aecmkj->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_aa(d,i)*t1_bb(c,l)*t3_abbabb(a,e,b,m,k,j)
    triples_res +=  1.000000000000000 * einsum('mlde,di,cl,aebmkj->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t1_aa(a,l)*t1_bb(b,m)*t3_abbabb(d,e,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lmde,al,bm,decijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t1_aa(a,l)*t1_bb(b,m)*t3_abbabb(e,d,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lmed,al,bm,edcijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t1_aa(a,l)*t1_bb(c,m)*t3_abbabb(d,e,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('lmde,al,cm,debijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t1_aa(a,l)*t1_bb(c,m)*t3_abbabb(e,d,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('lmed,al,cm,edbijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(b,l)*t1_bb(c,m)*t3_abbabb(a,e,d,i,j,k)
    triples_res +=  0.500000000000000 * einsum('mlde,bl,cm,aedijk->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,j)*t1_bb(b,l)*t2_abab(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,ej,bl,acim->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,j)*t1_bb(c,l)*t2_abab(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,ej,cl,abim->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t1_aa(a,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,ei,al,bcjm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t1_bb(b,l)*t2_abab(a,c,m,j)
    triples_res += -1.000000000000000 * einsum('mled,dk,ei,bl,acmj->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t1_aa(e,i)*t1_bb(c,l)*t2_abab(a,b,m,j)
    triples_res +=  1.000000000000000 * einsum('mled,dk,ei,cl,abmj->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(a,l)*t1_bb(b,m)*t2_abab(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('lmed,dk,al,bm,ecij->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t1_aa(a,l)*t1_bb(c,m)*t2_abab(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,al,cm,ebij->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(b,l)*t1_bb(c,m)*t2_abab(a,e,i,j)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,bl,cm,aeij->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,j)*t1_aa(e,i)*t1_aa(a,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmed,dj,ei,al,bckm->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_aa, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,j)*t1_aa(e,i)*t1_bb(b,l)*t2_abab(a,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mled,dj,ei,bl,acmk->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,j)*t1_aa(e,i)*t1_bb(c,l)*t2_abab(a,b,m,k)
    triples_res += -1.000000000000000 * einsum('mled,dj,ei,cl,abmk->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,j)*t1_aa(a,l)*t1_bb(b,m)*t2_abab(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('lmed,dj,al,bm,ecik->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,j)*t1_aa(a,l)*t1_bb(c,m)*t2_abab(e,b,i,k)
    triples_res += -1.000000000000000 * einsum('lmed,dj,al,cm,ebik->abcijk', g_abab[o, o, v, v], t1_bb, t1_aa, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t1_bb(b,l)*t1_bb(c,m)*t2_abab(a,e,i,k)
    triples_res += -1.000000000000000 * einsum('mlde,dj,bl,cm,aeik->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_abab, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t1_aa(a,l)*t1_bb(b,m)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lmde,di,al,bm,ecjk->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,i)*t1_aa(a,l)*t1_bb(c,m)*t2_bbbb(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('lmde,di,al,cm,ebjk->abcijk', g_abab[o, o, v, v], t1_aa, t1_aa, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    return triples_res
    
    
def ccsdt_t3_bbbbbb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):
    
    #    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :
    
    o = oa
    v = va
    
    #	 -1.0000 f_bb(l,k)*t3_bbbbbb(a,b,c,i,j,l)
    triples_res = -1.000000000000000 * einsum('lk,abcijl->abcijk', f_bb[o, o], t3_bbbbbb)
    
    #	  1.0000 f_bb(l,j)*t3_bbbbbb(a,b,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('lj,abcikl->abcijk', f_bb[o, o], t3_bbbbbb)
    
    #	 -1.0000 f_bb(l,i)*t3_bbbbbb(a,b,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('li,abcjkl->abcijk', f_bb[o, o], t3_bbbbbb)
    
    #	  1.0000 f_bb(a,d)*t3_bbbbbb(d,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('ad,dbcijk->abcijk', f_bb[v, v], t3_bbbbbb)
    
    #	 -1.0000 f_bb(b,d)*t3_bbbbbb(d,a,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('bd,dacijk->abcijk', f_bb[v, v], t3_bbbbbb)
    
    #	  1.0000 f_bb(c,d)*t3_bbbbbb(d,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('cd,dabijk->abcijk', f_bb[v, v], t3_bbbbbb)
    
    #	 -1.0000 f_bb(l,d)*t1_bb(d,k)*t3_bbbbbb(a,b,c,i,j,l)
    triples_res += -1.000000000000000 * einsum('ld,dk,abcijl->abcijk', f_bb[o, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(l,d)*t1_bb(d,j)*t3_bbbbbb(a,b,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('ld,dj,abcikl->abcijk', f_bb[o, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(l,d)*t1_bb(d,i)*t3_bbbbbb(a,b,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('ld,di,abcjkl->abcijk', f_bb[o, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(l,d)*t1_bb(a,l)*t3_bbbbbb(d,b,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('ld,al,dbcijk->abcijk', f_bb[o, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(l,d)*t1_bb(b,l)*t3_bbbbbb(d,a,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('ld,bl,dacijk->abcijk', f_bb[o, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(l,d)*t1_bb(c,l)*t3_bbbbbb(d,a,b,i,j,k)
    triples_res += -1.000000000000000 * einsum('ld,cl,dabijk->abcijk', f_bb[o, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(l,d)*t2_bbbb(d,a,j,k)*t2_bbbb(b,c,i,l)
    triples_res += -1.000000000000000 * einsum('ld,dajk,bcil->abcijk', f_bb[o, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(l,d)*t2_bbbb(d,a,i,k)*t2_bbbb(b,c,j,l)
    triples_res +=  1.000000000000000 * einsum('ld,daik,bcjl->abcijk', f_bb[o, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(l,d)*t2_bbbb(d,a,i,j)*t2_bbbb(b,c,k,l)
    triples_res += -1.000000000000000 * einsum('ld,daij,bckl->abcijk', f_bb[o, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(l,d)*t2_bbbb(d,b,j,k)*t2_bbbb(a,c,i,l)
    triples_res +=  1.000000000000000 * einsum('ld,dbjk,acil->abcijk', f_bb[o, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(l,d)*t2_bbbb(d,b,i,k)*t2_bbbb(a,c,j,l)
    triples_res += -1.000000000000000 * einsum('ld,dbik,acjl->abcijk', f_bb[o, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(l,d)*t2_bbbb(d,b,i,j)*t2_bbbb(a,c,k,l)
    triples_res +=  1.000000000000000 * einsum('ld,dbij,ackl->abcijk', f_bb[o, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(l,d)*t2_bbbb(d,c,j,k)*t2_bbbb(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('ld,dcjk,abil->abcijk', f_bb[o, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 f_bb(l,d)*t2_bbbb(d,c,i,k)*t2_bbbb(a,b,j,l)
    triples_res +=  1.000000000000000 * einsum('ld,dcik,abjl->abcijk', f_bb[o, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 f_bb(l,d)*t2_bbbb(d,c,i,j)*t2_bbbb(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('ld,dcij,abkl->abcijk', f_bb[o, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||j,k>_bbbb*t2_bbbb(b,c,i,l)
    triples_res += -1.000000000000000 * einsum('lajk,bcil->abcijk', g_bbbb[o, v, o, o], t2_bbbb)
    
    #	  1.0000 <l,a||i,k>_bbbb*t2_bbbb(b,c,j,l)
    triples_res +=  1.000000000000000 * einsum('laik,bcjl->abcijk', g_bbbb[o, v, o, o], t2_bbbb)
    
    #	 -1.0000 <l,a||i,j>_bbbb*t2_bbbb(b,c,k,l)
    triples_res += -1.000000000000000 * einsum('laij,bckl->abcijk', g_bbbb[o, v, o, o], t2_bbbb)
    
    #	  1.0000 <l,b||j,k>_bbbb*t2_bbbb(a,c,i,l)
    triples_res +=  1.000000000000000 * einsum('lbjk,acil->abcijk', g_bbbb[o, v, o, o], t2_bbbb)
    
    #	 -1.0000 <l,b||i,k>_bbbb*t2_bbbb(a,c,j,l)
    triples_res += -1.000000000000000 * einsum('lbik,acjl->abcijk', g_bbbb[o, v, o, o], t2_bbbb)
    
    #	  1.0000 <l,b||i,j>_bbbb*t2_bbbb(a,c,k,l)
    triples_res +=  1.000000000000000 * einsum('lbij,ackl->abcijk', g_bbbb[o, v, o, o], t2_bbbb)
    
    #	 -1.0000 <l,c||j,k>_bbbb*t2_bbbb(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('lcjk,abil->abcijk', g_bbbb[o, v, o, o], t2_bbbb)
    
    #	  1.0000 <l,c||i,k>_bbbb*t2_bbbb(a,b,j,l)
    triples_res +=  1.000000000000000 * einsum('lcik,abjl->abcijk', g_bbbb[o, v, o, o], t2_bbbb)
    
    #	 -1.0000 <l,c||i,j>_bbbb*t2_bbbb(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcij,abkl->abcijk', g_bbbb[o, v, o, o], t2_bbbb)
    
    #	 -1.0000 <a,b||d,k>_bbbb*t2_bbbb(d,c,i,j)
    triples_res += -1.000000000000000 * einsum('abdk,dcij->abcijk', g_bbbb[v, v, v, o], t2_bbbb)
    
    #	  1.0000 <a,b||d,j>_bbbb*t2_bbbb(d,c,i,k)
    triples_res +=  1.000000000000000 * einsum('abdj,dcik->abcijk', g_bbbb[v, v, v, o], t2_bbbb)
    
    #	 -1.0000 <a,b||d,i>_bbbb*t2_bbbb(d,c,j,k)
    triples_res += -1.000000000000000 * einsum('abdi,dcjk->abcijk', g_bbbb[v, v, v, o], t2_bbbb)
    
    #	  1.0000 <a,c||d,k>_bbbb*t2_bbbb(d,b,i,j)
    triples_res +=  1.000000000000000 * einsum('acdk,dbij->abcijk', g_bbbb[v, v, v, o], t2_bbbb)
    
    #	 -1.0000 <a,c||d,j>_bbbb*t2_bbbb(d,b,i,k)
    triples_res += -1.000000000000000 * einsum('acdj,dbik->abcijk', g_bbbb[v, v, v, o], t2_bbbb)
    
    #	  1.0000 <a,c||d,i>_bbbb*t2_bbbb(d,b,j,k)
    triples_res +=  1.000000000000000 * einsum('acdi,dbjk->abcijk', g_bbbb[v, v, v, o], t2_bbbb)
    
    #	 -1.0000 <b,c||d,k>_bbbb*t2_bbbb(d,a,i,j)
    triples_res += -1.000000000000000 * einsum('bcdk,daij->abcijk', g_bbbb[v, v, v, o], t2_bbbb)
    
    #	  1.0000 <b,c||d,j>_bbbb*t2_bbbb(d,a,i,k)
    triples_res +=  1.000000000000000 * einsum('bcdj,daik->abcijk', g_bbbb[v, v, v, o], t2_bbbb)
    
    #	 -1.0000 <b,c||d,i>_bbbb*t2_bbbb(d,a,j,k)
    triples_res += -1.000000000000000 * einsum('bcdi,dajk->abcijk', g_bbbb[v, v, v, o], t2_bbbb)
    
    #	  0.5000 <m,l||j,k>_bbbb*t3_bbbbbb(a,b,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mljk,abciml->abcijk', g_bbbb[o, o, o, o], t3_bbbbbb)
    
    #	 -0.5000 <m,l||i,k>_bbbb*t3_bbbbbb(a,b,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mlik,abcjml->abcijk', g_bbbb[o, o, o, o], t3_bbbbbb)
    
    #	  0.5000 <m,l||i,j>_bbbb*t3_bbbbbb(a,b,c,k,m,l)
    triples_res +=  0.500000000000000 * einsum('mlij,abckml->abcijk', g_bbbb[o, o, o, o], t3_bbbbbb)
    
    #	 -1.0000 <l,a||d,k>_abab*t3_abbabb(d,b,c,l,j,i)
    triples_res += -1.000000000000000 * einsum('ladk,dbclji->abcijk', g_abab[o, v, v, o], t3_abbabb)
    
    #	  1.0000 <l,a||d,k>_bbbb*t3_bbbbbb(d,b,c,i,j,l)
    triples_res +=  1.000000000000000 * einsum('ladk,dbcijl->abcijk', g_bbbb[o, v, v, o], t3_bbbbbb)
    
    #	  1.0000 <l,a||d,j>_abab*t3_abbabb(d,b,c,l,k,i)
    triples_res +=  1.000000000000000 * einsum('ladj,dbclki->abcijk', g_abab[o, v, v, o], t3_abbabb)
    
    #	 -1.0000 <l,a||d,j>_bbbb*t3_bbbbbb(d,b,c,i,k,l)
    triples_res += -1.000000000000000 * einsum('ladj,dbcikl->abcijk', g_bbbb[o, v, v, o], t3_bbbbbb)
    
    #	 -1.0000 <l,a||d,i>_abab*t3_abbabb(d,b,c,l,k,j)
    triples_res += -1.000000000000000 * einsum('ladi,dbclkj->abcijk', g_abab[o, v, v, o], t3_abbabb)
    
    #	  1.0000 <l,a||d,i>_bbbb*t3_bbbbbb(d,b,c,j,k,l)
    triples_res +=  1.000000000000000 * einsum('ladi,dbcjkl->abcijk', g_bbbb[o, v, v, o], t3_bbbbbb)
    
    #	  1.0000 <l,b||d,k>_abab*t3_abbabb(d,a,c,l,j,i)
    triples_res +=  1.000000000000000 * einsum('lbdk,daclji->abcijk', g_abab[o, v, v, o], t3_abbabb)
    
    #	 -1.0000 <l,b||d,k>_bbbb*t3_bbbbbb(d,a,c,i,j,l)
    triples_res += -1.000000000000000 * einsum('lbdk,dacijl->abcijk', g_bbbb[o, v, v, o], t3_bbbbbb)
    
    #	 -1.0000 <l,b||d,j>_abab*t3_abbabb(d,a,c,l,k,i)
    triples_res += -1.000000000000000 * einsum('lbdj,daclki->abcijk', g_abab[o, v, v, o], t3_abbabb)
    
    #	  1.0000 <l,b||d,j>_bbbb*t3_bbbbbb(d,a,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('lbdj,dacikl->abcijk', g_bbbb[o, v, v, o], t3_bbbbbb)
    
    #	  1.0000 <l,b||d,i>_abab*t3_abbabb(d,a,c,l,k,j)
    triples_res +=  1.000000000000000 * einsum('lbdi,daclkj->abcijk', g_abab[o, v, v, o], t3_abbabb)
    
    #	 -1.0000 <l,b||d,i>_bbbb*t3_bbbbbb(d,a,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('lbdi,dacjkl->abcijk', g_bbbb[o, v, v, o], t3_bbbbbb)
    
    #	 -1.0000 <l,c||d,k>_abab*t3_abbabb(d,a,b,l,j,i)
    triples_res += -1.000000000000000 * einsum('lcdk,dablji->abcijk', g_abab[o, v, v, o], t3_abbabb)
    
    #	  1.0000 <l,c||d,k>_bbbb*t3_bbbbbb(d,a,b,i,j,l)
    triples_res +=  1.000000000000000 * einsum('lcdk,dabijl->abcijk', g_bbbb[o, v, v, o], t3_bbbbbb)
    
    #	  1.0000 <l,c||d,j>_abab*t3_abbabb(d,a,b,l,k,i)
    triples_res +=  1.000000000000000 * einsum('lcdj,dablki->abcijk', g_abab[o, v, v, o], t3_abbabb)
    
    #	 -1.0000 <l,c||d,j>_bbbb*t3_bbbbbb(d,a,b,i,k,l)
    triples_res += -1.000000000000000 * einsum('lcdj,dabikl->abcijk', g_bbbb[o, v, v, o], t3_bbbbbb)
    
    #	 -1.0000 <l,c||d,i>_abab*t3_abbabb(d,a,b,l,k,j)
    triples_res += -1.000000000000000 * einsum('lcdi,dablkj->abcijk', g_abab[o, v, v, o], t3_abbabb)
    
    #	  1.0000 <l,c||d,i>_bbbb*t3_bbbbbb(d,a,b,j,k,l)
    triples_res +=  1.000000000000000 * einsum('lcdi,dabjkl->abcijk', g_bbbb[o, v, v, o], t3_bbbbbb)
    
    #	  0.5000 <a,b||d,e>_bbbb*t3_bbbbbb(d,e,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('abde,decijk->abcijk', g_bbbb[v, v, v, v], t3_bbbbbb)
    
    #	 -0.5000 <a,c||d,e>_bbbb*t3_bbbbbb(d,e,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('acde,debijk->abcijk', g_bbbb[v, v, v, v], t3_bbbbbb)
    
    #	  0.5000 <b,c||d,e>_bbbb*t3_bbbbbb(d,e,a,i,j,k)
    triples_res +=  0.500000000000000 * einsum('bcde,deaijk->abcijk', g_bbbb[v, v, v, v], t3_bbbbbb)
    
    #	  1.0000 <m,l||j,k>_bbbb*t1_bb(a,l)*t2_bbbb(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mljk,al,bcim->abcijk', g_bbbb[o, o, o, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||j,k>_bbbb*t1_bb(b,l)*t2_bbbb(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mljk,bl,acim->abcijk', g_bbbb[o, o, o, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||j,k>_bbbb*t1_bb(c,l)*t2_bbbb(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mljk,cl,abim->abcijk', g_bbbb[o, o, o, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,k>_bbbb*t1_bb(a,l)*t2_bbbb(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlik,al,bcjm->abcijk', g_bbbb[o, o, o, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,k>_bbbb*t1_bb(b,l)*t2_bbbb(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlik,bl,acjm->abcijk', g_bbbb[o, o, o, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,k>_bbbb*t1_bb(c,l)*t2_bbbb(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlik,cl,abjm->abcijk', g_bbbb[o, o, o, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,j>_bbbb*t1_bb(a,l)*t2_bbbb(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlij,al,bckm->abcijk', g_bbbb[o, o, o, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||i,j>_bbbb*t1_bb(b,l)*t2_bbbb(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlij,bl,ackm->abcijk', g_bbbb[o, o, o, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||i,j>_bbbb*t1_bb(c,l)*t2_bbbb(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mlij,cl,abkm->abcijk', g_bbbb[o, o, o, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,k>_bbbb*t1_bb(d,j)*t2_bbbb(b,c,i,l)
    triples_res += -1.000000000000000 * einsum('ladk,dj,bcil->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,k>_bbbb*t1_bb(d,i)*t2_bbbb(b,c,j,l)
    triples_res +=  1.000000000000000 * einsum('ladk,di,bcjl->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,k>_bbbb*t1_bb(b,l)*t2_bbbb(d,c,i,j)
    triples_res += -1.000000000000000 * einsum('ladk,bl,dcij->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,k>_bbbb*t1_bb(c,l)*t2_bbbb(d,b,i,j)
    triples_res +=  1.000000000000000 * einsum('ladk,cl,dbij->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,j>_bbbb*t1_bb(d,k)*t2_bbbb(b,c,i,l)
    triples_res +=  1.000000000000000 * einsum('ladj,dk,bcil->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,j>_bbbb*t1_bb(d,i)*t2_bbbb(b,c,k,l)
    triples_res += -1.000000000000000 * einsum('ladj,di,bckl->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,j>_bbbb*t1_bb(b,l)*t2_bbbb(d,c,i,k)
    triples_res +=  1.000000000000000 * einsum('ladj,bl,dcik->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,j>_bbbb*t1_bb(c,l)*t2_bbbb(d,b,i,k)
    triples_res += -1.000000000000000 * einsum('ladj,cl,dbik->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,i>_bbbb*t1_bb(d,k)*t2_bbbb(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('ladi,dk,bcjl->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,i>_bbbb*t1_bb(d,j)*t2_bbbb(b,c,k,l)
    triples_res +=  1.000000000000000 * einsum('ladi,dj,bckl->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,i>_bbbb*t1_bb(b,l)*t2_bbbb(d,c,j,k)
    triples_res += -1.000000000000000 * einsum('ladi,bl,dcjk->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,i>_bbbb*t1_bb(c,l)*t2_bbbb(d,b,j,k)
    triples_res +=  1.000000000000000 * einsum('ladi,cl,dbjk->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,k>_bbbb*t1_bb(d,j)*t2_bbbb(a,c,i,l)
    triples_res +=  1.000000000000000 * einsum('lbdk,dj,acil->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,k>_bbbb*t1_bb(d,i)*t2_bbbb(a,c,j,l)
    triples_res += -1.000000000000000 * einsum('lbdk,di,acjl->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,k>_bbbb*t1_bb(a,l)*t2_bbbb(d,c,i,j)
    triples_res +=  1.000000000000000 * einsum('lbdk,al,dcij->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,k>_bbbb*t1_bb(c,l)*t2_bbbb(d,a,i,j)
    triples_res += -1.000000000000000 * einsum('lbdk,cl,daij->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,j>_bbbb*t1_bb(d,k)*t2_bbbb(a,c,i,l)
    triples_res += -1.000000000000000 * einsum('lbdj,dk,acil->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,j>_bbbb*t1_bb(d,i)*t2_bbbb(a,c,k,l)
    triples_res +=  1.000000000000000 * einsum('lbdj,di,ackl->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,j>_bbbb*t1_bb(a,l)*t2_bbbb(d,c,i,k)
    triples_res += -1.000000000000000 * einsum('lbdj,al,dcik->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,j>_bbbb*t1_bb(c,l)*t2_bbbb(d,a,i,k)
    triples_res +=  1.000000000000000 * einsum('lbdj,cl,daik->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,i>_bbbb*t1_bb(d,k)*t2_bbbb(a,c,j,l)
    triples_res +=  1.000000000000000 * einsum('lbdi,dk,acjl->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,i>_bbbb*t1_bb(d,j)*t2_bbbb(a,c,k,l)
    triples_res += -1.000000000000000 * einsum('lbdi,dj,ackl->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,i>_bbbb*t1_bb(a,l)*t2_bbbb(d,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lbdi,al,dcjk->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,i>_bbbb*t1_bb(c,l)*t2_bbbb(d,a,j,k)
    triples_res += -1.000000000000000 * einsum('lbdi,cl,dajk->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,k>_bbbb*t1_bb(d,j)*t2_bbbb(a,b,i,l)
    triples_res += -1.000000000000000 * einsum('lcdk,dj,abil->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,k>_bbbb*t1_bb(d,i)*t2_bbbb(a,b,j,l)
    triples_res +=  1.000000000000000 * einsum('lcdk,di,abjl->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,k>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,i,j)
    triples_res += -1.000000000000000 * einsum('lcdk,al,dbij->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,k>_bbbb*t1_bb(b,l)*t2_bbbb(d,a,i,j)
    triples_res +=  1.000000000000000 * einsum('lcdk,bl,daij->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,j>_bbbb*t1_bb(d,k)*t2_bbbb(a,b,i,l)
    triples_res +=  1.000000000000000 * einsum('lcdj,dk,abil->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,j>_bbbb*t1_bb(d,i)*t2_bbbb(a,b,k,l)
    triples_res += -1.000000000000000 * einsum('lcdj,di,abkl->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,j>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,i,k)
    triples_res +=  1.000000000000000 * einsum('lcdj,al,dbik->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,j>_bbbb*t1_bb(b,l)*t2_bbbb(d,a,i,k)
    triples_res += -1.000000000000000 * einsum('lcdj,bl,daik->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,i>_bbbb*t1_bb(d,k)*t2_bbbb(a,b,j,l)
    triples_res += -1.000000000000000 * einsum('lcdi,dk,abjl->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,i>_bbbb*t1_bb(d,j)*t2_bbbb(a,b,k,l)
    triples_res +=  1.000000000000000 * einsum('lcdi,dj,abkl->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,i>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,j,k)
    triples_res += -1.000000000000000 * einsum('lcdi,al,dbjk->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,i>_bbbb*t1_bb(b,l)*t2_bbbb(d,a,j,k)
    triples_res +=  1.000000000000000 * einsum('lcdi,bl,dajk->abcijk', g_bbbb[o, v, v, o], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,b||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('abde,dk,ecij->abcijk', g_bbbb[v, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,b||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('abde,dj,ecik->abcijk', g_bbbb[v, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,b||d,e>_bbbb*t1_bb(d,i)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('abde,di,ecjk->abcijk', g_bbbb[v, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,c||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('acde,dk,ebij->abcijk', g_bbbb[v, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <a,c||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,b,i,k)
    triples_res +=  1.000000000000000 * einsum('acde,dj,ebik->abcijk', g_bbbb[v, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <a,c||d,e>_bbbb*t1_bb(d,i)*t2_bbbb(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('acde,di,ebjk->abcijk', g_bbbb[v, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,c||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,a,i,j)
    triples_res +=  1.000000000000000 * einsum('bcde,dk,eaij->abcijk', g_bbbb[v, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <b,c||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,a,i,k)
    triples_res += -1.000000000000000 * einsum('bcde,dj,eaik->abcijk', g_bbbb[v, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <b,c||d,e>_bbbb*t1_bb(d,i)*t2_bbbb(e,a,j,k)
    triples_res +=  1.000000000000000 * einsum('bcde,di,eajk->abcijk', g_bbbb[v, v, v, v], t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t1_aa(d,l)*t3_bbbbbb(a,b,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('lmdk,dl,abcijm->abcijk', g_abab[o, o, v, o], t1_aa, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t1_bb(d,l)*t3_bbbbbb(a,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dl,abcijm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,k>_bbbb*t1_bb(d,j)*t3_bbbbbb(a,b,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mldk,dj,abciml->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,k>_bbbb*t1_bb(d,i)*t3_bbbbbb(a,b,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mldk,di,abcjml->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_abab*t1_bb(a,l)*t3_abbabb(d,b,c,m,j,i)
    triples_res +=  1.000000000000000 * einsum('mldk,al,dbcmji->abcijk', g_abab[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t1_bb(a,l)*t3_bbbbbb(d,b,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,al,dbcijm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_abab*t1_bb(b,l)*t3_abbabb(d,a,c,m,j,i)
    triples_res += -1.000000000000000 * einsum('mldk,bl,dacmji->abcijk', g_abab[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t1_bb(b,l)*t3_bbbbbb(d,a,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,bl,dacijm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_abab*t1_bb(c,l)*t3_abbabb(d,a,b,m,j,i)
    triples_res +=  1.000000000000000 * einsum('mldk,cl,dabmji->abcijk', g_abab[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t1_bb(c,l)*t3_bbbbbb(d,a,b,i,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,cl,dabijm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,j>_abab*t1_aa(d,l)*t3_bbbbbb(a,b,c,i,k,m)
    triples_res +=  1.000000000000000 * einsum('lmdj,dl,abcikm->abcijk', g_abab[o, o, v, o], t1_aa, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t1_bb(d,l)*t3_bbbbbb(a,b,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('mldj,dl,abcikm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,j>_bbbb*t1_bb(d,k)*t3_bbbbbb(a,b,c,i,m,l)
    triples_res += -0.500000000000000 * einsum('mldj,dk,abciml->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,j>_bbbb*t1_bb(d,i)*t3_bbbbbb(a,b,c,k,m,l)
    triples_res +=  0.500000000000000 * einsum('mldj,di,abckml->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_abab*t1_bb(a,l)*t3_abbabb(d,b,c,m,k,i)
    triples_res += -1.000000000000000 * einsum('mldj,al,dbcmki->abcijk', g_abab[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_bbbb*t1_bb(a,l)*t3_bbbbbb(d,b,c,i,k,m)
    triples_res +=  1.000000000000000 * einsum('mldj,al,dbcikm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_abab*t1_bb(b,l)*t3_abbabb(d,a,c,m,k,i)
    triples_res +=  1.000000000000000 * einsum('mldj,bl,dacmki->abcijk', g_abab[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t1_bb(b,l)*t3_bbbbbb(d,a,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('mldj,bl,dacikm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_abab*t1_bb(c,l)*t3_abbabb(d,a,b,m,k,i)
    triples_res += -1.000000000000000 * einsum('mldj,cl,dabmki->abcijk', g_abab[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_bbbb*t1_bb(c,l)*t3_bbbbbb(d,a,b,i,k,m)
    triples_res +=  1.000000000000000 * einsum('mldj,cl,dabikm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,i>_abab*t1_aa(d,l)*t3_bbbbbb(a,b,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('lmdi,dl,abcjkm->abcijk', g_abab[o, o, v, o], t1_aa, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_bbbb*t1_bb(d,l)*t3_bbbbbb(a,b,c,j,k,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dl,abcjkm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,i>_bbbb*t1_bb(d,k)*t3_bbbbbb(a,b,c,j,m,l)
    triples_res +=  0.500000000000000 * einsum('mldi,dk,abcjml->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,i>_bbbb*t1_bb(d,j)*t3_bbbbbb(a,b,c,k,m,l)
    triples_res += -0.500000000000000 * einsum('mldi,dj,abckml->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_abab*t1_bb(a,l)*t3_abbabb(d,b,c,m,k,j)
    triples_res +=  1.000000000000000 * einsum('mldi,al,dbcmkj->abcijk', g_abab[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_bbbb*t1_bb(a,l)*t3_bbbbbb(d,b,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('mldi,al,dbcjkm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_abab*t1_bb(b,l)*t3_abbabb(d,a,c,m,k,j)
    triples_res += -1.000000000000000 * einsum('mldi,bl,dacmkj->abcijk', g_abab[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_bbbb*t1_bb(b,l)*t3_bbbbbb(d,a,c,j,k,m)
    triples_res +=  1.000000000000000 * einsum('mldi,bl,dacjkm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_abab*t1_bb(c,l)*t3_abbabb(d,a,b,m,k,j)
    triples_res +=  1.000000000000000 * einsum('mldi,cl,dabmkj->abcijk', g_abab[o, o, v, o], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_bbbb*t1_bb(c,l)*t3_bbbbbb(d,a,b,j,k,m)
    triples_res += -1.000000000000000 * einsum('mldi,cl,dabjkm->abcijk', g_bbbb[o, o, v, o], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_abab*t1_aa(d,l)*t3_bbbbbb(e,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lade,dl,ebcijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_bbbb*t1_bb(d,l)*t3_bbbbbb(e,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lade,dl,ebcijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||e,d>_abab*t1_bb(d,k)*t3_abbabb(e,b,c,l,j,i)
    triples_res += -1.000000000000000 * einsum('laed,dk,ebclji->abcijk', g_abab[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_bbbb*t1_bb(d,k)*t3_bbbbbb(e,b,c,i,j,l)
    triples_res += -1.000000000000000 * einsum('lade,dk,ebcijl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||e,d>_abab*t1_bb(d,j)*t3_abbabb(e,b,c,l,k,i)
    triples_res +=  1.000000000000000 * einsum('laed,dj,ebclki->abcijk', g_abab[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_bbbb*t1_bb(d,j)*t3_bbbbbb(e,b,c,i,k,l)
    triples_res +=  1.000000000000000 * einsum('lade,dj,ebcikl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||e,d>_abab*t1_bb(d,i)*t3_abbabb(e,b,c,l,k,j)
    triples_res += -1.000000000000000 * einsum('laed,di,ebclkj->abcijk', g_abab[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_bbbb*t1_bb(d,i)*t3_bbbbbb(e,b,c,j,k,l)
    triples_res += -1.000000000000000 * einsum('lade,di,ebcjkl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,a||d,e>_bbbb*t1_bb(b,l)*t3_bbbbbb(d,e,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lade,bl,decijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,a||d,e>_bbbb*t1_bb(c,l)*t3_bbbbbb(d,e,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('lade,cl,debijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_abab*t1_aa(d,l)*t3_bbbbbb(e,a,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('lbde,dl,eacijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t1_bb(d,l)*t3_bbbbbb(e,a,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('lbde,dl,eacijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||e,d>_abab*t1_bb(d,k)*t3_abbabb(e,a,c,l,j,i)
    triples_res +=  1.000000000000000 * einsum('lbed,dk,eaclji->abcijk', g_abab[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_bbbb*t1_bb(d,k)*t3_bbbbbb(e,a,c,i,j,l)
    triples_res +=  1.000000000000000 * einsum('lbde,dk,eacijl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||e,d>_abab*t1_bb(d,j)*t3_abbabb(e,a,c,l,k,i)
    triples_res += -1.000000000000000 * einsum('lbed,dj,eaclki->abcijk', g_abab[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t1_bb(d,j)*t3_bbbbbb(e,a,c,i,k,l)
    triples_res += -1.000000000000000 * einsum('lbde,dj,eacikl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||e,d>_abab*t1_bb(d,i)*t3_abbabb(e,a,c,l,k,j)
    triples_res +=  1.000000000000000 * einsum('lbed,di,eaclkj->abcijk', g_abab[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_bbbb*t1_bb(d,i)*t3_bbbbbb(e,a,c,j,k,l)
    triples_res +=  1.000000000000000 * einsum('lbde,di,eacjkl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,b||d,e>_bbbb*t1_bb(a,l)*t3_bbbbbb(d,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lbde,al,decijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,b||d,e>_bbbb*t1_bb(c,l)*t3_bbbbbb(d,e,a,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lbde,cl,deaijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_abab*t1_aa(d,l)*t3_bbbbbb(e,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dl,eabijk->abcijk', g_abab[o, v, v, v], t1_aa, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t1_bb(d,l)*t3_bbbbbb(e,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dl,eabijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||e,d>_abab*t1_bb(d,k)*t3_abbabb(e,a,b,l,j,i)
    triples_res += -1.000000000000000 * einsum('lced,dk,eablji->abcijk', g_abab[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t1_bb(d,k)*t3_bbbbbb(e,a,b,i,j,l)
    triples_res += -1.000000000000000 * einsum('lcde,dk,eabijl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||e,d>_abab*t1_bb(d,j)*t3_abbabb(e,a,b,l,k,i)
    triples_res +=  1.000000000000000 * einsum('lced,dj,eablki->abcijk', g_abab[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t1_bb(d,j)*t3_bbbbbb(e,a,b,i,k,l)
    triples_res +=  1.000000000000000 * einsum('lcde,dj,eabikl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||e,d>_abab*t1_bb(d,i)*t3_abbabb(e,a,b,l,k,j)
    triples_res += -1.000000000000000 * einsum('lced,di,eablkj->abcijk', g_abab[o, v, v, v], t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t1_bb(d,i)*t3_bbbbbb(e,a,b,j,k,l)
    triples_res += -1.000000000000000 * einsum('lcde,di,eabjkl->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,c||d,e>_bbbb*t1_bb(a,l)*t3_bbbbbb(d,e,b,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lcde,al,debijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,c||d,e>_bbbb*t1_bb(b,l)*t3_bbbbbb(d,e,a,i,j,k)
    triples_res += -0.500000000000000 * einsum('lcde,bl,deaijk->abcijk', g_bbbb[o, v, v, v], t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_abab(d,e,l,k)*t3_bbbbbb(a,b,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmde,delk,abcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(e,d,l,k)*t3_bbbbbb(a,b,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('lmed,edlk,abcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,e,k,l)*t3_bbbbbb(a,b,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,dekl,abcijm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_abab(d,e,l,j)*t3_bbbbbb(a,b,c,i,k,m)
    triples_res +=  0.500000000000000 * einsum('lmde,delj,abcikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_abab(e,d,l,j)*t3_bbbbbb(a,b,c,i,k,m)
    triples_res +=  0.500000000000000 * einsum('lmed,edlj,abcikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,e,j,l)*t3_bbbbbb(a,b,c,i,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,dejl,abcikm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_abab(d,e,l,i)*t3_bbbbbb(a,b,c,j,k,m)
    triples_res += -0.500000000000000 * einsum('lmde,deli,abcjkm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_abab(e,d,l,i)*t3_bbbbbb(a,b,c,j,k,m)
    triples_res += -0.500000000000000 * einsum('lmed,edli,abcjkm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,e,i,l)*t3_bbbbbb(a,b,c,j,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,deil,abcjkm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_bbbb*t2_bbbb(d,e,j,k)*t3_bbbbbb(a,b,c,i,m,l)
    triples_res +=  0.250000000000000 * einsum('mlde,dejk,abciml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.2500 <m,l||d,e>_bbbb*t2_bbbb(d,e,i,k)*t3_bbbbbb(a,b,c,j,m,l)
    triples_res += -0.250000000000000 * einsum('mlde,deik,abcjml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_bbbb*t2_bbbb(d,e,i,j)*t3_bbbbbb(a,b,c,k,m,l)
    triples_res +=  0.250000000000000 * einsum('mlde,deij,abckml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_abab(d,a,m,l)*t3_bbbbbb(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,daml,ebcijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_abab(d,a,l,m)*t3_bbbbbb(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('lmde,dalm,ebcijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,a,m,l)*t3_bbbbbb(e,b,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,daml,ebcijk->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_abab(d,a,l,k)*t3_abbabb(e,b,c,m,j,i)
    triples_res +=  1.000000000000010 * einsum('mlde,dalk,ebcmji->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_abab(d,a,l,k)*t3_bbbbbb(e,b,c,i,j,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dalk,ebcijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_bbbb(d,a,k,l)*t3_abbabb(e,b,c,m,j,i)
    triples_res +=  1.000000000000010 * einsum('mled,dakl,ebcmji->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,a,k,l)*t3_bbbbbb(e,b,c,i,j,m)
    triples_res +=  1.000000000000010 * einsum('mlde,dakl,ebcijm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_abab(d,a,l,j)*t3_abbabb(e,b,c,m,k,i)
    triples_res += -1.000000000000010 * einsum('mlde,dalj,ebcmki->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_abab(d,a,l,j)*t3_bbbbbb(e,b,c,i,k,m)
    triples_res += -1.000000000000010 * einsum('lmde,dalj,ebcikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_bbbb(d,a,j,l)*t3_abbabb(e,b,c,m,k,i)
    triples_res += -1.000000000000010 * einsum('mled,dajl,ebcmki->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,a,j,l)*t3_bbbbbb(e,b,c,i,k,m)
    triples_res += -1.000000000000010 * einsum('mlde,dajl,ebcikm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_abab(d,a,l,i)*t3_abbabb(e,b,c,m,k,j)
    triples_res +=  1.000000000000010 * einsum('mlde,dali,ebcmkj->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_abab(d,a,l,i)*t3_bbbbbb(e,b,c,j,k,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dali,ebcjkm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_bbbb(d,a,i,l)*t3_abbabb(e,b,c,m,k,j)
    triples_res +=  1.000000000000010 * einsum('mled,dail,ebcmkj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,a,i,l)*t3_bbbbbb(e,b,c,j,k,m)
    triples_res +=  1.000000000000010 * einsum('mlde,dail,ebcjkm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_bbbb(d,a,j,k)*t3_abbabb(e,b,c,m,i,l)
    triples_res += -0.500000000000000 * einsum('mled,dajk,ebcmil->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_bbbb(d,a,j,k)*t3_abbabb(e,b,c,l,m,i)
    triples_res +=  0.500000000000000 * einsum('lmed,dajk,ebclmi->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,a,j,k)*t3_bbbbbb(e,b,c,i,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dajk,ebciml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_bbbb(d,a,i,k)*t3_abbabb(e,b,c,m,j,l)
    triples_res +=  0.500000000000000 * einsum('mled,daik,ebcmjl->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_bbbb(d,a,i,k)*t3_abbabb(e,b,c,l,m,j)
    triples_res += -0.500000000000000 * einsum('lmed,daik,ebclmj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,a,i,k)*t3_bbbbbb(e,b,c,j,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,daik,ebcjml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_bbbb(d,a,i,j)*t3_abbabb(e,b,c,m,k,l)
    triples_res += -0.500000000000000 * einsum('mled,daij,ebcmkl->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_bbbb(d,a,i,j)*t3_abbabb(e,b,c,l,m,k)
    triples_res +=  0.500000000000000 * einsum('lmed,daij,ebclmk->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,a,i,j)*t3_bbbbbb(e,b,c,k,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,daij,ebckml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_abab(d,b,m,l)*t3_bbbbbb(e,a,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('mlde,dbml,eacijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,m||d,e>_abab*t2_abab(d,b,l,m)*t3_bbbbbb(e,a,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('lmde,dblm,eacijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,b,m,l)*t3_bbbbbb(e,a,c,i,j,k)
    triples_res +=  0.500000000000000 * einsum('mlde,dbml,eacijk->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_abab(d,b,l,k)*t3_abbabb(e,a,c,m,j,i)
    triples_res += -1.000000000000010 * einsum('mlde,dblk,eacmji->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_abab(d,b,l,k)*t3_bbbbbb(e,a,c,i,j,m)
    triples_res += -1.000000000000010 * einsum('lmde,dblk,eacijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_bbbb(d,b,k,l)*t3_abbabb(e,a,c,m,j,i)
    triples_res += -1.000000000000010 * einsum('mled,dbkl,eacmji->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,b,k,l)*t3_bbbbbb(e,a,c,i,j,m)
    triples_res += -1.000000000000010 * einsum('mlde,dbkl,eacijm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_abab(d,b,l,j)*t3_abbabb(e,a,c,m,k,i)
    triples_res +=  1.000000000000010 * einsum('mlde,dblj,eacmki->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_abab(d,b,l,j)*t3_bbbbbb(e,a,c,i,k,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dblj,eacikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_bbbb(d,b,j,l)*t3_abbabb(e,a,c,m,k,i)
    triples_res +=  1.000000000000010 * einsum('mled,dbjl,eacmki->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,b,j,l)*t3_bbbbbb(e,a,c,i,k,m)
    triples_res +=  1.000000000000010 * einsum('mlde,dbjl,eacikm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_abab(d,b,l,i)*t3_abbabb(e,a,c,m,k,j)
    triples_res += -1.000000000000010 * einsum('mlde,dbli,eacmkj->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_abab(d,b,l,i)*t3_bbbbbb(e,a,c,j,k,m)
    triples_res += -1.000000000000010 * einsum('lmde,dbli,eacjkm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_bbbb(d,b,i,l)*t3_abbabb(e,a,c,m,k,j)
    triples_res += -1.000000000000010 * einsum('mled,dbil,eacmkj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,b,i,l)*t3_bbbbbb(e,a,c,j,k,m)
    triples_res += -1.000000000000010 * einsum('mlde,dbil,eacjkm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_bbbb(d,b,j,k)*t3_abbabb(e,a,c,m,i,l)
    triples_res +=  0.500000000000000 * einsum('mled,dbjk,eacmil->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_bbbb(d,b,j,k)*t3_abbabb(e,a,c,l,m,i)
    triples_res += -0.500000000000000 * einsum('lmed,dbjk,eaclmi->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,b,j,k)*t3_bbbbbb(e,a,c,i,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dbjk,eaciml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_bbbb(d,b,i,k)*t3_abbabb(e,a,c,m,j,l)
    triples_res += -0.500000000000000 * einsum('mled,dbik,eacmjl->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_bbbb(d,b,i,k)*t3_abbabb(e,a,c,l,m,j)
    triples_res +=  0.500000000000000 * einsum('lmed,dbik,eaclmj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,b,i,k)*t3_bbbbbb(e,a,c,j,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dbik,eacjml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_bbbb(d,b,i,j)*t3_abbabb(e,a,c,m,k,l)
    triples_res +=  0.500000000000000 * einsum('mled,dbij,eacmkl->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_bbbb(d,b,i,j)*t3_abbabb(e,a,c,l,m,k)
    triples_res += -0.500000000000000 * einsum('lmed,dbij,eaclmk->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,b,i,j)*t3_bbbbbb(e,a,c,k,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dbij,eackml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_abab(d,c,m,l)*t3_bbbbbb(e,a,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,dcml,eabijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,m||d,e>_abab*t2_abab(d,c,l,m)*t3_bbbbbb(e,a,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('lmde,dclm,eabijk->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,c,m,l)*t3_bbbbbb(e,a,b,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,dcml,eabijk->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_abab(d,c,l,k)*t3_abbabb(e,a,b,m,j,i)
    triples_res +=  1.000000000000010 * einsum('mlde,dclk,eabmji->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_abab(d,c,l,k)*t3_bbbbbb(e,a,b,i,j,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dclk,eabijm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_bbbb(d,c,k,l)*t3_abbabb(e,a,b,m,j,i)
    triples_res +=  1.000000000000010 * einsum('mled,dckl,eabmji->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,c,k,l)*t3_bbbbbb(e,a,b,i,j,m)
    triples_res +=  1.000000000000010 * einsum('mlde,dckl,eabijm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_aaaa*t2_abab(d,c,l,j)*t3_abbabb(e,a,b,m,k,i)
    triples_res += -1.000000000000010 * einsum('mlde,dclj,eabmki->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t2_abab(d,c,l,j)*t3_bbbbbb(e,a,b,i,k,m)
    triples_res += -1.000000000000010 * einsum('lmde,dclj,eabikm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t2_bbbb(d,c,j,l)*t3_abbabb(e,a,b,m,k,i)
    triples_res += -1.000000000000010 * einsum('mled,dcjl,eabmki->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,c,j,l)*t3_bbbbbb(e,a,b,i,k,m)
    triples_res += -1.000000000000010 * einsum('mlde,dcjl,eabikm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_aaaa*t2_abab(d,c,l,i)*t3_abbabb(e,a,b,m,k,j)
    triples_res +=  1.000000000000010 * einsum('mlde,dcli,eabmkj->abcijk', g_aaaa[o, o, v, v], t2_abab, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t2_abab(d,c,l,i)*t3_bbbbbb(e,a,b,j,k,m)
    triples_res +=  1.000000000000010 * einsum('lmde,dcli,eabjkm->abcijk', g_abab[o, o, v, v], t2_abab, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t2_bbbb(d,c,i,l)*t3_abbabb(e,a,b,m,k,j)
    triples_res +=  1.000000000000010 * einsum('mled,dcil,eabmkj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t2_bbbb(d,c,i,l)*t3_bbbbbb(e,a,b,j,k,m)
    triples_res +=  1.000000000000010 * einsum('mlde,dcil,eabjkm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_bbbb(d,c,j,k)*t3_abbabb(e,a,b,m,i,l)
    triples_res += -0.500000000000000 * einsum('mled,dcjk,eabmil->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_bbbb(d,c,j,k)*t3_abbabb(e,a,b,l,m,i)
    triples_res +=  0.500000000000000 * einsum('lmed,dcjk,eablmi->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,c,j,k)*t3_bbbbbb(e,a,b,i,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dcjk,eabiml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_bbbb(d,c,i,k)*t3_abbabb(e,a,b,m,j,l)
    triples_res +=  0.500000000000000 * einsum('mled,dcik,eabmjl->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,m||e,d>_abab*t2_bbbb(d,c,i,k)*t3_abbabb(e,a,b,l,m,j)
    triples_res += -0.500000000000000 * einsum('lmed,dcik,eablmj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,c,i,k)*t3_bbbbbb(e,a,b,j,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dcik,eabjml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_bbbb(d,c,i,j)*t3_abbabb(e,a,b,m,k,l)
    triples_res += -0.500000000000000 * einsum('mled,dcij,eabmkl->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,m||e,d>_abab*t2_bbbb(d,c,i,j)*t3_abbabb(e,a,b,l,m,k)
    triples_res +=  0.500000000000000 * einsum('lmed,dcij,eablmk->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(d,c,i,j)*t3_bbbbbb(e,a,b,k,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dcij,eabkml->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_bbbb*t2_bbbb(a,b,m,l)*t3_bbbbbb(d,e,c,i,j,k)
    triples_res +=  0.250000000000000 * einsum('mlde,abml,decijk->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_bbbb(a,b,k,l)*t3_abbabb(d,e,c,m,j,i)
    triples_res +=  0.500000000000000 * einsum('mlde,abkl,decmji->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_bbbb(a,b,k,l)*t3_abbabb(e,d,c,m,j,i)
    triples_res +=  0.500000000000000 * einsum('mled,abkl,edcmji->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(a,b,k,l)*t3_bbbbbb(d,e,c,i,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,abkl,decijm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_bbbb(a,b,j,l)*t3_abbabb(d,e,c,m,k,i)
    triples_res += -0.500000000000000 * einsum('mlde,abjl,decmki->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_bbbb(a,b,j,l)*t3_abbabb(e,d,c,m,k,i)
    triples_res += -0.500000000000000 * einsum('mled,abjl,edcmki->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(a,b,j,l)*t3_bbbbbb(d,e,c,i,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,abjl,decikm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_bbbb(a,b,i,l)*t3_abbabb(d,e,c,m,k,j)
    triples_res +=  0.500000000000000 * einsum('mlde,abil,decmkj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_bbbb(a,b,i,l)*t3_abbabb(e,d,c,m,k,j)
    triples_res +=  0.500000000000000 * einsum('mled,abil,edcmkj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(a,b,i,l)*t3_bbbbbb(d,e,c,j,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,abil,decjkm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.2500 <m,l||d,e>_bbbb*t2_bbbb(a,c,m,l)*t3_bbbbbb(d,e,b,i,j,k)
    triples_res += -0.250000000000000 * einsum('mlde,acml,debijk->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_bbbb(a,c,k,l)*t3_abbabb(d,e,b,m,j,i)
    triples_res += -0.500000000000000 * einsum('mlde,ackl,debmji->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_bbbb(a,c,k,l)*t3_abbabb(e,d,b,m,j,i)
    triples_res += -0.500000000000000 * einsum('mled,ackl,edbmji->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(a,c,k,l)*t3_bbbbbb(d,e,b,i,j,m)
    triples_res +=  0.500000000000000 * einsum('mlde,ackl,debijm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_bbbb(a,c,j,l)*t3_abbabb(d,e,b,m,k,i)
    triples_res +=  0.500000000000000 * einsum('mlde,acjl,debmki->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_bbbb(a,c,j,l)*t3_abbabb(e,d,b,m,k,i)
    triples_res +=  0.500000000000000 * einsum('mled,acjl,edbmki->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(a,c,j,l)*t3_bbbbbb(d,e,b,i,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,acjl,debikm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_bbbb(a,c,i,l)*t3_abbabb(d,e,b,m,k,j)
    triples_res += -0.500000000000000 * einsum('mlde,acil,debmkj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_bbbb(a,c,i,l)*t3_abbabb(e,d,b,m,k,j)
    triples_res += -0.500000000000000 * einsum('mled,acil,edbmkj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(a,c,i,l)*t3_bbbbbb(d,e,b,j,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,acil,debjkm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.2500 <m,l||d,e>_bbbb*t2_bbbb(b,c,m,l)*t3_bbbbbb(d,e,a,i,j,k)
    triples_res +=  0.250000000000000 * einsum('mlde,bcml,deaijk->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_bbbb(b,c,k,l)*t3_abbabb(d,e,a,m,j,i)
    triples_res +=  0.500000000000000 * einsum('mlde,bckl,deamji->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_bbbb(b,c,k,l)*t3_abbabb(e,d,a,m,j,i)
    triples_res +=  0.500000000000000 * einsum('mled,bckl,edamji->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(b,c,k,l)*t3_bbbbbb(d,e,a,i,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,bckl,deaijm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_abab*t2_bbbb(b,c,j,l)*t3_abbabb(d,e,a,m,k,i)
    triples_res += -0.500000000000000 * einsum('mlde,bcjl,deamki->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||e,d>_abab*t2_bbbb(b,c,j,l)*t3_abbabb(e,d,a,m,k,i)
    triples_res += -0.500000000000000 * einsum('mled,bcjl,edamki->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t2_bbbb(b,c,j,l)*t3_bbbbbb(d,e,a,i,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,bcjl,deaikm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_abab*t2_bbbb(b,c,i,l)*t3_abbabb(d,e,a,m,k,j)
    triples_res +=  0.500000000000000 * einsum('mlde,bcil,deamkj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||e,d>_abab*t2_bbbb(b,c,i,l)*t3_abbabb(e,d,a,m,k,j)
    triples_res +=  0.500000000000000 * einsum('mled,bcil,edamkj->abcijk', g_abab[o, o, v, v], t2_bbbb, t3_abbabb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t2_bbbb(b,c,i,l)*t3_bbbbbb(d,e,a,j,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,bcil,deajkm->abcijk', g_bbbb[o, o, v, v], t2_bbbb, t3_bbbbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t2_abab(d,a,l,j)*t2_bbbb(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,dalj,bcim->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t2_bbbb(d,a,j,l)*t2_bbbb(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dajl,bcim->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t2_abab(d,a,l,i)*t2_bbbb(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmdk,dali,bcjm->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t2_bbbb(d,a,i,l)*t2_bbbb(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,dail,bcjm->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,k>_bbbb*t2_bbbb(d,a,i,j)*t2_bbbb(b,c,m,l)
    triples_res += -0.500000000000000 * einsum('mldk,daij,bcml->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t2_abab(d,b,l,j)*t2_bbbb(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmdk,dblj,acim->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t2_bbbb(d,b,j,l)*t2_bbbb(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mldk,dbjl,acim->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t2_abab(d,b,l,i)*t2_bbbb(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,dbli,acjm->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t2_bbbb(d,b,i,l)*t2_bbbb(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dbil,acjm->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,k>_bbbb*t2_bbbb(d,b,i,j)*t2_bbbb(a,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mldk,dbij,acml->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,k>_abab*t2_abab(d,c,l,j)*t2_bbbb(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('lmdk,dclj,abim->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t2_bbbb(d,c,j,l)*t2_bbbb(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dcjl,abim->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,k>_abab*t2_abab(d,c,l,i)*t2_bbbb(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('lmdk,dcli,abjm->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t2_bbbb(d,c,i,l)*t2_bbbb(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,dcil,abjm->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,k>_bbbb*t2_bbbb(d,c,i,j)*t2_bbbb(a,b,m,l)
    triples_res += -0.500000000000000 * einsum('mldk,dcij,abml->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,j>_abab*t2_abab(d,a,l,k)*t2_bbbb(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmdj,dalk,bcim->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t2_bbbb(d,a,k,l)*t2_bbbb(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('mldj,dakl,bcim->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,j>_abab*t2_abab(d,a,l,i)*t2_bbbb(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('lmdj,dali,bckm->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_bbbb*t2_bbbb(d,a,i,l)*t2_bbbb(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mldj,dail,bckm->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,j>_bbbb*t2_bbbb(d,a,i,k)*t2_bbbb(b,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mldj,daik,bcml->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,j>_abab*t2_abab(d,b,l,k)*t2_bbbb(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmdj,dblk,acim->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_bbbb*t2_bbbb(d,b,k,l)*t2_bbbb(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mldj,dbkl,acim->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,j>_abab*t2_abab(d,b,l,i)*t2_bbbb(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmdj,dbli,ackm->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t2_bbbb(d,b,i,l)*t2_bbbb(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('mldj,dbil,ackm->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,j>_bbbb*t2_bbbb(d,b,i,k)*t2_bbbb(a,c,m,l)
    triples_res += -0.500000000000000 * einsum('mldj,dbik,acml->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,j>_abab*t2_abab(d,c,l,k)*t2_bbbb(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('lmdj,dclk,abim->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t2_bbbb(d,c,k,l)*t2_bbbb(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mldj,dckl,abim->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,j>_abab*t2_abab(d,c,l,i)*t2_bbbb(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('lmdj,dcli,abkm->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_bbbb*t2_bbbb(d,c,i,l)*t2_bbbb(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mldj,dcil,abkm->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,j>_bbbb*t2_bbbb(d,c,i,k)*t2_bbbb(a,b,m,l)
    triples_res +=  0.500000000000000 * einsum('mldj,dcik,abml->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,i>_abab*t2_abab(d,a,l,k)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmdi,dalk,bcjm->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_bbbb*t2_bbbb(d,a,k,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dakl,bcjm->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,i>_abab*t2_abab(d,a,l,j)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmdi,dalj,bckm->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_bbbb*t2_bbbb(d,a,j,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mldi,dajl,bckm->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,i>_bbbb*t2_bbbb(d,a,j,k)*t2_bbbb(b,c,m,l)
    triples_res += -0.500000000000000 * einsum('mldi,dajk,bcml->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,i>_abab*t2_abab(d,b,l,k)*t2_bbbb(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmdi,dblk,acjm->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_bbbb*t2_bbbb(d,b,k,l)*t2_bbbb(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('mldi,dbkl,acjm->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,m||d,i>_abab*t2_abab(d,b,l,j)*t2_bbbb(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('lmdi,dblj,ackm->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_bbbb*t2_bbbb(d,b,j,l)*t2_bbbb(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dbjl,ackm->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <m,l||d,i>_bbbb*t2_bbbb(d,b,j,k)*t2_bbbb(a,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mldi,dbjk,acml->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,i>_abab*t2_abab(d,c,l,k)*t2_bbbb(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('lmdi,dclk,abjm->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_bbbb*t2_bbbb(d,c,k,l)*t2_bbbb(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dckl,abjm->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,m||d,i>_abab*t2_abab(d,c,l,j)*t2_bbbb(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('lmdi,dclj,abkm->abcijk', g_abab[o, o, v, o], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_bbbb*t2_bbbb(d,c,j,l)*t2_bbbb(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('mldi,dcjl,abkm->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <m,l||d,i>_bbbb*t2_bbbb(d,c,j,k)*t2_bbbb(a,b,m,l)
    triples_res += -0.500000000000000 * einsum('mldi,dcjk,abml->abcijk', g_bbbb[o, o, v, o], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,a||d,e>_bbbb*t2_bbbb(d,e,j,k)*t2_bbbb(b,c,i,l)
    triples_res += -0.500000000000000 * einsum('lade,dejk,bcil->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,a||d,e>_bbbb*t2_bbbb(d,e,i,k)*t2_bbbb(b,c,j,l)
    triples_res +=  0.500000000000000 * einsum('lade,deik,bcjl->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,a||d,e>_bbbb*t2_bbbb(d,e,i,j)*t2_bbbb(b,c,k,l)
    triples_res += -0.500000000000000 * einsum('lade,deij,bckl->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_abab*t2_abab(d,b,l,k)*t2_bbbb(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('lade,dblk,ecij->abcijk', g_abab[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_bbbb*t2_bbbb(d,b,k,l)*t2_bbbb(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('lade,dbkl,ecij->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_abab*t2_abab(d,b,l,j)*t2_bbbb(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('lade,dblj,ecik->abcijk', g_abab[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_bbbb*t2_bbbb(d,b,j,l)*t2_bbbb(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('lade,dbjl,ecik->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_abab*t2_abab(d,b,l,i)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('lade,dbli,ecjk->abcijk', g_abab[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_bbbb*t2_bbbb(d,b,i,l)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lade,dbil,ecjk->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,a||e,d>_abab*t2_bbbb(d,b,j,k)*t2_abab(e,c,l,i)
    triples_res +=  1.000000000000000 * einsum('laed,dbjk,ecli->abcijk', g_abab[o, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_bbbb*t2_bbbb(d,b,j,k)*t2_bbbb(e,c,i,l)
    triples_res +=  1.000000000000000 * einsum('lade,dbjk,ecil->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||e,d>_abab*t2_bbbb(d,b,i,k)*t2_abab(e,c,l,j)
    triples_res += -1.000000000000000 * einsum('laed,dbik,eclj->abcijk', g_abab[o, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_bbbb*t2_bbbb(d,b,i,k)*t2_bbbb(e,c,j,l)
    triples_res += -1.000000000000000 * einsum('lade,dbik,ecjl->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||e,d>_abab*t2_bbbb(d,b,i,j)*t2_abab(e,c,l,k)
    triples_res +=  1.000000000000000 * einsum('laed,dbij,eclk->abcijk', g_abab[o, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_bbbb*t2_bbbb(d,b,i,j)*t2_bbbb(e,c,k,l)
    triples_res +=  1.000000000000000 * einsum('lade,dbij,eckl->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <l,b||d,e>_bbbb*t2_bbbb(d,e,j,k)*t2_bbbb(a,c,i,l)
    triples_res +=  0.500000000000000 * einsum('lbde,dejk,acil->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,b||d,e>_bbbb*t2_bbbb(d,e,i,k)*t2_bbbb(a,c,j,l)
    triples_res += -0.500000000000000 * einsum('lbde,deik,acjl->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,b||d,e>_bbbb*t2_bbbb(d,e,i,j)*t2_bbbb(a,c,k,l)
    triples_res +=  0.500000000000000 * einsum('lbde,deij,ackl->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_abab*t2_abab(d,a,l,k)*t2_bbbb(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('lbde,dalk,ecij->abcijk', g_abab[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t2_bbbb(d,a,k,l)*t2_bbbb(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('lbde,dakl,ecij->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_abab*t2_abab(d,a,l,j)*t2_bbbb(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('lbde,dalj,ecik->abcijk', g_abab[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_bbbb*t2_bbbb(d,a,j,l)*t2_bbbb(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('lbde,dajl,ecik->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_abab*t2_abab(d,a,l,i)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lbde,dali,ecjk->abcijk', g_abab[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t2_bbbb(d,a,i,l)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('lbde,dail,ecjk->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,b||e,d>_abab*t2_bbbb(d,a,j,k)*t2_abab(e,c,l,i)
    triples_res += -1.000000000000000 * einsum('lbed,dajk,ecli->abcijk', g_abab[o, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t2_bbbb(d,a,j,k)*t2_bbbb(e,c,i,l)
    triples_res += -1.000000000000000 * einsum('lbde,dajk,ecil->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||e,d>_abab*t2_bbbb(d,a,i,k)*t2_abab(e,c,l,j)
    triples_res +=  1.000000000000000 * einsum('lbed,daik,eclj->abcijk', g_abab[o, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_bbbb*t2_bbbb(d,a,i,k)*t2_bbbb(e,c,j,l)
    triples_res +=  1.000000000000000 * einsum('lbde,daik,ecjl->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||e,d>_abab*t2_bbbb(d,a,i,j)*t2_abab(e,c,l,k)
    triples_res += -1.000000000000000 * einsum('lbed,daij,eclk->abcijk', g_abab[o, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t2_bbbb(d,a,i,j)*t2_bbbb(e,c,k,l)
    triples_res += -1.000000000000000 * einsum('lbde,daij,eckl->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -0.5000 <l,c||d,e>_bbbb*t2_bbbb(d,e,j,k)*t2_bbbb(a,b,i,l)
    triples_res += -0.500000000000000 * einsum('lcde,dejk,abil->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <l,c||d,e>_bbbb*t2_bbbb(d,e,i,k)*t2_bbbb(a,b,j,l)
    triples_res +=  0.500000000000000 * einsum('lcde,deik,abjl->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <l,c||d,e>_bbbb*t2_bbbb(d,e,i,j)*t2_bbbb(a,b,k,l)
    triples_res += -0.500000000000000 * einsum('lcde,deij,abkl->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_abab*t2_abab(d,a,l,k)*t2_bbbb(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('lcde,dalk,ebij->abcijk', g_abab[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t2_bbbb(d,a,k,l)*t2_bbbb(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('lcde,dakl,ebij->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_abab*t2_abab(d,a,l,j)*t2_bbbb(e,b,i,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dalj,ebik->abcijk', g_abab[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t2_bbbb(d,a,j,l)*t2_bbbb(e,b,i,k)
    triples_res += -1.000000000000000 * einsum('lcde,dajl,ebik->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_abab*t2_abab(d,a,l,i)*t2_bbbb(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('lcde,dali,ebjk->abcijk', g_abab[o, v, v, v], t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t2_bbbb(d,a,i,l)*t2_bbbb(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dail,ebjk->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <l,c||e,d>_abab*t2_bbbb(d,a,j,k)*t2_abab(e,b,l,i)
    triples_res +=  1.000000000000000 * einsum('lced,dajk,ebli->abcijk', g_abab[o, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t2_bbbb(d,a,j,k)*t2_bbbb(e,b,i,l)
    triples_res +=  1.000000000000000 * einsum('lcde,dajk,ebil->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||e,d>_abab*t2_bbbb(d,a,i,k)*t2_abab(e,b,l,j)
    triples_res += -1.000000000000000 * einsum('lced,daik,eblj->abcijk', g_abab[o, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t2_bbbb(d,a,i,k)*t2_bbbb(e,b,j,l)
    triples_res += -1.000000000000000 * einsum('lcde,daik,ebjl->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||e,d>_abab*t2_bbbb(d,a,i,j)*t2_abab(e,b,l,k)
    triples_res +=  1.000000000000000 * einsum('lced,daij,eblk->abcijk', g_abab[o, v, v, v], t2_bbbb, t2_abab, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t2_bbbb(d,a,i,j)*t2_bbbb(e,b,k,l)
    triples_res +=  1.000000000000000 * einsum('lcde,daij,ebkl->abcijk', g_bbbb[o, v, v, v], t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_bbbb(e,a,j,k)*t2_bbbb(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmde,dl,eajk,bcim->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_bbbb(e,a,j,k)*t2_bbbb(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,eajk,bcim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_bbbb(e,a,i,k)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dl,eaik,bcjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_bbbb(e,a,i,k)*t2_bbbb(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,eaik,bcjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_bbbb(e,a,i,j)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,dl,eaij,bckm->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_bbbb(e,a,i,j)*t2_bbbb(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,eaij,bckm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_bbbb(e,b,j,k)*t2_bbbb(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dl,ebjk,acim->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_bbbb(e,b,j,k)*t2_bbbb(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ebjk,acim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_bbbb(e,b,i,k)*t2_bbbb(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmde,dl,ebik,acjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_bbbb(e,b,i,k)*t2_bbbb(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ebik,acjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_bbbb(e,b,i,j)*t2_bbbb(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dl,ebij,ackm->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_bbbb(e,b,i,j)*t2_bbbb(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ebij,ackm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_bbbb(e,c,j,k)*t2_bbbb(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('lmde,dl,ecjk,abim->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_bbbb(e,c,j,k)*t2_bbbb(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ecjk,abim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_bbbb(e,c,i,k)*t2_bbbb(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dl,ecik,abjm->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_bbbb(e,c,i,k)*t2_bbbb(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ecik,abjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t2_bbbb(e,c,i,j)*t2_bbbb(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,dl,ecij,abkm->abcijk', g_abab[o, o, v, v], t1_aa, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t2_bbbb(e,c,i,j)*t2_bbbb(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ecij,abkm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t2_abab(e,a,l,j)*t2_bbbb(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,ealj,bcim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,a,j,l)*t2_bbbb(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,eajl,bcim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t2_abab(e,a,l,i)*t2_bbbb(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmed,dk,eali,bcjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,a,i,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,eail,bcjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,a,i,j)*t2_bbbb(b,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dk,eaij,bcml->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t2_abab(e,b,l,j)*t2_bbbb(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmed,dk,eblj,acim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,b,j,l)*t2_bbbb(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,ebjl,acim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t2_abab(e,b,l,i)*t2_bbbb(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,ebli,acjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,b,i,l)*t2_bbbb(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,ebil,acjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,b,i,j)*t2_bbbb(a,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dk,ebij,acml->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t2_abab(e,c,l,j)*t2_bbbb(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dk,eclj,abim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,c,j,l)*t2_bbbb(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,ecjl,abim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,k)*t2_abab(e,c,l,i)*t2_bbbb(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('lmed,dk,ecli,abjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,c,i,l)*t2_bbbb(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,ecil,abjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(d,k)*t2_bbbb(e,c,i,j)*t2_bbbb(a,b,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dk,ecij,abml->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,j)*t2_abab(e,a,l,k)*t2_bbbb(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('lmed,dj,ealk,bcim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,a,k,l)*t2_bbbb(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,eakl,bcim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,j)*t2_abab(e,a,l,i)*t2_bbbb(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dj,eali,bckm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,a,i,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,eail,bckm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,a,i,k)*t2_bbbb(b,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dj,eaik,bcml->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,j)*t2_abab(e,b,l,k)*t2_bbbb(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dj,eblk,acim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,b,k,l)*t2_bbbb(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,ebkl,acim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,j)*t2_abab(e,b,l,i)*t2_bbbb(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmed,dj,ebli,ackm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,b,i,l)*t2_bbbb(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,ebil,ackm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,b,i,k)*t2_bbbb(a,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dj,ebik,acml->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,j)*t2_abab(e,c,l,k)*t2_bbbb(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('lmed,dj,eclk,abim->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,c,k,l)*t2_bbbb(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,eckl,abim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,j)*t2_abab(e,c,l,i)*t2_bbbb(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('lmed,dj,ecli,abkm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,c,i,l)*t2_bbbb(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,ecil,abkm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(d,j)*t2_bbbb(e,c,i,k)*t2_bbbb(a,b,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dj,ecik,abml->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,i)*t2_abab(e,a,l,k)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('lmed,di,ealk,bcjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,i)*t2_bbbb(e,a,k,l)*t2_bbbb(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,di,eakl,bcjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,i)*t2_abab(e,a,l,j)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('lmed,di,ealj,bckm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,i)*t2_bbbb(e,a,j,l)*t2_bbbb(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,eajl,bckm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(d,i)*t2_bbbb(e,a,j,k)*t2_bbbb(b,c,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,di,eajk,bcml->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,i)*t2_abab(e,b,l,k)*t2_bbbb(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('lmed,di,eblk,acjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,i)*t2_bbbb(e,b,k,l)*t2_bbbb(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,ebkl,acjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,i)*t2_abab(e,b,l,j)*t2_bbbb(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('lmed,di,eblj,ackm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,i)*t2_bbbb(e,b,j,l)*t2_bbbb(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,di,ebjl,ackm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(d,i)*t2_bbbb(e,b,j,k)*t2_bbbb(a,c,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,di,ebjk,acml->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <l,m||e,d>_abab*t1_bb(d,i)*t2_abab(e,c,l,k)*t2_bbbb(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('lmed,di,eclk,abjm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,i)*t2_bbbb(e,c,k,l)*t2_bbbb(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,di,eckl,abjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||e,d>_abab*t1_bb(d,i)*t2_abab(e,c,l,j)*t2_bbbb(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('lmed,di,eclj,abkm->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,i)*t2_bbbb(e,c,j,l)*t2_bbbb(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,ecjl,abkm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(d,i)*t2_bbbb(e,c,j,k)*t2_bbbb(a,b,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,di,ecjk,abml->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(d,e,j,k)*t2_bbbb(b,c,i,m)
    triples_res +=  0.500000000000000 * einsum('mlde,al,dejk,bcim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(d,e,i,k)*t2_bbbb(b,c,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,al,deik,bcjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(d,e,i,j)*t2_bbbb(b,c,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,al,deij,bckm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_bb(a,l)*t2_abab(d,b,m,k)*t2_bbbb(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('mlde,al,dbmk,ecij->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,k,m)*t2_bbbb(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('mlde,al,dbkm,ecij->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_bb(a,l)*t2_abab(d,b,m,j)*t2_bbbb(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('mlde,al,dbmj,ecik->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,j,m)*t2_bbbb(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('mlde,al,dbjm,ecik->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_bb(a,l)*t2_abab(d,b,m,i)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,al,dbmi,ecjk->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,i,m)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,al,dbim,ecjk->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(a,l)*t2_bbbb(d,b,j,k)*t2_abab(e,c,m,i)
    triples_res += -1.000000000000000 * einsum('mled,al,dbjk,ecmi->abcijk', g_abab[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,j,k)*t2_bbbb(e,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,al,dbjk,ecim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(a,l)*t2_bbbb(d,b,i,k)*t2_abab(e,c,m,j)
    triples_res +=  1.000000000000000 * einsum('mled,al,dbik,ecmj->abcijk', g_abab[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,i,k)*t2_bbbb(e,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,al,dbik,ecjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(a,l)*t2_bbbb(d,b,i,j)*t2_abab(e,c,m,k)
    triples_res += -1.000000000000000 * einsum('mled,al,dbij,ecmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(a,l)*t2_bbbb(d,b,i,j)*t2_bbbb(e,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,al,dbij,eckm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(b,l)*t2_bbbb(d,e,j,k)*t2_bbbb(a,c,i,m)
    triples_res += -0.500000000000000 * einsum('mlde,bl,dejk,acim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(b,l)*t2_bbbb(d,e,i,k)*t2_bbbb(a,c,j,m)
    triples_res +=  0.500000000000000 * einsum('mlde,bl,deik,acjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(b,l)*t2_bbbb(d,e,i,j)*t2_bbbb(a,c,k,m)
    triples_res += -0.500000000000000 * einsum('mlde,bl,deij,ackm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_bb(b,l)*t2_abab(d,a,m,k)*t2_bbbb(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('mlde,bl,damk,ecij->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(b,l)*t2_bbbb(d,a,k,m)*t2_bbbb(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('mlde,bl,dakm,ecij->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_bb(b,l)*t2_abab(d,a,m,j)*t2_bbbb(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('mlde,bl,damj,ecik->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(b,l)*t2_bbbb(d,a,j,m)*t2_bbbb(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('mlde,bl,dajm,ecik->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_bb(b,l)*t2_abab(d,a,m,i)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,bl,dami,ecjk->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(b,l)*t2_bbbb(d,a,i,m)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,bl,daim,ecjk->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(b,l)*t2_bbbb(d,a,j,k)*t2_abab(e,c,m,i)
    triples_res +=  1.000000000000000 * einsum('mled,bl,dajk,ecmi->abcijk', g_abab[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(b,l)*t2_bbbb(d,a,j,k)*t2_bbbb(e,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,bl,dajk,ecim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(b,l)*t2_bbbb(d,a,i,k)*t2_abab(e,c,m,j)
    triples_res += -1.000000000000000 * einsum('mled,bl,daik,ecmj->abcijk', g_abab[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(b,l)*t2_bbbb(d,a,i,k)*t2_bbbb(e,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,bl,daik,ecjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(b,l)*t2_bbbb(d,a,i,j)*t2_abab(e,c,m,k)
    triples_res +=  1.000000000000000 * einsum('mled,bl,daij,ecmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(b,l)*t2_bbbb(d,a,i,j)*t2_bbbb(e,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,bl,daij,eckm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_bbbb(d,e,j,k)*t2_bbbb(a,b,i,m)
    triples_res +=  0.500000000000000 * einsum('mlde,cl,dejk,abim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_bbbb(d,e,i,k)*t2_bbbb(a,b,j,m)
    triples_res += -0.500000000000000 * einsum('mlde,cl,deik,abjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_bbbb(d,e,i,j)*t2_bbbb(a,b,k,m)
    triples_res +=  0.500000000000000 * einsum('mlde,cl,deij,abkm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_bb(c,l)*t2_abab(d,a,m,k)*t2_bbbb(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('mlde,cl,damk,ebij->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_bbbb(d,a,k,m)*t2_bbbb(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('mlde,cl,dakm,ebij->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_abab*t1_bb(c,l)*t2_abab(d,a,m,j)*t2_bbbb(e,b,i,k)
    triples_res += -1.000000000000000 * einsum('mlde,cl,damj,ebik->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_bbbb(d,a,j,m)*t2_bbbb(e,b,i,k)
    triples_res +=  1.000000000000000 * einsum('mlde,cl,dajm,ebik->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_abab*t1_bb(c,l)*t2_abab(d,a,m,i)*t2_bbbb(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,cl,dami,ebjk->abcijk', g_abab[o, o, v, v], t1_bb, t2_abab, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_bbbb(d,a,i,m)*t2_bbbb(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,cl,daim,ebjk->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(c,l)*t2_bbbb(d,a,j,k)*t2_abab(e,b,m,i)
    triples_res += -1.000000000000000 * einsum('mled,cl,dajk,ebmi->abcijk', g_abab[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_bbbb(d,a,j,k)*t2_bbbb(e,b,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,cl,dajk,ebim->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(c,l)*t2_bbbb(d,a,i,k)*t2_abab(e,b,m,j)
    triples_res +=  1.000000000000000 * einsum('mled,cl,daik,ebmj->abcijk', g_abab[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_bbbb(d,a,i,k)*t2_bbbb(e,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,cl,daik,ebjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(c,l)*t2_bbbb(d,a,i,j)*t2_abab(e,b,m,k)
    triples_res += -1.000000000000000 * einsum('mled,cl,daij,ebmk->abcijk', g_abab[o, o, v, v], t1_bb, t2_bbbb, t2_abab, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(c,l)*t2_bbbb(d,a,i,j)*t2_bbbb(e,b,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,cl,daij,ebkm->abcijk', g_bbbb[o, o, v, v], t1_bb, t2_bbbb, t2_bbbb, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t1_bb(d,j)*t1_bb(a,l)*t2_bbbb(b,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dj,al,bcim->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t1_bb(d,j)*t1_bb(b,l)*t2_bbbb(a,c,i,m)
    triples_res += -1.000000000000000 * einsum('mldk,dj,bl,acim->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t1_bb(d,j)*t1_bb(c,l)*t2_bbbb(a,b,i,m)
    triples_res +=  1.000000000000000 * einsum('mldk,dj,cl,abim->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t1_bb(d,i)*t1_bb(a,l)*t2_bbbb(b,c,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,di,al,bcjm->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t1_bb(d,i)*t1_bb(b,l)*t2_bbbb(a,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mldk,di,bl,acjm->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t1_bb(d,i)*t1_bb(c,l)*t2_bbbb(a,b,j,m)
    triples_res += -1.000000000000000 * einsum('mldk,di,cl,abjm->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t1_bb(a,l)*t1_bb(b,m)*t2_bbbb(d,c,i,j)
    triples_res +=  1.000000000000000 * einsum('mldk,al,bm,dcij->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,k>_bbbb*t1_bb(a,l)*t1_bb(c,m)*t2_bbbb(d,b,i,j)
    triples_res += -1.000000000000000 * einsum('mldk,al,cm,dbij->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,k>_bbbb*t1_bb(b,l)*t1_bb(c,m)*t2_bbbb(d,a,i,j)
    triples_res +=  1.000000000000000 * einsum('mldk,bl,cm,daij->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t1_bb(d,k)*t1_bb(a,l)*t2_bbbb(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('mldj,dk,al,bcim->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_bbbb*t1_bb(d,k)*t1_bb(b,l)*t2_bbbb(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mldj,dk,bl,acim->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t1_bb(d,k)*t1_bb(c,l)*t2_bbbb(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mldj,dk,cl,abim->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_bbbb*t1_bb(d,i)*t1_bb(a,l)*t2_bbbb(b,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mldj,di,al,bckm->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t1_bb(d,i)*t1_bb(b,l)*t2_bbbb(a,c,k,m)
    triples_res += -1.000000000000000 * einsum('mldj,di,bl,ackm->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_bbbb*t1_bb(d,i)*t1_bb(c,l)*t2_bbbb(a,b,k,m)
    triples_res +=  1.000000000000000 * einsum('mldj,di,cl,abkm->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t1_bb(a,l)*t1_bb(b,m)*t2_bbbb(d,c,i,k)
    triples_res += -1.000000000000000 * einsum('mldj,al,bm,dcik->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,j>_bbbb*t1_bb(a,l)*t1_bb(c,m)*t2_bbbb(d,b,i,k)
    triples_res +=  1.000000000000000 * einsum('mldj,al,cm,dbik->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,j>_bbbb*t1_bb(b,l)*t1_bb(c,m)*t2_bbbb(d,a,i,k)
    triples_res += -1.000000000000000 * einsum('mldj,bl,cm,daik->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_bbbb*t1_bb(d,k)*t1_bb(a,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dk,al,bcjm->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_bbbb*t1_bb(d,k)*t1_bb(b,l)*t2_bbbb(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('mldi,dk,bl,acjm->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_bbbb*t1_bb(d,k)*t1_bb(c,l)*t2_bbbb(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dk,cl,abjm->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_bbbb*t1_bb(d,j)*t1_bb(a,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mldi,dj,al,bckm->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_bbbb*t1_bb(d,j)*t1_bb(b,l)*t2_bbbb(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mldi,dj,bl,ackm->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_bbbb*t1_bb(d,j)*t1_bb(c,l)*t2_bbbb(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('mldi,dj,cl,abkm->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_bbbb*t1_bb(a,l)*t1_bb(b,m)*t2_bbbb(d,c,j,k)
    triples_res +=  1.000000000000000 * einsum('mldi,al,bm,dcjk->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,i>_bbbb*t1_bb(a,l)*t1_bb(c,m)*t2_bbbb(d,b,j,k)
    triples_res += -1.000000000000000 * einsum('mldi,al,cm,dbjk->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,i>_bbbb*t1_bb(b,l)*t1_bb(c,m)*t2_bbbb(d,a,j,k)
    triples_res +=  1.000000000000000 * einsum('mldi,bl,cm,dajk->abcijk', g_bbbb[o, o, v, o], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,j)*t2_bbbb(b,c,i,l)
    triples_res +=  1.000000000000000 * einsum('lade,dk,ej,bcil->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,i)*t2_bbbb(b,c,j,l)
    triples_res += -1.000000000000000 * einsum('lade,dk,ei,bcjl->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_bbbb*t1_bb(d,k)*t1_bb(b,l)*t2_bbbb(e,c,i,j)
    triples_res +=  1.000000000000000 * einsum('lade,dk,bl,ecij->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_bbbb*t1_bb(d,k)*t1_bb(c,l)*t2_bbbb(e,b,i,j)
    triples_res += -1.000000000000000 * einsum('lade,dk,cl,ebij->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_bbbb*t1_bb(d,j)*t1_bb(e,i)*t2_bbbb(b,c,k,l)
    triples_res +=  1.000000000000000 * einsum('lade,dj,ei,bckl->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_bbbb*t1_bb(d,j)*t1_bb(b,l)*t2_bbbb(e,c,i,k)
    triples_res += -1.000000000000000 * einsum('lade,dj,bl,ecik->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_bbbb*t1_bb(d,j)*t1_bb(c,l)*t2_bbbb(e,b,i,k)
    triples_res +=  1.000000000000000 * einsum('lade,dj,cl,ebik->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,a||d,e>_bbbb*t1_bb(d,i)*t1_bb(b,l)*t2_bbbb(e,c,j,k)
    triples_res +=  1.000000000000000 * einsum('lade,di,bl,ecjk->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,a||d,e>_bbbb*t1_bb(d,i)*t1_bb(c,l)*t2_bbbb(e,b,j,k)
    triples_res += -1.000000000000000 * einsum('lade,di,cl,ebjk->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,j)*t2_bbbb(a,c,i,l)
    triples_res += -1.000000000000000 * einsum('lbde,dk,ej,acil->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,i)*t2_bbbb(a,c,j,l)
    triples_res +=  1.000000000000000 * einsum('lbde,dk,ei,acjl->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t1_bb(d,k)*t1_bb(a,l)*t2_bbbb(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('lbde,dk,al,ecij->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_bbbb*t1_bb(d,k)*t1_bb(c,l)*t2_bbbb(e,a,i,j)
    triples_res +=  1.000000000000000 * einsum('lbde,dk,cl,eaij->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t1_bb(d,j)*t1_bb(e,i)*t2_bbbb(a,c,k,l)
    triples_res += -1.000000000000000 * einsum('lbde,dj,ei,ackl->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_bbbb*t1_bb(d,j)*t1_bb(a,l)*t2_bbbb(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('lbde,dj,al,ecik->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t1_bb(d,j)*t1_bb(c,l)*t2_bbbb(e,a,i,k)
    triples_res += -1.000000000000000 * einsum('lbde,dj,cl,eaik->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,b||d,e>_bbbb*t1_bb(d,i)*t1_bb(a,l)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('lbde,di,al,ecjk->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,b||d,e>_bbbb*t1_bb(d,i)*t1_bb(c,l)*t2_bbbb(e,a,j,k)
    triples_res +=  1.000000000000000 * einsum('lbde,di,cl,eajk->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,j)*t2_bbbb(a,b,i,l)
    triples_res +=  1.000000000000000 * einsum('lcde,dk,ej,abil->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,i)*t2_bbbb(a,b,j,l)
    triples_res += -1.000000000000000 * einsum('lcde,dk,ei,abjl->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t1_bb(d,k)*t1_bb(a,l)*t2_bbbb(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('lcde,dk,al,ebij->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t1_bb(d,k)*t1_bb(b,l)*t2_bbbb(e,a,i,j)
    triples_res += -1.000000000000000 * einsum('lcde,dk,bl,eaij->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t1_bb(d,j)*t1_bb(e,i)*t2_bbbb(a,b,k,l)
    triples_res +=  1.000000000000000 * einsum('lcde,dj,ei,abkl->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t1_bb(d,j)*t1_bb(a,l)*t2_bbbb(e,b,i,k)
    triples_res += -1.000000000000000 * einsum('lcde,dj,al,ebik->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t1_bb(d,j)*t1_bb(b,l)*t2_bbbb(e,a,i,k)
    triples_res +=  1.000000000000000 * einsum('lcde,dj,bl,eaik->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,c||d,e>_bbbb*t1_bb(d,i)*t1_bb(a,l)*t2_bbbb(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('lcde,di,al,ebjk->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,c||d,e>_bbbb*t1_bb(d,i)*t1_bb(b,l)*t2_bbbb(e,a,j,k)
    triples_res += -1.000000000000000 * einsum('lcde,di,bl,eajk->abcijk', g_bbbb[o, v, v, v], t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t1_bb(e,k)*t3_bbbbbb(a,b,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('lmde,dl,ek,abcijm->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t1_bb(e,k)*t3_bbbbbb(a,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ek,abcijm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t1_bb(e,j)*t3_bbbbbb(a,b,c,i,k,m)
    triples_res +=  1.000000000000000 * einsum('lmde,dl,ej,abcikm->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t1_bb(e,j)*t3_bbbbbb(a,b,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dl,ej,abcikm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t1_bb(e,i)*t3_bbbbbb(a,b,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('lmde,dl,ei,abcjkm->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t1_bb(e,i)*t3_bbbbbb(a,b,c,j,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,ei,abcjkm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t1_bb(a,m)*t3_bbbbbb(e,b,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('lmde,dl,am,ebcijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t1_bb(a,m)*t3_bbbbbb(e,b,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,am,ebcijk->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t1_bb(b,m)*t3_bbbbbb(e,a,c,i,j,k)
    triples_res +=  1.000000000000000 * einsum('lmde,dl,bm,eacijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t1_bb(b,m)*t3_bbbbbb(e,a,c,i,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,dl,bm,eacijk->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <l,m||d,e>_abab*t1_aa(d,l)*t1_bb(c,m)*t3_bbbbbb(e,a,b,i,j,k)
    triples_res += -1.000000000000000 * einsum('lmde,dl,cm,eabijk->abcijk', g_abab[o, o, v, v], t1_aa, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,l)*t1_bb(c,m)*t3_bbbbbb(e,a,b,i,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dl,cm,eabijk->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,j)*t3_bbbbbb(a,b,c,i,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dk,ej,abciml->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,i)*t3_bbbbbb(a,b,c,j,m,l)
    triples_res +=  0.500000000000000 * einsum('mlde,dk,ei,abcjml->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t1_bb(a,l)*t3_abbabb(e,b,c,m,j,i)
    triples_res +=  1.000000000000000 * einsum('mled,dk,al,ebcmji->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(a,l)*t3_bbbbbb(e,b,c,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,al,ebcijm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t1_bb(b,l)*t3_abbabb(e,a,c,m,j,i)
    triples_res += -1.000000000000000 * einsum('mled,dk,bl,eacmji->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(b,l)*t3_bbbbbb(e,a,c,i,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,bl,eacijm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,k)*t1_bb(c,l)*t3_abbabb(e,a,b,m,j,i)
    triples_res +=  1.000000000000000 * einsum('mled,dk,cl,eabmji->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(c,l)*t3_bbbbbb(e,a,b,i,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,cl,eabijm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(d,j)*t1_bb(e,i)*t3_bbbbbb(a,b,c,k,m,l)
    triples_res += -0.500000000000000 * einsum('mlde,dj,ei,abckml->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,j)*t1_bb(a,l)*t3_abbabb(e,b,c,m,k,i)
    triples_res += -1.000000000000000 * einsum('mled,dj,al,ebcmki->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t1_bb(a,l)*t3_bbbbbb(e,b,c,i,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,al,ebcikm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,j)*t1_bb(b,l)*t3_abbabb(e,a,c,m,k,i)
    triples_res +=  1.000000000000000 * einsum('mled,dj,bl,eacmki->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t1_bb(b,l)*t3_bbbbbb(e,a,c,i,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,bl,eacikm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,j)*t1_bb(c,l)*t3_abbabb(e,a,b,m,k,i)
    triples_res += -1.000000000000000 * einsum('mled,dj,cl,eabmki->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t1_bb(c,l)*t3_bbbbbb(e,a,b,i,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,cl,eabikm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,i)*t1_bb(a,l)*t3_abbabb(e,b,c,m,k,j)
    triples_res +=  1.000000000000000 * einsum('mled,di,al,ebcmkj->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,i)*t1_bb(a,l)*t3_bbbbbb(e,b,c,j,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,al,ebcjkm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||e,d>_abab*t1_bb(d,i)*t1_bb(b,l)*t3_abbabb(e,a,c,m,k,j)
    triples_res += -1.000000000000000 * einsum('mled,di,bl,eacmkj->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,i)*t1_bb(b,l)*t3_bbbbbb(e,a,c,j,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,di,bl,eacjkm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||e,d>_abab*t1_bb(d,i)*t1_bb(c,l)*t3_abbabb(e,a,b,m,k,j)
    triples_res +=  1.000000000000000 * einsum('mled,di,cl,eabmkj->abcijk', g_abab[o, o, v, v], t1_bb, t1_bb, t3_abbabb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,i)*t1_bb(c,l)*t3_bbbbbb(e,a,b,j,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,di,cl,eabjkm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(a,l)*t1_bb(b,m)*t3_bbbbbb(d,e,c,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,al,bm,decijk->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  0.5000 <m,l||d,e>_bbbb*t1_bb(a,l)*t1_bb(c,m)*t3_bbbbbb(d,e,b,i,j,k)
    triples_res +=  0.500000000000000 * einsum('mlde,al,cm,debijk->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <m,l||d,e>_bbbb*t1_bb(b,l)*t1_bb(c,m)*t3_bbbbbb(d,e,a,i,j,k)
    triples_res += -0.500000000000000 * einsum('mlde,bl,cm,deaijk->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t3_bbbbbb, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,j)*t1_bb(a,l)*t2_bbbb(b,c,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,ej,al,bcim->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,j)*t1_bb(b,l)*t2_bbbb(a,c,i,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,ej,bl,acim->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,j)*t1_bb(c,l)*t2_bbbb(a,b,i,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,ej,cl,abim->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,i)*t1_bb(a,l)*t2_bbbb(b,c,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,ei,al,bcjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,i)*t1_bb(b,l)*t2_bbbb(a,c,j,m)
    triples_res += -1.000000000000000 * einsum('mlde,dk,ei,bl,acjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(e,i)*t1_bb(c,l)*t2_bbbb(a,b,j,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,ei,cl,abjm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(a,l)*t1_bb(b,m)*t2_bbbb(e,c,i,j)
    triples_res += -1.000000000000000 * einsum('mlde,dk,al,bm,ecij->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(a,l)*t1_bb(c,m)*t2_bbbb(e,b,i,j)
    triples_res +=  1.000000000000000 * einsum('mlde,dk,al,cm,ebij->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,k)*t1_bb(b,l)*t1_bb(c,m)*t2_bbbb(e,a,i,j)
    triples_res += -1.000000000000000 * einsum('mlde,dk,bl,cm,eaij->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t1_bb(e,i)*t1_bb(a,l)*t2_bbbb(b,c,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,ei,al,bckm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t1_bb(e,i)*t1_bb(b,l)*t2_bbbb(a,c,k,m)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,ei,bl,ackm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t1_bb(e,i)*t1_bb(c,l)*t2_bbbb(a,b,k,m)
    triples_res += -1.000000000000000 * einsum('mlde,dj,ei,cl,abkm->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t1_bb(a,l)*t1_bb(b,m)*t2_bbbb(e,c,i,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,al,bm,ecik->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t1_bb(a,l)*t1_bb(c,m)*t2_bbbb(e,b,i,k)
    triples_res += -1.000000000000000 * einsum('mlde,dj,al,cm,ebik->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,j)*t1_bb(b,l)*t1_bb(c,m)*t2_bbbb(e,a,i,k)
    triples_res +=  1.000000000000000 * einsum('mlde,dj,bl,cm,eaik->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,i)*t1_bb(a,l)*t1_bb(b,m)*t2_bbbb(e,c,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,di,al,bm,ecjk->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 <m,l||d,e>_bbbb*t1_bb(d,i)*t1_bb(a,l)*t1_bb(c,m)*t2_bbbb(e,b,j,k)
    triples_res +=  1.000000000000000 * einsum('mlde,di,al,cm,ebjk->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 <m,l||d,e>_bbbb*t1_bb(d,i)*t1_bb(b,l)*t1_bb(c,m)*t2_bbbb(e,a,j,k)
    triples_res += -1.000000000000000 * einsum('mlde,di,bl,cm,eajk->abcijk', g_bbbb[o, o, v, v], t1_bb, t1_bb, t1_bb, t2_bbbb, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    return triples_res
    
