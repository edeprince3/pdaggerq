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


def ccsdt_energy(t1, t2, t3, f, g, o, v):

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


def singles_residual(t1, t2, t3, f, g, o, v):

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


def doubles_residual(t1, t2, t3, f, g, o, v):

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


def triples_residual(t1, t2, t3, f, g, o, v):

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
    old_energy = ccsdt_energy(t1, t2, t3, fock, g, o, v)

    print("")
    print("    ==> CCSDT amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(max_iter):

        residual_singles = singles_residual(t1, t2, t3, fock, g, o, v)
        residual_doubles = doubles_residual(t1, t2, t3, fock, g, o, v)
        residual_triples = triples_residual(t1, t2, t3, fock, g, o, v)

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

        current_energy = ccsdt_energy(new_singles, new_doubles, new_triples, fock, g, o, v)
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

