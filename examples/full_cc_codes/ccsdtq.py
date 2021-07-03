"""
A full working spin-orbital CCSDTQ code generated with pdaggerq

If you want to run the example here you should install pyscf openfermion
and openfermion-pyscf The actual CCSDTQ code (cc_energy, singles_residual,
doubles_residual, triples_residual, quadruples_residual, kernel) do not 
depend on those packages but you must obtain integrals from somehwere.

The total energy for this example been checked against that produced by
the CCSDTQ implementation in NWChem. Note that there must be a discrepancy
in the definition of the angstrom/bohr conversion ... agreement with
NWChem past 7 decimals can only be achieved if the geometry is defined
in bohr in that program.

 CCSDTQ correlation energy / hartree =        -0.179815934953076
 CCSDTQ total energy / hartree       =      -100.009723511692869

the main() function is fairly straightforward.
"""

# set allow numpy built with MKL to consume more threads for tensordot
import os
os.environ["MKL_NUM_THREADS"] = "{}".format(os.cpu_count() - 1)

import numpy as np
from numpy import einsum


def cc_energy(t1, t2, f, g, o, v):
    """
    < 0 | e(-T) H e(T) | 0> :


    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param f: fock operator defined as soei + np.einsum('piiq->pq', astei[:, o, o, :])
              where soei is 1 electron integrals (spinorb) and astei is
              antisymmetric 2 electron integrals in openfermion format
              <12|21>.  <ij|kl> - <ij|lk>
    :param g: antisymmetric 2 electron integrals. See fock input.
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    """

    #	  1.0000 f(i,i)
    energy = 1.0 * einsum('ii', f[o, o])

    #	  1.0000 f(i,a)*t1(a,i)
    energy += 1.0 * einsum('ia,ai', f[o, v], t1)

    #	 -0.5000 <j,i||j,i>
    energy += -0.5 * einsum('jiji', g[o, o, o, o])

    #	  0.2500 <j,i||a,b>*t2(a,b,j,i)
    energy += 0.25 * einsum('jiab,abji', g[o, o, v, v], t2)

    #	 -0.5000 <j,i||a,b>*t1(a,i)*t1(b,j)
    energy += -0.5 * einsum('jiab,ai,bj', g[o, o, v, v], t1, t1,
                            optimize=['einsum_path', (0, 1), (0, 1)])

    return energy


def singles_residual(t1, t2, t3, f, g, o, v):
    """
    < 0 | m* e e(-T) H e(T) | 0>

    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param t3: spin-orbital t3 amplitudes (nvirt x nvirt x nvirt x nocc x nocc x nocc)
    :param f: fock operator defined as soei + np.einsum('piiq->pq', astei[:, o, o, :])
              where soei is 1 electron integrals (spinorb) and astei is
              antisymmetric 2 electron integrals in openfermion format
              <12|21>.  <ij|kl> - <ij|lk>
    :param g: antisymmetric 2 electron integrals. See fock input.
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    """
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
    
    #	  1.0000 <k,j||b,i>*t1(b,j)*t1(a,k)
    singles_res +=  1.000000000000000 * einsum('kjbi,bj,ak->ai', g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,a||b,c>*t1(b,j)*t1(c,i)
    singles_res +=  1.000000000000000 * einsum('jabc,bj,ci->ai', g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <k,j||b,c>*t1(b,j)*t2(c,a,i,k)
    singles_res +=  1.000000000000000 * einsum('kjbc,bj,caik->ai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <k,j||b,c>*t1(b,i)*t2(c,a,k,j)
    singles_res +=  0.500000000000000 * einsum('kjbc,bi,cakj->ai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <k,j||b,c>*t1(a,j)*t2(b,c,i,k)
    singles_res +=  0.500000000000000 * einsum('kjbc,aj,bcik->ai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <k,j||b,c>*t1(b,j)*t1(c,i)*t1(a,k)
    singles_res +=  1.000000000000000 * einsum('kjbc,bj,ci,ak->ai', g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    return singles_res


def doubles_residual(t1, t2, t3, t4, f, g, o, v):
    """
     < 0 | m* n* f e e(-T) H e(T) | 0>

    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param t3: spin-orbital t3 amplitudes (nvirt x nvirt x nvirt x nocc x nocc x nocc)
    :param t4: spin-orbital t4 amplitudes (nvirt x nvirt x nvirt x nvirt x nocc x nocc x nocc x nocc)
    :param f: fock operator defined as soei + np.einsum('piiq->pq', astei[:, o, o, :])
              where soei is 1 electron integrals (spinorb) and astei is
              antisymmetric 2 electron integrals in openfermion format
              <12|21>.  <ij|kl> - <ij|lk>
    :param g: antisymmetric 2 electron integrals. See fock input.
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    """
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
    
    #	  0.2500 <l,k||c,d>*t4(c,d,a,b,i,j,l,k)
    doubles_res +=  0.250000000000000 * einsum('lkcd,cdabijlk->abij', g[o, o, v, v], t4[:, :, :, :, :, :, :, :])
    
    #	 -1.0000 <l,k||i,j>*t1(a,k)*t1(b,l)
    doubles_res += -1.000000000000000 * einsum('lkij,ak,bl->abij', g[o, o, o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(i,j)*P(a,b)<k,a||c,j>*t1(c,i)*t1(b,k)
    contracted_intermediate =  1.000000000000000 * einsum('kacj,ci,bk->abij', g[o, v, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate)  + -1.00000 * einsum('abij->baij', contracted_intermediate)  +  1.00000 * einsum('abij->baji', contracted_intermediate) 
    
    #	 -1.0000 <a,b||c,d>*t1(c,j)*t1(d,i)
    doubles_res += -1.000000000000000 * einsum('abcd,cj,di->abij', g[v, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
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
    
    #	 -1.0000 P(i,j)<l,k||c,j>*t1(c,i)*t1(a,k)*t1(b,l)
    contracted_intermediate = -1.000000000000000 * einsum('lkcj,ci,ak,bl->abij', g[o, o, v, o], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->abji', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<k,a||c,d>*t1(c,j)*t1(d,i)*t1(b,k)
    contracted_intermediate = -1.000000000000000 * einsum('kacd,cj,di,bk->abij', g[o, v, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    doubles_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abij->baij', contracted_intermediate) 
    
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
    
    #	  1.0000 <l,k||c,d>*t1(c,j)*t1(d,i)*t1(a,k)*t1(b,l)
    doubles_res +=  1.000000000000000 * einsum('lkcd,cj,di,ak,bl->abij', g[o, o, v, v], t1, t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])

    return doubles_res

def triples_residual(t1, t2, t3, t4, f, g, o, v):
    """
     < 0 | i* j* k* c b a e(-T) H e(T) | 0>

    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param t3: spin-orbital t3 amplitudes (nvirt x nvirt x nvirt x nocc x nocc x nocc)
    :param t4: spin-orbital t4 amplitudes (nvirt x nvirt x nvirt x nvirt x nocc x nocc x nocc x nocc)
    :param f: fock operator defined as soei + np.einsum('piiq->pq', astei[:, o, o, :])
              where soei is 1 electron integrals (spinorb) and astei is
              antisymmetric 2 electron integrals in openfermion format
              <12|21>.  <ij|kl> - <ij|lk>
    :param g: antisymmetric 2 electron integrals. See fock input.
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    """

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
    
    #	 -1.0000 f(l,d)*t4(d,a,b,c,i,j,k,l)
    triples_res += -1.000000000000010 * einsum('ld,dabcijkl->abcijk', f[o, v], t4[:, :, :, :, :, :, :, :])
    
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
    
    #	 -0.5000 P(j,k)<m,l||d,k>*t4(d,a,b,c,i,j,m,l)
    contracted_intermediate = -0.500000000000010 * einsum('mldk,dabcijml->abcijk', g[o, o, v, o], t4[:, :, :, :, :, :, :, :])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	 -0.5000 <m,l||d,i>*t4(d,a,b,c,j,k,m,l)
    triples_res += -0.500000000000010 * einsum('mldi,dabcjkml->abcijk', g[o, o, v, o], t4[:, :, :, :, :, :, :, :])
    
    #	 -0.5000 P(a,b)<l,a||d,e>*t4(d,e,b,c,i,j,k,l)
    contracted_intermediate = -0.500000000000010 * einsum('lade,debcijkl->abcijk', g[o, v, v, v], t4[:, :, :, :, :, :, :, :])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	 -0.5000 <l,c||d,e>*t4(d,e,a,b,i,j,k,l)
    triples_res += -0.500000000000010 * einsum('lcde,deabijkl->abcijk', g[o, v, v, v], t4[:, :, :, :, :, :, :, :])
    
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
    
    #	  1.0000 <m,l||d,e>*t1(d,l)*t4(e,a,b,c,i,j,k,m)
    triples_res +=  1.000000000000020 * einsum('mlde,dl,eabcijkm->abcijk', g[o, o, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 P(j,k)<m,l||d,e>*t1(d,k)*t4(e,a,b,c,i,j,m,l)
    contracted_intermediate =  0.500000000000010 * einsum('mlde,dk,eabcijml->abcijk', g[o, o, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->abcikj', contracted_intermediate) 
    
    #	  0.5000 <m,l||d,e>*t1(d,i)*t4(e,a,b,c,j,k,m,l)
    triples_res +=  0.500000000000010 * einsum('mlde,di,eabcjkml->abcijk', g[o, o, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 P(a,b)<m,l||d,e>*t1(a,l)*t4(d,e,b,c,i,j,k,m)
    contracted_intermediate =  0.500000000000010 * einsum('mlde,al,debcijkm->abcijk', g[o, o, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    triples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcijk->bacijk', contracted_intermediate) 
    
    #	  0.5000 <m,l||d,e>*t1(c,l)*t4(d,e,a,b,i,j,k,m)
    triples_res +=  0.500000000000010 * einsum('mlde,cl,deabijkm->abcijk', g[o, o, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    
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

def quadruples_residual(t1, t2, t3, t4, f, g, o, v):
    """
     < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0>

    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param t3: spin-orbital t3 amplitudes (nvirt x nvirt x nvirt x nocc x nocc x nocc)
    :param t4: spin-orbital t4 amplitudes (nvirt x nvirt x nvirt x nvirt x nocc x nocc x nocc x nocc)
    :param f: fock operator defined as soei + np.einsum('piiq->pq', astei[:, o, o, :])
              where soei is 1 electron integrals (spinorb) and astei is
              antisymmetric 2 electron integrals in openfermion format
              <12|21>.  <ij|kl> - <ij|lk>
    :param g: antisymmetric 2 electron integrals. See fock input.
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    """

    #	 -1.0000 P(k,l)f(m,l)*t4(a,b,c,d,i,j,k,m)
    contracted_intermediate = -1.000000000000010 * einsum('ml,abcdijkm->abcdijkl', f[o, o], t4[:, :, :, :, :, :, :, :])
    quadruples_res =  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)f(m,j)*t4(a,b,c,d,i,k,l,m)
    contracted_intermediate = -1.000000000000010 * einsum('mj,abcdiklm->abcdijkl', f[o, o], t4[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)f(a,e)*t4(e,b,c,d,i,j,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('ae,ebcdijkl->abcdijkl', f[v, v], t4[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)f(c,e)*t4(e,a,b,d,i,j,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('ce,eabdijkl->abcdijkl', f[v, v], t4[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)f(m,e)*t1(e,l)*t4(a,b,c,d,i,j,k,m)
    contracted_intermediate = -1.000000000000020 * einsum('me,el,abcdijkm->abcdijkl', f[o, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)f(m,e)*t1(e,j)*t4(a,b,c,d,i,k,l,m)
    contracted_intermediate = -1.000000000000020 * einsum('me,ej,abcdiklm->abcdijkl', f[o, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)f(m,e)*t1(a,m)*t4(e,b,c,d,i,j,k,l)
    contracted_intermediate = -1.000000000000020 * einsum('me,am,ebcdijkl->abcdijkl', f[o, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)f(m,e)*t1(c,m)*t4(e,a,b,d,i,j,k,l)
    contracted_intermediate = -1.000000000000020 * einsum('me,cm,eabdijkl->abcdijkl', f[o, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)f(m,e)*t2(e,a,k,l)*t3(b,c,d,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('me,eakl,bcdijm->abcdijkl', f[o, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)f(m,e)*t2(e,a,i,l)*t3(b,c,d,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('me,eail,bcdjkm->abcdijkl', f[o, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)f(m,e)*t2(e,a,j,k)*t3(b,c,d,i,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('me,eajk,bcdilm->abcdijkl', f[o, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(c,d)f(m,e)*t2(e,c,k,l)*t3(a,b,d,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('me,eckl,abdijm->abcdijkl', f[o, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)f(m,e)*t2(e,c,i,l)*t3(a,b,d,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('me,ecil,abdjkm->abcdijkl', f[o, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(c,d)f(m,e)*t2(e,c,j,k)*t3(a,b,d,i,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('me,ecjk,abdilm->abcdijkl', f[o, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)f(m,e)*t2(a,b,l,m)*t3(e,c,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('me,ablm,ecdijk->abcdijkl', f[o, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)f(m,e)*t2(a,b,j,m)*t3(e,c,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('me,abjm,ecdikl->abcdijkl', f[o, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)f(m,e)*t2(a,d,l,m)*t3(e,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('me,adlm,ebcijk->abcdijkl', f[o, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)f(m,e)*t2(a,d,j,m)*t3(e,b,c,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('me,adjm,ebcikl->abcdijkl', f[o, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,d)f(m,e)*t2(b,c,l,m)*t3(e,a,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('me,bclm,eadijk->abcdijkl', f[o, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,d)f(m,e)*t2(b,c,j,m)*t3(e,a,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('me,bcjm,eadikl->abcdijkl', f[o, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,a||k,l>*t3(b,c,d,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('makl,bcdijm->abcdijkl', g[o, v, o, o], t3)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,a||i,l>*t3(b,c,d,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mail,bcdjkm->abcdijkl', g[o, v, o, o], t3)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,a||j,k>*t3(b,c,d,i,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('majk,bcdilm->abcdijkl', g[o, v, o, o], t3)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(c,d)<m,c||k,l>*t3(a,b,d,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mckl,abdijm->abcdijkl', g[o, v, o, o], t3)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,c||i,l>*t3(a,b,d,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcil,abdjkm->abcdijkl', g[o, v, o, o], t3)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(c,d)<m,c||j,k>*t3(a,b,d,i,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcjk,abdilm->abcdijkl', g[o, v, o, o], t3)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<a,b||e,l>*t3(e,c,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('abel,ecdijk->abcdijkl', g[v, v, v, o], t3)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<a,b||e,j>*t3(e,c,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('abej,ecdikl->abcdijkl', g[v, v, v, o], t3)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<a,d||e,l>*t3(e,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('adel,ebcijk->abcdijkl', g[v, v, v, o], t3)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<a,d||e,j>*t3(e,b,c,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('adej,ebcikl->abcdijkl', g[v, v, v, o], t3)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,d)<b,c||e,l>*t3(e,a,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('bcel,eadijk->abcdijkl', g[v, v, v, o], t3)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,d)<b,c||e,j>*t3(e,a,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('bcej,eadikl->abcdijkl', g[v, v, v, o], t3)
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)<n,m||k,l>*t4(a,b,c,d,i,j,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmkl,abcdijnm->abcdijkl', g[o, o, o, o], t4[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||i,l>*t4(a,b,c,d,j,k,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmil,abcdjknm->abcdijkl', g[o, o, o, o], t4[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,k)<n,m||j,k>*t4(a,b,c,d,i,l,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmjk,abcdilnm->abcdijkl', g[o, o, o, o], t4[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,a||e,l>*t4(e,b,c,d,i,j,k,m)
    contracted_intermediate =  1.000000000000020 * einsum('mael,ebcdijkm->abcdijkl', g[o, v, v, o], t4[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,j>*t4(e,b,c,d,i,k,l,m)
    contracted_intermediate =  1.000000000000020 * einsum('maej,ebcdiklm->abcdijkl', g[o, v, v, o], t4[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,c||e,l>*t4(e,a,b,d,i,j,k,m)
    contracted_intermediate =  1.000000000000020 * einsum('mcel,eabdijkm->abcdijkl', g[o, v, v, o], t4[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,j>*t4(e,a,b,d,i,k,l,m)
    contracted_intermediate =  1.000000000000020 * einsum('mcej,eabdiklm->abcdijkl', g[o, v, v, o], t4[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  0.5000 P(b,c)<a,b||e,f>*t4(e,f,c,d,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('abef,efcdijkl->abcdijkl', g[v, v, v, v], t4[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<a,d||e,f>*t4(e,f,b,c,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('adef,efbcijkl->abcdijkl', g[v, v, v, v], t4[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(b,d)<b,c||e,f>*t4(e,f,a,d,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('bcef,efadijkl->abcdijkl', g[v, v, v, v], t4[:, :, :, :, :, :, :, :])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||k,l>*t1(a,m)*t3(b,c,d,i,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,am,bcdijn->abcdijkl', g[o, o, o, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(c,d)<n,m||k,l>*t1(c,m)*t3(a,b,d,i,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,cm,abdijn->abcdijkl', g[o, o, o, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||i,l>*t1(a,m)*t3(b,c,d,j,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmil,am,bcdjkn->abcdijkl', g[o, o, o, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<n,m||i,l>*t1(c,m)*t3(a,b,d,j,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmil,cm,abdjkn->abcdijkl', g[o, o, o, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<n,m||j,k>*t1(a,m)*t3(b,c,d,i,l,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjk,am,bcdiln->abcdijkl', g[o, o, o, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(c,d)<n,m||j,k>*t1(c,m)*t3(a,b,d,i,l,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjk,cm,abdiln->abcdijkl', g[o, o, o, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,a||e,l>*t1(e,k)*t3(b,c,d,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,ek,bcdijm->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,a||e,l>*t1(e,i)*t3(b,c,d,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,ei,bcdjkm->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<m,a||e,l>*t1(b,m)*t3(e,c,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('mael,bm,ecdijk->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,a||e,l>*t1(d,m)*t3(e,b,c,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('mael,dm,ebcijk->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(a,b)<m,a||e,k>*t1(e,l)*t3(b,c,d,i,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('maek,el,bcdijm->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,a||e,j>*t1(e,l)*t3(b,c,d,i,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,el,bcdikm->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,j>*t1(e,i)*t3(b,c,d,k,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,ei,bcdklm->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<m,a||e,j>*t1(b,m)*t3(e,c,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('maej,bm,ecdikl->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,j>*t1(d,m)*t3(e,b,c,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('maej,dm,ebcikl->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,a||e,i>*t1(e,l)*t3(b,c,d,j,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('maei,el,bcdjkm->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,c)<m,b||e,l>*t1(a,m)*t3(e,c,d,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('mbel,am,ecdijk->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<m,b||e,j>*t1(a,m)*t3(e,c,d,i,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('mbej,am,ecdikl->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(c,d)<m,c||e,l>*t1(e,k)*t3(a,b,d,i,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,ek,abdijm->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,c||e,l>*t1(e,i)*t3(a,b,d,j,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,ei,abdjkm->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,c||e,l>*t1(a,m)*t3(e,b,d,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,am,ebdijk->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,c||e,l>*t1(d,m)*t3(e,a,b,i,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,dm,eabijk->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(c,d)<m,c||e,k>*t1(e,l)*t3(a,b,d,i,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcek,el,abdijm->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,c||e,j>*t1(e,l)*t3(a,b,d,i,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,el,abdikm->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,j>*t1(e,i)*t3(a,b,d,k,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,ei,abdklm->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,c||e,j>*t1(a,m)*t3(e,b,d,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,am,ebdikl->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,j>*t1(d,m)*t3(e,a,b,i,k,l)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,dm,eabikl->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,i>*t1(e,l)*t3(a,b,d,j,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcei,el,abdjkm->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,d||e,l>*t1(a,m)*t3(e,b,c,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,am,ebcijk->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,j>*t1(a,m)*t3(e,b,c,i,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,am,ebcikl->abcdijkl', g[o, v, v, o], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<a,b||e,f>*t1(e,l)*t3(f,c,d,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abef,el,fcdijk->abcdijkl', g[v, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<a,b||e,f>*t1(e,j)*t3(f,c,d,i,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('abef,ej,fcdikl->abcdijkl', g[v, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<a,d||e,f>*t1(e,l)*t3(f,b,c,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('adef,el,fbcijk->abcdijkl', g[v, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<a,d||e,f>*t1(e,j)*t3(f,b,c,i,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ej,fbcikl->abcdijkl', g[v, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,d)<b,c||e,f>*t1(e,l)*t3(f,a,d,i,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,el,fadijk->abcdijkl', g[v, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,d)<b,c||e,f>*t1(e,j)*t3(f,a,d,i,k,l)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,ej,fadikl->abcdijkl', g[v, v, v, v], t1, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,l>*t1(e,m)*t4(a,b,c,d,i,j,k,n)
    contracted_intermediate =  1.000000000000020 * einsum('nmel,em,abcdijkn->abcdijkl', g[o, o, v, o], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(j,k)<n,m||e,l>*t1(e,k)*t4(a,b,c,d,i,j,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmel,ek,abcdijnm->abcdijkl', g[o, o, v, o], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||e,l>*t1(e,i)*t4(a,b,c,d,j,k,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmel,ei,abcdjknm->abcdijkl', g[o, o, v, o], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,l>*t1(a,m)*t4(e,b,c,d,i,j,k,n)
    contracted_intermediate = -1.000000000000020 * einsum('nmel,am,ebcdijkn->abcdijkl', g[o, o, v, o], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<n,m||e,l>*t1(c,m)*t4(e,a,b,d,i,j,k,n)
    contracted_intermediate = -1.000000000000020 * einsum('nmel,cm,eabdijkn->abcdijkl', g[o, o, v, o], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(j,l)<n,m||e,k>*t1(e,l)*t4(a,b,c,d,i,j,n,m)
    contracted_intermediate = -0.500000000000010 * einsum('nmek,el,abcdijnm->abcdijkl', g[o, o, v, o], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,j>*t1(e,m)*t4(a,b,c,d,i,k,l,n)
    contracted_intermediate =  1.000000000000020 * einsum('nmej,em,abcdikln->abcdijkl', g[o, o, v, o], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)<n,m||e,j>*t1(e,l)*t4(a,b,c,d,i,k,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmej,el,abcdiknm->abcdijkl', g[o, o, v, o], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)<n,m||e,j>*t1(e,i)*t4(a,b,c,d,k,l,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmej,ei,abcdklnm->abcdijkl', g[o, o, v, o], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,j>*t1(a,m)*t4(e,b,c,d,i,k,l,n)
    contracted_intermediate = -1.000000000000020 * einsum('nmej,am,ebcdikln->abcdijkl', g[o, o, v, o], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<n,m||e,j>*t1(c,m)*t4(e,a,b,d,i,k,l,n)
    contracted_intermediate = -1.000000000000020 * einsum('nmej,cm,eabdikln->abcdijkl', g[o, o, v, o], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,i>*t1(e,l)*t4(a,b,c,d,j,k,n,m)
    contracted_intermediate = -0.500000000000010 * einsum('nmei,el,abcdjknm->abcdijkl', g[o, o, v, o], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<m,a||e,f>*t1(e,m)*t4(f,b,c,d,i,j,k,l)
    contracted_intermediate =  1.000000000000020 * einsum('maef,em,fbcdijkl->abcdijkl', g[o, v, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,a||e,f>*t1(e,l)*t4(f,b,c,d,i,j,k,m)
    contracted_intermediate = -1.000000000000020 * einsum('maef,el,fbcdijkm->abcdijkl', g[o, v, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,f>*t1(e,j)*t4(f,b,c,d,i,k,l,m)
    contracted_intermediate = -1.000000000000020 * einsum('maef,ej,fbcdiklm->abcdijkl', g[o, v, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  0.5000 P(b,c)<m,a||e,f>*t1(b,m)*t4(e,f,c,d,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('maef,bm,efcdijkl->abcdijkl', g[o, v, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,a||e,f>*t1(d,m)*t4(e,f,b,c,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('maef,dm,efbcijkl->abcdijkl', g[o, v, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,c)<m,b||e,f>*t1(a,m)*t4(e,f,c,d,i,j,k,l)
    contracted_intermediate = -0.500000000000010 * einsum('mbef,am,efcdijkl->abcdijkl', g[o, v, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<m,c||e,f>*t1(e,m)*t4(f,a,b,d,i,j,k,l)
    contracted_intermediate =  1.000000000000020 * einsum('mcef,em,fabdijkl->abcdijkl', g[o, v, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,f>*t1(e,l)*t4(f,a,b,d,i,j,k,m)
    contracted_intermediate = -1.000000000000020 * einsum('mcef,el,fabdijkm->abcdijkl', g[o, v, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<m,c||e,f>*t1(e,j)*t4(f,a,b,d,i,k,l,m)
    contracted_intermediate = -1.000000000000020 * einsum('mcef,ej,fabdiklm->abcdijkl', g[o, v, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  0.5000 P(a,b)<m,c||e,f>*t1(a,m)*t4(e,f,b,d,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('mcef,am,efbdijkl->abcdijkl', g[o, v, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  0.5000 P(c,d)<m,c||e,f>*t1(d,m)*t4(e,f,a,b,i,j,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('mcef,dm,efabijkl->abcdijkl', g[o, v, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<m,d||e,f>*t1(a,m)*t4(e,f,b,c,i,j,k,l)
    contracted_intermediate = -0.500000000000010 * einsum('mdef,am,efbcijkl->abcdijkl', g[o, v, v, v], t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||k,l>*t2(a,b,j,m)*t2(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,abjm,cdin->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||k,l>*t2(a,d,j,m)*t2(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmkl,adjm,bcin->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||j,l>*t2(a,b,k,m)*t2(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,abkm,cdin->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||j,l>*t2(a,d,k,m)*t2(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmjl,adkm,bcin->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||i,l>*t2(a,b,k,m)*t2(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmil,abkm,cdjn->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||i,l>*t2(a,d,k,m)*t2(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmil,adkm,bcjn->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)*P(b,c)<n,m||j,k>*t2(a,b,l,m)*t2(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjk,ablm,cdin->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)<n,m||j,k>*t2(a,d,l,m)*t2(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmjk,adlm,bcin->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(b,c)<n,m||i,k>*t2(a,b,l,m)*t2(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmik,ablm,cdjn->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<n,m||i,k>*t2(a,d,l,m)*t2(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmik,adlm,bcjn->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||i,j>*t2(a,b,l,m)*t2(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmij,ablm,cdkn->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||i,j>*t2(a,d,l,m)*t2(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmij,adlm,bckn->abcdijkl', g[o, o, o, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<m,a||e,l>*t2(e,b,j,k)*t2(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,ebjk,cdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<m,a||e,l>*t2(e,b,i,j)*t2(c,d,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,ebij,cdkm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,l>*t2(e,d,j,k)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,edjk,bcim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,a||e,l>*t2(e,d,i,j)*t2(b,c,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mael,edij,bckm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<m,a||e,k>*t2(e,b,j,l)*t2(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('maek,ebjl,cdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,k>*t2(e,d,j,l)*t2(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('maek,edjl,bcim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<m,a||e,j>*t2(e,b,k,l)*t2(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,ebkl,cdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<m,a||e,j>*t2(e,b,i,k)*t2(c,d,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,ebik,cdlm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,a||e,j>*t2(e,d,k,l)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,edkl,bcim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,j>*t2(e,d,i,k)*t2(b,c,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('maej,edik,bclm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<m,a||e,i>*t2(e,b,k,l)*t2(c,d,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('maei,ebkl,cdjm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,a||e,i>*t2(e,d,k,l)*t2(b,c,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('maei,edkl,bcjm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<m,b||e,l>*t2(e,a,j,k)*t2(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbel,eajk,cdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,c)<m,b||e,l>*t2(e,a,i,j)*t2(c,d,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbel,eaij,cdkm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,c)<m,b||e,k>*t2(e,a,j,l)*t2(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mbek,eajl,cdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,c)<m,b||e,j>*t2(e,a,k,l)*t2(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbej,eakl,cdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<m,b||e,j>*t2(e,a,i,k)*t2(c,d,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbej,eaik,cdlm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,c)<m,b||e,i>*t2(e,a,k,l)*t2(c,d,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mbei,eakl,cdjm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,c||e,l>*t2(e,a,j,k)*t2(b,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,eajk,bdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,c||e,l>*t2(e,a,i,j)*t2(b,d,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,eaij,bdkm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,l>*t2(e,d,j,k)*t2(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,edjk,abim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<m,c||e,l>*t2(e,d,i,j)*t2(a,b,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcel,edij,abkm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,c||e,k>*t2(e,a,j,l)*t2(b,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcek,eajl,bdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<m,c||e,k>*t2(e,d,j,l)*t2(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcek,edjl,abim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,c||e,j>*t2(e,a,k,l)*t2(b,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,eakl,bdim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,c||e,j>*t2(e,a,i,k)*t2(b,d,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,eaik,bdlm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(c,d)<m,c||e,j>*t2(e,d,k,l)*t2(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,edkl,abim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,j>*t2(e,d,i,k)*t2(a,b,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcej,edik,ablm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,c||e,i>*t2(e,a,k,l)*t2(b,d,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcei,eakl,bdjm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(c,d)<m,c||e,i>*t2(e,d,k,l)*t2(a,b,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcei,edkl,abjm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,l>*t2(e,a,j,k)*t2(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,eajk,bcim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,d||e,l>*t2(e,a,i,j)*t2(b,c,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdel,eaij,bckm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,k>*t2(e,a,j,l)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdek,eajl,bcim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,d||e,j>*t2(e,a,k,l)*t2(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,eakl,bcim->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,j>*t2(e,a,i,k)*t2(b,c,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdej,eaik,bclm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,d||e,i>*t2(e,a,k,l)*t2(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdei,eakl,bcjm->abcdijkl', g[o, v, v, o], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<a,b||e,f>*t2(e,c,k,l)*t2(f,d,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('abef,eckl,fdij->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<a,b||e,f>*t2(e,c,i,l)*t2(f,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('abef,ecil,fdjk->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<a,b||e,f>*t2(e,c,j,k)*t2(f,d,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('abef,ecjk,fdil->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<a,d||e,f>*t2(e,b,k,l)*t2(f,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebkl,fcij->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<a,d||e,f>*t2(e,b,i,l)*t2(f,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebil,fcjk->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<a,d||e,f>*t2(e,b,j,k)*t2(f,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('adef,ebjk,fcil->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,d)<b,c||e,f>*t2(e,a,k,l)*t2(f,d,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,eakl,fdij->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,d)<b,c||e,f>*t2(e,a,i,l)*t2(f,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,eail,fdjk->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,d)<b,c||e,f>*t2(e,a,j,k)*t2(f,d,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('bcef,eajk,fdil->abcdijkl', g[v, v, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||e,l>*t2(e,a,k,m)*t3(b,c,d,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmel,eakm,bcdijn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,l>*t2(e,a,i,m)*t3(b,c,d,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmel,eaim,bcdjkn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,l>*t2(e,a,j,k)*t3(b,c,d,i,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmel,eajk,bcdinm->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(a,b)<n,m||e,l>*t2(e,a,i,j)*t3(b,c,d,k,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmel,eaij,bcdknm->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(c,d)<n,m||e,l>*t2(e,c,k,m)*t3(a,b,d,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmel,eckm,abdijn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<n,m||e,l>*t2(e,c,i,m)*t3(a,b,d,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmel,ecim,abdjkn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(c,d)<n,m||e,l>*t2(e,c,j,k)*t3(a,b,d,i,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmel,ecjk,abdinm->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(c,d)<n,m||e,l>*t2(e,c,i,j)*t3(a,b,d,k,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmel,ecij,abdknm->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(b,c)<n,m||e,l>*t2(a,b,n,m)*t3(e,c,d,i,j,k)
    contracted_intermediate =  0.500000000000010 * einsum('nmel,abnm,ecdijk->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||e,l>*t2(a,b,k,m)*t3(e,c,d,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmel,abkm,ecdijn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||e,l>*t2(a,b,i,m)*t3(e,c,d,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmel,abim,ecdjkn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(a,b)<n,m||e,l>*t2(a,d,n,m)*t3(e,b,c,i,j,k)
    contracted_intermediate =  0.500000000000010 * einsum('nmel,adnm,ebcijk->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||e,l>*t2(a,d,k,m)*t3(e,b,c,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmel,adkm,ebcijn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,l>*t2(a,d,i,m)*t3(e,b,c,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmel,adim,ebcjkn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(b,d)<n,m||e,l>*t2(b,c,n,m)*t3(e,a,d,i,j,k)
    contracted_intermediate =  0.500000000000010 * einsum('nmel,bcnm,eadijk->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,d)<n,m||e,l>*t2(b,c,k,m)*t3(e,a,d,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmel,bckm,eadijn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,d)<n,m||e,l>*t2(b,c,i,m)*t3(e,a,d,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmel,bcim,eadjkn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(a,b)<n,m||e,k>*t2(e,a,l,m)*t3(b,c,d,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmek,ealm,bcdijn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdilkj', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,k>*t2(e,a,j,l)*t3(b,c,d,i,n,m)
    contracted_intermediate = -0.500000000000010 * einsum('nmek,eajl,bcdinm->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(c,d)<n,m||e,k>*t2(e,c,l,m)*t3(a,b,d,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmek,eclm,abdijn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcilkj', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(c,d)<n,m||e,k>*t2(e,c,j,l)*t3(a,b,d,i,n,m)
    contracted_intermediate = -0.500000000000010 * einsum('nmek,ecjl,abdinm->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(b,c)<n,m||e,k>*t2(a,b,l,m)*t3(e,c,d,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmek,ablm,ecdijn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(a,b)<n,m||e,k>*t2(a,d,l,m)*t3(e,b,c,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmek,adlm,ebcijn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(b,d)<n,m||e,k>*t2(b,c,l,m)*t3(e,a,d,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmek,bclm,eadijn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbilkj', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,j>*t2(e,a,l,m)*t3(b,c,d,i,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmej,ealm,bcdikn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,j>*t2(e,a,i,m)*t3(b,c,d,k,l,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmej,eaim,bcdkln->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(a,b)<n,m||e,j>*t2(e,a,k,l)*t3(b,c,d,i,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmej,eakl,bcdinm->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,j>*t2(e,a,i,k)*t3(b,c,d,l,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmej,eaik,bcdlnm->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<n,m||e,j>*t2(e,c,l,m)*t3(a,b,d,i,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmej,eclm,abdikn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<n,m||e,j>*t2(e,c,i,m)*t3(a,b,d,k,l,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmej,ecim,abdkln->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(c,d)<n,m||e,j>*t2(e,c,k,l)*t3(a,b,d,i,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmej,eckl,abdinm->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(c,d)<n,m||e,j>*t2(e,c,i,k)*t3(a,b,d,l,n,m)
    contracted_intermediate =  0.500000000000010 * einsum('nmej,ecik,abdlnm->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,c)<n,m||e,j>*t2(a,b,n,m)*t3(e,c,d,i,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('nmej,abnm,ecdikl->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||e,j>*t2(a,b,l,m)*t3(e,c,d,i,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmej,ablm,ecdikn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||e,j>*t2(a,b,i,m)*t3(e,c,d,k,l,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmej,abim,ecdkln->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,j>*t2(a,d,n,m)*t3(e,b,c,i,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('nmej,adnm,ebcikl->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,j>*t2(a,d,l,m)*t3(e,b,c,i,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmej,adlm,ebcikn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,j>*t2(a,d,i,m)*t3(e,b,c,k,l,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmej,adim,ebckln->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,d)<n,m||e,j>*t2(b,c,n,m)*t3(e,a,d,i,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('nmej,bcnm,eadikl->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,d)<n,m||e,j>*t2(b,c,l,m)*t3(e,a,d,i,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmej,bclm,eadikn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,d)<n,m||e,j>*t2(b,c,i,m)*t3(e,a,d,k,l,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmej,bcim,eadkln->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,i>*t2(e,a,l,m)*t3(b,c,d,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmei,ealm,bcdjkn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<n,m||e,i>*t2(e,a,k,l)*t3(b,c,d,j,n,m)
    contracted_intermediate = -0.500000000000010 * einsum('nmei,eakl,bcdjnm->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,i>*t2(e,c,l,m)*t3(a,b,d,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmei,eclm,abdjkn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(c,d)<n,m||e,i>*t2(e,c,k,l)*t3(a,b,d,j,n,m)
    contracted_intermediate = -0.500000000000010 * einsum('nmei,eckl,abdjnm->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,i>*t2(a,b,l,m)*t3(e,c,d,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmei,ablm,ecdjkn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,i>*t2(a,d,l,m)*t3(e,b,c,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmei,adlm,ebcjkn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,d)<n,m||e,i>*t2(b,c,l,m)*t3(e,a,d,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmei,bclm,eadjkn->abcdijkl', g[o, o, v, o], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(a,b)<m,a||e,f>*t2(e,f,k,l)*t3(b,c,d,i,j,m)
    contracted_intermediate =  0.500000000000010 * einsum('maef,efkl,bcdijm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(a,b)<m,a||e,f>*t2(e,f,i,l)*t3(b,c,d,j,k,m)
    contracted_intermediate =  0.500000000000010 * einsum('maef,efil,bcdjkm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(a,b)<m,a||e,f>*t2(e,f,j,k)*t3(b,c,d,i,l,m)
    contracted_intermediate =  0.500000000000010 * einsum('maef,efjk,bcdilm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<m,a||e,f>*t2(e,b,l,m)*t3(f,c,d,i,j,k)
    contracted_intermediate = -1.000000000000010 * einsum('maef,eblm,fcdijk->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<m,a||e,f>*t2(e,b,j,m)*t3(f,c,d,i,k,l)
    contracted_intermediate = -1.000000000000010 * einsum('maef,ebjm,fcdikl->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<m,a||e,f>*t2(e,b,k,l)*t3(f,c,d,i,j,m)
    contracted_intermediate = -1.000000000000010 * einsum('maef,ebkl,fcdijm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<m,a||e,f>*t2(e,b,i,l)*t3(f,c,d,j,k,m)
    contracted_intermediate = -1.000000000000010 * einsum('maef,ebil,fcdjkm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<m,a||e,f>*t2(e,b,j,k)*t3(f,c,d,i,l,m)
    contracted_intermediate = -1.000000000000010 * einsum('maef,ebjk,fcdilm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,a||e,f>*t2(e,d,l,m)*t3(f,b,c,i,j,k)
    contracted_intermediate = -1.000000000000010 * einsum('maef,edlm,fbcijk->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,f>*t2(e,d,j,m)*t3(f,b,c,i,k,l)
    contracted_intermediate = -1.000000000000010 * einsum('maef,edjm,fbcikl->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,a||e,f>*t2(e,d,k,l)*t3(f,b,c,i,j,m)
    contracted_intermediate = -1.000000000000010 * einsum('maef,edkl,fbcijm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,a||e,f>*t2(e,d,i,l)*t3(f,b,c,j,k,m)
    contracted_intermediate = -1.000000000000010 * einsum('maef,edil,fbcjkm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,a||e,f>*t2(e,d,j,k)*t3(f,b,c,i,l,m)
    contracted_intermediate = -1.000000000000010 * einsum('maef,edjk,fbcilm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(c,d)<m,a||e,f>*t2(b,c,l,m)*t3(e,f,d,i,j,k)
    contracted_intermediate =  0.500000000000010 * einsum('maef,bclm,efdijk->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(c,d)<m,a||e,f>*t2(b,c,j,m)*t3(e,f,d,i,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('maef,bcjm,efdikl->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(a,b)<m,a||e,f>*t2(c,d,l,m)*t3(e,f,b,i,j,k)
    contracted_intermediate =  0.500000000000010 * einsum('maef,cdlm,efbijk->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<m,a||e,f>*t2(c,d,j,m)*t3(e,f,b,i,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('maef,cdjm,efbikl->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,c)<m,b||e,f>*t2(e,a,l,m)*t3(f,c,d,i,j,k)
    contracted_intermediate =  1.000000000000010 * einsum('mbef,ealm,fcdijk->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,c)<m,b||e,f>*t2(e,a,j,m)*t3(f,c,d,i,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('mbef,eajm,fcdikl->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,c)<m,b||e,f>*t2(e,a,k,l)*t3(f,c,d,i,j,m)
    contracted_intermediate =  1.000000000000010 * einsum('mbef,eakl,fcdijm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,c)<m,b||e,f>*t2(e,a,i,l)*t3(f,c,d,j,k,m)
    contracted_intermediate =  1.000000000000010 * einsum('mbef,eail,fcdjkm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,c)<m,b||e,f>*t2(e,a,j,k)*t3(f,c,d,i,l,m)
    contracted_intermediate =  1.000000000000010 * einsum('mbef,eajk,fcdilm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadkjil', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(c,d)<m,b||e,f>*t2(a,c,l,m)*t3(e,f,d,i,j,k)
    contracted_intermediate = -0.500000000000010 * einsum('mbef,aclm,efdijk->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(c,d)<m,b||e,f>*t2(a,c,j,m)*t3(e,f,d,i,k,l)
    contracted_intermediate = -0.500000000000010 * einsum('mbef,acjm,efdikl->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(c,d)<m,c||e,f>*t2(e,f,k,l)*t3(a,b,d,i,j,m)
    contracted_intermediate =  0.500000000000010 * einsum('mcef,efkl,abdijm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(c,d)<m,c||e,f>*t2(e,f,i,l)*t3(a,b,d,j,k,m)
    contracted_intermediate =  0.500000000000010 * einsum('mcef,efil,abdjkm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(c,d)<m,c||e,f>*t2(e,f,j,k)*t3(a,b,d,i,l,m)
    contracted_intermediate =  0.500000000000010 * einsum('mcef,efjk,abdilm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,c||e,f>*t2(e,a,l,m)*t3(f,b,d,i,j,k)
    contracted_intermediate = -1.000000000000010 * einsum('mcef,ealm,fbdijk->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,c||e,f>*t2(e,a,j,m)*t3(f,b,d,i,k,l)
    contracted_intermediate = -1.000000000000010 * einsum('mcef,eajm,fbdikl->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,c||e,f>*t2(e,a,k,l)*t3(f,b,d,i,j,m)
    contracted_intermediate = -1.000000000000010 * einsum('mcef,eakl,fbdijm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,c||e,f>*t2(e,a,i,l)*t3(f,b,d,j,k,m)
    contracted_intermediate = -1.000000000000010 * einsum('mcef,eail,fbdjkm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,c||e,f>*t2(e,a,j,k)*t3(f,b,d,i,l,m)
    contracted_intermediate = -1.000000000000010 * einsum('mcef,eajk,fbdilm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,f>*t2(e,d,l,m)*t3(f,a,b,i,j,k)
    contracted_intermediate = -1.000000000000010 * einsum('mcef,edlm,fabijk->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<m,c||e,f>*t2(e,d,j,m)*t3(f,a,b,i,k,l)
    contracted_intermediate = -1.000000000000010 * einsum('mcef,edjm,fabikl->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(c,d)<m,c||e,f>*t2(e,d,k,l)*t3(f,a,b,i,j,m)
    contracted_intermediate = -1.000000000000010 * einsum('mcef,edkl,fabijm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,f>*t2(e,d,i,l)*t3(f,a,b,j,k,m)
    contracted_intermediate = -1.000000000000010 * einsum('mcef,edil,fabjkm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(c,d)<m,c||e,f>*t2(e,d,j,k)*t3(f,a,b,i,l,m)
    contracted_intermediate = -1.000000000000010 * einsum('mcef,edjk,fabilm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(b,d)<m,c||e,f>*t2(a,b,l,m)*t3(e,f,d,i,j,k)
    contracted_intermediate =  0.500000000000010 * einsum('mcef,ablm,efdijk->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,d)<m,c||e,f>*t2(a,b,j,m)*t3(e,f,d,i,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('mcef,abjm,efdikl->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(c,d)<m,c||e,f>*t2(b,d,l,m)*t3(e,f,a,i,j,k)
    contracted_intermediate =  0.500000000000010 * einsum('mcef,bdlm,efaijk->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(c,d)<m,c||e,f>*t2(b,d,j,m)*t3(e,f,a,i,k,l)
    contracted_intermediate =  0.500000000000010 * einsum('mcef,bdjm,efaikl->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,d||e,f>*t2(e,a,l,m)*t3(f,b,c,i,j,k)
    contracted_intermediate =  1.000000000000010 * einsum('mdef,ealm,fbcijk->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,f>*t2(e,a,j,m)*t3(f,b,c,i,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('mdef,eajm,fbcikl->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,d||e,f>*t2(e,a,k,l)*t3(f,b,c,i,j,m)
    contracted_intermediate =  1.000000000000010 * einsum('mdef,eakl,fbcijm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,d||e,f>*t2(e,a,i,l)*t3(f,b,c,j,k,m)
    contracted_intermediate =  1.000000000000010 * einsum('mdef,eail,fbcjkm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,d||e,f>*t2(e,a,j,k)*t3(f,b,c,i,l,m)
    contracted_intermediate =  1.000000000000010 * einsum('mdef,eajk,fbcilm->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<m,d||e,f>*t2(a,b,l,m)*t3(e,f,c,i,j,k)
    contracted_intermediate = -0.500000000000010 * einsum('mdef,ablm,efcijk->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,c)<m,d||e,f>*t2(a,b,j,m)*t3(e,f,c,i,k,l)
    contracted_intermediate = -0.500000000000010 * einsum('mdef,abjm,efcikl->abcdijkl', g[o, v, v, v], t2, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>*t2(e,f,l,m)*t4(a,b,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eflm,abcdijkn->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>*t2(e,f,j,m)*t4(a,b,c,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,efjm,abcdikln->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.2500 P(j,k)<n,m||e,f>*t2(e,f,k,l)*t4(a,b,c,d,i,j,n,m)
    contracted_intermediate =  0.249999999999970 * einsum('nmef,efkl,abcdijnm->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  0.2500 P(k,l)<n,m||e,f>*t2(e,f,i,l)*t4(a,b,c,d,j,k,n,m)
    contracted_intermediate =  0.249999999999970 * einsum('nmef,efil,abcdjknm->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  0.2500 P(i,k)<n,m||e,f>*t2(e,f,j,k)*t4(a,b,c,d,i,l,n,m)
    contracted_intermediate =  0.249999999999970 * einsum('nmef,efjk,abcdilnm->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||e,f>*t2(e,a,n,m)*t4(f,b,c,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eanm,fbcdijkl->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t2(e,a,l,m)*t4(f,b,c,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,ealm,fbcdijkn->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t2(e,a,j,m)*t4(f,b,c,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,eajm,fbcdikln->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<n,m||e,f>*t2(e,a,k,l)*t4(f,b,c,d,i,j,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eakl,fbcdijnm->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>*t2(e,a,i,l)*t4(f,b,c,d,j,k,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eail,fbcdjknm->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(a,b)<n,m||e,f>*t2(e,a,j,k)*t4(f,b,c,d,i,l,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eajk,fbcdilnm->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(c,d)<n,m||e,f>*t2(e,c,n,m)*t4(f,a,b,d,i,j,k,l)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecnm,fabdijkl->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>*t2(e,c,l,m)*t4(f,a,b,d,i,j,k,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,eclm,fabdijkn->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||e,f>*t2(e,c,j,m)*t4(f,a,b,d,i,k,l,n)
    contracted_intermediate =  0.999999999999840 * einsum('nmef,ecjm,fabdikln->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(c,d)<n,m||e,f>*t2(e,c,k,l)*t4(f,a,b,d,i,j,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,eckl,fabdijnm->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(c,d)<n,m||e,f>*t2(e,c,i,l)*t4(f,a,b,d,j,k,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecil,fabdjknm->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(c,d)<n,m||e,f>*t2(e,c,j,k)*t4(f,a,b,d,i,l,n,m)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ecjk,fabdilnm->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  0.2500 P(b,c)<n,m||e,f>*t2(a,b,n,m)*t4(e,f,c,d,i,j,k,l)
    contracted_intermediate =  0.249999999999970 * einsum('nmef,abnm,efcdijkl->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<n,m||e,f>*t2(a,b,l,m)*t4(e,f,c,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,ablm,efcdijkn->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,c)<n,m||e,f>*t2(a,b,j,m)*t4(e,f,c,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,abjm,efcdikln->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  0.2500 P(a,b)<n,m||e,f>*t2(a,d,n,m)*t4(e,f,b,c,i,j,k,l)
    contracted_intermediate =  0.249999999999970 * einsum('nmef,adnm,efbcijkl->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>*t2(a,d,l,m)*t4(e,f,b,c,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,adlm,efbcijkn->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>*t2(a,d,j,m)*t4(e,f,b,c,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,adjm,efbcikln->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  0.2500 P(b,d)<n,m||e,f>*t2(b,c,n,m)*t4(e,f,a,d,i,j,k,l)
    contracted_intermediate =  0.249999999999970 * einsum('nmef,bcnm,efadijkl->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,d)<n,m||e,f>*t2(b,c,l,m)*t4(e,f,a,d,i,j,k,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,bclm,efadijkn->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,d)<n,m||e,f>*t2(b,c,j,m)*t4(e,f,a,d,i,k,l,n)
    contracted_intermediate = -0.499999999999950 * einsum('nmef,bcjm,efadikln->abcdijkl', g[o, o, v, v], t2, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<n,m||e,f>*t3(e,f,a,k,l,m)*t3(b,c,d,i,j,n)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,efaklm,bcdijn->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>*t3(e,f,a,i,l,m)*t3(b,c,d,j,k,n)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,efailm,bcdjkn->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(a,b)<n,m||e,f>*t3(e,f,a,j,k,m)*t3(b,c,d,i,l,n)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,efajkm,bcdiln->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(a,b)<n,m||e,f>*t3(e,f,a,j,k,l)*t3(b,c,d,i,n,m)
    contracted_intermediate = -0.249999999999990 * einsum('nmef,efajkl,bcdinm->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.2500 P(k,l)*P(a,b)<n,m||e,f>*t3(e,f,a,i,j,l)*t3(b,c,d,k,n,m)
    contracted_intermediate = -0.249999999999990 * einsum('nmef,efaijl,bcdknm->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(c,d)<n,m||e,f>*t3(e,f,c,k,l,m)*t3(a,b,d,i,j,n)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,efcklm,abdijn->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(c,d)<n,m||e,f>*t3(e,f,c,i,l,m)*t3(a,b,d,j,k,n)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,efcilm,abdjkn->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(c,d)<n,m||e,f>*t3(e,f,c,j,k,m)*t3(a,b,d,i,l,n)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,efcjkm,abdiln->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	 -0.2500 P(i,j)*P(c,d)<n,m||e,f>*t3(e,f,c,j,k,l)*t3(a,b,d,i,n,m)
    contracted_intermediate = -0.249999999999990 * einsum('nmef,efcjkl,abdinm->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -0.2500 P(k,l)*P(c,d)<n,m||e,f>*t3(e,f,c,i,j,l)*t3(a,b,d,k,n,m)
    contracted_intermediate = -0.249999999999990 * einsum('nmef,efcijl,abdknm->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<n,m||e,f>*t3(e,a,b,l,n,m)*t3(f,c,d,i,j,k)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,eablnm,fcdijk->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,c)<n,m||e,f>*t3(e,a,b,j,n,m)*t3(f,c,d,i,k,l)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,eabjnm,fcdikl->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||e,f>*t3(e,a,b,k,l,m)*t3(f,c,d,i,j,n)
    contracted_intermediate = -1.000000000000090 * einsum('nmef,eabklm,fcdijn->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||e,f>*t3(e,a,b,i,l,m)*t3(f,c,d,j,k,n)
    contracted_intermediate = -1.000000000000090 * einsum('nmef,eabilm,fcdjkn->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<n,m||e,f>*t3(e,a,b,j,k,m)*t3(f,c,d,i,l,n)
    contracted_intermediate = -1.000000000000090 * einsum('nmef,eabjkm,fcdiln->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,c)<n,m||e,f>*t3(e,a,b,j,k,l)*t3(f,c,d,i,n,m)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,eabjkl,fcdinm->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<n,m||e,f>*t3(e,a,b,i,j,l)*t3(f,c,d,k,n,m)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,eabijl,fcdknm->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>*t3(e,a,d,l,n,m)*t3(f,b,c,i,j,k)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,eadlnm,fbcijk->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>*t3(e,a,d,j,n,m)*t3(f,b,c,i,k,l)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,eadjnm,fbcikl->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>*t3(e,a,d,k,l,m)*t3(f,b,c,i,j,n)
    contracted_intermediate = -1.000000000000090 * einsum('nmef,eadklm,fbcijn->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,f>*t3(e,a,d,i,l,m)*t3(f,b,c,j,k,n)
    contracted_intermediate = -1.000000000000090 * einsum('nmef,eadilm,fbcjkn->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<n,m||e,f>*t3(e,a,d,j,k,m)*t3(f,b,c,i,l,n)
    contracted_intermediate = -1.000000000000090 * einsum('nmef,eadjkm,fbciln->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 1), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>*t3(e,a,d,j,k,l)*t3(f,b,c,i,n,m)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,eadjkl,fbcinm->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>*t3(e,a,d,i,j,l)*t3(f,b,c,k,n,m)
    contracted_intermediate = -0.499999999999980 * einsum('nmef,eadijl,fbcknm->abcdijkl', g[o, o, v, v], t3, t3, optimize=['einsum_path', (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||e,l>*t1(e,k)*t1(a,m)*t3(b,c,d,i,j,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmel,ek,am,bcdijn->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(c,d)<n,m||e,l>*t1(e,k)*t1(c,m)*t3(a,b,d,i,j,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmel,ek,cm,abdijn->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,l>*t1(e,i)*t1(a,m)*t3(b,c,d,j,k,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmel,ei,am,bcdjkn->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<n,m||e,l>*t1(e,i)*t1(c,m)*t3(a,b,d,j,k,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmel,ei,cm,abdjkn->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||e,l>*t1(a,m)*t1(b,n)*t3(e,c,d,i,j,k)
    contracted_intermediate = -0.999999999999990 * einsum('nmel,am,bn,ecdijk->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,l>*t1(a,m)*t1(d,n)*t3(e,b,c,i,j,k)
    contracted_intermediate = -0.999999999999990 * einsum('nmel,am,dn,ebcijk->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,l>*t1(b,m)*t1(c,n)*t3(e,a,d,i,j,k)
    contracted_intermediate = -0.999999999999990 * einsum('nmel,bm,cn,eadijk->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,l>*t1(c,m)*t1(d,n)*t3(e,a,b,i,j,k)
    contracted_intermediate = -0.999999999999990 * einsum('nmel,cm,dn,eabijk->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(a,b)<n,m||e,k>*t1(e,l)*t1(a,m)*t3(b,c,d,i,j,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmek,el,am,bcdijn->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(c,d)<n,m||e,k>*t1(e,l)*t1(c,m)*t3(a,b,d,i,j,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmek,el,cm,abdijn->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcilkj', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,j>*t1(e,l)*t1(a,m)*t3(b,c,d,i,k,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmej,el,am,bcdikn->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<n,m||e,j>*t1(e,l)*t1(c,m)*t3(a,b,d,i,k,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmej,el,cm,abdikn->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,j>*t1(e,i)*t1(a,m)*t3(b,c,d,k,l,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmej,ei,am,bcdkln->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<n,m||e,j>*t1(e,i)*t1(c,m)*t3(a,b,d,k,l,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmej,ei,cm,abdkln->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||e,j>*t1(a,m)*t1(b,n)*t3(e,c,d,i,k,l)
    contracted_intermediate = -0.999999999999990 * einsum('nmej,am,bn,ecdikl->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,j>*t1(a,m)*t1(d,n)*t3(e,b,c,i,k,l)
    contracted_intermediate = -0.999999999999990 * einsum('nmej,am,dn,ebcikl->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,j>*t1(b,m)*t1(c,n)*t3(e,a,d,i,k,l)
    contracted_intermediate = -0.999999999999990 * einsum('nmej,bm,cn,eadikl->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,j>*t1(c,m)*t1(d,n)*t3(e,a,b,i,k,l)
    contracted_intermediate = -0.999999999999990 * einsum('nmej,cm,dn,eabikl->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,i>*t1(e,l)*t1(a,m)*t3(b,c,d,j,k,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmei,el,am,bcdjkn->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,i>*t1(e,l)*t1(c,m)*t3(a,b,d,j,k,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmei,el,cm,abdjkn->abcdijkl', g[o, o, v, o], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,a||e,f>*t1(e,l)*t1(f,k)*t3(b,c,d,i,j,m)
    contracted_intermediate = -0.999999999999990 * einsum('maef,el,fk,bcdijm->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,a||e,f>*t1(e,l)*t1(f,i)*t3(b,c,d,j,k,m)
    contracted_intermediate = -0.999999999999990 * einsum('maef,el,fi,bcdjkm->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<m,a||e,f>*t1(e,l)*t1(b,m)*t3(f,c,d,i,j,k)
    contracted_intermediate = -0.999999999999990 * einsum('maef,el,bm,fcdijk->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,a||e,f>*t1(e,l)*t1(d,m)*t3(f,b,c,i,j,k)
    contracted_intermediate = -0.999999999999990 * einsum('maef,el,dm,fbcijk->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,a||e,f>*t1(e,k)*t1(f,j)*t3(b,c,d,i,l,m)
    contracted_intermediate = -0.999999999999990 * einsum('maef,ek,fj,bcdilm->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(a,b)<m,a||e,f>*t1(e,j)*t1(f,i)*t3(b,c,d,k,l,m)
    contracted_intermediate = -0.999999999999990 * einsum('maef,ej,fi,bcdklm->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<m,a||e,f>*t1(e,j)*t1(b,m)*t3(f,c,d,i,k,l)
    contracted_intermediate = -0.999999999999990 * einsum('maef,ej,bm,fcdikl->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,f>*t1(e,j)*t1(d,m)*t3(f,b,c,i,k,l)
    contracted_intermediate = -0.999999999999990 * einsum('maef,ej,dm,fbcikl->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,c)<m,b||e,f>*t1(e,l)*t1(a,m)*t3(f,c,d,i,j,k)
    contracted_intermediate =  0.999999999999990 * einsum('mbef,el,am,fcdijk->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,c)<m,b||e,f>*t1(e,j)*t1(a,m)*t3(f,c,d,i,k,l)
    contracted_intermediate =  0.999999999999990 * einsum('mbef,ej,am,fcdikl->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(c,d)<m,c||e,f>*t1(e,l)*t1(f,k)*t3(a,b,d,i,j,m)
    contracted_intermediate = -0.999999999999990 * einsum('mcef,el,fk,abdijm->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,f>*t1(e,l)*t1(f,i)*t3(a,b,d,j,k,m)
    contracted_intermediate = -0.999999999999990 * einsum('mcef,el,fi,abdjkm->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,c||e,f>*t1(e,l)*t1(a,m)*t3(f,b,d,i,j,k)
    contracted_intermediate = -0.999999999999990 * einsum('mcef,el,am,fbdijk->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,f>*t1(e,l)*t1(d,m)*t3(f,a,b,i,j,k)
    contracted_intermediate = -0.999999999999990 * einsum('mcef,el,dm,fabijk->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)<m,c||e,f>*t1(e,k)*t1(f,j)*t3(a,b,d,i,l,m)
    contracted_intermediate = -0.999999999999990 * einsum('mcef,ek,fj,abdilm->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(c,d)<m,c||e,f>*t1(e,j)*t1(f,i)*t3(a,b,d,k,l,m)
    contracted_intermediate = -0.999999999999990 * einsum('mcef,ej,fi,abdklm->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,c||e,f>*t1(e,j)*t1(a,m)*t3(f,b,d,i,k,l)
    contracted_intermediate = -0.999999999999990 * einsum('mcef,ej,am,fbdikl->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<m,c||e,f>*t1(e,j)*t1(d,m)*t3(f,a,b,i,k,l)
    contracted_intermediate = -0.999999999999990 * einsum('mcef,ej,dm,fabikl->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,d||e,f>*t1(e,l)*t1(a,m)*t3(f,b,c,i,j,k)
    contracted_intermediate =  0.999999999999990 * einsum('mdef,el,am,fbcijk->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,f>*t1(e,j)*t1(a,m)*t3(f,b,c,i,k,l)
    contracted_intermediate =  0.999999999999990 * einsum('mdef,ej,am,fbcikl->abcdijkl', g[o, v, v, v], t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>*t1(e,m)*t1(f,l)*t4(a,b,c,d,i,j,k,n)
    contracted_intermediate =  1.000000000000020 * einsum('nmef,em,fl,abcdijkn->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>*t1(e,m)*t1(f,j)*t4(a,b,c,d,i,k,l,n)
    contracted_intermediate =  1.000000000000020 * einsum('nmef,em,fj,abcdikln->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>*t1(e,m)*t1(a,n)*t4(f,b,c,d,i,j,k,l)
    contracted_intermediate =  1.000000000000020 * einsum('nmef,em,an,fbcdijkl->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<n,m||e,f>*t1(e,m)*t1(c,n)*t4(f,a,b,d,i,j,k,l)
    contracted_intermediate =  1.000000000000020 * einsum('nmef,em,cn,fabdijkl->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>*t1(e,l)*t1(f,k)*t4(a,b,c,d,i,j,n,m)
    contracted_intermediate = -0.500000000000010 * einsum('nmef,el,fk,abcdijnm->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>*t1(e,l)*t1(f,i)*t4(a,b,c,d,j,k,n,m)
    contracted_intermediate = -0.500000000000010 * einsum('nmef,el,fi,abcdjknm->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,l)*t1(a,m)*t4(f,b,c,d,i,j,k,n)
    contracted_intermediate =  1.000000000000020 * einsum('nmef,el,am,fbcdijkn->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>*t1(e,l)*t1(c,m)*t4(f,a,b,d,i,j,k,n)
    contracted_intermediate =  1.000000000000020 * einsum('nmef,el,cm,fabdijkn->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 <n,m||e,f>*t1(e,k)*t1(f,j)*t4(a,b,c,d,i,l,n,m)
    quadruples_res += -0.500000000000010 * einsum('nmef,ek,fj,abcdilnm->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>*t1(e,j)*t1(f,i)*t4(a,b,c,d,k,l,n,m)
    quadruples_res += -0.500000000000010 * einsum('nmef,ej,fi,abcdklnm->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,j)*t1(a,m)*t4(f,b,c,d,i,k,l,n)
    contracted_intermediate =  1.000000000000020 * einsum('nmef,ej,am,fbcdikln->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||e,f>*t1(e,j)*t1(c,m)*t4(f,a,b,d,i,k,l,n)
    contracted_intermediate =  1.000000000000020 * einsum('nmef,ej,cm,fabdikln->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -0.5000 P(b,c)<n,m||e,f>*t1(a,m)*t1(b,n)*t4(e,f,c,d,i,j,k,l)
    contracted_intermediate = -0.500000000000010 * einsum('nmef,am,bn,efcdijkl->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate) 
    
    #	 -0.5000 P(a,b)<n,m||e,f>*t1(a,m)*t1(d,n)*t4(e,f,b,c,i,j,k,l)
    contracted_intermediate = -0.500000000000010 * einsum('nmef,am,dn,efbcijkl->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	 -0.5000 <n,m||e,f>*t1(b,m)*t1(c,n)*t4(e,f,a,d,i,j,k,l)
    quadruples_res += -0.500000000000010 * einsum('nmef,bm,cn,efadijkl->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -0.5000 <n,m||e,f>*t1(c,m)*t1(d,n)*t4(e,f,a,b,i,j,k,l)
    quadruples_res += -0.500000000000010 * einsum('nmef,cm,dn,efabijkl->abcdijkl', g[o, o, v, v], t1, t1, t4[:, :, :, :, :, :, :, :], optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||e,l>*t1(e,k)*t2(a,b,j,m)*t2(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmel,ek,abjm,cdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,l>*t1(e,k)*t2(a,d,j,m)*t2(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmel,ek,adjm,bcin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||e,l>*t1(e,j)*t2(a,b,k,m)*t2(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmel,ej,abkm,cdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||e,l>*t1(e,j)*t2(a,d,k,m)*t2(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmel,ej,adkm,bcin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||e,l>*t1(e,i)*t2(a,b,k,m)*t2(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmel,ei,abkm,cdjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,l>*t1(e,i)*t2(a,d,k,m)*t2(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmel,ei,adkm,bcjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||e,l>*t1(a,m)*t2(e,b,j,k)*t2(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmel,am,ebjk,cdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||e,l>*t1(a,m)*t2(e,b,i,j)*t2(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmel,am,ebij,cdkn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,l>*t1(a,m)*t2(e,d,j,k)*t2(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmel,am,edjk,bcin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,l>*t1(a,m)*t2(e,d,i,j)*t2(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmel,am,edij,bckn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,c)<n,m||e,l>*t1(b,m)*t2(e,a,j,k)*t2(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmel,bm,eajk,cdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,c)<n,m||e,l>*t1(b,m)*t2(e,a,i,j)*t2(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmel,bm,eaij,cdkn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,l>*t1(c,m)*t2(e,a,j,k)*t2(b,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmel,cm,eajk,bdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,l>*t1(c,m)*t2(e,a,i,j)*t2(b,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmel,cm,eaij,bdkn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<n,m||e,l>*t1(c,m)*t2(e,d,j,k)*t2(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmel,cm,edjk,abin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<n,m||e,l>*t1(c,m)*t2(e,d,i,j)*t2(a,b,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmel,cm,edij,abkn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,l>*t1(d,m)*t2(e,a,j,k)*t2(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmel,dm,eajk,bcin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,l>*t1(d,m)*t2(e,a,i,j)*t2(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmel,dm,eaij,bckn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,k>*t1(e,l)*t2(a,b,j,m)*t2(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmek,el,abjm,cdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,k>*t1(e,l)*t2(a,d,j,m)*t2(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmek,el,adjm,bcin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)*P(b,c)<n,m||e,k>*t1(e,j)*t2(a,b,l,m)*t2(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmek,ej,ablm,cdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	 -1.0000 P(i,l)<n,m||e,k>*t1(e,j)*t2(a,d,l,m)*t2(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmek,ej,adlm,bcin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	  1.0000 P(j,l)*P(b,c)<n,m||e,k>*t1(e,i)*t2(a,b,l,m)*t2(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmek,ei,ablm,cdjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  1.0000 P(j,l)<n,m||e,k>*t1(e,i)*t2(a,d,l,m)*t2(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmek,ei,adlm,bcjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,k>*t1(a,m)*t2(e,b,j,l)*t2(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmek,am,ebjl,cdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,k>*t1(a,m)*t2(e,d,j,l)*t2(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmek,am,edjl,bcin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<n,m||e,k>*t1(b,m)*t2(e,a,j,l)*t2(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmek,bm,eajl,cdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,k>*t1(c,m)*t2(e,a,j,l)*t2(b,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmek,cm,eajl,bdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||e,k>*t1(c,m)*t2(e,d,j,l)*t2(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmek,cm,edjl,abin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,k>*t1(d,m)*t2(e,a,j,l)*t2(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmek,dm,eajl,bcin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<n,m||e,j>*t1(e,l)*t2(a,b,k,m)*t2(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmej,el,abkm,cdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<n,m||e,j>*t1(e,l)*t2(a,d,k,m)*t2(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmej,el,adkm,bcin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,l)*P(b,c)<n,m||e,j>*t1(e,k)*t2(a,b,l,m)*t2(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmej,ek,ablm,cdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	  1.0000 P(i,l)<n,m||e,j>*t1(e,k)*t2(a,d,l,m)*t2(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmej,ek,adlm,bcin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||e,j>*t1(e,i)*t2(a,b,l,m)*t2(c,d,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmej,ei,ablm,cdkn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)<n,m||e,j>*t1(e,i)*t2(a,d,l,m)*t2(b,c,k,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmej,ei,adlm,bckn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<n,m||e,j>*t1(a,m)*t2(e,b,k,l)*t2(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmej,am,ebkl,cdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||e,j>*t1(a,m)*t2(e,b,i,k)*t2(c,d,l,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmej,am,ebik,cdln->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<n,m||e,j>*t1(a,m)*t2(e,d,k,l)*t2(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmej,am,edkl,bcin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,j>*t1(a,m)*t2(e,d,i,k)*t2(b,c,l,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmej,am,edik,bcln->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,c)<n,m||e,j>*t1(b,m)*t2(e,a,k,l)*t2(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmej,bm,eakl,cdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,c)<n,m||e,j>*t1(b,m)*t2(e,a,i,k)*t2(c,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmej,bm,eaik,cdln->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<n,m||e,j>*t1(c,m)*t2(e,a,k,l)*t2(b,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmej,cm,eakl,bdin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,j>*t1(c,m)*t2(e,a,i,k)*t2(b,d,l,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmej,cm,eaik,bdln->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(c,d)<n,m||e,j>*t1(c,m)*t2(e,d,k,l)*t2(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmej,cm,edkl,abin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<n,m||e,j>*t1(c,m)*t2(e,d,i,k)*t2(a,b,l,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmej,cm,edik,abln->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<n,m||e,j>*t1(d,m)*t2(e,a,k,l)*t2(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmej,dm,eakl,bcin->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,j>*t1(d,m)*t2(e,a,i,k)*t2(b,c,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmej,dm,eaik,bcln->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<n,m||e,i>*t1(e,l)*t2(a,b,k,m)*t2(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmei,el,abkm,cdjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||e,i>*t1(e,l)*t2(a,d,k,m)*t2(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmei,el,adkm,bcjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(b,c)<n,m||e,i>*t1(e,k)*t2(a,b,l,m)*t2(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmei,ek,ablm,cdjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)<n,m||e,i>*t1(e,k)*t2(a,d,l,m)*t2(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmei,ek,adlm,bcjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,i>*t1(e,j)*t2(a,b,l,m)*t2(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmei,ej,ablm,cdkn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,i>*t1(e,j)*t2(a,d,l,m)*t2(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmei,ej,adlm,bckn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<n,m||e,i>*t1(a,m)*t2(e,b,k,l)*t2(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmei,am,ebkl,cdjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<n,m||e,i>*t1(a,m)*t2(e,d,k,l)*t2(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmei,am,edkl,bcjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,c)<n,m||e,i>*t1(b,m)*t2(e,a,k,l)*t2(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmei,bm,eakl,cdjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<n,m||e,i>*t1(c,m)*t2(e,a,k,l)*t2(b,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmei,cm,eakl,bdjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(c,d)<n,m||e,i>*t1(c,m)*t2(e,d,k,l)*t2(a,b,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmei,cm,edkl,abjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||e,i>*t1(d,m)*t2(e,a,k,l)*t2(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmei,dm,eakl,bcjn->abcdijkl', g[o, o, v, o], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<m,a||e,f>*t1(e,l)*t2(f,b,j,k)*t2(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('maef,el,fbjk,cdim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<m,a||e,f>*t1(e,l)*t2(f,b,i,j)*t2(c,d,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('maef,el,fbij,cdkm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,f>*t1(e,l)*t2(f,d,j,k)*t2(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('maef,el,fdjk,bcim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,a||e,f>*t1(e,l)*t2(f,d,i,j)*t2(b,c,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('maef,el,fdij,bckm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<m,a||e,f>*t1(e,k)*t2(f,b,j,l)*t2(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('maef,ek,fbjl,cdim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,a||e,f>*t1(e,k)*t2(f,d,j,l)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('maef,ek,fdjl,bcim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<m,a||e,f>*t1(e,j)*t2(f,b,k,l)*t2(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('maef,ej,fbkl,cdim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<m,a||e,f>*t1(e,j)*t2(f,b,i,k)*t2(c,d,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('maef,ej,fbik,cdlm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,a||e,f>*t1(e,j)*t2(f,d,k,l)*t2(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('maef,ej,fdkl,bcim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,a||e,f>*t1(e,j)*t2(f,d,i,k)*t2(b,c,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('maef,ej,fdik,bclm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<m,a||e,f>*t1(e,i)*t2(f,b,k,l)*t2(c,d,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('maef,ei,fbkl,cdjm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,a||e,f>*t1(e,i)*t2(f,d,k,l)*t2(b,c,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('maef,ei,fdkl,bcjm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<m,a||e,f>*t1(b,m)*t2(e,c,k,l)*t2(f,d,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('maef,bm,eckl,fdij->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<m,a||e,f>*t1(b,m)*t2(e,c,i,l)*t2(f,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('maef,bm,ecil,fdjk->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<m,a||e,f>*t1(b,m)*t2(e,c,j,k)*t2(f,d,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('maef,bm,ecjk,fdil->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,a||e,f>*t1(d,m)*t2(e,b,k,l)*t2(f,c,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('maef,dm,ebkl,fcij->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,a||e,f>*t1(d,m)*t2(e,b,i,l)*t2(f,c,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('maef,dm,ebil,fcjk->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,a||e,f>*t1(d,m)*t2(e,b,j,k)*t2(f,c,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('maef,dm,ebjk,fcil->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,c)<m,b||e,f>*t1(e,l)*t2(f,a,j,k)*t2(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mbef,el,fajk,cdim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,c)<m,b||e,f>*t1(e,l)*t2(f,a,i,j)*t2(c,d,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mbef,el,faij,cdkm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<m,b||e,f>*t1(e,k)*t2(f,a,j,l)*t2(c,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbef,ek,fajl,cdim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,c)<m,b||e,f>*t1(e,j)*t2(f,a,k,l)*t2(c,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mbef,ej,fakl,cdim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,c)<m,b||e,f>*t1(e,j)*t2(f,a,i,k)*t2(c,d,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mbef,ej,faik,cdlm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,c)<m,b||e,f>*t1(e,i)*t2(f,a,k,l)*t2(c,d,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mbef,ei,fakl,cdjm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,c)<m,b||e,f>*t1(a,m)*t2(e,c,k,l)*t2(f,d,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('mbef,am,eckl,fdij->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,c)<m,b||e,f>*t1(a,m)*t2(e,c,i,l)*t2(f,d,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('mbef,am,ecil,fdjk->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,c)<m,b||e,f>*t1(a,m)*t2(e,c,j,k)*t2(f,d,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('mbef,am,ecjk,fdil->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,c||e,f>*t1(e,l)*t2(f,a,j,k)*t2(b,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,el,fajk,bdim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,c||e,f>*t1(e,l)*t2(f,a,i,j)*t2(b,d,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,el,faij,bdkm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<m,c||e,f>*t1(e,l)*t2(f,d,j,k)*t2(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,el,fdjk,abim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,f>*t1(e,l)*t2(f,d,i,j)*t2(a,b,k,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,el,fdij,abkm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,c||e,f>*t1(e,k)*t2(f,a,j,l)*t2(b,d,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcef,ek,fajl,bdim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<m,c||e,f>*t1(e,k)*t2(f,d,j,l)*t2(a,b,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcef,ek,fdjl,abim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,c||e,f>*t1(e,j)*t2(f,a,k,l)*t2(b,d,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,ej,fakl,bdim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,c||e,f>*t1(e,j)*t2(f,a,i,k)*t2(b,d,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,ej,faik,bdlm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(c,d)<m,c||e,f>*t1(e,j)*t2(f,d,k,l)*t2(a,b,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,ej,fdkl,abim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<m,c||e,f>*t1(e,j)*t2(f,d,i,k)*t2(a,b,l,m)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,ej,fdik,ablm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,c||e,f>*t1(e,i)*t2(f,a,k,l)*t2(b,d,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcef,ei,fakl,bdjm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(c,d)<m,c||e,f>*t1(e,i)*t2(f,d,k,l)*t2(a,b,j,m)
    contracted_intermediate =  1.000000000000000 * einsum('mcef,ei,fdkl,abjm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,c||e,f>*t1(a,m)*t2(e,b,k,l)*t2(f,d,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,am,ebkl,fdij->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<m,c||e,f>*t1(a,m)*t2(e,b,i,l)*t2(f,d,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,am,ebil,fdjk->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<m,c||e,f>*t1(a,m)*t2(e,b,j,k)*t2(f,d,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,am,ebjk,fdil->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(c,d)<m,c||e,f>*t1(d,m)*t2(e,a,k,l)*t2(f,b,i,j)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,dm,eakl,fbij->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<m,c||e,f>*t1(d,m)*t2(e,a,i,l)*t2(f,b,j,k)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,dm,eail,fbjk->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(c,d)<m,c||e,f>*t1(d,m)*t2(e,a,j,k)*t2(f,b,i,l)
    contracted_intermediate = -1.000000000000000 * einsum('mcef,dm,eajk,fbil->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,f>*t1(e,l)*t2(f,a,j,k)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdef,el,fajk,bcim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,d||e,f>*t1(e,l)*t2(f,a,i,j)*t2(b,c,k,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdef,el,faij,bckm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<m,d||e,f>*t1(e,k)*t2(f,a,j,l)*t2(b,c,i,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdef,ek,fajl,bcim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,d||e,f>*t1(e,j)*t2(f,a,k,l)*t2(b,c,i,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdef,ej,fakl,bcim->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<m,d||e,f>*t1(e,j)*t2(f,a,i,k)*t2(b,c,l,m)
    contracted_intermediate =  1.000000000000000 * einsum('mdef,ej,faik,bclm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<m,d||e,f>*t1(e,i)*t2(f,a,k,l)*t2(b,c,j,m)
    contracted_intermediate = -1.000000000000000 * einsum('mdef,ei,fakl,bcjm->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<m,d||e,f>*t1(a,m)*t2(e,b,k,l)*t2(f,c,i,j)
    contracted_intermediate =  1.000000000000000 * einsum('mdef,am,ebkl,fcij->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<m,d||e,f>*t1(a,m)*t2(e,b,i,l)*t2(f,c,j,k)
    contracted_intermediate =  1.000000000000000 * einsum('mdef,am,ebil,fcjk->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<m,d||e,f>*t1(a,m)*t2(e,b,j,k)*t2(f,c,i,l)
    contracted_intermediate =  1.000000000000000 * einsum('mdef,am,ebjk,fcil->abcdijkl', g[o, v, v, v], t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||e,f>*t1(e,m)*t2(f,a,k,l)*t3(b,c,d,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,em,fakl,bcdijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,m)*t2(f,a,i,l)*t3(b,c,d,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,em,fail,bcdjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<n,m||e,f>*t1(e,m)*t2(f,a,j,k)*t3(b,c,d,i,l,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,em,fajk,bcdiln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(c,d)<n,m||e,f>*t1(e,m)*t2(f,c,k,l)*t3(a,b,d,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,em,fckl,abdijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<n,m||e,f>*t1(e,m)*t2(f,c,i,l)*t3(a,b,d,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,em,fcil,abdjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(c,d)<n,m||e,f>*t1(e,m)*t2(f,c,j,k)*t3(a,b,d,i,l,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,em,fcjk,abdiln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||e,f>*t1(e,m)*t2(a,b,l,n)*t3(f,c,d,i,j,k)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,em,abln,fcdijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||e,f>*t1(e,m)*t2(a,b,j,n)*t3(f,c,d,i,k,l)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,em,abjn,fcdikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,m)*t2(a,d,l,n)*t3(f,b,c,i,j,k)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,em,adln,fbcijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,m)*t2(a,d,j,n)*t3(f,b,c,i,k,l)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,em,adjn,fbcikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,d)<n,m||e,f>*t1(e,m)*t2(b,c,l,n)*t3(f,a,d,i,j,k)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,em,bcln,fadijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,d)<n,m||e,f>*t1(e,m)*t2(b,c,j,n)*t3(f,a,d,i,k,l)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,em,bcjn,fadikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<n,m||e,f>*t1(e,l)*t2(f,a,k,m)*t3(b,c,d,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,fakm,bcdijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,l)*t2(f,a,i,m)*t3(b,c,d,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,faim,bcdjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,l)*t2(f,a,j,k)*t3(b,c,d,i,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,el,fajk,bcdinm->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,l)*t2(f,a,i,j)*t3(b,c,d,k,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,el,faij,bcdknm->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(c,d)<n,m||e,f>*t1(e,l)*t2(f,c,k,m)*t3(a,b,d,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,fckm,abdijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>*t1(e,l)*t2(f,c,i,m)*t3(a,b,d,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,fcim,abdjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(c,d)<n,m||e,f>*t1(e,l)*t2(f,c,j,k)*t3(a,b,d,i,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,el,fcjk,abdinm->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(c,d)<n,m||e,f>*t1(e,l)*t2(f,c,i,j)*t3(a,b,d,k,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,el,fcij,abdknm->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<n,m||e,f>*t1(e,l)*t2(a,b,n,m)*t3(f,c,d,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,el,abnm,fcdijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<n,m||e,f>*t1(e,l)*t2(a,b,k,m)*t3(f,c,d,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,abkm,fcdijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>*t1(e,l)*t2(a,b,i,m)*t3(f,c,d,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,abim,fcdjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,l)*t2(a,d,n,m)*t3(f,b,c,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,el,adnm,fbcijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<n,m||e,f>*t1(e,l)*t2(a,d,k,m)*t3(f,b,c,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,adkm,fbcijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,l)*t2(a,d,i,m)*t3(f,b,c,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,adim,fbcjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,d)<n,m||e,f>*t1(e,l)*t2(b,c,n,m)*t3(f,a,d,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,el,bcnm,fadijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,d)<n,m||e,f>*t1(e,l)*t2(b,c,k,m)*t3(f,a,d,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,bckm,fadijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,d)<n,m||e,f>*t1(e,l)*t2(b,c,i,m)*t3(f,a,d,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,bcim,fadjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(a,b)<n,m||e,f>*t1(e,k)*t2(f,a,l,m)*t3(b,c,d,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,ek,falm,bcdijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdilkj', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,k)*t2(f,a,j,l)*t3(b,c,d,i,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,ek,fajl,bcdinm->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(c,d)<n,m||e,f>*t1(e,k)*t2(f,c,l,m)*t3(a,b,d,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,ek,fclm,abdijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcilkj', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(c,d)<n,m||e,f>*t1(e,k)*t2(f,c,j,l)*t3(a,b,d,i,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,ek,fcjl,abdinm->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(b,c)<n,m||e,f>*t1(e,k)*t2(a,b,l,m)*t3(f,c,d,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,ek,ablm,fcdijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(a,b)<n,m||e,f>*t1(e,k)*t2(a,d,l,m)*t3(f,b,c,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,ek,adlm,fbcijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(b,d)<n,m||e,f>*t1(e,k)*t2(b,c,l,m)*t3(f,a,d,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,ek,bclm,fadijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,j)*t2(f,a,l,m)*t3(b,c,d,i,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,falm,bcdikn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,j)*t2(f,a,i,m)*t3(b,c,d,k,l,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,faim,bcdkln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(a,b)<n,m||e,f>*t1(e,j)*t2(f,a,k,l)*t3(b,c,d,i,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ej,fakl,bcdinm->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,j)*t2(f,a,i,k)*t3(b,c,d,l,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ej,faik,bcdlnm->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>*t1(e,j)*t2(f,c,l,m)*t3(a,b,d,i,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,fclm,abdikn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||e,f>*t1(e,j)*t2(f,c,i,m)*t3(a,b,d,k,l,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,fcim,abdkln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(c,d)<n,m||e,f>*t1(e,j)*t2(f,c,k,l)*t3(a,b,d,i,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ej,fckl,abdinm->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(c,d)<n,m||e,f>*t1(e,j)*t2(f,c,i,k)*t3(a,b,d,l,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ej,fcik,abdlnm->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,c)<n,m||e,f>*t1(e,j)*t2(a,b,n,m)*t3(f,c,d,i,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ej,abnm,fcdikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>*t1(e,j)*t2(a,b,l,m)*t3(f,c,d,i,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,ablm,fcdikn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>*t1(e,j)*t2(a,b,i,m)*t3(f,c,d,k,l,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,abim,fcdkln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,j)*t2(a,d,n,m)*t3(f,b,c,i,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ej,adnm,fbcikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,j)*t2(a,d,l,m)*t3(f,b,c,i,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,adlm,fbcikn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,j)*t2(a,d,i,m)*t3(f,b,c,k,l,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,adim,fbckln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,d)<n,m||e,f>*t1(e,j)*t2(b,c,n,m)*t3(f,a,d,i,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ej,bcnm,fadikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,d)<n,m||e,f>*t1(e,j)*t2(b,c,l,m)*t3(f,a,d,i,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,bclm,fadikn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,d)<n,m||e,f>*t1(e,j)*t2(b,c,i,m)*t3(f,a,d,k,l,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,bcim,fadkln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,i)*t2(f,a,l,m)*t3(b,c,d,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,ei,falm,bcdjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(a,b)<n,m||e,f>*t1(e,i)*t2(f,a,k,l)*t3(b,c,d,j,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,ei,fakl,bcdjnm->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(c,d)<n,m||e,f>*t1(e,i)*t2(f,c,l,m)*t3(a,b,d,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,ei,fclm,abdjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  0.5000 P(j,k)*P(c,d)<n,m||e,f>*t1(e,i)*t2(f,c,k,l)*t3(a,b,d,j,n,m)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,ei,fckl,abdjnm->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,c)<n,m||e,f>*t1(e,i)*t2(a,b,l,m)*t3(f,c,d,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,ei,ablm,fcdjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,i)*t2(a,d,l,m)*t3(f,b,c,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,ei,adlm,fbcjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(b,d)<n,m||e,f>*t1(e,i)*t2(b,c,l,m)*t3(f,a,d,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,ei,bclm,fadjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<n,m||e,f>*t1(a,m)*t2(e,f,k,l)*t3(b,c,d,i,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,am,efkl,bcdijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>*t1(a,m)*t2(e,f,i,l)*t3(b,c,d,j,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,am,efil,bcdjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(a,b)<n,m||e,f>*t1(a,m)*t2(e,f,j,k)*t3(b,c,d,i,l,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,am,efjk,bcdiln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>*t1(a,m)*t2(e,b,l,n)*t3(f,c,d,i,j,k)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,am,ebln,fcdijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>*t1(a,m)*t2(e,b,j,n)*t3(f,c,d,i,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,am,ebjn,fcdikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<n,m||e,f>*t1(a,m)*t2(e,b,k,l)*t3(f,c,d,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,am,ebkl,fcdijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>*t1(a,m)*t2(e,b,i,l)*t3(f,c,d,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,am,ebil,fcdjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||e,f>*t1(a,m)*t2(e,b,j,k)*t3(f,c,d,i,l,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,am,ebjk,fcdiln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(a,m)*t2(e,d,l,n)*t3(f,b,c,i,j,k)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,am,edln,fbcijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(a,m)*t2(e,d,j,n)*t3(f,b,c,i,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,am,edjn,fbcikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<n,m||e,f>*t1(a,m)*t2(e,d,k,l)*t3(f,b,c,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,am,edkl,fbcijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(a,m)*t2(e,d,i,l)*t3(f,b,c,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,am,edil,fbcjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<n,m||e,f>*t1(a,m)*t2(e,d,j,k)*t3(f,b,c,i,l,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,am,edjk,fbciln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(c,d)<n,m||e,f>*t1(a,m)*t2(b,c,l,n)*t3(e,f,d,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,am,bcln,efdijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(c,d)<n,m||e,f>*t1(a,m)*t2(b,c,j,n)*t3(e,f,d,i,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,am,bcjn,efdikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>*t1(a,m)*t2(c,d,l,n)*t3(e,f,b,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,am,cdln,efbijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(a,b)<n,m||e,f>*t1(a,m)*t2(c,d,j,n)*t3(e,f,b,i,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,am,cdjn,efbikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,c)<n,m||e,f>*t1(b,m)*t2(e,a,l,n)*t3(f,c,d,i,j,k)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,bm,ealn,fcdijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<n,m||e,f>*t1(b,m)*t2(e,a,j,n)*t3(f,c,d,i,k,l)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,bm,eajn,fcdikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,c)<n,m||e,f>*t1(b,m)*t2(e,a,k,l)*t3(f,c,d,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,bm,eakl,fcdijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,c)<n,m||e,f>*t1(b,m)*t2(e,a,i,l)*t3(f,c,d,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,bm,eail,fcdjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,c)<n,m||e,f>*t1(b,m)*t2(e,a,j,k)*t3(f,c,d,i,l,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,bm,eajk,fcdiln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadkjil', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(c,d)<n,m||e,f>*t1(b,m)*t2(a,c,l,n)*t3(e,f,d,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,bm,acln,efdijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(c,d)<n,m||e,f>*t1(b,m)*t2(a,c,j,n)*t3(e,f,d,i,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,bm,acjn,efdikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(c,d)<n,m||e,f>*t1(c,m)*t2(e,f,k,l)*t3(a,b,d,i,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,cm,efkl,abdijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(c,d)<n,m||e,f>*t1(c,m)*t2(e,f,i,l)*t3(a,b,d,j,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,cm,efil,abdjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(c,d)<n,m||e,f>*t1(c,m)*t2(e,f,j,k)*t3(a,b,d,i,l,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,cm,efjk,abdiln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(c,m)*t2(e,a,l,n)*t3(f,b,d,i,j,k)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,cm,ealn,fbdijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(c,m)*t2(e,a,j,n)*t3(f,b,d,i,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,cm,eajn,fbdikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<n,m||e,f>*t1(c,m)*t2(e,a,k,l)*t3(f,b,d,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,cm,eakl,fbdijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(c,m)*t2(e,a,i,l)*t3(f,b,d,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,cm,eail,fbdjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<n,m||e,f>*t1(c,m)*t2(e,a,j,k)*t3(f,b,d,i,l,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,cm,eajk,fbdiln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>*t1(c,m)*t2(e,d,l,n)*t3(f,a,b,i,j,k)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,cm,edln,fabijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||e,f>*t1(c,m)*t2(e,d,j,n)*t3(f,a,b,i,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,cm,edjn,fabikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(c,d)<n,m||e,f>*t1(c,m)*t2(e,d,k,l)*t3(f,a,b,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,cm,edkl,fabijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>*t1(c,m)*t2(e,d,i,l)*t3(f,a,b,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,cm,edil,fabjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(c,d)<n,m||e,f>*t1(c,m)*t2(e,d,j,k)*t3(f,a,b,i,l,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,cm,edjk,fabiln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,d)<n,m||e,f>*t1(c,m)*t2(a,b,l,n)*t3(e,f,d,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,cm,abln,efdijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,d)<n,m||e,f>*t1(c,m)*t2(a,b,j,n)*t3(e,f,d,i,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,cm,abjn,efdikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->adcbijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->adcbjikl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(c,d)<n,m||e,f>*t1(c,m)*t2(b,d,l,n)*t3(e,f,a,i,j,k)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,cm,bdln,efaijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(c,d)<n,m||e,f>*t1(c,m)*t2(b,d,j,n)*t3(e,f,a,i,k,l)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,cm,bdjn,efaikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(d,m)*t2(e,a,l,n)*t3(f,b,c,i,j,k)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,dm,ealn,fbcijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(d,m)*t2(e,a,j,n)*t3(f,b,c,i,k,l)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,dm,eajn,fbcikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||e,f>*t1(d,m)*t2(e,a,k,l)*t3(f,b,c,i,j,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,dm,eakl,fbcijn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(d,m)*t2(e,a,i,l)*t3(f,b,c,j,k,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,dm,eail,fbcjkn->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<n,m||e,f>*t1(d,m)*t2(e,a,j,k)*t3(f,b,c,i,l,n)
    contracted_intermediate = -1.000000000000010 * einsum('nmef,dm,eajk,fbciln->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  0.5000 P(k,l)*P(b,c)<n,m||e,f>*t1(d,m)*t2(a,b,l,n)*t3(e,f,c,i,j,k)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,dm,abln,efcijk->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  0.5000 P(i,j)*P(b,c)<n,m||e,f>*t1(d,m)*t2(a,b,j,n)*t3(e,f,c,i,k,l)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,dm,abjn,efcikl->abcdijkl', g[o, o, v, v], t1, t2, t3, optimize=['einsum_path', (0, 1), (1, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)*P(b,c)<n,m||e,f>*t2(e,f,k,l)*t2(a,b,j,m)*t2(c,d,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efkl,abjm,cdin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(i,j)<n,m||e,f>*t2(e,f,k,l)*t2(a,d,j,m)*t2(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efkl,adjm,bcin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  0.5000 P(i,k)*P(b,c)<n,m||e,f>*t2(e,f,j,l)*t2(a,b,k,m)*t2(c,d,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efjl,abkm,cdin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  0.5000 P(i,k)<n,m||e,f>*t2(e,f,j,l)*t2(a,d,k,m)*t2(b,c,i,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efjl,adkm,bcin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(b,c)<n,m||e,f>*t2(e,f,i,l)*t2(a,b,k,m)*t2(c,d,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efil,abkm,cdjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>*t2(e,f,i,l)*t2(a,d,k,m)*t2(b,c,j,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efil,adkm,bcjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(i,l)*P(b,c)<n,m||e,f>*t2(e,f,j,k)*t2(a,b,l,m)*t2(c,d,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efjk,ablm,cdin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,l)<n,m||e,f>*t2(e,f,j,k)*t2(a,d,l,m)*t2(b,c,i,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efjk,adlm,bcin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	  0.5000 P(j,l)*P(b,c)<n,m||e,f>*t2(e,f,i,k)*t2(a,b,l,m)*t2(c,d,j,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efik,ablm,cdjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	  0.5000 P(j,l)<n,m||e,f>*t2(e,f,i,k)*t2(a,d,l,m)*t2(b,c,j,n)
    contracted_intermediate =  0.500000000000000 * einsum('nmef,efik,adlm,bcjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<n,m||e,f>*t2(e,f,i,j)*t2(a,b,l,m)*t2(c,d,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efij,ablm,cdkn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>*t2(e,f,i,j)*t2(a,d,l,m)*t2(b,c,k,n)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,efij,adlm,bckn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>*t2(e,a,l,m)*t2(f,b,j,k)*t2(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ealm,fbjk,cdin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>*t2(e,a,l,m)*t2(f,b,i,j)*t2(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ealm,fbij,cdkn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t2(e,a,l,m)*t2(f,d,j,k)*t2(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ealm,fdjk,bcin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t2(e,a,l,m)*t2(f,d,i,j)*t2(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ealm,fdij,bckn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||e,f>*t2(e,a,k,m)*t2(f,b,j,l)*t2(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eakm,fbjl,cdin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>*t2(e,a,k,m)*t2(f,d,j,l)*t2(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eakm,fdjl,bcin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||e,f>*t2(e,a,j,m)*t2(f,b,k,l)*t2(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fbkl,cdin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>*t2(e,a,j,m)*t2(f,b,i,k)*t2(c,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fbik,cdln->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<n,m||e,f>*t2(e,a,j,m)*t2(f,d,k,l)*t2(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fdkl,bcin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t2(e,a,j,m)*t2(f,d,i,k)*t2(b,c,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajm,fdik,bcln->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||e,f>*t2(e,a,i,m)*t2(f,b,k,l)*t2(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaim,fbkl,cdjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||e,f>*t2(e,a,i,m)*t2(f,d,k,l)*t2(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaim,fdkl,bcjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>*t2(e,a,k,l)*t2(f,b,j,m)*t2(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eakl,fbjm,cdin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(b,c)<n,m||e,f>*t2(e,a,k,l)*t2(f,b,i,j)*t2(c,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eakl,fbij,cdnm->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t2(e,a,k,l)*t2(f,d,j,m)*t2(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eakl,fdjm,bcin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)*P(a,b)<n,m||e,f>*t2(e,a,k,l)*t2(f,d,i,j)*t2(b,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eakl,fdij,bcnm->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<n,m||e,f>*t2(e,a,j,l)*t2(f,b,k,m)*t2(c,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eajl,fbkm,cdin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<n,m||e,f>*t2(e,a,j,l)*t2(f,d,k,m)*t2(b,c,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eajl,fdkm,bcin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<n,m||e,f>*t2(e,a,i,l)*t2(f,b,k,m)*t2(c,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eail,fbkm,cdjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(b,c)<n,m||e,f>*t2(e,a,i,l)*t2(f,b,j,k)*t2(c,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eail,fbjk,cdnm->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<n,m||e,f>*t2(e,a,i,l)*t2(f,d,k,m)*t2(b,c,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eail,fdkm,bcjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)*P(a,b)<n,m||e,f>*t2(e,a,i,l)*t2(f,d,j,k)*t2(b,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eail,fdjk,bcnm->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,l)*P(b,c)<n,m||e,f>*t2(e,a,j,k)*t2(f,b,l,m)*t2(c,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajk,fblm,cdin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(b,c)<n,m||e,f>*t2(e,a,j,k)*t2(f,b,i,l)*t2(c,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eajk,fbil,cdnm->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,l)*P(a,b)<n,m||e,f>*t2(e,a,j,k)*t2(f,d,l,m)*t2(b,c,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eajk,fdlm,bcin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)*P(a,b)<n,m||e,f>*t2(e,a,j,k)*t2(f,d,i,l)*t2(b,c,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eajk,fdil,bcnm->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(b,c)<n,m||e,f>*t2(e,a,i,k)*t2(f,b,l,m)*t2(c,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaik,fblm,cdjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(a,b)<n,m||e,f>*t2(e,a,i,k)*t2(f,d,l,m)*t2(b,c,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eaik,fdlm,bcjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>*t2(e,a,i,j)*t2(f,b,l,m)*t2(c,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaij,fblm,cdkn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t2(e,a,i,j)*t2(f,d,l,m)*t2(b,c,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eaij,fdlm,bckn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>*t2(e,b,l,m)*t2(f,c,j,k)*t2(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eblm,fcjk,adin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>*t2(e,b,l,m)*t2(f,c,i,j)*t2(a,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eblm,fcij,adkn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>*t2(e,b,k,m)*t2(f,c,j,l)*t2(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebkm,fcjl,adin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||e,f>*t2(e,b,j,m)*t2(f,c,k,l)*t2(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebjm,fckl,adin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>*t2(e,b,j,m)*t2(f,c,i,k)*t2(a,d,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebjm,fcik,adln->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>*t2(e,b,i,m)*t2(f,c,k,l)*t2(a,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebim,fckl,adjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>*t2(e,b,k,l)*t2(f,c,j,m)*t2(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebkl,fcjm,adin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>*t2(e,b,k,l)*t2(f,c,i,j)*t2(a,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ebkl,fcij,adnm->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<n,m||e,f>*t2(e,b,j,l)*t2(f,c,k,m)*t2(a,d,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebjl,fckm,adin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||e,f>*t2(e,b,i,l)*t2(f,c,k,m)*t2(a,d,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebil,fckm,adjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>*t2(e,b,i,l)*t2(f,c,j,k)*t2(a,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ebil,fcjk,adnm->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,l)<n,m||e,f>*t2(e,b,j,k)*t2(f,c,l,m)*t2(a,d,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebjk,fclm,adin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)<n,m||e,f>*t2(e,b,j,k)*t2(f,c,i,l)*t2(a,d,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ebjk,fcil,adnm->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)<n,m||e,f>*t2(e,b,i,k)*t2(f,c,l,m)*t2(a,d,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ebik,fclm,adjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>*t2(e,b,i,j)*t2(f,c,l,m)*t2(a,d,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ebij,fclm,adkn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>*t2(e,c,l,m)*t2(f,d,j,k)*t2(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eclm,fdjk,abin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>*t2(e,c,l,m)*t2(f,d,i,j)*t2(a,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eclm,fdij,abkn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)<n,m||e,f>*t2(e,c,k,m)*t2(f,d,j,l)*t2(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,eckm,fdjl,abin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||e,f>*t2(e,c,j,m)*t2(f,d,k,l)*t2(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecjm,fdkl,abin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>*t2(e,c,j,m)*t2(f,d,i,k)*t2(a,b,l,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecjm,fdik,abln->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)<n,m||e,f>*t2(e,c,i,m)*t2(f,d,k,l)*t2(a,b,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecim,fdkl,abjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>*t2(e,c,k,l)*t2(f,d,j,m)*t2(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,eckl,fdjm,abin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -0.5000 P(j,k)<n,m||e,f>*t2(e,c,k,l)*t2(f,d,i,j)*t2(a,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,eckl,fdij,abnm->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<n,m||e,f>*t2(e,c,j,l)*t2(f,d,k,m)*t2(a,b,i,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecjl,fdkm,abin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||e,f>*t2(e,c,i,l)*t2(f,d,k,m)*t2(a,b,j,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecil,fdkm,abjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	 -0.5000 P(k,l)<n,m||e,f>*t2(e,c,i,l)*t2(f,d,j,k)*t2(a,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ecil,fdjk,abnm->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,l)<n,m||e,f>*t2(e,c,j,k)*t2(f,d,l,m)*t2(a,b,i,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecjk,fdlm,abin->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	 -0.5000 P(i,k)<n,m||e,f>*t2(e,c,j,k)*t2(f,d,i,l)*t2(a,b,n,m)
    contracted_intermediate = -0.500000000000000 * einsum('nmef,ecjk,fdil,abnm->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)<n,m||e,f>*t2(e,c,i,k)*t2(f,d,l,m)*t2(a,b,j,n)
    contracted_intermediate = -1.000000000000000 * einsum('nmef,ecik,fdlm,abjn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>*t2(e,c,i,j)*t2(f,d,l,m)*t2(a,b,k,n)
    contracted_intermediate =  1.000000000000000 * einsum('nmef,ecij,fdlm,abkn->abcdijkl', g[o, o, v, v], t2, t2, t2, optimize=['einsum_path', (0, 2), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<n,m||e,f>*t1(e,l)*t1(f,k)*t1(a,m)*t3(b,c,d,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,fk,am,bcdijn->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(c,d)<n,m||e,f>*t1(e,l)*t1(f,k)*t1(c,m)*t3(a,b,d,i,j,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,fk,cm,abdijn->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,l)*t1(f,i)*t1(a,m)*t3(b,c,d,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,fi,am,bcdjkn->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>*t1(e,l)*t1(f,i)*t1(c,m)*t3(a,b,d,j,k,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,fi,cm,abdjkn->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>*t1(e,l)*t1(a,m)*t1(b,n)*t3(f,c,d,i,j,k)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,am,bn,fcdijk->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,l)*t1(a,m)*t1(d,n)*t3(f,b,c,i,j,k)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,am,dn,fbcijk->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>*t1(e,l)*t1(b,m)*t1(c,n)*t3(f,a,d,i,j,k)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,bm,cn,fadijk->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>*t1(e,l)*t1(c,m)*t1(d,n)*t3(f,a,b,i,j,k)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,el,cm,dn,fabijk->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>*t1(e,k)*t1(f,j)*t1(a,m)*t3(b,c,d,i,l,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ek,fj,am,bcdiln->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<n,m||e,f>*t1(e,k)*t1(f,j)*t1(c,m)*t3(a,b,d,i,l,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ek,fj,cm,abdiln->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(a,b)<n,m||e,f>*t1(e,j)*t1(f,i)*t1(a,m)*t3(b,c,d,k,l,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,fi,am,bcdkln->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate) 
    
    #	  1.0000 P(c,d)<n,m||e,f>*t1(e,j)*t1(f,i)*t1(c,m)*t3(a,b,d,k,l,n)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,fi,cm,abdkln->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>*t1(e,j)*t1(a,m)*t1(b,n)*t3(f,c,d,i,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,am,bn,fcdikl->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,j)*t1(a,m)*t1(d,n)*t3(f,b,c,i,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,am,dn,fbcikl->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>*t1(e,j)*t1(b,m)*t1(c,n)*t3(f,a,d,i,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,bm,cn,fadikl->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>*t1(e,j)*t1(c,m)*t1(d,n)*t3(f,a,b,i,k,l)
    contracted_intermediate =  1.000000000000010 * einsum('nmef,ej,cm,dn,fabikl->abcdijkl', g[o, o, v, v], t1, t1, t1, t3, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>*t1(e,l)*t1(f,k)*t2(a,b,j,m)*t2(c,d,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,el,fk,abjm,cdin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)<n,m||e,f>*t1(e,l)*t1(f,k)*t2(a,d,j,m)*t2(b,c,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,el,fk,adjm,bcin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(b,c)<n,m||e,f>*t1(e,l)*t1(f,j)*t2(a,b,k,m)*t2(c,d,i,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,el,fj,abkm,cdin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)<n,m||e,f>*t1(e,l)*t1(f,j)*t2(a,d,k,m)*t2(b,c,i,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,el,fj,adkm,bcin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<n,m||e,f>*t1(e,l)*t1(f,i)*t2(a,b,k,m)*t2(c,d,j,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,el,fi,abkm,cdjn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||e,f>*t1(e,l)*t1(f,i)*t2(a,d,k,m)*t2(b,c,j,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,el,fi,adkm,bcjn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>*t1(e,l)*t1(a,m)*t2(f,b,j,k)*t2(c,d,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,el,am,fbjk,cdin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>*t1(e,l)*t1(a,m)*t2(f,b,i,j)*t2(c,d,k,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,el,am,fbij,cdkn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,l)*t1(a,m)*t2(f,d,j,k)*t2(b,c,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,el,am,fdjk,bcin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,l)*t1(a,m)*t2(f,d,i,j)*t2(b,c,k,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,el,am,fdij,bckn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<n,m||e,f>*t1(e,l)*t1(b,m)*t2(f,a,j,k)*t2(c,d,i,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,el,bm,fajk,cdin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,c)<n,m||e,f>*t1(e,l)*t1(b,m)*t2(f,a,i,j)*t2(c,d,k,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,el,bm,faij,cdkn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,l)*t1(c,m)*t2(f,a,j,k)*t2(b,d,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,el,cm,fajk,bdin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,l)*t1(c,m)*t2(f,a,i,j)*t2(b,d,k,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,el,cm,faij,bdkn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||e,f>*t1(e,l)*t1(c,m)*t2(f,d,j,k)*t2(a,b,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,el,cm,fdjk,abin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(c,d)<n,m||e,f>*t1(e,l)*t1(c,m)*t2(f,d,i,j)*t2(a,b,k,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,el,cm,fdij,abkn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcijlk', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,l)*t1(d,m)*t2(f,a,j,k)*t2(b,c,i,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,el,dm,fajk,bcin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(e,l)*t1(d,m)*t2(f,a,i,j)*t2(b,c,k,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,el,dm,faij,bckn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,l)*P(b,c)<n,m||e,f>*t1(e,k)*t1(f,j)*t2(a,b,l,m)*t2(c,d,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ek,fj,ablm,cdin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdljki', contracted_intermediate) 
    
    #	  1.0000 P(i,l)<n,m||e,f>*t1(e,k)*t1(f,j)*t2(a,d,l,m)*t2(b,c,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ek,fj,adlm,bcin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdljki', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)*P(b,c)<n,m||e,f>*t1(e,k)*t1(f,i)*t2(a,b,l,m)*t2(c,d,j,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ek,fi,ablm,cdjn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(j,l)<n,m||e,f>*t1(e,k)*t1(f,i)*t2(a,d,l,m)*t2(b,c,j,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ek,fi,adlm,bcjn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdilkj', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(b,c)<n,m||e,f>*t1(e,k)*t1(a,m)*t2(f,b,j,l)*t2(c,d,i,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ek,am,fbjl,cdin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,k)*t1(a,m)*t2(f,d,j,l)*t2(b,c,i,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ek,am,fdjl,bcin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,c)<n,m||e,f>*t1(e,k)*t1(b,m)*t2(f,a,j,l)*t2(c,d,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ek,bm,fajl,cdin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,k)*t1(c,m)*t2(f,a,j,l)*t2(b,d,i,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ek,cm,fajl,bdin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(c,d)<n,m||e,f>*t1(e,k)*t1(c,m)*t2(f,d,j,l)*t2(a,b,i,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ek,cm,fdjl,abin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,k)*t1(d,m)*t2(f,a,j,l)*t2(b,c,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ek,dm,fajl,bcin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>*t1(e,j)*t1(f,i)*t2(a,b,l,m)*t2(c,d,k,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ej,fi,ablm,cdkn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>*t1(e,j)*t1(f,i)*t2(a,d,l,m)*t2(b,c,k,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ej,fi,adlm,bckn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||e,f>*t1(e,j)*t1(a,m)*t2(f,b,k,l)*t2(c,d,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ej,am,fbkl,cdin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(b,c)<n,m||e,f>*t1(e,j)*t1(a,m)*t2(f,b,i,k)*t2(c,d,l,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ej,am,fbik,cdln->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<n,m||e,f>*t1(e,j)*t1(a,m)*t2(f,d,k,l)*t2(b,c,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ej,am,fdkl,bcin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,j)*t1(a,m)*t2(f,d,i,k)*t2(b,c,l,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ej,am,fdik,bcln->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,c)<n,m||e,f>*t1(e,j)*t1(b,m)*t2(f,a,k,l)*t2(c,d,i,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ej,bm,fakl,cdin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,c)<n,m||e,f>*t1(e,j)*t1(b,m)*t2(f,a,i,k)*t2(c,d,l,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ej,bm,faik,cdln->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<n,m||e,f>*t1(e,j)*t1(c,m)*t2(f,a,k,l)*t2(b,d,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ej,cm,fakl,bdin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,j)*t1(c,m)*t2(f,a,i,k)*t2(b,d,l,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ej,cm,faik,bdln->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(c,d)<n,m||e,f>*t1(e,j)*t1(c,m)*t2(f,d,k,l)*t2(a,b,i,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ej,cm,fdkl,abin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdckjil', contracted_intermediate) 
    
    #	  1.0000 P(i,j)*P(c,d)<n,m||e,f>*t1(e,j)*t1(c,m)*t2(f,d,i,k)*t2(a,b,l,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ej,cm,fdik,abln->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcjikl', contracted_intermediate) 
    
    #	 -1.0000 P(i,k)*P(a,b)<n,m||e,f>*t1(e,j)*t1(d,m)*t2(f,a,k,l)*t2(b,c,i,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ej,dm,fakl,bcin->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	 -1.0000 P(i,j)*P(a,b)<n,m||e,f>*t1(e,j)*t1(d,m)*t2(f,a,i,k)*t2(b,c,l,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ej,dm,faik,bcln->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdjikl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdjikl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(b,c)<n,m||e,f>*t1(e,i)*t1(a,m)*t2(f,b,k,l)*t2(c,d,j,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ei,am,fbkl,cdjn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||e,f>*t1(e,i)*t1(a,m)*t2(f,d,k,l)*t2(b,c,j,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ei,am,fdkl,bcjn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,c)<n,m||e,f>*t1(e,i)*t1(b,m)*t2(f,a,k,l)*t2(c,d,j,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ei,bm,fakl,cdjn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->cbadijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->cbadikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(a,b)<n,m||e,f>*t1(e,i)*t1(c,m)*t2(f,a,k,l)*t2(b,d,j,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ei,cm,fakl,bdjn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	 -1.0000 P(j,k)*P(c,d)<n,m||e,f>*t1(e,i)*t1(c,m)*t2(f,d,k,l)*t2(a,b,j,n)
    contracted_intermediate = -0.999999999999990 * einsum('nmef,ei,cm,fdkl,abjn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->abdcijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->abdcikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<n,m||e,f>*t1(e,i)*t1(d,m)*t2(f,a,k,l)*t2(b,c,j,n)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,ei,dm,fakl,bcjn->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(b,c)<n,m||e,f>*t1(a,m)*t1(b,n)*t2(e,c,k,l)*t2(f,d,i,j)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,am,bn,eckl,fdij->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(b,c)<n,m||e,f>*t1(a,m)*t1(b,n)*t2(e,c,i,l)*t2(f,d,j,k)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,am,bn,ecil,fdjk->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(b,c)<n,m||e,f>*t1(a,m)*t1(b,n)*t2(e,c,j,k)*t2(f,d,i,l)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,am,bn,ecjk,fdil->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->acbdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->acbdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)*P(a,b)<n,m||e,f>*t1(a,m)*t1(d,n)*t2(e,b,k,l)*t2(f,c,i,j)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,am,dn,ebkl,fcij->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)*P(a,b)<n,m||e,f>*t1(a,m)*t1(d,n)*t2(e,b,i,l)*t2(f,c,j,k)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,am,dn,ebil,fcjk->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)*P(a,b)<n,m||e,f>*t1(a,m)*t1(d,n)*t2(e,b,j,k)*t2(f,c,i,l)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,am,dn,ebjk,fcil->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate)  + -1.00000 * einsum('abcdijkl->bacdijkl', contracted_intermediate)  +  1.00000 * einsum('abcdijkl->bacdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||e,f>*t1(b,m)*t1(c,n)*t2(e,a,k,l)*t2(f,d,i,j)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,bm,cn,eakl,fdij->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>*t1(b,m)*t1(c,n)*t2(e,a,i,l)*t2(f,d,j,k)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,bm,cn,eail,fdjk->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||e,f>*t1(b,m)*t1(c,n)*t2(e,a,j,k)*t2(f,d,i,l)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,bm,cn,eajk,fdil->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 
    
    #	  1.0000 P(j,k)<n,m||e,f>*t1(c,m)*t1(d,n)*t2(e,a,k,l)*t2(f,b,i,j)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,cm,dn,eakl,fbij->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdikjl', contracted_intermediate) 
    
    #	  1.0000 P(k,l)<n,m||e,f>*t1(c,m)*t1(d,n)*t2(e,a,i,l)*t2(f,b,j,k)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,cm,dn,eail,fbjk->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdijlk', contracted_intermediate) 
    
    #	  1.0000 P(i,k)<n,m||e,f>*t1(c,m)*t1(d,n)*t2(e,a,j,k)*t2(f,b,i,l)
    contracted_intermediate =  0.999999999999990 * einsum('nmef,cm,dn,eajk,fbil->abcdijkl', g[o, o, v, v], t1, t1, t2, t2, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    quadruples_res +=  1.00000 * contracted_intermediate + -1.00000 * einsum('abcdijkl->abcdkjil', contracted_intermediate) 

    return quadruples_res

def kernel(t1, t2, t3, t4, fock, g, o, v, e_ai, e_abij, e_abcijk, e_abcdijkl, hf_energy, max_iter=100,
           stopping_eps=1.0E-8):
    """

    :param t1: spin-orbital t1 amplitudes (nvirt x nocc)
    :param t2: spin-orbital t2 amplitudes (nvirt x nvirt x nocc x nocc)
    :param t3: spin-orbital t3 amplitudes (nvirt x nvirt x nvirt x nocc x nocc x nocc)
    :param fock: fock operator defined as soei + np.einsum('piiq->pq', astei[:, o, o, :])
              where soei is 1 electron integrals (spinorb) and astei is
              antisymmetric 2 electron integrals in openfermion format
              <12|21>.  <ij|kl> - <ij|lk>
    :param g: antisymmetric 2 electron integrals. See fock input.
    :param o: slice(None, occ) where occ is number of occupied spin-orbitals
    :param v: slice(occ, None) whwere occ is number of occupied spin-orbitals
    :param e_ai: 1 / (-eps[v, n] + eps[n, o])
    :param e_abij: 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[n, n, n, o])
    :param hf_energy: the hartree-fock energy
    :param max_iter: Total number of CC iterations allowed
    :param stopping_eps: stopping criteria for residual l2-norm
    """
    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)
    fock_e_abcijk = np.reciprocal(e_abcijk)
    fock_e_abcdijkl = np.reciprocal(e_abcdijkl)
    old_energy = cc_energy(t1, t2, fock, g, o, v)

    print("    ==> CCSDTQ amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(max_iter):

        residual_singles    = singles_residual(t1, t2, t3, fock, g, o, v)
        residual_doubles    = doubles_residual(t1, t2, t3, t4, fock, g, o, v)
        residual_triples    = triples_residual(t1, t2, t3, t4, fock, g, o, v)
        residual_quadruples = quadruples_residual(t1, t2, t3, t4, fock, g, o, v)

        res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles) + np.linalg.norm(residual_triples) + np.linalg.norm(residual_quadruples)
        singles_res    = residual_singles + fock_e_ai * t1
        doubles_res    = residual_doubles + fock_e_abij * t2
        triples_res    = residual_triples + fock_e_abcijk * t3
        quadruples_res = residual_quadruples + fock_e_abcdijkl * t4

        new_singles    = singles_res * e_ai
        new_doubles    = doubles_res * e_abij
        new_triples    = triples_res * e_abcijk
        new_quadruples = quadruples_res * e_abcdijkl

        current_energy = cc_energy(new_singles, new_doubles, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < stopping_eps and res_norm < stopping_eps:
            # assign t1 and t2 variables for future use before breaking
            t1 = new_singles
            t2 = new_doubles
            t3 = new_triples
            t4 = new_quadruples
            break
        else:
            # assign t1 and t2 and old_energy for next iteration
            t1 = new_singles
            t2 = new_doubles
            t3 = new_triples
            t4 = new_quadruples
            old_energy = current_energy
            print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(idx, old_energy - hf_energy, delta_e, res_norm))
    else:
        raise ValueError("CCSDTQ iterations did not converge")

    return t1, t2, t3


def main():
    """
    Example for solving CCSDTQ amplitude equations
    """
    import pyscf
    import openfermion as of
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from openfermionpyscf import run_pyscf
    from pyscf.cc.addons import spatial2spin
    from pyscf import cc
    import numpy as np

    np.set_printoptions(linewidth=500)

    # run pyscf for some reason
    basis = '6-31g'
    mol = pyscf.M(
        atom='H 0 0 0; F 0 0 {}'.format(1.6),
        basis=basis)

    mf = mol.RHF()
    mf.verbose = 0
    mf.run()

    # build molecule and run pyscf again for some reason
    molecule = of.MolecularData(geometry=[['H', (0, 0, 0)], ['F', (0, 0, 1.6)]],
                                basis=basis, charge=0, multiplicity=1)
    molecule = run_pyscf(molecule,run_ccsd=False)

    # 1-, 2-electron integrals
    oei, tei = molecule.get_integrals()

    # Number of orbitals, number of electrons. apparently only works for closed shells
    occ = mf.mo_occ
    nele = int(sum(occ))
    nocc = nele // 2
    norbs = oei.shape[0]
    nsvirt = 2 * (norbs - nocc)
    nsocc = 2 * nocc

    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    gtei = astei.transpose(0, 1, 3, 2)

    eps = np.kron(molecule.orbital_energies, np.ones(2))
    n = np.newaxis
    o = slice(None, nsocc)
    v = slice(nsocc, None)

    e_abcdijkl = 1 / (- eps[v, n, n, n, n, n, n, n] - eps[n, v, n, n, n, n, n, n] - eps[n, n, v, n, n, n, n, n] - eps[n, n, n, v, n, n, n, n]
                      + eps[n, n, n, n, o, n, n, n] + eps[n, n, n, n, n, o, n, n] + eps[n, n, n, n, n, n, o, n] + eps[n, n, n, n, n, n, n, o] )

    e_abcijk = 1 / (- eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                    + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o] )

    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
    hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])
    hf_energy_test = 1.0 * einsum('ii', fock[o, o]) -0.5 * einsum('ijij', gtei[o, o, o, o])

    print("")
    print("    SCF Total Energy:          {: 20.12f}".format(hf_energy + molecule.nuclear_repulsion))
    print("")
    assert np.isclose(hf_energy, mf.e_tot - molecule.nuclear_repulsion)
    assert np.isclose(hf_energy_test, hf_energy)

    g = gtei

    t1z = np.zeros((nsvirt, nsocc))
    t2z = np.zeros((nsvirt, nsvirt, nsocc, nsocc))
    t3z = np.zeros((nsvirt, nsvirt, nsvirt, nsocc, nsocc, nsocc))
    t4z = np.zeros((nsvirt, nsvirt, nsvirt, nsvirt, nsocc, nsocc, nsocc, nsocc))

    t1f, t2f, t3f = kernel(t1z, t2z, t3z, t4z, fock, g, o, v, e_ai, e_abij,e_abcijk,e_abcdijkl, hf_energy,
                        stopping_eps=1e-10)


    en = cc_energy(t1f, t2f, fock, g, o, v) 
    print("")
    print("    CCSDTQ Correlation Energy: {: 20.12f}".format(en - hf_energy))
    print("    CCSDTQ Total Energy:       {: 20.12f}".format(en + molecule.nuclear_repulsion))
    print("")

    assert np.isclose(en-hf_energy,-0.179815934953076,atol=1e-9)
    assert np.isclose(en+molecule.nuclear_repulsion,-100.009723511692869,atol=1e-9)


if __name__ == "__main__":
    main()

