"""
A full working spin-orbital CCSDT code generated with pdaggerq

If you want to run the example here you should install pyscf openfermion
and openfermion-pyscf The actual CCSDT code (cc_energy, singles_residual,
doubles_residual, triples_residual, kernel) do not depend on those
packages but you must obtain integrals from somehwere.

The total energy for this example been checked against that produced by
the CCSDT implementation in NWChem. Note that there must be a discrepancy
in the definition of the angstrom/bohr conversion ... agreement with
NWChem past 7 decimals can only be achieved if the geometry is defined
in bohr in that program.

 CCSDT correlation energy / hartree =        -0.179049024111075
 CCSDT total energy / hartree       =      -100.008956600850908

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


def residuals(t1, t2, t3, f, eri):

    tmps_ = {}
    perm_tmps = {}

    # rt1  = +1.00 f(a,i)
    rt1  = 1.00 * f["vo"]

    # rt1 += -1.00 f(j,i) t1(a,j)
    # flops: o1v1 += o2v1
    #  mems: o1v1 += o1v1
    rt1 -= np.einsum('aj,ji->ai',t1,f["oo"])

    # rt1 += +1.00 f(a,b) t1(b,i)
    # flops: o1v1 += o1v2
    #  mems: o1v1 += o1v1
    rt1 += np.einsum('ab,bi->ai',f["vv"],t1)

    # rt1 += +1.00 <j,a||b,i> t1(b,j)
    # flops: o1v1 += o2v2
    #  mems: o1v1 += o1v1
    rt1 -= np.einsum('ajbi,bj->ai',eri["vovo"],t1)

    # rt1 += -1.00 f(j,b) t2(b,a,i,j)
    # flops: o1v1 += o2v2
    #  mems: o1v1 += o1v1
    rt1 -= np.einsum('jb,baij->ai',f["ov"],t2)

    # rt1 += -0.50 <k,j||b,i> t2(b,a,k,j)
    # flops: o1v1 += o3v2
    #  mems: o1v1 += o1v1
    rt1 += 0.50 * np.einsum('bakj,jkbi->ai',t2,eri["oovo"])

    # rt1 += -0.50 <j,a||b,c> t2(b,c,i,j)
    # flops: o1v1 += o2v3
    #  mems: o1v1 += o1v1
    rt1 += 0.50 * np.einsum('ajbc,bcij->ai',eri["vovv"],t2)

    # rt1 += +0.25 <k,j||b,c> t3(b,c,a,i,k,j)
    # flops: o1v1 += o3v3
    #  mems: o1v1 += o1v1
    rt1 -= 0.25 * np.einsum('jkbc,bcaikj->ai',eri["oovv"],t3)

    # rt1 += +0.50 <k,j||b,c> t1(a,j) t2(b,c,i,k)
    # flops: o1v1 += o3v2 o2v1
    #  mems: o1v1 += o2v0 o1v1
    rt1 -= 0.50 * einsum('ia->ai', np.einsum('jkbc,bcik,aj->ia',eri["oovv"],t2,t1,optimize='optimal') )

    # rt2  = +1.00 <a,b||i,j>
    rt2  = 1.00 * eri["vvoo"]

    # rt2 += +1.00 P(a,b) <k,a||i,j> t1(b,k)
    # flops: o2v2 += o3v2
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 1.00 * einsum('aijb->abij', np.einsum('akij,bk->aijb',eri["vooo"],t1) )
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('baij->abij', perm_tmps["vvoo"] )

    # rt2 += -1.00 P(i,j) f(k,j) t2(a,b,i,k)
    # flops: o2v2 += o3v2
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 1.00 * np.einsum('abik,kj->abij',t2,f["oo"])
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('abji->abij', perm_tmps["vvoo"] )

    # rt2 += +1.00 P(i,j) <a,b||c,j> t1(c,i)
    # flops: o2v2 += o2v3
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 1.00 * einsum('abji->abij', np.einsum('abcj,ci->abji',eri["vvvo"],t1) )
    rt2 += perm_tmps["vvoo"]
    rt2 -= einsum('abji->abij', perm_tmps["vvoo"] )

    # rt2 += +1.00 P(a,b) f(a,c) t2(c,b,i,j)
    # flops: o2v2 += o2v3
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 1.00 * np.einsum('ac,cbij->abij',f["vv"],t2)
    rt2 += perm_tmps["vvoo"]
    rt2 -= einsum('baij->abij', perm_tmps["vvoo"] )

    # rt2 += +0.50 <l,k||i,j> t2(a,b,l,k)
    # flops: o2v2 += o4v2
    #  mems: o2v2 += o2v2
    rt2 -= 0.50 * np.einsum('ablk,klij->abij',t2,eri["oooo"])

    # rt2 += +1.00 P(i,j) P(a,b) <k,a||c,j> t2(c,b,i,k)
    # flops: o2v2 += o3v3
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 1.00 * einsum('ajbi->abij', np.einsum('akcj,cbik->ajbi',eri["vovo"],t2) )
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('abji->abij', perm_tmps["vvoo"] )
    rt2 += einsum('baij->abij', perm_tmps["vvoo"] )
    rt2 -= einsum('baji->abij', perm_tmps["vvoo"] )

    # rt2 += +1.00 f(k,c) t3(c,a,b,i,j,k)
    # flops: o2v2 += o3v3
    #  mems: o2v2 += o2v2
    rt2 += np.einsum('kc,cabijk->abij',f["ov"],t3)

    # rt2 += +0.50 <a,b||c,d> t2(c,d,i,j)
    # flops: o2v2 += o2v4
    #  mems: o2v2 += o2v2
    rt2 += 0.50 * np.einsum('abcd,cdij->abij',eri["vvvv"],t2)

    # rt2 += +0.50 P(i,j) <l,k||c,j> t3(c,a,b,i,l,k)
    # flops: o2v2 += o4v3
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 0.50 * np.einsum('cabilk,klcj->abij',t3,eri["oovo"])
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('abji->abij', perm_tmps["vvoo"] )

    # rt2 += +0.50 P(a,b) <k,a||c,d> t3(c,d,b,i,j,k)
    # flops: o2v2 += o3v4
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 0.50 * np.einsum('akcd,cdbijk->abij',eri["vovv"],t3)
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('baij->abij', perm_tmps["vvoo"] )

    # rt2 += -1.00 <l,k||i,j> t1(a,k) t1(b,l)
    # flops: o2v2 += o4v1 o3v2
    #  mems: o2v2 += o3v1 o2v2
    rt2 += einsum('aijb->abij', np.einsum('ak,klij,bl->aijb',t1,eri["oooo"],t1,optimize='optimal') )

    # rt2 += +1.00 P(i,j) P(a,b) <k,a||c,j> t1(b,k) t1(c,i)
    # flops: o2v2 += o3v2 o3v2
    #  mems: o2v2 += o3v1 o2v2
    perm_tmps["vvoo"]  = 1.00 * einsum('ajib->abij', np.einsum('akcj,ci,bk->ajib',eri["vovo"],t1,t1,optimize='optimal') )
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('abji->abij', perm_tmps["vvoo"] )
    rt2 += einsum('baij->abij', perm_tmps["vvoo"] )
    rt2 -= einsum('baji->abij', perm_tmps["vvoo"] )

    # rt2 += -1.00 P(a,b) f(k,c) t1(a,k) t2(c,b,i,j)
    # flops: o2v2 += o3v2 o3v2
    #  mems: o2v2 += o3v1 o2v2
    perm_tmps["vvoo"]  = 1.00 * einsum('bija->abij', np.einsum('kc,cbij,ak->bija',f["ov"],t2,t1,optimize='optimal') )
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('baij->abij', perm_tmps["vvoo"] )

    # rt2 += -0.50 P(i,j) <l,k||c,d> t2(a,b,i,l) t2(c,d,j,k)
    # flops: o2v2 += o3v2 o3v2
    #  mems: o2v2 += o2v0 o2v2
    perm_tmps["vvoo"]  = 0.50 * einsum('jabi->abij', np.einsum('klcd,cdjk,abil->jabi',eri["oovv"],t2,t2,optimize='optimal') )
    rt2 += perm_tmps["vvoo"]
    rt2 -= einsum('abji->abij', perm_tmps["vvoo"] )

    # rt2 += -0.50 <l,k||c,d> t2(c,a,l,k) t2(d,b,i,j)
    # flops: o2v2 += o2v3 o2v3
    #  mems: o2v2 += o0v2 o2v2
    rt2 += 0.50 * np.einsum('klcd,calk,dbij->abij',eri["oovv"],t2,t2,optimize='optimal')

    # rt2 += -0.50 <l,k||c,d> t2(c,a,i,j) t2(d,b,l,k)
    # flops: o2v2 += o2v3 o2v3
    #  mems: o2v2 += o0v2 o2v2
    rt2 += 0.50 * einsum('baij->abij', np.einsum('klcd,dblk,caij->baij',eri["oovv"],t2,t2,optimize='optimal') )

    # rt2 += -1.00 <a,b||c,d> t1(c,j) t1(d,i)
    # flops: o2v2 += o1v4 o2v3
    #  mems: o2v2 += o1v3 o2v2
    rt2 -= np.einsum('abcd,di,cj->abij',eri["vvvv"],t1,t1,optimize='optimal')

    # rt2 += -1.00 P(i,j) P(a,b) <l,k||c,j> t1(a,k) t2(c,b,i,l)
    # flops: o2v2 += o4v2 o3v2
    #  mems: o2v2 += o3v1 o2v2
    perm_tmps["vvoo"]  = 1.00 * einsum('bija->abij', np.einsum('cbil,klcj,ak->bija',t2,eri["oovo"],t1,optimize='optimal') )
    rt2 += perm_tmps["vvoo"]
    rt2 -= einsum('abji->abij', perm_tmps["vvoo"] )
    rt2 -= einsum('baij->abij', perm_tmps["vvoo"] )
    rt2 += einsum('baji->abij', perm_tmps["vvoo"] )

    # rt2 += +0.50 P(a,b) <k,a||c,d> t1(b,k) t2(c,d,i,j)
    # flops: o2v2 += o3v3 o3v2
    #  mems: o2v2 += o3v1 o2v2
    perm_tmps["vvoo"]  = 0.50 * einsum('aijb->abij', np.einsum('akcd,cdij,bk->aijb',eri["vovv"],t2,t1,optimize='optimal') )
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('baij->abij', perm_tmps["vvoo"] )

    # rt2 += -0.50 P(a,b) <l,k||c,d> t1(a,k) t3(c,d,b,i,j,l)
    # flops: o2v2 += o4v3 o3v2
    #  mems: o2v2 += o3v1 o2v2
    perm_tmps["vvoo"]  = 0.50 * einsum('bija->abij', np.einsum('klcd,cdbijl,ak->bija',eri["oovv"],t3,t1,optimize='optimal') )
    rt2 += perm_tmps["vvoo"]
    rt2 -= einsum('baij->abij', perm_tmps["vvoo"] )

    # flops: o3v3  = o4v4 o4v3
    #  mems: o3v3  = o4v2 o3v3
    tmps_["1_vvooov"]  = 1.00 * np.einsum('alde,decijk,bl->acijkb',eri["vovv"],t3,t1,optimize='optimal')

    # rt3  = -0.50 P(a,c) <l,b||d,e> t1(a,l) t3(d,e,c,i,j,k)
    rt3  = 0.50 * einsum('bcijka->abcijk', tmps_["1_vvooov"] )
    rt3 -= 0.50 * einsum('baijkc->abcijk', tmps_["1_vvooov"] )

    # rt3 += +0.50 P(b,c) <l,a||d,e> t1(b,l) t3(d,e,c,i,j,k)
    rt3 -= 0.50 * einsum('acijkb->abcijk', tmps_["1_vvooov"] )
    rt3 += 0.50 * einsum('abijkc->abcijk', tmps_["1_vvooov"] )

    # rt3 += +0.50 P(a,b) <l,c||d,e> t1(a,l) t3(d,e,b,i,j,k)
    rt3 -= 0.50 * einsum('cbijka->abcijk', tmps_["1_vvooov"] )
    rt3 += 0.50 * einsum('caijkb->abcijk', tmps_["1_vvooov"] )
    del tmps_["1_vvooov"]

    # flops: o5v1  = o5v3
    #  mems: o5v1  = o5v1
    tmps_["2_vooooo"]  = 1.00 * einsum('lmaijk->aijklm', np.einsum('lmde,deaijk->lmaijk',eri["oovv"],t3) )

    # flops: o3v3  = o3v5 o5v3 o3v3 o5v2 o4v3 o3v3
    #  mems: o3v3  = o3v3 o3v3 o3v3 o4v2 o3v3 o3v3
    tmps_["25_vvvooo"]  = 1.00 * np.einsum('abde,decijk->abcijk',eri["vvvv"],t3)
    tmps_["25_vvvooo"] -= 0.50 * np.einsum('abml,cijklm->abcijk',t2,tmps_["2_vooooo"])
    tmps_["25_vvvooo"] += einsum('acijkb->abcijk', np.einsum('al,cijklm,bm->acijkb',t1,tmps_["2_vooooo"],t1,optimize='optimal') )
    del tmps_["2_vooooo"]

    # rt3 += +0.50 P(b,c) <a,b||d,e> t3(d,e,c,i,j,k)
    #     += +0.25 P(b,c) <m,l||d,e> t2(a,b,m,l) t3(d,e,c,i,j,k)
    #     += -0.50 P(b,c) <m,l||d,e> t1(a,l) t1(b,m) t3(d,e,c,i,j,k)
    rt3 += 0.50 * tmps_["25_vvvooo"]
    rt3 -= 0.50 * einsum('acbijk->abcijk', tmps_["25_vvvooo"] )

    # rt3 += +0.50 <b,c||d,e> t3(d,e,a,i,j,k)
    #     += +0.25 <m,l||d,e> t3(d,e,a,i,j,k) t2(b,c,m,l)
    #     += -0.50 <m,l||d,e> t3(d,e,a,i,j,k) t1(b,l) t1(c,m)
    rt3 += 0.50 * einsum('bcaijk->abcijk', tmps_["25_vvvooo"] )
    del tmps_["25_vvvooo"]

    # flops: o3v3  = o4v1 o5v3
    #  mems: o3v3  = o4v0 o3v3
    tmps_["3_oovvvo"]  = 1.00 * einsum('jkabci->kjabci', np.einsum('dj,lmdk,abciml->jkabci',t1,eri["oovo"],t3,optimize='optimal') )

    # rt3 += +0.50 P(i,j) <m,l||d,k> t3(a,b,c,i,m,l) t1(d,j)
    rt3 -= 0.50 * einsum('kjabci->abcijk', tmps_["3_oovvvo"] )
    rt3 += 0.50 * einsum('kiabcj->abcijk', tmps_["3_oovvvo"] )

    # rt3 += +0.50 P(j,k) <m,l||d,i> t3(a,b,c,j,m,l) t1(d,k)
    rt3 -= 0.50 * einsum('ikabcj->abcijk', tmps_["3_oovvvo"] )
    rt3 += 0.50 * einsum('ijabck->abcijk', tmps_["3_oovvvo"] )

    # rt3 += -0.50 P(i,k) <m,l||d,j> t3(a,b,c,i,m,l) t1(d,k)
    rt3 += 0.50 * einsum('jkabci->abcijk', tmps_["3_oovvvo"] )
    rt3 -= 0.50 * einsum('jiabck->abcijk', tmps_["3_oovvvo"] )
    del tmps_["3_oovvvo"]

    # flops: o2v2  = o2v3
    #  mems: o2v2  = o2v2
    tmps_["8_vovo"]  = 1.00 * np.einsum('akcd,cj->akdj',eri["vovv"],t1)

    # flops: o3v3  = o2v3 o4v3 o4v3
    #  mems: o3v3  = o2v2 o4v2 o3v3
    tmps_["4_vovoov"]  = 1.00 * np.einsum('blde,dk,ecij,al->bkcija',eri["vovv"],t1,t2,t1,optimize='optimal')

    # rt3 += -1.00 P(a,c) <l,b||d,e> t1(a,l) t2(e,c,j,k) t1(d,i)
    rt3 += einsum('bicjka->abcijk', tmps_["4_vovoov"] )
    rt3 -= einsum('biajkc->abcijk', tmps_["4_vovoov"] )

    # rt3 += +1.00 P(j,k) P(b,c) <l,a||d,e> t1(b,l) t2(e,c,i,j) t1(d,k)
    rt3 -= einsum('akcijb->abcijk', tmps_["4_vovoov"] )
    rt3 += einsum('ajcikb->abcijk', tmps_["4_vovoov"] )
    rt3 += einsum('akbijc->abcijk', tmps_["4_vovoov"] )
    rt3 -= einsum('ajbikc->abcijk', tmps_["4_vovoov"] )

    # rt3 += +1.00 P(j,k) P(a,b) <l,c||d,e> t1(a,l) t2(e,b,i,j) t1(d,k)
    rt3 -= einsum('ckbija->abcijk', tmps_["4_vovoov"] )
    rt3 += einsum('cjbika->abcijk', tmps_["4_vovoov"] )
    rt3 += einsum('ckaijb->abcijk', tmps_["4_vovoov"] )
    rt3 -= einsum('cjaikb->abcijk', tmps_["4_vovoov"] )

    # rt3 += +1.00 P(a,b) <l,c||d,e> t1(a,l) t2(e,b,j,k) t1(d,i)
    rt3 -= einsum('cibjka->abcijk', tmps_["4_vovoov"] )
    rt3 += einsum('ciajkb->abcijk', tmps_["4_vovoov"] )

    # rt3 += -1.00 P(j,k) P(a,c) <l,b||d,e> t1(a,l) t2(e,c,i,j) t1(d,k)
    rt3 += einsum('bkcija->abcijk', tmps_["4_vovoov"] )
    rt3 -= einsum('bjcika->abcijk', tmps_["4_vovoov"] )
    rt3 -= einsum('bkaijc->abcijk', tmps_["4_vovoov"] )
    rt3 += einsum('bjaikc->abcijk', tmps_["4_vovoov"] )

    # rt3 += +1.00 P(b,c) <l,a||d,e> t1(b,l) t2(e,c,j,k) t1(d,i)
    rt3 -= einsum('aicjkb->abcijk', tmps_["4_vovoov"] )
    rt3 += einsum('aibjkc->abcijk', tmps_["4_vovoov"] )
    del tmps_["4_vovoov"]

    # flops: o3v3  = o4v3 o4v3
    #  mems: o3v3  = o4v2 o3v3
    tmps_["5_vovoov"]  = 1.00 * np.einsum('bldk,dcij,al->bkcija',eri["vovo"],t2,t1,optimize='optimal')

    # rt3 += +1.00 P(a,c) <l,b||d,i> t1(a,l) t2(d,c,j,k)
    rt3 -= einsum('bicjka->abcijk', tmps_["5_vovoov"] )
    rt3 += einsum('biajkc->abcijk', tmps_["5_vovoov"] )

    # rt3 += -1.00 P(j,k) P(b,c) <l,a||d,k> t1(b,l) t2(d,c,i,j)
    rt3 += einsum('akcijb->abcijk', tmps_["5_vovoov"] )
    rt3 -= einsum('ajcikb->abcijk', tmps_["5_vovoov"] )
    rt3 -= einsum('akbijc->abcijk', tmps_["5_vovoov"] )
    rt3 += einsum('ajbikc->abcijk', tmps_["5_vovoov"] )

    # rt3 += -1.00 P(j,k) P(a,b) <l,c||d,k> t1(a,l) t2(d,b,i,j)
    rt3 += einsum('ckbija->abcijk', tmps_["5_vovoov"] )
    rt3 -= einsum('cjbika->abcijk', tmps_["5_vovoov"] )
    rt3 -= einsum('ckaijb->abcijk', tmps_["5_vovoov"] )
    rt3 += einsum('cjaikb->abcijk', tmps_["5_vovoov"] )

    # rt3 += -1.00 P(a,b) <l,c||d,i> t1(a,l) t2(d,b,j,k)
    rt3 += einsum('cibjka->abcijk', tmps_["5_vovoov"] )
    rt3 -= einsum('ciajkb->abcijk', tmps_["5_vovoov"] )

    # rt3 += +1.00 P(j,k) P(a,c) <l,b||d,k> t1(a,l) t2(d,c,i,j)
    rt3 -= einsum('bkcija->abcijk', tmps_["5_vovoov"] )
    rt3 += einsum('bjcika->abcijk', tmps_["5_vovoov"] )
    rt3 += einsum('bkaijc->abcijk', tmps_["5_vovoov"] )
    rt3 -= einsum('bjaikc->abcijk', tmps_["5_vovoov"] )

    # rt3 += -1.00 P(b,c) <l,a||d,i> t1(b,l) t2(d,c,j,k)
    rt3 += einsum('aicjkb->abcijk', tmps_["5_vovoov"] )
    rt3 -= einsum('aibjkc->abcijk', tmps_["5_vovoov"] )
    del tmps_["5_vovoov"]

    # flops: o2v2  = o3v3
    #  mems: o2v2  = o2v2
    tmps_["6_voov"]  = 1.00 * einsum('ldbi->bild', np.einsum('lmde,ebim->ldbi',eri["oovv"],t2) )

    # rt1 += +1.00 <k,j||b,c> t2(c,a,i,k) t1(b,j)
    # flops: o1v1 += o2v2
    #  mems: o1v1 += o1v1
    rt1 -= np.einsum('bj,aijb->ai',t1,tmps_["6_voov"])

    # rt2 += +1.00 P(i,j) <l,k||c,d> t2(c,a,j,k) t2(d,b,i,l)
    # flops: o2v2 += o3v3
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 1.00 * einsum('ajbi->abij', np.einsum('cajk,bikc->ajbi',t2,tmps_["6_voov"]) )
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('abji->abij', perm_tmps["vvoo"] )

    # rt2 += +1.00 P(i,j) P(a,b) <l,k||c,d> t1(a,k) t2(d,b,i,l) t1(c,j)
    # flops: o2v2 += o3v2 o3v2
    #  mems: o2v2 += o3v1 o2v2
    perm_tmps["vvoo"]  = 1.00 * einsum('jbia->abij', np.einsum('cj,bikc,ak->jbia',t1,tmps_["6_voov"],t1,optimize='optimal') )
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('abji->abij', perm_tmps["vvoo"] )
    rt2 += einsum('baij->abij', perm_tmps["vvoo"] )
    rt2 -= einsum('baji->abij', perm_tmps["vvoo"] )
    del tmps_["6_voov"]

    # flops: o4v0  = o4v2
    #  mems: o4v0  = o4v0
    tmps_["7_oooo"]  = 1.00 * einsum('klij->ijkl', np.einsum('klcd,cdij->klij',eri["oovv"],t2) )

    # rt2 += +0.25 <l,k||c,d> t2(a,b,l,k) t2(c,d,i,j)
    # flops: o2v2 += o4v2
    #  mems: o2v2 += o2v2
    rt2 -= 0.25 * np.einsum('ablk,ijkl->abij',t2,tmps_["7_oooo"])

    # rt2 += -0.50 <l,k||c,d> t1(a,k) t1(b,l) t2(c,d,i,j)
    # flops: o2v2 += o4v1 o3v2
    #  mems: o2v2 += o3v1 o2v2
    rt2 += 0.50 * einsum('aijb->abij', np.einsum('ak,ijkl,bl->aijb',t1,tmps_["7_oooo"],t1,optimize='optimal') )
    del tmps_["7_oooo"]

    # rt2 += -1.00 P(i,j) P(a,b) <k,a||c,d> t2(d,b,i,k) t1(c,j)
    # flops: o2v2 += o3v3
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 1.00 * einsum('biaj->abij', np.einsum('dbik,akdj->biaj',t2,tmps_["8_vovo"]) )
    rt2 += perm_tmps["vvoo"]
    rt2 -= einsum('abji->abij', perm_tmps["vvoo"] )
    rt2 -= einsum('baij->abij', perm_tmps["vvoo"] )
    rt2 += einsum('baji->abij', perm_tmps["vvoo"] )

    # rt2 += -1.00 P(a,b) <k,a||c,d> t1(b,k) t1(c,j) t1(d,i)
    # flops: o2v2 += o3v2 o3v2
    #  mems: o2v2 += o3v1 o2v2
    perm_tmps["vvoo"]  = 1.00 * einsum('iajb->abij', np.einsum('di,akdj,bk->iajb',t1,tmps_["8_vovo"],t1,optimize='optimal') )
    rt2 += perm_tmps["vvoo"]
    rt2 -= einsum('baij->abij', perm_tmps["vvoo"] )

    # flops: o1v1  = o2v2
    #  mems: o1v1  = o1v1
    tmps_["12_ov"]  = 1.00 * np.einsum('jkbc,bj->kc',eri["oovv"],t1)

    # flops: o3v1  = o3v2
    #  mems: o3v1  = o3v1
    tmps_["9_ooov"]  = 1.00 * einsum('jkci->ijkc', np.einsum('jkbc,bi->jkci',eri["oovv"],t1) )

    # flops: o3v3  = o4v1 o4v1 o4v3 o3v2 o4v3 o3v3 o2v4 o3v4 o3v3 o3v4 o3v4 o3v3 o3v3 o4v3 o4v3 o3v3 o4v3 o4v1 o4v3 o3v3 o3v3 o4v3 o3v3 o3v2 o4v3 o3v3 o3v2 o4v3 o3v3 o4v2 o4v1 o4v3 o3v3 o3v3
    #  mems: o3v3  = o4v0 o3v1 o3v3 o3v1 o3v3 o3v3 o1v3 o3v3 o3v3 o1v3 o3v3 o3v3 o2v2 o4v2 o3v3 o3v3 o3v3 o3v1 o3v3 o3v3 o3v1 o3v3 o3v3 o3v1 o3v3 o3v3 o3v1 o3v3 o3v3 o4v0 o3v1 o3v3 o3v3 o3v3
    tmps_["21_oovvvo"]  = 1.00 * einsum('jkcabi->kjcabi', np.einsum('ej,klme,cl,abim->jkcabi',t1,tmps_["9_ooov"],t1,t2,optimize='optimal') )
    tmps_["21_oovvvo"] -= einsum('jckabi->kjcabi', np.einsum('ej,clek,abil->jckabi',t1,tmps_["8_vovo"],t2,optimize='optimal') )
    tmps_["21_oovvvo"] -= einsum('cbiajk->kjcabi', np.einsum('clde,ebil,dajk->cbiajk',eri["vovv"],t2,t2,optimize='optimal') )
    tmps_["21_oovvvo"] += 0.50 * einsum('abicjk->kjcabi', np.einsum('lmde,eabiml,dcjk->abicjk',eri["oovv"],t3,t2,optimize='optimal') )
    tmps_["21_oovvvo"] += einsum('biajkc->kjcabi', np.einsum('lmde,ebim,dajk,cl->biajkc',eri["oovv"],t2,t2,t1,optimize='optimal') )
    tmps_["21_oovvvo"] += einsum('cjkabi->kjcabi', np.einsum('cljk,abil->cjkabi',eri["vooo"],t2) )
    tmps_["21_oovvvo"] -= einsum('jkcabi->kjcabi', np.einsum('lmjk,cl,abim->jkcabi',eri["oooo"],t1,t2,optimize='optimal') )
    tmps_["21_oovvvo"] += 0.50 * einsum('cjkabi->kjcabi', np.einsum('clde,dejk,abil->cjkabi',eri["vovv"],t2,t2,optimize='optimal') )
    tmps_["21_oovvvo"] -= einsum('cjkabi->kjcabi', np.einsum('ecjk,me,abim->cjkabi',t2,tmps_["12_ov"],t2,optimize='optimal') )
    tmps_["21_oovvvo"] -= einsum('cjkabi->kjcabi', np.einsum('ld,dcjk,abil->cjkabi',f["ov"],t2,t2,optimize='optimal') )
    tmps_["21_oovvvo"] -= 0.50 * einsum('jkcabi->kjcabi', np.einsum('lmde,dejk,cl,abim->jkcabi',eri["oovv"],t2,t1,t2,optimize='optimal') )

    # rt3 += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(b,c,i,m) t1(d,k) t1(e,j)
    #     += +1.00 P(i,j) P(a,b) <l,a||d,e> t2(b,c,i,l) t1(d,k) t1(e,j)
    #     += +1.00 P(i,j) P(a,b) <l,a||d,e> t2(d,b,j,k) t2(e,c,i,l)
    #     += -0.50 P(i,j) P(a,b) <m,l||d,e> t2(d,a,j,k) t3(e,b,c,i,m,l)
    #     += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,j,k) t2(e,c,i,m)
    #     += -1.00 P(i,j) P(a,b) <l,a||j,k> t2(b,c,i,l)
    #     += +1.00 P(i,j) P(a,b) <m,l||j,k> t1(a,l) t2(b,c,i,m)
    #     += -0.50 P(i,j) P(a,b) <l,a||d,e> t2(b,c,i,l) t2(d,e,j,k)
    #     += +1.00 P(i,j) P(a,b) <m,l||d,e> t2(e,a,j,k) t2(b,c,i,m) t1(d,l)
    #     += -1.00 P(i,j) P(a,b) f(l,d) t2(d,a,j,k) t2(b,c,i,l)
    #     += +0.50 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(b,c,i,m) t2(d,e,j,k)
    rt3 += einsum('kjabci->abcijk', tmps_["21_oovvvo"] )
    rt3 -= einsum('kiabcj->abcijk', tmps_["21_oovvvo"] )
    rt3 -= einsum('kjbaci->abcijk', tmps_["21_oovvvo"] )
    rt3 += einsum('kibacj->abcijk', tmps_["21_oovvvo"] )

    # rt3 += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(b,c,k,m) t1(d,j) t1(e,i)
    #     += +1.00 P(a,b) <l,a||d,e> t2(b,c,k,l) t1(d,j) t1(e,i)
    #     += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,j) t2(e,c,k,l)
    #     += -0.50 P(a,b) <m,l||d,e> t2(d,a,i,j) t3(e,b,c,k,m,l)
    #     += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,j) t2(e,c,k,m)
    #     += -1.00 P(a,b) <l,a||i,j> t2(b,c,k,l)
    #     += +1.00 P(a,b) <m,l||i,j> t1(a,l) t2(b,c,k,m)
    #     += -0.50 P(a,b) <l,a||d,e> t2(b,c,k,l) t2(d,e,i,j)
    #     += +1.00 P(a,b) <m,l||d,e> t2(e,a,i,j) t2(b,c,k,m) t1(d,l)
    #     += -1.00 P(a,b) f(l,d) t2(d,a,i,j) t2(b,c,k,l)
    #     += +0.50 P(a,b) <m,l||d,e> t1(a,l) t2(b,c,k,m) t2(d,e,i,j)
    rt3 += einsum('jiabck->abcijk', tmps_["21_oovvvo"] )
    rt3 -= einsum('jiback->abcijk', tmps_["21_oovvvo"] )

    # rt3 += -1.00 P(i,j) <m,l||d,e> t2(a,b,i,m) t1(c,l) t1(d,k) t1(e,j)
    #     += +1.00 P(i,j) <l,c||d,e> t2(a,b,i,l) t1(d,k) t1(e,j)
    #     += +1.00 P(i,j) <l,c||d,e> t2(d,a,j,k) t2(e,b,i,l)
    #     += -0.50 P(i,j) <m,l||d,e> t3(e,a,b,i,m,l) t2(d,c,j,k)
    #     += -1.00 P(i,j) <m,l||d,e> t2(d,a,j,k) t2(e,b,i,m) t1(c,l)
    #     += -1.00 P(i,j) <l,c||j,k> t2(a,b,i,l)
    #     += +1.00 P(i,j) <m,l||j,k> t2(a,b,i,m) t1(c,l)
    #     += -0.50 P(i,j) <l,c||d,e> t2(a,b,i,l) t2(d,e,j,k)
    #     += +1.00 P(i,j) <m,l||d,e> t2(a,b,i,m) t2(e,c,j,k) t1(d,l)
    #     += -1.00 P(i,j) f(l,d) t2(a,b,i,l) t2(d,c,j,k)
    #     += +0.50 P(i,j) <m,l||d,e> t2(a,b,i,m) t1(c,l) t2(d,e,j,k)
    rt3 += einsum('kjcabi->abcijk', tmps_["21_oovvvo"] )
    rt3 -= einsum('kicabj->abcijk', tmps_["21_oovvvo"] )

    # rt3 += -1.00 <m,l||d,e> t2(a,b,k,m) t1(c,l) t1(d,j) t1(e,i)
    #     += +1.00 <l,c||d,e> t2(a,b,k,l) t1(d,j) t1(e,i)
    #     += +1.00 <l,c||d,e> t2(d,a,i,j) t2(e,b,k,l)
    #     += -0.50 <m,l||d,e> t3(e,a,b,k,m,l) t2(d,c,i,j)
    #     += -1.00 <m,l||d,e> t2(d,a,i,j) t2(e,b,k,m) t1(c,l)
    #     += -1.00 <l,c||i,j> t2(a,b,k,l)
    #     += +1.00 <m,l||i,j> t2(a,b,k,m) t1(c,l)
    #     += -0.50 <l,c||d,e> t2(a,b,k,l) t2(d,e,i,j)
    #     += +1.00 <m,l||d,e> t2(a,b,k,m) t2(e,c,i,j) t1(d,l)
    #     += -1.00 f(l,d) t2(a,b,k,l) t2(d,c,i,j)
    #     += +0.50 <m,l||d,e> t2(a,b,k,m) t1(c,l) t2(d,e,i,j)
    rt3 += einsum('jicabk->abcijk', tmps_["21_oovvvo"] )
    del tmps_["21_oovvvo"]

    # flops: o3v3  = o3v3 o3v4 o3v3 o4v3 o4v3 o2v4 o3v4 o3v3 o3v2 o5v3 o4v3 o3v3 o5v3 o4v3 o3v3 o3v3 o4v4 o3v3 o4v4 o3v3 o4v4 o3v3 o3v3 o3v2 o3v3 o3v4 o3v3
    #  mems: o3v3  = o1v3 o3v3 o2v2 o4v2 o3v3 o1v3 o3v3 o3v3 o3v1 o4v2 o3v3 o3v3 o4v2 o3v3 o3v3 o2v2 o3v3 o3v3 o3v3 o3v3 o3v3 o3v3 o3v3 o3v1 o1v3 o3v3 o3v3
    tmps_["22_ovvvoo"]  = 1.00 * einsum('abkcij->kabcij', np.einsum('abml,lmdk,dcij->abkcij',t2,eri["oovo"],t2,optimize='optimal') )
    tmps_["22_ovvvoo"] += 2.00 * einsum('akbijc->kabcij', np.einsum('lmde,dakm,ebij,cl->akbijc',eri["oovv"],t2,t2,t1,optimize='optimal') )
    tmps_["22_ovvvoo"] -= 2.00 * einsum('cakbij->kabcij', np.einsum('clde,dakl,ebij->cakbij',eri["vovv"],t2,t2,optimize='optimal') )
    tmps_["22_ovvvoo"] -= 2.00 * einsum('kabijc->kabcij', np.einsum('lmde,dk,eabijm,cl->kabijc',eri["oovv"],t1,t3,t1,optimize='optimal') )
    tmps_["22_ovvvoo"] += 2.00 * einsum('kabijc->kabcij', np.einsum('lmdk,dabijm,cl->kabijc',eri["oovo"],t3,t1,optimize='optimal') )
    tmps_["22_ovvvoo"] -= 2.00 * einsum('ckabij->kabcij', np.einsum('lmde,dckl,eabijm->ckabij',eri["oovv"],t2,t3,optimize='optimal') )
    tmps_["22_ovvvoo"] += 2.00 * einsum('ckabij->kabcij', np.einsum('clek,eabijl->ckabij',tmps_["8_vovo"],t3) )
    tmps_["22_ovvvoo"] -= 2.00 * einsum('ckabij->kabcij', np.einsum('cldk,dabijl->ckabij',eri["vovo"],t3) )
    tmps_["22_ovvvoo"] -= np.einsum('lmde,dk,abml,ecij->kabcij',eri["oovv"],t1,t2,t2,optimize='optimal')
    del tmps_["8_vovo"]

    # rt3 += -0.50 P(a,b) <m,l||d,i> t2(d,a,j,k) t2(b,c,m,l)
    #     += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,m) t2(e,c,j,k)
    #     += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,l) t2(e,c,j,k)
    #     += +1.00 P(a,b) <m,l||d,e> t1(a,l) t3(e,b,c,j,k,m) t1(d,i)
    #     += -1.00 P(a,b) <m,l||d,i> t1(a,l) t3(d,b,c,j,k,m)
    #     += +1.00 P(a,b) <m,l||d,e> t2(d,a,i,l) t3(e,b,c,j,k,m)
    #     += -1.00 P(a,b) <l,a||d,e> t3(e,b,c,j,k,l) t1(d,i)
    #     += +1.00 P(a,b) <l,a||d,i> t3(d,b,c,j,k,l)
    #     += +0.50 P(a,b) <m,l||d,e> t2(e,a,j,k) t2(b,c,m,l) t1(d,i)
    rt3 += 0.50 * einsum('ibcajk->abcijk', tmps_["22_ovvvoo"] )
    rt3 -= 0.50 * einsum('iacbjk->abcijk', tmps_["22_ovvvoo"] )

    # rt3 += -0.50 P(j,k) <m,l||d,k> t2(a,b,m,l) t2(d,c,i,j)
    #     += -1.00 P(j,k) <m,l||d,e> t2(d,a,k,m) t2(e,b,i,j) t1(c,l)
    #     += +1.00 P(j,k) <l,c||d,e> t2(d,a,k,l) t2(e,b,i,j)
    #     += +1.00 P(j,k) <m,l||d,e> t3(e,a,b,i,j,m) t1(c,l) t1(d,k)
    #     += -1.00 P(j,k) <m,l||d,k> t3(d,a,b,i,j,m) t1(c,l)
    #     += +1.00 P(j,k) <m,l||d,e> t3(e,a,b,i,j,m) t2(d,c,k,l)
    #     += -1.00 P(j,k) <l,c||d,e> t3(e,a,b,i,j,l) t1(d,k)
    #     += +1.00 P(j,k) <l,c||d,k> t3(d,a,b,i,j,l)
    #     += +0.50 P(j,k) <m,l||d,e> t2(a,b,m,l) t2(e,c,i,j) t1(d,k)
    rt3 += 0.50 * einsum('kabcij->abcijk', tmps_["22_ovvvoo"] )
    rt3 -= 0.50 * einsum('jabcik->abcijk', tmps_["22_ovvvoo"] )

    # rt3 += -0.50 <m,l||d,i> t2(a,b,m,l) t2(d,c,j,k)
    #     += -1.00 <m,l||d,e> t2(d,a,i,m) t2(e,b,j,k) t1(c,l)
    #     += +1.00 <l,c||d,e> t2(d,a,i,l) t2(e,b,j,k)
    #     += +1.00 <m,l||d,e> t3(e,a,b,j,k,m) t1(c,l) t1(d,i)
    #     += -1.00 <m,l||d,i> t3(d,a,b,j,k,m) t1(c,l)
    #     += +1.00 <m,l||d,e> t3(e,a,b,j,k,m) t2(d,c,i,l)
    #     += -1.00 <l,c||d,e> t3(e,a,b,j,k,l) t1(d,i)
    #     += +1.00 <l,c||d,i> t3(d,a,b,j,k,l)
    #     += +0.50 <m,l||d,e> t2(a,b,m,l) t2(e,c,j,k) t1(d,i)
    rt3 += 0.50 * einsum('iabcjk->abcijk', tmps_["22_ovvvoo"] )

    # rt3 += -0.50 P(j,k) P(a,b) <m,l||d,k> t2(d,a,i,j) t2(b,c,m,l)
    #     += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,k,m) t2(e,c,i,j)
    #     += +1.00 P(j,k) P(a,b) <l,a||d,e> t2(d,b,k,l) t2(e,c,i,j)
    #     += +1.00 P(j,k) P(a,b) <m,l||d,e> t1(a,l) t3(e,b,c,i,j,m) t1(d,k)
    #     += -1.00 P(j,k) P(a,b) <m,l||d,k> t1(a,l) t3(d,b,c,i,j,m)
    #     += +1.00 P(j,k) P(a,b) <m,l||d,e> t2(d,a,k,l) t3(e,b,c,i,j,m)
    #     += -1.00 P(j,k) P(a,b) <l,a||d,e> t3(e,b,c,i,j,l) t1(d,k)
    #     += +1.00 P(j,k) P(a,b) <l,a||d,k> t3(d,b,c,i,j,l)
    #     += +0.50 P(j,k) P(a,b) <m,l||d,e> t2(e,a,i,j) t2(b,c,m,l) t1(d,k)
    rt3 += 0.50 * einsum('kbcaij->abcijk', tmps_["22_ovvvoo"] )
    rt3 -= 0.50 * einsum('jbcaik->abcijk', tmps_["22_ovvvoo"] )
    rt3 -= 0.50 * einsum('kacbij->abcijk', tmps_["22_ovvvoo"] )
    rt3 += 0.50 * einsum('jacbik->abcijk', tmps_["22_ovvvoo"] )
    del tmps_["22_ovvvoo"]

    # rt1 += +0.50 <k,j||b,c> t2(c,a,k,j) t1(b,i)
    # flops: o1v1 += o3v2
    #  mems: o1v1 += o1v1
    rt1 -= 0.50 * np.einsum('cakj,ijkc->ai',t2,tmps_["9_ooov"])

    # rt2 += -0.50 P(i,j) <l,k||c,d> t3(d,a,b,i,l,k) t1(c,j)
    # flops: o2v2 += o4v3
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 0.50 * np.einsum('dabilk,jkld->abij',t3,tmps_["9_ooov"])
    rt2 += perm_tmps["vvoo"]
    rt2 -= einsum('abji->abij', perm_tmps["vvoo"] )

    # flops: o3v3  = o4v2 o4v0 o4v1 o4v0 o5v3
    #  mems: o3v3  = o4v0 o4v0 o4v0 o4v0 o3v3
    tmps_["17_oovvvo"]  = 1.00 * einsum('abcijk->jkabci', np.einsum('abciml,lmjk->abcijk',t3,(eri["oooo"] + 0.50 * np.einsum('lmde,dejk->lmjk',eri["oovv"],t2) + np.einsum('jklm->lmjk',-1.00 * np.einsum('ej,klme->jklm',t1,tmps_["9_ooov"])))) )

    # rt3 += +0.50 <m,l||i,j> t3(a,b,c,k,m,l)
    #     += +0.25 <m,l||d,e> t3(a,b,c,k,m,l) t2(d,e,i,j)
    #     += -0.50 <m,l||d,e> t3(a,b,c,k,m,l) t1(d,j) t1(e,i)
    rt3 -= 0.50 * einsum('ijabck->abcijk', tmps_["17_oovvvo"] )

    # rt3 += +0.50 P(i,j) <m,l||j,k> t3(a,b,c,i,m,l)
    #     += +0.25 P(i,j) <m,l||d,e> t3(a,b,c,i,m,l) t2(d,e,j,k)
    #     += -0.50 P(i,j) <m,l||d,e> t3(a,b,c,i,m,l) t1(d,k) t1(e,j)
    rt3 -= 0.50 * einsum('jkabci->abcijk', tmps_["17_oovvvo"] )
    rt3 += 0.50 * einsum('ikabcj->abcijk', tmps_["17_oovvvo"] )
    del tmps_["17_oovvvo"]

    # flops: o4v0  = o4v1
    #  mems: o4v0  = o4v0
    tmps_["23_oooo"]  = 1.00 * np.einsum('di,jkld->ijkl',t1,tmps_["9_ooov"])
    del tmps_["9_ooov"]

    # rt2 += -0.50 <l,k||c,d> t2(a,b,l,k) t1(c,j) t1(d,i)
    # flops: o2v2 += o4v2
    #  mems: o2v2 += o2v2
    rt2 += 0.50 * np.einsum('ablk,ijkl->abij',t2,tmps_["23_oooo"])

    # rt2 += +1.00 <l,k||c,d> t1(a,k) t1(b,l) t1(c,j) t1(d,i)
    # flops: o2v2 += o4v1 o3v2
    #  mems: o2v2 += o3v1 o2v2
    rt2 -= einsum('aijb->abij', np.einsum('ak,ijkl,bl->aijb',t1,tmps_["23_oooo"],t1,optimize='optimal') )
    del tmps_["23_oooo"]

    # flops: o4v0  = o4v1
    #  mems: o4v0  = o4v0
    tmps_["10_oooo"]  = 1.00 * einsum('lmjk->klmj', np.einsum('lmdj,dk->lmjk',eri["oovo"],t1) )

    # rt2 += +0.50 P(i,j) <l,k||c,j> t2(a,b,l,k) t1(c,i)
    # flops: o2v2 += o4v2
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 0.50 * np.einsum('ablk,iklj->abij',t2,tmps_["10_oooo"])
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('abji->abij', perm_tmps["vvoo"] )

    # rt2 += -1.00 P(i,j) <l,k||c,j> t1(a,k) t1(b,l) t1(c,i)
    # flops: o2v2 += o4v1 o3v2
    #  mems: o2v2 += o3v1 o2v2
    perm_tmps["vvoo"]  = 1.00 * einsum('aijb->abij', np.einsum('ak,iklj,bl->aijb',t1,tmps_["10_oooo"],t1,optimize='optimal') )
    rt2 += perm_tmps["vvoo"]
    rt2 -= einsum('abji->abij', perm_tmps["vvoo"] )
    del tmps_["10_oooo"]

    # flops: o0v2  = o1v3
    #  mems: o0v2  = o0v2
    tmps_["11_vv"]  = 1.00 * np.einsum('ajbc,bj->ac',eri["vovv"],t1)

    # rt1 += +1.00 <j,a||b,c> t1(b,j) t1(c,i)
    # flops: o1v1 += o1v2
    #  mems: o1v1 += o1v1
    rt1 -= einsum('ia->ai', np.einsum('ci,ac->ia',t1,tmps_["11_vv"]) )

    # rt2 += +1.00 P(a,b) <k,a||c,d> t2(d,b,i,j) t1(c,k)
    # flops: o2v2 += o2v3
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 1.00 * einsum('bija->abij', np.einsum('dbij,ad->bija',t2,tmps_["11_vv"]) )
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('baij->abij', perm_tmps["vvoo"] )

    # rt2 += -1.00 <l,k||c,d> t3(d,a,b,i,j,l) t1(c,k)
    # flops: o2v2 += o3v3
    #  mems: o2v2 += o2v2
    rt2 += np.einsum('dabijl,ld->abij',t3,tmps_["12_ov"])

    # rt2 += +1.00 P(a,b) <l,k||c,d> t1(a,l) t2(d,b,i,j) t1(c,k)
    # flops: o2v2 += o3v2 o3v2
    #  mems: o2v2 += o3v1 o2v2
    perm_tmps["vvoo"]  = 1.00 * einsum('bija->abij', np.einsum('dbij,ld,al->bija',t2,tmps_["12_ov"],t1,optimize='optimal') )
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('baij->abij', perm_tmps["vvoo"] )

    # flops: o3v3  = o4v3 o4v3 o4v3 o4v3 o3v3 o3v4 o3v3 o2v3 o3v4 o3v3 o3v4 o3v3
    #  mems: o3v3  = o4v2 o3v3 o4v2 o3v3 o3v3 o3v3 o3v3 o0v2 o3v3 o3v3 o3v3 o3v3
    tmps_["18_vvooov"]  = 1.00 * np.einsum('ld,dbcijk,al->bcijka',f["ov"],t3,t1,optimize='optimal')
    tmps_["18_vvooov"] += np.einsum('ebcijk,me,am->bcijka',t3,tmps_["12_ov"],t1,optimize='optimal')
    tmps_["18_vvooov"] -= einsum('abcijk->bcijka', np.einsum('ad,dbcijk->abcijk',f["vv"],t3) )
    tmps_["18_vvooov"] -= 0.50 * einsum('abcijk->bcijka', np.einsum('lmde,daml,ebcijk->abcijk',eri["oovv"],t2,t3,optimize='optimal') )
    tmps_["18_vvooov"] += np.einsum('ebcijk,ae->bcijka',t3,tmps_["11_vv"])
    del tmps_["11_vv"]

    # rt3 += -1.00 f(l,d) t3(d,a,b,i,j,k) t1(c,l)
    #     += +1.00 <m,l||d,e> t3(e,a,b,i,j,k) t1(c,m) t1(d,l)
    #     += +1.00 f(c,d) t3(d,a,b,i,j,k)
    #     += -0.50 <m,l||d,e> t3(e,a,b,i,j,k) t2(d,c,m,l)
    #     += +1.00 <l,c||d,e> t3(e,a,b,i,j,k) t1(d,l)
    rt3 -= einsum('abijkc->abcijk', tmps_["18_vvooov"] )

    # rt3 += -1.00 P(a,b) f(l,d) t1(a,l) t3(d,b,c,i,j,k)
    #     += +1.00 P(a,b) <m,l||d,e> t1(a,m) t3(e,b,c,i,j,k) t1(d,l)
    #     += +1.00 P(a,b) f(a,d) t3(d,b,c,i,j,k)
    #     += -0.50 P(a,b) <m,l||d,e> t2(d,a,m,l) t3(e,b,c,i,j,k)
    #     += +1.00 P(a,b) <l,a||d,e> t3(e,b,c,i,j,k) t1(d,l)
    rt3 -= einsum('bcijka->abcijk', tmps_["18_vvooov"] )
    rt3 += einsum('acijkb->abcijk', tmps_["18_vvooov"] )
    del tmps_["18_vvooov"]

    # flops: o2v0  = o2v1
    #  mems: o2v0  = o2v0
    tmps_["24_oo"]  = 1.00 * np.einsum('ci,kc->ik',t1,tmps_["12_ov"])

    # rt1 += +1.00 <k,j||b,c> t1(a,k) t1(b,j) t1(c,i)
    # flops: o1v1 += o2v1
    #  mems: o1v1 += o1v1
    rt1 -= np.einsum('ak,ik->ai',t1,tmps_["24_oo"])

    # rt2 += +1.00 P(i,j) <l,k||c,d> t2(a,b,i,l) t1(c,k) t1(d,j)
    # flops: o2v2 += o3v2
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 1.00 * np.einsum('abil,jl->abij',t2,tmps_["24_oo"])
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('abji->abij', perm_tmps["vvoo"] )
    del tmps_["24_oo"]

    # flops: o2v0  = o3v1
    #  mems: o2v0  = o2v0
    tmps_["13_oo"]  = 1.00 * np.einsum('jkbi,bj->ki',eri["oovo"],t1)

    # rt1 += +1.00 <k,j||b,i> t1(a,k) t1(b,j)
    # flops: o1v1 += o2v1
    #  mems: o1v1 += o1v1
    rt1 -= np.einsum('ak,ki->ai',t1,tmps_["13_oo"])

    # rt2 += +1.00 P(i,j) <l,k||c,j> t2(a,b,i,l) t1(c,k)
    # flops: o2v2 += o3v2
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 1.00 * np.einsum('abil,lj->abij',t2,tmps_["13_oo"])
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('abji->abij', perm_tmps["vvoo"] )

    # flops: o3v3  = o4v3 o2v1 o4v3 o3v3 o3v2 o4v3 o3v3 o2v1 o4v3 o3v3 o4v3 o3v3
    #  mems: o3v3  = o3v3 o2v0 o3v3 o3v3 o2v0 o3v3 o3v3 o2v0 o3v3 o3v3 o3v3 o3v3
    tmps_["19_ovvvoo"]  = 1.00 * einsum('abcijk->kabcij', np.einsum('abcijl,lk->abcijk',t3,f["oo"]) )
    tmps_["19_ovvvoo"] += np.einsum('ek,me,abcijm->kabcij',t1,tmps_["12_ov"],t3,optimize='optimal')
    tmps_["19_ovvvoo"] -= 0.50 * np.einsum('lmde,dekl,abcijm->kabcij',eri["oovv"],t2,t3,optimize='optimal')
    tmps_["19_ovvvoo"] += np.einsum('ld,dk,abcijl->kabcij',f["ov"],t1,t3,optimize='optimal')
    tmps_["19_ovvvoo"] += einsum('abcijk->kabcij', np.einsum('abcijm,mk->abcijk',t3,tmps_["13_oo"]) )
    del tmps_["13_oo"]
    del tmps_["12_ov"]

    # rt3 += -1.00 P(j,k) f(l,k) t3(a,b,c,i,j,l)
    #     += +1.00 P(j,k) <m,l||d,e> t3(a,b,c,i,j,m) t1(d,l) t1(e,k)
    #     += -0.50 P(j,k) <m,l||d,e> t3(a,b,c,i,j,m) t2(d,e,k,l)
    #     += -1.00 P(j,k) f(l,d) t3(a,b,c,i,j,l) t1(d,k)
    #     += +1.00 P(j,k) <m,l||d,k> t3(a,b,c,i,j,m) t1(d,l)
    rt3 -= einsum('kabcij->abcijk', tmps_["19_ovvvoo"] )
    rt3 += einsum('jabcik->abcijk', tmps_["19_ovvvoo"] )

    # rt3 += -1.00 f(l,i) t3(a,b,c,j,k,l)
    #     += +1.00 <m,l||d,e> t3(a,b,c,j,k,m) t1(d,l) t1(e,i)
    #     += -0.50 <m,l||d,e> t3(a,b,c,j,k,m) t2(d,e,i,l)
    #     += -1.00 f(l,d) t3(a,b,c,j,k,l) t1(d,i)
    #     += +1.00 <m,l||d,i> t3(a,b,c,j,k,m) t1(d,l)
    rt3 -= einsum('iabcjk->abcijk', tmps_["19_ovvvoo"] )
    del tmps_["19_ovvvoo"]

    # flops: o2v0  = o2v1
    #  mems: o2v0  = o2v0
    tmps_["14_oo"]  = 1.00 * einsum('ji->ij', np.einsum('jb,bi->ji',f["ov"],t1) )

    # rt1 += -1.00 f(j,b) t1(a,j) t1(b,i)
    # flops: o1v1 += o2v1
    #  mems: o1v1 += o1v1
    rt1 -= np.einsum('aj,ij->ai',t1,tmps_["14_oo"])

    # rt2 += -1.00 P(i,j) f(k,c) t2(a,b,i,k) t1(c,j)
    # flops: o2v2 += o3v2
    #  mems: o2v2 += o2v2
    perm_tmps["vvoo"]  = 1.00 * np.einsum('abik,jk->abij',t2,tmps_["14_oo"])
    rt2 -= perm_tmps["vvoo"]
    rt2 += einsum('abji->abij', perm_tmps["vvoo"] )
    del tmps_["14_oo"]

    # flops: o3v3  = o3v2 o4v3 o3v2 o4v2 o4v3 o3v3
    #  mems: o3v3  = o3v1 o3v3 o3v1 o3v1 o3v3 o3v3
    tmps_["15_voovvo"]  = 1.00 * np.einsum('cldi,dk,abjl->cikabj',eri["vovo"],t1,t2,optimize='optimal')
    tmps_["15_voovvo"] += einsum('ickabj->cikabj', np.einsum('lmde,di,eckl,abjm->ickabj',eri["oovv"],t1,t2,t2,optimize='optimal') )

    # rt3 += -1.00 P(i,j) P(a,b) <l,a||d,k> t2(b,c,i,l) t1(d,j)
    #     += -1.00 P(i,j) P(a,b) <m,l||d,e> t2(e,a,j,l) t2(b,c,i,m) t1(d,k)
    rt3 += einsum('akjbci->abcijk', tmps_["15_voovvo"] )
    rt3 -= einsum('akibcj->abcijk', tmps_["15_voovvo"] )
    rt3 -= einsum('bkjaci->abcijk', tmps_["15_voovvo"] )
    rt3 += einsum('bkiacj->abcijk', tmps_["15_voovvo"] )

    # rt3 += +1.00 P(i,k) P(a,b) <l,a||d,j> t2(b,c,i,l) t1(d,k)
    #     += +1.00 P(i,k) P(a,b) <m,l||d,e> t2(e,a,k,l) t2(b,c,i,m) t1(d,j)
    rt3 -= einsum('ajkbci->abcijk', tmps_["15_voovvo"] )
    rt3 += einsum('ajibck->abcijk', tmps_["15_voovvo"] )
    rt3 += einsum('bjkaci->abcijk', tmps_["15_voovvo"] )
    rt3 -= einsum('bjiack->abcijk', tmps_["15_voovvo"] )

    # rt3 += -1.00 P(j,k) P(a,b) <l,a||d,i> t2(b,c,j,l) t1(d,k)
    #     += -1.00 P(j,k) P(a,b) <m,l||d,e> t2(e,a,k,l) t2(b,c,j,m) t1(d,i)
    rt3 += einsum('aikbcj->abcijk', tmps_["15_voovvo"] )
    rt3 -= einsum('aijbck->abcijk', tmps_["15_voovvo"] )
    rt3 -= einsum('bikacj->abcijk', tmps_["15_voovvo"] )
    rt3 += einsum('bijack->abcijk', tmps_["15_voovvo"] )

    # rt3 += -1.00 P(j,k) <l,c||d,i> t2(a,b,j,l) t1(d,k)
    #     += -1.00 P(j,k) <m,l||d,e> t2(a,b,j,m) t2(e,c,k,l) t1(d,i)
    rt3 += einsum('cikabj->abcijk', tmps_["15_voovvo"] )
    rt3 -= einsum('cijabk->abcijk', tmps_["15_voovvo"] )

    # rt3 += -1.00 P(i,j) <l,c||d,k> t2(a,b,i,l) t1(d,j)
    #     += -1.00 P(i,j) <m,l||d,e> t2(a,b,i,m) t2(e,c,j,l) t1(d,k)
    rt3 += einsum('ckjabi->abcijk', tmps_["15_voovvo"] )
    rt3 -= einsum('ckiabj->abcijk', tmps_["15_voovvo"] )

    # rt3 += +1.00 P(i,k) <l,c||d,j> t2(a,b,i,l) t1(d,k)
    #     += +1.00 P(i,k) <m,l||d,e> t2(a,b,i,m) t2(e,c,k,l) t1(d,j)
    rt3 -= einsum('cjkabi->abcijk', tmps_["15_voovvo"] )
    rt3 += einsum('cjiabk->abcijk', tmps_["15_voovvo"] )
    del tmps_["15_voovvo"]

    # flops: o3v3  = o4v1 o4v1 o4v3 o4v2 o4v3 o3v3
    #  mems: o3v3  = o4v0 o3v1 o3v3 o3v1 o3v3 o3v3
    tmps_["16_oovvvo"]  = 1.00 * np.einsum('lmdj,dk,cl,abim->jkcabi',eri["oovo"],t1,t1,t2,optimize='optimal')
    tmps_["16_oovvvo"] += einsum('jckabi->jkcabi', np.einsum('lmdj,dckl,abim->jckabi',eri["oovo"],t2,t2,optimize='optimal') )

    # rt3 += +1.00 P(j,k) <m,l||d,i> t2(a,b,j,m) t1(c,l) t1(d,k)
    #     += +1.00 P(j,k) <m,l||d,i> t2(a,b,j,m) t2(d,c,k,l)
    rt3 -= einsum('ikcabj->abcijk', tmps_["16_oovvvo"] )
    rt3 += einsum('ijcabk->abcijk', tmps_["16_oovvvo"] )

    # rt3 += +1.00 P(i,j) P(a,b) <m,l||d,k> t1(a,l) t2(b,c,i,m) t1(d,j)
    #     += +1.00 P(i,j) P(a,b) <m,l||d,k> t2(d,a,j,l) t2(b,c,i,m)
    rt3 -= einsum('kjabci->abcijk', tmps_["16_oovvvo"] )
    rt3 += einsum('kiabcj->abcijk', tmps_["16_oovvvo"] )
    rt3 += einsum('kjbaci->abcijk', tmps_["16_oovvvo"] )
    rt3 -= einsum('kibacj->abcijk', tmps_["16_oovvvo"] )

    # rt3 += -1.00 P(i,k) <m,l||d,j> t2(a,b,i,m) t1(c,l) t1(d,k)
    #     += -1.00 P(i,k) <m,l||d,j> t2(a,b,i,m) t2(d,c,k,l)
    rt3 += einsum('jkcabi->abcijk', tmps_["16_oovvvo"] )
    rt3 -= einsum('jicabk->abcijk', tmps_["16_oovvvo"] )

    # rt3 += +1.00 P(j,k) P(a,b) <m,l||d,i> t1(a,l) t2(b,c,j,m) t1(d,k)
    #     += +1.00 P(j,k) P(a,b) <m,l||d,i> t2(d,a,k,l) t2(b,c,j,m)
    rt3 -= einsum('ikabcj->abcijk', tmps_["16_oovvvo"] )
    rt3 += einsum('ijabck->abcijk', tmps_["16_oovvvo"] )
    rt3 += einsum('ikbacj->abcijk', tmps_["16_oovvvo"] )
    rt3 -= einsum('ijback->abcijk', tmps_["16_oovvvo"] )

    # rt3 += -1.00 P(i,k) P(a,b) <m,l||d,j> t1(a,l) t2(b,c,i,m) t1(d,k)
    #     += -1.00 P(i,k) P(a,b) <m,l||d,j> t2(d,a,k,l) t2(b,c,i,m)
    rt3 += einsum('jkabci->abcijk', tmps_["16_oovvvo"] )
    rt3 -= einsum('jiabck->abcijk', tmps_["16_oovvvo"] )
    rt3 -= einsum('jkbaci->abcijk', tmps_["16_oovvvo"] )
    rt3 += einsum('jiback->abcijk', tmps_["16_oovvvo"] )

    # rt3 += +1.00 P(i,j) <m,l||d,k> t2(a,b,i,m) t1(c,l) t1(d,j)
    #     += +1.00 P(i,j) <m,l||d,k> t2(a,b,i,m) t2(d,c,j,l)
    rt3 -= einsum('kjcabi->abcijk', tmps_["16_oovvvo"] )
    rt3 += einsum('kicabj->abcijk', tmps_["16_oovvvo"] )
    del tmps_["16_oovvvo"]

    # flops: o3v3  = o3v2 o5v2 o5v2 o5v1 o5v2 o4v3 o4v3 o4v3 o3v3 o1v4 o3v4 o3v3 o3v4 o3v3
    #  mems: o3v3  = o3v1 o5v1 o5v1 o5v1 o4v2 o3v3 o3v1 o3v3 o3v3 o1v3 o3v3 o3v3 o3v3 o3v3
    tmps_["20_vovoov"]  = 1.00 * np.einsum('bl,klmaij,cm->bkaijc',t1,(-1.00 * np.einsum('dk,lmde,eaij->klmaij',t1,eri["oovv"],t2,optimize='optimal') + np.einsum('aijlmk->klmaij',np.einsum('daij,lmdk->aijlmk',t2,eri["oovo"]))),t1,optimize='optimal')
    tmps_["20_vovoov"] -= 0.50 * einsum('aijbck->bkaijc', np.einsum('lmde,deaijm,bckl->aijbck',eri["oovv"],t3,t2,optimize='optimal') )
    tmps_["20_vovoov"] -= einsum('bckaij->bkaijc', np.einsum('bcde,dk,eaij->bckaij',eri["vvvv"],t1,t2,optimize='optimal') )
    tmps_["20_vovoov"] += einsum('bckaij->bkaijc', np.einsum('bcdk,daij->bckaij',eri["vvvo"],t2) )

    # rt3 += +1.00 P(j,k) <m,l||d,k> t2(d,a,i,j) t1(b,l) t1(c,m)
    #     += -1.00 P(j,k) <m,l||d,e> t2(e,a,i,j) t1(b,l) t1(c,m) t1(d,k)
    #     += -0.50 P(j,k) <m,l||d,e> t3(d,e,a,i,j,m) t2(b,c,k,l)
    #     += +1.00 P(j,k) <b,c||d,e> t2(e,a,i,j) t1(d,k)
    #     += -1.00 P(j,k) <b,c||d,k> t2(d,a,i,j)
    rt3 -= einsum('bkaijc->abcijk', tmps_["20_vovoov"] )
    rt3 += einsum('bjaikc->abcijk', tmps_["20_vovoov"] )

    # rt3 += +1.00 <m,l||d,i> t2(d,a,j,k) t1(b,l) t1(c,m)
    #     += -1.00 <m,l||d,e> t2(e,a,j,k) t1(b,l) t1(c,m) t1(d,i)
    #     += -0.50 <m,l||d,e> t3(d,e,a,j,k,m) t2(b,c,i,l)
    #     += +1.00 <b,c||d,e> t2(e,a,j,k) t1(d,i)
    #     += -1.00 <b,c||d,i> t2(d,a,j,k)
    rt3 -= einsum('biajkc->abcijk', tmps_["20_vovoov"] )

    # rt3 += +1.00 P(j,k) P(b,c) <m,l||d,k> t1(a,l) t1(b,m) t2(d,c,i,j)
    #     += -1.00 P(j,k) P(b,c) <m,l||d,e> t1(a,l) t1(b,m) t2(e,c,i,j) t1(d,k)
    #     += -0.50 P(j,k) P(b,c) <m,l||d,e> t2(a,b,k,l) t3(d,e,c,i,j,m)
    #     += +1.00 P(j,k) P(b,c) <a,b||d,e> t2(e,c,i,j) t1(d,k)
    #     += -1.00 P(j,k) P(b,c) <a,b||d,k> t2(d,c,i,j)
    rt3 -= einsum('akcijb->abcijk', tmps_["20_vovoov"] )
    rt3 += einsum('ajcikb->abcijk', tmps_["20_vovoov"] )
    rt3 += einsum('akbijc->abcijk', tmps_["20_vovoov"] )
    rt3 -= einsum('ajbikc->abcijk', tmps_["20_vovoov"] )

    # rt3 += +1.00 P(b,c) <m,l||d,i> t1(a,l) t1(b,m) t2(d,c,j,k)
    #     += -1.00 P(b,c) <m,l||d,e> t1(a,l) t1(b,m) t2(e,c,j,k) t1(d,i)
    #     += -0.50 P(b,c) <m,l||d,e> t2(a,b,i,l) t3(d,e,c,j,k,m)
    #     += +1.00 P(b,c) <a,b||d,e> t2(e,c,j,k) t1(d,i)
    #     += -1.00 P(b,c) <a,b||d,i> t2(d,c,j,k)
    rt3 -= einsum('aicjkb->abcijk', tmps_["20_vovoov"] )
    rt3 += einsum('aibjkc->abcijk', tmps_["20_vovoov"] )
    del tmps_["20_vovoov"]


    singles_resid = rt1
    doubles_resid = rt2
    triples_resid = rt3

    return singles_resid, doubles_resid, triples_resid


def integral_maps(f, eri, o, v):
    eri_ = {}
    eri_["oooo"] = eri[o,o,o,o]
    eri_["oovo"] = eri[o,o,v,o]
    eri_["oovv"] = eri[o,o,v,v]
    eri_["vooo"] = eri[v,o,o,o]
    eri_["vovo"] = eri[v,o,v,o]
    eri_["vovv"] = eri[v,o,v,v]
    eri_["vvoo"] = eri[v,v,o,o]
    eri_["vvvo"] = eri[v,v,v,o]
    eri_["vvvv"] = eri[v,v,v,v]

    f_ = {}
    f_["oo"] = f[o,o]
    f_["ov"] = f[o,v]
    f_["vo"] = f[v,o]
    f_["vv"] = f[v,v]

    return f_, eri_


def kernel(t1, t2, t3, fock, g, o, v, e_ai, e_abij, e_abcijk, hf_energy, max_iter=100,
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
    old_energy = cc_energy(t1, t2, fock, g, o, v)
    f_map, g_map = integral_maps(fock, g, o, v)

    print("    ==> CCSDT amplitude equations <==")
    print("")
    print("     Iter               Energy                 |dE|                 |dT|")
    for idx in range(max_iter):

        residual_singles, residual_doubles, residual_triples = residuals(t1, t2, t3, f_map, g_map)

        res_norm = np.linalg.norm(residual_singles) + np.linalg.norm(residual_doubles) + np.linalg.norm(residual_triples)
        singles_res = residual_singles + fock_e_ai * t1
        doubles_res = residual_doubles + fock_e_abij * t2
        triples_res = residual_triples + fock_e_abcijk * t3


        new_singles = singles_res * e_ai
        new_doubles = doubles_res * e_abij
        new_triples = triples_res * e_abcijk

        current_energy = cc_energy(new_singles, new_doubles, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        print("    {: 5d} {: 20.12f} {: 20.12f} {: 20.12f}".format(idx, current_energy - hf_energy, delta_e, res_norm))
        if delta_e < stopping_eps and res_norm < stopping_eps:
            # assign t1 and t2 variables for future use before breaking
            t1 = new_singles
            t2 = new_doubles
            t3 = new_triples
            break
        else:
            # assign t1 and t2 and old_energy for next iteration
            t1 = new_singles
            t2 = new_doubles
            t3 = new_triples
            old_energy = current_energy
    else:
        raise ValueError("CCSDT iterations did not converge")

    return t1, t2, t3


def main():
    """
    Example for solving CCSDT amplitude equations
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

    e_abcijk = 1 / (- eps[v, n, n, n, n, n] - eps[n, v, n, n, n, n] - eps[n, n, v, n, n, n]
                    + eps[n, n, n, o, n, n] + eps[n, n, n, n, o, n] + eps[n, n, n, n, n, o] )

    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
    hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])
    hf_energy_test = 1.0 * einsum('ii', fock[o, o]) -0.5 * einsum('ijij', gtei[o, o, o, o])

    print("")
    print("    SCF Total Energy:         {: 20.12f}".format(hf_energy + molecule.nuclear_repulsion))
    print("")
    assert np.isclose(hf_energy, mf.e_tot - molecule.nuclear_repulsion)
    assert np.isclose(hf_energy_test, hf_energy)

    g = gtei

    t1z = np.zeros((nsvirt, nsocc))
    t2z = np.zeros((nsvirt, nsvirt, nsocc, nsocc))
    t3z = np.zeros((nsvirt, nsvirt, nsvirt, nsocc, nsocc, nsocc))

    t1f, t2f, t3f = kernel(t1z, t2z, t3z, fock, g, o, v, e_ai, e_abij,e_abcijk,hf_energy,
                        stopping_eps=1e-10)


    en = cc_energy(t1f, t2f, fock, g, o, v) 
    print("")
    print("    CCSDT Correlation Energy: {: 20.12f}".format(en - hf_energy))
    print("    CCSDT Total Energy:       {: 20.12f}".format(en + molecule.nuclear_repulsion))
    print("")

    assert np.isclose(en-hf_energy,-0.179049024111075,atol=1e-9)
    assert np.isclose(en+molecule.nuclear_repulsion,-100.008956600850908,atol=1e-9)


if __name__ == "__main__":
    main()



