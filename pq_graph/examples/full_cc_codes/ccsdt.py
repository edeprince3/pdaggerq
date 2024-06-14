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

    includes_ = {
        "t1": True,
        "t2": True,
        "t3": True,
    }

    # singles_resid = +1.00 f(a,i)  // flops: o1v1 = o1v1 | mem: o1v1 = o1v1
    singles_resid = 1.00 * einsum('ai->ai', f["vo"])

    # doubles_resid = +1.00 <a,b||i,j>  // flops: o2v2 = o2v2 | mem: o2v2 = o2v2
    doubles_resid = 1.00 * einsum('abij->abij', eri["vvoo"])

    # singles_resid += -1.00 f(j,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid -= einsum('aj,ji->ai', t1, f["oo"])

    # singles_resid += +1.00 f(a,b) t1(b,i)  // flops: o1v1 += o1v2 | mem: o1v1 += o1v1
    singles_resid += einsum('ab,bi->ai', f["vv"], t1)

    # singles_resid += -1.00 f(j,b) t2(b,a,i,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid -= einsum('jb,baij->ai', f["ov"], t2)

    # singles_resid += +1.00 <j,a||b,i> t1(b,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid -= einsum('ajbi,bj->ai', eri["vovo"], t1)

    # singles_resid += -0.50 <k,j||b,i> t2(b,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
    singles_resid -= 0.50 * einsum('bakj,kjbi->ai', t2, eri["oovo"])

    # singles_resid += -0.50 <j,a||b,c> t2(b,c,i,j)  // flops: o1v1 += o2v3 | mem: o1v1 += o1v1
    singles_resid += 0.50 * einsum('ajbc,bcij->ai', eri["vovv"], t2)

    # doubles_resid += +0.50 <l,k||i,j> t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
    doubles_resid += 0.50 * einsum('ablk,lkij->abij', t2, eri["oooo"])

    # doubles_resid += +1.00 f(k,c) t3(c,a,b,i,j,k)  // flops: o2v2 += o3v3 | mem: o2v2 += o2v2
    doubles_resid += einsum('kc,cabijk->abij', f["ov"], t3)

    # singles_resid += +0.25 <k,j||b,c> t3(b,c,a,i,k,j)  // flops: o1v1 += o3v3 | mem: o1v1 += o1v1
    singles_resid += 0.25 * einsum('kjbc,bcaikj->ai', eri["oovv"], t3)

    # doubles_resid += +0.50 <a,b||c,d> t2(c,d,i,j)  // flops: o2v2 += o2v4 | mem: o2v2 += o2v2
    doubles_resid += 0.50 * einsum('abcd,cdij->abij', eri["vvvv"], t2)

    # singles_resid += +0.50 <k,j||b,c> t1(a,j) t2(b,c,i,k)  // flops: o1v1 += o3v2 o2v1 | mem: o1v1 += o2v0 o1v1
    singles_resid += 0.50 * einsum('kjbc,bcik,aj->ai', eri["oovv"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

    # tmps_[1_vvvooo](a,c,b,i,j,k) = 0.50 eri[vvvv](a,c,d,e) * t3(d,e,b,i,j,k) // flops: o3v3 = o3v5 | mem: o3v3 = o3v3
    tmps_["1_vvvooo"] = 0.50 * einsum('acde,debijk->acbijk', eri["vvvv"], t3)
    triples_resid = -1.00 * einsum('acbijk->abcijk', tmps_["1_vvvooo"])

    # triples_resid += +0.50 P(b,c) <a,b||d,e> t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["1_vvvooo"])

    # triples_resid += +0.50 <b,c||d,e> t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcaijk->abcijk', tmps_["1_vvvooo"])
    del tmps_["1_vvvooo"]

    # tmps_[2_vovvoo](a,j,b,c,i,k) = 1.00 eri[vovo](a,l,d,j) * t3(d,b,c,i,k,l) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
    tmps_["2_vovvoo"] = einsum('aldj,dbcikl->ajbcik', eri["vovo"], t3)

    # triples_resid += +1.00 P(a,b) <l,a||d,i> t3(d,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aibcjk->abcijk', tmps_["2_vovvoo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <l,a||d,k> t3(d,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('akbcij->abcijk', tmps_["2_vovvoo"])
    triples_resid -= einsum('bjacik->abcijk', tmps_["2_vovvoo"])

    # triples_resid += +1.00 P(j,k) <l,c||d,k> t3(d,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ckabij->abcijk', tmps_["2_vovvoo"])
    triples_resid += einsum('cjabik->abcijk', tmps_["2_vovvoo"])

    # triples_resid += +1.00 <l,c||d,i> t3(d,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ciabjk->abcijk', tmps_["2_vovvoo"])
    triples_resid += einsum('ajbcik->abcijk', tmps_["2_vovvoo"])
    triples_resid += einsum('biacjk->abcijk', tmps_["2_vovvoo"])
    triples_resid += einsum('bkacij->abcijk', tmps_["2_vovvoo"])
    del tmps_["2_vovvoo"]

    # tmps_[3_vovooo](c,l,b,i,j,k) = 0.50 eri[vovv](c,l,d,e) * t3(d,e,b,i,j,k) // flops: o4v2 = o4v4 | mem: o4v2 = o4v2
    tmps_["3_vovooo"] = 0.50 * einsum('clde,debijk->clbijk', eri["vovv"], t3)

    # tmps_[80_vvooov](a,c,i,j,k,b) = 1.00 eri[vovv](a,l,d,e) * t3(d,e,c,i,j,k) * t1(b,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["80_vvooov"] = einsum('alcijk,bl->acijkb', tmps_["3_vovooo"], t1)
    del tmps_["3_vovooo"]
    triples_resid -= einsum('baijkc->abcijk', tmps_["80_vvooov"])

    # triples_resid += -0.50 P(a,c) <l,b||d,e> t1(a,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijka->abcijk', tmps_["80_vvooov"])

    # triples_resid += +0.50 P(a,b) <l,c||d,e> t1(a,l) t3(d,e,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbijka->abcijk', tmps_["80_vvooov"])
    triples_resid += einsum('abijkc->abcijk', tmps_["80_vvooov"])
    triples_resid += einsum('caijkb->abcijk', tmps_["80_vvooov"])

    # triples_resid += +0.50 P(b,c) <l,a||d,e> t1(b,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acijkb->abcijk', tmps_["80_vvooov"])
    del tmps_["80_vvooov"]

    # tmps_[4_oovvoo](l,k,b,c,i,j) = 1.00 eri[oovo](m,l,d,k) * t3(d,b,c,i,j,m) // flops: o4v2 = o5v3 | mem: o4v2 = o4v2
    tmps_["4_oovvoo"] = einsum('mldk,dbcijm->lkbcij', eri["oovo"], t3)

    # tmps_[77_vovvoo](c,i,a,b,j,k) = 1.00 t1(c,l) * eri[oovo](m,l,d,i) * t3(d,a,b,j,k,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["77_vovvoo"] = einsum('cl,liabjk->ciabjk', t1, tmps_["4_oovvoo"])
    del tmps_["4_oovvoo"]
    triples_resid += einsum('ajbcik->abcijk', tmps_["77_vovvoo"])

    # triples_resid += -1.00 P(j,k) <m,l||d,k> t1(c,l) t3(d,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ckabij->abcijk', tmps_["77_vovvoo"])

    # triples_resid += -1.00 <m,l||d,i> t1(c,l) t3(d,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ciabjk->abcijk', tmps_["77_vovvoo"])

    # triples_resid += -1.00 P(a,b) <m,l||d,i> t1(a,l) t3(d,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aibcjk->abcijk', tmps_["77_vovvoo"])
    triples_resid += einsum('bkacij->abcijk', tmps_["77_vovvoo"])
    triples_resid += einsum('biacjk->abcijk', tmps_["77_vovvoo"])
    triples_resid += einsum('cjabik->abcijk', tmps_["77_vovvoo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,k> t1(a,l) t3(d,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('akbcij->abcijk', tmps_["77_vovvoo"])
    triples_resid -= einsum('bjacik->abcijk', tmps_["77_vovvoo"])
    del tmps_["77_vovvoo"]

    # tmps_[5_oovooo](m,l,c,i,j,k) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,c,i,j,k) // flops: o5v1 = o5v3 | mem: o5v1 = o5v1
    tmps_["5_oovooo"] = 0.50 * einsum('mlde,decijk->mlcijk', eri["oovv"], t3)

    # tmps_[53_vooovv](c,i,j,k,a,b) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,c,i,j,k) * t2(a,b,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["53_vooovv"] = 0.50 * einsum('mlcijk,abml->cijkab', tmps_["5_oovooo"], t2)
    triples_resid -= einsum('bijkac->abcijk', tmps_["53_vooovv"])

    # triples_resid += +0.25 P(b,c) <m,l||d,e> t2(a,b,m,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cijkab->abcijk', tmps_["53_vooovv"])

    # triples_resid += +0.25 <m,l||d,e> t2(b,c,m,l) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aijkbc->abcijk', tmps_["53_vooovv"])
    del tmps_["53_vooovv"]

    # tmps_[82_vvooov](a,c,i,j,k,b) = 1.00 t1(a,l) * eri[oovv](m,l,d,e) * t3(d,e,c,i,j,k) * t1(b,m) // flops: o3v3 = o5v2 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["82_vvooov"] = einsum('al,mlcijk,bm->acijkb', t1, tmps_["5_oovooo"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["5_oovooo"]

    # triples_resid += -0.50 P(b,c) <m,l||d,e> t1(a,l) t1(b,m) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acijkb->abcijk', tmps_["82_vvooov"])

    # triples_resid += -0.50 <m,l||d,e> t1(b,l) t1(c,m) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('baijkc->abcijk', tmps_["82_vvooov"])
    triples_resid += einsum('abijkc->abcijk', tmps_["82_vvooov"])
    del tmps_["82_vvooov"]

    # tmps_[6_oovvvo](i,k,a,b,c,j) = 0.50 eri[oooo](m,l,i,k) * t3(a,b,c,j,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["6_oovvvo"] = 0.50 * einsum('mlik,abcjml->ikabcj', eri["oooo"], t3)

    # triples_resid += +0.50 P(i,j) <m,l||j,k> t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('jkabci->abcijk', tmps_["6_oovvvo"])

    # triples_resid += +0.50 <m,l||i,j> t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ijabck->abcijk', tmps_["6_oovvvo"])
    triples_resid -= einsum('ikabcj->abcijk', tmps_["6_oovvvo"])
    del tmps_["6_oovvvo"]

    # tmps_[7_vvvo](d,a,c,i) = 0.50 eri[oovv](m,l,d,e) * t3(e,a,c,i,m,l) // flops: o1v3 = o3v4 | mem: o1v3 = o1v3
    tmps_["7_vvvo"] = 0.50 * einsum('mlde,eaciml->daci', eri["oovv"], t3)

    # tmps_[58_voovvo](c,j,k,a,b,i) = 1.00 t2(d,c,j,k) * eri[oovv](m,l,d,e) * t3(e,a,b,i,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["58_voovvo"] = einsum('dcjk,dabi->cjkabi', t2, tmps_["7_vvvo"])

    # triples_resid += -0.50 <m,l||d,e> t2(d,c,i,j) t3(e,a,b,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cijabk->abcijk', tmps_["58_voovvo"])
    triples_resid += einsum('bjkaci->abcijk', tmps_["58_voovvo"])

    # triples_resid += -0.50 P(i,j) P(a,b) <m,l||d,e> t2(d,a,j,k) t3(e,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkbci->abcijk', tmps_["58_voovvo"])
    triples_resid += einsum('bijack->abcijk', tmps_["58_voovvo"])
    triples_resid -= einsum('bikacj->abcijk', tmps_["58_voovvo"])

    # triples_resid += -0.50 P(i,j) <m,l||d,e> t2(d,c,j,k) t3(e,a,b,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjkabi->abcijk', tmps_["58_voovvo"])
    triples_resid += einsum('aikbcj->abcijk', tmps_["58_voovvo"])

    # triples_resid += -0.50 P(a,b) <m,l||d,e> t2(d,a,i,j) t3(e,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aijbck->abcijk', tmps_["58_voovvo"])
    triples_resid += einsum('cikabj->abcijk', tmps_["58_voovvo"])
    del tmps_["58_voovvo"]

    # tmps_[93_vvoo](a,b,j,i) = 1.00 eri[oovv](l,k,c,d) * t3(d,a,b,j,l,k) * t1(c,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["93_vvoo"] = einsum('cabj,ci->abji', tmps_["7_vvvo"], t1)
    del tmps_["7_vvvo"]
    doubles_resid += einsum('abji->abij', tmps_["93_vvoo"])

    # doubles_resid += -0.50 P(i,j) <l,k||c,d> t1(c,j) t3(d,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["93_vvoo"])
    del tmps_["93_vvoo"]

    # tmps_[8_vvovoo](a,c,i,b,j,k) = 1.00 eri[vvvo](a,c,d,i) * t2(d,b,j,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["8_vvovoo"] = einsum('acdi,dbjk->acibjk', eri["vvvo"], t2)
    triples_resid += einsum('abjcik->abcijk', tmps_["8_vvovoo"])

    # triples_resid += -1.00 <b,c||d,i> t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bciajk->abcijk', tmps_["8_vvovoo"])

    # triples_resid += -1.00 P(j,k) P(b,c) <a,b||d,k> t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abkcij->abcijk', tmps_["8_vvovoo"])

    # triples_resid += -1.00 P(j,k) <b,c||d,k> t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bckaij->abcijk', tmps_["8_vvovoo"])
    triples_resid += einsum('ackbij->abcijk', tmps_["8_vvovoo"])
    triples_resid += einsum('bcjaik->abcijk', tmps_["8_vvovoo"])

    # triples_resid += -1.00 P(b,c) <a,b||d,i> t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["8_vvovoo"])
    triples_resid -= einsum('acjbik->abcijk', tmps_["8_vvovoo"])
    triples_resid += einsum('acibjk->abcijk', tmps_["8_vvovoo"])
    del tmps_["8_vvovoo"]

    # tmps_[9_vvvooo](a,b,c,i,j,k) = 1.00 f[vv](a,d) * t3(d,b,c,i,j,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["9_vvvooo"] = einsum('ad,dbcijk->abcijk', f["vv"], t3)

    # triples_resid += +1.00 f(c,d) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cabijk->abcijk', tmps_["9_vvvooo"])
    triples_resid -= einsum('bacijk->abcijk', tmps_["9_vvvooo"])

    # triples_resid += +1.00 P(a,b) f(a,d) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["9_vvvooo"])
    del tmps_["9_vvvooo"]

    # tmps_[10_vvoo](a,b,i,j) = 0.50 eri[vovv](a,k,c,d) * t3(c,d,b,i,j,k) // flops: o2v2 = o3v4 | mem: o2v2 = o2v2
    tmps_["10_vvoo"] = 0.50 * einsum('akcd,cdbijk->abij', eri["vovv"], t3)
    doubles_resid += einsum('baij->abij', tmps_["10_vvoo"])

    # doubles_resid += +0.50 P(a,b) <k,a||c,d> t3(c,d,b,i,j,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["10_vvoo"])
    del tmps_["10_vvoo"]

    # tmps_[11_voovoo](c,l,k,a,i,j) = 1.00 eri[vovo](c,l,d,k) * t2(d,a,i,j) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
    tmps_["11_voovoo"] = einsum('cldk,daij->clkaij', eri["vovo"], t2)

    # tmps_[66_vvovoo](b,c,j,a,i,k) = 1.00 t1(b,l) * eri[vovo](c,l,d,j) * t2(d,a,i,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["66_vvovoo"] = einsum('bl,cljaik->bcjaik', t1, tmps_["11_voovoo"])
    del tmps_["11_voovoo"]
    triples_resid -= einsum('cbjaik->abcijk', tmps_["66_vvovoo"])
    triples_resid -= einsum('caibjk->abcijk', tmps_["66_vvovoo"])
    triples_resid -= einsum('bckaij->abcijk', tmps_["66_vvovoo"])
    triples_resid += einsum('bcjaik->abcijk', tmps_["66_vvovoo"])

    # triples_resid += -1.00 P(j,k) P(b,c) <l,a||d,k> t1(b,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bakcij->abcijk', tmps_["66_vvovoo"])
    triples_resid -= einsum('bciajk->abcijk', tmps_["66_vvovoo"])
    triples_resid += einsum('abjcik->abcijk', tmps_["66_vvovoo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,c||d,k> t1(a,l) t2(d,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ackbij->abcijk', tmps_["66_vvovoo"])

    # triples_resid += -1.00 P(b,c) <l,a||d,i> t1(b,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('baicjk->abcijk', tmps_["66_vvovoo"])

    # triples_resid += -1.00 P(a,b) <l,c||d,i> t1(a,l) t2(d,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acibjk->abcijk', tmps_["66_vvovoo"])

    # triples_resid += +1.00 P(j,k) P(a,c) <l,b||d,k> t1(a,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abkcij->abcijk', tmps_["66_vvovoo"])
    triples_resid -= einsum('acjbik->abcijk', tmps_["66_vvovoo"])
    triples_resid += einsum('cbkaij->abcijk', tmps_["66_vvovoo"])
    triples_resid -= einsum('bajcik->abcijk', tmps_["66_vvovoo"])
    triples_resid += einsum('cajbik->abcijk', tmps_["66_vvovoo"])
    triples_resid += einsum('cbiajk->abcijk', tmps_["66_vvovoo"])
    triples_resid -= einsum('cakbij->abcijk', tmps_["66_vvovoo"])

    # triples_resid += +1.00 P(a,c) <l,b||d,i> t1(a,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["66_vvovoo"])
    del tmps_["66_vvovoo"]

    # tmps_[12_ovoo](l,c,i,j) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,c,i,j,m) // flops: o3v1 = o4v3 | mem: o3v1 = o3v1
    tmps_["12_ovoo"] = 0.50 * einsum('mlde,decijm->lcij', eri["oovv"], t3)

    # tmps_[76_vvovoo](a,b,j,c,i,k) = 1.00 t2(a,b,j,l) * eri[oovv](m,l,d,e) * t3(d,e,c,i,k,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["76_vvovoo"] = einsum('abjl,lcik->abjcik', t2, tmps_["12_ovoo"])

    # triples_resid += -0.50 <m,l||d,e> t2(b,c,i,l) t3(d,e,a,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bciajk->abcijk', tmps_["76_vvovoo"])
    triples_resid += einsum('acibjk->abcijk', tmps_["76_vvovoo"])
    triples_resid += einsum('abjcik->abcijk', tmps_["76_vvovoo"])

    # triples_resid += -0.50 P(b,c) <m,l||d,e> t2(a,b,i,l) t3(d,e,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["76_vvovoo"])

    # triples_resid += -0.50 P(j,k) P(b,c) <m,l||d,e> t2(a,b,k,l) t3(d,e,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abkcij->abcijk', tmps_["76_vvovoo"])

    # triples_resid += -0.50 P(j,k) <m,l||d,e> t2(b,c,k,l) t3(d,e,a,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bckaij->abcijk', tmps_["76_vvovoo"])
    triples_resid += einsum('ackbij->abcijk', tmps_["76_vvovoo"])
    triples_resid += einsum('bcjaik->abcijk', tmps_["76_vvovoo"])
    triples_resid -= einsum('acjbik->abcijk', tmps_["76_vvovoo"])
    del tmps_["76_vvovoo"]

    # tmps_[102_voov](a,i,j,b) = 1.00 eri[oovv](l,k,c,d) * t3(c,d,a,i,j,l) * t1(b,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["102_voov"] = einsum('kaij,bk->aijb', tmps_["12_ovoo"], t1)
    del tmps_["12_ovoo"]
    doubles_resid += einsum('aijb->abij', tmps_["102_voov"])

    # doubles_resid += -0.50 P(a,b) <l,k||c,d> t1(a,k) t3(c,d,b,i,j,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('bija->abij', tmps_["102_voov"])
    del tmps_["102_voov"]

    # tmps_[13_voovvo](a,i,k,b,c,j) = 1.00 eri[vooo](a,l,i,k) * t2(b,c,j,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["13_voovvo"] = einsum('alik,bcjl->aikbcj', eri["vooo"], t2)
    triples_resid -= einsum('bjkaci->abcijk', tmps_["13_voovvo"])

    # triples_resid += -1.00 P(i,j) P(a,b) <l,a||j,k> t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ajkbci->abcijk', tmps_["13_voovvo"])
    triples_resid -= einsum('aikbcj->abcijk', tmps_["13_voovvo"])
    triples_resid -= einsum('cikabj->abcijk', tmps_["13_voovvo"])
    triples_resid += einsum('bikacj->abcijk', tmps_["13_voovvo"])
    triples_resid -= einsum('bijack->abcijk', tmps_["13_voovvo"])

    # triples_resid += -1.00 P(i,j) <l,c||j,k> t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cjkabi->abcijk', tmps_["13_voovvo"])

    # triples_resid += -1.00 <l,c||i,j> t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cijabk->abcijk', tmps_["13_voovvo"])

    # triples_resid += -1.00 P(a,b) <l,a||i,j> t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aijbck->abcijk', tmps_["13_voovvo"])
    del tmps_["13_voovvo"]

    # tmps_[14_ovvooo](l,a,c,i,j,k) = 1.00 f[ov](l,d) * t3(d,a,c,i,j,k) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
    tmps_["14_ovvooo"] = einsum('ld,dacijk->lacijk', f["ov"], t3)

    # tmps_[87_vvooov](a,c,i,j,k,b) = 1.00 f[ov](l,d) * t3(d,a,c,i,j,k) * t1(b,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["87_vvooov"] = einsum('lacijk,bl->acijkb', tmps_["14_ovvooo"], t1)
    del tmps_["14_ovvooo"]

    # triples_resid += -1.00 P(a,b) f(l,d) t1(a,l) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijka->abcijk', tmps_["87_vvooov"])
    triples_resid += einsum('acijkb->abcijk', tmps_["87_vvooov"])

    # triples_resid += -1.00 f(l,d) t1(c,l) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijkc->abcijk', tmps_["87_vvooov"])
    del tmps_["87_vvooov"]

    # tmps_[15_ovvvoo](j,a,b,c,i,k) = 1.00 f[oo](l,j) * t3(a,b,c,i,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["15_ovvvoo"] = einsum('lj,abcikl->jabcik', f["oo"], t3)
    triples_resid += einsum('jabcik->abcijk', tmps_["15_ovvvoo"])

    # triples_resid += -1.00 P(j,k) f(l,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('kabcij->abcijk', tmps_["15_ovvvoo"])

    # triples_resid += -1.00 f(l,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('iabcjk->abcijk', tmps_["15_ovvvoo"])
    del tmps_["15_ovvvoo"]

    # tmps_[16_ovvo](j,a,b,i) = 0.50 eri[oovo](l,k,c,j) * t3(c,a,b,i,l,k) // flops: o2v2 = o4v3 | mem: o2v2 = o2v2
    tmps_["16_ovvo"] = 0.50 * einsum('lkcj,cabilk->jabi', eri["oovo"], t3)

    # doubles_resid += +0.50 P(i,j) <l,k||c,j> t3(c,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('jabi->abij', tmps_["16_ovvo"])
    doubles_resid -= einsum('iabj->abij', tmps_["16_ovvo"])
    del tmps_["16_ovvo"]

    # tmps_[17_ooovoo](m,l,k,c,i,j) = 1.00 eri[oovo](m,l,d,k) * t2(d,c,i,j) // flops: o5v1 = o5v2 | mem: o5v1 = o5v1
    tmps_["17_ooovoo"] = einsum('mldk,dcij->mlkcij', eri["oovo"], t2)

    # tmps_[71_vovoov](a,i,c,j,k,b) = 1.00 t1(a,l) * eri[oovo](m,l,d,i) * t2(d,c,j,k) * t1(b,m) // flops: o3v3 = o5v2 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["71_vovoov"] = einsum('al,mlicjk,bm->aicjkb', t1, tmps_["17_ooovoo"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["17_ooovoo"]

    # triples_resid += +1.00 P(b,c) <m,l||d,i> t1(a,l) t1(b,m) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aicjkb->abcijk', tmps_["71_vovoov"])
    triples_resid -= einsum('akbijc->abcijk', tmps_["71_vovoov"])
    triples_resid -= einsum('ajcikb->abcijk', tmps_["71_vovoov"])

    # triples_resid += +1.00 P(j,k) <m,l||d,k> t1(b,l) t1(c,m) t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bkaijc->abcijk', tmps_["71_vovoov"])

    # triples_resid += +1.00 P(j,k) P(b,c) <m,l||d,k> t1(a,l) t1(b,m) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('akcijb->abcijk', tmps_["71_vovoov"])
    triples_resid += einsum('ajbikc->abcijk', tmps_["71_vovoov"])
    triples_resid -= einsum('bjaikc->abcijk', tmps_["71_vovoov"])
    triples_resid -= einsum('aibjkc->abcijk', tmps_["71_vovoov"])

    # triples_resid += +1.00 <m,l||d,i> t1(b,l) t1(c,m) t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('biajkc->abcijk', tmps_["71_vovoov"])
    del tmps_["71_vovoov"]

    # tmps_[18_vvvo](c,d,b,i) = 1.00 eri[vovv](c,l,d,e) * t2(e,b,i,l) // flops: o1v3 = o2v4 | mem: o1v3 = o1v3
    tmps_["18_vvvo"] = einsum('clde,ebil->cdbi', eri["vovv"], t2)

    # tmps_[56_vvovoo](a,c,j,b,i,k) = 1.00 eri[vovv](a,l,d,e) * t2(e,c,j,l) * t2(d,b,i,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["56_vvovoo"] = einsum('adcj,dbik->acjbik', tmps_["18_vvvo"], t2)
    triples_resid += einsum('cbjaik->abcijk', tmps_["56_vvovoo"])
    triples_resid += einsum('bciajk->abcijk', tmps_["56_vvovoo"])

    # triples_resid += +1.00 <l,c||d,e> t2(d,a,i,j) t2(e,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbkaij->abcijk', tmps_["56_vvovoo"])
    triples_resid -= einsum('bcjaik->abcijk', tmps_["56_vvovoo"])

    # triples_resid += +1.00 P(i,j) P(a,b) <l,a||d,e> t2(d,b,j,k) t2(e,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acibjk->abcijk', tmps_["56_vvovoo"])
    triples_resid += einsum('bckaij->abcijk', tmps_["56_vvovoo"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,j) t2(e,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ackbij->abcijk', tmps_["56_vvovoo"])
    triples_resid += einsum('acjbik->abcijk', tmps_["56_vvovoo"])

    # triples_resid += +1.00 P(i,j) <l,c||d,e> t2(d,a,j,k) t2(e,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbiajk->abcijk', tmps_["56_vvovoo"])
    del tmps_["56_vvovoo"]

    # tmps_[90_vvoo](b,a,i,j) = 1.00 eri[vovv](b,k,c,d) * t2(d,a,i,k) * t1(c,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["90_vvoo"] = einsum('bcai,cj->baij', tmps_["18_vvvo"], t1)
    del tmps_["18_vvvo"]
    doubles_resid -= einsum('abji->abij', tmps_["90_vvoo"])
    doubles_resid += einsum('baji->abij', tmps_["90_vvoo"])

    # doubles_resid += -1.00 P(i,j) P(a,b) <k,a||c,d> t1(c,j) t2(d,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["90_vvoo"])
    doubles_resid -= einsum('baij->abij', tmps_["90_vvoo"])
    del tmps_["90_vvoo"]

    # tmps_[19_vvvo](a,e,b,i) = 1.00 eri[vovv](a,l,d,e) * t2(d,b,i,l) // flops: o1v3 = o2v4 | mem: o1v3 = o1v3
    tmps_["19_vvvo"] = einsum('alde,dbil->aebi', eri["vovv"], t2)

    # tmps_[57_vvovoo](b,a,k,c,i,j) = 1.00 eri[vovv](b,l,d,e) * t2(d,a,k,l) * t2(e,c,i,j) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["57_vvovoo"] = einsum('beak,ecij->bakcij', tmps_["19_vvvo"], t2)
    del tmps_["19_vvvo"]

    # triples_resid += +1.00 P(j,k) <l,c||d,e> t2(d,a,k,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cakbij->abcijk', tmps_["57_vvovoo"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["57_vvovoo"])
    triples_resid += einsum('cajbik->abcijk', tmps_["57_vvovoo"])

    # triples_resid += +1.00 <l,c||d,e> t2(d,a,i,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('caibjk->abcijk', tmps_["57_vvovoo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <l,a||d,e> t2(d,b,k,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abkcij->abcijk', tmps_["57_vvovoo"])
    triples_resid += einsum('bakcij->abcijk', tmps_["57_vvovoo"])
    triples_resid += einsum('abjcik->abcijk', tmps_["57_vvovoo"])
    triples_resid -= einsum('bajcik->abcijk', tmps_["57_vvovoo"])
    triples_resid += einsum('baicjk->abcijk', tmps_["57_vvovoo"])
    del tmps_["57_vvovoo"]

    # tmps_[20_ovvo](l,d,b,i) = 1.00 eri[oovv](m,l,d,e) * t2(e,b,i,m) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["20_ovvo"] = einsum('mlde,ebim->ldbi', eri["oovv"], t2)

    # singles_resid += +1.00 <k,j||b,c> t1(b,j) t2(c,a,i,k)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid += einsum('jbai,bj->ai', tmps_["20_ovvo"], t1)

    # tmps_[70_vovoov](c,k,a,i,j,b) = 1.00 eri[oovv](m,l,d,e) * t2(e,c,k,m) * t2(d,a,i,j) * t1(b,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["70_vovoov"] = einsum('ldck,daij,bl->ckaijb', tmps_["20_ovvo"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(c,l) t2(d,a,j,k) t2(e,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('biajkc->abcijk', tmps_["70_vovoov"])
    triples_resid += einsum('ciajkb->abcijk', tmps_["70_vovoov"])

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,j) t2(e,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ckbija->abcijk', tmps_["70_vovoov"])

    # triples_resid += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,j) t2(e,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bkaijc->abcijk', tmps_["70_vovoov"])
    triples_resid += einsum('ckaijb->abcijk', tmps_["70_vovoov"])
    triples_resid += einsum('bjaikc->abcijk', tmps_["70_vovoov"])
    triples_resid -= einsum('cjaikb->abcijk', tmps_["70_vovoov"])

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,j,k) t2(e,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cibjka->abcijk', tmps_["70_vovoov"])
    triples_resid += einsum('cjbika->abcijk', tmps_["70_vovoov"])
    del tmps_["70_vovoov"]

    # tmps_[89_vovo](b,i,a,j) = 1.00 eri[oovv](l,k,c,d) * t2(d,b,i,l) * t2(c,a,j,k) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["89_vovo"] = einsum('kcbi,cajk->biaj', tmps_["20_ovvo"], t2)
    doubles_resid -= einsum('bjai->abij', tmps_["89_vovo"])

    # doubles_resid += +1.00 P(i,j) <l,k||c,d> t2(c,a,j,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('biaj->abij', tmps_["89_vovo"])
    del tmps_["89_vovo"]

    # tmps_[94_vvoo](a,b,j,i) = 1.00 t1(c,i) * eri[oovv](l,k,c,d) * t2(d,b,j,l) * t1(a,k) // flops: o2v2 = o3v2 o3v2 | mem: o2v2 = o3v1 o2v2
    tmps_["94_vvoo"] = einsum('ci,kcbj,ak->abji', t1, tmps_["20_ovvo"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["20_ovvo"]

    # doubles_resid += +1.00 P(i,j) P(a,b) <l,k||c,d> t1(c,j) t1(a,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["94_vvoo"])
    doubles_resid -= einsum('abji->abij', tmps_["94_vvoo"])
    doubles_resid += einsum('baji->abij', tmps_["94_vvoo"])
    doubles_resid -= einsum('baij->abij', tmps_["94_vvoo"])
    del tmps_["94_vvoo"]

    # tmps_[21_vooo](b,k,i,j) = 0.50 eri[vovv](b,k,c,d) * t2(c,d,i,j) // flops: o3v1 = o3v3 | mem: o3v1 = o3v1
    tmps_["21_vooo"] = 0.50 * einsum('bkcd,cdij->bkij', eri["vovv"], t2)

    # tmps_[79_vvovoo](a,b,k,c,i,j) = 1.00 t2(a,b,k,l) * eri[vovv](c,l,d,e) * t2(d,e,i,j) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["79_vvovoo"] = einsum('abkl,clij->abkcij', t2, tmps_["21_vooo"])
    triples_resid -= einsum('acibjk->abcijk', tmps_["79_vvovoo"])
    triples_resid += einsum('acjbik->abcijk', tmps_["79_vvovoo"])

    # triples_resid += -0.50 P(i,j) <l,c||d,e> t2(d,e,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abicjk->abcijk', tmps_["79_vvovoo"])

    # triples_resid += -0.50 P(i,j) P(a,b) <l,a||d,e> t2(d,e,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bciajk->abcijk', tmps_["79_vvovoo"])

    # triples_resid += -0.50 P(a,b) <l,a||d,e> t2(d,e,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bckaij->abcijk', tmps_["79_vvovoo"])
    triples_resid -= einsum('ackbij->abcijk', tmps_["79_vvovoo"])
    triples_resid -= einsum('abjcik->abcijk', tmps_["79_vvovoo"])

    # triples_resid += -0.50 <l,c||d,e> t2(d,e,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abkcij->abcijk', tmps_["79_vvovoo"])
    triples_resid -= einsum('bcjaik->abcijk', tmps_["79_vvovoo"])
    del tmps_["79_vvovoo"]

    # tmps_[103_voov](b,i,j,a) = 1.00 eri[vovv](b,k,c,d) * t2(c,d,i,j) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["103_voov"] = einsum('bkij,ak->bija', tmps_["21_vooo"], t1)
    del tmps_["21_vooo"]
    doubles_resid += einsum('bija->abij', tmps_["103_voov"])

    # doubles_resid += +0.50 P(a,b) <k,a||c,d> t1(b,k) t2(c,d,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('aijb->abij', tmps_["103_voov"])
    del tmps_["103_voov"]

    # tmps_[22_vovv](c,i,a,b) = 0.50 eri[oovo](l,k,c,i) * t2(a,b,l,k) // flops: o1v3 = o3v3 | mem: o1v3 = o1v3
    tmps_["22_vovv"] = 0.50 * einsum('lkci,ablk->ciab', eri["oovo"], t2)

    # tmps_[60_vooovv](a,i,k,j,b,c) = 1.00 t2(d,a,i,k) * eri[oovo](m,l,d,j) * t2(b,c,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["60_vooovv"] = einsum('daik,djbc->aikjbc', t2, tmps_["22_vovv"])

    # triples_resid += -0.50 P(j,k) <m,l||d,k> t2(d,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cijkab->abcijk', tmps_["60_vooovv"])
    triples_resid += einsum('bjkiac->abcijk', tmps_["60_vooovv"])
    triples_resid += einsum('aikjbc->abcijk', tmps_["60_vooovv"])

    # triples_resid += -0.50 P(j,k) P(a,b) <m,l||d,k> t2(d,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aijkbc->abcijk', tmps_["60_vooovv"])
    triples_resid += einsum('cikjab->abcijk', tmps_["60_vooovv"])
    triples_resid -= einsum('bikjac->abcijk', tmps_["60_vooovv"])

    # triples_resid += -0.50 P(a,b) <m,l||d,i> t2(d,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkibc->abcijk', tmps_["60_vooovv"])
    triples_resid += einsum('bijkac->abcijk', tmps_["60_vooovv"])

    # triples_resid += -0.50 <m,l||d,i> t2(d,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjkiab->abcijk', tmps_["60_vooovv"])
    del tmps_["60_vooovv"]

    # tmps_[92_ovvo](i,a,b,j) = 1.00 eri[oovo](l,k,c,i) * t2(a,b,l,k) * t1(c,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["92_ovvo"] = einsum('ciab,cj->iabj', tmps_["22_vovv"], t1)
    del tmps_["22_vovv"]

    # doubles_resid += +0.50 P(i,j) <l,k||c,j> t1(c,i) t2(a,b,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('jabi->abij', tmps_["92_ovvo"])
    doubles_resid -= einsum('iabj->abij', tmps_["92_ovvo"])
    del tmps_["92_ovvo"]

    # tmps_[23_ovvo](m,e,b,i) = 1.00 eri[oovv](m,l,d,e) * t2(d,b,i,l) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["23_ovvo"] = einsum('mlde,dbil->mebi', eri["oovv"], t2)

    # tmps_[48_vvoovo](a,b,j,k,c,i) = 1.00 t3(e,a,b,j,k,m) * eri[oovv](m,l,d,e) * t2(d,c,i,l) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
    tmps_["48_vvoovo"] = einsum('eabjkm,meci->abjkci', t3, tmps_["23_ovvo"])
    del tmps_["23_ovvo"]
    triples_resid -= einsum('bcikaj->abcijk', tmps_["48_vvoovo"])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t2(d,c,k,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijck->abcijk', tmps_["48_vvoovo"])
    triples_resid -= einsum('acjkbi->abcijk', tmps_["48_vvoovo"])

    # triples_resid += +1.00 <m,l||d,e> t2(d,c,i,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abjkci->abcijk', tmps_["48_vvoovo"])
    triples_resid += einsum('acikbj->abcijk', tmps_["48_vvoovo"])
    triples_resid -= einsum('acijbk->abcijk', tmps_["48_vvoovo"])
    triples_resid -= einsum('abikcj->abcijk', tmps_["48_vvoovo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,e> t2(d,a,k,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijak->abcijk', tmps_["48_vvoovo"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t2(d,a,i,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcjkai->abcijk', tmps_["48_vvoovo"])
    del tmps_["48_vvoovo"]

    # tmps_[24_ovvo](l,e,a,k) = 1.00 eri[oovv](m,l,d,e) * t2(d,a,k,m) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["24_ovvo"] = einsum('mlde,dakm->leak', eri["oovv"], t2)

    # tmps_[69_voovov](b,j,k,a,i,c) = 1.00 t2(e,b,j,k) * eri[oovv](m,l,d,e) * t2(d,a,i,m) * t1(c,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["69_voovov"] = einsum('ebjk,leai,cl->bjkaic', t2, tmps_["24_ovvo"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["24_ovvo"]

    # triples_resid += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,m) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bjkaic->abcijk', tmps_["69_voovov"])
    triples_resid += einsum('bikajc->abcijk', tmps_["69_voovov"])
    triples_resid += einsum('cikbja->abcijk', tmps_["69_voovov"])

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjkbia->abcijk', tmps_["69_voovov"])
    triples_resid += einsum('cijakb->abcijk', tmps_["69_voovov"])
    triples_resid += einsum('cjkaib->abcijk', tmps_["69_voovov"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(c,l) t2(d,a,k,m) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bijakc->abcijk', tmps_["69_voovov"])
    triples_resid -= einsum('cikajb->abcijk', tmps_["69_voovov"])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,k,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cijbka->abcijk', tmps_["69_voovov"])
    del tmps_["69_voovov"]

    # tmps_[25_vovo](b,j,a,i) = 1.00 eri[vovo](b,k,c,j) * t2(c,a,i,k) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["25_vovo"] = einsum('bkcj,caik->bjai', eri["vovo"], t2)
    doubles_resid -= einsum('biaj->abij', tmps_["25_vovo"])
    doubles_resid += einsum('aibj->abij', tmps_["25_vovo"])

    # doubles_resid += +1.00 P(i,j) P(a,b) <k,a||c,j> t2(c,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('ajbi->abij', tmps_["25_vovo"])
    doubles_resid += einsum('bjai->abij', tmps_["25_vovo"])
    del tmps_["25_vovo"]

    # tmps_[26_oovo](m,i,c,k) = 1.00 eri[oovo](m,l,d,i) * t2(d,c,k,l) // flops: o3v1 = o4v2 | mem: o3v1 = o3v1
    tmps_["26_oovo"] = einsum('mldi,dckl->mick', eri["oovo"], t2)

    # tmps_[67_ovovvo](j,a,i,b,c,k) = 1.00 eri[oovo](m,l,d,j) * t2(d,a,i,l) * t2(b,c,k,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["67_ovovvo"] = einsum('mjai,bckm->jaibck', tmps_["26_oovo"], t2)
    del tmps_["26_oovo"]
    triples_resid -= einsum('kaibcj->abcijk', tmps_["67_ovovvo"])
    triples_resid += einsum('jbkaci->abcijk', tmps_["67_ovovvo"])
    triples_resid -= einsum('kbjaci->abcijk', tmps_["67_ovovvo"])

    # triples_resid += +1.00 P(j,k) <m,l||d,i> t2(d,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ickabj->abcijk', tmps_["67_ovovvo"])
    triples_resid -= einsum('jbiack->abcijk', tmps_["67_ovovvo"])
    triples_resid -= einsum('ibkacj->abcijk', tmps_["67_ovovvo"])
    triples_resid -= einsum('iajbck->abcijk', tmps_["67_ovovvo"])
    triples_resid += einsum('jciabk->abcijk', tmps_["67_ovovvo"])

    # triples_resid += -1.00 P(i,k) <m,l||d,j> t2(d,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('jckabi->abcijk', tmps_["67_ovovvo"])
    triples_resid += einsum('kbiacj->abcijk', tmps_["67_ovovvo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,i> t2(d,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('iakbcj->abcijk', tmps_["67_ovovvo"])
    triples_resid -= einsum('icjabk->abcijk', tmps_["67_ovovvo"])

    # triples_resid += -1.00 P(i,k) P(a,b) <m,l||d,j> t2(d,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('jakbci->abcijk', tmps_["67_ovovvo"])

    # triples_resid += +1.00 P(i,j) <m,l||d,k> t2(d,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kcjabi->abcijk', tmps_["67_ovovvo"])
    triples_resid += einsum('ibjack->abcijk', tmps_["67_ovovvo"])
    triples_resid += einsum('jaibck->abcijk', tmps_["67_ovovvo"])
    triples_resid -= einsum('kciabj->abcijk', tmps_["67_ovovvo"])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,k> t2(d,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kajbci->abcijk', tmps_["67_ovovvo"])
    del tmps_["67_ovovvo"]

    # tmps_[27_oooo](l,k,i,j) = 0.50 eri[oovv](l,k,c,d) * t2(c,d,i,j) // flops: o4v0 = o4v2 | mem: o4v0 = o4v0
    tmps_["27_oooo"] = 0.50 * einsum('lkcd,cdij->lkij', eri["oovv"], t2)

    # doubles_resid += +0.25 <l,k||c,d> t2(c,d,i,j) t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
    doubles_resid += 0.50 * einsum('ablk,lkij->abij', t2, tmps_["27_oooo"])

    # doubles_resid += -0.50 <l,k||c,d> t1(a,k) t1(b,l) t2(c,d,i,j)  // flops: o2v2 += o4v1 o3v2 | mem: o2v2 += o3v1 o2v2
    doubles_resid -= einsum('lkij,bl,ak->abij', tmps_["27_oooo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])

    # tmps_[54_oovvvo](j,k,a,b,c,i) = 0.50 eri[oovv](m,l,d,e) * t2(d,e,j,k) * t3(a,b,c,i,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["54_oovvvo"] = 0.50 * einsum('mljk,abciml->jkabci', tmps_["27_oooo"], t3)

    # triples_resid += +0.25 P(i,j) <m,l||d,e> t2(d,e,j,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('jkabci->abcijk', tmps_["54_oovvvo"])
    triples_resid -= einsum('ikabcj->abcijk', tmps_["54_oovvvo"])

    # triples_resid += +0.25 <m,l||d,e> t2(d,e,i,j) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ijabck->abcijk', tmps_["54_oovvvo"])
    del tmps_["54_oovvvo"]

    # tmps_[74_vvooov](a,b,i,j,k,c) = 1.00 eri[oovv](m,l,d,e) * t2(d,e,j,k) * t1(c,l) * t2(a,b,i,m) // flops: o3v3 = o4v1 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["74_vvooov"] = einsum('mljk,cl,abim->abijkc', tmps_["27_oooo"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["27_oooo"]
    triples_resid -= einsum('abjikc->abcijk', tmps_["74_vvooov"])

    # triples_resid += +0.50 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,e,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijka->abcijk', tmps_["74_vvooov"])
    triples_resid -= einsum('ackijb->abcijk', tmps_["74_vvooov"])

    # triples_resid += +0.50 P(a,b) <m,l||d,e> t1(a,l) t2(d,e,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bckija->abcijk', tmps_["74_vvooov"])

    # triples_resid += +0.50 P(i,j) <m,l||d,e> t1(c,l) t2(d,e,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijkc->abcijk', tmps_["74_vvooov"])
    triples_resid += einsum('acjikb->abcijk', tmps_["74_vvooov"])

    # triples_resid += +0.50 <m,l||d,e> t1(c,l) t2(d,e,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abkijc->abcijk', tmps_["74_vvooov"])
    triples_resid -= einsum('bcjika->abcijk', tmps_["74_vvooov"])
    triples_resid -= einsum('acijkb->abcijk', tmps_["74_vvooov"])
    del tmps_["74_vvooov"]

    # tmps_[28_oovo](k,j,a,i) = 1.00 eri[oovo](l,k,c,j) * t2(c,a,i,l) // flops: o3v1 = o4v2 | mem: o3v1 = o3v1
    tmps_["28_oovo"] = einsum('lkcj,cail->kjai', eri["oovo"], t2)

    # tmps_[95_ovov](j,b,i,a) = 1.00 eri[oovo](l,k,c,j) * t2(c,b,i,l) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["95_ovov"] = einsum('kjbi,ak->jbia', tmps_["28_oovo"], t1)
    del tmps_["28_oovo"]
    doubles_resid += einsum('jaib->abij', tmps_["95_ovov"])
    doubles_resid += einsum('ibja->abij', tmps_["95_ovov"])

    # doubles_resid += -1.00 P(i,j) P(a,b) <l,k||c,j> t1(a,k) t2(c,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('jbia->abij', tmps_["95_ovov"])
    doubles_resid -= einsum('iajb->abij', tmps_["95_ovov"])
    del tmps_["95_ovov"]

    # tmps_[29_vvvo](a,b,d,j) = 1.00 eri[vvvv](a,b,c,d) * t1(c,j) // flops: o1v3 = o1v4 | mem: o1v3 = o1v3
    tmps_["29_vvvo"] = einsum('abcd,cj->abdj', eri["vvvv"], t1)

    # doubles_resid += -1.00 <a,b||c,d> t1(c,j) t1(d,i)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= einsum('di,abdj->abij', t1, tmps_["29_vvvo"])

    # tmps_[59_vvovoo](a,b,j,c,i,k) = 1.00 eri[vvvv](a,b,d,e) * t1(d,j) * t2(e,c,i,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["59_vvovoo"] = einsum('abej,ecik->abjcik', tmps_["29_vvvo"], t2)
    del tmps_["29_vvvo"]
    triples_resid -= einsum('acibjk->abcijk', tmps_["59_vvovoo"])
    triples_resid -= einsum('ackbij->abcijk', tmps_["59_vvovoo"])
    triples_resid -= einsum('abjcik->abcijk', tmps_["59_vvovoo"])

    # triples_resid += +1.00 P(b,c) <a,b||d,e> t1(d,i) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abicjk->abcijk', tmps_["59_vvovoo"])

    # triples_resid += +1.00 P(j,k) P(b,c) <a,b||d,e> t1(d,k) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abkcij->abcijk', tmps_["59_vvovoo"])

    # triples_resid += +1.00 <b,c||d,e> t1(d,i) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bciajk->abcijk', tmps_["59_vvovoo"])
    triples_resid += einsum('acjbik->abcijk', tmps_["59_vvovoo"])
    triples_resid -= einsum('bcjaik->abcijk', tmps_["59_vvovoo"])

    # triples_resid += +1.00 P(j,k) <b,c||d,e> t1(d,k) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bckaij->abcijk', tmps_["59_vvovoo"])
    del tmps_["59_vvovoo"]

    # tmps_[30_vovo](b,k,d,i) = 1.00 eri[vovv](b,k,c,d) * t1(c,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["30_vovo"] = einsum('bkcd,ci->bkdi', eri["vovv"], t1)

    # tmps_[49_vvoovo](a,c,i,k,b,j) = 1.00 t3(e,a,c,i,k,l) * eri[vovv](b,l,d,e) * t1(d,j) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
    tmps_["49_vvoovo"] = einsum('eacikl,blej->acikbj', t3, tmps_["30_vovo"])
    triples_resid -= einsum('bcikaj->abcijk', tmps_["49_vvoovo"])

    # triples_resid += -1.00 P(a,b) <l,a||d,e> t1(d,i) t3(e,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcjkai->abcijk', tmps_["49_vvoovo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,a||d,e> t1(d,k) t3(e,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijak->abcijk', tmps_["49_vvoovo"])
    triples_resid += einsum('acikbj->abcijk', tmps_["49_vvoovo"])

    # triples_resid += -1.00 P(j,k) <l,c||d,e> t1(d,k) t3(e,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijck->abcijk', tmps_["49_vvoovo"])
    triples_resid -= einsum('acjkbi->abcijk', tmps_["49_vvoovo"])

    # triples_resid += -1.00 <l,c||d,e> t1(d,i) t3(e,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abjkci->abcijk', tmps_["49_vvoovo"])
    triples_resid -= einsum('abikcj->abcijk', tmps_["49_vvoovo"])
    triples_resid -= einsum('acijbk->abcijk', tmps_["49_vvoovo"])
    del tmps_["49_vvoovo"]

    # tmps_[63_vovoov](a,i,b,j,k,c) = 1.00 eri[vovv](a,l,d,e) * t1(d,i) * t2(e,b,j,k) * t1(c,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["63_vovoov"] = einsum('alei,ebjk,cl->aibjkc', tmps_["30_vovo"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])
    triples_resid -= einsum('biajkc->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('ciajkb->abcijk', tmps_["63_vovoov"])

    # triples_resid += +1.00 P(b,c) <l,a||d,e> t1(d,i) t1(b,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aicjkb->abcijk', tmps_["63_vovoov"])
    triples_resid -= einsum('ajbikc->abcijk', tmps_["63_vovoov"])
    triples_resid -= einsum('bkaijc->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('ckaijb->abcijk', tmps_["63_vovoov"])
    triples_resid -= einsum('bjcika->abcijk', tmps_["63_vovoov"])

    # triples_resid += +1.00 P(j,k) P(a,b) <l,c||d,e> t1(d,k) t1(a,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ckbija->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('akbijc->abcijk', tmps_["63_vovoov"])

    # triples_resid += -1.00 P(j,k) P(a,c) <l,b||d,e> t1(d,k) t1(a,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bkcija->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('aibjkc->abcijk', tmps_["63_vovoov"])

    # triples_resid += +1.00 P(a,b) <l,c||d,e> t1(d,i) t1(a,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cibjka->abcijk', tmps_["63_vovoov"])
    triples_resid -= einsum('cjaikb->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('ajcikb->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('bjaikc->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('cjbika->abcijk', tmps_["63_vovoov"])

    # triples_resid += +1.00 P(j,k) P(b,c) <l,a||d,e> t1(d,k) t1(b,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('akcijb->abcijk', tmps_["63_vovoov"])

    # triples_resid += -1.00 P(a,c) <l,b||d,e> t1(d,i) t1(a,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bicjka->abcijk', tmps_["63_vovoov"])
    del tmps_["63_vovoov"]

    # tmps_[73_voovvo](a,k,j,b,c,i) = 1.00 eri[vovv](a,l,d,e) * t1(d,k) * t1(e,j) * t2(b,c,i,l) // flops: o3v3 = o3v2 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["73_voovvo"] = einsum('alek,ej,bcil->akjbci', tmps_["30_vovo"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])
    triples_resid += einsum('akibcj->abcijk', tmps_["73_voovvo"])
    triples_resid += einsum('bkjaci->abcijk', tmps_["73_voovvo"])
    triples_resid -= einsum('bkiacj->abcijk', tmps_["73_voovvo"])

    # triples_resid += +1.00 P(i,j) <l,c||d,e> t1(d,k) t1(e,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ckjabi->abcijk', tmps_["73_voovvo"])

    # triples_resid += +1.00 <l,c||d,e> t1(d,j) t1(e,i) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjiabk->abcijk', tmps_["73_voovvo"])
    triples_resid += einsum('bjiack->abcijk', tmps_["73_voovvo"])

    # triples_resid += +1.00 P(i,j) P(a,b) <l,a||d,e> t1(d,k) t1(e,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('akjbci->abcijk', tmps_["73_voovvo"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t1(d,j) t1(e,i) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajibck->abcijk', tmps_["73_voovvo"])
    triples_resid += einsum('ckiabj->abcijk', tmps_["73_voovvo"])
    del tmps_["73_voovvo"]

    # tmps_[98_vvoo](a,b,j,i) = 1.00 t1(d,i) * eri[vovv](b,k,c,d) * t1(c,j) * t1(a,k) // flops: o2v2 = o3v2 o3v2 | mem: o2v2 = o3v1 o2v2
    tmps_["98_vvoo"] = einsum('di,bkdj,ak->abji', t1, tmps_["30_vovo"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["30_vovo"]
    doubles_resid -= einsum('abji->abij', tmps_["98_vvoo"])

    # doubles_resid += -1.00 P(a,b) <k,a||c,d> t1(c,j) t1(d,i) t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('baji->abij', tmps_["98_vvoo"])
    del tmps_["98_vvoo"]

    # tmps_[31_vv](d,a) = 0.50 eri[oovv](l,k,c,d) * t2(c,a,l,k) // flops: o0v2 = o2v3 | mem: o0v2 = o0v2
    tmps_["31_vv"] = 0.50 * einsum('lkcd,calk->da', eri["oovv"], t2)

    # doubles_resid += -0.50 <l,k||c,d> t2(c,a,l,k) t2(d,b,i,j)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= einsum('da,dbij->abij', tmps_["31_vv"], t2)

    # tmps_[62_vvooov](b,c,i,j,k,a) = 1.00 t3(e,b,c,i,j,k) * eri[oovv](m,l,d,e) * t2(d,a,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["62_vvooov"] = einsum('ebcijk,ea->bcijka', t3, tmps_["31_vv"])
    del tmps_["31_vv"]

    # triples_resid += -0.50 P(a,b) <m,l||d,e> t2(d,a,m,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijka->abcijk', tmps_["62_vvooov"])

    # triples_resid += -0.50 <m,l||d,e> t2(d,c,m,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijkc->abcijk', tmps_["62_vvooov"])
    triples_resid += einsum('acijkb->abcijk', tmps_["62_vvooov"])
    del tmps_["62_vvooov"]

    # tmps_[32_vvoo](b,a,i,j) = 1.00 f[vv](b,c) * t2(c,a,i,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["32_vvoo"] = einsum('bc,caij->baij', f["vv"], t2)

    # doubles_resid += +1.00 P(a,b) f(a,c) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["32_vvoo"])
    doubles_resid -= einsum('baij->abij', tmps_["32_vvoo"])
    del tmps_["32_vvoo"]

    # tmps_[33_vvoo](a,b,j,i) = 1.00 eri[vvvo](a,b,c,j) * t1(c,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["33_vvoo"] = einsum('abcj,ci->abji', eri["vvvo"], t1)

    # doubles_resid += +1.00 P(i,j) <a,b||c,j> t1(c,i)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abji->abij', tmps_["33_vvoo"])
    doubles_resid -= einsum('abij->abij', tmps_["33_vvoo"])
    del tmps_["33_vvoo"]

    # tmps_[34_oovo](l,k,d,i) = 1.00 eri[oovv](l,k,c,d) * t1(c,i) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["34_oovo"] = einsum('lkcd,ci->lkdi', eri["oovv"], t1)

    # singles_resid += +0.50 <k,j||b,c> t1(b,i) t2(c,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
    singles_resid += 0.50 * einsum('cakj,kjci->ai', t2, tmps_["34_oovo"])

    # doubles_resid += -0.50 <l,k||c,d> t1(c,j) t1(d,i) t2(a,b,l,k)  // flops: o2v2 += o4v1 o4v2 | mem: o2v2 += o4v0 o2v2
    doubles_resid -= 0.50 * einsum('lkdj,di,ablk->abij', tmps_["34_oovo"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])

    # tmps_[50_vvooov](a,b,i,j,k,c) = 1.00 t3(e,a,b,i,j,m) * eri[oovv](m,l,d,e) * t1(d,k) * t1(c,l) // flops: o3v3 = o5v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["50_vvooov"] = einsum('eabijm,mlek,cl->abijkc', t3, tmps_["34_oovo"], t1, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t1(a,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijka->abcijk', tmps_["50_vvooov"])
    triples_resid -= einsum('abikjc->abcijk', tmps_["50_vvooov"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,i) t1(c,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abjkic->abcijk', tmps_["50_vvooov"])
    triples_resid -= einsum('acjkib->abcijk', tmps_["50_vvooov"])
    triples_resid -= einsum('acijkb->abcijk', tmps_["50_vvooov"])
    triples_resid -= einsum('bcikja->abcijk', tmps_["50_vvooov"])
    triples_resid += einsum('acikjb->abcijk', tmps_["50_vvooov"])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t1(d,k) t1(c,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijkc->abcijk', tmps_["50_vvooov"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,i) t1(a,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcjkia->abcijk', tmps_["50_vvooov"])
    del tmps_["50_vvooov"]

    # tmps_[52_oovvvo](i,k,a,b,c,j) = 0.50 t1(e,i) * eri[oovv](m,l,d,e) * t1(d,k) * t3(a,b,c,j,m,l) // flops: o3v3 = o4v1 o5v3 | mem: o3v3 = o4v0 o3v3
    tmps_["52_oovvvo"] = 0.50 * einsum('ei,mlek,abcjml->ikabcj', t1, tmps_["34_oovo"], t3, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += -0.50 <m,l||d,e> t1(d,j) t1(e,i) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ijabck->abcijk', tmps_["52_oovvvo"])
    triples_resid += einsum('ikabcj->abcijk', tmps_["52_oovvvo"])

    # triples_resid += -0.50 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('jkabci->abcijk', tmps_["52_oovvvo"])
    del tmps_["52_oovvvo"]

    # tmps_[55_ovvvoo](i,a,c,b,j,k) = 0.50 eri[oovv](m,l,d,e) * t1(d,i) * t2(a,c,m,l) * t2(e,b,j,k) // flops: o3v3 = o3v3 o3v4 | mem: o3v3 = o1v3 o3v3
    tmps_["55_ovvvoo"] = 0.50 * einsum('mlei,acml,ebjk->iacbjk', tmps_["34_oovo"], t2, t2, optimize=['einsum_path',(0,1),(0,1)])
    triples_resid -= einsum('jabcik->abcijk', tmps_["55_ovvvoo"])

    # triples_resid += +0.50 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kbcaij->abcijk', tmps_["55_ovvvoo"])
    triples_resid -= einsum('jbcaik->abcijk', tmps_["55_ovvvoo"])

    # triples_resid += +0.50 P(a,b) <m,l||d,e> t1(d,i) t2(e,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ibcajk->abcijk', tmps_["55_ovvvoo"])

    # triples_resid += +0.50 P(j,k) <m,l||d,e> t1(d,k) t2(e,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kabcij->abcijk', tmps_["55_ovvvoo"])
    triples_resid += einsum('jacbik->abcijk', tmps_["55_ovvvoo"])

    # triples_resid += +0.50 <m,l||d,e> t1(d,i) t2(e,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('iabcjk->abcijk', tmps_["55_ovvvoo"])
    triples_resid -= einsum('iacbjk->abcijk', tmps_["55_ovvvoo"])
    triples_resid -= einsum('kacbij->abcijk', tmps_["55_ovvvoo"])
    del tmps_["55_ovvvoo"]

    # tmps_[64_voovvo](b,k,j,a,c,i) = 1.00 t2(e,b,k,l) * eri[oovv](m,l,d,e) * t1(d,j) * t2(a,c,i,m) // flops: o3v3 = o4v2 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["64_voovvo"] = einsum('ebkl,mlej,acim->bkjaci', t2, tmps_["34_oovo"], t2, optimize=['einsum_path',(0,1),(0,1)])
    triples_resid += einsum('bjkaci->abcijk', tmps_["64_voovvo"])
    triples_resid -= einsum('cijabk->abcijk', tmps_["64_voovvo"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(d,i) t2(e,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ckiabj->abcijk', tmps_["64_voovvo"])
    triples_resid += einsum('bijack->abcijk', tmps_["64_voovvo"])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(d,k) t2(e,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjkabi->abcijk', tmps_["64_voovvo"])

    # triples_resid += +1.00 P(i,k) P(a,b) <m,l||d,e> t1(d,j) t2(e,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('akjbci->abcijk', tmps_["64_voovvo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,i) t2(e,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('akibcj->abcijk', tmps_["64_voovvo"])

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkbci->abcijk', tmps_["64_voovvo"])
    triples_resid -= einsum('aijbck->abcijk', tmps_["64_voovvo"])
    triples_resid += einsum('cikabj->abcijk', tmps_["64_voovvo"])
    triples_resid -= einsum('bkjaci->abcijk', tmps_["64_voovvo"])
    triples_resid -= einsum('bikacj->abcijk', tmps_["64_voovvo"])

    # triples_resid += +1.00 P(i,k) <m,l||d,e> t1(d,j) t2(e,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ckjabi->abcijk', tmps_["64_voovvo"])
    triples_resid -= einsum('bjiack->abcijk', tmps_["64_voovvo"])
    triples_resid += einsum('ajibck->abcijk', tmps_["64_voovvo"])
    triples_resid += einsum('bkiacj->abcijk', tmps_["64_voovvo"])
    triples_resid += einsum('aikbcj->abcijk', tmps_["64_voovvo"])
    triples_resid += einsum('cjiabk->abcijk', tmps_["64_voovvo"])
    del tmps_["64_voovvo"]

    # tmps_[88_voovoo](b,m,j,a,i,k) = 1.00 eri[oovv](m,l,d,e) * t1(d,j) * t2(e,a,i,k) * t1(b,l) // flops: o4v2 = o5v2 o5v2 | mem: o4v2 = o5v1 o4v2
    tmps_["88_voovoo"] = einsum('mlej,eaik,bl->bmjaik', tmps_["34_oovo"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

    # tmps_[109_vvovoo](c,a,i,b,j,k) = 1.00 t1(c,m) * t1(a,l) * tmps_[34_oovo](m,l,e,i) * t2(e,b,j,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["109_vvovoo"] = einsum('cm,amibjk->caibjk', t1, tmps_["88_voovoo"])
    del tmps_["88_voovoo"]

    # triples_resid += -1.00 P(j,k) P(b,c) <m,l||d,e> t1(d,k) t1(a,l) t1(b,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bakcij->abcijk', tmps_["109_vvovoo"])
    triples_resid += einsum('cbjaik->abcijk', tmps_["109_vvovoo"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(d,k) t1(b,l) t1(c,m) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbkaij->abcijk', tmps_["109_vvovoo"])

    # triples_resid += -1.00 P(b,c) <m,l||d,e> t1(d,i) t1(a,l) t1(b,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('baicjk->abcijk', tmps_["109_vvovoo"])
    triples_resid += einsum('caibjk->abcijk', tmps_["109_vvovoo"])

    # triples_resid += -1.00 <m,l||d,e> t1(d,i) t1(b,l) t1(c,m) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbiajk->abcijk', tmps_["109_vvovoo"])
    triples_resid -= einsum('cajbik->abcijk', tmps_["109_vvovoo"])
    triples_resid += einsum('cakbij->abcijk', tmps_["109_vvovoo"])
    triples_resid += einsum('bajcik->abcijk', tmps_["109_vvovoo"])
    del tmps_["109_vvovoo"]

    # tmps_[107_vooo](c,m,k,i) = 1.00 eri[oovv](m,l,d,e) * t1(d,k) * t1(e,i) * t1(c,l) // flops: o3v1 = o4v1 o4v1 | mem: o3v1 = o4v0 o3v1
    tmps_["107_vooo"] = einsum('mlek,ei,cl->cmki', tmps_["34_oovo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["34_oovo"]

    # doubles_resid += +1.00 <l,k||c,d> t1(c,j) t1(d,i) t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('bl,alji->abij', t1, tmps_["107_vooo"])

    # tmps_[108_voovvo](c,k,j,a,b,i) = 1.00 t1(c,l) * tmps_[34_oovo](m,l,e,k) * t1(e,j) * t2(a,b,i,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["108_voovvo"] = einsum('cmkj,abim->ckjabi', tmps_["107_vooo"], t2)
    del tmps_["107_vooo"]
    triples_resid += einsum('akibcj->abcijk', tmps_["108_voovvo"])
    triples_resid += einsum('bjiack->abcijk', tmps_["108_voovvo"])

    # triples_resid += -1.00 <m,l||d,e> t1(d,j) t1(e,i) t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjiabk->abcijk', tmps_["108_voovvo"])

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(d,j) t1(e,i) t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajibck->abcijk', tmps_["108_voovvo"])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ckjabi->abcijk', tmps_["108_voovvo"])
    triples_resid += einsum('bkjaci->abcijk', tmps_["108_voovvo"])

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t1(e,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('akjbci->abcijk', tmps_["108_voovvo"])
    triples_resid -= einsum('bkiacj->abcijk', tmps_["108_voovvo"])
    triples_resid += einsum('ckiabj->abcijk', tmps_["108_voovvo"])
    del tmps_["108_voovvo"]

    # tmps_[35_vooo](a,k,j,i) = 1.00 eri[vovo](a,k,c,j) * t1(c,i) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["35_vooo"] = einsum('akcj,ci->akji', eri["vovo"], t1)

    # tmps_[68_voovvo](b,j,k,a,c,i) = 1.00 eri[vovo](b,l,d,j) * t1(d,k) * t2(a,c,i,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["68_voovvo"] = einsum('bljk,acil->bjkaci', tmps_["35_vooo"], t2)
    triples_resid += einsum('bijack->abcijk', tmps_["68_voovvo"])
    triples_resid -= einsum('bjiack->abcijk', tmps_["68_voovvo"])
    triples_resid -= einsum('aijbck->abcijk', tmps_["68_voovvo"])
    triples_resid -= einsum('bkjaci->abcijk', tmps_["68_voovvo"])
    triples_resid += einsum('bkiacj->abcijk', tmps_["68_voovvo"])
    triples_resid -= einsum('bikacj->abcijk', tmps_["68_voovvo"])

    # triples_resid += -1.00 P(i,j) P(a,b) <l,a||d,k> t1(d,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('akjbci->abcijk', tmps_["68_voovvo"])

    # triples_resid += +1.00 P(i,k) P(a,b) <l,a||d,j> t1(d,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkbci->abcijk', tmps_["68_voovvo"])
    triples_resid += einsum('cjiabk->abcijk', tmps_["68_voovvo"])
    triples_resid -= einsum('ckiabj->abcijk', tmps_["68_voovvo"])

    # triples_resid += +1.00 P(i,k) <l,c||d,j> t1(d,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjkabi->abcijk', tmps_["68_voovvo"])
    triples_resid += einsum('ajibck->abcijk', tmps_["68_voovvo"])
    triples_resid -= einsum('cijabk->abcijk', tmps_["68_voovvo"])
    triples_resid += einsum('bjkaci->abcijk', tmps_["68_voovvo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,a||d,i> t1(d,k) t2(b,c,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aikbcj->abcijk', tmps_["68_voovvo"])

    # triples_resid += -1.00 P(j,k) <l,c||d,i> t1(d,k) t2(a,b,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cikabj->abcijk', tmps_["68_voovvo"])
    triples_resid -= einsum('akibcj->abcijk', tmps_["68_voovvo"])

    # triples_resid += -1.00 P(i,j) <l,c||d,k> t1(d,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ckjabi->abcijk', tmps_["68_voovvo"])
    del tmps_["68_voovvo"]

    # tmps_[96_voov](a,i,j,b) = 1.00 eri[vovo](a,k,c,i) * t1(c,j) * t1(b,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["96_voov"] = einsum('akij,bk->aijb', tmps_["35_vooo"], t1)
    del tmps_["35_vooo"]
    doubles_resid -= einsum('bija->abij', tmps_["96_voov"])
    doubles_resid += einsum('aijb->abij', tmps_["96_voov"])
    doubles_resid += einsum('bjia->abij', tmps_["96_voov"])

    # doubles_resid += +1.00 P(i,j) P(a,b) <k,a||c,j> t1(c,i) t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('ajib->abij', tmps_["96_voov"])
    del tmps_["96_voov"]

    # tmps_[36_oovo](l,k,c,i) = 1.00 eri[oovv](l,k,c,d) * t1(d,i) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["36_oovo"] = einsum('lkcd,di->lkci', eri["oovv"], t1)

    # singles_resid += +1.00 <k,j||b,c> t1(b,j) t1(c,i) t1(a,k)  // flops: o1v1 += o3v1 o2v1 | mem: o1v1 += o2v0 o1v1
    singles_resid += einsum('kjbi,bj,ak->ai', tmps_["36_oovo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])

    # tmps_[83_ovvvoo](i,a,b,c,j,k) = 1.00 t1(d,l) * eri[oovv](m,l,d,e) * t1(e,i) * t3(a,b,c,j,k,m) // flops: o3v3 = o3v1 o4v3 | mem: o3v3 = o2v0 o3v3
    tmps_["83_ovvvoo"] = einsum('dl,mldi,abcjkm->iabcjk', t1, tmps_["36_oovo"], t3, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t1(e,i) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('iabcjk->abcijk', tmps_["83_ovvvoo"])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t1(d,l) t1(e,k) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kabcij->abcijk', tmps_["83_ovvvoo"])
    triples_resid -= einsum('jabcik->abcijk', tmps_["83_ovvvoo"])
    del tmps_["83_ovvvoo"]

    # tmps_[100_vvoo](a,b,i,j) = 1.00 eri[oovv](l,k,c,d) * t1(d,j) * t1(c,k) * t2(a,b,i,l) // flops: o2v2 = o3v1 o3v2 | mem: o2v2 = o2v0 o2v2
    tmps_["100_vvoo"] = einsum('lkcj,ck,abil->abij', tmps_["36_oovo"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["36_oovo"]
    doubles_resid -= einsum('abji->abij', tmps_["100_vvoo"])

    # doubles_resid += +1.00 P(i,j) <l,k||c,d> t1(c,k) t1(d,j) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["100_vvoo"])
    del tmps_["100_vvoo"]

    # tmps_[37_ovoo](k,a,i,j) = 1.00 f[ov](k,c) * t2(c,a,i,j) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["37_ovoo"] = einsum('kc,caij->kaij', f["ov"], t2)

    # tmps_[75_voovvo](a,i,j,b,c,k) = 1.00 f[ov](l,d) * t2(d,a,i,j) * t2(b,c,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["75_voovvo"] = einsum('laij,bckl->aijbck', tmps_["37_ovoo"], t2)
    triples_resid += einsum('bjkaci->abcijk', tmps_["75_voovvo"])
    triples_resid += einsum('bijack->abcijk', tmps_["75_voovvo"])

    # triples_resid += -1.00 P(a,b) f(l,d) t2(d,a,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aijbck->abcijk', tmps_["75_voovvo"])
    triples_resid += einsum('aikbcj->abcijk', tmps_["75_voovvo"])

    # triples_resid += -1.00 P(i,j) P(a,b) f(l,d) t2(d,a,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkbci->abcijk', tmps_["75_voovvo"])
    triples_resid -= einsum('bikacj->abcijk', tmps_["75_voovvo"])

    # triples_resid += -1.00 P(i,j) f(l,d) t2(d,c,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjkabi->abcijk', tmps_["75_voovvo"])
    triples_resid += einsum('cikabj->abcijk', tmps_["75_voovvo"])

    # triples_resid += -1.00 f(l,d) t2(d,c,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cijabk->abcijk', tmps_["75_voovvo"])
    del tmps_["75_voovvo"]

    # tmps_[104_vvoo](a,b,i,j) = 1.00 t1(a,k) * f[ov](k,c) * t2(c,b,i,j) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["104_vvoo"] = einsum('ak,kbij->abij', t1, tmps_["37_ovoo"])
    del tmps_["37_ovoo"]

    # doubles_resid += -1.00 P(a,b) f(k,c) t1(a,k) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["104_vvoo"])
    doubles_resid += einsum('baij->abij', tmps_["104_vvoo"])
    del tmps_["104_vvoo"]

    # tmps_[38_oo](l,i) = 0.50 eri[oovv](l,k,c,d) * t2(c,d,i,k) // flops: o2v0 = o3v2 | mem: o2v0 = o2v0
    tmps_["38_oo"] = 0.50 * einsum('lkcd,cdik->li', eri["oovv"], t2)

    # tmps_[86_vvvooo](a,b,c,j,k,i) = 1.00 t3(a,b,c,j,k,m) * eri[oovv](m,l,d,e) * t2(d,e,i,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["86_vvvooo"] = einsum('abcjkm,mi->abcjki', t3, tmps_["38_oo"])

    # triples_resid += -0.50 P(j,k) <m,l||d,e> t2(d,e,k,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcijk->abcijk', tmps_["86_vvvooo"])

    # triples_resid += -0.50 <m,l||d,e> t2(d,e,i,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcjki->abcijk', tmps_["86_vvvooo"])
    triples_resid += einsum('abcikj->abcijk', tmps_["86_vvvooo"])
    del tmps_["86_vvvooo"]

    # tmps_[106_ovvo](j,a,b,i) = 1.00 eri[oovv](l,k,c,d) * t2(c,d,j,k) * t2(a,b,i,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["106_ovvo"] = einsum('lj,abil->jabi', tmps_["38_oo"], t2)
    del tmps_["38_oo"]
    doubles_resid += einsum('iabj->abij', tmps_["106_ovvo"])

    # doubles_resid += -0.50 P(i,j) <l,k||c,d> t2(c,d,j,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('jabi->abij', tmps_["106_ovvo"])
    del tmps_["106_ovvo"]

    # tmps_[39_vv](c,b) = 1.00 eri[oovv](l,k,c,d) * t2(d,b,l,k) // flops: o0v2 = o2v3 | mem: o0v2 = o0v2
    tmps_["39_vv"] = einsum('lkcd,dblk->cb', eri["oovv"], t2)

    # doubles_resid += -0.50 <l,k||c,d> t2(c,a,i,j) t2(d,b,l,k)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= 0.50 * einsum('cb,caij->abij', tmps_["39_vv"], t2)
    del tmps_["39_vv"]

    # tmps_[40_voov](b,i,j,a) = 1.00 eri[vooo](b,k,i,j) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["40_voov"] = einsum('bkij,ak->bija', eri["vooo"], t1)
    doubles_resid += einsum('bija->abij', tmps_["40_voov"])

    # doubles_resid += +1.00 P(a,b) <k,a||i,j> t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('aijb->abij', tmps_["40_voov"])
    del tmps_["40_voov"]

    # tmps_[41_ovvo](j,a,b,i) = 1.00 f[oo](k,j) * t2(a,b,i,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["41_ovvo"] = einsum('kj,abik->jabi', f["oo"], t2)

    # doubles_resid += -1.00 P(i,j) f(k,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('jabi->abij', tmps_["41_ovvo"])
    doubles_resid += einsum('iabj->abij', tmps_["41_ovvo"])
    del tmps_["41_ovvo"]

    # tmps_[42_oooo](l,k,i,j) = 1.00 eri[oovo](l,k,c,i) * t1(c,j) // flops: o4v0 = o4v1 | mem: o4v0 = o4v0
    tmps_["42_oooo"] = einsum('lkci,cj->lkij', eri["oovo"], t1)

    # tmps_[51_oovvvo](i,j,a,b,c,k) = 0.50 eri[oovo](m,l,d,i) * t1(d,j) * t3(a,b,c,k,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["51_oovvvo"] = 0.50 * einsum('mlij,abckml->ijabck', tmps_["42_oooo"], t3)
    triples_resid += einsum('jiabck->abcijk', tmps_["51_oovvvo"])

    # triples_resid += +0.50 P(j,k) <m,l||d,i> t1(d,k) t3(a,b,c,j,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ikabcj->abcijk', tmps_["51_oovvvo"])
    triples_resid -= einsum('kiabcj->abcijk', tmps_["51_oovvvo"])

    # triples_resid += +0.50 P(i,j) <m,l||d,k> t1(d,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kjabci->abcijk', tmps_["51_oovvvo"])

    # triples_resid += -0.50 P(i,k) <m,l||d,j> t1(d,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('jkabci->abcijk', tmps_["51_oovvvo"])
    triples_resid -= einsum('ijabck->abcijk', tmps_["51_oovvvo"])
    del tmps_["51_oovvvo"]

    # tmps_[65_vvooov](a,b,j,k,i,c) = 1.00 eri[oovo](m,l,d,k) * t1(d,i) * t1(c,l) * t2(a,b,j,m) // flops: o3v3 = o4v1 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["65_vvooov"] = einsum('mlki,cl,abjm->abjkic', tmps_["42_oooo"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])
    triples_resid -= einsum('abkijc->abcijk', tmps_["65_vvooov"])
    triples_resid += einsum('acjkib->abcijk', tmps_["65_vvooov"])
    triples_resid += einsum('bckjia->abcijk', tmps_["65_vvooov"])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,k> t1(d,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcikja->abcijk', tmps_["65_vvooov"])

    # triples_resid += +1.00 P(i,j) <m,l||d,k> t1(d,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abikjc->abcijk', tmps_["65_vvooov"])

    # triples_resid += +1.00 P(j,k) <m,l||d,i> t1(d,k) t1(c,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abjikc->abcijk', tmps_["65_vvooov"])
    triples_resid += einsum('acijkb->abcijk', tmps_["65_vvooov"])
    triples_resid -= einsum('bcjkia->abcijk', tmps_["65_vvooov"])
    triples_resid += einsum('ackijb->abcijk', tmps_["65_vvooov"])
    triples_resid -= einsum('abjkic->abcijk', tmps_["65_vvooov"])

    # triples_resid += -1.00 P(i,k) P(a,b) <m,l||d,j> t1(d,k) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijka->abcijk', tmps_["65_vvooov"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,i> t1(d,k) t1(a,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcjika->abcijk', tmps_["65_vvooov"])
    triples_resid += einsum('abkjic->abcijk', tmps_["65_vvooov"])
    triples_resid -= einsum('acikjb->abcijk', tmps_["65_vvooov"])
    triples_resid -= einsum('ackjib->abcijk', tmps_["65_vvooov"])
    triples_resid -= einsum('acjikb->abcijk', tmps_["65_vvooov"])

    # triples_resid += -1.00 P(i,k) <m,l||d,j> t1(d,k) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijkc->abcijk', tmps_["65_vvooov"])
    triples_resid -= einsum('bckija->abcijk', tmps_["65_vvooov"])
    del tmps_["65_vvooov"]

    # tmps_[99_oovv](j,i,b,a) = 1.00 eri[oovo](l,k,c,j) * t1(c,i) * t1(b,l) * t1(a,k) // flops: o2v2 = o4v1 o3v2 | mem: o2v2 = o3v1 o2v2
    tmps_["99_oovv"] = einsum('lkji,bl,ak->jiba', tmps_["42_oooo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["42_oooo"]

    # doubles_resid += -1.00 P(i,j) <l,k||c,j> t1(c,i) t1(a,k) t1(b,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('jiba->abij', tmps_["99_oovv"])
    doubles_resid += einsum('ijba->abij', tmps_["99_oovv"])
    del tmps_["99_oovv"]

    # tmps_[43_ooov](l,i,j,a) = 1.00 eri[oooo](l,k,i,j) * t1(a,k) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["43_ooov"] = einsum('lkij,ak->lija', eri["oooo"], t1)

    # doubles_resid += -1.00 <l,k||i,j> t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('lija,bl->abij', tmps_["43_ooov"], t1)

    # tmps_[78_vvooov](b,c,j,i,k,a) = 1.00 t2(b,c,j,m) * eri[oooo](m,l,i,k) * t1(a,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["78_vvooov"] = einsum('bcjm,mika->bcjika', t2, tmps_["43_ooov"])
    del tmps_["43_ooov"]

    # triples_resid += +1.00 P(a,b) <m,l||i,j> t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bckija->abcijk', tmps_["78_vvooov"])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||j,k> t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijka->abcijk', tmps_["78_vvooov"])
    triples_resid -= einsum('ackijb->abcijk', tmps_["78_vvooov"])

    # triples_resid += +1.00 P(i,j) <m,l||j,k> t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijkc->abcijk', tmps_["78_vvooov"])
    triples_resid -= einsum('abjikc->abcijk', tmps_["78_vvooov"])
    triples_resid += einsum('acjikb->abcijk', tmps_["78_vvooov"])
    triples_resid -= einsum('bcjika->abcijk', tmps_["78_vvooov"])
    triples_resid -= einsum('acijkb->abcijk', tmps_["78_vvooov"])

    # triples_resid += +1.00 <m,l||i,j> t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abkijc->abcijk', tmps_["78_vvooov"])
    del tmps_["78_vvooov"]

    # tmps_[44_vv](a,d) = 1.00 eri[vovv](a,k,c,d) * t1(c,k) // flops: o0v2 = o1v3 | mem: o0v2 = o0v2
    tmps_["44_vv"] = einsum('akcd,ck->ad', eri["vovv"], t1)

    # singles_resid += +1.00 <j,a||b,c> t1(b,j) t1(c,i)  // flops: o1v1 += o1v2 | mem: o1v1 += o1v1
    singles_resid -= einsum('ac,ci->ai', tmps_["44_vv"], t1)

    # tmps_[61_vvvooo](c,a,b,i,j,k) = 1.00 eri[vovv](c,l,d,e) * t1(d,l) * t3(e,a,b,i,j,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["61_vvvooo"] = einsum('ce,eabijk->cabijk', tmps_["44_vv"], t3)

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t1(d,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcijk->abcijk', tmps_["61_vvvooo"])

    # triples_resid += +1.00 <l,c||d,e> t1(d,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabijk->abcijk', tmps_["61_vvvooo"])
    triples_resid += einsum('bacijk->abcijk', tmps_["61_vvvooo"])
    del tmps_["61_vvvooo"]

    # tmps_[91_voov](b,i,j,a) = 1.00 t2(d,b,i,j) * eri[vovv](a,k,c,d) * t1(c,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["91_voov"] = einsum('dbij,ad->bija', t2, tmps_["44_vv"])
    del tmps_["44_vv"]
    doubles_resid += einsum('aijb->abij', tmps_["91_voov"])

    # doubles_resid += +1.00 P(a,b) <k,a||c,d> t1(c,k) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('bija->abij', tmps_["91_voov"])
    del tmps_["91_voov"]

    # tmps_[45_ov](l,d) = 1.00 eri[oovv](l,k,c,d) * t1(c,k) // flops: o1v1 = o2v2 | mem: o1v1 = o1v1
    tmps_["45_ov"] = einsum('lkcd,ck->ld', eri["oovv"], t1)

    # doubles_resid += -1.00 <l,k||c,d> t1(c,k) t3(d,a,b,i,j,l)  // flops: o2v2 += o3v3 | mem: o2v2 += o2v2
    doubles_resid -= einsum('ld,dabijl->abij', tmps_["45_ov"], t3)

    # tmps_[72_voovvo](a,i,j,b,c,k) = 1.00 t2(e,a,i,j) * eri[oovv](m,l,d,e) * t1(d,l) * t2(b,c,k,m) // flops: o3v3 = o3v2 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["72_voovvo"] = einsum('eaij,me,bckm->aijbck', t2, tmps_["45_ov"], t2, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t2(e,c,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cijabk->abcijk', tmps_["72_voovvo"])
    triples_resid += einsum('bikacj->abcijk', tmps_["72_voovvo"])
    triples_resid -= einsum('bjkaci->abcijk', tmps_["72_voovvo"])

    # triples_resid += +1.00 P(i,j) <m,l||d,e> t1(d,l) t2(e,c,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cjkabi->abcijk', tmps_["72_voovvo"])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,l) t2(e,a,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ajkbci->abcijk', tmps_["72_voovvo"])
    triples_resid -= einsum('cikabj->abcijk', tmps_["72_voovvo"])
    triples_resid -= einsum('aikbcj->abcijk', tmps_["72_voovvo"])
    triples_resid -= einsum('bijack->abcijk', tmps_["72_voovvo"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,l) t2(e,a,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aijbck->abcijk', tmps_["72_voovvo"])
    del tmps_["72_voovvo"]

    # tmps_[81_vvooov](b,c,i,j,k,a) = 1.00 t3(e,b,c,i,j,k) * eri[oovv](m,l,d,e) * t1(d,l) * t1(a,m) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["81_vvooov"] = einsum('ebcijk,me,am->bcijka', t3, tmps_["45_ov"], t1, optimize=['einsum_path',(0,1),(0,1)])
    triples_resid -= einsum('acijkb->abcijk', tmps_["81_vvooov"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t1(c,m) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijkc->abcijk', tmps_["81_vvooov"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,l) t1(a,m) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijka->abcijk', tmps_["81_vvooov"])
    del tmps_["81_vvooov"]

    # tmps_[97_vvoo](a,b,i,j) = 1.00 eri[oovv](l,k,c,d) * t1(c,k) * t2(d,b,i,j) * t1(a,l) // flops: o2v2 = o3v2 o3v2 | mem: o2v2 = o3v1 o2v2
    tmps_["97_vvoo"] = einsum('ld,dbij,al->abij', tmps_["45_ov"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["45_ov"]
    doubles_resid -= einsum('baij->abij', tmps_["97_vvoo"])

    # doubles_resid += +1.00 P(a,b) <l,k||c,d> t1(c,k) t1(a,l) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["97_vvoo"])
    del tmps_["97_vvoo"]

    # tmps_[46_oo](l,i) = 1.00 eri[oovo](l,k,c,i) * t1(c,k) // flops: o2v0 = o3v1 | mem: o2v0 = o2v0
    tmps_["46_oo"] = einsum('lkci,ck->li', eri["oovo"], t1)

    # singles_resid += +1.00 <k,j||b,i> t1(b,j) t1(a,k)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid += einsum('ak,ki->ai', t1, tmps_["46_oo"])

    # tmps_[85_ovvvoo](k,a,b,c,i,j) = 1.00 eri[oovo](m,l,d,k) * t1(d,l) * t3(a,b,c,i,j,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["85_ovvvoo"] = einsum('mk,abcijm->kabcij', tmps_["46_oo"], t3)

    # triples_resid += +1.00 <m,l||d,i> t1(d,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('iabcjk->abcijk', tmps_["85_ovvvoo"])
    triples_resid -= einsum('jabcik->abcijk', tmps_["85_ovvvoo"])

    # triples_resid += +1.00 P(j,k) <m,l||d,k> t1(d,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kabcij->abcijk', tmps_["85_ovvvoo"])
    del tmps_["85_ovvvoo"]

    # tmps_[105_vvoo](a,b,j,i) = 1.00 t2(a,b,j,l) * eri[oovo](l,k,c,i) * t1(c,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["105_vvoo"] = einsum('abjl,li->abji', t2, tmps_["46_oo"])
    del tmps_["46_oo"]

    # doubles_resid += +1.00 P(i,j) <l,k||c,j> t1(c,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["105_vvoo"])
    doubles_resid -= einsum('abji->abij', tmps_["105_vvoo"])
    del tmps_["105_vvoo"]

    # tmps_[47_oo](k,j) = 1.00 f[ov](k,c) * t1(c,j) // flops: o2v0 = o2v1 | mem: o2v0 = o2v0
    tmps_["47_oo"] = einsum('kc,cj->kj', f["ov"], t1)

    # singles_resid += -1.00 f(j,b) t1(b,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid -= einsum('aj,ji->ai', t1, tmps_["47_oo"])

    # tmps_[84_vvvooo](a,b,c,j,k,i) = 1.00 t3(a,b,c,j,k,l) * f[ov](l,d) * t1(d,i) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["84_vvvooo"] = einsum('abcjkl,li->abcjki', t3, tmps_["47_oo"])

    # triples_resid += -1.00 f(l,d) t1(d,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcjki->abcijk', tmps_["84_vvvooo"])
    triples_resid += einsum('abcikj->abcijk', tmps_["84_vvvooo"])

    # triples_resid += -1.00 P(j,k) f(l,d) t1(d,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcijk->abcijk', tmps_["84_vvvooo"])
    del tmps_["84_vvvooo"]

    # tmps_[101_vvoo](a,b,j,i) = 1.00 t2(a,b,j,k) * f[ov](k,c) * t1(c,i) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["101_vvoo"] = einsum('abjk,ki->abji', t2, tmps_["47_oo"])
    del tmps_["47_oo"]

    # doubles_resid += -1.00 P(i,j) f(k,c) t1(c,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["101_vvoo"])
    doubles_resid += einsum('abji->abij', tmps_["101_vvoo"])
    del tmps_["101_vvoo"]

    return singles_residual, doubles_residual, triples_residual


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



