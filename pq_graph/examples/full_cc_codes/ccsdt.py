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

    # tmps_[1_vvvooo](a,b,c,i,j,k) = 0.50 eri[vvvv](a,b,d,e) * t3(d,e,c,i,j,k) // flops: o3v3 = o3v5 | mem: o3v3 = o3v3
    tmps_["1_vvvooo"] = 0.50 * einsum('abde,decijk->abcijk', eri["vvvv"], t3)
    triples_resid = -1.00 * einsum('acbijk->abcijk', tmps_["1_vvvooo"])

    # triples_resid += +0.50 <b,c||d,e> t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcaijk->abcijk', tmps_["1_vvvooo"])

    # triples_resid += +0.50 P(b,c) <a,b||d,e> t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["1_vvvooo"])
    del tmps_["1_vvvooo"]

    # tmps_[2_vovvoo](a,k,b,c,i,j) = 1.00 eri[vovo](a,l,d,k) * t3(d,b,c,i,j,l) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
    tmps_["2_vovvoo"] = einsum('aldk,dbcijl->akbcij', eri["vovo"], t3)
    triples_resid += einsum('cjabik->abcijk', tmps_["2_vovvoo"])
    triples_resid += einsum('bkacij->abcijk', tmps_["2_vovvoo"])
    triples_resid -= einsum('bjacik->abcijk', tmps_["2_vovvoo"])
    triples_resid += einsum('ajbcik->abcijk', tmps_["2_vovvoo"])

    # triples_resid += +1.00 <l,c||d,i> t3(d,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ciabjk->abcijk', tmps_["2_vovvoo"])
    triples_resid += einsum('biacjk->abcijk', tmps_["2_vovvoo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <l,a||d,k> t3(d,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('akbcij->abcijk', tmps_["2_vovvoo"])

    # triples_resid += +1.00 P(a,b) <l,a||d,i> t3(d,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aibcjk->abcijk', tmps_["2_vovvoo"])

    # triples_resid += +1.00 P(j,k) <l,c||d,k> t3(d,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ckabij->abcijk', tmps_["2_vovvoo"])
    del tmps_["2_vovvoo"]

    # tmps_[3_vovooo](c,l,a,i,j,k) = 0.50 eri[vovv](c,l,d,e) * t3(d,e,a,i,j,k) // flops: o4v2 = o4v4 | mem: o4v2 = o4v2
    tmps_["3_vovooo"] = 0.50 * einsum('clde,deaijk->claijk', eri["vovv"], t3)

    # tmps_[80_vvvooo](b,c,a,i,j,k) = 1.00 t1(b,l) * eri[vovv](c,l,d,e) * t3(d,e,a,i,j,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["80_vvvooo"] = einsum('bl,claijk->bcaijk', t1, tmps_["3_vovooo"])
    del tmps_["3_vovooo"]

    # triples_resid += -0.50 P(a,c) <l,b||d,e> t1(a,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["80_vvvooo"])

    # triples_resid += +0.50 P(b,c) <l,a||d,e> t1(b,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bacijk->abcijk', tmps_["80_vvvooo"])

    # triples_resid += +0.50 P(a,b) <l,c||d,e> t1(a,l) t3(d,e,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbijk->abcijk', tmps_["80_vvvooo"])
    triples_resid -= einsum('cbaijk->abcijk', tmps_["80_vvvooo"])
    triples_resid += einsum('bcaijk->abcijk', tmps_["80_vvvooo"])
    triples_resid += einsum('cabijk->abcijk', tmps_["80_vvvooo"])
    del tmps_["80_vvvooo"]

    # tmps_[4_oovvoo](l,i,a,c,j,k) = 1.00 eri[oovo](m,l,d,i) * t3(d,a,c,j,k,m) // flops: o4v2 = o5v3 | mem: o4v2 = o4v2
    tmps_["4_oovvoo"] = einsum('mldi,dacjkm->liacjk', eri["oovo"], t3)

    # tmps_[78_vovvoo](c,k,a,b,i,j) = 1.00 t1(c,l) * eri[oovo](m,l,d,k) * t3(d,a,b,i,j,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["78_vovvoo"] = einsum('cl,lkabij->ckabij', t1, tmps_["4_oovvoo"])
    del tmps_["4_oovvoo"]
    triples_resid += einsum('cjabik->abcijk', tmps_["78_vovvoo"])
    triples_resid -= einsum('bjacik->abcijk', tmps_["78_vovvoo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,k> t1(a,l) t3(d,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('akbcij->abcijk', tmps_["78_vovvoo"])
    triples_resid += einsum('ajbcik->abcijk', tmps_["78_vovvoo"])
    triples_resid += einsum('bkacij->abcijk', tmps_["78_vovvoo"])

    # triples_resid += -1.00 P(a,b) <m,l||d,i> t1(a,l) t3(d,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aibcjk->abcijk', tmps_["78_vovvoo"])

    # triples_resid += -1.00 P(j,k) <m,l||d,k> t1(c,l) t3(d,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ckabij->abcijk', tmps_["78_vovvoo"])

    # triples_resid += -1.00 <m,l||d,i> t1(c,l) t3(d,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ciabjk->abcijk', tmps_["78_vovvoo"])
    triples_resid += einsum('biacjk->abcijk', tmps_["78_vovvoo"])
    del tmps_["78_vovvoo"]

    # tmps_[5_oovooo](m,l,a,i,j,k) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,a,i,j,k) // flops: o5v1 = o5v3 | mem: o5v1 = o5v1
    tmps_["5_oovooo"] = 0.50 * einsum('mlde,deaijk->mlaijk', eri["oovv"], t3)

    # tmps_[53_vooovv](a,i,j,k,b,c) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,a,i,j,k) * t2(b,c,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["53_vooovv"] = 0.50 * einsum('mlaijk,bcml->aijkbc', tmps_["5_oovooo"], t2)

    # triples_resid += +0.25 P(b,c) <m,l||d,e> t2(a,b,m,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cijkab->abcijk', tmps_["53_vooovv"])

    # triples_resid += +0.25 <m,l||d,e> t2(b,c,m,l) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aijkbc->abcijk', tmps_["53_vooovv"])
    triples_resid -= einsum('bijkac->abcijk', tmps_["53_vooovv"])
    del tmps_["53_vooovv"]

    # tmps_[82_vvooov](a,c,i,j,k,b) = 1.00 t1(a,l) * eri[oovv](m,l,d,e) * t3(d,e,c,i,j,k) * t1(b,m) // flops: o3v3 = o5v2 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["82_vvooov"] = einsum('al,mlcijk,bm->acijkb', t1, tmps_["5_oovooo"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["5_oovooo"]
    triples_resid += einsum('abijkc->abcijk', tmps_["82_vvooov"])

    # triples_resid += -0.50 <m,l||d,e> t1(b,l) t1(c,m) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('baijkc->abcijk', tmps_["82_vvooov"])

    # triples_resid += -0.50 P(b,c) <m,l||d,e> t1(a,l) t1(b,m) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acijkb->abcijk', tmps_["82_vvooov"])
    del tmps_["82_vvooov"]

    # tmps_[6_oovvvo](j,k,a,b,c,i) = 0.50 eri[oooo](m,l,j,k) * t3(a,b,c,i,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["6_oovvvo"] = 0.50 * einsum('mljk,abciml->jkabci', eri["oooo"], t3)

    # triples_resid += +0.50 P(i,j) <m,l||j,k> t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('jkabci->abcijk', tmps_["6_oovvvo"])
    triples_resid -= einsum('ikabcj->abcijk', tmps_["6_oovvvo"])

    # triples_resid += +0.50 <m,l||i,j> t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ijabck->abcijk', tmps_["6_oovvvo"])
    del tmps_["6_oovvvo"]

    # tmps_[7_vvvo](c,a,b,i) = 0.50 eri[oovv](l,k,c,d) * t3(d,a,b,i,l,k) // flops: o1v3 = o3v4 | mem: o1v3 = o1v3
    tmps_["7_vvvo"] = 0.50 * einsum('lkcd,dabilk->cabi', eri["oovv"], t3)

    # tmps_[57_vvovoo](a,c,j,b,i,k) = 1.00 eri[oovv](m,l,d,e) * t3(e,a,c,j,m,l) * t2(d,b,i,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["57_vvovoo"] = einsum('dacj,dbik->acjbik', tmps_["7_vvvo"], t2)
    triples_resid -= einsum('acjbik->abcijk', tmps_["57_vvovoo"])

    # triples_resid += -0.50 P(i,j) P(a,b) <m,l||d,e> t2(d,a,j,k) t3(e,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bciajk->abcijk', tmps_["57_vvovoo"])
    triples_resid += einsum('ackbij->abcijk', tmps_["57_vvovoo"])
    triples_resid += einsum('acibjk->abcijk', tmps_["57_vvovoo"])
    triples_resid += einsum('bcjaik->abcijk', tmps_["57_vvovoo"])

    # triples_resid += -0.50 P(a,b) <m,l||d,e> t2(d,a,i,j) t3(e,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bckaij->abcijk', tmps_["57_vvovoo"])

    # triples_resid += -0.50 P(i,j) <m,l||d,e> t2(d,c,j,k) t3(e,a,b,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["57_vvovoo"])
    triples_resid += einsum('abjcik->abcijk', tmps_["57_vvovoo"])

    # triples_resid += -0.50 <m,l||d,e> t2(d,c,i,j) t3(e,a,b,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abkcij->abcijk', tmps_["57_vvovoo"])
    del tmps_["57_vvovoo"]

    # tmps_[93_ovvo](j,a,b,i) = 1.00 t1(c,j) * eri[oovv](l,k,c,d) * t3(d,a,b,i,l,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["93_ovvo"] = einsum('cj,cabi->jabi', t1, tmps_["7_vvvo"])
    del tmps_["7_vvvo"]

    # doubles_resid += -0.50 P(i,j) <l,k||c,d> t1(c,j) t3(d,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('jabi->abij', tmps_["93_ovvo"])
    doubles_resid += einsum('iabj->abij', tmps_["93_ovvo"])
    del tmps_["93_ovvo"]

    # tmps_[8_vvovoo](b,c,j,a,i,k) = 1.00 eri[vvvo](b,c,d,j) * t2(d,a,i,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["8_vvovoo"] = einsum('bcdj,daik->bcjaik', eri["vvvo"], t2)
    triples_resid += einsum('bcjaik->abcijk', tmps_["8_vvovoo"])

    # triples_resid += -1.00 P(j,k) P(b,c) <a,b||d,k> t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abkcij->abcijk', tmps_["8_vvovoo"])

    # triples_resid += -1.00 <b,c||d,i> t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bciajk->abcijk', tmps_["8_vvovoo"])
    triples_resid -= einsum('acjbik->abcijk', tmps_["8_vvovoo"])

    # triples_resid += -1.00 P(b,c) <a,b||d,i> t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["8_vvovoo"])
    triples_resid += einsum('acibjk->abcijk', tmps_["8_vvovoo"])
    triples_resid += einsum('abjcik->abcijk', tmps_["8_vvovoo"])
    triples_resid += einsum('ackbij->abcijk', tmps_["8_vvovoo"])

    # triples_resid += -1.00 P(j,k) <b,c||d,k> t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bckaij->abcijk', tmps_["8_vvovoo"])
    del tmps_["8_vvovoo"]

    # tmps_[9_vvvooo](c,a,b,i,j,k) = 1.00 f[vv](c,d) * t3(d,a,b,i,j,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["9_vvvooo"] = einsum('cd,dabijk->cabijk', f["vv"], t3)
    triples_resid -= einsum('bacijk->abcijk', tmps_["9_vvvooo"])

    # triples_resid += +1.00 f(c,d) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cabijk->abcijk', tmps_["9_vvvooo"])

    # triples_resid += +1.00 P(a,b) f(a,d) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["9_vvvooo"])
    del tmps_["9_vvvooo"]

    # tmps_[10_vvoo](a,b,i,j) = 0.50 eri[vovv](a,k,c,d) * t3(c,d,b,i,j,k) // flops: o2v2 = o3v4 | mem: o2v2 = o2v2
    tmps_["10_vvoo"] = 0.50 * einsum('akcd,cdbijk->abij', eri["vovv"], t3)
    doubles_resid += einsum('baij->abij', tmps_["10_vvoo"])

    # doubles_resid += +0.50 P(a,b) <k,a||c,d> t3(c,d,b,i,j,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["10_vvoo"])
    del tmps_["10_vvoo"]

    # tmps_[11_voovoo](b,l,k,c,i,j) = 1.00 eri[vovo](b,l,d,k) * t2(d,c,i,j) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
    tmps_["11_voovoo"] = einsum('bldk,dcij->blkcij', eri["vovo"], t2)

    # tmps_[66_vovoov](a,j,c,i,k,b) = 1.00 eri[vovo](a,l,d,j) * t2(d,c,i,k) * t1(b,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["66_vovoov"] = einsum('aljcik,bl->ajcikb', tmps_["11_voovoo"], t1)
    del tmps_["11_voovoo"]
    triples_resid += einsum('cjaikb->abcijk', tmps_["66_vovoov"])
    triples_resid += einsum('bjcika->abcijk', tmps_["66_vovoov"])
    triples_resid -= einsum('akbijc->abcijk', tmps_["66_vovoov"])
    triples_resid -= einsum('aibjkc->abcijk', tmps_["66_vovoov"])

    # triples_resid += -1.00 P(a,b) <l,c||d,i> t1(a,l) t2(d,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cibjka->abcijk', tmps_["66_vovoov"])
    triples_resid += einsum('biajkc->abcijk', tmps_["66_vovoov"])

    # triples_resid += -1.00 P(j,k) P(b,c) <l,a||d,k> t1(b,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('akcijb->abcijk', tmps_["66_vovoov"])

    # triples_resid += -1.00 P(b,c) <l,a||d,i> t1(b,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aicjkb->abcijk', tmps_["66_vovoov"])
    triples_resid -= einsum('ciajkb->abcijk', tmps_["66_vovoov"])
    triples_resid -= einsum('ajcikb->abcijk', tmps_["66_vovoov"])

    # triples_resid += +1.00 P(j,k) P(a,c) <l,b||d,k> t1(a,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bkcija->abcijk', tmps_["66_vovoov"])
    triples_resid -= einsum('cjbika->abcijk', tmps_["66_vovoov"])
    triples_resid += einsum('bkaijc->abcijk', tmps_["66_vovoov"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,c||d,k> t1(a,l) t2(d,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ckbija->abcijk', tmps_["66_vovoov"])
    triples_resid += einsum('ajbikc->abcijk', tmps_["66_vovoov"])
    triples_resid -= einsum('ckaijb->abcijk', tmps_["66_vovoov"])
    triples_resid -= einsum('bjaikc->abcijk', tmps_["66_vovoov"])

    # triples_resid += +1.00 P(a,c) <l,b||d,i> t1(a,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bicjka->abcijk', tmps_["66_vovoov"])
    del tmps_["66_vovoov"]

    # tmps_[12_ovoo](k,b,i,j) = 0.50 eri[oovv](l,k,c,d) * t3(c,d,b,i,j,l) // flops: o3v1 = o4v3 | mem: o3v1 = o3v1
    tmps_["12_ovoo"] = 0.50 * einsum('lkcd,cdbijl->kbij', eri["oovv"], t3)

    # tmps_[76_vvovoo](b,c,k,a,i,j) = 1.00 t2(b,c,k,l) * eri[oovv](m,l,d,e) * t3(d,e,a,i,j,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["76_vvovoo"] = einsum('bckl,laij->bckaij', t2, tmps_["12_ovoo"])
    triples_resid -= einsum('acjbik->abcijk', tmps_["76_vvovoo"])
    triples_resid += einsum('abjcik->abcijk', tmps_["76_vvovoo"])

    # triples_resid += -0.50 P(b,c) <m,l||d,e> t2(a,b,i,l) t3(d,e,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["76_vvovoo"])

    # triples_resid += -0.50 <m,l||d,e> t2(b,c,i,l) t3(d,e,a,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bciajk->abcijk', tmps_["76_vvovoo"])

    # triples_resid += -0.50 P(j,k) P(b,c) <m,l||d,e> t2(a,b,k,l) t3(d,e,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abkcij->abcijk', tmps_["76_vvovoo"])

    # triples_resid += -0.50 P(j,k) <m,l||d,e> t2(b,c,k,l) t3(d,e,a,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bckaij->abcijk', tmps_["76_vvovoo"])
    triples_resid += einsum('ackbij->abcijk', tmps_["76_vvovoo"])
    triples_resid += einsum('acibjk->abcijk', tmps_["76_vvovoo"])
    triples_resid += einsum('bcjaik->abcijk', tmps_["76_vvovoo"])
    del tmps_["76_vvovoo"]

    # tmps_[102_voov](b,i,j,a) = 1.00 eri[oovv](l,k,c,d) * t3(c,d,b,i,j,l) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["102_voov"] = einsum('kbij,ak->bija', tmps_["12_ovoo"], t1)
    del tmps_["12_ovoo"]

    # doubles_resid += -0.50 P(a,b) <l,k||c,d> t1(a,k) t3(c,d,b,i,j,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('bija->abij', tmps_["102_voov"])
    doubles_resid += einsum('aijb->abij', tmps_["102_voov"])
    del tmps_["102_voov"]

    # tmps_[13_voovvo](b,j,k,a,c,i) = 1.00 eri[vooo](b,l,j,k) * t2(a,c,i,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["13_voovvo"] = einsum('bljk,acil->bjkaci', eri["vooo"], t2)
    triples_resid -= einsum('cikabj->abcijk', tmps_["13_voovvo"])
    triples_resid -= einsum('aikbcj->abcijk', tmps_["13_voovvo"])
    triples_resid -= einsum('bjkaci->abcijk', tmps_["13_voovvo"])

    # triples_resid += -1.00 P(i,j) P(a,b) <l,a||j,k> t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ajkbci->abcijk', tmps_["13_voovvo"])

    # triples_resid += -1.00 P(i,j) <l,c||j,k> t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cjkabi->abcijk', tmps_["13_voovvo"])
    triples_resid -= einsum('bijack->abcijk', tmps_["13_voovvo"])
    triples_resid += einsum('bikacj->abcijk', tmps_["13_voovvo"])

    # triples_resid += -1.00 P(a,b) <l,a||i,j> t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aijbck->abcijk', tmps_["13_voovvo"])

    # triples_resid += -1.00 <l,c||i,j> t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cijabk->abcijk', tmps_["13_voovvo"])
    del tmps_["13_voovvo"]

    # tmps_[14_ovvooo](l,a,b,i,j,k) = 1.00 f[ov](l,d) * t3(d,a,b,i,j,k) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
    tmps_["14_ovvooo"] = einsum('ld,dabijk->labijk', f["ov"], t3)

    # tmps_[86_vvooov](a,c,i,j,k,b) = 1.00 f[ov](l,d) * t3(d,a,c,i,j,k) * t1(b,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["86_vvooov"] = einsum('lacijk,bl->acijkb', tmps_["14_ovvooo"], t1)
    del tmps_["14_ovvooo"]

    # triples_resid += -1.00 f(l,d) t1(c,l) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijkc->abcijk', tmps_["86_vvooov"])

    # triples_resid += -1.00 P(a,b) f(l,d) t1(a,l) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijka->abcijk', tmps_["86_vvooov"])
    triples_resid += einsum('acijkb->abcijk', tmps_["86_vvooov"])
    del tmps_["86_vvooov"]

    # tmps_[15_ovvvoo](i,a,b,c,j,k) = 1.00 f[oo](l,i) * t3(a,b,c,j,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["15_ovvvoo"] = einsum('li,abcjkl->iabcjk', f["oo"], t3)
    triples_resid += einsum('jabcik->abcijk', tmps_["15_ovvvoo"])

    # triples_resid += -1.00 P(j,k) f(l,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('kabcij->abcijk', tmps_["15_ovvvoo"])

    # triples_resid += -1.00 f(l,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('iabcjk->abcijk', tmps_["15_ovvvoo"])
    del tmps_["15_ovvvoo"]

    # tmps_[16_ovvo](i,a,b,j) = 0.50 eri[oovo](l,k,c,i) * t3(c,a,b,j,l,k) // flops: o2v2 = o4v3 | mem: o2v2 = o2v2
    tmps_["16_ovvo"] = 0.50 * einsum('lkci,cabjlk->iabj', eri["oovo"], t3)
    doubles_resid -= einsum('iabj->abij', tmps_["16_ovvo"])

    # doubles_resid += +0.50 P(i,j) <l,k||c,j> t3(c,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('jabi->abij', tmps_["16_ovvo"])
    del tmps_["16_ovvo"]

    # tmps_[17_ooovoo](m,l,k,a,i,j) = 1.00 eri[oovo](m,l,d,k) * t2(d,a,i,j) // flops: o5v1 = o5v2 | mem: o5v1 = o5v1
    tmps_["17_ooovoo"] = einsum('mldk,daij->mlkaij', eri["oovo"], t2)

    # tmps_[71_vovoov](b,i,c,j,k,a) = 1.00 t1(b,m) * eri[oovo](m,l,d,i) * t2(d,c,j,k) * t1(a,l) // flops: o3v3 = o5v2 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["71_vovoov"] = einsum('bm,mlicjk,al->bicjka', t1, tmps_["17_ooovoo"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["17_ooovoo"]

    # triples_resid += +1.00 P(j,k) P(b,c) <m,l||d,k> t1(a,l) t1(b,m) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bkcija->abcijk', tmps_["71_vovoov"])
    triples_resid -= einsum('cibjka->abcijk', tmps_["71_vovoov"])

    # triples_resid += +1.00 <m,l||d,i> t1(b,l) t1(c,m) t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ciajkb->abcijk', tmps_["71_vovoov"])
    triples_resid -= einsum('ckbija->abcijk', tmps_["71_vovoov"])
    triples_resid -= einsum('cjaikb->abcijk', tmps_["71_vovoov"])
    triples_resid -= einsum('bjcika->abcijk', tmps_["71_vovoov"])
    triples_resid += einsum('cjbika->abcijk', tmps_["71_vovoov"])

    # triples_resid += +1.00 P(b,c) <m,l||d,i> t1(a,l) t1(b,m) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bicjka->abcijk', tmps_["71_vovoov"])

    # triples_resid += +1.00 P(j,k) <m,l||d,k> t1(b,l) t1(c,m) t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ckaijb->abcijk', tmps_["71_vovoov"])
    del tmps_["71_vovoov"]

    # tmps_[18_vvvo](a,c,b,i) = 1.00 eri[vovv](a,k,c,d) * t2(d,b,i,k) // flops: o1v3 = o2v4 | mem: o1v3 = o1v3
    tmps_["18_vvvo"] = einsum('akcd,dbik->acbi', eri["vovv"], t2)

    # tmps_[56_voovvo](a,i,k,c,b,j) = 1.00 t2(d,a,i,k) * eri[vovv](c,l,d,e) * t2(e,b,j,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["56_voovvo"] = einsum('daik,cdbj->aikcbj', t2, tmps_["18_vvvo"])
    triples_resid -= einsum('aikbcj->abcijk', tmps_["56_voovvo"])
    triples_resid += einsum('aijbck->abcijk', tmps_["56_voovvo"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,j) t2(e,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bijack->abcijk', tmps_["56_voovvo"])

    # triples_resid += +1.00 <l,c||d,e> t2(d,a,i,j) t2(e,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aijcbk->abcijk', tmps_["56_voovvo"])
    triples_resid += einsum('bikacj->abcijk', tmps_["56_voovvo"])
    triples_resid += einsum('aikcbj->abcijk', tmps_["56_voovvo"])

    # triples_resid += +1.00 P(i,j) <l,c||d,e> t2(d,a,j,k) t2(e,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkcbi->abcijk', tmps_["56_voovvo"])
    triples_resid += einsum('ajkbci->abcijk', tmps_["56_voovvo"])

    # triples_resid += +1.00 P(i,j) P(a,b) <l,a||d,e> t2(d,b,j,k) t2(e,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bjkaci->abcijk', tmps_["56_voovvo"])
    del tmps_["56_voovvo"]

    # tmps_[90_ovvo](j,a,b,i) = 1.00 t1(c,j) * eri[vovv](a,k,c,d) * t2(d,b,i,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["90_ovvo"] = einsum('cj,acbi->jabi', t1, tmps_["18_vvvo"])
    del tmps_["18_vvvo"]
    doubles_resid -= einsum('jbai->abij', tmps_["90_ovvo"])

    # doubles_resid += -1.00 P(i,j) P(a,b) <k,a||c,d> t1(c,j) t2(d,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('jabi->abij', tmps_["90_ovvo"])
    doubles_resid -= einsum('iabj->abij', tmps_["90_ovvo"])
    doubles_resid += einsum('ibaj->abij', tmps_["90_ovvo"])
    del tmps_["90_ovvo"]

    # tmps_[19_vvvo](b,e,a,k) = 1.00 eri[vovv](b,l,d,e) * t2(d,a,k,l) // flops: o1v3 = o2v4 | mem: o1v3 = o1v3
    tmps_["19_vvvo"] = einsum('blde,dakl->beak', eri["vovv"], t2)

    # tmps_[58_voovvo](b,i,j,c,a,k) = 1.00 t2(e,b,i,j) * eri[vovv](c,l,d,e) * t2(d,a,k,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["58_voovvo"] = einsum('ebij,ceak->bijcak', t2, tmps_["19_vvvo"])
    del tmps_["19_vvvo"]
    triples_resid += einsum('cikabj->abcijk', tmps_["58_voovvo"])
    triples_resid += einsum('cjkbai->abcijk', tmps_["58_voovvo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <l,a||d,e> t2(d,b,k,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cijabk->abcijk', tmps_["58_voovvo"])

    # triples_resid += +1.00 <l,c||d,e> t2(d,a,i,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bjkcai->abcijk', tmps_["58_voovvo"])
    triples_resid += einsum('bikcaj->abcijk', tmps_["58_voovvo"])
    triples_resid += einsum('cijbak->abcijk', tmps_["58_voovvo"])

    # triples_resid += +1.00 P(j,k) <l,c||d,e> t2(d,a,k,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bijcak->abcijk', tmps_["58_voovvo"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjkabi->abcijk', tmps_["58_voovvo"])
    triples_resid -= einsum('cikbaj->abcijk', tmps_["58_voovvo"])
    del tmps_["58_voovvo"]

    # tmps_[20_ovvo](k,c,b,j) = 1.00 eri[oovv](l,k,c,d) * t2(d,b,j,l) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["20_ovvo"] = einsum('lkcd,dbjl->kcbj', eri["oovv"], t2)

    # singles_resid += +1.00 <k,j||b,c> t1(b,j) t2(c,a,i,k)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid += einsum('jbai,bj->ai', tmps_["20_ovvo"], t1)

    # tmps_[70_voovov](a,i,j,c,k,b) = 1.00 t2(d,a,i,j) * eri[oovv](m,l,d,e) * t2(e,c,k,m) * t1(b,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["70_voovov"] = einsum('daij,ldck,bl->aijckb', t2, tmps_["20_ovvo"], t1, optimize=['einsum_path',(0,1),(0,1)])
    triples_resid += einsum('aikbjc->abcijk', tmps_["70_voovov"])
    triples_resid -= einsum('aikcjb->abcijk', tmps_["70_voovov"])
    triples_resid += einsum('ajkcib->abcijk', tmps_["70_voovov"])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(c,l) t2(d,a,j,k) t2(e,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkbic->abcijk', tmps_["70_voovov"])
    triples_resid += einsum('aijckb->abcijk', tmps_["70_voovov"])

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,j,k) t2(e,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bjkcia->abcijk', tmps_["70_voovov"])

    # triples_resid += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,j) t2(e,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aijbkc->abcijk', tmps_["70_voovov"])

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,j) t2(e,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bijcka->abcijk', tmps_["70_voovov"])
    triples_resid += einsum('bikcja->abcijk', tmps_["70_voovov"])
    del tmps_["70_voovov"]

    # tmps_[89_vovo](a,i,b,j) = 1.00 t2(c,a,i,k) * eri[oovv](l,k,c,d) * t2(d,b,j,l) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["89_vovo"] = einsum('caik,kcbj->aibj', t2, tmps_["20_ovvo"])
    doubles_resid -= einsum('aibj->abij', tmps_["89_vovo"])

    # doubles_resid += +1.00 P(i,j) <l,k||c,d> t2(c,a,j,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('ajbi->abij', tmps_["89_vovo"])
    del tmps_["89_vovo"]

    # tmps_[94_vvoo](a,b,i,j) = 1.00 t1(c,j) * eri[oovv](l,k,c,d) * t2(d,b,i,l) * t1(a,k) // flops: o2v2 = o3v2 o3v2 | mem: o2v2 = o3v1 o2v2
    tmps_["94_vvoo"] = einsum('cj,kcbi,ak->abij', t1, tmps_["20_ovvo"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["20_ovvo"]
    doubles_resid -= einsum('abji->abij', tmps_["94_vvoo"])
    doubles_resid += einsum('baji->abij', tmps_["94_vvoo"])
    doubles_resid -= einsum('baij->abij', tmps_["94_vvoo"])

    # doubles_resid += +1.00 P(i,j) P(a,b) <l,k||c,d> t1(c,j) t1(a,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["94_vvoo"])
    del tmps_["94_vvoo"]

    # tmps_[21_vooo](a,k,i,j) = 0.50 eri[vovv](a,k,c,d) * t2(c,d,i,j) // flops: o3v1 = o3v3 | mem: o3v1 = o3v1
    tmps_["21_vooo"] = 0.50 * einsum('akcd,cdij->akij', eri["vovv"], t2)

    # tmps_[79_vvovoo](a,b,j,c,i,k) = 1.00 t2(a,b,j,l) * eri[vovv](c,l,d,e) * t2(d,e,i,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["79_vvovoo"] = einsum('abjl,clik->abjcik', t2, tmps_["21_vooo"])
    triples_resid -= einsum('acibjk->abcijk', tmps_["79_vvovoo"])

    # triples_resid += -0.50 P(i,j) P(a,b) <l,a||d,e> t2(d,e,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bciajk->abcijk', tmps_["79_vvovoo"])

    # triples_resid += -0.50 <l,c||d,e> t2(d,e,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abkcij->abcijk', tmps_["79_vvovoo"])
    triples_resid -= einsum('bcjaik->abcijk', tmps_["79_vvovoo"])
    triples_resid += einsum('acjbik->abcijk', tmps_["79_vvovoo"])

    # triples_resid += -0.50 P(a,b) <l,a||d,e> t2(d,e,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bckaij->abcijk', tmps_["79_vvovoo"])

    # triples_resid += -0.50 P(i,j) <l,c||d,e> t2(d,e,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abicjk->abcijk', tmps_["79_vvovoo"])
    triples_resid -= einsum('ackbij->abcijk', tmps_["79_vvovoo"])
    triples_resid -= einsum('abjcik->abcijk', tmps_["79_vvovoo"])
    del tmps_["79_vvovoo"]

    # tmps_[105_voov](a,i,j,b) = 1.00 eri[vovv](a,k,c,d) * t2(c,d,i,j) * t1(b,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["105_voov"] = einsum('akij,bk->aijb', tmps_["21_vooo"], t1)
    del tmps_["21_vooo"]
    doubles_resid += einsum('bija->abij', tmps_["105_voov"])

    # doubles_resid += +0.50 P(a,b) <k,a||c,d> t1(b,k) t2(c,d,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('aijb->abij', tmps_["105_voov"])
    del tmps_["105_voov"]

    # tmps_[22_vovv](c,i,a,b) = 0.50 eri[oovo](l,k,c,i) * t2(a,b,l,k) // flops: o1v3 = o3v3 | mem: o1v3 = o1v3
    tmps_["22_vovv"] = 0.50 * einsum('lkci,ablk->ciab', eri["oovo"], t2)

    # tmps_[59_vooovv](a,j,k,i,b,c) = 1.00 t2(d,a,j,k) * eri[oovo](m,l,d,i) * t2(b,c,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["59_vooovv"] = einsum('dajk,dibc->ajkibc', t2, tmps_["22_vovv"])
    triples_resid += einsum('aikjbc->abcijk', tmps_["59_vooovv"])

    # triples_resid += -0.50 P(a,b) <m,l||d,i> t2(d,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkibc->abcijk', tmps_["59_vooovv"])

    # triples_resid += -0.50 <m,l||d,i> t2(d,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjkiab->abcijk', tmps_["59_vooovv"])
    triples_resid += einsum('bjkiac->abcijk', tmps_["59_vooovv"])
    triples_resid += einsum('bijkac->abcijk', tmps_["59_vooovv"])

    # triples_resid += -0.50 P(j,k) <m,l||d,k> t2(d,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cijkab->abcijk', tmps_["59_vooovv"])
    triples_resid -= einsum('bikjac->abcijk', tmps_["59_vooovv"])
    triples_resid += einsum('cikjab->abcijk', tmps_["59_vooovv"])

    # triples_resid += -0.50 P(j,k) P(a,b) <m,l||d,k> t2(d,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aijkbc->abcijk', tmps_["59_vooovv"])
    del tmps_["59_vooovv"]

    # tmps_[91_oovv](i,j,a,b) = 1.00 t1(c,i) * eri[oovo](l,k,c,j) * t2(a,b,l,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["91_oovv"] = einsum('ci,cjab->ijab', t1, tmps_["22_vovv"])
    del tmps_["22_vovv"]
    doubles_resid -= einsum('jiab->abij', tmps_["91_oovv"])

    # doubles_resid += +0.50 P(i,j) <l,k||c,j> t1(c,i) t2(a,b,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('ijab->abij', tmps_["91_oovv"])
    del tmps_["91_oovv"]

    # tmps_[23_ovvo](l,e,a,i) = 1.00 eri[oovv](m,l,d,e) * t2(d,a,i,m) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["23_ovvo"] = einsum('mlde,daim->leai', eri["oovv"], t2)

    # tmps_[69_vovoov](a,k,c,i,j,b) = 1.00 eri[oovv](m,l,d,e) * t2(d,a,k,m) * t2(e,c,i,j) * t1(b,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["69_vovoov"] = einsum('leak,ecij,bl->akcijb', tmps_["23_ovvo"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["23_ovvo"]

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(c,l) t2(d,a,k,m) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('akbijc->abcijk', tmps_["69_vovoov"])
    triples_resid += einsum('bjcika->abcijk', tmps_["69_vovoov"])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,k,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bkcija->abcijk', tmps_["69_vovoov"])
    triples_resid -= einsum('ajcikb->abcijk', tmps_["69_vovoov"])
    triples_resid += einsum('ajbikc->abcijk', tmps_["69_vovoov"])
    triples_resid += einsum('aicjkb->abcijk', tmps_["69_vovoov"])

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bicjka->abcijk', tmps_["69_vovoov"])

    # triples_resid += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,m) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aibjkc->abcijk', tmps_["69_vovoov"])
    triples_resid += einsum('akcijb->abcijk', tmps_["69_vovoov"])
    del tmps_["69_vovoov"]

    # tmps_[24_ovvo](l,d,a,i) = 1.00 eri[oovv](l,k,c,d) * t2(c,a,i,k) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["24_ovvo"] = einsum('lkcd,caik->ldai', eri["oovv"], t2)

    # tmps_[49_vvoovo](a,c,j,k,b,i) = 1.00 t3(e,a,c,j,k,m) * eri[oovv](m,l,d,e) * t2(d,b,i,l) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
    tmps_["49_vvoovo"] = einsum('eacjkm,mebi->acjkbi', t3, tmps_["24_ovvo"])
    del tmps_["24_ovvo"]
    triples_resid -= einsum('acijbk->abcijk', tmps_["49_vvoovo"])
    triples_resid -= einsum('abikcj->abcijk', tmps_["49_vvoovo"])
    triples_resid -= einsum('acjkbi->abcijk', tmps_["49_vvoovo"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t2(d,a,i,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcjkai->abcijk', tmps_["49_vvoovo"])
    triples_resid += einsum('acikbj->abcijk', tmps_["49_vvoovo"])
    triples_resid -= einsum('bcikaj->abcijk', tmps_["49_vvoovo"])

    # triples_resid += +1.00 <m,l||d,e> t2(d,c,i,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abjkci->abcijk', tmps_["49_vvoovo"])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t2(d,c,k,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijck->abcijk', tmps_["49_vvoovo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,e> t2(d,a,k,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijak->abcijk', tmps_["49_vvoovo"])
    del tmps_["49_vvoovo"]

    # tmps_[25_vovo](a,i,b,j) = 1.00 eri[vovo](a,k,c,i) * t2(c,b,j,k) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["25_vovo"] = einsum('akci,cbjk->aibj', eri["vovo"], t2)
    doubles_resid += einsum('bjai->abij', tmps_["25_vovo"])

    # doubles_resid += +1.00 P(i,j) P(a,b) <k,a||c,j> t2(c,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('ajbi->abij', tmps_["25_vovo"])
    doubles_resid += einsum('aibj->abij', tmps_["25_vovo"])
    doubles_resid -= einsum('biaj->abij', tmps_["25_vovo"])
    del tmps_["25_vovo"]

    # tmps_[26_oovo](m,k,b,i) = 1.00 eri[oovo](m,l,d,k) * t2(d,b,i,l) // flops: o3v1 = o4v2 | mem: o3v1 = o3v1
    tmps_["26_oovo"] = einsum('mldk,dbil->mkbi', eri["oovo"], t2)

    # tmps_[68_vvoovo](a,c,k,i,b,j) = 1.00 t2(a,c,k,m) * eri[oovo](m,l,d,i) * t2(d,b,j,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["68_vvoovo"] = einsum('ackm,mibj->ackibj', t2, tmps_["26_oovo"])
    del tmps_["26_oovo"]
    triples_resid += einsum('acijbk->abcijk', tmps_["68_vvoovo"])
    triples_resid -= einsum('acikbj->abcijk', tmps_["68_vvoovo"])
    triples_resid += einsum('ackibj->abcijk', tmps_["68_vvoovo"])
    triples_resid -= einsum('acjibk->abcijk', tmps_["68_vvoovo"])
    triples_resid -= einsum('bcjkai->abcijk', tmps_["68_vvoovo"])

    # triples_resid += -1.00 P(i,k) <m,l||d,j> t2(d,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijck->abcijk', tmps_["68_vvoovo"])
    triples_resid -= einsum('abjkci->abcijk', tmps_["68_vvoovo"])
    triples_resid += einsum('bckjai->abcijk', tmps_["68_vvoovo"])
    triples_resid -= einsum('ackjbi->abcijk', tmps_["68_vvoovo"])

    # triples_resid += +1.00 P(j,k) <m,l||d,i> t2(d,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abjick->abcijk', tmps_["68_vvoovo"])
    triples_resid -= einsum('abkicj->abcijk', tmps_["68_vvoovo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,i> t2(d,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcjiak->abcijk', tmps_["68_vvoovo"])
    triples_resid -= einsum('bckiaj->abcijk', tmps_["68_vvoovo"])
    triples_resid += einsum('abkjci->abcijk', tmps_["68_vvoovo"])

    # triples_resid += -1.00 P(i,k) P(a,b) <m,l||d,j> t2(d,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijak->abcijk', tmps_["68_vvoovo"])

    # triples_resid += +1.00 P(i,j) <m,l||d,k> t2(d,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abikcj->abcijk', tmps_["68_vvoovo"])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,k> t2(d,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcikaj->abcijk', tmps_["68_vvoovo"])
    triples_resid += einsum('acjkbi->abcijk', tmps_["68_vvoovo"])
    del tmps_["68_vvoovo"]

    # tmps_[27_oooo](l,k,i,j) = 0.50 eri[oovv](l,k,c,d) * t2(c,d,i,j) // flops: o4v0 = o4v2 | mem: o4v0 = o4v0
    tmps_["27_oooo"] = 0.50 * einsum('lkcd,cdij->lkij', eri["oovv"], t2)

    # doubles_resid += +0.25 <l,k||c,d> t2(c,d,i,j) t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
    doubles_resid += 0.50 * einsum('ablk,lkij->abij', t2, tmps_["27_oooo"])

    # doubles_resid += -0.50 <l,k||c,d> t1(a,k) t1(b,l) t2(c,d,i,j)  // flops: o2v2 += o4v1 o3v2 | mem: o2v2 += o3v1 o2v2
    doubles_resid -= einsum('lkij,bl,ak->abij', tmps_["27_oooo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])

    # tmps_[54_vvvooo](a,b,c,j,i,k) = 0.50 t3(a,b,c,j,m,l) * eri[oovv](m,l,d,e) * t2(d,e,i,k) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["54_vvvooo"] = 0.50 * einsum('abcjml,mlik->abcjik', t3, tmps_["27_oooo"])

    # triples_resid += +0.25 <m,l||d,e> t2(d,e,i,j) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abckij->abcijk', tmps_["54_vvvooo"])
    triples_resid -= einsum('abcjik->abcijk', tmps_["54_vvvooo"])

    # triples_resid += +0.25 P(i,j) <m,l||d,e> t2(d,e,j,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["54_vvvooo"])
    del tmps_["54_vvvooo"]

    # tmps_[74_vvooov](b,c,i,j,k,a) = 1.00 t1(a,l) * eri[oovv](m,l,d,e) * t2(d,e,j,k) * t2(b,c,i,m) // flops: o3v3 = o4v1 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["74_vvooov"] = einsum('al,mljk,bcim->bcijka', t1, tmps_["27_oooo"], t2, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["27_oooo"]

    # triples_resid += +0.50 P(i,j) <m,l||d,e> t1(c,l) t2(d,e,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijkc->abcijk', tmps_["74_vvooov"])

    # triples_resid += +0.50 P(a,b) <m,l||d,e> t1(a,l) t2(d,e,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bckija->abcijk', tmps_["74_vvooov"])
    triples_resid -= einsum('ackijb->abcijk', tmps_["74_vvooov"])

    # triples_resid += +0.50 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,e,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijka->abcijk', tmps_["74_vvooov"])
    triples_resid -= einsum('bcjika->abcijk', tmps_["74_vvooov"])
    triples_resid -= einsum('abjikc->abcijk', tmps_["74_vvooov"])
    triples_resid += einsum('acjikb->abcijk', tmps_["74_vvooov"])
    triples_resid -= einsum('acijkb->abcijk', tmps_["74_vvooov"])

    # triples_resid += +0.50 <m,l||d,e> t1(c,l) t2(d,e,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abkijc->abcijk', tmps_["74_vvooov"])
    del tmps_["74_vvooov"]

    # tmps_[28_oovo](k,j,a,i) = 1.00 eri[oovo](l,k,c,j) * t2(c,a,i,l) // flops: o3v1 = o4v2 | mem: o3v1 = o3v1
    tmps_["28_oovo"] = einsum('lkcj,cail->kjai', eri["oovo"], t2)

    # tmps_[95_vovo](a,i,b,j) = 1.00 t1(a,k) * eri[oovo](l,k,c,i) * t2(c,b,j,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["95_vovo"] = einsum('ak,kibj->aibj', t1, tmps_["28_oovo"])
    del tmps_["28_oovo"]
    doubles_resid += einsum('aibj->abij', tmps_["95_vovo"])
    doubles_resid -= einsum('biaj->abij', tmps_["95_vovo"])
    doubles_resid += einsum('bjai->abij', tmps_["95_vovo"])

    # doubles_resid += -1.00 P(i,j) P(a,b) <l,k||c,j> t1(a,k) t2(c,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('ajbi->abij', tmps_["95_vovo"])
    del tmps_["95_vovo"]

    # tmps_[29_vvvo](a,b,d,j) = 1.00 eri[vvvv](a,b,c,d) * t1(c,j) // flops: o1v3 = o1v4 | mem: o1v3 = o1v3
    tmps_["29_vvvo"] = einsum('abcd,cj->abdj', eri["vvvv"], t1)

    # doubles_resid += -1.00 <a,b||c,d> t1(c,j) t1(d,i)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= einsum('di,abdj->abij', t1, tmps_["29_vvvo"])

    # tmps_[60_vvovoo](a,b,k,c,i,j) = 1.00 eri[vvvv](a,b,d,e) * t1(d,k) * t2(e,c,i,j) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["60_vvovoo"] = einsum('abek,ecij->abkcij', tmps_["29_vvvo"], t2)
    del tmps_["29_vvvo"]

    # triples_resid += +1.00 P(j,k) P(b,c) <a,b||d,e> t1(d,k) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abkcij->abcijk', tmps_["60_vvovoo"])

    # triples_resid += +1.00 <b,c||d,e> t1(d,i) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bciajk->abcijk', tmps_["60_vvovoo"])
    triples_resid -= einsum('acibjk->abcijk', tmps_["60_vvovoo"])
    triples_resid -= einsum('ackbij->abcijk', tmps_["60_vvovoo"])
    triples_resid -= einsum('abjcik->abcijk', tmps_["60_vvovoo"])
    triples_resid += einsum('acjbik->abcijk', tmps_["60_vvovoo"])
    triples_resid -= einsum('bcjaik->abcijk', tmps_["60_vvovoo"])

    # triples_resid += +1.00 P(j,k) <b,c||d,e> t1(d,k) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bckaij->abcijk', tmps_["60_vvovoo"])

    # triples_resid += +1.00 P(b,c) <a,b||d,e> t1(d,i) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abicjk->abcijk', tmps_["60_vvovoo"])
    del tmps_["60_vvovoo"]

    # tmps_[30_vovo](a,k,d,j) = 1.00 eri[vovv](a,k,c,d) * t1(c,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["30_vovo"] = einsum('akcd,cj->akdj', eri["vovv"], t1)

    # tmps_[48_vovvoo](a,j,b,c,i,k) = 1.00 eri[vovv](a,l,d,e) * t1(d,j) * t3(e,b,c,i,k,l) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
    tmps_["48_vovvoo"] = einsum('alej,ebcikl->ajbcik', tmps_["30_vovo"], t3)

    # triples_resid += -1.00 P(a,b) <l,a||d,e> t1(d,i) t3(e,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aibcjk->abcijk', tmps_["48_vovvoo"])
    triples_resid -= einsum('ajbcik->abcijk', tmps_["48_vovvoo"])
    triples_resid -= einsum('biacjk->abcijk', tmps_["48_vovvoo"])
    triples_resid += einsum('bjacik->abcijk', tmps_["48_vovvoo"])
    triples_resid -= einsum('cjabik->abcijk', tmps_["48_vovvoo"])
    triples_resid -= einsum('bkacij->abcijk', tmps_["48_vovvoo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,a||d,e> t1(d,k) t3(e,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('akbcij->abcijk', tmps_["48_vovvoo"])

    # triples_resid += -1.00 <l,c||d,e> t1(d,i) t3(e,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ciabjk->abcijk', tmps_["48_vovvoo"])

    # triples_resid += -1.00 P(j,k) <l,c||d,e> t1(d,k) t3(e,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ckabij->abcijk', tmps_["48_vovvoo"])
    del tmps_["48_vovvoo"]

    # tmps_[63_vovoov](a,k,c,i,j,b) = 1.00 eri[vovv](a,l,d,e) * t1(d,k) * t2(e,c,i,j) * t1(b,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["63_vovoov"] = einsum('alek,ecij,bl->akcijb', tmps_["30_vovo"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])
    triples_resid -= einsum('biajkc->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('akbijc->abcijk', tmps_["63_vovoov"])

    # triples_resid += +1.00 P(j,k) P(b,c) <l,a||d,e> t1(d,k) t1(b,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('akcijb->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('cjbika->abcijk', tmps_["63_vovoov"])

    # triples_resid += -1.00 P(a,c) <l,b||d,e> t1(d,i) t1(a,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bicjka->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('bjaikc->abcijk', tmps_["63_vovoov"])

    # triples_resid += +1.00 P(j,k) P(a,b) <l,c||d,e> t1(d,k) t1(a,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ckbija->abcijk', tmps_["63_vovoov"])

    # triples_resid += +1.00 P(a,b) <l,c||d,e> t1(d,i) t1(a,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cibjka->abcijk', tmps_["63_vovoov"])
    triples_resid -= einsum('bkaijc->abcijk', tmps_["63_vovoov"])

    # triples_resid += -1.00 P(j,k) P(a,c) <l,b||d,e> t1(d,k) t1(a,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bkcija->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('ciajkb->abcijk', tmps_["63_vovoov"])

    # triples_resid += +1.00 P(b,c) <l,a||d,e> t1(d,i) t1(b,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aicjkb->abcijk', tmps_["63_vovoov"])
    triples_resid -= einsum('ajbikc->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('ajcikb->abcijk', tmps_["63_vovoov"])
    triples_resid -= einsum('cjaikb->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('ckaijb->abcijk', tmps_["63_vovoov"])
    triples_resid += einsum('aibjkc->abcijk', tmps_["63_vovoov"])
    triples_resid -= einsum('bjcika->abcijk', tmps_["63_vovoov"])
    del tmps_["63_vovoov"]

    # tmps_[72_ovovvo](i,b,j,a,c,k) = 1.00 t1(e,i) * eri[vovv](b,l,d,e) * t1(d,j) * t2(a,c,k,l) // flops: o3v3 = o3v2 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["72_ovovvo"] = einsum('ei,blej,ackl->ibjack', t1, tmps_["30_vovo"], t2, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 P(i,j) P(a,b) <l,a||d,e> t1(d,k) t1(e,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('jakbci->abcijk', tmps_["72_ovovvo"])

    # triples_resid += +1.00 P(i,j) <l,c||d,e> t1(d,k) t1(e,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('jckabi->abcijk', tmps_["72_ovovvo"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t1(d,j) t1(e,i) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('iajbck->abcijk', tmps_["72_ovovvo"])
    triples_resid += einsum('jbkaci->abcijk', tmps_["72_ovovvo"])
    triples_resid += einsum('iakbcj->abcijk', tmps_["72_ovovvo"])
    triples_resid += einsum('ickabj->abcijk', tmps_["72_ovovvo"])
    triples_resid += einsum('ibjack->abcijk', tmps_["72_ovovvo"])
    triples_resid -= einsum('ibkacj->abcijk', tmps_["72_ovovvo"])

    # triples_resid += +1.00 <l,c||d,e> t1(d,j) t1(e,i) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('icjabk->abcijk', tmps_["72_ovovvo"])
    del tmps_["72_ovovvo"]

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

    # tmps_[61_vvvooo](c,a,b,i,j,k) = 1.00 eri[oovv](m,l,d,e) * t2(d,c,m,l) * t3(e,a,b,i,j,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["61_vvvooo"] = einsum('ec,eabijk->cabijk', tmps_["31_vv"], t3)
    del tmps_["31_vv"]

    # triples_resid += -0.50 <m,l||d,e> t2(d,c,m,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabijk->abcijk', tmps_["61_vvvooo"])
    triples_resid += einsum('bacijk->abcijk', tmps_["61_vvvooo"])

    # triples_resid += -0.50 P(a,b) <m,l||d,e> t2(d,a,m,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcijk->abcijk', tmps_["61_vvvooo"])
    del tmps_["61_vvvooo"]

    # tmps_[32_vvoo](b,a,i,j) = 1.00 f[vv](b,c) * t2(c,a,i,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["32_vvoo"] = einsum('bc,caij->baij', f["vv"], t2)
    doubles_resid -= einsum('baij->abij', tmps_["32_vvoo"])

    # doubles_resid += +1.00 P(a,b) f(a,c) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["32_vvoo"])
    del tmps_["32_vvoo"]

    # tmps_[33_vvoo](a,b,i,j) = 1.00 eri[vvvo](a,b,c,i) * t1(c,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["33_vvoo"] = einsum('abci,cj->abij', eri["vvvo"], t1)

    # doubles_resid += +1.00 P(i,j) <a,b||c,j> t1(c,i)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abji->abij', tmps_["33_vvoo"])
    doubles_resid -= einsum('abij->abij', tmps_["33_vvoo"])
    del tmps_["33_vvoo"]

    # tmps_[34_oovo](l,k,d,j) = 1.00 eri[oovv](l,k,c,d) * t1(c,j) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["34_oovo"] = einsum('lkcd,cj->lkdj', eri["oovv"], t1)

    # singles_resid += +0.50 <k,j||b,c> t1(b,i) t2(c,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
    singles_resid += 0.50 * einsum('cakj,kjci->ai', t2, tmps_["34_oovo"])

    # doubles_resid += -0.50 <l,k||c,d> t1(c,j) t1(d,i) t2(a,b,l,k)  // flops: o2v2 += o4v1 o4v2 | mem: o2v2 += o4v0 o2v2
    doubles_resid -= 0.50 * einsum('lkdj,di,ablk->abij', tmps_["34_oovo"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])

    # tmps_[50_vvooov](a,b,i,k,j,c) = 1.00 t3(e,a,b,i,k,m) * eri[oovv](m,l,d,e) * t1(d,j) * t1(c,l) // flops: o3v3 = o5v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["50_vvooov"] = einsum('eabikm,mlej,cl->abikjc', t3, tmps_["34_oovo"], t1, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t1(a,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijka->abcijk', tmps_["50_vvooov"])
    triples_resid += einsum('acikjb->abcijk', tmps_["50_vvooov"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,i) t1(a,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcjkia->abcijk', tmps_["50_vvooov"])
    triples_resid -= einsum('acijkb->abcijk', tmps_["50_vvooov"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,i) t1(c,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abjkic->abcijk', tmps_["50_vvooov"])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t1(d,k) t1(c,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijkc->abcijk', tmps_["50_vvooov"])
    triples_resid -= einsum('bcikja->abcijk', tmps_["50_vvooov"])
    triples_resid -= einsum('acjkib->abcijk', tmps_["50_vvooov"])
    triples_resid -= einsum('abikjc->abcijk', tmps_["50_vvooov"])
    del tmps_["50_vvooov"]

    # tmps_[52_vvvooo](a,b,c,i,k,j) = 0.50 eri[oovv](m,l,d,e) * t1(d,k) * t1(e,j) * t3(a,b,c,i,m,l) // flops: o3v3 = o4v1 o5v3 | mem: o3v3 = o4v0 o3v3
    tmps_["52_vvvooo"] = 0.50 * einsum('mlek,ej,abciml->abcikj', tmps_["34_oovo"], t1, t3, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += -0.50 <m,l||d,e> t1(d,j) t1(e,i) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abckji->abcijk', tmps_["52_vvvooo"])

    # triples_resid += -0.50 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcikj->abcijk', tmps_["52_vvvooo"])
    triples_resid += einsum('abcjki->abcijk', tmps_["52_vvvooo"])
    del tmps_["52_vvvooo"]

    # tmps_[55_vvovoo](a,c,i,b,j,k) = 0.50 t2(a,c,m,l) * eri[oovv](m,l,d,e) * t1(d,i) * t2(e,b,j,k) // flops: o3v3 = o3v3 o3v4 | mem: o3v3 = o1v3 o3v3
    tmps_["55_vvovoo"] = 0.50 * einsum('acml,mlei,ebjk->acibjk', t2, tmps_["34_oovo"], t2, optimize=['einsum_path',(0,1),(0,1)])
    triples_resid -= einsum('acibjk->abcijk', tmps_["55_vvovoo"])

    # triples_resid += +0.50 P(a,b) <m,l||d,e> t1(d,i) t2(e,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bciajk->abcijk', tmps_["55_vvovoo"])

    # triples_resid += +0.50 P(j,k) <m,l||d,e> t1(d,k) t2(e,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abkcij->abcijk', tmps_["55_vvovoo"])
    triples_resid -= einsum('bcjaik->abcijk', tmps_["55_vvovoo"])
    triples_resid += einsum('acjbik->abcijk', tmps_["55_vvovoo"])

    # triples_resid += +0.50 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bckaij->abcijk', tmps_["55_vvovoo"])
    triples_resid -= einsum('ackbij->abcijk', tmps_["55_vvovoo"])
    triples_resid -= einsum('abjcik->abcijk', tmps_["55_vvovoo"])

    # triples_resid += +0.50 <m,l||d,e> t1(d,i) t2(e,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abicjk->abcijk', tmps_["55_vvovoo"])
    del tmps_["55_vvovoo"]

    # tmps_[64_voovvo](a,i,j,b,c,k) = 1.00 t2(e,a,i,l) * eri[oovv](m,l,d,e) * t1(d,j) * t2(b,c,k,m) // flops: o3v3 = o4v2 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["64_voovvo"] = einsum('eail,mlej,bckm->aijbck', t2, tmps_["34_oovo"], t2, optimize=['einsum_path',(0,1),(0,1)])
    triples_resid -= einsum('bkjaci->abcijk', tmps_["64_voovvo"])
    triples_resid += einsum('cjiabk->abcijk', tmps_["64_voovvo"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(d,i) t2(e,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ckiabj->abcijk', tmps_["64_voovvo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,i) t2(e,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('akibcj->abcijk', tmps_["64_voovvo"])
    triples_resid += einsum('bjkaci->abcijk', tmps_["64_voovvo"])
    triples_resid += einsum('ajibck->abcijk', tmps_["64_voovvo"])

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkbci->abcijk', tmps_["64_voovvo"])
    triples_resid -= einsum('cijabk->abcijk', tmps_["64_voovvo"])

    # triples_resid += +1.00 P(i,k) P(a,b) <m,l||d,e> t1(d,j) t2(e,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('akjbci->abcijk', tmps_["64_voovvo"])

    # triples_resid += +1.00 P(i,k) <m,l||d,e> t1(d,j) t2(e,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ckjabi->abcijk', tmps_["64_voovvo"])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(d,k) t2(e,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjkabi->abcijk', tmps_["64_voovvo"])
    triples_resid -= einsum('bikacj->abcijk', tmps_["64_voovvo"])
    triples_resid += einsum('aikbcj->abcijk', tmps_["64_voovvo"])
    triples_resid -= einsum('aijbck->abcijk', tmps_["64_voovvo"])
    triples_resid -= einsum('bjiack->abcijk', tmps_["64_voovvo"])
    triples_resid += einsum('cikabj->abcijk', tmps_["64_voovvo"])
    triples_resid += einsum('bijack->abcijk', tmps_["64_voovvo"])
    triples_resid += einsum('bkiacj->abcijk', tmps_["64_voovvo"])
    del tmps_["64_voovvo"]

    # tmps_[88_oovoov](l,i,c,j,k,b) = 1.00 eri[oovv](m,l,d,e) * t1(d,i) * t2(e,c,j,k) * t1(b,m) // flops: o4v2 = o5v2 o5v2 | mem: o4v2 = o5v1 o4v2
    tmps_["88_oovoov"] = einsum('mlei,ecjk,bm->licjkb', tmps_["34_oovo"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

    # tmps_[108_vovoov](a,k,b,i,j,c) = 1.00 t1(a,l) * tmps_[34_oovo](m,l,e,k) * t2(e,b,i,j) * t1(c,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["108_vovoov"] = einsum('al,lkbijc->akbijc', t1, tmps_["88_oovoov"])
    del tmps_["88_oovoov"]
    triples_resid += einsum('ajcikb->abcijk', tmps_["108_vovoov"])
    triples_resid -= einsum('ajbikc->abcijk', tmps_["108_vovoov"])
    triples_resid += einsum('akbijc->abcijk', tmps_["108_vovoov"])
    triples_resid += einsum('aibjkc->abcijk', tmps_["108_vovoov"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(d,k) t1(b,l) t1(c,m) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bkaijc->abcijk', tmps_["108_vovoov"])

    # triples_resid += -1.00 <m,l||d,e> t1(d,i) t1(b,l) t1(c,m) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('biajkc->abcijk', tmps_["108_vovoov"])
    triples_resid += einsum('bjaikc->abcijk', tmps_["108_vovoov"])

    # triples_resid += -1.00 P(j,k) P(b,c) <m,l||d,e> t1(d,k) t1(a,l) t1(b,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('akcijb->abcijk', tmps_["108_vovoov"])

    # triples_resid += -1.00 P(b,c) <m,l||d,e> t1(d,i) t1(a,l) t1(b,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aicjkb->abcijk', tmps_["108_vovoov"])
    del tmps_["108_vovoov"]

    # tmps_[107_vooo](a,i,l,j) = 1.00 eri[oovv](l,k,c,d) * t1(c,j) * t1(d,i) * t1(a,k) // flops: o3v1 = o4v1 o4v1 | mem: o3v1 = o4v0 o3v1
    tmps_["107_vooo"] = einsum('lkdj,di,ak->ailj', tmps_["34_oovo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["34_oovo"]

    # doubles_resid += +1.00 <l,k||c,d> t1(c,j) t1(d,i) t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('bl,ailj->abij', t1, tmps_["107_vooo"])

    # tmps_[109_vvovoo](a,b,k,c,i,j) = 1.00 t2(a,b,k,m) * t1(c,l) * t1(e,i) * eri[oovv](m,l,d,e) * t1(d,j) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["109_vvovoo"] = einsum('abkm,cimj->abkcij', t2, tmps_["107_vooo"])
    del tmps_["107_vooo"]

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(d,j) t1(e,i) t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bckaij->abcijk', tmps_["109_vvovoo"])

    # triples_resid += -1.00 <m,l||d,e> t1(d,j) t1(e,i) t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abkcij->abcijk', tmps_["109_vvovoo"])
    triples_resid += einsum('ackbij->abcijk', tmps_["109_vvovoo"])
    triples_resid += einsum('abjcik->abcijk', tmps_["109_vvovoo"])

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t1(e,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bciajk->abcijk', tmps_["109_vvovoo"])
    triples_resid += einsum('acibjk->abcijk', tmps_["109_vvovoo"])
    triples_resid += einsum('bcjaik->abcijk', tmps_["109_vvovoo"])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["109_vvovoo"])
    triples_resid -= einsum('acjbik->abcijk', tmps_["109_vvovoo"])
    del tmps_["109_vvovoo"]

    # tmps_[35_vooo](b,k,i,j) = 1.00 eri[vovo](b,k,c,i) * t1(c,j) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["35_vooo"] = einsum('bkci,cj->bkij', eri["vovo"], t1)

    # tmps_[67_voovvo](c,k,i,a,b,j) = 1.00 eri[vovo](c,l,d,k) * t1(d,i) * t2(a,b,j,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["67_voovvo"] = einsum('clki,abjl->ckiabj', tmps_["35_vooo"], t2)

    # triples_resid += -1.00 P(j,k) P(a,b) <l,a||d,i> t1(d,k) t2(b,c,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aikbcj->abcijk', tmps_["67_voovvo"])

    # triples_resid += +1.00 P(i,k) <l,c||d,j> t1(d,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjkabi->abcijk', tmps_["67_voovvo"])

    # triples_resid += -1.00 P(i,j) P(a,b) <l,a||d,k> t1(d,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('akjbci->abcijk', tmps_["67_voovvo"])

    # triples_resid += -1.00 P(i,j) <l,c||d,k> t1(d,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ckjabi->abcijk', tmps_["67_voovvo"])
    triples_resid -= einsum('akibcj->abcijk', tmps_["67_voovvo"])
    triples_resid += einsum('cjiabk->abcijk', tmps_["67_voovvo"])
    triples_resid -= einsum('aijbck->abcijk', tmps_["67_voovvo"])
    triples_resid -= einsum('bikacj->abcijk', tmps_["67_voovvo"])
    triples_resid += einsum('bkiacj->abcijk', tmps_["67_voovvo"])
    triples_resid -= einsum('ckiabj->abcijk', tmps_["67_voovvo"])
    triples_resid += einsum('bjkaci->abcijk', tmps_["67_voovvo"])
    triples_resid += einsum('bijack->abcijk', tmps_["67_voovvo"])

    # triples_resid += +1.00 P(i,k) P(a,b) <l,a||d,j> t1(d,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkbci->abcijk', tmps_["67_voovvo"])

    # triples_resid += -1.00 P(j,k) <l,c||d,i> t1(d,k) t2(a,b,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cikabj->abcijk', tmps_["67_voovvo"])
    triples_resid -= einsum('cijabk->abcijk', tmps_["67_voovvo"])
    triples_resid += einsum('ajibck->abcijk', tmps_["67_voovvo"])
    triples_resid -= einsum('bjiack->abcijk', tmps_["67_voovvo"])
    triples_resid -= einsum('bkjaci->abcijk', tmps_["67_voovvo"])
    del tmps_["67_voovvo"]

    # tmps_[96_voov](a,j,i,b) = 1.00 eri[vovo](a,k,c,j) * t1(c,i) * t1(b,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["96_voov"] = einsum('akji,bk->ajib', tmps_["35_vooo"], t1)
    del tmps_["35_vooo"]

    # doubles_resid += +1.00 P(i,j) P(a,b) <k,a||c,j> t1(c,i) t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('ajib->abij', tmps_["96_voov"])
    doubles_resid += einsum('bjia->abij', tmps_["96_voov"])
    doubles_resid -= einsum('bija->abij', tmps_["96_voov"])
    doubles_resid += einsum('aijb->abij', tmps_["96_voov"])
    del tmps_["96_voov"]

    # tmps_[36_oovo](l,k,c,i) = 1.00 eri[oovv](l,k,c,d) * t1(d,i) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["36_oovo"] = einsum('lkcd,di->lkci', eri["oovv"], t1)

    # singles_resid += +1.00 <k,j||b,c> t1(b,j) t1(c,i) t1(a,k)  // flops: o1v1 += o3v1 o2v1 | mem: o1v1 += o2v0 o1v1
    singles_resid += einsum('kjbi,bj,ak->ai', tmps_["36_oovo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])

    # tmps_[83_ovvvoo](j,a,b,c,i,k) = 1.00 eri[oovv](m,l,d,e) * t1(e,j) * t1(d,l) * t3(a,b,c,i,k,m) // flops: o3v3 = o3v1 o4v3 | mem: o3v3 = o2v0 o3v3
    tmps_["83_ovvvoo"] = einsum('mldj,dl,abcikm->jabcik', tmps_["36_oovo"], t1, t3, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t1(d,l) t1(e,k) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kabcij->abcijk', tmps_["83_ovvvoo"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t1(e,i) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('iabcjk->abcijk', tmps_["83_ovvvoo"])
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

    # tmps_[77_vvovoo](a,c,i,b,j,k) = 1.00 t2(a,c,i,l) * f[ov](l,d) * t2(d,b,j,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["77_vvovoo"] = einsum('acil,lbjk->acibjk', t2, tmps_["37_ovoo"])

    # triples_resid += -1.00 P(a,b) f(l,d) t2(d,a,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bckaij->abcijk', tmps_["77_vvovoo"])
    triples_resid += einsum('bcjaik->abcijk', tmps_["77_vvovoo"])
    triples_resid += einsum('acibjk->abcijk', tmps_["77_vvovoo"])
    triples_resid += einsum('abjcik->abcijk', tmps_["77_vvovoo"])

    # triples_resid += -1.00 P(i,j) P(a,b) f(l,d) t2(d,a,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bciajk->abcijk', tmps_["77_vvovoo"])
    triples_resid -= einsum('acjbik->abcijk', tmps_["77_vvovoo"])

    # triples_resid += -1.00 P(i,j) f(l,d) t2(d,c,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["77_vvovoo"])
    triples_resid += einsum('ackbij->abcijk', tmps_["77_vvovoo"])

    # triples_resid += -1.00 f(l,d) t2(d,c,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abkcij->abcijk', tmps_["77_vvovoo"])
    del tmps_["77_vvovoo"]

    # tmps_[106_voov](b,i,j,a) = 1.00 f[ov](k,c) * t2(c,b,i,j) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["106_voov"] = einsum('kbij,ak->bija', tmps_["37_ovoo"], t1)
    del tmps_["37_ovoo"]

    # doubles_resid += -1.00 P(a,b) f(k,c) t1(a,k) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('bija->abij', tmps_["106_voov"])
    doubles_resid += einsum('aijb->abij', tmps_["106_voov"])
    del tmps_["106_voov"]

    # tmps_[38_oo](l,i) = 0.50 eri[oovv](l,k,c,d) * t2(c,d,i,k) // flops: o2v0 = o3v2 | mem: o2v0 = o2v0
    tmps_["38_oo"] = 0.50 * einsum('lkcd,cdik->li', eri["oovv"], t2)

    # tmps_[85_vvvooo](a,b,c,j,k,i) = 1.00 t3(a,b,c,j,k,m) * eri[oovv](m,l,d,e) * t2(d,e,i,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["85_vvvooo"] = einsum('abcjkm,mi->abcjki', t3, tmps_["38_oo"])

    # triples_resid += -0.50 P(j,k) <m,l||d,e> t2(d,e,k,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcijk->abcijk', tmps_["85_vvvooo"])
    triples_resid += einsum('abcikj->abcijk', tmps_["85_vvvooo"])

    # triples_resid += -0.50 <m,l||d,e> t2(d,e,i,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcjki->abcijk', tmps_["85_vvvooo"])
    del tmps_["85_vvvooo"]

    # tmps_[103_vvoo](a,b,j,i) = 1.00 t2(a,b,j,l) * eri[oovv](l,k,c,d) * t2(c,d,i,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["103_vvoo"] = einsum('abjl,li->abji', t2, tmps_["38_oo"])
    del tmps_["38_oo"]
    doubles_resid += einsum('abji->abij', tmps_["103_vvoo"])

    # doubles_resid += -0.50 P(i,j) <l,k||c,d> t2(c,d,j,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["103_vvoo"])
    del tmps_["103_vvoo"]

    # tmps_[39_vv](c,b) = 1.00 eri[oovv](l,k,c,d) * t2(d,b,l,k) // flops: o0v2 = o2v3 | mem: o0v2 = o0v2
    tmps_["39_vv"] = einsum('lkcd,dblk->cb', eri["oovv"], t2)

    # doubles_resid += -0.50 <l,k||c,d> t2(c,a,i,j) t2(d,b,l,k)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= 0.50 * einsum('cb,caij->abij', tmps_["39_vv"], t2)
    del tmps_["39_vv"]

    # tmps_[40_voov](a,i,j,b) = 1.00 eri[vooo](a,k,i,j) * t1(b,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["40_voov"] = einsum('akij,bk->aijb', eri["vooo"], t1)

    # doubles_resid += +1.00 P(a,b) <k,a||i,j> t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('aijb->abij', tmps_["40_voov"])
    doubles_resid += einsum('bija->abij', tmps_["40_voov"])
    del tmps_["40_voov"]

    # tmps_[41_ovvo](j,a,b,i) = 1.00 f[oo](k,j) * t2(a,b,i,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["41_ovvo"] = einsum('kj,abik->jabi', f["oo"], t2)
    doubles_resid += einsum('iabj->abij', tmps_["41_ovvo"])

    # doubles_resid += -1.00 P(i,j) f(k,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('jabi->abij', tmps_["41_ovvo"])
    del tmps_["41_ovvo"]

    # tmps_[42_oooo](l,k,i,j) = 1.00 eri[oovo](l,k,c,i) * t1(c,j) // flops: o4v0 = o4v1 | mem: o4v0 = o4v0
    tmps_["42_oooo"] = einsum('lkci,cj->lkij', eri["oovo"], t1)

    # tmps_[51_oovvvo](j,k,a,b,c,i) = 0.50 eri[oovo](m,l,d,j) * t1(d,k) * t3(a,b,c,i,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["51_oovvvo"] = 0.50 * einsum('mljk,abciml->jkabci', tmps_["42_oooo"], t3)

    # triples_resid += +0.50 P(i,j) <m,l||d,k> t1(d,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kjabci->abcijk', tmps_["51_oovvvo"])
    triples_resid += einsum('jiabck->abcijk', tmps_["51_oovvvo"])

    # triples_resid += +0.50 P(j,k) <m,l||d,i> t1(d,k) t3(a,b,c,j,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ikabcj->abcijk', tmps_["51_oovvvo"])
    triples_resid -= einsum('kiabcj->abcijk', tmps_["51_oovvvo"])
    triples_resid -= einsum('ijabck->abcijk', tmps_["51_oovvvo"])

    # triples_resid += -0.50 P(i,k) <m,l||d,j> t1(d,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('jkabci->abcijk', tmps_["51_oovvvo"])
    del tmps_["51_oovvvo"]

    # tmps_[65_vvooov](a,c,j,i,k,b) = 1.00 eri[oovo](m,l,d,i) * t1(d,k) * t1(b,l) * t2(a,c,j,m) // flops: o3v3 = o4v1 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["65_vvooov"] = einsum('mlik,bl,acjm->acjikb', tmps_["42_oooo"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])
    triples_resid -= einsum('bcjkia->abcijk', tmps_["65_vvooov"])
    triples_resid -= einsum('bckija->abcijk', tmps_["65_vvooov"])
    triples_resid -= einsum('ackjib->abcijk', tmps_["65_vvooov"])
    triples_resid -= einsum('abkijc->abcijk', tmps_["65_vvooov"])

    # triples_resid += -1.00 P(i,k) P(a,b) <m,l||d,j> t1(d,k) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijka->abcijk', tmps_["65_vvooov"])
    triples_resid -= einsum('acikjb->abcijk', tmps_["65_vvooov"])
    triples_resid -= einsum('acjikb->abcijk', tmps_["65_vvooov"])
    triples_resid += einsum('acijkb->abcijk', tmps_["65_vvooov"])
    triples_resid += einsum('abkjic->abcijk', tmps_["65_vvooov"])

    # triples_resid += -1.00 P(i,k) <m,l||d,j> t1(d,k) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijkc->abcijk', tmps_["65_vvooov"])
    triples_resid -= einsum('abjkic->abcijk', tmps_["65_vvooov"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,i> t1(d,k) t1(a,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcjika->abcijk', tmps_["65_vvooov"])

    # triples_resid += +1.00 P(j,k) <m,l||d,i> t1(d,k) t1(c,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abjikc->abcijk', tmps_["65_vvooov"])
    triples_resid += einsum('bckjia->abcijk', tmps_["65_vvooov"])
    triples_resid += einsum('ackijb->abcijk', tmps_["65_vvooov"])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,k> t1(d,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcikja->abcijk', tmps_["65_vvooov"])
    triples_resid += einsum('acjkib->abcijk', tmps_["65_vvooov"])

    # triples_resid += +1.00 P(i,j) <m,l||d,k> t1(d,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abikjc->abcijk', tmps_["65_vvooov"])
    del tmps_["65_vvooov"]

    # tmps_[99_vvoo](a,b,i,j) = 1.00 eri[oovo](l,k,c,i) * t1(c,j) * t1(b,l) * t1(a,k) // flops: o2v2 = o4v1 o3v2 | mem: o2v2 = o3v1 o2v2
    tmps_["99_vvoo"] = einsum('lkij,bl,ak->abij', tmps_["42_oooo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["42_oooo"]
    doubles_resid += einsum('abij->abij', tmps_["99_vvoo"])

    # doubles_resid += -1.00 P(i,j) <l,k||c,j> t1(c,i) t1(a,k) t1(b,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abji->abij', tmps_["99_vvoo"])
    del tmps_["99_vvoo"]

    # tmps_[43_ooov](l,i,j,a) = 1.00 eri[oooo](l,k,i,j) * t1(a,k) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["43_ooov"] = einsum('lkij,ak->lija', eri["oooo"], t1)

    # doubles_resid += -1.00 <l,k||i,j> t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('lija,bl->abij', tmps_["43_ooov"], t1)

    # tmps_[75_vvooov](a,b,j,i,k,c) = 1.00 t2(a,b,j,m) * eri[oooo](m,l,i,k) * t1(c,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["75_vvooov"] = einsum('abjm,mikc->abjikc', t2, tmps_["43_ooov"])
    del tmps_["43_ooov"]

    # triples_resid += +1.00 P(i,j) <m,l||j,k> t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijkc->abcijk', tmps_["75_vvooov"])

    # triples_resid += +1.00 P(a,b) <m,l||i,j> t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bckija->abcijk', tmps_["75_vvooov"])
    triples_resid -= einsum('abjikc->abcijk', tmps_["75_vvooov"])
    triples_resid -= einsum('acijkb->abcijk', tmps_["75_vvooov"])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||j,k> t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijka->abcijk', tmps_["75_vvooov"])
    triples_resid -= einsum('bcjika->abcijk', tmps_["75_vvooov"])
    triples_resid += einsum('acjikb->abcijk', tmps_["75_vvooov"])

    # triples_resid += +1.00 <m,l||i,j> t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abkijc->abcijk', tmps_["75_vvooov"])
    triples_resid -= einsum('ackijb->abcijk', tmps_["75_vvooov"])
    del tmps_["75_vvooov"]

    # tmps_[44_vv](b,d) = 1.00 eri[vovv](b,k,c,d) * t1(c,k) // flops: o0v2 = o1v3 | mem: o0v2 = o0v2
    tmps_["44_vv"] = einsum('bkcd,ck->bd', eri["vovv"], t1)

    # singles_resid += +1.00 <j,a||b,c> t1(b,j) t1(c,i)  // flops: o1v1 += o1v2 | mem: o1v1 += o1v1
    singles_resid -= einsum('ac,ci->ai', tmps_["44_vv"], t1)

    # tmps_[62_vvooov](a,c,i,j,k,b) = 1.00 t3(e,a,c,i,j,k) * eri[vovv](b,l,d,e) * t1(d,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["62_vvooov"] = einsum('eacijk,be->acijkb', t3, tmps_["44_vv"])
    triples_resid += einsum('acijkb->abcijk', tmps_["62_vvooov"])

    # triples_resid += +1.00 <l,c||d,e> t1(d,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijkc->abcijk', tmps_["62_vvooov"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t1(d,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijka->abcijk', tmps_["62_vvooov"])
    del tmps_["62_vvooov"]

    # tmps_[92_voov](b,i,j,a) = 1.00 t2(d,b,i,j) * eri[vovv](a,k,c,d) * t1(c,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["92_voov"] = einsum('dbij,ad->bija', t2, tmps_["44_vv"])
    del tmps_["44_vv"]

    # doubles_resid += +1.00 P(a,b) <k,a||c,d> t1(c,k) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('bija->abij', tmps_["92_voov"])
    doubles_resid += einsum('aijb->abij', tmps_["92_voov"])
    del tmps_["92_voov"]

    # tmps_[45_ov](l,d) = 1.00 eri[oovv](l,k,c,d) * t1(c,k) // flops: o1v1 = o2v2 | mem: o1v1 = o1v1
    tmps_["45_ov"] = einsum('lkcd,ck->ld', eri["oovv"], t1)

    # doubles_resid += -1.00 <l,k||c,d> t1(c,k) t3(d,a,b,i,j,l)  // flops: o2v2 += o3v3 | mem: o2v2 += o2v2
    doubles_resid -= einsum('ld,dabijl->abij', tmps_["45_ov"], t3)

    # tmps_[73_voovvo](a,i,k,b,c,j) = 1.00 t2(e,a,i,k) * eri[oovv](m,l,d,e) * t1(d,l) * t2(b,c,j,m) // flops: o3v3 = o3v2 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["73_voovvo"] = einsum('eaik,me,bcjm->aikbcj', t2, tmps_["45_ov"], t2, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,l) t2(e,a,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ajkbci->abcijk', tmps_["73_voovvo"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,l) t2(e,a,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aijbck->abcijk', tmps_["73_voovvo"])
    triples_resid -= einsum('aikbcj->abcijk', tmps_["73_voovvo"])

    # triples_resid += +1.00 P(i,j) <m,l||d,e> t1(d,l) t2(e,c,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cjkabi->abcijk', tmps_["73_voovvo"])
    triples_resid -= einsum('cikabj->abcijk', tmps_["73_voovvo"])
    triples_resid += einsum('bikacj->abcijk', tmps_["73_voovvo"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t2(e,c,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cijabk->abcijk', tmps_["73_voovvo"])
    triples_resid -= einsum('bjkaci->abcijk', tmps_["73_voovvo"])
    triples_resid -= einsum('bijack->abcijk', tmps_["73_voovvo"])
    del tmps_["73_voovvo"]

    # tmps_[81_vvooov](a,c,i,j,k,b) = 1.00 eri[oovv](m,l,d,e) * t1(d,l) * t3(e,a,c,i,j,k) * t1(b,m) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["81_vvooov"] = einsum('me,eacijk,bm->acijkb', tmps_["45_ov"], t3, t1, optimize=['einsum_path',(0,1),(0,1)])
    triples_resid -= einsum('acijkb->abcijk', tmps_["81_vvooov"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t1(c,m) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijkc->abcijk', tmps_["81_vvooov"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,l) t1(a,m) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijka->abcijk', tmps_["81_vvooov"])
    del tmps_["81_vvooov"]

    # tmps_[97_vvoo](b,a,i,j) = 1.00 t2(d,a,i,j) * eri[oovv](l,k,c,d) * t1(c,k) * t1(b,l) // flops: o2v2 = o3v2 o3v2 | mem: o2v2 = o3v1 o2v2
    tmps_["97_vvoo"] = einsum('daij,ld,bl->baij', t2, tmps_["45_ov"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["45_ov"]
    doubles_resid -= einsum('baij->abij', tmps_["97_vvoo"])

    # doubles_resid += +1.00 P(a,b) <l,k||c,d> t1(c,k) t1(a,l) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["97_vvoo"])
    del tmps_["97_vvoo"]

    # tmps_[46_oo](l,j) = 1.00 eri[oovo](l,k,c,j) * t1(c,k) // flops: o2v0 = o3v1 | mem: o2v0 = o2v0
    tmps_["46_oo"] = einsum('lkcj,ck->lj', eri["oovo"], t1)

    # singles_resid += +1.00 <k,j||b,i> t1(b,j) t1(a,k)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid += einsum('ak,ki->ai', t1, tmps_["46_oo"])

    # tmps_[84_vvvooo](a,b,c,j,k,i) = 1.00 t3(a,b,c,j,k,m) * eri[oovo](m,l,d,i) * t1(d,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["84_vvvooo"] = einsum('abcjkm,mi->abcjki', t3, tmps_["46_oo"])
    triples_resid -= einsum('abcikj->abcijk', tmps_["84_vvvooo"])

    # triples_resid += +1.00 <m,l||d,i> t1(d,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcjki->abcijk', tmps_["84_vvvooo"])

    # triples_resid += +1.00 P(j,k) <m,l||d,k> t1(d,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["84_vvvooo"])
    del tmps_["84_vvvooo"]

    # tmps_[101_vvoo](a,b,j,i) = 1.00 t2(a,b,j,l) * eri[oovo](l,k,c,i) * t1(c,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["101_vvoo"] = einsum('abjl,li->abji', t2, tmps_["46_oo"])
    del tmps_["46_oo"]
    doubles_resid -= einsum('abji->abij', tmps_["101_vvoo"])

    # doubles_resid += +1.00 P(i,j) <l,k||c,j> t1(c,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["101_vvoo"])
    del tmps_["101_vvoo"]

    # tmps_[47_oo](k,i) = 1.00 f[ov](k,c) * t1(c,i) // flops: o2v0 = o2v1 | mem: o2v0 = o2v0
    tmps_["47_oo"] = einsum('kc,ci->ki', f["ov"], t1)

    # singles_resid += -1.00 f(j,b) t1(b,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid -= einsum('aj,ji->ai', t1, tmps_["47_oo"])

    # tmps_[87_vvvooo](a,b,c,j,k,i) = 1.00 t3(a,b,c,j,k,l) * f[ov](l,d) * t1(d,i) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["87_vvvooo"] = einsum('abcjkl,li->abcjki', t3, tmps_["47_oo"])

    # triples_resid += -1.00 f(l,d) t1(d,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcjki->abcijk', tmps_["87_vvvooo"])

    # triples_resid += -1.00 P(j,k) f(l,d) t1(d,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcijk->abcijk', tmps_["87_vvvooo"])
    triples_resid += einsum('abcikj->abcijk', tmps_["87_vvvooo"])
    del tmps_["87_vvvooo"]

    # tmps_[104_vvoo](a,b,j,i) = 1.00 t2(a,b,j,k) * f[ov](k,c) * t1(c,i) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["104_vvoo"] = einsum('abjk,ki->abji', t2, tmps_["47_oo"])
    del tmps_["47_oo"]

    # doubles_resid += -1.00 P(i,j) f(k,c) t1(c,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["104_vvoo"])
    doubles_resid += einsum('abji->abij', tmps_["104_vvoo"])
    del tmps_["104_vvoo"]

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



