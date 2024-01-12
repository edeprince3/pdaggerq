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

    ### End of Scalars ###

    ##########  Evaluate Equations  ##########


    # doubles_resid = +1.00 f(k,c) t3(c,a,b,i,j,k)  // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    doubles_resid = einsum('kc,cabijk->abij', f["ov"], t3)

    # doubles_resid += +1.00 <a,b||i,j>  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', eri["vvoo"])

    # doubles_resid += +0.50 <l,k||i,j> t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
    doubles_resid += 0.50 * einsum('lkij,ablk->abij', eri["oooo"], t2)

    # doubles_resid += +0.50 <a,b||c,d> t2(c,d,i,j)  // flops: o2v2 += o2v4 | mem: o2v2 += o2v2
    doubles_resid += 0.50 * einsum('abcd,cdij->abij', eri["vvvv"], t2)

    # doubles_resid += -0.50 <l,k||c,d> t2(c,a,i,j) t2(d,b,l,k)  // flops: o2v2 += o2v3 o2v3 | mem: o2v2 += o0v2 o2v2
    doubles_resid -= 0.50 * einsum('lkcd,dblk,caij->abij', eri["oovv"], t2, t2, optimize=['einsum_path',(0,1),(0,1)])

    # singles_resid = +1.00 f(a,i)  // flops: o1v1 = o1v1 | mem: o1v1 = o1v1
    singles_resid = 1.00 * einsum('ai->ai', f["vo"])

    # singles_resid += -1.00 f(j,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid -= einsum('ji,aj->ai', f["oo"], t1)

    # singles_resid += +1.00 f(a,b) t1(b,i)  // flops: o1v1 += o1v2 | mem: o1v1 += o1v1
    singles_resid += einsum('ab,bi->ai', f["vv"], t1)

    # singles_resid += -1.00 f(j,b) t2(b,a,i,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid -= einsum('jb,baij->ai', f["ov"], t2)

    # singles_resid += +1.00 <j,a||b,i> t1(b,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid -= einsum('ajbi,bj->ai', eri["vovo"], t1)

    # singles_resid += -0.50 <k,j||b,i> t2(b,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
    singles_resid -= 0.50 * einsum('kjbi,bakj->ai', eri["oovo"], t2)

    # singles_resid += -0.50 <j,a||b,c> t2(b,c,i,j)  // flops: o1v1 += o2v3 | mem: o1v1 += o1v1
    singles_resid += 0.50 * einsum('ajbc,bcij->ai', eri["vovv"], t2)

    # singles_resid += +0.25 <k,j||b,c> t3(b,c,a,i,k,j)  // flops: o1v1 += o3v3 | mem: o1v1 += o1v1
    singles_resid += 0.25 * einsum('kjbc,bcaikj->ai', eri["oovv"], t3)

    # singles_resid += +0.50 <k,j||b,c> t1(a,j) t2(b,c,i,k)  // flops: o1v1 += o3v2 o2v1 | mem: o1v1 += o2v0 o1v1
    singles_resid += 0.50 * einsum('kjbc,bcik,aj->ai', eri["oovv"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

    # singles_resid += +1.00 <j,a||b,c> t1(b,j) t1(c,i)  // flops: o1v1 += o1v3 o1v2 | mem: o1v1 += o0v2 o1v1
    singles_resid -= einsum('ajbc,bj,ci->ai', eri["vovv"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])

    # tmps_[vvvooo_1](a,b,c,i,j,k) = 0.50 eri[vvvv](a,b,d,e) * t3(d,e,c,i,j,k) // flops: o3v3 = o3v5 | mem: o3v3 = o3v3
    tmps_["vvvooo_1"] = 0.50 * einsum('abde,decijk->abcijk', eri["vvvv"], t3)

    # triples_resid = +0.50 P(b,c) <a,b||d,e> t3(d,e,c,i,j,k)  // flops: o3v3 = o3v3 | mem: o3v3 = o3v3
    triples_resid = 1.00 * einsum('abcijk->abcijk', tmps_["vvvooo_1"])
    triples_resid -= einsum('acbijk->abcijk', tmps_["vvvooo_1"])

    # triples_resid += +0.50 <b,c||d,e> t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcaijk->abcijk', tmps_["vvvooo_1"])
    del tmps_["vvvooo_1"]

    # tmps_[vovvoo_2](a,i,b,c,j,k) = 1.00 eri[vovo](a,l,d,i) * t3(d,b,c,j,k,l) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
    tmps_["vovvoo_2"] = einsum('aldi,dbcjkl->aibcjk', eri["vovo"], t3)

    # triples_resid += +1.00 P(j,k) P(a,b) <l,a||d,k> t3(d,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('akbcij->abcijk', tmps_["vovvoo_2"])
    triples_resid += einsum('ajbcik->abcijk', tmps_["vovvoo_2"])
    triples_resid += einsum('bkacij->abcijk', tmps_["vovvoo_2"])
    triples_resid -= einsum('bjacik->abcijk', tmps_["vovvoo_2"])

    # triples_resid += +1.00 P(a,b) <l,a||d,i> t3(d,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aibcjk->abcijk', tmps_["vovvoo_2"])
    triples_resid += einsum('biacjk->abcijk', tmps_["vovvoo_2"])

    # triples_resid += +1.00 P(j,k) <l,c||d,k> t3(d,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ckabij->abcijk', tmps_["vovvoo_2"])
    triples_resid += einsum('cjabik->abcijk', tmps_["vovvoo_2"])

    # triples_resid += +1.00 <l,c||d,i> t3(d,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ciabjk->abcijk', tmps_["vovvoo_2"])
    del tmps_["vovvoo_2"]

    # tmps_[oovooo_5](m,l,c,i,j,k) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,c,i,j,k) // flops: o5v1 = o5v3 | mem: o5v1 = o5v1
    tmps_["oovooo_5"] = 0.50 * einsum('mlde,decijk->mlcijk', eri["oovv"], t3)

    # tmps_[vvvooo_53](a,c,b,i,j,k) = 0.50 t2(a,c,m,l) * eri[oovv](m,l,d,e) * t3(d,e,b,i,j,k) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_53"] = 0.50 * einsum('acml,mlbijk->acbijk', t2, tmps_["oovooo_5"])

    # triples_resid += +0.25 P(b,c) <m,l||d,e> t2(a,b,m,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["vvvooo_53"])
    triples_resid -= einsum('acbijk->abcijk', tmps_["vvvooo_53"])

    # triples_resid += +0.25 <m,l||d,e> t2(b,c,m,l) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcaijk->abcijk', tmps_["vvvooo_53"])
    del tmps_["vvvooo_53"]

    # tmps_[vvooov_117](c,b,i,j,k,a) = 1.00 t1(c,m) * eri[oovv](m,l,d,e) * t3(d,e,b,i,j,k) * t1(a,l) // flops: o3v3 = o5v2 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["vvooov_117"] = einsum('cm,mlbijk,al->cbijka', t1, tmps_["oovooo_5"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["oovooo_5"]

    # triples_resid += -0.50 P(b,c) <m,l||d,e> t1(a,l) t1(b,m) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijka->abcijk', tmps_["vvooov_117"])
    triples_resid += einsum('cbijka->abcijk', tmps_["vvooov_117"])

    # triples_resid += -0.50 <m,l||d,e> t1(b,l) t1(c,m) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('caijkb->abcijk', tmps_["vvooov_117"])
    del tmps_["vvooov_117"]

    # tmps_[oovvvo_6](j,k,a,b,c,i) = 0.50 eri[oooo](m,l,j,k) * t3(a,b,c,i,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["oovvvo_6"] = 0.50 * einsum('mljk,abciml->jkabci', eri["oooo"], t3)

    # triples_resid += +0.50 P(i,j) <m,l||j,k> t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('jkabci->abcijk', tmps_["oovvvo_6"])
    triples_resid -= einsum('ikabcj->abcijk', tmps_["oovvvo_6"])

    # triples_resid += +0.50 <m,l||i,j> t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ijabck->abcijk', tmps_["oovvvo_6"])
    del tmps_["oovvvo_6"]

    # tmps_[vvvo_7](c,a,b,j) = 0.50 eri[oovv](l,k,c,d) * t3(d,a,b,j,l,k) // flops: o1v3 = o3v4 | mem: o1v3 = o1v3
    tmps_["vvvo_7"] = 0.50 * einsum('lkcd,dabjlk->cabj', eri["oovv"], t3)

    # tmps_[voovvo_57](a,j,k,b,c,i) = 1.00 t2(d,a,j,k) * eri[oovv](m,l,d,e) * t3(e,b,c,i,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["voovvo_57"] = einsum('dajk,dbci->ajkbci', t2, tmps_["vvvo_7"])

    # triples_resid += -0.50 P(i,j) P(a,b) <m,l||d,e> t2(d,a,j,k) t3(e,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkbci->abcijk', tmps_["voovvo_57"])
    triples_resid += einsum('aikbcj->abcijk', tmps_["voovvo_57"])
    triples_resid += einsum('bjkaci->abcijk', tmps_["voovvo_57"])
    triples_resid -= einsum('bikacj->abcijk', tmps_["voovvo_57"])

    # triples_resid += -0.50 P(a,b) <m,l||d,e> t2(d,a,i,j) t3(e,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aijbck->abcijk', tmps_["voovvo_57"])
    triples_resid += einsum('bijack->abcijk', tmps_["voovvo_57"])

    # triples_resid += -0.50 P(i,j) <m,l||d,e> t2(d,c,j,k) t3(e,a,b,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjkabi->abcijk', tmps_["voovvo_57"])
    triples_resid += einsum('cikabj->abcijk', tmps_["voovvo_57"])

    # triples_resid += -0.50 <m,l||d,e> t2(d,c,i,j) t3(e,a,b,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cijabk->abcijk', tmps_["voovvo_57"])
    del tmps_["voovvo_57"]

    # tmps_[ovvo_86](i,a,b,j) = 1.00 t1(c,i) * eri[oovv](l,k,c,d) * t3(d,a,b,j,l,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["ovvo_86"] = einsum('ci,cabj->iabj', t1, tmps_["vvvo_7"])
    del tmps_["vvvo_7"]

    # doubles_resid += -0.50 P(i,j) <l,k||c,d> t1(c,j) t3(d,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('jabi->abij', tmps_["ovvo_86"])
    doubles_resid += einsum('iabj->abij', tmps_["ovvo_86"])
    del tmps_["ovvo_86"]

    # tmps_[vvovoo_8](a,b,j,c,i,k) = 1.00 eri[vvvo](a,b,d,j) * t2(d,c,i,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvovoo_8"] = einsum('abdj,dcik->abjcik', eri["vvvo"], t2)

    # triples_resid += -1.00 P(j,k) P(b,c) <a,b||d,k> t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abkcij->abcijk', tmps_["vvovoo_8"])
    triples_resid += einsum('abjcik->abcijk', tmps_["vvovoo_8"])
    triples_resid += einsum('ackbij->abcijk', tmps_["vvovoo_8"])
    triples_resid -= einsum('acjbik->abcijk', tmps_["vvovoo_8"])

    # triples_resid += -1.00 P(b,c) <a,b||d,i> t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["vvovoo_8"])
    triples_resid += einsum('acibjk->abcijk', tmps_["vvovoo_8"])

    # triples_resid += -1.00 P(j,k) <b,c||d,k> t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bckaij->abcijk', tmps_["vvovoo_8"])
    triples_resid += einsum('bcjaik->abcijk', tmps_["vvovoo_8"])

    # triples_resid += -1.00 <b,c||d,i> t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bciajk->abcijk', tmps_["vvovoo_8"])
    del tmps_["vvovoo_8"]

    # tmps_[vvvooo_9](a,b,c,i,j,k) = 1.00 f[vv](a,d) * t3(d,b,c,i,j,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvvooo_9"] = einsum('ad,dbcijk->abcijk', f["vv"], t3)

    # triples_resid += +1.00 P(a,b) f(a,d) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["vvvooo_9"])
    triples_resid -= einsum('bacijk->abcijk', tmps_["vvvooo_9"])

    # triples_resid += +1.00 f(c,d) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cabijk->abcijk', tmps_["vvvooo_9"])
    del tmps_["vvvooo_9"]

    # tmps_[vvoo_10](a,b,i,j) = 0.50 eri[vovv](a,k,c,d) * t3(c,d,b,i,j,k) // flops: o2v2 = o3v4 | mem: o2v2 = o2v2
    tmps_["vvoo_10"] = 0.50 * einsum('akcd,cdbijk->abij', eri["vovv"], t3)

    # doubles_resid += +0.50 P(a,b) <k,a||c,d> t3(c,d,b,i,j,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["vvoo_10"])
    doubles_resid += einsum('baij->abij', tmps_["vvoo_10"])
    del tmps_["vvoo_10"]

    # tmps_[ovoo_12](l,a,i,j) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,a,i,j,m) // flops: o3v1 = o4v3 | mem: o3v1 = o3v1
    tmps_["ovoo_12"] = 0.50 * einsum('mlde,deaijm->laij', eri["oovv"], t3)

    # tmps_[voovvo_71](c,i,j,a,b,k) = 1.00 eri[oovv](m,l,d,e) * t3(d,e,c,i,j,m) * t2(a,b,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["voovvo_71"] = einsum('lcij,abkl->cijabk', tmps_["ovoo_12"], t2)

    # triples_resid += -0.50 P(j,k) P(b,c) <m,l||d,e> t2(a,b,k,l) t3(d,e,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cijabk->abcijk', tmps_["voovvo_71"])
    triples_resid += einsum('cikabj->abcijk', tmps_["voovvo_71"])
    triples_resid += einsum('bijack->abcijk', tmps_["voovvo_71"])
    triples_resid -= einsum('bikacj->abcijk', tmps_["voovvo_71"])

    # triples_resid += -0.50 P(b,c) <m,l||d,e> t2(a,b,i,l) t3(d,e,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjkabi->abcijk', tmps_["voovvo_71"])
    triples_resid += einsum('bjkaci->abcijk', tmps_["voovvo_71"])

    # triples_resid += -0.50 P(j,k) <m,l||d,e> t2(b,c,k,l) t3(d,e,a,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aijbck->abcijk', tmps_["voovvo_71"])
    triples_resid += einsum('aikbcj->abcijk', tmps_["voovvo_71"])

    # triples_resid += -0.50 <m,l||d,e> t2(b,c,i,l) t3(d,e,a,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkbci->abcijk', tmps_["voovvo_71"])
    del tmps_["voovvo_71"]

    # tmps_[vvoo_95](a,b,i,j) = 1.00 t1(a,k) * eri[oovv](l,k,c,d) * t3(c,d,b,i,j,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_95"] = einsum('ak,kbij->abij', t1, tmps_["ovoo_12"])
    del tmps_["ovoo_12"]

    # doubles_resid += -0.50 P(a,b) <l,k||c,d> t1(a,k) t3(c,d,b,i,j,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["vvoo_95"])
    doubles_resid += einsum('baij->abij', tmps_["vvoo_95"])
    del tmps_["vvoo_95"]

    # tmps_[voovvo_13](a,i,k,b,c,j) = 1.00 eri[vooo](a,l,i,k) * t2(b,c,j,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["voovvo_13"] = einsum('alik,bcjl->aikbcj', eri["vooo"], t2)

    # triples_resid += -1.00 P(i,j) P(a,b) <l,a||j,k> t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ajkbci->abcijk', tmps_["voovvo_13"])
    triples_resid -= einsum('aikbcj->abcijk', tmps_["voovvo_13"])
    triples_resid -= einsum('bjkaci->abcijk', tmps_["voovvo_13"])
    triples_resid += einsum('bikacj->abcijk', tmps_["voovvo_13"])

    # triples_resid += -1.00 P(a,b) <l,a||i,j> t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aijbck->abcijk', tmps_["voovvo_13"])
    triples_resid -= einsum('bijack->abcijk', tmps_["voovvo_13"])

    # triples_resid += -1.00 P(i,j) <l,c||j,k> t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cjkabi->abcijk', tmps_["voovvo_13"])
    triples_resid -= einsum('cikabj->abcijk', tmps_["voovvo_13"])

    # triples_resid += -1.00 <l,c||i,j> t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cijabk->abcijk', tmps_["voovvo_13"])
    del tmps_["voovvo_13"]

    # tmps_[ovvvoo_15](k,a,b,c,i,j) = 1.00 f[oo](l,k) * t3(a,b,c,i,j,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["ovvvoo_15"] = einsum('lk,abcijl->kabcij', f["oo"], t3)

    # triples_resid += -1.00 P(j,k) f(l,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('kabcij->abcijk', tmps_["ovvvoo_15"])
    triples_resid += einsum('jabcik->abcijk', tmps_["ovvvoo_15"])

    # triples_resid += -1.00 f(l,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('iabcjk->abcijk', tmps_["ovvvoo_15"])
    del tmps_["ovvvoo_15"]

    # tmps_[ovvo_16](j,a,b,i) = 0.50 eri[oovo](l,k,c,j) * t3(c,a,b,i,l,k) // flops: o2v2 = o4v3 | mem: o2v2 = o2v2
    tmps_["ovvo_16"] = 0.50 * einsum('lkcj,cabilk->jabi', eri["oovo"], t3)

    # doubles_resid += +0.50 P(i,j) <l,k||c,j> t3(c,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('jabi->abij', tmps_["ovvo_16"])
    doubles_resid -= einsum('iabj->abij', tmps_["ovvo_16"])
    del tmps_["ovvo_16"]

    # tmps_[vvvo_18](b,c,a,i) = 1.00 eri[vovv](b,k,c,d) * t2(d,a,i,k) // flops: o1v3 = o2v4 | mem: o1v3 = o1v3
    tmps_["vvvo_18"] = einsum('bkcd,daik->bcai', eri["vovv"], t2)

    # tmps_[voovvo_58](a,i,k,c,b,j) = 1.00 t2(d,a,i,k) * eri[vovv](c,l,d,e) * t2(e,b,j,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["voovvo_58"] = einsum('daik,cdbj->aikcbj', t2, tmps_["vvvo_18"])

    # triples_resid += +1.00 P(i,j) P(a,b) <l,a||d,e> t2(d,b,j,k) t2(e,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bjkaci->abcijk', tmps_["voovvo_58"])
    triples_resid += einsum('bikacj->abcijk', tmps_["voovvo_58"])
    triples_resid += einsum('ajkbci->abcijk', tmps_["voovvo_58"])
    triples_resid -= einsum('aikbcj->abcijk', tmps_["voovvo_58"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,j) t2(e,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bijack->abcijk', tmps_["voovvo_58"])
    triples_resid += einsum('aijbck->abcijk', tmps_["voovvo_58"])

    # triples_resid += +1.00 P(i,j) <l,c||d,e> t2(d,a,j,k) t2(e,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkcbi->abcijk', tmps_["voovvo_58"])
    triples_resid += einsum('aikcbj->abcijk', tmps_["voovvo_58"])

    # triples_resid += +1.00 <l,c||d,e> t2(d,a,i,j) t2(e,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aijcbk->abcijk', tmps_["voovvo_58"])
    del tmps_["voovvo_58"]

    # tmps_[ovvo_85](i,a,b,j) = 1.00 t1(c,i) * eri[vovv](a,k,c,d) * t2(d,b,j,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["ovvo_85"] = einsum('ci,acbj->iabj', t1, tmps_["vvvo_18"])
    del tmps_["vvvo_18"]

    # doubles_resid += -1.00 P(i,j) P(a,b) <k,a||c,d> t1(c,j) t2(d,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('jabi->abij', tmps_["ovvo_85"])
    doubles_resid -= einsum('iabj->abij', tmps_["ovvo_85"])
    doubles_resid -= einsum('jbai->abij', tmps_["ovvo_85"])
    doubles_resid += einsum('ibaj->abij', tmps_["ovvo_85"])
    del tmps_["ovvo_85"]

    # tmps_[ovvo_20](k,c,b,j) = 1.00 eri[oovv](l,k,c,d) * t2(d,b,j,l) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["ovvo_20"] = einsum('lkcd,dbjl->kcbj', eri["oovv"], t2)

    # singles_resid += +1.00 <k,j||b,c> t1(b,j) t2(c,a,i,k)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid += einsum('jbai,bj->ai', tmps_["ovvo_20"], t1)

    # tmps_[vovo_83](a,j,b,i) = 1.00 t2(c,a,j,k) * eri[oovv](l,k,c,d) * t2(d,b,i,l) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["vovo_83"] = einsum('cajk,kcbi->ajbi', t2, tmps_["ovvo_20"])

    # doubles_resid += +1.00 P(i,j) <l,k||c,d> t2(c,a,j,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('ajbi->abij', tmps_["vovo_83"])
    doubles_resid -= einsum('aibj->abij', tmps_["vovo_83"])
    del tmps_["vovo_83"]

    # tmps_[vvoovo_114](a,b,j,k,c,i) = 1.00 t2(d,b,j,k) * eri[oovv](m,l,d,e) * t2(e,c,i,m) * t1(a,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["vvoovo_114"] = einsum('dbjk,ldci,al->abjkci', t2, tmps_["ovvo_20"], t1, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,j,k) t2(e,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abjkci->abcijk', tmps_["vvoovo_114"])
    triples_resid += einsum('abikcj->abcijk', tmps_["vvoovo_114"])
    triples_resid += einsum('bajkci->abcijk', tmps_["vvoovo_114"])
    triples_resid -= einsum('baikcj->abcijk', tmps_["vvoovo_114"])

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,j) t2(e,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijck->abcijk', tmps_["vvoovo_114"])
    triples_resid += einsum('baijck->abcijk', tmps_["vvoovo_114"])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(c,l) t2(d,a,j,k) t2(e,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cajkbi->abcijk', tmps_["vvoovo_114"])
    triples_resid += einsum('caikbj->abcijk', tmps_["vvoovo_114"])

    # triples_resid += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,j) t2(e,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('caijbk->abcijk', tmps_["vvoovo_114"])
    del tmps_["vvoovo_114"]

    # tmps_[vovo_120](a,j,b,i) = 1.00 t1(c,j) * eri[oovv](l,k,c,d) * t2(d,b,i,l) * t1(a,k) // flops: o2v2 = o3v2 o3v2 | mem: o2v2 = o3v1 o2v2
    tmps_["vovo_120"] = einsum('cj,kcbi,ak->ajbi', t1, tmps_["ovvo_20"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["ovvo_20"]

    # doubles_resid += +1.00 P(i,j) P(a,b) <l,k||c,d> t1(c,j) t1(a,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('ajbi->abij', tmps_["vovo_120"])
    doubles_resid -= einsum('aibj->abij', tmps_["vovo_120"])
    doubles_resid -= einsum('bjai->abij', tmps_["vovo_120"])
    doubles_resid += einsum('biaj->abij', tmps_["vovo_120"])
    del tmps_["vovo_120"]

    # tmps_[vooo_21](c,l,i,j) = 0.50 eri[vovv](c,l,d,e) * t2(d,e,i,j) // flops: o3v1 = o3v3 | mem: o3v1 = o3v1
    tmps_["vooo_21"] = 0.50 * einsum('clde,deij->clij', eri["vovv"], t2)

    # tmps_[voovvo_70](c,i,k,a,b,j) = 1.00 eri[vovv](c,l,d,e) * t2(d,e,i,k) * t2(a,b,j,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["voovvo_70"] = einsum('clik,abjl->cikabj', tmps_["vooo_21"], t2)

    # triples_resid += -0.50 P(i,j) P(a,b) <l,a||d,e> t2(d,e,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ajkbci->abcijk', tmps_["voovvo_70"])
    triples_resid -= einsum('aikbcj->abcijk', tmps_["voovvo_70"])
    triples_resid -= einsum('bjkaci->abcijk', tmps_["voovvo_70"])
    triples_resid += einsum('bikacj->abcijk', tmps_["voovvo_70"])

    # triples_resid += -0.50 P(a,b) <l,a||d,e> t2(d,e,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aijbck->abcijk', tmps_["voovvo_70"])
    triples_resid -= einsum('bijack->abcijk', tmps_["voovvo_70"])

    # triples_resid += -0.50 P(i,j) <l,c||d,e> t2(d,e,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cjkabi->abcijk', tmps_["voovvo_70"])
    triples_resid -= einsum('cikabj->abcijk', tmps_["voovvo_70"])

    # triples_resid += -0.50 <l,c||d,e> t2(d,e,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cijabk->abcijk', tmps_["voovvo_70"])
    del tmps_["voovvo_70"]

    # tmps_[vvoo_94](a,b,i,j) = 1.00 t1(a,k) * eri[vovv](b,k,c,d) * t2(c,d,i,j) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_94"] = einsum('ak,bkij->abij', t1, tmps_["vooo_21"])
    del tmps_["vooo_21"]

    # doubles_resid += +0.50 P(a,b) <k,a||c,d> t1(b,k) t2(c,d,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('baij->abij', tmps_["vvoo_94"])
    doubles_resid += einsum('abij->abij', tmps_["vvoo_94"])
    del tmps_["vvoo_94"]

    # tmps_[vovv_23](c,i,a,b) = 0.50 eri[oovo](l,k,c,i) * t2(a,b,l,k) // flops: o1v3 = o3v3 | mem: o1v3 = o1v3
    tmps_["vovv_23"] = 0.50 * einsum('lkci,ablk->ciab', eri["oovo"], t2)

    # tmps_[vooovv_59](a,i,j,k,b,c) = 1.00 t2(d,a,i,j) * eri[oovo](m,l,d,k) * t2(b,c,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vooovv_59"] = einsum('daij,dkbc->aijkbc', t2, tmps_["vovv_23"])

    # triples_resid += -0.50 P(j,k) P(a,b) <m,l||d,k> t2(d,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('aijkbc->abcijk', tmps_["vooovv_59"])
    triples_resid += einsum('aikjbc->abcijk', tmps_["vooovv_59"])
    triples_resid += einsum('bijkac->abcijk', tmps_["vooovv_59"])
    triples_resid -= einsum('bikjac->abcijk', tmps_["vooovv_59"])

    # triples_resid += -0.50 P(j,k) <m,l||d,k> t2(d,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cijkab->abcijk', tmps_["vooovv_59"])
    triples_resid += einsum('cikjab->abcijk', tmps_["vooovv_59"])

    # triples_resid += -0.50 P(a,b) <m,l||d,i> t2(d,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ajkibc->abcijk', tmps_["vooovv_59"])
    triples_resid += einsum('bjkiac->abcijk', tmps_["vooovv_59"])

    # triples_resid += -0.50 <m,l||d,i> t2(d,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cjkiab->abcijk', tmps_["vooovv_59"])
    del tmps_["vooovv_59"]

    # tmps_[oovv_87](i,j,a,b) = 1.00 t1(c,i) * eri[oovo](l,k,c,j) * t2(a,b,l,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["oovv_87"] = einsum('ci,cjab->ijab', t1, tmps_["vovv_23"])
    del tmps_["vovv_23"]

    # doubles_resid += +0.50 P(i,j) <l,k||c,j> t1(c,i) t2(a,b,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('ijab->abij', tmps_["oovv_87"])
    doubles_resid -= einsum('jiab->abij', tmps_["oovv_87"])
    del tmps_["oovv_87"]

    # tmps_[vovo_25](a,j,b,i) = 1.00 eri[vovo](a,k,c,j) * t2(c,b,i,k) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["vovo_25"] = einsum('akcj,cbik->ajbi', eri["vovo"], t2)

    # doubles_resid += +1.00 P(i,j) P(a,b) <k,a||c,j> t2(c,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('ajbi->abij', tmps_["vovo_25"])
    doubles_resid += einsum('aibj->abij', tmps_["vovo_25"])
    doubles_resid += einsum('bjai->abij', tmps_["vovo_25"])
    doubles_resid -= einsum('biaj->abij', tmps_["vovo_25"])
    del tmps_["vovo_25"]

    # tmps_[oooo_27](m,l,i,j) = 0.50 eri[oovv](m,l,d,e) * t2(d,e,i,j) // flops: o4v0 = o4v2 | mem: o4v0 = o4v0
    tmps_["oooo_27"] = 0.50 * einsum('mlde,deij->mlij', eri["oovv"], t2)

    # doubles_resid += +0.25 <l,k||c,d> t2(c,d,i,j) t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
    doubles_resid += 0.50 * einsum('lkij,ablk->abij', tmps_["oooo_27"], t2)

    # tmps_[oovvvo_54](i,k,a,b,c,j) = 0.50 eri[oovv](m,l,d,e) * t2(d,e,i,k) * t3(a,b,c,j,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["oovvvo_54"] = 0.50 * einsum('mlik,abcjml->ikabcj', tmps_["oooo_27"], t3)

    # triples_resid += +0.25 P(i,j) <m,l||d,e> t2(d,e,j,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('jkabci->abcijk', tmps_["oovvvo_54"])
    triples_resid -= einsum('ikabcj->abcijk', tmps_["oovvvo_54"])

    # triples_resid += +0.25 <m,l||d,e> t2(d,e,i,j) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ijabck->abcijk', tmps_["oovvvo_54"])
    del tmps_["oovvvo_54"]

    # tmps_[vooo_102](c,m,i,k) = 1.00 t1(c,l) * eri[oovv](m,l,d,e) * t2(d,e,i,k) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["vooo_102"] = einsum('cl,mlik->cmik', t1, tmps_["oooo_27"])
    del tmps_["oooo_27"]

    # doubles_resid += -0.50 <l,k||c,d> t1(a,k) t1(b,l) t2(c,d,i,j)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('alij,bl->abij', tmps_["vooo_102"], t1)

    # tmps_[vvovoo_109](b,c,k,a,i,j) = 1.00 t2(b,c,k,m) * t1(a,l) * eri[oovv](m,l,d,e) * t2(d,e,i,j) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvovoo_109"] = einsum('bckm,amij->bckaij', t2, tmps_["vooo_102"])
    del tmps_["vooo_102"]

    # triples_resid += +0.50 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,e,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bciajk->abcijk', tmps_["vvovoo_109"])
    triples_resid -= einsum('bcjaik->abcijk', tmps_["vvovoo_109"])
    triples_resid -= einsum('acibjk->abcijk', tmps_["vvovoo_109"])
    triples_resid += einsum('acjbik->abcijk', tmps_["vvovoo_109"])

    # triples_resid += +0.50 P(a,b) <m,l||d,e> t1(a,l) t2(d,e,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bckaij->abcijk', tmps_["vvovoo_109"])
    triples_resid -= einsum('ackbij->abcijk', tmps_["vvovoo_109"])

    # triples_resid += +0.50 P(i,j) <m,l||d,e> t1(c,l) t2(d,e,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abicjk->abcijk', tmps_["vvovoo_109"])
    triples_resid -= einsum('abjcik->abcijk', tmps_["vvovoo_109"])

    # triples_resid += +0.50 <m,l||d,e> t1(c,l) t2(d,e,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abkcij->abcijk', tmps_["vvovoo_109"])
    del tmps_["vvovoo_109"]

    # tmps_[vvvo_29](a,b,d,j) = 1.00 eri[vvvv](a,b,c,d) * t1(c,j) // flops: o1v3 = o1v4 | mem: o1v3 = o1v3
    tmps_["vvvo_29"] = einsum('abcd,cj->abdj', eri["vvvv"], t1)

    # doubles_resid += -1.00 <a,b||c,d> t1(c,j) t1(d,i)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abdj,di->abij', tmps_["vvvo_29"], t1)

    # tmps_[vvovoo_56](a,b,i,c,j,k) = 1.00 eri[vvvv](a,b,d,e) * t1(d,i) * t2(e,c,j,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvovoo_56"] = einsum('abei,ecjk->abicjk', tmps_["vvvo_29"], t2)
    del tmps_["vvvo_29"]

    # triples_resid += +1.00 P(j,k) P(b,c) <a,b||d,e> t1(d,k) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abkcij->abcijk', tmps_["vvovoo_56"])
    triples_resid -= einsum('abjcik->abcijk', tmps_["vvovoo_56"])
    triples_resid -= einsum('ackbij->abcijk', tmps_["vvovoo_56"])
    triples_resid += einsum('acjbik->abcijk', tmps_["vvovoo_56"])

    # triples_resid += +1.00 P(b,c) <a,b||d,e> t1(d,i) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abicjk->abcijk', tmps_["vvovoo_56"])
    triples_resid -= einsum('acibjk->abcijk', tmps_["vvovoo_56"])

    # triples_resid += +1.00 P(j,k) <b,c||d,e> t1(d,k) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bckaij->abcijk', tmps_["vvovoo_56"])
    triples_resid -= einsum('bcjaik->abcijk', tmps_["vvovoo_56"])

    # triples_resid += +1.00 <b,c||d,e> t1(d,i) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bciajk->abcijk', tmps_["vvovoo_56"])
    del tmps_["vvovoo_56"]

    # tmps_[vovo_30](b,k,d,j) = 1.00 eri[vovv](b,k,c,d) * t1(c,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["vovo_30"] = einsum('bkcd,cj->bkdj', eri["vovv"], t1)

    # tmps_[vvoovo_50](b,c,j,k,a,i) = 1.00 t3(e,b,c,j,k,l) * eri[vovv](a,l,d,e) * t1(d,i) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
    tmps_["vvoovo_50"] = einsum('ebcjkl,alei->bcjkai', t3, tmps_["vovo_30"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,a||d,e> t1(d,k) t3(e,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijak->abcijk', tmps_["vvoovo_50"])
    triples_resid -= einsum('bcikaj->abcijk', tmps_["vvoovo_50"])
    triples_resid -= einsum('acijbk->abcijk', tmps_["vvoovo_50"])
    triples_resid += einsum('acikbj->abcijk', tmps_["vvoovo_50"])

    # triples_resid += -1.00 P(a,b) <l,a||d,e> t1(d,i) t3(e,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcjkai->abcijk', tmps_["vvoovo_50"])
    triples_resid -= einsum('acjkbi->abcijk', tmps_["vvoovo_50"])

    # triples_resid += -1.00 P(j,k) <l,c||d,e> t1(d,k) t3(e,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijck->abcijk', tmps_["vvoovo_50"])
    triples_resid -= einsum('abikcj->abcijk', tmps_["vvoovo_50"])

    # triples_resid += -1.00 <l,c||d,e> t1(d,i) t3(e,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abjkci->abcijk', tmps_["vvoovo_50"])
    del tmps_["vvoovo_50"]

    # tmps_[ovoo_89](i,c,l,j) = 1.00 t1(e,i) * eri[vovv](c,l,d,e) * t1(d,j) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["ovoo_89"] = einsum('ei,clej->iclj', t1, tmps_["vovo_30"])

    # tmps_[vvoovo_111](b,c,j,i,a,k) = 1.00 t2(b,c,j,l) * t1(e,i) * eri[vovv](a,l,d,e) * t1(d,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvoovo_111"] = einsum('bcjl,ialk->bcjiak', t2, tmps_["ovoo_89"])

    # triples_resid += +1.00 P(i,j) P(a,b) <l,a||d,e> t1(d,k) t1(e,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijak->abcijk', tmps_["vvoovo_111"])
    triples_resid += einsum('bcjiak->abcijk', tmps_["vvoovo_111"])
    triples_resid += einsum('acijbk->abcijk', tmps_["vvoovo_111"])
    triples_resid -= einsum('acjibk->abcijk', tmps_["vvoovo_111"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t1(d,j) t1(e,i) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bckiaj->abcijk', tmps_["vvoovo_111"])
    triples_resid += einsum('ackibj->abcijk', tmps_["vvoovo_111"])

    # triples_resid += +1.00 P(i,j) <l,c||d,e> t1(d,k) t1(e,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijck->abcijk', tmps_["vvoovo_111"])
    triples_resid += einsum('abjick->abcijk', tmps_["vvoovo_111"])

    # triples_resid += +1.00 <l,c||d,e> t1(d,j) t1(e,i) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abkicj->abcijk', tmps_["vvoovo_111"])
    del tmps_["vvoovo_111"]

    # tmps_[vovo_124](b,i,a,j) = 1.00 t1(b,k) * t1(d,i) * eri[vovv](a,k,c,d) * t1(c,j) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vovo_124"] = einsum('bk,iakj->biaj', t1, tmps_["ovoo_89"])
    del tmps_["ovoo_89"]

    # doubles_resid += -1.00 P(a,b) <k,a||c,d> t1(c,j) t1(d,i) t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('biaj->abij', tmps_["vovo_124"])
    doubles_resid -= einsum('aibj->abij', tmps_["vovo_124"])
    del tmps_["vovo_124"]

    # tmps_[vvoovo_108](b,c,i,j,a,k) = 1.00 t2(e,c,i,j) * eri[vovv](a,l,d,e) * t1(d,k) * t1(b,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["vvoovo_108"] = einsum('ecij,alek,bl->bcijak', t2, tmps_["vovo_30"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["vovo_30"]

    # triples_resid += +1.00 P(j,k) P(b,c) <l,a||d,e> t1(d,k) t1(b,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijak->abcijk', tmps_["vvoovo_108"])
    triples_resid += einsum('bcikaj->abcijk', tmps_["vvoovo_108"])
    triples_resid += einsum('cbijak->abcijk', tmps_["vvoovo_108"])
    triples_resid -= einsum('cbikaj->abcijk', tmps_["vvoovo_108"])

    # triples_resid += +1.00 P(b,c) <l,a||d,e> t1(d,i) t1(b,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcjkai->abcijk', tmps_["vvoovo_108"])
    triples_resid += einsum('cbjkai->abcijk', tmps_["vvoovo_108"])

    # triples_resid += -1.00 P(j,k) P(a,c) <l,b||d,e> t1(d,k) t1(a,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acijbk->abcijk', tmps_["vvoovo_108"])
    triples_resid -= einsum('acikbj->abcijk', tmps_["vvoovo_108"])
    triples_resid -= einsum('caijbk->abcijk', tmps_["vvoovo_108"])
    triples_resid += einsum('caikbj->abcijk', tmps_["vvoovo_108"])

    # triples_resid += -1.00 P(a,c) <l,b||d,e> t1(d,i) t1(a,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acjkbi->abcijk', tmps_["vvoovo_108"])
    triples_resid -= einsum('cajkbi->abcijk', tmps_["vvoovo_108"])

    # triples_resid += +1.00 P(j,k) P(a,b) <l,c||d,e> t1(d,k) t1(a,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijck->abcijk', tmps_["vvoovo_108"])
    triples_resid += einsum('abikcj->abcijk', tmps_["vvoovo_108"])
    triples_resid += einsum('baijck->abcijk', tmps_["vvoovo_108"])
    triples_resid -= einsum('baikcj->abcijk', tmps_["vvoovo_108"])

    # triples_resid += +1.00 P(a,b) <l,c||d,e> t1(d,i) t1(a,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abjkci->abcijk', tmps_["vvoovo_108"])
    triples_resid += einsum('bajkci->abcijk', tmps_["vvoovo_108"])
    del tmps_["vvoovo_108"]

    # tmps_[vv_32](d,a) = 0.50 eri[oovv](l,k,c,d) * t2(c,a,l,k) // flops: o0v2 = o2v3 | mem: o0v2 = o0v2
    tmps_["vv_32"] = 0.50 * einsum('lkcd,calk->da', eri["oovv"], t2)

    # doubles_resid += -0.50 <l,k||c,d> t2(c,a,l,k) t2(d,b,i,j)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= einsum('da,dbij->abij', tmps_["vv_32"], t2)

    # tmps_[vvooov_60](b,c,i,j,k,a) = 1.00 t3(e,b,c,i,j,k) * eri[oovv](m,l,d,e) * t2(d,a,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvooov_60"] = einsum('ebcijk,ea->bcijka', t3, tmps_["vv_32"])
    del tmps_["vv_32"]

    # triples_resid += -0.50 P(a,b) <m,l||d,e> t2(d,a,m,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijka->abcijk', tmps_["vvooov_60"])
    triples_resid += einsum('acijkb->abcijk', tmps_["vvooov_60"])

    # triples_resid += -0.50 <m,l||d,e> t2(d,c,m,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijkc->abcijk', tmps_["vvooov_60"])
    del tmps_["vvooov_60"]

    # tmps_[vvoo_33](a,b,j,i) = 1.00 eri[vvvo](a,b,c,j) * t1(c,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["vvoo_33"] = einsum('abcj,ci->abji', eri["vvvo"], t1)

    # doubles_resid += +1.00 P(i,j) <a,b||c,j> t1(c,i)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abji->abij', tmps_["vvoo_33"])
    doubles_resid -= einsum('abij->abij', tmps_["vvoo_33"])
    del tmps_["vvoo_33"]

    # tmps_[vvoo_34](b,a,i,j) = 1.00 f[vv](b,c) * t2(c,a,i,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["vvoo_34"] = einsum('bc,caij->baij', f["vv"], t2)

    # doubles_resid += +1.00 P(a,b) f(a,c) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["vvoo_34"])
    doubles_resid -= einsum('baij->abij', tmps_["vvoo_34"])
    del tmps_["vvoo_34"]

    # tmps_[oovo_35](l,k,d,i) = 1.00 eri[oovv](l,k,c,d) * t1(c,i) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["oovo_35"] = einsum('lkcd,ci->lkdi', eri["oovv"], t1)

    # singles_resid += +0.50 <k,j||b,c> t1(b,i) t2(c,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
    singles_resid += 0.50 * einsum('kjci,cakj->ai', tmps_["oovo_35"], t2)

    # tmps_[vovv_82](e,i,b,c) = 0.50 eri[oovv](m,l,d,e) * t1(d,i) * t2(b,c,m,l) // flops: o1v3 = o3v3 | mem: o1v3 = o1v3
    tmps_["vovv_82"] = 0.50 * einsum('mlei,bcml->eibc', tmps_["oovo_35"], t2)

    # doubles_resid += -0.50 <l,k||c,d> t1(c,j) t1(d,i) t2(a,b,l,k)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= einsum('djab,di->abij', tmps_["vovv_82"], t1)

    # tmps_[vooovv_105](a,i,j,k,b,c) = 1.00 t2(e,a,i,j) * eri[oovv](m,l,d,e) * t1(d,k) * t2(b,c,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vooovv_105"] = einsum('eaij,ekbc->aijkbc', t2, tmps_["vovv_82"])
    del tmps_["vovv_82"]

    # triples_resid += +0.50 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aijkbc->abcijk', tmps_["vooovv_105"])
    triples_resid -= einsum('aikjbc->abcijk', tmps_["vooovv_105"])
    triples_resid -= einsum('bijkac->abcijk', tmps_["vooovv_105"])
    triples_resid += einsum('bikjac->abcijk', tmps_["vooovv_105"])

    # triples_resid += +0.50 P(j,k) <m,l||d,e> t1(d,k) t2(e,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cijkab->abcijk', tmps_["vooovv_105"])
    triples_resid -= einsum('cikjab->abcijk', tmps_["vooovv_105"])

    # triples_resid += +0.50 P(a,b) <m,l||d,e> t1(d,i) t2(e,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ajkibc->abcijk', tmps_["vooovv_105"])
    triples_resid -= einsum('bjkiac->abcijk', tmps_["vooovv_105"])

    # triples_resid += +0.50 <m,l||d,e> t1(d,i) t2(e,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cjkiab->abcijk', tmps_["vooovv_105"])
    del tmps_["vooovv_105"]

    # tmps_[oooo_101](i,m,l,j) = 1.00 t1(e,i) * eri[oovv](m,l,d,e) * t1(d,j) // flops: o4v0 = o4v1 | mem: o4v0 = o4v0
    tmps_["oooo_101"] = einsum('ei,mlej->imlj', t1, tmps_["oovo_35"])

    # tmps_[vvvooo_104](a,b,c,j,i,k) = 0.50 t3(a,b,c,j,m,l) * t1(e,i) * eri[oovv](m,l,d,e) * t1(d,k) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_104"] = 0.50 * einsum('abcjml,imlk->abcjik', t3, tmps_["oooo_101"])

    # triples_resid += -0.50 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcijk->abcijk', tmps_["vvvooo_104"])
    triples_resid += einsum('abcjik->abcijk', tmps_["vvvooo_104"])

    # triples_resid += -0.50 <m,l||d,e> t1(d,j) t1(e,i) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abckij->abcijk', tmps_["vvvooo_104"])
    del tmps_["vvvooo_104"]

    # tmps_[ooov_125](i,m,j,c) = 1.00 t1(e,i) * eri[oovv](m,l,d,e) * t1(d,j) * t1(c,l) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["ooov_125"] = einsum('imlj,cl->imjc', tmps_["oooo_101"], t1)
    del tmps_["oooo_101"]

    # doubles_resid += +1.00 <l,k||c,d> t1(c,j) t1(d,i) t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('ilja,bl->abij', tmps_["ooov_125"], t1)

    # tmps_[vvooov_126](b,c,i,j,k,a) = 1.00 t2(b,c,i,m) * t1(e,j) * eri[oovv](m,l,d,e) * t1(d,k) * t1(a,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvooov_126"] = einsum('bcim,jmka->bcijka', t2, tmps_["ooov_125"])
    del tmps_["ooov_125"]

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t1(e,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijka->abcijk', tmps_["vvooov_126"])
    triples_resid += einsum('bcjika->abcijk', tmps_["vvooov_126"])
    triples_resid += einsum('acijkb->abcijk', tmps_["vvooov_126"])
    triples_resid -= einsum('acjikb->abcijk', tmps_["vvooov_126"])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijkc->abcijk', tmps_["vvooov_126"])
    triples_resid += einsum('abjikc->abcijk', tmps_["vvooov_126"])

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(d,j) t1(e,i) t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bckija->abcijk', tmps_["vvooov_126"])
    triples_resid += einsum('ackijb->abcijk', tmps_["vvooov_126"])

    # triples_resid += -1.00 <m,l||d,e> t1(d,j) t1(e,i) t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abkijc->abcijk', tmps_["vvooov_126"])
    del tmps_["vvooov_126"]

    # tmps_[vvovoo_107](a,b,k,c,i,j) = 1.00 t2(e,c,i,l) * eri[oovv](m,l,d,e) * t1(d,j) * t2(a,b,k,m) // flops: o3v3 = o4v2 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["vvovoo_107"] = einsum('ecil,mlej,abkm->abkcij', t2, tmps_["oovo_35"], t2, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bciajk->abcijk', tmps_["vvovoo_107"])
    triples_resid += einsum('bcjaik->abcijk', tmps_["vvovoo_107"])
    triples_resid += einsum('acibjk->abcijk', tmps_["vvovoo_107"])
    triples_resid -= einsum('acjbik->abcijk', tmps_["vvovoo_107"])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(d,k) t2(e,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["vvovoo_107"])
    triples_resid += einsum('abjcik->abcijk', tmps_["vvovoo_107"])

    # triples_resid += +1.00 P(i,k) P(a,b) <m,l||d,e> t1(d,j) t2(e,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bciakj->abcijk', tmps_["vvovoo_107"])
    triples_resid -= einsum('bckaij->abcijk', tmps_["vvovoo_107"])
    triples_resid -= einsum('acibkj->abcijk', tmps_["vvovoo_107"])
    triples_resid += einsum('ackbij->abcijk', tmps_["vvovoo_107"])

    # triples_resid += +1.00 P(i,k) <m,l||d,e> t1(d,j) t2(e,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abickj->abcijk', tmps_["vvovoo_107"])
    triples_resid -= einsum('abkcij->abcijk', tmps_["vvovoo_107"])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,i) t2(e,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcjaki->abcijk', tmps_["vvovoo_107"])
    triples_resid += einsum('bckaji->abcijk', tmps_["vvovoo_107"])
    triples_resid += einsum('acjbki->abcijk', tmps_["vvovoo_107"])
    triples_resid -= einsum('ackbji->abcijk', tmps_["vvovoo_107"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(d,i) t2(e,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abjcki->abcijk', tmps_["vvovoo_107"])
    triples_resid += einsum('abkcji->abcijk', tmps_["vvovoo_107"])
    del tmps_["vvovoo_107"]

    # tmps_[vvvooo_115](a,b,c,i,j,k) = 1.00 t3(e,b,c,i,j,m) * eri[oovv](m,l,d,e) * t1(d,k) * t1(a,l) // flops: o3v3 = o5v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["vvvooo_115"] = einsum('ebcijm,mlek,al->abcijk', t3, tmps_["oovo_35"], t1, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t1(a,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["vvvooo_115"])
    triples_resid -= einsum('abcikj->abcijk', tmps_["vvvooo_115"])
    triples_resid -= einsum('bacijk->abcijk', tmps_["vvvooo_115"])
    triples_resid += einsum('bacikj->abcijk', tmps_["vvvooo_115"])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t1(d,k) t1(c,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cabijk->abcijk', tmps_["vvvooo_115"])
    triples_resid -= einsum('cabikj->abcijk', tmps_["vvvooo_115"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,i) t1(a,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcjki->abcijk', tmps_["vvvooo_115"])
    triples_resid -= einsum('bacjki->abcijk', tmps_["vvvooo_115"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,i) t1(c,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cabjki->abcijk', tmps_["vvvooo_115"])
    del tmps_["vvvooo_115"]

    # tmps_[vovoov_127](c,k,b,i,j,a) = 1.00 eri[oovv](m,l,d,e) * t1(d,k) * t2(e,b,i,j) * t1(c,m) * t1(a,l) // flops: o3v3 = o5v2 o5v2 o4v3 | mem: o3v3 = o5v1 o4v2 o3v3
    tmps_["vovoov_127"] = einsum('mlek,ebij,cm,al->ckbija', tmps_["oovo_35"], t2, t1, t1, optimize=['einsum_path',(0,1),(0,1),(0,1)])
    del tmps_["oovo_35"]

    # triples_resid += -1.00 P(j,k) P(b,c) <m,l||d,e> t1(d,k) t1(a,l) t1(b,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bkcija->abcijk', tmps_["vovoov_127"])
    triples_resid += einsum('bjcika->abcijk', tmps_["vovoov_127"])
    triples_resid += einsum('ckbija->abcijk', tmps_["vovoov_127"])
    triples_resid -= einsum('cjbika->abcijk', tmps_["vovoov_127"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(d,k) t1(b,l) t1(c,m) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ckaijb->abcijk', tmps_["vovoov_127"])
    triples_resid += einsum('cjaikb->abcijk', tmps_["vovoov_127"])

    # triples_resid += -1.00 P(b,c) <m,l||d,e> t1(d,i) t1(a,l) t1(b,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bicjka->abcijk', tmps_["vovoov_127"])
    triples_resid += einsum('cibjka->abcijk', tmps_["vovoov_127"])

    # triples_resid += -1.00 <m,l||d,e> t1(d,i) t1(b,l) t1(c,m) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ciajkb->abcijk', tmps_["vovoov_127"])
    del tmps_["vovoov_127"]

    # tmps_[vooo_36](c,l,k,i) = 1.00 eri[vovo](c,l,d,k) * t1(d,i) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["vooo_36"] = einsum('cldk,di->clki', eri["vovo"], t1)

    # tmps_[vvovoo_65](a,b,k,c,j,i) = 1.00 t2(a,b,k,l) * eri[vovo](c,l,d,j) * t1(d,i) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvovoo_65"] = einsum('abkl,clji->abkcji', t2, tmps_["vooo_36"])

    # triples_resid += -1.00 P(i,j) P(a,b) <l,a||d,k> t1(d,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bciakj->abcijk', tmps_["vvovoo_65"])
    triples_resid -= einsum('bcjaki->abcijk', tmps_["vvovoo_65"])
    triples_resid -= einsum('acibkj->abcijk', tmps_["vvovoo_65"])
    triples_resid += einsum('acjbki->abcijk', tmps_["vvovoo_65"])

    # triples_resid += +1.00 P(i,k) P(a,b) <l,a||d,j> t1(d,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bciajk->abcijk', tmps_["vvovoo_65"])
    triples_resid += einsum('bckaji->abcijk', tmps_["vvovoo_65"])
    triples_resid += einsum('acibjk->abcijk', tmps_["vvovoo_65"])
    triples_resid -= einsum('ackbji->abcijk', tmps_["vvovoo_65"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,a||d,i> t1(d,k) t2(b,c,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcjaik->abcijk', tmps_["vvovoo_65"])
    triples_resid -= einsum('bckaij->abcijk', tmps_["vvovoo_65"])
    triples_resid -= einsum('acjbik->abcijk', tmps_["vvovoo_65"])
    triples_resid += einsum('ackbij->abcijk', tmps_["vvovoo_65"])

    # triples_resid += -1.00 P(i,j) <l,c||d,k> t1(d,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abickj->abcijk', tmps_["vvovoo_65"])
    triples_resid -= einsum('abjcki->abcijk', tmps_["vvovoo_65"])

    # triples_resid += +1.00 P(i,k) <l,c||d,j> t1(d,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["vvovoo_65"])
    triples_resid += einsum('abkcji->abcijk', tmps_["vvovoo_65"])

    # triples_resid += -1.00 P(j,k) <l,c||d,i> t1(d,k) t2(a,b,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abjcik->abcijk', tmps_["vvovoo_65"])
    triples_resid -= einsum('abkcij->abcijk', tmps_["vvovoo_65"])
    del tmps_["vvovoo_65"]

    # tmps_[voov_93](b,j,i,a) = 1.00 eri[vovo](b,k,c,j) * t1(c,i) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["voov_93"] = einsum('bkji,ak->bjia', tmps_["vooo_36"], t1)
    del tmps_["vooo_36"]

    # doubles_resid += +1.00 P(i,j) P(a,b) <k,a||c,j> t1(c,i) t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('ajib->abij', tmps_["voov_93"])
    doubles_resid += einsum('aijb->abij', tmps_["voov_93"])
    doubles_resid += einsum('bjia->abij', tmps_["voov_93"])
    doubles_resid -= einsum('bija->abij', tmps_["voov_93"])
    del tmps_["voov_93"]

    # tmps_[ovoo_38](l,a,i,j) = 1.00 f[ov](l,d) * t2(d,a,i,j) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["ovoo_38"] = einsum('ld,daij->laij', f["ov"], t2)

    # tmps_[vvovoo_72](b,c,i,a,j,k) = 1.00 t2(b,c,i,l) * f[ov](l,d) * t2(d,a,j,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvovoo_72"] = einsum('bcil,lajk->bciajk', t2, tmps_["ovoo_38"])

    # triples_resid += -1.00 P(i,j) P(a,b) f(l,d) t2(d,a,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bciajk->abcijk', tmps_["vvovoo_72"])
    triples_resid += einsum('bcjaik->abcijk', tmps_["vvovoo_72"])
    triples_resid += einsum('acibjk->abcijk', tmps_["vvovoo_72"])
    triples_resid -= einsum('acjbik->abcijk', tmps_["vvovoo_72"])

    # triples_resid += -1.00 P(a,b) f(l,d) t2(d,a,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bckaij->abcijk', tmps_["vvovoo_72"])
    triples_resid += einsum('ackbij->abcijk', tmps_["vvovoo_72"])

    # triples_resid += -1.00 P(i,j) f(l,d) t2(d,c,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["vvovoo_72"])
    triples_resid += einsum('abjcik->abcijk', tmps_["vvovoo_72"])

    # triples_resid += -1.00 f(l,d) t2(d,c,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abkcij->abcijk', tmps_["vvovoo_72"])
    del tmps_["vvovoo_72"]

    # tmps_[vvoo_96](a,b,i,j) = 1.00 t1(a,k) * f[ov](k,c) * t2(c,b,i,j) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_96"] = einsum('ak,kbij->abij', t1, tmps_["ovoo_38"])
    del tmps_["ovoo_38"]

    # doubles_resid += -1.00 P(a,b) f(k,c) t1(a,k) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["vvoo_96"])
    doubles_resid += einsum('baij->abij', tmps_["vvoo_96"])
    del tmps_["vvoo_96"]

    # tmps_[oo_39](l,j) = 0.50 eri[oovv](l,k,c,d) * t2(c,d,j,k) // flops: o2v0 = o3v2 | mem: o2v0 = o2v0
    tmps_["oo_39"] = 0.50 * einsum('lkcd,cdjk->lj', eri["oovv"], t2)

    # tmps_[vvvooo_76](a,b,c,i,k,j) = 1.00 t3(a,b,c,i,k,m) * eri[oovv](m,l,d,e) * t2(d,e,j,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_76"] = einsum('abcikm,mj->abcikj', t3, tmps_["oo_39"])

    # triples_resid += -0.50 P(j,k) <m,l||d,e> t2(d,e,k,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcijk->abcijk', tmps_["vvvooo_76"])
    triples_resid += einsum('abcikj->abcijk', tmps_["vvvooo_76"])

    # triples_resid += -0.50 <m,l||d,e> t2(d,e,i,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcjki->abcijk', tmps_["vvvooo_76"])
    del tmps_["vvvooo_76"]

    # tmps_[vvoo_97](a,b,j,i) = 1.00 t2(a,b,j,l) * eri[oovv](l,k,c,d) * t2(c,d,i,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_97"] = einsum('abjl,li->abji', t2, tmps_["oo_39"])
    del tmps_["oo_39"]

    # doubles_resid += -0.50 P(i,j) <l,k||c,d> t2(c,d,j,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["vvoo_97"])
    doubles_resid += einsum('abji->abij', tmps_["vvoo_97"])
    del tmps_["vvoo_97"]

    # tmps_[voov_41](b,i,j,a) = 1.00 eri[vooo](b,k,i,j) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["voov_41"] = einsum('bkij,ak->bija', eri["vooo"], t1)

    # doubles_resid += +1.00 P(a,b) <k,a||i,j> t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('aijb->abij', tmps_["voov_41"])
    doubles_resid += einsum('bija->abij', tmps_["voov_41"])
    del tmps_["voov_41"]

    # tmps_[ovvo_42](j,a,b,i) = 1.00 f[oo](k,j) * t2(a,b,i,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["ovvo_42"] = einsum('kj,abik->jabi', f["oo"], t2)

    # doubles_resid += -1.00 P(i,j) f(k,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('jabi->abij', tmps_["ovvo_42"])
    doubles_resid += einsum('iabj->abij', tmps_["ovvo_42"])
    del tmps_["ovvo_42"]

    # tmps_[oooo_43](l,k,j,i) = 1.00 eri[oovo](l,k,c,j) * t1(c,i) // flops: o4v0 = o4v1 | mem: o4v0 = o4v0
    tmps_["oooo_43"] = einsum('lkcj,ci->lkji', eri["oovo"], t1)

    # tmps_[vvvooo_52](a,b,c,i,k,j) = 0.50 t3(a,b,c,i,m,l) * eri[oovo](m,l,d,k) * t1(d,j) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_52"] = 0.50 * einsum('abciml,mlkj->abcikj', t3, tmps_["oooo_43"])

    # triples_resid += +0.50 P(i,j) <m,l||d,k> t1(d,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcikj->abcijk', tmps_["vvvooo_52"])
    triples_resid -= einsum('abcjki->abcijk', tmps_["vvvooo_52"])

    # triples_resid += -0.50 P(i,k) <m,l||d,j> t1(d,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcijk->abcijk', tmps_["vvvooo_52"])
    triples_resid += einsum('abckji->abcijk', tmps_["vvvooo_52"])

    # triples_resid += +0.50 P(j,k) <m,l||d,i> t1(d,k) t3(a,b,c,j,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcjik->abcijk', tmps_["vvvooo_52"])
    triples_resid -= einsum('abckij->abcijk', tmps_["vvvooo_52"])
    del tmps_["vvvooo_52"]

    # tmps_[vooo_100](c,m,i,k) = 1.00 t1(c,l) * eri[oovo](m,l,d,i) * t1(d,k) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["vooo_100"] = einsum('cl,mlik->cmik', t1, tmps_["oooo_43"])
    del tmps_["oooo_43"]

    # tmps_[vvovoo_106](a,b,j,c,k,i) = 1.00 t2(a,b,j,m) * t1(c,l) * eri[oovo](m,l,d,k) * t1(d,i) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvovoo_106"] = einsum('abjm,cmki->abjcki', t2, tmps_["vooo_100"])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,k> t1(d,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bciakj->abcijk', tmps_["vvovoo_106"])
    triples_resid -= einsum('bcjaki->abcijk', tmps_["vvovoo_106"])
    triples_resid -= einsum('acibkj->abcijk', tmps_["vvovoo_106"])
    triples_resid += einsum('acjbki->abcijk', tmps_["vvovoo_106"])

    # triples_resid += +1.00 P(i,j) <m,l||d,k> t1(d,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abickj->abcijk', tmps_["vvovoo_106"])
    triples_resid -= einsum('abjcki->abcijk', tmps_["vvovoo_106"])

    # triples_resid += -1.00 P(i,k) P(a,b) <m,l||d,j> t1(d,k) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bciajk->abcijk', tmps_["vvovoo_106"])
    triples_resid += einsum('bckaji->abcijk', tmps_["vvovoo_106"])
    triples_resid += einsum('acibjk->abcijk', tmps_["vvovoo_106"])
    triples_resid -= einsum('ackbji->abcijk', tmps_["vvovoo_106"])

    # triples_resid += -1.00 P(i,k) <m,l||d,j> t1(d,k) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abicjk->abcijk', tmps_["vvovoo_106"])
    triples_resid += einsum('abkcji->abcijk', tmps_["vvovoo_106"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,i> t1(d,k) t1(a,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcjaik->abcijk', tmps_["vvovoo_106"])
    triples_resid -= einsum('bckaij->abcijk', tmps_["vvovoo_106"])
    triples_resid -= einsum('acjbik->abcijk', tmps_["vvovoo_106"])
    triples_resid += einsum('ackbij->abcijk', tmps_["vvovoo_106"])

    # triples_resid += +1.00 P(j,k) <m,l||d,i> t1(d,k) t1(c,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abjcik->abcijk', tmps_["vvovoo_106"])
    triples_resid -= einsum('abkcij->abcijk', tmps_["vvovoo_106"])
    del tmps_["vvovoo_106"]

    # tmps_[vvoo_122](b,a,j,i) = 1.00 t1(b,l) * t1(a,k) * eri[oovo](l,k,c,j) * t1(c,i) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_122"] = einsum('bl,alji->baji', t1, tmps_["vooo_100"])
    del tmps_["vooo_100"]

    # doubles_resid += -1.00 P(i,j) <l,k||c,j> t1(c,i) t1(a,k) t1(b,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('baji->abij', tmps_["vvoo_122"])
    doubles_resid += einsum('baij->abij', tmps_["vvoo_122"])
    del tmps_["vvoo_122"]

    # tmps_[ooov_44](l,i,j,a) = 1.00 eri[oooo](l,k,i,j) * t1(a,k) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["ooov_44"] = einsum('lkij,ak->lija', eri["oooo"], t1)

    # doubles_resid += -1.00 <l,k||i,j> t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('lija,bl->abij', tmps_["ooov_44"], t1)

    # tmps_[vvooov_68](a,b,k,i,j,c) = 1.00 t2(a,b,k,m) * eri[oooo](m,l,i,j) * t1(c,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvooov_68"] = einsum('abkm,mijc->abkijc', t2, tmps_["ooov_44"])
    del tmps_["ooov_44"]

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||j,k> t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijka->abcijk', tmps_["vvooov_68"])
    triples_resid -= einsum('bcjika->abcijk', tmps_["vvooov_68"])
    triples_resid -= einsum('acijkb->abcijk', tmps_["vvooov_68"])
    triples_resid += einsum('acjikb->abcijk', tmps_["vvooov_68"])

    # triples_resid += +1.00 P(i,j) <m,l||j,k> t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijkc->abcijk', tmps_["vvooov_68"])
    triples_resid -= einsum('abjikc->abcijk', tmps_["vvooov_68"])

    # triples_resid += +1.00 P(a,b) <m,l||i,j> t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bckija->abcijk', tmps_["vvooov_68"])
    triples_resid -= einsum('ackijb->abcijk', tmps_["vvooov_68"])

    # triples_resid += +1.00 <m,l||i,j> t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abkijc->abcijk', tmps_["vvooov_68"])
    del tmps_["vvooov_68"]

    # tmps_[vv_45](b,d) = 1.00 eri[vovv](b,k,c,d) * t1(c,k) // flops: o0v2 = o1v3 | mem: o0v2 = o0v2
    tmps_["vv_45"] = einsum('bkcd,ck->bd', eri["vovv"], t1)

    # tmps_[vvooov_61](b,c,i,j,k,a) = 1.00 t3(e,b,c,i,j,k) * eri[vovv](a,l,d,e) * t1(d,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvooov_61"] = einsum('ebcijk,ae->bcijka', t3, tmps_["vv_45"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t1(d,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijka->abcijk', tmps_["vvooov_61"])
    triples_resid += einsum('acijkb->abcijk', tmps_["vvooov_61"])

    # triples_resid += +1.00 <l,c||d,e> t1(d,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijkc->abcijk', tmps_["vvooov_61"])
    del tmps_["vvooov_61"]

    # tmps_[voov_88](a,i,j,b) = 1.00 t2(d,a,i,j) * eri[vovv](b,k,c,d) * t1(c,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["voov_88"] = einsum('daij,bd->aijb', t2, tmps_["vv_45"])
    del tmps_["vv_45"]

    # doubles_resid += +1.00 P(a,b) <k,a||c,d> t1(c,k) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('bija->abij', tmps_["voov_88"])
    doubles_resid += einsum('aijb->abij', tmps_["voov_88"])
    del tmps_["voov_88"]

    # tmps_[ov_46](l,d) = 1.00 eri[oovv](l,k,c,d) * t1(c,k) // flops: o1v1 = o2v2 | mem: o1v1 = o1v1
    tmps_["ov_46"] = einsum('lkcd,ck->ld', eri["oovv"], t1)

    # doubles_resid += -1.00 <l,k||c,d> t1(c,k) t3(d,a,b,i,j,l)  // flops: o2v2 += o3v3 | mem: o2v2 += o2v2
    doubles_resid -= einsum('ld,dabijl->abij', tmps_["ov_46"], t3)

    # tmps_[ovoo_90](m,b,i,j) = 1.00 eri[oovv](m,l,d,e) * t1(d,l) * t2(e,b,i,j) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["ovoo_90"] = einsum('me,ebij->mbij', tmps_["ov_46"], t2)

    # tmps_[voovvo_110](c,j,k,a,b,i) = 1.00 eri[oovv](m,l,d,e) * t1(d,l) * t2(e,c,j,k) * t2(a,b,i,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["voovvo_110"] = einsum('mcjk,abim->cjkabi', tmps_["ovoo_90"], t2)

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,l) t2(e,a,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ajkbci->abcijk', tmps_["voovvo_110"])
    triples_resid -= einsum('aikbcj->abcijk', tmps_["voovvo_110"])
    triples_resid -= einsum('bjkaci->abcijk', tmps_["voovvo_110"])
    triples_resid += einsum('bikacj->abcijk', tmps_["voovvo_110"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,l) t2(e,a,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aijbck->abcijk', tmps_["voovvo_110"])
    triples_resid -= einsum('bijack->abcijk', tmps_["voovvo_110"])

    # triples_resid += +1.00 P(i,j) <m,l||d,e> t1(d,l) t2(e,c,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cjkabi->abcijk', tmps_["voovvo_110"])
    triples_resid -= einsum('cikabj->abcijk', tmps_["voovvo_110"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t2(e,c,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cijabk->abcijk', tmps_["voovvo_110"])
    del tmps_["voovvo_110"]

    # tmps_[voov_123](b,i,j,a) = 1.00 eri[oovv](l,k,c,d) * t1(c,k) * t2(d,b,i,j) * t1(a,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["voov_123"] = einsum('lbij,al->bija', tmps_["ovoo_90"], t1)
    del tmps_["ovoo_90"]

    # doubles_resid += +1.00 P(a,b) <l,k||c,d> t1(c,k) t1(a,l) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('bija->abij', tmps_["voov_123"])
    doubles_resid -= einsum('aijb->abij', tmps_["voov_123"])
    del tmps_["voov_123"]

    # tmps_[vvvooo_118](c,a,b,i,j,k) = 1.00 t3(e,a,b,i,j,k) * eri[oovv](m,l,d,e) * t1(d,l) * t1(c,m) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["vvvooo_118"] = einsum('eabijk,me,cm->cabijk', t3, tmps_["ov_46"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["ov_46"]

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,l) t1(a,m) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["vvvooo_118"])
    triples_resid -= einsum('bacijk->abcijk', tmps_["vvvooo_118"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t1(c,m) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cabijk->abcijk', tmps_["vvvooo_118"])
    del tmps_["vvvooo_118"]

    # tmps_[oo_47](l,i) = 1.00 eri[oovo](l,k,c,i) * t1(c,k) // flops: o2v0 = o3v1 | mem: o2v0 = o2v0
    tmps_["oo_47"] = einsum('lkci,ck->li', eri["oovo"], t1)

    # singles_resid += +1.00 <k,j||b,i> t1(b,j) t1(a,k)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid += einsum('ki,ak->ai', tmps_["oo_47"], t1)

    # tmps_[ovvvoo_75](i,a,b,c,j,k) = 1.00 eri[oovo](m,l,d,i) * t1(d,l) * t3(a,b,c,j,k,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["ovvvoo_75"] = einsum('mi,abcjkm->iabcjk', tmps_["oo_47"], t3)

    # triples_resid += +1.00 P(j,k) <m,l||d,k> t1(d,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kabcij->abcijk', tmps_["ovvvoo_75"])
    triples_resid -= einsum('jabcik->abcijk', tmps_["ovvvoo_75"])

    # triples_resid += +1.00 <m,l||d,i> t1(d,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('iabcjk->abcijk', tmps_["ovvvoo_75"])
    del tmps_["ovvvoo_75"]

    # tmps_[vvoo_98](a,b,i,j) = 1.00 t2(a,b,i,l) * eri[oovo](l,k,c,j) * t1(c,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_98"] = einsum('abil,lj->abij', t2, tmps_["oo_47"])
    del tmps_["oo_47"]

    # doubles_resid += +1.00 P(i,j) <l,k||c,j> t1(c,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["vvoo_98"])
    doubles_resid -= einsum('abji->abij', tmps_["vvoo_98"])
    del tmps_["vvoo_98"]

    # tmps_[oo_48](k,j) = 1.00 f[ov](k,c) * t1(c,j) // flops: o2v0 = o2v1 | mem: o2v0 = o2v0
    tmps_["oo_48"] = einsum('kc,cj->kj', f["ov"], t1)

    # singles_resid += -1.00 f(j,b) t1(b,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid -= einsum('ji,aj->ai', tmps_["oo_48"], t1)

    # tmps_[ovvvoo_77](k,a,b,c,i,j) = 1.00 f[ov](l,d) * t1(d,k) * t3(a,b,c,i,j,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["ovvvoo_77"] = einsum('lk,abcijl->kabcij', tmps_["oo_48"], t3)

    # triples_resid += -1.00 P(j,k) f(l,d) t1(d,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('kabcij->abcijk', tmps_["ovvvoo_77"])
    triples_resid += einsum('jabcik->abcijk', tmps_["ovvvoo_77"])

    # triples_resid += -1.00 f(l,d) t1(d,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('iabcjk->abcijk', tmps_["ovvvoo_77"])
    del tmps_["ovvvoo_77"]

    # tmps_[vvoo_99](a,b,j,i) = 1.00 t2(a,b,j,k) * f[ov](k,c) * t1(c,i) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_99"] = einsum('abjk,ki->abji', t2, tmps_["oo_48"])
    del tmps_["oo_48"]

    # doubles_resid += -1.00 P(i,j) f(k,c) t1(c,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["vvoo_99"])
    doubles_resid += einsum('abji->abij', tmps_["vvoo_99"])
    del tmps_["vvoo_99"]

    # tmps_[vvoovo_49](b,c,i,j,a,k) = 1.00 eri[oovv](m,l,d,e) * t2(d,a,k,l) * t3(e,b,c,i,j,m) // flops: o3v3 = o3v3 o4v4 | mem: o3v3 = o2v2 o3v3
    tmps_["vvoovo_49"] = einsum('mlde,dakl,ebcijm->bcijak', eri["oovv"], t2, t3, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,e> t2(d,a,k,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcijak->abcijk', tmps_["vvoovo_49"])
    triples_resid -= einsum('bcikaj->abcijk', tmps_["vvoovo_49"])
    triples_resid -= einsum('acijbk->abcijk', tmps_["vvoovo_49"])
    triples_resid += einsum('acikbj->abcijk', tmps_["vvoovo_49"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t2(d,a,i,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcjkai->abcijk', tmps_["vvoovo_49"])
    triples_resid -= einsum('acjkbi->abcijk', tmps_["vvoovo_49"])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t2(d,c,k,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abijck->abcijk', tmps_["vvoovo_49"])
    triples_resid -= einsum('abikcj->abcijk', tmps_["vvoovo_49"])

    # triples_resid += +1.00 <m,l||d,e> t2(d,c,i,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abjkci->abcijk', tmps_["vvoovo_49"])
    del tmps_["vvoovo_49"]

    # tmps_[vvoovo_55](a,c,j,k,b,i) = 1.00 eri[vovv](a,l,d,e) * t2(d,b,i,l) * t2(e,c,j,k) // flops: o3v3 = o2v4 o3v4 | mem: o3v3 = o1v3 o3v3
    tmps_["vvoovo_55"] = einsum('alde,dbil,ecjk->acjkbi', eri["vovv"], t2, t2, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 P(j,k) P(a,b) <l,a||d,e> t2(d,b,k,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acijbk->abcijk', tmps_["vvoovo_55"])
    triples_resid += einsum('acikbj->abcijk', tmps_["vvoovo_55"])
    triples_resid += einsum('bcijak->abcijk', tmps_["vvoovo_55"])
    triples_resid -= einsum('bcikaj->abcijk', tmps_["vvoovo_55"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acjkbi->abcijk', tmps_["vvoovo_55"])
    triples_resid += einsum('bcjkai->abcijk', tmps_["vvoovo_55"])

    # triples_resid += +1.00 P(j,k) <l,c||d,e> t2(d,a,k,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbijak->abcijk', tmps_["vvoovo_55"])
    triples_resid += einsum('cbikaj->abcijk', tmps_["vvoovo_55"])

    # triples_resid += +1.00 <l,c||d,e> t2(d,a,i,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbjkai->abcijk', tmps_["vvoovo_55"])
    del tmps_["vvoovo_55"]

    # tmps_[vovoov_63](a,j,b,i,k,c) = 1.00 eri[vovo](a,l,d,j) * t2(d,b,i,k) * t1(c,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["vovoov_63"] = einsum('aldj,dbik,cl->ajbikc', eri["vovo"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += -1.00 P(j,k) P(b,c) <l,a||d,k> t1(b,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('akcijb->abcijk', tmps_["vovoov_63"])
    triples_resid -= einsum('ajcikb->abcijk', tmps_["vovoov_63"])
    triples_resid -= einsum('akbijc->abcijk', tmps_["vovoov_63"])
    triples_resid += einsum('ajbikc->abcijk', tmps_["vovoov_63"])

    # triples_resid += -1.00 P(b,c) <l,a||d,i> t1(b,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('aicjkb->abcijk', tmps_["vovoov_63"])
    triples_resid -= einsum('aibjkc->abcijk', tmps_["vovoov_63"])

    # triples_resid += +1.00 P(j,k) P(a,c) <l,b||d,k> t1(a,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bkcija->abcijk', tmps_["vovoov_63"])
    triples_resid += einsum('bjcika->abcijk', tmps_["vovoov_63"])
    triples_resid += einsum('bkaijc->abcijk', tmps_["vovoov_63"])
    triples_resid -= einsum('bjaikc->abcijk', tmps_["vovoov_63"])

    # triples_resid += +1.00 P(a,c) <l,b||d,i> t1(a,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bicjka->abcijk', tmps_["vovoov_63"])
    triples_resid += einsum('biajkc->abcijk', tmps_["vovoov_63"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,c||d,k> t1(a,l) t2(d,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ckbija->abcijk', tmps_["vovoov_63"])
    triples_resid -= einsum('cjbika->abcijk', tmps_["vovoov_63"])
    triples_resid -= einsum('ckaijb->abcijk', tmps_["vovoov_63"])
    triples_resid += einsum('cjaikb->abcijk', tmps_["vovoov_63"])

    # triples_resid += -1.00 P(a,b) <l,c||d,i> t1(a,l) t2(d,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cibjka->abcijk', tmps_["vovoov_63"])
    triples_resid -= einsum('ciajkb->abcijk', tmps_["vovoov_63"])
    del tmps_["vovoov_63"]

    # tmps_[ovvovo_64](k,b,c,i,a,j) = 1.00 eri[oovo](m,l,d,k) * t2(d,a,j,l) * t2(b,c,i,m) // flops: o3v3 = o4v2 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["ovvovo_64"] = einsum('mldk,dajl,bcim->kbciaj', eri["oovo"], t2, t2, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,k> t2(d,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kbciaj->abcijk', tmps_["ovvovo_64"])
    triples_resid -= einsum('kbcjai->abcijk', tmps_["ovvovo_64"])
    triples_resid -= einsum('kacibj->abcijk', tmps_["ovvovo_64"])
    triples_resid += einsum('kacjbi->abcijk', tmps_["ovvovo_64"])

    # triples_resid += +1.00 P(i,j) <m,l||d,k> t2(d,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kabicj->abcijk', tmps_["ovvovo_64"])
    triples_resid -= einsum('kabjci->abcijk', tmps_["ovvovo_64"])

    # triples_resid += -1.00 P(i,k) P(a,b) <m,l||d,j> t2(d,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('jbciak->abcijk', tmps_["ovvovo_64"])
    triples_resid += einsum('jbckai->abcijk', tmps_["ovvovo_64"])
    triples_resid += einsum('jacibk->abcijk', tmps_["ovvovo_64"])
    triples_resid -= einsum('jackbi->abcijk', tmps_["ovvovo_64"])

    # triples_resid += -1.00 P(i,k) <m,l||d,j> t2(d,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('jabick->abcijk', tmps_["ovvovo_64"])
    triples_resid += einsum('jabkci->abcijk', tmps_["ovvovo_64"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,i> t2(d,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ibcjak->abcijk', tmps_["ovvovo_64"])
    triples_resid -= einsum('ibckaj->abcijk', tmps_["ovvovo_64"])
    triples_resid -= einsum('iacjbk->abcijk', tmps_["ovvovo_64"])
    triples_resid += einsum('iackbj->abcijk', tmps_["ovvovo_64"])

    # triples_resid += +1.00 P(j,k) <m,l||d,i> t2(d,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('iabjck->abcijk', tmps_["ovvovo_64"])
    triples_resid -= einsum('iabkcj->abcijk', tmps_["ovvovo_64"])
    del tmps_["ovvovo_64"]

    # tmps_[ovvoov_69](k,b,c,i,j,a) = 1.00 eri[oovo](m,l,d,k) * t3(d,b,c,i,j,m) * t1(a,l) // flops: o3v3 = o5v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["ovvoov_69"] = einsum('mldk,dbcijm,al->kbcija', eri["oovo"], t3, t1, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,k> t1(a,l) t3(d,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('kbcija->abcijk', tmps_["ovvoov_69"])
    triples_resid += einsum('jbcika->abcijk', tmps_["ovvoov_69"])
    triples_resid += einsum('kacijb->abcijk', tmps_["ovvoov_69"])
    triples_resid -= einsum('jacikb->abcijk', tmps_["ovvoov_69"])

    # triples_resid += -1.00 P(j,k) <m,l||d,k> t1(c,l) t3(d,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('kabijc->abcijk', tmps_["ovvoov_69"])
    triples_resid += einsum('jabikc->abcijk', tmps_["ovvoov_69"])

    # triples_resid += -1.00 P(a,b) <m,l||d,i> t1(a,l) t3(d,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('ibcjka->abcijk', tmps_["ovvoov_69"])
    triples_resid += einsum('iacjkb->abcijk', tmps_["ovvoov_69"])

    # triples_resid += -1.00 <m,l||d,i> t1(c,l) t3(d,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('iabjkc->abcijk', tmps_["ovvoov_69"])
    del tmps_["ovvoov_69"]

    # tmps_[vvvooo_73](a,b,c,i,j,k) = 0.50 eri[vovv](a,l,d,e) * t3(d,e,c,i,j,k) * t1(b,l) // flops: o3v3 = o4v4 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["vvvooo_73"] = 0.50 * einsum('alde,decijk,bl->abcijk', eri["vovv"], t3, t1, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +0.50 P(b,c) <l,a||d,e> t1(b,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcijk->abcijk', tmps_["vvvooo_73"])
    triples_resid += einsum('acbijk->abcijk', tmps_["vvvooo_73"])

    # triples_resid += -0.50 P(a,c) <l,b||d,e> t1(a,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bacijk->abcijk', tmps_["vvvooo_73"])
    triples_resid -= einsum('bcaijk->abcijk', tmps_["vvvooo_73"])

    # triples_resid += +0.50 P(a,b) <l,c||d,e> t1(a,l) t3(d,e,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabijk->abcijk', tmps_["vvvooo_73"])
    triples_resid += einsum('cbaijk->abcijk', tmps_["vvvooo_73"])
    del tmps_["vvvooo_73"]

    # tmps_[vvooov_78](b,c,i,j,k,a) = 1.00 f[ov](l,d) * t3(d,b,c,i,j,k) * t1(a,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["vvooov_78"] = einsum('ld,dbcijk,al->bcijka', f["ov"], t3, t1, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += -1.00 P(a,b) f(l,d) t1(a,l) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcijka->abcijk', tmps_["vvooov_78"])
    triples_resid += einsum('acijkb->abcijk', tmps_["vvooov_78"])

    # triples_resid += -1.00 f(l,d) t1(c,l) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abijkc->abcijk', tmps_["vvooov_78"])
    del tmps_["vvooov_78"]

    # tmps_[ovvo_92](i,a,b,j) = 1.00 eri[oovo](l,k,c,i) * t2(c,b,j,l) * t1(a,k) // flops: o2v2 = o4v2 o3v2 | mem: o2v2 = o3v1 o2v2
    tmps_["ovvo_92"] = einsum('lkci,cbjl,ak->iabj', eri["oovo"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

    # doubles_resid += -1.00 P(i,j) P(a,b) <l,k||c,j> t1(a,k) t2(c,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('jabi->abij', tmps_["ovvo_92"])
    doubles_resid += einsum('iabj->abij', tmps_["ovvo_92"])
    doubles_resid += einsum('jbai->abij', tmps_["ovvo_92"])
    doubles_resid -= einsum('ibaj->abij', tmps_["ovvo_92"])
    del tmps_["ovvo_92"]

    # tmps_[oo_103](l,i) = 1.00 eri[oovv](l,k,c,d) * t1(c,k) * t1(d,i) // flops: o2v0 = o2v2 o2v1 | mem: o2v0 = o1v1 o2v0
    tmps_["oo_103"] = einsum('lkcd,ck,di->li', eri["oovv"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])

    # singles_resid += +1.00 <k,j||b,c> t1(b,j) t1(c,i) t1(a,k)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid += einsum('ki,ak->ai', tmps_["oo_103"], t1)

    # tmps_[vvvooo_116](a,b,c,j,k,i) = 1.00 t3(a,b,c,j,k,m) * eri[oovv](m,l,d,e) * t1(d,l) * t1(e,i) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_116"] = einsum('abcjkm,mi->abcjki', t3, tmps_["oo_103"])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t1(d,l) t1(e,k) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["vvvooo_116"])
    triples_resid -= einsum('abcikj->abcijk', tmps_["vvvooo_116"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t1(e,i) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcjki->abcijk', tmps_["vvvooo_116"])
    del tmps_["vvvooo_116"]

    # tmps_[vvoo_121](a,b,j,i) = 1.00 t2(a,b,j,l) * eri[oovv](l,k,c,d) * t1(c,k) * t1(d,i) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_121"] = einsum('abjl,li->abji', t2, tmps_["oo_103"])
    del tmps_["oo_103"]

    # doubles_resid += +1.00 P(i,j) <l,k||c,d> t1(c,k) t1(d,j) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["vvoo_121"])
    doubles_resid -= einsum('abji->abij', tmps_["vvoo_121"])
    del tmps_["vvoo_121"]

    # tmps_[ovvoov_112](k,b,c,i,j,a) = 1.00 eri[oovo](m,l,d,k) * t2(d,c,i,j) * t1(b,m) * t1(a,l) // flops: o3v3 = o5v2 o5v2 o4v3 | mem: o3v3 = o5v1 o4v2 o3v3
    tmps_["ovvoov_112"] = einsum('mldk,dcij,bm,al->kbcija', eri["oovo"], t2, t1, t1, optimize=['einsum_path',(0,1),(0,1),(0,1)])

    # triples_resid += +1.00 P(j,k) P(b,c) <m,l||d,k> t1(a,l) t1(b,m) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kbcija->abcijk', tmps_["ovvoov_112"])
    triples_resid -= einsum('jbcika->abcijk', tmps_["ovvoov_112"])
    triples_resid -= einsum('kcbija->abcijk', tmps_["ovvoov_112"])
    triples_resid += einsum('jcbika->abcijk', tmps_["ovvoov_112"])

    # triples_resid += +1.00 P(j,k) <m,l||d,k> t1(b,l) t1(c,m) t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('kcaijb->abcijk', tmps_["ovvoov_112"])
    triples_resid -= einsum('jcaikb->abcijk', tmps_["ovvoov_112"])

    # triples_resid += +1.00 P(b,c) <m,l||d,i> t1(a,l) t1(b,m) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('ibcjka->abcijk', tmps_["ovvoov_112"])
    triples_resid -= einsum('icbjka->abcijk', tmps_["ovvoov_112"])

    # triples_resid += +1.00 <m,l||d,i> t1(b,l) t1(c,m) t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('icajkb->abcijk', tmps_["ovvoov_112"])
    del tmps_["ovvoov_112"]

    # tmps_[vvoovo_113](a,c,j,k,b,i) = 1.00 eri[oovv](m,l,d,e) * t2(d,b,i,m) * t2(e,c,j,k) * t1(a,l) // flops: o3v3 = o3v3 o4v3 o4v3 | mem: o3v3 = o2v2 o4v2 o3v3
    tmps_["vvoovo_113"] = einsum('mlde,dbim,ecjk,al->acjkbi', eri["oovv"], t2, t2, t1, optimize=['einsum_path',(0,1),(0,1),(0,1)])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,k,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acijbk->abcijk', tmps_["vvoovo_113"])
    triples_resid += einsum('acikbj->abcijk', tmps_["vvoovo_113"])
    triples_resid += einsum('bcijak->abcijk', tmps_["vvoovo_113"])
    triples_resid -= einsum('bcikaj->abcijk', tmps_["vvoovo_113"])

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acjkbi->abcijk', tmps_["vvoovo_113"])
    triples_resid += einsum('bcjkai->abcijk', tmps_["vvoovo_113"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(c,l) t2(d,a,k,m) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbijak->abcijk', tmps_["vvoovo_113"])
    triples_resid += einsum('cbikaj->abcijk', tmps_["vvoovo_113"])

    # triples_resid += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,m) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbjkai->abcijk', tmps_["vvoovo_113"])
    del tmps_["vvoovo_113"]

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



