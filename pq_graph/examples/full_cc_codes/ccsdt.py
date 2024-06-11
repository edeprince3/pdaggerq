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

    # singles_residual = +1.00 f(a,i)  // flops: o1v1 = o1v1 | mem: o1v1 = o1v1
    singles_residual = 1.00 * einsum('ai->ai', f["vo"])

    # doubles_residual = +1.00 <a,b||i,j>  // flops: o2v2 = o2v2 | mem: o2v2 = o2v2
    doubles_residual = 1.00 * einsum('abij->abij', eri["vvoo"])

    # singles_residual += -1.00 f(j,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_residual -= einsum('aj,ji->ai', t1, f["oo"])

    # singles_residual += +1.00 f(a,b) t1(b,i)  // flops: o1v1 += o1v2 | mem: o1v1 += o1v1
    singles_residual += einsum('ab,bi->ai', f["vv"], t1)

    # singles_residual += -1.00 f(j,b) t2(b,a,i,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_residual -= einsum('jb,baij->ai', f["ov"], t2)

    # singles_residual += +1.00 <j,a||b,i> t1(b,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_residual -= einsum('ajbi,bj->ai', eri["vovo"], t1)

    # singles_residual += -0.50 <k,j||b,i> t2(b,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
    singles_residual -= 0.50 * einsum('bakj,kjbi->ai', t2, eri["oovo"])

    # singles_residual += -0.50 <j,a||b,c> t2(b,c,i,j)  // flops: o1v1 += o2v3 | mem: o1v1 += o1v1
    singles_residual += 0.50 * einsum('ajbc,bcij->ai', eri["vovv"], t2)

    # doubles_residual += +0.50 <l,k||i,j> t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
    doubles_residual += 0.50 * einsum('ablk,lkij->abij', t2, eri["oooo"])

    # doubles_residual += +0.50 <a,b||c,d> t2(c,d,i,j)  // flops: o2v2 += o2v4 | mem: o2v2 += o2v2
    doubles_residual += 0.50 * einsum('abcd,cdij->abij', eri["vvvv"], t2)

    # singles_residual += +0.50 <k,j||b,c> t1(a,j) t2(b,c,i,k)  // flops: o1v1 += o3v2 o2v1 | mem: o1v1 += o2v0 o1v1
    singles_residual += 0.50 * einsum('kjbc,bcik,aj->ai', eri["oovv"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

    # doubles_residual += -0.50 <l,k||c,d> t2(c,a,i,j) t2(d,b,l,k)  // flops: o2v2 += o2v3 o2v3 | mem: o2v2 += o0v2 o2v2
    doubles_residual -= 0.50 * einsum('lkcd,dblk,caij->abij', eri["oovv"], t2, t2, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t3"]:

        # doubles_residual += +1.00 f(k,c) t3(c,a,b,i,j,k)  // flops: o2v2 += o3v3 | mem: o2v2 += o2v2
        doubles_residual += einsum('kc,cabijk->abij', f["ov"], t3)

        # singles_residual += +0.25 <k,j||b,c> t3(b,c,a,i,k,j)  // flops: o1v1 += o3v3 | mem: o1v1 += o1v1
        singles_residual += 0.25 * einsum('kjbc,bcaikj->ai', eri["oovv"], t3)

        # tmps_[1_vvvooo](b,c,a,i,j,k) = 0.50 eri[vvvv](b,c,d,e) * t3(d,e,a,i,j,k) // flops: o3v3 = o3v5 | mem: o3v3 = o3v3
        tmps_["1_vvvooo"] = 0.50 * einsum('bcde,deaijk->bcaijk', eri["vvvv"], t3)

        # triples_residual = +0.50 P(b,c) <a,b||d,e> t3(d,e,c,i,j,k)  // flops: o3v3 = o3v3 | mem: o3v3 = o3v3
        triples_residual = 1.00 * einsum('abcijk->abcijk', tmps_["1_vvvooo"])
        triples_residual -= einsum('acbijk->abcijk', tmps_["1_vvvooo"])

        # triples_residual += +0.50 <b,c||d,e> t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bcaijk->abcijk', tmps_["1_vvvooo"])
        del tmps_["1_vvvooo"]

        # tmps_[2_vovvoo](b,j,a,c,i,k) = 1.00 eri[vovo](b,l,d,j) * t3(d,a,c,i,k,l) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
        tmps_["2_vovvoo"] = einsum('bldj,dacikl->bjacik', eri["vovo"], t3)
        triples_residual += einsum('cjabik->abcijk', tmps_["2_vovvoo"])
        triples_residual += einsum('biacjk->abcijk', tmps_["2_vovvoo"])

        # triples_residual += +1.00 P(j,k) <l,c||d,k> t3(d,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('ckabij->abcijk', tmps_["2_vovvoo"])

        # triples_residual += +1.00 P(j,k) P(a,b) <l,a||d,k> t3(d,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('akbcij->abcijk', tmps_["2_vovvoo"])
        triples_residual -= einsum('bjacik->abcijk', tmps_["2_vovvoo"])

        # triples_residual += +1.00 <l,c||d,i> t3(d,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('ciabjk->abcijk', tmps_["2_vovvoo"])
        triples_residual += einsum('ajbcik->abcijk', tmps_["2_vovvoo"])

        # triples_residual += +1.00 P(a,b) <l,a||d,i> t3(d,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('aibcjk->abcijk', tmps_["2_vovvoo"])
        triples_residual += einsum('bkacij->abcijk', tmps_["2_vovvoo"])
        del tmps_["2_vovvoo"]

        # tmps_[3_vovooo](c,l,b,i,j,k) = 0.50 eri[vovv](c,l,d,e) * t3(d,e,b,i,j,k) // flops: o4v2 = o4v4 | mem: o4v2 = o4v2
        tmps_["3_vovooo"] = 0.50 * einsum('clde,debijk->clbijk', eri["vovv"], t3)

        # tmps_[78_vvvooo](b,a,c,i,j,k) = 1.00 t1(b,l) * eri[vovv](a,l,d,e) * t3(d,e,c,i,j,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["78_vvvooo"] = einsum('bl,alcijk->bacijk', t1, tmps_["3_vovooo"])
        del tmps_["3_vovooo"]

        # triples_residual += -0.50 P(a,c) <l,b||d,e> t1(a,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcijk->abcijk', tmps_["78_vvvooo"])

        # triples_residual += +0.50 P(a,b) <l,c||d,e> t1(a,l) t3(d,e,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbijk->abcijk', tmps_["78_vvvooo"])
        triples_residual += einsum('cabijk->abcijk', tmps_["78_vvvooo"])

        # triples_residual += +0.50 P(b,c) <l,a||d,e> t1(b,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bacijk->abcijk', tmps_["78_vvvooo"])
        triples_residual += einsum('bcaijk->abcijk', tmps_["78_vvvooo"])
        triples_residual -= einsum('cbaijk->abcijk', tmps_["78_vvvooo"])
        del tmps_["78_vvvooo"]

        # tmps_[4_oovvoo](l,j,a,c,i,k) = 1.00 eri[oovo](m,l,d,j) * t3(d,a,c,i,k,m) // flops: o4v2 = o5v3 | mem: o4v2 = o4v2
        tmps_["4_oovvoo"] = einsum('mldj,dacikm->ljacik', eri["oovo"], t3)

        # tmps_[75_vovvoo](b,i,a,c,j,k) = 1.00 t1(b,l) * eri[oovo](m,l,d,i) * t3(d,a,c,j,k,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["75_vovvoo"] = einsum('bl,liacjk->biacjk', t1, tmps_["4_oovvoo"])
        del tmps_["4_oovvoo"]
        triples_residual += einsum('bkacij->abcijk', tmps_["75_vovvoo"])

        # triples_residual += -1.00 P(a,b) <m,l||d,i> t1(a,l) t3(d,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('aibcjk->abcijk', tmps_["75_vovvoo"])

        # triples_residual += -1.00 <m,l||d,i> t1(c,l) t3(d,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('ciabjk->abcijk', tmps_["75_vovvoo"])

        # triples_residual += -1.00 P(j,k) P(a,b) <m,l||d,k> t1(a,l) t3(d,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('akbcij->abcijk', tmps_["75_vovvoo"])
        triples_residual += einsum('cjabik->abcijk', tmps_["75_vovvoo"])
        triples_residual += einsum('biacjk->abcijk', tmps_["75_vovvoo"])
        triples_residual += einsum('ajbcik->abcijk', tmps_["75_vovvoo"])
        triples_residual -= einsum('bjacik->abcijk', tmps_["75_vovvoo"])

        # triples_residual += -1.00 P(j,k) <m,l||d,k> t1(c,l) t3(d,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('ckabij->abcijk', tmps_["75_vovvoo"])
        del tmps_["75_vovvoo"]

        # tmps_[5_oovooo](m,l,c,i,j,k) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,c,i,j,k) // flops: o5v1 = o5v3 | mem: o5v1 = o5v1
        tmps_["5_oovooo"] = 0.50 * einsum('mlde,decijk->mlcijk', eri["oovv"], t3)

        # tmps_[52_vooovv](a,i,j,k,b,c) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,a,i,j,k) * t2(b,c,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
        tmps_["52_vooovv"] = 0.50 * einsum('mlaijk,bcml->aijkbc', tmps_["5_oovooo"], t2)

        # triples_residual += +0.25 P(b,c) <m,l||d,e> t2(a,b,m,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cijkab->abcijk', tmps_["52_vooovv"])

        # triples_residual += +0.25 <m,l||d,e> t2(b,c,m,l) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('aijkbc->abcijk', tmps_["52_vooovv"])
        triples_residual -= einsum('bijkac->abcijk', tmps_["52_vooovv"])
        del tmps_["52_vooovv"]

        # tmps_[80_vooovv](b,i,j,k,a,c) = 1.00 eri[oovv](m,l,d,e) * t3(d,e,b,i,j,k) * t1(a,l) * t1(c,m) // flops: o3v3 = o5v2 o4v3 | mem: o3v3 = o4v2 o3v3
        tmps_["80_vooovv"] = einsum('mlbijk,al,cm->bijkac', tmps_["5_oovooo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])
        del tmps_["5_oovooo"]

        # triples_residual += -0.50 P(b,c) <m,l||d,e> t1(a,l) t1(b,m) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cijkab->abcijk', tmps_["80_vooovv"])
        triples_residual += einsum('bijkac->abcijk', tmps_["80_vooovv"])

        # triples_residual += -0.50 <m,l||d,e> t1(b,l) t1(c,m) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('aijkbc->abcijk', tmps_["80_vooovv"])
        del tmps_["80_vooovv"]

        # tmps_[6_oovvvo](j,k,a,b,c,i) = 0.50 eri[oooo](m,l,j,k) * t3(a,b,c,i,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
        tmps_["6_oovvvo"] = 0.50 * einsum('mljk,abciml->jkabci', eri["oooo"], t3)

        # triples_residual += +0.50 <m,l||i,j> t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('ijabck->abcijk', tmps_["6_oovvvo"])
        triples_residual -= einsum('ikabcj->abcijk', tmps_["6_oovvvo"])

        # triples_residual += +0.50 P(i,j) <m,l||j,k> t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('jkabci->abcijk', tmps_["6_oovvvo"])
        del tmps_["6_oovvvo"]

        # tmps_[7_vvvo](d,a,c,i) = 0.50 eri[oovv](m,l,d,e) * t3(e,a,c,i,m,l) // flops: o1v3 = o3v4 | mem: o1v3 = o1v3
        tmps_["7_vvvo"] = 0.50 * einsum('mlde,eaciml->daci', eri["oovv"], t3)

        # tmps_[54_vvovoo](a,b,k,c,i,j) = 1.00 eri[oovv](m,l,d,e) * t3(e,a,b,k,m,l) * t2(d,c,i,j) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
        tmps_["54_vvovoo"] = einsum('dabk,dcij->abkcij', tmps_["7_vvvo"], t2)

        # triples_residual += -0.50 P(a,b) <m,l||d,e> t2(d,a,i,j) t3(e,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bckaij->abcijk', tmps_["54_vvovoo"])

        # triples_residual += -0.50 P(i,j) P(a,b) <m,l||d,e> t2(d,a,j,k) t3(e,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bciajk->abcijk', tmps_["54_vvovoo"])
        triples_residual += einsum('bcjaik->abcijk', tmps_["54_vvovoo"])
        triples_residual += einsum('acibjk->abcijk', tmps_["54_vvovoo"])

        # triples_residual += -0.50 P(i,j) <m,l||d,e> t2(d,c,j,k) t3(e,a,b,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abicjk->abcijk', tmps_["54_vvovoo"])
        triples_residual += einsum('ackbij->abcijk', tmps_["54_vvovoo"])
        triples_residual += einsum('abjcik->abcijk', tmps_["54_vvovoo"])

        # triples_residual += -0.50 <m,l||d,e> t2(d,c,i,j) t3(e,a,b,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abkcij->abcijk', tmps_["54_vvovoo"])
        triples_residual -= einsum('acjbik->abcijk', tmps_["54_vvovoo"])
        del tmps_["54_vvovoo"]

        # tmps_[91_vvoo](a,b,j,i) = 1.00 eri[oovv](l,k,c,d) * t3(d,a,b,j,l,k) * t1(c,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
        tmps_["91_vvoo"] = einsum('cabj,ci->abji', tmps_["7_vvvo"], t1)
        del tmps_["7_vvvo"]

        # doubles_residual += -0.50 P(i,j) <l,k||c,d> t1(c,j) t3(d,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('abij->abij', tmps_["91_vvoo"])
        doubles_residual += einsum('abji->abij', tmps_["91_vvoo"])
        del tmps_["91_vvoo"]

    # tmps_[8_vvovoo](a,b,i,c,j,k) = 1.00 eri[vvvo](a,b,d,i) * t2(d,c,j,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["8_vvovoo"] = einsum('abdi,dcjk->abicjk', eri["vvvo"], t2)
    triples_residual += einsum('abjcik->abcijk', tmps_["8_vvovoo"])

    # triples_residual += -1.00 P(j,k) P(b,c) <a,b||d,k> t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('abkcij->abcijk', tmps_["8_vvovoo"])

    # triples_residual += -1.00 P(j,k) <b,c||d,k> t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bckaij->abcijk', tmps_["8_vvovoo"])
    triples_residual -= einsum('acjbik->abcijk', tmps_["8_vvovoo"])
    triples_residual += einsum('bcjaik->abcijk', tmps_["8_vvovoo"])
    triples_residual += einsum('ackbij->abcijk', tmps_["8_vvovoo"])

    # triples_residual += -1.00 P(b,c) <a,b||d,i> t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('abicjk->abcijk', tmps_["8_vvovoo"])
    triples_residual += einsum('acibjk->abcijk', tmps_["8_vvovoo"])

    # triples_residual += -1.00 <b,c||d,i> t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bciajk->abcijk', tmps_["8_vvovoo"])
    del tmps_["8_vvovoo"]

    if includes_["t3"]:

        # tmps_[9_vvvooo](b,a,c,i,j,k) = 1.00 f[vv](b,d) * t3(d,a,c,i,j,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
        tmps_["9_vvvooo"] = einsum('bd,dacijk->bacijk', f["vv"], t3)
        triples_residual -= einsum('bacijk->abcijk', tmps_["9_vvvooo"])

        # triples_residual += +1.00 f(c,d) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cabijk->abcijk', tmps_["9_vvvooo"])

        # triples_residual += +1.00 P(a,b) f(a,d) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcijk->abcijk', tmps_["9_vvvooo"])
        del tmps_["9_vvvooo"]

        # tmps_[10_vvoo](a,b,i,j) = 0.50 eri[vovv](a,k,c,d) * t3(c,d,b,i,j,k) // flops: o2v2 = o3v4 | mem: o2v2 = o2v2
        tmps_["10_vvoo"] = 0.50 * einsum('akcd,cdbijk->abij', eri["vovv"], t3)
        doubles_residual += einsum('baij->abij', tmps_["10_vvoo"])

        # doubles_residual += +0.50 P(a,b) <k,a||c,d> t3(c,d,b,i,j,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('abij->abij', tmps_["10_vvoo"])
        del tmps_["10_vvoo"]

    # tmps_[11_voovoo](b,l,i,a,j,k) = 1.00 eri[vovo](b,l,d,i) * t2(d,a,j,k) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
    tmps_["11_voovoo"] = einsum('bldi,dajk->bliajk', eri["vovo"], t2)

    # tmps_[66_vovoov](b,i,a,j,k,c) = 1.00 eri[vovo](b,l,d,i) * t2(d,a,j,k) * t1(c,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["66_vovoov"] = einsum('bliajk,cl->biajkc', tmps_["11_voovoo"], t1)
    del tmps_["11_voovoo"]

    # triples_residual += -1.00 P(j,k) P(b,c) <l,a||d,k> t1(b,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('akcijb->abcijk', tmps_["66_vovoov"])
    triples_residual -= einsum('cjbika->abcijk', tmps_["66_vovoov"])
    triples_residual -= einsum('akbijc->abcijk', tmps_["66_vovoov"])
    triples_residual -= einsum('ciajkb->abcijk', tmps_["66_vovoov"])
    triples_residual += einsum('ajbikc->abcijk', tmps_["66_vovoov"])

    # triples_residual += -1.00 P(a,b) <l,c||d,i> t1(a,l) t2(d,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('cibjka->abcijk', tmps_["66_vovoov"])

    # triples_residual += +1.00 P(j,k) P(a,c) <l,b||d,k> t1(a,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bkcija->abcijk', tmps_["66_vovoov"])
    triples_residual += einsum('cjaikb->abcijk', tmps_["66_vovoov"])
    triples_residual += einsum('biajkc->abcijk', tmps_["66_vovoov"])

    # triples_residual += -1.00 P(b,c) <l,a||d,i> t1(b,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('aicjkb->abcijk', tmps_["66_vovoov"])
    triples_residual -= einsum('ckaijb->abcijk', tmps_["66_vovoov"])
    triples_residual -= einsum('ajcikb->abcijk', tmps_["66_vovoov"])
    triples_residual += einsum('bkaijc->abcijk', tmps_["66_vovoov"])

    # triples_residual += -1.00 P(j,k) P(a,b) <l,c||d,k> t1(a,l) t2(d,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('ckbija->abcijk', tmps_["66_vovoov"])
    triples_residual -= einsum('aibjkc->abcijk', tmps_["66_vovoov"])
    triples_residual -= einsum('bjaikc->abcijk', tmps_["66_vovoov"])
    triples_residual += einsum('bjcika->abcijk', tmps_["66_vovoov"])

    # triples_residual += +1.00 P(a,c) <l,b||d,i> t1(a,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bicjka->abcijk', tmps_["66_vovoov"])
    del tmps_["66_vovoov"]

    if includes_["t3"]:

        # tmps_[12_ovoo](l,b,i,j) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,b,i,j,m) // flops: o3v1 = o4v3 | mem: o3v1 = o3v1
        tmps_["12_ovoo"] = 0.50 * einsum('mlde,debijm->lbij', eri["oovv"], t3)

        # tmps_[74_voovvo](a,i,j,b,c,k) = 1.00 eri[oovv](m,l,d,e) * t3(d,e,a,i,j,m) * t2(b,c,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["74_voovvo"] = einsum('laij,bckl->aijbck', tmps_["12_ovoo"], t2)

        # triples_residual += -0.50 P(j,k) <m,l||d,e> t2(b,c,k,l) t3(d,e,a,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('aijbck->abcijk', tmps_["74_voovvo"])
        triples_residual += einsum('bjkaci->abcijk', tmps_["74_voovvo"])
        triples_residual += einsum('bijack->abcijk', tmps_["74_voovvo"])

        # triples_residual += -0.50 <m,l||d,e> t2(b,c,i,l) t3(d,e,a,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('ajkbci->abcijk', tmps_["74_voovvo"])
        triples_residual += einsum('aikbcj->abcijk', tmps_["74_voovvo"])
        triples_residual -= einsum('bikacj->abcijk', tmps_["74_voovvo"])

        # triples_residual += -0.50 P(j,k) P(b,c) <m,l||d,e> t2(a,b,k,l) t3(d,e,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cijabk->abcijk', tmps_["74_voovvo"])
        triples_residual += einsum('cikabj->abcijk', tmps_["74_voovvo"])

        # triples_residual += -0.50 P(b,c) <m,l||d,e> t2(a,b,i,l) t3(d,e,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cjkabi->abcijk', tmps_["74_voovvo"])
        del tmps_["74_voovvo"]

        # tmps_[103_voov](b,i,j,a) = 1.00 eri[oovv](l,k,c,d) * t3(c,d,b,i,j,l) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
        tmps_["103_voov"] = einsum('kbij,ak->bija', tmps_["12_ovoo"], t1)
        del tmps_["12_ovoo"]
        doubles_residual += einsum('aijb->abij', tmps_["103_voov"])

        # doubles_residual += -0.50 P(a,b) <l,k||c,d> t1(a,k) t3(c,d,b,i,j,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('bija->abij', tmps_["103_voov"])
        del tmps_["103_voov"]

    # tmps_[13_voovvo](b,j,k,a,c,i) = 1.00 eri[vooo](b,l,j,k) * t2(a,c,i,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["13_voovvo"] = einsum('bljk,acil->bjkaci', eri["vooo"], t2)
    triples_residual -= einsum('cikabj->abcijk', tmps_["13_voovvo"])

    # triples_residual += -1.00 P(i,j) P(a,b) <l,a||j,k> t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('ajkbci->abcijk', tmps_["13_voovvo"])

    # triples_residual += -1.00 P(a,b) <l,a||i,j> t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('aijbck->abcijk', tmps_["13_voovvo"])
    triples_residual -= einsum('aikbcj->abcijk', tmps_["13_voovvo"])

    # triples_residual += -1.00 <l,c||i,j> t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('cijabk->abcijk', tmps_["13_voovvo"])

    # triples_residual += -1.00 P(i,j) <l,c||j,k> t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('cjkabi->abcijk', tmps_["13_voovvo"])
    triples_residual -= einsum('bjkaci->abcijk', tmps_["13_voovvo"])
    triples_residual -= einsum('bijack->abcijk', tmps_["13_voovvo"])
    triples_residual += einsum('bikacj->abcijk', tmps_["13_voovvo"])
    del tmps_["13_voovvo"]

    if includes_["t3"]:

        # tmps_[14_ovvooo](l,b,c,i,j,k) = 1.00 f[ov](l,d) * t3(d,b,c,i,j,k) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
        tmps_["14_ovvooo"] = einsum('ld,dbcijk->lbcijk', f["ov"], t3)

        # tmps_[85_vvooov](a,b,i,j,k,c) = 1.00 f[ov](l,d) * t3(d,a,b,i,j,k) * t1(c,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["85_vvooov"] = einsum('labijk,cl->abijkc', tmps_["14_ovvooo"], t1)
        del tmps_["14_ovvooo"]

        # triples_residual += -1.00 f(l,d) t1(c,l) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abijkc->abcijk', tmps_["85_vvooov"])
        triples_residual += einsum('acijkb->abcijk', tmps_["85_vvooov"])

        # triples_residual += -1.00 P(a,b) f(l,d) t1(a,l) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bcijka->abcijk', tmps_["85_vvooov"])
        del tmps_["85_vvooov"]

        # tmps_[15_ovvvoo](i,a,b,c,j,k) = 1.00 f[oo](l,i) * t3(a,b,c,j,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["15_ovvvoo"] = einsum('li,abcjkl->iabcjk', f["oo"], t3)
        triples_residual += einsum('jabcik->abcijk', tmps_["15_ovvvoo"])

        # triples_residual += -1.00 f(l,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('iabcjk->abcijk', tmps_["15_ovvvoo"])

        # triples_residual += -1.00 P(j,k) f(l,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('kabcij->abcijk', tmps_["15_ovvvoo"])
        del tmps_["15_ovvvoo"]

        # tmps_[16_ovvo](j,a,b,i) = 0.50 eri[oovo](l,k,c,j) * t3(c,a,b,i,l,k) // flops: o2v2 = o4v3 | mem: o2v2 = o2v2
        tmps_["16_ovvo"] = 0.50 * einsum('lkcj,cabilk->jabi', eri["oovo"], t3)

        # doubles_residual += +0.50 P(i,j) <l,k||c,j> t3(c,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual += einsum('jabi->abij', tmps_["16_ovvo"])
        doubles_residual -= einsum('iabj->abij', tmps_["16_ovvo"])
        del tmps_["16_ovvo"]

    # tmps_[17_ooovoo](m,l,i,c,j,k) = 1.00 eri[oovo](m,l,d,i) * t2(d,c,j,k) // flops: o5v1 = o5v2 | mem: o5v1 = o5v1
    tmps_["17_ooovoo"] = einsum('mldi,dcjk->mlicjk', eri["oovo"], t2)

    # tmps_[69_vovoov](a,j,c,i,k,b) = 1.00 t1(a,l) * eri[oovo](m,l,d,j) * t2(d,c,i,k) * t1(b,m) // flops: o3v3 = o5v2 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["69_vovoov"] = einsum('al,mljcik,bm->ajcikb', t1, tmps_["17_ooovoo"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["17_ooovoo"]
    triples_residual += einsum('ajbikc->abcijk', tmps_["69_vovoov"])

    # triples_residual += +1.00 P(j,k) <m,l||d,k> t1(b,l) t1(c,m) t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('bkaijc->abcijk', tmps_["69_vovoov"])

    # triples_residual += +1.00 P(j,k) P(b,c) <m,l||d,k> t1(a,l) t1(b,m) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('akcijb->abcijk', tmps_["69_vovoov"])
    triples_residual -= einsum('aibjkc->abcijk', tmps_["69_vovoov"])
    triples_residual -= einsum('ajcikb->abcijk', tmps_["69_vovoov"])
    triples_residual -= einsum('bjaikc->abcijk', tmps_["69_vovoov"])

    # triples_residual += +1.00 <m,l||d,i> t1(b,l) t1(c,m) t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('biajkc->abcijk', tmps_["69_vovoov"])

    # triples_residual += +1.00 P(b,c) <m,l||d,i> t1(a,l) t1(b,m) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('aicjkb->abcijk', tmps_["69_vovoov"])
    triples_residual -= einsum('akbijc->abcijk', tmps_["69_vovoov"])
    del tmps_["69_vovoov"]

    # tmps_[18_vvvo](a,d,c,i) = 1.00 eri[vovv](a,l,d,e) * t2(e,c,i,l) // flops: o1v3 = o2v4 | mem: o1v3 = o1v3
    tmps_["18_vvvo"] = einsum('alde,ecil->adci', eri["vovv"], t2)

    # tmps_[57_voovvo](a,i,k,b,c,j) = 1.00 t2(d,a,i,k) * eri[vovv](b,l,d,e) * t2(e,c,j,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["57_voovvo"] = einsum('daik,bdcj->aikbcj', t2, tmps_["18_vvvo"])
    triples_residual += einsum('aikcbj->abcijk', tmps_["57_voovvo"])

    # triples_residual += +1.00 P(i,j) <l,c||d,e> t2(d,a,j,k) t2(e,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('ajkcbi->abcijk', tmps_["57_voovvo"])

    # triples_residual += +1.00 <l,c||d,e> t2(d,a,i,j) t2(e,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('aijcbk->abcijk', tmps_["57_voovvo"])
    triples_residual -= einsum('aikbcj->abcijk', tmps_["57_voovvo"])
    triples_residual += einsum('ajkbci->abcijk', tmps_["57_voovvo"])
    triples_residual += einsum('aijbck->abcijk', tmps_["57_voovvo"])

    # triples_residual += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,j) t2(e,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bijack->abcijk', tmps_["57_voovvo"])
    triples_residual += einsum('bikacj->abcijk', tmps_["57_voovvo"])

    # triples_residual += +1.00 P(i,j) P(a,b) <l,a||d,e> t2(d,b,j,k) t2(e,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bjkaci->abcijk', tmps_["57_voovvo"])
    del tmps_["57_voovvo"]

    # tmps_[88_ovvo](j,a,b,i) = 1.00 t1(c,j) * eri[vovv](a,k,c,d) * t2(d,b,i,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["88_ovvo"] = einsum('cj,acbi->jabi', t1, tmps_["18_vvvo"])
    del tmps_["18_vvvo"]
    doubles_residual += einsum('ibaj->abij', tmps_["88_ovvo"])
    doubles_residual -= einsum('iabj->abij', tmps_["88_ovvo"])

    # doubles_residual += -1.00 P(i,j) P(a,b) <k,a||c,d> t1(c,j) t2(d,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual += einsum('jabi->abij', tmps_["88_ovvo"])
    doubles_residual -= einsum('jbai->abij', tmps_["88_ovvo"])
    del tmps_["88_ovvo"]

    # tmps_[19_vvvo](c,e,a,k) = 1.00 eri[vovv](c,l,d,e) * t2(d,a,k,l) // flops: o1v3 = o2v4 | mem: o1v3 = o1v3
    tmps_["19_vvvo"] = einsum('clde,dakl->ceak', eri["vovv"], t2)

    # tmps_[55_vvovoo](a,b,k,c,i,j) = 1.00 eri[vovv](a,l,d,e) * t2(d,b,k,l) * t2(e,c,i,j) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["55_vvovoo"] = einsum('aebk,ecij->abkcij', tmps_["19_vvvo"], t2)
    del tmps_["19_vvvo"]
    triples_residual += einsum('cajbik->abcijk', tmps_["55_vvovoo"])
    triples_residual -= einsum('bajcik->abcijk', tmps_["55_vvovoo"])

    # triples_residual += +1.00 <l,c||d,e> t2(d,a,i,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('caibjk->abcijk', tmps_["55_vvovoo"])

    # triples_residual += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('abicjk->abcijk', tmps_["55_vvovoo"])

    # triples_residual += +1.00 P(j,k) <l,c||d,e> t2(d,a,k,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('cakbij->abcijk', tmps_["55_vvovoo"])
    triples_residual += einsum('bakcij->abcijk', tmps_["55_vvovoo"])
    triples_residual += einsum('abjcik->abcijk', tmps_["55_vvovoo"])

    # triples_residual += +1.00 P(j,k) P(a,b) <l,a||d,e> t2(d,b,k,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('abkcij->abcijk', tmps_["55_vvovoo"])
    triples_residual += einsum('baicjk->abcijk', tmps_["55_vvovoo"])
    del tmps_["55_vvovoo"]

    # tmps_[20_ovvo](l,d,c,i) = 1.00 eri[oovv](m,l,d,e) * t2(e,c,i,m) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["20_ovvo"] = einsum('mlde,ecim->ldci', eri["oovv"], t2)

    # singles_residual += +1.00 <k,j||b,c> t1(b,j) t2(c,a,i,k)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_residual += einsum('jbai,bj->ai', tmps_["20_ovvo"], t1)

    # tmps_[67_voovov](a,i,j,c,k,b) = 1.00 t2(d,a,i,j) * eri[oovv](m,l,d,e) * t2(e,c,k,m) * t1(b,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["67_voovov"] = einsum('daij,ldck,bl->aijckb', t2, tmps_["20_ovvo"], t1, optimize=['einsum_path',(0,1),(0,1)])

    # triples_residual += -1.00 P(i,j) <m,l||d,e> t1(c,l) t2(d,a,j,k) t2(e,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('ajkbic->abcijk', tmps_["67_voovov"])
    triples_residual -= einsum('aikcjb->abcijk', tmps_["67_voovov"])

    # triples_residual += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,j,k) t2(e,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bjkcia->abcijk', tmps_["67_voovov"])
    triples_residual += einsum('aikbjc->abcijk', tmps_["67_voovov"])

    # triples_residual += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,j) t2(e,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('aijbkc->abcijk', tmps_["67_voovov"])
    triples_residual += einsum('bikcja->abcijk', tmps_["67_voovov"])
    triples_residual += einsum('aijckb->abcijk', tmps_["67_voovov"])

    # triples_residual += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,j) t2(e,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bijcka->abcijk', tmps_["67_voovov"])
    triples_residual += einsum('ajkcib->abcijk', tmps_["67_voovov"])
    del tmps_["67_voovov"]

    # tmps_[87_vovo](a,j,b,i) = 1.00 t2(c,a,j,k) * eri[oovv](l,k,c,d) * t2(d,b,i,l) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["87_vovo"] = einsum('cajk,kcbi->ajbi', t2, tmps_["20_ovvo"])

    # doubles_residual += +1.00 P(i,j) <l,k||c,d> t2(c,a,j,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual += einsum('ajbi->abij', tmps_["87_vovo"])
    doubles_residual -= einsum('aibj->abij', tmps_["87_vovo"])
    del tmps_["87_vovo"]

    # tmps_[92_vvoo](b,a,i,j) = 1.00 eri[oovv](l,k,c,d) * t2(d,a,i,l) * t1(c,j) * t1(b,k) // flops: o2v2 = o3v2 o3v2 | mem: o2v2 = o3v1 o2v2
    tmps_["92_vvoo"] = einsum('kcai,cj,bk->baij', tmps_["20_ovvo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["20_ovvo"]
    doubles_residual -= einsum('baij->abij', tmps_["92_vvoo"])

    # doubles_residual += +1.00 P(i,j) P(a,b) <l,k||c,d> t1(c,j) t1(a,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual += einsum('abij->abij', tmps_["92_vvoo"])
    doubles_residual += einsum('baji->abij', tmps_["92_vvoo"])
    doubles_residual -= einsum('abji->abij', tmps_["92_vvoo"])
    del tmps_["92_vvoo"]

    # tmps_[21_vooo](b,l,i,j) = 0.50 eri[vovv](b,l,d,e) * t2(d,e,i,j) // flops: o3v1 = o3v3 | mem: o3v1 = o3v1
    tmps_["21_vooo"] = 0.50 * einsum('blde,deij->blij', eri["vovv"], t2)

    # tmps_[77_voovvo](c,i,j,a,b,k) = 1.00 eri[vovv](c,l,d,e) * t2(d,e,i,j) * t2(a,b,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["77_voovvo"] = einsum('clij,abkl->cijabk', tmps_["21_vooo"], t2)

    # triples_residual += -0.50 P(a,b) <l,a||d,e> t2(d,e,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('aijbck->abcijk', tmps_["77_voovvo"])
    triples_residual -= einsum('cikabj->abcijk', tmps_["77_voovvo"])
    triples_residual -= einsum('bjkaci->abcijk', tmps_["77_voovvo"])

    # triples_residual += -0.50 <l,c||d,e> t2(d,e,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('cijabk->abcijk', tmps_["77_voovvo"])

    # triples_residual += -0.50 P(i,j) <l,c||d,e> t2(d,e,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('cjkabi->abcijk', tmps_["77_voovvo"])

    # triples_residual += -0.50 P(i,j) P(a,b) <l,a||d,e> t2(d,e,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('ajkbci->abcijk', tmps_["77_voovvo"])
    triples_residual += einsum('bikacj->abcijk', tmps_["77_voovvo"])
    triples_residual -= einsum('bijack->abcijk', tmps_["77_voovvo"])
    triples_residual -= einsum('aikbcj->abcijk', tmps_["77_voovvo"])
    del tmps_["77_voovvo"]

    # tmps_[102_voov](a,i,j,b) = 1.00 eri[vovv](a,k,c,d) * t2(c,d,i,j) * t1(b,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["102_voov"] = einsum('akij,bk->aijb', tmps_["21_vooo"], t1)
    del tmps_["21_vooo"]
    doubles_residual += einsum('bija->abij', tmps_["102_voov"])

    # doubles_residual += +0.50 P(a,b) <k,a||c,d> t1(b,k) t2(c,d,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual -= einsum('aijb->abij', tmps_["102_voov"])
    del tmps_["102_voov"]

    # tmps_[22_vovv](d,i,a,c) = 0.50 eri[oovo](m,l,d,i) * t2(a,c,m,l) // flops: o1v3 = o3v3 | mem: o1v3 = o1v3
    tmps_["22_vovv"] = 0.50 * einsum('mldi,acml->diac', eri["oovo"], t2)

    # tmps_[58_vooovv](c,i,j,k,a,b) = 1.00 t2(d,c,i,j) * eri[oovo](m,l,d,k) * t2(a,b,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["58_vooovv"] = einsum('dcij,dkab->cijkab', t2, tmps_["22_vovv"])

    # triples_residual += -0.50 P(j,k) <m,l||d,k> t2(d,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('cijkab->abcijk', tmps_["58_vooovv"])

    # triples_residual += -0.50 <m,l||d,i> t2(d,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('cjkiab->abcijk', tmps_["58_vooovv"])
    triples_residual += einsum('aikjbc->abcijk', tmps_["58_vooovv"])
    triples_residual += einsum('cikjab->abcijk', tmps_["58_vooovv"])

    # triples_residual += -0.50 P(j,k) P(a,b) <m,l||d,k> t2(d,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('aijkbc->abcijk', tmps_["58_vooovv"])
    triples_residual -= einsum('bikjac->abcijk', tmps_["58_vooovv"])
    triples_residual += einsum('bijkac->abcijk', tmps_["58_vooovv"])

    # triples_residual += -0.50 P(a,b) <m,l||d,i> t2(d,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('ajkibc->abcijk', tmps_["58_vooovv"])
    triples_residual += einsum('bjkiac->abcijk', tmps_["58_vooovv"])
    del tmps_["58_vooovv"]

    # tmps_[89_ovvo](i,a,b,j) = 1.00 eri[oovo](l,k,c,i) * t2(a,b,l,k) * t1(c,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["89_ovvo"] = einsum('ciab,cj->iabj', tmps_["22_vovv"], t1)
    del tmps_["22_vovv"]

    # doubles_residual += +0.50 P(i,j) <l,k||c,j> t1(c,i) t2(a,b,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual += einsum('jabi->abij', tmps_["89_ovvo"])
    doubles_residual -= einsum('iabj->abij', tmps_["89_ovvo"])
    del tmps_["89_ovvo"]

    # tmps_[23_ovvo](l,e,a,k) = 1.00 eri[oovv](m,l,d,e) * t2(d,a,k,m) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["23_ovvo"] = einsum('mlde,dakm->leak', eri["oovv"], t2)

    # tmps_[68_voovov](c,j,k,a,i,b) = 1.00 t2(e,c,j,k) * eri[oovv](m,l,d,e) * t2(d,a,i,m) * t1(b,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["68_voovov"] = einsum('ecjk,leai,bl->cjkaib', t2, tmps_["23_ovvo"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["23_ovvo"]
    triples_residual += einsum('cikbja->abcijk', tmps_["68_voovov"])

    # triples_residual += -1.00 P(j,k) <m,l||d,e> t1(c,l) t2(d,a,k,m) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bijakc->abcijk', tmps_["68_voovov"])
    triples_residual += einsum('bikajc->abcijk', tmps_["68_voovov"])
    triples_residual -= einsum('cikajb->abcijk', tmps_["68_voovov"])

    # triples_residual += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('cjkbia->abcijk', tmps_["68_voovov"])
    triples_residual += einsum('cijakb->abcijk', tmps_["68_voovov"])

    # triples_residual += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,k,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('cijbka->abcijk', tmps_["68_voovov"])
    triples_residual += einsum('cjkaib->abcijk', tmps_["68_voovov"])

    # triples_residual += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,m) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bjkaic->abcijk', tmps_["68_voovov"])
    del tmps_["68_voovov"]

    # tmps_[24_ovvo](m,e,c,i) = 1.00 eri[oovv](m,l,d,e) * t2(d,c,i,l) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["24_ovvo"] = einsum('mlde,dcil->meci', eri["oovv"], t2)

    if includes_["t3"]:

        # tmps_[46_vvoovo](a,c,j,k,b,i) = 1.00 t3(e,a,c,j,k,m) * eri[oovv](m,l,d,e) * t2(d,b,i,l) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
        tmps_["46_vvoovo"] = einsum('eacjkm,mebi->acjkbi', t3, tmps_["24_ovvo"])
    del tmps_["24_ovvo"]

    if includes_["t3"]:
        triples_residual -= einsum('abikcj->abcijk', tmps_["46_vvoovo"])

        # triples_residual += +1.00 P(a,b) <m,l||d,e> t2(d,a,i,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bcjkai->abcijk', tmps_["46_vvoovo"])

        # triples_residual += +1.00 P(j,k) P(a,b) <m,l||d,e> t2(d,a,k,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bcijak->abcijk', tmps_["46_vvoovo"])
        triples_residual += einsum('acikbj->abcijk', tmps_["46_vvoovo"])

        # triples_residual += +1.00 <m,l||d,e> t2(d,c,i,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abjkci->abcijk', tmps_["46_vvoovo"])
        triples_residual -= einsum('acijbk->abcijk', tmps_["46_vvoovo"])
        triples_residual -= einsum('bcikaj->abcijk', tmps_["46_vvoovo"])

        # triples_residual += +1.00 P(j,k) <m,l||d,e> t2(d,c,k,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abijck->abcijk', tmps_["46_vvoovo"])
        triples_residual -= einsum('acjkbi->abcijk', tmps_["46_vvoovo"])
        del tmps_["46_vvoovo"]

    # tmps_[25_vovo](a,i,b,j) = 1.00 eri[vovo](a,k,c,i) * t2(c,b,j,k) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["25_vovo"] = einsum('akci,cbjk->aibj', eri["vovo"], t2)

    # doubles_residual += +1.00 P(i,j) P(a,b) <k,a||c,j> t2(c,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual -= einsum('ajbi->abij', tmps_["25_vovo"])
    doubles_residual -= einsum('biaj->abij', tmps_["25_vovo"])
    doubles_residual += einsum('bjai->abij', tmps_["25_vovo"])
    doubles_residual += einsum('aibj->abij', tmps_["25_vovo"])
    del tmps_["25_vovo"]

    # tmps_[26_oovo](m,i,b,j) = 1.00 eri[oovo](m,l,d,i) * t2(d,b,j,l) // flops: o3v1 = o4v2 | mem: o3v1 = o3v1
    tmps_["26_oovo"] = einsum('mldi,dbjl->mibj', eri["oovo"], t2)

    # tmps_[64_vvoovo](b,c,i,j,a,k) = 1.00 t2(b,c,i,m) * eri[oovo](m,l,d,j) * t2(d,a,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["64_vvoovo"] = einsum('bcim,mjak->bcijak', t2, tmps_["26_oovo"])
    del tmps_["26_oovo"]
    triples_residual += einsum('acjkbi->abcijk', tmps_["64_vvoovo"])

    # triples_residual += +1.00 P(j,k) P(a,b) <m,l||d,i> t2(d,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('bcjiak->abcijk', tmps_["64_vvoovo"])
    triples_residual += einsum('bckjai->abcijk', tmps_["64_vvoovo"])
    triples_residual -= einsum('bcjkai->abcijk', tmps_["64_vvoovo"])
    triples_residual -= einsum('abjkci->abcijk', tmps_["64_vvoovo"])

    # triples_residual += -1.00 P(i,k) P(a,b) <m,l||d,j> t2(d,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bcijak->abcijk', tmps_["64_vvoovo"])
    triples_residual -= einsum('ackjbi->abcijk', tmps_["64_vvoovo"])
    triples_residual -= einsum('acikbj->abcijk', tmps_["64_vvoovo"])

    # triples_residual += +1.00 P(i,j) P(a,b) <m,l||d,k> t2(d,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('bcikaj->abcijk', tmps_["64_vvoovo"])
    triples_residual += einsum('acijbk->abcijk', tmps_["64_vvoovo"])
    triples_residual += einsum('abkjci->abcijk', tmps_["64_vvoovo"])

    # triples_residual += -1.00 P(i,k) <m,l||d,j> t2(d,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('abijck->abcijk', tmps_["64_vvoovo"])
    triples_residual -= einsum('abkicj->abcijk', tmps_["64_vvoovo"])
    triples_residual -= einsum('bckiaj->abcijk', tmps_["64_vvoovo"])
    triples_residual += einsum('ackibj->abcijk', tmps_["64_vvoovo"])
    triples_residual -= einsum('acjibk->abcijk', tmps_["64_vvoovo"])

    # triples_residual += +1.00 P(i,j) <m,l||d,k> t2(d,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('abikcj->abcijk', tmps_["64_vvoovo"])

    # triples_residual += +1.00 P(j,k) <m,l||d,i> t2(d,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('abjick->abcijk', tmps_["64_vvoovo"])
    del tmps_["64_vvoovo"]

    # tmps_[27_oooo](m,l,i,j) = 0.50 eri[oovv](m,l,d,e) * t2(d,e,i,j) // flops: o4v0 = o4v2 | mem: o4v0 = o4v0
    tmps_["27_oooo"] = 0.50 * einsum('mlde,deij->mlij', eri["oovv"], t2)

    # doubles_residual += +0.25 <l,k||c,d> t2(c,d,i,j) t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
    doubles_residual += 0.50 * einsum('ablk,lkij->abij', t2, tmps_["27_oooo"])

    # doubles_residual += -0.50 <l,k||c,d> t1(a,k) t1(b,l) t2(c,d,i,j)  // flops: o2v2 += o4v1 o3v2 | mem: o2v2 += o3v1 o2v2
    doubles_residual -= einsum('lkij,bl,ak->abij', tmps_["27_oooo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t3"]:

        # tmps_[51_oovvvo](i,j,a,b,c,k) = 0.50 eri[oovv](m,l,d,e) * t2(d,e,i,j) * t3(a,b,c,k,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
        tmps_["51_oovvvo"] = 0.50 * einsum('mlij,abckml->ijabck', tmps_["27_oooo"], t3)

        # triples_residual += +0.25 P(i,j) <m,l||d,e> t2(d,e,j,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('jkabci->abcijk', tmps_["51_oovvvo"])

        # triples_residual += +0.25 <m,l||d,e> t2(d,e,i,j) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('ijabck->abcijk', tmps_["51_oovvvo"])
        triples_residual -= einsum('ikabcj->abcijk', tmps_["51_oovvvo"])
        del tmps_["51_oovvvo"]

    # tmps_[72_oovvov](i,k,a,b,j,c) = 1.00 t1(c,l) * eri[oovv](m,l,d,e) * t2(d,e,i,k) * t2(a,b,j,m) // flops: o3v3 = o4v1 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["72_oovvov"] = einsum('cl,mlik,abjm->ikabjc', t1, tmps_["27_oooo"], t2, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["27_oooo"]

    # triples_residual += +0.50 P(a,b) <m,l||d,e> t1(a,l) t2(d,e,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('ijbcka->abcijk', tmps_["72_oovvov"])
    triples_residual -= einsum('jkacib->abcijk', tmps_["72_oovvov"])

    # triples_residual += +0.50 <m,l||d,e> t1(c,l) t2(d,e,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('ijabkc->abcijk', tmps_["72_oovvov"])
    triples_residual += einsum('ikacjb->abcijk', tmps_["72_oovvov"])
    triples_residual -= einsum('ikabjc->abcijk', tmps_["72_oovvov"])

    # triples_residual += +0.50 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,e,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('jkbcia->abcijk', tmps_["72_oovvov"])

    # triples_residual += +0.50 P(i,j) <m,l||d,e> t1(c,l) t2(d,e,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('jkabic->abcijk', tmps_["72_oovvov"])
    triples_residual -= einsum('ijackb->abcijk', tmps_["72_oovvov"])
    triples_residual -= einsum('ikbcja->abcijk', tmps_["72_oovvov"])
    del tmps_["72_oovvov"]

    # tmps_[28_oovo](k,j,b,i) = 1.00 eri[oovo](l,k,c,j) * t2(c,b,i,l) // flops: o3v1 = o4v2 | mem: o3v1 = o3v1
    tmps_["28_oovo"] = einsum('lkcj,cbil->kjbi', eri["oovo"], t2)

    # tmps_[94_vovo](b,j,a,i) = 1.00 t1(b,k) * eri[oovo](l,k,c,j) * t2(c,a,i,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["94_vovo"] = einsum('bk,kjai->bjai', t1, tmps_["28_oovo"])
    del tmps_["28_oovo"]
    doubles_residual += einsum('bjai->abij', tmps_["94_vovo"])

    # doubles_residual += -1.00 P(i,j) P(a,b) <l,k||c,j> t1(a,k) t2(c,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual -= einsum('ajbi->abij', tmps_["94_vovo"])
    doubles_residual -= einsum('biaj->abij', tmps_["94_vovo"])
    doubles_residual += einsum('aibj->abij', tmps_["94_vovo"])
    del tmps_["94_vovo"]

    # tmps_[29_vvvo](a,c,e,i) = 1.00 eri[vvvv](a,c,d,e) * t1(d,i) // flops: o1v3 = o1v4 | mem: o1v3 = o1v3
    tmps_["29_vvvo"] = einsum('acde,di->acei', eri["vvvv"], t1)

    # doubles_residual += -1.00 <a,b||c,d> t1(c,j) t1(d,i)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_residual -= einsum('di,abdj->abij', t1, tmps_["29_vvvo"])

    # tmps_[56_vvovoo](a,b,k,c,i,j) = 1.00 eri[vvvv](a,b,d,e) * t1(d,k) * t2(e,c,i,j) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["56_vvovoo"] = einsum('abek,ecij->abkcij', tmps_["29_vvvo"], t2)
    del tmps_["29_vvvo"]
    triples_residual += einsum('acjbik->abcijk', tmps_["56_vvovoo"])

    # triples_residual += +1.00 P(b,c) <a,b||d,e> t1(d,i) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('abicjk->abcijk', tmps_["56_vvovoo"])
    triples_residual -= einsum('bcjaik->abcijk', tmps_["56_vvovoo"])
    triples_residual -= einsum('ackbij->abcijk', tmps_["56_vvovoo"])
    triples_residual -= einsum('acibjk->abcijk', tmps_["56_vvovoo"])

    # triples_residual += +1.00 P(j,k) P(b,c) <a,b||d,e> t1(d,k) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('abkcij->abcijk', tmps_["56_vvovoo"])

    # triples_residual += +1.00 <b,c||d,e> t1(d,i) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('bciajk->abcijk', tmps_["56_vvovoo"])
    triples_residual -= einsum('abjcik->abcijk', tmps_["56_vvovoo"])

    # triples_residual += +1.00 P(j,k) <b,c||d,e> t1(d,k) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('bckaij->abcijk', tmps_["56_vvovoo"])
    del tmps_["56_vvovoo"]

    # tmps_[30_vovo](c,l,e,i) = 1.00 eri[vovv](c,l,d,e) * t1(d,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["30_vovo"] = einsum('clde,di->clei', eri["vovv"], t1)

    if includes_["t3"]:

        # tmps_[47_vvoovo](a,b,i,k,c,j) = 1.00 t3(e,a,b,i,k,l) * eri[vovv](c,l,d,e) * t1(d,j) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
        tmps_["47_vvoovo"] = einsum('eabikl,clej->abikcj', t3, tmps_["30_vovo"])

        # triples_residual += -1.00 P(a,b) <l,a||d,e> t1(d,i) t3(e,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bcjkai->abcijk', tmps_["47_vvoovo"])
        triples_residual -= einsum('acjkbi->abcijk', tmps_["47_vvoovo"])
        triples_residual -= einsum('acijbk->abcijk', tmps_["47_vvoovo"])
        triples_residual += einsum('acikbj->abcijk', tmps_["47_vvoovo"])

        # triples_residual += -1.00 P(j,k) <l,c||d,e> t1(d,k) t3(e,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abijck->abcijk', tmps_["47_vvoovo"])

        # triples_residual += -1.00 <l,c||d,e> t1(d,i) t3(e,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abjkci->abcijk', tmps_["47_vvoovo"])
        triples_residual -= einsum('bcikaj->abcijk', tmps_["47_vvoovo"])

        # triples_residual += -1.00 P(j,k) P(a,b) <l,a||d,e> t1(d,k) t3(e,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bcijak->abcijk', tmps_["47_vvoovo"])
        triples_residual -= einsum('abikcj->abcijk', tmps_["47_vvoovo"])
        del tmps_["47_vvoovo"]

    # tmps_[61_voovov](a,i,j,b,k,c) = 1.00 t2(e,a,i,j) * eri[vovv](b,l,d,e) * t1(d,k) * t1(c,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["61_voovov"] = einsum('eaij,blek,cl->aijbkc', t2, tmps_["30_vovo"], t1, optimize=['einsum_path',(0,1),(0,1)])
    triples_residual -= einsum('bikajc->abcijk', tmps_["61_voovov"])

    # triples_residual += +1.00 P(b,c) <l,a||d,e> t1(d,i) t1(b,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('cjkaib->abcijk', tmps_["61_voovov"])
    triples_residual -= einsum('aikcjb->abcijk', tmps_["61_voovov"])

    # triples_residual += +1.00 P(j,k) P(b,c) <l,a||d,e> t1(d,k) t1(b,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('cijakb->abcijk', tmps_["61_voovov"])

    # triples_residual += +1.00 P(a,b) <l,c||d,e> t1(d,i) t1(a,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bjkcia->abcijk', tmps_["61_voovov"])
    triples_residual -= einsum('ajkbic->abcijk', tmps_["61_voovov"])
    triples_residual += einsum('ajkcib->abcijk', tmps_["61_voovov"])
    triples_residual -= einsum('aijbkc->abcijk', tmps_["61_voovov"])
    triples_residual += einsum('bjkaic->abcijk', tmps_["61_voovov"])

    # triples_residual += -1.00 P(a,c) <l,b||d,e> t1(d,i) t1(a,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('cjkbia->abcijk', tmps_["61_voovov"])

    # triples_residual += -1.00 P(j,k) P(a,c) <l,b||d,e> t1(d,k) t1(a,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('cijbka->abcijk', tmps_["61_voovov"])
    triples_residual += einsum('cikajb->abcijk', tmps_["61_voovov"])
    triples_residual -= einsum('cikbja->abcijk', tmps_["61_voovov"])
    triples_residual += einsum('bijakc->abcijk', tmps_["61_voovov"])
    triples_residual += einsum('aikbjc->abcijk', tmps_["61_voovov"])
    triples_residual += einsum('aijckb->abcijk', tmps_["61_voovov"])

    # triples_residual += +1.00 P(j,k) P(a,b) <l,c||d,e> t1(d,k) t1(a,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bijcka->abcijk', tmps_["61_voovov"])
    triples_residual += einsum('bikcja->abcijk', tmps_["61_voovov"])
    del tmps_["61_voovov"]

    # tmps_[71_ovovvo](i,a,k,b,c,j) = 1.00 t1(e,i) * eri[vovv](a,l,d,e) * t1(d,k) * t2(b,c,j,l) // flops: o3v3 = o3v2 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["71_ovovvo"] = einsum('ei,alek,bcjl->iakbcj', t1, tmps_["30_vovo"], t2, optimize=['einsum_path',(0,1),(0,1)])

    # triples_residual += +1.00 <l,c||d,e> t1(d,j) t1(e,i) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('icjabk->abcijk', tmps_["71_ovovvo"])
    triples_residual += einsum('jbkaci->abcijk', tmps_["71_ovovvo"])

    # triples_residual += +1.00 P(i,j) P(a,b) <l,a||d,e> t1(d,k) t1(e,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('jakbci->abcijk', tmps_["71_ovovvo"])
    triples_residual += einsum('ibjack->abcijk', tmps_["71_ovovvo"])

    # triples_residual += +1.00 P(a,b) <l,a||d,e> t1(d,j) t1(e,i) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('iajbck->abcijk', tmps_["71_ovovvo"])

    # triples_residual += +1.00 P(i,j) <l,c||d,e> t1(d,k) t1(e,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('jckabi->abcijk', tmps_["71_ovovvo"])
    triples_residual += einsum('iakbcj->abcijk', tmps_["71_ovovvo"])
    triples_residual += einsum('ickabj->abcijk', tmps_["71_ovovvo"])
    triples_residual -= einsum('ibkacj->abcijk', tmps_["71_ovovvo"])
    del tmps_["71_ovovvo"]

    # tmps_[96_vvoo](a,b,j,i) = 1.00 t1(d,i) * eri[vovv](b,k,c,d) * t1(c,j) * t1(a,k) // flops: o2v2 = o3v2 o3v2 | mem: o2v2 = o3v1 o2v2
    tmps_["96_vvoo"] = einsum('di,bkdj,ak->abji', t1, tmps_["30_vovo"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["30_vovo"]
    doubles_residual -= einsum('abji->abij', tmps_["96_vvoo"])

    # doubles_residual += -1.00 P(a,b) <k,a||c,d> t1(c,j) t1(d,i) t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual += einsum('baji->abij', tmps_["96_vvoo"])
    del tmps_["96_vvoo"]

    # tmps_[31_vv](d,a) = 0.50 eri[oovv](l,k,c,d) * t2(c,a,l,k) // flops: o0v2 = o2v3 | mem: o0v2 = o0v2
    tmps_["31_vv"] = 0.50 * einsum('lkcd,calk->da', eri["oovv"], t2)

    # doubles_residual += -0.50 <l,k||c,d> t2(c,a,l,k) t2(d,b,i,j)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_residual -= einsum('da,dbij->abij', tmps_["31_vv"], t2)

    if includes_["t3"]:

        # tmps_[60_vvooov](a,b,i,j,k,c) = 1.00 t3(e,a,b,i,j,k) * eri[oovv](m,l,d,e) * t2(d,c,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
        tmps_["60_vvooov"] = einsum('eabijk,ec->abijkc', t3, tmps_["31_vv"])
    del tmps_["31_vv"]

    if includes_["t3"]:
        triples_residual += einsum('acijkb->abcijk', tmps_["60_vvooov"])

        # triples_residual += -0.50 <m,l||d,e> t2(d,c,m,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abijkc->abcijk', tmps_["60_vvooov"])

        # triples_residual += -0.50 P(a,b) <m,l||d,e> t2(d,a,m,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bcijka->abcijk', tmps_["60_vvooov"])
        del tmps_["60_vvooov"]

    # tmps_[32_vvoo](a,b,j,i) = 1.00 eri[vvvo](a,b,c,j) * t1(c,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["32_vvoo"] = einsum('abcj,ci->abji', eri["vvvo"], t1)
    doubles_residual -= einsum('abij->abij', tmps_["32_vvoo"])

    # doubles_residual += +1.00 P(i,j) <a,b||c,j> t1(c,i)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual += einsum('abji->abij', tmps_["32_vvoo"])
    del tmps_["32_vvoo"]

    # tmps_[33_vvoo](b,a,i,j) = 1.00 f[vv](b,c) * t2(c,a,i,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["33_vvoo"] = einsum('bc,caij->baij', f["vv"], t2)

    # doubles_residual += +1.00 P(a,b) f(a,c) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual += einsum('abij->abij', tmps_["33_vvoo"])
    doubles_residual -= einsum('baij->abij', tmps_["33_vvoo"])
    del tmps_["33_vvoo"]

    # tmps_[34_oovo](l,k,d,j) = 1.00 eri[oovv](l,k,c,d) * t1(c,j) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["34_oovo"] = einsum('lkcd,cj->lkdj', eri["oovv"], t1)

    # singles_residual += +0.50 <k,j||b,c> t1(b,i) t2(c,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
    singles_residual += 0.50 * einsum('cakj,kjci->ai', t2, tmps_["34_oovo"])

    # doubles_residual += -0.50 <l,k||c,d> t1(c,j) t1(d,i) t2(a,b,l,k)  // flops: o2v2 += o4v1 o4v2 | mem: o2v2 += o4v0 o2v2
    doubles_residual -= 0.50 * einsum('lkdj,di,ablk->abij', tmps_["34_oovo"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t3"]:

        # tmps_[48_vvooov](b,c,i,j,k,a) = 1.00 t3(e,b,c,i,j,m) * eri[oovv](m,l,d,e) * t1(d,k) * t1(a,l) // flops: o3v3 = o5v3 o4v3 | mem: o3v3 = o4v2 o3v3
        tmps_["48_vvooov"] = einsum('ebcijm,mlek,al->bcijka', t3, tmps_["34_oovo"], t1, optimize=['einsum_path',(0,1),(0,1)])
        triples_residual += einsum('acikjb->abcijk', tmps_["48_vvooov"])
        triples_residual -= einsum('abikjc->abcijk', tmps_["48_vvooov"])
        triples_residual -= einsum('acijkb->abcijk', tmps_["48_vvooov"])

        # triples_residual += +1.00 P(j,k) <m,l||d,e> t1(d,k) t1(c,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abijkc->abcijk', tmps_["48_vvooov"])
        triples_residual -= einsum('acjkib->abcijk', tmps_["48_vvooov"])

        # triples_residual += +1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t1(a,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bcijka->abcijk', tmps_["48_vvooov"])

        # triples_residual += +1.00 P(a,b) <m,l||d,e> t1(d,i) t1(a,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bcjkia->abcijk', tmps_["48_vvooov"])
        triples_residual -= einsum('bcikja->abcijk', tmps_["48_vvooov"])

        # triples_residual += +1.00 <m,l||d,e> t1(d,i) t1(c,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abjkic->abcijk', tmps_["48_vvooov"])
        del tmps_["48_vvooov"]

        # tmps_[50_vvvooo](a,b,c,j,k,i) = 0.50 eri[oovv](m,l,d,e) * t1(d,k) * t1(e,i) * t3(a,b,c,j,m,l) // flops: o3v3 = o4v1 o5v3 | mem: o3v3 = o4v0 o3v3
        tmps_["50_vvvooo"] = 0.50 * einsum('mlek,ei,abcjml->abcjki', tmps_["34_oovo"], t1, t3, optimize=['einsum_path',(0,1),(0,1)])
        triples_residual += einsum('abcjki->abcijk', tmps_["50_vvvooo"])

        # triples_residual += -0.50 <m,l||d,e> t1(d,j) t1(e,i) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abckji->abcijk', tmps_["50_vvvooo"])

        # triples_residual += -0.50 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcikj->abcijk', tmps_["50_vvvooo"])
        del tmps_["50_vvvooo"]

    # tmps_[53_vvovoo](a,b,k,c,i,j) = 0.50 t2(a,b,m,l) * eri[oovv](m,l,d,e) * t1(d,k) * t2(e,c,i,j) // flops: o3v3 = o3v3 o3v4 | mem: o3v3 = o1v3 o3v3
    tmps_["53_vvovoo"] = 0.50 * einsum('abml,mlek,ecij->abkcij', t2, tmps_["34_oovo"], t2, optimize=['einsum_path',(0,1),(0,1)])

    # triples_residual += +0.50 P(a,b) <m,l||d,e> t1(d,i) t2(e,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('bciajk->abcijk', tmps_["53_vvovoo"])
    triples_residual -= einsum('acibjk->abcijk', tmps_["53_vvovoo"])
    triples_residual -= einsum('bcjaik->abcijk', tmps_["53_vvovoo"])

    # triples_residual += +0.50 <m,l||d,e> t1(d,i) t2(e,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('abicjk->abcijk', tmps_["53_vvovoo"])
    triples_residual += einsum('acjbik->abcijk', tmps_["53_vvovoo"])

    # triples_residual += +0.50 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('bckaij->abcijk', tmps_["53_vvovoo"])
    triples_residual -= einsum('abjcik->abcijk', tmps_["53_vvovoo"])

    # triples_residual += +0.50 P(j,k) <m,l||d,e> t1(d,k) t2(e,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('abkcij->abcijk', tmps_["53_vvovoo"])
    triples_residual -= einsum('ackbij->abcijk', tmps_["53_vvovoo"])
    del tmps_["53_vvovoo"]

    # tmps_[62_voovvo](c,j,k,a,b,i) = 1.00 t2(e,c,j,l) * eri[oovv](m,l,d,e) * t1(d,k) * t2(a,b,i,m) // flops: o3v3 = o4v2 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["62_voovvo"] = einsum('ecjl,mlek,abim->cjkabi', t2, tmps_["34_oovo"], t2, optimize=['einsum_path',(0,1),(0,1)])

    # triples_residual += -1.00 P(i,j) <m,l||d,e> t1(d,k) t2(e,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('cjkabi->abcijk', tmps_["62_voovvo"])

    # triples_residual += -1.00 P(j,k) <m,l||d,e> t1(d,i) t2(e,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('ckiabj->abcijk', tmps_["62_voovvo"])

    # triples_residual += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('ajkbci->abcijk', tmps_["62_voovvo"])
    triples_residual -= einsum('cijabk->abcijk', tmps_["62_voovvo"])
    triples_residual += einsum('cjiabk->abcijk', tmps_["62_voovvo"])
    triples_residual -= einsum('bikacj->abcijk', tmps_["62_voovvo"])

    # triples_residual += +1.00 P(i,k) <m,l||d,e> t1(d,j) t2(e,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('ckjabi->abcijk', tmps_["62_voovvo"])
    triples_residual += einsum('bkiacj->abcijk', tmps_["62_voovvo"])
    triples_residual += einsum('bijack->abcijk', tmps_["62_voovvo"])
    triples_residual -= einsum('bkjaci->abcijk', tmps_["62_voovvo"])

    # triples_residual += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,i) t2(e,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('akibcj->abcijk', tmps_["62_voovvo"])
    triples_residual += einsum('ajibck->abcijk', tmps_["62_voovvo"])
    triples_residual += einsum('cikabj->abcijk', tmps_["62_voovvo"])
    triples_residual += einsum('aikbcj->abcijk', tmps_["62_voovvo"])
    triples_residual += einsum('bjkaci->abcijk', tmps_["62_voovvo"])
    triples_residual -= einsum('aijbck->abcijk', tmps_["62_voovvo"])

    # triples_residual += +1.00 P(i,k) P(a,b) <m,l||d,e> t1(d,j) t2(e,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('akjbci->abcijk', tmps_["62_voovvo"])
    triples_residual -= einsum('bjiack->abcijk', tmps_["62_voovvo"])
    del tmps_["62_voovvo"]

    # tmps_[86_oovvoo](l,j,c,b,i,k) = 1.00 eri[oovv](m,l,d,e) * t1(d,j) * t2(e,b,i,k) * t1(c,m) // flops: o4v2 = o5v2 o5v2 | mem: o4v2 = o5v1 o4v2
    tmps_["86_oovvoo"] = einsum('mlej,ebik,cm->ljcbik', tmps_["34_oovo"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

    # tmps_[106_vovvoo](a,j,b,c,i,k) = 1.00 t1(a,l) * tmps_[34_oovo](m,l,e,j) * t1(b,m) * t2(e,c,i,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["106_vovvoo"] = einsum('al,ljbcik->ajbcik', t1, tmps_["86_oovvoo"])
    del tmps_["86_oovvoo"]

    # triples_residual += -1.00 <m,l||d,e> t1(d,i) t1(b,l) t1(c,m) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bicajk->abcijk', tmps_["106_vovvoo"])

    # triples_residual += -1.00 P(j,k) <m,l||d,e> t1(d,k) t1(b,l) t1(c,m) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bkcaij->abcijk', tmps_["106_vovvoo"])
    triples_residual += einsum('bjcaik->abcijk', tmps_["106_vovvoo"])
    triples_residual += einsum('akcbij->abcijk', tmps_["106_vovvoo"])
    triples_residual += einsum('aicbjk->abcijk', tmps_["106_vovvoo"])

    # triples_residual += -1.00 P(j,k) P(b,c) <m,l||d,e> t1(d,k) t1(a,l) t1(b,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('akbcij->abcijk', tmps_["106_vovvoo"])

    # triples_residual += -1.00 P(b,c) <m,l||d,e> t1(d,i) t1(a,l) t1(b,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('aibcjk->abcijk', tmps_["106_vovvoo"])
    triples_residual += einsum('ajbcik->abcijk', tmps_["106_vovvoo"])
    triples_residual -= einsum('ajcbik->abcijk', tmps_["106_vovvoo"])
    del tmps_["106_vovvoo"]

    # tmps_[105_vooo](c,m,j,i) = 1.00 eri[oovv](m,l,d,e) * t1(d,j) * t1(e,i) * t1(c,l) // flops: o3v1 = o4v1 o4v1 | mem: o3v1 = o4v0 o3v1
    tmps_["105_vooo"] = einsum('mlej,ei,cl->cmji', tmps_["34_oovo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["34_oovo"]

    # doubles_residual += +1.00 <l,k||c,d> t1(c,j) t1(d,i) t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_residual += einsum('bl,alji->abij', t1, tmps_["105_vooo"])

    # tmps_[107_voovvo](c,k,j,a,b,i) = 1.00 t1(c,l) * tmps_[34_oovo](m,l,e,k) * t1(e,j) * t2(a,b,i,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["107_voovvo"] = einsum('cmkj,abim->ckjabi', tmps_["105_vooo"], t2)
    del tmps_["105_vooo"]
    triples_residual += einsum('ckiabj->abcijk', tmps_["107_voovvo"])
    triples_residual -= einsum('bkiacj->abcijk', tmps_["107_voovvo"])
    triples_residual += einsum('bjiack->abcijk', tmps_["107_voovvo"])

    # triples_residual += -1.00 P(a,b) <m,l||d,e> t1(d,j) t1(e,i) t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('ajibck->abcijk', tmps_["107_voovvo"])
    triples_residual += einsum('akibcj->abcijk', tmps_["107_voovvo"])

    # triples_residual += -1.00 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('ckjabi->abcijk', tmps_["107_voovvo"])

    # triples_residual += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t1(e,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('akjbci->abcijk', tmps_["107_voovvo"])

    # triples_residual += -1.00 <m,l||d,e> t1(d,j) t1(e,i) t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('cjiabk->abcijk', tmps_["107_voovvo"])
    triples_residual += einsum('bkjaci->abcijk', tmps_["107_voovvo"])
    del tmps_["107_voovvo"]

    # tmps_[35_vooo](c,l,i,k) = 1.00 eri[vovo](c,l,d,i) * t1(d,k) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["35_vooo"] = einsum('cldi,dk->clik', eri["vovo"], t1)

    # tmps_[65_vvovoo](a,b,i,c,k,j) = 1.00 t2(a,b,i,l) * eri[vovo](c,l,d,k) * t1(d,j) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["65_vvovoo"] = einsum('abil,clkj->abickj', t2, tmps_["35_vooo"])
    triples_residual -= einsum('acjbik->abcijk', tmps_["65_vvovoo"])
    triples_residual -= einsum('bcjaki->abcijk', tmps_["65_vvovoo"])
    triples_residual -= einsum('abkcij->abcijk', tmps_["65_vvovoo"])
    triples_residual -= einsum('acibkj->abcijk', tmps_["65_vvovoo"])
    triples_residual += einsum('acjbki->abcijk', tmps_["65_vvovoo"])
    triples_residual -= einsum('ackbji->abcijk', tmps_["65_vvovoo"])
    triples_residual += einsum('acibjk->abcijk', tmps_["65_vvovoo"])

    # triples_residual += -1.00 P(i,j) <l,c||d,k> t1(d,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('abickj->abcijk', tmps_["65_vvovoo"])

    # triples_residual += -1.00 P(i,j) P(a,b) <l,a||d,k> t1(d,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('bciakj->abcijk', tmps_["65_vvovoo"])
    triples_residual += einsum('abkcji->abcijk', tmps_["65_vvovoo"])

    # triples_residual += +1.00 P(i,k) P(a,b) <l,a||d,j> t1(d,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bciajk->abcijk', tmps_["65_vvovoo"])

    # triples_residual += +1.00 P(i,k) <l,c||d,j> t1(d,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('abicjk->abcijk', tmps_["65_vvovoo"])
    triples_residual -= einsum('bckaij->abcijk', tmps_["65_vvovoo"])
    triples_residual += einsum('bckaji->abcijk', tmps_["65_vvovoo"])
    triples_residual -= einsum('abjcki->abcijk', tmps_["65_vvovoo"])
    triples_residual += einsum('ackbij->abcijk', tmps_["65_vvovoo"])

    # triples_residual += -1.00 P(j,k) <l,c||d,i> t1(d,k) t2(a,b,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('abjcik->abcijk', tmps_["65_vvovoo"])

    # triples_residual += -1.00 P(j,k) P(a,b) <l,a||d,i> t1(d,k) t2(b,c,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('bcjaik->abcijk', tmps_["65_vvovoo"])
    del tmps_["65_vvovoo"]

    # tmps_[93_voov](b,j,i,a) = 1.00 eri[vovo](b,k,c,j) * t1(c,i) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["93_voov"] = einsum('bkji,ak->bjia', tmps_["35_vooo"], t1)
    del tmps_["35_vooo"]
    doubles_residual -= einsum('bija->abij', tmps_["93_voov"])
    doubles_residual += einsum('aijb->abij', tmps_["93_voov"])
    doubles_residual += einsum('bjia->abij', tmps_["93_voov"])

    # doubles_residual += +1.00 P(i,j) P(a,b) <k,a||c,j> t1(c,i) t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual -= einsum('ajib->abij', tmps_["93_voov"])
    del tmps_["93_voov"]

    # tmps_[36_ovoo](l,c,i,k) = 1.00 f[ov](l,d) * t2(d,c,i,k) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["36_ovoo"] = einsum('ld,dcik->lcik', f["ov"], t2)

    # tmps_[73_vvovoo](a,b,i,c,j,k) = 1.00 t2(a,b,i,l) * f[ov](l,d) * t2(d,c,j,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["73_vvovoo"] = einsum('abil,lcjk->abicjk', t2, tmps_["36_ovoo"])

    # triples_residual += -1.00 P(a,b) f(l,d) t2(d,a,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bckaij->abcijk', tmps_["73_vvovoo"])

    # triples_residual += -1.00 P(i,j) P(a,b) f(l,d) t2(d,a,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bciajk->abcijk', tmps_["73_vvovoo"])
    triples_residual += einsum('abjcik->abcijk', tmps_["73_vvovoo"])

    # triples_residual += -1.00 f(l,d) t2(d,c,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('abkcij->abcijk', tmps_["73_vvovoo"])
    triples_residual += einsum('acibjk->abcijk', tmps_["73_vvovoo"])
    triples_residual -= einsum('acjbik->abcijk', tmps_["73_vvovoo"])
    triples_residual += einsum('ackbij->abcijk', tmps_["73_vvovoo"])
    triples_residual += einsum('bcjaik->abcijk', tmps_["73_vvovoo"])

    # triples_residual += -1.00 P(i,j) f(l,d) t2(d,c,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('abicjk->abcijk', tmps_["73_vvovoo"])
    del tmps_["73_vvovoo"]

    # tmps_[101_voov](b,i,j,a) = 1.00 f[ov](k,c) * t2(c,b,i,j) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["101_voov"] = einsum('kbij,ak->bija', tmps_["36_ovoo"], t1)
    del tmps_["36_ovoo"]
    doubles_residual += einsum('aijb->abij', tmps_["101_voov"])

    # doubles_residual += -1.00 P(a,b) f(k,c) t1(a,k) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual -= einsum('bija->abij', tmps_["101_voov"])
    del tmps_["101_voov"]

    # tmps_[37_oo](l,j) = 0.50 eri[oovv](l,k,c,d) * t2(c,d,j,k) // flops: o2v0 = o3v2 | mem: o2v0 = o2v0
    tmps_["37_oo"] = 0.50 * einsum('lkcd,cdjk->lj', eri["oovv"], t2)

    if includes_["t3"]:

        # tmps_[83_vvvooo](a,b,c,j,k,i) = 1.00 t3(a,b,c,j,k,m) * eri[oovv](m,l,d,e) * t2(d,e,i,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["83_vvvooo"] = einsum('abcjkm,mi->abcjki', t3, tmps_["37_oo"])

        # triples_residual += -0.50 <m,l||d,e> t2(d,e,i,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcjki->abcijk', tmps_["83_vvvooo"])

        # triples_residual += -0.50 P(j,k) <m,l||d,e> t2(d,e,k,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcijk->abcijk', tmps_["83_vvvooo"])
        triples_residual += einsum('abcikj->abcijk', tmps_["83_vvvooo"])
        del tmps_["83_vvvooo"]

    # tmps_[104_vvoo](a,b,j,i) = 1.00 t2(a,b,j,l) * eri[oovv](l,k,c,d) * t2(c,d,i,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["104_vvoo"] = einsum('abjl,li->abji', t2, tmps_["37_oo"])
    del tmps_["37_oo"]
    doubles_residual += einsum('abji->abij', tmps_["104_vvoo"])

    # doubles_residual += -0.50 P(i,j) <l,k||c,d> t2(c,d,j,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual -= einsum('abij->abij', tmps_["104_vvoo"])
    del tmps_["104_vvoo"]

    # tmps_[38_voov](b,i,j,a) = 1.00 eri[vooo](b,k,i,j) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["38_voov"] = einsum('bkij,ak->bija', eri["vooo"], t1)

    # doubles_residual += +1.00 P(a,b) <k,a||i,j> t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual -= einsum('aijb->abij', tmps_["38_voov"])
    doubles_residual += einsum('bija->abij', tmps_["38_voov"])
    del tmps_["38_voov"]

    # tmps_[39_ovvo](j,a,b,i) = 1.00 f[oo](k,j) * t2(a,b,i,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["39_ovvo"] = einsum('kj,abik->jabi', f["oo"], t2)

    # doubles_residual += -1.00 P(i,j) f(k,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual -= einsum('jabi->abij', tmps_["39_ovvo"])
    doubles_residual += einsum('iabj->abij', tmps_["39_ovvo"])
    del tmps_["39_ovvo"]

    # tmps_[40_oooo](l,k,j,i) = 1.00 eri[oovo](l,k,c,j) * t1(c,i) // flops: o4v0 = o4v1 | mem: o4v0 = o4v0
    tmps_["40_oooo"] = einsum('lkcj,ci->lkji', eri["oovo"], t1)

    if includes_["t3"]:

        # tmps_[49_vvvooo](a,b,c,k,i,j) = 0.50 t3(a,b,c,k,m,l) * eri[oovo](m,l,d,i) * t1(d,j) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
        tmps_["49_vvvooo"] = 0.50 * einsum('abckml,mlij->abckij', t3, tmps_["40_oooo"])
        triples_residual -= einsum('abcjki->abcijk', tmps_["49_vvvooo"])
        triples_residual += einsum('abckji->abcijk', tmps_["49_vvvooo"])
        triples_residual -= einsum('abckij->abcijk', tmps_["49_vvvooo"])

        # triples_residual += +0.50 P(i,j) <m,l||d,k> t1(d,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcikj->abcijk', tmps_["49_vvvooo"])

        # triples_residual += -0.50 P(i,k) <m,l||d,j> t1(d,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcijk->abcijk', tmps_["49_vvvooo"])

        # triples_residual += +0.50 P(j,k) <m,l||d,i> t1(d,k) t3(a,b,c,j,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcjik->abcijk', tmps_["49_vvvooo"])
        del tmps_["49_vvvooo"]

    # tmps_[63_vvooov](a,b,j,i,k,c) = 1.00 eri[oovo](m,l,d,i) * t1(d,k) * t1(c,l) * t2(a,b,j,m) // flops: o3v3 = o4v1 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["63_vvooov"] = einsum('mlik,cl,abjm->abjikc', tmps_["40_oooo"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])
    triples_residual -= einsum('abjkic->abcijk', tmps_["63_vvooov"])

    # triples_residual += +1.00 P(j,k) <m,l||d,i> t1(d,k) t1(c,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('abjikc->abcijk', tmps_["63_vvooov"])

    # triples_residual += +1.00 P(i,j) <m,l||d,k> t1(d,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('abikjc->abcijk', tmps_["63_vvooov"])
    triples_residual -= einsum('ackjib->abcijk', tmps_["63_vvooov"])
    triples_residual -= einsum('bcjkia->abcijk', tmps_["63_vvooov"])
    triples_residual -= einsum('abkijc->abcijk', tmps_["63_vvooov"])
    triples_residual += einsum('acjkib->abcijk', tmps_["63_vvooov"])

    # triples_residual += +1.00 P(j,k) P(a,b) <m,l||d,i> t1(d,k) t1(a,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('bcjika->abcijk', tmps_["63_vvooov"])
    triples_residual += einsum('bckjia->abcijk', tmps_["63_vvooov"])
    triples_residual += einsum('abkjic->abcijk', tmps_["63_vvooov"])
    triples_residual += einsum('ackijb->abcijk', tmps_["63_vvooov"])

    # triples_residual += -1.00 P(i,k) P(a,b) <m,l||d,j> t1(d,k) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('bcijka->abcijk', tmps_["63_vvooov"])

    # triples_residual += -1.00 P(i,k) <m,l||d,j> t1(d,k) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual -= einsum('abijkc->abcijk', tmps_["63_vvooov"])

    # triples_residual += +1.00 P(i,j) P(a,b) <m,l||d,k> t1(d,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('bcikja->abcijk', tmps_["63_vvooov"])
    triples_residual += einsum('acijkb->abcijk', tmps_["63_vvooov"])
    triples_residual -= einsum('acikjb->abcijk', tmps_["63_vvooov"])
    triples_residual -= einsum('acjikb->abcijk', tmps_["63_vvooov"])
    triples_residual -= einsum('bckija->abcijk', tmps_["63_vvooov"])
    del tmps_["63_vvooov"]

    # tmps_[97_vvoo](b,a,j,i) = 1.00 eri[oovo](l,k,c,j) * t1(c,i) * t1(b,l) * t1(a,k) // flops: o2v2 = o4v1 o3v2 | mem: o2v2 = o3v1 o2v2
    tmps_["97_vvoo"] = einsum('lkji,bl,ak->baji', tmps_["40_oooo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["40_oooo"]
    doubles_residual += einsum('baij->abij', tmps_["97_vvoo"])

    # doubles_residual += -1.00 P(i,j) <l,k||c,j> t1(c,i) t1(a,k) t1(b,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual -= einsum('baji->abij', tmps_["97_vvoo"])
    del tmps_["97_vvoo"]

    # tmps_[41_ooov](m,i,j,c) = 1.00 eri[oooo](m,l,i,j) * t1(c,l) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["41_ooov"] = einsum('mlij,cl->mijc', eri["oooo"], t1)

    # doubles_residual += -1.00 <l,k||i,j> t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_residual -= einsum('lija,bl->abij', tmps_["41_ooov"], t1)

    # tmps_[76_oovvvo](i,k,b,a,c,j) = 1.00 eri[oooo](m,l,i,k) * t1(b,l) * t2(a,c,j,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["76_oovvvo"] = einsum('mikb,acjm->ikbacj', tmps_["41_ooov"], t2)
    del tmps_["41_ooov"]
    triples_residual -= einsum('jkbaci->abcijk', tmps_["76_oovvvo"])
    triples_residual -= einsum('ikcabj->abcijk', tmps_["76_oovvvo"])

    # triples_residual += +1.00 P(i,j) P(a,b) <m,l||j,k> t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('jkabci->abcijk', tmps_["76_oovvvo"])

    # triples_residual += +1.00 P(a,b) <m,l||i,j> t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('ijabck->abcijk', tmps_["76_oovvvo"])
    triples_residual -= einsum('ikabcj->abcijk', tmps_["76_oovvvo"])
    triples_residual -= einsum('ijback->abcijk', tmps_["76_oovvvo"])

    # triples_residual += +1.00 P(i,j) <m,l||j,k> t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('jkcabi->abcijk', tmps_["76_oovvvo"])

    # triples_residual += +1.00 <m,l||i,j> t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('ijcabk->abcijk', tmps_["76_oovvvo"])
    triples_residual += einsum('ikbacj->abcijk', tmps_["76_oovvvo"])
    del tmps_["76_oovvvo"]

    # tmps_[42_vv](b,d) = 1.00 eri[vovv](b,k,c,d) * t1(c,k) // flops: o0v2 = o1v3 | mem: o0v2 = o0v2
    tmps_["42_vv"] = einsum('bkcd,ck->bd', eri["vovv"], t1)

    # singles_residual += +1.00 <j,a||b,c> t1(b,j) t1(c,i)  // flops: o1v1 += o1v2 | mem: o1v1 += o1v1
    singles_residual -= einsum('ac,ci->ai', tmps_["42_vv"], t1)

    if includes_["t3"]:

        # tmps_[59_vvooov](b,c,i,j,k,a) = 1.00 t3(e,b,c,i,j,k) * eri[vovv](a,l,d,e) * t1(d,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
        tmps_["59_vvooov"] = einsum('ebcijk,ae->bcijka', t3, tmps_["42_vv"])

        # triples_residual += +1.00 P(a,b) <l,a||d,e> t1(d,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bcijka->abcijk', tmps_["59_vvooov"])

        # triples_residual += +1.00 <l,c||d,e> t1(d,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abijkc->abcijk', tmps_["59_vvooov"])
        triples_residual += einsum('acijkb->abcijk', tmps_["59_vvooov"])
        del tmps_["59_vvooov"]

    # tmps_[90_voov](a,i,j,b) = 1.00 t2(d,a,i,j) * eri[vovv](b,k,c,d) * t1(c,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["90_voov"] = einsum('daij,bd->aijb', t2, tmps_["42_vv"])
    del tmps_["42_vv"]

    # doubles_residual += +1.00 P(a,b) <k,a||c,d> t1(c,k) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual -= einsum('bija->abij', tmps_["90_voov"])
    doubles_residual += einsum('aijb->abij', tmps_["90_voov"])
    del tmps_["90_voov"]

    # tmps_[43_ov](l,d) = 1.00 eri[oovv](l,k,c,d) * t1(c,k) // flops: o1v1 = o2v2 | mem: o1v1 = o1v1
    tmps_["43_ov"] = einsum('lkcd,ck->ld', eri["oovv"], t1)

    # singles_residual += +1.00 <k,j||b,c> t1(b,j) t1(c,i) t1(a,k)  // flops: o1v1 += o2v1 o2v1 | mem: o1v1 += o2v0 o1v1
    singles_residual += einsum('kc,ci,ak->ai', tmps_["43_ov"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t3"]:

        # doubles_residual += -1.00 <l,k||c,d> t1(c,k) t3(d,a,b,i,j,l)  // flops: o2v2 += o3v3 | mem: o2v2 += o2v2
        doubles_residual -= einsum('ld,dabijl->abij', tmps_["43_ov"], t3)

    # tmps_[70_voovvo](c,i,j,a,b,k) = 1.00 t2(e,c,i,j) * eri[oovv](m,l,d,e) * t1(d,l) * t2(a,b,k,m) // flops: o3v3 = o3v2 o4v3 | mem: o3v3 = o3v1 o3v3
    tmps_["70_voovvo"] = einsum('ecij,me,abkm->cijabk', t2, tmps_["43_ov"], t2, optimize=['einsum_path',(0,1),(0,1)])
    triples_residual -= einsum('cikabj->abcijk', tmps_["70_voovvo"])
    triples_residual += einsum('bikacj->abcijk', tmps_["70_voovvo"])
    triples_residual -= einsum('bijack->abcijk', tmps_["70_voovvo"])

    # triples_residual += +1.00 P(i,j) <m,l||d,e> t1(d,l) t2(e,c,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('cjkabi->abcijk', tmps_["70_voovvo"])

    # triples_residual += +1.00 <m,l||d,e> t1(d,l) t2(e,c,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('cijabk->abcijk', tmps_["70_voovvo"])

    # triples_residual += +1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,l) t2(e,a,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('ajkbci->abcijk', tmps_["70_voovvo"])

    # triples_residual += +1.00 P(a,b) <m,l||d,e> t1(d,l) t2(e,a,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_residual += einsum('aijbck->abcijk', tmps_["70_voovvo"])
    triples_residual -= einsum('bjkaci->abcijk', tmps_["70_voovvo"])
    triples_residual -= einsum('aikbcj->abcijk', tmps_["70_voovvo"])
    del tmps_["70_voovvo"]

    if includes_["t3"]:

        # tmps_[79_vvooov](a,c,i,j,k,b) = 1.00 t3(e,a,c,i,j,k) * eri[oovv](m,l,d,e) * t1(d,l) * t1(b,m) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
        tmps_["79_vvooov"] = einsum('eacijk,me,bm->acijkb', t3, tmps_["43_ov"], t1, optimize=['einsum_path',(0,1),(0,1)])

        # triples_residual += +1.00 <m,l||d,e> t1(d,l) t1(c,m) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abijkc->abcijk', tmps_["79_vvooov"])
        triples_residual -= einsum('acijkb->abcijk', tmps_["79_vvooov"])

        # triples_residual += +1.00 P(a,b) <m,l||d,e> t1(d,l) t1(a,m) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bcijka->abcijk', tmps_["79_vvooov"])
        del tmps_["79_vvooov"]

        # tmps_[81_ovvvoo](k,a,b,c,i,j) = 1.00 t1(e,k) * eri[oovv](m,l,d,e) * t1(d,l) * t3(a,b,c,i,j,m) // flops: o3v3 = o2v1 o4v3 | mem: o3v3 = o2v0 o3v3
        tmps_["81_ovvvoo"] = einsum('ek,me,abcijm->kabcij', t1, tmps_["43_ov"], t3, optimize=['einsum_path',(0,1),(0,1)])
        triples_residual -= einsum('jabcik->abcijk', tmps_["81_ovvvoo"])

        # triples_residual += +1.00 P(j,k) <m,l||d,e> t1(d,l) t1(e,k) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('kabcij->abcijk', tmps_["81_ovvvoo"])

        # triples_residual += +1.00 <m,l||d,e> t1(d,l) t1(e,i) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('iabcjk->abcijk', tmps_["81_ovvvoo"])
        del tmps_["81_ovvvoo"]

    # tmps_[95_vvoo](b,a,i,j) = 1.00 t2(d,a,i,j) * eri[oovv](l,k,c,d) * t1(c,k) * t1(b,l) // flops: o2v2 = o3v2 o3v2 | mem: o2v2 = o3v1 o2v2
    tmps_["95_vvoo"] = einsum('daij,ld,bl->baij', t2, tmps_["43_ov"], t1, optimize=['einsum_path',(0,1),(0,1)])

    # doubles_residual += +1.00 P(a,b) <l,k||c,d> t1(c,k) t1(a,l) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual += einsum('abij->abij', tmps_["95_vvoo"])
    doubles_residual -= einsum('baij->abij', tmps_["95_vvoo"])
    del tmps_["95_vvoo"]

    # tmps_[98_vvoo](a,b,i,j) = 1.00 eri[oovv](l,k,c,d) * t1(c,k) * t1(d,j) * t2(a,b,i,l) // flops: o2v2 = o2v1 o3v2 | mem: o2v2 = o2v0 o2v2
    tmps_["98_vvoo"] = einsum('ld,dj,abil->abij', tmps_["43_ov"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["43_ov"]
    doubles_residual -= einsum('abji->abij', tmps_["98_vvoo"])

    # doubles_residual += +1.00 P(i,j) <l,k||c,d> t1(c,k) t1(d,j) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual += einsum('abij->abij', tmps_["98_vvoo"])
    del tmps_["98_vvoo"]

    # tmps_[44_oo](l,i) = 1.00 eri[oovo](l,k,c,i) * t1(c,k) // flops: o2v0 = o3v1 | mem: o2v0 = o2v0
    tmps_["44_oo"] = einsum('lkci,ck->li', eri["oovo"], t1)

    # singles_residual += +1.00 <k,j||b,i> t1(b,j) t1(a,k)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_residual += einsum('ak,ki->ai', t1, tmps_["44_oo"])

    if includes_["t3"]:

        # tmps_[84_vvvooo](a,b,c,i,k,j) = 1.00 t3(a,b,c,i,k,m) * eri[oovo](m,l,d,j) * t1(d,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["84_vvvooo"] = einsum('abcikm,mj->abcikj', t3, tmps_["44_oo"])

        # triples_residual += +1.00 <m,l||d,i> t1(d,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcjki->abcijk', tmps_["84_vvvooo"])
        triples_residual -= einsum('abcikj->abcijk', tmps_["84_vvvooo"])

        # triples_residual += +1.00 P(j,k) <m,l||d,k> t1(d,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcijk->abcijk', tmps_["84_vvvooo"])
        del tmps_["84_vvvooo"]

    # tmps_[99_ovvo](j,a,b,i) = 1.00 eri[oovo](l,k,c,j) * t1(c,k) * t2(a,b,i,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["99_ovvo"] = einsum('lj,abil->jabi', tmps_["44_oo"], t2)
    del tmps_["44_oo"]
    doubles_residual -= einsum('iabj->abij', tmps_["99_ovvo"])

    # doubles_residual += +1.00 P(i,j) <l,k||c,j> t1(c,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual += einsum('jabi->abij', tmps_["99_ovvo"])
    del tmps_["99_ovvo"]

    # tmps_[45_oo](k,j) = 1.00 f[ov](k,c) * t1(c,j) // flops: o2v0 = o2v1 | mem: o2v0 = o2v0
    tmps_["45_oo"] = einsum('kc,cj->kj', f["ov"], t1)

    # singles_residual += -1.00 f(j,b) t1(b,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_residual -= einsum('aj,ji->ai', t1, tmps_["45_oo"])

    if includes_["t3"]:

        # tmps_[82_vvvooo](a,b,c,i,k,j) = 1.00 t3(a,b,c,i,k,l) * f[ov](l,d) * t1(d,j) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["82_vvvooo"] = einsum('abcikl,lj->abcikj', t3, tmps_["45_oo"])

        # triples_residual += -1.00 P(j,k) f(l,d) t1(d,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcijk->abcijk', tmps_["82_vvvooo"])

        # triples_residual += -1.00 f(l,d) t1(d,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcjki->abcijk', tmps_["82_vvvooo"])
        triples_residual += einsum('abcikj->abcijk', tmps_["82_vvvooo"])
        del tmps_["82_vvvooo"]

    # tmps_[100_ovvo](j,a,b,i) = 1.00 f[ov](k,c) * t1(c,j) * t2(a,b,i,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["100_ovvo"] = einsum('kj,abik->jabi', tmps_["45_oo"], t2)
    del tmps_["45_oo"]
    doubles_residual += einsum('iabj->abij', tmps_["100_ovvo"])

    # doubles_residual += -1.00 P(i,j) f(k,c) t1(c,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_residual -= einsum('jabi->abij', tmps_["100_ovvo"])
    del tmps_["100_ovvo"]

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



