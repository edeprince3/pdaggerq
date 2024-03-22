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

    if includes_["t1"]:

        # singles_residual += -1.00 f(j,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
        singles_residual -= einsum('ji,aj->ai', f["oo"], t1)

        # singles_residual += +1.00 f(a,b) t1(b,i)  // flops: o1v1 += o1v2 | mem: o1v1 += o1v1
        singles_residual += einsum('ab,bi->ai', f["vv"], t1)

        # singles_residual += +1.00 <j,a||b,i> t1(b,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
        singles_residual -= einsum('ajbi,bj->ai', eri["vovo"], t1)

    if includes_["t1"] and includes_["t2"]:

        # singles_residual += +0.50 <k,j||b,c> t1(a,j) t2(b,c,i,k)  // flops: o1v1 += o3v2 o2v1 | mem: o1v1 += o2v0 o1v1
        singles_residual += 0.50 * einsum('kjbc,bcik,aj->ai', eri["oovv"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t2"]:

        # singles_residual += -1.00 f(j,b) t2(b,a,i,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
        singles_residual -= einsum('jb,baij->ai', f["ov"], t2)

        # singles_residual += -0.50 <k,j||b,i> t2(b,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
        singles_residual -= 0.50 * einsum('kjbi,bakj->ai', eri["oovo"], t2)

        # singles_residual += -0.50 <j,a||b,c> t2(b,c,i,j)  // flops: o1v1 += o2v3 | mem: o1v1 += o1v1
        singles_residual += 0.50 * einsum('ajbc,bcij->ai', eri["vovv"], t2)

        # doubles_residual += +0.50 <l,k||i,j> t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
        doubles_residual += 0.50 * einsum('lkij,ablk->abij', eri["oooo"], t2)

        # doubles_residual += +0.50 <a,b||c,d> t2(c,d,i,j)  // flops: o2v2 += o2v4 | mem: o2v2 += o2v2
        doubles_residual += 0.50 * einsum('abcd,cdij->abij', eri["vvvv"], t2)

    if includes_["t3"]:

        # doubles_residual += +1.00 f(k,c) t3(c,a,b,i,j,k)  // flops: o2v2 += o3v3 | mem: o2v2 += o2v2
        doubles_residual += einsum('kc,cabijk->abij', f["ov"], t3)

        # singles_residual += +0.25 <k,j||b,c> t3(b,c,a,i,k,j)  // flops: o1v1 += o3v3 | mem: o1v1 += o1v1
        singles_residual += 0.25 * einsum('kjbc,bcaikj->ai', eri["oovv"], t3)

        # tmps_[1_vvvooo](a,c,b,k,j,i) = 0.50 eri[vvvv](b,c,d,e) * t3(d,e,a,i,j,k) // flops: o3v3 = o3v5 | mem: o3v3 = o3v3
        tmps_["1_vvvooo"] = 0.50 * einsum('bcde,deaijk->acbkji', eri["vvvv"], t3)

        # triples_residual = +0.50 P(b,c) <a,b||d,e> t3(d,e,c,i,j,k)  // flops: o3v3 = o3v3 | mem: o3v3 = o3v3
        triples_residual = 1.00 * einsum('cbakji->abcijk', tmps_["1_vvvooo"])

        # triples_residual += +0.50 <b,c||d,e> t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbkji->abcijk', tmps_["1_vvvooo"])
        triples_residual -= einsum('bcakji->abcijk', tmps_["1_vvvooo"])
        del tmps_["1_vvvooo"]


        # tmps_[2_vvvooo](b,a,c,j,i,k) = 1.00 eri[vovo](c,l,d,k) * t3(d,a,b,i,j,l) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
        tmps_["2_vvvooo"] = einsum('cldk,dabijl->bacjik', eri["vovo"], t3)
        triples_residual += einsum('cabjik->abcijk', tmps_["2_vvvooo"])

        # triples_residual += +1.00 <l,c||d,i> t3(d,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('backji->abcijk', tmps_["2_vvvooo"])

        # triples_residual += +1.00 P(a,b) <l,a||d,i> t3(d,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbakji->abcijk', tmps_["2_vvvooo"])
        triples_residual += einsum('cbakij->abcijk', tmps_["2_vvvooo"])

        # triples_residual += +1.00 P(j,k) <l,c||d,k> t3(d,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bacjik->abcijk', tmps_["2_vvvooo"])
        triples_residual += einsum('backij->abcijk', tmps_["2_vvvooo"])
        triples_residual += einsum('cabkji->abcijk', tmps_["2_vvvooo"])

        # triples_residual += +1.00 P(j,k) P(a,b) <l,a||d,k> t3(d,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbajik->abcijk', tmps_["2_vvvooo"])
        triples_residual -= einsum('cabkij->abcijk', tmps_["2_vvvooo"])
        del tmps_["2_vvvooo"]


        # tmps_[3_vvoooo](a,c,k,j,i,l) = 0.50 eri[vovv](c,l,d,e) * t3(d,e,a,i,j,k) // flops: o4v2 = o4v4 | mem: o4v2 = o4v2
        tmps_["3_vvoooo"] = 0.50 * einsum('clde,deaijk->ackjil', eri["vovv"], t3)

    if includes_["t1"] and includes_["t3"]:

        # tmps_[80_vvvooo](c,b,a,i,j,k) = 1.00 t1(a,l) * eri[vovv](c,l,d,e) * t3(d,e,b,i,j,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["80_vvvooo"] = einsum('al,bckjil->cbaijk', t1, tmps_["3_vvoooo"])

    if includes_["t3"]:
        del tmps_["3_vvoooo"]


    if includes_["t1"] and includes_["t3"]:
        triples_residual += einsum('abcijk->abcijk', tmps_["80_vvvooo"])

        # triples_residual += +0.50 P(b,c) <l,a||d,e> t1(b,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbijk->abcijk', tmps_["80_vvvooo"])
        triples_residual += einsum('cabijk->abcijk', tmps_["80_vvvooo"])

        # triples_residual += +0.50 P(a,b) <l,c||d,e> t1(a,l) t3(d,e,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbaijk->abcijk', tmps_["80_vvvooo"])
        triples_residual -= einsum('bacijk->abcijk', tmps_["80_vvvooo"])

        # triples_residual += -0.50 P(a,c) <l,b||d,e> t1(a,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bcaijk->abcijk', tmps_["80_vvvooo"])
        del tmps_["80_vvvooo"]


    if includes_["t3"]:

        # tmps_[4_vvoooo](c,a,k,i,j,l) = 1.00 eri[oovo](m,l,d,j) * t3(d,a,c,i,k,m) // flops: o4v2 = o5v3 | mem: o4v2 = o4v2
        tmps_["4_vvoooo"] = einsum('mldj,dacikm->cakijl', eri["oovo"], t3)

    if includes_["t1"] and includes_["t3"]:

        # tmps_[75_vvvooo](b,a,c,i,j,k) = 1.00 eri[oovo](m,l,d,i) * t3(d,a,c,j,k,m) * t1(b,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["75_vvvooo"] = einsum('cakjil,bl->bacijk', tmps_["4_vvoooo"], t1)

    if includes_["t3"]:
        del tmps_["4_vvoooo"]


    if includes_["t1"] and includes_["t3"]:
        triples_residual += einsum('abcjik->abcijk', tmps_["75_vvvooo"])

        # triples_residual += -1.00 P(j,k) <m,l||d,k> t1(c,l) t3(d,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cabkij->abcijk', tmps_["75_vvvooo"])
        triples_residual -= einsum('bacjik->abcijk', tmps_["75_vvvooo"])
        triples_residual += einsum('cabjik->abcijk', tmps_["75_vvvooo"])
        triples_residual += einsum('bacijk->abcijk', tmps_["75_vvvooo"])

        # triples_residual += -1.00 P(j,k) P(a,b) <m,l||d,k> t1(a,l) t3(d,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abckij->abcijk', tmps_["75_vvvooo"])
        triples_residual += einsum('backij->abcijk', tmps_["75_vvvooo"])

        # triples_residual += -1.00 <m,l||d,i> t1(c,l) t3(d,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cabijk->abcijk', tmps_["75_vvvooo"])

        # triples_residual += -1.00 P(a,b) <m,l||d,i> t1(a,l) t3(d,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcijk->abcijk', tmps_["75_vvvooo"])
        del tmps_["75_vvvooo"]


    if includes_["t3"]:

        # tmps_[5_vooooo](a,k,j,i,l,m) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,a,i,j,k) // flops: o5v1 = o5v3 | mem: o5v1 = o5v1
        tmps_["5_vooooo"] = 0.50 * einsum('mlde,deaijk->akjilm', eri["oovv"], t3)

    if includes_["t2"] and includes_["t3"]:

        # tmps_[53_vvvooo](c,b,a,i,j,k) = 0.50 t2(a,b,m,l) * eri[oovv](m,l,d,e) * t3(d,e,c,i,j,k) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
        tmps_["53_vvvooo"] = 0.50 * einsum('abml,ckjilm->cbaijk', t2, tmps_["5_vooooo"])
        triples_residual -= einsum('bcaijk->abcijk', tmps_["53_vvvooo"])

        # triples_residual += +0.25 <m,l||d,e> t2(b,c,m,l) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbijk->abcijk', tmps_["53_vvvooo"])

        # triples_residual += +0.25 P(b,c) <m,l||d,e> t2(a,b,m,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaijk->abcijk', tmps_["53_vvvooo"])
        del tmps_["53_vvvooo"]


    if includes_["t1"] and includes_["t3"]:

        # tmps_[82_vvvooo](b,a,c,k,j,i) = 1.00 eri[oovv](m,l,d,e) * t3(d,e,a,i,j,k) * t1(c,m) * t1(b,l) // flops: o3v3 = o5v2 o4v3 | mem: o3v3 = o4v2 o3v3
        tmps_["82_vvvooo"] = einsum('akjilm,cm,bl->backji', tmps_["5_vooooo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t3"]:
        del tmps_["5_vooooo"]


    if includes_["t1"] and includes_["t3"]:

        # triples_residual += -0.50 P(b,c) <m,l||d,e> t1(a,l) t1(b,m) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbkji->abcijk', tmps_["82_vvvooo"])

        # triples_residual += -0.50 <m,l||d,e> t1(b,l) t1(c,m) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('backji->abcijk', tmps_["82_vvvooo"])
        triples_residual += einsum('abckji->abcijk', tmps_["82_vvvooo"])
        del tmps_["82_vvvooo"]


    if includes_["t3"]:

        # tmps_[6_vvvooo](c,b,a,j,k,i) = 0.50 eri[oooo](m,l,i,k) * t3(a,b,c,j,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
        tmps_["6_vvvooo"] = 0.50 * einsum('mlik,abcjml->cbajki', eri["oooo"], t3)

        # triples_residual += +0.50 <m,l||i,j> t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbakji->abcijk', tmps_["6_vvvooo"])

        # triples_residual += +0.50 P(i,j) <m,l||j,k> t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaikj->abcijk', tmps_["6_vvvooo"])
        triples_residual -= einsum('cbajki->abcijk', tmps_["6_vvvooo"])
        del tmps_["6_vvvooo"]


        # tmps_[7_vvvo](b,a,d,i) = 0.50 eri[oovv](m,l,d,e) * t3(e,a,b,i,m,l) // flops: o1v3 = o3v4 | mem: o1v3 = o1v3
        tmps_["7_vvvo"] = 0.50 * einsum('mlde,eabiml->badi', eri["oovv"], t3)

    if includes_["t2"] and includes_["t3"]:

        # tmps_[60_vvvooo](a,b,c,k,j,i) = 1.00 t2(d,c,i,j) * eri[oovv](m,l,d,e) * t3(e,a,b,k,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
        tmps_["60_vvvooo"] = einsum('dcij,badk->abckji', t2, tmps_["7_vvvo"])

        # triples_residual += -0.50 P(i,j) P(a,b) <m,l||d,e> t2(d,a,j,k) t3(e,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bcaikj->abcijk', tmps_["60_vvvooo"])
        triples_residual += einsum('bcajki->abcijk', tmps_["60_vvvooo"])
        triples_residual += einsum('acbikj->abcijk', tmps_["60_vvvooo"])
        triples_residual += einsum('acbkji->abcijk', tmps_["60_vvvooo"])

        # triples_residual += -0.50 P(a,b) <m,l||d,e> t2(d,a,i,j) t3(e,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bcakji->abcijk', tmps_["60_vvvooo"])

        # triples_residual += -0.50 <m,l||d,e> t2(d,c,i,j) t3(e,a,b,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abckji->abcijk', tmps_["60_vvvooo"])
        triples_residual += einsum('abcjki->abcijk', tmps_["60_vvvooo"])
        triples_residual -= einsum('acbjki->abcijk', tmps_["60_vvvooo"])

        # triples_residual += -0.50 P(i,j) <m,l||d,e> t2(d,c,j,k) t3(e,a,b,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcikj->abcijk', tmps_["60_vvvooo"])
        del tmps_["60_vvvooo"]


    if includes_["t1"] and includes_["t3"]:

        # tmps_[91_vvoo](a,b,i,j) = 1.00 t1(c,j) * eri[oovv](l,k,c,d) * t3(d,a,b,i,l,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
        tmps_["91_vvoo"] = einsum('cj,baci->abij', t1, tmps_["7_vvvo"])

    if includes_["t3"]:
        del tmps_["7_vvvo"]


    if includes_["t1"] and includes_["t3"]:

        # doubles_residual += -0.50 P(i,j) <l,k||c,d> t1(c,j) t3(d,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('abij->abij', tmps_["91_vvoo"])
        doubles_residual += einsum('abji->abij', tmps_["91_vvoo"])
        del tmps_["91_vvoo"]


    if includes_["t2"]:

        # tmps_[8_vvvooo](a,c,b,k,j,i) = 1.00 eri[vvvo](b,c,d,i) * t2(d,a,j,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
        tmps_["8_vvvooo"] = einsum('bcdi,dajk->acbkji', eri["vvvo"], t2)
        triples_residual += einsum('cbakij->abcijk', tmps_["8_vvvooo"])

        # triples_residual += -1.00 P(j,k) <b,c||d,k> t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbjik->abcijk', tmps_["8_vvvooo"])
        triples_residual += einsum('bcajik->abcijk', tmps_["8_vvvooo"])
        triples_residual -= einsum('bcakij->abcijk', tmps_["8_vvvooo"])

        # triples_residual += -1.00 P(j,k) P(b,c) <a,b||d,k> t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbajik->abcijk', tmps_["8_vvvooo"])
        triples_residual += einsum('bcakji->abcijk', tmps_["8_vvvooo"])

        # triples_residual += -1.00 <b,c||d,i> t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbkji->abcijk', tmps_["8_vvvooo"])

        # triples_residual += -1.00 P(b,c) <a,b||d,i> t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbakji->abcijk', tmps_["8_vvvooo"])
        triples_residual += einsum('acbkij->abcijk', tmps_["8_vvvooo"])
        del tmps_["8_vvvooo"]


    if includes_["t3"]:

        # tmps_[9_vvvooo](c,a,b,k,j,i) = 1.00 f[vv](b,d) * t3(d,a,c,i,j,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
        tmps_["9_vvvooo"] = einsum('bd,dacijk->cabkji', f["vv"], t3)

        # triples_residual += +1.00 f(c,d) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('backji->abcijk', tmps_["9_vvvooo"])
        triples_residual -= einsum('cabkji->abcijk', tmps_["9_vvvooo"])

        # triples_residual += +1.00 P(a,b) f(a,d) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbakji->abcijk', tmps_["9_vvvooo"])
        del tmps_["9_vvvooo"]


        # tmps_[10_vvoo](a,b,j,i) = 0.50 eri[vovv](b,k,c,d) * t3(c,d,a,i,j,k) // flops: o2v2 = o3v4 | mem: o2v2 = o2v2
        tmps_["10_vvoo"] = 0.50 * einsum('bkcd,cdaijk->abji', eri["vovv"], t3)
        doubles_residual += einsum('abji->abij', tmps_["10_vvoo"])

        # doubles_residual += +0.50 P(a,b) <k,a||c,d> t3(c,d,b,i,j,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('baji->abij', tmps_["10_vvoo"])
        del tmps_["10_vvoo"]


    if includes_["t2"]:

        # tmps_[11_vvoooo](c,b,k,j,i,l) = 1.00 eri[vovo](b,l,d,i) * t2(d,c,j,k) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
        tmps_["11_vvoooo"] = einsum('bldi,dcjk->cbkjil', eri["vovo"], t2)

    if includes_["t1"] and includes_["t2"]:

        # tmps_[66_vvvooo](b,c,a,j,i,k) = 1.00 t1(a,l) * eri[vovo](b,l,d,j) * t2(d,c,i,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["66_vvvooo"] = einsum('al,cbkijl->bcajik', t1, tmps_["11_vvoooo"])

    if includes_["t2"]:
        del tmps_["11_vvoooo"]


    if includes_["t1"] and includes_["t2"]:
        triples_residual -= einsum('cabkij->abcijk', tmps_["66_vvvooo"])

        # triples_residual += -1.00 P(j,k) P(a,b) <l,c||d,k> t1(a,l) t2(d,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbakij->abcijk', tmps_["66_vvvooo"])
        triples_residual += einsum('backij->abcijk', tmps_["66_vvvooo"])
        triples_residual += einsum('bcajik->abcijk', tmps_["66_vvvooo"])

        # triples_residual += +1.00 P(j,k) P(a,c) <l,b||d,k> t1(a,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bcakij->abcijk', tmps_["66_vvvooo"])
        triples_residual += einsum('abcjik->abcijk', tmps_["66_vvvooo"])
        triples_residual += einsum('cabjik->abcijk', tmps_["66_vvvooo"])

        # triples_residual += -1.00 P(a,b) <l,c||d,i> t1(a,l) t2(d,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaijk->abcijk', tmps_["66_vvvooo"])

        # triples_residual += -1.00 P(b,c) <l,a||d,i> t1(b,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbijk->abcijk', tmps_["66_vvvooo"])

        # triples_residual += -1.00 P(j,k) P(b,c) <l,a||d,k> t1(b,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbkij->abcijk', tmps_["66_vvvooo"])

        # triples_residual += +1.00 P(a,c) <l,b||d,i> t1(a,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bcaijk->abcijk', tmps_["66_vvvooo"])
        triples_residual -= einsum('bacjik->abcijk', tmps_["66_vvvooo"])
        triples_residual -= einsum('cbajik->abcijk', tmps_["66_vvvooo"])
        triples_residual -= einsum('abckij->abcijk', tmps_["66_vvvooo"])
        triples_residual -= einsum('acbjik->abcijk', tmps_["66_vvvooo"])
        triples_residual -= einsum('cabijk->abcijk', tmps_["66_vvvooo"])
        triples_residual -= einsum('abcijk->abcijk', tmps_["66_vvvooo"])
        triples_residual += einsum('bacijk->abcijk', tmps_["66_vvvooo"])
        del tmps_["66_vvvooo"]


    if includes_["t3"]:

        # tmps_[12_vooo](a,k,i,l) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,a,i,k,m) // flops: o3v1 = o4v3 | mem: o3v1 = o3v1
        tmps_["12_vooo"] = 0.50 * einsum('mlde,deaikm->akil', eri["oovv"], t3)

    if includes_["t2"] and includes_["t3"]:

        # tmps_[77_vvvooo](c,b,a,j,k,i) = 1.00 t2(a,b,i,l) * eri[oovv](m,l,d,e) * t3(d,e,c,j,k,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["77_vvvooo"] = einsum('abil,ckjl->cbajki', t2, tmps_["12_vooo"])

        # triples_residual += -0.50 P(b,c) <m,l||d,e> t2(a,b,i,l) t3(d,e,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbajki->abcijk', tmps_["77_vvvooo"])
        triples_residual -= einsum('bcaikj->abcijk', tmps_["77_vvvooo"])
        triples_residual += einsum('bcaijk->abcijk', tmps_["77_vvvooo"])
        triples_residual += einsum('bcajki->abcijk', tmps_["77_vvvooo"])
        triples_residual += einsum('acbikj->abcijk', tmps_["77_vvvooo"])

        # triples_residual += -0.50 <m,l||d,e> t2(b,c,i,l) t3(d,e,a,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbjki->abcijk', tmps_["77_vvvooo"])

        # triples_residual += -0.50 P(j,k) <m,l||d,e> t2(b,c,k,l) t3(d,e,a,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbijk->abcijk', tmps_["77_vvvooo"])

        # triples_residual += -0.50 P(j,k) P(b,c) <m,l||d,e> t2(a,b,k,l) t3(d,e,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbaijk->abcijk', tmps_["77_vvvooo"])
        triples_residual += einsum('cbaikj->abcijk', tmps_["77_vvvooo"])
        del tmps_["77_vvvooo"]


    if includes_["t1"] and includes_["t3"]:

        # tmps_[104_vvoo](b,a,i,j) = 1.00 t1(a,k) * eri[oovv](l,k,c,d) * t3(c,d,b,i,j,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
        tmps_["104_vvoo"] = einsum('ak,bjik->baij', t1, tmps_["12_vooo"])

    if includes_["t3"]:
        del tmps_["12_vooo"]


    if includes_["t1"] and includes_["t3"]:
        doubles_residual += einsum('abij->abij', tmps_["104_vvoo"])

        # doubles_residual += -0.50 P(a,b) <l,k||c,d> t1(a,k) t3(c,d,b,i,j,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('baij->abij', tmps_["104_vvoo"])
        del tmps_["104_vvoo"]


    if includes_["t2"]:

        # tmps_[13_vvvooo](b,a,c,i,k,j) = 1.00 eri[vooo](c,l,j,k) * t2(a,b,i,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["13_vvvooo"] = einsum('cljk,abil->bacikj', eri["vooo"], t2)
        triples_residual -= einsum('cabkji->abcijk', tmps_["13_vvvooo"])

        # triples_residual += -1.00 P(a,b) <l,a||i,j> t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbakji->abcijk', tmps_["13_vvvooo"])
        triples_residual -= einsum('cabikj->abcijk', tmps_["13_vvvooo"])
        triples_residual += einsum('cabjki->abcijk', tmps_["13_vvvooo"])

        # triples_residual += -1.00 P(i,j) P(a,b) <l,a||j,k> t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaikj->abcijk', tmps_["13_vvvooo"])

        # triples_residual += -1.00 <l,c||i,j> t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('backji->abcijk', tmps_["13_vvvooo"])

        # triples_residual += -1.00 P(i,j) <l,c||j,k> t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bacikj->abcijk', tmps_["13_vvvooo"])
        triples_residual -= einsum('bacjki->abcijk', tmps_["13_vvvooo"])
        triples_residual -= einsum('cbajki->abcijk', tmps_["13_vvvooo"])
        del tmps_["13_vvvooo"]


    if includes_["t3"]:

        # tmps_[14_vvoooo](c,b,k,j,i,l) = 1.00 f[ov](l,d) * t3(d,b,c,i,j,k) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
        tmps_["14_vvoooo"] = einsum('ld,dbcijk->cbkjil', f["ov"], t3)

    if includes_["t1"] and includes_["t3"]:

        # tmps_[86_vvvooo](a,b,c,i,j,k) = 1.00 f[ov](l,d) * t3(d,b,c,i,j,k) * t1(a,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["86_vvvooo"] = einsum('cbkjil,al->abcijk', tmps_["14_vvoooo"], t1)

    if includes_["t3"]:
        del tmps_["14_vvoooo"]


    if includes_["t1"] and includes_["t3"]:
        triples_residual += einsum('bacijk->abcijk', tmps_["86_vvvooo"])

        # triples_residual += -1.00 P(a,b) f(l,d) t1(a,l) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcijk->abcijk', tmps_["86_vvvooo"])

        # triples_residual += -1.00 f(l,d) t1(c,l) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cabijk->abcijk', tmps_["86_vvvooo"])
        del tmps_["86_vvvooo"]


    if includes_["t3"]:

        # tmps_[15_vvvooo](c,b,a,k,j,i) = 1.00 f[oo](l,i) * t3(a,b,c,j,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["15_vvvooo"] = einsum('li,abcjkl->cbakji', f["oo"], t3)
        triples_residual += einsum('cbakij->abcijk', tmps_["15_vvvooo"])

        # triples_residual += -1.00 P(j,k) f(l,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbajik->abcijk', tmps_["15_vvvooo"])

        # triples_residual += -1.00 f(l,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbakji->abcijk', tmps_["15_vvvooo"])
        del tmps_["15_vvvooo"]


        # tmps_[16_vvoo](b,a,j,i) = 0.50 eri[oovo](l,k,c,i) * t3(c,a,b,j,l,k) // flops: o2v2 = o4v3 | mem: o2v2 = o2v2
        tmps_["16_vvoo"] = 0.50 * einsum('lkci,cabjlk->baji', eri["oovo"], t3)
        doubles_residual -= einsum('baji->abij', tmps_["16_vvoo"])

        # doubles_residual += +0.50 P(i,j) <l,k||c,j> t3(c,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual += einsum('baij->abij', tmps_["16_vvoo"])
        del tmps_["16_vvoo"]


    if includes_["t2"]:

        # tmps_[17_vooooo](a,k,i,j,l,m) = 1.00 eri[oovo](m,l,d,j) * t2(d,a,i,k) // flops: o5v1 = o5v2 | mem: o5v1 = o5v1
        tmps_["17_vooooo"] = einsum('mldj,daik->akijlm', eri["oovo"], t2)

    if includes_["t1"] and includes_["t2"]:

        # tmps_[71_vvvooo](b,c,a,j,i,k) = 1.00 t1(c,m) * eri[oovo](m,l,d,k) * t2(d,a,i,j) * t1(b,l) // flops: o3v3 = o5v2 o4v3 | mem: o3v3 = o4v2 o3v3
        tmps_["71_vvvooo"] = einsum('cm,ajiklm,bl->bcajik', t1, tmps_["17_vooooo"], t1, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t2"]:
        del tmps_["17_vooooo"]


    if includes_["t1"] and includes_["t2"]:

        # triples_residual += +1.00 <m,l||d,i> t1(b,l) t1(c,m) t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bcakji->abcijk', tmps_["71_vvvooo"])

        # triples_residual += +1.00 P(b,c) <m,l||d,i> t1(a,l) t1(b,m) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abckji->abcijk', tmps_["71_vvvooo"])
        triples_residual -= einsum('acbkji->abcijk', tmps_["71_vvvooo"])

        # triples_residual += +1.00 P(j,k) P(b,c) <m,l||d,k> t1(a,l) t1(b,m) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcjik->abcijk', tmps_["71_vvvooo"])
        triples_residual -= einsum('abckij->abcijk', tmps_["71_vvvooo"])
        triples_residual += einsum('acbkij->abcijk', tmps_["71_vvvooo"])
        triples_residual -= einsum('bcakij->abcijk', tmps_["71_vvvooo"])

        # triples_residual += +1.00 P(j,k) <m,l||d,k> t1(b,l) t1(c,m) t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bcajik->abcijk', tmps_["71_vvvooo"])
        triples_residual -= einsum('acbjik->abcijk', tmps_["71_vvvooo"])
        del tmps_["71_vvvooo"]


    if includes_["t2"]:

        # tmps_[18_vvvo](c,d,a,i) = 1.00 eri[vovv](a,l,d,e) * t2(e,c,i,l) // flops: o1v3 = o2v4 | mem: o1v3 = o1v3
        tmps_["18_vvvo"] = einsum('alde,ecil->cdai', eri["vovv"], t2)

        # tmps_[56_vvvooo](b,c,a,k,j,i) = 1.00 t2(d,a,i,j) * eri[vovv](b,l,d,e) * t2(e,c,k,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
        tmps_["56_vvvooo"] = einsum('daij,cdbk->bcakji', t2, tmps_["18_vvvo"])

        # triples_residual += +1.00 P(i,j) P(a,b) <l,a||d,e> t2(d,b,j,k) t2(e,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbikj->abcijk', tmps_["56_vvvooo"])
        triples_residual += einsum('bcaikj->abcijk', tmps_["56_vvvooo"])
        triples_residual += einsum('cbajki->abcijk', tmps_["56_vvvooo"])
        triples_residual += einsum('acbjki->abcijk', tmps_["56_vvvooo"])
        triples_residual += einsum('bcakji->abcijk', tmps_["56_vvvooo"])
        triples_residual -= einsum('bcajki->abcijk', tmps_["56_vvvooo"])

        # triples_residual += +1.00 <l,c||d,e> t2(d,a,i,j) t2(e,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbakji->abcijk', tmps_["56_vvvooo"])

        # triples_residual += +1.00 P(i,j) <l,c||d,e> t2(d,a,j,k) t2(e,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbaikj->abcijk', tmps_["56_vvvooo"])

        # triples_residual += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,j) t2(e,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbkji->abcijk', tmps_["56_vvvooo"])
        del tmps_["56_vvvooo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[90_vvoo](b,a,j,i) = 1.00 t1(c,i) * eri[vovv](b,k,c,d) * t2(d,a,j,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
        tmps_["90_vvoo"] = einsum('ci,acbj->baji', t1, tmps_["18_vvvo"])

    if includes_["t2"]:
        del tmps_["18_vvvo"]


    if includes_["t1"] and includes_["t2"]:
        doubles_residual -= einsum('baij->abij', tmps_["90_vvoo"])

        # doubles_residual += -1.00 P(i,j) P(a,b) <k,a||c,d> t1(c,j) t2(d,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual += einsum('abij->abij', tmps_["90_vvoo"])
        doubles_residual += einsum('baji->abij', tmps_["90_vvoo"])
        doubles_residual -= einsum('abji->abij', tmps_["90_vvoo"])
        del tmps_["90_vvoo"]


    if includes_["t2"]:

        # tmps_[19_vvvo](a,e,b,i) = 1.00 eri[vovv](b,l,d,e) * t2(d,a,i,l) // flops: o1v3 = o2v4 | mem: o1v3 = o1v3
        tmps_["19_vvvo"] = einsum('blde,dail->aebi', eri["vovv"], t2)

        # tmps_[57_vvvooo](a,b,c,j,k,i) = 1.00 t2(e,c,i,k) * eri[vovv](a,l,d,e) * t2(d,b,j,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
        tmps_["57_vvvooo"] = einsum('ecik,beaj->abcjki', t2, tmps_["19_vvvo"])
        del tmps_["19_vvvo"]


        # triples_residual += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcikj->abcijk', tmps_["57_vvvooo"])
        triples_residual += einsum('abcjki->abcijk', tmps_["57_vvvooo"])

        # triples_residual += +1.00 P(j,k) P(a,b) <l,a||d,e> t2(d,b,k,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abckji->abcijk', tmps_["57_vvvooo"])
        triples_residual += einsum('bacikj->abcijk', tmps_["57_vvvooo"])
        triples_residual -= einsum('bacjki->abcijk', tmps_["57_vvvooo"])
        triples_residual += einsum('backji->abcijk', tmps_["57_vvvooo"])

        # triples_residual += +1.00 P(j,k) <l,c||d,e> t2(d,a,k,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cabkji->abcijk', tmps_["57_vvvooo"])

        # triples_residual += +1.00 <l,c||d,e> t2(d,a,i,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cabikj->abcijk', tmps_["57_vvvooo"])
        triples_residual += einsum('cabjki->abcijk', tmps_["57_vvvooo"])
        del tmps_["57_vvvooo"]


        # tmps_[20_vvoo](c,d,i,l) = 1.00 eri[oovv](m,l,d,e) * t2(e,c,i,m) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
        tmps_["20_vvoo"] = einsum('mlde,ecim->cdil', eri["oovv"], t2)

    if includes_["t1"] and includes_["t2"]:

        # singles_residual += +1.00 <k,j||b,c> t1(b,j) t2(c,a,i,k)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
        singles_residual += einsum('abij,bj->ai', tmps_["20_vvoo"], t1)

        # tmps_[69_vvvooo](a,b,c,i,k,j) = 1.00 t2(d,b,i,k) * eri[oovv](m,l,d,e) * t2(e,c,j,m) * t1(a,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
        tmps_["69_vvvooo"] = einsum('dbik,cdjl,al->abcikj', t2, tmps_["20_vvoo"], t1, optimize=['einsum_path',(0,1),(0,1)])
        triples_residual -= einsum('bacikj->abcijk', tmps_["69_vvvooo"])

        # triples_residual += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,j) t2(e,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cabijk->abcijk', tmps_["69_vvvooo"])
        triples_residual += einsum('bacijk->abcijk', tmps_["69_vvvooo"])

        # triples_residual += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,j) t2(e,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcijk->abcijk', tmps_["69_vvvooo"])
        triples_residual += einsum('cabikj->abcijk', tmps_["69_vvvooo"])
        triples_residual += einsum('bacjki->abcijk', tmps_["69_vvvooo"])

        # triples_residual += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,j,k) t2(e,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcjki->abcijk', tmps_["69_vvvooo"])

        # triples_residual += -1.00 P(i,j) <m,l||d,e> t1(c,l) t2(d,a,j,k) t2(e,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cabjki->abcijk', tmps_["69_vvvooo"])
        triples_residual += einsum('abcikj->abcijk', tmps_["69_vvvooo"])
        del tmps_["69_vvvooo"]


    if includes_["t2"]:

        # tmps_[89_vvoo](b,a,j,i) = 1.00 t2(c,a,i,k) * eri[oovv](l,k,c,d) * t2(d,b,j,l) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
        tmps_["89_vvoo"] = einsum('caik,bcjk->baji', t2, tmps_["20_vvoo"])
        doubles_residual -= einsum('baji->abij', tmps_["89_vvoo"])

        # doubles_residual += +1.00 P(i,j) <l,k||c,d> t2(c,a,j,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual += einsum('baij->abij', tmps_["89_vvoo"])
        del tmps_["89_vvoo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[94_vvoo](a,b,j,i) = 1.00 t1(c,i) * eri[oovv](l,k,c,d) * t2(d,a,j,l) * t1(b,k) // flops: o2v2 = o3v2 o3v2 | mem: o2v2 = o3v1 o2v2
        tmps_["94_vvoo"] = einsum('ci,acjk,bk->abji', t1, tmps_["20_vvoo"], t1, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t2"]:
        del tmps_["20_vvoo"]


    if includes_["t1"] and includes_["t2"]:
        doubles_residual += einsum('abji->abij', tmps_["94_vvoo"])

        # doubles_residual += +1.00 P(i,j) P(a,b) <l,k||c,d> t1(c,j) t1(a,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual += einsum('baij->abij', tmps_["94_vvoo"])
        doubles_residual -= einsum('abij->abij', tmps_["94_vvoo"])
        doubles_residual -= einsum('baji->abij', tmps_["94_vvoo"])
        del tmps_["94_vvoo"]


    if includes_["t2"]:

        # tmps_[21_vooo](b,j,i,l) = 0.50 eri[vovv](b,l,d,e) * t2(d,e,i,j) // flops: o3v1 = o3v3 | mem: o3v1 = o3v1
        tmps_["21_vooo"] = 0.50 * einsum('blde,deij->bjil', eri["vovv"], t2)

        # tmps_[76_vvvooo](c,b,a,i,j,k) = 1.00 t2(a,b,k,l) * eri[vovv](c,l,d,e) * t2(d,e,i,j) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["76_vvvooo"] = einsum('abkl,cjil->cbaijk', t2, tmps_["21_vooo"])
        triples_residual -= einsum('acbikj->abcijk', tmps_["76_vvvooo"])
        triples_residual += einsum('bcaikj->abcijk', tmps_["76_vvvooo"])

        # triples_residual += -0.50 P(i,j) <l,c||d,e> t2(d,e,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbajki->abcijk', tmps_["76_vvvooo"])

        # triples_residual += -0.50 P(i,j) P(a,b) <l,a||d,e> t2(d,e,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbjki->abcijk', tmps_["76_vvvooo"])

        # triples_residual += -0.50 P(a,b) <l,a||d,e> t2(d,e,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbijk->abcijk', tmps_["76_vvvooo"])
        triples_residual -= einsum('bcaijk->abcijk', tmps_["76_vvvooo"])
        triples_residual -= einsum('bcajki->abcijk', tmps_["76_vvvooo"])
        triples_residual -= einsum('cbaikj->abcijk', tmps_["76_vvvooo"])

        # triples_residual += -0.50 <l,c||d,e> t2(d,e,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaijk->abcijk', tmps_["76_vvvooo"])
        del tmps_["76_vvvooo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[105_vvoo](a,b,i,j) = 1.00 eri[vovv](b,k,c,d) * t2(c,d,i,j) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
        tmps_["105_vvoo"] = einsum('bjik,ak->abij', tmps_["21_vooo"], t1)

    if includes_["t2"]:
        del tmps_["21_vooo"]


    if includes_["t1"] and includes_["t2"]:
        doubles_residual += einsum('abij->abij', tmps_["105_vvoo"])

        # doubles_residual += +0.50 P(a,b) <k,a||c,d> t1(b,k) t2(c,d,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('baij->abij', tmps_["105_vvoo"])
        del tmps_["105_vvoo"]


    if includes_["t2"]:

        # tmps_[22_vvvo](c,a,d,i) = 0.50 eri[oovo](m,l,d,i) * t2(a,c,m,l) // flops: o1v3 = o3v3 | mem: o1v3 = o1v3
        tmps_["22_vvvo"] = 0.50 * einsum('mldi,acml->cadi', eri["oovo"], t2)

        # tmps_[58_vvvooo](a,c,b,k,j,i) = 1.00 t2(d,b,i,j) * eri[oovo](m,l,d,k) * t2(a,c,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
        tmps_["58_vvvooo"] = einsum('dbij,cadk->acbkji', t2, tmps_["22_vvvo"])
        triples_residual += einsum('acbkji->abcijk', tmps_["58_vvvooo"])
        triples_residual -= einsum('acbjki->abcijk', tmps_["58_vvvooo"])

        # triples_residual += -0.50 <m,l||d,i> t2(d,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcikj->abcijk', tmps_["58_vvvooo"])
        triples_residual += einsum('acbikj->abcijk', tmps_["58_vvvooo"])
        triples_residual += einsum('abcjki->abcijk', tmps_["58_vvvooo"])

        # triples_residual += -0.50 P(j,k) P(a,b) <m,l||d,k> t2(d,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bcakji->abcijk', tmps_["58_vvvooo"])

        # triples_residual += -0.50 P(a,b) <m,l||d,i> t2(d,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bcaikj->abcijk', tmps_["58_vvvooo"])
        triples_residual += einsum('bcajki->abcijk', tmps_["58_vvvooo"])

        # triples_residual += -0.50 P(j,k) <m,l||d,k> t2(d,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abckji->abcijk', tmps_["58_vvvooo"])
        del tmps_["58_vvvooo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[92_vvoo](a,b,j,i) = 1.00 t1(c,i) * eri[oovo](l,k,c,j) * t2(a,b,l,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
        tmps_["92_vvoo"] = einsum('ci,bacj->abji', t1, tmps_["22_vvvo"])

    if includes_["t2"]:
        del tmps_["22_vvvo"]


    if includes_["t1"] and includes_["t2"]:

        # doubles_residual += +0.50 P(i,j) <l,k||c,j> t1(c,i) t2(a,b,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual += einsum('abji->abij', tmps_["92_vvoo"])
        doubles_residual -= einsum('abij->abij', tmps_["92_vvoo"])
        del tmps_["92_vvoo"]


    if includes_["t2"]:

        # tmps_[23_vvoo](b,e,k,l) = 1.00 eri[oovv](m,l,d,e) * t2(d,b,k,m) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
        tmps_["23_vvoo"] = einsum('mlde,dbkm->bekl', eri["oovv"], t2)

    if includes_["t1"] and includes_["t2"]:

        # tmps_[70_vvvooo](b,c,a,j,k,i) = 1.00 t2(e,c,j,k) * eri[oovv](m,l,d,e) * t2(d,a,i,m) * t1(b,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
        tmps_["70_vvvooo"] = einsum('ecjk,aeil,bl->bcajki', t2, tmps_["23_vvoo"], t1, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t2"]:
        del tmps_["23_vvoo"]


    if includes_["t1"] and includes_["t2"]:

        # triples_residual += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,k,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbijk->abcijk', tmps_["70_vvvooo"])
        triples_residual -= einsum('bcaikj->abcijk', tmps_["70_vvvooo"])

        # triples_residual += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbjki->abcijk', tmps_["70_vvvooo"])
        triples_residual += einsum('bcaijk->abcijk', tmps_["70_vvvooo"])

        # triples_residual += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,m) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbajki->abcijk', tmps_["70_vvvooo"])
        triples_residual += einsum('bcajki->abcijk', tmps_["70_vvvooo"])
        triples_residual += einsum('cbaikj->abcijk', tmps_["70_vvvooo"])
        triples_residual += einsum('acbikj->abcijk', tmps_["70_vvvooo"])

        # triples_residual += -1.00 P(j,k) <m,l||d,e> t1(c,l) t2(d,a,k,m) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbaijk->abcijk', tmps_["70_vvvooo"])
        del tmps_["70_vvvooo"]


    if includes_["t2"]:

        # tmps_[24_vvoo](a,e,i,m) = 1.00 eri[oovv](m,l,d,e) * t2(d,a,i,l) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
        tmps_["24_vvoo"] = einsum('mlde,dail->aeim', eri["oovv"], t2)

    if includes_["t2"] and includes_["t3"]:

        # tmps_[48_vvvooo](a,c,b,i,k,j) = 1.00 t3(e,b,c,j,k,m) * eri[oovv](m,l,d,e) * t2(d,a,i,l) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
        tmps_["48_vvvooo"] = einsum('ebcjkm,aeim->acbikj', t3, tmps_["24_vvoo"])

    if includes_["t2"]:
        del tmps_["24_vvoo"]


    if includes_["t2"] and includes_["t3"]:

        # triples_residual += +1.00 P(j,k) P(a,b) <m,l||d,e> t2(d,a,k,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbkji->abcijk', tmps_["48_vvvooo"])
        triples_residual -= einsum('bcaikj->abcijk', tmps_["48_vvvooo"])
        triples_residual += einsum('bcajki->abcijk', tmps_["48_vvvooo"])

        # triples_residual += +1.00 P(a,b) <m,l||d,e> t2(d,a,i,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbikj->abcijk', tmps_["48_vvvooo"])
        triples_residual -= einsum('cbajki->abcijk', tmps_["48_vvvooo"])
        triples_residual -= einsum('bcakji->abcijk', tmps_["48_vvvooo"])

        # triples_residual += +1.00 P(j,k) <m,l||d,e> t2(d,c,k,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbakji->abcijk', tmps_["48_vvvooo"])

        # triples_residual += +1.00 <m,l||d,e> t2(d,c,i,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaikj->abcijk', tmps_["48_vvvooo"])
        triples_residual -= einsum('acbjki->abcijk', tmps_["48_vvvooo"])
        del tmps_["48_vvvooo"]


    if includes_["t2"]:

        # tmps_[25_vvoo](a,b,i,j) = 1.00 eri[vovo](b,k,c,j) * t2(c,a,i,k) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
        tmps_["25_vvoo"] = einsum('bkcj,caik->abij', eri["vovo"], t2)

        # doubles_residual += +1.00 P(i,j) P(a,b) <k,a||c,j> t2(c,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('baij->abij', tmps_["25_vvoo"])
        doubles_residual -= einsum('abji->abij', tmps_["25_vvoo"])
        doubles_residual += einsum('baji->abij', tmps_["25_vvoo"])
        doubles_residual += einsum('abij->abij', tmps_["25_vvoo"])
        del tmps_["25_vvoo"]


        # tmps_[26_vooo](b,j,k,m) = 1.00 eri[oovo](m,l,d,k) * t2(d,b,j,l) // flops: o3v1 = o4v2 | mem: o3v1 = o3v1
        tmps_["26_vooo"] = einsum('mldk,dbjl->bjkm', eri["oovo"], t2)

        # tmps_[67_vvvooo](c,b,a,i,k,j) = 1.00 t2(a,b,j,m) * eri[oovo](m,l,d,i) * t2(d,c,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["67_vvvooo"] = einsum('abjm,ckim->cbaikj', t2, tmps_["26_vooo"])
        del tmps_["26_vooo"]

        triples_residual -= einsum('acbkij->abcijk', tmps_["67_vvvooo"])

        # triples_residual += -1.00 P(i,k) <m,l||d,j> t2(d,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbajki->abcijk', tmps_["67_vvvooo"])
        triples_residual += einsum('bcaijk->abcijk', tmps_["67_vvvooo"])
        triples_residual += einsum('cbajik->abcijk', tmps_["67_vvvooo"])

        # triples_residual += +1.00 P(j,k) <m,l||d,i> t2(d,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaikj->abcijk', tmps_["67_vvvooo"])

        # triples_residual += +1.00 P(i,j) P(a,b) <m,l||d,k> t2(d,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbkji->abcijk', tmps_["67_vvvooo"])
        triples_residual -= einsum('cbaijk->abcijk', tmps_["67_vvvooo"])

        # triples_residual += -1.00 P(i,k) P(a,b) <m,l||d,j> t2(d,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbjki->abcijk', tmps_["67_vvvooo"])
        triples_residual -= einsum('cbakij->abcijk', tmps_["67_vvvooo"])

        # triples_residual += +1.00 P(j,k) P(a,b) <m,l||d,i> t2(d,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbikj->abcijk', tmps_["67_vvvooo"])

        # triples_residual += +1.00 P(i,j) <m,l||d,k> t2(d,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbakji->abcijk', tmps_["67_vvvooo"])
        triples_residual -= einsum('bcakji->abcijk', tmps_["67_vvvooo"])
        triples_residual -= einsum('bcaikj->abcijk', tmps_["67_vvvooo"])
        triples_residual -= einsum('bcajik->abcijk', tmps_["67_vvvooo"])
        triples_residual += einsum('acbjik->abcijk', tmps_["67_vvvooo"])
        triples_residual += einsum('bcakij->abcijk', tmps_["67_vvvooo"])
        triples_residual += einsum('bcajki->abcijk', tmps_["67_vvvooo"])
        triples_residual -= einsum('acbijk->abcijk', tmps_["67_vvvooo"])
        del tmps_["67_vvvooo"]


        # tmps_[27_oooo](k,i,l,m) = 0.50 eri[oovv](m,l,d,e) * t2(d,e,i,k) // flops: o4v0 = o4v2 | mem: o4v0 = o4v0
        tmps_["27_oooo"] = 0.50 * einsum('mlde,deik->kilm', eri["oovv"], t2)

    if includes_["t1"] and includes_["t2"]:

        # doubles_residual += -0.50 <l,k||c,d> t1(a,k) t1(b,l) t2(c,d,i,j)  // flops: o2v2 += o4v1 o3v2 | mem: o2v2 += o3v1 o2v2
        doubles_residual -= einsum('jikl,ak,bl->abij', tmps_["27_oooo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t2"]:

        # doubles_residual += +0.25 <l,k||c,d> t2(c,d,i,j) t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
        doubles_residual += 0.50 * einsum('jikl,ablk->abij', tmps_["27_oooo"], t2)

    if includes_["t2"] and includes_["t3"]:

        # tmps_[54_vvvooo](c,b,a,j,k,i) = 0.50 t3(a,b,c,i,m,l) * eri[oovv](m,l,d,e) * t2(d,e,j,k) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
        tmps_["54_vvvooo"] = 0.50 * einsum('abciml,kjlm->cbajki', t3, tmps_["27_oooo"])

        # triples_residual += +0.25 P(i,j) <m,l||d,e> t2(d,e,j,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbajki->abcijk', tmps_["54_vvvooo"])
        triples_residual -= einsum('cbaikj->abcijk', tmps_["54_vvvooo"])

        # triples_residual += +0.25 <m,l||d,e> t2(d,e,i,j) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaijk->abcijk', tmps_["54_vvvooo"])
        del tmps_["54_vvvooo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[74_vvvooo](c,a,b,k,j,i) = 1.00 eri[oovv](m,l,d,e) * t2(d,e,j,k) * t1(c,l) * t2(a,b,i,m) // flops: o3v3 = o4v1 o4v3 | mem: o3v3 = o3v1 o3v3
        tmps_["74_vvvooo"] = einsum('kjlm,cl,abim->cabkji', tmps_["27_oooo"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t2"]:
        del tmps_["27_oooo"]


    if includes_["t1"] and includes_["t2"]:

        # triples_residual += +0.50 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,e,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abckji->abcijk', tmps_["74_vvvooo"])

        # triples_residual += +0.50 P(i,j) <m,l||d,e> t1(c,l) t2(d,e,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cabkji->abcijk', tmps_["74_vvvooo"])
        triples_residual -= einsum('bacjik->abcijk', tmps_["74_vvvooo"])
        triples_residual -= einsum('abckij->abcijk', tmps_["74_vvvooo"])
        triples_residual -= einsum('backji->abcijk', tmps_["74_vvvooo"])

        # triples_residual += +0.50 <m,l||d,e> t1(c,l) t2(d,e,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cabjik->abcijk', tmps_["74_vvvooo"])

        # triples_residual += +0.50 P(a,b) <m,l||d,e> t1(a,l) t2(d,e,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcjik->abcijk', tmps_["74_vvvooo"])
        triples_residual += einsum('backij->abcijk', tmps_["74_vvvooo"])
        triples_residual -= einsum('cabkij->abcijk', tmps_["74_vvvooo"])
        del tmps_["74_vvvooo"]


    if includes_["t2"]:

        # tmps_[28_vooo](a,j,i,k) = 1.00 eri[oovo](l,k,c,i) * t2(c,a,j,l) // flops: o3v1 = o4v2 | mem: o3v1 = o3v1
        tmps_["28_vooo"] = einsum('lkci,cajl->ajik', eri["oovo"], t2)

    if includes_["t1"] and includes_["t2"]:

        # tmps_[95_vvoo](a,b,i,j) = 1.00 t1(b,k) * eri[oovo](l,k,c,i) * t2(c,a,j,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
        tmps_["95_vvoo"] = einsum('bk,ajik->abij', t1, tmps_["28_vooo"])

    if includes_["t2"]:
        del tmps_["28_vooo"]


    if includes_["t1"] and includes_["t2"]:

        # doubles_residual += -1.00 P(i,j) P(a,b) <l,k||c,j> t1(a,k) t2(c,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('baji->abij', tmps_["95_vvoo"])
        doubles_residual += einsum('baij->abij', tmps_["95_vvoo"])
        doubles_residual += einsum('abji->abij', tmps_["95_vvoo"])
        doubles_residual -= einsum('abij->abij', tmps_["95_vvoo"])
        del tmps_["95_vvoo"]


    if includes_["t1"]:

        # tmps_[29_vvvo](e,c,b,i) = 1.00 eri[vvvv](b,c,d,e) * t1(d,i) // flops: o1v3 = o1v4 | mem: o1v3 = o1v3
        tmps_["29_vvvo"] = einsum('bcde,di->ecbi', eri["vvvv"], t1)

        # doubles_residual += -1.00 <a,b||c,d> t1(c,j) t1(d,i)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
        doubles_residual -= einsum('dbaj,di->abij', tmps_["29_vvvo"], t1)

    if includes_["t1"] and includes_["t2"]:

        # tmps_[59_vvvooo](b,c,a,k,j,i) = 1.00 t2(e,a,i,j) * eri[vvvv](b,c,d,e) * t1(d,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
        tmps_["59_vvvooo"] = einsum('eaij,ecbk->bcakji', t2, tmps_["29_vvvo"])

    if includes_["t1"]:
        del tmps_["29_vvvo"]


    if includes_["t1"] and includes_["t2"]:

        # triples_residual += +1.00 P(j,k) <b,c||d,e> t1(d,k) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bcakji->abcijk', tmps_["59_vvvooo"])

        # triples_residual += +1.00 <b,c||d,e> t1(d,i) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bcaikj->abcijk', tmps_["59_vvvooo"])

        # triples_residual += +1.00 P(j,k) P(b,c) <a,b||d,e> t1(d,k) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abckji->abcijk', tmps_["59_vvvooo"])
        triples_residual -= einsum('acbikj->abcijk', tmps_["59_vvvooo"])
        triples_residual += einsum('acbjki->abcijk', tmps_["59_vvvooo"])
        triples_residual -= einsum('acbkji->abcijk', tmps_["59_vvvooo"])
        triples_residual -= einsum('bcajki->abcijk', tmps_["59_vvvooo"])
        triples_residual -= einsum('abcjki->abcijk', tmps_["59_vvvooo"])

        # triples_residual += +1.00 P(b,c) <a,b||d,e> t1(d,i) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcikj->abcijk', tmps_["59_vvvooo"])
        del tmps_["59_vvvooo"]


    if includes_["t1"]:

        # tmps_[30_vvoo](e,c,i,l) = 1.00 eri[vovv](c,l,d,e) * t1(d,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
        tmps_["30_vvoo"] = einsum('clde,di->ecil', eri["vovv"], t1)

    if includes_["t1"] and includes_["t3"]:

        # tmps_[49_vvvooo](b,c,a,i,k,j) = 1.00 t3(e,a,c,j,k,l) * eri[vovv](b,l,d,e) * t1(d,i) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
        tmps_["49_vvvooo"] = einsum('eacjkl,ebil->bcaikj', t3, tmps_["30_vvoo"])

        # triples_residual += -1.00 P(a,b) <l,a||d,e> t1(d,i) t3(e,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbikj->abcijk', tmps_["49_vvvooo"])

        # triples_residual += -1.00 P(j,k) P(a,b) <l,a||d,e> t1(d,k) t3(e,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbkji->abcijk', tmps_["49_vvvooo"])
        triples_residual -= einsum('bcakji->abcijk', tmps_["49_vvvooo"])

        # triples_residual += -1.00 <l,c||d,e> t1(d,i) t3(e,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaikj->abcijk', tmps_["49_vvvooo"])
        triples_residual -= einsum('bcaikj->abcijk', tmps_["49_vvvooo"])
        triples_residual += einsum('bcajki->abcijk', tmps_["49_vvvooo"])
        triples_residual -= einsum('acbjki->abcijk', tmps_["49_vvvooo"])

        # triples_residual += -1.00 P(j,k) <l,c||d,e> t1(d,k) t3(e,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbakji->abcijk', tmps_["49_vvvooo"])
        triples_residual -= einsum('cbajki->abcijk', tmps_["49_vvvooo"])
        del tmps_["49_vvvooo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[63_vvvooo](a,c,b,i,j,k) = 1.00 t2(e,c,i,j) * eri[vovv](b,l,d,e) * t1(d,k) * t1(a,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
        tmps_["63_vvvooo"] = einsum('ecij,ebkl,al->acbijk', t2, tmps_["30_vvoo"], t1, optimize=['einsum_path',(0,1),(0,1)])
        triples_residual += einsum('bacjki->abcijk', tmps_["63_vvvooo"])
        triples_residual -= einsum('acbikj->abcijk', tmps_["63_vvvooo"])

        # triples_residual += +1.00 P(a,b) <l,c||d,e> t1(d,i) t1(a,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcjki->abcijk', tmps_["63_vvvooo"])
        triples_residual += einsum('abcikj->abcijk', tmps_["63_vvvooo"])
        triples_residual -= einsum('bacikj->abcijk', tmps_["63_vvvooo"])

        # triples_residual += +1.00 P(b,c) <l,a||d,e> t1(d,i) t1(b,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bcajki->abcijk', tmps_["63_vvvooo"])
        triples_residual += einsum('cabikj->abcijk', tmps_["63_vvvooo"])

        # triples_residual += -1.00 P(a,c) <l,b||d,e> t1(d,i) t1(a,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbjki->abcijk', tmps_["63_vvvooo"])
        triples_residual -= einsum('cabjki->abcijk', tmps_["63_vvvooo"])

        # triples_residual += +1.00 P(j,k) P(a,b) <l,c||d,e> t1(d,k) t1(a,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcijk->abcijk', tmps_["63_vvvooo"])
        triples_residual += einsum('bcaikj->abcijk', tmps_["63_vvvooo"])

        # triples_residual += -1.00 P(j,k) P(a,c) <l,b||d,e> t1(d,k) t1(a,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbijk->abcijk', tmps_["63_vvvooo"])
        triples_residual += einsum('cbaijk->abcijk', tmps_["63_vvvooo"])
        triples_residual -= einsum('cbaikj->abcijk', tmps_["63_vvvooo"])
        triples_residual -= einsum('cabijk->abcijk', tmps_["63_vvvooo"])
        triples_residual += einsum('cbajki->abcijk', tmps_["63_vvvooo"])

        # triples_residual += +1.00 P(j,k) P(b,c) <l,a||d,e> t1(d,k) t1(b,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bcaijk->abcijk', tmps_["63_vvvooo"])
        triples_residual += einsum('bacijk->abcijk', tmps_["63_vvvooo"])
        del tmps_["63_vvvooo"]


        # tmps_[72_vvvooo](c,a,b,k,i,j) = 1.00 t1(e,i) * eri[vovv](b,l,d,e) * t1(d,j) * t2(a,c,k,l) // flops: o3v3 = o3v2 o4v3 | mem: o3v3 = o3v1 o3v3
        tmps_["72_vvvooo"] = einsum('ei,ebjl,ackl->cabkij', t1, tmps_["30_vvoo"], t2, optimize=['einsum_path',(0,1),(0,1)])
        triples_residual -= einsum('cabjik->abcijk', tmps_["72_vvvooo"])
        triples_residual += einsum('cabkij->abcijk', tmps_["72_vvvooo"])
        triples_residual += einsum('cabijk->abcijk', tmps_["72_vvvooo"])

        # triples_residual += +1.00 <l,c||d,e> t1(d,j) t1(e,i) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('backij->abcijk', tmps_["72_vvvooo"])

        # triples_residual += +1.00 P(i,j) P(a,b) <l,a||d,e> t1(d,k) t1(e,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbaijk->abcijk', tmps_["72_vvvooo"])

        # triples_residual += +1.00 P(a,b) <l,a||d,e> t1(d,j) t1(e,i) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbakij->abcijk', tmps_["72_vvvooo"])

        # triples_residual += +1.00 P(i,j) <l,c||d,e> t1(d,k) t1(e,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bacijk->abcijk', tmps_["72_vvvooo"])
        triples_residual += einsum('bacjik->abcijk', tmps_["72_vvvooo"])
        triples_residual += einsum('cbajik->abcijk', tmps_["72_vvvooo"])
        del tmps_["72_vvvooo"]


    if includes_["t1"]:

        # tmps_[98_vvoo](b,a,j,i) = 1.00 t1(d,i) * eri[vovv](b,k,c,d) * t1(c,j) * t1(a,k) // flops: o2v2 = o3v2 o3v2 | mem: o2v2 = o3v1 o2v2
        tmps_["98_vvoo"] = einsum('di,dbjk,ak->baji', t1, tmps_["30_vvoo"], t1, optimize=['einsum_path',(0,1),(0,1)])
        del tmps_["30_vvoo"]


        # doubles_residual += -1.00 P(a,b) <k,a||c,d> t1(c,j) t1(d,i) t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual += einsum('abji->abij', tmps_["98_vvoo"])
        doubles_residual -= einsum('baji->abij', tmps_["98_vvoo"])
        del tmps_["98_vvoo"]


    if includes_["t2"]:

        # tmps_[31_vv](a,d) = 0.50 eri[oovv](l,k,c,d) * t2(c,a,l,k) // flops: o0v2 = o2v3 | mem: o0v2 = o0v2
        tmps_["31_vv"] = 0.50 * einsum('lkcd,calk->ad', eri["oovv"], t2)

        # doubles_residual += -0.50 <l,k||c,d> t2(c,a,l,k) t2(d,b,i,j)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
        doubles_residual -= einsum('ad,dbij->abij', tmps_["31_vv"], t2)

    if includes_["t2"] and includes_["t3"]:

        # tmps_[61_vvvooo](a,c,b,k,j,i) = 1.00 t3(e,b,c,i,j,k) * eri[oovv](m,l,d,e) * t2(d,a,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
        tmps_["61_vvvooo"] = einsum('ebcijk,ae->acbkji', t3, tmps_["31_vv"])

    if includes_["t2"]:
        del tmps_["31_vv"]


    if includes_["t2"] and includes_["t3"]:

        # triples_residual += -0.50 P(a,b) <m,l||d,e> t2(d,a,m,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbkji->abcijk', tmps_["61_vvvooo"])
        triples_residual += einsum('bcakji->abcijk', tmps_["61_vvvooo"])

        # triples_residual += -0.50 <m,l||d,e> t2(d,c,m,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbakji->abcijk', tmps_["61_vvvooo"])
        del tmps_["61_vvvooo"]


    if includes_["t1"]:

        # tmps_[32_vvoo](b,a,i,j) = 1.00 eri[vvvo](a,b,c,j) * t1(c,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
        tmps_["32_vvoo"] = einsum('abcj,ci->baij', eri["vvvo"], t1)

        # doubles_residual += +1.00 P(i,j) <a,b||c,j> t1(c,i)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual += einsum('baij->abij', tmps_["32_vvoo"])
        doubles_residual -= einsum('baji->abij', tmps_["32_vvoo"])
        del tmps_["32_vvoo"]


    if includes_["t2"]:

        # tmps_[33_vvoo](a,b,j,i) = 1.00 f[vv](b,c) * t2(c,a,i,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
        tmps_["33_vvoo"] = einsum('bc,caij->abji', f["vv"], t2)
        doubles_residual -= einsum('abji->abij', tmps_["33_vvoo"])

        # doubles_residual += +1.00 P(a,b) f(a,c) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual += einsum('baji->abij', tmps_["33_vvoo"])
        del tmps_["33_vvoo"]


    if includes_["t1"]:

        # tmps_[34_vooo](d,i,k,l) = 1.00 eri[oovv](l,k,c,d) * t1(c,i) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
        tmps_["34_vooo"] = einsum('lkcd,ci->dikl', eri["oovv"], t1)

    if includes_["t1"] and includes_["t2"]:

        # singles_residual += +0.50 <k,j||b,c> t1(b,i) t2(c,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
        singles_residual += 0.50 * einsum('cijk,cakj->ai', tmps_["34_vooo"], t2)

        # doubles_residual += -0.50 <l,k||c,d> t1(c,j) t1(d,i) t2(a,b,l,k)  // flops: o2v2 += o4v1 o4v2 | mem: o2v2 += o4v0 o2v2
        doubles_residual -= 0.50 * einsum('djkl,di,ablk->abij', tmps_["34_vooo"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t1"] and includes_["t3"]:

        # tmps_[50_vvvooo](b,a,c,i,j,k) = 1.00 t3(e,a,c,i,j,m) * eri[oovv](m,l,d,e) * t1(d,k) * t1(b,l) // flops: o3v3 = o5v3 o4v3 | mem: o3v3 = o4v2 o3v3
        tmps_["50_vvvooo"] = einsum('eacijm,eklm,bl->bacijk', t3, tmps_["34_vooo"], t1, optimize=['einsum_path',(0,1),(0,1)])

        # triples_residual += +1.00 P(a,b) <m,l||d,e> t1(d,i) t1(a,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcjki->abcijk', tmps_["50_vvvooo"])
        triples_residual -= einsum('bacjki->abcijk', tmps_["50_vvvooo"])
        triples_residual -= einsum('bacijk->abcijk', tmps_["50_vvvooo"])
        triples_residual -= einsum('abcikj->abcijk', tmps_["50_vvvooo"])
        triples_residual += einsum('bacikj->abcijk', tmps_["50_vvvooo"])

        # triples_residual += +1.00 P(j,k) <m,l||d,e> t1(d,k) t1(c,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cabijk->abcijk', tmps_["50_vvvooo"])

        # triples_residual += +1.00 <m,l||d,e> t1(d,i) t1(c,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cabjki->abcijk', tmps_["50_vvvooo"])
        triples_residual -= einsum('cabikj->abcijk', tmps_["50_vvvooo"])

        # triples_residual += +1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t1(a,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcijk->abcijk', tmps_["50_vvvooo"])
        del tmps_["50_vvvooo"]


        # tmps_[52_vvvooo](c,b,a,i,k,j) = 0.50 eri[oovv](m,l,d,e) * t1(d,k) * t1(e,j) * t3(a,b,c,i,m,l) // flops: o3v3 = o4v1 o5v3 | mem: o3v3 = o4v0 o3v3
        tmps_["52_vvvooo"] = 0.50 * einsum('eklm,ej,abciml->cbaikj', tmps_["34_vooo"], t1, t3, optimize=['einsum_path',(0,1),(0,1)])

        # triples_residual += -0.50 <m,l||d,e> t1(d,j) t1(e,i) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbakji->abcijk', tmps_["52_vvvooo"])

        # triples_residual += -0.50 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbaikj->abcijk', tmps_["52_vvvooo"])
        triples_residual += einsum('cbajki->abcijk', tmps_["52_vvvooo"])
        del tmps_["52_vvvooo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[55_vvvooo](b,a,c,k,j,i) = 0.50 t2(a,c,m,l) * eri[oovv](m,l,d,e) * t1(d,i) * t2(e,b,j,k) // flops: o3v3 = o3v3 o3v4 | mem: o3v3 = o1v3 o3v3
        tmps_["55_vvvooo"] = 0.50 * einsum('acml,eilm,ebjk->backji', t2, tmps_["34_vooo"], t2, optimize=['einsum_path',(0,1),(0,1)])
        triples_residual -= einsum('cabkij->abcijk', tmps_["55_vvvooo"])

        # triples_residual += +0.50 <m,l||d,e> t1(d,i) t2(e,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cabkji->abcijk', tmps_["55_vvvooo"])
        triples_residual -= einsum('abckij->abcijk', tmps_["55_vvvooo"])

        # triples_residual += +0.50 P(a,b) <m,l||d,e> t1(d,i) t2(e,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abckji->abcijk', tmps_["55_vvvooo"])
        triples_residual -= einsum('bacjik->abcijk', tmps_["55_vvvooo"])

        # triples_residual += +0.50 P(j,k) <m,l||d,e> t1(d,k) t2(e,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cabjik->abcijk', tmps_["55_vvvooo"])
        triples_residual -= einsum('backji->abcijk', tmps_["55_vvvooo"])

        # triples_residual += +0.50 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcjik->abcijk', tmps_["55_vvvooo"])
        triples_residual += einsum('backij->abcijk', tmps_["55_vvvooo"])
        del tmps_["55_vvvooo"]


        # tmps_[64_vvvooo](c,a,b,k,j,i) = 1.00 t2(e,b,j,l) * eri[oovv](m,l,d,e) * t1(d,i) * t2(a,c,k,m) // flops: o3v3 = o4v2 o4v3 | mem: o3v3 = o3v1 o3v3
        tmps_["64_vvvooo"] = einsum('ebjl,eilm,ackm->cabkji', t2, tmps_["34_vooo"], t2, optimize=['einsum_path',(0,1),(0,1)])
        triples_residual += einsum('cabkij->abcijk', tmps_["64_vvvooo"])
        triples_residual += einsum('cabjki->abcijk', tmps_["64_vvvooo"])
        triples_residual -= einsum('cabkji->abcijk', tmps_["64_vvvooo"])

        # triples_residual += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,i) t2(e,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbajki->abcijk', tmps_["64_vvvooo"])

        # triples_residual += -1.00 P(j,k) <m,l||d,e> t1(d,i) t2(e,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bacjki->abcijk', tmps_["64_vvvooo"])
        triples_residual -= einsum('cbakij->abcijk', tmps_["64_vvvooo"])
        triples_residual += einsum('bacjik->abcijk', tmps_["64_vvvooo"])
        triples_residual += einsum('cbajik->abcijk', tmps_["64_vvvooo"])
        triples_residual += einsum('cbakji->abcijk', tmps_["64_vvvooo"])

        # triples_residual += +1.00 P(i,k) <m,l||d,e> t1(d,j) t2(e,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bacikj->abcijk', tmps_["64_vvvooo"])
        triples_residual -= einsum('cabjik->abcijk', tmps_["64_vvvooo"])

        # triples_residual += +1.00 P(i,k) P(a,b) <m,l||d,e> t1(d,j) t2(e,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaikj->abcijk', tmps_["64_vvvooo"])

        # triples_residual += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbaijk->abcijk', tmps_["64_vvvooo"])
        triples_residual += einsum('backji->abcijk', tmps_["64_vvvooo"])
        triples_residual -= einsum('cabikj->abcijk', tmps_["64_vvvooo"])

        # triples_residual += -1.00 P(i,j) <m,l||d,e> t1(d,k) t2(e,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('bacijk->abcijk', tmps_["64_vvvooo"])
        triples_residual -= einsum('backij->abcijk', tmps_["64_vvvooo"])
        triples_residual += einsum('cabijk->abcijk', tmps_["64_vvvooo"])
        del tmps_["64_vvvooo"]


        # tmps_[88_vvoooo](a,c,i,m,j,k) = 1.00 eri[oovv](m,l,d,e) * t1(d,i) * t2(e,c,j,k) * t1(a,l) // flops: o4v2 = o5v2 o5v2 | mem: o4v2 = o5v1 o4v2
        tmps_["88_vvoooo"] = einsum('eilm,ecjk,al->acimjk', tmps_["34_vooo"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

        # tmps_[109_vvvooo](b,a,c,k,j,i) = 1.00 t1(c,m) * tmps_[34_vooo](e,i,l,m) * t2(e,b,j,k) * t1(a,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["109_vvvooo"] = einsum('cm,abimjk->backji', t1, tmps_["88_vvoooo"])
        del tmps_["88_vvoooo"]

        triples_residual += einsum('cabkij->abcijk', tmps_["109_vvvooo"])

        # triples_residual += -1.00 P(j,k) P(b,c) <m,l||d,e> t1(d,k) t1(a,l) t1(b,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cabjik->abcijk', tmps_["109_vvvooo"])
        triples_residual += einsum('bacjik->abcijk', tmps_["109_vvvooo"])
        triples_residual += einsum('abckij->abcijk', tmps_["109_vvvooo"])

        # triples_residual += -1.00 P(j,k) <m,l||d,e> t1(d,k) t1(b,l) t1(c,m) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcjik->abcijk', tmps_["109_vvvooo"])

        # triples_residual += -1.00 P(b,c) <m,l||d,e> t1(d,i) t1(a,l) t1(b,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cabkji->abcijk', tmps_["109_vvvooo"])
        triples_residual += einsum('backji->abcijk', tmps_["109_vvvooo"])

        # triples_residual += -1.00 <m,l||d,e> t1(d,i) t1(b,l) t1(c,m) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abckji->abcijk', tmps_["109_vvvooo"])
        triples_residual -= einsum('backij->abcijk', tmps_["109_vvvooo"])
        del tmps_["109_vvvooo"]


    if includes_["t1"]:

        # tmps_[107_vooo](c,m,j,i) = 1.00 t1(e,i) * eri[oovv](m,l,d,e) * t1(d,j) * t1(c,l) // flops: o3v1 = o4v1 o4v1 | mem: o3v1 = o4v0 o3v1
        tmps_["107_vooo"] = einsum('ei,ejlm,cl->cmji', t1, tmps_["34_vooo"], t1, optimize=['einsum_path',(0,1),(0,1)])
        del tmps_["34_vooo"]


        # doubles_residual += +1.00 <l,k||c,d> t1(c,j) t1(d,i) t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
        doubles_residual += einsum('alji,bl->abij', tmps_["107_vooo"], t1)

    if includes_["t1"] and includes_["t2"]:

        # tmps_[108_vvvooo](b,c,a,j,k,i) = 1.00 t2(a,c,i,m) * t1(e,j) * t1(b,l) * eri[oovv](m,l,d,e) * t1(d,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["108_vvvooo"] = einsum('acim,bmkj->bcajki', t2, tmps_["107_vooo"])

    if includes_["t1"]:
        del tmps_["107_vooo"]


    if includes_["t1"] and includes_["t2"]:

        # triples_residual += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t1(e,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbjki->abcijk', tmps_["108_vvvooo"])
        triples_residual += einsum('bcaijk->abcijk', tmps_["108_vvvooo"])
        triples_residual += einsum('cbaikj->abcijk', tmps_["108_vvvooo"])
        triples_residual += einsum('bcajki->abcijk', tmps_["108_vvvooo"])

        # triples_residual += -1.00 P(a,b) <m,l||d,e> t1(d,j) t1(e,i) t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbijk->abcijk', tmps_["108_vvvooo"])
        triples_residual += einsum('acbikj->abcijk', tmps_["108_vvvooo"])

        # triples_residual += -1.00 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbajki->abcijk', tmps_["108_vvvooo"])

        # triples_residual += -1.00 <m,l||d,e> t1(d,j) t1(e,i) t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbaijk->abcijk', tmps_["108_vvvooo"])
        triples_residual -= einsum('bcaikj->abcijk', tmps_["108_vvvooo"])
        del tmps_["108_vvvooo"]


    if includes_["t1"]:

        # tmps_[35_vooo](c,k,i,l) = 1.00 eri[vovo](c,l,d,i) * t1(d,k) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
        tmps_["35_vooo"] = einsum('cldi,dk->ckil', eri["vovo"], t1)

    if includes_["t1"] and includes_["t2"]:

        # tmps_[68_vvvooo](c,b,a,k,i,j) = 1.00 t2(a,b,j,l) * eri[vovo](c,l,d,k) * t1(d,i) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["68_vvvooo"] = einsum('abjl,cikl->cbakij', t2, tmps_["35_vooo"])
        triples_residual -= einsum('bcaikj->abcijk', tmps_["68_vvvooo"])
        triples_residual -= einsum('cbaijk->abcijk', tmps_["68_vvvooo"])
        triples_residual += einsum('bcaijk->abcijk', tmps_["68_vvvooo"])
        triples_residual -= einsum('acbijk->abcijk', tmps_["68_vvvooo"])

        # triples_residual += +1.00 P(i,k) <l,c||d,j> t1(d,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbajki->abcijk', tmps_["68_vvvooo"])

        # triples_residual += -1.00 P(i,j) <l,c||d,k> t1(d,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbakji->abcijk', tmps_["68_vvvooo"])
        triples_residual -= einsum('bcakji->abcijk', tmps_["68_vvvooo"])

        # triples_residual += -1.00 P(j,k) P(a,b) <l,a||d,i> t1(d,k) t2(b,c,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbikj->abcijk', tmps_["68_vvvooo"])

        # triples_residual += -1.00 P(j,k) <l,c||d,i> t1(d,k) t2(a,b,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaikj->abcijk', tmps_["68_vvvooo"])
        triples_residual -= einsum('cbakij->abcijk', tmps_["68_vvvooo"])

        # triples_residual += -1.00 P(i,j) P(a,b) <l,a||d,k> t1(d,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbkji->abcijk', tmps_["68_vvvooo"])
        triples_residual += einsum('acbjik->abcijk', tmps_["68_vvvooo"])
        triples_residual += einsum('bcajki->abcijk', tmps_["68_vvvooo"])

        # triples_residual += +1.00 P(i,k) P(a,b) <l,a||d,j> t1(d,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbjki->abcijk', tmps_["68_vvvooo"])
        triples_residual -= einsum('bcajik->abcijk', tmps_["68_vvvooo"])
        triples_residual += einsum('bcakij->abcijk', tmps_["68_vvvooo"])
        triples_residual += einsum('cbajik->abcijk', tmps_["68_vvvooo"])
        triples_residual -= einsum('acbkij->abcijk', tmps_["68_vvvooo"])
        del tmps_["68_vvvooo"]


    if includes_["t1"]:

        # tmps_[96_vvoo](b,a,i,j) = 1.00 t1(a,k) * eri[vovo](b,k,c,i) * t1(c,j) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
        tmps_["96_vvoo"] = einsum('ak,bjik->baij', t1, tmps_["35_vooo"])
        del tmps_["35_vooo"]

        doubles_residual -= einsum('baij->abij', tmps_["96_vvoo"])

        # doubles_residual += +1.00 P(i,j) P(a,b) <k,a||c,j> t1(c,i) t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('abji->abij', tmps_["96_vvoo"])
        doubles_residual += einsum('abij->abij', tmps_["96_vvoo"])
        doubles_residual += einsum('baji->abij', tmps_["96_vvoo"])
        del tmps_["96_vvoo"]


        # tmps_[36_vooo](c,i,k,l) = 1.00 eri[oovv](l,k,c,d) * t1(d,i) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
        tmps_["36_vooo"] = einsum('lkcd,di->cikl', eri["oovv"], t1)

        # singles_residual += +1.00 <k,j||b,c> t1(b,j) t1(c,i) t1(a,k)  // flops: o1v1 += o3v1 o2v1 | mem: o1v1 += o2v0 o1v1
        singles_residual += einsum('bijk,bj,ak->ai', tmps_["36_vooo"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t1"] and includes_["t3"]:

        # tmps_[83_vvvooo](c,b,a,k,j,i) = 1.00 t1(d,l) * eri[oovv](m,l,d,e) * t1(e,i) * t3(a,b,c,j,k,m) // flops: o3v3 = o3v1 o4v3 | mem: o3v3 = o2v0 o3v3
        tmps_["83_vvvooo"] = einsum('dl,dilm,abcjkm->cbakji', t1, tmps_["36_vooo"], t3, optimize=['einsum_path',(0,1),(0,1)])

        # triples_residual += +1.00 P(j,k) <m,l||d,e> t1(d,l) t1(e,k) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbajik->abcijk', tmps_["83_vvvooo"])
        triples_residual -= einsum('cbakij->abcijk', tmps_["83_vvvooo"])

        # triples_residual += +1.00 <m,l||d,e> t1(d,l) t1(e,i) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbakji->abcijk', tmps_["83_vvvooo"])
        del tmps_["83_vvvooo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[100_vvoo](a,b,j,i) = 1.00 eri[oovv](l,k,c,d) * t1(d,i) * t1(c,k) * t2(a,b,j,l) // flops: o2v2 = o3v1 o3v2 | mem: o2v2 = o2v0 o2v2
        tmps_["100_vvoo"] = einsum('cikl,ck,abjl->abji', tmps_["36_vooo"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t1"]:
        del tmps_["36_vooo"]


    if includes_["t1"] and includes_["t2"]:

        # doubles_residual += +1.00 P(i,j) <l,k||c,d> t1(c,k) t1(d,j) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual += einsum('abij->abij', tmps_["100_vvoo"])
        doubles_residual -= einsum('abji->abij', tmps_["100_vvoo"])
        del tmps_["100_vvoo"]


    if includes_["t2"]:

        # tmps_[37_vooo](b,j,i,l) = 1.00 f[ov](l,d) * t2(d,b,i,j) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
        tmps_["37_vooo"] = einsum('ld,dbij->bjil', f["ov"], t2)

        # tmps_[79_vvvooo](c,b,a,j,k,i) = 1.00 t2(a,b,i,l) * f[ov](l,d) * t2(d,c,j,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["79_vvvooo"] = einsum('abil,ckjl->cbajki', t2, tmps_["37_vooo"])
        triples_residual -= einsum('bcaikj->abcijk', tmps_["79_vvvooo"])

        # triples_residual += -1.00 P(a,b) f(l,d) t2(d,a,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbijk->abcijk', tmps_["79_vvvooo"])
        triples_residual += einsum('bcaijk->abcijk', tmps_["79_vvvooo"])

        # triples_residual += -1.00 f(l,d) t2(d,c,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbaijk->abcijk', tmps_["79_vvvooo"])

        # triples_residual += -1.00 P(i,j) P(a,b) f(l,d) t2(d,a,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbjki->abcijk', tmps_["79_vvvooo"])
        triples_residual += einsum('cbaikj->abcijk', tmps_["79_vvvooo"])
        triples_residual += einsum('acbikj->abcijk', tmps_["79_vvvooo"])
        triples_residual += einsum('bcajki->abcijk', tmps_["79_vvvooo"])

        # triples_residual += -1.00 P(i,j) f(l,d) t2(d,c,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbajki->abcijk', tmps_["79_vvvooo"])
        del tmps_["79_vvvooo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[106_vvoo](b,a,i,j) = 1.00 t1(a,k) * f[ov](k,c) * t2(c,b,i,j) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
        tmps_["106_vvoo"] = einsum('ak,bjik->baij', t1, tmps_["37_vooo"])

    if includes_["t2"]:
        del tmps_["37_vooo"]


    if includes_["t1"] and includes_["t2"]:
        doubles_residual += einsum('abij->abij', tmps_["106_vvoo"])

        # doubles_residual += -1.00 P(a,b) f(k,c) t1(a,k) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('baij->abij', tmps_["106_vvoo"])
        del tmps_["106_vvoo"]


    if includes_["t2"]:

        # tmps_[38_oo](j,l) = 0.50 eri[oovv](l,k,c,d) * t2(c,d,j,k) // flops: o2v0 = o3v2 | mem: o2v0 = o2v0
        tmps_["38_oo"] = 0.50 * einsum('lkcd,cdjk->jl', eri["oovv"], t2)

    if includes_["t2"] and includes_["t3"]:

        # tmps_[85_vvvooo](c,b,a,i,k,j) = 1.00 t3(a,b,c,j,k,m) * eri[oovv](m,l,d,e) * t2(d,e,i,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["85_vvvooo"] = einsum('abcjkm,im->cbaikj', t3, tmps_["38_oo"])
        triples_residual += einsum('cbajki->abcijk', tmps_["85_vvvooo"])

        # triples_residual += -0.50 P(j,k) <m,l||d,e> t2(d,e,k,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbakji->abcijk', tmps_["85_vvvooo"])

        # triples_residual += -0.50 <m,l||d,e> t2(d,e,i,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbaikj->abcijk', tmps_["85_vvvooo"])
        del tmps_["85_vvvooo"]


    if includes_["t2"]:

        # tmps_[102_vvoo](b,a,j,i) = 1.00 eri[oovv](l,k,c,d) * t2(c,d,i,k) * t2(a,b,j,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
        tmps_["102_vvoo"] = einsum('il,abjl->baji', tmps_["38_oo"], t2)
        del tmps_["38_oo"]


        # doubles_residual += -0.50 P(i,j) <l,k||c,d> t2(c,d,j,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('baij->abij', tmps_["102_vvoo"])
        doubles_residual += einsum('baji->abij', tmps_["102_vvoo"])
        del tmps_["102_vvoo"]


        # tmps_[39_vv](b,c) = 1.00 eri[oovv](l,k,c,d) * t2(d,b,l,k) // flops: o0v2 = o2v3 | mem: o0v2 = o0v2
        tmps_["39_vv"] = einsum('lkcd,dblk->bc', eri["oovv"], t2)

        # doubles_residual += -0.50 <l,k||c,d> t2(c,a,i,j) t2(d,b,l,k)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
        doubles_residual -= 0.50 * einsum('bc,caij->abij', tmps_["39_vv"], t2)
        del tmps_["39_vv"]


    if includes_["t1"]:

        # tmps_[40_vvoo](b,a,j,i) = 1.00 eri[vooo](a,k,i,j) * t1(b,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
        tmps_["40_vvoo"] = einsum('akij,bk->baji', eri["vooo"], t1)
        doubles_residual += einsum('abji->abij', tmps_["40_vvoo"])

        # doubles_residual += +1.00 P(a,b) <k,a||i,j> t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('baji->abij', tmps_["40_vvoo"])
        del tmps_["40_vvoo"]


    if includes_["t2"]:

        # tmps_[41_vvoo](b,a,i,j) = 1.00 f[oo](k,j) * t2(a,b,i,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
        tmps_["41_vvoo"] = einsum('kj,abik->baij', f["oo"], t2)

        # doubles_residual += -1.00 P(i,j) f(k,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('baij->abij', tmps_["41_vvoo"])
        doubles_residual += einsum('baji->abij', tmps_["41_vvoo"])
        del tmps_["41_vvoo"]


    if includes_["t1"]:

        # tmps_[42_oooo](i,j,k,l) = 1.00 eri[oovo](l,k,c,j) * t1(c,i) // flops: o4v0 = o4v1 | mem: o4v0 = o4v0
        tmps_["42_oooo"] = einsum('lkcj,ci->ijkl', eri["oovo"], t1)

    if includes_["t1"] and includes_["t3"]:

        # tmps_[51_vvvooo](c,b,a,k,i,j) = 0.50 t3(a,b,c,j,m,l) * eri[oovo](m,l,d,k) * t1(d,i) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
        tmps_["51_vvvooo"] = 0.50 * einsum('abcjml,iklm->cbakij', t3, tmps_["42_oooo"])
        triples_residual -= einsum('cbakij->abcijk', tmps_["51_vvvooo"])

        # triples_residual += +0.50 P(i,j) <m,l||d,k> t1(d,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbakji->abcijk', tmps_["51_vvvooo"])
        triples_residual -= einsum('cbaijk->abcijk', tmps_["51_vvvooo"])

        # triples_residual += +0.50 P(j,k) <m,l||d,i> t1(d,k) t3(a,b,c,j,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaikj->abcijk', tmps_["51_vvvooo"])

        # triples_residual += -0.50 P(i,k) <m,l||d,j> t1(d,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbajki->abcijk', tmps_["51_vvvooo"])
        triples_residual += einsum('cbajik->abcijk', tmps_["51_vvvooo"])
        del tmps_["51_vvvooo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[65_vvvooo](c,a,b,i,k,j) = 1.00 eri[oovo](m,l,d,j) * t1(d,k) * t1(c,l) * t2(a,b,i,m) // flops: o3v3 = o4v1 o4v3 | mem: o3v3 = o3v1 o3v3
        tmps_["65_vvvooo"] = einsum('kjlm,cl,abim->cabikj', tmps_["42_oooo"], t1, t2, optimize=['einsum_path',(0,1),(0,1)])
        triples_residual -= einsum('bacjki->abcijk', tmps_["65_vvvooo"])
        triples_residual += einsum('bacjik->abcijk', tmps_["65_vvvooo"])

        # triples_residual += +1.00 P(j,k) <m,l||d,i> t1(d,k) t1(c,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cabjki->abcijk', tmps_["65_vvvooo"])
        triples_residual -= einsum('cabkji->abcijk', tmps_["65_vvvooo"])
        triples_residual += einsum('cabkij->abcijk', tmps_["65_vvvooo"])
        triples_residual -= einsum('abcjik->abcijk', tmps_["65_vvvooo"])

        # triples_residual += +1.00 P(j,k) P(a,b) <m,l||d,i> t1(d,k) t1(a,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcjki->abcijk', tmps_["65_vvvooo"])
        triples_residual -= einsum('cabjik->abcijk', tmps_["65_vvvooo"])

        # triples_residual += -1.00 P(i,k) <m,l||d,j> t1(d,k) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cabikj->abcijk', tmps_["65_vvvooo"])
        triples_residual -= einsum('bacijk->abcijk', tmps_["65_vvvooo"])
        triples_residual += einsum('backji->abcijk', tmps_["65_vvvooo"])
        triples_residual -= einsum('abckji->abcijk', tmps_["65_vvvooo"])

        # triples_residual += -1.00 P(i,k) P(a,b) <m,l||d,j> t1(d,k) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('abcikj->abcijk', tmps_["65_vvvooo"])
        triples_residual += einsum('abckij->abcijk', tmps_["65_vvvooo"])

        # triples_residual += +1.00 P(i,j) P(a,b) <m,l||d,k> t1(d,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcijk->abcijk', tmps_["65_vvvooo"])
        triples_residual += einsum('bacikj->abcijk', tmps_["65_vvvooo"])

        # triples_residual += +1.00 P(i,j) <m,l||d,k> t1(d,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cabijk->abcijk', tmps_["65_vvvooo"])
        triples_residual -= einsum('backij->abcijk', tmps_["65_vvvooo"])
        del tmps_["65_vvvooo"]


    if includes_["t1"]:

        # tmps_[99_vvoo](b,a,j,i) = 1.00 t1(b,l) * eri[oovo](l,k,c,j) * t1(c,i) * t1(a,k) // flops: o2v2 = o4v1 o3v2 | mem: o2v2 = o3v1 o2v2
        tmps_["99_vvoo"] = einsum('bl,ijkl,ak->baji', t1, tmps_["42_oooo"], t1, optimize=['einsum_path',(0,1),(0,1)])
        del tmps_["42_oooo"]


        # doubles_residual += -1.00 P(i,j) <l,k||c,j> t1(c,i) t1(a,k) t1(b,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('baji->abij', tmps_["99_vvoo"])
        doubles_residual += einsum('baij->abij', tmps_["99_vvoo"])
        del tmps_["99_vvoo"]


        # tmps_[43_vooo](c,j,i,m) = 1.00 eri[oooo](m,l,i,j) * t1(c,l) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
        tmps_["43_vooo"] = einsum('mlij,cl->cjim', eri["oooo"], t1)

        # doubles_residual += -1.00 <l,k||i,j> t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('ajil,bl->abij', tmps_["43_vooo"], t1)

    if includes_["t1"] and includes_["t2"]:

        # tmps_[78_vvvooo](b,c,a,i,j,k) = 1.00 t2(a,c,k,m) * eri[oooo](m,l,i,j) * t1(b,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["78_vvvooo"] = einsum('ackm,bjim->bcaijk', t2, tmps_["43_vooo"])

    if includes_["t1"]:
        del tmps_["43_vooo"]


    if includes_["t1"] and includes_["t2"]:

        # triples_residual += +1.00 P(i,j) P(a,b) <m,l||j,k> t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbjki->abcijk', tmps_["78_vvvooo"])
        triples_residual -= einsum('bcaijk->abcijk', tmps_["78_vvvooo"])
        triples_residual += einsum('bcaikj->abcijk', tmps_["78_vvvooo"])

        # triples_residual += +1.00 P(i,j) <m,l||j,k> t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbajki->abcijk', tmps_["78_vvvooo"])
        triples_residual -= einsum('bcajki->abcijk', tmps_["78_vvvooo"])

        # triples_residual += +1.00 P(a,b) <m,l||i,j> t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('acbijk->abcijk', tmps_["78_vvvooo"])
        triples_residual -= einsum('acbikj->abcijk', tmps_["78_vvvooo"])

        # triples_residual += +1.00 <m,l||i,j> t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaijk->abcijk', tmps_["78_vvvooo"])
        triples_residual -= einsum('cbaikj->abcijk', tmps_["78_vvvooo"])
        del tmps_["78_vvvooo"]


    if includes_["t1"]:

        # tmps_[44_vv](d,b) = 1.00 eri[vovv](b,k,c,d) * t1(c,k) // flops: o0v2 = o1v3 | mem: o0v2 = o0v2
        tmps_["44_vv"] = einsum('bkcd,ck->db', eri["vovv"], t1)

        # singles_residual += +1.00 <j,a||b,c> t1(b,j) t1(c,i)  // flops: o1v1 += o1v2 | mem: o1v1 += o1v1
        singles_residual -= einsum('ca,ci->ai', tmps_["44_vv"], t1)

    if includes_["t1"] and includes_["t3"]:

        # tmps_[62_vvvooo](b,c,a,k,j,i) = 1.00 t3(e,a,c,i,j,k) * eri[vovv](b,l,d,e) * t1(d,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
        tmps_["62_vvvooo"] = einsum('eacijk,eb->bcakji', t3, tmps_["44_vv"])

        # triples_residual += +1.00 P(a,b) <l,a||d,e> t1(d,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('acbkji->abcijk', tmps_["62_vvvooo"])
        triples_residual += einsum('bcakji->abcijk', tmps_["62_vvvooo"])

        # triples_residual += +1.00 <l,c||d,e> t1(d,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbakji->abcijk', tmps_["62_vvvooo"])
        del tmps_["62_vvvooo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[93_vvoo](a,b,j,i) = 1.00 t2(d,b,i,j) * eri[vovv](a,k,c,d) * t1(c,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
        tmps_["93_vvoo"] = einsum('dbij,da->abji', t2, tmps_["44_vv"])

    if includes_["t1"]:
        del tmps_["44_vv"]


    if includes_["t1"] and includes_["t2"]:
        doubles_residual += einsum('baji->abij', tmps_["93_vvoo"])

        # doubles_residual += +1.00 P(a,b) <k,a||c,d> t1(c,k) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('abji->abij', tmps_["93_vvoo"])
        del tmps_["93_vvoo"]


    if includes_["t1"]:

        # tmps_[45_vo](d,l) = 1.00 eri[oovv](l,k,c,d) * t1(c,k) // flops: o1v1 = o2v2 | mem: o1v1 = o1v1
        tmps_["45_vo"] = einsum('lkcd,ck->dl', eri["oovv"], t1)

    if includes_["t1"] and includes_["t3"]:

        # doubles_residual += -1.00 <l,k||c,d> t1(c,k) t3(d,a,b,i,j,l)  // flops: o2v2 += o3v3 | mem: o2v2 += o2v2
        doubles_residual -= einsum('dl,dabijl->abij', tmps_["45_vo"], t3)

    if includes_["t1"] and includes_["t2"]:

        # tmps_[73_vvvooo](c,a,b,j,i,k) = 1.00 t2(e,b,i,k) * eri[oovv](m,l,d,e) * t1(d,l) * t2(a,c,j,m) // flops: o3v3 = o3v2 o4v3 | mem: o3v3 = o3v1 o3v3
        tmps_["73_vvvooo"] = einsum('ebik,em,acjm->cabjik', t2, tmps_["45_vo"], t2, optimize=['einsum_path',(0,1),(0,1)])
        triples_residual -= einsum('cabkij->abcijk', tmps_["73_vvvooo"])

        # triples_residual += +1.00 P(a,b) <m,l||d,e> t1(d,l) t2(e,a,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbakij->abcijk', tmps_["73_vvvooo"])
        triples_residual -= einsum('bacjik->abcijk', tmps_["73_vvvooo"])
        triples_residual -= einsum('cbajik->abcijk', tmps_["73_vvvooo"])
        triples_residual += einsum('cabjik->abcijk', tmps_["73_vvvooo"])

        # triples_residual += +1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,l) t2(e,a,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaijk->abcijk', tmps_["73_vvvooo"])

        # triples_residual += +1.00 P(i,j) <m,l||d,e> t1(d,l) t2(e,c,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('bacijk->abcijk', tmps_["73_vvvooo"])

        # triples_residual += +1.00 <m,l||d,e> t1(d,l) t2(e,c,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('backij->abcijk', tmps_["73_vvvooo"])
        triples_residual -= einsum('cabijk->abcijk', tmps_["73_vvvooo"])
        del tmps_["73_vvvooo"]


    if includes_["t1"] and includes_["t3"]:

        # tmps_[81_vvvooo](b,a,c,i,j,k) = 1.00 t3(e,a,c,i,j,k) * eri[oovv](m,l,d,e) * t1(d,l) * t1(b,m) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
        tmps_["81_vvvooo"] = einsum('eacijk,em,bm->bacijk', t3, tmps_["45_vo"], t1, optimize=['einsum_path',(0,1),(0,1)])
        triples_residual -= einsum('bacijk->abcijk', tmps_["81_vvvooo"])

        # triples_residual += +1.00 <m,l||d,e> t1(d,l) t1(c,m) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cabijk->abcijk', tmps_["81_vvvooo"])

        # triples_residual += +1.00 P(a,b) <m,l||d,e> t1(d,l) t1(a,m) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('abcijk->abcijk', tmps_["81_vvvooo"])
        del tmps_["81_vvvooo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[97_vvoo](a,b,j,i) = 1.00 eri[oovv](l,k,c,d) * t1(c,k) * t2(d,a,i,j) * t1(b,l) // flops: o2v2 = o3v2 o3v2 | mem: o2v2 = o3v1 o2v2
        tmps_["97_vvoo"] = einsum('dl,daij,bl->abji', tmps_["45_vo"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

    if includes_["t1"]:
        del tmps_["45_vo"]


    if includes_["t1"] and includes_["t2"]:
        doubles_residual -= einsum('abji->abij', tmps_["97_vvoo"])

        # doubles_residual += +1.00 P(a,b) <l,k||c,d> t1(c,k) t1(a,l) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual += einsum('baji->abij', tmps_["97_vvoo"])
        del tmps_["97_vvoo"]


    if includes_["t1"]:

        # tmps_[46_oo](j,l) = 1.00 eri[oovo](l,k,c,j) * t1(c,k) // flops: o2v0 = o3v1 | mem: o2v0 = o2v0
        tmps_["46_oo"] = einsum('lkcj,ck->jl', eri["oovo"], t1)

        # singles_residual += +1.00 <k,j||b,i> t1(b,j) t1(a,k)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
        singles_residual += einsum('ik,ak->ai', tmps_["46_oo"], t1)
    if includes_["t1"] and includes_["t3"]:

        # tmps_[87_vvvooo](c,b,a,j,k,i) = 1.00 t3(a,b,c,i,k,m) * eri[oovo](m,l,d,j) * t1(d,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["87_vvvooo"] = einsum('abcikm,jm->cbajki', t3, tmps_["46_oo"])

        # triples_residual += +1.00 P(j,k) <m,l||d,k> t1(d,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbakji->abcijk', tmps_["87_vvvooo"])

        # triples_residual += +1.00 <m,l||d,i> t1(d,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual += einsum('cbaikj->abcijk', tmps_["87_vvvooo"])
        triples_residual -= einsum('cbajki->abcijk', tmps_["87_vvvooo"])
        del tmps_["87_vvvooo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[103_vvoo](b,a,j,i) = 1.00 t2(a,b,i,l) * eri[oovo](l,k,c,j) * t1(c,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
        tmps_["103_vvoo"] = einsum('abil,jl->baji', t2, tmps_["46_oo"])

    if includes_["t1"]:
        del tmps_["46_oo"]


    if includes_["t1"] and includes_["t2"]:
        doubles_residual -= einsum('baij->abij', tmps_["103_vvoo"])

        # doubles_residual += +1.00 P(i,j) <l,k||c,j> t1(c,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual += einsum('baji->abij', tmps_["103_vvoo"])
        del tmps_["103_vvoo"]


    if includes_["t1"]:

        # tmps_[47_oo](i,k) = 1.00 f[ov](k,c) * t1(c,i) // flops: o2v0 = o2v1 | mem: o2v0 = o2v0
        tmps_["47_oo"] = einsum('kc,ci->ik', f["ov"], t1)

        # singles_residual += -1.00 f(j,b) t1(b,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
        singles_residual -= einsum('ij,aj->ai', tmps_["47_oo"], t1)

    if includes_["t1"] and includes_["t3"]:

        # tmps_[84_vvvooo](c,b,a,k,j,i) = 1.00 t3(a,b,c,i,j,l) * f[ov](l,d) * t1(d,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
        tmps_["84_vvvooo"] = einsum('abcijl,kl->cbakji', t3, tmps_["47_oo"])

        # triples_residual += -1.00 f(l,d) t1(d,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbaikj->abcijk', tmps_["84_vvvooo"])

        # triples_residual += -1.00 P(j,k) f(l,d) t1(d,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
        triples_residual -= einsum('cbakji->abcijk', tmps_["84_vvvooo"])
        triples_residual += einsum('cbajki->abcijk', tmps_["84_vvvooo"])
        del tmps_["84_vvvooo"]


    if includes_["t1"] and includes_["t2"]:

        # tmps_[101_vvoo](b,a,i,j) = 1.00 t2(a,b,j,k) * f[ov](k,c) * t1(c,i) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
        tmps_["101_vvoo"] = einsum('abjk,ik->baij', t2, tmps_["47_oo"])

    if includes_["t1"]:
        del tmps_["47_oo"]


    if includes_["t1"] and includes_["t2"]:

        # doubles_residual += -1.00 P(i,j) f(k,c) t1(c,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
        doubles_residual -= einsum('baji->abij', tmps_["101_vvoo"])
        doubles_residual += einsum('baij->abij', tmps_["101_vvoo"])
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



