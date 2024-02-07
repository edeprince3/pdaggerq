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

    # doubles_resid = +1.00 f(k,c) t3(c,a,b,i,j,k)  // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    doubles_resid = einsum('kc,cabijk->abij', f["ov"], t3)

    # doubles_resid += +0.50 <a,b||c,d> t2(c,d,i,j)  // flops: o2v2 += o2v4 | mem: o2v2 += o2v2
    doubles_resid += 0.50 * einsum('abcd,cdij->abij', eri["vvvv"], t2)

    # doubles_resid += +0.50 <l,k||i,j> t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
    doubles_resid += 0.50 * einsum('lkij,ablk->abij', eri["oooo"], t2)

    # doubles_resid += +1.00 <a,b||i,j>  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', eri["vvoo"])

    # singles_resid = +1.00 f(a,i)  // flops: o1v1 = o1v1 | mem: o1v1 = o1v1
    singles_resid = 1.00 * einsum('ai->ai', f["vo"])

    # singles_resid += -1.00 f(j,b) t2(b,a,i,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid -= einsum('jb,baij->ai', f["ov"], t2)

    # singles_resid += -1.00 f(j,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid -= einsum('ji,aj->ai', f["oo"], t1)

    # singles_resid += +0.50 <k,j||b,c> t1(a,j) t2(b,c,i,k)  // flops: o1v1 += o3v2 o2v1 | mem: o1v1 += o2v0 o1v1
    singles_resid += 0.50 * einsum('kjbc,bcik,aj->ai', eri["oovv"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

    # singles_resid += +1.00 <j,a||b,i> t1(b,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid -= einsum('ajbi,bj->ai', eri["vovo"], t1)

    # singles_resid += +0.25 <k,j||b,c> t3(b,c,a,i,k,j)  // flops: o1v1 += o3v3 | mem: o1v1 += o1v1
    singles_resid += 0.25 * einsum('kjbc,bcaikj->ai', eri["oovv"], t3)

    # singles_resid += -0.50 <k,j||b,i> t2(b,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
    singles_resid -= 0.50 * einsum('kjbi,bakj->ai', eri["oovo"], t2)

    # singles_resid += +1.00 f(a,b) t1(b,i)  // flops: o1v1 += o1v2 | mem: o1v1 += o1v1
    singles_resid += einsum('ab,bi->ai', f["vv"], t1)

    # singles_resid += -0.50 <j,a||b,c> t2(b,c,i,j)  // flops: o1v1 += o2v3 | mem: o1v1 += o1v1
    singles_resid += 0.50 * einsum('ajbc,bcij->ai', eri["vovv"], t2)

    # tmps_[1_vvvooo](a,c,b,k,j,i) = 0.50 eri[vvvv](b,c,d,e) * t3(d,e,a,i,j,k) // flops: o3v3 = o3v5 | mem: o3v3 = o3v3
    tmps_["1_vvvooo"] = 0.50 * einsum('bcde,deaijk->acbkji', eri["vvvv"], t3)

    # triples_resid = +0.50 <b,c||d,e> t3(d,e,a,i,j,k)  // flops: o3v3 = o3v3 | mem: o3v3 = o3v3
    triples_resid = 1.00 * einsum('acbkji->abcijk', tmps_["1_vvvooo"])
    triples_resid -= einsum('bcakji->abcijk', tmps_["1_vvvooo"])

    # triples_resid += +0.50 P(b,c) <a,b||d,e> t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakji->abcijk', tmps_["1_vvvooo"])
    del tmps_["1_vvvooo"]

    # tmps_[2_vvvooo](b,a,c,j,i,k) = 1.00 eri[vovo](c,l,d,k) * t3(d,a,b,i,j,l) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
    tmps_["2_vvvooo"] = einsum('cldk,dabijl->bacjik', eri["vovo"], t3)
    triples_resid -= einsum('cabkij->abcijk', tmps_["2_vvvooo"])
    triples_resid += einsum('cabjik->abcijk', tmps_["2_vvvooo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <l,a||d,k> t3(d,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbajik->abcijk', tmps_["2_vvvooo"])

    # triples_resid += +1.00 P(a,b) <l,a||d,i> t3(d,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakji->abcijk', tmps_["2_vvvooo"])

    # triples_resid += +1.00 <l,c||d,i> t3(d,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('backji->abcijk', tmps_["2_vvvooo"])
    triples_resid += einsum('backij->abcijk', tmps_["2_vvvooo"])
    triples_resid += einsum('cbakij->abcijk', tmps_["2_vvvooo"])
    triples_resid += einsum('cabkji->abcijk', tmps_["2_vvvooo"])

    # triples_resid += +1.00 P(j,k) <l,c||d,k> t3(d,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bacjik->abcijk', tmps_["2_vvvooo"])
    del tmps_["2_vvvooo"]

    # tmps_[3_vvoooo](c,a,k,j,i,l) = 0.50 eri[vovv](a,l,d,e) * t3(d,e,c,i,j,k) // flops: o4v2 = o4v4 | mem: o4v2 = o4v2
    tmps_["3_vvoooo"] = 0.50 * einsum('alde,decijk->cakjil', eri["vovv"], t3)

    # tmps_[73_vvvooo](b,a,c,i,j,k) = 1.00 eri[vovv](a,l,d,e) * t3(d,e,c,i,j,k) * t1(b,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["73_vvvooo"] = einsum('cakjil,bl->bacijk', tmps_["3_vvoooo"], t1)
    del tmps_["3_vvoooo"]

    # triples_resid += +0.50 P(b,c) <l,a||d,e> t1(b,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bacijk->abcijk', tmps_["73_vvvooo"])
    triples_resid += einsum('cabijk->abcijk', tmps_["73_vvvooo"])
    triples_resid -= einsum('cbaijk->abcijk', tmps_["73_vvvooo"])

    # triples_resid += +0.50 P(a,b) <l,c||d,e> t1(a,l) t3(d,e,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbijk->abcijk', tmps_["73_vvvooo"])
    triples_resid += einsum('bcaijk->abcijk', tmps_["73_vvvooo"])

    # triples_resid += -0.50 P(a,c) <l,b||d,e> t1(a,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["73_vvvooo"])
    del tmps_["73_vvvooo"]

    # tmps_[4_vvoooo](b,a,k,j,i,l) = 1.00 eri[oovo](m,l,d,i) * t3(d,a,b,j,k,m) // flops: o4v2 = o5v3 | mem: o4v2 = o4v2
    tmps_["4_vvoooo"] = einsum('mldi,dabjkm->bakjil', eri["oovo"], t3)

    # tmps_[69_vvvooo](b,a,c,i,j,k) = 1.00 eri[oovo](m,l,d,i) * t3(d,a,c,j,k,m) * t1(b,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["69_vvvooo"] = einsum('cakjil,bl->bacijk', tmps_["4_vvoooo"], t1)
    del tmps_["4_vvoooo"]
    triples_resid += einsum('bacijk->abcijk', tmps_["69_vvvooo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,k> t1(a,l) t3(d,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abckij->abcijk', tmps_["69_vvvooo"])
    triples_resid += einsum('cabjik->abcijk', tmps_["69_vvvooo"])
    triples_resid += einsum('backij->abcijk', tmps_["69_vvvooo"])
    triples_resid += einsum('abcjik->abcijk', tmps_["69_vvvooo"])

    # triples_resid += -1.00 <m,l||d,i> t1(c,l) t3(d,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabijk->abcijk', tmps_["69_vvvooo"])

    # triples_resid += -1.00 P(a,b) <m,l||d,i> t1(a,l) t3(d,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcijk->abcijk', tmps_["69_vvvooo"])

    # triples_resid += -1.00 P(j,k) <m,l||d,k> t1(c,l) t3(d,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabkij->abcijk', tmps_["69_vvvooo"])
    triples_resid -= einsum('bacjik->abcijk', tmps_["69_vvvooo"])
    del tmps_["69_vvvooo"]

    # tmps_[5_vooooo](a,k,j,i,l,m) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,a,i,j,k) // flops: o5v1 = o5v3 | mem: o5v1 = o5v1
    tmps_["5_vooooo"] = 0.50 * einsum('mlde,deaijk->akjilm', eri["oovv"], t3)

    # tmps_[53_vvvooo](c,a,b,i,j,k) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,b,i,j,k) * t2(a,c,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["53_vvvooo"] = 0.50 * einsum('bkjilm,acml->cabijk', tmps_["5_vooooo"], t2)

    # triples_resid += +0.25 <m,l||d,e> t2(b,c,m,l) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaijk->abcijk', tmps_["53_vvvooo"])

    # triples_resid += +0.25 P(b,c) <m,l||d,e> t2(a,b,m,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bacijk->abcijk', tmps_["53_vvvooo"])
    triples_resid -= einsum('cabijk->abcijk', tmps_["53_vvvooo"])
    del tmps_["53_vvvooo"]

    # tmps_[81_vvoooo](a,c,l,i,j,k) = 1.00 t1(c,m) * eri[oovv](m,l,d,e) * t3(d,e,a,i,j,k) // flops: o4v2 = o5v2 | mem: o4v2 = o4v2
    tmps_["81_vvoooo"] = einsum('cm,akjilm->aclijk', t1, tmps_["5_vooooo"])
    del tmps_["5_vooooo"]

    # tmps_[117_vvvooo](b,c,a,k,j,i) = 1.00 t1(c,m) * eri[oovv](m,l,d,e) * t3(d,e,a,i,j,k) * t1(b,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["117_vvvooo"] = einsum('aclijk,bl->bcakji', tmps_["81_vvoooo"], t1)
    del tmps_["81_vvoooo"]

    # triples_resid += -0.50 <m,l||d,e> t1(b,l) t1(c,m) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcakji->abcijk', tmps_["117_vvvooo"])
    triples_resid += einsum('acbkji->abcijk', tmps_["117_vvvooo"])

    # triples_resid += -0.50 P(b,c) <m,l||d,e> t1(a,l) t1(b,m) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abckji->abcijk', tmps_["117_vvvooo"])
    del tmps_["117_vvvooo"]

    # tmps_[6_vvvooo](c,b,a,j,k,i) = 0.50 eri[oooo](m,l,i,k) * t3(a,b,c,j,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["6_vvvooo"] = 0.50 * einsum('mlik,abcjml->cbajki', eri["oooo"], t3)
    triples_resid -= einsum('cbajki->abcijk', tmps_["6_vvvooo"])

    # triples_resid += +0.50 P(i,j) <m,l||j,k> t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaikj->abcijk', tmps_["6_vvvooo"])

    # triples_resid += +0.50 <m,l||i,j> t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakji->abcijk', tmps_["6_vvvooo"])
    del tmps_["6_vvvooo"]

    # tmps_[7_vvvo](c,a,d,i) = 0.50 eri[oovv](m,l,d,e) * t3(e,a,c,i,m,l) // flops: o1v3 = o3v4 | mem: o1v3 = o1v3
    tmps_["7_vvvo"] = 0.50 * einsum('mlde,eaciml->cadi', eri["oovv"], t3)

    # tmps_[57_vvvooo](a,b,c,k,i,j) = 1.00 eri[oovv](m,l,d,e) * t3(e,b,c,j,m,l) * t2(d,a,i,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["57_vvvooo"] = einsum('cbdj,daik->abckij', tmps_["7_vvvo"], t2)
    triples_resid += einsum('backji->abcijk', tmps_["57_vvvooo"])
    triples_resid += einsum('bacjik->abcijk', tmps_["57_vvvooo"])
    triples_resid += einsum('abckij->abcijk', tmps_["57_vvvooo"])

    # triples_resid += -0.50 P(a,b) <m,l||d,e> t2(d,a,i,j) t3(e,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcjik->abcijk', tmps_["57_vvvooo"])

    # triples_resid += -0.50 P(i,j) P(a,b) <m,l||d,e> t2(d,a,j,k) t3(e,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abckji->abcijk', tmps_["57_vvvooo"])
    triples_resid -= einsum('backij->abcijk', tmps_["57_vvvooo"])
    triples_resid += einsum('cabkij->abcijk', tmps_["57_vvvooo"])

    # triples_resid += -0.50 <m,l||d,e> t2(d,c,i,j) t3(e,a,b,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabjik->abcijk', tmps_["57_vvvooo"])

    # triples_resid += -0.50 P(i,j) <m,l||d,e> t2(d,c,j,k) t3(e,a,b,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabkji->abcijk', tmps_["57_vvvooo"])
    del tmps_["57_vvvooo"]

    # tmps_[86_vvoo](a,b,i,j) = 1.00 t1(c,j) * eri[oovv](l,k,c,d) * t3(d,a,b,i,l,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["86_vvoo"] = einsum('cj,baci->abij', t1, tmps_["7_vvvo"])
    del tmps_["7_vvvo"]

    # doubles_resid += -0.50 P(i,j) <l,k||c,d> t1(c,j) t3(d,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["86_vvoo"])
    doubles_resid += einsum('abji->abij', tmps_["86_vvoo"])
    del tmps_["86_vvoo"]

    # tmps_[8_vvvooo](c,b,a,k,i,j) = 1.00 eri[vvvo](a,b,d,j) * t2(d,c,i,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["8_vvvooo"] = einsum('abdj,dcik->cbakij', eri["vvvo"], t2)

    # triples_resid += -1.00 P(j,k) P(b,c) <a,b||d,k> t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbajik->abcijk', tmps_["8_vvvooo"])

    # triples_resid += -1.00 P(b,c) <a,b||d,i> t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakji->abcijk', tmps_["8_vvvooo"])

    # triples_resid += -1.00 <b,c||d,i> t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbkji->abcijk', tmps_["8_vvvooo"])
    triples_resid += einsum('bcakji->abcijk', tmps_["8_vvvooo"])
    triples_resid += einsum('acbkij->abcijk', tmps_["8_vvvooo"])
    triples_resid += einsum('cbakij->abcijk', tmps_["8_vvvooo"])

    # triples_resid += -1.00 P(j,k) <b,c||d,k> t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbjik->abcijk', tmps_["8_vvvooo"])
    triples_resid -= einsum('bcakij->abcijk', tmps_["8_vvvooo"])
    triples_resid += einsum('bcajik->abcijk', tmps_["8_vvvooo"])
    del tmps_["8_vvvooo"]

    # tmps_[9_vvvooo](c,a,b,k,j,i) = 1.00 f[vv](b,d) * t3(d,a,c,i,j,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["9_vvvooo"] = einsum('bd,dacijk->cabkji', f["vv"], t3)
    triples_resid -= einsum('cabkji->abcijk', tmps_["9_vvvooo"])

    # triples_resid += +1.00 f(c,d) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('backji->abcijk', tmps_["9_vvvooo"])

    # triples_resid += +1.00 P(a,b) f(a,d) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakji->abcijk', tmps_["9_vvvooo"])
    del tmps_["9_vvvooo"]

    # tmps_[10_vvoo](b,a,j,i) = 0.50 eri[vovv](a,k,c,d) * t3(c,d,b,i,j,k) // flops: o2v2 = o3v4 | mem: o2v2 = o2v2
    tmps_["10_vvoo"] = 0.50 * einsum('akcd,cdbijk->baji', eri["vovv"], t3)

    # doubles_resid += +0.50 P(a,b) <k,a||c,d> t3(c,d,b,i,j,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('baji->abij', tmps_["10_vvoo"])
    doubles_resid += einsum('abji->abij', tmps_["10_vvoo"])
    del tmps_["10_vvoo"]

    # tmps_[11_vvoooo](c,a,j,i,k,l) = 1.00 eri[vovo](a,l,d,k) * t2(d,c,i,j) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
    tmps_["11_vvoooo"] = einsum('aldk,dcij->cajikl', eri["vovo"], t2)

    # tmps_[63_vvvooo](b,a,c,j,i,k) = 1.00 t1(c,l) * eri[vovo](b,l,d,j) * t2(d,a,i,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["63_vvvooo"] = einsum('cl,abkijl->bacjik', t1, tmps_["11_vvoooo"])
    del tmps_["11_vvoooo"]

    # triples_resid += -1.00 P(b,c) <l,a||d,i> t1(b,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbijk->abcijk', tmps_["63_vvvooo"])
    triples_resid -= einsum('acbjik->abcijk', tmps_["63_vvvooo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,c||d,k> t1(a,l) t2(d,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakij->abcijk', tmps_["63_vvvooo"])
    triples_resid -= einsum('bacjik->abcijk', tmps_["63_vvvooo"])
    triples_resid += einsum('backij->abcijk', tmps_["63_vvvooo"])
    triples_resid += einsum('bcajik->abcijk', tmps_["63_vvvooo"])

    # triples_resid += +1.00 P(a,c) <l,b||d,i> t1(a,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcaijk->abcijk', tmps_["63_vvvooo"])

    # triples_resid += +1.00 P(j,k) P(a,c) <l,b||d,k> t1(a,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcakij->abcijk', tmps_["63_vvvooo"])

    # triples_resid += -1.00 P(j,k) P(b,c) <l,a||d,k> t1(b,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbkij->abcijk', tmps_["63_vvvooo"])
    triples_resid -= einsum('cabijk->abcijk', tmps_["63_vvvooo"])
    triples_resid += einsum('cabjik->abcijk', tmps_["63_vvvooo"])

    # triples_resid += -1.00 P(a,b) <l,c||d,i> t1(a,l) t2(d,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaijk->abcijk', tmps_["63_vvvooo"])
    triples_resid -= einsum('abcijk->abcijk', tmps_["63_vvvooo"])
    triples_resid += einsum('abcjik->abcijk', tmps_["63_vvvooo"])
    triples_resid -= einsum('abckij->abcijk', tmps_["63_vvvooo"])
    triples_resid += einsum('bacijk->abcijk', tmps_["63_vvvooo"])
    triples_resid -= einsum('cabkij->abcijk', tmps_["63_vvvooo"])
    triples_resid -= einsum('cbajik->abcijk', tmps_["63_vvvooo"])
    del tmps_["63_vvvooo"]

    # tmps_[12_vooo](a,k,i,l) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,a,i,k,m) // flops: o3v1 = o4v3 | mem: o3v1 = o3v1
    tmps_["12_vooo"] = 0.50 * einsum('mlde,deaikm->akil', eri["oovv"], t3)

    # tmps_[68_vvvooo](c,b,a,j,i,k) = 1.00 eri[oovv](m,l,d,e) * t3(d,e,a,i,k,m) * t2(b,c,j,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["68_vvvooo"] = einsum('akil,bcjl->cbajik', tmps_["12_vooo"], t2)
    triples_resid += einsum('cbajik->abcijk', tmps_["68_vvvooo"])

    # triples_resid += -0.50 <m,l||d,e> t2(b,c,i,l) t3(d,e,a,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbaijk->abcijk', tmps_["68_vvvooo"])
    triples_resid += einsum('bacjik->abcijk', tmps_["68_vvvooo"])

    # triples_resid += -0.50 P(b,c) <m,l||d,e> t2(a,b,i,l) t3(d,e,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bacijk->abcijk', tmps_["68_vvvooo"])
    triples_resid += einsum('cabijk->abcijk', tmps_["68_vvvooo"])

    # triples_resid += -0.50 P(j,k) P(b,c) <m,l||d,e> t2(a,b,k,l) t3(d,e,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('backij->abcijk', tmps_["68_vvvooo"])
    triples_resid += einsum('cabkij->abcijk', tmps_["68_vvvooo"])
    triples_resid -= einsum('cabjik->abcijk', tmps_["68_vvvooo"])

    # triples_resid += -0.50 P(j,k) <m,l||d,e> t2(b,c,k,l) t3(d,e,a,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakij->abcijk', tmps_["68_vvvooo"])
    del tmps_["68_vvvooo"]

    # tmps_[94_vvoo](b,a,i,j) = 1.00 t1(a,k) * eri[oovv](l,k,c,d) * t3(c,d,b,i,j,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["94_vvoo"] = einsum('ak,bjik->baij', t1, tmps_["12_vooo"])
    del tmps_["12_vooo"]

    # doubles_resid += -0.50 P(a,b) <l,k||c,d> t1(a,k) t3(c,d,b,i,j,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('baij->abij', tmps_["94_vvoo"])
    doubles_resid += einsum('abij->abij', tmps_["94_vvoo"])
    del tmps_["94_vvoo"]

    # tmps_[13_vvvooo](b,a,c,i,k,j) = 1.00 eri[vooo](c,l,j,k) * t2(a,b,i,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["13_vvvooo"] = einsum('cljk,abil->bacikj', eri["vooo"], t2)
    triples_resid -= einsum('cabkji->abcijk', tmps_["13_vvvooo"])
    triples_resid -= einsum('cabikj->abcijk', tmps_["13_vvvooo"])
    triples_resid -= einsum('cbajki->abcijk', tmps_["13_vvvooo"])
    triples_resid += einsum('cabjki->abcijk', tmps_["13_vvvooo"])
    triples_resid -= einsum('bacjki->abcijk', tmps_["13_vvvooo"])

    # triples_resid += -1.00 <l,c||i,j> t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('backji->abcijk', tmps_["13_vvvooo"])

    # triples_resid += -1.00 P(a,b) <l,a||i,j> t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakji->abcijk', tmps_["13_vvvooo"])

    # triples_resid += -1.00 P(i,j) P(a,b) <l,a||j,k> t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaikj->abcijk', tmps_["13_vvvooo"])

    # triples_resid += -1.00 P(i,j) <l,c||j,k> t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bacikj->abcijk', tmps_["13_vvvooo"])
    del tmps_["13_vvvooo"]

    # tmps_[14_vvoooo](b,a,k,j,i,l) = 1.00 f[ov](l,d) * t3(d,a,b,i,j,k) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
    tmps_["14_vvoooo"] = einsum('ld,dabijk->bakjil', f["ov"], t3)

    # tmps_[76_vvvooo](a,b,c,i,j,k) = 1.00 t1(c,l) * f[ov](l,d) * t3(d,a,b,i,j,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["76_vvvooo"] = einsum('cl,bakjil->abcijk', t1, tmps_["14_vvoooo"])
    del tmps_["14_vvoooo"]

    # triples_resid += -1.00 P(a,b) f(l,d) t1(a,l) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcaijk->abcijk', tmps_["76_vvvooo"])

    # triples_resid += -1.00 f(l,d) t1(c,l) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcijk->abcijk', tmps_["76_vvvooo"])
    triples_resid += einsum('acbijk->abcijk', tmps_["76_vvvooo"])
    del tmps_["76_vvvooo"]

    # tmps_[15_vvvooo](c,b,a,k,i,j) = 1.00 f[oo](l,j) * t3(a,b,c,i,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["15_vvvooo"] = einsum('lj,abcikl->cbakij', f["oo"], t3)
    triples_resid += einsum('cbakij->abcijk', tmps_["15_vvvooo"])

    # triples_resid += -1.00 f(l,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakji->abcijk', tmps_["15_vvvooo"])

    # triples_resid += -1.00 P(j,k) f(l,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbajik->abcijk', tmps_["15_vvvooo"])
    del tmps_["15_vvvooo"]

    # tmps_[16_vvoo](b,a,i,j) = 0.50 eri[oovo](l,k,c,j) * t3(c,a,b,i,l,k) // flops: o2v2 = o4v3 | mem: o2v2 = o2v2
    tmps_["16_vvoo"] = 0.50 * einsum('lkcj,cabilk->baij', eri["oovo"], t3)

    # doubles_resid += +0.50 P(i,j) <l,k||c,j> t3(c,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('baij->abij', tmps_["16_vvoo"])
    doubles_resid -= einsum('baji->abij', tmps_["16_vvoo"])
    del tmps_["16_vvoo"]

    # tmps_[17_vooooo](a,k,j,i,l,m) = 1.00 eri[oovo](m,l,d,i) * t2(d,a,j,k) // flops: o5v1 = o5v2 | mem: o5v1 = o5v1
    tmps_["17_vooooo"] = einsum('mldi,dajk->akjilm', eri["oovo"], t2)

    # tmps_[80_vvoooo](a,b,m,i,j,k) = 1.00 eri[oovo](m,l,d,i) * t2(d,b,j,k) * t1(a,l) // flops: o4v2 = o5v2 | mem: o4v2 = o4v2
    tmps_["80_vvoooo"] = einsum('bkjilm,al->abmijk', tmps_["17_vooooo"], t1)
    del tmps_["17_vooooo"]

    # tmps_[112_vvvooo](b,a,c,k,j,i) = 1.00 t1(c,m) * eri[oovo](m,l,d,i) * t2(d,b,j,k) * t1(a,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["112_vvvooo"] = einsum('cm,abmijk->backji', t1, tmps_["80_vvoooo"])
    del tmps_["80_vvoooo"]
    triples_resid -= einsum('backji->abcijk', tmps_["112_vvvooo"])
    triples_resid -= einsum('abckij->abcijk', tmps_["112_vvvooo"])

    # triples_resid += +1.00 P(j,k) <m,l||d,k> t1(b,l) t1(c,m) t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcjik->abcijk', tmps_["112_vvvooo"])
    triples_resid += einsum('backij->abcijk', tmps_["112_vvvooo"])
    triples_resid -= einsum('cabkij->abcijk', tmps_["112_vvvooo"])
    triples_resid -= einsum('bacjik->abcijk', tmps_["112_vvvooo"])

    # triples_resid += +1.00 P(j,k) P(b,c) <m,l||d,k> t1(a,l) t1(b,m) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cabjik->abcijk', tmps_["112_vvvooo"])

    # triples_resid += +1.00 <m,l||d,i> t1(b,l) t1(c,m) t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abckji->abcijk', tmps_["112_vvvooo"])

    # triples_resid += +1.00 P(b,c) <m,l||d,i> t1(a,l) t1(b,m) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cabkji->abcijk', tmps_["112_vvvooo"])
    del tmps_["112_vvvooo"]

    # tmps_[18_vvvo](b,d,c,i) = 1.00 eri[vovv](c,l,d,e) * t2(e,b,i,l) // flops: o1v3 = o2v4 | mem: o1v3 = o1v3
    tmps_["18_vvvo"] = einsum('clde,ebil->bdci', eri["vovv"], t2)

    # tmps_[55_vvvooo](a,c,b,k,j,i) = 1.00 t2(d,b,i,j) * eri[vovv](a,l,d,e) * t2(e,c,k,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["55_vvvooo"] = einsum('dbij,cdak->acbkji', t2, tmps_["18_vvvo"])
    triples_resid -= einsum('bcajki->abcijk', tmps_["55_vvvooo"])

    # triples_resid += +1.00 P(i,j) <l,c||d,e> t2(d,a,j,k) t2(e,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbaikj->abcijk', tmps_["55_vvvooo"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,j) t2(e,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbkji->abcijk', tmps_["55_vvvooo"])
    triples_resid += einsum('bcaikj->abcijk', tmps_["55_vvvooo"])

    # triples_resid += +1.00 P(i,j) P(a,b) <l,a||d,e> t2(d,b,j,k) t2(e,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbikj->abcijk', tmps_["55_vvvooo"])
    triples_resid += einsum('acbjki->abcijk', tmps_["55_vvvooo"])
    triples_resid += einsum('bcakji->abcijk', tmps_["55_vvvooo"])
    triples_resid += einsum('cbajki->abcijk', tmps_["55_vvvooo"])

    # triples_resid += +1.00 <l,c||d,e> t2(d,a,i,j) t2(e,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakji->abcijk', tmps_["55_vvvooo"])
    del tmps_["55_vvvooo"]

    # tmps_[85_vvoo](a,b,i,j) = 1.00 t1(c,j) * eri[vovv](a,k,c,d) * t2(d,b,i,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["85_vvoo"] = einsum('cj,bcai->abij', t1, tmps_["18_vvvo"])
    del tmps_["18_vvvo"]

    # doubles_resid += -1.00 P(i,j) P(a,b) <k,a||c,d> t1(c,j) t2(d,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["85_vvoo"])
    doubles_resid -= einsum('baij->abij', tmps_["85_vvoo"])
    doubles_resid -= einsum('abji->abij', tmps_["85_vvoo"])
    doubles_resid += einsum('baji->abij', tmps_["85_vvoo"])
    del tmps_["85_vvoo"]

    # tmps_[19_vvvo](a,e,b,k) = 1.00 eri[vovv](b,l,d,e) * t2(d,a,k,l) // flops: o1v3 = o2v4 | mem: o1v3 = o1v3
    tmps_["19_vvvo"] = einsum('blde,dakl->aebk', eri["vovv"], t2)

    # tmps_[56_vvvooo](b,a,c,j,k,i) = 1.00 t2(e,c,i,k) * eri[vovv](b,l,d,e) * t2(d,a,j,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["56_vvvooo"] = einsum('ecik,aebj->bacjki', t2, tmps_["19_vvvo"])
    del tmps_["19_vvvo"]
    triples_resid += einsum('backji->abcijk', tmps_["56_vvvooo"])

    # triples_resid += +1.00 <l,c||d,e> t2(d,a,i,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabikj->abcijk', tmps_["56_vvvooo"])
    triples_resid += einsum('abcjki->abcijk', tmps_["56_vvvooo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <l,a||d,e> t2(d,b,k,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abckji->abcijk', tmps_["56_vvvooo"])
    triples_resid -= einsum('bacjki->abcijk', tmps_["56_vvvooo"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcikj->abcijk', tmps_["56_vvvooo"])
    triples_resid += einsum('bacikj->abcijk', tmps_["56_vvvooo"])

    # triples_resid += +1.00 P(j,k) <l,c||d,e> t2(d,a,k,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabkji->abcijk', tmps_["56_vvvooo"])
    triples_resid += einsum('cabjki->abcijk', tmps_["56_vvvooo"])
    del tmps_["56_vvvooo"]

    # tmps_[20_vvoo](b,d,i,l) = 1.00 eri[oovv](m,l,d,e) * t2(e,b,i,m) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["20_vvoo"] = einsum('mlde,ebim->bdil', eri["oovv"], t2)

    # singles_resid += +1.00 <k,j||b,c> t1(b,j) t2(c,a,i,k)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid += einsum('abij,bj->ai', tmps_["20_vvoo"], t1)

    # tmps_[66_vvoooo](b,a,l,i,k,j) = 1.00 t2(d,a,j,k) * eri[oovv](m,l,d,e) * t2(e,b,i,m) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
    tmps_["66_vvoooo"] = einsum('dajk,bdil->balikj', t2, tmps_["20_vvoo"])

    # tmps_[114_vvvooo](c,a,b,j,k,i) = 1.00 t2(d,a,j,k) * eri[oovv](m,l,d,e) * t2(e,b,i,m) * t1(c,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["114_vvvooo"] = einsum('balikj,cl->cabjki', tmps_["66_vvoooo"], t1)
    del tmps_["66_vvoooo"]

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(c,l) t2(d,a,j,k) t2(e,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabjki->abcijk', tmps_["114_vvvooo"])

    # triples_resid += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,j) t2(e,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabijk->abcijk', tmps_["114_vvvooo"])
    triples_resid += einsum('cabikj->abcijk', tmps_["114_vvvooo"])
    triples_resid += einsum('bacijk->abcijk', tmps_["114_vvvooo"])
    triples_resid += einsum('abcikj->abcijk', tmps_["114_vvvooo"])
    triples_resid += einsum('bacjki->abcijk', tmps_["114_vvvooo"])

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,j) t2(e,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcijk->abcijk', tmps_["114_vvvooo"])
    triples_resid -= einsum('bacikj->abcijk', tmps_["114_vvvooo"])

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,j,k) t2(e,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcjki->abcijk', tmps_["114_vvvooo"])
    del tmps_["114_vvvooo"]

    # tmps_[83_vvoo](b,a,j,i) = 1.00 t2(c,a,i,k) * eri[oovv](l,k,c,d) * t2(d,b,j,l) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["83_vvoo"] = einsum('caik,bcjk->baji', t2, tmps_["20_vvoo"])

    # doubles_resid += +1.00 P(i,j) <l,k||c,d> t2(c,a,j,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('baij->abij', tmps_["83_vvoo"])
    doubles_resid -= einsum('baji->abij', tmps_["83_vvoo"])
    del tmps_["83_vvoo"]

    # tmps_[91_vooo](b,k,j,i) = 1.00 t1(c,i) * eri[oovv](l,k,c,d) * t2(d,b,j,l) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["91_vooo"] = einsum('ci,bcjk->bkji', t1, tmps_["20_vvoo"])
    del tmps_["20_vvoo"]

    # tmps_[120_vvoo](b,a,j,i) = 1.00 t1(c,j) * eri[oovv](l,k,c,d) * t2(d,a,i,l) * t1(b,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["120_vvoo"] = einsum('akij,bk->baji', tmps_["91_vooo"], t1)
    del tmps_["91_vooo"]
    doubles_resid -= einsum('baji->abij', tmps_["120_vvoo"])
    doubles_resid += einsum('baij->abij', tmps_["120_vvoo"])
    doubles_resid -= einsum('abij->abij', tmps_["120_vvoo"])

    # doubles_resid += +1.00 P(i,j) P(a,b) <l,k||c,d> t1(c,j) t1(a,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abji->abij', tmps_["120_vvoo"])
    del tmps_["120_vvoo"]

    # tmps_[21_vooo](a,k,i,l) = 0.50 eri[vovv](a,l,d,e) * t2(d,e,i,k) // flops: o3v1 = o3v3 | mem: o3v1 = o3v1
    tmps_["21_vooo"] = 0.50 * einsum('alde,deik->akil', eri["vovv"], t2)

    # tmps_[70_vvvooo](b,a,c,k,i,j) = 1.00 eri[vovv](c,l,d,e) * t2(d,e,i,j) * t2(a,b,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["70_vvvooo"] = einsum('cjil,abkl->backij', tmps_["21_vooo"], t2)

    # triples_resid += -0.50 P(i,j) <l,c||d,e> t2(d,e,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bacijk->abcijk', tmps_["70_vvvooo"])
    triples_resid -= einsum('cbajik->abcijk', tmps_["70_vvvooo"])
    triples_resid -= einsum('bacjik->abcijk', tmps_["70_vvvooo"])
    triples_resid -= einsum('cabkij->abcijk', tmps_["70_vvvooo"])

    # triples_resid += -0.50 P(i,j) P(a,b) <l,a||d,e> t2(d,e,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaijk->abcijk', tmps_["70_vvvooo"])
    triples_resid -= einsum('cabijk->abcijk', tmps_["70_vvvooo"])
    triples_resid += einsum('cabjik->abcijk', tmps_["70_vvvooo"])

    # triples_resid += -0.50 P(a,b) <l,a||d,e> t2(d,e,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakij->abcijk', tmps_["70_vvvooo"])

    # triples_resid += -0.50 <l,c||d,e> t2(d,e,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('backij->abcijk', tmps_["70_vvvooo"])
    del tmps_["70_vvvooo"]

    # tmps_[95_vvoo](a,b,i,j) = 1.00 t1(b,k) * eri[vovv](a,k,c,d) * t2(c,d,i,j) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["95_vvoo"] = einsum('bk,ajik->abij', t1, tmps_["21_vooo"])
    del tmps_["21_vooo"]
    doubles_resid += einsum('baij->abij', tmps_["95_vvoo"])

    # doubles_resid += +0.50 P(a,b) <k,a||c,d> t1(b,k) t2(c,d,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["95_vvoo"])
    del tmps_["95_vvoo"]

    # tmps_[22_vvoo](a,e,i,m) = 1.00 eri[oovv](m,l,d,e) * t2(d,a,i,l) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["22_vvoo"] = einsum('mlde,dail->aeim', eri["oovv"], t2)

    # tmps_[49_vvvooo](c,b,a,j,k,i) = 1.00 t3(e,a,b,i,k,m) * eri[oovv](m,l,d,e) * t2(d,c,j,l) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
    tmps_["49_vvvooo"] = einsum('eabikm,cejm->cbajki', t3, tmps_["22_vvoo"])
    del tmps_["22_vvoo"]

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t2(d,a,i,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbikj->abcijk', tmps_["49_vvvooo"])
    triples_resid += einsum('bcajki->abcijk', tmps_["49_vvvooo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,e> t2(d,a,k,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbkji->abcijk', tmps_["49_vvvooo"])
    triples_resid -= einsum('acbjki->abcijk', tmps_["49_vvvooo"])
    triples_resid -= einsum('bcaikj->abcijk', tmps_["49_vvvooo"])

    # triples_resid += +1.00 <m,l||d,e> t2(d,c,i,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaikj->abcijk', tmps_["49_vvvooo"])
    triples_resid -= einsum('bcakji->abcijk', tmps_["49_vvvooo"])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t2(d,c,k,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakji->abcijk', tmps_["49_vvvooo"])
    triples_resid -= einsum('cbajki->abcijk', tmps_["49_vvvooo"])
    del tmps_["49_vvvooo"]

    # tmps_[23_vvvo](c,b,d,i) = 0.50 eri[oovo](m,l,d,i) * t2(b,c,m,l) // flops: o1v3 = o3v3 | mem: o1v3 = o1v3
    tmps_["23_vvvo"] = 0.50 * einsum('mldi,bcml->cbdi', eri["oovo"], t2)

    # tmps_[58_vvvooo](a,b,c,j,i,k) = 1.00 eri[oovo](m,l,d,k) * t2(b,c,m,l) * t2(d,a,i,j) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["58_vvvooo"] = einsum('cbdk,daij->abcjik', tmps_["23_vvvo"], t2)

    # triples_resid += -0.50 P(j,k) P(a,b) <m,l||d,k> t2(d,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcjik->abcijk', tmps_["58_vvvooo"])
    triples_resid -= einsum('backij->abcijk', tmps_["58_vvvooo"])

    # triples_resid += -0.50 P(a,b) <m,l||d,i> t2(d,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abckji->abcijk', tmps_["58_vvvooo"])

    # triples_resid += -0.50 P(j,k) <m,l||d,k> t2(d,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabjik->abcijk', tmps_["58_vvvooo"])
    triples_resid += einsum('abckij->abcijk', tmps_["58_vvvooo"])
    triples_resid += einsum('bacjik->abcijk', tmps_["58_vvvooo"])
    triples_resid += einsum('cabkij->abcijk', tmps_["58_vvvooo"])

    # triples_resid += -0.50 <m,l||d,i> t2(d,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabkji->abcijk', tmps_["58_vvvooo"])
    triples_resid += einsum('backji->abcijk', tmps_["58_vvvooo"])
    del tmps_["58_vvvooo"]

    # tmps_[87_vvoo](a,b,j,i) = 1.00 t1(c,i) * eri[oovo](l,k,c,j) * t2(a,b,l,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["87_vvoo"] = einsum('ci,bacj->abji', t1, tmps_["23_vvvo"])
    del tmps_["23_vvvo"]

    # doubles_resid += +0.50 P(i,j) <l,k||c,j> t1(c,i) t2(a,b,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abji->abij', tmps_["87_vvoo"])
    doubles_resid -= einsum('abij->abij', tmps_["87_vvoo"])
    del tmps_["87_vvoo"]

    # tmps_[24_vvoo](a,e,i,l) = 1.00 eri[oovv](m,l,d,e) * t2(d,a,i,m) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["24_vvoo"] = einsum('mlde,daim->aeil', eri["oovv"], t2)

    # tmps_[67_vvoooo](a,b,l,i,k,j) = 1.00 t2(e,b,j,k) * eri[oovv](m,l,d,e) * t2(d,a,i,m) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
    tmps_["67_vvoooo"] = einsum('ebjk,aeil->ablikj', t2, tmps_["24_vvoo"])
    del tmps_["24_vvoo"]

    # tmps_[113_vvvooo](c,b,a,j,k,i) = 1.00 t1(a,l) * t2(e,c,j,k) * eri[oovv](m,l,d,e) * t2(d,b,i,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["113_vvvooo"] = einsum('al,bclikj->cbajki', t1, tmps_["67_vvoooo"])
    del tmps_["67_vvoooo"]

    # triples_resid += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,m) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bacjki->abcijk', tmps_["113_vvvooo"])
    triples_resid += einsum('bacikj->abcijk', tmps_["113_vvvooo"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(c,l) t2(d,a,k,m) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bacijk->abcijk', tmps_["113_vvvooo"])

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbajki->abcijk', tmps_["113_vvvooo"])
    triples_resid -= einsum('cabikj->abcijk', tmps_["113_vvvooo"])
    triples_resid += einsum('cabjki->abcijk', tmps_["113_vvvooo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,k,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbaijk->abcijk', tmps_["113_vvvooo"])
    triples_resid += einsum('cbaikj->abcijk', tmps_["113_vvvooo"])
    triples_resid += einsum('cabijk->abcijk', tmps_["113_vvvooo"])
    del tmps_["113_vvvooo"]

    # tmps_[25_vvoo](b,a,i,j) = 1.00 eri[vovo](a,k,c,j) * t2(c,b,i,k) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["25_vvoo"] = einsum('akcj,cbik->baij', eri["vovo"], t2)
    doubles_resid += einsum('abij->abij', tmps_["25_vvoo"])

    # doubles_resid += +1.00 P(i,j) P(a,b) <k,a||c,j> t2(c,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('baij->abij', tmps_["25_vvoo"])
    doubles_resid += einsum('baji->abij', tmps_["25_vvoo"])
    doubles_resid -= einsum('abji->abij', tmps_["25_vvoo"])
    del tmps_["25_vvoo"]

    # tmps_[26_vooo](b,i,k,m) = 1.00 eri[oovo](m,l,d,k) * t2(d,b,i,l) // flops: o3v1 = o4v2 | mem: o3v1 = o3v1
    tmps_["26_vooo"] = einsum('mldk,dbil->bikm', eri["oovo"], t2)

    # tmps_[64_vvvooo](b,c,a,k,i,j) = 1.00 t2(a,c,j,m) * eri[oovo](m,l,d,k) * t2(d,b,i,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["64_vvvooo"] = einsum('acjm,bikm->bcakij', t2, tmps_["26_vooo"])
    del tmps_["26_vooo"]
    triples_resid += einsum('bcakij->abcijk', tmps_["64_vvvooo"])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,k> t2(d,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbkji->abcijk', tmps_["64_vvvooo"])
    triples_resid += einsum('cbajik->abcijk', tmps_["64_vvvooo"])
    triples_resid -= einsum('acbijk->abcijk', tmps_["64_vvvooo"])
    triples_resid -= einsum('bcaikj->abcijk', tmps_["64_vvvooo"])
    triples_resid += einsum('acbjik->abcijk', tmps_["64_vvvooo"])

    # triples_resid += -1.00 P(i,k) P(a,b) <m,l||d,j> t2(d,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbjki->abcijk', tmps_["64_vvvooo"])
    triples_resid -= einsum('bcajik->abcijk', tmps_["64_vvvooo"])
    triples_resid -= einsum('cbaijk->abcijk', tmps_["64_vvvooo"])
    triples_resid -= einsum('acbkij->abcijk', tmps_["64_vvvooo"])
    triples_resid -= einsum('bcakji->abcijk', tmps_["64_vvvooo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,i> t2(d,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbikj->abcijk', tmps_["64_vvvooo"])
    triples_resid += einsum('bcaijk->abcijk', tmps_["64_vvvooo"])

    # triples_resid += +1.00 P(i,j) <m,l||d,k> t2(d,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakji->abcijk', tmps_["64_vvvooo"])
    triples_resid -= einsum('cbakij->abcijk', tmps_["64_vvvooo"])

    # triples_resid += +1.00 P(j,k) <m,l||d,i> t2(d,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaikj->abcijk', tmps_["64_vvvooo"])

    # triples_resid += -1.00 P(i,k) <m,l||d,j> t2(d,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbajki->abcijk', tmps_["64_vvvooo"])
    triples_resid += einsum('bcajki->abcijk', tmps_["64_vvvooo"])
    del tmps_["64_vvvooo"]

    # tmps_[27_oooo](j,i,l,m) = 0.50 eri[oovv](m,l,d,e) * t2(d,e,i,j) // flops: o4v0 = o4v2 | mem: o4v0 = o4v0
    tmps_["27_oooo"] = 0.50 * einsum('mlde,deij->jilm', eri["oovv"], t2)

    # doubles_resid += +0.25 <l,k||c,d> t2(c,d,i,j) t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
    doubles_resid += 0.50 * einsum('jikl,ablk->abij', tmps_["27_oooo"], t2)

    # tmps_[54_vvvooo](c,b,a,j,i,k) = 0.50 eri[oovv](m,l,d,e) * t2(d,e,i,k) * t3(a,b,c,j,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["54_vvvooo"] = 0.50 * einsum('kilm,abcjml->cbajik', tmps_["27_oooo"], t3)
    triples_resid -= einsum('cbajik->abcijk', tmps_["54_vvvooo"])

    # triples_resid += +0.25 <m,l||d,e> t2(d,e,i,j) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakij->abcijk', tmps_["54_vvvooo"])

    # triples_resid += +0.25 P(i,j) <m,l||d,e> t2(d,e,j,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaijk->abcijk', tmps_["54_vvvooo"])
    del tmps_["54_vvvooo"]

    # tmps_[102_vooo](c,m,i,k) = 1.00 t1(c,l) * eri[oovv](m,l,d,e) * t2(d,e,i,k) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["102_vooo"] = einsum('cl,kilm->cmik', t1, tmps_["27_oooo"])
    del tmps_["27_oooo"]

    # doubles_resid += -0.50 <l,k||c,d> t1(a,k) t1(b,l) t2(c,d,i,j)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('alij,bl->abij', tmps_["102_vooo"], t1)

    # tmps_[109_vvvooo](b,c,a,j,i,k) = 1.00 t2(a,c,k,m) * t1(b,l) * eri[oovv](m,l,d,e) * t2(d,e,i,j) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["109_vvvooo"] = einsum('ackm,bmij->bcajik', t2, tmps_["102_vooo"])
    del tmps_["102_vooo"]
    triples_resid -= einsum('cbakij->abcijk', tmps_["109_vvvooo"])

    # triples_resid += +0.50 P(a,b) <m,l||d,e> t1(a,l) t2(d,e,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbjik->abcijk', tmps_["109_vvvooo"])
    triples_resid += einsum('bcakij->abcijk', tmps_["109_vvvooo"])

    # triples_resid += +0.50 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,e,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbkji->abcijk', tmps_["109_vvvooo"])
    triples_resid -= einsum('bcajik->abcijk', tmps_["109_vvvooo"])
    triples_resid -= einsum('acbkij->abcijk', tmps_["109_vvvooo"])
    triples_resid -= einsum('bcakji->abcijk', tmps_["109_vvvooo"])

    # triples_resid += +0.50 P(i,j) <m,l||d,e> t1(c,l) t2(d,e,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakji->abcijk', tmps_["109_vvvooo"])

    # triples_resid += +0.50 <m,l||d,e> t1(c,l) t2(d,e,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbajik->abcijk', tmps_["109_vvvooo"])
    del tmps_["109_vvvooo"]

    # tmps_[28_vooo](b,j,i,k) = 1.00 eri[oovo](l,k,c,i) * t2(c,b,j,l) // flops: o3v1 = o4v2 | mem: o3v1 = o3v1
    tmps_["28_vooo"] = einsum('lkci,cbjl->bjik', eri["oovo"], t2)

    # tmps_[92_vvoo](b,a,i,j) = 1.00 t1(a,k) * eri[oovo](l,k,c,i) * t2(c,b,j,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["92_vvoo"] = einsum('ak,bjik->baij', t1, tmps_["28_vooo"])
    del tmps_["28_vooo"]
    doubles_resid += einsum('baij->abij', tmps_["92_vvoo"])

    # doubles_resid += -1.00 P(i,j) P(a,b) <l,k||c,j> t1(a,k) t2(c,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('baji->abij', tmps_["92_vvoo"])
    doubles_resid += einsum('abji->abij', tmps_["92_vvoo"])
    doubles_resid -= einsum('abij->abij', tmps_["92_vvoo"])
    del tmps_["92_vvoo"]

    # tmps_[29_vvvo](e,c,b,j) = 1.00 eri[vvvv](b,c,d,e) * t1(d,j) // flops: o1v3 = o1v4 | mem: o1v3 = o1v3
    tmps_["29_vvvo"] = einsum('bcde,dj->ecbj', eri["vvvv"], t1)

    # doubles_resid += -1.00 <a,b||c,d> t1(c,j) t1(d,i)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= einsum('dbaj,di->abij', tmps_["29_vvvo"], t1)

    # tmps_[59_vvvooo](a,c,b,i,k,j) = 1.00 t2(e,b,j,k) * eri[vvvv](a,c,d,e) * t1(d,i) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["59_vvvooo"] = einsum('ebjk,ecai->acbikj', t2, tmps_["29_vvvo"])
    del tmps_["29_vvvo"]
    triples_resid += einsum('acbjki->abcijk', tmps_["59_vvvooo"])
    triples_resid -= einsum('acbkji->abcijk', tmps_["59_vvvooo"])
    triples_resid -= einsum('bcajki->abcijk', tmps_["59_vvvooo"])

    # triples_resid += +1.00 P(b,c) <a,b||d,e> t1(d,i) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcikj->abcijk', tmps_["59_vvvooo"])
    triples_resid -= einsum('acbikj->abcijk', tmps_["59_vvvooo"])

    # triples_resid += +1.00 P(j,k) P(b,c) <a,b||d,e> t1(d,k) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abckji->abcijk', tmps_["59_vvvooo"])

    # triples_resid += +1.00 <b,c||d,e> t1(d,i) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcaikj->abcijk', tmps_["59_vvvooo"])

    # triples_resid += +1.00 P(j,k) <b,c||d,e> t1(d,k) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcakji->abcijk', tmps_["59_vvvooo"])
    triples_resid -= einsum('abcjki->abcijk', tmps_["59_vvvooo"])
    del tmps_["59_vvvooo"]

    # tmps_[30_vvoo](e,c,i,l) = 1.00 eri[vovv](c,l,d,e) * t1(d,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["30_vvoo"] = einsum('clde,di->ecil', eri["vovv"], t1)

    # tmps_[50_vvvooo](a,c,b,k,j,i) = 1.00 t3(e,b,c,i,j,l) * eri[vovv](a,l,d,e) * t1(d,k) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
    tmps_["50_vvvooo"] = einsum('ebcijl,eakl->acbkji', t3, tmps_["30_vvoo"])
    triples_resid += einsum('bcajki->abcijk', tmps_["50_vvvooo"])
    triples_resid -= einsum('bcaikj->abcijk', tmps_["50_vvvooo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,a||d,e> t1(d,k) t3(e,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbkji->abcijk', tmps_["50_vvvooo"])
    triples_resid -= einsum('bcakji->abcijk', tmps_["50_vvvooo"])

    # triples_resid += -1.00 <l,c||d,e> t1(d,i) t3(e,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaikj->abcijk', tmps_["50_vvvooo"])

    # triples_resid += -1.00 P(j,k) <l,c||d,e> t1(d,k) t3(e,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakji->abcijk', tmps_["50_vvvooo"])
    triples_resid -= einsum('cbajki->abcijk', tmps_["50_vvvooo"])
    triples_resid -= einsum('acbjki->abcijk', tmps_["50_vvvooo"])

    # triples_resid += -1.00 P(a,b) <l,a||d,e> t1(d,i) t3(e,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbikj->abcijk', tmps_["50_vvvooo"])
    del tmps_["50_vvvooo"]

    # tmps_[62_vvoooo](b,a,l,k,j,i) = 1.00 t2(e,a,i,j) * eri[vovv](b,l,d,e) * t1(d,k) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
    tmps_["62_vvoooo"] = einsum('eaij,ebkl->balkji', t2, tmps_["30_vvoo"])

    # tmps_[108_vvvooo](c,b,a,i,k,j) = 1.00 t1(a,l) * t2(e,c,i,k) * eri[vovv](b,l,d,e) * t1(d,j) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["108_vvvooo"] = einsum('al,bcljki->cbaikj', t1, tmps_["62_vvoooo"])
    del tmps_["62_vvoooo"]
    triples_resid += einsum('acbjki->abcijk', tmps_["108_vvvooo"])

    # triples_resid += +1.00 P(a,b) <l,c||d,e> t1(d,i) t1(a,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcajki->abcijk', tmps_["108_vvvooo"])
    triples_resid -= einsum('acbikj->abcijk', tmps_["108_vvvooo"])
    triples_resid += einsum('acbijk->abcijk', tmps_["108_vvvooo"])
    triples_resid += einsum('bcaikj->abcijk', tmps_["108_vvvooo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <l,c||d,e> t1(d,k) t1(a,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcaijk->abcijk', tmps_["108_vvvooo"])
    triples_resid -= einsum('abcjki->abcijk', tmps_["108_vvvooo"])
    triples_resid += einsum('abcikj->abcijk', tmps_["108_vvvooo"])
    triples_resid -= einsum('abcijk->abcijk', tmps_["108_vvvooo"])

    # triples_resid += -1.00 P(j,k) P(a,c) <l,b||d,e> t1(d,k) t1(a,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaijk->abcijk', tmps_["108_vvvooo"])
    triples_resid += einsum('bacjki->abcijk', tmps_["108_vvvooo"])

    # triples_resid += -1.00 P(a,c) <l,b||d,e> t1(d,i) t1(a,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbajki->abcijk', tmps_["108_vvvooo"])
    triples_resid += einsum('bacijk->abcijk', tmps_["108_vvvooo"])
    triples_resid += einsum('cabikj->abcijk', tmps_["108_vvvooo"])

    # triples_resid += +1.00 P(j,k) P(b,c) <l,a||d,e> t1(d,k) t1(b,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabijk->abcijk', tmps_["108_vvvooo"])
    triples_resid -= einsum('bacikj->abcijk', tmps_["108_vvvooo"])
    triples_resid -= einsum('cbaikj->abcijk', tmps_["108_vvvooo"])

    # triples_resid += +1.00 P(b,c) <l,a||d,e> t1(d,i) t1(b,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cabjki->abcijk', tmps_["108_vvvooo"])
    del tmps_["108_vvvooo"]

    # tmps_[89_vooo](a,i,l,j) = 1.00 eri[vovv](a,l,d,e) * t1(d,j) * t1(e,i) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["89_vooo"] = einsum('eajl,ei->ailj', tmps_["30_vvoo"], t1)
    del tmps_["30_vvoo"]

    # tmps_[111_vvvooo](c,b,a,j,i,k) = 1.00 t2(a,b,k,l) * eri[vovv](c,l,d,e) * t1(d,j) * t1(e,i) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["111_vvvooo"] = einsum('abkl,cilj->cbajik', t2, tmps_["89_vooo"])

    # triples_resid += +1.00 <l,c||d,e> t1(d,j) t1(e,i) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbajik->abcijk', tmps_["111_vvvooo"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t1(d,j) t1(e,i) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbjik->abcijk', tmps_["111_vvvooo"])
    triples_resid -= einsum('bcakij->abcijk', tmps_["111_vvvooo"])
    triples_resid += einsum('acbkij->abcijk', tmps_["111_vvvooo"])

    # triples_resid += +1.00 P(i,j) P(a,b) <l,a||d,e> t1(d,k) t1(e,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbkji->abcijk', tmps_["111_vvvooo"])
    triples_resid += einsum('cbakij->abcijk', tmps_["111_vvvooo"])
    triples_resid += einsum('bcajik->abcijk', tmps_["111_vvvooo"])
    triples_resid += einsum('bcakji->abcijk', tmps_["111_vvvooo"])

    # triples_resid += +1.00 P(i,j) <l,c||d,e> t1(d,k) t1(e,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakji->abcijk', tmps_["111_vvvooo"])
    del tmps_["111_vvvooo"]

    # tmps_[123_vvoo](a,b,j,i) = 1.00 t1(b,k) * eri[vovv](a,k,c,d) * t1(c,j) * t1(d,i) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["123_vvoo"] = einsum('bk,aikj->abji', t1, tmps_["89_vooo"])
    del tmps_["89_vooo"]

    # doubles_resid += -1.00 P(a,b) <k,a||c,d> t1(c,j) t1(d,i) t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abji->abij', tmps_["123_vvoo"])
    doubles_resid -= einsum('baji->abij', tmps_["123_vvoo"])
    del tmps_["123_vvoo"]

    # tmps_[31_vvoo](d,c,i,l) = 1.00 eri[vovv](c,l,d,e) * t1(e,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["31_vvoo"] = einsum('clde,ei->dcil', eri["vovv"], t1)

    # singles_resid += +1.00 <j,a||b,c> t1(b,j) t1(c,i)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid -= einsum('baij,bj->ai', tmps_["31_vvoo"], t1)
    del tmps_["31_vvoo"]

    # tmps_[32_vv](a,d) = 0.50 eri[oovv](l,k,c,d) * t2(c,a,l,k) // flops: o0v2 = o2v3 | mem: o0v2 = o0v2
    tmps_["32_vv"] = 0.50 * einsum('lkcd,calk->ad', eri["oovv"], t2)

    # doubles_resid += -0.50 <l,k||c,d> t2(c,a,l,k) t2(d,b,i,j)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= einsum('ad,dbij->abij', tmps_["32_vv"], t2)

    # tmps_[61_vvvooo](a,c,b,k,j,i) = 1.00 t3(e,b,c,i,j,k) * eri[oovv](m,l,d,e) * t2(d,a,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["61_vvvooo"] = einsum('ebcijk,ae->acbkji', t3, tmps_["32_vv"])
    del tmps_["32_vv"]

    # triples_resid += -0.50 P(a,b) <m,l||d,e> t2(d,a,m,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbkji->abcijk', tmps_["61_vvvooo"])
    triples_resid += einsum('bcakji->abcijk', tmps_["61_vvvooo"])

    # triples_resid += -0.50 <m,l||d,e> t2(d,c,m,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakji->abcijk', tmps_["61_vvvooo"])
    del tmps_["61_vvvooo"]

    # tmps_[33_vvoo](b,a,j,i) = 1.00 eri[vvvo](a,b,c,i) * t1(c,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["33_vvoo"] = einsum('abci,cj->baji', eri["vvvo"], t1)
    doubles_resid -= einsum('baji->abij', tmps_["33_vvoo"])

    # doubles_resid += +1.00 P(i,j) <a,b||c,j> t1(c,i)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('baij->abij', tmps_["33_vvoo"])
    del tmps_["33_vvoo"]

    # tmps_[34_vvoo](b,a,j,i) = 1.00 f[vv](a,c) * t2(c,b,i,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["34_vvoo"] = einsum('ac,cbij->baji', f["vv"], t2)

    # doubles_resid += +1.00 P(a,b) f(a,c) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('baji->abij', tmps_["34_vvoo"])
    doubles_resid -= einsum('abji->abij', tmps_["34_vvoo"])
    del tmps_["34_vvoo"]

    # tmps_[35_vooo](d,i,k,l) = 1.00 eri[oovv](l,k,c,d) * t1(c,i) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["35_vooo"] = einsum('lkcd,ci->dikl', eri["oovv"], t1)

    # singles_resid += +0.50 <k,j||b,c> t1(b,i) t2(c,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
    singles_resid += 0.50 * einsum('cijk,cakj->ai', tmps_["35_vooo"], t2)

    # tmps_[51_vvoooo](c,a,j,i,l,k) = 1.00 eri[oovv](m,l,d,e) * t1(d,k) * t3(e,a,c,i,j,m) // flops: o4v2 = o5v3 | mem: o4v2 = o4v2
    tmps_["51_vvoooo"] = einsum('eklm,eacijm->cajilk', tmps_["35_vooo"], t3)

    # tmps_[115_vvvooo](b,a,c,i,j,k) = 1.00 eri[oovv](m,l,d,e) * t1(d,i) * t3(e,a,c,j,k,m) * t1(b,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["115_vvvooo"] = einsum('cakjli,bl->bacijk', tmps_["51_vvoooo"], t1)
    del tmps_["51_vvoooo"]
    triples_resid -= einsum('cabjik->abcijk', tmps_["115_vvvooo"])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t1(d,k) t1(c,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cabkij->abcijk', tmps_["115_vvvooo"])
    triples_resid += einsum('bacjik->abcijk', tmps_["115_vvvooo"])
    triples_resid -= einsum('backij->abcijk', tmps_["115_vvvooo"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,i) t1(c,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cabijk->abcijk', tmps_["115_vvvooo"])
    triples_resid -= einsum('bacijk->abcijk', tmps_["115_vvvooo"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,i) t1(a,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["115_vvvooo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t1(a,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abckij->abcijk', tmps_["115_vvvooo"])
    triples_resid -= einsum('abcjik->abcijk', tmps_["115_vvvooo"])
    del tmps_["115_vvvooo"]

    # tmps_[79_vooooo](b,m,l,j,k,i) = 1.00 t2(e,b,i,k) * eri[oovv](m,l,d,e) * t1(d,j) // flops: o5v1 = o5v2 | mem: o5v1 = o5v1
    tmps_["79_vooooo"] = einsum('ebik,ejlm->bmljki', t2, tmps_["35_vooo"])

    # tmps_[119_vvoooo](a,c,i,k,j,l) = 1.00 t1(c,m) * t2(e,a,i,k) * eri[oovv](m,l,d,e) * t1(d,j) // flops: o4v2 = o5v2 | mem: o4v2 = o4v2
    tmps_["119_vvoooo"] = einsum('cm,amljki->acikjl', t1, tmps_["79_vooooo"])
    del tmps_["79_vooooo"]

    # tmps_[127_vvvooo](b,c,a,j,k,i) = 1.00 t1(c,m) * t2(e,a,i,k) * eri[oovv](m,l,d,e) * t1(d,j) * t1(b,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["127_vvvooo"] = einsum('acikjl,bl->bcajki', tmps_["119_vvoooo"], t1)
    del tmps_["119_vvoooo"]
    triples_resid += einsum('acbikj->abcijk', tmps_["127_vvvooo"])

    # triples_resid += -1.00 P(b,c) <m,l||d,e> t1(d,i) t1(a,l) t1(b,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abcikj->abcijk', tmps_["127_vvvooo"])
    triples_resid += einsum('bcajki->abcijk', tmps_["127_vvvooo"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(d,k) t1(b,l) t1(c,m) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcakji->abcijk', tmps_["127_vvvooo"])
    triples_resid -= einsum('acbjki->abcijk', tmps_["127_vvvooo"])
    triples_resid += einsum('abcjki->abcijk', tmps_["127_vvvooo"])
    triples_resid += einsum('acbkji->abcijk', tmps_["127_vvvooo"])

    # triples_resid += -1.00 P(j,k) P(b,c) <m,l||d,e> t1(d,k) t1(a,l) t1(b,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('abckji->abcijk', tmps_["127_vvvooo"])

    # triples_resid += -1.00 <m,l||d,e> t1(d,i) t1(b,l) t1(c,m) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bcaikj->abcijk', tmps_["127_vvvooo"])
    del tmps_["127_vvvooo"]

    # tmps_[82_vvvo](e,c,b,i) = 0.50 t2(b,c,m,l) * eri[oovv](m,l,d,e) * t1(d,i) // flops: o1v3 = o3v3 | mem: o1v3 = o1v3
    tmps_["82_vvvo"] = 0.50 * einsum('bcml,eilm->ecbi', t2, tmps_["35_vooo"])

    # doubles_resid += -0.50 <l,k||c,d> t1(c,j) t1(d,i) t2(a,b,l,k)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= einsum('di,dbaj->abij', t1, tmps_["82_vvvo"])

    # tmps_[105_vvvooo](c,a,b,k,i,j) = 1.00 t2(a,b,m,l) * eri[oovv](m,l,d,e) * t1(d,j) * t2(e,c,i,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["105_vvvooo"] = einsum('ebaj,ecik->cabkij', tmps_["82_vvvo"], t2)
    del tmps_["82_vvvo"]
    triples_resid -= einsum('cabkij->abcijk', tmps_["105_vvvooo"])

    # triples_resid += +0.50 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcjik->abcijk', tmps_["105_vvvooo"])
    triples_resid += einsum('backij->abcijk', tmps_["105_vvvooo"])

    # triples_resid += +0.50 P(a,b) <m,l||d,e> t1(d,i) t2(e,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abckji->abcijk', tmps_["105_vvvooo"])
    triples_resid -= einsum('bacjik->abcijk', tmps_["105_vvvooo"])
    triples_resid -= einsum('abckij->abcijk', tmps_["105_vvvooo"])
    triples_resid -= einsum('backji->abcijk', tmps_["105_vvvooo"])

    # triples_resid += +0.50 <m,l||d,e> t1(d,i) t2(e,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cabkji->abcijk', tmps_["105_vvvooo"])

    # triples_resid += +0.50 P(j,k) <m,l||d,e> t1(d,k) t2(e,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cabjik->abcijk', tmps_["105_vvvooo"])
    del tmps_["105_vvvooo"]

    # tmps_[84_vooo](c,i,m,j) = 1.00 eri[oovv](m,l,d,e) * t1(d,j) * t2(e,c,i,l) // flops: o3v1 = o4v2 | mem: o3v1 = o3v1
    tmps_["84_vooo"] = einsum('ejlm,ecil->cimj', tmps_["35_vooo"], t2)

    # tmps_[107_vvvooo](b,c,a,i,j,k) = 1.00 t2(a,c,k,m) * eri[oovv](m,l,d,e) * t1(d,i) * t2(e,b,j,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["107_vvvooo"] = einsum('ackm,bjmi->bcaijk', t2, tmps_["84_vooo"])
    del tmps_["84_vooo"]
    triples_resid -= einsum('cbajik->abcijk', tmps_["107_vvvooo"])
    triples_resid += einsum('bcajik->abcijk', tmps_["107_vvvooo"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(d,i) t2(e,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbaikj->abcijk', tmps_["107_vvvooo"])
    triples_resid += einsum('acbijk->abcijk', tmps_["107_vvvooo"])
    triples_resid += einsum('bcakji->abcijk', tmps_["107_vvvooo"])
    triples_resid -= einsum('acbjik->abcijk', tmps_["107_vvvooo"])
    triples_resid += einsum('bcaikj->abcijk', tmps_["107_vvvooo"])
    triples_resid += einsum('cbakij->abcijk', tmps_["107_vvvooo"])
    triples_resid += einsum('acbkij->abcijk', tmps_["107_vvvooo"])
    triples_resid -= einsum('bcakij->abcijk', tmps_["107_vvvooo"])

    # triples_resid += +1.00 P(i,k) P(a,b) <m,l||d,e> t1(d,j) t2(e,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbjki->abcijk', tmps_["107_vvvooo"])
    triples_resid -= einsum('bcaijk->abcijk', tmps_["107_vvvooo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,i) t2(e,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbikj->abcijk', tmps_["107_vvvooo"])
    triples_resid -= einsum('bcajki->abcijk', tmps_["107_vvvooo"])

    # triples_resid += +1.00 P(i,k) <m,l||d,e> t1(d,j) t2(e,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbajki->abcijk', tmps_["107_vvvooo"])
    triples_resid += einsum('cbaijk->abcijk', tmps_["107_vvvooo"])

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbkji->abcijk', tmps_["107_vvvooo"])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(d,k) t2(e,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakji->abcijk', tmps_["107_vvvooo"])
    del tmps_["107_vvvooo"]

    # tmps_[101_oooo](l,k,j,i) = 1.00 t1(d,i) * eri[oovv](l,k,c,d) * t1(c,j) // flops: o4v0 = o4v1 | mem: o4v0 = o4v0
    tmps_["101_oooo"] = einsum('di,djkl->lkji', t1, tmps_["35_vooo"])
    del tmps_["35_vooo"]

    # tmps_[104_vvvooo](c,b,a,i,j,k) = 0.50 t1(e,j) * eri[oovv](m,l,d,e) * t1(d,k) * t3(a,b,c,i,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["104_vvvooo"] = 0.50 * einsum('mlkj,abciml->cbaijk', tmps_["101_oooo"], t3)
    triples_resid += einsum('cbajik->abcijk', tmps_["104_vvvooo"])

    # triples_resid += -0.50 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbaijk->abcijk', tmps_["104_vvvooo"])

    # triples_resid += -0.50 <m,l||d,e> t1(d,j) t1(e,i) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakij->abcijk', tmps_["104_vvvooo"])
    del tmps_["104_vvvooo"]

    # tmps_[125_vooo](a,i,j,l) = 1.00 t1(a,k) * t1(d,i) * eri[oovv](l,k,c,d) * t1(c,j) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["125_vooo"] = einsum('ak,lkji->aijl', t1, tmps_["101_oooo"])
    del tmps_["101_oooo"]

    # doubles_resid += +1.00 <l,k||c,d> t1(c,j) t1(d,i) t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('aijl,bl->abij', tmps_["125_vooo"], t1)

    # tmps_[126_vvvooo](b,c,a,j,i,k) = 1.00 t2(a,c,k,m) * t1(b,l) * t1(e,i) * eri[oovv](m,l,d,e) * t1(d,j) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["126_vvvooo"] = einsum('ackm,bijm->bcajik', t2, tmps_["125_vooo"])
    del tmps_["125_vooo"]

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(d,j) t1(e,i) t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbjik->abcijk', tmps_["126_vvvooo"])

    # triples_resid += -1.00 <m,l||d,e> t1(d,j) t1(e,i) t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbajik->abcijk', tmps_["126_vvvooo"])
    triples_resid += einsum('bcajik->abcijk', tmps_["126_vvvooo"])
    triples_resid += einsum('cbakij->abcijk', tmps_["126_vvvooo"])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakji->abcijk', tmps_["126_vvvooo"])
    triples_resid += einsum('acbkij->abcijk', tmps_["126_vvvooo"])

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t1(e,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbkji->abcijk', tmps_["126_vvvooo"])
    triples_resid -= einsum('bcakij->abcijk', tmps_["126_vvvooo"])
    triples_resid += einsum('bcakji->abcijk', tmps_["126_vvvooo"])
    del tmps_["126_vvvooo"]

    # tmps_[36_vooo](c,j,i,l) = 1.00 eri[vovo](c,l,d,i) * t1(d,j) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["36_vooo"] = einsum('cldi,dj->cjil', eri["vovo"], t1)

    # tmps_[65_vvvooo](a,c,b,j,i,k) = 1.00 t2(b,c,k,l) * eri[vovo](a,l,d,j) * t1(d,i) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["65_vvvooo"] = einsum('bckl,aijl->acbjik', t2, tmps_["36_vooo"])
    triples_resid -= einsum('bcajik->abcijk', tmps_["65_vvvooo"])
    triples_resid += einsum('bcaijk->abcijk', tmps_["65_vvvooo"])
    triples_resid -= einsum('cbakij->abcijk', tmps_["65_vvvooo"])
    triples_resid -= einsum('acbijk->abcijk', tmps_["65_vvvooo"])
    triples_resid += einsum('bcajki->abcijk', tmps_["65_vvvooo"])
    triples_resid += einsum('acbjik->abcijk', tmps_["65_vvvooo"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,a||d,i> t1(d,k) t2(b,c,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbikj->abcijk', tmps_["65_vvvooo"])
    triples_resid -= einsum('bcaikj->abcijk', tmps_["65_vvvooo"])
    triples_resid -= einsum('acbkij->abcijk', tmps_["65_vvvooo"])

    # triples_resid += -1.00 P(i,j) P(a,b) <l,a||d,k> t1(d,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbkji->abcijk', tmps_["65_vvvooo"])
    triples_resid += einsum('bcakij->abcijk', tmps_["65_vvvooo"])

    # triples_resid += +1.00 P(i,k) <l,c||d,j> t1(d,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbajki->abcijk', tmps_["65_vvvooo"])

    # triples_resid += +1.00 P(i,k) P(a,b) <l,a||d,j> t1(d,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbjki->abcijk', tmps_["65_vvvooo"])
    triples_resid += einsum('cbajik->abcijk', tmps_["65_vvvooo"])
    triples_resid -= einsum('cbaijk->abcijk', tmps_["65_vvvooo"])

    # triples_resid += -1.00 P(j,k) <l,c||d,i> t1(d,k) t2(a,b,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaikj->abcijk', tmps_["65_vvvooo"])
    triples_resid -= einsum('bcakji->abcijk', tmps_["65_vvvooo"])

    # triples_resid += -1.00 P(i,j) <l,c||d,k> t1(d,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakji->abcijk', tmps_["65_vvvooo"])
    del tmps_["65_vvvooo"]

    # tmps_[93_vvoo](b,a,j,i) = 1.00 t1(a,k) * eri[vovo](b,k,c,j) * t1(c,i) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["93_vvoo"] = einsum('ak,bijk->baji', t1, tmps_["36_vooo"])
    del tmps_["36_vooo"]
    doubles_resid -= einsum('baij->abij', tmps_["93_vvoo"])
    doubles_resid += einsum('baji->abij', tmps_["93_vvoo"])
    doubles_resid += einsum('abij->abij', tmps_["93_vvoo"])

    # doubles_resid += +1.00 P(i,j) P(a,b) <k,a||c,j> t1(c,i) t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abji->abij', tmps_["93_vvoo"])
    del tmps_["93_vvoo"]

    # tmps_[37_vooo](c,i,k,l) = 1.00 eri[oovv](l,k,c,d) * t1(d,i) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["37_vooo"] = einsum('lkcd,di->cikl', eri["oovv"], t1)

    # tmps_[103_oo](l,i) = 1.00 t1(c,k) * eri[oovv](l,k,c,d) * t1(d,i) // flops: o2v0 = o3v1 | mem: o2v0 = o2v0
    tmps_["103_oo"] = einsum('ck,cikl->li', t1, tmps_["37_vooo"])
    del tmps_["37_vooo"]

    # singles_resid += +1.00 <k,j||b,c> t1(b,j) t1(c,i) t1(a,k)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid += einsum('ki,ak->ai', tmps_["103_oo"], t1)

    # tmps_[116_vvvooo](c,b,a,k,j,i) = 1.00 t1(d,l) * eri[oovv](m,l,d,e) * t1(e,i) * t3(a,b,c,j,k,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["116_vvvooo"] = einsum('mi,abcjkm->cbakji', tmps_["103_oo"], t3)

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t1(e,i) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakji->abcijk', tmps_["116_vvvooo"])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t1(d,l) t1(e,k) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbajik->abcijk', tmps_["116_vvvooo"])
    triples_resid -= einsum('cbakij->abcijk', tmps_["116_vvvooo"])
    del tmps_["116_vvvooo"]

    # tmps_[121_vvoo](b,a,i,j) = 1.00 t2(a,b,j,l) * t1(c,k) * eri[oovv](l,k,c,d) * t1(d,i) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["121_vvoo"] = einsum('abjl,li->baij', t2, tmps_["103_oo"])
    del tmps_["103_oo"]
    doubles_resid -= einsum('baij->abij', tmps_["121_vvoo"])

    # doubles_resid += +1.00 P(i,j) <l,k||c,d> t1(c,k) t1(d,j) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('baji->abij', tmps_["121_vvoo"])
    del tmps_["121_vvoo"]

    # tmps_[38_vooo](c,k,i,l) = 1.00 f[ov](l,d) * t2(d,c,i,k) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["38_vooo"] = einsum('ld,dcik->ckil', f["ov"], t2)

    # tmps_[71_vvvooo](b,a,c,k,i,j) = 1.00 f[ov](l,d) * t2(d,c,i,j) * t2(a,b,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["71_vvvooo"] = einsum('cjil,abkl->backij', tmps_["38_vooo"], t2)

    # triples_resid += -1.00 f(l,d) t2(d,c,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('backij->abcijk', tmps_["71_vvvooo"])
    triples_resid += einsum('bacjik->abcijk', tmps_["71_vvvooo"])
    triples_resid += einsum('cabijk->abcijk', tmps_["71_vvvooo"])

    # triples_resid += -1.00 P(a,b) f(l,d) t2(d,a,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakij->abcijk', tmps_["71_vvvooo"])
    triples_resid += einsum('cabkij->abcijk', tmps_["71_vvvooo"])
    triples_resid += einsum('cbajik->abcijk', tmps_["71_vvvooo"])

    # triples_resid += -1.00 P(i,j) P(a,b) f(l,d) t2(d,a,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbaijk->abcijk', tmps_["71_vvvooo"])
    triples_resid -= einsum('cabjik->abcijk', tmps_["71_vvvooo"])

    # triples_resid += -1.00 P(i,j) f(l,d) t2(d,c,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('bacijk->abcijk', tmps_["71_vvvooo"])
    del tmps_["71_vvvooo"]

    # tmps_[98_vvoo](a,b,i,j) = 1.00 t1(b,k) * f[ov](k,c) * t2(c,a,i,j) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["98_vvoo"] = einsum('bk,ajik->abij', t1, tmps_["38_vooo"])
    del tmps_["38_vooo"]
    doubles_resid += einsum('abij->abij', tmps_["98_vvoo"])

    # doubles_resid += -1.00 P(a,b) f(k,c) t1(a,k) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('baij->abij', tmps_["98_vvoo"])
    del tmps_["98_vvoo"]

    # tmps_[39_oo](j,l) = 0.50 eri[oovv](l,k,c,d) * t2(c,d,j,k) // flops: o2v0 = o3v2 | mem: o2v0 = o2v0
    tmps_["39_oo"] = 0.50 * einsum('lkcd,cdjk->jl', eri["oovv"], t2)

    # tmps_[75_vvvooo](c,b,a,k,j,i) = 1.00 eri[oovv](m,l,d,e) * t2(d,e,i,l) * t3(a,b,c,j,k,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["75_vvvooo"] = einsum('im,abcjkm->cbakji', tmps_["39_oo"], t3)

    # triples_resid += -0.50 <m,l||d,e> t2(d,e,i,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakji->abcijk', tmps_["75_vvvooo"])

    # triples_resid += -0.50 P(j,k) <m,l||d,e> t2(d,e,k,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbajik->abcijk', tmps_["75_vvvooo"])
    triples_resid += einsum('cbakij->abcijk', tmps_["75_vvvooo"])
    del tmps_["75_vvvooo"]

    # tmps_[96_vvoo](b,a,i,j) = 1.00 eri[oovv](l,k,c,d) * t2(c,d,j,k) * t2(a,b,i,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["96_vvoo"] = einsum('jl,abil->baij', tmps_["39_oo"], t2)
    del tmps_["39_oo"]

    # doubles_resid += -0.50 P(i,j) <l,k||c,d> t2(c,d,j,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('baij->abij', tmps_["96_vvoo"])
    doubles_resid += einsum('baji->abij', tmps_["96_vvoo"])
    del tmps_["96_vvoo"]

    # tmps_[40_vv](b,c) = 1.00 eri[oovv](l,k,c,d) * t2(d,b,l,k) // flops: o0v2 = o2v3 | mem: o0v2 = o0v2
    tmps_["40_vv"] = einsum('lkcd,dblk->bc', eri["oovv"], t2)

    # doubles_resid += -0.50 <l,k||c,d> t2(c,a,i,j) t2(d,b,l,k)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= 0.50 * einsum('bc,caij->abij', tmps_["40_vv"], t2)
    del tmps_["40_vv"]

    # tmps_[41_vvoo](b,a,i,j) = 1.00 f[oo](k,j) * t2(a,b,i,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["41_vvoo"] = einsum('kj,abik->baij', f["oo"], t2)

    # doubles_resid += -1.00 P(i,j) f(k,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('baij->abij', tmps_["41_vvoo"])
    doubles_resid += einsum('baji->abij', tmps_["41_vvoo"])
    del tmps_["41_vvoo"]

    # tmps_[42_vvoo](a,b,j,i) = 1.00 eri[vooo](b,k,i,j) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["42_vvoo"] = einsum('bkij,ak->abji', eri["vooo"], t1)
    doubles_resid += einsum('abji->abij', tmps_["42_vvoo"])

    # doubles_resid += +1.00 P(a,b) <k,a||i,j> t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('baji->abij', tmps_["42_vvoo"])
    del tmps_["42_vvoo"]

    # tmps_[43_oooo](i,j,k,l) = 1.00 eri[oovo](l,k,c,j) * t1(c,i) // flops: o4v0 = o4v1 | mem: o4v0 = o4v0
    tmps_["43_oooo"] = einsum('lkcj,ci->ijkl', eri["oovo"], t1)

    # tmps_[52_vvvooo](c,b,a,k,i,j) = 0.50 t3(a,b,c,j,m,l) * eri[oovo](m,l,d,k) * t1(d,i) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["52_vvvooo"] = 0.50 * einsum('abcjml,iklm->cbakij', t3, tmps_["43_oooo"])
    triples_resid -= einsum('cbakij->abcijk', tmps_["52_vvvooo"])

    # triples_resid += +0.50 P(j,k) <m,l||d,i> t1(d,k) t3(a,b,c,j,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaikj->abcijk', tmps_["52_vvvooo"])

    # triples_resid += -0.50 P(i,k) <m,l||d,j> t1(d,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbajki->abcijk', tmps_["52_vvvooo"])
    triples_resid -= einsum('cbaijk->abcijk', tmps_["52_vvvooo"])

    # triples_resid += +0.50 P(i,j) <m,l||d,k> t1(d,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakji->abcijk', tmps_["52_vvvooo"])
    triples_resid += einsum('cbajik->abcijk', tmps_["52_vvvooo"])
    del tmps_["52_vvvooo"]

    # tmps_[100_vooo](c,m,j,i) = 1.00 t1(c,l) * eri[oovo](m,l,d,j) * t1(d,i) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["100_vooo"] = einsum('cl,ijlm->cmji', t1, tmps_["43_oooo"])
    del tmps_["43_oooo"]

    # tmps_[106_vvvooo](a,c,b,j,i,k) = 1.00 t2(b,c,k,m) * t1(a,l) * eri[oovo](m,l,d,i) * t1(d,j) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["106_vvvooo"] = einsum('bckm,amij->acbjik', t2, tmps_["100_vooo"])
    triples_resid -= einsum('cbajik->abcijk', tmps_["106_vvvooo"])

    # triples_resid += +1.00 P(j,k) <m,l||d,i> t1(d,k) t1(c,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakij->abcijk', tmps_["106_vvvooo"])
    triples_resid -= einsum('acbjik->abcijk', tmps_["106_vvvooo"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,i> t1(d,k) t1(a,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbkij->abcijk', tmps_["106_vvvooo"])

    # triples_resid += -1.00 P(i,k) <m,l||d,j> t1(d,k) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakji->abcijk', tmps_["106_vvvooo"])
    triples_resid += einsum('bcakji->abcijk', tmps_["106_vvvooo"])
    triples_resid += einsum('acbijk->abcijk', tmps_["106_vvvooo"])

    # triples_resid += -1.00 P(i,k) P(a,b) <m,l||d,j> t1(d,k) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbkji->abcijk', tmps_["106_vvvooo"])
    triples_resid -= einsum('bcakij->abcijk', tmps_["106_vvvooo"])
    triples_resid += einsum('cbaijk->abcijk', tmps_["106_vvvooo"])
    triples_resid -= einsum('bcaijk->abcijk', tmps_["106_vvvooo"])
    triples_resid -= einsum('cbaikj->abcijk', tmps_["106_vvvooo"])
    triples_resid += einsum('bcaikj->abcijk', tmps_["106_vvvooo"])
    triples_resid -= einsum('acbikj->abcijk', tmps_["106_vvvooo"])

    # triples_resid += +1.00 P(i,j) <m,l||d,k> t1(d,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbajki->abcijk', tmps_["106_vvvooo"])
    triples_resid -= einsum('bcajki->abcijk', tmps_["106_vvvooo"])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,k> t1(d,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbjki->abcijk', tmps_["106_vvvooo"])
    triples_resid += einsum('bcajik->abcijk', tmps_["106_vvvooo"])
    del tmps_["106_vvvooo"]

    # tmps_[122_vvoo](a,b,j,i) = 1.00 t1(b,l) * t1(a,k) * eri[oovo](l,k,c,i) * t1(c,j) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["122_vvoo"] = einsum('bl,alij->abji', t1, tmps_["100_vooo"])
    del tmps_["100_vooo"]

    # doubles_resid += -1.00 P(i,j) <l,k||c,j> t1(c,i) t1(a,k) t1(b,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abij->abij', tmps_["122_vvoo"])
    doubles_resid += einsum('abji->abij', tmps_["122_vvoo"])
    del tmps_["122_vvoo"]

    # tmps_[44_vooo](c,k,i,m) = 1.00 eri[oooo](m,l,i,k) * t1(c,l) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["44_vooo"] = einsum('mlik,cl->ckim', eri["oooo"], t1)

    # doubles_resid += -1.00 <l,k||i,j> t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('ajil,bl->abij', tmps_["44_vooo"], t1)

    # tmps_[72_vvvooo](c,b,a,k,i,j) = 1.00 eri[oooo](m,l,i,j) * t1(a,l) * t2(b,c,k,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["72_vvvooo"] = einsum('ajim,bckm->cbakij', tmps_["44_vooo"], t2)
    del tmps_["44_vooo"]

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||j,k> t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaijk->abcijk', tmps_["72_vvvooo"])

    # triples_resid += +1.00 P(a,b) <m,l||i,j> t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakij->abcijk', tmps_["72_vvvooo"])
    triples_resid -= einsum('bacjik->abcijk', tmps_["72_vvvooo"])

    # triples_resid += +1.00 P(i,j) <m,l||j,k> t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bacijk->abcijk', tmps_["72_vvvooo"])
    triples_resid -= einsum('cabkij->abcijk', tmps_["72_vvvooo"])
    triples_resid -= einsum('cabijk->abcijk', tmps_["72_vvvooo"])
    triples_resid -= einsum('cbajik->abcijk', tmps_["72_vvvooo"])
    triples_resid += einsum('cabjik->abcijk', tmps_["72_vvvooo"])

    # triples_resid += +1.00 <m,l||i,j> t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('backij->abcijk', tmps_["72_vvvooo"])
    del tmps_["72_vvvooo"]

    # tmps_[45_vv](d,a) = 1.00 eri[vovv](a,k,c,d) * t1(c,k) // flops: o0v2 = o1v3 | mem: o0v2 = o0v2
    tmps_["45_vv"] = einsum('akcd,ck->da', eri["vovv"], t1)

    # tmps_[60_vvvooo](c,b,a,k,j,i) = 1.00 t3(e,a,b,i,j,k) * eri[vovv](c,l,d,e) * t1(d,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["60_vvvooo"] = einsum('eabijk,ec->cbakji', t3, tmps_["45_vv"])
    triples_resid += einsum('bcakji->abcijk', tmps_["60_vvvooo"])

    # triples_resid += +1.00 <l,c||d,e> t1(d,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakji->abcijk', tmps_["60_vvvooo"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t1(d,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('acbkji->abcijk', tmps_["60_vvvooo"])
    del tmps_["60_vvvooo"]

    # tmps_[88_vvoo](a,b,j,i) = 1.00 t2(d,b,i,j) * eri[vovv](a,k,c,d) * t1(c,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["88_vvoo"] = einsum('dbij,da->abji', t2, tmps_["45_vv"])
    del tmps_["45_vv"]

    # doubles_resid += +1.00 P(a,b) <k,a||c,d> t1(c,k) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('abji->abij', tmps_["88_vvoo"])
    doubles_resid += einsum('baji->abij', tmps_["88_vvoo"])
    del tmps_["88_vvoo"]

    # tmps_[46_vo](d,l) = 1.00 eri[oovv](l,k,c,d) * t1(c,k) // flops: o1v1 = o2v2 | mem: o1v1 = o1v1
    tmps_["46_vo"] = einsum('lkcd,ck->dl', eri["oovv"], t1)

    # doubles_resid += -1.00 <l,k||c,d> t1(c,k) t3(d,a,b,i,j,l)  // flops: o2v2 += o3v3 | mem: o2v2 += o2v2
    doubles_resid -= einsum('dl,dabijl->abij', tmps_["46_vo"], t3)

    # tmps_[74_vvoooo](c,b,k,j,i,m) = 1.00 eri[oovv](m,l,d,e) * t1(d,l) * t3(e,b,c,i,j,k) // flops: o4v2 = o4v3 | mem: o4v2 = o4v2
    tmps_["74_vvoooo"] = einsum('em,ebcijk->cbkjim', tmps_["46_vo"], t3)

    # tmps_[118_vvvooo](a,c,b,i,j,k) = 1.00 t1(b,m) * eri[oovv](m,l,d,e) * t1(d,l) * t3(e,a,c,i,j,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["118_vvvooo"] = einsum('bm,cakjim->acbijk', t1, tmps_["74_vvoooo"])
    del tmps_["74_vvoooo"]

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t1(c,m) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('abcijk->abcijk', tmps_["118_vvvooo"])
    triples_resid -= einsum('acbijk->abcijk', tmps_["118_vvvooo"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,l) t1(a,m) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('bcaijk->abcijk', tmps_["118_vvvooo"])
    del tmps_["118_vvvooo"]

    # tmps_[90_vooo](a,j,i,m) = 1.00 eri[oovv](m,l,d,e) * t1(d,l) * t2(e,a,i,j) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["90_vooo"] = einsum('em,eaij->ajim', tmps_["46_vo"], t2)
    del tmps_["46_vo"]

    # tmps_[110_vvvooo](b,c,a,i,k,j) = 1.00 t2(a,c,j,m) * eri[oovv](m,l,d,e) * t1(d,l) * t2(e,b,i,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["110_vvvooo"] = einsum('acjm,bkim->bcaikj', t2, tmps_["90_vooo"])
    triples_resid -= einsum('cbaikj->abcijk', tmps_["110_vvvooo"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,l) t2(e,a,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbijk->abcijk', tmps_["110_vvvooo"])
    triples_resid += einsum('bcaikj->abcijk', tmps_["110_vvvooo"])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,l) t2(e,a,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('acbjki->abcijk', tmps_["110_vvvooo"])
    triples_resid -= einsum('bcaijk->abcijk', tmps_["110_vvvooo"])
    triples_resid -= einsum('acbikj->abcijk', tmps_["110_vvvooo"])
    triples_resid -= einsum('bcajki->abcijk', tmps_["110_vvvooo"])

    # triples_resid += +1.00 P(i,j) <m,l||d,e> t1(d,l) t2(e,c,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbajki->abcijk', tmps_["110_vvvooo"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t2(e,c,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbaijk->abcijk', tmps_["110_vvvooo"])
    del tmps_["110_vvvooo"]

    # tmps_[124_vvoo](b,a,i,j) = 1.00 eri[oovv](l,k,c,d) * t1(c,k) * t2(d,a,i,j) * t1(b,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["124_vvoo"] = einsum('ajil,bl->baij', tmps_["90_vooo"], t1)
    del tmps_["90_vooo"]
    doubles_resid -= einsum('baij->abij', tmps_["124_vvoo"])

    # doubles_resid += +1.00 P(a,b) <l,k||c,d> t1(c,k) t1(a,l) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('abij->abij', tmps_["124_vvoo"])
    del tmps_["124_vvoo"]

    # tmps_[47_oo](i,l) = 1.00 eri[oovo](l,k,c,i) * t1(c,k) // flops: o2v0 = o3v1 | mem: o2v0 = o2v0
    tmps_["47_oo"] = einsum('lkci,ck->il', eri["oovo"], t1)

    # singles_resid += +1.00 <k,j||b,i> t1(b,j) t1(a,k)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid += einsum('ik,ak->ai', tmps_["47_oo"], t1)

    # tmps_[78_vvvooo](c,b,a,k,j,i) = 1.00 eri[oovo](m,l,d,i) * t1(d,l) * t3(a,b,c,j,k,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["78_vvvooo"] = einsum('im,abcjkm->cbakji', tmps_["47_oo"], t3)

    # triples_resid += +1.00 P(j,k) <m,l||d,k> t1(d,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbajik->abcijk', tmps_["78_vvvooo"])
    triples_resid -= einsum('cbakij->abcijk', tmps_["78_vvvooo"])

    # triples_resid += +1.00 <m,l||d,i> t1(d,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += einsum('cbakji->abcijk', tmps_["78_vvvooo"])
    del tmps_["78_vvvooo"]

    # tmps_[99_vvoo](b,a,j,i) = 1.00 t2(a,b,i,l) * eri[oovo](l,k,c,j) * t1(c,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["99_vvoo"] = einsum('abil,jl->baji', t2, tmps_["47_oo"])
    del tmps_["47_oo"]
    doubles_resid -= einsum('baij->abij', tmps_["99_vvoo"])

    # doubles_resid += +1.00 P(i,j) <l,k||c,j> t1(c,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += einsum('baji->abij', tmps_["99_vvoo"])
    del tmps_["99_vvoo"]

    # tmps_[48_oo](j,k) = 1.00 f[ov](k,c) * t1(c,j) // flops: o2v0 = o2v1 | mem: o2v0 = o2v0
    tmps_["48_oo"] = einsum('kc,cj->jk', f["ov"], t1)

    # singles_resid += -1.00 f(j,b) t1(b,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid -= einsum('ij,aj->ai', tmps_["48_oo"], t1)

    # tmps_[77_vvvooo](c,b,a,k,i,j) = 1.00 f[ov](l,d) * t1(d,j) * t3(a,b,c,i,k,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["77_vvvooo"] = einsum('jl,abcikl->cbakij', tmps_["48_oo"], t3)
    triples_resid += einsum('cbakij->abcijk', tmps_["77_vvvooo"])

    # triples_resid += -1.00 f(l,d) t1(d,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbakji->abcijk', tmps_["77_vvvooo"])

    # triples_resid += -1.00 P(j,k) f(l,d) t1(d,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= einsum('cbajik->abcijk', tmps_["77_vvvooo"])
    del tmps_["77_vvvooo"]

    # tmps_[97_vvoo](b,a,i,j) = 1.00 f[ov](k,c) * t1(c,j) * t2(a,b,i,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["97_vvoo"] = einsum('jk,abik->baij', tmps_["48_oo"], t2)
    del tmps_["48_oo"]

    # doubles_resid += -1.00 P(i,j) f(k,c) t1(c,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= einsum('baij->abij', tmps_["97_vvoo"])
    doubles_resid += einsum('baji->abij', tmps_["97_vvoo"])
    del tmps_["97_vvoo"]

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



