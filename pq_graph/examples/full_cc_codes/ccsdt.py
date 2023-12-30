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

    ##########  Evaluate Equations  ##########

    # doubles_resid = +1.00 f(k,c) t3(c,a,b,i,j,k)  // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    doubles_resid = np.einsum('kc,cabijk->abij', f["ov"], t3)

    # doubles_resid += +1.00 <a,b||i,j>  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += np.einsum('abij->abij', eri["vvoo"])

    # doubles_resid += +0.50 <l,k||i,j> t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
    doubles_resid += 0.50 * np.einsum('lkij,ablk->abij', eri["oooo"], t2)

    # doubles_resid += +0.50 <a,b||c,d> t2(c,d,i,j)  // flops: o2v2 += o2v4 | mem: o2v2 += o2v2
    doubles_resid += 0.50 * np.einsum('abcd,cdij->abij', eri["vvvv"], t2)

    # doubles_resid += -0.50 <l,k||c,d> t2(c,a,i,j) t2(d,b,l,k)  // flops: o2v2 += o2v3 o2v3 | mem: o2v2 += o0v2 o2v2
    doubles_resid -= 0.50 * np.einsum('lkcd,dblk,caij->abij', eri["oovv"], t2, t2, optimize=['einsum_path',(0,1),(0,1)])

    # singles_resid = +1.00 f(a,i)  // flops: o1v1 = o1v1 | mem: o1v1 = o1v1
    singles_resid = 1.00 * np.einsum('ai->ai', f["vo"])

    # singles_resid += -1.00 f(j,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid -= np.einsum('ji,aj->ai', f["oo"], t1)

    # singles_resid += +1.00 f(a,b) t1(b,i)  // flops: o1v1 += o1v2 | mem: o1v1 += o1v1
    singles_resid += np.einsum('ab,bi->ai', f["vv"], t1)

    # singles_resid += -1.00 f(j,b) t2(b,a,i,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid -= np.einsum('jb,baij->ai', f["ov"], t2)

    # singles_resid += +1.00 <j,a||b,i> t1(b,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid -= np.einsum('ajbi,bj->ai', eri["vovo"], t1)

    # singles_resid += -0.50 <k,j||b,i> t2(b,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
    singles_resid -= 0.50 * np.einsum('kjbi,bakj->ai', eri["oovo"], t2)

    # singles_resid += -0.50 <j,a||b,c> t2(b,c,i,j)  // flops: o1v1 += o2v3 | mem: o1v1 += o1v1
    singles_resid += 0.50 * np.einsum('ajbc,bcij->ai', eri["vovv"], t2)

    # singles_resid += +0.25 <k,j||b,c> t3(b,c,a,i,k,j)  // flops: o1v1 += o3v3 | mem: o1v1 += o1v1
    singles_resid += 0.25 * np.einsum('kjbc,bcaikj->ai', eri["oovv"], t3)

    # singles_resid += +0.50 <k,j||b,c> t1(a,j) t2(b,c,i,k)  // flops: o1v1 += o3v2 o2v1 | mem: o1v1 += o2v0 o1v1
    singles_resid += 0.50 * np.einsum('kjbc,bcik,aj->ai', eri["oovv"], t2, t1, optimize=['einsum_path',(0,1),(0,1)])

    # singles_resid += +1.00 <j,a||b,c> t1(b,j) t1(c,i)  // flops: o1v1 += o2v3 o2v2 | mem: o1v1 += o2v2 o1v1
    singles_resid -= np.einsum('ajbc,ci,bj->ai', eri["vovv"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])

    # tmps_[vvvooo_1](a,c,b,i,j,k) = 0.50 eri[vvvv](a,c,d,e) * t3(d,e,b,i,j,k) // flops: o3v3 = o3v5 | mem: o3v3 = o3v3
    tmps_["vvvooo_1"] = 0.50 * np.einsum('acde,debijk->acbijk', eri["vvvv"], t3)

    # triples_resid = +0.50 P(b,c) <a,b||d,e> t3(d,e,c,i,j,k)  // flops: o3v3 = o3v3 | mem: o3v3 = o3v3
    triples_resid = 1.00 * np.einsum('abcijk->abcijk', tmps_["vvvooo_1"])
    triples_resid -= np.einsum('acbijk->abcijk', tmps_["vvvooo_1"])

    # triples_resid += +0.50 <b,c||d,e> t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcaijk->abcijk', tmps_["vvvooo_1"])
    del tmps_["vvvooo_1"]

    # tmps_[vvvooo_2](c,a,b,i,j,k) = 1.00 eri[vovo](c,l,d,i) * t3(d,a,b,j,k,l) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
    tmps_["vvvooo_2"] = np.einsum('cldi,dabjkl->cabijk', eri["vovo"], t3)

    # triples_resid += +1.00 P(j,k) P(a,b) <l,a||d,k> t3(d,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abckij->abcijk', tmps_["vvvooo_2"])
    triples_resid += np.einsum('abcjik->abcijk', tmps_["vvvooo_2"])
    triples_resid += np.einsum('backij->abcijk', tmps_["vvvooo_2"])
    triples_resid -= np.einsum('bacjik->abcijk', tmps_["vvvooo_2"])

    # triples_resid += +1.00 P(a,b) <l,a||d,i> t3(d,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_2"])
    triples_resid += np.einsum('bacijk->abcijk', tmps_["vvvooo_2"])

    # triples_resid += +1.00 P(j,k) <l,c||d,k> t3(d,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cabkij->abcijk', tmps_["vvvooo_2"])
    triples_resid += np.einsum('cabjik->abcijk', tmps_["vvvooo_2"])

    # triples_resid += +1.00 <l,c||d,i> t3(d,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cabijk->abcijk', tmps_["vvvooo_2"])
    del tmps_["vvvooo_2"]

    # tmps_[vooooo_5](c,m,l,i,j,k) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,c,i,j,k) // flops: o5v1 = o5v3 | mem: o5v1 = o5v1
    tmps_["vooooo_5"] = 0.50 * np.einsum('mlde,decijk->cmlijk', eri["oovv"], t3)

    # tmps_[vvvooo_53](c,a,b,i,j,k) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,c,i,j,k) * t2(a,b,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_53"] = 0.50 * np.einsum('cmlijk,abml->cabijk', tmps_["vooooo_5"], t2)

    # triples_resid += +0.25 P(b,c) <m,l||d,e> t2(a,b,m,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabijk->abcijk', tmps_["vvvooo_53"])
    triples_resid -= np.einsum('bacijk->abcijk', tmps_["vvvooo_53"])

    # triples_resid += +0.25 <m,l||d,e> t2(b,c,m,l) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_53"])
    del tmps_["vvvooo_53"]

    # tmps_[vvvooo_117](a,c,b,i,j,k) = 1.00 t1(a,l) * eri[oovv](m,l,d,e) * t3(d,e,c,i,j,k) * t1(b,m) // flops: o3v3 = o5v2 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["vvvooo_117"] = np.einsum('al,cmlijk,bm->acbijk', t1, tmps_["vooooo_5"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["vooooo_5"]

    # triples_resid += -0.50 P(b,c) <m,l||d,e> t1(a,l) t1(b,m) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('acbijk->abcijk', tmps_["vvvooo_117"])
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_117"])

    # triples_resid += -0.50 <m,l||d,e> t1(b,l) t1(c,m) t3(d,e,a,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bacijk->abcijk', tmps_["vvvooo_117"])
    del tmps_["vvvooo_117"]

    # tmps_[vvvooo_6](a,b,c,j,k,i) = 0.50 eri[oooo](m,l,j,k) * t3(a,b,c,i,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_6"] = 0.50 * np.einsum('mljk,abciml->abcjki', eri["oooo"], t3)

    # triples_resid += +0.50 P(i,j) <m,l||j,k> t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcjki->abcijk', tmps_["vvvooo_6"])
    triples_resid -= np.einsum('abcikj->abcijk', tmps_["vvvooo_6"])

    # triples_resid += +0.50 <m,l||i,j> t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_6"])
    del tmps_["vvvooo_6"]

    # tmps_[vvvo_7](d,a,b,i) = 0.50 eri[oovv](m,l,d,e) * t3(e,a,b,i,m,l) // flops: o1v3 = o3v4 | mem: o1v3 = o1v3
    tmps_["vvvo_7"] = 0.50 * np.einsum('mlde,eabiml->dabi', eri["oovv"], t3)

    # tmps_[vvvooo_58](a,b,c,j,k,i) = 1.00 t2(d,a,j,k) * eri[oovv](m,l,d,e) * t3(e,b,c,i,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvvooo_58"] = np.einsum('dajk,dbci->abcjki', t2, tmps_["vvvo_7"])

    # triples_resid += -0.50 P(i,j) P(a,b) <m,l||d,e> t2(d,a,j,k) t3(e,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcjki->abcijk', tmps_["vvvooo_58"])
    triples_resid += np.einsum('abcikj->abcijk', tmps_["vvvooo_58"])
    triples_resid += np.einsum('bacjki->abcijk', tmps_["vvvooo_58"])
    triples_resid -= np.einsum('bacikj->abcijk', tmps_["vvvooo_58"])

    # triples_resid += -0.50 P(a,b) <m,l||d,e> t2(d,a,i,j) t3(e,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_58"])
    triples_resid += np.einsum('bacijk->abcijk', tmps_["vvvooo_58"])

    # triples_resid += -0.50 P(i,j) <m,l||d,e> t2(d,c,j,k) t3(e,a,b,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cabjki->abcijk', tmps_["vvvooo_58"])
    triples_resid += np.einsum('cabikj->abcijk', tmps_["vvvooo_58"])

    # triples_resid += -0.50 <m,l||d,e> t2(d,c,i,j) t3(e,a,b,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cabijk->abcijk', tmps_["vvvooo_58"])
    del tmps_["vvvooo_58"]

    # tmps_[vvoo_88](a,b,j,i) = 1.00 t1(c,j) * eri[oovv](l,k,c,d) * t3(d,a,b,i,l,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["vvoo_88"] = np.einsum('cj,cabi->abji', t1, tmps_["vvvo_7"])
    del tmps_["vvvo_7"]

    # doubles_resid += -0.50 P(i,j) <l,k||c,d> t1(c,j) t3(d,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('abji->abij', tmps_["vvoo_88"])
    doubles_resid += np.einsum('abij->abij', tmps_["vvoo_88"])
    del tmps_["vvoo_88"]

    # tmps_[vvvooo_8](a,b,c,k,i,j) = 1.00 eri[vvvo](a,b,d,k) * t2(d,c,i,j) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvvooo_8"] = np.einsum('abdk,dcij->abckij', eri["vvvo"], t2)

    # triples_resid += -1.00 P(j,k) P(b,c) <a,b||d,k> t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abckij->abcijk', tmps_["vvvooo_8"])
    triples_resid += np.einsum('abcjik->abcijk', tmps_["vvvooo_8"])
    triples_resid += np.einsum('acbkij->abcijk', tmps_["vvvooo_8"])
    triples_resid -= np.einsum('acbjik->abcijk', tmps_["vvvooo_8"])

    # triples_resid += -1.00 P(b,c) <a,b||d,i> t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_8"])
    triples_resid += np.einsum('acbijk->abcijk', tmps_["vvvooo_8"])

    # triples_resid += -1.00 P(j,k) <b,c||d,k> t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcakij->abcijk', tmps_["vvvooo_8"])
    triples_resid += np.einsum('bcajik->abcijk', tmps_["vvvooo_8"])

    # triples_resid += -1.00 <b,c||d,i> t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcaijk->abcijk', tmps_["vvvooo_8"])
    del tmps_["vvvooo_8"]

    # tmps_[vvvooo_9](a,b,c,i,j,k) = 1.00 f[vv](a,d) * t3(d,b,c,i,j,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvvooo_9"] = np.einsum('ad,dbcijk->abcijk', f["vv"], t3)

    # triples_resid += +1.00 P(a,b) f(a,d) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_9"])
    triples_resid -= np.einsum('bacijk->abcijk', tmps_["vvvooo_9"])

    # triples_resid += +1.00 f(c,d) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabijk->abcijk', tmps_["vvvooo_9"])
    del tmps_["vvvooo_9"]

    # tmps_[vvoo_10](a,b,i,j) = 0.50 eri[vovv](a,k,c,d) * t3(c,d,b,i,j,k) // flops: o2v2 = o3v4 | mem: o2v2 = o2v2
    tmps_["vvoo_10"] = 0.50 * np.einsum('akcd,cdbijk->abij', eri["vovv"], t3)

    # doubles_resid += +0.50 P(a,b) <k,a||c,d> t3(c,d,b,i,j,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('abij->abij', tmps_["vvoo_10"])
    doubles_resid += np.einsum('baij->abij', tmps_["vvoo_10"])
    del tmps_["vvoo_10"]

    # tmps_[vooo_12](c,l,i,j) = 0.50 eri[oovv](m,l,d,e) * t3(d,e,c,i,j,m) // flops: o3v1 = o4v3 | mem: o3v1 = o3v1
    tmps_["vooo_12"] = 0.50 * np.einsum('mlde,decijm->clij', eri["oovv"], t3)

    # tmps_[vvvooo_69](a,b,c,k,i,j) = 1.00 t2(a,b,k,l) * eri[oovv](m,l,d,e) * t3(d,e,c,i,j,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_69"] = np.einsum('abkl,clij->abckij', t2, tmps_["vooo_12"])

    # triples_resid += -0.50 P(j,k) P(b,c) <m,l||d,e> t2(a,b,k,l) t3(d,e,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abckij->abcijk', tmps_["vvvooo_69"])
    triples_resid += np.einsum('abcjik->abcijk', tmps_["vvvooo_69"])
    triples_resid += np.einsum('acbkij->abcijk', tmps_["vvvooo_69"])
    triples_resid -= np.einsum('acbjik->abcijk', tmps_["vvvooo_69"])

    # triples_resid += -0.50 P(b,c) <m,l||d,e> t2(a,b,i,l) t3(d,e,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_69"])
    triples_resid += np.einsum('acbijk->abcijk', tmps_["vvvooo_69"])

    # triples_resid += -0.50 P(j,k) <m,l||d,e> t2(b,c,k,l) t3(d,e,a,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcakij->abcijk', tmps_["vvvooo_69"])
    triples_resid += np.einsum('bcajik->abcijk', tmps_["vvvooo_69"])

    # triples_resid += -0.50 <m,l||d,e> t2(b,c,i,l) t3(d,e,a,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcaijk->abcijk', tmps_["vvvooo_69"])
    del tmps_["vvvooo_69"]

    # tmps_[vvoo_99](b,a,i,j) = 1.00 eri[oovv](l,k,c,d) * t3(c,d,b,i,j,l) * t1(a,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_99"] = np.einsum('bkij,ak->baij', tmps_["vooo_12"], t1)
    del tmps_["vooo_12"]

    # doubles_resid += -0.50 P(a,b) <l,k||c,d> t1(a,k) t3(c,d,b,i,j,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('baij->abij', tmps_["vvoo_99"])
    doubles_resid += np.einsum('abij->abij', tmps_["vvoo_99"])
    del tmps_["vvoo_99"]

    # tmps_[vvvooo_13](c,a,b,j,k,i) = 1.00 eri[vooo](c,l,j,k) * t2(a,b,i,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_13"] = np.einsum('cljk,abil->cabjki', eri["vooo"], t2)

    # triples_resid += -1.00 P(i,j) P(a,b) <l,a||j,k> t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcjki->abcijk', tmps_["vvvooo_13"])
    triples_resid -= np.einsum('abcikj->abcijk', tmps_["vvvooo_13"])
    triples_resid -= np.einsum('bacjki->abcijk', tmps_["vvvooo_13"])
    triples_resid += np.einsum('bacikj->abcijk', tmps_["vvvooo_13"])

    # triples_resid += -1.00 P(a,b) <l,a||i,j> t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_13"])
    triples_resid -= np.einsum('bacijk->abcijk', tmps_["vvvooo_13"])

    # triples_resid += -1.00 P(i,j) <l,c||j,k> t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabjki->abcijk', tmps_["vvvooo_13"])
    triples_resid -= np.einsum('cabikj->abcijk', tmps_["vvvooo_13"])

    # triples_resid += -1.00 <l,c||i,j> t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabijk->abcijk', tmps_["vvvooo_13"])
    del tmps_["vvvooo_13"]

    # tmps_[vvvooo_15](a,b,c,k,i,j) = 1.00 f[oo](l,k) * t3(a,b,c,i,j,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_15"] = np.einsum('lk,abcijl->abckij', f["oo"], t3)

    # triples_resid += -1.00 P(j,k) f(l,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abckij->abcijk', tmps_["vvvooo_15"])
    triples_resid += np.einsum('abcjik->abcijk', tmps_["vvvooo_15"])

    # triples_resid += -1.00 f(l,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_15"])
    del tmps_["vvvooo_15"]

    # tmps_[vvoo_16](a,b,j,i) = 0.50 eri[oovo](l,k,c,j) * t3(c,a,b,i,l,k) // flops: o2v2 = o4v3 | mem: o2v2 = o2v2
    tmps_["vvoo_16"] = 0.50 * np.einsum('lkcj,cabilk->abji', eri["oovo"], t3)

    # doubles_resid += +0.50 P(i,j) <l,k||c,j> t3(c,a,b,i,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += np.einsum('abji->abij', tmps_["vvoo_16"])
    doubles_resid -= np.einsum('abij->abij', tmps_["vvoo_16"])
    del tmps_["vvoo_16"]

    # tmps_[vvvo_18](a,d,c,i) = 1.00 eri[vovv](a,l,d,e) * t2(e,c,i,l) // flops: o1v3 = o2v4 | mem: o1v3 = o1v3
    tmps_["vvvo_18"] = np.einsum('alde,ecil->adci', eri["vovv"], t2)

    # tmps_[vvvooo_59](b,a,c,j,k,i) = 1.00 t2(d,b,j,k) * eri[vovv](a,l,d,e) * t2(e,c,i,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvvooo_59"] = np.einsum('dbjk,adci->bacjki', t2, tmps_["vvvo_18"])

    # triples_resid += +1.00 P(i,j) P(a,b) <l,a||d,e> t2(d,b,j,k) t2(e,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bacjki->abcijk', tmps_["vvvooo_59"])
    triples_resid += np.einsum('bacikj->abcijk', tmps_["vvvooo_59"])
    triples_resid += np.einsum('abcjki->abcijk', tmps_["vvvooo_59"])
    triples_resid -= np.einsum('abcikj->abcijk', tmps_["vvvooo_59"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,j) t2(e,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bacijk->abcijk', tmps_["vvvooo_59"])
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_59"])

    # triples_resid += +1.00 P(i,j) <l,c||d,e> t2(d,a,j,k) t2(e,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('acbjki->abcijk', tmps_["vvvooo_59"])
    triples_resid += np.einsum('acbikj->abcijk', tmps_["vvvooo_59"])

    # triples_resid += +1.00 <l,c||d,e> t2(d,a,i,j) t2(e,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('acbijk->abcijk', tmps_["vvvooo_59"])
    del tmps_["vvvooo_59"]

    # tmps_[vvoo_85](a,b,j,i) = 1.00 eri[vovv](a,k,c,d) * t2(d,b,j,k) * t1(c,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["vvoo_85"] = np.einsum('acbj,ci->abji', tmps_["vvvo_18"], t1)
    del tmps_["vvvo_18"]

    # doubles_resid += -1.00 P(i,j) P(a,b) <k,a||c,d> t1(c,j) t2(d,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += np.einsum('abij->abij', tmps_["vvoo_85"])
    doubles_resid -= np.einsum('abji->abij', tmps_["vvoo_85"])
    doubles_resid -= np.einsum('baij->abij', tmps_["vvoo_85"])
    doubles_resid += np.einsum('baji->abij', tmps_["vvoo_85"])
    del tmps_["vvoo_85"]

    # tmps_[vvoo_20](d,b,l,i) = 1.00 eri[oovv](m,l,d,e) * t2(e,b,i,m) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["vvoo_20"] = np.einsum('mlde,ebim->dbli', eri["oovv"], t2)

    # singles_resid += +1.00 <k,j||b,c> t1(b,j) t2(c,a,i,k)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    singles_resid += np.einsum('baji,bj->ai', tmps_["vvoo_20"], t1)

    # tmps_[vvoo_83](b,a,j,i) = 1.00 eri[oovv](l,k,c,d) * t2(d,b,j,l) * t2(c,a,i,k) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["vvoo_83"] = np.einsum('cbkj,caik->baji', tmps_["vvoo_20"], t2)

    # doubles_resid += +1.00 P(i,j) <l,k||c,d> t2(c,a,j,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += np.einsum('baij->abij', tmps_["vvoo_83"])
    doubles_resid -= np.einsum('baji->abij', tmps_["vvoo_83"])
    del tmps_["vvoo_83"]

    # tmps_[vvvooo_113](a,b,c,j,k,i) = 1.00 t1(a,l) * t2(d,b,j,k) * eri[oovv](m,l,d,e) * t2(e,c,i,m) // flops: o3v3 = o3v3 o4v4 | mem: o3v3 = o3v3 o3v3
    tmps_["vvvooo_113"] = np.einsum('al,dbjk,dcli->abcjki', t1, t2, tmps_["vvoo_20"], optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,j,k) t2(e,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcjki->abcijk', tmps_["vvvooo_113"])
    triples_resid += np.einsum('abcikj->abcijk', tmps_["vvvooo_113"])
    triples_resid += np.einsum('bacjki->abcijk', tmps_["vvvooo_113"])
    triples_resid -= np.einsum('bacikj->abcijk', tmps_["vvvooo_113"])

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,j) t2(e,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_113"])
    triples_resid += np.einsum('bacijk->abcijk', tmps_["vvvooo_113"])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(c,l) t2(d,a,j,k) t2(e,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cabjki->abcijk', tmps_["vvvooo_113"])
    triples_resid += np.einsum('cabikj->abcijk', tmps_["vvvooo_113"])

    # triples_resid += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,j) t2(e,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cabijk->abcijk', tmps_["vvvooo_113"])
    del tmps_["vvvooo_113"]

    # tmps_[vvoo_120](a,b,i,j) = 1.00 t1(a,k) * eri[oovv](l,k,c,d) * t2(d,b,i,l) * t1(c,j) // flops: o2v2 = o2v3 o2v3 | mem: o2v2 = o1v3 o2v2
    tmps_["vvoo_120"] = np.einsum('ak,cbki,cj->abij', t1, tmps_["vvoo_20"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["vvoo_20"]

    # doubles_resid += +1.00 P(i,j) P(a,b) <l,k||c,d> t1(c,j) t1(a,k) t2(d,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += np.einsum('abij->abij', tmps_["vvoo_120"])
    doubles_resid -= np.einsum('abji->abij', tmps_["vvoo_120"])
    doubles_resid -= np.einsum('baij->abij', tmps_["vvoo_120"])
    doubles_resid += np.einsum('baji->abij', tmps_["vvoo_120"])
    del tmps_["vvoo_120"]

    # tmps_[vooo_21](c,l,i,j) = 0.50 eri[vovv](c,l,d,e) * t2(d,e,i,j) // flops: o3v1 = o3v3 | mem: o3v1 = o3v1
    tmps_["vooo_21"] = 0.50 * np.einsum('clde,deij->clij', eri["vovv"], t2)

    # tmps_[vvvooo_70](a,b,c,j,k,i) = 1.00 eri[vovv](a,l,d,e) * t2(d,e,j,k) * t2(b,c,i,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_70"] = np.einsum('aljk,bcil->abcjki', tmps_["vooo_21"], t2)

    # triples_resid += -0.50 P(i,j) P(a,b) <l,a||d,e> t2(d,e,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcjki->abcijk', tmps_["vvvooo_70"])
    triples_resid -= np.einsum('abcikj->abcijk', tmps_["vvvooo_70"])
    triples_resid -= np.einsum('bacjki->abcijk', tmps_["vvvooo_70"])
    triples_resid += np.einsum('bacikj->abcijk', tmps_["vvvooo_70"])

    # triples_resid += -0.50 P(a,b) <l,a||d,e> t2(d,e,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_70"])
    triples_resid -= np.einsum('bacijk->abcijk', tmps_["vvvooo_70"])

    # triples_resid += -0.50 P(i,j) <l,c||d,e> t2(d,e,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabjki->abcijk', tmps_["vvvooo_70"])
    triples_resid -= np.einsum('cabikj->abcijk', tmps_["vvvooo_70"])

    # triples_resid += -0.50 <l,c||d,e> t2(d,e,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabijk->abcijk', tmps_["vvvooo_70"])
    del tmps_["vvvooo_70"]

    # tmps_[vvoo_94](a,b,i,j) = 1.00 eri[vovv](a,k,c,d) * t2(c,d,i,j) * t1(b,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_94"] = np.einsum('akij,bk->abij', tmps_["vooo_21"], t1)
    del tmps_["vooo_21"]

    # doubles_resid += +0.50 P(a,b) <k,a||c,d> t1(b,k) t2(c,d,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('abij->abij', tmps_["vvoo_94"])
    doubles_resid += np.einsum('baij->abij', tmps_["vvoo_94"])
    del tmps_["vvoo_94"]

    # tmps_[vvvo_23](d,b,c,i) = 0.50 eri[oovo](m,l,d,i) * t2(b,c,m,l) // flops: o1v3 = o3v3 | mem: o1v3 = o1v3
    tmps_["vvvo_23"] = 0.50 * np.einsum('mldi,bcml->dbci', eri["oovo"], t2)

    # tmps_[vvvooo_57](c,a,b,i,j,k) = 1.00 t2(d,c,i,j) * eri[oovo](m,l,d,k) * t2(a,b,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvvooo_57"] = np.einsum('dcij,dabk->cabijk', t2, tmps_["vvvo_23"])

    # triples_resid += -0.50 P(j,k) P(a,b) <m,l||d,k> t2(d,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_57"])
    triples_resid += np.einsum('abcikj->abcijk', tmps_["vvvooo_57"])
    triples_resid += np.einsum('bacijk->abcijk', tmps_["vvvooo_57"])
    triples_resid -= np.einsum('bacikj->abcijk', tmps_["vvvooo_57"])

    # triples_resid += -0.50 P(j,k) <m,l||d,k> t2(d,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cabijk->abcijk', tmps_["vvvooo_57"])
    triples_resid += np.einsum('cabikj->abcijk', tmps_["vvvooo_57"])

    # triples_resid += -0.50 P(a,b) <m,l||d,i> t2(d,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcjki->abcijk', tmps_["vvvooo_57"])
    triples_resid += np.einsum('bacjki->abcijk', tmps_["vvvooo_57"])

    # triples_resid += -0.50 <m,l||d,i> t2(d,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cabjki->abcijk', tmps_["vvvooo_57"])
    del tmps_["vvvooo_57"]

    # tmps_[vvoo_86](a,b,i,j) = 1.00 t1(c,i) * eri[oovo](l,k,c,j) * t2(a,b,l,k) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["vvoo_86"] = np.einsum('ci,cabj->abij', t1, tmps_["vvvo_23"])
    del tmps_["vvvo_23"]

    # doubles_resid += +0.50 P(i,j) <l,k||c,j> t1(c,i) t2(a,b,l,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += np.einsum('abij->abij', tmps_["vvoo_86"])
    doubles_resid -= np.einsum('abji->abij', tmps_["vvoo_86"])
    del tmps_["vvoo_86"]

    # tmps_[vvoo_25](a,b,j,i) = 1.00 eri[vovo](a,k,c,j) * t2(c,b,i,k) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["vvoo_25"] = np.einsum('akcj,cbik->abji', eri["vovo"], t2)

    # doubles_resid += +1.00 P(i,j) P(a,b) <k,a||c,j> t2(c,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('abji->abij', tmps_["vvoo_25"])
    doubles_resid += np.einsum('abij->abij', tmps_["vvoo_25"])
    doubles_resid += np.einsum('baji->abij', tmps_["vvoo_25"])
    doubles_resid -= np.einsum('baij->abij', tmps_["vvoo_25"])
    del tmps_["vvoo_25"]

    # tmps_[oooo_27](m,l,i,k) = 0.50 eri[oovv](m,l,d,e) * t2(d,e,i,k) // flops: o4v0 = o4v2 | mem: o4v0 = o4v0
    tmps_["oooo_27"] = 0.50 * np.einsum('mlde,deik->mlik', eri["oovv"], t2)

    # doubles_resid += +0.25 <l,k||c,d> t2(c,d,i,j) t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
    doubles_resid += 0.50 * np.einsum('lkij,ablk->abij', tmps_["oooo_27"], t2)

    # tmps_[vvvooo_54](a,b,c,i,j,k) = 0.50 t3(a,b,c,i,m,l) * eri[oovv](m,l,d,e) * t2(d,e,j,k) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_54"] = 0.50 * np.einsum('abciml,mljk->abcijk', t3, tmps_["oooo_27"])

    # triples_resid += +0.25 P(i,j) <m,l||d,e> t2(d,e,j,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_54"])
    triples_resid -= np.einsum('abcjik->abcijk', tmps_["vvvooo_54"])

    # triples_resid += +0.25 <m,l||d,e> t2(d,e,i,j) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abckij->abcijk', tmps_["vvvooo_54"])
    del tmps_["vvvooo_54"]

    # tmps_[vooo_102](c,m,i,k) = 1.00 eri[oovv](m,l,d,e) * t2(d,e,i,k) * t1(c,l) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["vooo_102"] = np.einsum('mlik,cl->cmik', tmps_["oooo_27"], t1)
    del tmps_["oooo_27"]

    # doubles_resid += -0.50 <l,k||c,d> t1(a,k) t1(b,l) t2(c,d,i,j)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('alij,bl->abij', tmps_["vooo_102"], t1)

    # tmps_[vvvooo_109](b,c,a,j,i,k) = 1.00 t2(b,c,j,m) * eri[oovv](m,l,d,e) * t2(d,e,i,k) * t1(a,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_109"] = np.einsum('bcjm,amik->bcajik', t2, tmps_["vooo_102"])
    del tmps_["vooo_102"]

    # triples_resid += +0.50 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,e,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcaijk->abcijk', tmps_["vvvooo_109"])
    triples_resid -= np.einsum('bcajik->abcijk', tmps_["vvvooo_109"])
    triples_resid -= np.einsum('acbijk->abcijk', tmps_["vvvooo_109"])
    triples_resid += np.einsum('acbjik->abcijk', tmps_["vvvooo_109"])

    # triples_resid += +0.50 P(a,b) <m,l||d,e> t1(a,l) t2(d,e,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcakij->abcijk', tmps_["vvvooo_109"])
    triples_resid -= np.einsum('acbkij->abcijk', tmps_["vvvooo_109"])

    # triples_resid += +0.50 P(i,j) <m,l||d,e> t1(c,l) t2(d,e,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_109"])
    triples_resid -= np.einsum('abcjik->abcijk', tmps_["vvvooo_109"])

    # triples_resid += +0.50 <m,l||d,e> t1(c,l) t2(d,e,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abckij->abcijk', tmps_["vvvooo_109"])
    del tmps_["vvvooo_109"]

    # tmps_[vvvo_29](a,c,e,i) = 1.00 eri[vvvv](a,c,d,e) * t1(d,i) // flops: o1v3 = o1v4 | mem: o1v3 = o1v3
    tmps_["vvvo_29"] = np.einsum('acde,di->acei', eri["vvvv"], t1)

    # doubles_resid += -1.00 <a,b||c,d> t1(c,j) t1(d,i)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('abdj,di->abij', tmps_["vvvo_29"], t1)

    # tmps_[vvvooo_56](c,a,b,i,j,k) = 1.00 t2(e,c,i,j) * eri[vvvv](a,b,d,e) * t1(d,k) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvvooo_56"] = np.einsum('ecij,abek->cabijk', t2, tmps_["vvvo_29"])
    del tmps_["vvvo_29"]

    # triples_resid += +1.00 P(j,k) P(b,c) <a,b||d,e> t1(d,k) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabijk->abcijk', tmps_["vvvooo_56"])
    triples_resid -= np.einsum('cabikj->abcijk', tmps_["vvvooo_56"])
    triples_resid -= np.einsum('bacijk->abcijk', tmps_["vvvooo_56"])
    triples_resid += np.einsum('bacikj->abcijk', tmps_["vvvooo_56"])

    # triples_resid += +1.00 P(b,c) <a,b||d,e> t1(d,i) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabjki->abcijk', tmps_["vvvooo_56"])
    triples_resid -= np.einsum('bacjki->abcijk', tmps_["vvvooo_56"])

    # triples_resid += +1.00 P(j,k) <b,c||d,e> t1(d,k) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_56"])
    triples_resid -= np.einsum('abcikj->abcijk', tmps_["vvvooo_56"])

    # triples_resid += +1.00 <b,c||d,e> t1(d,i) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcjki->abcijk', tmps_["vvvooo_56"])
    del tmps_["vvvooo_56"]

    # tmps_[vvoo_30](c,e,l,i) = 1.00 eri[vovv](c,l,d,e) * t1(d,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["vvoo_30"] = np.einsum('clde,di->celi', eri["vovv"], t1)

    # tmps_[vvvooo_49](a,b,c,j,k,i) = 1.00 t3(e,a,b,j,k,l) * eri[vovv](c,l,d,e) * t1(d,i) // flops: o3v3 = o4v4 | mem: o3v3 = o3v3
    tmps_["vvvooo_49"] = np.einsum('eabjkl,celi->abcjki', t3, tmps_["vvoo_30"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,a||d,e> t1(d,k) t3(e,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcaijk->abcijk', tmps_["vvvooo_49"])
    triples_resid -= np.einsum('bcaikj->abcijk', tmps_["vvvooo_49"])
    triples_resid -= np.einsum('acbijk->abcijk', tmps_["vvvooo_49"])
    triples_resid += np.einsum('acbikj->abcijk', tmps_["vvvooo_49"])

    # triples_resid += -1.00 P(a,b) <l,a||d,e> t1(d,i) t3(e,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcajki->abcijk', tmps_["vvvooo_49"])
    triples_resid -= np.einsum('acbjki->abcijk', tmps_["vvvooo_49"])

    # triples_resid += -1.00 P(j,k) <l,c||d,e> t1(d,k) t3(e,a,b,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_49"])
    triples_resid -= np.einsum('abcikj->abcijk', tmps_["vvvooo_49"])

    # triples_resid += -1.00 <l,c||d,e> t1(d,i) t3(e,a,b,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcjki->abcijk', tmps_["vvvooo_49"])
    del tmps_["vvvooo_49"]

    # tmps_[vooo_89](c,i,l,j) = 1.00 t1(e,i) * eri[vovv](c,l,d,e) * t1(d,j) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["vooo_89"] = np.einsum('ei,celj->cilj', t1, tmps_["vvoo_30"])

    # tmps_[vvvooo_111](a,c,b,j,i,k) = 1.00 t2(a,c,j,l) * t1(e,i) * eri[vovv](b,l,d,e) * t1(d,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_111"] = np.einsum('acjl,bilk->acbjik', t2, tmps_["vooo_89"])

    # triples_resid += +1.00 P(i,j) P(a,b) <l,a||d,e> t1(d,k) t1(e,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcaijk->abcijk', tmps_["vvvooo_111"])
    triples_resid += np.einsum('bcajik->abcijk', tmps_["vvvooo_111"])
    triples_resid += np.einsum('acbijk->abcijk', tmps_["vvvooo_111"])
    triples_resid -= np.einsum('acbjik->abcijk', tmps_["vvvooo_111"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t1(d,j) t1(e,i) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcakij->abcijk', tmps_["vvvooo_111"])
    triples_resid += np.einsum('acbkij->abcijk', tmps_["vvvooo_111"])

    # triples_resid += +1.00 P(i,j) <l,c||d,e> t1(d,k) t1(e,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_111"])
    triples_resid += np.einsum('abcjik->abcijk', tmps_["vvvooo_111"])

    # triples_resid += +1.00 <l,c||d,e> t1(d,j) t1(e,i) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abckij->abcijk', tmps_["vvvooo_111"])
    del tmps_["vvvooo_111"]

    # tmps_[vvoo_124](b,a,i,j) = 1.00 t1(b,k) * t1(d,i) * eri[vovv](a,k,c,d) * t1(c,j) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_124"] = np.einsum('bk,aikj->baij', t1, tmps_["vooo_89"])
    del tmps_["vooo_89"]

    # doubles_resid += -1.00 P(a,b) <k,a||c,d> t1(c,j) t1(d,i) t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += np.einsum('baij->abij', tmps_["vvoo_124"])
    doubles_resid -= np.einsum('abij->abij', tmps_["vvoo_124"])
    del tmps_["vvoo_124"]

    # tmps_[vvvooo_108](b,c,a,i,j,k) = 1.00 t2(e,b,i,j) * eri[vovv](c,l,d,e) * t1(d,k) * t1(a,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["vvvooo_108"] = np.einsum('ebij,celk,al->bcaijk', t2, tmps_["vvoo_30"], t1, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["vvoo_30"]

    # triples_resid += +1.00 P(j,k) P(b,c) <l,a||d,e> t1(d,k) t1(b,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cabijk->abcijk', tmps_["vvvooo_108"])
    triples_resid += np.einsum('cabikj->abcijk', tmps_["vvvooo_108"])
    triples_resid += np.einsum('bacijk->abcijk', tmps_["vvvooo_108"])
    triples_resid -= np.einsum('bacikj->abcijk', tmps_["vvvooo_108"])

    # triples_resid += +1.00 P(b,c) <l,a||d,e> t1(d,i) t1(b,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cabjki->abcijk', tmps_["vvvooo_108"])
    triples_resid += np.einsum('bacjki->abcijk', tmps_["vvvooo_108"])

    # triples_resid += -1.00 P(j,k) P(a,c) <l,b||d,e> t1(d,k) t1(a,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cbaijk->abcijk', tmps_["vvvooo_108"])
    triples_resid -= np.einsum('cbaikj->abcijk', tmps_["vvvooo_108"])
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_108"])
    triples_resid += np.einsum('abcikj->abcijk', tmps_["vvvooo_108"])

    # triples_resid += -1.00 P(a,c) <l,b||d,e> t1(d,i) t1(a,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cbajki->abcijk', tmps_["vvvooo_108"])
    triples_resid -= np.einsum('abcjki->abcijk', tmps_["vvvooo_108"])

    # triples_resid += +1.00 P(j,k) P(a,b) <l,c||d,e> t1(d,k) t1(a,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcaijk->abcijk', tmps_["vvvooo_108"])
    triples_resid += np.einsum('bcaikj->abcijk', tmps_["vvvooo_108"])
    triples_resid += np.einsum('acbijk->abcijk', tmps_["vvvooo_108"])
    triples_resid -= np.einsum('acbikj->abcijk', tmps_["vvvooo_108"])

    # triples_resid += +1.00 P(a,b) <l,c||d,e> t1(d,i) t1(a,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcajki->abcijk', tmps_["vvvooo_108"])
    triples_resid += np.einsum('acbjki->abcijk', tmps_["vvvooo_108"])
    del tmps_["vvvooo_108"]

    # tmps_[vv_32](d,a) = 0.50 eri[oovv](l,k,c,d) * t2(c,a,l,k) // flops: o0v2 = o2v3 | mem: o0v2 = o0v2
    tmps_["vv_32"] = 0.50 * np.einsum('lkcd,calk->da', eri["oovv"], t2)

    # doubles_resid += -0.50 <l,k||c,d> t2(c,a,l,k) t2(d,b,i,j)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('da,dbij->abij', tmps_["vv_32"], t2)

    # tmps_[vvvooo_60](b,c,a,i,j,k) = 1.00 t3(e,b,c,i,j,k) * eri[oovv](m,l,d,e) * t2(d,a,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvvooo_60"] = np.einsum('ebcijk,ea->bcaijk', t3, tmps_["vv_32"])
    del tmps_["vv_32"]

    # triples_resid += -0.50 P(a,b) <m,l||d,e> t2(d,a,m,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcaijk->abcijk', tmps_["vvvooo_60"])
    triples_resid += np.einsum('acbijk->abcijk', tmps_["vvvooo_60"])

    # triples_resid += -0.50 <m,l||d,e> t2(d,c,m,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_60"])
    del tmps_["vvvooo_60"]

    # tmps_[vvoo_33](a,b,i,j) = 1.00 eri[vvvo](a,b,c,i) * t1(c,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["vvoo_33"] = np.einsum('abci,cj->abij', eri["vvvo"], t1)

    # doubles_resid += +1.00 P(i,j) <a,b||c,j> t1(c,i)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += np.einsum('abji->abij', tmps_["vvoo_33"])
    doubles_resid -= np.einsum('abij->abij', tmps_["vvoo_33"])
    del tmps_["vvoo_33"]

    # tmps_[vvoo_34](a,b,i,j) = 1.00 f[vv](a,c) * t2(c,b,i,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["vvoo_34"] = np.einsum('ac,cbij->abij', f["vv"], t2)

    # doubles_resid += +1.00 P(a,b) f(a,c) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += np.einsum('abij->abij', tmps_["vvoo_34"])
    doubles_resid -= np.einsum('baij->abij', tmps_["vvoo_34"])
    del tmps_["vvoo_34"]

    # tmps_[vooo_35](d,l,k,j) = 1.00 eri[oovv](l,k,c,d) * t1(c,j) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["vooo_35"] = np.einsum('lkcd,cj->dlkj', eri["oovv"], t1)

    # singles_resid += +0.50 <k,j||b,c> t1(b,i) t2(c,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
    singles_resid += 0.50 * np.einsum('ckji,cakj->ai', tmps_["vooo_35"], t2)

    # tmps_[vvvo_82](e,a,b,j) = 0.50 eri[oovv](m,l,d,e) * t1(d,j) * t2(a,b,m,l) // flops: o1v3 = o3v3 | mem: o1v3 = o1v3
    tmps_["vvvo_82"] = 0.50 * np.einsum('emlj,abml->eabj', tmps_["vooo_35"], t2)

    # doubles_resid += -0.50 <l,k||c,d> t1(c,j) t1(d,i) t2(a,b,l,k)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('dabj,di->abij', tmps_["vvvo_82"], t1)

    # tmps_[vvvooo_105](a,b,c,i,j,k) = 1.00 t2(e,a,i,j) * eri[oovv](m,l,d,e) * t1(d,k) * t2(b,c,m,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvvooo_105"] = np.einsum('eaij,ebck->abcijk', t2, tmps_["vvvo_82"])
    del tmps_["vvvo_82"]

    # triples_resid += +0.50 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,i,j) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_105"])
    triples_resid -= np.einsum('abcikj->abcijk', tmps_["vvvooo_105"])
    triples_resid -= np.einsum('bacijk->abcijk', tmps_["vvvooo_105"])
    triples_resid += np.einsum('bacikj->abcijk', tmps_["vvvooo_105"])

    # triples_resid += +0.50 P(j,k) <m,l||d,e> t1(d,k) t2(e,c,i,j) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabijk->abcijk', tmps_["vvvooo_105"])
    triples_resid -= np.einsum('cabikj->abcijk', tmps_["vvvooo_105"])

    # triples_resid += +0.50 P(a,b) <m,l||d,e> t1(d,i) t2(e,a,j,k) t2(b,c,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcjki->abcijk', tmps_["vvvooo_105"])
    triples_resid -= np.einsum('bacjki->abcijk', tmps_["vvvooo_105"])

    # triples_resid += +0.50 <m,l||d,e> t1(d,i) t2(e,c,j,k) t2(a,b,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabjki->abcijk', tmps_["vvvooo_105"])
    del tmps_["vvvooo_105"]

    # tmps_[oooo_101](m,l,j,i) = 1.00 eri[oovv](m,l,d,e) * t1(d,j) * t1(e,i) // flops: o4v0 = o4v1 | mem: o4v0 = o4v0
    tmps_["oooo_101"] = np.einsum('emlj,ei->mlji', tmps_["vooo_35"], t1)

    # tmps_[vvvooo_104](a,b,c,i,k,j) = 0.50 t3(a,b,c,i,m,l) * eri[oovv](m,l,d,e) * t1(d,k) * t1(e,j) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_104"] = 0.50 * np.einsum('abciml,mlkj->abcikj', t3, tmps_["oooo_101"])

    # triples_resid += -0.50 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcikj->abcijk', tmps_["vvvooo_104"])
    triples_resid += np.einsum('abcjki->abcijk', tmps_["vvvooo_104"])

    # triples_resid += -0.50 <m,l||d,e> t1(d,j) t1(e,i) t3(a,b,c,k,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abckji->abcijk', tmps_["vvvooo_104"])
    del tmps_["vvvooo_104"]

    # tmps_[vooo_125](a,l,j,i) = 1.00 t1(a,k) * eri[oovv](l,k,c,d) * t1(c,j) * t1(d,i) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["vooo_125"] = np.einsum('ak,lkji->alji', t1, tmps_["oooo_101"])
    del tmps_["oooo_101"]

    # doubles_resid += +1.00 <l,k||c,d> t1(c,j) t1(d,i) t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_resid += np.einsum('alji,bl->abij', tmps_["vooo_125"], t1)

    # tmps_[vvvooo_126](b,c,a,i,k,j) = 1.00 t2(b,c,i,m) * t1(a,l) * eri[oovv](m,l,d,e) * t1(d,k) * t1(e,j) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_126"] = np.einsum('bcim,amkj->bcaikj', t2, tmps_["vooo_125"])
    del tmps_["vooo_125"]

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t1(e,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcaikj->abcijk', tmps_["vvvooo_126"])
    triples_resid += np.einsum('bcajki->abcijk', tmps_["vvvooo_126"])
    triples_resid += np.einsum('acbikj->abcijk', tmps_["vvvooo_126"])
    triples_resid -= np.einsum('acbjki->abcijk', tmps_["vvvooo_126"])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(d,k) t1(e,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcikj->abcijk', tmps_["vvvooo_126"])
    triples_resid += np.einsum('abcjki->abcijk', tmps_["vvvooo_126"])

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(d,j) t1(e,i) t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcakji->abcijk', tmps_["vvvooo_126"])
    triples_resid += np.einsum('acbkji->abcijk', tmps_["vvvooo_126"])

    # triples_resid += -1.00 <m,l||d,e> t1(d,j) t1(e,i) t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abckji->abcijk', tmps_["vvvooo_126"])
    del tmps_["vvvooo_126"]

    # tmps_[vvvooo_107](b,c,a,i,j,k) = 1.00 t2(b,c,i,m) * t2(e,a,j,l) * eri[oovv](m,l,d,e) * t1(d,k) // flops: o3v3 = o4v4 o5v4 | mem: o3v3 = o4v4 o3v3
    tmps_["vvvooo_107"] = np.einsum('bcim,eajl,emlk->bcaijk', t2, t2, tmps_["vooo_35"], optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,k) t2(e,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcaijk->abcijk', tmps_["vvvooo_107"])
    triples_resid += np.einsum('bcajik->abcijk', tmps_["vvvooo_107"])
    triples_resid += np.einsum('acbijk->abcijk', tmps_["vvvooo_107"])
    triples_resid -= np.einsum('acbjik->abcijk', tmps_["vvvooo_107"])

    # triples_resid += -1.00 P(i,j) <m,l||d,e> t1(d,k) t2(e,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_107"])
    triples_resid += np.einsum('abcjik->abcijk', tmps_["vvvooo_107"])

    # triples_resid += +1.00 P(i,k) P(a,b) <m,l||d,e> t1(d,j) t2(e,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcaikj->abcijk', tmps_["vvvooo_107"])
    triples_resid -= np.einsum('bcakij->abcijk', tmps_["vvvooo_107"])
    triples_resid -= np.einsum('acbikj->abcijk', tmps_["vvvooo_107"])
    triples_resid += np.einsum('acbkij->abcijk', tmps_["vvvooo_107"])

    # triples_resid += +1.00 P(i,k) <m,l||d,e> t1(d,j) t2(e,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcikj->abcijk', tmps_["vvvooo_107"])
    triples_resid -= np.einsum('abckij->abcijk', tmps_["vvvooo_107"])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,i) t2(e,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcajki->abcijk', tmps_["vvvooo_107"])
    triples_resid += np.einsum('bcakji->abcijk', tmps_["vvvooo_107"])
    triples_resid += np.einsum('acbjki->abcijk', tmps_["vvvooo_107"])
    triples_resid -= np.einsum('acbkji->abcijk', tmps_["vvvooo_107"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(d,i) t2(e,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcjki->abcijk', tmps_["vvvooo_107"])
    triples_resid += np.einsum('abckji->abcijk', tmps_["vvvooo_107"])
    del tmps_["vvvooo_107"]

    # tmps_[vvvooo_115](a,b,c,k,i,j) = 1.00 t1(a,l) * eri[oovv](m,l,d,e) * t1(d,k) * t3(e,b,c,i,j,m) // flops: o3v3 = o3v2 o4v4 | mem: o3v3 = o2v2 o3v3
    tmps_["vvvooo_115"] = np.einsum('al,emlk,ebcijm->abckij', t1, tmps_["vooo_35"], t3, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,e> t1(d,k) t1(a,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abckij->abcijk', tmps_["vvvooo_115"])
    triples_resid -= np.einsum('abcjik->abcijk', tmps_["vvvooo_115"])
    triples_resid -= np.einsum('backij->abcijk', tmps_["vvvooo_115"])
    triples_resid += np.einsum('bacjik->abcijk', tmps_["vvvooo_115"])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t1(d,k) t1(c,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabkij->abcijk', tmps_["vvvooo_115"])
    triples_resid -= np.einsum('cabjik->abcijk', tmps_["vvvooo_115"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,i) t1(a,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_115"])
    triples_resid -= np.einsum('bacijk->abcijk', tmps_["vvvooo_115"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,i) t1(c,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabijk->abcijk', tmps_["vvvooo_115"])
    del tmps_["vvvooo_115"]

    # tmps_[vvvooo_127](a,b,c,i,j,k) = 1.00 t1(a,l) * t2(e,b,i,j) * eri[oovv](m,l,d,e) * t1(d,k) * t1(c,m) // flops: o3v3 = o3v3 o5v3 o4v3 | mem: o3v3 = o3v3 o4v2 o3v3
    tmps_["vvvooo_127"] = np.einsum('al,ebij,emlk,cm->abcijk', t1, t2, tmps_["vooo_35"], t1, optimize=['einsum_path',(0,1),(0,1),(0,1)])
    del tmps_["vooo_35"]

    # triples_resid += -1.00 P(j,k) P(b,c) <m,l||d,e> t1(d,k) t1(a,l) t1(b,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('acbijk->abcijk', tmps_["vvvooo_127"])
    triples_resid += np.einsum('acbikj->abcijk', tmps_["vvvooo_127"])
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_127"])
    triples_resid -= np.einsum('abcikj->abcijk', tmps_["vvvooo_127"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(d,k) t1(b,l) t1(c,m) t2(e,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bacijk->abcijk', tmps_["vvvooo_127"])
    triples_resid += np.einsum('bacikj->abcijk', tmps_["vvvooo_127"])

    # triples_resid += -1.00 P(b,c) <m,l||d,e> t1(d,i) t1(a,l) t1(b,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('acbjki->abcijk', tmps_["vvvooo_127"])
    triples_resid += np.einsum('abcjki->abcijk', tmps_["vvvooo_127"])

    # triples_resid += -1.00 <m,l||d,e> t1(d,i) t1(b,l) t1(c,m) t2(e,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bacjki->abcijk', tmps_["vvvooo_127"])
    del tmps_["vvvooo_127"]

    # tmps_[vooo_36](c,l,i,k) = 1.00 eri[vovo](c,l,d,i) * t1(d,k) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["vooo_36"] = np.einsum('cldi,dk->clik', eri["vovo"], t1)

    # tmps_[vvvooo_64](a,c,b,j,i,k) = 1.00 t2(a,c,j,l) * eri[vovo](b,l,d,i) * t1(d,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_64"] = np.einsum('acjl,blik->acbjik', t2, tmps_["vooo_36"])

    # triples_resid += -1.00 P(i,j) P(a,b) <l,a||d,k> t1(d,j) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcaikj->abcijk', tmps_["vvvooo_64"])
    triples_resid -= np.einsum('bcajki->abcijk', tmps_["vvvooo_64"])
    triples_resid -= np.einsum('acbikj->abcijk', tmps_["vvvooo_64"])
    triples_resid += np.einsum('acbjki->abcijk', tmps_["vvvooo_64"])

    # triples_resid += +1.00 P(i,k) P(a,b) <l,a||d,j> t1(d,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcaijk->abcijk', tmps_["vvvooo_64"])
    triples_resid += np.einsum('bcakji->abcijk', tmps_["vvvooo_64"])
    triples_resid += np.einsum('acbijk->abcijk', tmps_["vvvooo_64"])
    triples_resid -= np.einsum('acbkji->abcijk', tmps_["vvvooo_64"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,a||d,i> t1(d,k) t2(b,c,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcajik->abcijk', tmps_["vvvooo_64"])
    triples_resid -= np.einsum('bcakij->abcijk', tmps_["vvvooo_64"])
    triples_resid -= np.einsum('acbjik->abcijk', tmps_["vvvooo_64"])
    triples_resid += np.einsum('acbkij->abcijk', tmps_["vvvooo_64"])

    # triples_resid += -1.00 P(i,j) <l,c||d,k> t1(d,j) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcikj->abcijk', tmps_["vvvooo_64"])
    triples_resid -= np.einsum('abcjki->abcijk', tmps_["vvvooo_64"])

    # triples_resid += +1.00 P(i,k) <l,c||d,j> t1(d,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_64"])
    triples_resid += np.einsum('abckji->abcijk', tmps_["vvvooo_64"])

    # triples_resid += -1.00 P(j,k) <l,c||d,i> t1(d,k) t2(a,b,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcjik->abcijk', tmps_["vvvooo_64"])
    triples_resid -= np.einsum('abckij->abcijk', tmps_["vvvooo_64"])
    del tmps_["vvvooo_64"]

    # tmps_[vvoo_92](b,a,j,i) = 1.00 t1(b,k) * eri[vovo](a,k,c,j) * t1(c,i) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_92"] = np.einsum('bk,akji->baji', t1, tmps_["vooo_36"])
    del tmps_["vooo_36"]

    # doubles_resid += +1.00 P(i,j) P(a,b) <k,a||c,j> t1(c,i) t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('baji->abij', tmps_["vvoo_92"])
    doubles_resid += np.einsum('baij->abij', tmps_["vvoo_92"])
    doubles_resid += np.einsum('abji->abij', tmps_["vvoo_92"])
    doubles_resid -= np.einsum('abij->abij', tmps_["vvoo_92"])
    del tmps_["vvoo_92"]

    # tmps_[vooo_38](a,l,i,k) = 1.00 f[ov](l,d) * t2(d,a,i,k) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["vooo_38"] = np.einsum('ld,daik->alik', f["ov"], t2)

    # tmps_[vvvooo_71](a,b,c,k,i,j) = 1.00 t2(a,b,k,l) * f[ov](l,d) * t2(d,c,i,j) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_71"] = np.einsum('abkl,clij->abckij', t2, tmps_["vooo_38"])

    # triples_resid += -1.00 P(i,j) P(a,b) f(l,d) t2(d,a,j,k) t2(b,c,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcaijk->abcijk', tmps_["vvvooo_71"])
    triples_resid += np.einsum('bcajik->abcijk', tmps_["vvvooo_71"])
    triples_resid += np.einsum('acbijk->abcijk', tmps_["vvvooo_71"])
    triples_resid -= np.einsum('acbjik->abcijk', tmps_["vvvooo_71"])

    # triples_resid += -1.00 P(a,b) f(l,d) t2(d,a,i,j) t2(b,c,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcakij->abcijk', tmps_["vvvooo_71"])
    triples_resid += np.einsum('acbkij->abcijk', tmps_["vvvooo_71"])

    # triples_resid += -1.00 P(i,j) f(l,d) t2(d,c,j,k) t2(a,b,i,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_71"])
    triples_resid += np.einsum('abcjik->abcijk', tmps_["vvvooo_71"])

    # triples_resid += -1.00 f(l,d) t2(d,c,i,j) t2(a,b,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abckij->abcijk', tmps_["vvvooo_71"])
    del tmps_["vvvooo_71"]

    # tmps_[vvoo_98](b,a,i,j) = 1.00 t1(b,k) * f[ov](k,c) * t2(c,a,i,j) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_98"] = np.einsum('bk,akij->baij', t1, tmps_["vooo_38"])
    del tmps_["vooo_38"]

    # doubles_resid += -1.00 P(a,b) f(k,c) t1(a,k) t2(c,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('abij->abij', tmps_["vvoo_98"])
    doubles_resid += np.einsum('baij->abij', tmps_["vvoo_98"])
    del tmps_["vvoo_98"]

    # tmps_[oo_39](l,i) = 0.50 eri[oovv](l,k,c,d) * t2(c,d,i,k) // flops: o2v0 = o3v2 | mem: o2v0 = o2v0
    tmps_["oo_39"] = 0.50 * np.einsum('lkcd,cdik->li', eri["oovv"], t2)

    # tmps_[vvvooo_77](a,b,c,i,j,k) = 1.00 eri[oovv](m,l,d,e) * t2(d,e,i,l) * t3(a,b,c,j,k,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_77"] = np.einsum('mi,abcjkm->abcijk', tmps_["oo_39"], t3)

    # triples_resid += -0.50 P(j,k) <m,l||d,e> t2(d,e,k,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abckij->abcijk', tmps_["vvvooo_77"])
    triples_resid += np.einsum('abcjik->abcijk', tmps_["vvvooo_77"])

    # triples_resid += -0.50 <m,l||d,e> t2(d,e,i,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_77"])
    del tmps_["vvvooo_77"]

    # tmps_[vvoo_97](a,b,i,j) = 1.00 t2(a,b,i,l) * eri[oovv](l,k,c,d) * t2(c,d,j,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_97"] = np.einsum('abil,lj->abij', t2, tmps_["oo_39"])
    del tmps_["oo_39"]

    # doubles_resid += -0.50 P(i,j) <l,k||c,d> t2(c,d,j,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('abij->abij', tmps_["vvoo_97"])
    doubles_resid += np.einsum('abji->abij', tmps_["vvoo_97"])
    del tmps_["vvoo_97"]

    # tmps_[vvoo_41](a,b,i,j) = 1.00 eri[vooo](a,k,i,j) * t1(b,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_41"] = np.einsum('akij,bk->abij', eri["vooo"], t1)

    # doubles_resid += +1.00 P(a,b) <k,a||i,j> t1(b,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('abij->abij', tmps_["vvoo_41"])
    doubles_resid += np.einsum('baij->abij', tmps_["vvoo_41"])
    del tmps_["vvoo_41"]

    # tmps_[vvoo_42](a,b,j,i) = 1.00 f[oo](k,j) * t2(a,b,i,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_42"] = np.einsum('kj,abik->abji', f["oo"], t2)

    # doubles_resid += -1.00 P(i,j) f(k,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('abji->abij', tmps_["vvoo_42"])
    doubles_resid += np.einsum('abij->abij', tmps_["vvoo_42"])
    del tmps_["vvoo_42"]

    # tmps_[oooo_43](l,k,i,j) = 1.00 eri[oovo](l,k,c,i) * t1(c,j) // flops: o4v0 = o4v1 | mem: o4v0 = o4v0
    tmps_["oooo_43"] = np.einsum('lkci,cj->lkij', eri["oovo"], t1)

    # tmps_[vvvooo_52](a,b,c,k,j,i) = 0.50 eri[oovo](m,l,d,k) * t1(d,j) * t3(a,b,c,i,m,l) // flops: o3v3 = o5v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_52"] = 0.50 * np.einsum('mlkj,abciml->abckji', tmps_["oooo_43"], t3)

    # triples_resid += +0.50 P(i,j) <m,l||d,k> t1(d,j) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abckji->abcijk', tmps_["vvvooo_52"])
    triples_resid -= np.einsum('abckij->abcijk', tmps_["vvvooo_52"])

    # triples_resid += -0.50 P(i,k) <m,l||d,j> t1(d,k) t3(a,b,c,i,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcjki->abcijk', tmps_["vvvooo_52"])
    triples_resid += np.einsum('abcjik->abcijk', tmps_["vvvooo_52"])

    # triples_resid += +0.50 P(j,k) <m,l||d,i> t1(d,k) t3(a,b,c,j,m,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcikj->abcijk', tmps_["vvvooo_52"])
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_52"])
    del tmps_["vvvooo_52"]

    # tmps_[vooo_100](c,m,k,i) = 1.00 t1(c,l) * eri[oovo](m,l,d,k) * t1(d,i) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["vooo_100"] = np.einsum('cl,mlki->cmki', t1, tmps_["oooo_43"])
    del tmps_["oooo_43"]

    # tmps_[vvvooo_106](a,b,c,k,j,i) = 1.00 t2(a,b,k,m) * t1(c,l) * eri[oovo](m,l,d,j) * t1(d,i) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_106"] = np.einsum('abkm,cmji->abckji', t2, tmps_["vooo_100"])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,k> t1(d,j) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcaikj->abcijk', tmps_["vvvooo_106"])
    triples_resid -= np.einsum('bcajki->abcijk', tmps_["vvvooo_106"])
    triples_resid -= np.einsum('acbikj->abcijk', tmps_["vvvooo_106"])
    triples_resid += np.einsum('acbjki->abcijk', tmps_["vvvooo_106"])

    # triples_resid += +1.00 P(i,j) <m,l||d,k> t1(d,j) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcikj->abcijk', tmps_["vvvooo_106"])
    triples_resid -= np.einsum('abcjki->abcijk', tmps_["vvvooo_106"])

    # triples_resid += -1.00 P(i,k) P(a,b) <m,l||d,j> t1(d,k) t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcaijk->abcijk', tmps_["vvvooo_106"])
    triples_resid += np.einsum('bcakji->abcijk', tmps_["vvvooo_106"])
    triples_resid += np.einsum('acbijk->abcijk', tmps_["vvvooo_106"])
    triples_resid -= np.einsum('acbkji->abcijk', tmps_["vvvooo_106"])

    # triples_resid += -1.00 P(i,k) <m,l||d,j> t1(d,k) t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_106"])
    triples_resid += np.einsum('abckji->abcijk', tmps_["vvvooo_106"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,i> t1(d,k) t1(a,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcajik->abcijk', tmps_["vvvooo_106"])
    triples_resid -= np.einsum('bcakij->abcijk', tmps_["vvvooo_106"])
    triples_resid -= np.einsum('acbjik->abcijk', tmps_["vvvooo_106"])
    triples_resid += np.einsum('acbkij->abcijk', tmps_["vvvooo_106"])

    # triples_resid += +1.00 P(j,k) <m,l||d,i> t1(d,k) t1(c,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcjik->abcijk', tmps_["vvvooo_106"])
    triples_resid -= np.einsum('abckij->abcijk', tmps_["vvvooo_106"])
    del tmps_["vvvooo_106"]

    # tmps_[vvoo_122](a,b,j,i) = 1.00 t1(a,k) * eri[oovo](l,k,c,j) * t1(c,i) * t1(b,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_122"] = np.einsum('alji,bl->abji', tmps_["vooo_100"], t1)
    del tmps_["vooo_100"]

    # doubles_resid += -1.00 P(i,j) <l,k||c,j> t1(c,i) t1(a,k) t1(b,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('abji->abij', tmps_["vvoo_122"])
    doubles_resid += np.einsum('abij->abij', tmps_["vvoo_122"])
    del tmps_["vvoo_122"]

    # tmps_[vooo_44](c,m,i,k) = 1.00 eri[oooo](m,l,i,k) * t1(c,l) // flops: o3v1 = o4v1 | mem: o3v1 = o3v1
    tmps_["vooo_44"] = np.einsum('mlik,cl->cmik', eri["oooo"], t1)

    # doubles_resid += -1.00 <l,k||i,j> t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('alij,bl->abij', tmps_["vooo_44"], t1)

    # tmps_[vvvooo_72](a,c,b,i,j,k) = 1.00 t2(a,c,i,m) * eri[oooo](m,l,j,k) * t1(b,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_72"] = np.einsum('acim,bmjk->acbijk', t2, tmps_["vooo_44"])
    del tmps_["vooo_44"]

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||j,k> t1(a,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcaijk->abcijk', tmps_["vvvooo_72"])
    triples_resid -= np.einsum('bcajik->abcijk', tmps_["vvvooo_72"])
    triples_resid -= np.einsum('acbijk->abcijk', tmps_["vvvooo_72"])
    triples_resid += np.einsum('acbjik->abcijk', tmps_["vvvooo_72"])

    # triples_resid += +1.00 P(i,j) <m,l||j,k> t1(c,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_72"])
    triples_resid -= np.einsum('abcjik->abcijk', tmps_["vvvooo_72"])

    # triples_resid += +1.00 P(a,b) <m,l||i,j> t1(a,l) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcakij->abcijk', tmps_["vvvooo_72"])
    triples_resid -= np.einsum('acbkij->abcijk', tmps_["vvvooo_72"])

    # triples_resid += +1.00 <m,l||i,j> t1(c,l) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abckij->abcijk', tmps_["vvvooo_72"])
    del tmps_["vvvooo_72"]

    # tmps_[vv_45](a,d) = 1.00 eri[vovv](a,k,c,d) * t1(c,k) // flops: o0v2 = o1v3 | mem: o0v2 = o0v2
    tmps_["vv_45"] = np.einsum('akcd,ck->ad', eri["vovv"], t1)

    # tmps_[vvvooo_61](b,c,a,i,j,k) = 1.00 t3(e,b,c,i,j,k) * eri[vovv](a,l,d,e) * t1(d,l) // flops: o3v3 = o3v4 | mem: o3v3 = o3v3
    tmps_["vvvooo_61"] = np.einsum('ebcijk,ae->bcaijk', t3, tmps_["vv_45"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t1(d,l) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcaijk->abcijk', tmps_["vvvooo_61"])
    triples_resid += np.einsum('acbijk->abcijk', tmps_["vvvooo_61"])

    # triples_resid += +1.00 <l,c||d,e> t1(d,l) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_61"])
    del tmps_["vvvooo_61"]

    # tmps_[vvoo_87](a,b,i,j) = 1.00 eri[vovv](a,k,c,d) * t1(c,k) * t2(d,b,i,j) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["vvoo_87"] = np.einsum('ad,dbij->abij', tmps_["vv_45"], t2)
    del tmps_["vv_45"]

    # doubles_resid += +1.00 P(a,b) <k,a||c,d> t1(c,k) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('abij->abij', tmps_["vvoo_87"])
    doubles_resid += np.einsum('baij->abij', tmps_["vvoo_87"])
    del tmps_["vvoo_87"]

    # tmps_[vo_46](d,l) = 1.00 eri[oovv](l,k,c,d) * t1(c,k) // flops: o1v1 = o2v2 | mem: o1v1 = o1v1
    tmps_["vo_46"] = np.einsum('lkcd,ck->dl', eri["oovv"], t1)

    # doubles_resid += -1.00 <l,k||c,d> t1(c,k) t3(d,a,b,i,j,l)  // flops: o2v2 += o3v3 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('dl,dabijl->abij', tmps_["vo_46"], t3)

    # tmps_[vooo_90](b,i,j,m) = 1.00 t2(e,b,i,j) * eri[oovv](m,l,d,e) * t1(d,l) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["vooo_90"] = np.einsum('ebij,em->bijm', t2, tmps_["vo_46"])

    # tmps_[vvvooo_110](a,b,c,i,j,k) = 1.00 t2(a,b,i,m) * t2(e,c,j,k) * eri[oovv](m,l,d,e) * t1(d,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_110"] = np.einsum('abim,cjkm->abcijk', t2, tmps_["vooo_90"])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,e> t1(d,l) t2(e,a,j,k) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcaijk->abcijk', tmps_["vvvooo_110"])
    triples_resid -= np.einsum('bcajik->abcijk', tmps_["vvvooo_110"])
    triples_resid -= np.einsum('acbijk->abcijk', tmps_["vvvooo_110"])
    triples_resid += np.einsum('acbjik->abcijk', tmps_["vvvooo_110"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,l) t2(e,a,i,j) t2(b,c,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcakij->abcijk', tmps_["vvvooo_110"])
    triples_resid -= np.einsum('acbkij->abcijk', tmps_["vvvooo_110"])

    # triples_resid += +1.00 P(i,j) <m,l||d,e> t1(d,l) t2(e,c,j,k) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_110"])
    triples_resid -= np.einsum('abcjik->abcijk', tmps_["vvvooo_110"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t2(e,c,i,j) t2(a,b,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abckij->abcijk', tmps_["vvvooo_110"])
    del tmps_["vvvooo_110"]

    # tmps_[vvoo_123](b,a,i,j) = 1.00 t2(d,b,i,j) * eri[oovv](l,k,c,d) * t1(c,k) * t1(a,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_123"] = np.einsum('bijl,al->baij', tmps_["vooo_90"], t1)
    del tmps_["vooo_90"]

    # doubles_resid += +1.00 P(a,b) <l,k||c,d> t1(c,k) t1(a,l) t2(d,b,i,j)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += np.einsum('baij->abij', tmps_["vvoo_123"])
    doubles_resid -= np.einsum('abij->abij', tmps_["vvoo_123"])
    del tmps_["vvoo_123"]

    # tmps_[vvvooo_118](a,b,c,i,j,k) = 1.00 t1(a,m) * eri[oovv](m,l,d,e) * t1(d,l) * t3(e,b,c,i,j,k) // flops: o3v3 = o1v2 o3v4 | mem: o3v3 = o0v2 o3v3
    tmps_["vvvooo_118"] = np.einsum('am,em,ebcijk->abcijk', t1, tmps_["vo_46"], t3, optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["vo_46"]

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t1(d,l) t1(a,m) t3(e,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_118"])
    triples_resid -= np.einsum('bacijk->abcijk', tmps_["vvvooo_118"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t1(c,m) t3(e,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabijk->abcijk', tmps_["vvvooo_118"])
    del tmps_["vvvooo_118"]

    # tmps_[oo_47](l,i) = 1.00 eri[oovo](l,k,c,i) * t1(c,k) // flops: o2v0 = o3v1 | mem: o2v0 = o2v0
    tmps_["oo_47"] = np.einsum('lkci,ck->li', eri["oovo"], t1)

    # singles_resid += +1.00 <k,j||b,i> t1(b,j) t1(a,k)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid += np.einsum('ki,ak->ai', tmps_["oo_47"], t1)

    # tmps_[vvvooo_76](a,b,c,j,k,i) = 1.00 t3(a,b,c,j,k,m) * eri[oovo](m,l,d,i) * t1(d,l) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_76"] = np.einsum('abcjkm,mi->abcjki', t3, tmps_["oo_47"])

    # triples_resid += +1.00 P(j,k) <m,l||d,k> t1(d,l) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_76"])
    triples_resid -= np.einsum('abcikj->abcijk', tmps_["vvvooo_76"])

    # triples_resid += +1.00 <m,l||d,i> t1(d,l) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcjki->abcijk', tmps_["vvvooo_76"])
    del tmps_["vvvooo_76"]

    # tmps_[vvoo_96](a,b,j,i) = 1.00 eri[oovo](l,k,c,j) * t1(c,k) * t2(a,b,i,l) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_96"] = np.einsum('lj,abil->abji', tmps_["oo_47"], t2)
    del tmps_["oo_47"]

    # doubles_resid += +1.00 P(i,j) <l,k||c,j> t1(c,k) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += np.einsum('abji->abij', tmps_["vvoo_96"])
    doubles_resid -= np.einsum('abij->abij', tmps_["vvoo_96"])
    del tmps_["vvoo_96"]

    # tmps_[oo_48](k,j) = 1.00 f[ov](k,c) * t1(c,j) // flops: o2v0 = o2v1 | mem: o2v0 = o2v0
    tmps_["oo_48"] = np.einsum('kc,cj->kj', f["ov"], t1)

    # singles_resid += -1.00 f(j,b) t1(b,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid -= np.einsum('ji,aj->ai', tmps_["oo_48"], t1)

    # tmps_[vvvooo_78](a,b,c,i,j,k) = 1.00 t3(a,b,c,i,j,l) * f[ov](l,d) * t1(d,k) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_78"] = np.einsum('abcijl,lk->abcijk', t3, tmps_["oo_48"])

    # triples_resid += -1.00 P(j,k) f(l,d) t1(d,k) t3(a,b,c,i,j,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_78"])
    triples_resid += np.einsum('abcikj->abcijk', tmps_["vvvooo_78"])

    # triples_resid += -1.00 f(l,d) t1(d,i) t3(a,b,c,j,k,l)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcjki->abcijk', tmps_["vvvooo_78"])
    del tmps_["vvvooo_78"]

    # tmps_[vvoo_95](a,b,j,i) = 1.00 f[ov](k,c) * t1(c,j) * t2(a,b,i,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_95"] = np.einsum('kj,abik->abji', tmps_["oo_48"], t2)
    del tmps_["oo_48"]

    # doubles_resid += -1.00 P(i,j) f(k,c) t1(c,j) t2(a,b,i,k)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('abji->abij', tmps_["vvoo_95"])
    doubles_resid += np.einsum('abij->abij', tmps_["vvoo_95"])
    del tmps_["vvoo_95"]

    # tmps_[vvvooo_50](b,c,a,j,k,i) = 1.00 t3(e,b,c,j,k,m) * eri[oovv](m,l,d,e) * t2(d,a,i,l) // flops: o3v3 = o4v4 o4v4 | mem: o3v3 = o3v3 o3v3
    tmps_["vvvooo_50"] = np.einsum('ebcjkm,mlde,dail->bcajki', t3, eri["oovv"], t2, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,e> t2(d,a,k,l) t3(e,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcaijk->abcijk', tmps_["vvvooo_50"])
    triples_resid -= np.einsum('bcaikj->abcijk', tmps_["vvvooo_50"])
    triples_resid -= np.einsum('acbijk->abcijk', tmps_["vvvooo_50"])
    triples_resid += np.einsum('acbikj->abcijk', tmps_["vvvooo_50"])

    # triples_resid += +1.00 P(a,b) <m,l||d,e> t2(d,a,i,l) t3(e,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcajki->abcijk', tmps_["vvvooo_50"])
    triples_resid -= np.einsum('acbjki->abcijk', tmps_["vvvooo_50"])

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t2(d,c,k,l) t3(e,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_50"])
    triples_resid -= np.einsum('abcikj->abcijk', tmps_["vvvooo_50"])

    # triples_resid += +1.00 <m,l||d,e> t2(d,c,i,l) t3(e,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcjki->abcijk', tmps_["vvvooo_50"])
    del tmps_["vvvooo_50"]

    # tmps_[vvvooo_55](a,c,b,i,j,k) = 1.00 t2(e,c,i,j) * eri[vovv](a,l,d,e) * t2(d,b,k,l) // flops: o3v3 = o3v4 o4v4 | mem: o3v3 = o3v3 o3v3
    tmps_["vvvooo_55"] = np.einsum('ecij,alde,dbkl->acbijk', t2, eri["vovv"], t2, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 P(j,k) P(a,b) <l,a||d,e> t2(d,b,k,l) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('acbijk->abcijk', tmps_["vvvooo_55"])
    triples_resid += np.einsum('acbikj->abcijk', tmps_["vvvooo_55"])
    triples_resid += np.einsum('bcaijk->abcijk', tmps_["vvvooo_55"])
    triples_resid -= np.einsum('bcaikj->abcijk', tmps_["vvvooo_55"])

    # triples_resid += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,l) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('acbjki->abcijk', tmps_["vvvooo_55"])
    triples_resid += np.einsum('bcajki->abcijk', tmps_["vvvooo_55"])

    # triples_resid += +1.00 P(j,k) <l,c||d,e> t2(d,a,k,l) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cbaijk->abcijk', tmps_["vvvooo_55"])
    triples_resid += np.einsum('cbaikj->abcijk', tmps_["vvvooo_55"])

    # triples_resid += +1.00 <l,c||d,e> t2(d,a,i,l) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cbajki->abcijk', tmps_["vvvooo_55"])
    del tmps_["vvvooo_55"]

    # tmps_[vvvooo_63](a,b,c,k,i,j) = 1.00 t1(b,l) * eri[vovo](a,l,d,k) * t2(d,c,i,j) // flops: o3v3 = o2v3 o3v4 | mem: o3v3 = o1v3 o3v3
    tmps_["vvvooo_63"] = np.einsum('bl,aldk,dcij->abckij', t1, eri["vovo"], t2, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += -1.00 P(j,k) P(b,c) <l,a||d,k> t1(b,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abckij->abcijk', tmps_["vvvooo_63"])
    triples_resid -= np.einsum('abcjik->abcijk', tmps_["vvvooo_63"])
    triples_resid -= np.einsum('acbkij->abcijk', tmps_["vvvooo_63"])
    triples_resid += np.einsum('acbjik->abcijk', tmps_["vvvooo_63"])

    # triples_resid += -1.00 P(b,c) <l,a||d,i> t1(b,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_63"])
    triples_resid -= np.einsum('acbijk->abcijk', tmps_["vvvooo_63"])

    # triples_resid += +1.00 P(j,k) P(a,c) <l,b||d,k> t1(a,l) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('backij->abcijk', tmps_["vvvooo_63"])
    triples_resid += np.einsum('bacjik->abcijk', tmps_["vvvooo_63"])
    triples_resid += np.einsum('bcakij->abcijk', tmps_["vvvooo_63"])
    triples_resid -= np.einsum('bcajik->abcijk', tmps_["vvvooo_63"])

    # triples_resid += +1.00 P(a,c) <l,b||d,i> t1(a,l) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bacijk->abcijk', tmps_["vvvooo_63"])
    triples_resid += np.einsum('bcaijk->abcijk', tmps_["vvvooo_63"])

    # triples_resid += -1.00 P(j,k) P(a,b) <l,c||d,k> t1(a,l) t2(d,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabkij->abcijk', tmps_["vvvooo_63"])
    triples_resid -= np.einsum('cabjik->abcijk', tmps_["vvvooo_63"])
    triples_resid -= np.einsum('cbakij->abcijk', tmps_["vvvooo_63"])
    triples_resid += np.einsum('cbajik->abcijk', tmps_["vvvooo_63"])

    # triples_resid += -1.00 P(a,b) <l,c||d,i> t1(a,l) t2(d,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabijk->abcijk', tmps_["vvvooo_63"])
    triples_resid -= np.einsum('cbaijk->abcijk', tmps_["vvvooo_63"])
    del tmps_["vvvooo_63"]

    # tmps_[vvvooo_65](b,c,a,k,j,i) = 1.00 t2(b,c,j,m) * eri[oovo](m,l,d,k) * t2(d,a,i,l) // flops: o3v3 = o4v3 o4v4 | mem: o3v3 = o3v3 o3v3
    tmps_["vvvooo_65"] = np.einsum('bcjm,mldk,dail->bcakji', t2, eri["oovo"], t2, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +1.00 P(i,j) P(a,b) <m,l||d,k> t2(d,a,j,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcakij->abcijk', tmps_["vvvooo_65"])
    triples_resid -= np.einsum('bcakji->abcijk', tmps_["vvvooo_65"])
    triples_resid -= np.einsum('acbkij->abcijk', tmps_["vvvooo_65"])
    triples_resid += np.einsum('acbkji->abcijk', tmps_["vvvooo_65"])

    # triples_resid += +1.00 P(i,j) <m,l||d,k> t2(d,c,j,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abckij->abcijk', tmps_["vvvooo_65"])
    triples_resid -= np.einsum('abckji->abcijk', tmps_["vvvooo_65"])

    # triples_resid += -1.00 P(i,k) P(a,b) <m,l||d,j> t2(d,a,k,l) t2(b,c,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcajik->abcijk', tmps_["vvvooo_65"])
    triples_resid += np.einsum('bcajki->abcijk', tmps_["vvvooo_65"])
    triples_resid += np.einsum('acbjik->abcijk', tmps_["vvvooo_65"])
    triples_resid -= np.einsum('acbjki->abcijk', tmps_["vvvooo_65"])

    # triples_resid += -1.00 P(i,k) <m,l||d,j> t2(d,c,k,l) t2(a,b,i,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcjik->abcijk', tmps_["vvvooo_65"])
    triples_resid += np.einsum('abcjki->abcijk', tmps_["vvvooo_65"])

    # triples_resid += +1.00 P(j,k) P(a,b) <m,l||d,i> t2(d,a,k,l) t2(b,c,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcaijk->abcijk', tmps_["vvvooo_65"])
    triples_resid -= np.einsum('bcaikj->abcijk', tmps_["vvvooo_65"])
    triples_resid -= np.einsum('acbijk->abcijk', tmps_["vvvooo_65"])
    triples_resid += np.einsum('acbikj->abcijk', tmps_["vvvooo_65"])

    # triples_resid += +1.00 P(j,k) <m,l||d,i> t2(d,c,k,l) t2(a,b,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_65"])
    triples_resid -= np.einsum('abcikj->abcijk', tmps_["vvvooo_65"])
    del tmps_["vvvooo_65"]

    # tmps_[vvvooo_68](b,c,a,k,i,j) = 1.00 eri[oovo](m,l,d,k) * t3(d,b,c,i,j,m) * t1(a,l) // flops: o3v3 = o5v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["vvvooo_68"] = np.einsum('mldk,dbcijm,al->bcakij', eri["oovo"], t3, t1, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,k> t1(a,l) t3(d,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcakij->abcijk', tmps_["vvvooo_68"])
    triples_resid += np.einsum('bcajik->abcijk', tmps_["vvvooo_68"])
    triples_resid += np.einsum('acbkij->abcijk', tmps_["vvvooo_68"])
    triples_resid -= np.einsum('acbjik->abcijk', tmps_["vvvooo_68"])

    # triples_resid += -1.00 P(j,k) <m,l||d,k> t1(c,l) t3(d,a,b,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abckij->abcijk', tmps_["vvvooo_68"])
    triples_resid += np.einsum('abcjik->abcijk', tmps_["vvvooo_68"])

    # triples_resid += -1.00 P(a,b) <m,l||d,i> t1(a,l) t3(d,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcaijk->abcijk', tmps_["vvvooo_68"])
    triples_resid += np.einsum('acbijk->abcijk', tmps_["vvvooo_68"])

    # triples_resid += -1.00 <m,l||d,i> t1(c,l) t3(d,a,b,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_68"])
    del tmps_["vvvooo_68"]

    # tmps_[vvvooo_73](a,c,b,i,j,k) = 0.50 t1(c,l) * eri[vovv](a,l,d,e) * t3(d,e,b,i,j,k) // flops: o3v3 = o1v4 o3v5 | mem: o3v3 = o0v4 o3v3
    tmps_["vvvooo_73"] = 0.50 * np.einsum('cl,alde,debijk->acbijk', t1, eri["vovv"], t3, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += +0.50 P(b,c) <l,a||d,e> t1(b,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_73"])
    triples_resid += np.einsum('acbijk->abcijk', tmps_["vvvooo_73"])

    # triples_resid += -0.50 P(a,c) <l,b||d,e> t1(a,l) t3(d,e,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bacijk->abcijk', tmps_["vvvooo_73"])
    triples_resid -= np.einsum('bcaijk->abcijk', tmps_["vvvooo_73"])

    # triples_resid += +0.50 P(a,b) <l,c||d,e> t1(a,l) t3(d,e,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cabijk->abcijk', tmps_["vvvooo_73"])
    triples_resid += np.einsum('cbaijk->abcijk', tmps_["vvvooo_73"])
    del tmps_["vvvooo_73"]

    # tmps_[vvvooo_75](b,c,a,i,j,k) = 1.00 f[ov](l,d) * t3(d,b,c,i,j,k) * t1(a,l) // flops: o3v3 = o4v3 o4v3 | mem: o3v3 = o4v2 o3v3
    tmps_["vvvooo_75"] = np.einsum('ld,dbcijk,al->bcaijk', f["ov"], t3, t1, optimize=['einsum_path',(0,1),(0,1)])

    # triples_resid += -1.00 P(a,b) f(l,d) t1(a,l) t3(d,b,c,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('bcaijk->abcijk', tmps_["vvvooo_75"])
    triples_resid += np.einsum('acbijk->abcijk', tmps_["vvvooo_75"])

    # triples_resid += -1.00 f(l,d) t1(c,l) t3(d,a,b,i,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('abcijk->abcijk', tmps_["vvvooo_75"])
    del tmps_["vvvooo_75"]

    # tmps_[vvoo_93](a,b,j,i) = 1.00 t1(a,k) * eri[oovo](l,k,c,j) * t2(c,b,i,l) // flops: o2v2 = o3v2 o3v3 | mem: o2v2 = o2v2 o2v2
    tmps_["vvoo_93"] = np.einsum('ak,lkcj,cbil->abji', t1, eri["oovo"], t2, optimize=['einsum_path',(0,1),(0,1)])

    # doubles_resid += -1.00 P(i,j) P(a,b) <l,k||c,j> t1(a,k) t2(c,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid -= np.einsum('abji->abij', tmps_["vvoo_93"])
    doubles_resid += np.einsum('abij->abij', tmps_["vvoo_93"])
    doubles_resid += np.einsum('baji->abij', tmps_["vvoo_93"])
    doubles_resid -= np.einsum('baij->abij', tmps_["vvoo_93"])
    del tmps_["vvoo_93"]

    # tmps_[oo_103](l,i) = 1.00 eri[oovv](l,k,c,d) * t1(d,i) * t1(c,k) // flops: o2v0 = o3v2 o3v1 | mem: o2v0 = o3v1 o2v0
    tmps_["oo_103"] = np.einsum('lkcd,di,ck->li', eri["oovv"], t1, t1, optimize=['einsum_path',(0,1),(0,1)])

    # singles_resid += +1.00 <k,j||b,c> t1(b,j) t1(c,i) t1(a,k)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    singles_resid += np.einsum('ki,ak->ai', tmps_["oo_103"], t1)

    # tmps_[vvvooo_116](a,b,c,k,i,j) = 1.00 eri[oovv](m,l,d,e) * t1(e,k) * t1(d,l) * t3(a,b,c,i,j,m) // flops: o3v3 = o4v3 | mem: o3v3 = o3v3
    tmps_["vvvooo_116"] = np.einsum('mk,abcijm->abckij', tmps_["oo_103"], t3)

    # triples_resid += +1.00 P(j,k) <m,l||d,e> t1(d,l) t1(e,k) t3(a,b,c,i,j,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abckij->abcijk', tmps_["vvvooo_116"])
    triples_resid -= np.einsum('abcjik->abcijk', tmps_["vvvooo_116"])

    # triples_resid += +1.00 <m,l||d,e> t1(d,l) t1(e,i) t3(a,b,c,j,k,m)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('abcijk->abcijk', tmps_["vvvooo_116"])
    del tmps_["vvvooo_116"]

    # tmps_[vvoo_121](a,b,i,j) = 1.00 t2(a,b,i,l) * eri[oovv](l,k,c,d) * t1(d,j) * t1(c,k) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["vvoo_121"] = np.einsum('abil,lj->abij', t2, tmps_["oo_103"])
    del tmps_["oo_103"]

    # doubles_resid += +1.00 P(i,j) <l,k||c,d> t1(c,k) t1(d,j) t2(a,b,i,l)  // flops: o2v2 += o2v2 | mem: o2v2 += o2v2
    doubles_resid += np.einsum('abij->abij', tmps_["vvoo_121"])
    doubles_resid -= np.einsum('abji->abij', tmps_["vvoo_121"])
    del tmps_["vvoo_121"]

    # tmps_[vvvooo_112](b,c,a,k,i,j) = 1.00 eri[oovo](m,l,d,k) * t1(b,m) * t2(d,c,i,j) * t1(a,l) // flops: o3v3 = o3v2 o4v3 o4v3 | mem: o3v3 = o2v2 o4v2 o3v3
    tmps_["vvvooo_112"] = np.einsum('mldk,bm,dcij,al->bcakij', eri["oovo"], t1, t2, t1, optimize=['einsum_path',(0,1),(0,1),(0,1)])

    # triples_resid += +1.00 P(j,k) P(b,c) <m,l||d,k> t1(a,l) t1(b,m) t2(d,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcakij->abcijk', tmps_["vvvooo_112"])
    triples_resid -= np.einsum('bcajik->abcijk', tmps_["vvvooo_112"])
    triples_resid -= np.einsum('cbakij->abcijk', tmps_["vvvooo_112"])
    triples_resid += np.einsum('cbajik->abcijk', tmps_["vvvooo_112"])

    # triples_resid += +1.00 P(j,k) <m,l||d,k> t1(b,l) t1(c,m) t2(d,a,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabkij->abcijk', tmps_["vvvooo_112"])
    triples_resid -= np.einsum('cabjik->abcijk', tmps_["vvvooo_112"])

    # triples_resid += +1.00 P(b,c) <m,l||d,i> t1(a,l) t1(b,m) t2(d,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('bcaijk->abcijk', tmps_["vvvooo_112"])
    triples_resid -= np.einsum('cbaijk->abcijk', tmps_["vvvooo_112"])

    # triples_resid += +1.00 <m,l||d,i> t1(b,l) t1(c,m) t2(d,a,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid += np.einsum('cabijk->abcijk', tmps_["vvvooo_112"])
    del tmps_["vvvooo_112"]

    # tmps_[vvvooo_114](b,c,a,i,j,k) = 1.00 t1(b,l) * eri[oovv](m,l,d,e) * t2(e,c,i,j) * t2(d,a,k,m) // flops: o3v3 = o2v3 o3v4 o4v4 | mem: o3v3 = o1v3 o3v3 o3v3
    tmps_["vvvooo_114"] = np.einsum('bl,mlde,ecij,dakm->bcaijk', t1, eri["oovv"], t2, t2, optimize=['einsum_path',(0,1),(0,1),(0,1)])

    # triples_resid += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,k,m) t2(e,c,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('acbijk->abcijk', tmps_["vvvooo_114"])
    triples_resid += np.einsum('acbikj->abcijk', tmps_["vvvooo_114"])
    triples_resid += np.einsum('bcaijk->abcijk', tmps_["vvvooo_114"])
    triples_resid -= np.einsum('bcaikj->abcijk', tmps_["vvvooo_114"])

    # triples_resid += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,m) t2(e,c,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('acbjki->abcijk', tmps_["vvvooo_114"])
    triples_resid += np.einsum('bcajki->abcijk', tmps_["vvvooo_114"])

    # triples_resid += -1.00 P(j,k) <m,l||d,e> t1(c,l) t2(d,a,k,m) t2(e,b,i,j)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cbaijk->abcijk', tmps_["vvvooo_114"])
    triples_resid += np.einsum('cbaikj->abcijk', tmps_["vvvooo_114"])

    # triples_resid += -1.00 <m,l||d,e> t1(c,l) t2(d,a,i,m) t2(e,b,j,k)  // flops: o3v3 += o3v3 | mem: o3v3 += o3v3
    triples_resid -= np.einsum('cbajki->abcijk', tmps_["vvvooo_114"])
    del tmps_["vvvooo_114"]

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



