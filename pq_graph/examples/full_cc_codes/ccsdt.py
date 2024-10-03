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

    # rt1  = +1.00 f(a,i)
    rt1  = 1.00 * np.einsum('ai->ai',f["vo"])

    # rt2  = +1.00 <a,b||i,j>
    rt2  = 1.00 * np.einsum('abij->abij',eri["vvoo"])

    # rt1 += -1.00 f(j,i) t1(a,j)
    # flops: o1v1 += o2v1
    #  mems: o1v1 += o1v1
    rt1 -= einsum('ia->ai', np.einsum('ji,aj->ia',f["oo"],t1) )

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
    rt1 += 0.50 * einsum('ia->ai', np.einsum('jkbi,bakj->ia',eri["oovo"],t2) )

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

    # rt2 += -1.00 P(i,j) f(k,j) t2(a,b,i,k)
    # flops: o2v2 += o3v2
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 1.00 * einsum('jabi->abij', np.einsum('kj,abik->jabi',f["oo"],t2) )
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += +1.00 P(a,b) <k,a||i,j> t1(b,k)
    # flops: o2v2 += o3v2
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 1.00 * einsum('aijb->abij', np.einsum('akij,bk->aijb',eri["vooo"],t1) )
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += +1.00 P(a,b) f(a,c) t2(c,b,i,j)
    # flops: o2v2 += o2v3
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 1.00 * np.einsum('ac,cbij->abij',f["vv"],t2)
    rt2 += np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 -= einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += +1.00 P(i,j) <a,b||c,j> t1(c,i)
    # flops: o2v2 += o2v3
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 1.00 * einsum('abji->abij', np.einsum('abcj,ci->abji',eri["vvvo"],t1) )
    rt2 += np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 -= einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += +0.50 <l,k||i,j> t2(a,b,l,k)
    # flops: o2v2 += o4v2
    #  mems: o2v2 += o2v2
    rt2 -= 0.50 * einsum('ijab->abij', np.einsum('klij,ablk->ijab',eri["oooo"],t2) )

    # rt2 += +1.00 f(k,c) t3(c,a,b,i,j,k)
    # flops: o2v2 += o3v3
    #  mems: o2v2 += o2v2
    rt2 += np.einsum('kc,cabijk->abij',f["ov"],t3)

    # rt2 += +1.00 P(i,j) P(a,b) <k,a||c,j> t2(c,b,i,k)
    # flops: o2v2 += o3v3
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 1.00 * einsum('ajbi->abij', np.einsum('akcj,cbik->ajbi',eri["vovo"],t2) )
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    rt2 += einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    rt2 -= einsum('baji->abij', np.einsum('baji->baji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += +0.50 <a,b||c,d> t2(c,d,i,j)
    # flops: o2v2 += o2v4
    #  mems: o2v2 += o2v2
    rt2 += 0.50 * np.einsum('abcd,cdij->abij',eri["vvvv"],t2)

    # rt2 += +0.50 P(i,j) <l,k||c,j> t3(c,a,b,i,l,k)
    # flops: o2v2 += o4v3
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 0.50 * einsum('jabi->abij', np.einsum('klcj,cabilk->jabi',eri["oovo"],t3) )
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += +0.50 P(a,b) <k,a||c,d> t3(c,d,b,i,j,k)
    # flops: o2v2 += o3v4
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 0.50 * np.einsum('akcd,cdbijk->abij',eri["vovv"],t3)
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += -1.00 <l,k||i,j> t1(a,k) t1(b,l)
    # flops: o2v2 += o4v1 o3v2
    #  mems: o2v2 += o3v1 o2v2
    rt2 += einsum('bija->abij', np.einsum('bl,klij,ak->bija',t1,eri["oooo"],t1,optimize='optimal') )

    # rt2 += +1.00 P(i,j) P(a,b) <k,a||c,j> t1(b,k) t1(c,i)
    # flops: o2v2 += o3v2 o3v2
    #  mems: o2v2 += o3v1 o2v2
    tmps_["perm_vvoo"]  = 1.00 * einsum('ajib->abij', np.einsum('akcj,ci,bk->ajib',eri["vovo"],t1,t1,optimize='optimal') )
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    rt2 += einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    rt2 -= einsum('baji->abij', np.einsum('baji->baji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += -1.00 P(a,b) f(k,c) t1(a,k) t2(c,b,i,j)
    # flops: o2v2 += o3v2 o3v2
    #  mems: o2v2 += o3v1 o2v2
    tmps_["perm_vvoo"]  = 1.00 * einsum('bija->abij', np.einsum('kc,cbij,ak->bija',f["ov"],t2,t1,optimize='optimal') )
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += -0.50 P(i,j) <l,k||c,d> t2(a,b,i,l) t2(c,d,j,k)
    # flops: o2v2 += o3v2 o3v2
    #  mems: o2v2 += o2v0 o2v2
    tmps_["perm_vvoo"]  = 0.50 * einsum('jabi->abij', np.einsum('klcd,cdjk,abil->jabi',eri["oovv"],t2,t2,optimize='optimal') )
    rt2 += np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 -= einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

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
    rt2 -= einsum('abji->abij', np.einsum('abcd,cj,di->abji',eri["vvvv"],t1,t1,optimize='optimal') )

    # rt2 += -1.00 P(i,j) P(a,b) <l,k||c,j> t1(a,k) t2(c,b,i,l)
    # flops: o2v2 += o4v2 o3v2
    #  mems: o2v2 += o3v1 o2v2
    tmps_["perm_vvoo"]  = 1.00 * einsum('jbia->abij', np.einsum('klcj,cbil,ak->jbia',eri["oovo"],t2,t1,optimize='optimal') )
    rt2 += np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 -= einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    rt2 -= einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    rt2 += einsum('baji->abij', np.einsum('baji->baji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += +0.50 P(a,b) <k,a||c,d> t1(b,k) t2(c,d,i,j)
    # flops: o2v2 += o3v3 o3v2
    #  mems: o2v2 += o3v1 o2v2
    tmps_["perm_vvoo"]  = 0.50 * einsum('aijb->abij', np.einsum('akcd,cdij,bk->aijb',eri["vovv"],t2,t1,optimize='optimal') )
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += -0.50 P(a,b) <l,k||c,d> t1(a,k) t3(c,d,b,i,j,l)
    # flops: o2v2 += o4v3 o3v2
    #  mems: o2v2 += o3v1 o2v2
    tmps_["perm_vvoo"]  = 0.50 * einsum('bija->abij', np.einsum('klcd,cdbijl,ak->bija',eri["oovv"],t3,t1,optimize='optimal') )
    rt2 += np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 -= einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # flops: o1v1  = o2v2
    #  mems: o1v1  = o1v1
    tmps_["1_ov"]  = 1.00 * np.einsum('jkbc,bj->kc',eri["oovv"],t1)

    # rt2 += -1.00 <l,k||c,d> t3(d,a,b,i,j,l) t1(c,k)
    # flops: o2v2 += o3v3
    #  mems: o2v2 += o2v2
    rt2 += np.einsum('dabijl,ld->abij',t3,tmps_["1_ov"])

    # rt2 += +1.00 P(a,b) <l,k||c,d> t1(a,l) t2(d,b,i,j) t1(c,k)
    # flops: o2v2 += o3v2 o3v2
    #  mems: o2v2 += o3v1 o2v2
    tmps_["perm_vvoo"]  = 1.00 * einsum('bija->abij', np.einsum('dbij,ld,al->bija',t2,tmps_["1_ov"],t1,optimize='optimal') )
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # flops: o2v0  = o2v1
    #  mems: o2v0  = o2v0
    tmps_["2_oo"]  = 1.00 * np.einsum('ci,kc->ik',t1,tmps_["1_ov"])

    # rt1 += +1.00 <k,j||b,c> t1(a,k) t1(b,j) t1(c,i)
    # flops: o1v1 += o2v1
    #  mems: o1v1 += o1v1
    rt1 -= np.einsum('ak,ik->ai',t1,tmps_["2_oo"])

    # rt2 += +1.00 P(i,j) <l,k||c,d> t2(a,b,i,l) t1(c,k) t1(d,j)
    # flops: o2v2 += o3v2
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 1.00 * np.einsum('abil,jl->abij',t2,tmps_["2_oo"])
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]
    del tmps_["2_oo"]

    # flops: o2v0  = o2v1
    #  mems: o2v0  = o2v0
    tmps_["3_oo"]  = 1.00 * np.einsum('bi,jb->ij',t1,f["ov"])

    # rt1 += -1.00 f(j,b) t1(a,j) t1(b,i)
    # flops: o1v1 += o2v1
    #  mems: o1v1 += o1v1
    rt1 -= np.einsum('aj,ij->ai',t1,tmps_["3_oo"])

    # rt2 += -1.00 P(i,j) f(k,c) t2(a,b,i,k) t1(c,j)
    # flops: o2v2 += o3v2
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 1.00 * np.einsum('abik,jk->abij',t2,tmps_["3_oo"])
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]
    del tmps_["3_oo"]

    # flops: o0v2  = o1v3
    #  mems: o0v2  = o0v2
    tmps_["4_vv"]  = 1.00 * np.einsum('ajbc,bj->ac',eri["vovv"],t1)

    # rt1 += +1.00 <j,a||b,c> t1(b,j) t1(c,i)
    # flops: o1v1 += o1v2
    #  mems: o1v1 += o1v1
    rt1 -= np.einsum('ac,ci->ai',tmps_["4_vv"],t1)

    # rt2 += +1.00 P(a,b) <k,a||c,d> t2(d,b,i,j) t1(c,k)
    # flops: o2v2 += o2v3
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 1.00 * np.einsum('ad,dbij->abij',tmps_["4_vv"],t2)
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # flops: o2v2  = o3v3
    #  mems: o2v2  = o2v2
    tmps_["5_voov"]  = 1.00 * np.einsum('eckm,lmde->ckld',t2,eri["oovv"])

    # rt1 += +1.00 <k,j||b,c> t2(c,a,i,k) t1(b,j)
    # flops: o1v1 += o2v2
    #  mems: o1v1 += o1v1
    rt1 -= np.einsum('bj,aijb->ai',t1,tmps_["5_voov"])

    # rt2 += +1.00 P(i,j) <l,k||c,d> t2(c,a,j,k) t2(d,b,i,l)
    # flops: o2v2 += o3v3
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 1.00 * einsum('ajbi->abij', np.einsum('cajk,bikc->ajbi',t2,tmps_["5_voov"]) )
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += +1.00 P(i,j) P(a,b) <l,k||c,d> t1(a,k) t2(d,b,i,l) t1(c,j)
    # flops: o2v2 += o3v2 o3v2
    #  mems: o2v2 += o3v1 o2v2
    tmps_["perm_vvoo"]  = 1.00 * einsum('bija->abij', np.einsum('bikc,cj,ak->bija',tmps_["5_voov"],t1,t1,optimize='optimal') )
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    rt2 += einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    rt2 -= einsum('baji->abij', np.einsum('baji->baji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]
    del tmps_["5_voov"]

    # flops: o3v1  = o3v2
    #  mems: o3v1  = o3v1
    tmps_["6_ooov"]  = 1.00 * np.einsum('bi,jkbc->ijkc',t1,eri["oovv"])

    # rt1 += +0.50 <k,j||b,c> t2(c,a,k,j) t1(b,i)
    # flops: o1v1 += o3v2
    #  mems: o1v1 += o1v1
    rt1 -= 0.50 * np.einsum('cakj,ijkc->ai',t2,tmps_["6_ooov"])

    # rt2 += -0.50 P(i,j) <l,k||c,d> t3(d,a,b,i,l,k) t1(c,j)
    # flops: o2v2 += o4v3
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 0.50 * np.einsum('dabilk,jkld->abij',t3,tmps_["6_ooov"])
    rt2 += np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 -= einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # flops: o2v0  = o3v1
    #  mems: o2v0  = o2v0
    tmps_["7_oo"]  = 1.00 * np.einsum('jkbi,bj->ki',eri["oovo"],t1)

    # rt1 += +1.00 <k,j||b,i> t1(a,k) t1(b,j)
    # flops: o1v1 += o2v1
    #  mems: o1v1 += o1v1
    rt1 -= np.einsum('ak,ki->ai',t1,tmps_["7_oo"])

    # rt2 += +1.00 P(i,j) <l,k||c,j> t2(a,b,i,l) t1(c,k)
    # flops: o2v2 += o3v2
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 1.00 * np.einsum('abil,lj->abij',t2,tmps_["7_oo"])
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # flops: o4v0  = o4v1
    #  mems: o4v0  = o4v0
    tmps_["8_oooo"]  = 1.00 * np.einsum('di,jkld->ijkl',t1,tmps_["6_ooov"])

    # rt2 += -0.50 <l,k||c,d> t2(a,b,l,k) t1(c,j) t1(d,i)
    # flops: o2v2 += o4v2
    #  mems: o2v2 += o2v2
    rt2 += 0.50 * np.einsum('ablk,ijkl->abij',t2,tmps_["8_oooo"])

    # rt2 += +1.00 <l,k||c,d> t1(a,k) t1(b,l) t1(c,j) t1(d,i)
    # flops: o2v2 += o4v1 o3v2
    #  mems: o2v2 += o3v1 o2v2
    rt2 -= einsum('aijb->abij', np.einsum('ak,ijkl,bl->aijb',t1,tmps_["8_oooo"],t1,optimize='optimal') )
    del tmps_["8_oooo"]

    # flops: o4v0  = o4v1
    #  mems: o4v0  = o4v0
    tmps_["9_oooo"]  = 1.00 * np.einsum('dk,lmdi->klmi',t1,eri["oovo"])

    # rt2 += +0.50 P(i,j) <l,k||c,j> t2(a,b,l,k) t1(c,i)
    # flops: o2v2 += o4v2
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 0.50 * np.einsum('ablk,iklj->abij',t2,tmps_["9_oooo"])
    rt2 -= np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 += einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += -1.00 P(i,j) <l,k||c,j> t1(a,k) t1(b,l) t1(c,i)
    # flops: o2v2 += o4v1 o3v2
    #  mems: o2v2 += o3v1 o2v2
    tmps_["perm_vvoo"]  = 1.00 * einsum('aijb->abij', np.einsum('ak,iklj,bl->aijb',t1,tmps_["9_oooo"],t1,optimize='optimal') )
    rt2 += np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 -= einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]
    del tmps_["9_oooo"]

    # flops: o4v0  = o4v2
    #  mems: o4v0  = o4v0
    tmps_["10_oooo"]  = 1.00 * np.einsum('cdij,klcd->ijkl',t2,eri["oovv"])

    # rt2 += +0.25 <l,k||c,d> t2(a,b,l,k) t2(c,d,i,j)
    # flops: o2v2 += o4v2
    #  mems: o2v2 += o2v2
    rt2 -= 0.25 * np.einsum('ablk,ijkl->abij',t2,tmps_["10_oooo"])

    # rt2 += -0.50 <l,k||c,d> t1(a,k) t1(b,l) t2(c,d,i,j)
    # flops: o2v2 += o4v1 o3v2
    #  mems: o2v2 += o3v1 o2v2
    rt2 += 0.50 * einsum('aijb->abij', np.einsum('ak,ijkl,bl->aijb',t1,tmps_["10_oooo"],t1,optimize='optimal') )
    del tmps_["10_oooo"]

    # flops: o2v2  = o2v3
    #  mems: o2v2  = o2v2
    tmps_["11_vovo"]  = 1.00 * np.einsum('akcd,cj->akdj',eri["vovv"],t1)

    # rt2 += -1.00 P(i,j) P(a,b) <k,a||c,d> t2(d,b,i,k) t1(c,j)
    # flops: o2v2 += o3v3
    #  mems: o2v2 += o2v2
    tmps_["perm_vvoo"]  = 1.00 * einsum('ajbi->abij', np.einsum('akdj,dbik->ajbi',tmps_["11_vovo"],t2) )
    rt2 += np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 -= einsum('abji->abij', np.einsum('abji->abji',tmps_["perm_vvoo"]) )
    rt2 -= einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    rt2 += einsum('baji->abij', np.einsum('baji->baji',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # rt2 += -1.00 P(a,b) <k,a||c,d> t1(b,k) t1(c,j) t1(d,i)
    # flops: o2v2 += o3v2 o3v2
    #  mems: o2v2 += o3v1 o2v2
    tmps_["perm_vvoo"]  = 1.00 * einsum('ajib->abij', np.einsum('akdj,di,bk->ajib',tmps_["11_vovo"],t1,t1,optimize='optimal') )
    rt2 += np.einsum('abij->abij',tmps_["perm_vvoo"])
    rt2 -= einsum('baij->abij', np.einsum('baij->baij',tmps_["perm_vvoo"]) )
    del tmps_["perm_vvoo"]

    # flops: o5v1  = o5v3
    #  mems: o5v1  = o5v1
    tmps_["12_vooooo"]  = 1.00 * np.einsum('decijk,lmde->cijklm',t3,eri["oovv"])

    # flops: o3v3  = o3v5 o5v3 o3v3 o5v2 o4v3 o3v3
    #  mems: o3v3  = o3v3 o3v3 o3v3 o4v2 o3v3 o3v3
    tmps_["13_vvvooo"]  = 1.00 * einsum('aijkbc->bcaijk', np.einsum('deaijk,bcde->aijkbc',t3,eri["vvvv"]) )
    tmps_["13_vvvooo"] -= 0.50 * np.einsum('bcml,aijklm->bcaijk',t2,tmps_["12_vooooo"])
    tmps_["13_vvvooo"] += einsum('caijkb->bcaijk', np.einsum('cm,aijklm,bl->caijkb',t1,tmps_["12_vooooo"],t1,optimize='optimal') )
    del tmps_["12_vooooo"]

    # rt3  = +0.50 <b,c||d,e> t3(d,e,a,i,j,k)
    #     += +0.25 <m,l||d,e> t3(d,e,a,i,j,k) t2(b,c,m,l)
    #     += -0.50 <m,l||d,e> t3(d,e,a,i,j,k) t1(b,l) t1(c,m)
    rt3  = 0.50 * einsum('bcaijk->abcijk', tmps_["13_vvvooo"] )

    # rt3 += +0.50 P(b,c) <a,b||d,e> t3(d,e,c,i,j,k)
    #     += +0.25 P(b,c) <m,l||d,e> t2(a,b,m,l) t3(d,e,c,i,j,k)
    #     += -0.50 P(b,c) <m,l||d,e> t1(a,l) t1(b,m) t3(d,e,c,i,j,k)
    rt3 += 0.50 * tmps_["13_vvvooo"]
    rt3 -= 0.50 * einsum('acbijk->abcijk', tmps_["13_vvvooo"] )
    del tmps_["13_vvvooo"]

    # flops: o3v3  = o4v3 o3v4 o3v4 o3v3 o3v3 o4v3 o3v3 o3v3 o4v3 o4v3 o3v3 o4v1 o4v3 o3v2 o4v3 o3v3 o3v2 o4v3 o3v3 o3v2 o4v3 o3v3 o4v2 o4v1 o4v3 o3v3 o2v4 o3v4 o3v3 o4v1 o4v1 o4v3 o3v3 o3v3
    #  mems: o3v3  = o3v3 o1v3 o3v3 o3v3 o3v1 o3v3 o3v3 o2v2 o4v2 o3v3 o3v3 o3v1 o3v3 o3v1 o3v3 o3v3 o3v1 o3v3 o3v3 o3v1 o3v3 o3v3 o4v0 o3v1 o3v3 o3v3 o1v3 o3v3 o3v3 o4v0 o3v1 o3v3 o3v3 o3v3
    tmps_["14_voovvo"]  = 1.00 * np.einsum('clij,abkl->cijabk',eri["vooo"],t2)
    tmps_["14_voovvo"] += 0.50 * einsum('abkcij->cijabk', np.einsum('lmde,eabkml,dcij->abkcij',eri["oovv"],t3,t2,optimize='optimal') )
    tmps_["14_voovvo"] += 0.50 * np.einsum('clde,deij,abkl->cijabk',eri["vovv"],t2,t2,optimize='optimal')
    tmps_["14_voovvo"] += einsum('bkaijc->cijabk', np.einsum('lmde,ebkm,daij,cl->bkaijc',eri["oovv"],t2,t2,t1,optimize='optimal') )
    tmps_["14_voovvo"] -= einsum('ijcabk->cijabk', np.einsum('lmij,cl,abkm->ijcabk',eri["oooo"],t1,t2,optimize='optimal') )
    tmps_["14_voovvo"] -= np.einsum('ld,dcij,abkl->cijabk',f["ov"],t2,t2,optimize='optimal')
    tmps_["14_voovvo"] -= np.einsum('me,ecij,abkm->cijabk',tmps_["1_ov"],t2,t2,optimize='optimal')
    tmps_["14_voovvo"] -= einsum('cjiabk->cijabk', np.einsum('clej,ei,abkl->cjiabk',tmps_["11_vovo"],t1,t2,optimize='optimal') )
    tmps_["14_voovvo"] -= 0.50 * einsum('ijcabk->cijabk', np.einsum('lmde,deij,cl,abkm->ijcabk',eri["oovv"],t2,t1,t2,optimize='optimal') )
    tmps_["14_voovvo"] -= einsum('cbkaij->cijabk', np.einsum('clde,ebkl,daij->cbkaij',eri["vovv"],t2,t2,optimize='optimal') )
    tmps_["14_voovvo"] += einsum('jicabk->cijabk', np.einsum('jlme,ei,cl,abkm->jicabk',tmps_["6_ooov"],t1,t1,t2,optimize='optimal') )

    # rt3 += -1.00 P(i,j) <l,c||j,k> t2(a,b,i,l)
    #     += -0.50 P(i,j) <m,l||d,e> t3(e,a,b,i,m,l) t2(d,c,j,k)
    #     += -0.50 P(i,j) <l,c||d,e> t2(a,b,i,l) t2(d,e,j,k)
    #     += -1.00 P(i,j) <m,l||d,e> t2(d,a,j,k) t2(e,b,i,m) t1(c,l)
    #     += +1.00 P(i,j) <m,l||j,k> t2(a,b,i,m) t1(c,l)
    #     += -1.00 P(i,j) f(l,d) t2(a,b,i,l) t2(d,c,j,k)
    #     += +1.00 P(i,j) <m,l||d,e> t2(a,b,i,m) t2(e,c,j,k) t1(d,l)
    #     += +1.00 P(i,j) <l,c||d,e> t2(a,b,i,l) t1(d,k) t1(e,j)
    #     += +0.50 P(i,j) <m,l||d,e> t2(a,b,i,m) t1(c,l) t2(d,e,j,k)
    #     += +1.00 P(i,j) <l,c||d,e> t2(d,a,j,k) t2(e,b,i,l)
    #     += -1.00 P(i,j) <m,l||d,e> t2(a,b,i,m) t1(c,l) t1(d,k) t1(e,j)
    rt3 += einsum('cjkabi->abcijk', tmps_["14_voovvo"] )
    rt3 -= einsum('cikabj->abcijk', tmps_["14_voovvo"] )

    # rt3 += -1.00 P(a,b) <l,a||i,j> t2(b,c,k,l)
    #     += -0.50 P(a,b) <m,l||d,e> t2(d,a,i,j) t3(e,b,c,k,m,l)
    #     += -0.50 P(a,b) <l,a||d,e> t2(b,c,k,l) t2(d,e,i,j)
    #     += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,j) t2(e,c,k,m)
    #     += +1.00 P(a,b) <m,l||i,j> t1(a,l) t2(b,c,k,m)
    #     += -1.00 P(a,b) f(l,d) t2(d,a,i,j) t2(b,c,k,l)
    #     += +1.00 P(a,b) <m,l||d,e> t2(e,a,i,j) t2(b,c,k,m) t1(d,l)
    #     += +1.00 P(a,b) <l,a||d,e> t2(b,c,k,l) t1(d,j) t1(e,i)
    #     += +0.50 P(a,b) <m,l||d,e> t1(a,l) t2(b,c,k,m) t2(d,e,i,j)
    #     += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,j) t2(e,c,k,l)
    #     += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(b,c,k,m) t1(d,j) t1(e,i)
    rt3 += einsum('aijbck->abcijk', tmps_["14_voovvo"] )
    rt3 -= einsum('bijack->abcijk', tmps_["14_voovvo"] )

    # rt3 += -1.00 P(i,j) P(a,b) <l,a||j,k> t2(b,c,i,l)
    #     += -0.50 P(i,j) P(a,b) <m,l||d,e> t2(d,a,j,k) t3(e,b,c,i,m,l)
    #     += -0.50 P(i,j) P(a,b) <l,a||d,e> t2(b,c,i,l) t2(d,e,j,k)
    #     += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,j,k) t2(e,c,i,m)
    #     += +1.00 P(i,j) P(a,b) <m,l||j,k> t1(a,l) t2(b,c,i,m)
    #     += -1.00 P(i,j) P(a,b) f(l,d) t2(d,a,j,k) t2(b,c,i,l)
    #     += +1.00 P(i,j) P(a,b) <m,l||d,e> t2(e,a,j,k) t2(b,c,i,m) t1(d,l)
    #     += +1.00 P(i,j) P(a,b) <l,a||d,e> t2(b,c,i,l) t1(d,k) t1(e,j)
    #     += +0.50 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(b,c,i,m) t2(d,e,j,k)
    #     += +1.00 P(i,j) P(a,b) <l,a||d,e> t2(d,b,j,k) t2(e,c,i,l)
    #     += -1.00 P(i,j) P(a,b) <m,l||d,e> t1(a,l) t2(b,c,i,m) t1(d,k) t1(e,j)
    rt3 += einsum('ajkbci->abcijk', tmps_["14_voovvo"] )
    rt3 -= einsum('aikbcj->abcijk', tmps_["14_voovvo"] )
    rt3 -= einsum('bjkaci->abcijk', tmps_["14_voovvo"] )
    rt3 += einsum('bikacj->abcijk', tmps_["14_voovvo"] )

    # rt3 += -1.00 <l,c||i,j> t2(a,b,k,l)
    #     += -0.50 <m,l||d,e> t3(e,a,b,k,m,l) t2(d,c,i,j)
    #     += -0.50 <l,c||d,e> t2(a,b,k,l) t2(d,e,i,j)
    #     += -1.00 <m,l||d,e> t2(d,a,i,j) t2(e,b,k,m) t1(c,l)
    #     += +1.00 <m,l||i,j> t2(a,b,k,m) t1(c,l)
    #     += -1.00 f(l,d) t2(a,b,k,l) t2(d,c,i,j)
    #     += +1.00 <m,l||d,e> t2(a,b,k,m) t2(e,c,i,j) t1(d,l)
    #     += +1.00 <l,c||d,e> t2(a,b,k,l) t1(d,j) t1(e,i)
    #     += +0.50 <m,l||d,e> t2(a,b,k,m) t1(c,l) t2(d,e,i,j)
    #     += +1.00 <l,c||d,e> t2(d,a,i,j) t2(e,b,k,l)
    #     += -1.00 <m,l||d,e> t2(a,b,k,m) t1(c,l) t1(d,j) t1(e,i)
    rt3 += einsum('cijabk->abcijk', tmps_["14_voovvo"] )
    del tmps_["14_voovvo"]

    # flops: o3v3  = o4v2 o4v1 o4v0 o4v0 o5v3
    #  mems: o3v3  = o4v0 o4v0 o4v0 o4v0 o3v3
    tmps_["15_vvvooo"]  = 1.00 * np.einsum('abckml,ijlm->abckij',t3,(np.einsum('deij,lmde->ijlm',t2,eri["oovv"]) + np.einsum('lmij->ijlm',2.00 * np.einsum('lmij->lmij',(np.einsum('lmij->lmij',eri["oooo"]) + np.einsum('ijlm->lmij',-1.00 * np.einsum('ei,jlme->ijlm',t1,tmps_["6_ooov"])))))))
    del tmps_["6_ooov"]

    # rt3 += +0.25 P(i,j) <m,l||d,e> t3(a,b,c,i,m,l) t2(d,e,j,k)
    #     += +0.50 P(i,j) <m,l||j,k> t3(a,b,c,i,m,l)
    #     += -0.50 P(i,j) <m,l||d,e> t3(a,b,c,i,m,l) t1(d,k) t1(e,j)
    rt3 -= 0.25 * tmps_["15_vvvooo"]
    rt3 += 0.25 * einsum('abcjik->abcijk', tmps_["15_vvvooo"] )

    # rt3 += +0.25 <m,l||d,e> t3(a,b,c,k,m,l) t2(d,e,i,j)
    #     += +0.50 <m,l||i,j> t3(a,b,c,k,m,l)
    #     += -0.50 <m,l||d,e> t3(a,b,c,k,m,l) t1(d,j) t1(e,i)
    rt3 -= 0.25 * einsum('abckij->abcijk', tmps_["15_vvvooo"] )
    del tmps_["15_vvvooo"]

    # flops: o3v3  = o4v3 o2v1 o4v3 o3v3 o3v2 o4v3 o3v3 o2v1 o4v3 o3v3 o4v3 o3v3
    #  mems: o3v3  = o3v3 o2v0 o3v3 o3v3 o2v0 o3v3 o3v3 o2v0 o3v3 o3v3 o3v3 o3v3
    tmps_["16_ovvvoo"]  = 1.00 * np.einsum('li,abcjkl->iabcjk',f["oo"],t3)
    tmps_["16_ovvvoo"] += np.einsum('ei,me,abcjkm->iabcjk',t1,tmps_["1_ov"],t3,optimize='optimal')
    tmps_["16_ovvvoo"] -= 0.50 * np.einsum('lmde,deil,abcjkm->iabcjk',eri["oovv"],t2,t3,optimize='optimal')
    tmps_["16_ovvvoo"] += np.einsum('ld,di,abcjkl->iabcjk',f["ov"],t1,t3,optimize='optimal')
    tmps_["16_ovvvoo"] += einsum('abcjki->iabcjk', np.einsum('abcjkm,mi->abcjki',t3,tmps_["7_oo"]) )
    del tmps_["7_oo"]

    # rt3 += -1.00 P(j,k) f(l,k) t3(a,b,c,i,j,l)
    #     += +1.00 P(j,k) <m,l||d,e> t3(a,b,c,i,j,m) t1(d,l) t1(e,k)
    #     += -0.50 P(j,k) <m,l||d,e> t3(a,b,c,i,j,m) t2(d,e,k,l)
    #     += -1.00 P(j,k) f(l,d) t3(a,b,c,i,j,l) t1(d,k)
    #     += +1.00 P(j,k) <m,l||d,k> t3(a,b,c,i,j,m) t1(d,l)
    rt3 -= einsum('kabcij->abcijk', tmps_["16_ovvvoo"] )
    rt3 += einsum('jabcik->abcijk', tmps_["16_ovvvoo"] )

    # rt3 += -1.00 f(l,i) t3(a,b,c,j,k,l)
    #     += +1.00 <m,l||d,e> t3(a,b,c,j,k,m) t1(d,l) t1(e,i)
    #     += -0.50 <m,l||d,e> t3(a,b,c,j,k,m) t2(d,e,i,l)
    #     += -1.00 f(l,d) t3(a,b,c,j,k,l) t1(d,i)
    #     += +1.00 <m,l||d,i> t3(a,b,c,j,k,m) t1(d,l)
    rt3 -= einsum('iabcjk->abcijk', tmps_["16_ovvvoo"] )
    del tmps_["16_ovvvoo"]

    # flops: o3v3  = o4v3 o4v3 o4v3 o4v3 o3v3 o2v3 o3v4 o3v3 o3v4 o3v3 o3v4 o3v3
    #  mems: o3v3  = o4v2 o3v3 o4v2 o3v3 o3v3 o0v2 o3v3 o3v3 o3v3 o3v3 o3v3 o3v3
    tmps_["17_vvooov"]  = 1.00 * np.einsum('ld,dabijk,cl->abijkc',f["ov"],t3,t1,optimize='optimal')
    tmps_["17_vvooov"] += np.einsum('eabijk,me,cm->abijkc',t3,tmps_["1_ov"],t1,optimize='optimal')
    tmps_["17_vvooov"] -= 0.50 * einsum('cabijk->abijkc', np.einsum('lmde,dcml,eabijk->cabijk',eri["oovv"],t2,t3,optimize='optimal') )
    tmps_["17_vvooov"] -= einsum('cabijk->abijkc', np.einsum('cd,dabijk->cabijk',f["vv"],t3) )
    tmps_["17_vvooov"] += np.einsum('eabijk,ce->abijkc',t3,tmps_["4_vv"])
    del tmps_["4_vv"]
    del tmps_["1_ov"]

    # rt3 += -1.00 f(l,d) t3(d,a,b,i,j,k) t1(c,l)
    #     += +1.00 <m,l||d,e> t3(e,a,b,i,j,k) t1(c,m) t1(d,l)
    #     += -0.50 <m,l||d,e> t3(e,a,b,i,j,k) t2(d,c,m,l)
    #     += +1.00 f(c,d) t3(d,a,b,i,j,k)
    #     += +1.00 <l,c||d,e> t3(e,a,b,i,j,k) t1(d,l)
    rt3 -= einsum('abijkc->abcijk', tmps_["17_vvooov"] )

    # rt3 += -1.00 P(a,b) f(l,d) t1(a,l) t3(d,b,c,i,j,k)
    #     += +1.00 P(a,b) <m,l||d,e> t1(a,m) t3(e,b,c,i,j,k) t1(d,l)
    #     += -0.50 P(a,b) <m,l||d,e> t2(d,a,m,l) t3(e,b,c,i,j,k)
    #     += +1.00 P(a,b) f(a,d) t3(d,b,c,i,j,k)
    #     += +1.00 P(a,b) <l,a||d,e> t3(e,b,c,i,j,k) t1(d,l)
    rt3 -= einsum('bcijka->abcijk', tmps_["17_vvooov"] )
    rt3 += einsum('acijkb->abcijk', tmps_["17_vvooov"] )
    del tmps_["17_vvooov"]

    # flops: o3v3  = o4v1 o4v1 o4v3 o4v2 o4v3 o3v3
    #  mems: o3v3  = o4v0 o3v1 o3v3 o3v1 o3v3 o3v3
    tmps_["18_oovvvo"]  = 1.00 * np.einsum('lmdj,dk,al,bcim->jkabci',eri["oovo"],t1,t1,t2,optimize='optimal')
    tmps_["18_oovvvo"] += einsum('jakbci->jkabci', np.einsum('lmdj,dakl,bcim->jakbci',eri["oovo"],t2,t2,optimize='optimal') )

    # rt3 += +1.00 P(j,k) P(a,b) <m,l||d,i> t1(a,l) t2(b,c,j,m) t1(d,k)
    #     += +1.00 P(j,k) P(a,b) <m,l||d,i> t2(d,a,k,l) t2(b,c,j,m)
    rt3 -= einsum('ikabcj->abcijk', tmps_["18_oovvvo"] )
    rt3 += einsum('ijabck->abcijk', tmps_["18_oovvvo"] )
    rt3 += einsum('ikbacj->abcijk', tmps_["18_oovvvo"] )
    rt3 -= einsum('ijback->abcijk', tmps_["18_oovvvo"] )

    # rt3 += -1.00 P(i,k) P(a,b) <m,l||d,j> t1(a,l) t2(b,c,i,m) t1(d,k)
    #     += -1.00 P(i,k) P(a,b) <m,l||d,j> t2(d,a,k,l) t2(b,c,i,m)
    rt3 += einsum('jkabci->abcijk', tmps_["18_oovvvo"] )
    rt3 -= einsum('jiabck->abcijk', tmps_["18_oovvvo"] )
    rt3 -= einsum('jkbaci->abcijk', tmps_["18_oovvvo"] )
    rt3 += einsum('jiback->abcijk', tmps_["18_oovvvo"] )

    # rt3 += +1.00 P(i,j) <m,l||d,k> t2(a,b,i,m) t1(c,l) t1(d,j)
    #     += +1.00 P(i,j) <m,l||d,k> t2(a,b,i,m) t2(d,c,j,l)
    rt3 -= einsum('kjcabi->abcijk', tmps_["18_oovvvo"] )
    rt3 += einsum('kicabj->abcijk', tmps_["18_oovvvo"] )

    # rt3 += -1.00 P(i,k) <m,l||d,j> t2(a,b,i,m) t1(c,l) t1(d,k)
    #     += -1.00 P(i,k) <m,l||d,j> t2(a,b,i,m) t2(d,c,k,l)
    rt3 += einsum('jkcabi->abcijk', tmps_["18_oovvvo"] )
    rt3 -= einsum('jicabk->abcijk', tmps_["18_oovvvo"] )

    # rt3 += +1.00 P(i,j) P(a,b) <m,l||d,k> t1(a,l) t2(b,c,i,m) t1(d,j)
    #     += +1.00 P(i,j) P(a,b) <m,l||d,k> t2(d,a,j,l) t2(b,c,i,m)
    rt3 -= einsum('kjabci->abcijk', tmps_["18_oovvvo"] )
    rt3 += einsum('kiabcj->abcijk', tmps_["18_oovvvo"] )
    rt3 += einsum('kjbaci->abcijk', tmps_["18_oovvvo"] )
    rt3 -= einsum('kibacj->abcijk', tmps_["18_oovvvo"] )

    # rt3 += +1.00 P(j,k) <m,l||d,i> t2(a,b,j,m) t1(c,l) t1(d,k)
    #     += +1.00 P(j,k) <m,l||d,i> t2(a,b,j,m) t2(d,c,k,l)
    rt3 -= einsum('ikcabj->abcijk', tmps_["18_oovvvo"] )
    rt3 += einsum('ijcabk->abcijk', tmps_["18_oovvvo"] )
    del tmps_["18_oovvvo"]

    # flops: o3v3  = o5v2 o5v2 o4v3 o3v4 o3v3 o1v4 o3v4 o3v3 o4v3 o4v3 o3v3 o3v2 o5v2 o5v2 o4v3 o3v3
    #  mems: o3v3  = o5v1 o4v2 o3v3 o3v3 o3v3 o1v3 o3v3 o3v3 o3v1 o3v3 o3v3 o3v1 o5v1 o4v2 o3v3 o3v3
    tmps_["19_ovoovv"]  = 1.00 * np.einsum('lmdi,dajk,bl,cm->iajkbc',eri["oovo"],t2,t1,t1,optimize='optimal')
    tmps_["19_ovoovv"] += einsum('bciajk->iajkbc', np.einsum('bcdi,dajk->bciajk',eri["vvvo"],t2) )
    tmps_["19_ovoovv"] -= einsum('bciajk->iajkbc', np.einsum('bcde,di,eajk->bciajk',eri["vvvv"],t1,t2,optimize='optimal') )
    tmps_["19_ovoovv"] -= 0.50 * einsum('ajkbci->iajkbc', np.einsum('lmde,deajkm,bcil->ajkbci',eri["oovv"],t3,t2,optimize='optimal') )
    tmps_["19_ovoovv"] -= np.einsum('lmde,di,eajk,bl,cm->iajkbc',eri["oovv"],t1,t2,t1,t1,optimize='optimal')

    # rt3 += +1.00 <m,l||d,i> t2(d,a,j,k) t1(b,l) t1(c,m)
    #     += -1.00 <b,c||d,i> t2(d,a,j,k)
    #     += +1.00 <b,c||d,e> t2(e,a,j,k) t1(d,i)
    #     += -0.50 <m,l||d,e> t3(d,e,a,j,k,m) t2(b,c,i,l)
    #     += -1.00 <m,l||d,e> t2(e,a,j,k) t1(b,l) t1(c,m) t1(d,i)
    rt3 -= einsum('iajkbc->abcijk', tmps_["19_ovoovv"] )

    # rt3 += +1.00 P(j,k) <m,l||d,k> t2(d,a,i,j) t1(b,l) t1(c,m)
    #     += -1.00 P(j,k) <b,c||d,k> t2(d,a,i,j)
    #     += +1.00 P(j,k) <b,c||d,e> t2(e,a,i,j) t1(d,k)
    #     += -0.50 P(j,k) <m,l||d,e> t3(d,e,a,i,j,m) t2(b,c,k,l)
    #     += -1.00 P(j,k) <m,l||d,e> t2(e,a,i,j) t1(b,l) t1(c,m) t1(d,k)
    rt3 -= einsum('kaijbc->abcijk', tmps_["19_ovoovv"] )
    rt3 += einsum('jaikbc->abcijk', tmps_["19_ovoovv"] )

    # rt3 += +1.00 P(b,c) <m,l||d,i> t1(a,l) t1(b,m) t2(d,c,j,k)
    #     += -1.00 P(b,c) <a,b||d,i> t2(d,c,j,k)
    #     += +1.00 P(b,c) <a,b||d,e> t2(e,c,j,k) t1(d,i)
    #     += -0.50 P(b,c) <m,l||d,e> t2(a,b,i,l) t3(d,e,c,j,k,m)
    #     += -1.00 P(b,c) <m,l||d,e> t1(a,l) t1(b,m) t2(e,c,j,k) t1(d,i)
    rt3 -= einsum('icjkab->abcijk', tmps_["19_ovoovv"] )
    rt3 += einsum('ibjkac->abcijk', tmps_["19_ovoovv"] )

    # rt3 += +1.00 P(j,k) P(b,c) <m,l||d,k> t1(a,l) t1(b,m) t2(d,c,i,j)
    #     += -1.00 P(j,k) P(b,c) <a,b||d,k> t2(d,c,i,j)
    #     += +1.00 P(j,k) P(b,c) <a,b||d,e> t2(e,c,i,j) t1(d,k)
    #     += -0.50 P(j,k) P(b,c) <m,l||d,e> t2(a,b,k,l) t3(d,e,c,i,j,m)
    #     += -1.00 P(j,k) P(b,c) <m,l||d,e> t1(a,l) t1(b,m) t2(e,c,i,j) t1(d,k)
    rt3 -= einsum('kcijab->abcijk', tmps_["19_ovoovv"] )
    rt3 += einsum('jcikab->abcijk', tmps_["19_ovoovv"] )
    rt3 += einsum('kbijac->abcijk', tmps_["19_ovoovv"] )
    rt3 -= einsum('jbikac->abcijk', tmps_["19_ovoovv"] )
    del tmps_["19_ovoovv"]

    # flops: o3v3  = o4v3 o4v3
    #  mems: o3v3  = o4v2 o3v3
    tmps_["20_vovoov"]  = 1.00 * np.einsum('bldi,dcjk,al->bicjka',eri["vovo"],t2,t1,optimize='optimal')

    # rt3 += -1.00 P(j,k) P(b,c) <l,a||d,k> t1(b,l) t2(d,c,i,j)
    rt3 += einsum('akcijb->abcijk', tmps_["20_vovoov"] )
    rt3 -= einsum('ajcikb->abcijk', tmps_["20_vovoov"] )
    rt3 -= einsum('akbijc->abcijk', tmps_["20_vovoov"] )
    rt3 += einsum('ajbikc->abcijk', tmps_["20_vovoov"] )

    # rt3 += -1.00 P(b,c) <l,a||d,i> t1(b,l) t2(d,c,j,k)
    rt3 += einsum('aicjkb->abcijk', tmps_["20_vovoov"] )
    rt3 -= einsum('aibjkc->abcijk', tmps_["20_vovoov"] )

    # rt3 += -1.00 P(a,b) <l,c||d,i> t1(a,l) t2(d,b,j,k)
    rt3 += einsum('cibjka->abcijk', tmps_["20_vovoov"] )
    rt3 -= einsum('ciajkb->abcijk', tmps_["20_vovoov"] )

    # rt3 += +1.00 P(a,c) <l,b||d,i> t1(a,l) t2(d,c,j,k)
    rt3 -= einsum('bicjka->abcijk', tmps_["20_vovoov"] )
    rt3 += einsum('biajkc->abcijk', tmps_["20_vovoov"] )

    # rt3 += -1.00 P(j,k) P(a,b) <l,c||d,k> t1(a,l) t2(d,b,i,j)
    rt3 += einsum('ckbija->abcijk', tmps_["20_vovoov"] )
    rt3 -= einsum('cjbika->abcijk', tmps_["20_vovoov"] )
    rt3 -= einsum('ckaijb->abcijk', tmps_["20_vovoov"] )
    rt3 += einsum('cjaikb->abcijk', tmps_["20_vovoov"] )

    # rt3 += +1.00 P(j,k) P(a,c) <l,b||d,k> t1(a,l) t2(d,c,i,j)
    rt3 -= einsum('bkcija->abcijk', tmps_["20_vovoov"] )
    rt3 += einsum('bjcika->abcijk', tmps_["20_vovoov"] )
    rt3 += einsum('bkaijc->abcijk', tmps_["20_vovoov"] )
    rt3 -= einsum('bjaikc->abcijk', tmps_["20_vovoov"] )
    del tmps_["20_vovoov"]

    # flops: o3v3  = o3v2 o4v2 o4v3 o3v2 o4v3 o3v3
    #  mems: o3v3  = o3v1 o3v1 o3v3 o3v1 o3v3 o3v3
    tmps_["21_ovovvo"]  = 1.00 * np.einsum('lmde,di,eakl,bcjm->iakbcj',eri["oovv"],t1,t2,t2,optimize='optimal')
    tmps_["21_ovovvo"] += einsum('aikbcj->iakbcj', np.einsum('aldi,dk,bcjl->aikbcj',eri["vovo"],t1,t2,optimize='optimal') )

    # rt3 += -1.00 P(i,j) P(a,b) <m,l||d,e> t2(e,a,j,l) t2(b,c,i,m) t1(d,k)
    #     += -1.00 P(i,j) P(a,b) <l,a||d,k> t2(b,c,i,l) t1(d,j)
    rt3 += einsum('kajbci->abcijk', tmps_["21_ovovvo"] )
    rt3 -= einsum('kaibcj->abcijk', tmps_["21_ovovvo"] )
    rt3 -= einsum('kbjaci->abcijk', tmps_["21_ovovvo"] )
    rt3 += einsum('kbiacj->abcijk', tmps_["21_ovovvo"] )

    # rt3 += +1.00 P(i,k) P(a,b) <m,l||d,e> t2(e,a,k,l) t2(b,c,i,m) t1(d,j)
    #     += +1.00 P(i,k) P(a,b) <l,a||d,j> t2(b,c,i,l) t1(d,k)
    rt3 -= einsum('jakbci->abcijk', tmps_["21_ovovvo"] )
    rt3 += einsum('jaibck->abcijk', tmps_["21_ovovvo"] )
    rt3 += einsum('jbkaci->abcijk', tmps_["21_ovovvo"] )
    rt3 -= einsum('jbiack->abcijk', tmps_["21_ovovvo"] )

    # rt3 += -1.00 P(i,j) <m,l||d,e> t2(a,b,i,m) t2(e,c,j,l) t1(d,k)
    #     += -1.00 P(i,j) <l,c||d,k> t2(a,b,i,l) t1(d,j)
    rt3 += einsum('kcjabi->abcijk', tmps_["21_ovovvo"] )
    rt3 -= einsum('kciabj->abcijk', tmps_["21_ovovvo"] )

    # rt3 += +1.00 P(i,k) <m,l||d,e> t2(a,b,i,m) t2(e,c,k,l) t1(d,j)
    #     += +1.00 P(i,k) <l,c||d,j> t2(a,b,i,l) t1(d,k)
    rt3 -= einsum('jckabi->abcijk', tmps_["21_ovovvo"] )
    rt3 += einsum('jciabk->abcijk', tmps_["21_ovovvo"] )

    # rt3 += -1.00 P(j,k) P(a,b) <m,l||d,e> t2(e,a,k,l) t2(b,c,j,m) t1(d,i)
    #     += -1.00 P(j,k) P(a,b) <l,a||d,i> t2(b,c,j,l) t1(d,k)
    rt3 += einsum('iakbcj->abcijk', tmps_["21_ovovvo"] )
    rt3 -= einsum('iajbck->abcijk', tmps_["21_ovovvo"] )
    rt3 -= einsum('ibkacj->abcijk', tmps_["21_ovovvo"] )
    rt3 += einsum('ibjack->abcijk', tmps_["21_ovovvo"] )

    # rt3 += -1.00 P(j,k) <m,l||d,e> t2(a,b,j,m) t2(e,c,k,l) t1(d,i)
    #     += -1.00 P(j,k) <l,c||d,i> t2(a,b,j,l) t1(d,k)
    rt3 += einsum('ickabj->abcijk', tmps_["21_ovovvo"] )
    rt3 -= einsum('icjabk->abcijk', tmps_["21_ovovvo"] )
    del tmps_["21_ovovvo"]

    # flops: o3v3  = o2v3 o4v3 o4v3
    #  mems: o3v3  = o2v2 o4v2 o3v3
    tmps_["22_vovoov"]  = 1.00 * np.einsum('alde,di,ecjk,bl->aicjkb',eri["vovv"],t1,t2,t1,optimize='optimal')

    # rt3 += -1.00 P(a,c) <l,b||d,e> t1(a,l) t2(e,c,j,k) t1(d,i)
    rt3 += einsum('bicjka->abcijk', tmps_["22_vovoov"] )
    rt3 -= einsum('biajkc->abcijk', tmps_["22_vovoov"] )

    # rt3 += -1.00 P(j,k) P(a,c) <l,b||d,e> t1(a,l) t2(e,c,i,j) t1(d,k)
    rt3 += einsum('bkcija->abcijk', tmps_["22_vovoov"] )
    rt3 -= einsum('bjcika->abcijk', tmps_["22_vovoov"] )
    rt3 -= einsum('bkaijc->abcijk', tmps_["22_vovoov"] )
    rt3 += einsum('bjaikc->abcijk', tmps_["22_vovoov"] )

    # rt3 += +1.00 P(a,b) <l,c||d,e> t1(a,l) t2(e,b,j,k) t1(d,i)
    rt3 -= einsum('cibjka->abcijk', tmps_["22_vovoov"] )
    rt3 += einsum('ciajkb->abcijk', tmps_["22_vovoov"] )

    # rt3 += +1.00 P(j,k) P(b,c) <l,a||d,e> t1(b,l) t2(e,c,i,j) t1(d,k)
    rt3 -= einsum('akcijb->abcijk', tmps_["22_vovoov"] )
    rt3 += einsum('ajcikb->abcijk', tmps_["22_vovoov"] )
    rt3 += einsum('akbijc->abcijk', tmps_["22_vovoov"] )
    rt3 -= einsum('ajbikc->abcijk', tmps_["22_vovoov"] )

    # rt3 += +1.00 P(j,k) P(a,b) <l,c||d,e> t1(a,l) t2(e,b,i,j) t1(d,k)
    rt3 -= einsum('ckbija->abcijk', tmps_["22_vovoov"] )
    rt3 += einsum('cjbika->abcijk', tmps_["22_vovoov"] )
    rt3 += einsum('ckaijb->abcijk', tmps_["22_vovoov"] )
    rt3 -= einsum('cjaikb->abcijk', tmps_["22_vovoov"] )

    # rt3 += +1.00 P(b,c) <l,a||d,e> t1(b,l) t2(e,c,j,k) t1(d,i)
    rt3 -= einsum('aicjkb->abcijk', tmps_["22_vovoov"] )
    rt3 += einsum('aibjkc->abcijk', tmps_["22_vovoov"] )
    del tmps_["22_vovoov"]

    # flops: o3v3  = o4v1 o5v3
    #  mems: o3v3  = o4v0 o3v3
    tmps_["23_oovvvo"]  = 1.00 * np.einsum('lmdi,dk,abcjml->ikabcj',eri["oovo"],t1,t3,optimize='optimal')

    # rt3 += -0.50 P(i,k) <m,l||d,j> t3(a,b,c,i,m,l) t1(d,k)
    rt3 += 0.50 * einsum('jkabci->abcijk', tmps_["23_oovvvo"] )
    rt3 -= 0.50 * einsum('jiabck->abcijk', tmps_["23_oovvvo"] )

    # rt3 += +0.50 P(i,j) <m,l||d,k> t3(a,b,c,i,m,l) t1(d,j)
    rt3 -= 0.50 * einsum('kjabci->abcijk', tmps_["23_oovvvo"] )
    rt3 += 0.50 * einsum('kiabcj->abcijk', tmps_["23_oovvvo"] )

    # rt3 += +0.50 P(j,k) <m,l||d,i> t3(a,b,c,j,m,l) t1(d,k)
    rt3 -= 0.50 * einsum('ikabcj->abcijk', tmps_["23_oovvvo"] )
    rt3 += 0.50 * einsum('ijabck->abcijk', tmps_["23_oovvvo"] )
    del tmps_["23_oovvvo"]

    # flops: o3v3  = o3v3 o4v3 o4v3 o3v3 o4v4 o3v3 o4v4 o3v3 o3v2 o5v3 o4v3 o3v3 o5v3 o4v3 o3v3 o2v4 o3v4 o3v3 o4v4 o3v3 o3v3 o3v4 o3v3 o3v2 o3v3 o3v4 o3v3
    #  mems: o3v3  = o2v2 o4v2 o3v3 o2v2 o3v3 o3v3 o3v3 o3v3 o3v1 o4v2 o3v3 o3v3 o4v2 o3v3 o3v3 o1v3 o3v3 o3v3 o3v3 o3v3 o1v3 o3v3 o3v3 o3v1 o1v3 o3v3 o3v3
    tmps_["24_vovoov"]  = 1.00 * np.einsum('lmde,daim,ebjk,cl->aibjkc',eri["oovv"],t2,t2,t1,optimize='optimal')
    tmps_["24_vovoov"] -= einsum('ciabjk->aibjkc', np.einsum('lmde,dcil,eabjkm->ciabjk',eri["oovv"],t2,t3,optimize='optimal') )
    tmps_["24_vovoov"] += einsum('abjkci->aibjkc', np.einsum('eabjkl,clei->abjkci',t3,tmps_["11_vovo"]) )
    tmps_["24_vovoov"] -= einsum('iabjkc->aibjkc', np.einsum('lmde,di,eabjkm,cl->iabjkc',eri["oovv"],t1,t3,t1,optimize='optimal') )
    tmps_["24_vovoov"] += einsum('iabjkc->aibjkc', np.einsum('lmdi,dabjkm,cl->iabjkc',eri["oovo"],t3,t1,optimize='optimal') )
    tmps_["24_vovoov"] -= einsum('caibjk->aibjkc', np.einsum('clde,dail,ebjk->caibjk',eri["vovv"],t2,t2,optimize='optimal') )
    tmps_["24_vovoov"] -= einsum('ciabjk->aibjkc', np.einsum('cldi,dabjkl->ciabjk',eri["vovo"],t3) )
    tmps_["24_vovoov"] += 0.50 * einsum('iabcjk->aibjkc', np.einsum('lmdi,abml,dcjk->iabcjk',eri["oovo"],t2,t2,optimize='optimal') )
    tmps_["24_vovoov"] -= 0.50 * einsum('iabcjk->aibjkc', np.einsum('lmde,di,abml,ecjk->iabcjk',eri["oovv"],t1,t2,t2,optimize='optimal') )
    del tmps_["11_vovo"]

    # rt3 += -1.00 P(j,k) <m,l||d,e> t2(d,a,k,m) t2(e,b,i,j) t1(c,l)
    #     += +1.00 P(j,k) <m,l||d,e> t3(e,a,b,i,j,m) t2(d,c,k,l)
    #     += -1.00 P(j,k) <l,c||d,e> t3(e,a,b,i,j,l) t1(d,k)
    #     += +1.00 P(j,k) <m,l||d,e> t3(e,a,b,i,j,m) t1(c,l) t1(d,k)
    #     += -1.00 P(j,k) <m,l||d,k> t3(d,a,b,i,j,m) t1(c,l)
    #     += +1.00 P(j,k) <l,c||d,e> t2(d,a,k,l) t2(e,b,i,j)
    #     += +1.00 P(j,k) <l,c||d,k> t3(d,a,b,i,j,l)
    #     += -0.50 P(j,k) <m,l||d,k> t2(a,b,m,l) t2(d,c,i,j)
    #     += +0.50 P(j,k) <m,l||d,e> t2(a,b,m,l) t2(e,c,i,j) t1(d,k)
    rt3 += einsum('akbijc->abcijk', tmps_["24_vovoov"] )
    rt3 -= einsum('ajbikc->abcijk', tmps_["24_vovoov"] )

    # rt3 += -1.00 <m,l||d,e> t2(d,a,i,m) t2(e,b,j,k) t1(c,l)
    #     += +1.00 <m,l||d,e> t3(e,a,b,j,k,m) t2(d,c,i,l)
    #     += -1.00 <l,c||d,e> t3(e,a,b,j,k,l) t1(d,i)
    #     += +1.00 <m,l||d,e> t3(e,a,b,j,k,m) t1(c,l) t1(d,i)
    #     += -1.00 <m,l||d,i> t3(d,a,b,j,k,m) t1(c,l)
    #     += +1.00 <l,c||d,e> t2(d,a,i,l) t2(e,b,j,k)
    #     += +1.00 <l,c||d,i> t3(d,a,b,j,k,l)
    #     += -0.50 <m,l||d,i> t2(a,b,m,l) t2(d,c,j,k)
    #     += +0.50 <m,l||d,e> t2(a,b,m,l) t2(e,c,j,k) t1(d,i)
    rt3 += einsum('aibjkc->abcijk', tmps_["24_vovoov"] )

    # rt3 += -1.00 P(j,k) P(a,b) <m,l||d,e> t1(a,l) t2(d,b,k,m) t2(e,c,i,j)
    #     += +1.00 P(j,k) P(a,b) <m,l||d,e> t2(d,a,k,l) t3(e,b,c,i,j,m)
    #     += -1.00 P(j,k) P(a,b) <l,a||d,e> t3(e,b,c,i,j,l) t1(d,k)
    #     += +1.00 P(j,k) P(a,b) <m,l||d,e> t1(a,l) t3(e,b,c,i,j,m) t1(d,k)
    #     += -1.00 P(j,k) P(a,b) <m,l||d,k> t1(a,l) t3(d,b,c,i,j,m)
    #     += +1.00 P(j,k) P(a,b) <l,a||d,e> t2(d,b,k,l) t2(e,c,i,j)
    #     += +1.00 P(j,k) P(a,b) <l,a||d,k> t3(d,b,c,i,j,l)
    #     += -0.50 P(j,k) P(a,b) <m,l||d,k> t2(d,a,i,j) t2(b,c,m,l)
    #     += +0.50 P(j,k) P(a,b) <m,l||d,e> t2(e,a,i,j) t2(b,c,m,l) t1(d,k)
    rt3 += einsum('bkcija->abcijk', tmps_["24_vovoov"] )
    rt3 -= einsum('bjcika->abcijk', tmps_["24_vovoov"] )
    rt3 -= einsum('akcijb->abcijk', tmps_["24_vovoov"] )
    rt3 += einsum('ajcikb->abcijk', tmps_["24_vovoov"] )

    # rt3 += -1.00 P(a,b) <m,l||d,e> t1(a,l) t2(d,b,i,m) t2(e,c,j,k)
    #     += +1.00 P(a,b) <m,l||d,e> t2(d,a,i,l) t3(e,b,c,j,k,m)
    #     += -1.00 P(a,b) <l,a||d,e> t3(e,b,c,j,k,l) t1(d,i)
    #     += +1.00 P(a,b) <m,l||d,e> t1(a,l) t3(e,b,c,j,k,m) t1(d,i)
    #     += -1.00 P(a,b) <m,l||d,i> t1(a,l) t3(d,b,c,j,k,m)
    #     += +1.00 P(a,b) <l,a||d,e> t2(d,b,i,l) t2(e,c,j,k)
    #     += +1.00 P(a,b) <l,a||d,i> t3(d,b,c,j,k,l)
    #     += -0.50 P(a,b) <m,l||d,i> t2(d,a,j,k) t2(b,c,m,l)
    #     += +0.50 P(a,b) <m,l||d,e> t2(e,a,j,k) t2(b,c,m,l) t1(d,i)
    rt3 += einsum('bicjka->abcijk', tmps_["24_vovoov"] )
    rt3 -= einsum('aicjkb->abcijk', tmps_["24_vovoov"] )
    del tmps_["24_vovoov"]

    # flops: o3v3  = o4v4 o4v3
    #  mems: o3v3  = o4v2 o3v3
    tmps_["25_vvooov"]  = 1.00 * np.einsum('alde,decijk,bl->acijkb',eri["vovv"],t3,t1,optimize='optimal')

    # rt3 += +0.50 P(a,b) <l,c||d,e> t1(a,l) t3(d,e,b,i,j,k)
    rt3 -= 0.50 * einsum('cbijka->abcijk', tmps_["25_vvooov"] )
    rt3 += 0.50 * einsum('caijkb->abcijk', tmps_["25_vvooov"] )

    # rt3 += -0.50 P(a,c) <l,b||d,e> t1(a,l) t3(d,e,c,i,j,k)
    rt3 += 0.50 * einsum('bcijka->abcijk', tmps_["25_vvooov"] )
    rt3 -= 0.50 * einsum('baijkc->abcijk', tmps_["25_vvooov"] )

    # rt3 += +0.50 P(b,c) <l,a||d,e> t1(b,l) t3(d,e,c,i,j,k)
    rt3 -= 0.50 * einsum('acijkb->abcijk', tmps_["25_vvooov"] )
    rt3 += 0.50 * einsum('abijkc->abcijk', tmps_["25_vvooov"] )
    del tmps_["25_vvooov"]


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
           stopping_eps=1.0E-8, diis_size=None, diis_start_cycle=4):
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

        # diis update
        if diis_size is not None:
            vectorized_iterate = np.hstack(
                (new_singles.flatten(), new_doubles.flatten(), new_triples.flatten()))
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                 error_vec)
            new_singles = new_vectorized_iterate[:t1_dim].reshape(t1.shape)
            new_doubles = new_vectorized_iterate[t1_dim:(t1_dim + t2_dim)].reshape(t2.shape)
            new_triples = new_vectorized_iterate[(t1_dim + t2_dim):].reshape(t3.shape)
            old_vec = new_vectorized_iterate

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
                        stopping_eps=1e-10, diis_size=8, diis_start_cycle=4)


    en = cc_energy(t1f, t2f, fock, g, o, v) 
    print("")
    print("    CCSDT Correlation Energy: {: 20.12f}".format(en - hf_energy))
    print("    CCSDT Total Energy:       {: 20.12f}".format(en + molecule.nuclear_repulsion))
    print("")

    assert np.isclose(en-hf_energy,-0.179049024111075,atol=1e-9)
    assert np.isclose(en+molecule.nuclear_repulsion,-100.008956600850908,atol=1e-9)


if __name__ == "__main__":
    main()



