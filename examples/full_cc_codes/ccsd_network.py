"""
A full working spin-orbital CCSD code generated with pdaggerq

If you want to run the example here you should install pyscf openfermion and openfermion-pyscf
The actual CCSD code (ccsd_energy, singles_residual, doubles_residual, kernel)
do not depend on those packages but you gotta get integrals frome somehwere.

We also check the code by using pyscfs functionality for generating spin-orbital
t-amplitudes from RCCSD.  the main() function is fairly straightforward.
"""
import numpy as np
from numpy import einsum


def ccsd_energy(t1, t2, f, g, o, v):
    """
    < 0 | e(-T) H e(T) | 0> :

    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
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


def residuals(t1, t2, f, eri, o, v):

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

    t1_  = {}
    rt1_ = {}
    t2_  = {}
    rt2_ = {}

    t1_["vo"]   = t1
    t2_["vvoo"] = t2

    perm_tmps_= {}
    perm_tmps_["vvoo"] = np.zeros_like(t2)

    tmps_ = {}

    ##########  Evaluate Equations  ##########

    # rt1_[vo] = +1.00 f(a,i)  // flops: o1v1 | mem: o1v1
    rt1_["vo"] = np.einsum('ai->ai', f_["vo"])

    # rt1_[vo] += -1.00 f(j,i) t1(a,j)  // flops: o1v1 += o2v1 | mem: o1v1 += o1v1
    rt1_["vo"] -= np.einsum('ji,aj->ai', f_["oo"], t1_["vo"])

    # rt1_[vo] += +1.00 f(a,b) t1(b,i)  // flops: o1v1 += o1v2 | mem: o1v1 += o1v1
    rt1_["vo"] += np.einsum('ab,bi->ai', f_["vv"], t1_["vo"])

    # rt1_[vo] += -1.00 f(j,b) t2(b,a,i,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    rt1_["vo"] -= np.einsum('jb,baij->ai', f_["ov"], t2_["vvoo"])

    # rt1_[vo] += -1.00 f(j,b) t1(b,i) t1(a,j)  // flops: o1v1 += o2v1 o2v1 | mem: o1v1 += o2v0 o1v1
    rt1_["vo"] -= np.einsum('jb,bi,aj->ai', f_["ov"], t1_["vo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1)])

    # rt1_[vo] += +1.00 <j,a||b,i> t1(b,j)  // flops: o1v1 += o2v2 | mem: o1v1 += o1v1
    rt1_["vo"] -= np.einsum('ajbi,bj->ai', eri_["vovo"], t1_["vo"])

    # rt1_[vo] += -0.50 <k,j||b,i> t2(b,a,k,j)  // flops: o1v1 += o3v2 | mem: o1v1 += o1v1
    rt1_["vo"] -= 0.50 * np.einsum('kjbi,bakj->ai', eri_["oovo"], t2_["vvoo"])

    # rt1_[vo] += -0.50 <j,a||b,c> t2(b,c,i,j)  // flops: o1v1 += o2v3 | mem: o1v1 += o1v1
    rt1_["vo"] += 0.50 * np.einsum('ajbc,bcij->ai', eri_["vovv"], t2_["vvoo"])

    # rt1_[vo] += +1.00 <k,j||b,c> t1(b,j) t2(c,a,i,k)  // flops: o1v1 += o2v2 o2v2 | mem: o1v1 += o1v1 o1v1
    rt1_["vo"] += np.einsum('kjbc,bj,caik->ai', eri_["oovv"], t1_["vo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1)])

    # rt1_[vo] += +0.50 <k,j||b,c> t1(b,i) t2(c,a,k,j)  // flops: o1v1 += o3v2 o3v2 | mem: o1v1 += o3v1 o1v1
    rt1_["vo"] += 0.50 * np.einsum('kjbc,bi,cakj->ai', eri_["oovv"], t1_["vo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1)])

    # rt1_[vo] += +0.50 <k,j||b,c> t1(a,j) t2(b,c,i,k)  // flops: o1v1 += o3v2 o2v1 | mem: o1v1 += o2v0 o1v1
    rt1_["vo"] += 0.50 * np.einsum('kjbc,bcik,aj->ai', eri_["oovv"], t2_["vvoo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1)])

    # rt1_[vo] += +1.00 <k,j||b,i> t1(b,j) t1(a,k)  // flops: o1v1 += o3v1 o2v1 | mem: o1v1 += o2v0 o1v1
    rt1_["vo"] += np.einsum('kjbi,bj,ak->ai', eri_["oovo"], t1_["vo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1)])

    # rt1_[vo] += +1.00 <j,a||b,c> t1(b,j) t1(c,i)  // flops: o1v1 += o1v3 o1v2 | mem: o1v1 += o0v2 o1v1
    rt1_["vo"] -= np.einsum('ajbc,bj,ci->ai', eri_["vovv"], t1_["vo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1)])

    # rt1_[vo] += +1.00 <k,j||b,c> t1(b,j) t1(c,i) t1(a,k)  // flops: o1v1 += o2v2 o2v1 o2v1 | mem: o1v1 += o1v1 o2v0 o1v1
    rt1_["vo"] += np.einsum('kjbc,bj,ci,ak->ai', eri_["oovv"], t1_["vo"], t1_["vo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1),(0,1)])

    # rt2_[vvoo] = -1.00 P(i,j) f(k,j) t2(a,b,i,k)  // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    perm_tmps_["vvoo"] = np.einsum('kj,abik->abij', f_["oo"], t2_["vvoo"])
    rt2_["vvoo"] = -1.00 * np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('abji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += +1.00 P(a,b) f(a,c) t2(c,b,i,j)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    perm_tmps_["vvoo"] = np.einsum('ac,cbij->abij', f_["vv"], t2_["vvoo"])
    rt2_["vvoo"] += np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('baij->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += -1.00 P(i,j) f(k,c) t1(c,j) t2(a,b,i,k)  // flops: o2v2 += o2v1 o3v2 | mem: o2v2 += o2v0 o2v2
    perm_tmps_["vvoo"] = np.einsum('kc,cj,abik->abij', f_["ov"], t1_["vo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1)])
    rt2_["vvoo"] -= np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('abji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += -1.00 P(a,b) f(k,c) t1(a,k) t2(c,b,i,j)  // flops: o2v2 += o3v2 o3v2 | mem: o2v2 += o3v1 o2v2
    perm_tmps_["vvoo"] = np.einsum('kc,cbij,ak->abij', f_["ov"], t2_["vvoo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1)])
    rt2_["vvoo"] -= np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('baij->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += +1.00 <a,b||i,j>  // flops: o2v2 | mem: o2v2
    rt2_["vvoo"] += np.einsum('abij->abij', eri_["vvoo"])

    # rt2_[vvoo] += +1.00 P(a,b) <k,a||i,j> t1(b,k)  // flops: o2v2 += o3v2 | mem: o2v2 += o2v2
    perm_tmps_["vvoo"] = np.einsum('akij,bk->abij', eri_["vooo"], t1_["vo"])
    rt2_["vvoo"] -= np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('baij->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += +1.00 P(i,j) <a,b||c,j> t1(c,i)  // flops: o2v2 += o2v3 | mem: o2v2 += o2v2
    perm_tmps_["vvoo"] = np.einsum('abcj,ci->abij', eri_["vvvo"], t1_["vo"])
    rt2_["vvoo"] += np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('abji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += +0.50 <l,k||i,j> t2(a,b,l,k)  // flops: o2v2 += o4v2 | mem: o2v2 += o2v2
    rt2_["vvoo"] += 0.50 * np.einsum('lkij,ablk->abij', eri_["oooo"], t2_["vvoo"])

    # rt2_[vvoo] += +1.00 P(i,j) P(a,b) <k,a||c,j> t2(c,b,i,k)  // flops: o2v2 += o3v3 | mem: o2v2 += o2v2
    perm_tmps_["vvoo"] = np.einsum('akcj,cbik->abij', eri_["vovo"], t2_["vvoo"])
    rt2_["vvoo"] -= np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('abji->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('baij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('baji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += +0.50 <a,b||c,d> t2(c,d,i,j)  // flops: o2v2 += o2v4 | mem: o2v2 += o2v2
    rt2_["vvoo"] += 0.50 * np.einsum('abcd,cdij->abij', eri_["vvvv"], t2_["vvoo"])

    # rt2_[vvoo] += +1.00 P(i,j) <l,k||c,j> t1(c,k) t2(a,b,i,l)  // flops: o2v2 += o3v1 o3v2 | mem: o2v2 += o2v0 o2v2
    perm_tmps_["vvoo"] = np.einsum('lkcj,ck,abil->abij', eri_["oovo"], t1_["vo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1)])
    rt2_["vvoo"] += np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('abji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += +0.50 P(i,j) <l,k||c,j> t1(c,i) t2(a,b,l,k)  // flops: o2v2 += o4v1 o4v2 | mem: o2v2 += o4v0 o2v2
    perm_tmps_["vvoo"] = 0.50 * np.einsum('lkcj,ci,ablk->abij', eri_["oovo"], t1_["vo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1)])
    rt2_["vvoo"] += np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('abji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += -1.00 P(i,j) P(a,b) <l,k||c,j> t1(a,k) t2(c,b,i,l)  // flops: o2v2 += o4v2 o3v2 | mem: o2v2 += o3v1 o2v2
    perm_tmps_["vvoo"] = np.einsum('lkcj,cbil,ak->abij', eri_["oovo"], t2_["vvoo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1)])
    rt2_["vvoo"] -= np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('abji->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('baij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('baji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += +1.00 P(a,b) <k,a||c,d> t1(c,k) t2(d,b,i,j)  // flops: o2v2 += o1v3 o2v3 | mem: o2v2 += o0v2 o2v2
    perm_tmps_["vvoo"] = np.einsum('akcd,ck,dbij->abij', eri_["vovv"], t1_["vo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1)])
    rt2_["vvoo"] -= np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('baij->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += -1.00 P(i,j) P(a,b) <k,a||c,d> t1(c,j) t2(d,b,i,k)  // flops: o2v2 += o2v3 o3v3 | mem: o2v2 += o2v2 o2v2
    perm_tmps_["vvoo"] = np.einsum('akcd,cj,dbik->abij', eri_["vovv"], t1_["vo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1)])
    rt2_["vvoo"] += np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('abji->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('baij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('baji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += +0.50 P(a,b) <k,a||c,d> t1(b,k) t2(c,d,i,j)  // flops: o2v2 += o3v3 o3v2 | mem: o2v2 += o3v1 o2v2
    perm_tmps_["vvoo"] = 0.50 * np.einsum('akcd,cdij,bk->abij', eri_["vovv"], t2_["vvoo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1)])
    rt2_["vvoo"] -= np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('baij->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += -1.00 <l,k||i,j> t1(a,k) t1(b,l)  // flops: o2v2 += o4v1 o3v2 | mem: o2v2 += o3v1 o2v2
    rt2_["vvoo"] -= np.einsum('lkij,ak,bl->abij', eri_["oooo"], t1_["vo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1)])

    # rt2_[vvoo] += +1.00 P(i,j) P(a,b) <k,a||c,j> t1(c,i) t1(b,k)  // flops: o2v2 += o3v2 o3v2 | mem: o2v2 += o3v1 o2v2
    perm_tmps_["vvoo"] = np.einsum('akcj,ci,bk->abij', eri_["vovo"], t1_["vo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1)])
    rt2_["vvoo"] -= np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('abji->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('baij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('baji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += -1.00 <a,b||c,d> t1(c,j) t1(d,i)  // flops: o2v2 += o1v4 o2v3 | mem: o2v2 += o1v3 o2v2
    rt2_["vvoo"] -= np.einsum('abcd,cj,di->abij', eri_["vvvv"], t1_["vo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1)])

    # rt2_[vvoo] += -0.50 P(i,j) <l,k||c,d> t2(c,d,j,k) t2(a,b,i,l)  // flops: o2v2 += o3v2 o3v2 | mem: o2v2 += o2v0 o2v2
    perm_tmps_["vvoo"] = 0.50 * np.einsum('lkcd,cdjk,abil->abij', eri_["oovv"], t2_["vvoo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1)])
    rt2_["vvoo"] -= np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('abji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += +0.25 <l,k||c,d> t2(c,d,i,j) t2(a,b,l,k)  // flops: o2v2 += o4v2 o4v2 | mem: o2v2 += o4v0 o2v2
    rt2_["vvoo"] += 0.25 * np.einsum('lkcd,cdij,ablk->abij', eri_["oovv"], t2_["vvoo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1)])

    # rt2_[vvoo] += -0.50 <l,k||c,d> t2(c,a,l,k) t2(d,b,i,j)  // flops: o2v2 += o2v3 o2v3 | mem: o2v2 += o0v2 o2v2
    rt2_["vvoo"] -= 0.50 * np.einsum('lkcd,calk,dbij->abij', eri_["oovv"], t2_["vvoo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1)])

    # rt2_[vvoo] += +1.00 P(i,j) <l,k||c,d> t2(c,a,j,k) t2(d,b,i,l)  // flops: o2v2 += o3v3 o3v3 | mem: o2v2 += o2v2 o2v2
    perm_tmps_["vvoo"] = np.einsum('lkcd,cajk,dbil->abij', eri_["oovv"], t2_["vvoo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1)])
    rt2_["vvoo"] += np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('abji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += -0.50 <l,k||c,d> t2(c,a,i,j) t2(d,b,l,k)  // flops: o2v2 += o2v3 o2v3 | mem: o2v2 += o0v2 o2v2
    rt2_["vvoo"] -= 0.50 * np.einsum('lkcd,dblk,caij->abij', eri_["oovv"], t2_["vvoo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1)])

    # rt2_[vvoo] += +1.00 P(i,j) <l,k||c,d> t1(c,k) t1(d,j) t2(a,b,i,l)  // flops: o2v2 += o2v2 o2v1 o3v2 | mem: o2v2 += o1v1 o2v0 o2v2
    perm_tmps_["vvoo"] = np.einsum('lkcd,ck,dj,abil->abij', eri_["oovv"], t1_["vo"], t1_["vo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1),(0,1)])
    rt2_["vvoo"] += np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('abji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += +1.00 P(a,b) <l,k||c,d> t1(c,k) t1(a,l) t2(d,b,i,j)  // flops: o2v2 += o2v2 o3v2 o3v2 | mem: o2v2 += o1v1 o3v1 o2v2
    perm_tmps_["vvoo"] = np.einsum('lkcd,ck,dbij,al->abij', eri_["oovv"], t1_["vo"], t2_["vvoo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1),(0,1)])
    rt2_["vvoo"] += np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('baij->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += -0.50 <l,k||c,d> t1(c,j) t1(d,i) t2(a,b,l,k)  // flops: o2v2 += o3v2 o4v1 o4v2 | mem: o2v2 += o3v1 o4v0 o2v2
    rt2_["vvoo"] -= 0.50 * np.einsum('lkcd,cj,di,ablk->abij', eri_["oovv"], t1_["vo"], t1_["vo"], t2_["vvoo"], optimize=['einsum_path',(0,1),(0,1),(0,1)])

    # rt2_[vvoo] += +1.00 P(i,j) P(a,b) <l,k||c,d> t1(c,j) t1(a,k) t2(d,b,i,l)  // flops: o2v2 += o3v2 o4v2 o3v2 | mem: o2v2 += o3v1 o3v1 o2v2
    perm_tmps_["vvoo"] = np.einsum('lkcd,cj,dbil,ak->abij', eri_["oovv"], t1_["vo"], t2_["vvoo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1),(0,1)])
    rt2_["vvoo"] += np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('abji->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('baij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('baji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += -0.50 <l,k||c,d> t1(a,k) t1(b,l) t2(c,d,i,j)  // flops: o2v2 += o4v2 o4v1 o3v2 | mem: o2v2 += o4v0 o3v1 o2v2
    rt2_["vvoo"] -= 0.50 * np.einsum('lkcd,cdij,ak,bl->abij', eri_["oovv"], t2_["vvoo"], t1_["vo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1),(0,1)])

    # rt2_[vvoo] += -1.00 P(i,j) <l,k||c,j> t1(c,i) t1(a,k) t1(b,l)  // flops: o2v2 += o4v1 o4v1 o3v2 | mem: o2v2 += o4v0 o3v1 o2v2
    perm_tmps_["vvoo"] = np.einsum('lkcj,ci,ak,bl->abij', eri_["oovo"], t1_["vo"], t1_["vo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1),(0,1)])
    rt2_["vvoo"] -= np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] += np.einsum('abji->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += -1.00 P(a,b) <k,a||c,d> t1(c,j) t1(d,i) t1(b,k)  // flops: o2v2 += o2v3 o3v2 o3v2 | mem: o2v2 += o2v2 o3v1 o2v2
    perm_tmps_["vvoo"] = np.einsum('akcd,cj,di,bk->abij', eri_["vovv"], t1_["vo"], t1_["vo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1),(0,1)])
    rt2_["vvoo"] += np.einsum('abij->abij', perm_tmps_["vvoo"])
    rt2_["vvoo"] -= np.einsum('baij->abij', perm_tmps_["vvoo"])

    # rt2_[vvoo] += +1.00 <l,k||c,d> t1(c,j) t1(d,i) t1(a,k) t1(b,l)  // flops: o2v2 += o3v2 o4v1 o4v1 o3v2 | mem: o2v2 += o3v1 o4v0 o3v1 o2v2
    rt2_["vvoo"] += np.einsum('lkcd,cj,di,ak,bl->abij', eri_["oovv"], t1_["vo"], t1_["vo"], t1_["vo"], t1_["vo"], optimize=['einsum_path',(0,1),(0,1),(0,1),(0,1)])

    return rt1_["vo"], rt2_["vvoo"]


def kernel(t1, t2, fock, g, o, v, e_ai, e_abij, max_iter=100, stopping_eps=1.0E-8,
           diis_size=None, diis_start_cycle=4):

    # initialize diis if diis_size is not None
    # else normal scf iterate
    if diis_size is not None:
        from diis import DIIS
        diis_update = DIIS(diis_size, start_iter=diis_start_cycle)
        t1_dim = t1.size
        old_vec = np.hstack((t1.flatten(), t2.flatten()))

    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)
    old_energy = ccsd_energy(t1, t2, fock, g, o, v)
    for idx in range(max_iter):

        singles_res, doubles_res = residuals(t1, t2, fock, g, o, v)

        # singles_res  = singles_residual(t1, t2, fock, g, o, v)
        # doubles_res  = doubles_residual(t1, t2, fock, g, o, v)

        singles_res  += fock_e_ai * t1
        doubles_res  += fock_e_abij * t2

        new_singles = singles_res * e_ai
        new_doubles = doubles_res * e_abij

        # diis update
        if diis_size is not None:
            vectorized_iterate = np.hstack(
                (new_singles.flatten(), new_doubles.flatten()))
            error_vec = old_vec - vectorized_iterate
            new_vectorized_iterate = diis_update.compute_new_vec(vectorized_iterate,
                                                                 error_vec)
            new_singles = new_vectorized_iterate[:t1_dim].reshape(t1.shape)
            new_doubles = new_vectorized_iterate[t1_dim:].reshape(t2.shape)
            old_vec = new_vectorized_iterate

        current_energy = ccsd_energy(new_singles, new_doubles, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < stopping_eps:
            return new_singles, new_doubles
        else:
            t1 = new_singles
            t2 = new_doubles
            old_energy = current_energy
            print("\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}".format(idx, old_energy, delta_e))
    else:
        print("Did not converge")
        return new_singles, new_doubles


def main():
    from itertools import product
    import pyscf
    import openfermion as of
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from openfermionpyscf import run_pyscf
    from pyscf.cc.addons import spatial2spin
    import numpy as np


    basis = 'cc-pvdz'
    mol = pyscf.M(
        atom='H 0 0 0; B 0 0 {}'.format(1.6),
        basis=basis)

    mf = mol.RHF().run()
    mycc = mf.CCSD().run()
    print('CCSD correlation energy', mycc.e_corr)

    molecule = of.MolecularData(geometry=[['H', (0, 0, 0)], ['B', (0, 0, 1.6)]],
                                basis=basis, charge=0, multiplicity=1)
    molecule = run_pyscf(molecule, run_ccsd=True)
    oei, tei = molecule.get_integrals()
    norbs = int(mf.mo_coeff.shape[1])
    occ = mf.mo_occ
    nele = int(sum(occ))
    nocc = nele // 2
    assert np.allclose(np.transpose(mycc.t2, [1, 0, 3, 2]), mycc.t2)

    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)

    # put in physics notation. OpenFermion stores <12|2'1'>
    gtei = astei.transpose(0, 1, 3, 2)

    eps = np.kron(molecule.orbital_energies, np.ones(2))
    n = np.newaxis
    o = slice(None, 2 * nocc)
    v = slice(2 * nocc, None)

    e_abij = 1 / (-eps[v, n, n, n] - eps[n, v, n, n] + eps[n, n, o, n] + eps[
        n, n, n, o])
    e_ai = 1 / (-eps[v, n] + eps[n, o])

    fock = soei + np.einsum('piiq->pq', astei[:, o, o, :])
    hf_energy = 0.5 * np.einsum('ii', (fock + soei)[o, o])
    hf_energy_test = 1.0 * einsum('ii', fock[o, o]) -0.5 * einsum('ijij', gtei[o, o, o, o])
    print(hf_energy_test, hf_energy)
    assert np.isclose(hf_energy + molecule.nuclear_repulsion, molecule.hf_energy)

    g = gtei
    nsvirt = 2 * (norbs - nocc)
    nsocc = 2 * nocc
    t1f, t2f = kernel(np.zeros((nsvirt, nsocc)), np.zeros((nsvirt, nsvirt, nsocc, nsocc)), fock, g, o, v, e_ai, e_abij)
    print(ccsd_energy(t1f, t2f, fock, g, o, v) - hf_energy)

    t1f, t2f = kernel(np.zeros((nsvirt, nsocc)), np.zeros((nsvirt, nsvirt, nsocc, nsocc)), fock, g, o, v, e_ai, e_abij,
                      diis_size=8, diis_start_cycle=4)
    print(ccsd_energy(t1f, t2f, fock, g, o, v) - hf_energy)




if __name__ == "__main__":
    main()