"""
Full diagonalization of the EOM-CCSD similarity-transformed Hamiltonian

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


def singles_residual(t1, t2, f, g, o, v):
    """
    < 0 | m* e e(-T) H e(T) | 0>
    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
    """
    #	  1.0000 f(e,m)
    singles_res = 1.0 * einsum('em->em', f[v, o])

    #	 -1.0000 f(i,m)*t1(e,i)
    singles_res += -1.0 * einsum('im,ei->em', f[o, o], t1)

    #	  1.0000 f(e,a)*t1(a,m)
    singles_res += 1.0 * einsum('ea,am->em', f[v, v], t1)

    #	 -1.0000 f(i,a)*t2(a,e,m,i)
    singles_res += -1.0 * einsum('ia,aemi->em', f[o, v], t2)

    #	 -1.0000 f(i,a)*t1(a,m)*t1(e,i)
    singles_res += -1.0 * einsum('ia,am,ei->em', f[o, v], t1, t1,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,m>*t1(a,i)
    singles_res += 1.0 * einsum('ieam,ai->em', g[o, v, v, o], t1)

    #	 -0.5000 <j,i||a,m>*t2(a,e,j,i)
    singles_res += -0.5 * einsum('jiam,aeji->em', g[o, o, v, o], t2)

    #	 -0.5000 <i,e||a,b>*t2(a,b,m,i)
    singles_res += -0.5 * einsum('ieab,abmi->em', g[o, v, v, v], t2)

    #	  1.0000 <j,i||a,m>*t1(a,i)*t1(e,j)
    singles_res += 1.0 * einsum('jiam,ai,ej->em', g[o, o, v, o], t1, t1,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,b>*t1(a,i)*t1(b,m)
    singles_res += 1.0 * einsum('ieab,ai,bm->em', g[o, v, v, v], t1, t1,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <j,i||a,b>*t1(a,i)*t2(b,e,m,j)
    singles_res += 1.0 * einsum('jiab,ai,bemj->em', g[o, o, v, v], t1, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <j,i||a,b>*t1(a,m)*t2(b,e,j,i)
    singles_res += 0.5 * einsum('jiab,am,beji->em', g[o, o, v, v], t1, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <j,i||a,b>*t1(e,i)*t2(a,b,m,j)
    singles_res += 0.5 * einsum('jiab,ei,abmj->em', g[o, o, v, v], t1, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,i||a,b>*t1(a,i)*t1(b,m)*t1(e,j)
    singles_res += 1.0 * einsum('jiab,ai,bm,ej->em', g[o, o, v, v], t1, t1, t1,
                                optimize=['einsum_path', (0, 1), (0, 2),
                                          (0, 1)])
    return singles_res


def doubles_residual(t1, t2, f, g, o, v):
    """
     < 0 | m* n* f e e(-T) H e(T) | 0>

    :param f:
    :param g:
    :param t1:
    :param t2:
    :param o:
    :param v:
    :return:
    """
    #	 -1.0000 P(m,n)f(i,n)*t2(e,f,m,i)
    contracted_intermediate = -1.0 * einsum('in,efmi->efmn', f[o, o], t2)
    doubles_res = 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate)

    #	  1.0000 P(e,f)f(e,a)*t2(a,f,m,n)
    contracted_intermediate = 1.0 * einsum('ea,afmn->efmn', f[v, v], t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->femn', contracted_intermediate)

    #	 -1.0000 P(m,n)f(i,a)*t1(a,n)*t2(e,f,m,i)
    contracted_intermediate = -1.0 * einsum('ia,an,efmi->efmn', f[o, v], t1, t2,
                                            optimize=['einsum_path', (0, 1),
                                                      (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate)

    #	 -1.0000 P(e,f)f(i,a)*t1(e,i)*t2(a,f,m,n)
    contracted_intermediate = -1.0 * einsum('ia,ei,afmn->efmn', f[o, v], t1, t2,
                                            optimize=['einsum_path', (0, 1),
                                                      (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->femn', contracted_intermediate)

    #	  1.0000 <e,f||m,n>
    doubles_res += 1.0 * einsum('efmn->efmn', g[v, v, o, o])

    #	  1.0000 P(e,f)<i,e||m,n>*t1(f,i)
    contracted_intermediate = 1.0 * einsum('iemn,fi->efmn', g[o, v, o, o], t1)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->femn', contracted_intermediate)

    #	  1.0000 P(m,n)<e,f||a,n>*t1(a,m)
    contracted_intermediate = 1.0 * einsum('efan,am->efmn', g[v, v, v, o], t1)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate)

    #	  0.5000 <j,i||m,n>*t2(e,f,j,i)
    doubles_res += 0.5 * einsum('jimn,efji->efmn', g[o, o, o, o], t2)

    #	  1.0000 P(m,n)*P(e,f)<i,e||a,n>*t2(a,f,m,i)
    contracted_intermediate = 1.0 * einsum('iean,afmi->efmn', g[o, v, v, o], t2)
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate) + -1.00000 * einsum('efmn->femn',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'efmn->fenm', contracted_intermediate)

    #	  0.5000 <e,f||a,b>*t2(a,b,m,n)
    doubles_res += 0.5 * einsum('efab,abmn->efmn', g[v, v, v, v], t2)

    #	 -1.0000 <j,i||m,n>*t1(e,i)*t1(f,j)
    doubles_res += -1.0 * einsum('jimn,ei,fj->efmn', g[o, o, o, o], t1, t1,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(m,n)*P(e,f)<i,e||a,n>*t1(a,m)*t1(f,i)
    contracted_intermediate = 1.0 * einsum('iean,am,fi->efmn', g[o, v, v, o],
                                           t1, t1,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate) + -1.00000 * einsum('efmn->femn',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'efmn->fenm', contracted_intermediate)

    #	 -1.0000 <e,f||a,b>*t1(a,n)*t1(b,m)
    doubles_res += -1.0 * einsum('efab,an,bm->efmn', g[v, v, v, v], t1, t1,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(m,n)<j,i||a,n>*t1(a,i)*t2(e,f,m,j)
    contracted_intermediate = 1.0 * einsum('jian,ai,efmj->efmn', g[o, o, v, o],
                                           t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate)

    #	  0.5000 P(m,n)<j,i||a,n>*t1(a,m)*t2(e,f,j,i)
    contracted_intermediate = 0.5 * einsum('jian,am,efji->efmn', g[o, o, v, o],
                                           t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate)

    #	 -1.0000 P(m,n)*P(e,f)<j,i||a,n>*t1(e,i)*t2(a,f,m,j)
    contracted_intermediate = -1.0 * einsum('jian,ei,afmj->efmn', g[o, o, v, o],
                                            t1, t2,
                                            optimize=['einsum_path', (0, 1),
                                                      (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate) + -1.00000 * einsum('efmn->femn',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'efmn->fenm', contracted_intermediate)

    #	  1.0000 P(e,f)<i,e||a,b>*t1(a,i)*t2(b,f,m,n)
    contracted_intermediate = 1.0 * einsum('ieab,ai,bfmn->efmn', g[o, v, v, v],
                                           t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->femn', contracted_intermediate)

    #	 -1.0000 P(m,n)*P(e,f)<i,e||a,b>*t1(a,n)*t2(b,f,m,i)
    contracted_intermediate = -1.0 * einsum('ieab,an,bfmi->efmn', g[o, v, v, v],
                                            t1, t2,
                                            optimize=['einsum_path', (0, 1),
                                                      (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate) + -1.00000 * einsum('efmn->femn',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'efmn->fenm', contracted_intermediate)

    #	  0.5000 P(e,f)<i,e||a,b>*t1(f,i)*t2(a,b,m,n)
    contracted_intermediate = 0.5 * einsum('ieab,fi,abmn->efmn', g[o, v, v, v],
                                           t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->femn', contracted_intermediate)

    #	 -0.5000 P(m,n)<j,i||a,b>*t2(a,b,n,i)*t2(e,f,m,j)
    contracted_intermediate = -0.5 * einsum('jiab,abni,efmj->efmn',
                                            g[o, o, v, v], t2, t2,
                                            optimize=['einsum_path', (0, 1),
                                                      (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate)

    #	  0.2500 <j,i||a,b>*t2(a,b,m,n)*t2(e,f,j,i)
    doubles_res += 0.25 * einsum('jiab,abmn,efji->efmn', g[o, o, v, v], t2, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <j,i||a,b>*t2(a,e,j,i)*t2(b,f,m,n)
    doubles_res += -0.5 * einsum('jiab,aeji,bfmn->efmn', g[o, o, v, v], t2, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(m,n)<j,i||a,b>*t2(a,e,n,i)*t2(b,f,m,j)
    contracted_intermediate = 1.0 * einsum('jiab,aeni,bfmj->efmn',
                                           g[o, o, v, v], t2, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate)

    #	 -0.5000 <j,i||a,b>*t2(a,e,m,n)*t2(b,f,j,i)
    doubles_res += -0.5 * einsum('jiab,aemn,bfji->efmn', g[o, o, v, v], t2, t2,
                                 optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(m,n)<j,i||a,n>*t1(a,m)*t1(e,i)*t1(f,j)
    contracted_intermediate = -1.0 * einsum('jian,am,ei,fj->efmn',
                                            g[o, o, v, o], t1, t1, t1,
                                            optimize=['einsum_path', (0, 1),
                                                      (0, 2), (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate)

    #	 -1.0000 P(e,f)<i,e||a,b>*t1(a,n)*t1(b,m)*t1(f,i)
    contracted_intermediate = -1.0 * einsum('ieab,an,bm,fi->efmn',
                                            g[o, v, v, v], t1, t1, t1,
                                            optimize=['einsum_path', (0, 1),
                                                      (0, 2), (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->femn', contracted_intermediate)

    #	  1.0000 P(m,n)<j,i||a,b>*t1(a,i)*t1(b,n)*t2(e,f,m,j)
    contracted_intermediate = 1.0 * einsum('jiab,ai,bn,efmj->efmn',
                                           g[o, o, v, v], t1, t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 2), (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate)

    #	  1.0000 P(e,f)<j,i||a,b>*t1(a,i)*t1(e,j)*t2(b,f,m,n)
    contracted_intermediate = 1.0 * einsum('jiab,ai,ej,bfmn->efmn',
                                           g[o, o, v, v], t1, t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 2), (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->femn', contracted_intermediate)

    #	 -0.5000 <j,i||a,b>*t1(a,n)*t1(b,m)*t2(e,f,j,i)
    doubles_res += -0.5 * einsum('jiab,an,bm,efji->efmn', g[o, o, v, v], t1, t1,
                                 t2, optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	  1.0000 P(m,n)*P(e,f)<j,i||a,b>*t1(a,n)*t1(e,i)*t2(b,f,m,j)
    contracted_intermediate = 1.0 * einsum('jiab,an,ei,bfmj->efmn',
                                           g[o, o, v, v], t1, t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 2), (0, 1)])
    doubles_res += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'efmn->efnm', contracted_intermediate) + -1.00000 * einsum('efmn->femn',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'efmn->fenm', contracted_intermediate)

    #	 -0.5000 <j,i||a,b>*t1(e,i)*t1(f,j)*t2(a,b,m,n)
    doubles_res += -0.5 * einsum('jiab,ei,fj,abmn->efmn', g[o, o, v, v], t1, t1,
                                 t2, optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	  1.0000 <j,i||a,b>*t1(a,n)*t1(b,m)*t1(e,i)*t1(f,j)
    doubles_res += 1.0 * einsum('jiab,an,bm,ei,fj->efmn', g[o, o, v, v], t1, t1,
                                t1, t1,
                                optimize=['einsum_path', (0, 1), (0, 3), (0, 2),
                                          (0, 1)])

    return doubles_res

def integral_maps(f, eri, kd, o, v):
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

    Id_ = {}
    Id_["oo"] = kd[o,o]
    Id_["vv"] = kd[v,v]

    return f_, eri_, Id_

def build_ea_Hbar(Id, f, eri, o, v, t1, t2):

    scalars_ = {}
    tmps_ = {}

    #####  Scalars  #####
    scalars_["0"] = 0.0 * einsum('ij,ij->', Id["oo"], f["oo"])
    scalars_["1"] = 0.0 * einsum('ib,bi->', f["ov"], t1)
    scalars_["2"] = 0.0 * einsum('jk,jikl,ik->', Id["oo"], eri["oooo"], Id["oo"], optimize=['einsum_path',(0,1),(0,1)])
    scalars_["3"] = 0.0 * einsum('jibc,bcji->', eri["oovv"], t2)
    scalars_["4"] = 0.0 * einsum('cj,jibc,bi->', t1, eri["oovv"], t1, optimize=['einsum_path',(0,1),(0,1)])

    ### End of Scalars ###

    ##########  Evaluate Equations  ##########


    # H11 = +1.00 f(a,e)  // flops: o0v2 = o0v2 | mem: o0v2 = o0v2
    H11 = 1.00 * einsum('ae->ae', f["vv"])

    # H12 = -1.00 <m,a||e,f>  // flops: o1v3 = o1v3 | mem: o1v3 = o1v3
    H12 = 1.00 * einsum('amef->aefm', eri["vovv"])

    # H21 = +1.00 <a,b||e,i>  // flops: o1v3 = o1v3 | mem: o1v3 = o1v3
    H21 = 1.00 * einsum('abei->abie', eri["vvvo"])

    # H11 += +0.25 d(a,e) <j,i||b,c> t2(b,c,j,i)  // flops: o0v2 += o0v2 | mem: o0v2 += o0v2
    H11 += 0.25 * scalars_["3"] * einsum('ae->ae', Id["vv"])

    # H11 += -0.50 d(a,e) <j,i||j,i>  // flops: o0v2 += o0v2 | mem: o0v2 += o0v2
    H11 -= 0.50 * scalars_["2"] * einsum('ae->ae', Id["vv"])

    # H11 += -0.50 d(a,e) <j,i||b,c> t1(b,i) t1(c,j)  // flops: o0v2 += o0v2 | mem: o0v2 += o0v2
    H11 -= 0.50 * scalars_["4"] * einsum('ae->ae', Id["vv"])

    # H11 += +1.00 d(a,e) f(i,b) t1(b,i)  // flops: o0v2 += o0v2 | mem: o0v2 += o0v2
    H11 += scalars_["1"] * einsum('ae->ae', Id["vv"])

    # H11 += +1.00 d(a,e) f(i,i)  // flops: o0v2 += o0v2 | mem: o0v2 += o0v2
    H11 += scalars_["0"] * einsum('ae->ae', Id["vv"])

    # H12 += -1.00 d(a,f) f(m,e)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H12 -= einsum('af,me->aefm', Id["vv"], f["ov"])

    # H12 += +1.00 d(a,e) f(m,f)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H12 += einsum('ae,mf->aefm', Id["vv"], f["ov"])

    # H21 += +1.00 d(a,e) f(b,i)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('ae,bi->abie', Id["vv"], f["vo"])

    # H21 += -1.00 d(b,e) f(a,i)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('be,ai->abie', Id["vv"], f["vo"])

    # H21 += +1.00 f(j,e) t2(a,b,i,j)  // flops: o1v3 += o2v3 | mem: o1v3 += o1v3
    H21 += einsum('je,abij->abie', f["ov"], t2)

    # H21 += -1.00 <a,b||c,e> t1(c,i)  // flops: o1v3 += o1v4 | mem: o1v3 += o1v3
    H21 -= einsum('abce,ci->abie', eri["vvvv"], t1)

    # H21 += +0.50 <k,j||e,i> t2(a,b,k,j)  // flops: o1v3 += o3v3 | mem: o1v3 += o1v3
    H21 += 0.50 * einsum('kjei,abkj->abie', eri["oovo"], t2)

    # H22 = +1.00 d(i,m) <a,b||e,f>  // flops: o2v4 = o2v4 | mem: o2v4 = o2v4
    H22 = einsum('im,abef->abiefm', Id["oo"], eri["vvvv"])

    # H22 += +1.00 d(a,e) <m,b||f,i>  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('ae,bmfi->abiefm', Id["vv"], eri["vovo"])

    # H22 += -1.00 d(a,f) <m,b||e,i>  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('af,bmei->abiefm', Id["vv"], eri["vovo"])

    # H22 += -1.00 d(b,e) <m,a||f,i>  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('be,amfi->abiefm', Id["vv"], eri["vovo"])

    # H22 += +1.00 d(b,f) <m,a||e,i>  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('bf,amei->abiefm', Id["vv"], eri["vovo"])

    # H22 += -1.00 <m,j||e,f> t2(a,b,i,j)  // flops: o2v4 += o3v4 | mem: o2v4 += o2v4
    H22 -= einsum('mjef,abij->abiefm', eri["oovv"], t2)

    # tmps_[1_vvvo](a,e,b,i) = 1.00 eri[vovv](b,j,c,e) * t2(c,a,i,j) // flops: o1v3 = o2v4 | mem: o1v3 = o1v3
    tmps_["1_vvvo"] = einsum('bjce,caij->aebi', eri["vovv"], t2)

    # H21 += -1.00 P(a,b) <j,a||c,e> t2(c,b,i,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('beai->abie', tmps_["1_vvvo"])
    H21 -= einsum('aebi->abie', tmps_["1_vvvo"])
    del tmps_["1_vvvo"]

    # tmps_[2_vvoo](a,f,i,m) = 1.00 eri[oovv](m,j,c,f) * t2(c,a,i,j) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["2_vvoo"] = einsum('mjcf,caij->afim', eri["oovv"], t2)

    # H22 += -1.00 d(a,f) <m,j||c,e> t2(c,b,i,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('beim,af->abiefm', tmps_["2_vvoo"], Id["vv"])

    # H22 += -1.00 d(b,e) <m,j||c,f> t2(c,a,i,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('afim,be->abiefm', tmps_["2_vvoo"], Id["vv"])

    # H22 += +1.00 d(b,f) <m,j||c,e> t2(c,a,i,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('aeim,bf->abiefm', tmps_["2_vvoo"], Id["vv"])

    # H22 += +1.00 d(a,e) <m,j||c,f> t2(c,b,i,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('bfim,ae->abiefm', tmps_["2_vvoo"], Id["vv"])
    del tmps_["2_vvoo"]

    # tmps_[3_vvoo](a,e,i,j) = 1.00 eri[oovv](k,j,c,e) * t2(c,a,i,k) // flops: o2v2 = o3v3 | mem: o2v2 = o2v2
    tmps_["3_vvoo"] = einsum('kjce,caik->aeij', eri["oovv"], t2)

    # tmps_[28_vvvo](a,e,b,i) = 1.00 eri[oovv](k,j,c,e) * t2(c,b,i,k) * t1(a,j) // flops: o1v3 = o2v3 | mem: o1v3 = o1v3
    tmps_["28_vvvo"] = einsum('beij,aj->aebi', tmps_["3_vvoo"], t1)
    del tmps_["3_vvoo"]
    H21 -= einsum('beai->abie', tmps_["28_vvvo"])

    # H21 += +1.00 P(a,b) <k,j||c,e> t1(a,j) t2(c,b,i,k)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('aebi->abie', tmps_["28_vvvo"])
    del tmps_["28_vvvo"]

    # tmps_[4_vvvv](b,a,f,e) = 1.00 eri[oovv](k,j,e,f) * t2(a,b,k,j) // flops: o0v4 = o2v4 | mem: o0v4 = o0v4
    tmps_["4_vvvv"] = einsum('kjef,abkj->bafe', eri["oovv"], t2)

    # H22 += +0.50 d(i,m) <k,j||e,f> t2(a,b,k,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += 0.50 * einsum('bafe,im->abiefm', tmps_["4_vvvv"], Id["oo"])
    del tmps_["4_vvvv"]

    # tmps_[5_vvvv](b,e,c,a) = 1.00 eri[vovv](a,j,c,e) * t1(b,j) // flops: o0v4 = o1v4 | mem: o0v4 = o0v4
    tmps_["5_vvvv"] = einsum('ajce,bj->beca', eri["vovv"], t1)
    H22 += einsum('afeb,im->abiefm', tmps_["5_vvvv"], Id["oo"])

    # H22 += +1.00 P(a,b) d(i,m) <j,a||e,f> t1(b,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('bfea,im->abiefm', tmps_["5_vvvv"], Id["oo"])
    del tmps_["5_vvvv"]

    # tmps_[6_vvoo](e,a,i,j) = 1.00 eri[vovv](a,j,c,e) * t1(c,i) // flops: o2v2 = o2v3 | mem: o2v2 = o2v2
    tmps_["6_vvoo"] = einsum('ajce,ci->eaij', eri["vovv"], t1)

    # H22 += -1.00 d(a,e) <m,b||c,f> t1(c,i)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('fbim,ae->abiefm', tmps_["6_vvoo"], Id["vv"])

    # H22 += -1.00 d(b,f) <m,a||c,e> t1(c,i)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('eaim,bf->abiefm', tmps_["6_vvoo"], Id["vv"])

    # H22 += +1.00 d(a,f) <m,b||c,e> t1(c,i)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('ebim,af->abiefm', tmps_["6_vvoo"], Id["vv"])

    # H22 += +1.00 d(b,e) <m,a||c,f> t1(c,i)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('faim,be->abiefm', tmps_["6_vvoo"], Id["vv"])

    # tmps_[27_vvvo](a,e,b,i) = 1.00 t1(b,j) * eri[vovv](a,j,c,e) * t1(c,i) // flops: o1v3 = o2v3 | mem: o1v3 = o1v3
    tmps_["27_vvvo"] = einsum('bj,eaij->aebi', t1, tmps_["6_vvoo"])
    del tmps_["6_vvoo"]

    # H21 += -1.00 P(a,b) <j,a||c,e> t1(c,i) t1(b,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('aebi->abie', tmps_["27_vvvo"])
    H21 -= einsum('beai->abie', tmps_["27_vvvo"])
    del tmps_["27_vvvo"]

    # tmps_[7_vv](a,e) = 0.50 eri[oovv](k,j,c,e) * t2(c,a,k,j) // flops: o0v2 = o2v3 | mem: o0v2 = o0v2
    tmps_["7_vv"] = 0.50 * einsum('kjce,cakj->ae', eri["oovv"], t2)

    # H11 += -0.50 <j,i||b,e> t2(b,a,j,i)  // flops: o0v2 += o0v2 | mem: o0v2 += o0v2
    H11 -= einsum('ae->ae', tmps_["7_vv"])

    # tmps_[17_vvoo](f,b,m,i) = 1.00 Id[oo](i,m) * Id[vv](b,f) // flops: o2v2 = o2v2 | mem: o2v2 = o2v2
    tmps_["17_vvoo"] = einsum('im,bf->fbmi', Id["oo"], Id["vv"])

    # H22 += +0.50 d(b,e) d(i,m) <k,j||c,f> t2(c,a,k,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('af,ebmi->abiefm', tmps_["7_vv"], tmps_["17_vvoo"])

    # H22 += +0.50 d(a,f) d(i,m) <k,j||c,e> t2(c,b,k,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('be,fami->abiefm', tmps_["7_vv"], tmps_["17_vvoo"])

    # H22 += -0.50 d(b,f) d(i,m) <k,j||c,e> t2(c,a,k,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('ae,fbmi->abiefm', tmps_["7_vv"], tmps_["17_vvoo"])

    # H22 += -0.50 d(a,e) d(i,m) <k,j||c,f> t2(c,b,k,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('bf,eami->abiefm', tmps_["7_vv"], tmps_["17_vvoo"])
    del tmps_["7_vv"]

    # tmps_[8_vo](b,i) = 0.50 eri[vovv](b,j,c,d) * t2(c,d,i,j) // flops: o1v1 = o2v3 | mem: o1v1 = o1v1
    tmps_["8_vo"] = 0.50 * einsum('bjcd,cdij->bi', eri["vovv"], t2)

    # H21 += +0.50 d(b,e) <j,a||c,d> t2(c,d,i,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('ai,be->abie', tmps_["8_vo"], Id["vv"])

    # H21 += -0.50 d(a,e) <j,b||c,d> t2(c,d,i,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('bi,ae->abie', tmps_["8_vo"], Id["vv"])
    del tmps_["8_vo"]

    # tmps_[9_vvvo](a,f,e,k) = 1.00 eri[oovv](k,j,e,f) * t1(a,j) // flops: o1v3 = o2v3 | mem: o1v3 = o1v3
    tmps_["9_vvvo"] = einsum('kjef,aj->afek', eri["oovv"], t1)

    # H12 += +1.00 <m,i||e,f> t1(a,i)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H12 += einsum('afem->aefm', tmps_["9_vvvo"])

    # H22 += -1.00 d(i,m) <k,j||e,f> t1(a,j) t1(b,k)  // flops: o2v4 += o1v4 o2v4 | mem: o2v4 += o0v4 o2v4
    H22 -= einsum('afek,bk,im->abiefm', tmps_["9_vvvo"], t1, Id["oo"], optimize=['einsum_path',(0,1),(0,1)])
    del tmps_["9_vvvo"]

    # tmps_[10_vvvo](a,e,b,i) = 1.00 eri[vovo](b,j,e,i) * t1(a,j) // flops: o1v3 = o2v3 | mem: o1v3 = o1v3
    tmps_["10_vvvo"] = einsum('bjei,aj->aebi', eri["vovo"], t1)

    # H21 += +1.00 P(a,b) <j,a||e,i> t1(b,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('beai->abie', tmps_["10_vvvo"])
    H21 += einsum('aebi->abie', tmps_["10_vvvo"])
    del tmps_["10_vvvo"]

    # tmps_[11_vooo](f,i,j,m) = 1.00 eri[oovv](m,j,c,f) * t1(c,i) // flops: o3v1 = o3v2 | mem: o3v1 = o3v1
    tmps_["11_vooo"] = einsum('mjcf,ci->fijm', eri["oovv"], t1)

    # H21 += -0.50 <k,j||c,e> t1(c,i) t2(a,b,k,j)  // flops: o1v3 += o3v3 | mem: o1v3 += o1v3
    H21 -= 0.50 * einsum('eijk,abkj->abie', tmps_["11_vooo"], t2)

    # tmps_[29_vvoo](f,a,m,i) = 1.00 t1(a,j) * eri[oovv](m,j,c,f) * t1(c,i) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["29_vvoo"] = einsum('aj,fijm->fami', t1, tmps_["11_vooo"])

    # H21 += +1.00 <k,j||c,e> t1(c,i) t1(a,j) t1(b,k)  // flops: o1v3 += o2v3 | mem: o1v3 += o1v3
    H21 += einsum('eaki,bk->abie', tmps_["29_vvoo"], t1)

    # H22 += +1.00 d(a,e) <m,j||c,f> t1(c,i) t1(b,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('fbmi,ae->abiefm', tmps_["29_vvoo"], Id["vv"])

    # H22 += -1.00 d(b,e) <m,j||c,f> t1(c,i) t1(a,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('fami,be->abiefm', tmps_["29_vvoo"], Id["vv"])

    # H22 += -1.00 d(a,f) <m,j||c,e> t1(c,i) t1(b,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('ebmi,af->abiefm', tmps_["29_vvoo"], Id["vv"])

    # H22 += +1.00 d(b,f) <m,j||c,e> t1(c,i) t1(a,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('eami,bf->abiefm', tmps_["29_vvoo"], Id["vv"])
    del tmps_["29_vvoo"]

    # tmps_[30_vo](a,i) = 0.50 t2(d,a,k,j) * eri[oovv](k,j,c,d) * t1(c,i) // flops: o1v1 = o3v2 | mem: o1v1 = o1v1
    tmps_["30_vo"] = 0.50 * einsum('dakj,dijk->ai', t2, tmps_["11_vooo"])
    del tmps_["11_vooo"]

    # H21 += +0.50 d(a,e) <k,j||c,d> t1(c,i) t2(d,b,k,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('bi,ae->abie', tmps_["30_vo"], Id["vv"])

    # H21 += -0.50 d(b,e) <k,j||c,d> t1(c,i) t2(d,a,k,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('ai,be->abie', tmps_["30_vo"], Id["vv"])
    del tmps_["30_vo"]

    # tmps_[12_vvoo](a,e,i,m) = 1.00 eri[oovo](m,j,e,i) * t1(a,j) // flops: o2v2 = o3v2 | mem: o2v2 = o2v2
    tmps_["12_vvoo"] = einsum('mjei,aj->aeim', eri["oovo"], t1)

    # H21 += -1.00 <k,j||e,i> t1(a,j) t1(b,k)  // flops: o1v3 += o2v3 | mem: o1v3 += o1v3
    H21 -= einsum('aeik,bk->abie', tmps_["12_vvoo"], t1)

    # H22 += +1.00 d(b,e) <m,j||f,i> t1(a,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('afim,be->abiefm', tmps_["12_vvoo"], Id["vv"])

    # H22 += -1.00 d(b,f) <m,j||e,i> t1(a,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('aeim,bf->abiefm', tmps_["12_vvoo"], Id["vv"])

    # H22 += +1.00 d(a,f) <m,j||e,i> t1(b,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('beim,af->abiefm', tmps_["12_vvoo"], Id["vv"])

    # H22 += -1.00 d(a,e) <m,j||f,i> t1(b,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('bfim,ae->abiefm', tmps_["12_vvoo"], Id["vv"])
    del tmps_["12_vvoo"]

    # tmps_[13_oo](i,m) = 0.50 eri[oovv](m,j,c,d) * t2(c,d,i,j) // flops: o2v0 = o3v2 | mem: o2v0 = o2v0
    tmps_["13_oo"] = 0.50 * einsum('mjcd,cdij->im', eri["oovv"], t2)

    # tmps_[40_vvoo](e,a,m,i) = 1.00 Id[vv](a,e) * eri[oovv](m,j,c,d) * t2(c,d,i,j) // flops: o2v2 = o2v2 | mem: o2v2 = o2v2
    tmps_["40_vvoo"] = einsum('ae,im->eami', Id["vv"], tmps_["13_oo"])
    del tmps_["13_oo"]

    # H22 += -0.50 d(a,e) d(b,f) <m,j||c,d> t2(c,d,i,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('eami,bf->abiefm', tmps_["40_vvoo"], Id["vv"])

    # H22 += +0.50 d(b,e) d(a,f) <m,j||c,d> t2(c,d,i,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('ebmi,af->abiefm', tmps_["40_vvoo"], Id["vv"])
    del tmps_["40_vvoo"]

    # tmps_[14_oo](i,j) = 0.50 eri[oovv](k,j,c,d) * t2(c,d,i,k) // flops: o2v0 = o3v2 | mem: o2v0 = o2v0
    tmps_["14_oo"] = 0.50 * einsum('kjcd,cdik->ij', eri["oovv"], t2)

    # tmps_[37_vo](b,i) = 1.00 t1(b,j) * eri[oovv](k,j,c,d) * t2(c,d,i,k) // flops: o1v1 = o2v1 | mem: o1v1 = o1v1
    tmps_["37_vo"] = einsum('bj,ij->bi', t1, tmps_["14_oo"])
    del tmps_["14_oo"]

    # H21 += +0.50 d(a,e) <k,j||c,d> t1(b,j) t2(c,d,i,k)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('bi,ae->abie', tmps_["37_vo"], Id["vv"])

    # H21 += -0.50 d(b,e) <k,j||c,d> t1(a,j) t2(c,d,i,k)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('ai,be->abie', tmps_["37_vo"], Id["vv"])
    del tmps_["37_vo"]

    # tmps_[15_vo](a,i) = 0.50 eri[oovo](k,j,c,i) * t2(c,a,k,j) // flops: o1v1 = o3v2 | mem: o1v1 = o1v1
    tmps_["15_vo"] = 0.50 * einsum('kjci,cakj->ai', eri["oovo"], t2)

    # H21 += -0.50 d(a,e) <k,j||c,i> t2(c,b,k,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('bi,ae->abie', tmps_["15_vo"], Id["vv"])

    # H21 += +0.50 d(b,e) <k,j||c,i> t2(c,a,k,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('ai,be->abie', tmps_["15_vo"], Id["vv"])
    del tmps_["15_vo"]

    # tmps_[16_vv](d,a) = 1.00 eri[vovv](a,j,c,d) * t1(c,j) // flops: o0v2 = o1v3 | mem: o0v2 = o0v2
    tmps_["16_vv"] = einsum('ajcd,cj->da', eri["vovv"], t1)

    # H11 += +1.00 <i,a||b,e> t1(b,i)  // flops: o0v2 += o0v2 | mem: o0v2 += o0v2
    H11 -= einsum('ea->ae', tmps_["16_vv"])

    # H22 += +1.00 d(a,e) d(i,m) <j,b||c,f> t1(c,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('fb,eami->abiefm', tmps_["16_vv"], tmps_["17_vvoo"])

    # H22 += -1.00 d(a,f) d(i,m) <j,b||c,e> t1(c,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('eb,fami->abiefm', tmps_["16_vv"], tmps_["17_vvoo"])

    # H22 += +1.00 d(b,f) d(i,m) <j,a||c,e> t1(c,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('ea,fbmi->abiefm', tmps_["16_vv"], tmps_["17_vvoo"])

    # H22 += -1.00 d(b,e) d(i,m) <j,a||c,f> t1(c,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('fa,ebmi->abiefm', tmps_["16_vv"], tmps_["17_vvoo"])

    # tmps_[33_vo](b,i) = 1.00 t1(d,i) * eri[vovv](b,j,c,d) * t1(c,j) // flops: o1v1 = o1v2 | mem: o1v1 = o1v1
    tmps_["33_vo"] = einsum('di,db->bi', t1, tmps_["16_vv"])
    del tmps_["16_vv"]

    # H21 += +1.00 d(a,e) <j,b||c,d> t1(c,j) t1(d,i)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('bi,ae->abie', tmps_["33_vo"], Id["vv"])

    # H21 += -1.00 d(b,e) <j,a||c,d> t1(c,j) t1(d,i)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('ai,be->abie', tmps_["33_vo"], Id["vv"])
    del tmps_["33_vo"]

    # H22 += -1.00 d(a,f) d(i,m) f(b,e)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('fami,be->abiefm', tmps_["17_vvoo"], f["vv"])

    # H22 += +1.00 d(b,f) d(i,m) f(a,e)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('fbmi,ae->abiefm', tmps_["17_vvoo"], f["vv"])

    # H22 += -1.00 d(b,e) d(i,m) f(a,f)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('ebmi,af->abiefm', tmps_["17_vvoo"], f["vv"])

    # H22 += +1.00 d(a,e) d(i,m) f(b,f)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('eami,bf->abiefm', tmps_["17_vvoo"], f["vv"])

    # H22 += +1.00 d(a,e) d(b,f) d(i,m) f(j,c) t1(c,j)  // flops: o2v4 += o0v2 o2v4 | mem: o2v4 += o0v2 o2v4
    H22 += scalars_["1"] * einsum('bf,eami->abiefm', Id["vv"], tmps_["17_vvoo"])

    # H22 += +1.00 d(a,e) d(b,f) d(i,m) f(j,j)  // flops: o2v4 += o0v2 o2v4 | mem: o2v4 += o0v2 o2v4
    H22 += scalars_["0"] * einsum('bf,eami->abiefm', Id["vv"], tmps_["17_vvoo"])

    # H22 += +0.25 d(a,e) d(b,f) d(i,m) <k,j||c,d> t2(c,d,k,j)  // flops: o2v4 += o0v2 o2v4 | mem: o2v4 += o0v2 o2v4
    H22 += 0.25 * scalars_["3"] * einsum('bf,eami->abiefm', Id["vv"], tmps_["17_vvoo"])

    # H22 += +0.50 d(b,e) d(a,f) d(i,m) <k,j||k,j>  // flops: o2v4 += o0v2 o2v4 | mem: o2v4 += o0v2 o2v4
    H22 += 0.50 * scalars_["2"] * einsum('af,ebmi->abiefm', Id["vv"], tmps_["17_vvoo"])

    # H22 += -0.25 d(b,e) d(a,f) d(i,m) <k,j||c,d> t2(c,d,k,j)  // flops: o2v4 += o0v2 o2v4 | mem: o2v4 += o0v2 o2v4
    H22 -= 0.25 * scalars_["3"] * einsum('af,ebmi->abiefm', Id["vv"], tmps_["17_vvoo"])

    # H22 += -0.50 d(a,e) d(b,f) d(i,m) <k,j||c,d> t1(c,j) t1(d,k)  // flops: o2v4 += o0v2 o2v4 | mem: o2v4 += o0v2 o2v4
    H22 -= 0.50 * scalars_["4"] * einsum('bf,eami->abiefm', Id["vv"], tmps_["17_vvoo"])

    # H22 += -1.00 d(b,e) d(a,f) d(i,m) f(j,j)  // flops: o2v4 += o0v2 o2v4 | mem: o2v4 += o0v2 o2v4
    H22 -= scalars_["0"] * einsum('af,ebmi->abiefm', Id["vv"], tmps_["17_vvoo"])

    # H22 += +0.50 d(b,e) d(a,f) d(i,m) <k,j||c,d> t1(c,j) t1(d,k)  // flops: o2v4 += o0v2 o2v4 | mem: o2v4 += o0v2 o2v4
    H22 += 0.50 * scalars_["4"] * einsum('af,ebmi->abiefm', Id["vv"], tmps_["17_vvoo"])

    # H22 += -1.00 d(b,e) d(a,f) d(i,m) f(j,c) t1(c,j)  // flops: o2v4 += o0v2 o2v4 | mem: o2v4 += o0v2 o2v4
    H22 -= scalars_["1"] * einsum('af,ebmi->abiefm', Id["vv"], tmps_["17_vvoo"])

    # H22 += -0.50 d(a,e) d(b,f) d(i,m) <k,j||k,j>  // flops: o2v4 += o0v2 o2v4 | mem: o2v4 += o0v2 o2v4
    H22 -= 0.50 * scalars_["2"] * einsum('bf,eami->abiefm', Id["vv"], tmps_["17_vvoo"])

    # tmps_[18_vo](e,k) = 1.00 eri[oovv](k,j,c,e) * t1(c,j) // flops: o1v1 = o2v2 | mem: o1v1 = o1v1
    tmps_["18_vo"] = einsum('kjce,cj->ek', eri["oovv"], t1)

    # tmps_[32_vv](e,a) = 1.00 t1(a,j) * eri[oovv](j,i,b,e) * t1(b,i) // flops: o0v2 = o1v2 | mem: o0v2 = o0v2
    tmps_["32_vv"] = einsum('aj,ej->ea', t1, tmps_["18_vo"])

    # H22 += -1.00 d(b,e) d(i,m) <k,j||c,f> t1(c,j) t1(a,k)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('fa,ebmi->abiefm', tmps_["32_vv"], tmps_["17_vvoo"])

    # H22 += +1.00 d(b,f) d(i,m) <k,j||c,e> t1(c,j) t1(a,k)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('ea,fbmi->abiefm', tmps_["32_vv"], tmps_["17_vvoo"])

    # H22 += +1.00 d(a,e) d(i,m) <k,j||c,f> t1(c,j) t1(b,k)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('fb,eami->abiefm', tmps_["32_vv"], tmps_["17_vvoo"])

    # H22 += -1.00 d(a,f) d(i,m) <k,j||c,e> t1(c,j) t1(b,k)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('eb,fami->abiefm', tmps_["32_vv"], tmps_["17_vvoo"])

    # tmps_[22_vv](a,e) = 1.00 f[ov](i,e) * t1(a,i) // flops: o0v2 = o1v2 | mem: o0v2 = o0v2
    tmps_["22_vv"] = einsum('ie,ai->ae', f["ov"], t1)

    # H22 += +1.00 d(b,e) d(i,m) f(j,f) t1(a,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('af,ebmi->abiefm', tmps_["22_vv"], tmps_["17_vvoo"])

    # H22 += -1.00 d(a,e) d(i,m) f(j,f) t1(b,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('bf,eami->abiefm', tmps_["22_vv"], tmps_["17_vvoo"])

    # H22 += +1.00 d(a,f) d(i,m) f(j,e) t1(b,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('be,fami->abiefm', tmps_["22_vv"], tmps_["17_vvoo"])

    # H22 += -1.00 d(b,f) d(i,m) f(j,e) t1(a,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('ae,fbmi->abiefm', tmps_["22_vv"], tmps_["17_vvoo"])
    del tmps_["17_vvoo"]

    # H12 += -1.00 d(a,e) <m,i||b,f> t1(b,i)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H12 -= einsum('fm,ae->aefm', tmps_["18_vo"], Id["vv"])

    # H12 += +1.00 d(a,f) <m,i||b,e> t1(b,i)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H12 += einsum('em,af->aefm', tmps_["18_vo"], Id["vv"])

    # H21 += -1.00 <k,j||c,e> t1(c,j) t2(a,b,i,k)  // flops: o1v3 += o2v3 | mem: o1v3 += o1v3
    H21 -= einsum('ek,abik->abie', tmps_["18_vo"], t2)

    # tmps_[31_vo](a,i) = 1.00 t2(d,a,i,k) * eri[oovv](k,j,c,d) * t1(c,j) // flops: o1v1 = o2v2 | mem: o1v1 = o1v1
    tmps_["31_vo"] = einsum('daik,dk->ai', t2, tmps_["18_vo"])

    # H21 += +1.00 d(a,e) <k,j||c,d> t1(c,j) t2(d,b,i,k)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('bi,ae->abie', tmps_["31_vo"], Id["vv"])

    # H21 += -1.00 d(b,e) <k,j||c,d> t1(c,j) t2(d,a,i,k)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('ai,be->abie', tmps_["31_vo"], Id["vv"])
    del tmps_["31_vo"]

    # H11 += +1.00 <j,i||b,e> t1(b,i) t1(a,j)  // flops: o0v2 += o0v2 | mem: o0v2 += o0v2
    H11 += einsum('ea->ae', tmps_["32_vv"])
    del tmps_["32_vv"]

    # tmps_[34_oo](k,i) = 1.00 t1(d,i) * eri[oovv](k,j,c,d) * t1(c,j) // flops: o2v0 = o2v1 | mem: o2v0 = o2v0
    tmps_["34_oo"] = einsum('di,dk->ki', t1, tmps_["18_vo"])
    del tmps_["18_vo"]

    # tmps_[41_vo](b,i) = 1.00 t1(b,k) * t1(d,i) * eri[oovv](k,j,c,d) * t1(c,j) // flops: o1v1 = o2v1 | mem: o1v1 = o1v1
    tmps_["41_vo"] = einsum('bk,ki->bi', t1, tmps_["34_oo"])

    # H21 += +1.00 d(a,e) <k,j||c,d> t1(c,j) t1(d,i) t1(b,k)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('bi,ae->abie', tmps_["41_vo"], Id["vv"])

    # H21 += -1.00 d(b,e) <k,j||c,d> t1(c,j) t1(d,i) t1(a,k)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('ai,be->abie', tmps_["41_vo"], Id["vv"])
    del tmps_["41_vo"]

    # tmps_[42_vvoo](e,a,i,k) = 1.00 Id[vv](a,e) * t1(d,i) * eri[oovv](k,j,c,d) * t1(c,j) // flops: o2v2 = o2v2 | mem: o2v2 = o2v2
    tmps_["42_vvoo"] = einsum('ae,ki->eaik', Id["vv"], tmps_["34_oo"])
    del tmps_["34_oo"]

    # H22 += -1.00 d(b,e) d(a,f) <m,j||c,d> t1(c,j) t1(d,i)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('ebim,af->abiefm', tmps_["42_vvoo"], Id["vv"])

    # H22 += +1.00 d(a,e) d(b,f) <m,j||c,d> t1(c,j) t1(d,i)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('eaim,bf->abiefm', tmps_["42_vvoo"], Id["vv"])
    del tmps_["42_vvoo"]

    # tmps_[19_vo](a,i) = 1.00 f[ov](j,c) * t2(c,a,i,j) // flops: o1v1 = o2v2 | mem: o1v1 = o1v1
    tmps_["19_vo"] = einsum('jc,caij->ai', f["ov"], t2)

    # H21 += +1.00 d(b,e) f(j,c) t2(c,a,i,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('ai,be->abie', tmps_["19_vo"], Id["vv"])

    # H21 += -1.00 d(a,e) f(j,c) t2(c,b,i,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('bi,ae->abie', tmps_["19_vo"], Id["vv"])
    del tmps_["19_vo"]

    # tmps_[20_vo](b,i) = 1.00 eri[vovo](b,j,c,i) * t1(c,j) // flops: o1v1 = o2v2 | mem: o1v1 = o1v1
    tmps_["20_vo"] = einsum('bjci,cj->bi', eri["vovo"], t1)

    # H21 += -1.00 d(b,e) <j,a||c,i> t1(c,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('ai,be->abie', tmps_["20_vo"], Id["vv"])

    # H21 += +1.00 d(a,e) <j,b||c,i> t1(c,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('bi,ae->abie', tmps_["20_vo"], Id["vv"])
    del tmps_["20_vo"]

    # tmps_[21_oo](i,k) = 1.00 eri[oovo](k,j,c,i) * t1(c,j) // flops: o2v0 = o3v1 | mem: o2v0 = o2v0
    tmps_["21_oo"] = einsum('kjci,cj->ik', eri["oovo"], t1)

    # tmps_[35_vo](b,i) = 1.00 t1(b,k) * eri[oovo](k,j,c,i) * t1(c,j) // flops: o1v1 = o2v1 | mem: o1v1 = o1v1
    tmps_["35_vo"] = einsum('bk,ik->bi', t1, tmps_["21_oo"])

    # H21 += +1.00 d(a,e) <k,j||c,i> t1(c,j) t1(b,k)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('bi,ae->abie', tmps_["35_vo"], Id["vv"])

    # H21 += -1.00 d(b,e) <k,j||c,i> t1(c,j) t1(a,k)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('ai,be->abie', tmps_["35_vo"], Id["vv"])
    del tmps_["35_vo"]

    # tmps_[38_vvoo](e,a,m,i) = 1.00 Id[vv](a,e) * eri[oovo](m,j,c,i) * t1(c,j) // flops: o2v2 = o2v2 | mem: o2v2 = o2v2
    tmps_["38_vvoo"] = einsum('ae,im->eami', Id["vv"], tmps_["21_oo"])
    del tmps_["21_oo"]

    # H22 += -1.00 d(b,e) d(a,f) <m,j||c,i> t1(c,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('ebmi,af->abiefm', tmps_["38_vvoo"], Id["vv"])

    # H22 += +1.00 d(a,e) d(b,f) <m,j||c,i> t1(c,j)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('eami,bf->abiefm', tmps_["38_vvoo"], Id["vv"])
    del tmps_["38_vvoo"]

    # H11 += -1.00 f(i,e) t1(a,i)  // flops: o0v2 += o0v2 | mem: o0v2 += o0v2
    H11 -= einsum('ae->ae', tmps_["22_vv"])
    del tmps_["22_vv"]

    # tmps_[23_vo](a,i) = 1.00 f[vv](a,c) * t1(c,i) // flops: o1v1 = o1v2 | mem: o1v1 = o1v1
    tmps_["23_vo"] = einsum('ac,ci->ai', f["vv"], t1)

    # H21 += +1.00 d(a,e) f(b,c) t1(c,i)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('bi,ae->abie', tmps_["23_vo"], Id["vv"])

    # H21 += -1.00 d(b,e) f(a,c) t1(c,i)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('ai,be->abie', tmps_["23_vo"], Id["vv"])
    del tmps_["23_vo"]

    # tmps_[24_oo](i,j) = 1.00 f[ov](j,c) * t1(c,i) // flops: o2v0 = o2v1 | mem: o2v0 = o2v0
    tmps_["24_oo"] = einsum('jc,ci->ij', f["ov"], t1)

    # tmps_[36_vo](a,i) = 1.00 t1(a,j) * f[ov](j,c) * t1(c,i) // flops: o1v1 = o2v1 | mem: o1v1 = o1v1
    tmps_["36_vo"] = einsum('aj,ij->ai', t1, tmps_["24_oo"])

    # H21 += +1.00 d(b,e) f(j,c) t1(c,i) t1(a,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('ai,be->abie', tmps_["36_vo"], Id["vv"])

    # H21 += -1.00 d(a,e) f(j,c) t1(c,i) t1(b,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('bi,ae->abie', tmps_["36_vo"], Id["vv"])
    del tmps_["36_vo"]

    # tmps_[39_vvoo](e,b,j,i) = 1.00 Id[vv](b,e) * f[ov](j,c) * t1(c,i) // flops: o2v2 = o2v2 | mem: o2v2 = o2v2
    tmps_["39_vvoo"] = einsum('be,ij->ebji', Id["vv"], tmps_["24_oo"])
    del tmps_["24_oo"]

    # H22 += +1.00 d(b,e) d(a,f) f(m,c) t1(c,i)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('ebmi,af->abiefm', tmps_["39_vvoo"], Id["vv"])

    # H22 += -1.00 d(a,e) d(b,f) f(m,c) t1(c,i)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('eami,bf->abiefm', tmps_["39_vvoo"], Id["vv"])
    del tmps_["39_vvoo"]

    # tmps_[25_vo](a,i) = 1.00 f[oo](j,i) * t1(a,j) // flops: o1v1 = o2v1 | mem: o1v1 = o1v1
    tmps_["25_vo"] = einsum('ji,aj->ai', f["oo"], t1)

    # H21 += +1.00 d(b,e) f(j,i) t1(a,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 += einsum('ai,be->abie', tmps_["25_vo"], Id["vv"])

    # H21 += -1.00 d(a,e) f(j,i) t1(b,j)  // flops: o1v3 += o1v3 | mem: o1v3 += o1v3
    H21 -= einsum('bi,ae->abie', tmps_["25_vo"], Id["vv"])
    del tmps_["25_vo"]

    # tmps_[26_vvoo](e,b,i,j) = 1.00 Id[vv](b,e) * f[oo](j,i) // flops: o2v2 = o2v2 | mem: o2v2 = o2v2
    tmps_["26_vvoo"] = einsum('be,ji->ebij', Id["vv"], f["oo"])

    # H22 += -1.00 d(a,e) d(b,f) f(m,i)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 -= einsum('eaim,bf->abiefm', tmps_["26_vvoo"], Id["vv"])

    # H22 += +1.00 d(b,e) d(a,f) f(m,i)  // flops: o2v4 += o2v4 | mem: o2v4 += o2v4
    H22 += einsum('ebim,af->abiefm', tmps_["26_vvoo"], Id["vv"])
    del tmps_["26_vvoo"]

    return H11, H12, H21, H22
    
def kernel(t1, t2, fock, g, o, v, e_ai, e_abij, max_iter=100, stopping_eps=1.0E-14,
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

        singles_res = singles_residual(t1, t2, fock, g, o, v) + fock_e_ai * t1
        doubles_res = doubles_residual(t1, t2, fock, g, o, v) + fock_e_abij * t2

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


    basis = '6-31g'
    mol = pyscf.gto.Mole(
        atom='Ne 0 0 0',
        charge=0,
        spin=0,
        basis=basis)

    print(mol.charge, mol.spin, mol.nelectron)

    mf = mol.RHF().run()
    mycc = mf.CCSD().run()
    print('CCSD correlation energy', mycc.e_corr)

    molecule = of.MolecularData(geometry=[['Ne', (0, 0, 0)]],
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
    #print(hf_energy_test, hf_energy)
    assert np.isclose(hf_energy + molecule.nuclear_repulsion, molecule.hf_energy)

    g = gtei
    nsvirt = 2 * (norbs - nocc)
    nsocc = 2 * nocc
    #t1f, t2f = kernel(np.zeros((nsvirt, nsocc)), np.zeros((nsvirt, nsvirt, nsocc, nsocc)), fock, g, o, v, e_ai, e_abij)
    #print(ccsd_energy(t1f, t2f, fock, g, o, v) - hf_energy)

    t1f, t2f = kernel(np.zeros((nsvirt, nsocc)), np.zeros((nsvirt, nsvirt, nsocc, nsocc)), fock, g, o, v, e_ai, e_abij,
                      diis_size=8, diis_start_cycle=4)
    #print(ccsd_energy(t1f, t2f, fock, g, o, v) - hf_energy)

    # now eom-ccsd?
    #kd = np.eye(fock.shape[0])
    kd = np.zeros((2*norbs,2*norbs))
    for i in range (0,2*norbs):
        kd[i,i] = 1.0

    f_map, eri_map, Id_map = integral_maps(fock,g,kd,o,v)
    Hss, Hsd, Hds, Hdd = build_ea_Hbar(Id_map, f_map, eri_map, o, v, t1f, t2f)
    print(Hss.shape, Hsd.shape, Hds.shape, Hdd.shape)

    print("MATRIX")
    for i in range(0, Hss.shape[0]):
        for j in range(0, Hss.shape[1]):
            print(f"{Hss[i, j]:>7.2f}", end=' ')
        print()

    #
    dim = nsvirt + nsvirt*(nsvirt-1)//2*nsocc
    H = np.zeros((dim,dim))
    print(H.shape)

    # ss block
    for a in range(0, nsvirt):
        for e in range(0, nsvirt):
            H[a, e] = Hss[a, e]

    # sd ds block
    for a in range(0, nsvirt):
        efm = nsvirt
        for e in range(0, nsvirt):
            for f in range(e+1, nsvirt):
                for m in range(0, nsocc):
                    H[a, efm] = Hsd[a,e,f,m]
                    H[efm, a] = Hds[e,f,m,a]
                    efm += 1


    # dd block
    abi = nsvirt
    for a in range(0, nsvirt):
        for b in range(a+1, nsvirt):
            for i in range(0, nsocc):

                efm = nsvirt
                for e in range(0, nsvirt):
                    for f in range(e+1, nsvirt):
                        for m in range(0, nsocc):
                            H[abi, efm] = Hdd[a,b,i,e,f,m]
                            efm += 1
                abi += 1


    cc_energy = ccsd_energy(t1f, t2f, fock, g, o, v)
    print('    ccsd energy: %20.12f' % (cc_energy + molecule.nuclear_repulsion) )

    for i in range(dim):
        H[i,i] += cc_energy + molecule.nuclear_repulsion

    print('')
    print('    eigenvalues of e(-T) H e(T):')
    print('')

    print('    %20s %20s' % ('total energy','excitation energy'))
    en, vec = np.linalg.eig(H)
    args = np.argsort(en)
    en = en[args]
    vec = vec[args,args]
    last_en = 0.0
    for i in range (0,len(en)):
        if abs(en[i]-last_en) > 1.0E-12:
            print('    %5d %20.12f %20.12f' % ( i,en[i] - molecule.nuclear_repulsion,en[i]-cc_energy))
        last_en = en[i]
    print('')

if __name__ == "__main__":
    main()

