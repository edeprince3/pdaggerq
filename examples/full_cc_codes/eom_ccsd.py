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

def build_Hbar(kd, f, g, o, v, t1, t2):

    #    H(0;0) = <0| e(-T) H e(T) |0>
    
    #	  1.0000 f(i,i)
    H00 =  1.000000000000000 * einsum('ii', f[o, o])
    
    #	  1.0000 f(i,a)*t1(a,i)
    H00 +=  1.000000000000000 * einsum('ia,ai', f[o, v], t1)
    
    #	 -0.5000 <j,i||j,i>
    H00 += -0.500000000000000 * einsum('jiji', g[o, o, o, o])
    
    #	  0.2500 <j,i||a,b>*t2(a,b,j,i)
    H00 +=  0.250000000000000 * einsum('jiab,abji', g[o, o, v, v], t2)
    
    #	 -0.5000 <j,i||a,b>*t1(a,i)*t1(b,j)
    H00 += -0.500000000000000 * einsum('jiab,ai,bj', g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    
    #    H(m,e;0) = <0|e1(m,e) e(-T) H e(T) |0>
    
    #	  1.0000 f(e,m)
    Hs0 =  1.000000000000000 * einsum('em->em', f[v, o])
    
    #	 -1.0000 f(i,m)*t1(e,i)
    Hs0 += -1.000000000000000 * einsum('im,ei->em', f[o, o], t1)
    
    #	  1.0000 f(e,a)*t1(a,m)
    Hs0 +=  1.000000000000000 * einsum('ea,am->em', f[v, v], t1)
    
    #	 -1.0000 f(i,a)*t2(a,e,m,i)
    Hs0 += -1.000000000000000 * einsum('ia,aemi->em', f[o, v], t2)
    
    #	 -1.0000 f(i,a)*t1(a,m)*t1(e,i)
    Hs0 += -1.000000000000000 * einsum('ia,am,ei->em', f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <i,e||a,m>*t1(a,i)
    Hs0 +=  1.000000000000000 * einsum('ieam,ai->em', g[o, v, v, o], t1)
    
    #	 -0.5000 <j,i||a,m>*t2(a,e,j,i)
    Hs0 += -0.500000000000000 * einsum('jiam,aeji->em', g[o, o, v, o], t2)
    
    #	 -0.5000 <i,e||a,b>*t2(a,b,m,i)
    Hs0 += -0.500000000000000 * einsum('ieab,abmi->em', g[o, v, v, v], t2)
    
    #	  1.0000 <j,i||a,b>*t1(a,i)*t2(b,e,m,j)
    Hs0 +=  1.000000000000000 * einsum('jiab,ai,bemj->em', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  0.5000 <j,i||a,b>*t1(a,m)*t2(b,e,j,i)
    Hs0 +=  0.500000000000000 * einsum('jiab,am,beji->em', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  0.5000 <j,i||a,b>*t1(e,i)*t2(a,b,m,j)
    Hs0 +=  0.500000000000000 * einsum('jiab,ei,abmj->em', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <j,i||a,m>*t1(a,i)*t1(e,j)
    Hs0 +=  1.000000000000000 * einsum('jiam,ai,ej->em', g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <i,e||a,b>*t1(a,i)*t1(b,m)
    Hs0 +=  1.000000000000000 * einsum('ieab,ai,bm->em', g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 <j,i||a,b>*t1(a,i)*t1(b,m)*t1(e,j)
    Hs0 +=  1.000000000000000 * einsum('jiab,ai,bm,ej->em', g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    
    #    H(0;i,a) = <0| e(-T) H e(T) e1(a,i)|0>
    
    #	  1.0000 f(i,a)
    H0s =  1.000000000000000 * einsum('ia->ai', f[o, v])
    
    #	 -1.0000 <i,j||b,a>*t1(b,j)
    H0s += -1.000000000000000 * einsum('ijba,bj->ai', g[o, o, v, v], t1)
    
    
    #    H(m,n,e,f;0) = <0|e2(m,n,f,e) e(-T) H e(T) |0>
    
    #	 -1.0000 P(m,n)f(i,n)*t2(e,f,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('in,efmi->efmn', f[o, o], t2)
    Hd0 =  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(e,f)f(e,a)*t2(a,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('ea,afmn->efmn', f[v, v], t2)
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -1.0000 P(m,n)f(i,a)*t1(a,n)*t2(e,f,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('ia,an,efmi->efmn', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -1.0000 P(e,f)f(i,a)*t1(e,i)*t2(a,f,m,n)
    contracted_intermediate = -1.000000000000000 * einsum('ia,ei,afmn->efmn', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  1.0000 <e,f||m,n>
    Hd0 +=  1.000000000000000 * einsum('efmn->efmn', g[v, v, o, o])
    
    #	  1.0000 P(e,f)<i,e||m,n>*t1(f,i)
    contracted_intermediate =  1.000000000000000 * einsum('iemn,fi->efmn', g[o, v, o, o], t1)
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  1.0000 P(m,n)<e,f||a,n>*t1(a,m)
    contracted_intermediate =  1.000000000000000 * einsum('efan,am->efmn', g[v, v, v, o], t1)
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.5000 <j,i||m,n>*t2(e,f,j,i)
    Hd0 +=  0.500000000000000 * einsum('jimn,efji->efmn', g[o, o, o, o], t2)
    
    #	  1.0000 P(m,n)*P(e,f)<i,e||a,n>*t2(a,f,m,i)
    contracted_intermediate =  1.000000000000000 * einsum('iean,afmi->efmn', g[o, v, v, o], t2)
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  0.5000 <e,f||a,b>*t2(a,b,m,n)
    Hd0 +=  0.500000000000000 * einsum('efab,abmn->efmn', g[v, v, v, v], t2)
    
    #	  1.0000 P(m,n)<j,i||a,n>*t1(a,i)*t2(e,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jian,ai,efmj->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.5000 P(m,n)<j,i||a,n>*t1(a,m)*t2(e,f,j,i)
    contracted_intermediate =  0.500000000000000 * einsum('jian,am,efji->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -1.0000 P(m,n)*P(e,f)<j,i||a,n>*t1(e,i)*t2(a,f,m,j)
    contracted_intermediate = -1.000000000000000 * einsum('jian,ei,afmj->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  1.0000 P(e,f)<i,e||a,b>*t1(a,i)*t2(b,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('ieab,ai,bfmn->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -1.0000 P(m,n)*P(e,f)<i,e||a,b>*t1(a,n)*t2(b,f,m,i)
    contracted_intermediate = -1.000000000000000 * einsum('ieab,an,bfmi->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	  0.5000 P(e,f)<i,e||a,b>*t1(f,i)*t2(a,b,m,n)
    contracted_intermediate =  0.500000000000000 * einsum('ieab,fi,abmn->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -1.0000 <j,i||m,n>*t1(e,i)*t1(f,j)
    Hd0 += -1.000000000000000 * einsum('jimn,ei,fj->efmn', g[o, o, o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(m,n)*P(e,f)<i,e||a,n>*t1(a,m)*t1(f,i)
    contracted_intermediate =  1.000000000000000 * einsum('iean,am,fi->efmn', g[o, v, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	 -1.0000 <e,f||a,b>*t1(a,n)*t1(b,m)
    Hd0 += -1.000000000000000 * einsum('efab,an,bm->efmn', g[v, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 P(m,n)<j,i||a,b>*t2(a,b,n,i)*t2(e,f,m,j)
    contracted_intermediate = -0.500000000000000 * einsum('jiab,abni,efmj->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  0.2500 <j,i||a,b>*t2(a,b,m,n)*t2(e,f,j,i)
    Hd0 +=  0.250000000000000 * einsum('jiab,abmn,efji->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	 -0.5000 <j,i||a,b>*t2(a,e,j,i)*t2(b,f,m,n)
    Hd0 += -0.500000000000000 * einsum('jiab,aeji,bfmn->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    
    #	  1.0000 P(m,n)<j,i||a,b>*t2(a,e,n,i)*t2(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,aeni,bfmj->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -0.5000 <j,i||a,b>*t2(a,e,m,n)*t2(b,f,j,i)
    Hd0 += -0.500000000000000 * einsum('jiab,aemn,bfji->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 P(m,n)<j,i||a,b>*t1(a,i)*t1(b,n)*t2(e,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,ai,bn,efmj->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	  1.0000 P(e,f)<j,i||a,b>*t1(a,i)*t1(e,j)*t2(b,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,ai,ej,bfmn->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	 -0.5000 <j,i||a,b>*t1(a,n)*t1(b,m)*t2(e,f,j,i)
    Hd0 += -0.500000000000000 * einsum('jiab,an,bm,efji->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 P(m,n)*P(e,f)<j,i||a,b>*t1(a,n)*t1(e,i)*t2(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('jiab,an,ei,bfmj->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate)  + -1.00000 * einsum('efmn->femn', contracted_intermediate)  +  1.00000 * einsum('efmn->fenm', contracted_intermediate) 
    
    #	 -0.5000 <j,i||a,b>*t1(e,i)*t1(f,j)*t2(a,b,m,n)
    Hd0 += -0.500000000000000 * einsum('jiab,ei,fj,abmn->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(m,n)<j,i||a,n>*t1(a,m)*t1(e,i)*t1(f,j)
    contracted_intermediate = -1.000000000000000 * einsum('jian,am,ei,fj->efmn', g[o, o, v, o], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->efnm', contracted_intermediate) 
    
    #	 -1.0000 P(e,f)<i,e||a,b>*t1(a,n)*t1(b,m)*t1(f,i)
    contracted_intermediate = -1.000000000000000 * einsum('ieab,an,bm,fi->efmn', g[o, v, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
    Hd0 +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmn->femn', contracted_intermediate) 
    
    #	  1.0000 <j,i||a,b>*t1(a,n)*t1(b,m)*t1(e,i)*t1(f,j)
    Hd0 +=  1.000000000000000 * einsum('jiab,an,bm,ei,fj->efmn', g[o, o, v, v], t1, t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])
    
    
    #    H(0;i,j,a,b) = <0| e(-T) H e(T) e2(a,b,j,i)|0>
    
    #	  1.0000 <i,j||a,b>
    H0d =  1.000000000000000 * einsum('ijab->abij', g[o, o, v, v])
    
    
    #    H(m,e;i,a) = <0|e1(m,e) e(-T) H e(T) e1(a,i)|0>
    
    #	  1.0000 d(e,a)*d(m,i)*f(j,j)
    Hss =  1.000000000000000 * einsum('ea,mi,jj->emai', kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*f(i,m)
    Hss += -1.000000000000000 * einsum('ea,im->emai', kd[v, v], f[o, o])
    
    #	  1.0000 d(m,i)*f(e,a)
    Hss +=  1.000000000000000 * einsum('mi,ea->emai', kd[o, o], f[v, v])
    
    #	  1.0000 d(e,a)*d(m,i)*f(j,b)*t1(b,j)
    Hss +=  1.000000000000000 * einsum('ea,mi,jb,bj->emai', kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*f(i,b)*t1(b,m)
    Hss += -1.000000000000000 * einsum('ea,ib,bm->emai', kd[v, v], f[o, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(m,i)*f(j,a)*t1(e,j)
    Hss += -1.000000000000000 * einsum('mi,ja,ej->emai', kd[o, o], f[o, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(m,i)*<k,j||k,j>
    Hss += -0.500000000000000 * einsum('ea,mi,kjkj->emai', kd[v, v], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 2), (0, 1)])
    
    #	  1.0000 <i,e||a,m>
    Hss +=  1.000000000000000 * einsum('ieam->emai', g[o, v, v, o])
    
    #	  1.0000 d(e,a)*<i,j||b,m>*t1(b,j)
    Hss +=  1.000000000000000 * einsum('ea,ijbm,bj->emai', kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 <i,j||a,m>*t1(e,j)
    Hss += -1.000000000000000 * einsum('ijam,ej->emai', g[o, o, v, o], t1)
    
    #	  1.0000 d(m,i)*<j,e||b,a>*t1(b,j)
    Hss +=  1.000000000000000 * einsum('mi,jeba,bj->emai', kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 <i,e||b,a>*t1(b,m)
    Hss += -1.000000000000000 * einsum('ieba,bm->emai', g[o, v, v, v], t1)
    
    #	  0.2500 d(e,a)*d(m,i)*<k,j||b,c>*t2(b,c,k,j)
    Hss +=  0.250000000000000 * einsum('ea,mi,kjbc,bckj->emai', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (2, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*<i,j||b,c>*t2(b,c,m,j)
    Hss += -0.500000000000000 * einsum('ea,ijbc,bcmj->emai', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(m,i)*<k,j||b,a>*t2(b,e,k,j)
    Hss += -0.500000000000000 * einsum('mi,kjba,bekj->emai', kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 <i,j||b,a>*t2(b,e,m,j)
    Hss +=  1.000000000000000 * einsum('ijba,bemj->emai', g[o, o, v, v], t2)
    
    #	 -0.5000 d(e,a)*d(m,i)*<k,j||b,c>*t1(b,j)*t1(c,k)
    Hss += -0.500000000000000 * einsum('ea,mi,kjbc,bj,ck->emai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (2, 3), (2, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*<i,j||b,c>*t1(b,j)*t1(c,m)
    Hss +=  1.000000000000000 * einsum('ea,ijbc,bj,cm->emai', kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(m,i)*<k,j||b,a>*t1(b,j)*t1(e,k)
    Hss +=  1.000000000000000 * einsum('mi,kjba,bj,ek->emai', kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 <i,j||b,a>*t1(b,m)*t1(e,j)
    Hss +=  1.000000000000000 * einsum('ijba,bm,ej->emai', g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])
    
    
    #    H(m,e;i,j,a,b) = <0|e1(m,e) e(-T) H e(T) e2(a,b,j,i)|0>
    
    #	 -1.0000 d(e,b)*d(m,i)*f(j,a)
    Hsd = -1.000000000000000 * einsum('eb,mi,ja->emabij', kd[v, v], kd[o, o], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,b)*d(m,j)*f(i,a)
    Hsd +=  1.000000000000000 * einsum('eb,mj,ia->emabij', kd[v, v], kd[o, o], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,a)*d(m,i)*f(j,b)
    Hsd +=  1.000000000000000 * einsum('ea,mi,jb->emabij', kd[v, v], kd[o, o], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,a)*d(m,j)*f(i,b)
    Hsd += -1.000000000000000 * einsum('ea,mj,ib->emabij', kd[v, v], kd[o, o], f[o, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,b)*<i,j||a,m>
    Hsd += -1.000000000000000 * einsum('eb,ijam->emabij', kd[v, v], g[o, o, v, o])
    
    #	  1.0000 d(e,a)*<i,j||b,m>
    Hsd +=  1.000000000000000 * einsum('ea,ijbm->emabij', kd[v, v], g[o, o, v, o])
    
    #	 -1.0000 d(m,i)*<j,e||a,b>
    Hsd += -1.000000000000000 * einsum('mi,jeab->emabij', kd[o, o], g[o, v, v, v])
    
    #	  1.0000 d(m,j)*<i,e||a,b>
    Hsd +=  1.000000000000000 * einsum('mj,ieab->emabij', kd[o, o], g[o, v, v, v])
    
    #	  1.0000 d(e,b)*d(m,i)*<j,k||c,a>*t1(c,k)
    Hsd +=  1.000000000000000 * einsum('eb,mi,jkca,ck->emabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(m,j)*<i,k||c,a>*t1(c,k)
    Hsd += -1.000000000000000 * einsum('eb,mj,ikca,ck->emabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,i)*<j,k||c,b>*t1(c,k)
    Hsd += -1.000000000000000 * einsum('ea,mi,jkcb,ck->emabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,j)*<i,k||c,b>*t1(c,k)
    Hsd +=  1.000000000000000 * einsum('ea,mj,ikcb,ck->emabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*<i,j||c,a>*t1(c,m)
    Hsd +=  1.000000000000000 * einsum('eb,ijca,cm->emabij', kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*<i,j||c,b>*t1(c,m)
    Hsd += -1.000000000000000 * einsum('ea,ijcb,cm->emabij', kd[v, v], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(m,i)*<j,k||a,b>*t1(e,k)
    Hsd +=  1.000000000000000 * einsum('mi,jkab,ek->emabij', kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(m,j)*<i,k||a,b>*t1(e,k)
    Hsd += -1.000000000000000 * einsum('mj,ikab,ek->emabij', kd[o, o], g[o, o, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    
    #    H(m,n,e,f;i,a) = <0|e2(m,n,f,e) e(-T) H e(T) e1(a,i)|0>
    
    #	 -1.0000 d(f,a)*d(m,i)*f(e,n)
    Hds = -1.000000000000000 * einsum('fa,mi,en->efmnai', kd[v, v], kd[o, o], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(f,a)*d(n,i)*f(e,m)
    Hds +=  1.000000000000000 * einsum('fa,ni,em->efmnai', kd[v, v], kd[o, o], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,a)*d(m,i)*f(f,n)
    Hds +=  1.000000000000000 * einsum('ea,mi,fn->efmnai', kd[v, v], kd[o, o], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,a)*d(n,i)*f(f,m)
    Hds += -1.000000000000000 * einsum('ea,ni,fm->efmnai', kd[v, v], kd[o, o], f[v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(f,a)*d(m,i)*f(j,n)*t1(e,j)
    Hds +=  1.000000000000000 * einsum('fa,mi,jn,ej->efmnai', kd[v, v], kd[o, o], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,i)*f(j,n)*t1(f,j)
    Hds += -1.000000000000000 * einsum('ea,mi,jn,fj->efmnai', kd[v, v], kd[o, o], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,i)*f(j,m)*t1(e,j)
    Hds += -1.000000000000000 * einsum('fa,ni,jm,ej->efmnai', kd[v, v], kd[o, o], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,i)*f(j,m)*t1(f,j)
    Hds +=  1.000000000000000 * einsum('ea,ni,jm,fj->efmnai', kd[v, v], kd[o, o], f[o, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*f(e,b)*t1(b,n)
    Hds += -1.000000000000000 * einsum('fa,mi,eb,bn->efmnai', kd[v, v], kd[o, o], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*f(e,b)*t1(b,m)
    Hds +=  1.000000000000000 * einsum('fa,ni,eb,bm->efmnai', kd[v, v], kd[o, o], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,i)*f(f,b)*t1(b,n)
    Hds +=  1.000000000000000 * einsum('ea,mi,fb,bn->efmnai', kd[v, v], kd[o, o], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*f(f,b)*t1(b,m)
    Hds += -1.000000000000000 * einsum('ea,ni,fb,bm->efmnai', kd[v, v], kd[o, o], f[v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,i)*f(j,b)*t2(b,e,n,j)
    Hds +=  1.000000000000000 * einsum('fa,mi,jb,benj->efmnai', kd[v, v], kd[o, o], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,i)*f(j,b)*t2(b,e,m,j)
    Hds += -1.000000000000000 * einsum('fa,ni,jb,bemj->efmnai', kd[v, v], kd[o, o], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*f(i,b)*t2(b,e,m,n)
    Hds +=  1.000000000000000 * einsum('fa,ib,bemn->efmnai', kd[v, v], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,i)*f(j,b)*t2(b,f,n,j)
    Hds += -1.000000000000000 * einsum('ea,mi,jb,bfnj->efmnai', kd[v, v], kd[o, o], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,i)*f(j,b)*t2(b,f,m,j)
    Hds +=  1.000000000000000 * einsum('ea,ni,jb,bfmj->efmnai', kd[v, v], kd[o, o], f[o, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*f(i,b)*t2(b,f,m,n)
    Hds += -1.000000000000000 * einsum('ea,ib,bfmn->efmnai', kd[v, v], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(m,i)*f(j,a)*t2(e,f,n,j)
    Hds +=  1.000000000000000 * einsum('mi,ja,efnj->efmnai', kd[o, o], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(n,i)*f(j,a)*t2(e,f,m,j)
    Hds += -1.000000000000000 * einsum('ni,ja,efmj->efmnai', kd[o, o], f[o, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,i)*f(j,b)*t1(b,n)*t1(e,j)
    Hds +=  1.000000000000000 * einsum('fa,mi,jb,bn,ej->efmnai', kd[v, v], kd[o, o], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,i)*f(j,b)*t1(b,n)*t1(f,j)
    Hds += -1.000000000000000 * einsum('ea,mi,jb,bn,fj->efmnai', kd[v, v], kd[o, o], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,i)*f(j,b)*t1(b,m)*t1(e,j)
    Hds += -1.000000000000000 * einsum('fa,ni,jb,bm,ej->efmnai', kd[v, v], kd[o, o], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,i)*f(j,b)*t1(b,m)*t1(f,j)
    Hds +=  1.000000000000000 * einsum('ea,ni,jb,bm,fj->efmnai', kd[v, v], kd[o, o], f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*<i,e||m,n>
    Hds +=  1.000000000000000 * einsum('fa,iemn->efmnai', kd[v, v], g[o, v, o, o])
    
    #	 -1.0000 d(e,a)*<i,f||m,n>
    Hds += -1.000000000000000 * einsum('ea,ifmn->efmnai', kd[v, v], g[o, v, o, o])
    
    #	  1.0000 d(m,i)*<e,f||a,n>
    Hds +=  1.000000000000000 * einsum('mi,efan->efmnai', kd[o, o], g[v, v, v, o])
    
    #	 -1.0000 d(n,i)*<e,f||a,m>
    Hds += -1.000000000000000 * einsum('ni,efam->efmnai', kd[o, o], g[v, v, v, o])
    
    #	 -1.0000 d(f,a)*<i,j||m,n>*t1(e,j)
    Hds += -1.000000000000000 * einsum('fa,ijmn,ej->efmnai', kd[v, v], g[o, o, o, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*<i,j||m,n>*t1(f,j)
    Hds +=  1.000000000000000 * einsum('ea,ijmn,fj->efmnai', kd[v, v], g[o, o, o, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<j,e||b,n>*t1(b,j)
    Hds += -1.000000000000000 * einsum('fa,mi,jebn,bj->efmnai', kd[v, v], kd[o, o], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 P(m,n)d(f,a)*<i,e||b,n>*t1(b,m)
    contracted_intermediate =  1.000000000000000 * einsum('fa,iebn,bm->efmnai', kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	  1.0000 P(e,f)d(m,i)*<j,e||a,n>*t1(f,j)
    contracted_intermediate =  1.000000000000000 * einsum('mi,jean,fj->efmnai', kd[o, o], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  1.0000 d(f,a)*d(n,i)*<j,e||b,m>*t1(b,j)
    Hds +=  1.000000000000000 * einsum('fa,ni,jebm,bj->efmnai', kd[v, v], kd[o, o], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 P(e,f)d(n,i)*<j,e||a,m>*t1(f,j)
    contracted_intermediate = -1.000000000000000 * einsum('ni,jeam,fj->efmnai', kd[o, o], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  1.0000 d(e,a)*d(m,i)*<j,f||b,n>*t1(b,j)
    Hds +=  1.000000000000000 * einsum('ea,mi,jfbn,bj->efmnai', kd[v, v], kd[o, o], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 P(m,n)d(e,a)*<i,f||b,n>*t1(b,m)
    contracted_intermediate = -1.000000000000000 * einsum('ea,ifbn,bm->efmnai', kd[v, v], g[o, v, v, o], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	 -1.0000 d(e,a)*d(n,i)*<j,f||b,m>*t1(b,j)
    Hds += -1.000000000000000 * einsum('ea,ni,jfbm,bj->efmnai', kd[v, v], kd[o, o], g[o, v, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(m,i)*<e,f||b,a>*t1(b,n)
    Hds += -1.000000000000000 * einsum('mi,efba,bn->efmnai', kd[o, o], g[v, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(n,i)*<e,f||b,a>*t1(b,m)
    Hds +=  1.000000000000000 * einsum('ni,efba,bm->efmnai', kd[o, o], g[v, v, v, v], t1, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(m,i)*<k,j||b,n>*t2(b,e,k,j)
    Hds +=  0.500000000000000 * einsum('fa,mi,kjbn,bekj->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 P(m,n)d(f,a)*<i,j||b,n>*t2(b,e,m,j)
    contracted_intermediate = -1.000000000000000 * einsum('fa,ijbn,bemj->efmnai', kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	 -0.5000 d(e,a)*d(m,i)*<k,j||b,n>*t2(b,f,k,j)
    Hds += -0.500000000000000 * einsum('ea,mi,kjbn,bfkj->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 P(m,n)d(e,a)*<i,j||b,n>*t2(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('ea,ijbn,bfmj->efmnai', kd[v, v], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	  0.5000 d(m,i)*<k,j||a,n>*t2(e,f,k,j)
    Hds +=  0.500000000000000 * einsum('mi,kjan,efkj->efmnai', kd[o, o], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 P(m,n)<i,j||a,n>*t2(e,f,m,j)
    contracted_intermediate = -1.000000000000000 * einsum('ijan,efmj->efmnai', g[o, o, v, o], t2)
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	 -0.5000 d(f,a)*d(n,i)*<k,j||b,m>*t2(b,e,k,j)
    Hds += -0.500000000000000 * einsum('fa,ni,kjbm,bekj->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(e,a)*d(n,i)*<k,j||b,m>*t2(b,f,k,j)
    Hds +=  0.500000000000000 * einsum('ea,ni,kjbm,bfkj->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(n,i)*<k,j||a,m>*t2(e,f,k,j)
    Hds += -0.500000000000000 * einsum('ni,kjam,efkj->efmnai', kd[o, o], g[o, o, v, o], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(m,i)*<j,e||b,c>*t2(b,c,n,j)
    Hds +=  0.500000000000000 * einsum('fa,mi,jebc,bcnj->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(n,i)*<j,e||b,c>*t2(b,c,m,j)
    Hds += -0.500000000000000 * einsum('fa,ni,jebc,bcmj->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(f,a)*<i,e||b,c>*t2(b,c,m,n)
    Hds +=  0.500000000000000 * einsum('fa,iebc,bcmn->efmnai', kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 P(e,f)d(m,i)*<j,e||b,a>*t2(b,f,n,j)
    contracted_intermediate = -1.000000000000000 * einsum('mi,jeba,bfnj->efmnai', kd[o, o], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  1.0000 P(e,f)d(n,i)*<j,e||b,a>*t2(b,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('ni,jeba,bfmj->efmnai', kd[o, o], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	 -1.0000 P(e,f)<i,e||b,a>*t2(b,f,m,n)
    contracted_intermediate = -1.000000000000000 * einsum('ieba,bfmn->efmnai', g[o, v, v, v], t2)
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	 -0.5000 d(e,a)*d(m,i)*<j,f||b,c>*t2(b,c,n,j)
    Hds += -0.500000000000000 * einsum('ea,mi,jfbc,bcnj->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.5000 d(e,a)*d(n,i)*<j,f||b,c>*t2(b,c,m,j)
    Hds +=  0.500000000000000 * einsum('ea,ni,jfbc,bcmj->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(e,a)*<i,f||b,c>*t2(b,c,m,n)
    Hds += -0.500000000000000 * einsum('ea,ifbc,bcmn->efmnai', kd[v, v], g[o, v, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<k,j||b,c>*t1(b,j)*t2(c,e,n,k)
    Hds += -1.000000000000000 * einsum('fa,mi,kjbc,bj,cenk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*<k,j||b,c>*t1(b,j)*t2(c,e,m,k)
    Hds +=  1.000000000000000 * einsum('fa,ni,kjbc,bj,cemk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*<i,j||b,c>*t1(b,j)*t2(c,e,m,n)
    Hds += -1.000000000000000 * einsum('fa,ijbc,bj,cemn->efmnai', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,i)*<k,j||b,c>*t1(b,j)*t2(c,f,n,k)
    Hds +=  1.000000000000000 * einsum('ea,mi,kjbc,bj,cfnk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<k,j||b,c>*t1(b,j)*t2(c,f,m,k)
    Hds += -1.000000000000000 * einsum('ea,ni,kjbc,bj,cfmk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*<i,j||b,c>*t1(b,j)*t2(c,f,m,n)
    Hds +=  1.000000000000000 * einsum('ea,ijbc,bj,cfmn->efmnai', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(m,i)*<k,j||b,a>*t1(b,j)*t2(e,f,n,k)
    Hds += -1.000000000000000 * einsum('mi,kjba,bj,efnk->efmnai', kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(n,i)*<k,j||b,a>*t1(b,j)*t2(e,f,m,k)
    Hds +=  1.000000000000000 * einsum('ni,kjba,bj,efmk->efmnai', kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(m,i)*<k,j||b,c>*t1(b,n)*t2(c,e,k,j)
    Hds += -0.500000000000000 * einsum('fa,mi,kjbc,bn,cekj->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 P(m,n)d(f,a)*<i,j||b,c>*t1(b,n)*t2(c,e,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('fa,ijbc,bn,cemj->efmnai', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	  0.5000 d(e,a)*d(m,i)*<k,j||b,c>*t1(b,n)*t2(c,f,k,j)
    Hds +=  0.500000000000000 * einsum('ea,mi,kjbc,bn,cfkj->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 P(m,n)d(e,a)*<i,j||b,c>*t1(b,n)*t2(c,f,m,j)
    contracted_intermediate = -1.000000000000000 * einsum('ea,ijbc,bn,cfmj->efmnai', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	 -0.5000 d(m,i)*<k,j||b,a>*t1(b,n)*t2(e,f,k,j)
    Hds += -0.500000000000000 * einsum('mi,kjba,bn,efkj->efmnai', kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 P(m,n)<i,j||b,a>*t1(b,n)*t2(e,f,m,j)
    contracted_intermediate =  1.000000000000000 * einsum('ijba,bn,efmj->efmnai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	  0.5000 d(f,a)*d(n,i)*<k,j||b,c>*t1(b,m)*t2(c,e,k,j)
    Hds +=  0.500000000000000 * einsum('fa,ni,kjbc,bm,cekj->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(n,i)*<k,j||b,c>*t1(b,m)*t2(c,f,k,j)
    Hds += -0.500000000000000 * einsum('ea,ni,kjbc,bm,cfkj->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(n,i)*<k,j||b,a>*t1(b,m)*t2(e,f,k,j)
    Hds +=  0.500000000000000 * einsum('ni,kjba,bm,efkj->efmnai', kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(m,i)*<k,j||b,c>*t1(e,j)*t2(b,c,n,k)
    Hds += -0.500000000000000 * einsum('fa,mi,kjbc,ej,bcnk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(n,i)*<k,j||b,c>*t1(e,j)*t2(b,c,m,k)
    Hds +=  0.500000000000000 * einsum('fa,ni,kjbc,ej,bcmk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*<i,j||b,c>*t1(e,j)*t2(b,c,m,n)
    Hds += -0.500000000000000 * einsum('fa,ijbc,ej,bcmn->efmnai', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 P(e,f)d(m,i)*<k,j||b,a>*t1(e,j)*t2(b,f,n,k)
    contracted_intermediate =  1.000000000000000 * einsum('mi,kjba,ej,bfnk->efmnai', kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	 -1.0000 P(e,f)d(n,i)*<k,j||b,a>*t1(e,j)*t2(b,f,m,k)
    contracted_intermediate = -1.000000000000000 * einsum('ni,kjba,ej,bfmk->efmnai', kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  1.0000 P(e,f)<i,j||b,a>*t1(e,j)*t2(b,f,m,n)
    contracted_intermediate =  1.000000000000000 * einsum('ijba,ej,bfmn->efmnai', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  0.5000 d(e,a)*d(m,i)*<k,j||b,c>*t1(f,j)*t2(b,c,n,k)
    Hds +=  0.500000000000000 * einsum('ea,mi,kjbc,fj,bcnk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(n,i)*<k,j||b,c>*t1(f,j)*t2(b,c,m,k)
    Hds += -0.500000000000000 * einsum('ea,ni,kjbc,fj,bcmk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*<i,j||b,c>*t1(f,j)*t2(b,c,m,n)
    Hds +=  0.500000000000000 * einsum('ea,ijbc,fj,bcmn->efmnai', kd[v, v], g[o, o, v, v], t1, t2, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<k,j||b,n>*t1(b,j)*t1(e,k)
    Hds += -1.000000000000000 * einsum('fa,mi,kjbn,bj,ek->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,i)*<k,j||b,n>*t1(b,j)*t1(f,k)
    Hds +=  1.000000000000000 * einsum('ea,mi,kjbn,bj,fk->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 P(m,n)d(f,a)*<i,j||b,n>*t1(b,m)*t1(e,j)
    contracted_intermediate = -1.000000000000000 * einsum('fa,ijbn,bm,ej->efmnai', kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	  1.0000 P(m,n)d(e,a)*<i,j||b,n>*t1(b,m)*t1(f,j)
    contracted_intermediate =  1.000000000000000 * einsum('ea,ijbn,bm,fj->efmnai', kd[v, v], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->efnmai', contracted_intermediate) 
    
    #	 -1.0000 d(m,i)*<k,j||a,n>*t1(e,j)*t1(f,k)
    Hds += -1.000000000000000 * einsum('mi,kjan,ej,fk->efmnai', kd[o, o], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*<k,j||b,m>*t1(b,j)*t1(e,k)
    Hds +=  1.000000000000000 * einsum('fa,ni,kjbm,bj,ek->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<k,j||b,m>*t1(b,j)*t1(f,k)
    Hds += -1.000000000000000 * einsum('ea,ni,kjbm,bj,fk->efmnai', kd[v, v], kd[o, o], g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(n,i)*<k,j||a,m>*t1(e,j)*t1(f,k)
    Hds +=  1.000000000000000 * einsum('ni,kjam,ej,fk->efmnai', kd[o, o], g[o, o, v, o], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<j,e||b,c>*t1(b,j)*t1(c,n)
    Hds += -1.000000000000000 * einsum('fa,mi,jebc,bj,cn->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*<j,e||b,c>*t1(b,j)*t1(c,m)
    Hds +=  1.000000000000000 * einsum('fa,ni,jebc,bj,cm->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*<i,e||b,c>*t1(b,n)*t1(c,m)
    Hds += -1.000000000000000 * einsum('fa,iebc,bn,cm->efmnai', kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 P(e,f)d(m,i)*<j,e||b,a>*t1(b,n)*t1(f,j)
    contracted_intermediate = -1.000000000000000 * einsum('mi,jeba,bn,fj->efmnai', kd[o, o], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  1.0000 P(e,f)d(n,i)*<j,e||b,a>*t1(b,m)*t1(f,j)
    contracted_intermediate =  1.000000000000000 * einsum('ni,jeba,bm,fj->efmnai', kd[o, o], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    Hds +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnai->femnai', contracted_intermediate) 
    
    #	  1.0000 d(e,a)*d(m,i)*<j,f||b,c>*t1(b,j)*t1(c,n)
    Hds +=  1.000000000000000 * einsum('ea,mi,jfbc,bj,cn->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<j,f||b,c>*t1(b,j)*t1(c,m)
    Hds += -1.000000000000000 * einsum('ea,ni,jfbc,bj,cm->efmnai', kd[v, v], kd[o, o], g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*<i,f||b,c>*t1(b,n)*t1(c,m)
    Hds +=  1.000000000000000 * einsum('ea,ifbc,bn,cm->efmnai', kd[v, v], g[o, v, v, v], t1, t1, optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<k,j||b,c>*t1(b,j)*t1(c,n)*t1(e,k)
    Hds += -1.000000000000000 * einsum('fa,mi,kjbc,bj,cn,ek->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,i)*<k,j||b,c>*t1(b,j)*t1(c,n)*t1(f,k)
    Hds +=  1.000000000000000 * einsum('ea,mi,kjbc,bj,cn,fk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*<k,j||b,c>*t1(b,j)*t1(c,m)*t1(e,k)
    Hds +=  1.000000000000000 * einsum('fa,ni,kjbc,bj,cm,ek->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<k,j||b,c>*t1(b,j)*t1(c,m)*t1(f,k)
    Hds += -1.000000000000000 * einsum('ea,ni,kjbc,bj,cm,fk->efmnai', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*<i,j||b,c>*t1(b,n)*t1(c,m)*t1(e,j)
    Hds +=  1.000000000000000 * einsum('fa,ijbc,bn,cm,ej->efmnai', kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*<i,j||b,c>*t1(b,n)*t1(c,m)*t1(f,j)
    Hds += -1.000000000000000 * einsum('ea,ijbc,bn,cm,fj->efmnai', kd[v, v], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    #	  1.0000 d(m,i)*<k,j||b,a>*t1(b,n)*t1(e,j)*t1(f,k)
    Hds +=  1.000000000000000 * einsum('mi,kjba,bn,ej,fk->efmnai', kd[o, o], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    #	 -1.0000 d(n,i)*<k,j||b,a>*t1(b,m)*t1(e,j)*t1(f,k)
    Hds += -1.000000000000000 * einsum('ni,kjba,bm,ej,fk->efmnai', kd[o, o], g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (1, 2), (1, 3), (1, 2), (0, 1)])
    
    
    #    H(m,n,e,f;i,j,a,b) = <0|e2(m,n,f,e) e(-T) H e(T) e2(a,b,j,i)|0>
    
    #	  1.0000 d(e,a)*d(f,b)*d(n,j)*d(m,i)*f(k,k)
    Hdd =  1.000000000000000 * einsum('ea,fb,nj,mi,kk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(m,j)*d(n,i)*f(k,k)
    Hdd += -1.000000000000000 * einsum('ea,fb,mj,ni,kk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(n,j)*d(m,i)*f(k,k)
    Hdd += -1.000000000000000 * einsum('fa,eb,nj,mi,kk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(m,j)*d(n,i)*f(k,k)
    Hdd +=  1.000000000000000 * einsum('fa,eb,mj,ni,kk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(m,i)*f(j,n)
    Hdd += -1.000000000000000 * einsum('ea,fb,mi,jn->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(m,j)*f(i,n)
    Hdd +=  1.000000000000000 * einsum('ea,fb,mj,in->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(m,i)*f(j,n)
    Hdd +=  1.000000000000000 * einsum('fa,eb,mi,jn->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(m,j)*f(i,n)
    Hdd += -1.000000000000000 * einsum('fa,eb,mj,in->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(n,i)*f(j,m)
    Hdd +=  1.000000000000000 * einsum('ea,fb,ni,jm->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(n,j)*f(i,m)
    Hdd += -1.000000000000000 * einsum('ea,fb,nj,im->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(n,i)*f(j,m)
    Hdd += -1.000000000000000 * einsum('fa,eb,ni,jm->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(n,j)*f(i,m)
    Hdd +=  1.000000000000000 * einsum('fa,eb,nj,im->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, o], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(f,b)*d(n,j)*d(m,i)*f(e,a)
    Hdd +=  1.000000000000000 * einsum('fb,nj,mi,ea->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(f,b)*d(m,j)*d(n,i)*f(e,a)
    Hdd += -1.000000000000000 * einsum('fb,mj,ni,ea->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(f,a)*d(n,j)*d(m,i)*f(e,b)
    Hdd += -1.000000000000000 * einsum('fa,nj,mi,eb->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(f,a)*d(m,j)*d(n,i)*f(e,b)
    Hdd +=  1.000000000000000 * einsum('fa,mj,ni,eb->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(e,b)*d(n,j)*d(m,i)*f(f,a)
    Hdd += -1.000000000000000 * einsum('eb,nj,mi,fa->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(e,b)*d(m,j)*d(n,i)*f(f,a)
    Hdd +=  1.000000000000000 * einsum('eb,mj,ni,fa->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(e,a)*d(n,j)*d(m,i)*f(f,b)
    Hdd +=  1.000000000000000 * einsum('ea,nj,mi,fb->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	 -1.0000 d(e,a)*d(m,j)*d(n,i)*f(f,b)
    Hdd += -1.000000000000000 * einsum('ea,mj,ni,fb->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[v, v], optimize=['einsum_path', (0, 1, 2, 3)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(n,j)*d(m,i)*f(k,c)*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('ea,fb,nj,mi,kc,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(m,j)*d(n,i)*f(k,c)*t1(c,k)
    Hdd += -1.000000000000000 * einsum('ea,fb,mj,ni,kc,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(n,j)*d(m,i)*f(k,c)*t1(c,k)
    Hdd += -1.000000000000000 * einsum('fa,eb,nj,mi,kc,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(m,j)*d(n,i)*f(k,c)*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('fa,eb,mj,ni,kc,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(m,i)*f(j,c)*t1(c,n)
    Hdd += -1.000000000000000 * einsum('ea,fb,mi,jc,cn->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(m,j)*f(i,c)*t1(c,n)
    Hdd +=  1.000000000000000 * einsum('ea,fb,mj,ic,cn->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(m,i)*f(j,c)*t1(c,n)
    Hdd +=  1.000000000000000 * einsum('fa,eb,mi,jc,cn->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(m,j)*f(i,c)*t1(c,n)
    Hdd += -1.000000000000000 * einsum('fa,eb,mj,ic,cn->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(n,i)*f(j,c)*t1(c,m)
    Hdd +=  1.000000000000000 * einsum('ea,fb,ni,jc,cm->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(n,j)*f(i,c)*t1(c,m)
    Hdd += -1.000000000000000 * einsum('ea,fb,nj,ic,cm->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(n,i)*f(j,c)*t1(c,m)
    Hdd += -1.000000000000000 * einsum('fa,eb,ni,jc,cm->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(n,j)*f(i,c)*t1(c,m)
    Hdd +=  1.000000000000000 * einsum('fa,eb,nj,ic,cm->efmnabij', kd[v, v], kd[v, v], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(n,j)*d(m,i)*f(k,a)*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fb,nj,mi,ka,ek->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,b)*d(m,j)*d(n,i)*f(k,a)*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fb,mj,ni,ka,ek->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,j)*d(m,i)*f(k,b)*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fa,nj,mi,kb,ek->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,j)*d(n,i)*f(k,b)*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fa,mj,ni,kb,ek->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,b)*d(n,j)*d(m,i)*f(k,a)*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('eb,nj,mi,ka,fk->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(m,j)*d(n,i)*f(k,a)*t1(f,k)
    Hdd += -1.000000000000000 * einsum('eb,mj,ni,ka,fk->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,j)*d(m,i)*f(k,b)*t1(f,k)
    Hdd += -1.000000000000000 * einsum('ea,nj,mi,kb,fk->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,j)*d(n,i)*f(k,b)*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('ea,mj,ni,kb,fk->efmnabij', kd[v, v], kd[o, o], kd[o, o], f[o, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(f,b)*d(n,j)*d(m,i)*<l,k||l,k>
    Hdd += -0.500000000000000 * einsum('ea,fb,nj,mi,lklk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*d(f,b)*d(m,j)*d(n,i)*<l,k||l,k>
    Hdd +=  0.500000000000000 * einsum('ea,fb,mj,ni,lklk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(e,b)*d(n,j)*d(m,i)*<l,k||l,k>
    Hdd +=  0.500000000000000 * einsum('fa,eb,nj,mi,lklk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(e,b)*d(m,j)*d(n,i)*<l,k||l,k>
    Hdd += -0.500000000000000 * einsum('fa,eb,mj,ni,lklk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, o, o], optimize=['einsum_path', (0, 1), (0, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(f,b)*<i,j||m,n>
    Hdd +=  1.000000000000000 * einsum('ea,fb,ijmn->efmnabij', kd[v, v], kd[v, v], g[o, o, o, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(f,a)*d(e,b)*<i,j||m,n>
    Hdd += -1.000000000000000 * einsum('fa,eb,ijmn->efmnabij', kd[v, v], kd[v, v], g[o, o, o, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(f,b)*d(m,i)*<j,e||a,n>
    Hdd +=  1.000000000000000 * einsum('fb,mi,jean->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(f,b)*d(m,j)*<i,e||a,n>
    Hdd += -1.000000000000000 * einsum('fb,mj,iean->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<j,e||b,n>
    Hdd += -1.000000000000000 * einsum('fa,mi,jebn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(f,a)*d(m,j)*<i,e||b,n>
    Hdd +=  1.000000000000000 * einsum('fa,mj,iebn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(f,b)*d(n,i)*<j,e||a,m>
    Hdd += -1.000000000000000 * einsum('fb,ni,jeam->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(f,b)*d(n,j)*<i,e||a,m>
    Hdd +=  1.000000000000000 * einsum('fb,nj,ieam->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(f,a)*d(n,i)*<j,e||b,m>
    Hdd +=  1.000000000000000 * einsum('fa,ni,jebm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(f,a)*d(n,j)*<i,e||b,m>
    Hdd += -1.000000000000000 * einsum('fa,nj,iebm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,b)*d(m,i)*<j,f||a,n>
    Hdd += -1.000000000000000 * einsum('eb,mi,jfan->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,b)*d(m,j)*<i,f||a,n>
    Hdd +=  1.000000000000000 * einsum('eb,mj,ifan->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,a)*d(m,i)*<j,f||b,n>
    Hdd +=  1.000000000000000 * einsum('ea,mi,jfbn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,a)*d(m,j)*<i,f||b,n>
    Hdd += -1.000000000000000 * einsum('ea,mj,ifbn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,b)*d(n,i)*<j,f||a,m>
    Hdd +=  1.000000000000000 * einsum('eb,ni,jfam->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,b)*d(n,j)*<i,f||a,m>
    Hdd += -1.000000000000000 * einsum('eb,nj,ifam->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<j,f||b,m>
    Hdd += -1.000000000000000 * einsum('ea,ni,jfbm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,a)*d(n,j)*<i,f||b,m>
    Hdd +=  1.000000000000000 * einsum('ea,nj,ifbm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, o], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(n,j)*d(m,i)*<e,f||a,b>
    Hdd +=  1.000000000000000 * einsum('nj,mi,efab->efmnabij', kd[o, o], kd[o, o], g[v, v, v, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	 -1.0000 d(m,j)*d(n,i)*<e,f||a,b>
    Hdd += -1.000000000000000 * einsum('mj,ni,efab->efmnabij', kd[o, o], kd[o, o], g[v, v, v, v], optimize=['einsum_path', (0, 1, 2)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(m,i)*<j,k||c,n>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('ea,fb,mi,jkcn,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(m,j)*<i,k||c,n>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('ea,fb,mj,ikcn,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(m,i)*<j,k||c,n>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('fa,eb,mi,jkcn,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(m,j)*<i,k||c,n>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('fa,eb,mj,ikcn,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 P(m,n)d(e,a)*d(f,b)*<i,j||c,n>*t1(c,m)
    contracted_intermediate =  1.000000000000000 * einsum('ea,fb,ijcn,cm->efmnabij', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    Hdd +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnabij->efnmabij', contracted_intermediate) 
    
    #	 -1.0000 P(m,n)d(f,a)*d(e,b)*<i,j||c,n>*t1(c,m)
    contracted_intermediate = -1.000000000000000 * einsum('fa,eb,ijcn,cm->efmnabij', kd[v, v], kd[v, v], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    Hdd +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnabij->efnmabij', contracted_intermediate) 
    
    #	 -1.0000 d(f,b)*d(m,i)*<j,k||a,n>*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fb,mi,jkan,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,b)*d(m,j)*<i,k||a,n>*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fb,mj,ikan,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,i)*<j,k||b,n>*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fa,mi,jkbn,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,j)*<i,k||b,n>*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fa,mj,ikbn,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*d(m,i)*<j,k||a,n>*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('eb,mi,jkan,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(m,j)*<i,k||a,n>*t1(f,k)
    Hdd += -1.000000000000000 * einsum('eb,mj,ikan,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,i)*<j,k||b,n>*t1(f,k)
    Hdd += -1.000000000000000 * einsum('ea,mi,jkbn,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,j)*<i,k||b,n>*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('ea,mj,ikbn,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(n,i)*<j,k||c,m>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('ea,fb,ni,jkcm,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(n,j)*<i,k||c,m>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('ea,fb,nj,ikcm,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(n,i)*<j,k||c,m>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('fa,eb,ni,jkcm,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(n,j)*<i,k||c,m>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('fa,eb,nj,ikcm,ck->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,b)*d(n,i)*<j,k||a,m>*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fb,ni,jkam,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(n,j)*<i,k||a,m>*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fb,nj,ikam,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,i)*<j,k||b,m>*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fa,ni,jkbm,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,j)*<i,k||b,m>*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fa,nj,ikbm,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(n,i)*<j,k||a,m>*t1(f,k)
    Hdd += -1.000000000000000 * einsum('eb,ni,jkam,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*d(n,j)*<i,k||a,m>*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('eb,nj,ikam,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,i)*<j,k||b,m>*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('ea,ni,jkbm,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,j)*<i,k||b,m>*t1(f,k)
    Hdd += -1.000000000000000 * einsum('ea,nj,ikbm,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, o], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,b)*d(n,j)*d(m,i)*<k,e||c,a>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('fb,nj,mi,keca,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(m,j)*d(n,i)*<k,e||c,a>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('fb,mj,ni,keca,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,j)*d(m,i)*<k,e||c,b>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('fa,nj,mi,kecb,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,j)*d(n,i)*<k,e||c,b>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('fa,mj,ni,kecb,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(m,i)*<j,e||c,a>*t1(c,n)
    Hdd += -1.000000000000000 * einsum('fb,mi,jeca,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,b)*d(m,j)*<i,e||c,a>*t1(c,n)
    Hdd +=  1.000000000000000 * einsum('fb,mj,ieca,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,i)*<j,e||c,b>*t1(c,n)
    Hdd +=  1.000000000000000 * einsum('fa,mi,jecb,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,j)*<i,e||c,b>*t1(c,n)
    Hdd += -1.000000000000000 * einsum('fa,mj,iecb,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,b)*d(n,i)*<j,e||c,a>*t1(c,m)
    Hdd +=  1.000000000000000 * einsum('fb,ni,jeca,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(n,j)*<i,e||c,a>*t1(c,m)
    Hdd += -1.000000000000000 * einsum('fb,nj,ieca,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,i)*<j,e||c,b>*t1(c,m)
    Hdd += -1.000000000000000 * einsum('fa,ni,jecb,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,j)*<i,e||c,b>*t1(c,m)
    Hdd +=  1.000000000000000 * einsum('fa,nj,iecb,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 P(e,f)d(n,j)*d(m,i)*<k,e||a,b>*t1(f,k)
    contracted_intermediate =  1.000000000000000 * einsum('nj,mi,keab,fk->efmnabij', kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    Hdd +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnabij->femnabij', contracted_intermediate) 
    
    #	 -1.0000 P(e,f)d(m,j)*d(n,i)*<k,e||a,b>*t1(f,k)
    contracted_intermediate = -1.000000000000000 * einsum('mj,ni,keab,fk->efmnabij', kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    Hdd +=  1.00000 * contracted_intermediate + -1.00000 * einsum('efmnabij->femnabij', contracted_intermediate) 
    
    #	 -1.0000 d(e,b)*d(n,j)*d(m,i)*<k,f||c,a>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('eb,nj,mi,kfca,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,b)*d(m,j)*d(n,i)*<k,f||c,a>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('eb,mj,ni,kfca,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,j)*d(m,i)*<k,f||c,b>*t1(c,k)
    Hdd +=  1.000000000000000 * einsum('ea,nj,mi,kfcb,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,j)*d(n,i)*<k,f||c,b>*t1(c,k)
    Hdd += -1.000000000000000 * einsum('ea,mj,ni,kfcb,ck->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,b)*d(m,i)*<j,f||c,a>*t1(c,n)
    Hdd +=  1.000000000000000 * einsum('eb,mi,jfca,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(m,j)*<i,f||c,a>*t1(c,n)
    Hdd += -1.000000000000000 * einsum('eb,mj,ifca,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,i)*<j,f||c,b>*t1(c,n)
    Hdd += -1.000000000000000 * einsum('ea,mi,jfcb,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,j)*<i,f||c,b>*t1(c,n)
    Hdd +=  1.000000000000000 * einsum('ea,mj,ifcb,cn->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(n,i)*<j,f||c,a>*t1(c,m)
    Hdd += -1.000000000000000 * einsum('eb,ni,jfca,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*d(n,j)*<i,f||c,a>*t1(c,m)
    Hdd +=  1.000000000000000 * einsum('eb,nj,ifca,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,i)*<j,f||c,b>*t1(c,m)
    Hdd +=  1.000000000000000 * einsum('ea,ni,jfcb,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,j)*<i,f||c,b>*t1(c,m)
    Hdd += -1.000000000000000 * einsum('ea,nj,ifcb,cm->efmnabij', kd[v, v], kd[o, o], g[o, v, v, v], t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  0.2500 d(e,a)*d(f,b)*d(n,j)*d(m,i)*<l,k||c,d>*t2(c,d,l,k)
    Hdd +=  0.250000000000000 * einsum('ea,fb,nj,mi,lkcd,cdlk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 d(e,a)*d(f,b)*d(m,j)*d(n,i)*<l,k||c,d>*t2(c,d,l,k)
    Hdd += -0.250000000000000 * einsum('ea,fb,mj,ni,lkcd,cdlk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.2500 d(f,a)*d(e,b)*d(n,j)*d(m,i)*<l,k||c,d>*t2(c,d,l,k)
    Hdd += -0.250000000000000 * einsum('fa,eb,nj,mi,lkcd,cdlk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	  0.2500 d(f,a)*d(e,b)*d(m,j)*d(n,i)*<l,k||c,d>*t2(c,d,l,k)
    Hdd +=  0.250000000000000 * einsum('fa,eb,mj,ni,lkcd,cdlk->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (2, 3), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(f,b)*d(m,i)*<j,k||c,d>*t2(c,d,n,k)
    Hdd += -0.500000000000000 * einsum('ea,fb,mi,jkcd,cdnk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*d(f,b)*d(m,j)*<i,k||c,d>*t2(c,d,n,k)
    Hdd +=  0.500000000000000 * einsum('ea,fb,mj,ikcd,cdnk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(e,b)*d(m,i)*<j,k||c,d>*t2(c,d,n,k)
    Hdd +=  0.500000000000000 * einsum('fa,eb,mi,jkcd,cdnk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(e,b)*d(m,j)*<i,k||c,d>*t2(c,d,n,k)
    Hdd += -0.500000000000000 * einsum('fa,eb,mj,ikcd,cdnk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*d(f,b)*d(n,i)*<j,k||c,d>*t2(c,d,m,k)
    Hdd +=  0.500000000000000 * einsum('ea,fb,ni,jkcd,cdmk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(f,b)*d(n,j)*<i,k||c,d>*t2(c,d,m,k)
    Hdd += -0.500000000000000 * einsum('ea,fb,nj,ikcd,cdmk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(e,b)*d(n,i)*<j,k||c,d>*t2(c,d,m,k)
    Hdd += -0.500000000000000 * einsum('fa,eb,ni,jkcd,cdmk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(e,b)*d(n,j)*<i,k||c,d>*t2(c,d,m,k)
    Hdd +=  0.500000000000000 * einsum('fa,eb,nj,ikcd,cdmk->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*d(f,b)*<i,j||c,d>*t2(c,d,m,n)
    Hdd +=  0.500000000000000 * einsum('ea,fb,ijcd,cdmn->efmnabij', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(e,b)*<i,j||c,d>*t2(c,d,m,n)
    Hdd += -0.500000000000000 * einsum('fa,eb,ijcd,cdmn->efmnabij', kd[v, v], kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(f,b)*d(n,j)*d(m,i)*<l,k||c,a>*t2(c,e,l,k)
    Hdd += -0.500000000000000 * einsum('fb,nj,mi,lkca,celk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,b)*d(m,j)*d(n,i)*<l,k||c,a>*t2(c,e,l,k)
    Hdd +=  0.500000000000000 * einsum('fb,mj,ni,lkca,celk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(n,j)*d(m,i)*<l,k||c,b>*t2(c,e,l,k)
    Hdd +=  0.500000000000000 * einsum('fa,nj,mi,lkcb,celk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(m,j)*d(n,i)*<l,k||c,b>*t2(c,e,l,k)
    Hdd += -0.500000000000000 * einsum('fa,mj,ni,lkcb,celk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,b)*d(m,i)*<j,k||c,a>*t2(c,e,n,k)
    Hdd +=  1.000000000000000 * einsum('fb,mi,jkca,cenk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(m,j)*<i,k||c,a>*t2(c,e,n,k)
    Hdd += -1.000000000000000 * einsum('fb,mj,ikca,cenk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<j,k||c,b>*t2(c,e,n,k)
    Hdd += -1.000000000000000 * einsum('fa,mi,jkcb,cenk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,j)*<i,k||c,b>*t2(c,e,n,k)
    Hdd +=  1.000000000000000 * einsum('fa,mj,ikcb,cenk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(n,i)*<j,k||c,a>*t2(c,e,m,k)
    Hdd += -1.000000000000000 * einsum('fb,ni,jkca,cemk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,b)*d(n,j)*<i,k||c,a>*t2(c,e,m,k)
    Hdd +=  1.000000000000000 * einsum('fb,nj,ikca,cemk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*<j,k||c,b>*t2(c,e,m,k)
    Hdd +=  1.000000000000000 * einsum('fa,ni,jkcb,cemk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,j)*<i,k||c,b>*t2(c,e,m,k)
    Hdd += -1.000000000000000 * einsum('fa,nj,ikcb,cemk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(f,b)*<i,j||c,a>*t2(c,e,m,n)
    Hdd += -1.000000000000000 * einsum('fb,ijca,cemn->efmnabij', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*<i,j||c,b>*t2(c,e,m,n)
    Hdd +=  1.000000000000000 * einsum('fa,ijcb,cemn->efmnabij', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(e,b)*d(n,j)*d(m,i)*<l,k||c,a>*t2(c,f,l,k)
    Hdd +=  0.500000000000000 * einsum('eb,nj,mi,lkca,cflk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,b)*d(m,j)*d(n,i)*<l,k||c,a>*t2(c,f,l,k)
    Hdd += -0.500000000000000 * einsum('eb,mj,ni,lkca,cflk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(n,j)*d(m,i)*<l,k||c,b>*t2(c,f,l,k)
    Hdd += -0.500000000000000 * einsum('ea,nj,mi,lkcb,cflk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*d(m,j)*d(n,i)*<l,k||c,b>*t2(c,f,l,k)
    Hdd +=  0.500000000000000 * einsum('ea,mj,ni,lkcb,cflk->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (1, 2), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(m,i)*<j,k||c,a>*t2(c,f,n,k)
    Hdd += -1.000000000000000 * einsum('eb,mi,jkca,cfnk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*d(m,j)*<i,k||c,a>*t2(c,f,n,k)
    Hdd +=  1.000000000000000 * einsum('eb,mj,ikca,cfnk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,i)*<j,k||c,b>*t2(c,f,n,k)
    Hdd +=  1.000000000000000 * einsum('ea,mi,jkcb,cfnk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,j)*<i,k||c,b>*t2(c,f,n,k)
    Hdd += -1.000000000000000 * einsum('ea,mj,ikcb,cfnk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*d(n,i)*<j,k||c,a>*t2(c,f,m,k)
    Hdd +=  1.000000000000000 * einsum('eb,ni,jkca,cfmk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(n,j)*<i,k||c,a>*t2(c,f,m,k)
    Hdd += -1.000000000000000 * einsum('eb,nj,ikca,cfmk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<j,k||c,b>*t2(c,f,m,k)
    Hdd += -1.000000000000000 * einsum('ea,ni,jkcb,cfmk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,j)*<i,k||c,b>*t2(c,f,m,k)
    Hdd +=  1.000000000000000 * einsum('ea,nj,ikcb,cfmk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	  1.0000 d(e,b)*<i,j||c,a>*t2(c,f,m,n)
    Hdd +=  1.000000000000000 * einsum('eb,ijca,cfmn->efmnabij', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*<i,j||c,b>*t2(c,f,m,n)
    Hdd += -1.000000000000000 * einsum('ea,ijcb,cfmn->efmnabij', kd[v, v], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  0.5000 d(n,j)*d(m,i)*<l,k||a,b>*t2(e,f,l,k)
    Hdd +=  0.500000000000000 * einsum('nj,mi,lkab,eflk->efmnabij', kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -0.5000 d(m,j)*d(n,i)*<l,k||a,b>*t2(e,f,l,k)
    Hdd += -0.500000000000000 * einsum('mj,ni,lkab,eflk->efmnabij', kd[o, o], kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (0, 1), (0, 1), (0, 1)])
    
    #	 -1.0000 d(m,i)*<j,k||a,b>*t2(e,f,n,k)
    Hdd += -1.000000000000000 * einsum('mi,jkab,efnk->efmnabij', kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(m,j)*<i,k||a,b>*t2(e,f,n,k)
    Hdd +=  1.000000000000000 * einsum('mj,ikab,efnk->efmnabij', kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	  1.0000 d(n,i)*<j,k||a,b>*t2(e,f,m,k)
    Hdd +=  1.000000000000000 * einsum('ni,jkab,efmk->efmnabij', kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -1.0000 d(n,j)*<i,k||a,b>*t2(e,f,m,k)
    Hdd += -1.000000000000000 * einsum('nj,ikab,efmk->efmnabij', kd[o, o], g[o, o, v, v], t2, optimize=['einsum_path', (1, 2), (0, 1)])
    
    #	 -0.5000 d(e,a)*d(f,b)*d(n,j)*d(m,i)*<l,k||c,d>*t1(c,k)*t1(d,l)
    Hdd += -0.500000000000000 * einsum('ea,fb,nj,mi,lkcd,ck,dl->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(e,a)*d(f,b)*d(m,j)*d(n,i)*<l,k||c,d>*t1(c,k)*t1(d,l)
    Hdd +=  0.500000000000000 * einsum('ea,fb,mj,ni,lkcd,ck,dl->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  0.5000 d(f,a)*d(e,b)*d(n,j)*d(m,i)*<l,k||c,d>*t1(c,k)*t1(d,l)
    Hdd +=  0.500000000000000 * einsum('fa,eb,nj,mi,lkcd,ck,dl->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	 -0.5000 d(f,a)*d(e,b)*d(m,j)*d(n,i)*<l,k||c,d>*t1(c,k)*t1(d,l)
    Hdd += -0.500000000000000 * einsum('fa,eb,mj,ni,lkcd,ck,dl->efmnabij', kd[v, v], kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (2, 3), (2, 4), (0, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(m,i)*<j,k||c,d>*t1(c,k)*t1(d,n)
    Hdd +=  1.000000000000000 * einsum('ea,fb,mi,jkcd,ck,dn->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(m,j)*<i,k||c,d>*t1(c,k)*t1(d,n)
    Hdd += -1.000000000000000 * einsum('ea,fb,mj,ikcd,ck,dn->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(m,i)*<j,k||c,d>*t1(c,k)*t1(d,n)
    Hdd += -1.000000000000000 * einsum('fa,eb,mi,jkcd,ck,dn->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(m,j)*<i,k||c,d>*t1(c,k)*t1(d,n)
    Hdd +=  1.000000000000000 * einsum('fa,eb,mj,ikcd,ck,dn->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*d(n,i)*<j,k||c,d>*t1(c,k)*t1(d,m)
    Hdd += -1.000000000000000 * einsum('ea,fb,ni,jkcd,ck,dm->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(f,b)*d(n,j)*<i,k||c,d>*t1(c,k)*t1(d,m)
    Hdd +=  1.000000000000000 * einsum('ea,fb,nj,ikcd,ck,dm->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*d(n,i)*<j,k||c,d>*t1(c,k)*t1(d,m)
    Hdd +=  1.000000000000000 * einsum('fa,eb,ni,jkcd,ck,dm->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(e,b)*d(n,j)*<i,k||c,d>*t1(c,k)*t1(d,m)
    Hdd += -1.000000000000000 * einsum('fa,eb,nj,ikcd,ck,dm->efmnabij', kd[v, v], kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,b)*d(n,j)*d(m,i)*<l,k||c,a>*t1(c,k)*t1(e,l)
    Hdd +=  1.000000000000000 * einsum('fb,nj,mi,lkca,ck,el->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(m,j)*d(n,i)*<l,k||c,a>*t1(c,k)*t1(e,l)
    Hdd += -1.000000000000000 * einsum('fb,mj,ni,lkca,ck,el->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,j)*d(m,i)*<l,k||c,b>*t1(c,k)*t1(e,l)
    Hdd += -1.000000000000000 * einsum('fa,nj,mi,lkcb,ck,el->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,j)*d(n,i)*<l,k||c,b>*t1(c,k)*t1(e,l)
    Hdd +=  1.000000000000000 * einsum('fa,mj,ni,lkcb,ck,el->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(n,j)*d(m,i)*<l,k||c,a>*t1(c,k)*t1(f,l)
    Hdd += -1.000000000000000 * einsum('eb,nj,mi,lkca,ck,fl->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,b)*d(m,j)*d(n,i)*<l,k||c,a>*t1(c,k)*t1(f,l)
    Hdd +=  1.000000000000000 * einsum('eb,mj,ni,lkca,ck,fl->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,j)*d(m,i)*<l,k||c,b>*t1(c,k)*t1(f,l)
    Hdd +=  1.000000000000000 * einsum('ea,nj,mi,lkcb,ck,fl->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,j)*d(n,i)*<l,k||c,b>*t1(c,k)*t1(f,l)
    Hdd += -1.000000000000000 * einsum('ea,mj,ni,lkcb,ck,fl->efmnabij', kd[v, v], kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (1, 2), (1, 3), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(f,b)*<i,j||c,d>*t1(c,n)*t1(d,m)
    Hdd += -1.000000000000000 * einsum('ea,fb,ijcd,cn,dm->efmnabij', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(e,b)*<i,j||c,d>*t1(c,n)*t1(d,m)
    Hdd +=  1.000000000000000 * einsum('fa,eb,ijcd,cn,dm->efmnabij', kd[v, v], kd[v, v], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,b)*d(m,i)*<j,k||c,a>*t1(c,n)*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fb,mi,jkca,cn,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(m,j)*<i,k||c,a>*t1(c,n)*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fb,mj,ikca,cn,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(m,i)*<j,k||c,b>*t1(c,n)*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fa,mi,jkcb,cn,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(m,j)*<i,k||c,b>*t1(c,n)*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fa,mj,ikcb,cn,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(m,i)*<j,k||c,a>*t1(c,n)*t1(f,k)
    Hdd += -1.000000000000000 * einsum('eb,mi,jkca,cn,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,b)*d(m,j)*<i,k||c,a>*t1(c,n)*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('eb,mj,ikca,cn,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(m,i)*<j,k||c,b>*t1(c,n)*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('ea,mi,jkcb,cn,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(m,j)*<i,k||c,b>*t1(c,n)*t1(f,k)
    Hdd += -1.000000000000000 * einsum('ea,mj,ikcb,cn,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,b)*d(n,i)*<j,k||c,a>*t1(c,m)*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fb,ni,jkca,cm,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,b)*d(n,j)*<i,k||c,a>*t1(c,m)*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fb,nj,ikca,cm,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(f,a)*d(n,i)*<j,k||c,b>*t1(c,m)*t1(e,k)
    Hdd +=  1.000000000000000 * einsum('fa,ni,jkcb,cm,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(f,a)*d(n,j)*<i,k||c,b>*t1(c,m)*t1(e,k)
    Hdd += -1.000000000000000 * einsum('fa,nj,ikcb,cm,ek->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,b)*d(n,i)*<j,k||c,a>*t1(c,m)*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('eb,ni,jkca,cm,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,b)*d(n,j)*<i,k||c,a>*t1(c,m)*t1(f,k)
    Hdd += -1.000000000000000 * einsum('eb,nj,ikca,cm,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(e,a)*d(n,i)*<j,k||c,b>*t1(c,m)*t1(f,k)
    Hdd += -1.000000000000000 * einsum('ea,ni,jkcb,cm,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(e,a)*d(n,j)*<i,k||c,b>*t1(c,m)*t1(f,k)
    Hdd +=  1.000000000000000 * einsum('ea,nj,ikcb,cm,fk->efmnabij', kd[v, v], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	 -1.0000 d(n,j)*d(m,i)*<l,k||a,b>*t1(e,k)*t1(f,l)
    Hdd += -1.000000000000000 * einsum('nj,mi,lkab,ek,fl->efmnabij', kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])
    
    #	  1.0000 d(m,j)*d(n,i)*<l,k||a,b>*t1(e,k)*t1(f,l)
    Hdd +=  1.000000000000000 * einsum('mj,ni,lkab,ek,fl->efmnabij', kd[o, o], kd[o, o], g[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1), (0, 2), (0, 1)])


    return H00, Hs0, H0s, Hd0, H0d, Hss, Hsd, Hds, Hdd
    
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
    mol = pyscf.M(
        atom='B 0 0 0; H 0 0 {}'.format(1.6),
        basis=basis)

    mf = mol.RHF().run()
    mycc = mf.CCSD().run()
    print('CCSD correlation energy', mycc.e_corr)

    molecule = of.MolecularData(geometry=[['B', (0, 0, 0)], ['H', (0, 0, 1.6)]],
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

    H00, Hs0, H0s, Hd0, H0d, Hss, Hsd, Hds, Hdd = build_Hbar(kd,fock, g, o, v, t1f, t2f)

    dim = int(1 + nsvirt*(nsvirt-1)/2*nsocc*(nsocc-1)/2 + nsvirt*nsocc)
    H = np.zeros((dim,dim))

    # 00 block
    H[0,0] = H00

    # 0s, s0 blocks
    for a in range (0,nsvirt):
        for i in range (0,nsocc):
            ai = 1 + a*nsocc + i
            H[ai,0] = Hs0[a,i]
            H[0,ai] = H0s[a,i]

    # ss block
    for a in range (0,nsvirt):
        for i in range (0,nsocc):
            ai = 1 + a*nsocc + i
            for e in range (0,nsvirt):
                for m in range (0,nsocc):
                    em = 1 + e*nsocc + m
                    H[ai,em] = Hss[a,i,e,m]

    # sd, ds blocks
    for a in range (0,nsvirt):
        for i in range (0,nsocc):
            ai = 1 + a*nsocc + i
            efmn = 1 + nsocc*nsvirt
            for e in range (0,nsvirt):
                for f in range (e+1,nsvirt):
                    for m in range (0,nsocc):
                        for n in range (m+1,nsocc):
                            H[ai,efmn] = Hsd[a,i,e,f,m,n]
                            H[efmn,ai] = Hds[e,f,m,n,a,i]
                            efmn += 1

    # 0d, d0 blocks
    abij = 1 + nsocc*nsvirt
    for a in range (0,nsvirt):
        for b in range (a+1,nsvirt):
            for i in range (0,nsocc):
                for j in range (i+1,nsocc):
                    H[abij,0] = Hd0[a,b,i,j]
                    H[0,abij] = H0d[a,b,i,j]
                    abij += 1

    # dd blocks
    abij = 1 + nsocc*nsvirt
    for a in range (0,nsvirt):
        for b in range (a+1,nsvirt):
            for i in range (0,nsocc):
                for j in range (i+1,nsocc):
                    efmn = 1 + nsocc*nsvirt
                    for e in range (0,nsvirt):
                        for f in range (e+1,nsvirt):
                            for m in range (0,nsocc):
                                for n in range (m+1,nsocc):
                                    H[abij,efmn] = Hdd[a,b,i,j,e,f,m,n]
                                    efmn += 1
                    abij += 1

    cc_energy = ccsd_energy(t1f, t2f, fock, g, o, v)
    print('    ccsd energy: %20.12f' % (cc_energy + molecule.nuclear_repulsion) )

    print('')
    print('    eigenvalues of e(-T) H e(T):')
    print('')

    print('    %20s %20s' % ('total energy','excitation energy'))
    en, vec = np.linalg.eig(H)
    for i in range (0,len(en)):
        print('    %20.12f %20.12f' % ( en[i] + molecule.nuclear_repulsion,en[i]-cc_energy))

    print('')

if __name__ == "__main__":
    main()

