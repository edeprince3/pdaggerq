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
    # HF energy
    #	  1.0000 f(i,i)
    energy = 1.0 * einsum('ii', f[o, o])

    #	 -0.5000 <i,j||i,j>
    energy += -0.5 * einsum('ijij', g[o, o, o, o])


    # F + t1 contraction
    #	  1.0000 f(i,a)*t1(a,i)
    energy += 1.0 * einsum('ia,ai', f[o, v], t1)

    # v + t2 contraction
    #	  0.5000 <i,j||a,b>*t1(a,i)*t1(b,j)
    energy += 0.5 * einsum('ijab,ai,bj', g[o, o, v, v], t1, t1,
                           optimize=['einsum_path', (0, 1), (0, 1)])

    # ?  should be +0.25
    #	 -0.2500 <i,j||a,b>*t2(a,b,j,i)
    energy += -0.25 * einsum('ijab,abji', g[o, o, v, v], t2)


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

    #	  1.0000 f(i,a)*t2(a,e,i,m)
    singles_res += 1.0 * einsum('ia,aeim->em', f[o, v], t2)

    #	 -1.0000 f(i,a)*t1(a,m)*t1(e,i)
    singles_res += -1.0 * einsum('ia,am,ei->em', f[o, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,m>*t1(a,i)
    singles_res += 1.0 * einsum('ieam,ai->em', g[o, v, v, o], t1)

    #	  0.5000 <i,j||a,m>*t2(a,e,j,i)
    singles_res += 0.5 * einsum('ijam,aeji->em', g[o, o, v, o], t2)

    #	  0.5000 <i,e||a,b>*t2(a,b,i,m)
    singles_res += 0.5 * einsum('ieab,abim->em', g[o, v, v, v], t2)

    #	 -1.0000 <i,j||a,m>*t1(a,i)*t1(e,j)
    singles_res += -1.0 * einsum('ijam,ai,ej->em', g[o, o, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,b>*t1(a,i)*t1(b,m)
    singles_res += 1.0 * einsum('ieab,ai,bm->em', g[o, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,i)*t2(b,e,j,m)
    singles_res += 1.0 * einsum('ijab,ai,bejm->em', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,j||a,b>*t1(a,m)*t2(b,e,j,i)
    singles_res += -0.5 * einsum('ijab,am,beji->em', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <i,j||a,b>*t1(e,i)*t2(a,b,j,m)
    singles_res += 0.5 * einsum('ijab,ei,abjm->em', g[o, o, v, v], t1, t2, optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,i)*t1(b,m)*t1(e,j)
    singles_res += -1.0 * einsum('ijab,ai,bm,ej->em', g[o, o, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])
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
    #	  1.0000 f(i,n)*t2(e,f,i,m)
    doubles_res = 1.0 * einsum('in,efim->efmn', f[o, o], t2)

    #	 -1.0000 f(i,m)*t2(e,f,i,n)
    doubles_res += -1.0 * einsum('im,efin->efmn', f[o, o], t2)

    #	  1.0000 f(e,a)*t2(a,f,m,n)
    doubles_res += 1.0 * einsum('ea,afmn->efmn', f[v, v], t2)

    #	 -1.0000 f(f,a)*t2(a,e,m,n)
    doubles_res += -1.0 * einsum('fa,aemn->efmn', f[v, v], t2)

    #	 -1.0000 f(i,a)*t1(a,n)*t2(e,f,m,i)
    doubles_res += -1.0 * einsum('ia,an,efmi->efmn', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 f(i,a)*t1(a,m)*t2(e,f,n,i)
    doubles_res += 1.0 * einsum('ia,am,efni->efmn', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 f(i,a)*t1(e,i)*t2(a,f,m,n)
    doubles_res += -1.0 * einsum('ia,ei,afmn->efmn', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 f(i,a)*t1(f,i)*t2(a,e,m,n)
    doubles_res += 1.0 * einsum('ia,fi,aemn->efmn', f[o, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <e,f||m,n>
    doubles_res += 1.0 * einsum('efmn->efmn', g[v, v, o, o])

    #	  1.0000 <i,e||m,n>*t1(f,i)
    doubles_res += 1.0 * einsum('iemn,fi->efmn', g[o, v, o, o], t1)

    #	 -1.0000 <i,f||m,n>*t1(e,i)
    doubles_res += -1.0 * einsum('ifmn,ei->efmn', g[o, v, o, o], t1)

    #	  1.0000 <e,f||a,n>*t1(a,m)
    doubles_res += 1.0 * einsum('efan,am->efmn', g[v, v, v, o], t1)

    #	 -1.0000 <e,f||a,m>*t1(a,n)
    doubles_res += -1.0 * einsum('efam,an->efmn', g[v, v, v, o], t1)

    #	 -0.5000 <i,j||m,n>*t2(e,f,j,i)
    doubles_res += -0.5 * einsum('ijmn,efji->efmn', g[o, o, o, o], t2)

    #	 -1.0000 <i,e||a,n>*t2(a,f,i,m)
    doubles_res += -1.0 * einsum('iean,afim->efmn', g[o, v, v, o], t2)

    #	  1.0000 <i,e||a,m>*t2(a,f,i,n)
    doubles_res += 1.0 * einsum('ieam,afin->efmn', g[o, v, v, o], t2)

    #	  1.0000 <i,f||a,n>*t2(a,e,i,m)
    doubles_res += 1.0 * einsum('ifan,aeim->efmn', g[o, v, v, o], t2)

    #	 -1.0000 <i,f||a,m>*t2(a,e,i,n)
    doubles_res += -1.0 * einsum('ifam,aein->efmn', g[o, v, v, o], t2)

    #	  0.5000 <e,f||a,b>*t2(a,b,m,n)
    doubles_res += 0.5 * einsum('efab,abmn->efmn', g[v, v, v, v], t2)

    #	  1.0000 <i,j||m,n>*t1(e,i)*t1(f,j)
    doubles_res += 1.0 * einsum('ijmn,ei,fj->efmn', g[o, o, o, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,n>*t1(a,m)*t1(f,i)
    doubles_res += 1.0 * einsum('iean,am,fi->efmn', g[o, v, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,e||a,m>*t1(a,n)*t1(f,i)
    doubles_res += -1.0 * einsum('ieam,an,fi->efmn', g[o, v, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,f||a,n>*t1(a,m)*t1(e,i)
    doubles_res += -1.0 * einsum('ifan,am,ei->efmn', g[o, v, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,f||a,m>*t1(a,n)*t1(e,i)
    doubles_res += 1.0 * einsum('ifam,an,ei->efmn', g[o, v, v, o], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <e,f||a,b>*t1(a,n)*t1(b,m)
    doubles_res += -1.0 * einsum('efab,an,bm->efmn', g[v, v, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,n>*t1(a,i)*t2(e,f,j,m)
    doubles_res += 1.0 * einsum('ijan,ai,efjm->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,j||a,n>*t1(a,m)*t2(e,f,j,i)
    doubles_res += -0.5 * einsum('ijan,am,efji->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,n>*t1(e,i)*t2(a,f,j,m)
    doubles_res += -1.0 * einsum('ijan,ei,afjm->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,n>*t1(f,i)*t2(a,e,j,m)
    doubles_res += 1.0 * einsum('ijan,fi,aejm->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,m>*t1(a,i)*t2(e,f,j,n)
    doubles_res += -1.0 * einsum('ijam,ai,efjn->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,j||a,m>*t1(a,n)*t2(e,f,j,i)
    doubles_res += 0.5 * einsum('ijam,an,efji->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,m>*t1(e,i)*t2(a,f,j,n)
    doubles_res += 1.0 * einsum('ijam,ei,afjn->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,m>*t1(f,i)*t2(a,e,j,n)
    doubles_res += -1.0 * einsum('ijam,fi,aejn->efmn', g[o, o, v, o], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,b>*t1(a,i)*t2(b,f,m,n)
    doubles_res += 1.0 * einsum('ieab,ai,bfmn->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,e||a,b>*t1(a,n)*t2(b,f,m,i)
    doubles_res += -1.0 * einsum('ieab,an,bfmi->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,b>*t1(a,m)*t2(b,f,n,i)
    doubles_res += 1.0 * einsum('ieab,am,bfni->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,e||a,b>*t1(f,i)*t2(a,b,m,n)
    doubles_res += 0.5 * einsum('ieab,fi,abmn->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,f||a,b>*t1(a,i)*t2(b,e,m,n)
    doubles_res += -1.0 * einsum('ifab,ai,bemn->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,f||a,b>*t1(a,n)*t2(b,e,m,i)
    doubles_res += 1.0 * einsum('ifab,an,bemi->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,f||a,b>*t1(a,m)*t2(b,e,n,i)
    doubles_res += -1.0 * einsum('ifab,am,beni->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,f||a,b>*t1(e,i)*t2(a,b,m,n)
    doubles_res += -0.5 * einsum('ifab,ei,abmn->efmn', g[o, v, v, v], t1, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,j||a,b>*t2(a,b,n,i)*t2(e,f,m,j)
    doubles_res += 0.5 * einsum('ijab,abni,efmj->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,j||a,b>*t2(a,b,m,i)*t2(e,f,n,j)
    doubles_res += -0.5 * einsum('ijab,abmi,efnj->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.2500 <i,j||a,b>*t2(a,b,m,n)*t2(e,f,j,i)
    doubles_res += -0.25 * einsum('ijab,abmn,efji->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,j||a,b>*t2(a,e,j,i)*t2(b,f,m,n)
    doubles_res += 0.5 * einsum('ijab,aeji,bfmn->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,b>*t2(a,e,n,i)*t2(b,f,m,j)
    doubles_res += -1.0 * einsum('ijab,aeni,bfmj->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,b>*t2(a,e,m,i)*t2(b,f,n,j)
    doubles_res += 1.0 * einsum('ijab,aemi,bfnj->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,j||a,b>*t2(a,e,m,n)*t2(b,f,j,i)
    doubles_res += 0.5 * einsum('ijab,aemn,bfji->efmn', g[o, o, v, v], t2, t2, optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <i,j||a,n>*t1(a,m)*t1(e,i)*t1(f,j)
    doubles_res += 1.0 * einsum('ijan,am,ei,fj->efmn', g[o, o, v, o], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <i,j||a,m>*t1(a,n)*t1(e,i)*t1(f,j)
    doubles_res += -1.0 * einsum('ijam,an,ei,fj->efmn', g[o, o, v, o], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <i,e||a,b>*t1(a,n)*t1(b,m)*t1(f,i)
    doubles_res += -1.0 * einsum('ieab,an,bm,fi->efmn', g[o, v, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <i,f||a,b>*t1(a,n)*t1(b,m)*t1(e,i)
    doubles_res += 1.0 * einsum('ifab,an,bm,ei->efmn', g[o, v, v, v], t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,i)*t1(b,n)*t2(e,f,m,j)
    doubles_res += -1.0 * einsum('ijab,ai,bn,efmj->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,i)*t1(b,m)*t2(e,f,n,j)
    doubles_res += 1.0 * einsum('ijab,ai,bm,efnj->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,i)*t1(e,j)*t2(b,f,m,n)
    doubles_res += -1.0 * einsum('ijab,ai,ej,bfmn->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,i)*t1(f,j)*t2(b,e,m,n)
    doubles_res += 1.0 * einsum('ijab,ai,fj,bemn->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  0.5000 <i,j||a,b>*t1(a,n)*t1(b,m)*t2(e,f,j,i)
    doubles_res += 0.5 * einsum('ijab,an,bm,efji->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,n)*t1(e,i)*t2(b,f,m,j)
    doubles_res += -1.0 * einsum('ijab,an,ei,bfmj->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,n)*t1(f,i)*t2(b,e,m,j)
    doubles_res += 1.0 * einsum('ijab,an,fi,bemj->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,m)*t1(e,i)*t2(b,f,n,j)
    doubles_res += 1.0 * einsum('ijab,am,ei,bfnj->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,m)*t1(f,i)*t2(b,e,n,j)
    doubles_res += -1.0 * einsum('ijab,am,fi,benj->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	  0.5000 <i,j||a,b>*t1(e,i)*t1(f,j)*t2(a,b,m,n)
    doubles_res += 0.5 * einsum('ijab,ei,fj,abmn->efmn', g[o, o, v, v], t1, t1, t2, optimize=['einsum_path', (0, 1), (0, 2), (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,n)*t1(b,m)*t1(e,i)*t1(f,j)
    doubles_res += -1.0 * einsum('ijab,an,bm,ei,fj->efmn', g[o, o, v, v], t1, t1, t1, t1, optimize=['einsum_path', (0, 1), (0, 3), (0, 2), (0, 1)])

    return doubles_res


def kernel(t1, t2, fock, g, o, v, e_ai, e_abij, max_iter=100, stopping_eps=1.0E-8):

    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)
    old_energy = ccsd_energy(t1, t2, fock, g, o, v)
    for idx in range(max_iter):

        singles_res = singles_residual(t1, t2, fock, g, o, v) + fock_e_ai * t1
        doubles_res = doubles_residual(t1, t2, fock, g, o, v) + fock_e_abij * t2

        new_singles = singles_res * e_ai
        new_doubles = doubles_res * e_abij

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

    t1s = spatial2spin(mycc.t1)
    t2s = spatial2spin(mycc.t2)
    t1 = t1s.transpose(1, 0)
    t2 = t2s.transpose(2, 3, 0, 1)

    cc_energy = np.einsum('ia,ia', fock[o, v], t1s) + 0.25 * np.einsum('ijba,ijab', astei[o, o, v, v], t2s) + 0.5 * np.einsum('ijba,ia,jb', astei[o, o, v, v], t1s, t1s)

    cc_energy2 = 1.0 * einsum('ii', fock[o, o]) + -0.5 * einsum('ijij', gtei[o, o, o, o]) \
               + 1.0 * einsum('ia,ai', fock[o, v], t1) \
               +  0.5 * einsum('ijab,ai,bj', gtei[o, o, v, v], t1, t1, optimize=['einsum_path', (0, 1), (0, 1)]) \
               + -0.25 * einsum('ijab,abji', gtei[o, o, v, v], t2)

    print("{: 5.10f}\tTest CC energy ".format(cc_energy))
    print("{: 5.10f}\tTest CC energy22 ".format(cc_energy2 - hf_energy))
    print("{: 5.10f}\tTest CC energy pdaggerq".format(ccsd_energy(t1, t2, fock, gtei, o, v) - hf_energy))
    print("{: 5.10f}\tpyscf CC energy".format(molecule.ccsd_energy - molecule.hf_energy))
    print("{: 5.10f}\t{: 5.10f}  CCSD pyscf, CCSD NCR".format(molecule.ccsd_energy, hf_energy + cc_energy + molecule.nuclear_repulsion))

    print("Check total energy ", ccsd_energy(t1s.transpose(1, 0), t2s.transpose(2, 3, 0, 1), fock, gtei, o, v) + molecule.nuclear_repulsion)
    print("electronic energy", ccsd_energy(t1s.transpose(1, 0), t2s.transpose(2, 3, 0, 1), fock, gtei, o, v) )


    g = gtei
    t1 = t1s.transpose(1, 0)
    t2 = t2s.transpose(2, 3, 0, 1)

    assert np.allclose(singles_residual(t1, t2, fock, g, o, v), 0)
    assert np.allclose(doubles_residual(t1, t2, fock, g, o, v), 0, atol=1.0E-6)

    t1f, t2f = kernel(np.zeros_like(t1), np.zeros_like(t2), fock, g, o, v, e_ai, e_abij)
    print(ccsd_energy(t1f, t2f, fock, g, o, v) - hf_energy)


if __name__ == "__main__":
    main()