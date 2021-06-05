"""
A full working spin-orbital lambda-CCSD code generated with pdaggerq

If you want to run the example here you should install pyscf openfermion and openfermion-pyscf
The actual CCSD code (ccsd_energy, singles_residual, doubles_residual, kernel)
do not depend on those packages but you gotta get integrals frome somehwere.

We also check the code by using pyscfs functionality for generating spin-orbital
t-amplitudes from RCCSD.  the main() function is fairly straightforward.
"""
import numpy as np
from numpy import einsum


def lagrangian_energy(t1, t2, l1, l2, f, g, o, v):
    l_energy = ccsd_energy(t1, t2, f, g, o, v)
    l_energy += np.einsum('me,em', l1, singles_residual(t1, t2, f, g, o, v))
    l_energy += np.einsum('mnef,efmn', l2, doubles_residual(t1, t2, f, g, o, v))
    return l_energy


def lambda_singles(t1, t2, l1, l2, f, g, o, v):
    #	  1.0000 f(m,e)
    lambda_one = 1.0 * einsum('me->em', f[o, v])
    #	 -1.0000 <i,m||e,a>*t1(a,i)
    lambda_one += -1.0 * einsum('imea,ai->em', g[o, o, v, v], t1)
    #	 -1.0000 f(m,i)*l1(i,e)
    lambda_one += -1.0 * einsum('mi,ie->em', f[o, o], l1)
    #	  1.0000 f(a,e)*l1(m,a)
    lambda_one += 1.0 * einsum('ae,ma->em', f[v, v], l1)
    #	 -1.0000 f(i,e)*l1(m,a)*t1(a,i)
    lambda_one += -1.0 * einsum('ie,ma,ai->em', f[o, v], l1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])
    #	 -1.0000 f(m,a)*l1(i,e)*t1(a,i)
    lambda_one += -1.0 * einsum('ma,ie,ai->em', f[o, v], l1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])
    #	 -0.5000 f(j,e)*l2(i,m,b,a)*t2(b,a,i,j)
    lambda_one += -0.5 * einsum('je,imba,baij->em', f[o, v], l2, t2,
                                optimize=['einsum_path', (1, 2), (0, 1)])
    #	 -0.5000 f(m,b)*l2(i,j,e,a)*t2(b,a,i,j)
    lambda_one += -0.5 * einsum('mb,ijea,baij->em', f[o, v], l2, t2,
                                optimize=['einsum_path', (1, 2), (0, 1)])
    #	  1.0000 <m,a||e,i>*l1(i,a)
    lambda_one += 1.0 * einsum('maei,ia->em', g[o, v, v, o], l1)
    #	  0.5000 <m,a||i,j>*l2(i,j,a,e)
    lambda_one += 0.5 * einsum('maij,ijae->em', g[o, v, o, o], l2)
    #	 -0.5000 <a,b||e,i>*l2(m,i,b,a)
    lambda_one += -0.5 * einsum('abei,miba->em', g[v, v, v, o], l2)
    #	  1.0000 <j,m||e,i>*l1(i,a)*t1(a,j)
    lambda_one += 1.0 * einsum('jmei,ia,aj->em', g[o, o, v, o], l1, t1,
                               optimize=['einsum_path', (1, 2), (0, 1)])
    #	  1.0000 <j,m||i,a>*l1(i,e)*t1(a,j)
    lambda_one += 1.0 * einsum('jmia,ie,aj->em', g[o, o, o, v], l1, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])
    #	  1.0000 <m,a||e,b>*l1(i,a)*t1(b,i)
    lambda_one += 1.0 * einsum('maeb,ia,bi->em', g[o, v, v, v], l1, t1,
                               optimize=['einsum_path', (1, 2), (0, 1)])
    #	 -1.0000 <i,a||e,b>*l1(m,a)*t1(b,i)
    lambda_one += -1.0 * einsum('iaeb,ma,bi->em', g[o, v, v, v], l1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])
    #	 -0.5000 <k,m||i,j>*l2(i,j,e,a)*t1(a,k)
    lambda_one += -0.5 * einsum('kmij,ijea,ak->em', g[o, o, o, o], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])
    #	  1.0000 <j,b||e,i>*l2(m,i,b,a)*t1(a,j)
    lambda_one += 1.0 * einsum('jbei,miba,aj->em', g[o, v, v, o], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])
    #	 -1.0000 <m,a||j,b>*l2(i,j,a,e)*t1(b,i)
    lambda_one += -1.0 * einsum('majb,ijae,bi->em', g[o, v, o, v], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])
    #	  0.5000 <a,b||e,c>*l2(i,m,b,a)*t1(c,i)
    lambda_one += 0.5 * einsum('abec,imba,ci->em', g[v, v, v, v], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])
    #	  1.0000 <j,m||e,b>*l1(i,a)*t2(b,a,i,j)
    lambda_one += 1.0 * einsum('jmeb,ia,baij->em', g[o, o, v, v], l1, t2,
                               optimize=['einsum_path', (1, 2), (0, 1)])
    #	 -0.5000 <i,j||e,b>*l1(m,a)*t2(b,a,j,i)
    lambda_one += -0.5 * einsum('ijeb,ma,baji->em', g[o, o, v, v], l1, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])
    #	  0.5000 <j,m||a,b>*l1(i,e)*t2(a,b,i,j)
    lambda_one += 0.5 * einsum('jmab,ie,abij->em', g[o, o, v, v], l1, t2,
                               optimize=['einsum_path', (0, 2), (0, 1)])
    #	  0.5000 <k,m||e,j>*l2(i,j,b,a)*t2(b,a,i,k)
    lambda_one += 0.5 * einsum('kmej,ijba,baik->em', g[o, o, v, o], l2, t2,
                               optimize=['einsum_path', (1, 2), (0, 1)])
    #	  0.2500 <k,j||e,i>*l2(m,i,b,a)*t2(b,a,k,j)
    lambda_one += 0.25 * einsum('kjei,miba,bakj->em', g[o, o, v, o], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])
    #	  1.0000 <k,m||j,b>*l2(i,j,e,a)*t2(b,a,i,k)
    lambda_one += 1.0 * einsum('kmjb,ijea,baik->em', g[o, o, o, v], l2, t2,
                               optimize=['einsum_path', (0, 2), (0, 1)])
    #	  0.5000 <m,b||e,c>*l2(i,j,b,a)*t2(c,a,i,j)
    lambda_one += 0.5 * einsum('mbec,ijba,caij->em', g[o, v, v, v], l2, t2,
                               optimize=['einsum_path', (1, 2), (0, 1)])
    #	 -1.0000 <j,b||e,c>*l2(i,m,b,a)*t2(c,a,i,j)
    lambda_one += -1.0 * einsum('jbec,imba,caij->em', g[o, v, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])
    #	  0.2500 <m,a||b,c>*l2(i,j,a,e)*t2(b,c,i,j)
    lambda_one += 0.25 * einsum('mabc,ijae,bcij->em', g[o, v, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])
    #	  1.0000 <j,m||e,b>*l1(i,a)*t1(b,i)*t1(a,j)
    lambda_one += 1.0 * einsum('jmeb,ia,bi,aj->em', g[o, o, v, v], l1, t1, t1,
                               optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])
    #	  1.0000 <i,j||e,b>*l1(m,a)*t1(b,i)*t1(a,j)
    lambda_one += 1.0 * einsum('ijeb,ma,bi,aj->em', g[o, o, v, v], l1, t1, t1,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    #	 -1.0000 <j,m||a,b>*l1(i,e)*t1(a,j)*t1(b,i)
    lambda_one += -1.0 * einsum('jmab,ie,aj,bi->em', g[o, o, v, v], l1, t1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1),
                                          (0, 1)])
    #	 -0.5000 <k,j||e,i>*l2(m,i,b,a)*t1(b,j)*t1(a,k)
    lambda_one += -0.5 * einsum('kjei,miba,bj,ak->em', g[o, o, v, o], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])
    #	  1.0000 <k,m||j,b>*l2(i,j,e,a)*t1(b,i)*t1(a,k)
    lambda_one += 1.0 * einsum('kmjb,ijea,bi,ak->em', g[o, o, o, v], l2, t1, t1,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    #	 -1.0000 <j,b||e,c>*l2(i,m,b,a)*t1(c,i)*t1(a,j)
    lambda_one += -1.0 * einsum('jbec,imba,ci,aj->em', g[o, v, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])
    #	 -0.5000 <m,a||b,c>*l2(i,j,a,e)*t1(b,j)*t1(c,i)
    lambda_one += -0.5 * einsum('mabc,ijae,bj,ci->em', g[o, v, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])
    #	  0.5000 <k,m||e,c>*l2(i,j,b,a)*t1(c,j)*t2(b,a,i,k)
    lambda_one += 0.5 * einsum('kmec,ijba,cj,baik->em', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    #	  0.5000 <k,m||e,c>*l2(i,j,b,a)*t1(b,k)*t2(c,a,i,j)
    lambda_one += 0.5 * einsum('kmec,ijba,bk,caij->em', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])
    #	 -0.2500 <k,j||e,c>*l2(i,m,b,a)*t1(c,j)*t2(b,a,i,k)
    lambda_one += -0.25 * einsum('kjec,imba,cj,baik->em', g[o, o, v, v], l2, t1,
                                 t2, optimize=['einsum_path', (0, 2), (0, 1),
                                               (0, 1)])
    #	 -0.2500 <k,j||e,c>*l2(i,m,b,a)*t1(c,i)*t2(b,a,k,j)
    lambda_one += -0.25 * einsum('kjec,imba,ci,bakj->em', g[o, o, v, v], l2, t1,
                                 t2, optimize=['einsum_path', (0, 2), (0, 1),
                                               (0, 1)])
    #	  0.5000 <k,j||e,c>*l2(i,m,b,a)*t1(b,j)*t2(c,a,i,k)
    lambda_one += 0.5 * einsum('kjec,imba,bj,caik->em', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    #	 -0.2500 <k,m||b,c>*l2(i,j,e,a)*t1(b,k)*t2(c,a,i,j)
    lambda_one += -0.25 * einsum('kmbc,ijea,bk,caij->em', g[o, o, v, v], l2, t1,
                                 t2, optimize=['einsum_path', (0, 2), (0, 1),
                                               (0, 1)])
    #	  0.5000 <k,m||b,c>*l2(i,j,e,a)*t1(b,j)*t2(c,a,i,k)
    lambda_one += 0.5 * einsum('kmbc,ijea,bj,caik->em', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    #	 -0.2500 <k,m||b,c>*l2(i,j,e,a)*t1(a,k)*t2(b,c,i,j)
    lambda_one += -0.25 * einsum('kmbc,ijea,ak,bcij->em', g[o, o, v, v], l2, t1,
                                 t2, optimize=['einsum_path', (0, 2), (0, 1),
                                               (0, 1)])
    #	  0.5000 <k,j||e,c>*l2(i,m,b,a)*t1(a,k)*t2(c,b,i,j)
    lambda_one += 0.5 * einsum('kjec,imba,ak,cbij->em', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    #	  0.2500 <k,j||e,c>*l2(i,m,b,a)*t1(c,k)*t2(b,a,i,j)
    lambda_one += 0.25 * einsum('kjec,imba,ck,baij->em', g[o, o, v, v], l2, t1,
                                t2, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])
    #	  0.5000 <k,m||b,c>*l2(i,j,e,a)*t1(c,i)*t2(b,a,j,k)
    lambda_one += 0.5 * einsum('kmbc,ijea,ci,bajk->em', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])
    #	  0.2500 <k,m||b,c>*l2(i,j,e,a)*t1(c,k)*t2(b,a,i,j)
    lambda_one += 0.25 * einsum('kmbc,ijea,ck,baij->em', g[o, o, v, v], l2, t1,
                                t2, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])
    #	  0.5000 <k,j||e,c>*l2(i,m,b,a)*t1(c,i)*t1(b,j)*t1(a,k)
    lambda_one += 0.5 * einsum('kjec,imba,ci,bj,ak->em', g[o, o, v, v], l2, t1,
                               t1, t1,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1),
                                         (0, 1)])
    #	  0.5000 <k,m||b,c>*l2(i,j,e,a)*t1(b,j)*t1(c,i)*t1(a,k)
    lambda_one += 0.5 * einsum('kmbc,ijea,bj,ci,ak->em', g[o, o, v, v], l2, t1,
                               t1, t1,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1),
                                         (0, 1)])
    return lambda_one.transpose(1, 0)


def lambda_doubles(t1, t2, l1, l2, f, g, o, v):
    #	  1.0000 <m,n||e,f>
    lambda_two = 1.0 * einsum('mnef->efmn', g[o, o, v, v])

    #	 -1.0000 f(n,e)*l1(m,f)
    lambda_two += -1.0 * einsum('ne,mf->efmn', f[o, v], l1)

    #	  1.0000 f(m,e)*l1(n,f)
    lambda_two += 1.0 * einsum('me,nf->efmn', f[o, v], l1)

    #	  1.0000 f(n,f)*l1(m,e)
    lambda_two += 1.0 * einsum('nf,me->efmn', f[o, v], l1)

    #	 -1.0000 f(m,f)*l1(n,e)
    lambda_two += -1.0 * einsum('mf,ne->efmn', f[o, v], l1)

    #	 -1.0000 f(n,i)*l2(m,i,e,f)
    lambda_two += -1.0 * einsum('ni,mief->efmn', f[o, o], l2)

    #	  1.0000 f(m,i)*l2(n,i,e,f)
    lambda_two += 1.0 * einsum('mi,nief->efmn', f[o, o], l2)

    #	  1.0000 f(a,e)*l2(m,n,a,f)
    lambda_two += 1.0 * einsum('ae,mnaf->efmn', f[v, v], l2)

    #	 -1.0000 f(a,f)*l2(m,n,a,e)
    lambda_two += -1.0 * einsum('af,mnae->efmn', f[v, v], l2)

    #	  1.0000 f(i,e)*l2(m,n,f,a)*t1(a,i)
    lambda_two += 1.0 * einsum('ie,mnfa,ai->efmn', f[o, v], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 f(i,f)*l2(m,n,e,a)*t1(a,i)
    lambda_two += -1.0 * einsum('if,mnea,ai->efmn', f[o, v], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 f(n,a)*l2(i,m,e,f)*t1(a,i)
    lambda_two += 1.0 * einsum('na,imef,ai->efmn', f[o, v], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 f(m,a)*l2(i,n,e,f)*t1(a,i)
    lambda_two += -1.0 * einsum('ma,inef,ai->efmn', f[o, v], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <m,n||e,i>*l1(i,f)
    lambda_two += -1.0 * einsum('mnei,if->efmn', g[o, o, v, o], l1)

    #	  1.0000 <m,n||f,i>*l1(i,e)
    lambda_two += 1.0 * einsum('mnfi,ie->efmn', g[o, o, v, o], l1)

    #	 -1.0000 <n,a||e,f>*l1(m,a)
    lambda_two += -1.0 * einsum('naef,ma->efmn', g[o, v, v, v], l1)

    #	  1.0000 <m,a||e,f>*l1(n,a)
    lambda_two += 1.0 * einsum('maef,na->efmn', g[o, v, v, v], l1)

    #	  0.5000 <m,n||i,j>*l2(i,j,e,f)
    lambda_two += 0.5 * einsum('mnij,ijef->efmn', g[o, o, o, o], l2)

    #	  1.0000 <n,a||e,i>*l2(m,i,a,f)
    lambda_two += 1.0 * einsum('naei,miaf->efmn', g[o, v, v, o], l2)

    #	 -1.0000 <m,a||e,i>*l2(n,i,a,f)
    lambda_two += -1.0 * einsum('maei,niaf->efmn', g[o, v, v, o], l2)

    #	 -1.0000 <n,a||f,i>*l2(m,i,a,e)
    lambda_two += -1.0 * einsum('nafi,miae->efmn', g[o, v, v, o], l2)

    #	  1.0000 <m,a||f,i>*l2(n,i,a,e)
    lambda_two += 1.0 * einsum('mafi,niae->efmn', g[o, v, v, o], l2)

    #	 -0.5000 <a,b||e,f>*l2(m,n,b,a)
    lambda_two += -0.5 * einsum('abef,mnba->efmn', g[v, v, v, v], l2)

    #	 -1.0000 <i,n||e,f>*l1(m,a)*t1(a,i)
    lambda_two += -1.0 * einsum('inef,ma,ai->efmn', g[o, o, v, v], l1, t1,
                                optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 <i,m||e,f>*l1(n,a)*t1(a,i)
    lambda_two += 1.0 * einsum('imef,na,ai->efmn', g[o, o, v, v], l1, t1,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 <m,n||e,a>*l1(i,f)*t1(a,i)
    lambda_two += -1.0 * einsum('mnea,if,ai->efmn', g[o, o, v, v], l1, t1,
                                optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 <i,n||e,a>*l1(m,f)*t1(a,i)
    lambda_two += 1.0 * einsum('inea,mf,ai->efmn', g[o, o, v, v], l1, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <i,m||e,a>*l1(n,f)*t1(a,i)
    lambda_two += -1.0 * einsum('imea,nf,ai->efmn', g[o, o, v, v], l1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,n||f,a>*l1(i,e)*t1(a,i)
    lambda_two += 1.0 * einsum('mnfa,ie,ai->efmn', g[o, o, v, v], l1, t1,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 <i,n||f,a>*l1(m,e)*t1(a,i)
    lambda_two += -1.0 * einsum('infa,me,ai->efmn', g[o, o, v, v], l1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <i,m||f,a>*l1(n,e)*t1(a,i)
    lambda_two += 1.0 * einsum('imfa,ne,ai->efmn', g[o, o, v, v], l1, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <j,n||e,i>*l2(m,i,f,a)*t1(a,j)
    lambda_two += -1.0 * einsum('jnei,mifa,aj->efmn', g[o, o, v, o], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,m||e,i>*l2(n,i,f,a)*t1(a,j)
    lambda_two += 1.0 * einsum('jmei,nifa,aj->efmn', g[o, o, v, o], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,n||f,i>*l2(m,i,e,a)*t1(a,j)
    lambda_two += 1.0 * einsum('jnfi,miea,aj->efmn', g[o, o, v, o], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <j,m||f,i>*l2(n,i,e,a)*t1(a,j)
    lambda_two += -1.0 * einsum('jmfi,niea,aj->efmn', g[o, o, v, o], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <m,n||j,a>*l2(i,j,e,f)*t1(a,i)
    lambda_two += -1.0 * einsum('mnja,ijef,ai->efmn', g[o, o, o, v], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,n||i,a>*l2(m,i,e,f)*t1(a,j)
    lambda_two += 1.0 * einsum('jnia,mief,aj->efmn', g[o, o, o, v], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <j,m||i,a>*l2(n,i,e,f)*t1(a,j)
    lambda_two += -1.0 * einsum('jmia,nief,aj->efmn', g[o, o, o, v], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <i,b||e,f>*l2(m,n,b,a)*t1(a,i)
    lambda_two += 1.0 * einsum('ibef,mnba,ai->efmn', g[o, v, v, v], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <n,a||e,b>*l2(i,m,a,f)*t1(b,i)
    lambda_two += -1.0 * einsum('naeb,imaf,bi->efmn', g[o, v, v, v], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,a||e,b>*l2(i,n,a,f)*t1(b,i)
    lambda_two += 1.0 * einsum('maeb,inaf,bi->efmn', g[o, v, v, v], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <i,a||e,b>*l2(m,n,a,f)*t1(b,i)
    lambda_two += -1.0 * einsum('iaeb,mnaf,bi->efmn', g[o, v, v, v], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <n,a||f,b>*l2(i,m,a,e)*t1(b,i)
    lambda_two += 1.0 * einsum('nafb,imae,bi->efmn', g[o, v, v, v], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <m,a||f,b>*l2(i,n,a,e)*t1(b,i)
    lambda_two += -1.0 * einsum('mafb,inae,bi->efmn', g[o, v, v, v], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <i,a||f,b>*l2(m,n,a,e)*t1(b,i)
    lambda_two += 1.0 * einsum('iafb,mnae,bi->efmn', g[o, v, v, v], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <j,n||e,f>*l2(i,m,b,a)*t2(b,a,i,j)
    lambda_two += -0.5 * einsum('jnef,imba,baij->efmn', g[o, o, v, v], l2, t2,
                                optimize=['einsum_path', (1, 2), (0, 1)])

    #	  0.5000 <j,m||e,f>*l2(i,n,b,a)*t2(b,a,i,j)
    lambda_two += 0.5 * einsum('jmef,inba,baij->efmn', g[o, o, v, v], l2, t2,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -0.2500 <i,j||e,f>*l2(m,n,b,a)*t2(b,a,j,i)
    lambda_two += -0.25 * einsum('ijef,mnba,baji->efmn', g[o, o, v, v], l2, t2,
                                 optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <m,n||e,b>*l2(i,j,f,a)*t2(b,a,i,j)
    lambda_two += -0.5 * einsum('mneb,ijfa,baij->efmn', g[o, o, v, v], l2, t2,
                                optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 <j,n||e,b>*l2(i,m,f,a)*t2(b,a,i,j)
    lambda_two += 1.0 * einsum('jneb,imfa,baij->efmn', g[o, o, v, v], l2, t2,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <j,m||e,b>*l2(i,n,f,a)*t2(b,a,i,j)
    lambda_two += -1.0 * einsum('jmeb,infa,baij->efmn', g[o, o, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <i,j||e,b>*l2(m,n,f,a)*t2(b,a,j,i)
    lambda_two += 0.5 * einsum('ijeb,mnfa,baji->efmn', g[o, o, v, v], l2, t2,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,n||f,b>*l2(i,j,e,a)*t2(b,a,i,j)
    lambda_two += 0.5 * einsum('mnfb,ijea,baij->efmn', g[o, o, v, v], l2, t2,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 <j,n||f,b>*l2(i,m,e,a)*t2(b,a,i,j)
    lambda_two += -1.0 * einsum('jnfb,imea,baij->efmn', g[o, o, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,m||f,b>*l2(i,n,e,a)*t2(b,a,i,j)
    lambda_two += 1.0 * einsum('jmfb,inea,baij->efmn', g[o, o, v, v], l2, t2,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <i,j||f,b>*l2(m,n,e,a)*t2(b,a,j,i)
    lambda_two += -0.5 * einsum('ijfb,mnea,baji->efmn', g[o, o, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 <m,n||a,b>*l2(i,j,e,f)*t2(a,b,i,j)
    lambda_two += 0.25 * einsum('mnab,ijef,abij->efmn', g[o, o, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <j,n||a,b>*l2(i,m,e,f)*t2(a,b,i,j)
    lambda_two += -0.5 * einsum('jnab,imef,abij->efmn', g[o, o, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <j,m||a,b>*l2(i,n,e,f)*t2(a,b,i,j)
    lambda_two += 0.5 * einsum('jmab,inef,abij->efmn', g[o, o, v, v], l2, t2,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <i,j||e,f>*l2(m,n,b,a)*t1(b,i)*t1(a,j)
    lambda_two += 0.5 * einsum('ijef,mnba,bi,aj->efmn', g[o, o, v, v], l2, t1,
                               t1,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])

    #	  1.0000 <j,n||e,b>*l2(i,m,f,a)*t1(b,i)*t1(a,j)
    lambda_two += 1.0 * einsum('jneb,imfa,bi,aj->efmn', g[o, o, v, v], l2, t1,
                               t1,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])

    #	 -1.0000 <j,m||e,b>*l2(i,n,f,a)*t1(b,i)*t1(a,j)
    lambda_two += -1.0 * einsum('jmeb,infa,bi,aj->efmn', g[o, o, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	 -1.0000 <i,j||e,b>*l2(m,n,f,a)*t1(b,i)*t1(a,j)
    lambda_two += -1.0 * einsum('ijeb,mnfa,bi,aj->efmn', g[o, o, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (1, 2),
                                              (0, 1)])

    #	 -1.0000 <j,n||f,b>*l2(i,m,e,a)*t1(b,i)*t1(a,j)
    lambda_two += -1.0 * einsum('jnfb,imea,bi,aj->efmn', g[o, o, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	  1.0000 <j,m||f,b>*l2(i,n,e,a)*t1(b,i)*t1(a,j)
    lambda_two += 1.0 * einsum('jmfb,inea,bi,aj->efmn', g[o, o, v, v], l2, t1,
                               t1,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])

    #	  1.0000 <i,j||f,b>*l2(m,n,e,a)*t1(b,i)*t1(a,j)
    lambda_two += 1.0 * einsum('ijfb,mnea,bi,aj->efmn', g[o, o, v, v], l2, t1,
                               t1,
                               optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	 -0.5000 <m,n||a,b>*l2(i,j,e,f)*t1(a,j)*t1(b,i)
    lambda_two += -0.5 * einsum('mnab,ijef,aj,bi->efmn', g[o, o, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	  1.0000 <j,n||a,b>*l2(i,m,e,f)*t1(a,j)*t1(b,i)
    lambda_two += 1.0 * einsum('jnab,imef,aj,bi->efmn', g[o, o, v, v], l2, t1,
                               t1,
                               optimize=['einsum_path', (0, 2), (1, 2), (0, 1)])

    #	 -1.0000 <j,m||a,b>*l2(i,n,e,f)*t1(a,j)*t1(b,i)
    lambda_two += -1.0 * einsum('jmab,inef,aj,bi->efmn', g[o, o, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (1, 2),
                                              (0, 1)])
    return lambda_two.transpose(2, 3, 0, 1)


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
    singles_res += -1.0 * einsum('ia,am,ei->em', f[o, v], t1, t1,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,m>*t1(a,i)
    singles_res += 1.0 * einsum('ieam,ai->em', g[o, v, v, o], t1)

    #	  0.5000 <i,j||a,m>*t2(a,e,j,i)
    singles_res += 0.5 * einsum('ijam,aeji->em', g[o, o, v, o], t2)

    #	  0.5000 <i,e||a,b>*t2(a,b,i,m)
    singles_res += 0.5 * einsum('ieab,abim->em', g[o, v, v, v], t2)

    #	 -1.0000 <i,j||a,m>*t1(a,i)*t1(e,j)
    singles_res += -1.0 * einsum('ijam,ai,ej->em', g[o, o, v, o], t1, t1,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,b>*t1(a,i)*t1(b,m)
    singles_res += 1.0 * einsum('ieab,ai,bm->em', g[o, v, v, v], t1, t1,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,i)*t2(b,e,j,m)
    singles_res += 1.0 * einsum('ijab,ai,bejm->em', g[o, o, v, v], t1, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,j||a,b>*t1(a,m)*t2(b,e,j,i)
    singles_res += -0.5 * einsum('ijab,am,beji->em', g[o, o, v, v], t1, t2,
                                 optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <i,j||a,b>*t1(e,i)*t2(a,b,j,m)
    singles_res += 0.5 * einsum('ijab,ei,abjm->em', g[o, o, v, v], t1, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,i)*t1(b,m)*t1(e,j)
    singles_res += -1.0 * einsum('ijab,ai,bm,ej->em', g[o, o, v, v], t1, t1, t1,
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
    #	  1.0000 f(i,n)*t2(e,f,i,m)
    doubles_res = 1.0 * einsum('in,efim->efmn', f[o, o], t2)

    #	 -1.0000 f(i,m)*t2(e,f,i,n)
    doubles_res += -1.0 * einsum('im,efin->efmn', f[o, o], t2)

    #	  1.0000 f(e,a)*t2(a,f,m,n)
    doubles_res += 1.0 * einsum('ea,afmn->efmn', f[v, v], t2)

    #	 -1.0000 f(f,a)*t2(a,e,m,n)
    doubles_res += -1.0 * einsum('fa,aemn->efmn', f[v, v], t2)

    #	 -1.0000 f(i,a)*t1(a,n)*t2(e,f,m,i)
    doubles_res += -1.0 * einsum('ia,an,efmi->efmn', f[o, v], t1, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 f(i,a)*t1(a,m)*t2(e,f,n,i)
    doubles_res += 1.0 * einsum('ia,am,efni->efmn', f[o, v], t1, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 f(i,a)*t1(e,i)*t2(a,f,m,n)
    doubles_res += -1.0 * einsum('ia,ei,afmn->efmn', f[o, v], t1, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 f(i,a)*t1(f,i)*t2(a,e,m,n)
    doubles_res += 1.0 * einsum('ia,fi,aemn->efmn', f[o, v], t1, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

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
    doubles_res += 1.0 * einsum('ijmn,ei,fj->efmn', g[o, o, o, o], t1, t1,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,n>*t1(a,m)*t1(f,i)
    doubles_res += 1.0 * einsum('iean,am,fi->efmn', g[o, v, v, o], t1, t1,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,e||a,m>*t1(a,n)*t1(f,i)
    doubles_res += -1.0 * einsum('ieam,an,fi->efmn', g[o, v, v, o], t1, t1,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,f||a,n>*t1(a,m)*t1(e,i)
    doubles_res += -1.0 * einsum('ifan,am,ei->efmn', g[o, v, v, o], t1, t1,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,f||a,m>*t1(a,n)*t1(e,i)
    doubles_res += 1.0 * einsum('ifam,an,ei->efmn', g[o, v, v, o], t1, t1,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <e,f||a,b>*t1(a,n)*t1(b,m)
    doubles_res += -1.0 * einsum('efab,an,bm->efmn', g[v, v, v, v], t1, t1,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,n>*t1(a,i)*t2(e,f,j,m)
    doubles_res += 1.0 * einsum('ijan,ai,efjm->efmn', g[o, o, v, o], t1, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,j||a,n>*t1(a,m)*t2(e,f,j,i)
    doubles_res += -0.5 * einsum('ijan,am,efji->efmn', g[o, o, v, o], t1, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,n>*t1(e,i)*t2(a,f,j,m)
    doubles_res += -1.0 * einsum('ijan,ei,afjm->efmn', g[o, o, v, o], t1, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,n>*t1(f,i)*t2(a,e,j,m)
    doubles_res += 1.0 * einsum('ijan,fi,aejm->efmn', g[o, o, v, o], t1, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,m>*t1(a,i)*t2(e,f,j,n)
    doubles_res += -1.0 * einsum('ijam,ai,efjn->efmn', g[o, o, v, o], t1, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,j||a,m>*t1(a,n)*t2(e,f,j,i)
    doubles_res += 0.5 * einsum('ijam,an,efji->efmn', g[o, o, v, o], t1, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,m>*t1(e,i)*t2(a,f,j,n)
    doubles_res += 1.0 * einsum('ijam,ei,afjn->efmn', g[o, o, v, o], t1, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,m>*t1(f,i)*t2(a,e,j,n)
    doubles_res += -1.0 * einsum('ijam,fi,aejn->efmn', g[o, o, v, o], t1, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,b>*t1(a,i)*t2(b,f,m,n)
    doubles_res += 1.0 * einsum('ieab,ai,bfmn->efmn', g[o, v, v, v], t1, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,e||a,b>*t1(a,n)*t2(b,f,m,i)
    doubles_res += -1.0 * einsum('ieab,an,bfmi->efmn', g[o, v, v, v], t1, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,e||a,b>*t1(a,m)*t2(b,f,n,i)
    doubles_res += 1.0 * einsum('ieab,am,bfni->efmn', g[o, v, v, v], t1, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,e||a,b>*t1(f,i)*t2(a,b,m,n)
    doubles_res += 0.5 * einsum('ieab,fi,abmn->efmn', g[o, v, v, v], t1, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,f||a,b>*t1(a,i)*t2(b,e,m,n)
    doubles_res += -1.0 * einsum('ifab,ai,bemn->efmn', g[o, v, v, v], t1, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,f||a,b>*t1(a,n)*t2(b,e,m,i)
    doubles_res += 1.0 * einsum('ifab,an,bemi->efmn', g[o, v, v, v], t1, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,f||a,b>*t1(a,m)*t2(b,e,n,i)
    doubles_res += -1.0 * einsum('ifab,am,beni->efmn', g[o, v, v, v], t1, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,f||a,b>*t1(e,i)*t2(a,b,m,n)
    doubles_res += -0.5 * einsum('ifab,ei,abmn->efmn', g[o, v, v, v], t1, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,j||a,b>*t2(a,b,n,i)*t2(e,f,m,j)
    doubles_res += 0.5 * einsum('ijab,abni,efmj->efmn', g[o, o, v, v], t2, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 <i,j||a,b>*t2(a,b,m,i)*t2(e,f,n,j)
    doubles_res += -0.5 * einsum('ijab,abmi,efnj->efmn', g[o, o, v, v], t2, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.2500 <i,j||a,b>*t2(a,b,m,n)*t2(e,f,j,i)
    doubles_res += -0.25 * einsum('ijab,abmn,efji->efmn', g[o, o, v, v], t2, t2,
                                  optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,j||a,b>*t2(a,e,j,i)*t2(b,f,m,n)
    doubles_res += 0.5 * einsum('ijab,aeji,bfmn->efmn', g[o, o, v, v], t2, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 <i,j||a,b>*t2(a,e,n,i)*t2(b,f,m,j)
    doubles_res += -1.0 * einsum('ijab,aeni,bfmj->efmn', g[o, o, v, v], t2, t2,
                                 optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 <i,j||a,b>*t2(a,e,m,i)*t2(b,f,n,j)
    doubles_res += 1.0 * einsum('ijab,aemi,bfnj->efmn', g[o, o, v, v], t2, t2,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    #	  0.5000 <i,j||a,b>*t2(a,e,m,n)*t2(b,f,j,i)
    doubles_res += 0.5 * einsum('ijab,aemn,bfji->efmn', g[o, o, v, v], t2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <i,j||a,n>*t1(a,m)*t1(e,i)*t1(f,j)
    doubles_res += 1.0 * einsum('ijan,am,ei,fj->efmn', g[o, o, v, o], t1, t1,
                                t1, optimize=['einsum_path', (0, 1), (0, 2),
                                              (0, 1)])

    #	 -1.0000 <i,j||a,m>*t1(a,n)*t1(e,i)*t1(f,j)
    doubles_res += -1.0 * einsum('ijam,an,ei,fj->efmn', g[o, o, v, o], t1, t1,
                                 t1, optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	 -1.0000 <i,e||a,b>*t1(a,n)*t1(b,m)*t1(f,i)
    doubles_res += -1.0 * einsum('ieab,an,bm,fi->efmn', g[o, v, v, v], t1, t1,
                                 t1, optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	  1.0000 <i,f||a,b>*t1(a,n)*t1(b,m)*t1(e,i)
    doubles_res += 1.0 * einsum('ifab,an,bm,ei->efmn', g[o, v, v, v], t1, t1,
                                t1, optimize=['einsum_path', (0, 1), (0, 2),
                                              (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,i)*t1(b,n)*t2(e,f,m,j)
    doubles_res += -1.0 * einsum('ijab,ai,bn,efmj->efmn', g[o, o, v, v], t1, t1,
                                 t2, optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,i)*t1(b,m)*t2(e,f,n,j)
    doubles_res += 1.0 * einsum('ijab,ai,bm,efnj->efmn', g[o, o, v, v], t1, t1,
                                t2, optimize=['einsum_path', (0, 1), (0, 2),
                                              (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,i)*t1(e,j)*t2(b,f,m,n)
    doubles_res += -1.0 * einsum('ijab,ai,ej,bfmn->efmn', g[o, o, v, v], t1, t1,
                                 t2, optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,i)*t1(f,j)*t2(b,e,m,n)
    doubles_res += 1.0 * einsum('ijab,ai,fj,bemn->efmn', g[o, o, v, v], t1, t1,
                                t2, optimize=['einsum_path', (0, 1), (0, 2),
                                              (0, 1)])

    #	  0.5000 <i,j||a,b>*t1(a,n)*t1(b,m)*t2(e,f,j,i)
    doubles_res += 0.5 * einsum('ijab,an,bm,efji->efmn', g[o, o, v, v], t1, t1,
                                t2, optimize=['einsum_path', (0, 1), (0, 2),
                                              (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,n)*t1(e,i)*t2(b,f,m,j)
    doubles_res += -1.0 * einsum('ijab,an,ei,bfmj->efmn', g[o, o, v, v], t1, t1,
                                 t2, optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,n)*t1(f,i)*t2(b,e,m,j)
    doubles_res += 1.0 * einsum('ijab,an,fi,bemj->efmn', g[o, o, v, v], t1, t1,
                                t2, optimize=['einsum_path', (0, 1), (0, 2),
                                              (0, 1)])

    #	  1.0000 <i,j||a,b>*t1(a,m)*t1(e,i)*t2(b,f,n,j)
    doubles_res += 1.0 * einsum('ijab,am,ei,bfnj->efmn', g[o, o, v, v], t1, t1,
                                t2, optimize=['einsum_path', (0, 1), (0, 2),
                                              (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,m)*t1(f,i)*t2(b,e,n,j)
    doubles_res += -1.0 * einsum('ijab,am,fi,benj->efmn', g[o, o, v, v], t1, t1,
                                 t2, optimize=['einsum_path', (0, 1), (0, 2),
                                               (0, 1)])

    #	  0.5000 <i,j||a,b>*t1(e,i)*t1(f,j)*t2(a,b,m,n)
    doubles_res += 0.5 * einsum('ijab,ei,fj,abmn->efmn', g[o, o, v, v], t1, t1,
                                t2, optimize=['einsum_path', (0, 1), (0, 2),
                                              (0, 1)])

    #	 -1.0000 <i,j||a,b>*t1(a,n)*t1(b,m)*t1(e,i)*t1(f,j)
    doubles_res += -1.0 * einsum('ijab,an,bm,ei,fj->efmn', g[o, o, v, v], t1,
                                 t1, t1, t1,
                                 optimize=['einsum_path', (0, 1), (0, 3),
                                           (0, 2), (0, 1)])

    return doubles_res


def kernel(t1, t2, l1, l2, fock, g, o, v, e_ai, e_abij, max_iter=100,
           stopping_eps=1.0E-8):
    """
    Solve method of multipliers

    :param t1:
    :param t2:
    :param l1:
    :param l2:
    :param fock:
    :param g:
    :param o:
    :param v:
    :param e_ai:
    :param e_abij:
    :param max_iter:
    :param stopping_eps:
    :return:
    """
    fock_e_ai = np.reciprocal(e_ai)
    fock_e_abij = np.reciprocal(e_abij)
    old_energy = ccsd_energy(t1, t2, fock, g, o, v)

    print("\tSolving T-quations")
    for idx in range(max_iter):

        singles_res = singles_residual(t1, t2, fock, g, o, v) + fock_e_ai * t1
        doubles_res = doubles_residual(t1, t2, fock, g, o, v) + fock_e_abij * t2

        new_singles = singles_res * e_ai
        new_doubles = doubles_res * e_abij

        current_energy = ccsd_energy(new_singles, new_doubles, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < stopping_eps:
            # assign t1 and t2 variables for future use before breaking
            t1 = new_singles
            t2 = new_doubles
            break
        else:
            # assign t1 and t2 and old_energy for next iteration
            t1 = new_singles
            t2 = new_doubles
            old_energy = current_energy
            print("\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}".format(idx, old_energy, delta_e))
    else:
        raise ValueError("Did not converge")

    print("\n\tSolving lambda-quations\n")
    # inverse diagonal fock should be rearranged for lambdas
    lfock_e_ai = fock_e_ai.transpose(1, 0)
    lfock_e_abij = fock_e_abij.transpose(2, 3, 0, 1)
    # diagonal fock should be rearranged for lambda
    le_ai = e_ai.transpose(1, 0)
    le_abij = e_abij.transpose(2, 3, 0, 1)

    # set old energy with initial l1 and l2
    old_energy = lagrangian_energy(t1, t2, l1, l2, fock, g, o, v)
    # now solve for lambdas!
    for idx in range(max_iter):

        lsingles_res = lambda_singles(t1, t2, l1, l2,  fock, g, o, v)
        ldoubles_res = lambda_doubles(t1, t2, l1, l2, fock, g, o, v)

        total_lambda_res = np.linalg.norm(lsingles_res) + np.linalg.norm(ldoubles_res)
        lsingles_res += lfock_e_ai * l1
        ldoubles_res += lfock_e_abij * l2

        lnew_singles = lsingles_res * le_ai
        lnew_doubles = ldoubles_res * le_abij

        current_energy = lagrangian_energy(t1, t2, lnew_singles, lnew_doubles, fock, g, o, v)
        delta_e = np.abs(old_energy - current_energy)

        if delta_e < stopping_eps and total_lambda_res < stopping_eps:
            l1 = lnew_singles
            l2 = lnew_doubles
            break
        else:
            l1 = lnew_singles
            l2 = lnew_doubles
            old_energy = current_energy
            print("\tIteration {: 5d}\t{: 5.15f}\t{: 5.15f}\t{: 5.15f}".format(
                idx, old_energy, delta_e,
                np.linalg.norm(lambda_singles(t1, t2, l1, l2, fock, g, o, v)) +
                np.linalg.norm(lambda_doubles(t1, t2, l1, l2, fock, g, o, v))
            ))
    else:
        raise ValueError("Did not converge")

    return t1, t2, l1, l2


def main():
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
    occ = mf.mo_occ
    nele = int(sum(occ))
    nocc = nele // 2
    assert np.allclose(np.transpose(mycc.t2, [1, 0, 3, 2]), mycc.t2)

    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
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

    print("{: 5.10f}\tNick's CC energy ".format(cc_energy))
    print("{: 5.10f}\tNick's CC energy22 ".format(cc_energy2 - hf_energy))
    print("{: 5.10f}\tNick's CC energy pdaggerq".format(ccsd_energy(t1, t2, fock, gtei, o, v) - hf_energy))
    print("{: 5.10f}\tpyscf CC energy".format(molecule.ccsd_energy - molecule.hf_energy))
    print("{: 5.10f}\t{: 5.10f}  CCSD pyscf, CCSD NCR".format(molecule.ccsd_energy, hf_energy + cc_energy + molecule.nuclear_repulsion))

    print("Check total energy ", ccsd_energy(t1s.transpose(1, 0), t2s.transpose(2, 3, 0, 1), fock, gtei, o, v) + molecule.nuclear_repulsion)
    print("electronic energy", ccsd_energy(t1s.transpose(1, 0), t2s.transpose(2, 3, 0, 1), fock, gtei, o, v) )


    g = gtei
    t1 = t1s.transpose(1, 0)
    t2 = t2s.transpose(2, 3, 0, 1)

    assert np.allclose(singles_residual(t1, t2, fock, g, o, v), 0)
    assert np.allclose(doubles_residual(t1, t2, fock, g, o, v), 0, atol=1.0E-6)

    t1f, t2f, l1f, l2f = kernel(np.zeros_like(t1), np.zeros_like(t2), np.zeros_like(t1.transpose(1, 0)), np.zeros_like(t2.transpose(2, 3, 0, 1)), fock, g, o, v, e_ai, e_abij)
    print(ccsd_energy(t1f, t2f, fock, g, o, v) - hf_energy)



if __name__ == "__main__":
    main()



