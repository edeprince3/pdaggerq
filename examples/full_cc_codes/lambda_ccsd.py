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
    lambda_one = 1.0 * einsum('me->me', f[o, v])

    #	 -1.0000 <i,m||e,a>*t1(a,i)
    lambda_one += -1.0 * einsum('imea,ai->me', g[o, o, v, v], t1)

    #	 -1.0000 f(m,i)*l1(i,e)
    lambda_one += -1.0 * einsum('mi,ie->me', f[o, o], l1)

    #	  1.0000 f(a,e)*l1(m,a)
    lambda_one += 1.0 * einsum('ae,ma->me', f[v, v], l1)

    #	 -1.0000 f(i,e)*l1(m,a)*t1(a,i)
    lambda_one += -1.0 * einsum('ie,ma,ai->me', f[o, v], l1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 f(m,a)*l1(i,e)*t1(a,i)
    lambda_one += -1.0 * einsum('ma,ie,ai->me', f[o, v], l1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 f(j,e)*l2(i,m,b,a)*t2(b,a,i,j)
    lambda_one += -0.5 * einsum('je,imba,baij->me', f[o, v], l2, t2,
                                optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -0.5000 f(m,b)*l2(i,j,e,a)*t2(b,a,i,j)
    lambda_one += -0.5 * einsum('mb,ijea,baij->me', f[o, v], l2, t2,
                                optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 <m,a||e,i>*l1(i,a)
    lambda_one += 1.0 * einsum('maei,ia->me', g[o, v, v, o], l1)

    #	  0.5000 <m,a||i,j>*l2(i,j,a,e)
    lambda_one += 0.5 * einsum('maij,ijae->me', g[o, v, o, o], l2)

    #	  0.5000 <b,a||e,i>*l2(m,i,b,a)
    lambda_one += 0.5 * einsum('baei,miba->me', g[v, v, v, o], l2)

    #	  1.0000 <j,m||e,i>*l1(i,a)*t1(a,j)
    lambda_one += 1.0 * einsum('jmei,ia,aj->me', g[o, o, v, o], l1, t1,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 <j,m||a,i>*l1(i,e)*t1(a,j)
    lambda_one += -1.0 * einsum('jmai,ie,aj->me', g[o, o, v, o], l1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,a||e,b>*l1(i,a)*t1(b,i)
    lambda_one += 1.0 * einsum('maeb,ia,bi->me', g[o, v, v, v], l1, t1,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 <i,a||e,b>*l1(m,a)*t1(b,i)
    lambda_one += -1.0 * einsum('iaeb,ma,bi->me', g[o, v, v, v], l1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <k,m||i,j>*l2(i,j,e,a)*t1(a,k)
    lambda_one += -0.5 * einsum('kmij,ijea,ak->me', g[o, o, o, o], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,b||e,i>*l2(m,i,b,a)*t1(a,j)
    lambda_one += 1.0 * einsum('jbei,miba,aj->me', g[o, v, v, o], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <m,a||b,j>*l2(i,j,a,e)*t1(b,i)
    lambda_one += 1.0 * einsum('mabj,ijae,bi->me', g[o, v, v, o], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 <b,a||e,c>*l2(i,m,b,a)*t1(c,i)
    lambda_one += -0.5 * einsum('baec,imba,ci->me', g[v, v, v, v], l2, t1,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,m||e,b>*l1(i,a)*t2(b,a,i,j)
    lambda_one += 1.0 * einsum('jmeb,ia,baij->me', g[o, o, v, v], l1, t2,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	  0.5000 <j,i||e,b>*l1(m,a)*t2(b,a,j,i)
    lambda_one += 0.5 * einsum('jieb,ma,baji->me', g[o, o, v, v], l1, t2,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <j,m||a,b>*l1(i,e)*t2(a,b,i,j)
    lambda_one += 0.5 * einsum('jmab,ie,abij->me', g[o, o, v, v], l1, t2,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <k,m||e,j>*l2(i,j,b,a)*t2(b,a,i,k)
    lambda_one += 0.5 * einsum('kmej,ijba,baik->me', g[o, o, v, o], l2, t2,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	  0.2500 <k,j||e,i>*l2(m,i,b,a)*t2(b,a,k,j)
    lambda_one += 0.25 * einsum('kjei,miba,bakj->me', g[o, o, v, o], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 <k,m||b,j>*l2(i,j,e,a)*t2(b,a,i,k)
    lambda_one += -1.0 * einsum('kmbj,ijea,baik->me', g[o, o, v, o], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.5000 <m,b||e,c>*l2(i,j,b,a)*t2(c,a,i,j)
    lambda_one += 0.5 * einsum('mbec,ijba,caij->me', g[o, v, v, v], l2, t2,
                               optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 <j,b||e,c>*l2(i,m,b,a)*t2(c,a,i,j)
    lambda_one += -1.0 * einsum('jbec,imba,caij->me', g[o, v, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  0.2500 <m,a||b,c>*l2(i,j,a,e)*t2(b,c,i,j)
    lambda_one += 0.25 * einsum('mabc,ijae,bcij->me', g[o, v, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 <j,m||e,b>*l1(i,a)*t1(b,i)*t1(a,j)
    lambda_one += 1.0 * einsum('jmeb,ia,bi,aj->me', g[o, o, v, v], l1, t1, t1,
                               optimize=['einsum_path', (1, 2), (1, 2), (0, 1)])

    #	 -1.0000 <j,i||e,b>*l1(m,a)*t1(b,i)*t1(a,j)
    lambda_one += -1.0 * einsum('jieb,ma,bi,aj->me', g[o, o, v, v], l1, t1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1),
                                          (0, 1)])

    #	 -1.0000 <j,m||a,b>*l1(i,e)*t1(a,j)*t1(b,i)
    lambda_one += -1.0 * einsum('jmab,ie,aj,bi->me', g[o, o, v, v], l1, t1, t1,
                                optimize=['einsum_path', (0, 2), (0, 1),
                                          (0, 1)])

    #	 -0.5000 <k,j||e,i>*l2(m,i,b,a)*t1(b,j)*t1(a,k)
    lambda_one += -0.5 * einsum('kjei,miba,bj,ak->me', g[o, o, v, o], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	 -1.0000 <k,m||b,j>*l2(i,j,e,a)*t1(b,i)*t1(a,k)
    lambda_one += -1.0 * einsum('kmbj,ijea,bi,ak->me', g[o, o, v, o], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	 -1.0000 <j,b||e,c>*l2(i,m,b,a)*t1(c,i)*t1(a,j)
    lambda_one += -1.0 * einsum('jbec,imba,ci,aj->me', g[o, v, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	 -0.5000 <m,a||b,c>*l2(i,j,a,e)*t1(b,j)*t1(c,i)
    lambda_one += -0.5 * einsum('mabc,ijae,bj,ci->me', g[o, v, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	  0.5000 <k,m||e,c>*l2(i,j,b,a)*t1(c,j)*t2(b,a,i,k)
    lambda_one += 0.5 * einsum('kmec,ijba,cj,baik->me', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])

    #	  0.5000 <k,m||e,c>*l2(i,j,b,a)*t1(b,k)*t2(c,a,i,j)
    lambda_one += 0.5 * einsum('kmec,ijba,bk,caij->me', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (1, 3), (1, 2), (0, 1)])

    #	 -0.5000 <k,j||e,c>*l2(i,m,b,a)*t1(c,j)*t2(b,a,i,k)
    lambda_one += -0.5 * einsum('kjec,imba,cj,baik->me', g[o, o, v, v], l2, t1,
                                t2, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	 -0.2500 <k,j||e,c>*l2(i,m,b,a)*t1(c,i)*t2(b,a,k,j)
    lambda_one += -0.25 * einsum('kjec,imba,ci,bakj->me', g[o, o, v, v], l2, t1,
                                 t2, optimize=['einsum_path', (0, 2), (0, 1),
                                               (0, 1)])

    #	  1.0000 <k,j||e,c>*l2(i,m,b,a)*t1(b,j)*t2(c,a,i,k)
    lambda_one += 1.0 * einsum('kjec,imba,bj,caik->me', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])

    #	 -0.5000 <k,m||b,c>*l2(i,j,e,a)*t1(b,k)*t2(c,a,i,j)
    lambda_one += -0.5 * einsum('kmbc,ijea,bk,caij->me', g[o, o, v, v], l2, t1,
                                t2, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	  1.0000 <k,m||b,c>*l2(i,j,e,a)*t1(b,j)*t2(c,a,i,k)
    lambda_one += 1.0 * einsum('kmbc,ijea,bj,caik->me', g[o, o, v, v], l2, t1,
                               t2,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1)])

    #	 -0.2500 <k,m||b,c>*l2(i,j,e,a)*t1(a,k)*t2(b,c,i,j)
    lambda_one += -0.25 * einsum('kmbc,ijea,ak,bcij->me', g[o, o, v, v], l2, t1,
                                 t2, optimize=['einsum_path', (0, 2), (0, 1),
                                               (0, 1)])

    #	  0.5000 <k,j||e,c>*l2(i,m,b,a)*t1(c,i)*t1(b,j)*t1(a,k)
    lambda_one += 0.5 * einsum('kjec,imba,ci,bj,ak->me', g[o, o, v, v], l2, t1,
                               t1, t1,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1),
                                         (0, 1)])

    #	  0.5000 <k,m||b,c>*l2(i,j,e,a)*t1(b,j)*t1(c,i)*t1(a,k)
    lambda_one += 0.5 * einsum('kmbc,ijea,bj,ci,ak->me', g[o, o, v, v], l2, t1,
                               t1, t1,
                               optimize=['einsum_path', (0, 2), (0, 1), (0, 1),
                                         (0, 1)])
    return lambda_one


def lambda_doubles(t1, t2, l1, l2, f, g, o, v):
    #	  1.0000 <m,n||e,f>
    lambda_two = 1.0 * einsum('mnef->mnef', g[o, o, v, v])

    #	 -1.0000 P(m,n)*P(e,f)f(n,e)*l1(m,f)
    contracted_intermediate = -1.0 * einsum('ne,mf->mnef', f[o, v], l1)
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	 -1.0000 P(m,n)f(n,i)*l2(m,i,e,f)
    contracted_intermediate = -1.0 * einsum('ni,mief->mnef', f[o, o], l2)
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	  1.0000 P(e,f)f(a,e)*l2(m,n,a,f)
    contracted_intermediate = 1.0 * einsum('ae,mnaf->mnef', f[v, v], l2)
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	  1.0000 P(e,f)f(i,e)*l2(m,n,f,a)*t1(a,i)
    contracted_intermediate = 1.0 * einsum('ie,mnfa,ai->mnef', f[o, v], l2, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	  1.0000 P(m,n)f(n,a)*l2(i,m,e,f)*t1(a,i)
    contracted_intermediate = 1.0 * einsum('na,imef,ai->mnef', f[o, v], l2, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	 -1.0000 P(e,f)<m,n||e,i>*l1(i,f)
    contracted_intermediate = -1.0 * einsum('mnei,if->mnef', g[o, o, v, o], l1)
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	 -1.0000 P(m,n)<n,a||e,f>*l1(m,a)
    contracted_intermediate = -1.0 * einsum('naef,ma->mnef', g[o, v, v, v], l1)
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	  0.5000 <m,n||i,j>*l2(i,j,e,f)
    lambda_two += 0.5 * einsum('mnij,ijef->mnef', g[o, o, o, o], l2)

    #	  1.0000 P(m,n)*P(e,f)<n,a||e,i>*l2(m,i,a,f)
    contracted_intermediate = 1.0 * einsum('naei,miaf->mnef', g[o, v, v, o], l2)
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	  0.5000 <b,a||e,f>*l2(m,n,b,a)
    lambda_two += 0.5 * einsum('baef,mnba->mnef', g[v, v, v, v], l2)

    #	 -1.0000 P(m,n)<i,n||e,f>*l1(m,a)*t1(a,i)
    contracted_intermediate = -1.0 * einsum('inef,ma,ai->mnef', g[o, o, v, v],
                                            l1, t1,
                                            optimize=['einsum_path', (1, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	 -1.0000 P(e,f)<m,n||e,a>*l1(i,f)*t1(a,i)
    contracted_intermediate = -1.0 * einsum('mnea,if,ai->mnef', g[o, o, v, v],
                                            l1, t1,
                                            optimize=['einsum_path', (1, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	  1.0000 P(m,n)*P(e,f)<i,n||e,a>*l1(m,f)*t1(a,i)
    contracted_intermediate = 1.0 * einsum('inea,mf,ai->mnef', g[o, o, v, v],
                                           l1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	 -1.0000 P(m,n)*P(e,f)<j,n||e,i>*l2(m,i,f,a)*t1(a,j)
    contracted_intermediate = -1.0 * einsum('jnei,mifa,aj->mnef', g[o, o, v, o],
                                            l2, t1,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	  1.0000 <m,n||a,j>*l2(i,j,e,f)*t1(a,i)
    lambda_two += 1.0 * einsum('mnaj,ijef,ai->mnef', g[o, o, v, o], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(m,n)<j,n||a,i>*l2(m,i,e,f)*t1(a,j)
    contracted_intermediate = -1.0 * einsum('jnai,mief,aj->mnef', g[o, o, v, o],
                                            l2, t1,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	  1.0000 <i,b||e,f>*l2(m,n,b,a)*t1(a,i)
    lambda_two += 1.0 * einsum('ibef,mnba,ai->mnef', g[o, v, v, v], l2, t1,
                               optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -1.0000 P(m,n)*P(e,f)<n,a||e,b>*l2(i,m,a,f)*t1(b,i)
    contracted_intermediate = -1.0 * einsum('naeb,imaf,bi->mnef', g[o, v, v, v],
                                            l2, t1,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	 -1.0000 P(e,f)<i,a||e,b>*l2(m,n,a,f)*t1(b,i)
    contracted_intermediate = -1.0 * einsum('iaeb,mnaf,bi->mnef', g[o, v, v, v],
                                            l2, t1,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	 -0.5000 P(m,n)<j,n||e,f>*l2(i,m,b,a)*t2(b,a,i,j)
    contracted_intermediate = -0.5 * einsum('jnef,imba,baij->mnef',
                                            g[o, o, v, v], l2, t2,
                                            optimize=['einsum_path', (1, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	  0.2500 <j,i||e,f>*l2(m,n,b,a)*t2(b,a,j,i)
    lambda_two += 0.25 * einsum('jief,mnba,baji->mnef', g[o, o, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 P(e,f)<m,n||e,b>*l2(i,j,f,a)*t2(b,a,i,j)
    contracted_intermediate = -0.5 * einsum('mneb,ijfa,baij->mnef',
                                            g[o, o, v, v], l2, t2,
                                            optimize=['einsum_path', (1, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	  1.0000 P(m,n)*P(e,f)<j,n||e,b>*l2(i,m,f,a)*t2(b,a,i,j)
    contracted_intermediate = 1.0 * einsum('jneb,imfa,baij->mnef',
                                           g[o, o, v, v], l2, t2,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	 -0.5000 P(e,f)<j,i||e,b>*l2(m,n,f,a)*t2(b,a,j,i)
    contracted_intermediate = -0.5 * einsum('jieb,mnfa,baji->mnef',
                                            g[o, o, v, v], l2, t2,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	  0.2500 <m,n||a,b>*l2(i,j,e,f)*t2(a,b,i,j)
    lambda_two += 0.25 * einsum('mnab,ijef,abij->mnef', g[o, o, v, v], l2, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    #	 -0.5000 P(m,n)<j,n||a,b>*l2(i,m,e,f)*t2(a,b,i,j)
    contracted_intermediate = -0.5 * einsum('jnab,imef,abij->mnef',
                                            g[o, o, v, v], l2, t2,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)

    #	 -0.5000 <j,i||e,f>*l2(m,n,b,a)*t1(b,i)*t1(a,j)
    lambda_two += -0.5 * einsum('jief,mnba,bi,aj->mnef', g[o, o, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	  1.0000 P(m,n)*P(e,f)<j,n||e,b>*l2(i,m,f,a)*t1(b,i)*t1(a,j)
    contracted_intermediate = 1.0 * einsum('jneb,imfa,bi,aj->mnef',
                                           g[o, o, v, v], l2, t1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1), (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate) + -1.00000 * einsum('mnef->mnfe',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'mnef->nmfe', contracted_intermediate)

    #	  1.0000 P(e,f)<j,i||e,b>*l2(m,n,f,a)*t1(b,i)*t1(a,j)
    contracted_intermediate = 1.0 * einsum('jieb,mnfa,bi,aj->mnef',
                                           g[o, o, v, v], l2, t1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (1, 2), (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->mnfe', contracted_intermediate)

    #	 -0.5000 <m,n||a,b>*l2(i,j,e,f)*t1(a,j)*t1(b,i)
    lambda_two += -0.5 * einsum('mnab,ijef,aj,bi->mnef', g[o, o, v, v], l2, t1,
                                t1, optimize=['einsum_path', (0, 2), (0, 1),
                                              (0, 1)])

    #	  1.0000 P(m,n)<j,n||a,b>*l2(i,m,e,f)*t1(a,j)*t1(b,i)
    contracted_intermediate = 1.0 * einsum('jnab,imef,aj,bi->mnef',
                                           g[o, o, v, v], l2, t1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (1, 2), (0, 1)])
    lambda_two += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'mnef->nmef', contracted_intermediate)
    return lambda_two


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


def ccsd_d1(t1, t2, l1, l2, d1_fermi, o, v):
    d1 = d1_fermi.copy()
    opdm = np.zeros_like(d1)
    #    D1(m,n):

    # 	  1.0000 d1(m,n)
    opdm[o, o] += 1.0 * einsum('mn->mn', d1[o, o])

    # 	 -1.0000 l1(n,a)*t1(a,m)
    opdm[o, o] += -1.0 * einsum('na,am->mn', l1, t1)

    # 	 -0.5000 l2(i,n,b,a)*t2(b,a,i,m)
    opdm[o, o] += -0.5 * einsum('inba,baim->mn', l2, t2)

    #    D1(e,f):

    # 	  1.0000 l1(i,e)*t1(f,i)
    opdm[v, v] += 1.0 * einsum('ie,fi->ef', l1, t1)

    # 	  0.5000 l2(i,j,e,a)*t2(f,a,i,j)
    opdm[v, v] += 0.5 * einsum('ijea,faij->ef', l2, t2)

    #    D1(e,m):

    # 	  1.0000 l1(m,e)
    opdm[v, o] += 1.0 * einsum('me->em', l1)

    #    D1(m,e):

    # 	  1.0000 t1(e,m)
    opdm[o, v] += 1.0 * einsum('em->me', t1)

    # 	 -1.0000 l1(i,a)*t2(e,a,i,m)
    opdm[o, v] += -1.0 * einsum('ia,eaim->me', l1, t2)

    # 	 -1.0000 l1(i,a)*t1(e,i)*t1(a,m)
    opdm[o, v] += -1.0 * einsum('ia,ei,am->me', l1, t1, t1,
                                optimize=['einsum_path', (0, 1), (0, 1)])

    # 	 -0.5000 l2(i,j,b,a)*t1(e,j)*t2(b,a,i,m)
    opdm[o, v] += -0.5 * einsum('ijba,ej,baim->me', l2, t1, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])

    # 	 -0.5000 l2(i,j,b,a)*t1(b,m)*t2(e,a,i,j)
    opdm[o, v] += -0.5 * einsum('ijba,bm,eaij->me', l2, t1, t2,
                                optimize=['einsum_path', (0, 2), (0, 1)])
    return opdm


def ccsd_d2(t1, t2, l1, l2, d1fermi, o, v):
    d1 = d1fermi.copy()
    nso = d1.shape[0]
    tpdm = np.zeros((nso, nso, nso, nso))

    #    D2(i,j,k,l):

    #	  1.0000 d1(j,l)*d1(i,k)
    tpdm[o, o, o, o] += 1.0 * einsum('jl,ik->ijkl', d1[o, o], d1[o, o])

    #	 -1.0000 d1(i,l)*d1(j,k)
    tpdm[o, o, o, o] += -1.0 * einsum('il,jk->ijkl', d1[o, o], d1[o, o])

    #	 -1.0000 d1(j,l)*l1(k,a)*t1(a,i)
    tpdm[o, o, o, o] += -1.0 * einsum('jl,ka,ai->ijkl', d1[o, o], l1, t1,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 d1(i,l)*l1(k,a)*t1(a,j)
    tpdm[o, o, o, o] += 1.0 * einsum('il,ka,aj->ijkl', d1[o, o], l1, t1,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	  1.0000 d1(j,k)*l1(l,a)*t1(a,i)
    tpdm[o, o, o, o] += 1.0 * einsum('jk,la,ai->ijkl', d1[o, o], l1, t1,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -1.0000 d1(i,k)*l1(l,a)*t1(a,j)
    tpdm[o, o, o, o] += -1.0 * einsum('ik,la,aj->ijkl', d1[o, o], l1, t1,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -0.5000 d1(j,l)*l2(j,k,b,a)*t2(b,a,j,i)
    tpdm[o, o, o, o] += -0.5 * einsum('jl,jkba,baji->ijkl', d1[o, o], l2, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  0.5000 d1(i,l)*l2(i,k,b,a)*t2(b,a,i,j)
    tpdm[o, o, o, o] += 0.5 * einsum('il,ikba,baij->ijkl', d1[o, o], l2, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	  0.5000 d1(j,k)*l2(j,l,b,a)*t2(b,a,j,i)
    tpdm[o, o, o, o] += 0.5 * einsum('jk,jlba,baji->ijkl', d1[o, o], l2, t2,
                                     optimize=['einsum_path', (1, 2), (0, 1)])

    #	 -0.5000 d1(i,k)*l2(i,l,b,a)*t2(b,a,i,j)
    tpdm[o, o, o, o] += -0.5 * einsum('ik,ilba,baij->ijkl', d1[o, o], l2, t2,
                                      optimize=['einsum_path', (1, 2), (0, 1)])

    #	  0.5000 l2(k,l,b,a)*t2(b,a,i,j)
    tpdm[o, o, o, o] += 0.5 * einsum('klba,baij->ijkl', l2, t2)

    #	 -1.0000 l2(k,l,b,a)*t1(b,j)*t1(a,i)
    tpdm[o, o, o, o] += -1.0 * einsum('klba,bj,ai->ijkl', l2, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #    D2(a,b,c,d):

    #	  0.5000 l2(i,j,a,b)*t2(c,d,i,j)
    tpdm[v, v, v, v] += 0.5 * einsum('ijab,cdij->abcd', l2, t2)

    #	 -1.0000 l2(i,j,a,b)*t1(c,j)*t1(d,i)
    tpdm[v, v, v, v] += -1.0 * einsum('ijab,cj,di->abcd', l2, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #    D2(i,j,a,b):

    #	  1.0000 t2(a,b,i,j)
    tpdm[o, o, v, v] += 1.0 * einsum('abij->ijab', t2)

    #	 -1.0000 P(i,j)t1(a,j)*t1(b,i)
    contracted_intermediate = -1.0 * einsum('aj,bi->ijab', t1, t1)
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate)

    #	  1.0000 P(i,j)*P(a,b)l1(k,c)*t1(a,j)*t2(b,c,k,i)
    contracted_intermediate = 1.0 * einsum('kc,aj,bcki->ijab', l1, t1, t2,
                                           optimize=['einsum_path', (0, 2),
                                                     (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	  1.0000 P(a,b)l1(k,c)*t1(a,k)*t2(b,c,i,j)
    contracted_intermediate = 1.0 * einsum('kc,ak,bcij->ijab', l1, t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->ijba', contracted_intermediate)

    #	  1.0000 P(i,j)l1(k,c)*t1(c,j)*t2(a,b,k,i)
    contracted_intermediate = 1.0 * einsum('kc,cj,abki->ijab', l1, t1, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate)

    #	 -0.5000 P(i,j)l2(k,l,d,c)*t2(a,b,l,j)*t2(d,c,k,i)
    contracted_intermediate = -0.5 * einsum('kldc,ablj,dcki->ijab', l2, t2, t2,
                                            optimize=['einsum_path', (0, 2),
                                                      (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate)

    #	  0.2500 l2(k,l,d,c)*t2(a,b,k,l)*t2(d,c,i,j)
    tpdm[o, o, v, v] += 0.25 * einsum('kldc,abkl,dcij->ijab', l2, t2, t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -0.5000 l2(k,l,d,c)*t2(a,d,i,j)*t2(b,c,k,l)
    tpdm[o, o, v, v] += -0.5 * einsum('kldc,adij,bckl->ijab', l2, t2, t2,
                                      optimize=['einsum_path', (0, 2), (0, 1)])

    #	  1.0000 P(i,j)l2(k,l,d,c)*t2(a,d,l,j)*t2(b,c,k,i)
    contracted_intermediate = 1.0 * einsum('kldc,adlj,bcki->ijab', l2, t2, t2,
                                           optimize=['einsum_path', (0, 1),
                                                     (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate)

    #	 -0.5000 l2(k,l,d,c)*t2(a,d,k,l)*t2(b,c,i,j)
    tpdm[o, o, v, v] += -0.5 * einsum('kldc,adkl,bcij->ijab', l2, t2, t2,
                                      optimize=['einsum_path', (0, 1), (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)l1(k,c)*t1(a,j)*t1(b,k)*t1(c,i)
    contracted_intermediate = 1.0 * einsum('kc,aj,bk,ci->ijab', l1, t1, t1, t1,
                                           optimize=['einsum_path', (0, 2),
                                                     (1, 2), (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	  0.5000 P(i,j)*P(a,b)l2(k,l,d,c)*t1(a,j)*t1(b,l)*t2(d,c,k,i)
    contracted_intermediate = 0.5 * einsum('kldc,aj,bl,dcki->ijab', l2, t1, t1,
                                           t2, optimize=['einsum_path', (0, 3),
                                                         (1, 2), (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	  0.5000 P(i,j)*P(a,b)l2(k,l,d,c)*t1(a,j)*t1(d,i)*t2(b,c,k,l)
    contracted_intermediate = 0.5 * einsum('kldc,aj,di,bckl->ijab', l2, t1, t1,
                                           t2, optimize=['einsum_path', (0, 3),
                                                         (1, 2), (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	 -0.5000 l2(k,l,d,c)*t1(a,l)*t1(b,k)*t2(d,c,i,j)
    tpdm[o, o, v, v] += -0.5 * einsum('kldc,al,bk,dcij->ijab', l2, t1, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #	  1.0000 P(i,j)*P(a,b)l2(k,l,d,c)*t1(a,l)*t1(d,j)*t2(b,c,k,i)
    contracted_intermediate = 1.0 * einsum('kldc,al,dj,bcki->ijab', l2, t1, t1,
                                           t2, optimize=['einsum_path', (0, 1),
                                                         (0, 2), (0, 1)])
    tpdm[o, o, v, v] += 1.00000 * contracted_intermediate + -1.00000 * einsum(
        'ijab->jiab', contracted_intermediate) + -1.00000 * einsum('ijab->ijba',
                                                                   contracted_intermediate) + 1.00000 * einsum(
        'ijab->jiba', contracted_intermediate)

    #	 -0.5000 l2(k,l,d,c)*t1(d,j)*t1(c,i)*t2(a,b,k,l)
    tpdm[o, o, v, v] += -0.5 * einsum('kldc,dj,ci,abkl->ijab', l2, t1, t1, t2,
                                      optimize=['einsum_path', (0, 1), (0, 2),
                                                (0, 1)])

    #	  1.0000 l2(k,l,d,c)*t1(a,l)*t1(b,k)*t1(d,j)*t1(c,i)
    tpdm[o, o, v, v] += 1.0 * einsum('kldc,al,bk,dj,ci->ijab', l2, t1, t1, t1,
                                     t1,
                                     optimize=['einsum_path', (0, 1), (0, 3),
                                               (0, 2), (0, 1)])

    #    D2(a,b,i,j):

    #	  1.0000 l2(i,j,a,b)
    tpdm[v, v, o, o] += 1.0 * einsum('ijab->abij', l2)

    #    D2(i,a,j,b):

    #	  1.0000 d1(i,j)*l1(i,a)*t1(b,i)
    tpdm[o, v, o, v] += 1.0 * einsum('ij,ia,bi->iajb', d1[o, o], l1, t1,
                                     optimize=['einsum_path', (0, 1, 2)])

    #	 -1.0000 l1(j,a)*t1(b,i)
    tpdm[o, v, o, v] += -1.0 * einsum('ja,bi->iajb', l1, t1)

    #	  0.5000 d1(i,j)*l2(i,j,a,c)*t2(b,c,i,j)
    tpdm[o, v, o, v] += 0.5 * einsum('ij,ijac,bcij->iajb', d1[o, o], l2, t2,
                                     optimize=['einsum_path', (0, 1), (0, 1)])

    #	 -1.0000 l2(k,j,a,c)*t2(b,c,k,i)
    tpdm[o, v, o, v] += -1.0 * einsum('kjac,bcki->iajb', l2, t2)

    #	 -1.0000 l2(k,j,a,c)*t1(b,k)*t1(c,i)
    tpdm[o, v, o, v] += -1.0 * einsum('kjac,bk,ci->iajb', l2, t1, t1,
                                      optimize=['einsum_path', (0, 1), (0, 1)])
    return tpdm


def main():
    import pyscf
    import openfermion as of
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from openfermionpyscf import run_pyscf
    from pyscf.cc.addons import spatial2spin
    import numpy as np

    basis = 'sto-3g'
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
    norbs = oei.shape[0]
    nsvirt = 2 * (norbs - nocc)
    nsocc = 2 * nocc
    assert np.allclose(np.transpose(mycc.t2, [1, 0, 3, 2]), mycc.t2)

    soei, stei = spinorb_from_spatial(oei, tei)
    astei = np.einsum('ijkl', stei) - np.einsum('ijlk', stei)
    gtei = astei.transpose(0, 1, 3, 2)


    d1hf = np.diag([1.] * nsocc + [0.] * nsvirt)
    opdm = d1hf
    tpdm_wedge = 2 * of.wedge(opdm, opdm, (1, 1), (1, 1))
    rdm_energy = np.einsum('ij,ij', soei, opdm) + 0.25 * np.einsum('ijkl,ijkl', tpdm_wedge, astei)
    # print(rdm_energy + molecule.nuclear_repulsion, molecule.hf_energy)

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
    print("HF energies")
    print(hf_energy_test, hf_energy)
    assert np.isclose(hf_energy + molecule.nuclear_repulsion, molecule.hf_energy)


    g = gtei
    nsvirt = 2 * (norbs - nocc)
    nsocc = 2 * nocc
    t1z, t2z = np.zeros((nsvirt, nsocc)), np.zeros((nsvirt, nsvirt, nsocc, nsocc))
    t1f, t2f, l1f, l2f = kernel(t1z, t2z, np.zeros_like(t1z.transpose(1, 0)), np.zeros_like(t2z.transpose(2, 3, 0, 1)), fock, g, o, v, e_ai, e_abij)
    print("Final Correlation Energy")
    print(ccsd_energy(t1f, t2f, fock, g, o, v) - hf_energy)
    print("Lagrangian Energy - HF")
    print(lagrangian_energy(t1f, t2f, l1f, l2f, fock, g, o, v) - hf_energy)

    d1hf = np.diag([1.] * nsocc + [0.] * nsvirt)
    opdm = ccsd_d1(t1f, t2f, l1f, l2f, d1hf, o, v)
    tpdm = ccsd_d2(t1f, t2f, l1f, l2f, d1hf, o, v)
    tpdm = tpdm.transpose(0, 1, 3, 2) # openfermion ordering

    rdm_energy = np.einsum('ij,ij', soei, opdm) + 0.25 * np.einsum('ijkl,ijkl', tpdm, astei)
    print("Correlation Energy from RDMs")
    print(rdm_energy - hf_energy)




if __name__ == "__main__":
    main()



