
import sys
sys.path.insert(0, './..')

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("doubles")
ahat.set_print_level(0)

print('')
print('    < 0 | m* n* f e e(-T) H e(T) | 0> :')
print('')

# one-electron part: need only up to double commutators
# h
ahat.add_operator_product(1.0,['h(p,q)'])

# [h, T1]
ahat.add_commutator(1.0,['h(p,q)','t1(a,i)'])

# [h, T2]
ahat.add_commutator(1.0,['h(p,q)','t2(a,b,i,j)'])

# [[h, T1], T1]]
ahat.add_double_commutator(0.5, ['h(p,q)','t1(a,i)','t1(b,j)'])

# [[h, T1], T2]] + [[h, T2], T1]]
ahat.add_double_commutator( 1.0, ['h(p,q)','t1(a,i)','t2(c,d,k,l)'])

# [[h, T2, T2]]
ahat.add_double_commutator( 0.5, ['h(p,q)','t2(a,b,i,j)','t2(c,d,k,l)'])

# two-electron part: need up to quadruple commutators

# g
ahat.add_operator_product(1.0,['g(p,r,q,s)'])

# [g, T1]
ahat.add_commutator(1.0,['g(p,r,q,s)','t1(a,i)'])

# [g, T2]
ahat.add_commutator(1.0,['g(p,r,q,s)','t2(a,b,i,j)'])

# [[g, T1], T1]]
ahat.add_double_commutator(0.5, ['g(p,r,q,s)','t1(a,i)','t1(b,j)'])

# [[g, T1], T2]] + [[g, T2], T1]]
ahat.add_double_commutator( 1.0, ['g(p,r,q,s)','t1(a,i)','t2(c,d,k,l)'])

# [[g, T2, T2]]
ahat.add_double_commutator( 0.5, ['g(p,r,q,s)','t2(a,b,i,j)','t2(c,d,k,l)'])

# triple commutators

# [[[g, T1, T1], T1]
ahat.add_triple_commutator( 1.0/6.0, ['g(p,r,q,s)','t1(a,i)','t1(b,j)','t1(c,k)'])

# [[[g, T1, T1], T2] + [[[g, T1, T2], T1] + [[[g, T2, T1], T1]
ahat.add_triple_commutator( 1.0/2.0, ['g(p,r,q,s)','t1(a,i)','t1(b,j)','t2(c,d,k,l)'])

# [[[g, T1, T2], T2] + [[[g, T2, T1], T2] + [[[g, T2, T2], T1]
#ahat.add_triple_commutator( 1.0/2.0, ['g(p,r,q,s)','t1(a,i)','t2(c,d,k,l)','t2(e,f,m,n)'])

# [[[g, T2, T2], T2]
#ahat.add_triple_commutator( 1.0/6.0, ['g(p,r,q,s)','t2(a,b,i,j)','t2(c,d,k,l)','t2(e,f,m,n)'])

# quadruple commutators: the only one that yields non-zero contributions to the doubles equations will be

# [[[[g, T1], T1], T1], T1]
ahat.add_quadruple_commutator( 1.0/24.0, ['g(p,r,q,s)','t1(a,i)','t1(b,j)','t1(c,k)','t1(d,l)'])

# [[[[g, T1], T1], T1], T2]
#ahat.add_quadruple_commutator( 1.0/24.0, ['g(p,r,q,s)','t1(a,i)','t1(b,j)','t1(c,k)','t2(e,f,m,n)'])

ahat.simplify()

ahat.print_fully_contracted()

#ahat.print_one_body()

#ahat.print_two_body()

#ahat.print_one_body()

#ahat.print_two_body()

ahat.clear()

