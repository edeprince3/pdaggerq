import sys
sys.path.insert(0, './..')

# elements of the one-electron reduced density matrix 
# at the CCD level of theory: D(pq) = <psi|(1+lambda) e(-T) p*q e(T) |psi> 

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("vacuum")
ahat.set_print_level(0)

print('')
print('    D1(m,n):')
print('')

# D(mn) = <psi|(1+lambda) e(-T) e(m,n) e(T) |psi> 
# e(-T) e(m,n) e(T) = e + [e,t2] + 1/2 [[e,t2],t2] 

# first the bare part without lambda

# e
ahat.add_operator_product(1.0,['e(m,n)'])

# L2 [h, T2]
ahat.add_commutator(1.0,['e(m,n)','t2(a,b,i,j)'])

# L2 [[e, T2, T2]]
ahat.add_double_commutator( 0.5, ['e(m,n)','t2(a,b,i,j)','t2(c,d,k,l)'])

# now, we must include lambda

# L2 e
ahat.add_operator_product(1.0,['l2(k,l,c,d)','e(m,n)'])

# L2 [h, T2]
ahat.add_operator_product( 1.0,['l2(k,l,c,d)','e(m,n)','t2(a,b,i,j)'])
ahat.add_operator_product(-1.0,['l2(k,l,c,d)','t2(a,b,i,j)','e(m,n)'])

# L2 [[e, T2, T2]]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = 'e(m,n)'
B = 't2(a,b,i,j)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',C,B,A]) 

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(e,f):')
print('')

# D(ef) = <psi|(1+lambda) e(-T) e(e,f) e(T) |psi> 
# e(-T) e(m,n) e(T) = e + [e,t2] + 1/2 [[e,t2],t2] 

# first the bare part without lambda

# e
ahat.add_operator_product(1.0,['e(e,f)'])

# L2 [h, T2]
ahat.add_commutator(1.0,['e(e,f)','t2(a,b,i,j)'])

# L2 [[e, T2, T2]]
ahat.add_double_commutator( 0.5, ['e(e,f)','t2(a,b,i,j)','t2(c,d,k,l)'])

# now, we must include lambda

# L2 e
ahat.add_operator_product(1.0,['l2(k,l,c,d)','e(e,f)'])

# L2 [h, T2]
#ahat.add_operator_times_commutator(1.0,['l2(k,l,c,d)','e(m,n)','t2(a,b,i,j)'])
ahat.add_operator_product( 1.0,['l2(k,l,c,d)','e(e,f)','t2(a,b,i,j)'])
ahat.add_operator_product(-1.0,['l2(k,l,c,d)','t2(a,b,i,j)','e(e,f)'])

# L2 [[e, T2, T2]]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = 'e(e,f)'
B = 't2(a,b,i,j)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',C,B,A]) 

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()



