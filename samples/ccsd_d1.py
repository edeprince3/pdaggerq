import sys
sys.path.insert(0, './..')

# elements of the one-electron reduced density matrix 
# at the CCSD level of theory: D(pq) = <psi|(1+lambda) e(-T) p*q e(T) |psi> 

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("vacuum")
ahat.set_print_level(0)

print('')
print('    D1(m,n):')
print('')

# D(mn) = <psi|(1+l1 + l2) e(-T) e(m,n) e(T) |psi> 
# e(-T) Dop e(T) = e + [e,t1] + [e,t2] + 1/2 [[e,t1,],t1] + [[e,t1],t2] + 1/2 [[e,t2],t2] 

# first the bare part without lambda

Dop = 'e(m,n)'

# e
ahat.add_operator_product(1.0,[Dop])

# [h, T2]
ahat.add_commutator(1.0,[Dop,'t2(a,b,i,j)'])
# [h, T1]
ahat.add_commutator(1.0,[Dop,'t1(a,i)'])

# [[e, T2, T2]]
ahat.add_double_commutator( 0.5, [Dop,'t2(a,b,i,j)','t2(c,d,k,l)'])
# [[e, T1, T1]]
ahat.add_double_commutator( 0.5, [Dop,'t1(a,i)','t1(c,k)'])
# [[e, T1, T2]]
ahat.add_double_commutator( 1.0, [Dop,'t1(a,i)','t2(c,d,k,l)'])

# now, we must include lambda

# L1 e
ahat.add_operator_product(1.0,['l1(k,c)',Dop])

# L2 e
ahat.add_operator_product(1.0,['l2(k,l,c,d)',Dop])

# L1 [h, T2]
ahat.add_operator_product( 1.0,['l1(k,c)',Dop,'t2(a,b,i,j)'])
ahat.add_operator_product(-1.0,['l1(k,c)','t2(a,b,i,j)',Dop])

# L1 [h, T1]
ahat.add_operator_product( 1.0,['l1(k,c)',Dop,'t1(a,i)'])
ahat.add_operator_product(-1.0,['l1(k,c)','t1(a,i)',Dop])

# L2 [h, T2]
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',Dop,'t2(a,b,i,j)'])
ahat.add_operator_product(-1.0,['l2(k,l,c,d)','t2(a,b,i,j)',Dop])

# L2 [h, T1]
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',Dop,'t1(a,i)'])
ahat.add_operator_product(-1.0,['l2(k,l,c,d)','t1(a,i)',Dop])

# L2 [[e, T2], T2]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't2(a,b,i,j)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',C,B,A]) 

# L1 [[e, T2], T2]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't2(a,b,i,j)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',C,B,A]) 

# L2 [[e, T1], T2] + [[e, T2], T1]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',C,B,A]) 

# L1 [[e, T1], T2] + [[e, T2], T1]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0,['l1(k,c)',A,B,C]) 
ahat.add_operator_product(-1.0,['l1(k,c)',B,A,C]) 
ahat.add_operator_product(-1.0,['l1(k,c)',C,A,B]) 
ahat.add_operator_product( 1.0,['l1(k,c)',C,B,A]) 

# L2 [[e, T1], T1]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't1(a2,i2)'
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',C,B,A]) 

# L1 [[e, T2], T2]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't1(a2,i2)'
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',C,B,A]) 


ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(e,f):')
print('')

# D(ef) = <psi|(1+l1 + l2) e(-T) e(e,f) e(T) |psi> 
# e(-T) e(e,f) e(T) = e + [e,t1] + [e,t2] + 1/2 [[e,t1,],t1] + [[e,t1],t2] + 1/2 [[e,t2],t2] 

# first the bare part without lambda

Dop = 'e(e,f)'

# e
ahat.add_operator_product(1.0,[Dop])

# [h, T2]
ahat.add_commutator(1.0,[Dop,'t2(a,b,i,j)'])
# [h, T1]
ahat.add_commutator(1.0,[Dop,'t1(a,i)'])

# [[e, T2, T2]]
ahat.add_double_commutator( 0.5, [Dop,'t2(a,b,i,j)','t2(c,d,k,l)'])
# [[e, T1, T1]]
ahat.add_double_commutator( 0.5, [Dop,'t1(a,i)','t1(c,k)'])
# [[e, T1, T2]]
ahat.add_double_commutator( 1.0, [Dop,'t1(a,i)','t2(c,d,k,l)'])

# now, we must include lambda

# L1 e
ahat.add_operator_product(1.0,['l1(k,c)',Dop])

# L2 e
ahat.add_operator_product(1.0,['l2(k,l,c,d)',Dop])

# L1 [h, T2]
ahat.add_operator_product( 1.0,['l1(k,c)',Dop,'t2(a,b,i,j)'])
ahat.add_operator_product(-1.0,['l1(k,c)','t2(a,b,i,j)',Dop])

# L1 [h, T1]
ahat.add_operator_product( 1.0,['l1(k,c)',Dop,'t1(a,i)'])
ahat.add_operator_product(-1.0,['l1(k,c)','t1(a,i)',Dop])

# L2 [h, T2]
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',Dop,'t2(a,b,i,j)'])
ahat.add_operator_product(-1.0,['l2(k,l,c,d)','t2(a,b,i,j)',Dop])

# L2 [h, T1]
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',Dop,'t1(a,i)'])
ahat.add_operator_product(-1.0,['l2(k,l,c,d)','t1(a,i)',Dop])

# L2 [[e, T2], T2]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't2(a,b,i,j)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',C,B,A]) 

# L1 [[e, T2], T2]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't2(a,b,i,j)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',C,B,A]) 

# L2 [[e, T1], T2] + [[e, T2], T1]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',C,B,A]) 

# L1 [[e, T1], T2] + [[e, T2], T1]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0,['l1(k,c)',A,B,C]) 
ahat.add_operator_product(-1.0,['l1(k,c)',B,A,C]) 
ahat.add_operator_product(-1.0,['l1(k,c)',C,A,B]) 
ahat.add_operator_product( 1.0,['l1(k,c)',C,B,A]) 

# L2 [[e, T1], T1]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't1(a2,i2)'
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',C,B,A]) 

# L1 [[e, T2], T2]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't1(a2,i2)'
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',C,B,A]) 


ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()


print('')
print('    D1(e,m):')
print('')

# first the bare part without lambda

Dop = 'e(e,m)'

# e
ahat.add_operator_product(1.0,[Dop])

# [h, T2]
ahat.add_commutator(1.0,[Dop,'t2(a,b,i,j)'])
# [h, T1]
ahat.add_commutator(1.0,[Dop,'t1(a,i)'])

# [[e, T2, T2]]
ahat.add_double_commutator( 0.5, [Dop,'t2(a,b,i,j)','t2(c,d,k,l)'])
# [[e, T1, T1]]
ahat.add_double_commutator( 0.5, [Dop,'t1(a,i)','t1(c,k)'])
# [[e, T1, T2]]
ahat.add_double_commutator( 1.0, [Dop,'t1(a,i)','t2(c,d,k,l)'])

# now, we must include lambda

# L1 e
ahat.add_operator_product(1.0,['l1(k,c)',Dop])

# L2 e
ahat.add_operator_product(1.0,['l2(k,l,c,d)',Dop])

# L1 [h, T2]
ahat.add_operator_product( 1.0,['l1(k,c)',Dop,'t2(a,b,i,j)'])
ahat.add_operator_product(-1.0,['l1(k,c)','t2(a,b,i,j)',Dop])

# L1 [h, T1]
ahat.add_operator_product( 1.0,['l1(k,c)',Dop,'t1(a,i)'])
ahat.add_operator_product(-1.0,['l1(k,c)','t1(a,i)',Dop])

# L2 [h, T2]
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',Dop,'t2(a,b,i,j)'])
ahat.add_operator_product(-1.0,['l2(k,l,c,d)','t2(a,b,i,j)',Dop])

# L2 [h, T1]
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',Dop,'t1(a,i)'])
ahat.add_operator_product(-1.0,['l2(k,l,c,d)','t1(a,i)',Dop])

# L2 [[e, T2], T2]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't2(a,b,i,j)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',C,B,A]) 

# L1 [[e, T2], T2]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't2(a,b,i,j)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',C,B,A]) 

# L2 [[e, T1], T2] + [[e, T2], T1]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',C,B,A]) 

# L1 [[e, T1], T2] + [[e, T2], T1]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0,['l1(k,c)',A,B,C]) 
ahat.add_operator_product(-1.0,['l1(k,c)',B,A,C]) 
ahat.add_operator_product(-1.0,['l1(k,c)',C,A,B]) 
ahat.add_operator_product( 1.0,['l1(k,c)',C,B,A]) 

# L2 [[e, T1], T1]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't1(a2,i2)'
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',C,B,A]) 

# L1 [[e, T2], T2]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't1(a2,i2)'
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',C,B,A]) 


ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()


print('')
print('    D1(m,e):')
print('')

# first the bare part without lambda

Dop = 'e(m,e)'

# e
ahat.add_operator_product(1.0,[Dop])

# [h, T2]
ahat.add_commutator(1.0,[Dop,'t2(a,b,i,j)'])
# [h, T1]
ahat.add_commutator(1.0,[Dop,'t1(a,i)'])

# [[e, T2, T2]]
ahat.add_double_commutator( 0.5, [Dop,'t2(a,b,i,j)','t2(c,d,k,l)'])
# [[e, T1, T1]]
ahat.add_double_commutator( 0.5, [Dop,'t1(a,i)','t1(c,k)'])
# [[e, T1, T2]]
ahat.add_double_commutator( 1.0, [Dop,'t1(a,i)','t2(c,d,k,l)'])

# now, we must include lambda

# L1 e
ahat.add_operator_product(1.0,['l1(k,c)',Dop])

# L2 e
ahat.add_operator_product(1.0,['l2(k,l,c,d)',Dop])

# L1 [h, T2]
ahat.add_operator_product( 1.0,['l1(k,c)',Dop,'t2(a,b,i,j)'])
ahat.add_operator_product(-1.0,['l1(k,c)','t2(a,b,i,j)',Dop])

# L1 [h, T1]
ahat.add_operator_product( 1.0,['l1(k,c)',Dop,'t1(a,i)'])
ahat.add_operator_product(-1.0,['l1(k,c)','t1(a,i)',Dop])

# L2 [h, T2]
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',Dop,'t2(a,b,i,j)'])
ahat.add_operator_product(-1.0,['l2(k,l,c,d)','t2(a,b,i,j)',Dop])

# L2 [h, T1]
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',Dop,'t1(a,i)'])
ahat.add_operator_product(-1.0,['l2(k,l,c,d)','t1(a,i)',Dop])

# L2 [[e, T2], T2]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't2(a,b,i,j)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',C,B,A]) 

# L1 [[e, T2], T2]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't2(a,b,i,j)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',C,B,A]) 

# L2 [[e, T1], T2] + [[e, T2], T1]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0,['l2(k,l,c,d)',C,B,A]) 

# L1 [[e, T1], T2] + [[e, T2], T1]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't2(a2,a3,i2,i3)'
ahat.add_operator_product( 1.0,['l1(k,c)',A,B,C]) 
ahat.add_operator_product(-1.0,['l1(k,c)',B,A,C]) 
ahat.add_operator_product(-1.0,['l1(k,c)',C,A,B]) 
ahat.add_operator_product( 1.0,['l1(k,c)',C,B,A]) 

# L2 [[e, T1], T1]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't1(a2,i2)'
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l2(k,l,c,d)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l2(k,l,c,d)',C,B,A]) 

# L1 [[e, T2], T2]
#[A,B],C] = [A,B]C - C[A,B] = ABC - BAC - CAB + CBA
A = Dop
B = 't1(a,i)'
C = 't1(a2,i2)'
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',A,B,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',B,A,C]) 
ahat.add_operator_product(-1.0/2.0,['l1(k,c)',C,A,B]) 
ahat.add_operator_product( 1.0/2.0,['l1(k,c)',C,B,A]) 


ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

