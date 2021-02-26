import sys
sys.path.insert(0, './..')

# elements of the one-electron reduced density matrix 
# at the CID level of theory, using intermediate normalizatoin: 
# D(pq) = <0|(l0 + l2) p*q*rs (r0 + t2)|0> 

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("vacuum")
ahat.set_print_level(0)

left_hand_operators  = ['l0','l2(i1,i2,a1,a2)']
right_hand_operators = ['r0','r2(a3,a4,i3,i4)']

print('')
print('    D2(i,j,k,l) = <0|(l0 + l2) e2(i,j,l,k) (r0 + r2)|0>')
print('')

Dop = 'e2(i,j,l,k)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])
        
ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(a,b,c,d) = <0|(l0 + l2) e(a,b,d,c) (r0 + r2)|0>')
print('')

Dop = 'e2(a,b,d,c)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])
        
ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D2(i,j,a,b) = <0|(l0 + l2) e(i,j,b,a) (r0 + r2)|0>')
print('')

Dop = 'e2(i,j,b,a)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])
        
ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(a,b,i,j) = <0|(l0 + l2) e(a,b,j,i) (r0 + r2)|0>')
print('')

Dop = 'e2(a,b,j,i)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(i,a,j,b) = <0|(l0 + l2) e(i,a,b,j) (r0 + r2)|0>')
print('')

Dop = 'e2(i,a,b,j)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(i,a,b,j) = <0|(l0 + l2) e(i,a,j,b) (r0 + r2)|0>')
print('')

Dop = 'e2(i,a,j,b)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

