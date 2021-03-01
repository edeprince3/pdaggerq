import sys
sys.path.insert(0, './..')

# elements of the one-electron reduced density matrix 
# at the CID level of theory
# D2(p,q,r,s) = <0|(l0 + l2) p*q*sr (r0 + t2)|0> 

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("vacuum")
ahat.set_print_level(0)

left_hand_operators  = ['l0','l2(i,j,a,b)']
right_hand_operators = ['r0','r2(c,d,k,l)']

print('')
print('    D2(i1,i2,i3,i4) = <0|(l0 + l2) e2(i1,i2,i4,i3) (r0 + r2)|0>')
print('')

Dop = 'e2(i1,i2,i4,i3)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])
        
ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(a1,a2,a3,a4) = <0|(l0 + l2) e(a1,a2,a4,a3) (r0 + r2)|0>')
print('')

Dop = 'e2(a1,a2,a4,a3)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])
        
ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D2(i1,i2,a1,a2) = <0|(l0 + l2) e(i1,i2,a2,a1) (r0 + r2)|0>')
print('')

Dop = 'e2(i1,i2,a2,a1)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])
        
ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(a1,a2,i1,i2) = <0|(l0 + l2) e(a1,a2,i2,i1) (r0 + r2)|0>')
print('')

Dop = 'e2(a1,a2,i2,i1)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(i1,a1,i2,a1) = <0|(l0 + l2) e(i1,a1,a2,i2) (r0 + r2)|0>')
print('')

Dop = 'e2(i1,a1,a2,i2)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(i1,a1,a2,i2) = <0|(l0 + l2) e(i1,a1,i2,a2) (r0 + r2)|0>')
print('')

Dop = 'e2(i1,a1,i2,a2)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

