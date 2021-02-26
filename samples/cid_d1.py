import sys
sys.path.insert(0, './..')

# elements of the one-electron reduced density matrix 
# at the CID level of theory:
# D(pq) = <0|(l0 + l2) p*q (r0 + t2)|0> 

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("vacuum")
ahat.set_print_level(0)

left_hand_operators  = ['l0','l2(i1,i2,a1,a2)']
right_hand_operators = ['r0','r2(a3,a4,i3,i4)']

print('')
print('    D1(i,j) = <0|(l0 + l2) e(i,j) (r0 + r2)|0>')
print('')

Dop = 'e1(i,j)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])
        
ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(a,b) = <0|(l0 + l2) e(a,b) (r0 + r2)|0>')
print('')

Dop = 'e1(a,b)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])
        
ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(i,a) = <0|(l0 + l2) e(i,a) (r0 + r2)|0>')
print('')

Dop = 'e1(i,a)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])
        
ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(a,i) = <0|(l0 + l2) e(e,m) (r0 + r2)|0>')
print('')

Dop = 'e1(a,i)'
for L in left_hand_operators:
    for R in right_hand_operators:

        # L e R
        ahat.add_operator_product(1.0,[L,Dop,R])

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

