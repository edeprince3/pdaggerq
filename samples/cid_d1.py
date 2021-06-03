import sys
sys.path.insert(0, './..')

# elements of the one-electron reduced density matrix 
# at the CID level of theory:
# D(pq) = <0|(l0 + l2) p*q (r0 + t2)|0> 

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("vacuum")
ahat.set_print_level(0)

print('')
print('    D1(m,n) = <0|(l0 + l2) e(m,n) (r0 + r2)|0>')
print('')

ahat.set_left_operators(['l0','l2'])
ahat.set_right_operators(['r0','r2'])

ahat.add_operator_product(1.0,['e1(m,n)'])

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(e,f) = <0|(l0 + l2) e(e,f) (r0 + r2)|0>')
print('')

ahat.add_operator_product(1.0,['e1(e,f)'])
        
ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(m,e) = <0|(l0 + l2) e(m,e) (r0 + r2)|0>')
print('')

ahat.add_operator_product(1.0,['e1(m,e)'])
        
ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(e,m) = <0|(l0 + l2) e(e,m) (r0 + r2)|0>')
print('')

ahat.add_operator_product(1.0,['e1(e,m)'])

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

