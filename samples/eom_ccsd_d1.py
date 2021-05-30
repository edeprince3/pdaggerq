import sys
sys.path.insert(0, './..')

# elements of the one-electron reduced density matrix 
# at the EOM-CCSD level of theory: D(pq) = <0|(l0 + l1 + l2) e(-T) p*q e(T) (r0 + r1 + r2)|0> 

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("vacuum")
ahat.set_print_level(0)

# set right and left-hand operators
ahat.set_left_operators(['l0','l1','l2'])
ahat.set_right_operators(['r0','r1','r2'])

print('')
print('    D1(m,n) = <0|(l0 + l1 + l2) e(-T) e(m,n) e(T) (r0 + r1 + r2)|0>')
print('')

ahat.add_st_operator(1.0,['e1(m,n)'],['t1','t2'])

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(e,f) = <0|(l0 + l1 + l2) e(-T) e(e,f) e(T) (r0 + r1 + r2)|0>')
print('')

ahat.add_st_operator(1.0,['e1(e,f)'],['t1','t2'])

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(m,e) = <0|(l0 + l1 + l2) e(-T) e(m,e) e(T) (r0 + r1 + r2)|0>')
print('')

ahat.add_st_operator(1.0,['e1(m,e)'],['t1','t2'])

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

print('')
print('    D1(e,m) = <0|(l0 + l1 + l2) e(-T) e(e,m) e(T) (r0 + r1 + r2)|0>')
print('')

ahat.add_st_operator(1.0,['e1(e,m)'],['t1','t2'])

ahat.simplify()
ahat.print_fully_contracted()
ahat.clear()

