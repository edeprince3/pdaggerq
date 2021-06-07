# elements of the one-electron reduced density matrix 
# at the CID level of theory
# D2(p,q,r,s) = <0|(l0 + l2) p*q*sr (r0 + t2)|0> 

import pdaggerq

pq = pdaggerq.pq_helper("fermi")

pq.set_left_operators(['l0','l2'])
pq.set_right_operators(['r0','r2'])

print('')
print('    D2(i,j,k,l) = <0|(l0 + l2) e2(i,j,l,k) (r0 + r2)|0>')
print('')

pq.add_operator_product(1.0,['e2(i,j,k,l)'])
        
pq.simplify()
pq.print_fully_contracted()
pq.clear()

print('')
print('    D2(a,b,c,d) = <0|(l0 + l2) e(a,b,d,c) (r0 + r2)|0>')
print('')

pq.add_operator_product(1.0,['e2(a,b,d,c)'])
        
pq.simplify()
pq.print_fully_contracted()
pq.clear()

print('')
print('    D2(i,j,a,b) = <0|(l0 + l2) e(i,j,b,a) (r0 + r2)|0>')
print('')

pq.add_operator_product(1.0,['e2(i,j,b,a)'])
        
pq.simplify()
pq.print_fully_contracted()
pq.clear()

print('')
print('    D2(a,b,i,j) = <0|(l0 + l2) e(a,b,j,i) (r0 + r2)|0>')
print('')

pq.add_operator_product(1.0,['e2(a,b,j,i)'])

pq.simplify()
pq.print_fully_contracted()
pq.clear()

print('')
print('    D2(i,a,j,b) = <0|(l0 + l2) e(i,a,b,j) (r0 + r2)|0>')
print('')

pq.add_operator_product(1.0,['e2(i,a,b,j)'])

pq.simplify()
pq.print_fully_contracted()
pq.clear()

print('')
print('    D2(i,a,b,j) = <0|(l0 + l2) e(i,a,j,b) (r0 + r2)|0>')
print('')

pq.add_operator_product(1.0,['e2(i,a,j,b)'])

pq.simplify()
pq.print_fully_contracted()
pq.clear()

