# elements of the one-electron reduced density matrix 
# at the CID level of theory:
# D(pq) = <0|(l0 + l2) p*q (r0 + t2)|0> 

import pdaggerq

pq = pdaggerq.pq_helper("fermi")

print('')
print('    D1(m,n) = <0|(l0 + l2) e(m,n) (r0 + r2)|0>')
print('')

pq.set_left_operators([['l0'],['l2']])
pq.set_right_operators([['r0'],['r2']])

pq.add_operator_product(1.0,['e1(m,n)'])

pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.strings()
for my_term in terms:
    print(my_term)

pq.clear()

print('')
print('    D1(e,f) = <0|(l0 + l2) e(e,f) (r0 + r2)|0>')
print('')

pq.add_operator_product(1.0,['e1(e,f)'])
        
pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.strings()
for my_term in terms:
    print(my_term)

pq.clear()

print('')
print('    D1(m,e) = <0|(l0 + l2) e(m,e) (r0 + r2)|0>')
print('')

pq.add_operator_product(1.0,['e1(m,e)'])
        
pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.strings()
for my_term in terms:
    print(my_term)

pq.clear()

print('')
print('    D1(e,m) = <0|(l0 + l2) e(e,m) (r0 + r2)|0>')
print('')

pq.add_operator_product(1.0,['e1(e,m)'])

pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.strings()
for my_term in terms:
    print(my_term)

pq.clear()

