# elements of the one-electron reduced density matrix 
# at the CCSD level of theory: D(pq) = <psi|(1+lambda) e(-T) p*q e(T) |psi> 

import pdaggerq

pq = pdaggerq.pq_helper("fermi")

pq.set_bra("vacuum")
pq.set_print_level(0)

print('')
print('    D1(m,n):')
print('')

# D(mn) = <psi|(1+l1 + l2) e(-T) e(m,n) e(T) |psi> 
pq.set_left_operators(['1','l1','l2'])
pq.add_st_operator(1.0,['e1(m,n)'],['t1','t2'])
pq.simplify()

# grab list of fully-contracted strings, then print
d1_terms = pq.fully_contracted_strings()
for my_term in d1_terms:
    print(my_term)

pq.clear()

print('')
print('    D1(e,f):')
print('')

pq.set_left_operators(['1','l1','l2'])
pq.add_st_operator(1.0,['e1(e,f)'],['t1','t2'])
pq.simplify()

# grab list of fully-contracted strings, then print
d1_terms = pq.fully_contracted_strings()
for my_term in d1_terms:
    print(my_term)

pq.clear()
print('')
print('    D1(e,m):')
print('')

pq.set_left_operators(['1','l1','l2'])
pq.add_st_operator(1.0,['e1(e,m)'],['t1','t2'])
pq.simplify()

# grab list of fully-contracted strings, then print
d1_terms = pq.fully_contracted_strings()
for my_term in d1_terms:
    print(my_term)

pq.clear()

print('')
print('    D1(m,e):')
print('')

pq.set_left_operators(['1','l1','l2'])
pq.add_st_operator(1.0,['e1(m,e)'],['t1','t2'])
pq.simplify()

# grab list of fully-contracted strings, then print
d1_terms = pq.fully_contracted_strings()
for my_term in d1_terms:
    print(my_term)

pq.clear()

