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
ahat.set_left_operators(['1','l1','l2'])
ahat.add_st_operator(1.0,['e1(m,n)'],['t1','t2'])
ahat.simplify()

# grab list of fully-contracted strings, then print
d1_terms = ahat.fully_contracted_strings()
for my_term in d1_terms:
    print(my_term)

ahat.clear()

print('')
print('    D1(e,f):')
print('')

ahat.set_left_operators(['1','l1','l2'])
ahat.add_st_operator(1.0,['e1(e,f)'],['t1','t2'])
ahat.simplify()

# grab list of fully-contracted strings, then print
d1_terms = ahat.fully_contracted_strings()
for my_term in d1_terms:
    print(my_term)

ahat.clear()
print('')
print('    D1(e,m):')
print('')

ahat.set_left_operators(['1','l1','l2'])
ahat.add_st_operator(1.0,['e1(e,m)'],['t1','t2'])
ahat.simplify()

# grab list of fully-contracted strings, then print
d1_terms = ahat.fully_contracted_strings()
for my_term in d1_terms:
    print(my_term)

ahat.clear()

print('')
print('    D1(m,e):')
print('')

ahat.set_left_operators(['1','l1','l2'])
ahat.add_st_operator(1.0,['e1(m,e)'],['t1','t2'])
ahat.simplify()

# grab list of fully-contracted strings, then print
d1_terms = ahat.fully_contracted_strings()
for my_term in d1_terms:
    print(my_term)

ahat.clear()

