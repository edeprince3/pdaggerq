
import sys
sys.path.insert(0, './..')

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("doubles")
ahat.set_print_level(0)

print('')
print('    < 0 | m* n* f e e(-T) H e(T) | 0> :')
print('')

## one-electron part: need only up to double commutators
## h
#ahat.add_operator_product(1.0,['h'])
#
## [h, T1]
#ahat.add_commutator(1.0,['h'],['t1'])
#
## [h, T2]
#ahat.add_commutator(1.0,['h'],['t2'])
#
## [[h, T1], T1]]
#ahat.add_double_commutator(0.5, ['h'],['t1'],['t1'])
#
## [[h, T1], T2]] + [[h, T2], T1]]
#ahat.add_double_commutator( 1.0, ['h'],['t1'],['t2'])
#
## [[h, T2, T2]]
#ahat.add_double_commutator( 0.5, ['h'],['t2'],['t2'])
#
## two-electron part: need up to quadruple commutators
#
## g
#ahat.add_operator_product(1.0,['g'])
#
## [g, T1]
#ahat.add_commutator(1.0,['g'],['t1'])
#
## [g, T2]
#ahat.add_commutator(1.0,['g'],['t2'])
#
## [[g, T1], T1]]
#ahat.add_double_commutator(0.5, ['g'],['t1'],['t1'])
#
## [[g, T1], T2]] + [[g, T2], T1]]
#ahat.add_double_commutator( 1.0, ['g'],['t1'],['t2'])
#
## [[g, T2, T2]]
#ahat.add_double_commutator( 0.5, ['g'],['t2'],['t2'])
#
## triple commutators
#
## [[[g, T1, T1], T1]
#ahat.add_triple_commutator( 1.0/6.0, ['g'],['t1'],['t1'],['t1'])
#
## [[[g, T1, T1], T2] + [[[g, T1, T2], T1] + [[[g, T2, T1], T1]
#ahat.add_triple_commutator( 1.0/2.0, ['g'],['t1'],['t1'],['t2'])
#
## quadruple commutators: the only one that yields non-zero contributions to the doubles equations will be
#
## [[[[g, T1], T1], T1], T1]
#ahat.add_quadruple_commutator( 1.0/24.0, ['g'],['t1'],['t1'],['t1'],['t1'])

ahat.add_st_operator(1.0,['h'],['t1','t2'])
ahat.add_st_operator(1.0,['g'],['t1','t2'])

ahat.simplify()

# print fully-contracted strings
#ahat.print_fully_contracted()

# grab list of fully-contracted strings, then print
residual_terms = ahat.fully_contracted_strings()
for my_term in residual_terms:
    print(my_term)


ahat.clear()

