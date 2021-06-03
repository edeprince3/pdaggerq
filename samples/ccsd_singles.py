
import sys
sys.path.insert(0, './..')

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("singles")
ahat.set_print_level(0)

print('')
print('    < 0 | m* e e(-T) H e(T) | 0> :')
print('')

# one-electron part: need only up to double commutators
# h
ahat.add_operator_product(1.0,['f'])

# [h, T1]
ahat.add_commutator(1.0,['f'],['t1'])

# [h, T2]
ahat.add_commutator(1.0,['f'],['t2'])

# [[h, T1], T1]]
ahat.add_double_commutator(0.5, ['f'],['t1'],['t1'])

# [[h, T1], T2]] + [[h, T2], T1]]
ahat.add_double_commutator( 1.0, ['f'],['t1'],['t2'])

# [[h, T2, T2]]
ahat.add_double_commutator( 0.5, ['f'],['t2'],['t2'])

# two-electron part: need up to quadruple commutators

# g
ahat.add_operator_product(1.0,['v'])

# [g, T1]
ahat.add_commutator(1.0,['v'],['t1'])

# [g, T2]
ahat.add_commutator(1.0,['v'],['t2'])

# [[g, T1], T1]]
ahat.add_double_commutator(0.5, ['v'],['t1'],['t1'])

# [[g, T1], T2]] + [[g, T2], T1]]
ahat.add_double_commutator( 1.0, ['v'],['t1'],['t2'])

# [[g, T2, T2]]
ahat.add_double_commutator( 0.5, ['v'],['t2'],['t2'])

# triple commutators

# [[[g, T1, T1], T1]
ahat.add_triple_commutator( 1.0/6.0, ['v'],['t1'],['t1'],['t1'])

# [[[g, T1, T1], T2] + [[[g, T1, T2], T1] + [[[g, T2, T1], T1]
ahat.add_triple_commutator( 1.0/2.0, ['v'],['t1'],['t1'],['t2'])

# [[[[g, T1], T1], T1], T1]
ahat.add_quadruple_commutator( 1.0/24.0, ['v'],['t1'],['t1'],['t1'],['t1'])


#ahat.add_st_operator(1.0,['f'],['t1','t2'])
#ahat.add_st_operator(1.0,['v'],['t1','t2'])

ahat.simplify()

ahat.print_fully_contracted()

#ahat.print_one_body()

#ahat.print_two_body()

#ahat.print_one_body()

#ahat.print_two_body()

ahat.clear()

