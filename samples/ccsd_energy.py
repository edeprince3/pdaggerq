
import sys
sys.path.insert(0, './..')

import pdaggerq

ahat = pdaggerq.ahat_helper("fermi")

ahat.set_bra("")
ahat.set_print_level(0)

print('')
print('    < 0 | e(-T) H e(T) | 0> :')
print('')

# one-electron part: 
# h
ahat.add_operator_product(1.0,['f'])

# [h, T1]
ahat.add_commutator(1.0,['f'],['t1'])

# [h, T2]
ahat.add_commutator(1.0,['f'],['t2'])

# two-electron part: 

# g
ahat.add_operator_product(1.0,['v'])

# [g, T1]
ahat.add_commutator(1.0,['v'],['t1'])

# [g, T2]
ahat.add_commutator(1.0,['v'],['t2'])

# [[g, T1], T1]]
ahat.add_double_commutator(0.5, ['v'],['t1'],['t1'])

#ahat.add_st_operator(1.0,['f'],['t1','t2'])
#ahat.add_st_operator(1.0,['v'],['t1','t2'])

ahat.simplify()

ahat.print_fully_contracted()

ahat.clear()

