
import sys
sys.path.insert(0, './..')

import pdaggerq

pq = pdaggerq.pq_helper("fermi")

pq.set_bra("")
pq.set_print_level(0)

print('')
print('    < 0 | e(-T) H e(T) | 0> :')
print('')

# one-electron part: 
# h
pq.add_operator_product(1.0,['f'])

# [h, T1]
pq.add_commutator(1.0,['f'],['t1'])

# [h, T2]
pq.add_commutator(1.0,['f'],['t2'])

# two-electron part: 

# g
pq.add_operator_product(1.0,['v'])

# [g, T1]
pq.add_commutator(1.0,['v'],['t1'])

# [g, T2]
pq.add_commutator(1.0,['v'],['t2'])

# [[g, T1], T1]]
pq.add_double_commutator(0.5, ['v'],['t1'],['t1'])

#pq.add_st_operator(1.0,['f'],['t1','t2'])
#pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

pq.print_fully_contracted()

pq.clear()

