
import pdaggerq

pq = pdaggerq.pq_helper("fermi")


print('')
print('    < 0 | m* e e(-T) H e(T) | 0> :')
print('')

pq.set_left_operators([['e1(m,e)']])

# one-electron part: need only up to double commutators
# h
pq.add_operator_product(1.0,['f'])

# [h, T1]
pq.add_commutator(1.0,['f'],['t1'])

# [h, T2]
pq.add_commutator(1.0,['f'],['t2'])

# [[h, T1], T1]]
pq.add_double_commutator(0.5, ['f'],['t1'],['t1'])

# [[h, T1], T2]] + [[h, T2], T1]]
pq.add_double_commutator( 1.0, ['f'],['t1'],['t2'])

# [[h, T2, T2]]
pq.add_double_commutator( 0.5, ['f'],['t2'],['t2'])

# two-electron part: need up to quadruple commutators

# g
pq.add_operator_product(1.0,['v'])

# [g, T1]
pq.add_commutator(1.0,['v'],['t1'])

# [g, T2]
pq.add_commutator(1.0,['v'],['t2'])

# [[g, T1], T1]]
pq.add_double_commutator(0.5, ['v'],['t1'],['t1'])

# [[g, T1], T2]] + [[g, T2], T1]]
pq.add_double_commutator( 1.0, ['v'],['t1'],['t2'])

# [[g, T2, T2]]
pq.add_double_commutator( 0.5, ['v'],['t2'],['t2'])

# triple commutators

# [[[g, T1, T1], T1]
pq.add_triple_commutator( 1.0/6.0, ['v'],['t1'],['t1'],['t1'])

# [[[g, T1, T1], T2] + [[[g, T1, T2], T1] + [[[g, T2, T1], T1]
pq.add_triple_commutator( 1.0/2.0, ['v'],['t1'],['t1'],['t2'])

# [[[[g, T1], T1], T1], T1]
pq.add_quadruple_commutator( 1.0/24.0, ['v'],['t1'],['t1'],['t1'],['t1'])

pq.simplify()


# grab list of fully-contracted strings, then print
terms = pq.fully_contracted_strings()
for my_term in terms:
    print(my_term)


pq.clear()

