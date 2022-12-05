
import pdaggerq

pq = pdaggerq.pq_helper("fermi")

# set right and left-hand operators
pq.set_right_operators_type('DEA')
pq.set_left_operators([['a(b)', 'a(a)']])
pq.set_right_operators([['r2'], ['r3']])

print('')
print('    sigma(e) = <0|e e(-T) H e(T) (r0 + r1 + r2)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()
# grab list of fully-contracted strings, then print
doubles_residual_terms = pq.fully_contracted_strings()
for my_term in doubles_residual_terms:
    print(my_term)
pq.clear()

# set right and left-hand operators
pq.set_right_operators_type('DEA')
pq.set_left_operators([['a*(i)', 'a(c)', 'a(b)', 'a(a)']])
pq.set_right_operators([['r2'], ['r3']])

print('')
print('    sigma(e) = <0|e e(-T) H e(T) (r0 + r1 + r2)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()
# grab list of fully-contracted strings, then print
triples_residual_terms = pq.fully_contracted_strings()
for my_term in triples_residual_terms:
    print(my_term)
pq.clear()

