
import pdaggerq

pq = pdaggerq.pq_helper("fermi")

# set right and left-hand operators
pq.set_right_operators_type('EA')
pq.set_left_operators([['a(e)']])
pq.set_right_operators([['r0'],['r1'],['r2']])

print('')
print('    sigma(e) = <0|e e(-T) H e(T) (r0 + r1 + r2)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()
#pq.print_fully_contracted()
# grab list of fully-contracted strings, then print
singles_residual_terms = pq.fully_contracted_strings()
for my_term in singles_residual_terms:
    print(my_term)
pq.clear()

# set right and left-hand operators
pq.set_left_operators_type('EA')
pq.set_left_operators([['a*(m)','a(f)','a(e)']])
pq.set_right_operators([['r0'],['r1'],['r2']])

print('')
print('    sigma(e,f,m) = <0|m*f e(-T) H e(T) (r0 + r1 + r2)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()
#pq.print_fully_contracted()
# grab list of fully-contracted strings, then print
doubles_residual_terms = pq.fully_contracted_strings()
for my_term in doubles_residual_terms:
    print(my_term)
pq.clear()

