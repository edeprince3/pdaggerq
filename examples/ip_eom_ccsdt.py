
import pdaggerq

pq = pdaggerq.pq_helper("fermi")

# set right and left-hand operators
pq.set_right_operators_type('IP')
pq.set_left_operators([['a*(m)']])
pq.set_right_operators([['r0'],['r1'],['r2'],['r3']])

print('')
print('    sigma(m) = <0|m* e(-T) H e(T) (r0 + r1 + r2)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3'])

pq.simplify()
# grab list of fully-contracted strings, then print
singles_residual_terms = pq.fully_contracted_strings()
for my_term in singles_residual_terms:
    print(my_term)
pq.clear()

# set right and left-hand operators
pq.set_left_operators_type('IP')
pq.set_left_operators([['a*(m)','a*(n)','a(e)']])
pq.set_right_operators([['r0'],['r1'],['r2'],['r3']])

print('')
print('    sigma(e,m,n) = <0|m*n*e e(-T) H e(T) (r0 + r1 + r2)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3'])

pq.simplify()
# grab list of fully-contracted strings, then print
doubles_residual_terms = pq.fully_contracted_strings()
for my_term in doubles_residual_terms:
    print(my_term)
pq.clear()

# set right and left-hand operators
pq.set_left_operators_type('IP')
pq.set_left_operators([['a*(m)','a*(n)','a*(o)','a(f)','a(e)']])
pq.set_right_operators([['r0'],['r1'],['r2'],['r3']])

print('')
print('    sigma(e,f,m,n,o) = <0|m*n*o*fe e(-T) H e(T) (r0 + r1 + r2)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3'])

pq.simplify()
# grab list of fully-contracted strings, then print
doubles_residual_terms = pq.fully_contracted_strings()
for my_term in doubles_residual_terms:
    print(my_term)
pq.clear()

