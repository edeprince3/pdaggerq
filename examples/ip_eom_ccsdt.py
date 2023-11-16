
import pdaggerq

pq = pdaggerq.pq_helper("fermi")

# set right and left-hand operators
pq.set_right_operators_type('IP')
pq.set_left_operators([['a*(i)']])
pq.set_right_operators([['r0'],['r1'],['r2'],['r3']])

print('')
print('    sigma(i) = <0|i* e(-T) H e(T) (r0 + r1 + r2)|0>')
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
pq.set_left_operators([['a*(i)','a*(j)','a(a)']])
pq.set_right_operators([['r0'],['r1'],['r2'],['r3']])

print('')
print('    sigma(a,i,j) = <0|i*j*a e(-T) H e(T) (r0 + r1 + r2)|0>')
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
pq.set_left_operators([['a*(i)','a*(j)','a*(k)','a(b)','a(a)']])
pq.set_right_operators([['r0'],['r1'],['r2'],['r3']])

print('')
print('    sigma(a,b,i,j,k) = <0|i*j*k*ba e(-T) H e(T) (r0 + r1 + r2)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3'])

pq.simplify()
# grab list of fully-contracted strings, then print
doubles_residual_terms = pq.fully_contracted_strings()
for my_term in doubles_residual_terms:
    print(my_term)
pq.clear()

