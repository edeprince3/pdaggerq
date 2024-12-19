
import pdaggerq

pq = pdaggerq.pq_helper("fermi")

# set right and left-hand operators
pq.set_right_operators_type('EA')
pq.set_left_operators([['a(a)']])
pq.set_right_operators([['r0'],['r1'],['r2'],['r3']])

print('')
print('    sigma(a) = <0|a e(-T) H e(T) (r0 + r1 + r2)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3'])

pq.simplify()
# grab list of fully-contracted strings, then print
singles_residual_terms = pq.strings()
for my_term in singles_residual_terms:
    print(my_term)
pq.clear()

# set right and left-hand operators
pq.set_left_operators_type('EA')
pq.set_left_operators([['a*(i)','a(b)','a(a)']])
pq.set_right_operators([['r0'],['r1'],['r2'],['r3']])

print('')
print('    sigma(a,b,i) = <0|i*b a e(-T) H e(T) (r0 + r1 + r2)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3'])

pq.simplify()
# grab list of fully-contracted strings, then print
doubles_residual_terms = pq.strings()
for my_term in doubles_residual_terms:
    print(my_term)
pq.clear()

# set right and left-hand operators
pq.set_left_operators_type('EA')
pq.set_left_operators([['a*(i)','a*(j)','a(c)','a(b)','a(a)']])
pq.set_right_operators([['r0'],['r1'],['r2'],['r3']])

print('')
print('    sigma(a,b,c,i,j) = <0|i*j*c b a e(-T) H e(T) (r0 + r1 + r2)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3'])

pq.simplify()
# grab list of fully-contracted strings, then print
triples_residual_terms = pq.strings()
for my_term in triples_residual_terms:
    print(my_term)
pq.clear()

