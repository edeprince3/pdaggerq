# elements of the sigma vector using only the one-electron part hamiltonian

import pdaggerq

pq = pdaggerq.pq_helper("fermi")


print('')
print('    sigma_r(0) = <0| e(-T) h e(T) (r0 + r1 + r2)|0>')
print('')

# set right and left-hand operators
pq.set_left_operators([['1']])
pq.set_right_operators([['r0'],['r1'],['r2']])
pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.strings()
for my_term in terms:
    print(my_term)

pq.clear()

print('')
print('    sigma_r(ai) = <0|i*a e(-T) h e(T) (r0 + r1 + r2)|0>')
print('')

# set right and left-hand operators
pq.set_left_operators([['e1(i,a)']])
pq.set_right_operators([['r0'],['r1'],['r2']])
pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.strings()
for my_term in terms:
    print(my_term)

pq.clear()

print('')
print('    sigma_r(abij) = <0|i*j*ba e(-T) h e(T) (r0 + r1 + r2)|0>')
print('')

# set right and left-hand operators
pq.set_left_operators([['e2(i,j,b,a)']])
pq.set_right_operators([['r0'],['r1'],['r2']])
pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.strings()
for my_term in terms:
    print(my_term)

pq.clear()

print('')
print('    sigma_l(0) = <0|(l0 + l1 + l2) e(-T) h e(T) |0>')
print('')

# set right and left-hand operators
pq.set_left_operators([['l0'],['l1'],['l2']])
pq.set_right_operators([['1']])
pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.strings()
for my_term in terms:
    print(my_term)

pq.clear()

print('')
print('    sigma_l(ai) = <0|(l0 + l1 + l2) e(-T) h e(T) a*i|0>')
print('')

# set right and left-hand operators
pq.set_left_operators([['l0'],['l1'],['l2']])
pq.set_right_operators([['e1(a,i)']])
pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.strings()
for my_term in terms:
    print(my_term)

pq.clear()


print('')
print('    sigma_l(abij) = <0|(l0 + l1 + l2) e(-T) h e(T) a*b*ji|0>')
print('')

# set right and left-hand operators
pq.set_left_operators([['l0'],['l1'],['l2']])
pq.set_right_operators([['e2(a,b,j,i)']])
pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.strings()
for my_term in terms:
    print(my_term)

pq.clear()

