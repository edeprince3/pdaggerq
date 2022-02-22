
import pdaggerq

pq = pdaggerq.pq_helper("fermi")

# set right and left-hand operators
pq.set_left_operators([['e1(m,e)']])
pq.set_right_operators([['r0'],['r1'],['r2']])

print('')
print('    sigma(e,m) = <0|e1(m,e) e(-T) H e(T) (r0 + r1 + r2)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()
pq.print_fully_contracted()
pq.clear()

# set right and left-hand operators
pq.set_left_operators([['e2(m,n,f,e)']])
pq.set_right_operators([['r0'],['r1'],['r2']])

print('')
print('    sigma(e,f,m,n) = <0|e2(m,n,f,e) e(-T) H e(T) (r0 + r1 + r2)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()
pq.print_fully_contracted()
pq.clear()

