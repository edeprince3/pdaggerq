
import pdaggerq

pq = pdaggerq.pq_helper("fermi")

# set right and left-hand operators
pq.set_left_operators([['e1(m,e)']])
pq.set_right_operators([['e1(a,i)']])

print('')
print('    H(m,e;i,a) = <0|e1(m,e) e(-T) H e(T) e1(a,i)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()
pq.print_fully_contracted()
pq.clear()

# set right and left-hand operators
pq.set_left_operators([['e1(m,e)']])
pq.set_right_operators([['e2(a,b,j,i)']])

print('')
print('    H(m,e;i,j,a,b) = <0|e1(m,e) e(-T) H e(T) e2(a,b,j,i)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()
pq.print_fully_contracted()
pq.clear()

# set right and left-hand operators
pq.set_left_operators([['e2(m,n,f,e)']])
pq.set_right_operators([['e1(a,i)']])

print('')
print('    H(m,n,e,f;i,a) = <0|e2(m,n,f,e) e(-T) H e(T) e1(a,i)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()
pq.print_fully_contracted()
pq.clear()

# set right and left-hand operators
pq.set_left_operators([['e2(m,n,f,e)']])
pq.set_right_operators([['e2(a,b,j,i)']])

print('')
print('    H(m,n,e,f;i,j,a,b) = <0|e2(m,n,f,e) e(-T) H e(T) e2(a,b,j,i)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()
pq.print_fully_contracted()
pq.clear()

