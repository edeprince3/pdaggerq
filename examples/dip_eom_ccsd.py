
import pdaggerq

pq = pdaggerq.pq_helper("fermi")

pq.set_print_level(0)

# set right and left-hand operators
pq.set_right_operators_type('DIP')
pq.set_left_operators([['a*(i)', 'a*(j)']])
pq.set_right_operators([['r2'], ['r3']])

print('')
print('    sigma(ij) = <0|i* j* e(-T) H e(T) (r2 + r3)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.fully_contracted_strings()
for term in terms:
    print(term)
pq.clear()

# set right and left-hand operators
pq.set_right_operators_type('DIP')
pq.set_left_operators([['a*(i)', 'a*(j)', 'a*(k)', 'a(a)']])

pq.set_right_operators([['r2'], ['r3']])

print('')
print('    sigma(ija) = <0|i* j* a e(-T) H e(T) (r2 + r3)|0>')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.fully_contracted_strings()
for term in terms:
    print(term)
pq.clear()

