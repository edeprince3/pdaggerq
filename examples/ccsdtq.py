
import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

# energy equation

#pq.set_bra("")
pq.set_left_operators(['1'])

print('')
print('    < 0 | e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3','t4'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3','t4'])

pq.simplify()

# grab list of fully-contracted strings, then print
energy_terms = pq.fully_contracted_strings()
for my_term in energy_terms:
    print(my_term, flush=True)

pq.clear()

# singles equations

#pq.set_bra("singles")
pq.set_left_operators(['e1(i,a)'])

print('')
print('    < 0 | i* a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3','t4'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3','t4'])

pq.simplify()

# grab list of fully-contracted strings, then print
singles_residual_terms = pq.fully_contracted_strings()
for my_term in singles_residual_terms:
    print(my_term, flush=True)

pq.clear()

# doubles equations

#pq.set_bra("doubles")
pq.set_left_operators(['e2(i,j,b,a)'])

print('')
print('    < 0 | i* j* b a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3','t4'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3','t4'])

pq.simplify()

# grab list of fully-contracted strings, then print
doubles_residual_terms = pq.fully_contracted_strings()
for my_term in doubles_residual_terms:
    print(my_term, flush=True)

pq.clear()

# triples equations

pq.set_left_operators(['e3(i,j,k,c,b,a)'])

print('')
print('    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3','t4'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3','t4'])

pq.simplify()

# grab list of fully-contracted strings, then print
triples_residual_terms = pq.fully_contracted_strings()
for my_term in triples_residual_terms:
    print(my_term, flush=True)

pq.clear()

# quadruples equations

pq.set_left_operators(['e4(i,j,k,l,d,c,b,a)'])

print('')
print('    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3','t4'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3','t4'])

pq.simplify()

# grab list of fully-contracted strings, then print
triples_residual_terms = pq.fully_contracted_strings()
for my_term in triples_residual_terms:
    print(my_term, flush=True)

pq.clear()


