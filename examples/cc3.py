
import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

# energy equation

pq.set_left_operators([['1']])

print('')
print('    < 0 | e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3'])

pq.simplify()

# grab list of fully-contracted strings, then print
energy_terms = pq.strings()
for my_term in energy_terms:
    print(my_term)

pq.clear()

# singles equations

pq.set_left_operators([['e1(i,a)']])

print('')
print('    < 0 | i* a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3'])

pq.simplify()

# grab list of fully-contracted strings, then print
singles_residual_terms = pq.strings()
for my_term in singles_residual_terms:
    print(my_term)

singles_residual_terms = contracted_strings_to_tensor_terms(singles_residual_terms)
for my_term in singles_residual_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='singles_res',
                                output_variables=('a', 'i')))
    print()

pq.clear()

# doubles equations

pq.set_left_operators([['e2(i,j,b,a)']])

print('')
print('    < 0 | i* j* b a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3'])

pq.simplify()

# grab list of fully-contracted strings, then print
doubles_residual_terms = pq.strings()
for my_term in doubles_residual_terms:
    print(my_term)

doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
for my_term in doubles_residual_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='doubles_res',
                                output_variables=('a', 'b', 'i', 'j')))
    print()

pq.clear()

# triples equations

pq.set_left_operators([['e3(i,j,k,c,b,a)']])

print('')
print('    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1'])

# g
pq.add_operator_product(1.0,['v'])

# [g, T1]
#pq.add_commutator(1.0,['v'],['t1'])

# [g, T2]
pq.add_commutator(1.0,['v'],['t2'])

# [[g, T1], T1]]
#pq.add_double_commutator(0.5, ['v'],['t1'],['t1'])

# [[g, T1], T2]] + [[g, T2], T1]]
pq.add_double_commutator( 1.0, ['v'],['t1'],['t2'])

# [[g, T2, T2]]
#pq.add_double_commutator( 0.5, ['v'],['t2'],['t2'])

# triple commutators

# [[[g, T1, T1], T1]
#pq.add_triple_commutator( 1.0/6.0, ['v'],['t1'],['t1'],['t1'])

# [[[g, T1, T1], T2] + [[[g, T1, T2], T1] + [[[g, T2, T1], T1]
pq.add_triple_commutator( 1.0/2.0, ['v'],['t1'],['t1'],['t2'])

# quadruple commutators: the only one that yields non-zero contributions to the doubles equations will be

# [[[[g, T1], T1], T1], T1]
#pq.add_quadruple_commutator( 1.0/24.0, ['v'],['t1'],['t1'],['t1'],['t1'])

# [[[[g, T1], T1], T1], T2] + three others
pq.add_quadruple_commutator( 1.0/6.0, ['v'],['t1'],['t1'],['t1'],['t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
triples_residual_terms = pq.strings()
for my_term in triples_residual_terms:
    print(my_term)

triples_residual_terms = contracted_strings_to_tensor_terms(triples_residual_terms)
for my_term in triples_residual_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='triples_res',
                                output_variables=('a', 'b', 'c', 'i', 'j', 'k')))
    print()

pq.clear()


