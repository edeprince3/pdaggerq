
import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

T = ['t1','t2','t3','t4']

# energy equation

#pq.set_bra("")
pq.set_left_operators(['1'])

print('')
print('    < 0 | e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],T)
pq.add_st_operator(1.0,['v'],T)

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

pq.add_st_operator(1.0,['f'],T)
pq.add_st_operator(1.0,['v'],T)

pq.simplify()

# grab list of fully-contracted strings, then print
singles_residual_terms = pq.fully_contracted_strings()
for my_term in singles_residual_terms:
    print(my_term, flush=True)

singles_residual_terms = contracted_strings_to_tensor_terms(singles_residual_terms)
for my_term in singles_residual_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='singles_res',
                                output_variables=('a', 'i')))
    print(flush=True)

pq.clear()

# doubles equations

#pq.set_bra("doubles")
pq.set_left_operators(['e2(i,j,b,a)'])

print('')
print('    < 0 | i* j* b a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],T)
pq.add_st_operator(1.0,['v'],T)

pq.simplify()

# grab list of fully-contracted strings, then print
doubles_residual_terms = pq.fully_contracted_strings()
for my_term in doubles_residual_terms:
    print(my_term, flush=True)

doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
for my_term in doubles_residual_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='doubles_res',
                                output_variables=('a', 'b', 'i', 'j')))
    print(flush=True)

pq.clear()

# triples equations

pq.set_left_operators(['e3(i,j,k,c,b,a)'])

print('')
print('    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],T)
pq.add_st_operator(1.0,['v'],T)

pq.simplify()

# grab list of fully-contracted strings, then print
triples_residual_terms = pq.fully_contracted_strings()
for my_term in triples_residual_terms:
    print(my_term, flush=True)

triples_residual_terms = contracted_strings_to_tensor_terms(triples_residual_terms)
for my_term in triples_residual_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='triples_res',
                                output_variables=('a', 'b', 'c', 'i', 'j', 'k')))
    print(flush=True)

pq.clear()

# quadruples equations

pq.set_left_operators(['e4(i,j,k,l,d,c,b,a)'])

print('')
print('    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],T)
pq.add_st_operator(1.0,['v'],T)

pq.simplify()

# grab list of fully-contracted strings, then print
quadruples_residual_terms = pq.fully_contracted_strings()
for my_term in quadruples_residual_terms:
    print(my_term, flush=True)

quadruples_residual_terms = contracted_strings_to_tensor_terms(quadruples_residual_terms)
for my_term in quadruples_residual_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='quadruples_res',
                                output_variables=('a', 'b', 'c', 'd', 'i', 'j', 'k', 'l')))
    print(flush=True)

pq.clear()


