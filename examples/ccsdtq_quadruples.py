
import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

T = ['t1','t2','t3','t4']

# quadruples equations

pq.set_left_operators(['e4(i,j,k,l,d,c,b,a)'])

print('')
print('    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],T)
pq.add_st_operator(1.0,['v'],T)

pq.simplify()

# grab list of fully-contracted strings, then print
quadruples_residual_terms = pq.strings()
for my_term in quadruples_residual_terms:
    print(my_term, flush=True)

quadruples_residual_terms = contracted_strings_to_tensor_terms(quadruples_residual_terms)
for my_term in quadruples_residual_terms:
    print("#\t", my_term)
    print(my_term.einsum_string(update_val='quadruples_res',
                                output_variables=('a', 'b', 'c', 'd', 'i', 'j', 'k', 'l')))
    print(flush=True)

pq.clear()


