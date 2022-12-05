
import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms

def main():
    pq = pdaggerq.pq_helper("fermi")

    # set right and left-hand operators
    pq.set_right_operators_type('DEA')
    pq.set_left_operators_type('DEA')
    pq.set_left_operators([['a(b)', 'a(a)']])
    pq.set_right_operators([['a*(d)', 'a*(e)']])

    print('')
    print('def dea_eom_ccsd_hamiltonian_22(kd, f, g, o, v, t1, t2):')
    print('')
    print('#    H(a,b;d,e) = <0|b a e(-T) H e(T) d* e*|0>')
    print('')

    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])

    pq.simplify()

    H = pq.fully_contracted_strings()
    H = contracted_strings_to_tensor_terms(H)
    for my_term in H:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='H',
                                    output_variables=('a', 'b', 'd', 'e')))
        print()

    print('')
    print('return H')
    print('')

    pq.clear()
    
    # set right and left-hand operators
    pq.set_right_operators_type('DEA')
    pq.set_left_operators_type('DEA')
    pq.set_left_operators([['a*(i)', 'a(c)', 'a(b)', 'a(a)']])
    pq.set_right_operators([['a*(d)', 'a*(e)']])

    print('')
    print('def dea_eom_ccsd_hamiltonian_32(kd, f, g, o, v, t1, t2):')
    print('')
    print('#    H(a,b,c,i;d,e) = <0|i* c b a e(-T) H e(T) d* e*|0>')
    print('')

    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])

    pq.simplify()

    H = pq.fully_contracted_strings()
    H = contracted_strings_to_tensor_terms(H)
    for my_term in H:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='H',
                                    output_variables=('a', 'b', 'c', 'i', 'd', 'e')))
        print()

    print('')
    print('return H')
    print('')

    pq.clear()
    
    # set right and left-hand operators
    pq.set_right_operators_type('DEA')
    pq.set_left_operators_type('DEA')
    pq.set_left_operators([['a(b)', 'a(a)']])
    pq.set_right_operators([['a*(d)', 'a*(e)', 'a*(f)', 'a(j)']])

    print('')
    print('def dea_eom_ccsd_hamiltonian_23(kd, f, g, o, v, t1, t2):')
    print('')
    print('#    H(a,b;d,e,f,j) = <0|b a e(-T) H e(T) d* e* f* j|0>')
    print('')

    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])

    pq.simplify()

    H = pq.fully_contracted_strings()
    H = contracted_strings_to_tensor_terms(H)
    for my_term in H:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='H',
                                    output_variables=('a', 'b', 'd', 'e', 'f', 'j')))
        print()
    
    print('')
    print('return H')
    print('')

    pq.clear()

    # set right and left-hand operators
    pq.set_right_operators_type('DEA')
    pq.set_left_operators_type('DEA')
    pq.set_left_operators([['a*(i)', 'a(c)', 'a(b)', 'a(a)']])
    pq.set_right_operators([['a*(d)', 'a*(e)', 'a*(f)', 'a(j)']])

    print('')
    print('def dea_eom_ccsd_hamiltonian_33(kd, f, g, o, v, t1, t2):')
    print('')
    print('#    H(a,b,c,i;d,e,f,j) = <0|i* c b a e(-T) H e(T) d* e* f* j|0>')
    print('')

    pq.add_st_operator(1.0,['f'],['t1','t2'])
    pq.add_st_operator(1.0,['v'],['t1','t2'])

    pq.simplify()

    H = pq.fully_contracted_strings()
    H = contracted_strings_to_tensor_terms(H)
    for my_term in H:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='H',
                                    output_variables=('a', 'b', 'c', 'i', 'd', 'e', 'f', 'j')))
        print()
    
    print('')
    print('return H')
    print('')

    pq.clear()
    
if __name__ == "__main__":
    main()
