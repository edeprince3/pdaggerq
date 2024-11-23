import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


def main():
    pq = pdaggerq.pq_helper("fermi")

    # energy equation

    pq.set_left_operators([['1']])


    print('')
    print('def ccsd_energy(t1, t2, f, g, o, v):')
    print('')
    print('#    < 0 | e(-T) H e(T) | 0> :')
    print('')

    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    energy_terms = pq.strings()
    energy_terms = contracted_strings_to_tensor_terms(energy_terms)

    for my_term in energy_terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='energy')))
        print()

    print('return energy')
    print('')

    pq.clear()

    # CCSD singles equations

    pq.set_left_operators([['e1(m,e)']])


    print('')
    print('def singles_residual(t1, t2, f, g, o, v):')
    print('')
    print('#    < 0 | m* e e(-T) H e(T) | 0> :')
    print('')

    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    singles_residual_terms = pq.strings()
    singles_residual_terms = contracted_strings_to_tensor_terms(
        singles_residual_terms)
    for my_term in singles_residual_terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='singles_res',
                                    output_variables=('e', 'm'))))
        print()

    print('return singles_res')
    print('')

    pq.clear()

    # CCSD doubles equations

    pq.set_left_operators([['e2(m,n,f,e)']])


    print('')
    print('def doubles_residual(t1, t2, f, g, o, v):')
    print('')
    print('#    < 0 | m* n* f e e(-T) H e(T) | 0> :')
    print('')

    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    doubles_residual_terms = pq.strings()
    doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
    for my_term in doubles_residual_terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='doubles_res',
                                    output_variables=('e', 'f', 'm', 'n'))))
        print()

    print('return doubles_res')
    print('')

    pq.clear()

if __name__ == "__main__":
    main()
