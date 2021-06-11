import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


def main():
    pq = pdaggerq.pq_helper("fermi")
    pq.set_print_level(0)

    # energy equation

    pq.set_bra("")

    print('')
    print('    < 0 | e(-T) H e(T) | 0> :')
    print('')

    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    energy_terms = pq.fully_contracted_strings()
    for my_term in energy_terms:
        print(my_term)
    energy_terms = contracted_strings_to_tensor_terms(energy_terms)

    for my_term in energy_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='energy'))
        print()

    pq.clear()

    # singles equations

    # pq.set_bra("singles")
    pq.set_left_operators(['e1(m,e)'])

    print('')
    print('    < 0 | m* e e(-T) H e(T) | 0> :')
    print('')

    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    singles_residual_terms = pq.fully_contracted_strings()
    singles_residual_terms = contracted_strings_to_tensor_terms(
        singles_residual_terms)
    for my_term in singles_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='singles_res',
                                    output_variables=('e', 'm')))
        print()

    pq.clear()

    # doubles equations

    # pq.set_bra("doubles")
    pq.set_left_operators(['e2(m,n,f,e)'])

    print('')
    print('    < 0 | m* n* f e e(-T) H e(T) | 0> :')
    print('')

    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    doubles_residual_terms = pq.fully_contracted_strings()
    doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
    for my_term in doubles_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='doubles_res',
                                    output_variables=('e', 'f', 'm', 'n')))
        print()


    pq.clear()

if __name__ == "__main__":
    main()
