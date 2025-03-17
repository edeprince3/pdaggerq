import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms

def main():
    pq = pdaggerq.pq_helper("fermi")

    pq.set_unitary_cc(True)

    # energy equation

    pq.set_left_operators([['1']])

    print('')
    print('def ucc4_energy(t1, t2, f, g, o, v):')
    print('')
    print('#    < 0 | e(-T) H e(T) | 0> :')
    print('')

    # up to fourth order

    pq.add_operator_product(1.0, ['f']) # 0
    pq.add_operator_product(1.0, ['v']) # 1

    pq.add_commutator(1.0, ['f'],['t1']) # 2
    pq.add_commutator(1.0, ['f'],['t2']) # 1
    pq.add_commutator(1.0, ['v'],['t1']) # 3
    pq.add_commutator(1.0, ['v'],['t2']) # 2

    pq.add_double_commutator(0.5, ['f'],['t1'],['t1']) # 4
    pq.add_double_commutator(0.5, ['f'],['t1'],['t2']) # 3
    pq.add_double_commutator(0.5, ['f'],['t2'],['t1']) # 3
    pq.add_double_commutator(0.5, ['f'],['t2'],['t2']) # 2

    pq.add_double_commutator(0.5, ['v'],['t1'],['t2']) # 4
    pq.add_double_commutator(0.5, ['v'],['t2'],['t1']) # 4
    pq.add_double_commutator(0.5, ['v'],['t2'],['t2']) # 3

    pq.add_triple_commutator(1.0 / 6.0, ['f'],['t1'],['t2'],['t2']) # 4
    pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t1'],['t2']) # 4
    pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t2'],['t1']) # 4
    pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t2'],['t2']) # 3

    pq.add_triple_commutator(1.0 / 6.0, ['v'],['t2'],['t2'],['t2']) # 4

    pq.add_quadruple_commutator(1.0 / 24.0, ['f'],['t2'],['t2'],['t2'],['t2']) # 4

    pq.simplify()

    # optimize the equations
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
    print('def ucc4_singles_residual(t1, t2, f, g, o, v):')
    print('')
    print('#    < 0 | m* e e(-T) H e(T) | 0> :')
    print('')

    # up to second order
    pq.add_operator_product(1.0, ['f']) # 0
    pq.add_operator_product(1.0, ['v']) # 1

    pq.add_commutator(1.0, ['f'],['t1']) # 2
    pq.add_commutator(1.0, ['f'],['t2']) # 1
    #pq.add_commutator(1.0, ['v'],['t1']) # 3
    pq.add_commutator(1.0, ['v'],['t2']) # 2

    #pq.add_double_commutator(0.5, ['f'],['t1'],['t1']) # 4
    #pq.add_double_commutator(0.5, ['f'],['t1'],['t2']) # 3
    #pq.add_double_commutator(0.5, ['f'],['t2'],['t1']) # 3
    pq.add_double_commutator(0.5, ['f'],['t2'],['t2']) # 2

    #pq.add_double_commutator(0.5, ['v'],['t1'],['t1']) # 5
    #pq.add_double_commutator(0.5, ['v'],['t1'],['t2']) # 4
    #pq.add_double_commutator(0.5, ['v'],['t2'],['t1']) # 4
    #pq.add_double_commutator(0.5, ['v'],['t2'],['t2']) # 3

    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t1'],['t1'],['t1']) # 6
    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t1'],['t1'],['t2']) # 5
    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t1'],['t2'],['t1']) # 5
    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t1'],['t2'],['t2']) # 4
    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t1'],['t1']) # 5
    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t1'],['t2']) # 4
    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t2'],['t1']) # 4
    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t2'],['t2']) # 3

    pq.simplify()

    # grab list of fully-contracted strings, then print
    terms = pq.strings()
    terms = contracted_strings_to_tensor_terms(terms)
    for my_term in terms:
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
    print('def ucc4_doubles_residual(t1, t2, f, g, o, v):')
    print('')
    print('#    < 0 | m* n* f e e(-T) H e(T) | 0> :')
    print('')

    # up to third order
    pq.add_operator_product(1.0, ['f']) # 0
    pq.add_operator_product(1.0, ['v']) # 1

    pq.add_commutator(1.0, ['f'],['t1']) # 2
    pq.add_commutator(1.0, ['f'],['t2']) # 1
    pq.add_commutator(1.0, ['v'],['t1']) # 3
    pq.add_commutator(1.0, ['v'],['t2']) # 2

    #pq.add_double_commutator(0.5, ['f'],['t1'],['t1']) # 4
    pq.add_double_commutator(0.5, ['f'],['t1'],['t2']) # 3
    pq.add_double_commutator(0.5, ['f'],['t2'],['t1']) # 3
    pq.add_double_commutator(0.5, ['f'],['t2'],['t2']) # 2

    #pq.add_double_commutator(0.5, ['v'],['t1'],['t1']) # 5
    #pq.add_double_commutator(0.5, ['v'],['t1'],['t2']) # 4
    #pq.add_double_commutator(0.5, ['v'],['t2'],['t1']) # 4
    pq.add_double_commutator(0.5, ['v'],['t2'],['t2']) # 3

    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t1'],['t1'],['t1']) # 6
    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t1'],['t1'],['t2']) # 5
    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t1'],['t2'],['t1']) # 5
    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t1'],['t2'],['t2']) # 4
    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t1'],['t1']) # 5
    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t1'],['t2']) # 4
    #pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t2'],['t1']) # 4
    pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t2'],['t2']) # 3

    pq.simplify()

    # grab list of fully-contracted strings, then print
    terms = pq.strings()
    terms = contracted_strings_to_tensor_terms(terms)
    for my_term in terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='doubles_res',
                                    output_variables=('e', 'f', 'm', 'n'))))
        print()

    print('return doubles_res')
    print('')

    pq.clear()

if __name__ == "__main__":
    main()
