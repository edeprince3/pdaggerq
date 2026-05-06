import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


def main():
    pq = pdaggerq.pq_helper("fermi")

    pq.set_unitary_cc(True)

    # energy equation

    print('')
    print('def quccsd_energy(t1, t2, f, g, o, v):')
    print('')
    print('#    < 0 | e(-T) H e(T) | 0> :')
    print('')

    pq.add_operator_product(1.0, ['f'])
    pq.add_commutator(1.0, ['f'], ['t1'])
    pq.add_commutator(1.0, ['f'], ['t2'])

    pq.add_bernoulli_operator(1.0,['v'],['t1','t2'], 3)

    pq.simplify()

    terms = pq.strings()
    terms = contracted_strings_to_tensor_terms(terms)

    for term in terms:
        print("#\t", term)
        print("%s" % (term.einsum_string(update_val='energy')))
        print()

    print('return energy')
    print('')

    pq.clear()

    print('')
    print('def quccsd_singles_residual(t1, t2, f, g, o, v):')
    print('')
    print('#    < 0 | i* a e(-T) H e(T) | 0> :')
    print('')

    pq.set_left_operators([['a*(i)', 'a(a)']])

    pq.add_operator_product(1.0, ['f'])
    pq.add_commutator(1.0, ['f'], ['t1'])
    pq.add_commutator(1.0, ['f'], ['t2'])

    pq.add_bernoulli_operator(1.0,['v'],['t1','t2'], 2)

    pq.simplify()

    terms = pq.strings()
    terms = contracted_strings_to_tensor_terms(terms)
    for term in terms:
        print("#\t", term)
        print("%s" % (term.einsum_string(update_val='singles_res',
                                    output_variables=('a', 'i'))))
        print()

    print('return singles_res')
    print('')

    pq.clear()

    print('')
    print('def quccsd_doubles_residual(t1, t2, f, g, o, v):')
    print('')
    print('#    < 0 | i* j* b a e(-T) H e(T) | 0> :')
    print('')

    pq.set_left_operators([['a*(i)', 'a*(j)', 'a(b)', 'a(a)']])

    pq.add_operator_product(1.0, ['f'])
    pq.add_commutator(1.0, ['f'], ['t1'])
    pq.add_commutator(1.0, ['f'], ['t2'])

    pq.add_bernoulli_operator(1.0,['v'],['t1','t2'], 2)

    pq.simplify()

    terms = pq.strings()
    terms = contracted_strings_to_tensor_terms(terms)
    for term in terms:
        print("#\t", term)
        print("%s" % (term.einsum_string(update_val='doubles_res',
                                    output_variables=('a', 'b', 'i', 'j'))))
        print()

    print('return doubles_res')
    print('')

    pq.clear()
if __name__ == "__main__":
    main()
