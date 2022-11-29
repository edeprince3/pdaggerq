import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


def main():
    pq = pdaggerq.pq_helper("fermi")

    # energy equation

    pq.set_left_operators([['1']])


    print('')
    print('def ccsd_energy_with_spin(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])

    pq.simplify()

    # grab list of fully-contracted strings, then print

    energy_terms = pq.fully_contracted_strings_with_spin()
    energy_terms = contracted_strings_to_tensor_terms(energy_terms)

    for my_term in energy_terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='energy')))
        print()

    print('return energy')
    print('')

    pq.clear()

    # CCSD singles equations

    print('')
    print('def ccsd_t1_aa_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | m* e e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    pq.set_left_operators([['e1(m,e)']])

    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'm' : 'a',
        'e' : 'a'
    }
    singles_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    singles_residual_terms = contracted_strings_to_tensor_terms(
        singles_residual_terms)
    for my_term in singles_residual_terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='singles_res',
                                    output_variables=('e', 'm'))))
        print()

    print('return singles_res')
    print('')

    print('')
    print('def ccsd_t1_bb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | m* e e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'm' : 'b',
        'e' : 'b'
    }
    singles_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
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

    pq.add_st_operator(1.0, ['f'], ['t1', 't2'])
    pq.add_st_operator(1.0, ['v'], ['t1', 't2'])

    pq.simplify()

    print('')
    print('def ccsd_t2_aaaa_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | m* n* f e e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'm' : 'a',
        'n' : 'a',
        'e' : 'a',
        'f' : 'a'
    }
    doubles_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
    for my_term in doubles_residual_terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='doubles_res',
                                    output_variables=('e', 'f', 'm', 'n'))))
        print()

    print('return doubles_res')
    print('')

    print('')
    print('def ccsd_t2_bbbb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | m* n* f e e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'm' : 'b',
        'n' : 'b',
        'e' : 'b',
        'f' : 'b'
    }
    doubles_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
    for my_term in doubles_residual_terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='doubles_res',
                                    output_variables=('e', 'f', 'm', 'n'))))
        print()

    print('return doubles_res')
    print('')

    print('')
    print('def ccsd_t2_abab_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | m* n* f e e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'm' : 'a',
        'n' : 'b',
        'e' : 'a',
        'f' : 'b'
    }
    doubles_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
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
