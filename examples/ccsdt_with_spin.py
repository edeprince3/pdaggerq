
import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


def main():

    pq = pdaggerq.pq_helper("fermi")
    pq.set_print_level(0)
    
    # singles equations
    
    pq.set_left_operators([['e1(i,a)']])
    
    pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
    pq.add_st_operator(1.0,['v'],['t1','t2','t3'])
    
    pq.simplify()

    print('')
    print('def ccsdt_t1_aa_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'i' : 'a',
        'a' : 'a'
    }
    singles_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    singles_residual_terms = contracted_strings_to_tensor_terms(
        singles_residual_terms)
    for my_term in singles_residual_terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='singles_res',
                                    output_variables=('a', 'i'))))
        print()

    print('return singles_res')
    print('')

    print('')
    print('def ccsdt_t1_bb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'i' : 'b',
        'a' : 'b'
    }
    singles_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    singles_residual_terms = contracted_strings_to_tensor_terms(
        singles_residual_terms)
    for my_term in singles_residual_terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='singles_res',
                                    output_variables=('a', 'i'))))
        print()

    print('return singles_res')
    print('')

    pq.clear()
    
    # doubles equations
    
    pq.set_left_operators([['e2(i,j,b,a)']])
    
    pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
    pq.add_st_operator(1.0,['v'],['t1','t2','t3'])
    
    pq.simplify()

    print('')
    print('def ccsdt_t2_aaaa_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'i' : 'a',
        'j' : 'a',
        'a' : 'a',
        'b' : 'a'
    }
    doubles_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
    for my_term in doubles_residual_terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='doubles_res',
                                    output_variables=('a', 'b', 'i', 'j'))))
        print()

    print('return doubles_res')
    print('')

    print('')
    print('def ccsdt_t2_bbbb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'i' : 'b',
        'j' : 'b',
        'a' : 'b',
        'b' : 'b'
    }
    doubles_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
    for my_term in doubles_residual_terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='doubles_res',
                                    output_variables=('a', 'b', 'i', 'j'))))
        print()

    print('return doubles_res')
    print('')

    print('')
    print('def ccsdt_t2_abab_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'i' : 'a',
        'j' : 'b',
        'a' : 'a',
        'b' : 'b'
    }
    doubles_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
    for my_term in doubles_residual_terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='doubles_res',
                                    output_variables=('a', 'b', 'i', 'j'))))
        print()

    print('return doubles_res')
    print('')

    pq.clear()
    
    # triples equations
    
    pq.set_left_operators([['e3(i,j,k,c,b,a)']])
    
    print('')
    print('#    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
    print('')
    
    pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
    pq.add_st_operator(1.0,['v'],['t1','t2','t3'])
    
    pq.simplify()

    print('')
    print('def ccsdt_t3_aaaaaa_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')
    
    # grab list of fully-contracted strings, then print
    spin_labels = {
        'i' : 'a',
        'j' : 'a',
        'k' : 'a',
        'a' : 'a',
        'b' : 'a',
        'c' : 'a'
    }
    triples_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    triples_residual_terms = contracted_strings_to_tensor_terms(triples_residual_terms)
    for my_term in triples_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='triples_res',
                                    output_variables=('a', 'b', 'c', 'i', 'j', 'k')))
        print()

    print('return triples_res')
    print('')

    print('')
    print('def ccsdt_t3_aabaab_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'i' : 'a',
        'j' : 'a',
        'k' : 'b',
        'a' : 'a',
        'b' : 'a',
        'c' : 'b'
    }
    triples_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    triples_residual_terms = contracted_strings_to_tensor_terms(triples_residual_terms)
    for my_term in triples_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='triples_res',
                                    output_variables=('a', 'b', 'c', 'i', 'j', 'k')))
        print()

    print('return triples_res')
    print('')

    print('')
    print('def ccsdt_t3_abbabb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'i' : 'a',
        'j' : 'b',
        'k' : 'b',
        'a' : 'a',
        'b' : 'b',
        'c' : 'b'
    }
    triples_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    triples_residual_terms = contracted_strings_to_tensor_terms(triples_residual_terms)
    for my_term in triples_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='triples_res',
                                    output_variables=('a', 'b', 'c', 'i', 'j', 'k')))
        print()

    print('return triples_res')
    print('')

    print('')
    print('def ccsdt_t3_bbbbbb_residual(t1_aa, t1_bb, t2_aaaa, t2_bbbb, t2_abab, t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'i' : 'b',
        'j' : 'b',
        'k' : 'b',
        'a' : 'b',
        'b' : 'b',
        'c' : 'b'
    }
    triples_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    triples_residual_terms = contracted_strings_to_tensor_terms(triples_residual_terms)
    for my_term in triples_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='triples_res',
                                    output_variables=('a', 'b', 'c', 'i', 'j', 'k')))
        print()

    print('return triples_res')
    print('')
    
    pq.clear()

if __name__ == "__main__":
    main()
