
import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


def main():

    pq = pdaggerq.pq_helper("fermi")
    pq.set_print_level(0)
    
    T = ['t1','t2','t3','t4']
    
    # doubles equations
    
    pq.set_left_operators([['e2(i,j,b,a)']])
    
    pq.add_st_operator(1.0,['f'], T)
    pq.add_st_operator(1.0,['v'], T)
    
    pq.simplify()

    print('')
    print('def ccsdtq_t2_aaaa_residual(t1_aa, t1_bb, \n \
                           t2_aaaa, t2_bbbb, t2_abab, \n \
                           t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, \n \
                           t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, \n \
                           f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')
    
    # grab list of fully-contracted strings, then print
    spin_labels = {
        'a' : 'a',
        'b' : 'a',
        'i' : 'a',
        'j' : 'a'
    }
    doubles_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
    for my_term in doubles_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='doubles_res',
                                    output_variables=('a', 'b', 'i', 'j')))
        print(flush=True)

    print('return doubles_res')
    print('')

    print('')
    print('def ccsdtq_t2_abab_residual(t1_aa, t1_bb, \n \
                           t2_aaaa, t2_bbbb, t2_abab, \n \
                           t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, \n \
                           t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, \n \
                           f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'a' : 'a',
        'b' : 'b',
        'i' : 'a',
        'j' : 'b'
    }
    doubles_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
    for my_term in doubles_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='doubles_res',
                                    output_variables=('a', 'b', 'i', 'j')))
        print(flush=True)

    print('return doubles_res')
    print('')

    print('')
    print('def ccsdtq_t2_bbbb_residual(t1_aa, t1_bb, \n \
                           t2_aaaa, t2_bbbb, t2_abab, \n \
                           t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, \n \
                           t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, \n \
                           f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'a' : 'b',
        'b' : 'b',
        'i' : 'b',
        'j' : 'b'
    }
    doubles_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    doubles_residual_terms = contracted_strings_to_tensor_terms(doubles_residual_terms)
    for my_term in doubles_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='doubles_res',
                                    output_variables=('a', 'b', 'i', 'j')))
        print(flush=True)

    print('return doubles_res')
    print('')
    
    pq.clear()
    
    # triples equations
    
    pq.set_left_operators([['e3(i,j,k,c,b,a)']])
    
    print('')
    print('    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
    print('')
    
    pq.add_st_operator(1.0,['f'], T)
    pq.add_st_operator(1.0,['v'], T)
    
    pq.simplify()

    print('')
    print('def ccsdtq_t3_aaaaaa_residual(t1_aa, t1_bb, \n \
                             t2_aaaa, t2_bbbb, t2_abab, \n \
                             t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, \n \
                             t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, \n \
                             f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'a' : 'a',
        'b' : 'a',
        'c' : 'a',
        'i' : 'a',
        'j' : 'a',
        'k' : 'a'
    }
    triples_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    triples_residual_terms = contracted_strings_to_tensor_terms(triples_residual_terms)
    for my_term in triples_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='triples_res',
                                    output_variables=('a', 'b', 'c', 'i', 'j', 'k')))
        print(flush=True)
    
    print('return triples_res')
    print('')

    print('')
    print('def ccsdtq_t3_aabaab_residual(t1_aa, t1_bb, \n \
                             t2_aaaa, t2_bbbb, t2_abab, \n \
                             t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, \n \
                             t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, \n \
                             f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'a' : 'a',
        'b' : 'a',
        'c' : 'b',
        'i' : 'a',
        'j' : 'a',
        'k' : 'b'
    }
    triples_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    triples_residual_terms = contracted_strings_to_tensor_terms(triples_residual_terms)
    for my_term in triples_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='triples_res',
                                    output_variables=('a', 'b', 'c', 'i', 'j', 'k')))
        print(flush=True)
    
    print('return triples_res')
    print('')

    print('')
    print('def ccsdtq_t3_abbabb_residual(t1_aa, t1_bb, \n \
                             t2_aaaa, t2_bbbb, t2_abab, \n \
                             t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, \n \
                             t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, \n \
                             f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'a' : 'a',
        'b' : 'b',
        'c' : 'b',
        'i' : 'a',
        'j' : 'b',
        'k' : 'b'
    }
    triples_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    triples_residual_terms = contracted_strings_to_tensor_terms(triples_residual_terms)
    for my_term in triples_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='triples_res',
                                    output_variables=('a', 'b', 'c', 'i', 'j', 'k')))
        print(flush=True)
    
    print('return triples_res')
    print('')

    print('')
    print('def ccsdtq_t3_bbbbbb_residual(t1_aa, t1_bb, \n \
                             t2_aaaa, t2_bbbb, t2_abab, \n \
                             t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, \n \
                             t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, \n \
                             f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')

    # grab list of fully-contracted strings, then print
    spin_labels = {
        'a' : 'b',
        'b' : 'b',
        'c' : 'b',
        'i' : 'b',
        'j' : 'b',
        'k' : 'b'
    }
    triples_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    triples_residual_terms = contracted_strings_to_tensor_terms(triples_residual_terms)
    for my_term in triples_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='triples_res',
                                    output_variables=('a', 'b', 'c', 'i', 'j', 'k')))
        print(flush=True)
    
    print('return triples_res')
    print('')

    pq.clear()

    # quadruples equations
    
    pq.set_left_operators([['e4(i,j,k,l,d,c,b,a)']])
    
    pq.add_st_operator(1.0,['f'], T)
    pq.add_st_operator(1.0,['v'], T)
    
    pq.simplify()

    print('')
    print('def ccsdtq_t4_aaaaaaaa_residual(t1_aa, t1_bb, \n \
                               t2_aaaa, t2_bbbb, t2_abab, \n \
                               t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, \n \
                               t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, \n \
                               f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')
    
    # grab list of fully-contracted strings, then print
    spin_labels = {
        'a' : 'a',
        'b' : 'a',
        'c' : 'a',
        'd' : 'a',
        'i' : 'a',
        'j' : 'a',
        'k' : 'a',
        'l' : 'a'
    }
    quadruples_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    quadruples_residual_terms = contracted_strings_to_tensor_terms(quadruples_residual_terms)
    for my_term in quadruples_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='quadruples_res',
                                    output_variables=('a', 'b', 'c', 'd', 'i', 'j', 'k', 'l')))
        print(flush=True)

    print('return quadruples_res')
    print('')

    print('')
    print('def ccsdtq_t4_aaabaaab_residual(t1_aa, t1_bb, \n \
                               t2_aaaa, t2_bbbb, t2_abab, \n \
                               t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, \n \
                               t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, \n \
                               f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')
    
    # grab list of fully-contracted strings, then print
    spin_labels = {
        'a' : 'a',
        'b' : 'a',
        'c' : 'a',
        'd' : 'b',
        'i' : 'a',
        'j' : 'a',
        'k' : 'a',
        'l' : 'b'
    }
    quadruples_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    quadruples_residual_terms = contracted_strings_to_tensor_terms(quadruples_residual_terms)
    for my_term in quadruples_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='quadruples_res',
                                    output_variables=('a', 'b', 'c', 'd', 'i', 'j', 'k', 'l')))
        print(flush=True)

    print('return quadruples_res')
    print('')

    print('')
    print('def ccsdtq_t4_aabbaabb_residual(t1_aa, t1_bb, \n \
                               t2_aaaa, t2_bbbb, t2_abab, \n \
                               t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, \n \
                               t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, \n \
                               f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')
   
    # grab list of fully-contracted strings, then print
    spin_labels = {
        'a' : 'a',
        'b' : 'a',
        'c' : 'b',
        'd' : 'b',
        'i' : 'a',
        'j' : 'a',
        'k' : 'b',
        'l' : 'b'
    }
    quadruples_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    quadruples_residual_terms = contracted_strings_to_tensor_terms(quadruples_residual_terms)
    for my_term in quadruples_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='quadruples_res',
                                    output_variables=('a', 'b', 'c', 'd', 'i', 'j', 'k', 'l')))
        print(flush=True)

    print('return quadruples_res')
    print('')

    print('')
    print('def ccsdtq_t4_abbbabbb_residual(t1_aa, t1_bb, \n \
                               t2_aaaa, t2_bbbb, t2_abab, \n \
                               t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, \n \
                               t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, \n \
                               f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')
  
    # grab list of fully-contracted strings, then print
    spin_labels = {
        'a' : 'a',
        'b' : 'b',
        'c' : 'b',
        'd' : 'b',
        'i' : 'a',
        'j' : 'b',
        'k' : 'b',
        'l' : 'b'
    }
    quadruples_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    quadruples_residual_terms = contracted_strings_to_tensor_terms(quadruples_residual_terms)
    for my_term in quadruples_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='quadruples_res',
                                    output_variables=('a', 'b', 'c', 'd', 'i', 'j', 'k', 'l')))
        print(flush=True)

    print('return quadruples_res')
    print('')

    print('')
    print('def ccsdtq_t4_bbbbbbbb_residual(t1_aa, t1_bb, \n \
                               t2_aaaa, t2_bbbb, t2_abab, \n \
                               t3_aaaaaa, t3_aabaab, t3_abbabb, t3_bbbbbb, \n \
                               t4_aaaaaaaa, t4_aaabaaab, t4_aabbaabb, t4_abbbabbb, t4_bbbbbbbb, \n \
                               f_aa, f_bb, g_aaaa, g_bbbb, g_abab, oa, ob, va, vb):')
    print('')
    print('#    < 0 | i* j* k* l* d c b a e(-T) H e(T) | 0> :')
    print('')
    print('o = oa')
    print('v = va')
    print('')
 
    # grab list of fully-contracted strings, then print
    spin_labels = {
        'a' : 'b',
        'b' : 'b',
        'c' : 'b',
        'd' : 'b',
        'i' : 'b',
        'j' : 'b',
        'k' : 'b',
        'l' : 'b'
    }
    quadruples_residual_terms = pq.fully_contracted_strings_with_spin(spin_labels)
    quadruples_residual_terms = contracted_strings_to_tensor_terms(quadruples_residual_terms)
    for my_term in quadruples_residual_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='quadruples_res',
                                    output_variables=('a', 'b', 'c', 'd', 'i', 'j', 'k', 'l')))
        print(flush=True)
   
    print('return quadruples_res')
    print('')

    pq.clear()
    
if __name__ == "__main__":
    main()
