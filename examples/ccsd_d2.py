
import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms


def main():
    pq = pdaggerq.pq_helper("fermi")

    # D2(p,q,r,s) = <0|(1 + l1 + l2) e(-T) p*q*sr e(T) |0>
    pq.set_left_operators([['1'],['l1'],['l2']])

    print('\n', '#    D2(i,j,k,l):', '\n')
    pq.set_left_operators([['1'],['l1'],['l2']])
    pq.add_st_operator(1.0,['e2(i,j,l,k)'],['t1','t2'])
    pq.simplify()
    d2_terms_deprince = pq.strings()
    d2_terms_ncr = contracted_strings_to_tensor_terms(d2_terms_deprince)
    for my_term, deprince_term in zip(d2_terms_ncr, d2_terms_deprince):
        print("#\t", deprince_term)
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[o, o, o, o]',
                                    output_variables=['i', 'j', 'k', 'l']))
        print()
    pq.clear()

    print('\n', '#    D2(i,j,k,a):', '\n')
    pq.set_left_operators([['1'],['l1'],['l2']])
    pq.add_st_operator(1.0,['e2(i,j,a,k)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[o, o, o, v]',
                                    output_variables=['i', 'j', 'k', 'a']))
        print()
    pq.clear()

    print('\n', '#    D2(i,j,a,l):', '\n')
    pq.set_left_operators([['1'],['l1'],['l2']])
    pq.add_st_operator(1.0,['e2(i,j,l,a)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[o, o, v, o]',
                                    output_variables=['i', 'j', 'a', 'l']))
        print()
    pq.clear()

    print('\n', '#    D2(i,a,k,l):', '\n')
    pq.set_left_operators([['1'],['l1'],['l2']])
    pq.add_st_operator(1.0,['e2(i,a,l,k)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[o, v, o, o]',
                                    output_variables=['i', 'a', 'k', 'l']))
        print()
    pq.clear()

    print('\n', '#    D2(a,j,k,l):', '\n')
    pq.set_left_operators([['1'],['l1'],['l2']])
    pq.add_st_operator(1.0,['e2(a,j,l,k)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[v, o, o, o]',
                                    output_variables=['a', 'j', 'k', 'l']))
        print()
    pq.clear()

    print('\n', '#    D2(a,b,c,d):', '\n')
    pq.set_left_operators([['1'],['l1'],['l2']])
    pq.add_st_operator(1.0,['e2(a,b,d,c)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[v, v, v, v]',
                                    output_variables=['a', 'b', 'c', 'd']))
        print()
    pq.clear()

    print('\n', '#    D2(a,b,c,i):', '\n')
    pq.add_st_operator(1.0,['e2(a,b,i,c)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[v, v, v, o]',
                                    output_variables=['a', 'b', 'c', 'i']))
        print()
    pq.clear()

    print('\n', '#    D2(a,b,i,d):', '\n')
    pq.add_st_operator(1.0,['e2(a,b,d,i)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[v, v, o, v]',
                                    output_variables=['a', 'b', 'i', 'd']))
        print()
    pq.clear()

    print('\n', '#    D2(i,b,c,d):', '\n')
    pq.add_st_operator(1.0,['e2(i,b,d,c)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[o, v, v, v]',
                                    output_variables=['i', 'b', 'c', 'd']))
        print()
    pq.clear()

    print('\n', '#    D2(a,i,c,d):', '\n')
    pq.add_st_operator(1.0,['e2(a,i,d,c)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[v, o, v, v]',
                                    output_variables=['a', 'i', 'c', 'd']))
        print()
    pq.clear()

    print('\n', '#    D2(i,j,a,b):', '\n')
    pq.add_st_operator(1.0,['e2(i,j,b,a)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[o, o, v, v]',
                                    output_variables=['i', 'j', 'a', 'b']))
        print()
    pq.clear()

    print('\n', '#    D2(a,b,i,j):', '\n')
    pq.add_st_operator(1.0, ['e2(a,b,j,i)'], ['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[v, v, o, o]',
                                    output_variables=['a', 'b', 'i', 'j']))
        print()
    pq.clear()

    print('\n', '#    D2(i,a,j,b):', '\n')
    pq.add_st_operator(1.0,['e2(i,a,b,j)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[o, v, o, v]',
                                    output_variables=['i', 'a', 'j', 'b']))
        print()
    pq.clear()

    print('\n', '#    D2(a,i,j,b):', '\n')
    pq.add_st_operator(1.0,['e2(a,i,b,j)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[v, o, o, v]',
                                    output_variables=['a', 'i', 'j', 'b']))
        print()
    pq.clear()


    print('\n', '#    D2(i,a,b,j):', '\n')
    pq.add_st_operator(1.0, ['e2(i,a,j,b)'], ['t1', 't2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[o, v, v, o]',
                                    output_variables=['i', 'a', 'b', 'j']))
        print()
    pq.clear()


    print('\n', '#    D2(a,i,b,j):', '\n')
    pq.add_st_operator(1.0,['e2(a,i,j,b)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[v, o, v, o]',
                                    output_variables=['a', 'i', 'b', 'j']))
        print()
    pq.clear()


if __name__ == "__main__":
    main()
