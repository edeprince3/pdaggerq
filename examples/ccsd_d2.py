
import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms


def main():
    pq = pdaggerq.pq_helper("fermi")

    # D2(p,q,r,s) = <0|(1 + l1 + l2) e(-T) p*q*sr e(T) |0>
    pq.set_left_operators(['1','l1','l2'])

    print('\n', '#    D2(i,j,k,l):', '\n')
    pq.add_st_operator(1.0,['e2(i,j,l,k)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.fully_contracted_strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[o, o, o, o]',
                                    output_variables=['i', 'j', 'k', 'l']))
        print()
    pq.clear()

    print('\n', '#    D2(a,b,c,d):', '\n')
    pq.add_st_operator(1.0,['e2(a,b,d,c)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.fully_contracted_strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[v, v, v, v]',
                                    output_variables=['a', 'b', 'c', 'd']))
        print()
    pq.clear()

    print('\n', '#    D2(i,j,a,b):', '\n')
    pq.add_st_operator(1.0,['e2(i,j,b,a)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.fully_contracted_strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[o, o, v, v]',
                                    output_variables=['i', 'j', 'a', 'b']))
        print()
    pq.clear()

    print('\n', '#    D2(a,b,i,j):', '\n')
    pq.add_st_operator(1.0,['e2(a,b,j,i)'],['t1','t2'])
    pq.simplify()
    d2_terms = pq.fully_contracted_strings()
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
    d2_terms = pq.fully_contracted_strings()
    d2_terms = contracted_strings_to_tensor_terms(d2_terms)
    for my_term in d2_terms:
        print("#\t", my_term)
        print(my_term.einsum_string(update_val='tpdm[o, v, o, v]',
                                    output_variables=['i', 'a', 'j', 'b']))
        print()
    pq.clear()

if __name__ == "__main__":
    main()