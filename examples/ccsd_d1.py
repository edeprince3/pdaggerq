# elements of the one-electron reduced density matrix 
# at the CCSD level of theory: D(pq) = <psi|(1+lambda) e(-T) p*q e(T) |psi> 

import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms


def main():
    pq = pdaggerq.pq_helper("fermi")

    print('')
    print('#    D1(m,n):')
    print('')

    # D(mn) = <psi|(1+l1 + l2) e(-T) e(m,n) e(T) |psi>
    pq.set_left_operators(['1','l1','l2'])
    pq.add_st_operator(1.0,['e1(m,n)'],['t1','t2'])
    pq.simplify()

    # grab list of fully-contracted strings, then print
    d1_terms = pq.fully_contracted_strings()
    d1_terms = contracted_strings_to_tensor_terms(d1_terms)
    for my_term in d1_terms:
        print("# \t", my_term)
        print(my_term.einsum_string(update_val='opdm[o, o]',
                                    output_variables=('m', 'n')))
        print()
    pq.clear()


    print('')
    print('#    D1(e,f):')
    print('')

    pq.set_left_operators(['1','l1','l2'])
    pq.add_st_operator(1.0,['e1(e,f)'],['t1','t2'])
    pq.simplify()

    # grab list of fully-contracted strings, then print
    d1_terms = pq.fully_contracted_strings()
    d1_terms = contracted_strings_to_tensor_terms(d1_terms)
    for my_term in d1_terms:
        print("# \t", my_term)
        print(my_term.einsum_string(update_val='opdm[v, v]',
                                    output_variables=('e', 'f')))
        print()
    pq.clear()

    print('')
    print('#    D1(e,m):')
    print('')

    pq.set_left_operators(['1','l1','l2'])
    pq.add_st_operator(1.0,['e1(e,m)'],['t1','t2'])
    pq.simplify()

    # grab list of fully-contracted strings, then print
    d1_terms = pq.fully_contracted_strings()
    d1_terms = contracted_strings_to_tensor_terms(d1_terms)
    for my_term in d1_terms:
        print("# \t", my_term)
        print(my_term.einsum_string(update_val='opdm[v, o]',
                                    output_variables=('e', 'm')))
        print()
    pq.clear()

    print('')
    print('#    D1(m,e):')
    print('')

    pq.set_left_operators(['1','l1','l2'])
    pq.add_st_operator(1.0,['e1(m,e)'],['t1','t2'])
    pq.simplify()

    # grab list of fully-contracted strings, then print
    # note, this will be sorted e,m output so user must transpose
    d1_terms = pq.fully_contracted_strings()
    d1_terms = contracted_strings_to_tensor_terms(d1_terms)
    for my_term in d1_terms:
        print("# \t", my_term)
        print(my_term.einsum_string(update_val='opdm[o, v]',
                                    output_variables=('m', 'e')))
        print()

    pq.clear()


if __name__ == "__main__":
    main()