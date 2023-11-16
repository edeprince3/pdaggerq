# ci coefficients from cluster amplitudes
# c_i = <i| exp(t) | 0>

import pdaggerq

def main():

    pq = pdaggerq.pq_helper("fermi")

    # T = T1 + T2 + T3 + T4
    T = ['t1', 't2', 't3', 't4']

    # exp(T) ... expanded up to fourth order ... good enough to get c4
    eT = []

    # order = 0
    eT.append([1.0, ['1']])

    # order = 1
    for my_T in T :
        eT.append([1.0, [my_T]])

    # order = 2
    for my_T1 in T :
        for my_T2 in T :
            eT.append([0.5, [my_T1, my_T2]])

    # order = 3
    for my_T1 in T :
        for my_T2 in T :
            for my_T3 in T :
                eT.append([1.0 / 6.0, [my_T1, my_T2, my_T3]])

    # order = 4
    for my_T1 in T :
        for my_T2 in T :
            for my_T3 in T :
                for my_T4 in T :
                    eT.append([1.0 / 24.0, [my_T1, my_T2, my_T3, my_T4]])

    print('')
    print('#    c(a,i) = <0|i* a e(T)|0> |0>')
    print('')

    pq.set_left_operators([['e1(i,a)']])
    pq.set_right_operators([['1']])

    for term in eT:
        pq.add_operator_product(term[0], term[1])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    terms = pq.fully_contracted_strings()
    for my_term in terms:
        print(my_term)

    pq.clear()

    print('')
    print('#    c(ab,ij) = <0|i* j* b a e(T)|0> |0>')
    print('')

    pq.set_left_operators([['e2(i,j,b,a)']])
    pq.set_right_operators([['1']])

    for term in eT:
        pq.add_operator_product(term[0], term[1])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    terms = pq.fully_contracted_strings()
    for my_term in terms:
        print(my_term)

    pq.clear()

    print('')
    print('#    c(abc,ijk) = <0|i* j* k* c b a e(T)|0> |0>')
    print('')

    pq.set_left_operators([['e3(i,j,k,c,b,a)']])
    pq.set_right_operators([['1']])

    for term in eT:
        pq.add_operator_product(term[0], term[1])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    terms = pq.fully_contracted_strings()
    for my_term in terms:
        print(my_term)

    pq.clear()

    print('')
    print('#    c(abcd,ijkj) = <0|i* j* k* l* d c b a e(T)|0> |0>')
    print('')

    pq.set_left_operators([['e4(i,j,k,l,d,c,b,a)']])
    pq.set_right_operators([['1']])

    for term in eT:
        pq.add_operator_product(term[0], term[1])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    terms = pq.fully_contracted_strings()
    for my_term in terms:
        print(my_term)

    pq.clear()

if __name__ == "__main__":
    main()
