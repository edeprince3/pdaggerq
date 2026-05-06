import pdaggerq

from pdaggerq.parser import contracted_strings_to_tensor_terms


def main():
    pq = pdaggerq.pq_helper("fermi")

    pq.set_unitary_cc(True)

    # energy equation

    print('')
    print('#    < 0 | e(-T) H e(T) | 0> :')
    print('')

    # up to 3rd-order

    pq.add_operator_product(1.0,['f'])

    pq.add_operator_product(1.0,['v'])

    pq.add_commutator(1.0/1.0,['f'],['t1'])
    pq.add_commutator(1.0/1.0,['f'],['t2'])

    pq.add_commutator(1.0/1.0,['v'],['t1'])
    pq.add_commutator(1.0/1.0,['v'],['t2'])

    pq.add_double_commutator(1.0/2.0,['f'],['t1'],['t2'])
    pq.add_double_commutator(1.0/2.0,['f'],['t2'],['t1'])
    pq.add_double_commutator(1.0/2.0,['f'],['t2'],['t2'])

    pq.add_double_commutator(1.0/2.0,['v'],['t2'],['t2'])

    pq.add_triple_commutator(1.0/6.0,['f'],['t2'],['t2'],['t2'])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    terms = pq.strings()

    for term in terms:
        print(term)

    pq.clear()

    # ucc3 singles equations

    pq.set_left_operators([['e1(m,e)']])

    print('')
    print('#    < 0 | m* e e(-T) H e(T) | 0> :')
    print('')

    # up to 1st-order

    pq.add_operator_product(1.0,['f'])

    pq.add_operator_product(1.0,['v'])

    pq.add_commutator(1.0/1.0,['f'],['t2'])

    pq.simplify()

    # grab list of fully-contracted strings, then print
    terms = pq.strings()
    for term in terms:
        print(term)

    pq.clear()

    # ucc3 doubles equations

    pq.set_left_operators([['e2(m,n,f,e)']])

    print('')
    print('#    < 0 | m* n* f e e(-T) H e(T) | 0> :')
    print('')

    # up to 2nd-order

    pq.add_operator_product(1.0,['f'])

    pq.add_operator_product(1.0,['v'])

    pq.add_commutator(1.0/1.0,['f'],['t1'])
    pq.add_commutator(1.0/1.0,['f'],['t2'])

    pq.add_commutator(1.0/1.0,['v'],['t2'])
    
    pq.add_double_commutator(1.0/2.0,['f'],['t2'],['t2'])
    
    pq.simplify()

    # grab list of fully-contracted strings, then print
    terms = pq.strings()
    for term in terms:
        print(term)

    pq.clear()

if __name__ == "__main__":
    main()
