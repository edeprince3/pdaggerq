import pdaggerq

def main():
    pq = pdaggerq.pq_helper("fermi")

    pq.set_unitary_cc(True)

    # energy equation

    print('')
    print('#    < 0 | e(-T) H e(T) | 0> :')
    print('')

    # up to 4th-order

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

    terms = pq.strings()
    for term in terms:
        print(term)

    pq.clear()

    # ucc4 singles equations

    pq.set_left_operators([['e1(m,e)']])

    print('')
    print('#    < 0 | m* e e(-T) H e(T) | 0> :')
    print('')

    # up to 2nd order
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
    for term in terms:
        print(term)

    pq.clear()

    # ucc4 doubles equations

    pq.set_left_operators([['e2(m,n,f,e)']])

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
    for term in terms:
        print(term)

    pq.clear()

if __name__ == "__main__":
    main()
