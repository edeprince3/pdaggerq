"""
Example for vacuum normal ordering the T2 operator for 2-RDM theory
"""
import pdaggerq

def main():

    print("T2 mappings")
    pq = pdaggerq.pq_helper('true')
    pq.add_anticommutator(1.0, ['a*(i)', 'a*(j)', 'a(k)'], ['a*(n)', 'a(m)', 'a(l)'])

    pq.simplify()
    terms = pq.strings()
    for term in terms:
        print(term)
    pq.clear()

    print("T1 mappings")
    pq.add_anticommutator(1.0, ['a*(i)', 'a*(j)', 'a*(k)'], ['a(n)', 'a(m)', 'a(l)'])
    pq.simplify()
    terms = pq.strings()
    for term in terms:
        print(term)
    pq.clear()

    print("Q -> D")
    pq.add_operator_product(1.0, ['a(i)', 'a(j)', 'a*(k)', 'a*(l)'])
    pq.simplify()
    terms = pq.strings()
    for term in terms:
        print(term)
    pq.clear()

    print("G -> D")
    pq.add_operator_product(1.0, ['a*(i)', 'a(j)', 'a*(k)', 'a(l)'])
    pq.simplify()
    terms = pq.strings()
    for term in terms:
        print(term)
    pq.clear()

if __name__ == "__main__":
    main()
