"""
Example for vacuum normal ordering the T2 operator for 2-RDM theory
"""
import pdaggerq

def main():

    print("T2 mappings")
    pq = pdaggerq.pq_helper('true')
    pq.add_operator_product(1.0, ['a*(i)', 'a*(j)', 'a(k)', 'a*(n)', 'a(m)', 'a(l)'])
    pq.add_operator_product(1.0, ['a*(n)', 'a(m)', 'a(l)', 'a*(i)', 'a*(j)', 'a(k)'])
    pq.simplify()
    pq.print(string_type = 'all')
    pq.clear()

    print("T1 mappings")
    pq.add_operator_product(1.0, ['a*(i)', 'a*(j)', 'a*(k)', 'a(n)', 'a(m)', 'a(l)'])
    pq.add_operator_product(1.0, ['a(n)', 'a(m)', 'a(l)', 'a*(i)', 'a*(j)', 'a*(k)'])
    pq.simplify()
    pq.print(string_type = 'all')
    pq.clear()

    print("Q -> D")
    pq.add_operator_product(1.0, ['a(i)', 'a(j)', 'a*(k)', 'a*(l)'])
    pq.simplify()
    pq.print(string_type = 'all')
    pq.clear()

    print("G -> D")
    pq.add_operator_product(1.0, ['a*(i)', 'a(j)', 'a*(k)', 'a(l)'])
    pq.simplify()
    pq.print(string_type = 'all')
    pq.clear()

if __name__ == "__main__":
    main()
