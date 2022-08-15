"""
Example for vacuum normal ordering the T2 operator for 2-RDM theory
"""
import pdaggerq

def main():
    print("T2 mappings")
    pq = pdaggerq.pq_helper('true')
    pq.set_string(['i*','j*','k','n*','m', 'l'])
    pq.add_new_string()
    pq.set_string(['n*','m','l', 'i*','j*', 'k'])
    pq.add_new_string()
    pq.simplify()
    pq.print()
    pq.clear()

    print("T1 mappings")
    pq = pdaggerq.pq_helper('true')
    pq.set_string(['i*','j*','k*','n','m', 'l'])
    pq.add_new_string()
    pq.set_string(['n','m','l', 'i*','j*', 'k*'])
    pq.add_new_string()
    pq.simplify()
    pq.print()
    pq.clear()

    print("Q -> D")
    pq = pdaggerq.pq_helper('true')

    pq.set_string(['i', 'j', 'k*', 'l*'])
    pq.add_new_string()

    pq.simplify()
    pq.print()

    pq.clear()

    print("G -> D")
    pq = pdaggerq.pq_helper('true')
    pq.set_string(['i*', 'j', 'k*', 'l'])
    pq.add_new_string()
    pq.simplify()
    pq.print()
    pq.clear()

if __name__ == "__main__":
    main()
