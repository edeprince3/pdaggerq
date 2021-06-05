"""
Example for vacuum normal ordering the T2 operator for 2-RDM theory
"""
import pdaggerq

def main():
    print("T2 mappings")
    ahat = pdaggerq.pq_helper('true')
    ahat.set_string(['i*','j*','k','n*','m', 'l'])
    ahat.add_new_string()
    ahat.set_string(['n*','m','l', 'i*','j*', 'k'])
    ahat.add_new_string()
    ahat.simplify()
    ahat.print()
    ahat.clear()

    print("T1 mappings")
    ahat = pdaggerq.pq_helper('true')
    ahat.set_string(['i*','j*','k*','n','m', 'l'])
    ahat.add_new_string()
    ahat.set_string(['n','m','l', 'i*','j*', 'k*'])
    ahat.add_new_string()
    ahat.simplify()
    ahat.print()
    ahat.clear()

    print("Q -> D")
    ahat = pdaggerq.pq_helper('true')

    ahat.set_string(['i', 'j', 'k*', 'l*'])
    ahat.add_new_string()

    ahat.simplify()
    ahat.print()

    ahat.clear()

    print("G -> D")
    ahat = pdaggerq.pq_helper('true')
    ahat.set_string(['i*', 'j', 'k*', 'l'])
    ahat.add_new_string()
    ahat.simplify()
    ahat.print()
    ahat.clear()

if __name__ == "__main__":
    main()