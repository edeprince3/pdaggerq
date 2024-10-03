import pdaggerq
from extract_spins import *

def configure_graph():
    """
    Configure and return the pq_graph with specific settings.

    Returns:
        graph (pq_graph): Configured pq_graph object.
    """
    return pdaggerq.pq_graph({
        'batched': False,
        'print_level': 0,
        'opt_level': 6,
        'nthreads': -1,
    })

def derive_equation(proj_eqname, P, ops, coeffs, T, eqs):
    """
    Derive and simplify the equation for the given projection operator.

    Args:
        proj_eqname (str): Name of the projection equation.
        P (list): Projection operators.
        ops (list): Operators.
        coeffs (list): Coefficients for the operators.
        T (list): T-operators.
        eqs (dict): Dictionary to store the derived equations.
    """
    pq = pdaggerq.pq_helper("fermi")
    print("Deriving equation:", f"{proj_eqname} = <{P}| Hbar |0>", flush=True)

    pq.set_left_operators(P)
    for j, op in enumerate(ops):
        pq.add_st_operator(coeffs[j], op, T)

    pq.simplify()
    # block_by_spin(pq, proj_eqname, P, eqs)
    eqs[proj_eqname] = pq.clone()
    del pq

def main():
    """
    Main function to derive and simplify equations using pdaggerq library.
    """

    # Operators and their coefficients (fock matrix and two-electron integrals)
    ops = [['f'], ['v']]
    coeffs = [1.0, 1.0]

    # Cluster operators
    T = ['t1', 't2', 't3']

    # Projection operators for different equations
    proj = {
        "energy": [['1']],               # ground state energy
        "singles_resid":    [['e1(i,a)']],            # singles residual
        "doubles_resid":    [['e2(i,j,b,a)']],        # doubles residual
        "triples_resid":    [['e3(i,j,k,c,b,a)']],        # triples residual
    }

    # Dictionary to store the derived equations
    eqs = {}

    for proj_eqname, P in proj.items():
        # Derive normal and coherent state equations
        derive_equation(proj_eqname, P, ops, coeffs, T, eqs)

    # Enable and configure pq_graph
    graph = configure_graph()

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # Optimize and output the graph
    graph.optimize()
    graph.print("cpp")
    graph.analysis()

    return graph

if __name__ == "__main__":
    main()
