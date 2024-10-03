import pdaggerq
from extract_spins import *

def derive_equation(eqs, proj_eqname, P, ops, coeffs, T, type, S=None):
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

    # determine if the projections should be applied to the right or left
    if (type == "right"):
        print("Deriving equation:", f"{proj_eqname} = <{P}| Hbar |{S}>", flush=True)
    elif (type == "left"):
        print("Deriving equation:", f"{proj_eqname} = <{S}| Hbar |{P}>", flush=True)
    else:
        raise ValueError("Invalid type for building the sigma equations")

    if type is None:
        pq.set_left_operators(P)
        pq.set_right_operators([["1"]])
    elif type == "right":
        pq.set_left_operators(P)
        pq.set_right_operators(S)
    elif type == "left":
        pq.set_left_operators(S)
        pq.set_right_operators(P)

    for j, op in enumerate(ops):
        pq.add_st_operator(coeffs[j], op, T)

    pq.simplify()
    block_by_spin(pq, proj_eqname, P, eqs)
    del pq

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

def main():
    """
    Main function to derive and simplify equations using pdaggerq library.
    """

    # Operators and their coefficients
    ops = [['f'], ['v']]
    coeffs = [1.0, 1.0]

    # T-operators (assumes T1 transformed integrals)
    T = ['t2']

    # Exciation operators (truncated ground state)
    R = [
            #['r0'], 
        ['r1'], 
        ['r2'],
    ]
    L = [
            #['l0'], 
        ['l1'], 
        ['l2'],
    ]

    # Projection operators for different equations
    rproj = {
        "sigmar1":    [['e1(i,a)']],            # singles residual
        "sigmar2":    [['e2(i,j,b,a)']],        # doubles residual
    }

    lproj = {
        "sigmal1":    [['e1(a,i)']],            # singles residual
        "sigmal2":    [['e2(a,b,j,i)']],        # doubles residual
    }

    # Dictionary to store the derived equations
    eqs = {}

    for proj_eqname, P in rproj.items():
        # Derive normal and coherent state equations
        derive_equation(eqs, proj_eqname, P,   ops,   coeffs, T, "right", S=R)
        print()

    for proj_eqname, P in lproj.items():
        derive_equation(eqs, proj_eqname, P,   ops,   coeffs, T, "left", S=L)
        print()

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
