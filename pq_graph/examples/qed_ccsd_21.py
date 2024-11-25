import pdaggerq
from extract_spins import *

def derive_equation(eqs, proj_eqname, ops, coeffs, L = None, R = None, T = None, spin_block = False):
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

    if L is None:
        L = [['1']]
    if R is None:
        R = [['1']]

    # determine if the projections should be applied to the right or left
    print("Deriving equation:", f"{proj_eqname} = <{L}| {ops} |{R}>", flush=True)

    pq.set_left_operators( L)
    pq.set_right_operators(R)

    for j, op in enumerate(ops):
        if T is None:
            pq.add_operator(coeffs[j], op)
        else:
            pq.add_st_operator(coeffs[j], op, T)
    pq.simplify()

    if spin_block:
        block_by_spin(pq, proj_eqname, L + R + T + ops, eqs)
    else:
        eqs[proj_eqname] = pq.clone()
        # print the fully contracted strings
        print(f"Equation {proj_eqname}:", flush=True)
        for term in pq.strings():
            print(term, flush=True)
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
        'use_trial_index': True,
        'opt_level': 6,
        'nthreads': -1,
    })

def main():
    """
    Main function to derive and simplify equations using pdaggerq library.
    """

    # Operators and their coefficients
    ops = [['w0'], ['d+'], ['d-'], ['f'], ['v']]
    coeffs = [1.0, -1.0, -1.0, 1.0, 1.0]

    # Coherent state operators and their coefficients
    c_ops = [['B+'], ['B-']]
    c_coeffs = [1.0, 1.0]

    # T-operators (assumes T1 transformed integrals)
    T = ['t2', 't0,1', 't1,1', 't2,1']

    # Projection operators for different equations
    proj = {
        "energy": [['1']],                       # ground state energy
        "rt1": [['e1(i,a)']],                    # singles residual
        "rt2": [['e2(i,j,b,a)']],                # doubles residual

        "rt0_1": [['B-']],                       # ground state + hw
        "rt1_1": [['B-', 'e1(i,a)']],            # singles residual + hw
        "rt2_1": [['B-', 'e2(i,j,b,a)']],        # doubles residual + hw
    }

    # Dictionary to store the derived equations
    eqs = {}

    for proj_eqname, P in proj.items():
        # residual equations
        derive_equation(eqs, proj_eqname, ops, coeffs, L=P, T=T, spin_block=True)
        derive_equation(eqs, "c" + proj_eqname, c_ops, c_coeffs, L=P, T=T, spin_block=True)
        print()

    # Enable and configure pq_graph
    graph = configure_graph()

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # Optimize and output the graph
    graph.optimize()
    graph.print("c++")
    graph.analysis()

    return graph

if __name__ == "__main__":
    main()
