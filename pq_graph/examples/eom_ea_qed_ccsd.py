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

    pq.set_left_operators_type("EA")
    pq.set_right_operators_type("EA")

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
        for term in pq.fully_contracted_strings():
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
        'print_level': 3,
        'use_trial_index': True,
        'separate_sigma': True,
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

    # Exciation operators (truncated ground state)
    R = [
        ['r1'], 
        ['r2'],
        ['r1,1'],
        ['r2,1'],
    ]
    L = [
        ['l1'], 
        ['l2'],
        ['l1,1'], 
        ['l2,1'],
    ]

    # Projection operators for different equations
    rproj = {
        "sigmar1":    [['a(a)']],                       # singles residual
        "sigmar2":    [['a*(i)','a(b)','a(a)']],        # doubles residual
        "sigmar1_1":  [['B-', 'a(a)']],                 # singles residual + hw
        "sigmar2_1":  [['B-', 'a*(i)','a(b)','a(a)']],  # doubles residual + hw
    }

    lproj = {
        "sigmal1":    [['a*(a)']],                       # singles residual
        "sigmal2":    [['a*(a)','a*(b)','a(i)']],        # doubles residual
        "sigmal1_1":  [['a*(a)','B+']],                  # singles residual + hw
        "sigmal2_1":  [['a*(a)','a*(b)','a(i)','B+']],   # doubles residual + hw
    }

    # Dictionary to store the derived equations
    eqs = {}

    for proj_eqname, P in rproj.items():
        # right projected equations
        derive_equation(eqs, proj_eqname, ops, coeffs, L=P, R=R, T=T, spin_block=True)
        derive_equation(eqs, "c" + proj_eqname, c_ops, c_coeffs, L=P, R=R, T=T, spin_block=True)
        print()

    for proj_eqname, P in lproj.items():
        # left projected equations
        derive_equation(eqs, proj_eqname, ops, coeffs, L=L, R=P, T=T, spin_block=True)
        derive_equation(eqs, "c" + proj_eqname, c_ops, c_coeffs, L=L, R=P, T=T, spin_block=True)
        print()

    # Enable and configure pq_graph
    graph = configure_graph()

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname, ['a', 'b', 'i', 'j'])

    # Optimize and output the graph
    graph.optimize()
    graph.print("c++")
    graph.analysis()

    return graph

if __name__ == "__main__":
    main()
