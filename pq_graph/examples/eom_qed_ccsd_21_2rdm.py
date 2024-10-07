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
        'print_level': 0,
        'opt_level': 6,
        'use_trial_index': True,
        'nthreads': -1,
    })

def main():
    """
    Main function to derive and simplify equations using pdaggerq library.
    """

    # cluster operators
    T = ['t1', 't2', 't0,1', 't1,1', 't2,1']

    # left and right excitation operators
    L = [['l0'], ['l1'], ['l2'], ['l0,1'], ['l1,1'], ['l2,1']]
    R = [['r0'], ['r1'], ['r2'], ['r0,1'], ['r1,1'], ['r2,1']]

    rdms = {
        "D_vvvv": [['e2(a,b,c,d)']], # vvvv
        "D_vvvo": [['e2(a,b,c,i)']], # vvvo
        "D_vvov": [['e2(a,b,i,c)']], # vvov
        "D_vovv": [['e2(a,i,b,c)']], # vovv
        "D_ovvv": [['e2(i,a,b,c)']], # ovvv
        "D_ovvo": [['e2(i,a,b,j)']], # ovvo
        "D_ovov": [['e2(i,a,j,b)']], # ovov
        "D_oovv": [['e2(i,j,a,b)']], # oovv
        "D_oovo": [['e2(i,j,a,k)']], # oovo
        "D_ooov": [['e2(i,j,k,a)']], # ooov
        "D_oooo": [['e2(i,j,k,l)']], # oooo
    }

    # Dictionary to store the derived equations
    eqs = {}

    for eqname, rdm in rdms.items():
        # right projected equations
        derive_equation(eqs, eqname, rdm, [1.0], L=L, R=R, T=T, spin_block=True)
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
