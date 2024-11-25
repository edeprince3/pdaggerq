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

    # cluster operators
    T = ['t1', 't2']

    # left and right excitation operators
    L = [['l0'], ['l1'], ['l2']]
    R = [['r0'], ['r1'], ['r2']]

    rdms = {
        "D_vv": [['e1(a,b)']], # vv
        "D_vo": [['e1(a,i)']], # vo
        "D_ov": [['e1(i,a)']], # ov
        "D_oo": [['e1(i,j)']], # oo
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
    graph.add(eqs["D_vv_aa"], "D_vv_aa", ['a', 'b'])
    graph.add(eqs["D_vo_aa"], "D_vo_aa", ['a', 'i'])
    graph.add(eqs["D_ov_aa"], "D_ov_aa", ['i', 'a'])
    graph.add(eqs["D_oo_aa"], "D_oo_aa", ['i', 'j'])
    
    graph.add(eqs["D_vv_bb"], "D_vv_bb", ['a', 'b'])
    graph.add(eqs["D_vo_bb"], "D_vo_bb", ['a', 'i'])
    graph.add(eqs["D_ov_bb"], "D_ov_bb", ['i', 'a'])
    graph.add(eqs["D_oo_bb"], "D_oo_bb", ['i', 'j'])


    # Optimize and output the graph
    graph.optimize()
    graph.print("c++")
    graph.analysis()

    return graph

if __name__ == "__main__":
    main()
