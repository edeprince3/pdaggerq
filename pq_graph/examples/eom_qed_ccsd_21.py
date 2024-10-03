import pdaggerq
from extract_spins import *

def derive_equation(eqs, proj_eqname, P, ops, coeffs, T, use_coherent=False, type="ground", S=None):
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

    op_str = "Hbar"
    if use_coherent:
        proj_eqname = "c" + proj_eqname
        op_str = "B* + B"

    if type == "ground":
        print("Deriving equation:", f"{proj_eqname} = <{P}| {op_str} |0>", flush=True)
    elif (type == "right"):
        print("Deriving equation:", f"{proj_eqname} = <{P}| {op_str} |{S}>", flush=True)
    elif (type == "left"):
        print("Deriving equation:", f"{proj_eqname} = <{S}| {op_str} |{P}>", flush=True)
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
        'batched': True,
        'print_level': 3,
        'opt_level': 0,
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
    R = [
            #['r0'], 
        ['r1'], ['r2'],
        ['r0,1'], ['r1,1'], ['r2,1'],
    ]
    L = [
            #['l0'], 
        ['l1'], ['l2'],
        ['l0,1'], ['l1,1'], ['l2,1'],
    ]

    # Projection operators for different equations
    rproj = {
            #"sigmar0":    [['1']],                  # ground state energy
        "sigmar1":    [['e1(i,a)']],            # singles residual
        "sigmar2":    [['e2(i,j,b,a)']],        # doubles residual
        "sigmar0_1":  [['B-']],                 # ground state + hw
        "sigmar1_1":  [['B-', 'e1(i,a)']],      # singles residual + hw
        "sigmar2_1":  [['B-', 'e2(i,j,b,a)']],  # doubles residual + hw
    }

    lproj = {
            #"sigmal0":    [['1']],                  # ground state energy
        "sigmal1":    [['e1(a,i)']],            # singles residual
        "sigmal2":    [['e2(a,b,j,i)']],        # doubles residual
        "sigmal0_1":  [['B+']],                 # ground state + hw
        "sigmal1_1":  [['e1(a,i)','B+']],      # singles residual + hw
        "sigmal2_1":  [['e2(a,b,j,i)','B+']],  # doubles residual + hw
    }

    # Dictionary to store the derived equations
    eqs = {}

    for proj_eqname, P in rproj.items():
        # Derive normal and coherent state equations
        derive_equation(eqs, proj_eqname, P,   ops,   coeffs, T, use_coherent=False, type="right", S=R)
        derive_equation(eqs, proj_eqname, P, c_ops, c_coeffs, T,  use_coherent=True, type="right", S=R)
        print()

    for proj_eqname, P in lproj.items():
        derive_equation(eqs, proj_eqname, P,   ops,   coeffs, T, use_coherent=False, type="left", S=L)
        derive_equation(eqs, proj_eqname, P, c_ops, c_coeffs, T,  use_coherent=True, type="left", S=L)
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
