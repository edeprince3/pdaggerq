import pdaggerq
from extract_spins import *

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
    print("Deriving equation:", f"{proj_eqname} = <{P}| {ops} |0>", flush=True)

    pq.set_left_operators(P)
    for j, op in enumerate(ops):
        pq.add_st_operator(coeffs[j], op, T)

    pq.simplify()
    block_by_spin(pq, proj_eqname, P, eqs)
    del pq

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
    T = ['t2', 't0,1', 't1,1', 't2,1', 't0,2', 't1,2', 't2,2']

    # Projection operators for different equations
    proj = {
        "energy": [['1']],               # ground state energy
        "rt1":    [['e1(i,a)']],            # singles residual
        "rt2":    [['e2(i,j,b,a)']],        # doubles residual
        "rt0_1":  [['B-']],                 # ground state + hw
        "rt1_1":  [['B-', 'e1(i,a)']],      # singles residual + hw
        "rt2_1":  [['B-', 'e2(i,j,b,a)']],  # doubles residual + hw
        "rt0_2":  [['B-','B-']],                 # ground state + hw
        "rt1_2":  [['B-','B-', 'e1(i,a)']],      # singles residual + hw
        "rt2_2":  [['B-','B-', 'e2(i,j,b,a)']],  # doubles residual + hw
    }

    # Dictionary to store the derived equations
    eqs = {}

    for proj_eqname, P in proj.items():
        # Derive normal and coherent state equations
        derive_equation(proj_eqname, P, ops, coeffs, T, eqs)
        derive_equation("c_" + proj_eqname, P, c_ops, c_coeffs, T, eqs)


if __name__ == "__main__":
    main()
