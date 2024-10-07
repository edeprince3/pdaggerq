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
        # right projected equations
        derive_equation(eqs, proj_eqname, ops, coeffs, L=P, R=R, T=T, spin_block=True)
        derive_equation(eqs, "c" + proj_eqname, c_ops, c_coeffs, L=P, R=R, T=T, spin_block=True)
        print()

    for proj_eqname, P in lproj.items():
        # left projected equations
        derive_equation(eqs, proj_eqname, ops, coeffs, L=L, R=P, T=T, spin_block=True)
        derive_equation(eqs, "c" + proj_eqname, c_ops, c_coeffs, L=L, R=P, T=T, spin_block=True)
        print()

if __name__ == "__main__":
    main()
