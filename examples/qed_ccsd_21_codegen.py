import pdaggerq
from pdaggerq.parser import contracted_strings_to_tensor_terms
import re

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

    # indices pattern looks like '[...]([a-zA-Z],...,[a-zA-Z])'
    # with ChatGPT's help:
    # (?<=\() : Positive lookbehind to match ( but not include it in the result.
    # [^)]+   : Match one or more characters that are not a closing parenthesis ).
    # (?=\))  : Positive lookahead to match ) but not include it in the result.
    idx_pattern  = re.compile('(?<=\()[^)]+(?=\))')

    idx = idx_pattern.findall(L[0][-1])
    if len(idx)==0:
        idx = None
    else:
        tmp = idx[0].split(',')
        idx = []
        for i in range(len(tmp)//2):
            idx.append(tmp[-1-i])
        for i in range(len(tmp)//2):
            idx.append(tmp[i])

        idx = tuple(idx)

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
        block_by_spin(pq, proj_eqname, L + R + T + ops, eqs, idx)
    else:
        eqs[proj_eqname] = pq.clone()
        # print the fully contracted strings
        print(f"Equation {proj_eqname}:", flush=True)
        terms = pq.strings()
        terms = contracted_strings_to_tensor_terms(terms)
        for term in terms:
            print(f"# {term}", flush=True)
            if idx==None:
                print("# {}".format(term.einsum_string(update_val=proj_eqname)), flush=True)
            else:
                print("# {}".format(term.einsum_string(update_val=proj_eqname,output_variables=idx)), flush=True)
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

def get_spin_labels(ops):
    """
    Get spin labels for the given operators.

    Args:
        ops (list): List of operators.

    Returns:
        dict: Dictionary mapping spin types to label-spin mappings.
    """
    spin_map = {}
    labels = set()
    found = False

    # find all labels in the operators
    for op in ops:
        for subop in op:
            # no labels in the operator
            if "(" not in subop:
                continue

            # extract labels from the operator
            subop_labels = subop[subop.find("(") + 1:subop.find(")")].split(",")
            for label in subop_labels:
                # add the label to the set
                labels.add(label)
                found = True

    # no labels found in the operators; no spin blocking
    if not found:
        return {"": {}}

    # sort the labels and create spin types based on the number of unique labels
    labels = sorted(labels)
    spin_types = ["aaaa", "abab", "bbbb"] if len(labels) == 4 else (
        ["aaa", "abb", "aba", "bbb"] if len(labels) == 3 else (
            ["aa", "bb"] if len(labels) == 2 else (
                ["a", "b"] if len(labels) == 1 else []
            )
        )
    )

    # create a mapping of labels to spins for each spin type
    for spin in spin_types:
        if len(labels) != len(spin):
            continue
        label_to_spin = {label: spin[i] for i, label in enumerate(labels)}
        spin_map[spin] = label_to_spin

    return spin_map

def block_by_spin(pq, eqname, ops, eqs, idx=None):
    """
    Block the equation by spin and store the result in the equations dictionary.

    Args:
        pq (pq_helper): pdaggerq helper object.
        eqname (str): Name of the equation.
        ops (list): List of operators.
        eqs (dict): Dictionary to store the derived equations.
    """
    spin_map = get_spin_labels(ops)

    # print the blocking by spin
    print("Blocking by spin:", flush=True)
    for spins, label_to_spin in spin_map.items():
        print(f"{spins} ->", ", ".join(f"{label} -> {spin}" for label, spin in label_to_spin.items()), flush=True)
    print()

    # create equations for each spin block
    for spins, label_to_spin in spin_map.items():
        spin_eqname = eqname if spins == "" else eqname + "_" + spins
        pq.block_by_spin(label_to_spin)

        # store the equation in the dictionary
        eqs[spin_eqname] = pq.clone()

        # print the fully contracted strings
        print(f"Equation {spin_eqname}:", flush=True)
        terms = pq.strings()
        terms = contracted_strings_to_tensor_terms(terms)
        for term in terms:
            print(f"# {term}", flush=True)
            if idx == None:
                print("# {}".format(term.einsum_string(update_val=spin_eqname)), flush=True)
            else:
                print("# {}".format(term.einsum_string(update_val=spin_eqname, output_variables=idx)), flush=True)

if __name__ == "__main__":
    main()
