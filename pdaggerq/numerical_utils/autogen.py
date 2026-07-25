import pdaggerq 
import re

# Map fermionic order -> list of spin channels
SPIN_MAP = {
    0: [''],
    1: ['aa', 'bb'],
    2: ['aaaa', 'abab', 'bbbb'],
    3: ['aaaaaa', 'aabaab', 'abbabb', 'bbbbbb'],
    4: ['aaaaaaaa', 'aaabaaab', 'aabbaabb', 'abbbabbb', 'bbbbbbbb'],
}

def configure_graph(options = None):
    """
    Configure and return the pq_graph with specific settings.

    Returns:
        graph (pq_graph): Configured pq_graph object.
    """

    if options is None:
        options = {
            'batched': False,
            #'batched': True,
            #'batch_number': 100,
            'print_level': 0,
            'opt_level': 0,
            'nthreads': -1,
            'no_scalars': False,
            #'permute_eri': False,
        }

    return pdaggerq.pq_graph(options)

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
    spin_types = ["aaaaaa", "aabaab", "abbabb", "bbbbbb"] if len(labels) == 6 else (
        ["aaaaa", "aabaa", "abbab", "bbbbb"] if len(labels) == 5 else (
            ["aaaa", "abab", "bbbb"] if len(labels) == 4 else (
                ["aaa", "abb", "aba", "bbb"] if len(labels) == 3 else (
                    ["aa", "bb"] if len(labels) == 2 else (
                        ["a", "b"] if len(labels) == 1 else []
                    )
                )
            )
        )
    )

    if spin_types == [] and len(labels) != 0:
        raise ValueError("Invalid number of labels for spin blocking")

    # create a mapping of labels to spins for each spin type
    for spin in spin_types:
        if len(labels) != len(spin):
            continue
        label_to_spin = {label: spin[i] for i, label in enumerate(labels)}
        spin_map[spin] = label_to_spin

    return spin_map

def block_by_spin(pq, eqname, ops, eqs):
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
        for term in pq.strings():
            print(term, flush=True)

def function_initialization_string(extra_class = "", is_qed = False):
    """
    generate a string containing information required for function initialization

    :param extra_class: do amplitudes live in self or self.extra_class?
    :param is_qed: include qed-cc terms? 

    """
    if extra_class != "":
        extra_class += "."
    ret_string = \
f"""
    import numpy as np
    from numpy import einsum

    # cluster amplitudes

    t1 = dict(self.{extra_class}T.get('1', {{}}))
    t2 = dict(self.{extra_class}T.get('2', {{}}))
    t3 = dict(self.{extra_class}T.get('3', {{}}))
    t4 = dict(self.{extra_class}T.get('4', {{}}))
    
    # Photon-Coupled Amplitudes (1 Photon)

    # Photon creation only is special because it is a scalar
    t0_1p_dict = self.{extra_class}T.get('0_1p', {{}})
    t0_1p_val = t0_1p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    t0_1p = t0_1p_val.item() if hasattr(t0_1p_val, 'item') else t0_1p_val

    t1_1p = dict(self.{extra_class}T.get('1_1p', {{}}))
    t2_1p = dict(self.{extra_class}T.get('2_1p', {{}}))
    t3_1p = dict(self.{extra_class}T.get('3_1p', {{}}))
    t4_1p = dict(self.{extra_class}T.get('4_1p', {{}}))

    # Photon-Coupled Amplitudes (2 Photon)

    # Photon creation only is special because it is a scalar
    t0_2p_dict = self.{extra_class}T.get('0_2p', {{}})
    t0_2p_val = t0_2p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    t0_2p = t0_2p_val.item() if hasattr(t0_2p_val, 'item') else t0_2p_val

    t1_2p = dict(self.{extra_class}T.get('1_2p', {{}}))
    t2_2p = dict(self.{extra_class}T.get('2_2p', {{}}))
    t3_2p = dict(self.{extra_class}T.get('3_2p', {{}}))
    t4_2p = dict(self.{extra_class}T.get('4_2p', {{}}))

    # Photon-Coupled Amplitudes (3 Photon)

    # Photon creation only is special because it is a scalar
    t0_3p_dict = self.{extra_class}T.get('0_3p', {{}})
    t0_3p_val = t0_3p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    t0_3p = t0_3p_val.item() if hasattr(t0_3p_val, 'item') else t0_3p_val

    t1_3p = dict(self.{extra_class}T.get('1_3p', {{}}))
    t2_3p = dict(self.{extra_class}T.get('2_3p', {{}}))
    t3_3p = dict(self.{extra_class}T.get('3_3p', {{}}))
    t4_3p = dict(self.{extra_class}T.get('4_3p', {{}}))

    # Photon-Coupled Amplitudes (4 Photon)

    # Photon creation only is special because it is a scalar
    t0_4p_dict = self.{extra_class}T.get('0_4p', {{}})
    t0_4p_val = t0_4p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    t0_4p = t0_4p_val.item() if hasattr(t0_4p_val, 'item') else t0_4p_val

    t1_4p = dict(self.{extra_class}T.get('1_4p', {{}}))
    t2_4p = dict(self.{extra_class}T.get('2_4p', {{}}))
    t3_4p = dict(self.{extra_class}T.get('3_4p', {{}}))
    t4_4p = dict(self.{extra_class}T.get('4_4p', {{}}))

    # lambda amplitudes

    l1 = {{spin: tensor.transpose(1, 0) for spin, tensor in self.{extra_class}L.get('1', {{}}).items()}}
    l2 = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.{extra_class}L.get('2', {{}}).items()}}
    l3 = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.{extra_class}L.get('3', {{}}).items()}}
    l4 = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.{extra_class}L.get('4', {{}}).items()}}
    
    # Photon-Coupled Amplitudes (1 Photon)

    # Photon creation only is special because it is a scalar
    l0_1p_dict = self.{extra_class}L.get('0_1p', {{}})
    l0_1p_val = l0_1p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    l0_1p = l0_1p_val.item() if hasattr(l0_1p_val, 'item') else l0_1p_val

    l1_1p = {{spin: tensor.transpose(1, 0) for spin, tensor in self.{extra_class}L.get('1_1p', {{}}).items()}}
    l2_1p = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.{extra_class}L.get('2_1p', {{}}).items()}}
    l3_1p = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.{extra_class}L.get('3_1p', {{}}).items()}}
    l4_1p = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.{extra_class}L.get('4_1p', {{}}).items()}}

    # Photon-Coupled Amplitudes (2 Photon)

    # Photon creation only is special because it is a scalar
    l0_2p_dict = self.{extra_class}L.get('0_2p', {{}})
    l0_2p_val = l0_2p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    l0_2p = l0_2p_val.item() if hasattr(l0_2p_val, 'item') else l0_2p_val

    l1_2p = {{spin: tensor.transpose(1, 0) for spin, tensor in self.{extra_class}L.get('1_2p', {{}}).items()}}
    l2_2p = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.{extra_class}L.get('2_2p', {{}}).items()}}
    l3_2p = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.{extra_class}L.get('3_2p', {{}}).items()}}
    l4_2p = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.{extra_class}L.get('4_2p', {{}}).items()}}

    # Photon-Coupled Amplitudes (3 Photon)

    # Photon creation only is special because it is a scalar
    l0_3p_dict = self.{extra_class}L.get('0_3p', {{}})
    l0_3p_val = l0_3p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    l0_3p = l0_3p_val.item() if hasattr(l0_3p_val, 'item') else l0_3p_val

    l1_3p = {{spin: tensor.transpose(1, 0) for spin, tensor in self.{extra_class}L.get('1_3p', {{}}).items()}}
    l2_3p = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.{extra_class}L.get('2_3p', {{}}).items()}}
    l3_3p = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.{extra_class}L.get('3_3p', {{}}).items()}}
    l4_3p = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.{extra_class}L.get('4_3p', {{}}).items()}}

    # Photon-Coupled Amplitudes (4 Photon)

    # Photon creation only is special because it is a scalar
    l0_4p_dict = self.{extra_class}L.get('0_4p', {{}})
    l0_4p_val = l0_4p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    l0_4p = l0_4p_val.item() if hasattr(l0_4p_val, 'item') else l0_4p_val

    l1_4p = {{spin: tensor.transpose(1, 0) for spin, tensor in self.{extra_class}L.get('1_4p', {{}}).items()}}
    l2_4p = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.{extra_class}L.get('2_4p', {{}}).items()}}
    l3_4p = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.{extra_class}L.get('3_4p', {{}}).items()}}
    l4_4p = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.{extra_class}L.get('4_4p', {{}}).items()}}

    oa = self.{extra_class}oa
    ob = self.{extra_class}ob
    va = self.{extra_class}va
    vb = self.{extra_class}vb
    f = {{}}
    f['aa_oo'] = self.{extra_class}f_aa[oa, oa]
    f['aa_ov'] = self.{extra_class}f_aa[oa, va]
    f['aa_vo'] = self.{extra_class}f_aa[va, oa]
    f['aa_vv'] = self.{extra_class}f_aa[va, va]
    f['bb_oo'] = self.{extra_class}f_bb[ob, ob]
    f['bb_ov'] = self.{extra_class}f_bb[ob, vb]
    f['bb_vo'] = self.{extra_class}f_bb[vb, ob]
    f['bb_vv'] = self.{extra_class}f_bb[vb, vb]
    eri = {{}}
    eri['aaaa_oooo'] = self.{extra_class}g_aaaa[oa, oa, oa, oa]
    eri['aaaa_oovo'] = self.{extra_class}g_aaaa[oa, oa, va, oa]
    eri['aaaa_oovv'] = self.{extra_class}g_aaaa[oa, oa, va, va]
    eri['aaaa_vooo'] = self.{extra_class}g_aaaa[va, oa, oa, oa]
    eri['aaaa_vovo'] = self.{extra_class}g_aaaa[va, oa, va, oa]
    eri['aaaa_vovv'] = self.{extra_class}g_aaaa[va, oa, va, va]
    eri['aaaa_vvoo'] = self.{extra_class}g_aaaa[va, va, oa, oa]
    eri['aaaa_vvvo'] = self.{extra_class}g_aaaa[va, va, va, oa]
    eri['aaaa_vvvv'] = self.{extra_class}g_aaaa[va, va, va, va]
    eri['abab_oooo'] = self.{extra_class}g_abab[oa, ob, oa, ob]
    eri['abab_oovo'] = self.{extra_class}g_abab[oa, ob, va, ob]
    eri['abab_oovv'] = self.{extra_class}g_abab[oa, ob, va, vb]
    eri['abab_vooo'] = self.{extra_class}g_abab[va, ob, oa, ob]
    eri['abab_vovo'] = self.{extra_class}g_abab[va, ob, va, ob]
    eri['abab_vovv'] = self.{extra_class}g_abab[va, ob, va, vb]
    eri['abab_vvoo'] = self.{extra_class}g_abab[va, vb, oa, ob]
    eri['abab_vvvo'] = self.{extra_class}g_abab[va, vb, va, ob]
    eri['abab_vvvv'] = self.{extra_class}g_abab[va, vb, va, vb]
    eri['abba_oovo'] = -self.{extra_class}g_abab[oa, ob, oa, vb].transpose(0,1,3,2)
    eri['abba_vovo'] = -self.{extra_class}g_abab[va, ob, oa, vb].transpose(0,1,3,2)
    eri['abba_vvvo'] = -self.{extra_class}g_abab[va, vb, oa, vb].transpose(0,1,3,2)
    eri['baab_vooo'] = -self.{extra_class}g_abab[oa, vb, oa, ob].transpose(1,0,2,3)
    eri['baab_vovo'] = -self.{extra_class}g_abab[oa, vb, va, ob].transpose(1,0,2,3)
    eri['baab_vovv'] = -self.{extra_class}g_abab[oa, vb, va, vb].transpose(1,0,2,3)
    eri['baba_vovo'] = self.{extra_class}g_abab[oa, vb, oa, vb].transpose(1,0,3,2)
    eri['bbbb_oooo'] = self.{extra_class}g_bbbb[ob, ob, ob, ob]
    eri['bbbb_oovo'] = self.{extra_class}g_bbbb[ob, ob, vb, ob]
    eri['bbbb_oovv'] = self.{extra_class}g_bbbb[ob, ob, vb, vb]
    eri['bbbb_vooo'] = self.{extra_class}g_bbbb[vb, ob, ob, ob]
    eri['bbbb_vovo'] = self.{extra_class}g_bbbb[vb, ob, vb, ob]
    eri['bbbb_vovv'] = self.{extra_class}g_bbbb[vb, ob, vb, vb]
    eri['bbbb_vvoo'] = self.{extra_class}g_bbbb[vb, vb, ob, ob]
    eri['bbbb_vvvo'] = self.{extra_class}g_bbbb[vb, vb, vb, ob]
    eri['bbbb_vvvv'] = self.{extra_class}g_bbbb[vb, vb, vb, vb]
    Id = {{}}
    noa = t1['aa'].shape[1]
    nob = t1['bb'].shape[1]
    Id['aa_oo'] = np.eye(noa, noa)
    Id['bb_oo'] = np.eye(nob, nob)
    scalars_ = {{}}
    tmps_ = {{}}
"""
    if is_qed:
        ret_string += \
f"""
    dp = {{}}
    dp['aa_oo'] = self.{extra_class}dipole_aa[oa, oa]
    dp['aa_ov'] = self.{extra_class}dipole_aa[oa, va]
    dp['aa_vo'] = self.{extra_class}dipole_aa[va, oa]
    dp['aa_vv'] = self.{extra_class}dipole_aa[va, va]
    dp['bb_oo'] = self.{extra_class}dipole_bb[ob, ob]
    dp['bb_ov'] = self.{extra_class}dipole_bb[ob, vb]
    dp['bb_vo'] = self.{extra_class}dipole_bb[vb, ob]
    dp['bb_vv'] = self.{extra_class}dipole_bb[vb, vb]
    w0 = self.{extra_class}cavity_frequency
    N0 = self.{extra_class}nuc_dip * np.sqrt(0.5 * self.{extra_class}cavity_frequency)
"""

    return ret_string

def cc_residual(residual_name, 
    T, 
    L, 
    function_name, 
    spin_block = True, 
    write_function = False,
    is_qed = False,
    pq_graph_options = None):

    """
    derive equations for CC residual

    :param residual_name: name for the variable representing the left-hand side of the residual equation
    :param T: list of cluster operators
    :param L: left operator defining the bra / projection
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
    :param is_qed: include qed-cc terms? 
    :param pq_graph_options: options dictionary for pq_graph
    """

    if not spin_block:
        raise Exception("spin-orbital cc residual equations not implemented")

    pq = pdaggerq.pq_helper("fermi")

    # set bra
    pq.set_left_operators(L)

    # add similarity-transformed Hamiltonian
    ham_terms = [['f'], ['v']]
    pq.add_st_operator(1.0, ['f'], T)
    pq.add_st_operator(1.0, ['v'], T)

    if is_qed:
        ham_terms.append(['w0'])
        ham_terms.append(['d+'])
        ham_terms.append(['d-'])

        pq.add_st_operator(1.0, ['w0'], T)
        pq.add_st_operator(-1.0, ['d+'], T)
        pq.add_st_operator(-1.0, ['d-'], T)

        pq.add_st_operator(-1.0, ['ON', 'B+'], T) # nuclear part of bilinear coupling
        pq.add_st_operator(-1.0, ['ON', 'B-'], T) # nuclear part of bilinear coupling

    # cleanup
    pq.simplify()

    # dictionary to store the derived equations
    eqs = {}

    # spin blocking
    if spin_block:
        block_by_spin(pq, residual_name, L + T + ham_terms, eqs)
    else:
        eqs[residual_name] = pq.clone()
        # print the fully contracted strings
        print(f"Equation {residual_name}:", flush=True)
        for term in pq.strings():
            print(term, flush=True)

    # Enable and configure pq_graph
    graph = configure_graph(pq_graph_options)

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # Optimize the graph
    graph.optimize()

    # Initialization statements
    generated_code_string = f"""def {function_name}(self):"""

    generated_code_string += function_initialization_string(is_qed = is_qed)

    # pq graph output
    generated_code_string += graph.str("python")

    # Return statement

    # This regex looks for digit, optionally followed by '_Np'
    match = re.search(r'(\d+)(?:_(\d+)p)?$', residual_name)
    
    if not match:
        # Handles pure strings with no numbers (e.g., 'cc_energy', 'cc_residual')
        generated_code_string += f"\n    return {residual_name}\n"
    else:
        # Group 1 is guaranteed to be the fermion order (e.g., '2' from 'r2_1p')
        order = int(match.group(1))
        
        # Group 2 is the photon order if it exists (e.g., '1' from 'r2_1p')
        nph_suffix = f"{match.group(2)}p" if match.group(2) else None
    
        # Construct base_name (e.g., '1', '2_1p', '0_1p')
        base_name = f"{order}_{nph_suffix}" if nph_suffix else str(order)
        
        # Safely get the spins for this exact order
        spins = SPIN_MAP.get(order, [''])

        # Generate individual spin channel assignments
        assignments = []
        for spin in spins:
            var_name = f"{residual_name}_{spin}" if spin else residual_name
            assignments.append(f"    residual['{base_name}']['{spin}'] = {var_name}")
    
        assignments_str = "\n".join(assignments)
    
        generated_code_string += \
f"""
    residual = {{}}
    residual['{base_name}'] = {{}}
{assignments_str}
    return residual
"""

    # write function 
    if write_function:
        with open(f"generated_equations/{function_name}.py", "w") as file:
            file.write(generated_code_string)

    pq.clear()

    del pq

    return generated_code_string

def bernoulli_ucc_residual(rank, 
    residual_name, 
    T, 
    L, 
    function_name, 
    spin_block = True, 
    write_function = False,
    pq_graph_options = None):

    """
    derive equations for UCC, truncation based on Bernoulli expansion and commutator rank

    :param residual_name: name for the variable representing the left-hand side of the residual equation
    :param rank: commutator rank
    :param T: list of cluster operators
    :param L: left operator defining the bra / projection
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
    :param pq_graph_options: options dictionary for pq_graph
    """

    if not spin_block:
        raise Exception("spin-orbital cc residual equations not implemented")

    pq = pdaggerq.pq_helper("fermi")

    pq.set_unitary_cc(True)

    # set bra
    pq.set_left_operators(L)

    # add similarity-transformed Hamiltonian
    ham_terms = [['f'], ['v']]

    pq.add_operator_product(1.0, ['f'])
    for myT in T:
        pq.add_commutator(1.0, ['f'], [myT])

    pq.add_bernoulli_operator(1.0, ['v'], T, rank)

    # cleanup
    pq.simplify()

    # dictionary to store the derived equations
    eqs = {}

    # spin blocking
    if spin_block:
        block_by_spin(pq, residual_name, L + T + ham_terms, eqs)
    else:
        eqs[residual_name] = pq.clone()
        # print the fully contracted strings
        print(f"Equation {residual_name}:", flush=True)
        for term in pq.strings():
            print(term, flush=True)

    # Enable and configure pq_graph
    graph = configure_graph(pq_graph_options)

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # optimize the graph
    graph.optimize()

    # initialization statements
    generated_code_string = f"""def {function_name}(self):"""

    generated_code_string += function_initialization_string()

    # pq graph output
    generated_code_string += graph.str("python")

    # Return statement

    # This regex looks for digit, optionally followed by '_Np'
    match = re.search(r'(\d+)(?:_(\d+)p)?$', residual_name)
            
    if not match:
        # Handles pure strings with no numbers (e.g., 'cc_energy', 'cc_residual')
        generated_code_string += f"\n    return {residual_name}\n"
    else:
        # Group 1 is guaranteed to be the fermion order (e.g., '2' from 'r2_1p')
        order = int(match.group(1))
        
        # Group 2 is the photon order if it exists (e.g., '1' from 'r2_1p')
        nph_suffix = f"{match.group(2)}p" if match.group(2) else None
    
        # Construct base_name (e.g., '1', '2_1p', '0_1p')
        base_name = f"{order}_{nph_suffix}" if nph_suffix else str(order)
   
        # Safely get the spins for this exact order
        spins = SPIN_MAP.get(order, [''])
        
        # Generate individual spin channel assignments
        assignments = []
        for spin in spins:
            var_name = f"{residual_name}_{spin}" if spin else residual_name
            assignments.append(f"    residual['{base_name}']['{spin}'] = {var_name}")
    
        assignments_str = "\n".join(assignments)

        generated_code_string += \
f"""
    residual = {{}}
    residual['{base_name}'] = {{}}
{assignments_str}
    return residual
"""

    # write function 
    if write_function:
        with open(f"generated_equations/{function_name}.py", "w") as file:
            file.write(generated_code_string)

    pq.clear()

    del pq

    return generated_code_string

def uccsd_singles_residual(order, 
    residual_name, 
    L,
    function_name,
    spin_block = True, 
    write_function = False,
    pq_graph_options = None):

    """
    derive equations for UCCSD singles residual, truncation based on perturbation order

    :param order: the order in perturbation theory used to truncate the BCH expansion
    :param residual_name: name for the variable representing the left-hand side of the residual equation
    :param L: left operator defining the bra / projection
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
    :param pq_graph_options: options dictionary for pq_graph
    """

    if not spin_block:
        raise Exception("spin-orbital uccsd singles residual equations not implemented")

    pq = pdaggerq.pq_helper("fermi")

    pq.set_unitary_cc(True)

    # set bra
    pq.set_left_operators(L)

    # up to 2nd order

    if order > 0:
        pq.add_operator_product(1.0, ['f']) # 0

    if order > 1:
        pq.add_operator_product(1.0, ['v']) # 1
        pq.add_commutator(1.0, ['f'],['t2']) # 1

    if order > 2:
        pq.add_commutator(1.0, ['f'],['t1']) # 2
        pq.add_commutator(1.0, ['v'],['t2']) # 2
        pq.add_double_commutator(0.5, ['f'],['t2'],['t2']) # 2

    if order > 3:
        raise Exception("uccsd singles residual implemented only up to 3rd order")

    # cleanup
    pq.simplify()

    # dictionary to store the derived equations
    eqs = {}

    # spin blocking
    if spin_block:
        T = ['t1', 't2']
        block_by_spin(pq, residual_name, L + T + ['f'] + ['v'], eqs)
    else:
        eqs[residual_name] = pq.clone()
        # print the fully contracted strings
        print(f"Equation {residual_name}:", flush=True)
        for term in pq.strings():
            print(term, flush=True)

    # Enable and configure pq_graph
    graph = configure_graph(pq_graph_options)

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # optimize the graph
    graph.optimize()

    # initialization statements
    generated_code_string = f"""def {function_name}(self):"""

    generated_code_string += function_initialization_string()

    # pq graph output
    generated_code_string += graph.str("python")

    # return statement
    base_name = '1'
    # check for photons
    for part in residual_name.split('_'):
        if part.endswith('p') and part[:-1].isdigit():
            base_name += '_' + part[:-1] + 'p'
            break
    generated_code_string += \
f"""
    residual = {{}}
    residual['{base_name}'] = {{}}
    residual['{base_name}']['aa'] = {residual_name}_aa
    residual['{base_name}']['bb'] = {residual_name}_bb
    return residual
"""

    # write function 
    if write_function:
        with open(f"generated_equations/{function_name}.py", "w") as file:
            file.write(generated_code_string)

    pq.clear()

    del pq

    return generated_code_string

def uccsd_doubles_residual(order, 
    residual_name,
    L,
    function_name,
    spin_block = True,
    write_function = False,
    pq_graph_options = None):

    """
    derive equations for UCCSD doubles residual, truncation based on perturbation order

    :param order: the order in perturbation theory used to truncate the BCH expansion
    :param residual_name: name for the variable representing the left-hand side of the residual equation
    :param L: left operator defining the bra / projection
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
    :param pq_graph_options: options dictionary for pq_graph
    """

    if not spin_block:
        raise Exception("spin-orbital uccsd doubles residual equations not implemented")

    pq = pdaggerq.pq_helper("fermi")

    pq.set_unitary_cc(True)

    # set bra
    pq.set_left_operators(L)

    # up to 3rd order

    pq.add_operator_product(1.0, ['f']) # 0

    if order > 0:
        pq.add_operator_product(1.0, ['v']) # 1
        pq.add_commutator(1.0, ['f'],['t2']) # 1

    if order > 1:
        pq.add_commutator(1.0, ['f'],['t1']) # 2
        pq.add_commutator(1.0, ['v'],['t2']) # 2
        pq.add_double_commutator(0.5, ['f'],['t2'],['t2']) # 2

    if order > 2:
        pq.add_commutator(1.0, ['v'],['t1']) # 3
        pq.add_double_commutator(0.5, ['f'],['t1'],['t2']) # 3
        pq.add_double_commutator(0.5, ['f'],['t2'],['t1']) # 3
        pq.add_double_commutator(0.5, ['v'],['t2'],['t2']) # 3
        pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t2'],['t2']) # 3

    if order > 3:
        raise Exception("uccsd doubles residual implemented only up to 3rd order")

    # cleanup
    pq.simplify()

    # dictionary to store the derived equations
    eqs = {}

    # spin blocking
    if spin_block:
        T = ['t1', 't2']
        block_by_spin(pq, residual_name, L + T + ['f'] + ['v'], eqs)
    else:
        eqs[residual_name] = pq.clone()
        # print the fully contracted strings
        print(f"Equation {residual_name}:", flush=True)
        for term in pq.strings():
            print(term, flush=True)

    # Enable and configure pq_graph
    graph = configure_graph(pq_graph_options)

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # optimize the graph
    graph.optimize()

    # initialization statements 
    generated_code_string = f"""def {function_name}(self):""" 

    generated_code_string += function_initialization_string()
    
    # pq graph output
    generated_code_string += graph.str("python")
        
    # return statement
    base_name = '2' 
    # check for photons
    for part in residual_name.split('_'):
        if part.endswith('p') and part[:-1].isdigit():
            base_name += '_' + part[:-1] + 'p'
            break
    generated_code_string += \
f"""
    residual = {{}}
    residual['{base_name}'] = {{}}
    residual['{base_name}']['aaaa'] = {residual_name}_aaaa
    residual['{base_name}']['abab'] = {residual_name}_abab
    residual['{base_name}']['bbbb'] = {residual_name}_bbbb
    return residual
"""
    
    # write function 
    if write_function:
        with open(f"generated_equations/{function_name}.py", "w") as file:
            file.write(generated_code_string)

    pq.clear()

    del pq

    return generated_code_string

def uccsd_energy(order,
    energy_name,
    function_name,
    spin_block = True,
    write_function = False,
    pq_graph_options = None):

    """
    derive equations for UCCSD energy, truncation based on perturbation order

    :param order: the order in perturbation theory used to truncate the BCH expansion
    :param energy_name: name for the variable representing the energy
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
    :param pq_graph_options: options dictionary for pq_graph
    """

    if not spin_block:
        raise Exception("spin-orbital uccsd energy equations not implemented")

    pq = pdaggerq.pq_helper("fermi")

    pq.set_unitary_cc(True)

    # up to 4th-order

    pq.add_operator_product(1.0, ['f']) # 0

    if order > 0:
        pq.add_operator_product(1.0, ['v']) # 1
        pq.add_commutator(1.0, ['f'],['t2']) # 1

    if order > 1:
        pq.add_commutator(1.0, ['f'],['t1']) # 2
        pq.add_commutator(1.0, ['v'],['t2']) # 2
        pq.add_double_commutator(0.5, ['f'],['t2'],['t2']) # 2

    if order > 2:
        pq.add_commutator(1.0, ['v'],['t1']) # 3
        pq.add_double_commutator(0.5, ['f'],['t1'],['t2']) # 3
        pq.add_double_commutator(0.5, ['f'],['t2'],['t1']) # 3
        pq.add_double_commutator(0.5, ['v'],['t2'],['t2']) # 3
        pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t2'],['t2']) # 3

    if order > 3:
        pq.add_double_commutator(0.5, ['f'],['t1'],['t1']) # 4
        pq.add_double_commutator(0.5, ['v'],['t1'],['t2']) # 4
        pq.add_double_commutator(0.5, ['v'],['t2'],['t1']) # 4
        pq.add_triple_commutator(1.0 / 6.0, ['f'],['t1'],['t2'],['t2']) # 4
        pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t1'],['t2']) # 4
        pq.add_triple_commutator(1.0 / 6.0, ['f'],['t2'],['t2'],['t1']) # 4
        pq.add_triple_commutator(1.0 / 6.0, ['v'],['t2'],['t2'],['t2']) # 4
        pq.add_quadruple_commutator(1.0 / 24.0, ['f'],['t2'],['t2'],['t2'],['t2']) # 4

    if order > 4:
        raise Exception("uccsd energy implemented only up to 4th order")

    # cleanup
    pq.simplify()

    # dictionary to store the derived equations
    eqs = {}

    # spin blocking
    if spin_block:
        T = ['t1', 't2']
        block_by_spin(pq, energy_name, T + ['f'] + ['v'], eqs)
    else:
        eqs[energy_name] = pq.clone()
        # print the fully contracted strings
        print(f"Equation {energy_name}:", flush=True)
        for term in pq.strings():
            print(term, flush=True)

    # Enable and configure pq_graph
    graph = configure_graph(pq_graph_options)

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # optimize the graph
    graph.optimize()

    # initialization statements 
    generated_code_string = f"""def {function_name}(self):"""

    generated_code_string += function_initialization_string()

    # pq graph output
    generated_code_string += graph.str("python")
        
    # return statement
    generated_code_string += f"    return {energy_name}"
    
    # write function 
    if write_function:
        with open(f"generated_equations/{function_name}.py", "w") as file:
            file.write(generated_code_string)

    pq.clear()

    del pq

    return generated_code_string

def cc3_triples_residual(residual_name,
    L,
    function_name,
    spin_block = True,
    write_function = False,
    pq_graph_options = None):

    """
    derive equations for the CC3 triples residual

    :param residual_name: name for the variable representing the left-hand side of the residual equation
    :param L: left operator defining the bra / projection
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
    :param pq_graph_options: options dictionary for pq_graph
    """

    if not spin_block:
        raise Exception("spin-orbital cc3 triples residual equations not implemented")

    pq = pdaggerq.pq_helper("fermi")

    # set bra
    pq.set_left_operators(L)

    pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
    pq.add_st_operator(1.0,['v'],['t1'])
    
    # g
    pq.add_operator_product(1.0,['v'])
    
    # [g, T2]
    pq.add_commutator(1.0,['v'],['t2'])
    
    # [[g, T1], T2]] + [[g, T2], T1]]
    pq.add_double_commutator( 1.0, ['v'],['t1'],['t2'])
    
    # triple commutators
    
    # [[[g, T1, T1], T2] + [[[g, T1, T2], T1] + [[[g, T2, T1], T1]
    pq.add_triple_commutator( 1.0/2.0, ['v'],['t1'],['t1'],['t2'])
    
    # [[[[g, T1], T1], T1], T2] + three others
    pq.add_quadruple_commutator( 1.0/6.0, ['v'],['t1'],['t1'],['t1'],['t2'])

    # cleanup
    pq.simplify()

    # dictionary to store the derived equations
    eqs = {}

    # spin blocking
    if spin_block:
        block_by_spin(pq, residual_name, L + ['t1', 't2', 't3'] + ['f'] + ['v'], eqs)
    else:
        eqs[residual_name] = pq.clone()
        # print the fully contracted strings
        print(f"Equation {residual_name}:", flush=True)
        for term in pq.strings():
            print(term, flush=True)

    # Enable and configure pq_graph
    graph = configure_graph(pq_graph_options)

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # optimize the graph
    graph.optimize()

    # initialization statements 
    generated_code_string = f"""def {function_name}(self):"""

    generated_code_string += function_initialization_string()

    # pq graph output
    generated_code_string += graph.str("python")
        
    # return statement
    base_name = '3'
    # check for photons
    for part in residual_name.split('_'):
        if part.endswith('p') and part[:-1].isdigit():
            base_name += '_' + part[:-1] + 'p'
            break
    generated_code_string += \
f"""
    residual = {{}}
    residual['{base_name}'] = {{}}
    residual['{base_name}']['aaaaaa'] = {residual_name}_aaaaaa
    residual['{base_name}']['aabaab'] = {residual_name}_aabaab
    residual['{base_name}']['abbabb'] = {residual_name}_abbabb
    residual['{base_name}']['bbbbbb'] = {residual_name}_bbbbbb
    return residual
"""
    
    # write function 
    if write_function:
        with open(f"generated_equations/{function_name}.py", "w") as file:
            file.write(generated_code_string)

    pq.clear()

    del pq

    return generated_code_string

def lambda_cc_residual(residual_name,
    T,
    L,
    R, 
    function_name, 
    spin_block = True, 
    write_function = False,
    pq_graph_options = None):

    """
    derive equations for lambda CC residual

    :param residual_name: name for the variable representing the left-hand side of the residual equation
    :param T: list of cluster operators
    :param L: list of lambda amplitudes
    :param R: excitation operator defining the projection
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
    :param pq_graph_options: options dictionary for pq_graph
    """

    if not spin_block:
        raise Exception("spin-orbital ccsd lambda equations not implemented")

    pq = pdaggerq.pq_helper("fermi")

    #  <0| e(-T) H R e(T)|0>
    
    pq.set_left_operators([['1']])
    pq.set_right_operators([['1']])
    
    pq.add_st_operator(1.0,['f',R],T)
    pq.add_st_operator(1.0,['v',R],T)
    
    # <0| L e(-T) [H,R] e(T)|0>
    
    pq.set_left_operators(L)
    pq.set_right_operators([['1']])
    
    pq.add_st_operator( 1.0,['f',R],T)
    pq.add_st_operator( 1.0,['v',R],T)
    
    pq.add_st_operator(-1.0,[R,'f'],T)
    pq.add_st_operator(-1.0,[R,'v'],T)

    # cleanup
    pq.simplify()

    # dictionary to store the derived equations
    eqs = {}

    # spin blocking
    if spin_block:
        block_by_spin(pq, residual_name, L + T + ['f'] + ['v'] + [[R]], eqs)
    else:
        eqs[residual_name] = pq.clone()
        # print the fully contracted strings
        print(f"Equation {residual_name}:", flush=True)
        for term in pq.strings():
            print(term, flush=True)

    # Enable and configure pq_graph
    graph = configure_graph(pq_graph_options)

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # optimize the graph
    graph.optimize()

    # initialization statements 
    generated_code_string = f"""def {function_name}(self):"""

    generated_code_string += function_initialization_string()

    # pq graph output
    generated_code_string += graph.str("python")

    # Return statement

    # This regex looks for digit, optionally followed by '_Np'
    match = re.search(r'(\d+)(?:_(\d+)p)?$', residual_name)
    
    if not match:
        # Handles pure strings with no numbers (e.g., 'cc_energy', 'cc_residual')
        generated_code_string += f"\n    return {residual_name}\n"
    else:
        # Group 1 is guaranteed to be the fermion order (e.g., '2' from 'r2_1p')
        order = int(match.group(1))
        
        # Group 2 is the photon order if it exists (e.g., '1' from 'r2_1p')
        nph_suffix = f"{match.group(2)}p" if match.group(2) else None
    
        # Construct base_name (e.g., '1', '2_1p', '0_1p')
        base_name = f"{order}_{nph_suffix}" if nph_suffix else str(order)
        
        # Safely get the spins for this exact order
        spins = SPIN_MAP.get(order, [''])

        # Generate individual spin channel assignments
        assignments = []
        for spin in spins:
            var_name = f"{residual_name}_{spin}" if spin else residual_name
            assignments.append(f"    residual['{base_name}']['{spin}'] = {var_name}")
    
        assignments_str = "\n".join(assignments)

        generated_code_string += \
f"""    
    residual = {{}}
    residual['{base_name}'] = {{}}
{assignments_str}
    return residual
"""        

    # write function 
    if write_function:
        with open(f"generated_equations/{function_name}.py", "w") as file:
            file.write(generated_code_string)

    pq.clear()

    del pq

    return generated_code_string

def lambda_cc_pseudoenergy(energy_name,
    L,
    R,
    function_name,
    spin_block = True,
    write_function = False,
    pq_graph_options = None):

    """
    derive equations for lambda CC pseudoenergy

    :param name: name for the variable representing the pseudoenergy
    :param T: list of cluster operators
    :param L: list of lambda amplitudes
    :param R: excitation operator defining the projection
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
    :param pq_graph_options: options dictionary for pq_graph
    """

    if not spin_block:
        raise Exception("spin-orbital ccsd lambda equations not implemented")

    pq = pdaggerq.pq_helper("fermi")

    # set bra
    pq.set_left_operators(L)

    # set ket
    pq.set_right_operators(R)

    # bare Hamiltonian
    pq.add_operator_product(1.0, ['f'])
    pq.add_operator_product(1.0, ['v'])

    # cleanup
    pq.simplify()

    # dictionary to store the derived equations
    eqs = {}

    # spin blocking
    if spin_block:
        block_by_spin(pq, energy_name, L + R + ['f'] + ['v'], eqs)
    else:
        eqs[sigma_eqname] = pq.clone()
        # print the fully contracted strings
        print(f"Equation {sigma_eqname}:", flush=True)
        for term in pq.strings():
            print(term, flush=True)

    # Enable and configure pq_graph
    graph = configure_graph(pq_graph_options)

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # optimize the graph
    graph.optimize()

    # initialization statements 
    generated_code_string = f"""def {function_name}(self):"""         

    generated_code_string += function_initialization_string()
        
    # pq graph output
    generated_code_string += graph.str("python")
            
    # return statement
    generated_code_string += f"    return {energy_name}"

    # write function 
    if write_function:
        with open(f"generated_equations/{function_name}.py", "w") as file:
            file.write(generated_code_string)

    pq.clear()

    del pq

    return generated_code_string

def eomcc_sigma(sigma_name, 
    T,
    L,
    R, 
    function_name,
    spin_block = True,
    write_function = False,
    pq_graph_options = None):

    """
    derive equations for left/right EOMCC sigma equations
    
    :param sigma_name: name for the variable representing the left/right EOMCC sigma veector
    :param T: list of cluster operators
    :param L: list of left-hand operators
    :param R: list of right-hand operators
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
    :param pq_graph_options: options dictionary for pq_graph
    """ 

    if not spin_block:
        raise Exception("spin-orbital eomcc equations not implemented")

    # right- or left-hand sigma?
    is_right = True
    if len(L) > len(R):
        is_right = False

    pq = pdaggerq.pq_helper("fermi")

    # set bra
    pq.set_left_operators(L)

    # set ket
    pq.set_right_operators(R)

    # add similarity-transformed Hamiltonian (or bare Hamiltonian if no T)
    if len(T) > 0:
        pq.add_st_operator(1.0, ['f'], T)
        pq.add_st_operator(1.0, ['v'], T)
    else:
        pq.add_operator_product(1.0, ['f'])
        pq.add_operator_product(1.0, ['v'])

    # cleanup
    pq.simplify()

    # dictionary to store the derived equations
    eqs = {}

    # spin blocking
    if spin_block:
        block_by_spin(pq, sigma_name, L + T + R + ['f'] + ['v'], eqs)
    else:
        eqs[sigma_eqname] = pq.clone()
        # print the fully contracted strings
        print(f"Equation {sigma_eqname}:", flush=True)
        for term in pq.strings():
            print(term, flush=True)

    # Enable and configure pq_graph
    graph = configure_graph(pq_graph_options)

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # optimize the graph
    graph.optimize()

    # initialization statements 
    generated_code_string = f"def {function_name}(self):"
    generated_code_string += function_initialization_string(extra_class = "ccsd")

    generated_code_string += \
f"""
    # right-hand eom amplitudes

    # r0 is special because it is a scalar
    r0_dict = self.R.get('0', {{}})
    r0_val = r0_dict.get('', 0.0)

    # Unwrap numpy array/scalar to a raw float if necessary
    r0 = r0_val.item() if hasattr(r0_val, 'item') else r0_val

    r1 = dict(self.R.get('1', {{}}))
    r2 = dict(self.R.get('2', {{}}))
    r3 = dict(self.R.get('3', {{}}))
    r4 = dict(self.R.get('4', {{}}))
    
    # Photon-Coupled Amplitudes (1 Photon)

    # Photon creation only is special because it is a scalar
    r0_1p_dict = self.R.get('0_1p', {{}})
    r0_1p_val = r0_1p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    r0_1p = r0_1p_val.item() if hasattr(r0_1p_val, 'item') else r0_1p_val

    r1_1p = dict(self.R.get('1_1p', {{}}))
    r2_1p = dict(self.R.get('2_1p', {{}}))
    r3_1p = dict(self.R.get('3_1p', {{}}))
    r4_1p = dict(self.R.get('4_1p', {{}}))

    # Photon-Coupled Amplitudes (2 Photon)

    # Photon creation only is special because it is a scalar
    r0_2p_dict = self.R.get('0_2p', {{}})
    r0_2p_val = r0_2p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    r0_2p = r0_2p_val.item() if hasattr(r0_2p_val, 'item') else r0_2p_val

    r1_2p = dict(self.R.get('1_2p', {{}}))
    r2_2p = dict(self.R.get('2_2p', {{}}))
    r3_2p = dict(self.R.get('3_2p', {{}}))
    r4_2p = dict(self.R.get('4_2p', {{}}))

    # Photon-Coupled Amplitudes (3 Photon)

    # Photon creation only is special because it is a scalar
    r0_3p_dict = self.R.get('0_3p', {{}})
    r0_3p_val = r0_3p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    r0_3p = r0_3p_val.item() if hasattr(r0_3p_val, 'item') else r0_3p_val

    r1_3p = dict(self.R.get('1_3p', {{}}))
    r2_3p = dict(self.R.get('2_3p', {{}}))
    r3_3p = dict(self.R.get('3_3p', {{}}))
    r4_3p = dict(self.R.get('4_3p', {{}}))

    # Photon-Coupled Amplitudes (4 Photon)

    # Photon creation only is special because it is a scalar
    r0_4p_dict = self.R.get('0_4p', {{}})
    r0_4p_val = r0_4p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    r0_4p = r0_4p_val.item() if hasattr(r0_4p_val, 'item') else r0_4p_val

    r1_4p = dict(self.R.get('1_4p', {{}}))
    r2_4p = dict(self.R.get('2_4p', {{}}))
    r3_4p = dict(self.R.get('3_4p', {{}}))
    r4_4p = dict(self.R.get('4_4p', {{}}))

    # left-hand eom amplitudes

    # l0 is special because it is a scalar
    l0_dict = self.L.get('0', {{}})
    l0_val = l0_dict.get('', 0.0)

    # Unwrap numpy array/scalar to a raw float if necessary
    l0 = l0_val.item() if hasattr(l0_val, 'item') else l0_val
    
    l1 = {{spin: tensor.transpose(1, 0) for spin, tensor in self.L.get('1', {{}}).items()}}
    l2 = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.L.get('2', {{}}).items()}}
    l3 = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.L.get('3', {{}}).items()}}
    l4 = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.L.get('4', {{}}).items()}}
    
    # Photon-Coupled Amplitudes (1 Photon)

    # Photon creation only is special because it is a scalar
    l0_1p_dict = self.L.get('0_1p', {{}})
    l0_1p_val = l0_1p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    l0_1p = l0_1p_val.item() if hasattr(l0_1p_val, 'item') else l0_1p_val

    l1_1p = {{spin: tensor.transpose(1, 0) for spin, tensor in self.L.get('1_1p', {{}}).items()}}
    l2_1p = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.L.get('2_1p', {{}}).items()}}
    l3_1p = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.L.get('3_1p', {{}}).items()}}
    l4_1p = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.L.get('4_1p', {{}}).items()}}

    # Photon-Coupled Amplitudes (2 Photon)

    # Photon creation only is special because it is a scalar
    l0_2p_dict = self.L.get('0_2p', {{}})
    l0_2p_val = l0_2p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    l0_2p = l0_2p_val.item() if hasattr(l0_2p_val, 'item') else l0_2p_val

    l1_2p = {{spin: tensor.transpose(1, 0) for spin, tensor in self.L.get('1_2p', {{}}).items()}}
    l2_2p = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.L.get('2_2p', {{}}).items()}}
    l3_2p = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.L.get('3_2p', {{}}).items()}}
    l4_2p = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.L.get('4_2p', {{}}).items()}}

    # Photon-Coupled Amplitudes (3 Photon)

    # Photon creation only is special because it is a scalar
    l0_3p_dict = self.L.get('0_3p', {{}})
    l0_3p_val = l0_3p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    l0_3p = l0_3p_val.item() if hasattr(l0_3p_val, 'item') else l0_3p_val

    l1_3p = {{spin: tensor.transpose(1, 0) for spin, tensor in self.L.get('1_3p', {{}}).items()}}
    l2_3p = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.L.get('2_3p', {{}}).items()}}
    l3_3p = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.L.get('3_3p', {{}}).items()}}
    l4_3p = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.L.get('4_3p', {{}}).items()}}

    # Photon-Coupled Amplitudes (4 Photon)

    # Photon creation only is special because it is a scalar
    l0_4p_dict = self.L.get('0_4p', {{}})
    l0_4p_val = l0_4p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    l0_4p = l0_4p_val.item() if hasattr(l0_4p_val, 'item') else l0_4p_val

    l1_4p = {{spin: tensor.transpose(1, 0) for spin, tensor in self.L.get('1_4p', {{}}).items()}}
    l2_4p = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.L.get('2_4p', {{}}).items()}}
    l3_4p = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.L.get('3_4p', {{}}).items()}}
    l4_4p = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.L.get('4_4p', {{}}).items()}}
"""

    # pq graph output
    generated_code_string += graph.str("python")
        
    # Return statement
    # This regex looks for digit, optionally followed by '_Np'
    match = re.search(r'(\d+)(?:_(\d+)p)?$', sigma_name)
            
    if not match:
        # Handles pure strings with no numbers (e.g., 'cc_energy', 'cc_residual') ... although this shouldn't happen in EOM
        generated_code_string += f"\n    return {sigma_name}\n"
    else:
        # Group 1 is guaranteed to be the fermion order (e.g., '2' from 'r2_1p')
        order = int(match.group(1))
        
        # Group 2 is the photon order if it exists (e.g., '1' from 'r2_1p')
        nph_suffix = f"{match.group(2)}p" if match.group(2) else None

        # Construct base_name (e.g., '1', '2_1p', '0_1p')
        base_name = f"{order}_{nph_suffix}" if nph_suffix else str(order)

        # Safely get the spins for this exact order
        spins = SPIN_MAP.get(order, [''])

        # Generate individual spin channel assignments
        assignments = []
        for spin in spins:
            var_name = f"{sigma_name}_{spin}" if spin else sigma_name
            assignments.append(f"    sigma['{base_name}']['{spin}'] = {var_name}")

        assignments_str = "\n".join(assignments)

        generated_code_string += \
f"""
    sigma = {{}}
    sigma['{base_name}'] = {{}}
{assignments_str}
    return sigma
"""

    # write function 
    if write_function:
        with open(f"generated_equations/{function_name}.py", "w") as file:
            file.write(generated_code_string)

    pq.clear()

    del pq

    return generated_code_string

def eomcc_density_matrix(ret_name, 
    T,
    L,
    R, 
    function_name,
    spin_block = True,
    write_function = False,
    pq_graph_options = None):

    """
    derive equations for EOMCC (transition)density matrix equations
    
    :param ret_name: name for the variable representing the density matrix
    :param T: list of cluster operators
    :param L: list of left-hand operators
    :param R: list of right-hand operators
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
    :param pq_graph_options: options dictionary for pq_graph
    """ 

    if not spin_block:
        raise Exception("spin-orbital eomcc equations not implemented")

    blocks = {
        'oo' : 'e1(i,j)',
        'ov' : 'e1(i,a)',
        'vo' : 'e1(a,i)',
        'vv' : 'e1(a,b)',
    }

    # initialization statements 
    generated_code_string = \
f"""
def {function_name}(self, left_state, right_state):
    # right-hand eom amplitudes

    # r0 is special because it is a scalar
    r0_dict = self.R[right_state].get('0', {{}})
    r0_val = r0_dict.get('', 0.0)

    # Unwrap numpy array/scalar to a raw float if necessary
    r0 = r0_val.item() if hasattr(r0_val, 'item') else r0_val

    r1 = dict(self.R[right_state].get('1', {{}}))
    r2 = dict(self.R[right_state].get('2', {{}}))
    r3 = dict(self.R[right_state].get('3', {{}}))
    r4 = dict(self.R[right_state].get('4', {{}}))
    
    # Photon-Coupled Amplitudes (1 Photon)

    # Photon creation only is special because it is a scalar
    r0_1p_dict = self.R[right_state].get('0_1p', {{}})
    r0_1p_val = r0_1p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    r0_1p = r0_1p_val.item() if hasattr(r0_1p_val, 'item') else r0_1p_val

    r1_1p = dict(self.R[right_state].get('1_1p', {{}}))
    r2_1p = dict(self.R[right_state].get('2_1p', {{}}))
    r3_1p = dict(self.R[right_state].get('3_1p', {{}}))
    r4_1p = dict(self.R[right_state].get('4_1p', {{}}))

    # Photon-Coupled Amplitudes (2 Photon)

    # Photon creation only is special because it is a scalar
    r0_2p_dict = self.R[right_state].get('0_2p', {{}})
    r0_2p_val = r0_2p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    r0_2p = r0_2p_val.item() if hasattr(r0_2p_val, 'item') else r0_2p_val

    r1_2p = dict(self.R[right_state].get('1_2p', {{}}))
    r2_2p = dict(self.R[right_state].get('2_2p', {{}}))
    r3_2p = dict(self.R[right_state].get('3_2p', {{}}))
    r4_2p = dict(self.R[right_state].get('4_2p', {{}}))

    # Photon-Coupled Amplitudes (3 Photon)

    # Photon creation only is special because it is a scalar
    r0_3p_dict = self.R[right_state].get('0_3p', {{}})
    r0_3p_val = r0_3p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    r0_3p = r0_3p_val.item() if hasattr(r0_3p_val, 'item') else r0_3p_val

    r1_3p = dict(self.R[right_state].get('1_3p', {{}}))
    r2_3p = dict(self.R[right_state].get('2_3p', {{}}))
    r3_3p = dict(self.R[right_state].get('3_3p', {{}}))
    r4_3p = dict(self.R[right_state].get('4_3p', {{}}))

    # Photon-Coupled Amplitudes (4 Photon)

    # Photon creation only is special because it is a scalar
    r0_4p_dict = self.R[right_state].get('0_4p', {{}})
    r0_4p_val = r0_4p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    r0_4p = r0_4p_val.item() if hasattr(r0_4p_val, 'item') else r0_4p_val

    r1_4p = dict(self.R[right_state].get('1_4p', {{}}))
    r2_4p = dict(self.R[right_state].get('2_4p', {{}}))
    r3_4p = dict(self.R[right_state].get('3_4p', {{}}))
    r4_4p = dict(self.R[right_state].get('4_4p', {{}}))
"""

    # Enable and configure pq_graph
    graph = configure_graph(pq_graph_options)

    for block, op in blocks.items():

        pq = pdaggerq.pq_helper("fermi")

        # set bra
        pq.set_left_operators(L)

        # set ket
        pq.set_right_operators(R)

        # add similarity-transformed density operator (or bare Hamiltonian if no T)
        if len(T) > 0:
            pq.add_st_operator(1.0, [op], T)
        else:
            pq.add_operator_product(1.0, [op])

        # cleanup
        pq.simplify()

        # dictionary to store the derived equations
        eqs = {}

        # spin blocking
        block_by_spin(pq, ret_name + "_" + block, L + T + R + [[op]], eqs)

        # Add equations to graph
        for proj_eqname, eq in eqs.items():
            print(f"Adding equation {proj_eqname} to the graph", flush=True)
            graph.add(eq, proj_eqname)

        pq.clear()

        del pq

    # optimize the graph
    graph.optimize()

    generated_code_string += function_initialization_string(extra_class = "ccsd")

    # need to redefine l1/l2 because they currently point to the ccsd ones
    generated_code_string += \
f"""
    # left-hand eom amplitudes

    # l0 is special because it is a scalar
    l0_dict = self.L[left_state].get('0', {{}})
    l0_val = l0_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    l0 = l0_val.item() if hasattr(l0_val, 'item') else l0_val
    
    l1 = {{spin: tensor.transpose(1, 0) for spin, tensor in self.L[left_state].get('1', {{}}).items()}}
    l2 = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.L[left_state].get('2', {{}}).items()}}
    l3 = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.L[left_state].get('3', {{}}).items()}}
    l4 = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.L[left_state].get('4', {{}}).items()}}
    
    # Photon-Coupled Amplitudes (1 Photon)

    # Photon creation only is special because it is a scalar
    l0_1p_dict = self.L[left_state].get('0_1p', {{}})
    l0_1p_val = l0_1p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    l0_1p = l0_1p_val.item() if hasattr(l0_1p_val, 'item') else l0_1p_val

    l1_1p = {{spin: tensor.transpose(1, 0) for spin, tensor in self.L[left_state].get('1_1p', {{}}).items()}}
    l2_1p = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.L[left_state].get('2_1p', {{}}).items()}}
    l3_1p = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.L[left_state].get('3_1p', {{}}).items()}}
    l4_1p = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.L[left_state].get('4_1p', {{}}).items()}}
    
    # Photon-Coupled Amplitudes (2 Photon)
    
    # Photon creation only is special because it is a scalar
    l0_2p_dict = self.L[left_state].get('0_2p', {{}})
    l0_2p_val = l0_2p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    l0_2p = l0_2p_val.item() if hasattr(l0_2p_val, 'item') else l0_2p_val

    l1_2p = {{spin: tensor.transpose(1, 0) for spin, tensor in self.L[left_state].get('1_2p', {{}}).items()}}
    l2_2p = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.L[left_state].get('2_2p', {{}}).items()}}
    l3_2p = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.L[left_state].get('3_2p', {{}}).items()}}
    l4_2p = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.L[left_state].get('4_2p', {{}}).items()}}

    # Photon-Coupled Amplitudes (3 Photon)

    # Photon creation only is special because it is a scalar
    l0_3p_dict = self.L[left_state].get('0_3p', {{}})
    l0_3p_val = l0_3p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    l0_3p = l0_3p_val.item() if hasattr(l0_3p_val, 'item') else l0_3p_val

    l1_3p = {{spin: tensor.transpose(1, 0) for spin, tensor in self.L[left_state].get('1_3p', {{}}).items()}}
    l2_3p = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.L[left_state].get('2_3p', {{}}).items()}}
    l3_3p = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.L[left_state].get('3_3p', {{}}).items()}}
    l4_3p = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.L[left_state].get('4_3p', {{}}).items()}}

    # Photon-Coupled Amplitudes (4 Photon)

    # Photon creation only is special because it is a scalar
    l0_4p_dict = self.L[left_state].get('0_4p', {{}})
    l0_4p_val = l0_4p_dict.get('', 0.0)
    
    # Unwrap numpy array/scalar to a raw float if necessary
    l0_4p = l0_4p_val.item() if hasattr(l0_4p_val, 'item') else l0_4p_val

    l1_4p = {{spin: tensor.transpose(1, 0) for spin, tensor in self.L[left_state].get('1_4p', {{}}).items()}}
    l2_4p = {{spin: tensor.transpose(2, 3, 0, 1) for spin, tensor in self.L[left_state].get('2_4p', {{}}).items()}}
    l3_4p = {{spin: tensor.transpose(3, 4, 5, 0, 1, 2) for spin, tensor in self.L[left_state].get('3_4p', {{}}).items()}}
    l4_4p = {{spin: tensor.transpose(4, 5, 6, 7, 0, 1, 2, 3) for spin, tensor in self.L[left_state].get('4_4p', {{}}).items()}}
"""

    # pq graph output
    generated_code_string += graph.str("python")

    # return statement
    generated_code_string += \
f"""
    {ret_name} = {{}}
    {ret_name}['aa_oo'] = {ret_name}_oo_aa
    {ret_name}['aa_ov'] = {ret_name}_ov_aa.transpose(1,0)
    {ret_name}['aa_vo'] = {ret_name}_vo_aa
    {ret_name}['aa_vv'] = {ret_name}_vv_aa
    {ret_name}['bb_oo'] = {ret_name}_oo_bb
    {ret_name}['bb_ov'] = {ret_name}_ov_bb.transpose(1,0)
    {ret_name}['bb_vo'] = {ret_name}_vo_bb
    {ret_name}['bb_vv'] = {ret_name}_vv_bb

    return {ret_name}
"""

    # write function 
    if write_function:
        with open(f"generated_equations/{function_name}.py", "w") as file:
            file.write(generated_code_string)

    return generated_code_string
