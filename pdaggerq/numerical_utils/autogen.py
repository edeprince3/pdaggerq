import pdaggerq 

def configure_graph():
    """
    Configure and return the pq_graph with specific settings.

    Returns:
        graph (pq_graph): Configured pq_graph object.
    """
    return pdaggerq.pq_graph({
        'batched': False,
        #'batched': True,
        #'batch_number': 100,
        'print_level': 0,
        'opt_level': 0,
        'nthreads': -1,
        'no_scalars': False,
    })

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
    t1 = {{}}
    t1['aa'] = self.{extra_class}t1_aa
    t1['bb'] = self.{extra_class}t1_bb
    t2 = {{}}
    t2['aaaa'] = self.{extra_class}t2_aaaa
    t2['abab'] = self.{extra_class}t2_abab
    t2['bbbb'] = self.{extra_class}t2_bbbb
    t3 = {{}}
    t3['aaaaaa'] = self.{extra_class}t3_aaaaaa
    t3['aabaab'] = self.{extra_class}t3_aabaab
    t3['abbabb'] = self.{extra_class}t3_abbabb
    t3['bbbbbb'] = self.{extra_class}t3_bbbbbb
    l1 = {{}}
    l1['aa'] = self.{extra_class}l1_aa
    l1['bb'] = self.{extra_class}l1_bb
    l2 = {{}}
    l2['aaaa'] = self.{extra_class}l2_aaaa
    l2['abab'] = self.{extra_class}l2_abab
    l2['bbbb'] = self.{extra_class}l2_bbbb
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
    t0_1p = self.{extra_class}t0_1p
    t1_1p = {{}}
    t1_1p['aa'] = self.{extra_class}t1_1p_aa
    t1_1p['bb'] = self.{extra_class}t1_1p_bb
    t2_1p = {{}}
    t2_1p['aaaa'] = self.{extra_class}t2_1p_aaaa
    t2_1p['abab'] = self.{extra_class}t2_1p_abab
    t2_1p['bbbb'] = self.{extra_class}t2_1p_bbbb
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
"""

    return ret_string

def cc_residual(residual_name, 
    T, 
    L, 
    function_name, 
    spin_block = True, 
    write_function = False,
    is_qed = False):

    """
    derive equations for CC residual

    :param residual_name: name for the variable representing the left-hand side of the residual equation
    :param T: list of cluster operators
    :param L: left operator defining the bra / projection
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
    :param is_qed: include qed-cc terms? 
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
        pq.add_st_operator(1.0, ['d+'], T)
        pq.add_st_operator(1.0, ['d-'], T)

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
    graph = configure_graph()

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # optimize the graph
    graph.optimize()

    # initialization statements
    generated_code_string = f"""def {function_name}(self):"""

    generated_code_string += function_initialization_string(is_qed = is_qed)

    # pq graph output
    generated_code_string += graph.str("python")

    # return statement
    if '3' in residual_name:
        generated_code_string += f"    return {residual_name}_aaaaaa, {residual_name}_aabaab, {residual_name}_abbabb, {residual_name}_bbbbbb"
    elif '2' in residual_name:
        generated_code_string += f"    return {residual_name}_aaaa, {residual_name}_abab, r2_bbbb"
    elif '1' in residual_name:
        generated_code_string += f"    return {residual_name}_aa, {residual_name}_bb"
    else:
        generated_code_string += f"    return {residual_name}"

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
    write_function = False):

    """
    derive equations for UCCSD singles residual, truncation based on perturbation order

    :param order: the order in perturbation theory used to truncate the BCH expansion
    :param residual_name: name for the variable representing the left-hand side of the residual equation
    :param L: left operator defining the bra / projection
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
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
    graph = configure_graph()

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
    generated_code_string += f"    return {residual_name}_aa, {residual_name}_bb"

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
    write_function = False):

    """
    derive equations for UCCSD doubles residual, truncation based on perturbation order

    :param order: the order in perturbation theory used to truncate the BCH expansion
    :param residual_name: name for the variable representing the left-hand side of the residual equation
    :param L: left operator defining the bra / projection
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
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
    graph = configure_graph()

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
    generated_code_string += f"    return {residual_name}_aaaa, {residual_name}_abab, {residual_name}_bbbb"
    
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
    write_function = False):

    """
    derive equations for UCCSD energy, truncation based on perturbation order

    :param order: the order in perturbation theory used to truncate the BCH expansion
    :param energy_name: name for the variable representing the energy
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
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
    graph = configure_graph()

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
    write_function = False):

    """
    derive equations for the CC3 triples residual

    :param residual_name: name for the variable representing the left-hand side of the residual equation
    :param L: left operator defining the bra / projection
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
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
    graph = configure_graph()

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
    generated_code_string += f"    return {residual_name}_aaaaaa, {residual_name}_aabaab, {residual_name}_abbabb, {residual_name}_bbbbbb"
    
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
    write_function = False):

    """
    derive equations for lambda CC residual

    :param residual_name: name for the variable representing the left-hand side of the residual equation
    :param T: list of cluster operators
    :param L: list of lambda amplitudes
    :param R: excitation operator defining the projection
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
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
    graph = configure_graph()

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
    if residual_name == 'r3':
        generated_code_string += f"    return r3_aaaaaa.transpose.transpose(3,4,5,0,1,2), r3_aabaab.transpose(3,4,5,0,1,2), r3_abbabb.transpose(3,4,5,0,1,2), r3_bbbbbb.transpose(3,4,5,0,1,2)"
    elif residual_name == 'r2':
        generated_code_string += f"    return r2_aaaa.transpose(2,3,0,1), r2_abab.transpose(2,3,0,1), r2_bbbb.transpose(2,3,0,1)"
    elif residual_name == 'r1':
        generated_code_string += f"    return r1_aa.transpose(1,0), r1_bb.transpose(1,0)"
    else:
        generated_code_string += f"    return {residual_name}"

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
    write_function = False):

    """
    derive equations for lambda CC pseudoenergy

    :param name: name for the variable representing the pseudoenergy
    :param T: list of cluster operators
    :param L: list of lambda amplitudes
    :param R: excitation operator defining the projection
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
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
    graph = configure_graph()

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
    write_function = False):

    """
    derive equations for left/right EOMCC sigma equations
    
    :param sigma_name: name for the variable representing the left/right EOMCC sigma veector
    :param T: list of cluster operators
    :param L: list of left-hand operators
    :param R: list of right-hand operators
    :param function_name: name for the python function
    :param spin_block: do spin block the equations?
    :param write_function: do write function to disk?
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
    graph = configure_graph()

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # optimize the graph
    graph.optimize()

    # initialization statements 
    if is_right:
        generated_code_string = \
f"""
def {function_name}(self, r0, r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb):
    r1 = {{}}
    r1['aa'] = r1_aa
    r1['bb'] = r1_bb
    r2 = {{}}
    r2['aaaa'] = r2_aaaa
    r2['abab'] = r2_abab
    r2['bbbb'] = r2_bbbb
"""
    else:
        generated_code_string = f"def {function_name}(self, l0, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb):"

    generated_code_string += function_initialization_string("ccsd")

    # need to redefine l1/l2 because they currently point to the ccsd ones
    if not is_right:
        generated_code_string += \
f"""
    l1 = {{}}
    l1['aa'] = l1_aa
    l1['bb'] = l1_bb
    l2 = {{}}
    l2['aaaa'] = l2_aaaa
    l2['abab'] = l2_abab
    l2['bbbb'] = l2_bbbb
"""

    # pq graph output
    generated_code_string += graph.str("python")
        
    # return statement
    if is_right:
        if '2' in sigma_name:
            generated_code_string += f"    return {sigma_name}_aaaa, {sigma_name}_abab, {sigma_name}_bbbb"
        elif '1' in sigma_name:
            generated_code_string += f"    return {sigma_name}_aa, {sigma_name}_bb"
        else:
            generated_code_string += f"    return {sigma_name}"
    else:
        if '2' in sigma_name:
            generated_code_string += f"    return {sigma_name}_aaaa.transpose(2,3,0,1), {sigma_name}_abab.transpose(2,3,0,1), {sigma_name}_bbbb.transpose(2,3,0,1)"
        elif '1' in sigma_name:
            generated_code_string += f"    return {sigma_name}_aa.transpose(1,0), {sigma_name}_bb.transpose(1,0)"
        else:
            generated_code_string += f"    return {sigma_name}"

    # write function 
    if write_function:
        with open(f"generated_equations/{function_name}.py", "w") as file:
            file.write(generated_code_string)

    pq.clear()

    del pq

    return generated_code_string
