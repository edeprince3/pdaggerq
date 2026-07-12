import pdaggerq 

def configure_graph():
    """
    Configure and return the pq_graph with specific settings.

    Returns:
        graph (pq_graph): Configured pq_graph object.
    """
    return pdaggerq.pq_graph({
        'batched': False,
        'print_level': 3,
        'opt_level': 1,
        'nthreads': -1,
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

def cc_residual(residual_name, T, L, function_name, spin_block = True):
    """
    derive equations for CC residual
    """

    pq = pdaggerq.pq_helper("fermi")

    # set bra
    pq.set_left_operators(L)

    # add similarity-transformed Hamiltonian
    pq.add_st_operator(1.0, ['f'], T)
    pq.add_st_operator(1.0, ['v'], T)

    # cleanup
    pq.simplify()

    # dictionary to store the derived equations
    eqs = {}

    # spin blocking
    if spin_block:
        block_by_spin(pq, residual_name, L + T + ['f'] + ['v'], eqs)
    else:
        eqs[residual_eqname] = pq.clone()
        # print the fully contracted strings
        print(f"Equation {residual_eqname}:", flush=True)
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

    # write function 
    with open(f"generated_equations/{function_name}.py", "w") as file:

        # initialize
        file.write(f"import numpy as np\n")
        file.write(f"from numpy import einsum\n")
        file.write(f"def {function_name}(self):\n")
        file.write(f"    t1 = {{}}\n")
        file.write(f"    t1['aa'] = self.t1_aa\n")
        file.write(f"    t1['bb'] = self.t1_bb\n")
        file.write(f"    t2 = {{}}\n")
        file.write(f"    t2['aaaa'] = self.t2_aaaa\n")
        file.write(f"    t2['abab'] = self.t2_abab\n")
        file.write(f"    t2['bbbb'] = self.t2_bbbb\n")
        if 't3' in T or 'T3' in T:
            file.write(f"    t3 = {{}}\n")
            file.write(f"    t3['aaaaaa'] = self.t3_aaaaaa\n")
            file.write(f"    t3['aabaab'] = self.t3_aabaab\n")
            file.write(f"    t3['abbabb'] = self.t3_abbabb\n")
            file.write(f"    t3['bbbbbb'] = self.t3_bbbbbb\n")
        file.write(f"    oa = self.oa\n")
        file.write(f"    ob = self.ob\n")
        file.write(f"    va = self.va\n")
        file.write(f"    vb = self.vb\n")

        file.write(f"    f = {{}}\n")
        for spin in ['a', 'b']:
            for block1 in ['o', 'v']:
                for block2 in ['o', 'v']:
                    file.write(f"    f['{spin}{spin}_{block1}{block2}'] = self.f_{spin}{spin}[{block1}{spin}, {block2}{spin}]\n")

        file.write(f"    eri = {{}}\n")
        file.write(f"    eri['aaaa_oooo'] = self.g_aaaa[oa, oa, oa, oa]\n")
        file.write(f"    eri['aaaa_oovo'] = self.g_aaaa[oa, oa, va, oa]\n")
        file.write(f"    eri['aaaa_oovv'] = self.g_aaaa[oa, oa, va, va]\n")
        file.write(f"    eri['aaaa_vooo'] = self.g_aaaa[va, oa, oa, oa]\n")
        file.write(f"    eri['aaaa_vovo'] = self.g_aaaa[va, oa, va, oa]\n")
        file.write(f"    eri['aaaa_vovv'] = self.g_aaaa[va, oa, va, va]\n")
        file.write(f"    eri['aaaa_vvoo'] = self.g_aaaa[va, va, oa, oa]\n")
        file.write(f"    eri['aaaa_vvvo'] = self.g_aaaa[va, va, va, oa]\n")
        file.write(f"    eri['aaaa_vvvv'] = self.g_aaaa[va, va, va, va]\n")

        file.write(f"    eri['abab_oooo'] = self.g_abab[oa, ob, oa, ob]\n")
        file.write(f"    eri['abab_oovo'] = self.g_abab[oa, ob, va, ob]\n")
        file.write(f"    eri['abab_oovv'] = self.g_abab[oa, ob, va, vb]\n")
        file.write(f"    eri['abab_vooo'] = self.g_abab[va, ob, oa, ob]\n")
        file.write(f"    eri['abab_vovo'] = self.g_abab[va, ob, va, ob]\n")
        file.write(f"    eri['abab_vovv'] = self.g_abab[va, ob, va, vb]\n")
        file.write(f"    eri['abab_vvoo'] = self.g_abab[va, vb, oa, ob]\n")
        file.write(f"    eri['abab_vvvo'] = self.g_abab[va, vb, va, ob]\n")
        file.write(f"    eri['abab_vvvv'] = self.g_abab[va, vb, va, vb]\n")
        file.write(f"    eri['abba_oovo'] = -self.g_abab[oa, ob, oa, vb].transpose(0,1,3,2)\n")
        file.write(f"    eri['abba_vovo'] = -self.g_abab[va, ob, oa, vb].transpose(0,1,3,2)\n")
        file.write(f"    eri['abba_vvvo'] = -self.g_abab[va, vb, oa, vb].transpose(0,1,3,2)\n")
        file.write(f"    eri['baab_vooo'] = -self.g_abab[oa, vb, oa, ob].transpose(1,0,2,3)\n")
        file.write(f"    eri['baab_vovo'] = -self.g_abab[oa, vb, va, ob].transpose(1,0,2,3)\n")
        file.write(f"    eri['baab_vovv'] = -self.g_abab[oa, vb, va, vb].transpose(1,0,2,3)\n")
        file.write(f"    eri['baba_vovo'] = self.g_abab[oa, vb, oa, vb].transpose(1,0,3,2)\n")

        file.write(f"    eri['bbbb_oooo'] = self.g_bbbb[ob, ob, ob, ob]\n")
        file.write(f"    eri['bbbb_oovo'] = self.g_bbbb[ob, ob, vb, ob]\n")
        file.write(f"    eri['bbbb_oovv'] = self.g_bbbb[ob, ob, vb, vb]\n")
        file.write(f"    eri['bbbb_vooo'] = self.g_bbbb[vb, ob, ob, ob]\n")
        file.write(f"    eri['bbbb_vovo'] = self.g_bbbb[vb, ob, vb, ob]\n")
        file.write(f"    eri['bbbb_vovv'] = self.g_bbbb[vb, ob, vb, vb]\n")
        file.write(f"    eri['bbbb_vvoo'] = self.g_bbbb[vb, vb, ob, ob]\n")
        file.write(f"    eri['bbbb_vvvo'] = self.g_bbbb[vb, vb, vb, ob]\n")
        file.write(f"    eri['bbbb_vvvv'] = self.g_bbbb[vb, vb, vb, vb]\n")

        # kronecker delta
        file.write(f"    Id = {{}}\n")
        file.write(f"    noa = t1['aa'].shape[1]\n")
        file.write(f"    nob = t1['bb'].shape[1]\n")
        file.write(f"    Id['aa_oo'] = np.eye(noa, noa)\n")
        file.write(f"    Id['bb_oo'] = np.eye(nob, nob)\n")
        # scalars
        file.write(f"    scalars_ = {{}}\n")
        # temporary arrays
        file.write(f"    tmps_ = {{}}\n")

        # pq graph output
        file.write(graph.str("python"))

        if residual_name == 'r3':
            file.write(f"    return r3_aaaaaa, r3_aabaab, r3_abbabb, r3_bbbbbb\n")
        elif residual_name == 'r2':
            file.write(f"    return r2_aaaa, r2_abab, r2_bbbb\n")
        elif residual_name == 'r1':
            file.write(f"    return r1_aa, r1_bb\n")
        else:
            file.write(f"    return {residual_name}\n")

    #print("Code generation complete")

def eomcc_sigma(sigma_name, T, L, R, function_name, spin_block = True):

    # right- or left-hand sigma?
    is_right = True
    if len(L) > len(R):
        is_right = False

    pq = pdaggerq.pq_helper("fermi")

    # set bra
    pq.set_left_operators(L)

    # set ket
    pq.set_right_operators(R)

    # add similarity-transformed Hamiltonian
    pq.add_st_operator(1.0, ['f'], T)
    pq.add_st_operator(1.0, ['v'], T)

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

    # write function 
    with open(f"generated_equations/{function_name}.py", "w") as file:

        # initialize
        file.write(f"import numpy as np\n")
        file.write(f"from numpy import einsum\n")
        if is_right:
            file.write(f"def {function_name}(self, r0, r1_aa, r1_bb, r2_aaaa, r2_abab, r2_bbbb):\n")
        else:
            file.write(f"def {function_name}(self, l0, l1_aa, l1_bb, l2_aaaa, l2_abab, l2_bbbb):\n")
        file.write(f"    t1 = {{}}\n")
        file.write(f"    t1['aa'] = self.ccsd.t1_aa\n")
        file.write(f"    t1['bb'] = self.ccsd.t1_bb\n")
        file.write(f"    t2 = {{}}\n")
        file.write(f"    t2['aaaa'] = self.ccsd.t2_aaaa\n")
        file.write(f"    t2['abab'] = self.ccsd.t2_abab\n")
        file.write(f"    t2['bbbb'] = self.ccsd.t2_bbbb\n")
        if is_right:
            file.write(f"    r1 = {{}}\n")
            file.write(f"    r1['aa'] = r1_aa\n")
            file.write(f"    r1['bb'] = r1_bb\n")
            file.write(f"    r2 = {{}}\n")
            file.write(f"    r2['aaaa'] = r2_aaaa\n")
            file.write(f"    r2['abab'] = r2_abab\n")
            file.write(f"    r2['bbbb'] = r2_bbbb\n")
        else:
            file.write(f"    l1 = {{}}\n")
            file.write(f"    l1['aa'] = l1_aa\n")
            file.write(f"    l1['bb'] = l1_bb\n")
            file.write(f"    l2 = {{}}\n")
            file.write(f"    l2['aaaa'] = l2_aaaa\n")
            file.write(f"    l2['abab'] = l2_abab\n")
            file.write(f"    l2['bbbb'] = l2_bbbb\n")
        file.write(f"    oa = self.ccsd.oa\n")
        file.write(f"    ob = self.ccsd.ob\n")
        file.write(f"    va = self.ccsd.va\n")
        file.write(f"    vb = self.ccsd.vb\n")

        file.write(f"    f = {{}}\n")
        for spin in ['a', 'b']:
            for block1 in ['o', 'v']:
                for block2 in ['o', 'v']:
                    file.write(f"    f['{spin}{spin}_{block1}{block2}'] = self.ccsd.f_{spin}{spin}[{block1}{spin}, {block2}{spin}]\n")

        file.write(f"    eri = {{}}\n")
        file.write(f"    eri['aaaa_oooo'] = self.ccsd.g_aaaa[oa, oa, oa, oa]\n")
        file.write(f"    eri['aaaa_oovo'] = self.ccsd.g_aaaa[oa, oa, va, oa]\n")
        file.write(f"    eri['aaaa_oovv'] = self.ccsd.g_aaaa[oa, oa, va, va]\n")
        file.write(f"    eri['aaaa_vooo'] = self.ccsd.g_aaaa[va, oa, oa, oa]\n")
        file.write(f"    eri['aaaa_vovo'] = self.ccsd.g_aaaa[va, oa, va, oa]\n")
        file.write(f"    eri['aaaa_vovv'] = self.ccsd.g_aaaa[va, oa, va, va]\n")
        file.write(f"    eri['aaaa_vvoo'] = self.ccsd.g_aaaa[va, va, oa, oa]\n")
        file.write(f"    eri['aaaa_vvvo'] = self.ccsd.g_aaaa[va, va, va, oa]\n")
        file.write(f"    eri['aaaa_vvvv'] = self.ccsd.g_aaaa[va, va, va, va]\n")

        file.write(f"    eri['abab_oooo'] = self.ccsd.g_abab[oa, ob, oa, ob]\n")
        file.write(f"    eri['abab_oovo'] = self.ccsd.g_abab[oa, ob, va, ob]\n")
        file.write(f"    eri['abab_oovv'] = self.ccsd.g_abab[oa, ob, va, vb]\n")
        file.write(f"    eri['abab_vooo'] = self.ccsd.g_abab[va, ob, oa, ob]\n")
        file.write(f"    eri['abab_vovo'] = self.ccsd.g_abab[va, ob, va, ob]\n")
        file.write(f"    eri['abab_vovv'] = self.ccsd.g_abab[va, ob, va, vb]\n")
        file.write(f"    eri['abab_vvoo'] = self.ccsd.g_abab[va, vb, oa, ob]\n")
        file.write(f"    eri['abab_vvvo'] = self.ccsd.g_abab[va, vb, va, ob]\n")
        file.write(f"    eri['abab_vvvv'] = self.ccsd.g_abab[va, vb, va, vb]\n")
        file.write(f"    eri['abba_oovo'] = -self.ccsd.g_abab[oa, ob, oa, vb].transpose(0,1,3,2)\n")
        file.write(f"    eri['abba_vovo'] = -self.ccsd.g_abab[va, ob, oa, vb].transpose(0,1,3,2)\n")
        file.write(f"    eri['abba_vvvo'] = -self.ccsd.g_abab[va, vb, oa, vb].transpose(0,1,3,2)\n")
        file.write(f"    eri['baab_vooo'] = -self.ccsd.g_abab[oa, vb, oa, ob].transpose(1,0,2,3)\n")
        file.write(f"    eri['baab_vovo'] = -self.ccsd.g_abab[oa, vb, va, ob].transpose(1,0,2,3)\n")
        file.write(f"    eri['baab_vovv'] = -self.ccsd.g_abab[oa, vb, va, vb].transpose(1,0,2,3)\n")
        file.write(f"    eri['baba_vovo'] = self.ccsd.g_abab[oa, vb, oa, vb].transpose(1,0,3,2)\n")

        file.write(f"    eri['bbbb_oooo'] = self.ccsd.g_bbbb[ob, ob, ob, ob]\n")
        file.write(f"    eri['bbbb_oovo'] = self.ccsd.g_bbbb[ob, ob, vb, ob]\n")
        file.write(f"    eri['bbbb_oovv'] = self.ccsd.g_bbbb[ob, ob, vb, vb]\n")
        file.write(f"    eri['bbbb_vooo'] = self.ccsd.g_bbbb[vb, ob, ob, ob]\n")
        file.write(f"    eri['bbbb_vovo'] = self.ccsd.g_bbbb[vb, ob, vb, ob]\n")
        file.write(f"    eri['bbbb_vovv'] = self.ccsd.g_bbbb[vb, ob, vb, vb]\n")
        file.write(f"    eri['bbbb_vvoo'] = self.ccsd.g_bbbb[vb, vb, ob, ob]\n")
        file.write(f"    eri['bbbb_vvvo'] = self.ccsd.g_bbbb[vb, vb, vb, ob]\n")
        file.write(f"    eri['bbbb_vvvv'] = self.ccsd.g_bbbb[vb, vb, vb, vb]\n")

        # kronecker delta
        file.write(f"    Id = {{}}\n")
        file.write(f"    noa = t1['aa'].shape[1]\n")
        file.write(f"    nob = t1['bb'].shape[1]\n")
        file.write(f"    Id['aa_oo'] = np.eye(noa, noa)\n")
        file.write(f"    Id['bb_oo'] = np.eye(nob, nob)\n")
        # scalars
        file.write(f"    scalars_ = {{}}\n")
        # temporary arrays
        file.write(f"    tmps_ = {{}}\n")

        # pq graph output
        file.write(graph.str("python"))

        if is_right:
            if sigma_name == 'sigma2':
                file.write(f"    return sigma2_aaaa, sigma2_abab, sigma2_bbbb\n")
            elif sigma_name == 'sigma1':
                file.write(f"    return sigma1_aa, sigma1_bb\n")
            else:
                file.write(f"    return {sigma_name}\n")
        else:
            if sigma_name == 'sigma2':
                file.write(f"    return sigma2_aaaa.transpose(2,3,0,1), sigma2_abab.transpose(2,3,0,1), sigma2_bbbb.transpose(2,3,0,1)\n")
            elif sigma_name == 'sigma1':
                file.write(f"    return sigma1_aa.transpose(1,0), sigma1_bb.transpose(1,0)\n")
            else:
                file.write(f"    return {sigma_name}\n")

    #print("Code generation complete")
