def _is_nuclear(label):
    # multicomponent (NEO) nuclear orbital labels carry the 'n' prefix (ni/nj, na/nb)
    return len(label) > 1 and label[0] == 'n'


def _spin_types(n):
    # the distinct spin cases for n equivalent fermionic labels (one species)
    return {6: ["aaaaaa", "aabaab", "abbabb", "bbbbbb"],
            5: ["aaaaa", "aabaa", "abbab", "bbbbb"],
            4: ["aaaa", "abab", "bbbb"],
            3: ["aaa", "abb", "aba", "bbb"],
            2: ["aa", "bb"],
            1: ["a", "b"],
            0: [""]}.get(n, None)


def get_spin_labels(ops, nuclear_spin="high-spin"):
    """
    Get spin labels for the given operators.

    Electron labels are spin-blocked into the usual alpha/beta cases.  Multicomponent
    (NEO) nuclear labels (the 'n' prefix) are treated as their own spin species:

      nuclear_spin="high-spin"  -> a single spin channel (one quantum proton / positron,
                                   or several identical high-spin nuclei); nuclear labels
                                   are pinned to alpha.  This is the closed-shell-electron
                                   default and the case that matters for spin-tracing.
      nuclear_spin="full"       -> the nuclear species gets its own independent alpha/beta
                                   manifold, enumerated as a product with the electron
                                   cases (needed for >=2 quantum nuclei and nuclear pairing
                                   / superfluidity, where the alpha_n-beta_n channel is the
                                   pairing channel).

    With no nuclear labels this is byte-for-byte the original electron-only behavior.

    Args:
        ops (list): List of operators.
        nuclear_spin (str): "high-spin" or "full" treatment of the nuclear species.

    Returns:
        dict: Dictionary mapping spin-case names to label-spin mappings.
    """
    labels = set()
    found = False
    for op in ops:
        for subop in op:
            if "(" not in subop:
                continue
            subop_labels = subop[subop.find("(") + 1:subop.find(")")].split(",")
            for label in subop_labels:
                labels.add(label)
                found = True
    if not found:
        return {"": {}}

    elec = sorted(l for l in labels if not _is_nuclear(l))
    nuc = sorted(l for l in labels if _is_nuclear(l))

    e_types = _spin_types(len(elec))
    if e_types is None:
        raise ValueError("Invalid number of electron labels for spin blocking")

    # electron-only: original behavior exactly (spin-case name is the electron spin string)
    if not nuc:
        spin_map = {}
        for spin in e_types:
            if len(elec) != len(spin):
                continue
            spin_map[spin] = {label: spin[i] for i, label in enumerate(elec)}
        return spin_map

    # multicomponent: combine the electron cases with the nuclear treatment
    if nuclear_spin == "high-spin":
        n_types = [""]                       # single channel; nuclear pinned to alpha below
    elif nuclear_spin == "full":
        n_types = _spin_types(len(nuc))
        if n_types is None:
            raise ValueError("Invalid number of nuclear labels for spin blocking")
    else:
        raise ValueError("nuclear_spin must be 'high-spin' or 'full'")

    spin_map = {}
    for espin in e_types:
        if len(elec) != len(espin):
            continue
        emap = {label: espin[i] for i, label in enumerate(elec)}
        for nspin in n_types:
            if nuclear_spin == "high-spin":
                nmap = {label: "a" for label in nuc}
                name = espin + "_n"          # nuclear high-spin tag
            else:
                if len(nuc) != len(nspin):
                    continue
                nmap = {label: nspin[i] for i, label in enumerate(nuc)}
                name = espin + "_n" + nspin
            m = dict(emap); m.update(nmap)
            spin_map[name] = m
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