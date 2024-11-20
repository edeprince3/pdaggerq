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
        ["aaa", "abb"] if len(labels) == 3 else (
            ["aa", "bb"] if len(labels) == 2 else (
                ["a"] if len(labels) == 1 else []
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
        for term in pq.fully_contracted_strings():
            print(term, flush=True)
