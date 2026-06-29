"""Spin-case labels for spin-blocking CC equations (the spin axis of code gen).

`get_spin_labels(ops, nuclear_spin)` returns ``{case_name: {label: spin_char}}``;
feed one case's map to ``pq_helper.block_by_spin(...)`` (after ``simplify()``,
before ``pq_graph``) to derive that spin block. Electron labels get the usual
alpha/beta cases; multicomponent (NEO) nuclear labels (the 'n' prefix) are a
separate spin species -- ``"high-spin"`` pins them to one channel (closed-shell
default, one quantum proton), ``"full"`` gives them an independent alpha/beta
manifold (>=2 nuclei / pairing).

This is the package-canonical copy; ``pq_graph/tests/extract_spins.py`` carries
the same `get_spin_labels` for the standalone spin tests.
"""


def _is_nuclear(label):
    # NEO nuclear orbital labels carry the 'n' prefix (ni/nj occ, na/nb vir)
    return len(label) > 1 and label[0] == "n"


def _spin_types(n):
    # the distinct spin cases for n equivalent fermionic labels of one species
    return {6: ["aaaaaa", "aabaab", "abbabb", "bbbbbb"],
            5: ["aaaaa", "aabaa", "abbab", "bbbbb"],
            4: ["aaaa", "abab", "bbbb"],
            3: ["aaa", "abb", "aba", "bbb"],
            2: ["aa", "bb"],
            1: ["a", "b"],
            0: [""]}.get(n, None)


def get_spin_labels(ops, nuclear_spin="high-spin"):
    """Map each spin case to its ``{label: spin_char}`` blocking, derived from the
    external labels of ``ops`` (a list of operator-string lists, e.g.
    ``[["e2(i,ni,a,na)"]]``). With no nuclear labels this is the electron-only
    behavior. ``nuclear_spin`` is ``"high-spin"`` (single nuclear channel) or
    ``"full"`` (independent nuclear alpha/beta manifold)."""
    labels = set()
    found = False
    for op in ops:
        for subop in op:
            if "(" not in subop:
                continue
            for label in subop[subop.find("(") + 1:subop.find(")")].split(","):
                labels.add(label)
                found = True
    if not found:
        return {"": {}}

    elec = sorted(l for l in labels if not _is_nuclear(l))
    nuc = sorted(l for l in labels if _is_nuclear(l))

    e_types = _spin_types(len(elec))
    if e_types is None:
        raise ValueError("Invalid number of electron labels for spin blocking")

    if not nuc:
        return {spin: {label: spin[i] for i, label in enumerate(elec)}
                for spin in e_types if len(elec) == len(spin)}

    if nuclear_spin == "high-spin":
        n_types = [""]
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
                name = espin + "_n"
            else:
                if len(nuc) != len(nspin):
                    continue
                nmap = {label: nspin[i] for i, label in enumerate(nuc)}
                name = espin + "_n" + nspin
            m = dict(emap)
            m.update(nmap)
            spin_map[name] = m
    return spin_map
