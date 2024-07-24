import pdaggerq

def main():

    T = ['t1', 't2', 'u0', 'u1', 'u2']
    L = [['l0'], ['l1'], ['l2'], ['m0'], ['m1'], ['m2']]
    R = [['r0'], ['r1'], ['r2'], ['s0'], ['s1'], ['s2']]

    rdms = [
        ['e1(a,b)'], # vv
        ['e1(a,i)'], # vo
        ['e1(i,a)'], # ov
        ['e1(i,j)'], # oo
    ]


    eqnames = [ "rdm_vv", "rdm_vo", "rdm_ov", "rdm_oo" ]
    eqs = {}

    for i, rdm in enumerate(rdms):
        # set up pq
        pq = pdaggerq.pq_helper("fermi")

        # get name of eq
        eqname = eqnames[i]

        print("Deriving equation: ", f"{eqname} = <{L}| {rdm} |{R}>", flush=True)

        # set projection operators
        pq.set_left_operators(L)
        pq.set_right_operators(R)

        # add similarity transformed operators
        pq.add_st_operator(1.0, rdm, T)

        # simplify and block by spin
        pq.simplify()
        block_by_spin(pq, eqname, rdm, eqs)

        # remove pq
        del pq

    exit(0)
    # enable pq_graph
    graph = pdaggerq.pq_graph({
        'verbose':  True,
        'batched': False,
        'allow_merge': True,
        'use_trial_index': True,
        'conditions': {
            "u0": ["u0", "m0", "s0"],
            "u1": ["u1", "m1", "s1"],
            "u2": ["u2", "m2", "s2"]
        },
        'nthreads': -1,
    })

    # add equations to graph
    for proj_eqname, eq in eqs.items():
        graph.add(eq, proj_eqname, ['a', 'b', 'i', 'j'])

    # optimize graph
    graph.optimize()
    graph.print("cpp")
    graph.analysis()
    graph.write_dot("qed_1rdm.dot")

    return graph

def get_spin_labels(ops):
    spin_map = {}

    # make spin list using index

    labels = set()
    for op in ops: # only one operator
        if "(" not in op:
            continue # no labels

        # get all labels in parentheses
        op_labels = op[op.find("(")+1:op.find(")")].split(",")
        for label in op_labels:
            labels.add(label)
    labels = sorted(labels)
    labels = list(labels)
    
    # block by spin (aaaa, abab, and bbbb)
    spin_types = None
    if len(labels) == 4:
        spin_types = ["aaaa", "abab", "bbbb"]
    elif len(labels) == 3:
        spin_types = ["aaa", "abb", "aba", "bbb"]
    elif len(labels) == 2:
        spin_types = ["aa", "bb"]
    elif len(labels) == 1:
        spin_types = ["a", "b"]
    elif len(labels) == 0:
        spin_types = []

    for spin in spin_types: 
        if len(labels) != len(spin):
            continue # incompatible spin


        # create dictionary mapping labels to spins
        label_to_spin = {}
        for i, label in enumerate(labels):
            label_to_spin[label] = spin[i]

        spin_map[spin] = label_to_spin

    return spin_map

def block_by_spin(pq, eqname, ops, eqs):
    spin_map = get_spin_labels(ops)
    for spins, label_to_spin in spin_map.items():
        print(f"    blocking by labels:", flush=True) 
        for label, spin in label_to_spin.items():
            print(f"        {label} -> {spin}", flush=True)
        print()

        # create name for equation
        spin_eqname = eqname + "_" + spins
        
        
        pq.block_by_spin(label_to_spin)
        eqs[spin_eqname] = pq
    
        terms = pq.fully_contracted_strings()
        for term in terms:
            print(term)
        print()

if __name__ == "__main__":
    main()
