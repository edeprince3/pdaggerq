import pdaggerq

def main():

    ops = [['f'], ['v']]
    coeffs = [1.0, 1.0]

    T = ['t2']

    lproj = [
        # [['1']],
        [['a(e)']],
        [['a*(m)', 'a(f)', 'a(e)']],
    ]
    rproj = [
        # [['1']],
        [['a*(e)']],
        [['a*(e)', 'a*(f)', 'a(m)']],
    ]

    rproj_eqnames = ["sigmal1", "sigmal2"]
    lproj_eqnames = ["sigmar1", "sigmar2"]
    eqs = {}

    spin_list = {
        "a": {"e": "a"},
        "b": {"e": "b"},
        "aaa": {
            "e": "a",
            "f": "a",
            "m": "a"
        },
        "abb": {
            "e": "a",
            "f": "b",
            "m": "b"
        },
        "aba": {
            "e": "a",
            "f": "b",
            "m": "a"
        },
        "bbb": {
            "e": "b",
            "f": "b",
            "m": "b"
        },
    }

    for i, R in enumerate(rproj):
        for spin in spin_list.keys():
            # add to eqs
            if len(spin_list[spin]) != len(R[0]):
                continue

            # set up pq
            pq = pdaggerq.pq_helper("fermi")
            L = [['l1'], ['l2']]

            # get name of eq
            rproj_eqname = rproj_eqnames[i]
            print("Deriving equation: ", f"{rproj_eqname} = <{L}| Hbar |{R}>", flush=True)

            # set projection operators
            pq.set_left_operators_type("EA")
            pq.set_left_operators(L)
            pq.set_right_operators(R)


            # add similarity transformed operators
            for j, op in enumerate(ops):
                pq.add_st_operator(coeffs[j], op, T)

            # simplify and block by spin
            pq.simplify()

            print(rproj_eqname + "_" + spin)
            pq.block_by_spin(spin_list[spin])
            eqs[rproj_eqname + "_" + spin] = pq

            # remove pq
            del pq

    for i, L in enumerate(lproj):
        for spin in spin_list.keys():
            if len(spin_list[spin]) != len(L[0]):
                continue

            # set up pq
            pq = pdaggerq.pq_helper("fermi")
            R = [['r1'], ['r2']]

            # get name of eq
            lproj_eqname = lproj_eqnames[i]
            print("Deriving equation: ", f"{lproj_eqname} = <{L}| Hbar |{R}>", flush=True)

            # set projection operators
            pq.set_left_operators(L)
            pq.set_right_operators_type("EA")
            pq.set_right_operators(R)


            # add similarity transformed operators
            for j, op in enumerate(ops):
                pq.add_st_operator(coeffs[j], op, T)

            # simplify and block by spin
            pq.simplify()

            # add to eqs
            print(lproj_eqname + "_" + spin)
            pq.block_by_spin(spin_list[spin])
            eqs[lproj_eqname + "_" + spin] = pq

            # remove pq
            del pq

    # enable pq_graph
    graph = pdaggerq.pq_graph({
        'verbose': True,
        'batched':False,
        'allow_merge': False,
        'use_trial_index': True,
        'no_scalars': False,
        'nthreads': -1,
    })

    # add equations to graph
    for proj_eqname, eq in eqs.items():
        graph.add(eq, proj_eqname, ['a','b','i','e','f','m'])

    # optimize graph
    graph.optimize()
    #graph.reorder()
    graph.print("cpp")
    graph.analysis()
    graph.write_dot("test.dot")

    return graph

if __name__ == "__main__":
    main()
