import pdaggerq

def main():

    ops = [['w0'], ['f'], ['v'], ['d+'], ['d-']]
    coeffs = [1.0, 1.0, 1.0, -1.0, -1.0]

    c_ops = [['B+'], ['B-']]
    c_coeffs = [1.0, 1.0]

    T = ['t2', 't0,1', 't1,1', 't2,2']

    proj = {
        "energy": [['1']],
        "rt1": [['e1(i,a)']],
        "rt2": [['e2(i,j,b,a)']],
        "ru0": [['B-']],
        "ru1": [['B-', 'e1(i,a)']],
        "ru2": [['B-', 'e2(i,j,b,a)']],
    }

    eqs = {}

    for i, items in enumerate(proj.items()):

            proj, P = items

            # set up pq
            pq = pdaggerq.pq_helper("fermi")

            # get name of eq
            proj_eqname = proj

            print("Deriving equation: ", f"{proj_eqname} = <{P}| Hbar |0>", flush=True)

            # set projection operators
            pq.set_left_operators(P)

            # add similarity transformed operators
            for j, op in enumerate(ops):
                pq.add_st_operator(coeffs[j], op, T)

            # simplify and block by spin
            pq.simplify()
            block_by_spin(pq, proj_eqname, P, eqs)
            
            # print terms
            terms = pq.fully_contracted_strings()
            for term in terms:
                print(term)
            print()

            

            ### now do coherent state operators ###
            pq.clear()
            pq = pdaggerq.pq_helper("fermi")

            # get name of eq
            proj_eqname = "c" + proj_eqname
            print("Deriving equation: ", f"{proj_eqname} = <{P}| B* + B |0>", flush=True)

            # set projection operators
            pq.set_left_operators(P)

            # add similarity transformed operators
            for j, op in enumerate(c_ops):
                pq.add_st_operator(c_coeffs[j], op, T)

            # simplify and block by spin
            pq.simplify()
            block_by_spin(pq, proj_eqname, P, eqs)

            # print terms
            terms = pq.fully_contracted_strings()
            for term in terms:
                print(term)
            print()

            # remove pq
            del pq

def get_spin_labels(ops):
    spin_map = {}

    # make spin list using index

    labels = set()
    found = False
    for op in ops:       
        for subop in op:
            if "(" not in subop:
                continue # no labels

            # get all labels in parentheses
            subop_labels = subop[subop.find("(")+1:subop.find(")")].split(",")
            for label in subop_labels:
                labels.add(label)
                found = True

    if not found:
        return {"": {}}

    # sort labels
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
            continue  # incompatible spin


        # create dictionary mapping labels to spins
        label_to_spin = {}
        for i, label in enumerate(labels):
            label_to_spin[label] = spin[i]

        spin_map[spin] = label_to_spin

    return spin_map

def block_by_spin(pq, eqname, ops, eqs):
    spin_map = get_spin_labels(ops)
    print(f"    blocking by spin:", flush=True)
    for spins, label_to_spin in spin_map.items():
        print(f"        {spins} -> ", end="")
        for label, spin in label_to_spin.items():
            print(f"{label} -> {spin}", end=", ")
        print()
    print()
    for spins, label_to_spin in spin_map.items():

        # create name for equation
        spin_eqname = eqname + "_" + spins
        
        
        pq.block_by_spin(label_to_spin)
        eqs[spin_eqname] = pq.clone()
    
        terms = pq.fully_contracted_strings()
        for term in terms:
            print(term)
        print()

if __name__ == "__main__":
    main()
