import pdaggerq
from extract_spins import *

def main():

    # cluster operators
    T = ['t1', 't2', 't0,1', 't1,1', 't2,1']

    # left and right excitation operators
    L = [['l0'], ['l1'], ['l2'], ['l0,1'], ['l1,1'], ['l2,1']]
    R = [['r0'], ['r1'], ['r2'], ['r0,1'], ['r1,1'], ['r2,1']]

    rdms = [
        [['e1(a,b)']], # vv
        [['e1(a,i)']], # vo
        [['e1(i,a)']], # ov
        [['e1(i,j)']], # oo
    ]


    eqnames = [ "D_vv", "D_vo", "D_ov", "D_oo" ]
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
        pq.add_st_operator(1.0, rdm[0], T)

        # simplify and block by spin
        pq.simplify()
        block_by_spin(pq, eqname, rdm, eqs)

        # remove pq
        del pq

    graph = pdaggerq.pq_graph({
        'batched': False,
        'print_level': 0,
        'opt_level': 6,
        'nthreads': -1,
    })


    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # Optimize and output the graph
    graph.optimize()
    graph.print("cpp")
    graph.analysis()

if __name__ == "__main__":
    main()
