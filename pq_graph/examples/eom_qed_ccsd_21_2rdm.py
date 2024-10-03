import pdaggerq
from extract_spins import *

def main():

    # excitation operators
    T = ['t1', 't2', 't0,1', 't1,1', 't2,1']

    # left and right excitation operators
    L = [['l0'], ['l1'], ['l2'], ['l0,1'], ['l1,1'], ['l2,1']]
    R = [['r0'], ['r1'], ['r2'], ['r0,1'], ['r1,1'], ['r2,1']]

    rdms = [
        [['e2(a,b,c,d)']], # vvvv
        [['e2(a,b,c,i)']], # vvvo
        [['e2(a,b,i,c)']], # vvov
        [['e2(a,i,b,c)']], # vovv
        [['e2(i,a,b,c)']], # ovvv
        [['e2(i,a,b,j)']], # ovvo
        [['e2(i,a,j,b)']], # ovov
        [['e2(i,j,a,b)']], # oovv
        [['e2(i,j,a,k)']], # oovo
        [['e2(i,j,k,a)']], # ooov
        [['e2(i,j,k,l)']], # oooo
    ]

    eqnames = [ "D_vvvv",
                "D_vvvo", "D_vvov", "D_vovv", "D_ovvv",
                "D_ovvo", "D_ovov", "D_oovv",
                "D_oovo", "D_ooov", "D_oooo" ]
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

    # enable pq_graph
    graph = pdaggerq.pq_graph({
        'batched': True,
        'print_level': 3,
        'opt_level': 6,
        'nthreads': -1,
    })

    # add equations to graph
    for proj_eqname, eq in eqs.items():
        graph.add(eq, proj_eqname, ['a','b','c','d','i','j','k','l'])

    # optimize graph
    graph.optimize()
    graph.print("cpp")
    graph.analysis()
    graph.write_dot("qed_2rdm.dot")

    return graph

if __name__ == "__main__":
    main()
