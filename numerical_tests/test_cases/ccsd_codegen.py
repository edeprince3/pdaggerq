from pdaggerq.numerical_utils.autogen import derive_equation
from pdaggerq.numerical_utils.autogen import configure_graph

def cc_residual(residual_name, t_ops, r_ops, function_name):

    # hamiltonian
    ops = [['f'], ['v']]
    coeffs = [1.0, 1.0]

    # cluster operators
    T = t_ops 

    # projection operators and residuals
    proj = {
        residual_name: r_ops
    }

    # dictionary to store the derived equations
    eqs = {}

    # residual equations
    for proj_eqname, P in proj.items():
        derive_equation(eqs, proj_eqname, ops, coeffs, L=P, T=T, spin_block = True)

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

        if residual_name == 'r2':
            file.write(f"    return r2_aaaa, r2_abab, r2_bbbb\n")
        elif residual_name == 'r1':
            file.write(f"    return r1_aa, r1_bb\n")
        else:
            file.write(f"    return {residual_name}\n")

    #print("Code generation complete")

def main():
    """
    generate t1 and t2 residuals for ccsd
    """
    cc_residual('cc_energy', ['t1', 't2'], [['1']], 'cc_energy')
    cc_residual('r1', ['t1', 't2'], [['e1(i,a)']], 't1_residual')
    cc_residual('r2', ['t1', 't2'], [['e2(i,j,b,a)']], 't2_residual')

if __name__ == "__main__":
    main()

