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
        derive_equation(eqs, proj_eqname, ops, coeffs, L=P, T=T)

    # Enable and configure pq_graph
    graph = configure_graph()

    # Add equations to graph
    for proj_eqname, eq in eqs.items():
        print(f"Adding equation {proj_eqname} to the graph", flush=True)
        graph.add(eq, proj_eqname)

    # optimize the graph
    graph.optimize()

    # write function 
    with open(f"{function_name}.py", "w") as file:

        # initialize
        file.write(f"import numpy as np\n")
        file.write(f"from numpy import einsum\n")
        file.write(f"def {function_name}(self):\n")
        file.write(f"    t1 = self.t1\n")
        file.write(f"    t2 = self.t2\n")
        file.write(f"    o = self.o\n")
        file.write(f"    v = self.v\n")
        file.write(f"    f = {{}}\n")
        for block1 in ['o', 'v']:
            for block2 in ['o', 'v']:
                file.write(f"    f['{block1}{block2}'] = self.f[{block1}, {block2}]\n")
        file.write(f"    eri = {{}}\n")
        # we only need oovo but not ooov, vooo but not ovoo, etc
        blocks = ['v', 'o']
        for i, block1 in enumerate(blocks):
            for j, block2 in enumerate(blocks):
                if i > j: 
                    continue
                for k, block3 in enumerate(blocks):
                    for l, block4 in enumerate(blocks):
                        if k > l: 
                            continue
                        file.write(f"    eri['{block1}{block2}{block3}{block4}'] = self.g[{block1}, {block2}, {block3}, {block4}]\n")
        # kronecker delta
        file.write(f"    Id = {{}}\n")
        file.write(f"    no = t1.shape[1]\n")
        file.write(f"    Id['oo'] = np.eye(no, no)\n")
        # scalars
        file.write(f"    scalars_ = {{}}\n")
        # temporary arrays
        file.write(f"    tmps_ = {{}}\n")
        # pq graph output
        file.write(graph.str("python"))

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

