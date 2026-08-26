from pdaggerq.numerical.solvers.cc import cc 
from pdaggerq.numerical.codegen.autogen import cc_residual

class CCD:
    def __init__(self, wfn, mol, **kwargs):

        self.wfn = wfn
        self.mol = mol
        
        # Extract optional kwargs with defaults
        self.pq_graph_options = kwargs.get('pq_graph_options', None)
        self.nfzc = kwargs.get('nfzc', 0)
        self.e_convergence = kwargs.get('e_convergence', 1e-8)
        self.r_convergence = kwargs.get('r_convergence', 1e-6)
        self.max_iter = kwargs.get('max_iter', 50)

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}
        
        # Generate equations
        T = ['t2']
        
        cc_energy_func = cc_residual('cc_energy',
            T,
            [['1']],
            'cc_energy',
            pq_graph_options = self.pq_graph_options
        )
        
        t2_residual_func = cc_residual('r2',
            T,
            [['e2(i,j,b,a)']],
            't2_residual',
            indices = ['a', 'b', 'i', 'j'],
            pq_graph_options = self.pq_graph_options
        )
        
        # Execute the code strings in memory
        exec(t2_residual_func, globals(), local_namespace)

        # amplitude dictionaries to pass into the solver
        t2 = {
            'spaces' : ['vv','oo'],
            'spins' : [['aa','aa'], ['ab','ab'], ['bb','bb']],
            'residual' : local_namespace["t2_residual"]
        }
        self.T_list = [t2]

        self.cc_energy = {}
        exec(cc_energy_func, globals(), self.cc_energy)

    def t_solver(self):
        
        # Pass pq-generated functions into the cc solver
        self.cc_solver = cc(
            self.wfn,
            self.mol,
            nfzc = self.nfzc,
            e_convergence = self.e_convergence,
            r_convergence = self.r_convergence,
            cc_energy_func = self.cc_energy["cc_energy"],
            T_list = self.T_list
        )
        self.efzc = self.cc_solver.efzc
        self.nuclear_repulsion_energy = self.cc_solver.nuclear_repulsion_energy
        
        en = self.cc_solver.t_solver()

        return en

    def lambda_solver(self):

        # Import pq codegen functions 
        from pdaggerq.numerical.codegen.autogen import lambda_cc_residual
        from pdaggerq.numerical.codegen.autogen import lambda_cc_pseudoenergy

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}

        # Generate lambda equations
        T = ['t2']
        L = [['l2']]

        cc_pseudoenergy_func = lambda_cc_pseudoenergy('cc_pseudoenergy',
            L,
            [['1']],
            'cc_pseudoenergy',
            pq_graph_options = self.pq_graph_options
        )
        l2_residual_func = lambda_cc_residual('r2',
            T,
            L,
            ['e2(a,b,j,i)'],
            'l2_residual',
            indices = ['a', 'b', 'i', 'j'],
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(cc_pseudoenergy_func, globals(), local_namespace)
        exec(l2_residual_func, globals(), local_namespace)

        # lambda amplitude dictionaries to pass into the solver
        l2 = {
            'spaces' : ['vv','oo'],
            'spins' : [['aa','aa'], ['ab','ab'], ['bb','bb']],
            'residual' : local_namespace["l2_residual"]
        }

        # initialize lambdas in cc_solver
        self.cc_solver.initialize_lambda([l2], cc_pseudoenergy_func = local_namespace['cc_pseudoenergy'])

        en = self.cc_solver.lambda_solver()
        return en

    def opdm(self):

        # Import pq cc density matrix codegen function
        from pdaggerq.numerical.codegen.autogen import cc_density_matrix

        # Generate transition density matrix equations
        T = ['t2']
        L = [['1'], ['l2']]
        opdm_func = cc_density_matrix('opdm',
            T, 
            L, 
            'density_matrix',
            pq_graph_options = self.pq_graph_options
        )

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}
        exec(opdm_func, globals(), local_namespace)

        opdm_a, opdm_b = self.cc_solver.opdm(opdm_func = local_namespace['density_matrix'])

        return opdm_a, opdm_b

    def tpdm(self):

        # Import pq cc tpdm codegen function
        from pdaggerq.numerical.codegen.autogen import cc_tpdm

        # Generate transition density matrix equations
        T = ['t2']
        L = [['1'], ['l2']]
        tpdm_func = cc_tpdm('tpdm',
            T, 
            L, 
            'density_matrix',
            pq_graph_options = self.pq_graph_options
        )

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}
        exec(tpdm_func, globals(), local_namespace)

        tpdm_aaaa, tpdm_abab, tpdm_bbbb = self.cc_solver.tpdm(tpdm_func = local_namespace['density_matrix'])

        return tpdm_aaaa, tpdm_abab, tpdm_bbbb
