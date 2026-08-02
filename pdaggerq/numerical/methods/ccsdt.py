from pdaggerq.numerical.solvers.cc import cc 
from pdaggerq.numerical.codegen.autogen import cc_residual

class CCSDT:
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
        T = ['t1', 't2', 't3']
        
        cc_energy_func = cc_residual('cc_energy',
            T,
            [['1']],
            'cc_energy',
            pq_graph_options = self.pq_graph_options
        )
        
        t1_residual_func = cc_residual('r1',
            T,
            [['e1(i,a)']],
            't1_residual',
            pq_graph_options = self.pq_graph_options
        )
        
        t2_residual_func = cc_residual('r2',
            T,
            [['e2(i,j,b,a)']],
            't2_residual',
            pq_graph_options = self.pq_graph_options
        )

        t3_residual_func = cc_residual('r3',
            T,
            [['e3(i,j,k,c,b,a)']],
            't3_residual',
            pq_graph_options = self.pq_graph_options
        )
        
        # Execute the code strings in memory
        exec(t1_residual_func, globals(), local_namespace)
        exec(t2_residual_func, globals(), local_namespace)
        exec(t3_residual_func, globals(), local_namespace)

        # amplitude dictionaries to pass into the solver
        t1 = {
            'spaces' : 'vo',
            'spins' : ['aa', 'bb'],
            'residual' : local_namespace["t1_residual"]
        }
        t2 = {
            'spaces' : 'vvoo',
            'spins' : ['aaaa', 'abab', 'bbbb'],
            'residual' : local_namespace["t2_residual"]
        }
        t3 = {
            'spaces' : 'vvvooo',
            'spins' : ['aaaaaa', 'aabaab', 'abbabb', 'bbbbbb'],
            'residual' : local_namespace["t3_residual"]
        }
        self.T_list = [t1, t2, t3]

        self.cc_energy = {}
        exec(cc_energy_func, globals(), self.cc_energy)

    def t_solver(self):
        
        # Pass pq-generated functions into the cc solver
        self.cc_solver = cc(
            self.wfn,
            self.mol,
            nfzc = self.nfzc,
            cc_energy_func = self.cc_energy["cc_energy"],
            T_list = self.T_list
        )
        
        en = self.cc_solver.t_solver()

        return en

    def lambda_solver(self):

        # Import pq codegen functions 
        from pdaggerq.numerical.codegen.autogen import lambda_cc_residual
        from pdaggerq.numerical.codegen.autogen import lambda_cc_pseudoenergy

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}

        # Generate lambda equations
        T = ['t1', 't2', 't3']
        L = [['l1'], ['l2'], ['l3']]

        cc_pseudoenergy_func = lambda_cc_pseudoenergy('cc_pseudoenergy',
            L,
            [['1']],
            'cc_pseudoenergy',
            pq_graph_options = self.pq_graph_options
        )
        l1_residual_func = lambda_cc_residual('r1',
            T,
            L,
            ['e1(a,i)'],
            'l1_residual',
            pq_graph_options = self.pq_graph_options
        )

        l2_residual_func = lambda_cc_residual('r2',
            T,
            L,
            ['e2(a,b,j,i)'],
            'l2_residual',
            pq_graph_options = self.pq_graph_options
        )

        l3_residual_func = lambda_cc_residual('r3',
            T,
            L,
            ['e3(a,b,c,k,j,i)'],
            'l3_residual',
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(cc_pseudoenergy_func, globals(), local_namespace)
        exec(l1_residual_func, globals(), local_namespace)
        exec(l2_residual_func, globals(), local_namespace)
        exec(l3_residual_func, globals(), local_namespace)

        # lambda amplitude dictionaries to pass into the solver
        l1 = {
            'spaces' : 'vo',
            'spins' : ['aa', 'bb'],
            'residual' : local_namespace["l1_residual"]
        }

        l2 = {
            'spaces' : 'vvoo',
            'spins' : ['aaaa', 'abab', 'bbbb'],
            'residual' : local_namespace["l2_residual"]
        }

        l3 = {
            'spaces' : 'vvvooo',
            'spins' : ['aaaaaa', 'aabaab', 'abbabb', 'bbbbbb'],
            'residual' : local_namespace["l3_residual"]
        }

        # initialize lambdas in cc_solver
        self.cc_solver.initialize_lambda([l1, l2, l3], cc_pseudoenergy_func = local_namespace['cc_pseudoenergy'])

        en = self.cc_solver.lambda_solver()

        return en
