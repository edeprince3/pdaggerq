from pdaggerq.numerical.solvers.cc import cc 
from pdaggerq.numerical.codegen.autogen import cc_residual

class QED_CCSD_21:
    def __init__(self, wfn, mol, **kwargs):

        self.wfn = wfn
        self.mol = mol
        
        # Extract optional kwargs with defaults
        self.pq_graph_options = kwargs.get('pq_graph_options', None)
        self.nfzc = kwargs.get('nfzc', 0)
        self.e_convergence = kwargs.get('e_convergence', 1e-8)
        self.r_convergence = kwargs.get('r_convergence', 1e-6)
        self.max_iter = kwargs.get('max_iter', 50)
        self.cavity_lambda = kwargs.get('cavity_lambda', [0.0, 0.0, 0.0])
        self.cavity_frequency = kwargs.get('cavity_frequency', 0.07349864501573)

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}
       
        T = ['t1', 't2', 'tb1', 'teb11', 'teb21']

        # Generate equations
        cc_energy_func = cc_residual('cc_energy',
            T,
            [['1']],
            'cc_energy',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        t1_residual_func = cc_residual('r1',
            T,
            [['e1(i,a)']],
            't1_residual',
            indices = ['a', 'i'],
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        t2_residual_func = cc_residual('r2',
            T,
            [['e2(i,j,b,a)']],
            't2_residual',
            indices = ['a', 'b', 'i', 'j'],
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        t0_1p_residual_func = cc_residual('r0_1p',
            T,
            [['B-']],
            't0_1p_residual',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        t1_1p_residual_func = cc_residual('r1_1p',
            T,
            [['B-','e1(i,a)']],
            't1_1p_residual',
            indices = ['a', 'i'],
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        t2_1p_residual_func = cc_residual('r2_1p',
            T,
            [['B-','e2(i,j,b,a)']],
            't2_1p_residual',
            indices = ['a', 'b', 'i', 'j'],
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(cc_energy_func, globals(), local_namespace)
        exec(t1_residual_func, globals(), local_namespace)
        exec(t2_residual_func, globals(), local_namespace)
        exec(t0_1p_residual_func, globals(), local_namespace)
        exec(t1_1p_residual_func, globals(), local_namespace)
        exec(t2_1p_residual_func, globals(), local_namespace)

        # amplitude dictionaries to pass into the solver
        t1 = {
            'spaces' : ['v', 'o'],
            'spins' : [['a','a'], ['b','b']],
            'residual' : local_namespace["t1_residual"]
        }

        t2 = {
            'spaces' : ['vv','oo'],
            'spins' : [['aa','aa'], ['ab','ab'], ['bb','bb']],
            'residual' : local_namespace["t2_residual"]
        }

        t0_1p = {
            'nph' : 1,
            'residual' : local_namespace["t0_1p_residual"]
        }

        t1_1p = {
            'nph' : 1,
            'spaces' : ['v', 'o'],
            'spins' : [['a','a'], ['b','b']],
            'residual' : local_namespace["t1_1p_residual"]
        }

        t2_1p = {
            'nph' : 1,
            'spaces' : ['vv','oo'],
            'spins' : [['aa','aa'], ['ab','ab'], ['bb','bb']],
            'residual' : local_namespace["t2_1p_residual"]
        } 
        
        self.T_list = [t1, t2, t0_1p, t1_1p, t2_1p]

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
            is_qed = True,
            T_list = self.T_list,
            cavity_lambda = self.cavity_lambda,
            cavity_frequency= self.cavity_frequency
        )
        
        en = self.cc_solver.t_solver()
        self.efzc = self.cc_solver.efzc
        self.nuclear_repulsion_energy = self.cc_solver.nuclear_repulsion_energy
        self.enuc_dse = self.cc_solver.enuc_dse

        return en

    def lambda_solver(self):

        # Import pq codegen functions 
        from pdaggerq.numerical.codegen.autogen import lambda_cc_residual
        from pdaggerq.numerical.codegen.autogen import lambda_cc_pseudoenergy

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}

        # Generate lambda equations
        T = ['t1', 't2', 'tb1', 'teb11', 'teb21']
        L = [['l1'], ['l2'], ['lb1'], ['leb11'], ['leb21']]

        cc_pseudoenergy_func = lambda_cc_pseudoenergy('cc_pseudoenergy',
            L,
            [['1']],
            'cc_pseudoenergy',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )
        l1_residual_func = lambda_cc_residual('r1',
            T,
            L,
            ['e1(a,i)'],
            'l1_residual',
            indices = ['a', 'i'],
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        l2_residual_func = lambda_cc_residual('r2',
            T,
            L,
            ['e2(a,b,j,i)'],
            'l2_residual',
            indices = ['a', 'b', 'i', 'j'],
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        l0_1p_residual_func = lambda_cc_residual('r0_1p',
            T,
            L,
            ['B+'],
            'l0_1p_residual',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )   
            
        l1_1p_residual_func = lambda_cc_residual('r1_1p',
            T,
            L,
            ['B+','e1(a,i)'],
            'l1_1p_residual',
            indices = ['a', 'i'],
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )
        
        l2_1p_residual_func = lambda_cc_residual('r2_1p',
            T,
            L,
            ['B+','e2(a,b,j,i)'],
            'l2_1p_residual',
            indices = ['a', 'b', 'i', 'j'],
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(cc_pseudoenergy_func, globals(), local_namespace)
        exec(l1_residual_func, globals(), local_namespace)
        exec(l2_residual_func, globals(), local_namespace)
        exec(l0_1p_residual_func, globals(), local_namespace)
        exec(l1_1p_residual_func, globals(), local_namespace)
        exec(l2_1p_residual_func, globals(), local_namespace)

        # lambda amplitude dictionaries to pass into the solver
        l1 = {
            'spaces' : ['v', 'o'],
            'spins' : [['a','a'], ['b','b']],
            'residual' : local_namespace["l1_residual"]
        }

        l2 = {
            'spaces' : ['vv','oo'],
            'spins' : [['aa','aa'], ['ab','ab'], ['bb','bb']],
            'residual' : local_namespace["l2_residual"]
        }

        l0_1p = {
            'nph' : 1,
            'residual' : local_namespace["l0_1p_residual"]
        }

        l1_1p = {
            'nph' : 1,
            'spaces' : ['v', 'o'],
            'spins' : [['a','a'], ['b','b']],
            'residual' : local_namespace["l1_1p_residual"]
        }

        l2_1p = {
            'nph' : 1,
            'spaces' : ['vv','oo'],
            'spins' : [['aa','aa'], ['ab','ab'], ['bb','bb']],
            'residual' : local_namespace["l2_1p_residual"]
        } 

        # initialize lambdas in cc_solver
        self.cc_solver.initialize_lambda([l1, l2, l0_1p, l1_1p, l2_1p], cc_pseudoenergy_func = local_namespace['cc_pseudoenergy'])

        en = self.cc_solver.lambda_solver()

        return en
