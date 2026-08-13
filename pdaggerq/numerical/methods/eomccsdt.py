from pdaggerq.numerical.solvers.eomcc import eomcc 
from pdaggerq.numerical.codegen.autogen import eomcc_sigma

class EOMCCSDT:

    def __init__(self, cc, **kwargs):

        self.wfn = cc.wfn
        self.mol = cc.mol
        
        # Extract optional kwargs with defaults
        self.pq_graph_options = kwargs.get('pq_graph_options', None)
        self.nstates = kwargs.get('nstates', 5)

        # initialize empty solver
        self.eomcc_solver = eomcc(cc, nstates = self.nstates)

    def right_solver(self):

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}

        # Generate equations
        T = ['t1', 't2', 't3']

        # Generate right-hand sigma equations
        R = [['r0'], ['r1'], ['r2'], ['r3']]

        right_sigma0_func = eomcc_sigma('sigma0',
            T,
            [['1']],
            R,
            'right_sigma0',
            pq_graph_options = self.pq_graph_options
        )

        right_sigma1_func = eomcc_sigma('sigma1',
            T,
            [['e1(i,a)']],
            R,
            'right_sigma1',
            pq_graph_options = self.pq_graph_options
        )

        right_sigma2_func = eomcc_sigma('sigma2',
            T,
            [['e2(i,j,b,a)']],
            R,
            'right_sigma2',
            pq_graph_options = self.pq_graph_options
        )

        right_sigma3_func = eomcc_sigma('sigma3',
            T,
            [['e3(i,j,k,c,b,a)']],
            R,
            'right_sigma3',
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(right_sigma0_func, globals(), local_namespace)
        exec(right_sigma1_func, globals(), local_namespace)
        exec(right_sigma2_func, globals(), local_namespace)
        exec(right_sigma3_func, globals(), local_namespace)

        # right-hand amplitude dictionaries to pass into the solver
        r0 = {
            'sigma' : local_namespace["right_sigma0"]
        }
        r1 = {
            'spaces' : ['v', 'o'],
            'spins' : [['a', 'a'], ['b', 'b']],
            'sigma' : local_namespace["right_sigma1"]
        }
        r2 = {
            'spaces' : ['vv', 'oo'],
            'spins' : [['aa','aa'], ['ab','ab'], ['bb', 'bb']],
            'sigma' : local_namespace["right_sigma2"]
        }
        r3 = {
            'spaces' : ['vvv', 'ooo'],
            'spins' : [['aaa','aaa'], ['aab','aab'], ['abb', 'abb'], ['bbb', 'bbb']],
            'sigma' : local_namespace["right_sigma3"]
        }

        # call solver
        self.eomcc_solver.right_solver(R_list = [r0, r1, r2, r3])

        self.eomcc_energy = self.eomcc_solver.eomcc_energy
        self.R = self.eomcc_solver.R

    def left_solver(self):

        # Generate right-hand sigma equations
        local_namespace = {}

        T = ['t1', 't2', 't3']

        L = [['l0'], ['l1'], ['l2'], ['l3']]

        left_sigma0_func = eomcc_sigma('sigma0',
            T,
            L,
            [['1']],
            'left_sigma0',
            pq_graph_options = self.pq_graph_options
        )

        left_sigma1_func = eomcc_sigma('sigma1',
            T,
            L,
            [['e1(a,i)']],
            'left_sigma1',
            pq_graph_options = self.pq_graph_options
        )

        left_sigma2_func = eomcc_sigma('sigma2',
            T,
            L,
            [['e2(a,b,j,i)']],
            'left_sigma2',
            pq_graph_options = self.pq_graph_options
        )

        left_sigma3_func = eomcc_sigma('sigma3',
            T,
            L,
            [['e3(a,b,c,k,j,i)']],
            'left_sigma3',
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(left_sigma0_func, globals(), local_namespace)
        exec(left_sigma1_func, globals(), local_namespace)
        exec(left_sigma2_func, globals(), local_namespace)
        exec(left_sigma3_func, globals(), local_namespace)

        # left-hand amplitude dictionaries to pass into the solver
        l0 = {
            'sigma' : local_namespace["left_sigma0"]
        }
        l1 = {
            'spaces' : ['v', 'o'],
            'spins' : [['a', 'a'], ['b', 'b']],
            'sigma' : local_namespace["left_sigma1"]
        }
        l2 = {
            'spaces' : ['vv', 'oo'],
            'spins' : [['aa','aa'], ['ab','ab'], ['bb', 'bb']],
            'sigma' : local_namespace["left_sigma2"]
        }
        l3 = {
            'spaces' : ['vvv', 'ooo'],
            'spins' : [['aaa','aaa'], ['aab','aab'], ['abb', 'abb'], ['bbb', 'bbb']],
            'sigma' : local_namespace["left_sigma3"]
        }

        # call solver
        self.eomcc_solver.left_solver(L_list = [l0, l1, l2, l3])

        self.eomcc_energy = self.eomcc_solver.eomcc_energy
        self.L = self.eomcc_solver.L

    def oscillator_strengths(self):

        # Import pq eomcc density matrix codegen function
        from pdaggerq.numerical.codegen.autogen import eomcc_density_matrix
           
        # Generate transition density matrix equations
        T = ['t1', 't2', 't3']
        R = [['r0'], ['r1'], ['r2'], ['r3']]
        L = [['l0'], ['l1'], ['l2'], ['l3']]
        tdm_func = eomcc_density_matrix('tdm', 
            T, 
            L, 
            R, 
            'density_matrix', 
            pq_graph_options = self.pq_graph_options
        )
        
        # Execute the code string in memory
        local_namespace = {}
        exec(tdm_func, globals(), local_namespace)

        f = self.eomcc_solver.oscillator_strengths(density_matrix_func = local_namespace['density_matrix'])

        return f
