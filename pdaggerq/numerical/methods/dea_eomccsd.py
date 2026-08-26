from pdaggerq.numerical.solvers.eomcc import eomcc 
from pdaggerq.numerical.codegen.autogen import eomcc_sigma

class DEA_EOMCCSD:

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

        # Generate right-hand sigma equations
        T = ['t1', 't2']
        R = [['r2'], ['r3']]

        right_sigma2_func = eomcc_sigma('sigma2',
            T,
            [['a(b)', 'a(a)']],
            R,
            'right_sigma2',
            indices = ['a', 'b'],
            operator_type = 'DEA',
            pq_graph_options = self.pq_graph_options
        )

        right_sigma3_func = eomcc_sigma('sigma3',
            T,
            [['a*(i)', 'a(c)', 'a(b)', 'a(a)']],
            R,
            'right_sigma3',
            indices = ['a', 'b', 'c', 'i'],
            operator_type = 'DEA',
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(right_sigma2_func, globals(), local_namespace)
        exec(right_sigma3_func, globals(), local_namespace)

        # right-hand amplitude dictionaries to pass into the solver
        r2 = {
            'spaces' : ['vv', ''],
            'spins' : [['aa', ''], ['ab', ''], ['bb', '']],
            'sigma' : local_namespace["right_sigma2"]
        }
        r3 = {
            'spaces' : ['vvv', 'o'],
            'spins' : [['aaa', 'a'], ['aab','a'], ['aab', 'b'], ['abb', 'a'], ['abb', 'b'], ['bbb', 'b']],
            'sigma' : local_namespace["right_sigma3"]
        }

        # call solver
        self.eomcc_solver.right_solver(R_list = [r2, r3])

        self.eomcc_energy = self.eomcc_solver.eomcc_energy
        self.R = self.eomcc_solver.R

    def left_solver(self):

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}

        # Generate right-hand sigma equations
        T = ['t1', 't2']
        L = [['l2'], ['l3']]

        left_sigma2_func = eomcc_sigma('sigma2',
            T,
            L,
            [['a*(a)', 'a*(b)']],
            'left_sigma2',
            indices = ['a', 'b'],
            operator_type = 'DEA',
            pq_graph_options = self.pq_graph_options
        )

        left_sigma3_func = eomcc_sigma('sigma3',
            T,
            L,
            [['a*(a)', 'a*(b)', 'a*(c)', 'a(i)']],
            'left_sigma3',
            indices = ['a', 'b', 'c', 'i'],
            operator_type = 'DEA',
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(left_sigma2_func, globals(), local_namespace)
        exec(left_sigma3_func, globals(), local_namespace)

        # left-hand amplitude dictionaries to pass into the solver
        l2 = {
            'spaces' : ['vv', ''],
            'spins' : [['aa', ''], ['ab', ''], ['bb', '']],
            'sigma' : local_namespace["left_sigma2"]
        }
        l3 = {
            'spaces' : ['vvv', 'o'],
            'spins' : [['aaa', 'a'], ['aab','a'], ['aab', 'b'], ['abb', 'a'], ['abb', 'b'], ['bbb', 'b']],
            'sigma' : local_namespace["left_sigma3"]
        }

        # call solver
        self.eomcc_solver.left_solver(L_list = [l2, l3])

        self.eomcc_energy = self.eomcc_solver.eomcc_energy
        self.L = self.eomcc_solver.L

    def oscillator_strengths(self):
        raise Exception("oscillator strengths are not implemented for DEA-EOMCCSD")

