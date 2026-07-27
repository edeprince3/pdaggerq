from pdaggerq.numerical.solvers.eomcc import eomcc 
from pdaggerq.numerical.codegen.autogen import eomcc_sigma

class IP_EOMCCSD:

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
        R = [['r1'], ['r2']]
            
        right_sigma1_func = eomcc_sigma('sigma1',
            T,
            [['a*(i)']],
            R,
            'right_sigma1', 
            operator_type = 'IP',
            pq_graph_options = self.pq_graph_options
        )

        right_sigma2_func = eomcc_sigma('sigma2',
            T,
            [['a*(i)', 'a*(j)', 'a(a)']],
            R,
            'right_sigma2',
            operator_type = 'IP',
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(right_sigma1_func, globals(), local_namespace)
        exec(right_sigma2_func, globals(), local_namespace)

        # right-hand amplitude dictionaries to pass into the solver
        r1 = {
            'spaces' : ['', 'o'],
            'spins' : [['', 'a'], ['', 'b']],
            'sigma' : local_namespace["right_sigma1"]
        }
        r2 = {
            'spaces' : ['v', 'oo'],
            'spins' : [['a', 'aa'], ['a','ab'], ['b', 'ab'], ['b', 'bb']],
            'sigma' : local_namespace["right_sigma2"]
        }

        # call solver
        self.eomcc_solver.right_solver(R_list = [r1, r2])

        self.eomcc_energy = self.eomcc_solver.eomcc_energy
        self.R = self.eomcc_solver.R

    def left_solver(self):
        raise Exception("left-hand IP-EOMCCSD is not implemented")

