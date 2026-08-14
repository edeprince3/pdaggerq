from pdaggerq.numerical.solvers.eomcc import eomcc 
from pdaggerq.numerical.codegen.autogen import eomcc_sigma

class QED_EOMCCSD_21:

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
        T = ['t1', 't2', 't0,1', 't1,1', 't2,1']

        # Generate right-hand sigma equations
        R = [['r0'], ['r1'], ['r2'], ['r0,1'], ['r1,1'], ['r2,1']]

        right_sigma0_func = eomcc_sigma('sigma0',
            T,
            [['1']],
            R,
            'right_sigma0',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        right_sigma1_func = eomcc_sigma('sigma1',
            T,
            [['e1(i,a)']],
            R,
            'right_sigma1',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        right_sigma2_func = eomcc_sigma('sigma2',
            T,
            [['e2(i,j,b,a)']],
            R,
            'right_sigma2',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        right_sigma0_1p_func = eomcc_sigma('sigma0_1p',
            T,
            [['B-']],
            R,
            'right_sigma0_1p',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        right_sigma1_1p_func = eomcc_sigma('sigma1_1p',
            T,
            [['B-', 'e1(i,a)']],
            R,
            'right_sigma1_1p',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        right_sigma2_1p_func = eomcc_sigma('sigma2_1p',
            T,
            [['B-', 'e2(i,j,b,a)']],
            R,
            'right_sigma2_1p',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(right_sigma0_func, globals(), local_namespace)
        exec(right_sigma1_func, globals(), local_namespace)
        exec(right_sigma2_func, globals(), local_namespace)
        exec(right_sigma0_1p_func, globals(), local_namespace)
        exec(right_sigma1_1p_func, globals(), local_namespace)
        exec(right_sigma2_1p_func, globals(), local_namespace)


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
        r0_1p = {
            'nph' : 1,
            'sigma' : local_namespace["right_sigma0_1p"]
        }
        r1_1p = {
            'nph' : 1,
            'spaces' : ['v', 'o'],
            'spins' : [['a', 'a'], ['b', 'b']],
            'sigma' : local_namespace["right_sigma1_1p"]
        }
        r2_1p = {
            'nph' : 1,
            'spaces' : ['vv', 'oo'],
            'spins' : [['aa','aa'], ['ab','ab'], ['bb', 'bb']],
            'sigma' : local_namespace["right_sigma2_1p"]
        }

        # call solver
        self.eomcc_solver.right_solver(R_list = [r0, r1, r2, r0_1p, r1_1p, r2_1p])

        self.eomcc_energy = self.eomcc_solver.eomcc_energy
        self.R = self.eomcc_solver.R

    def left_solver(self, L_list = None):
        raise Exception("left-hand QED-EOMCCSD-21 is not implemented")

    def oscillator_strengths(self):
        raise Exception("oscillator strengths are not implemented for QED-EOMCCSD-21")
