from pdaggerq.numerical.solvers.cc_response import cc_response 
from pdaggerq.numerical.codegen.autogen import eomcc_sigma
from pdaggerq.numerical.codegen.autogen import cc_response_terms

class CCSDT_RESPONSE:

    def __init__(self, cc, **kwargs):
        """
        Initialize CCSDT response method
        """

        self.cc = cc
        
        # Extract optional kwargs with defaults
        self.pq_graph_options = kwargs.get('pq_graph_options', None)
        self.perturb = kwargs.get('perturb', 'dipole')
        self.omega = kwargs.get('omega', 0.0)
        self.r_convergence = kwargs.get('r_convergence', 1e-8)

    def first_order_response_solver(self):
        """
        Generate and execute first-order response equations
        """

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}

        # Generate equations
        T = ['t1', 't2', 't3']

        # Generate right-hand sigma equations
        R = [['r1'], ['r2'], ['r3']]

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
        exec(right_sigma1_func, globals(), local_namespace)
        exec(right_sigma2_func, globals(), local_namespace)
        exec(right_sigma3_func, globals(), local_namespace)

        # Generate xi_mu = <mu| Vbar |0> equations
        L = [[]]
        xi1_func = cc_response_terms('xi1',
            T,
            L,
            ['e1(i,a)'],
            'xi1',
            term_type = 'xi',
            pq_graph_options = self.pq_graph_options
        )

        xi2_func = cc_response_terms('xi2',
            T,
            L,
            ['e2(i,j,b,a)'],
            'xi2',
            term_type = 'xi',
            pq_graph_options = self.pq_graph_options
        )

        xi3_func = cc_response_terms('xi3',
            T,
            L,
            ['e3(i,j,k,c,b,a)'],
            'xi3',
            term_type = 'xi',
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(xi1_func, globals(), local_namespace)
        exec(xi2_func, globals(), local_namespace)
        exec(xi3_func, globals(), local_namespace)

        # Generate eta_nu = <0|(1+Lambda)[Vbar,tau_nu]|0> equations
        L = [['l1'], ['l2'], ['l3']]
        eta1_func = cc_response_terms('eta1',
            T,
            L,
            ['e1(a,i)'],
            'eta1',
            term_type = 'eta',
            pq_graph_options = self.pq_graph_options
        )

        eta2_func = cc_response_terms('eta2',
            T,
            L,
            ['e2(a,b,j,i)'],
            'eta2',
            term_type = 'eta',
            pq_graph_options = self.pq_graph_options
        )

        eta3_func = cc_response_terms('eta3',
            T,
            L,
            ['e3(a,b,c,k,j,i)'],
            'eta3',
            term_type = 'eta',
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(eta1_func, globals(), local_namespace)
        exec(eta2_func, globals(), local_namespace)
        exec(eta3_func, globals(), local_namespace)

        # right-hand amplitude dictionaries to pass into the solver
        r1 = {
            'spaces' : ['v', 'o'],
            'spins' : [['a', 'a'], ['b', 'b']],
            'sigma' : local_namespace["right_sigma1"],
            'xi' : local_namespace["xi1"],
            'eta' : local_namespace["eta1"],
        }

        r2 = {
            'spaces' : ['vv', 'oo'],
            'spins' : [['aa','aa'], ['ab','ab'], ['bb', 'bb']],
            'sigma' : local_namespace["right_sigma2"],
            'xi' : local_namespace["xi2"],
            'eta' : local_namespace["eta2"],
        }

        r3 = {
            'spaces' : ['vvv', 'ooo'],
            'spins' : [['aaa','aaa'], ['aab','aab'], ['abb', 'abb'], ['bbb', 'bbb']],
            'sigma' : local_namespace["right_sigma3"],
            'xi' : local_namespace["xi3"],
            'eta' : local_namespace["eta3"],
        }

        # initialize cc response solver
        self.cc_response = cc_response(self.cc, R_list = [r1, r2, r3], perturb = self.perturb)

        # call solver
        self.cc_response.first_order_response_solver(omega = self.omega, r_convergence = self.r_convergence)

    def polarizability(self):
        """
        Evaluate cc polarizability
        """

        print('')
        print('    ==> CC polarizability <==')
        print('')

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}

        # hessian term
        T = ['t1', 't2', 't3']
        L = [['1'], ['l1'], ['l2'], ['l3']]
        Ra = ['x1', 'x2', 'x3']
        Rb = ['r1', 'r2', 'r3']
        hessian_func = cc_response_terms('hessian',
            T,
            L,
            '1',
            'hessian',
            term_type = 'hessian',
            Ra = Ra,
            Rb = Rb,
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(hessian_func, globals(), local_namespace)

        return self.cc_response.polarizability(hessian_func = local_namespace['hessian'])
