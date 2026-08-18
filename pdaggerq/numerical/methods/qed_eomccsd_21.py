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

    def left_solver(self):

        # Generate right-hand sigma equations
        local_namespace = {}

        T = ['t1', 't2', 't0,1', 't1,1', 't2,1']

        L = [['l0'], ['l1'], ['l2'], ['l0,1'], ['l1,1'], ['l2,1']]

        left_sigma0_func = eomcc_sigma('sigma0',
            T,
            L,
            [['1']],
            'left_sigma0',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        left_sigma1_func = eomcc_sigma('sigma1',
            T,
            L,
            [['e1(a,i)']],
            'left_sigma1',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        left_sigma2_func = eomcc_sigma('sigma2',
            T,
            L,
            [['e2(a,b,j,i)']],
            'left_sigma2',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        left_sigma0_1p_func = eomcc_sigma('sigma0_1p',
            T,
            L,
            [['B+']],
            'left_sigma0_1p',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        left_sigma1_1p_func = eomcc_sigma('sigma1_1p',
            T,
            L,
            [['B+', 'e1(a,i)']],
            'left_sigma1_1p',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        left_sigma2_1p_func = eomcc_sigma('sigma2_1p',
            T,
            L,
            [['B+', 'e2(a,b,j,i)']],
            'left_sigma2_1p',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code strings in memory
        exec(left_sigma0_func, globals(), local_namespace)
        exec(left_sigma1_func, globals(), local_namespace)
        exec(left_sigma2_func, globals(), local_namespace)
        exec(left_sigma0_1p_func, globals(), local_namespace)
        exec(left_sigma1_1p_func, globals(), local_namespace)
        exec(left_sigma2_1p_func, globals(), local_namespace)

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
        l0_1p = {
            'nph' : 1,
            'sigma' : local_namespace["left_sigma0_1p"]
        }
        l1_1p = {
            'nph' : 1,
            'spaces' : ['v', 'o'],
            'spins' : [['a', 'a'], ['b', 'b']],
            'sigma' : local_namespace["left_sigma1_1p"]
        }
        l2_1p = {
            'nph' : 1,
            'spaces' : ['vv', 'oo'],
            'spins' : [['aa','aa'], ['ab','ab'], ['bb', 'bb']],
            'sigma' : local_namespace["left_sigma2_1p"]
        }

        # call solver
        self.eomcc_solver.left_solver(L_list = [l0, l1, l2, l0_1p, l1_1p, l2_1p])

        self.eomcc_energy = self.eomcc_solver.eomcc_energy
        self.L_meta = self.eomcc_solver.L_meta

    def oscillator_strengths(self):

        # Import pq eomcc density matrix codegen function
        from pdaggerq.numerical.codegen.autogen import eomcc_density_matrix

        # Generate transition density matrix equations
        T = ['t1', 't2', 't0,1', 't1,1', 't2,1']
        R = [['r0'], ['r1'], ['r2'], ['r0,1'], ['r1,1'], ['r2,1']]
        L = [['l0'], ['l1'], ['l2'], ['l0,1'], ['l1,1'], ['l2,1']]
        tdm_func = eomcc_density_matrix('tdm',
            T,
            L,
            R,
            'density_matrix',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        # Execute the code string in memory
        local_namespace = {}
        exec(tdm_func, globals(), local_namespace)

        f = self.eomcc_solver.oscillator_strengths(density_matrix_func = local_namespace['density_matrix'])

        return f

    def opdm(self):

        # Import pq cc density matrix codegen function
        from pdaggerq.numerical.codegen.autogen import eomcc_density_matrix

        # Generate transition density matrix equations
        T = ['t1', 't2', 't0,1', 't1,1', 't2,1']
        R = [['r0'], ['r1'], ['r2'], ['r0,1'], ['r1,1'], ['r2,1']]
        L = [['l0'], ['l1'], ['l2'], ['l0,1'], ['l1,1'], ['l2,1']]
        opdm_func = eomcc_density_matrix('opdm',
            T,
            L,
            R,
            'density_matrix',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}
        exec(opdm_func, globals(), local_namespace)

        opdm_a, opdm_b = self.eomcc_solver.opdm(opdm_func = local_namespace['density_matrix'])

        return opdm_a, opdm_b

    def tpdm(self):

        # Import pq cc tpdm codegen function
        from pdaggerq.numerical.codegen.autogen import eomcc_tpdm

        # Generate tpdm equations
        T = ['t1', 't2', 't0,1', 't1,1', 't2,1']
        R = [['r0'], ['r1'], ['r2'], ['r0,1'], ['r1,1'], ['r2,1']]
        L = [['l0'], ['l1'], ['l2'], ['l0,1'], ['l1,1'], ['l2,1']]
        tpdm_func = eomcc_tpdm('tpdm',
            T,
            L,
            R,
            'density_matrix',
            is_qed = True,
            operator_type = 'EE',
            pq_graph_options = self.pq_graph_options
        )

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}
        exec(tpdm_func, globals(), local_namespace)

        tpdm_aaaa, tpdm_abab, tpdm_bbbb = self.eomcc_solver.tpdm(tpdm_func = local_namespace['density_matrix'])

        return tpdm_aaaa, tpdm_abab, tpdm_bbbb


    def phdm(self):
        
        # Import pq cc density matrix codegen function
        from pdaggerq.numerical.codegen.autogen import eomcc_phdm
        
        # Generate transition density matrix equations
        T = ['t1', 't2', 't0,1', 't1,1', 't2,1']
        R = [['r0'], ['r1'], ['r2'], ['r0,1'], ['r1,1'], ['r2,1']]
        L = [['l0'], ['l1'], ['l2'], ['l0,1'], ['l1,1'], ['l2,1']]
        phdm_func = eomcc_phdm('phdm',
            T,
            L,
            R,
            'density_matrix',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )   
        
        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}
        exec(phdm_func, globals(), local_namespace)

        phdm = self.eomcc_solver.phdm(phdm_func = local_namespace['density_matrix'])

        return phdm
