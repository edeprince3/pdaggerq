from pdaggerq.numerical.solvers.cc import cc 
from pdaggerq.numerical.codegen.autogen import cc_residual

class QED_CCSD_22:
    def __init__(self, wfn, mol, **kwargs):

        self.wfn = wfn
        self.mol = mol
        
        # Extract optional kwargs with defaults
        self.pq_graph_options = kwargs.get('pq_graph_options', None)
        self.nfzc = kwargs.get('nfzc', 0)
        self.e_convergence = kwargs.get('e_convergence', 1e-8)
        self.r_convergence = kwargs.get('r_convergence', 1e-6)
        self.max_iter = kwargs.get('max_iter', 50)
        self.cavity_lambda = kwargs.get('cavity_lambda', 0.0)
        self.cavity_frequency = kwargs.get('cavity_frequency', 0.07349864501573)

        # Create an empty dictionary to hold the pq-generated equations
        local_namespace = {}
       
        T = ['t1', 't2', 'tb1', 'teb11', 'teb21', 'tb2', 'teb12', 'teb22']

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

        t0_2p_residual_func = cc_residual('r0_2p',
            T,
            [['B-', 'B-']],
            't0_2p_residual',
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        t1_2p_residual_func = cc_residual('r1_2p',
            T,
            [['B-','B-', 'e1(i,a)']],
            't1_2p_residual',
            indices = ['a', 'i'],
            is_qed = True,
            pq_graph_options = self.pq_graph_options
        )

        t2_2p_residual_func = cc_residual('r2_2p',
            T,
            [['B-', 'B-', 'e2(i,j,b,a)']],
            't2_2p_residual',
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
        exec(t0_2p_residual_func, globals(), local_namespace)
        exec(t1_2p_residual_func, globals(), local_namespace)
        exec(t2_2p_residual_func, globals(), local_namespace)

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
        t0_2p = {
            'nph' : 2,
            'residual' : local_namespace["t0_2p_residual"]
        }
        t1_2p = {
            'nph' : 2,
            'spaces' : ['v', 'o'],
            'spins' : [['a','a'], ['b','b']],
            'residual' : local_namespace["t1_2p_residual"]
        }
        t2_2p = {
            'nph' : 2,
            'spaces' : ['vv','oo'],
            'spins' : [['aa','aa'], ['ab','ab'], ['bb','bb']],
            'residual' : local_namespace["t2_2p_residual"]
        } 
        
        self.T_list = [t1, t2, t0_1p, t1_1p, t2_1p, t0_2p, t1_2p, t2_2p]

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
        self.efzc = self.cc_solver.efzc
        self.nuclear_repulsion_energy = self.cc_solver.nuclear_repulsion_energy
        self.enuc_dse = self.cc_solver.enuc_dse
        
        en = self.cc_solver.t_solver()

        return en
