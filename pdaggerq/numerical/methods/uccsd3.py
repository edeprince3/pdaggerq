from pdaggerq.numerical.solvers.cc import cc 
from pdaggerq.numerical.codegen.autogen import uccsd_energy
from pdaggerq.numerical.codegen.autogen import uccsd_singles_residual
from pdaggerq.numerical.codegen.autogen import uccsd_doubles_residual

class UCCSD3:
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
        T = ['t1', 't2']
      
        # Generate equations
        cc_energy_func = uccsd_energy(3,
            'cc_energy',
            'cc_energy',
            pq_graph_options = self.pq_graph_options
        )

        t1_residual_func = uccsd_singles_residual(2,
            'r1',
            [['e1(i,a)']],
            't1_residual',
            indices = ['a', 'i'],
            pq_graph_options = self.pq_graph_options
        )

        t2_residual_func = uccsd_doubles_residual(2,
            'r2',
            [['e2(i,j,b,a)']],
            't2_residual',
            indices = ['a', 'b', 'i', 'j'],
            pq_graph_options = self.pq_graph_options
        ) 
        
        # Execute the code strings in memory
        exec(t1_residual_func, globals(), local_namespace)
        exec(t2_residual_func, globals(), local_namespace)

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
        self.T_list = [t1, t2]

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
        
        en = self.cc_solver.t_solver()

        return en
