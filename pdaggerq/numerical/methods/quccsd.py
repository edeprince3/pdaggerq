from pdaggerq.numerical.solvers.cc import cc 
from pdaggerq.numerical.codegen.autogen import bernoulli_ucc_residual

class QUCCSD:
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
       
        cc_energy_func = bernoulli_ucc_residual(3,
            'cc_energy',
            T,
            [['1']], 
            'cc_energy',
            pq_graph_options = self.pq_graph_options
        )  

        t1_residual_func = bernoulli_ucc_residual(2,
            'r1',
            T,
            [['e1(i,a)']],
            't1_residual',
            pq_graph_options = self.pq_graph_options
        )  

        t2_residual_func = bernoulli_ucc_residual(2,
            'r2',
            T,
            [['e2(i,j,b,a)']],
            't2_residual',
            pq_graph_options = self.pq_graph_options
        )   
        
        # Execute the code strings in memory
        exec(t1_residual_func, globals(), local_namespace)
        exec(t2_residual_func, globals(), local_namespace)

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
        self.T_list = [t1, t2]

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
