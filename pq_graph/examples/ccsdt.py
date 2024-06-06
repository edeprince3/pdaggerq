import pdaggerq

# set up pq_graph
graph = pdaggerq.pq_graph({
    "verbose": True,         # print out verbose analysis?
    "permute_eri": True,     # permute ERI integrals to a common order? (ovov -> vovo; ovvo -> -vovo)
    "allow_merge": True,     # merge similar terms during optimization?
    "batched": False,         # substitute intermediates in batches?
    "batch_size": 100,       # batch size for substitution
    "max_temps": -1,         # maximum number of intermediates to find
    "max_depth": 2,          # maximum depth for chain of contractions
    "max_shape": {           # a map of maximum container size for intermediates
        'o':-1,            
        'v':-1,            
    },                     
    "allow_nesting": True,   # allow nested intermediates?
    "nthreads": -1,          # number of threads to use for optimization (-1 = all)
    "conditions": {          # map of the named conditions for each operator type
        "t3":  ['t3'],       # terms that have any of these operators will be in an if statement
    }
})

T = ['t1', 't2', 't3'] # cluster amplitudes
left_ops = { # projection equations
    "singles_residual": [['e1(i,a)']],         # singles ( 0 = <0| i* a e(-T) H e(T) |0> )
    "doubles_residual": [['e2(i,j,b,a)']],     # doubles ( 0 = <0| i* j* b a e(-T) H e(T) |0> )
    "triples_residual": [['e3(i,j,k,c,b,a)']], # triples ( 0 = <0| i* j* k* b a e(-T) H e(T) |0> )        
}

for eq_name, ops in left_ops.items():
    pq = pdaggerq.pq_helper('fermi')
    pq.set_left_operators(ops)
    pq.add_st_operator(1.0,['f'], T)
    pq.add_st_operator(1.0,['v'], T)
    pq.simplify()

    # queue up the equation for optimization:
    # 1) pass the pq_helper object and the name of the equation.
    # 2) the name is used to label the left-hand side (lhs) of the equation
    # 3) the last argument (optional) overrides the ordering of the lhs indices
    graph.add(pq, eq_name, ['a', 'b', 'c', 'i', 'j', 'k'])
    pq.clear()

# optimize the equations
#graph.reorder()        # reorder contractions for optimal performance (redundant if optimize is called)
graph.optimize()       # reorders contraction and generates intermediates
graph.print("cpp")  # print the optimized equations for Python.
graph.analysis()       # prints the FLOP scaling (permutations are expanded into repeated terms for analysis)

# create a DOT file for use with Graphviz
graph.write_dot("ccsd.dot") 
