import pdaggerq

def generate_pq():
    pq = pdaggerq.pq_helper("fermi")
    pq.set_print_level(0)
    
    # energy equation
    T = ['t1','t2','t3']
    
    pq.set_left_operators([['1']])
    
    print('')
    print('    < 0 | e(-T) H e(T) | 0> :')
    print('')
    
    pq.add_st_operator(1.0,['f'],T)
    pq.add_st_operator(1.0,['v'],T)
    
    pq.simplify()
    pq.save('energy.bin')
    pq.clear()
    
    # singles equations
    
    pq.set_left_operators([['e1(i,a)']])
    
    print('')
    print('    < 0 | i* a e(-T) H e(T) | 0> :')
    print('')
    
    pq.add_st_operator(1.0,['f'],T)
    pq.add_st_operator(1.0,['v'],T)
    
    pq.simplify()
    pq.save('singles_resid.bin')
    pq.clear()
    
    # doubles equations
    
    pq.set_left_operators([['e2(i,j,b,a)']])
    
    print('')
    print('    < 0 | i* j* b a e(-T) H e(T) | 0> :')
    print('')
    
    pq.add_st_operator(1.0,['f'],T)
    pq.add_st_operator(1.0,['v'],T)
    
    pq.simplify()
    pq.save('doubles_resid.bin')
    pq.clear()
    
    # triples equations
    
    pq.set_left_operators([['e3(i,j,k,c,b,a)']])
    
    print('')
    print('    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
    print('')
    
    pq.add_st_operator(1.0,['f'],T)
    pq.add_st_operator(1.0,['v'],T)
    
    pq.simplify()
    pq.save('triples_resid.bin')
    pq.clear()

def load_pq():
    
    # set up pq_helper
    pq = pdaggerq.pq_helper("fermi")
    pq.set_print_level(0)
    
    # set up pq_graph
    graph = pdaggerq.pq_graph({
        'verbose': True,
        'nthreads': -1,
    })
    
    # load energy
    pq.load("energy.bin")
    graph.add(pq, "energy")

    # load singles_resid
    pq.load("singles_resid.bin")
    graph.add(pq, "singles_resid")

    # load doubles_resid
    pq.load("doubles_resid.bin")
    graph.add(pq, "doubles_resid")

    # load triples_resid
    pq.load("triples_resid.bin")
    graph.add(pq, "triples_resid")
    
    return graph
    
generate_pq()
graph = load_pq()

graph.optimize()
graph.print("python")
graph.analysis()
