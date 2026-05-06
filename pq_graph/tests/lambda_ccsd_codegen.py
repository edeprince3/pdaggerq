# ccsd lambda equations
# L = <0| (1+L) e(-T) H e(T) |0>
# dL/dtu = <0| e(-T) H e(T) |u> + <0| L e(-T) H e(T) |u> - <0| L tu e(-T) H e(T) |0>

import pdaggerq

graph = pdaggerq.pq_graph({
    'batched': False,
    'print_level': 3,
    'use_trial_index': False,
    'opt_level': 6,
    'nthreads': -1,
})

# SINGLES

pq = pdaggerq.pq_helper('fermi')

print('')
print('    0 = <0| e(-T) H e*m e(T)|0> + <0| L e(-T) [H, e*m] e(T)|0>')
print('')

#  <0| e(-T) H e*m e(T)|0>

pq.set_left_operators([['1']])
pq.set_right_operators([['1']])

pq.add_st_operator(1.0,['f','e1(e,m)'],['t1','t2'])
pq.add_st_operator(1.0,['v','e1(e,m)'],['t1','t2'])

# <0| L e(-T) [H,e*m] e(T)|0>

pq.set_left_operators([['l1'],['l2']])

pq.add_st_operator( 1.0,['f','e1(e,m)'],['t1','t2'])
pq.add_st_operator( 1.0,['v','e1(e,m)'],['t1','t2'])

pq.add_st_operator(-1.0,['e1(e,m)','f'],['t1','t2'])
pq.add_st_operator(-1.0,['e1(e,m)','v'],['t1','t2'])

pq.simplify()

graph.add(pq.clone(), 'lsingles', ['m', 'e'])

pq.clear()


# DOUBLES

pq = pdaggerq.pq_helper("fermi")

print('')
print('    0 = <0| e(-T) H e*f*nm e(T)|0> + <0| L e(-T) [H, e*f*nm] e(T)|0>')
print('')

#  <0| e(-T) H e*f*nm e(T)|0>

pq.set_left_operators([['1']])
pq.set_right_operators([['1']])

pq.add_st_operator(1.0,['f','e2(e,f,n,m)'],['t1','t2'])
pq.add_st_operator(1.0,['v','e2(e,f,n,m)'],['t1','t2'])

# <0| L e(-T) [H,e*f*nm] e(T)|0>

pq.set_left_operators([['l1'],['l2']])

pq.add_st_operator( 1.0,['f','e2(e,f,n,m)'],['t1','t2'])
pq.add_st_operator( 1.0,['v','e2(e,f,n,m)'],['t1','t2'])

pq.add_st_operator(-1.0,['e2(e,f,n,m)','f'],['t1','t2'])
pq.add_st_operator(-1.0,['e2(e,f,n,m)','v'],['t1','t2'])

pq.simplify()

graph.add(pq.clone(), 'ldoubles', ['m', 'n', 'e', 'f'])

pq.clear()

# Optimize and output the graph
graph.optimize()
graph.print("python")
graph.analysis()

import os

# Generate code generator from the graph output
graph_string = graph.str("python")

file_path = os.path.dirname(os.path.realpath(__file__))

with open(f"{file_path}/lambda_ccsd_code.ref", "r") as file:
    codegen_lines = file.readlines()

with open(f"{file_path}/lambda_ccsd_code.py", "w") as file:
    for line in codegen_lines:
        if line.strip() == "# INSERTED CODE":
            file.write(graph_string)
        else:
            file.write(line)

print("Code generation complete")