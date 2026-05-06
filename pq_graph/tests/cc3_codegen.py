
import pdaggerq
import os


pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

graph = pdaggerq.pq_graph({
    'batched': False,
    'print_level': 3,
    'opt_level': 6,
    'nthreads': -1,
})

# singles equations

pq.set_left_operators([['e1(i,a)']])

print('')
print('    < 0 | i* a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3'])

pq.simplify()

graph.add(pq.clone(), 'singles_res', ["a","i"])

pq.clear()

# doubles equations

pq.set_left_operators([['e2(i,j,b,a)']])

print('')
print('    < 0 | i* j* b a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1','t2','t3'])

pq.simplify()
graph.add(pq.clone(), 'doubles_res', ["a","b","i","j"])
pq.clear()

# triples equations

pq.set_left_operators([['e3(i,j,k,c,b,a)']])

print('')
print('    < 0 | i* j* k* c b a e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2','t3'])
pq.add_st_operator(1.0,['v'],['t1'])

# g
pq.add_operator_product(1.0,['v'])

# [g, T1]
#pq.add_commutator(1.0,['v'],['t1'])

# [g, T2]
pq.add_commutator(1.0,['v'],['t2'])

# [[g, T1], T1]]
#pq.add_double_commutator(0.5, ['v'],['t1'],['t1'])

# [[g, T1], T2]] + [[g, T2], T1]]
pq.add_double_commutator( 1.0, ['v'],['t1'],['t2'])

# [[g, T2, T2]]
#pq.add_double_commutator( 0.5, ['v'],['t2'],['t2'])

# triple commutators

# [[[g, T1, T1], T1]
#pq.add_triple_commutator( 1.0/6.0, ['v'],['t1'],['t1'],['t1'])

# [[[g, T1, T1], T2] + [[[g, T1, T2], T1] + [[[g, T2, T1], T1]
pq.add_triple_commutator( 1.0/2.0, ['v'],['t1'],['t1'],['t2'])

# quadruple commutators: the only one that yields non-zero contributions to the doubles equations will be

# [[[[g, T1], T1], T1], T1]
#pq.add_quadruple_commutator( 1.0/24.0, ['v'],['t1'],['t1'],['t1'],['t1'])

# [[[[g, T1], T1], T1], T2] + three others
pq.add_quadruple_commutator( 1.0/6.0, ['v'],['t1'],['t1'],['t1'],['t2'])

pq.simplify()

graph.add(pq.clone(), 'triples_res', ["a","b","c","i","j","k"])

pq.clear()

# Optimize and output the graph
graph.optimize()
graph.print("python")
graph.analysis()

# Generate code generator from the graph output
graph_string = graph.str("python")

file_path = os.path.dirname(os.path.realpath(__file__))

with open(f"{file_path}/cc3_code.ref", "r") as file:
    codegen_lines = file.readlines()

with open(f"{file_path}/cc3_code.py", "w") as file:
    for line in codegen_lines:
        if line.strip() == "# INSERTED CODE":
            file.write(graph_string)
        else:
            file.write(line)

print("Code generation complete")

