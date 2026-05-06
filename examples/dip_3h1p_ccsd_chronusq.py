
import pdaggerq
from pdaggerq.chronus import to_chronus_string

pq = pdaggerq.pq_helper("fermi")
graph = pdaggerq.pq_graph({
    "print_level":3, # 1 prints substitution, 3 prints fusion
    "opt_level":6,   # before fusion is level 5, using 6 activates fusion\
    "separate_sigma": True,
#    "nthreads": 16,
    "low_memory":True,
#    "no_scalars": True,
    });

pq.set_print_level(0)

# set right and left-hand operators
pq.set_right_operators_type('DIP')
pq.set_left_operators([['a*(i)', 'a*(j)']])
pq.set_right_operators([['r2'], ['r3']])

#print('')
#print('    sigma(ij) = <0|i* j* e(-T) H e(T) (r2 + r3)|0>')
#print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.strings()
#for my_term in terms:
#    print(my_term)
graph.add(pq, "sigmaR2_ij")
pq.clear()

# set right and left-hand operators
pq.set_right_operators_type('DIP')
pq.set_left_operators([['a*(i)', 'a*(j)', 'a*(k)', 'a(a)']])

pq.set_right_operators([['r2'], ['r3']])

#print('')
#print('    sigma(ijka) = <0|i* j* k* a e(-T) H e(T) (r2 + r3 + r4)|0>')
#print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.strings()
#for my_term in terms:
#    print(my_term)
graph.add(pq, "sigmaR3_ijka")
pq.clear()

graph.optimize()
#graph.print("c++")
output = to_chronus_string(graph.str("c++"), "DIP_3h1p", is_active = False)
with open("chronus_output", "w") as file:
    file.write(output)


