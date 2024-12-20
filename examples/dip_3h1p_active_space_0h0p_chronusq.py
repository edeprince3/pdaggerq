# T1 residual equations for CCSDt

import pdaggerq
from pdaggerq.chronus import to_chronus_string

pq = pdaggerq.pq_helper("fermi")
graph = pdaggerq.pq_graph({
    'print_level': 3,
    "opt_level":6,   # before fusion is level 5, using 6 activates fusion\
    "separate_sigma": True,
#    "nthreads": 16,
    "low_memory":True,
#    "no_scalars": True,
    });

pq.set_print_level(0)
# set bra/ket
pq.set_right_operators_type('DIP')
pq.set_left_operators([['a*(i)', 'a*(j)']])
pq.set_right_operators([['r2'], ['r3']])

# add operators
pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

#print('')
#print('#    0 = <0|i* j* e(-T) H e(T) (r2 + r3)|0> ... (i = act, j = act)')
#print('')

label_ranges = {
    't2' : ['all', 'all', 'all', 'all'],
    't1' : ['all', 'all'],
    'r3' : ['act', 'act', 'act', 'act'],
    'r2' : ['act', 'act'],
    'i' : ['act'],
    'j' : ['act']
}

# grab list of fully-contracted strings, then print
terms = pq.strings(label_ranges = label_ranges)
#for my_term in terms:
    #print(my_term)
graph.add(pq, "sigmaR2_IJ") 

pq.clear()


### sigma_R3 terms ###
# set right and left-hand operators
pq.set_right_operators_type('DIP')
pq.set_left_operators([['a*(i)', 'a*(j)', 'a*(k)', 'a(a)']])
pq.set_right_operators([['r2'],['r3']])

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()


#print('')
#print('    sigma(e) = <0|i* j* k* a e(-T) H e(T) (r2 + r3)|0> ... (i = act, j = act, k = act, a = act)')
#print('')

label_ranges = {
    't2' : ['all', 'all', 'all', 'all'],
    't1' : ['all', 'all'],
    'r2' : ['act', 'act'],
    'r3' : ['act', 'act', 'act', 'act'],
    'i' : ['act'],
    'j' : ['act'],
    'k' : ['act'],
    'a' : ['act']
}

# grab list of fully-contracted strings, then print
terms = pq.strings(label_ranges = label_ranges)
#for my_term in terms:
    #print(my_term)
graph.add(pq, "sigmaR3_AIJK") 


graph.optimize()
#graph.print("c++")
output = to_chronus_string(graph.str("c++"), "DIP_active_0p0h", is_active = True)
with open("chronus_output", "w") as file:
    file.write(output)

