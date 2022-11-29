
import pdaggerq

pq = pdaggerq.pq_helper("fermi")
pq.set_print_level(0)

# energy equation

pq.set_left_operators([['1']])

print('')
print('#    < 0 | e(-T) H e(T) | 0> :')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
#pq.set_non_summed_spin_labels([[]])
energy_terms = pq.fully_contracted_strings_with_spin()
for my_term in energy_terms:
    print(my_term)

pq.clear()

# singles equations

print('')
print('#    < 0 | m* e e(-T) H e(T) | 0> (aa):')
print('')

pq.set_left_operators([['e1(m,e)']])

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
pq.set_non_summed_spin_labels([['m', 'a'], ['e', 'a']])
singles_residual_terms = pq.fully_contracted_strings_with_spin()
for my_term in singles_residual_terms:
    print(my_term)

print('')
print('#    < 0 | m* e e(-T) H e(T) | 0> (bb):')
print('')

# grab list of fully-contracted strings, then print
pq.set_non_summed_spin_labels([['m', 'b'], ['e', 'b']])
singles_residual_terms = pq.fully_contracted_strings_with_spin()
for my_term in singles_residual_terms:
    print(my_term)

pq.clear()

# doubles equations

print('')
print('#    < 0 | m* n* f e e(-T) H e(T) | 0> (aaaa):')
print('')

pq.set_left_operators([['e2(m,n,f,e)']])

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
pq.set_non_summed_spin_labels([['m', 'a'], ['n', 'a'], ['e', 'a'], ['f', 'a']])
doubles_residual_terms = pq.fully_contracted_strings_with_spin()
for my_term in doubles_residual_terms:
    print(my_term)

print('')
print('#    < 0 | m* n* f e e(-T) H e(T) | 0> (bbbb):')
print('')

# grab list of fully-contracted strings, then print
pq.set_non_summed_spin_labels([['m', 'b'], ['n', 'b'], ['e', 'b'], ['f', 'b']])
doubles_residual_terms = pq.fully_contracted_strings_with_spin()
for my_term in doubles_residual_terms:
    print(my_term)

print('')
print('#    < 0 | m* n* f e e(-T) H e(T) | 0> (abab):')
print('')

# grab list of fully-contracted strings, then print
pq.set_non_summed_spin_labels([['m', 'a'], ['n', 'b'], ['e', 'a'], ['f', 'b']])
doubles_residual_terms = pq.fully_contracted_strings_with_spin()
for my_term in doubles_residual_terms:
    print(my_term)

pq.clear()

