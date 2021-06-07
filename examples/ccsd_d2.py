
import pdaggerq

pq = pdaggerq.pq_helper("fermi")

# D2(p,q,r,s) = <0|(1 + l1 + l2) e(-T) p*q*sr e(T) |0> 
pq.set_left_operators(['1','l1','l2'])

print('\n', '    D2(i,j,k,l):', '\n')
pq.add_st_operator(1.0,['e2(i,j,l,k)'],['t1','t2'])
pq.simplify()
d1_terms = pq.fully_contracted_strings()
for my_term in d1_terms:
    print(my_term)
pq.clear()

print('\n', '    D2(a,b,c,d):', '\n')
pq.add_st_operator(1.0,['e2(a,b,d,c)'],['t1','t2'])
pq.simplify()
d1_terms = pq.fully_contracted_strings()
for my_term in d1_terms:
    print(my_term)
pq.clear()

print('\n', '    D2(i,j,a,b):', '\n')
pq.add_st_operator(1.0,['e2(i,j,b,a)'],['t1','t2'])
pq.simplify()
d1_terms = pq.fully_contracted_strings()
for my_term in d1_terms:
    print(my_term)
pq.clear()

print('\n', '    D2(a,b,i,j):', '\n')
pq.add_st_operator(1.0,['e2(a,b,j,i)'],['t1','t2'])
pq.simplify()
d1_terms = pq.fully_contracted_strings()
for my_term in d1_terms:
    print(my_term)
pq.clear()

print('\n', '    D2(i,a,j,b):', '\n')
pq.add_st_operator(1.0,['e2(i,a,b,j)'],['t1','t2'])
pq.simplify()
d1_terms = pq.fully_contracted_strings()
for my_term in d1_terms:
    print(my_term)
pq.clear()

