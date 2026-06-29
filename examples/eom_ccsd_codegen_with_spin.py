
import pdaggerq

pq = pdaggerq.pq_helper("fermi")

from pdaggerq.parser import contracted_strings_to_tensor_terms

# set right and left-hand operators
pq.set_left_operators([['1']])
pq.set_right_operators([['r0'],['r1'],['r2']])

print('')
print('def sigma_ref(r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab):')
print('')

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

# grab list of fully-contracted strings, then print
terms = pq.strings(spin_labels = {})
terms = contracted_strings_to_tensor_terms(terms)
for my_term in terms:
    print("#\t", my_term)
    print("%s" % (my_term.einsum_string(update_val='sigma0')))
    print()

print('return sigma0')
print('')

pq.clear()

# set right and left-hand operators
pq.set_left_operators([['e1(i,a)']])
pq.set_right_operators([['r0'],['r1'],['r2']])

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

for s in ['a', 'b']:
    print('')
    print('def sigma_singles_%s%s(r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab):' % (s, s))
    print('')
    
    # grab list of fully-contracted strings, then print
    spins = {
        'i' : s,
        'a' : s
    }
    terms = pq.strings(spin_labels = spins)
    terms = contracted_strings_to_tensor_terms(terms)
    for my_term in terms:
        print("#\t", my_term)
        print("%s" % (my_term.einsum_string(update_val='sigma1_' + s + s,
                                    output_variables=('a', 'i'))))
        print()
    
    print('return sigma1_' + s + s)
    print('')

pq.clear()

# set right and left-hand operators
pq.set_left_operators([['e2(i,j,b,a)']])
pq.set_right_operators([['r0'],['r1'],['r2']])

pq.add_st_operator(1.0,['f'],['t1','t2'])
pq.add_st_operator(1.0,['v'],['t1','t2'])

pq.simplify()

for s1 in ['a', 'b']:
    for s2 in ['a', 'b']:
        if s2 < s1 :
            continue

        spins = {
            'i' : s1,
            'a' : s1,
            'j' : s2,
            'b' : s2
        }
        
        print('')
        print('def sigma_doubles_%s%s%s%s(r0, r1_aa, r1_bb, r2_aaaa, r2_bbbb, r2_abab):' % (s1, s2, s1, s2))
        print('')

        # grab list of fully-contracted strings, then print
        terms = pq.strings(spin_labels = spins)
        terms = contracted_strings_to_tensor_terms(terms)
        for my_term in terms:
            print("#\t", my_term)
            print("%s" % (my_term.einsum_string(update_val='sigma2_' + s1 + s2 + s1 + s2,
                                        output_variables=('a', 'b', 'i', 'j'))))
            print()
        
        print('return sigma2_' + s1 + s2 + s1 + s2)
        print('')

pq.clear()

